"""The agent: an LLM consulted only at decision points, with typed JSON answers.

Decisions
- `decide_continuation`: does a table at the top of page N+1 continue the
  table at the bottom of page N? (only for cases the rules call ambiguous)
- `read_cells`: vision tie-breaker. Given a crop of a table and the cells
  where the two OCR engines disagree, read the printed value.

Any OpenAI-compatible server works: OpenRouter now (AGENT_BASE_URL default),
`llama-server` on the RTX 5070 later (AGENT_BASE_URL=http://localhost:8080/v1).
OpenRouter spend is written to a ledger and capped by OPENROUTER_BUDGET_USD.
When the agent is unavailable or over budget, every method returns None and
the caller falls back to its rules.
"""

from __future__ import annotations

import base64
import json
import re
import threading
import time
from pathlib import Path
from typing import Any

import requests

from src.config import (
    AGENT_BASE_URL,
    AGENT_MAX_INFLIGHT,
    AGENT_STARTS_PER_MIN,
    AGENT_MODEL,
    CACHE_DIR,
    INR_PER_USD,
    OPENROUTER_API_KEY,
    OPENROUTER_BUDGET_USD,
    OUT_DIR,
    VERIFIER_VLM_MODEL,
)
from src import cache
from src.rate_limit import CallFailed, Limiter, call_with_retry
from src.tables import normalize_digits

LEDGER = OUT_DIR / "openrouter_ledger.jsonl"
PRICES_CACHE = CACHE_DIR / "openrouter_models.json"

CONTINUATION_SCHEMA = {
    "type": "object",
    "properties": {
        "continues": {"type": "boolean"},
        "repeated_header_rows": {"type": "integer", "minimum": 0, "maximum": 6},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
    },
    "required": ["continues", "repeated_header_rows", "confidence", "reason"],
    "additionalProperties": False,
}

CELLS_SCHEMA = {
    "type": "object",
    "properties": {
        "answers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "value": {"type": "string"},
                    "choice": {"type": "string", "enum": ["a", "b", "neither", "unreadable"]},
                },
                "required": ["id", "value", "choice"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["answers"],
    "additionalProperties": False,
}


def _openrouter_prices() -> dict[str, tuple[float, float]]:
    """model id -> (US$ per input token, US$ per output token), cached for a day."""
    try:
        if PRICES_CACHE.is_file() and time.time() - PRICES_CACHE.stat().st_mtime < 86400:
            data = json.loads(PRICES_CACHE.read_text())
        else:
            data = requests.get("https://openrouter.ai/api/v1/models", timeout=30).json()
            PRICES_CACHE.parent.mkdir(parents=True, exist_ok=True)
            PRICES_CACHE.write_text(json.dumps(data))
        return {
            m["id"]: (float(m["pricing"].get("prompt") or 0), float(m["pricing"].get("completion") or 0))
            for m in data.get("data", [])
        }
    except Exception:  # noqa: BLE001 - prices are best effort; unknown => assume dearer
        return {}


class Agent:
    def __init__(
        self,
        base_url: str = AGENT_BASE_URL,
        model: str = AGENT_MODEL,
        vlm_model: str = VERIFIER_VLM_MODEL,
        api_key: str | None = None,
        budget_usd: float = OPENROUTER_BUDGET_USD,
        ledger: Path = LEDGER,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.vlm_model = vlm_model
        self.is_openrouter = "openrouter.ai" in self.base_url
        self.api_key = api_key if api_key is not None else (OPENROUTER_API_KEY if self.is_openrouter else "local")
        self.budget_usd = budget_usd
        self.ledger = ledger
        self._lock = threading.Lock()
        self._prices: dict[str, tuple[float, float]] | None = None
        self.limiter = Limiter(AGENT_STARTS_PER_MIN, 62.0, AGENT_MAX_INFLIGHT)
        self.calls: list[dict] = []
        self._client = None

    @property
    def available(self) -> bool:
        return bool(self.base_url) and bool(self.api_key)

    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, max_retries=0, timeout=120)
        return self._client

    # -- spend ------------------------------------------------------------

    def spent_usd(self) -> float:
        if not self.ledger.is_file():
            return 0.0
        total = 0.0
        for line in self.ledger.read_text(encoding="utf-8").splitlines():
            try:
                total += float(json.loads(line).get("usd", 0))
            except (ValueError, json.JSONDecodeError):
                continue
        return total

    def _cost_usd(self, model: str, usage: Any) -> float:
        if not self.is_openrouter:
            return 0.0
        if self._prices is None:
            self._prices = _openrouter_prices()
        pin, pout = self._prices.get(model, (3e-6, 1.5e-5))
        return (getattr(usage, "prompt_tokens", 0) or 0) * pin + (getattr(usage, "completion_tokens", 0) or 0) * pout

    def _record(self, purpose: str, model: str, usd: float, usage: Any, ok: bool) -> None:
        entry = {
            "t": round(time.time(), 3), "purpose": purpose, "model": model, "usd": round(usd, 6),
            "inr": round(usd * INR_PER_USD, 4), "ok": ok,
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
        }
        self.calls.append(entry)
        if self.is_openrouter:
            self.ledger.parent.mkdir(parents=True, exist_ok=True)
            with self._lock, self.ledger.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")

    # -- core call --------------------------------------------------------

    def _chat(self, purpose: str, model: str, messages: list[dict], schema: dict, max_tokens: int,
              extra_body: dict | None = None) -> dict | None:
        if not self.available:
            return None
        key = cache.make_key(
            json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8"),
            "agent", model, f"{purpose}|{self.base_url}|{json.dumps(schema, sort_keys=True)}",
        )
        hit = cache.get(key)
        if hit is not None:  # same question, same model: free on re-assembly
            parsed = json.loads(hit.decode("utf-8"))
            parsed["_cost_inr"] = 0.0
            parsed["_cached"] = True
            return parsed
        if self.is_openrouter and self.spent_usd() >= self.budget_usd:
            print(f"agent: OpenRouter budget US${self.budget_usd} reached; rules decide")
            return None
        extra: dict[str, Any] = extra_body if extra_body is not None else (
            {"reasoning": {"enabled": False}} if self.is_openrouter
            else {"chat_template_kwargs": {"enable_thinking": False}}
        )
        fmt = {"type": "json_schema", "json_schema": {"name": purpose, "strict": True, "schema": schema}}
        try:
            completion = call_with_retry(
                lambda: self.client().chat.completions.create(
                    model=model, messages=messages, temperature=0, max_tokens=max_tokens,
                    response_format=fmt, extra_body=extra,
                ),
                limiter=self.limiter,
                label=f"agent {purpose}",
                max_tries=3,
            )
        except CallFailed as exc:
            print(f"agent {purpose} failed: {exc.outcome.reason}")
            return None
        usage = getattr(completion, "usage", None)
        usd = self._cost_usd(model, usage)
        text = (completion.choices[0].message.content or "") if completion.choices else ""
        parsed = _parse_json(text)
        self._record(purpose, model, usd, usage, parsed is not None)
        if parsed is not None:
            parsed["_model"] = model
            cache.put(key, json.dumps(parsed, ensure_ascii=False).encode("utf-8"))
            parsed["_cost_inr"] = round(usd * INR_PER_USD, 4)
        return parsed

    # -- decisions --------------------------------------------------------

    def decide_continuation(self, evidence: dict) -> dict | None:
        messages = [
            {"role": "system", "content": (
                "You check scanned or born-digital documents for tables that continue across a page break. "
                "Given the bottom table on page N and the top table on page N+1, decide whether the second "
                "continues the first (same table split by the page break). Judge by structure, not by topic: "
                "a continuation has the same columns and no title or caption of its own, and its rows simply "
                "carry on (new row labels or categories are normal). A NEW table usually starts with its own "
                "caption/title and header row, or has different columns. If it continues, say how many of its "
                "leading rows repeat the first table's header. Answer with JSON only."
            )},
            {"role": "user", "content": json.dumps(evidence, ensure_ascii=False)},
        ]
        return self._chat("table_continuation", self.model, messages, CONTINUATION_SCHEMA, 300)

    def read_cells(self, image_png: bytes, questions: list[dict]) -> dict | None:
        """questions: [{"id", "row_label", "column_header", "a", "b"}]."""
        url = "data:image/png;base64," + base64.b64encode(image_png).decode("ascii")
        prompt = (
            "This image is a crop of a table from a document. Two OCR engines disagree on some cells. "
            "For each question, find the cell at the given row label and column header, read the number "
            "exactly as printed (keep the original digits/script), and say whether it equals candidate "
            "'a', candidate 'b', 'neither', or is 'unreadable'.\n\n"
            + json.dumps(questions, ensure_ascii=False)
        )
        messages = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": url}},
        ]}]
        return self._chat("cell_tiebreak", self.vlm_model, messages, CELLS_SCHEMA, 120 + 60 * len(questions))

    def judge_cells(self, sheet_png: bytes, questions: list[dict], style: str, model: str | None = None,
                    extra: dict | None = None, think_tokens: int = 3000) -> dict | None:
        """The image check. questions: [{"id", "where", "candidates": {"A": text, ...}}] (no candidates for audits).
        Returns {"answers": [{"id", "reading", "match", "confidence"}]} or None."""
        url = "data:image/png;base64," + base64.b64encode(sheet_png).decode("ascii")
        prompt = JUDGE_INTRO + JUDGE_STYLE[style] + "\n\nQuestions:\n" + json.dumps(questions, ensure_ascii=False)
        messages = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": url}},
        ]}]
        budget = 200 + (30 if style == "choose" else 90) * len(questions)
        if extra and "reasoning" in extra and extra["reasoning"].get("effort"):
            budget += think_tokens  # reasoning tokens count against max_tokens (measured: a truncated answer otherwise)
        answer = self._chat(f"table_{style}", model or self.vlm_model, messages, JUDGE_SCHEMA, budget, extra)
        if answer:  # some models rename ids ("Q1_red_box"): keep the question number only
            for a in answer.get("answers", []):
                m = re.match(r"\s*(Q\d+)", str(a.get("id", "")), re.I)
                if m:
                    a["id"] = m.group(1).upper()
        return answer


JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "answers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "reading": {"type": "string"},
                    "match": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                },
                "required": ["id", "reading", "match", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["answers"],
    "additionalProperties": False,
}

JUDGE_INTRO = (
    "The image is a sheet of strips cut from one table in a printed or scanned document. The top strip "
    "(HEAD) is the table's header; every other strip is labelled with a question id (Q1, Q2, ...). "
    "Each question points at one cell: either the cell inside the red box of its strip, or, when there is "
    "no box, the cell described in words (its column header and other text in the same row). "
    "Read the cell exactly as printed: keep the script of every letter and digit (do not turn Indian "
    "digits into 0-9 or the reverse), keep punctuation, commas, decimal points, brackets and dashes. "
    "If the cell is empty, the reading is an empty string. If you cannot read it, set confidence to low."
)
JUDGE_STYLE = {
    "dispute": " Then compare your reading with the candidates and set match to the letter of the candidate "
               "that is exactly what is printed, or to 'none' if no candidate is exactly right.",
    "audit": " There are no candidates: set match to '-'.",
    "choose": " Compare the printed cell with the candidates and set match to the letter of the candidate that is "
              "exactly what is printed, or to 'none' if no candidate is exactly right. Only when match is 'none', set "
              "reading to what is printed; otherwise set reading to an empty string.",
}


def _parse_json(text: str) -> dict | None:
    """The JSON object in a reply. A bare list of answers (some open models skip the wrapper) becomes {"answers": [...]}."""
    text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if isinstance(data, list) and all(isinstance(x, dict) for x in data):
        return {"answers": data}
    return data if isinstance(data, dict) else None


def same_number(a: str, b: str) -> bool:
    norm = lambda s: normalize_digits(s).replace(",", "").replace(" ", "").strip()  # noqa: E731
    return norm(a) == norm(b)
