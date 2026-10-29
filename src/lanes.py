"""Engine lanes for one document run.

- SarvamLane: one thread. Groups pending pages into jobs of several upright
  chunk PDFs (<= 10 pages each), submits, polls, downloads, and maps results
  back to original page numbers. Pages already in the cache never reach a job.
- PageLane: N worker threads for a per-page engine (Bodhan, optional Qwen).
  Renders upright pages, uses the cache, calls the backend, and may take
  ("steal") pages from the end of Sarvam's queue when Sarvam has a backlog.
- Router: which engine reads which page, and what happens on failure.
- Budget: rupee cap shared by all lanes.
"""

from __future__ import annotations

import io
import json
import os
import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

import pymupdf

from src import cache
from src import sarvam_digitise as sarvam_api
from src.blocks import Block, from_bodhan, from_sarvam, from_text_page, page_markdown
from src.bodhan_ocr import MAX_TOKENS, bodhan_request
from src.langid import page_languages
from src.config import (
    BODHAN_BACKEND,
    BODHAN_INR_PER_PAGE,
    BODHAN_MODEL,
    INR_PER_USD,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    QWEN_MODEL,
    QWEN_USD_PER_M_IN,
    QWEN_USD_PER_M_OUT,
    SARVAM_INR_PER_PAGE,
    SARVAM_MAX_PAGES_PER_PDF,
)
from src.manifest import EventLog, Manifest
from src.pdf_pages import RenderSpec, build_chunk_pdf, render_page
from src.rate_limit import CallFailed, FatalError, Limiter, Outcome, RetriesExhausted, Verdict, call_with_retry

FATAL_LANE_REASONS = {"http_401", "http_402", "http_403", "credit_exhausted"}
POLICIES = ("sarvam-primary", "bodhan-primary", "both", "sarvam-only", "bodhan-only")
JOB_TIMEOUT_S = 60 * 60


@dataclass
class Settings:
    """Everything about a run that changes its output. The fingerprint is checked when a run is resumed."""
    language: str
    policy: str = "sarvam-primary"
    dpi: int = 200
    table_format: str = "html"
    chunk_pages: int = SARVAM_MAX_PAGES_PER_PDF
    files_per_job: int = 5
    max_open_jobs: int = 3
    bodhan_workers: int = 4
    steal: bool = True
    qwen: bool = False
    max_pages: int | None = None
    max_cost_inr: float | None = None
    pages: list[int] | None = None  # explicit page selection (1-based), e.g. windows of a long PDF
    doc_type: str = "auto"  # page_class: auto (per page) | digital | scanned
    lead: str = "sarvam"  # the engine that reads every page and supplies the document (modes.default_lead)

    def __post_init__(self) -> None:
        # the primary policies name their lead; "both" takes it from `lead`
        if self.policy == "bodhan-primary":
            self.lead = "bodhan"
        elif self.policy == "sarvam-primary":
            self.lead = "sarvam"

    def fingerprint(self) -> dict:
        """Settings that change outputs. A resume must match these."""
        return {
            "language": self.language,
            "policy": self.policy,
            "dpi": self.dpi,
            "table_format": self.table_format,
            "bodhan_model": BODHAN_MODEL,
            "sarvam_output": sarvam_api.OUTPUT_FORMAT,
            "qwen": self.qwen,
            **({"doc_type": self.doc_type} if self.doc_type != "auto" else {}),  # absent for auto: older runs still resume
            **({"bodhan_backend": _local_backend_tag()} if BODHAN_BACKEND == "local" else {}),  # absent for the API: older runs still resume
            **({"lead": self.lead} if self.lead != "sarvam" else {}),  # absent for Sarvam: older runs still resume
        }


def _local_backend_tag() -> str:
    from src.indic_local import local_cache_extra

    return "local|" + local_cache_extra("")


class Budget:
    """Spending cap shared by all lanes: reserve before a call, settle after it."""
    def __init__(self, cap_inr: float | None, spent_inr: float = 0.0) -> None:
        self.cap = cap_inr
        self.spent = spent_inr
        self.reserved = 0.0
        self._lock = threading.Lock()

    def reserve(self, amount: float, *, force: bool = False) -> bool:
        with self._lock:
            if not force and self.cap is not None and self.spent + self.reserved + amount > self.cap + 1e-9:
                return False
            self.reserved += amount
            return True

    def settle(self, reserved: float, actual: float) -> None:
        with self._lock:
            self.reserved = max(0.0, self.reserved - reserved)
            self.spent += actual


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


class Context:
    """Everything the lanes share for one run."""

    def __init__(
        self,
        pdf_path: Path,
        out_dir: Path,
        settings: Settings,
        manifest: Manifest,
        events: EventLog,
        budget: Budget,
        src_sha: str,
    ) -> None:
        self.pdf_path = pdf_path
        self.out_dir = out_dir
        self.settings = settings
        self.manifest = manifest
        self.events = events
        self.budget = budget
        self.src_sha = src_sha
        self.stop = threading.Event()  # graceful: take no new work
        self.hard_stop = threading.Event()  # second Ctrl-C: leave now
        self.done = threading.Event()  # the run is finished; lanes exit
        self.triage_done = threading.Event()
        self.budget_exhausted = threading.Event()
        self.alive: dict[str, bool] = {"sarvam": True, "bodhan": True, "qwen": True}
        self._tls = threading.local()

    def doc(self) -> pymupdf.Document:
        """PyMuPDF is not thread-safe: one open document per thread."""
        doc = getattr(self._tls, "doc", None)
        if doc is None:
            doc = pymupdf.open(self.pdf_path)
            self._tls.doc = doc
        return doc

    def path(self, *parts: str) -> Path:
        return self.out_dir.joinpath(*parts)

    def accepting(self) -> bool:
        return not (self.stop.is_set() or self.budget_exhausted.is_set() or self.hard_stop.is_set())

    def language_prior(self) -> str | None:
        lang = self.settings.language
        return lang if lang not in ("auto", "en-IN") else None

    def save_page_outputs(self, page: int, engine: str, blocks: list[Block]) -> dict:
        """Tag each block's language from its text, save, and return the page's languages."""
        per_block, page_langs = page_languages(
            [b.text for b in blocks], prior=self.language_prior(), kinds=[b.kind for b in blocks]
        )
        for block, found in zip(blocks, per_block):
            block.lang = found.dominant
        summary = page_langs.to_dict()
        summary["indic_share"] = round(sum(v for k, v in page_langs.shares.items() if k != "en-IN"), 3)
        self.manifest.update_detected(page, engine, summary)
        write_json(self.path("blocks", f"page_{page:05d}.{engine}.json"), [b.to_dict() for b in blocks])
        md = self.path("md", f"page_{page:05d}.{engine}.md")
        md.parent.mkdir(parents=True, exist_ok=True)
        md.write_text(page_markdown(blocks), encoding="utf-8")
        return summary


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


class Router:
    """Decides which engine reads which page and what happens when a page or a lane fails. It never calls an API."""
    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx
        self.policy = ctx.settings.policy

    def engines(self) -> set[str]:
        used = {
            "sarvam-primary": {"sarvam", "bodhan"},
            "bodhan-primary": {"sarvam", "bodhan"},
            "both": {"sarvam", "bodhan"},
            "sarvam-only": {"sarvam"},
            "bodhan-only": {"bodhan"},
        }[self.policy]
        return used | ({"qwen"} if self.ctx.settings.qwen else set())

    def initial_tasks(self, flags: Any) -> dict[str, tuple[str, int]]:
        engines = self.engines()
        if flags.blank:
            return {e: ("skipped", 0) for e in engines if e != "qwen"}
        alive = self.ctx.alive
        tasks: dict[str, tuple[str, int]] = {}
        if self.policy == "bodhan-primary":
            # Bodhan reads every page; Sarvam reads the table pages, or the page when Bodhan is down
            if alive["bodhan"]:
                tasks["bodhan"] = ("pending", 0)
                if flags.has_table and alive["sarvam"]:
                    tasks["sarvam"] = ("pending", 1)
            elif alive["sarvam"]:
                tasks["sarvam"] = ("pending", 2)
            if "qwen" in engines and flags.language.startswith("en"):
                tasks["qwen"] = ("pending", 0)
            return tasks
        if "sarvam" in engines:
            if alive["sarvam"]:
                tasks["sarvam"] = ("pending", 0)
            elif self.policy != "sarvam-only" and alive["bodhan"]:
                tasks["bodhan"] = ("pending", 2)
        if self.policy in ("both", "bodhan-only"):
            tasks["bodhan"] = ("pending", 0)
        if self.policy == "sarvam-primary" and flags.has_table and alive["bodhan"]:
            tasks.setdefault("bodhan", ("pending", 1))
        if "qwen" in engines and flags.language.startswith("en"):
            tasks["qwen"] = ("pending", 0)
        return tasks

    def after_sarvam_page(self, page: int, blocks: list[Block]) -> None:
        """Sarvam found a table the text layer did not: let Bodhan check it."""
        if self.policy == "sarvam-primary" and self.ctx.alive["bodhan"]:
            if any(b.kind == "table" for b in blocks) and self.ctx.manifest.ensure_task(page, "bodhan", 1):
                self.ctx.events.log("verify_table", page=page)

    def after_bodhan_page(self, page: int, blocks: list[Block]) -> None:
        """Bodhan leads and found a table triage did not: let Sarvam give the second reading."""
        if self.policy == "bodhan-primary" and self.ctx.alive["sarvam"]:
            if any(b.kind == "table" for b in blocks) and self.ctx.manifest.ensure_task(page, "sarvam", 1):
                self.ctx.events.log("verify_table", page=page, engine="sarvam")

    def on_page_failed(self, page: int, engine: str) -> None:
        if engine == "bodhan" and self.policy == "bodhan-primary" and self.ctx.alive["sarvam"]:
            if self.ctx.manifest.ensure_task(page, "sarvam", 2):
                self.ctx.events.log("fallback", page=page, to="sarvam")
            return
        if engine == "sarvam" and self.policy not in ("sarvam-only", "bodhan-primary") and self.ctx.alive["bodhan"]:
            if self.ctx.manifest.ensure_task(page, "bodhan", 2):
                self.ctx.events.log("fallback", page=page, to="bodhan")
        if engine == "bodhan" and self.ctx.manifest.reopen_stolen(page):
            self.ctx.events.log("fallback", page=page, to="sarvam")

    def on_lane_dead(self, engine: str, reason: str) -> None:
        self.ctx.alive[engine] = False
        pages = self.ctx.manifest.fail_lane(engine, reason)
        self.ctx.events.log("lane_dead", engine=engine, reason=reason, pages=len(pages))
        print(f"{engine} lane stopped: {reason}")
        if engine == "sarvam" and reason in ("http_402", "credit_exhausted"):
            print("  Sarvam has no credit. Its pages now go to Bodhan alone: no second reading, so every table will need review. "
                  "Add credit and run again with --fresh (not --resume: failed pages are not retried; the cache keeps what was paid for).")
        for page in pages:
            self.on_page_failed(page, engine)


# ---------------------------------------------------------------------------
# Per-page backends
# ---------------------------------------------------------------------------


class PageBackend(Protocol):
    """What a per-page engine (Bodhan API, local Bodhan, Qwen) has to provide to the page lane."""
    engine: str
    cache_ns: str
    model: str
    price_inr: float
    fatal_reasons: set[str]

    def cache_extra(self) -> str: ...
    def ocr(self, image: bytes, mime: str, on_outcome: Any) -> dict: ...
    def to_blocks(self, payload: dict, page: int, width: int, height: int) -> list[Block]: ...
    def cost_inr(self, payload: dict) -> float: ...


class BodhanAPIBackend:
    """Bodhan through its hosted HTTP API: ₹0.20 a page, 8 requests a minute."""
    engine = "bodhan"
    cache_ns = "indic-ocr"
    model = BODHAN_MODEL
    price_inr = BODHAN_INR_PER_PAGE
    fatal_reasons = FATAL_LANE_REASONS

    def __init__(self, table_format: str = "html") -> None:
        self.table_format = table_format

    def cache_extra(self) -> str:
        return f"{self.table_format}|{MAX_TOKENS}"

    def ocr(self, image: bytes, mime: str, on_outcome: Any) -> dict:
        return bodhan_request(image, mime, table_format=self.table_format, on_outcome=on_outcome)

    def to_blocks(self, payload: dict, page: int, width: int, height: int) -> list[Block]:
        return from_bodhan(payload, page, width, height)

    def cost_inr(self, payload: dict) -> float:
        return self.price_inr


class BodhanLocalBackend:
    """Local Bodhan. MLX BF16 on Apple Silicon, official PyTorch on CUDA. Cost is 0."""

    engine = "bodhan"
    cache_ns = "indic-ocr-local"
    model = "bodhan-ai/indic-ocr"
    price_inr = 0.0
    fatal_reasons = FATAL_LANE_REASONS | {"missing_runtime", "missing_weights", "gated_model"}

    def __init__(self, table_format: str = "html") -> None:
        from src.indic_local import active_model

        self.table_format = table_format
        self.model = active_model()

    def cache_extra(self) -> str:
        from src.indic_local import local_cache_extra

        return local_cache_extra(self.table_format)

    def ocr(self, image: bytes, mime: str, on_outcome: Any) -> dict:
        from src.indic_local import local_ocr

        try:
            payload = local_ocr(image, mime, table_format=self.table_format)
        except ImportError as exc:
            raise FatalError(Outcome(Verdict.FATAL, "missing_runtime"), 1) from exc
        except FileNotFoundError as exc:
            raise FatalError(Outcome(Verdict.FATAL, "missing_weights"), 1) from exc
        except Exception as exc:
            name = type(exc).__name__
            if name in ("GatedRepoError", "LocalEntryNotFoundError"):
                raise FatalError(Outcome(Verdict.FATAL, "gated_model"), 1) from exc
            if name == "VllmUnavailable":
                raise FatalError(Outcome(Verdict.FATAL, "missing_runtime"), 1) from exc
            raise
        if on_outcome is not None:
            on_outcome(Outcome(Verdict.OK))
        return payload

    def to_blocks(self, payload: dict, page: int, width: int, height: int) -> list[Block]:
        return from_bodhan(payload, page, width, height)

    def cost_inr(self, payload: dict) -> float:
        return 0.0


QWEN_PROMPT = (
    "OCR this document page.\n"
    "Return only the page text.\n"
    "Keep reading order.\n"
    "Use HTML tables with rowspan/colspan for merged cells.\n"
    "Keep the original script. Do not translate.\n"
    "Do not add a title, preface, or closing comment."
)


class QwenBackend:
    """Optional English-page reader through OpenRouter (no layout blocks)."""

    engine = "qwen"
    cache_ns = "openrouter-qwen-page"
    model = QWEN_MODEL
    price_inr = 0.05  # reservation; actual cost comes from token usage
    fatal_reasons = FATAL_LANE_REASONS | {"http_404"}

    def __init__(self) -> None:
        from openai import OpenAI

        self.client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY, max_retries=0, timeout=120)
        self.limiter = Limiter(60, 62.0, 4)

    def cache_extra(self) -> str:
        return "v2|html-tables|4000|no-reasoning"

    def ocr(self, image: bytes, mime: str, on_outcome: Any) -> dict:
        import base64

        url = f"data:{mime};base64," + base64.b64encode(image).decode("ascii")
        completion = call_with_retry(
            lambda: self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                max_tokens=4000,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": QWEN_PROMPT},
                            {"type": "image_url", "image_url": {"url": url}},
                        ],
                    }
                ],
                extra_body={"reasoning": {"enabled": False}},
            ),
            limiter=self.limiter,
            label="qwen",
            on_outcome=on_outcome,
        )
        usage = getattr(completion, "usage", None)
        return {
            "markdown": (completion.choices[0].message.content or "") if completion.choices else "",
            "usage": {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(usage, "total_tokens", 0) or 0,
            },
            "model": self.model,
        }

    def to_blocks(self, payload: dict, page: int, width: int, height: int) -> list[Block]:
        return from_text_page(payload.get("markdown") or "", page, self.engine)

    def cost_inr(self, payload: dict) -> float:
        u = payload.get("usage") or {}
        usd = (u.get("prompt_tokens", 0) * QWEN_USD_PER_M_IN + u.get("completion_tokens", 0) * QWEN_USD_PER_M_OUT) / 1e6
        return usd * INR_PER_USD


# ---------------------------------------------------------------------------
# Per-page lane
# ---------------------------------------------------------------------------


class PageLane:
    """Worker threads that read pages with one per-page engine, serve cache hits and save the blocks."""
    def __init__(
        self,
        ctx: Context,
        router: Router,
        backend: PageBackend,
        workers: int,
        *,
        steal_from: str | None = None,
        steal_min_backlog: int = 0,
        max_attempts: int = 3,
    ) -> None:
        self.ctx = ctx
        self.router = router
        self.backend = backend
        self.engine = backend.engine
        self.workers = max(1, workers)
        self.steal_from = steal_from
        self.steal_min_backlog = steal_min_backlog
        self.max_attempts = max_attempts
        self.threads: list[threading.Thread] = []
        self._busy = 0
        self._busy_lock = threading.Lock()

    @property
    def busy(self) -> bool:
        with self._busy_lock:
            return self._busy > 0

    def start(self) -> None:
        for i in range(self.workers):
            t = threading.Thread(target=self._worker, name=f"{self.engine}-{i}", daemon=True)
            t.start()
            self.threads.append(t)

    def join(self, timeout: float | None = None) -> None:
        for t in self.threads:
            t.join(timeout)

    def _on_outcome(self, outcome: Outcome) -> None:
        self.ctx.events.log(
            "outcome", engine=self.engine, verdict=outcome.verdict.value, reason=outcome.reason, wait_s=outcome.wait_s
        )

    def _worker(self) -> None:
        ctx = self.ctx
        while not (ctx.done.is_set() or ctx.hard_stop.is_set()):
            if not ctx.accepting() or not ctx.alive[self.engine]:
                return
            steal = self.steal_from if (ctx.settings.steal and self.steal_from and ctx.alive[self.steal_from]) else None
            claim = ctx.manifest.claim_page_task(
                self.engine, steal_from=steal, steal_min_backlog=self.steal_min_backlog
            )
            if claim is None:
                time.sleep(0.4)
                continue
            page, stolen = claim
            with self._busy_lock:
                self._busy += 1
            try:
                if stolen:
                    ctx.events.log("steal", page=page, engine=self.engine, owner=self.steal_from)
                self._process(page)
            except Exception as exc:  # noqa: BLE001 - one page must not kill the lane
                reason = f"error:{type(exc).__name__}:{str(exc)[:120]}"
                ctx.events.log("page_error", page=page, engine=self.engine, reason=reason)
                if ctx.manifest.retry_or_fail(page, self.engine, reason, self.max_attempts) == "failed":
                    self.router.on_page_failed(page, self.engine)
            finally:
                with self._busy_lock:
                    self._busy -= 1

    def _process(self, page: int) -> None:
        ctx, backend = self.ctx, self.backend
        info = ctx.manifest.page_info(page)
        rendered = render_page(ctx.doc(), page, RenderSpec(dpi=ctx.settings.dpi, rotate=info["rotate"]))
        key = cache.make_key(rendered.data, backend.cache_ns, backend.model, backend.cache_extra())
        started = time.perf_counter()
        raw = cache.get(key)
        if raw is not None:
            payload, cost, hit = json.loads(raw.decode("utf-8")), 0.0, True
        else:
            if not ctx.budget.reserve(backend.price_inr):
                ctx.manifest.release([page], self.engine, "budget")
                ctx.budget_exhausted.set()
                ctx.events.log("budget_stop", engine=self.engine, spent=round(ctx.budget.spent, 2))
                return
            try:
                payload = backend.ocr(rendered.data, rendered.mime, self._on_outcome)
            except CallFailed as exc:
                ctx.budget.settle(backend.price_inr, 0.0)
                reason = exc.outcome.reason
                ctx.events.log("page_error", page=page, engine=self.engine, reason=reason)
                if reason in backend.fatal_reasons:
                    ctx.manifest.release([page], self.engine, reason)
                    self.router.on_lane_dead(self.engine, reason)
                elif ctx.manifest.retry_or_fail(page, self.engine, reason, self.max_attempts) == "failed":
                    self.router.on_page_failed(page, self.engine)
                return
            cost, hit = backend.cost_inr(payload), False
            ctx.budget.settle(backend.price_inr, cost)
            cache.put(key, json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        latency = time.perf_counter() - started
        result_path = ctx.path(self.engine, "pages", f"page_{page:05d}.json")
        write_json(
            result_path,
            {
                "engine": self.engine,
                "page": page,
                "render": {
                    "width": rendered.width,
                    "height": rendered.height,
                    "dpi": round(rendered.dpi, 2),
                    "rotate": rendered.rotate,
                    "transform": list(rendered.transform),
                },
                "payload": payload,
            },
        )
        blocks = backend.to_blocks(payload, page, rendered.width, rendered.height)
        ctx.save_page_outputs(page, self.engine, blocks)
        if self.engine == "bodhan":
            self.router.after_bodhan_page(page, blocks)  # before `finish`, so the run cannot end in between
        usage = payload.get("usage") or {}
        degraded = (payload.get("_infinite") or {}).get("degraded") or None
        ctx.manifest.finish(
            page,
            self.engine,
            result_path=str(result_path.relative_to(ctx.out_dir)),
            latency_s=None if hit else round(latency, 3),
            tokens=usage.get("total_tokens"),
            cost_inr=cost,
            cache_hit=int(hit),
            degraded=degraded,
        )
        ctx.events.log(
            "page_done", page=page, engine=self.engine, cache_hit=hit, cost_inr=round(cost, 4),
            latency_s=None if hit else round(latency, 2), degraded=degraded,
        )


# ---------------------------------------------------------------------------
# Sarvam lane
# ---------------------------------------------------------------------------


def job_file_name(job: int, index: int) -> str:
    return f"j{job:05d}f{index:02d}.pdf"


def map_job_pages(zip_bytes: bytes, job: int, files: list[list[int]]) -> dict[int, dict]:
    """ZIP `<file>/metadata/page_NNN.json` -> original page number -> page JSON."""
    out: dict[int, dict] = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        names = set(archive.namelist())
        for index, pages in enumerate(files):
            folder = job_file_name(job, index)
            for position, page in enumerate(pages, start=1):
                name = f"{folder}/metadata/page_{position:03d}.json"
                if name not in names:
                    continue
                data = json.loads(archive.read(name))
                if int(data.get("page_num", position)) != position:
                    continue
                out[page] = data
    return out


@dataclass
class OpenJob:
    """A Sarvam job that has been submitted and is being polled."""
    job: int
    job_id: str
    n_pages: int
    next_poll: float
    delay: float
    reserved: float
    submitted_at: float
    fetch_failures: int = 0


class SarvamLane:
    """The thread that submits page chunks to Sarvam, polls the jobs, maps pages back and saves the results."""
    def __init__(self, ctx: Context, router: Router, api: ModuleType | Any = sarvam_api, max_attempts: int = 2) -> None:
        self.ctx = ctx
        self.router = router
        self.api = api
        self.max_attempts = max_attempts
        self.open: dict[int, OpenJob] = {}
        self.busy = False
        self.thread = threading.Thread(target=self.run, name="sarvam", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def join(self, timeout: float | None = None) -> None:
        self.thread.join(timeout)

    @property
    def per_job(self) -> int:
        s = self.ctx.settings
        return min(s.chunk_pages, SARVAM_MAX_PAGES_PER_PDF) * max(1, s.files_per_job)

    def _on_outcome(self, outcome: Outcome) -> None:
        self.ctx.events.log(
            "outcome", engine="sarvam", verdict=outcome.verdict.value, reason=outcome.reason, wait_s=outcome.wait_s
        )

    def run(self) -> None:
        ctx = self.ctx
        try:
            self.busy = True
            self._restore()
            while not (ctx.done.is_set() or ctx.hard_stop.is_set()):
                self.busy = True
                now = time.monotonic()
                for st in list(self.open.values()):
                    if now >= st.next_poll:
                        self._poll(st)
                if ctx.accepting() and ctx.alive["sarvam"] and len(self.open) < ctx.settings.max_open_jobs:
                    self._submit_next()
                self.busy = False
                if not self.open and (not ctx.accepting() or not ctx.alive["sarvam"]):
                    return
                time.sleep(0.25)
        except Exception as exc:  # noqa: BLE001 - surface, do not hang the run
            ctx.events.log("lane_crash", engine="sarvam", error=repr(exc)[:300])
            print(f"sarvam lane crashed: {exc!r}")
            self.open.clear()
            self.router.on_lane_dead("sarvam", "lane_crash")
        finally:
            self.busy = False

    # -- resume -----------------------------------------------------------

    def _restore(self) -> None:
        m = self.ctx.manifest
        for row in m.open_jobs():
            pages = [p for f in row["files"] for p in f]
            if row["state"] == "submitting" or not row["job_id"]:
                # Crashed between recording and submitting: the job may or may not exist.
                m.job_finished(row["id"], "abandoned", error="crashed while submitting")
                m.release(pages, "sarvam", "resubmit after crash")
                self.ctx.events.log("job_abandoned", job=row["id"], pages=len(pages), possible_duplicate=True)
                continue
            reserved = len(pages) * SARVAM_INR_PER_PAGE
            self.ctx.budget.reserve(reserved, force=True)
            now = time.monotonic()
            self.open[row["id"]] = OpenJob(row["id"], row["job_id"], len(pages), now, 5.0, reserved, now)
            self.ctx.events.log("job_resumed", job=row["id"], job_id=row["job_id"], pages=len(pages))
        orphans = [p for p, job in m.queued("sarvam").items() if job not in self.open]
        if orphans:
            m.release(orphans, "sarvam", "orphan after resume")

    # -- per-page cache ---------------------------------------------------

    def _page_key(self, page: int) -> str:
        info = self.ctx.manifest.page_info(page)
        ident = f"{self.ctx.src_sha}|p{page}|r{info['rotate']}".encode("utf-8")
        return cache.make_key(ident, "sarvam-vision-page", "digitise", f"{info['language']}|{sarvam_api.OUTPUT_FORMAT}")

    def _ingest(self, page: int, page_json: dict, *, cache_hit: bool, cost: float, job: int | None) -> bool:
        """Save a page result. Returns False if the page goes back for an Indic-mode re-read."""
        ctx = self.ctx
        if not cache_hit:
            cache.put(self._page_key(page), json.dumps(page_json, ensure_ascii=False).encode("utf-8"))
        result_path = ctx.path("sarvam", "pages", f"page_{page:05d}.json")
        write_json(result_path, {"engine": "sarvam", "page": page, "job": job, "payload": page_json})
        blocks = from_sarvam(page_json, page)
        languages = ctx.save_page_outputs(page, "sarvam", blocks)
        info = ctx.manifest.page_info(page)
        if info.get("ocr_mode") == "en" and languages["indic_share"] >= 0.03 and not info.get("mode_retried"):
            # Read in English mode but the page has Indic script: English mode
            # corrupts Indic text, so read it again in Indic mode, once.
            indic = ctx.settings.language if ctx.settings.language not in ("auto", "en-IN") else "auto"
            ctx.manifest.set_page_mode(page, "indic", indic, "ocr_output", retried=True)
            ctx.manifest.add_extra("sarvam_mode_retry", cost, page, {"indic_share": languages["indic_share"]})
            ctx.manifest.release([page], "sarvam", "re-read in Indic mode")
            ctx.events.log("mode_retry", page=page, indic_share=languages["indic_share"], shares=languages["shares"])
            return False
        self.router.after_sarvam_page(page, blocks)  # before `done`, so the run cannot end in between
        ctx.manifest.finish(
            page, "sarvam", result_path=str(result_path.relative_to(ctx.out_dir)),
            cost_inr=cost, cache_hit=int(cache_hit), job=job,
        )
        ctx.events.log("page_done", page=page, engine="sarvam", cache_hit=cache_hit, cost_inr=cost, job=job,
                       languages=languages["shares"])
        return True

    # -- submit -----------------------------------------------------------

    def _submit_next(self) -> None:
        ctx, m = self.ctx, self.ctx.manifest
        pending = m.count("sarvam", ("pending",))
        if pending == 0:
            return
        if not ctx.triage_done.is_set() and pending < self.per_job:
            return  # let the next job fill up while triage runs
        language, pages = m.claim_sarvam(self.per_job)
        if not pages or language is None:
            return
        todo: list[int] = []
        for page in pages:
            raw = cache.get(self._page_key(page))
            if raw is not None:
                self._ingest(page, json.loads(raw.decode("utf-8")), cache_hit=True, cost=0.0, job=None)
            else:
                todo.append(page)
        if not todo:
            return
        reserved = len(todo) * SARVAM_INR_PER_PAGE
        if not ctx.budget.reserve(reserved):
            m.release(todo, "sarvam", "budget")
            ctx.budget_exhausted.set()
            ctx.events.log("budget_stop", engine="sarvam", spent=round(ctx.budget.spent, 2))
            return
        size = min(ctx.settings.chunk_pages, SARVAM_MAX_PAGES_PER_PDF)
        files = [todo[i : i + size] for i in range(0, len(todo), size)]
        job = m.new_job(files, language)
        rotations = {p: m.page_info(p)["rotate"] for p in todo}
        pdfs = [(job_file_name(job, k), build_chunk_pdf(ctx.doc(), f, rotations)) for k, f in enumerate(files)]
        try:
            job_id = self.api.submit(pdfs, None if language == "auto" else language, on_outcome=self._on_outcome)
        except CallFailed as exc:
            ctx.budget.settle(reserved, 0.0)
            reason = exc.outcome.reason
            ctx.events.log("job_submit_failed", job=job, reason=reason, pages=len(todo))
            if isinstance(exc, RetriesExhausted) and reason == "http_429":
                m.job_finished(job, "abandoned", error=reason)
                m.release(todo, "sarvam", reason)
                return
            uncertain = reason.startswith("submit_uncertain")
            m.job_finished(job, "abandoned" if uncertain else "failed", error=reason)
            if uncertain:
                ctx.events.log("possible_duplicate_job", job=job)
            if reason in FATAL_LANE_REASONS:
                m.release(todo, "sarvam", reason)
                self.router.on_lane_dead("sarvam", reason)
                return
            for page in todo:
                if m.retry_or_fail(page, "sarvam", reason, self.max_attempts) == "failed":
                    self.router.on_page_failed(page, "sarvam")
            return
        m.job_submitted(job, job_id)
        now = time.monotonic()
        self.open[job] = OpenJob(job, job_id, len(todo), now + self.api.first_poll_delay(len(todo)), 5.0, reserved, now)
        ctx.events.log("job_submitted", job=job, job_id=job_id, files=len(files), pages=len(todo), language=language)
        print(f"sarvam job {job} submitted: {len(todo)} pages in {len(files)} files ({job_id[:8]})")

    # -- poll and complete ------------------------------------------------

    def _poll(self, st: OpenJob) -> None:
        try:
            status, data = self.api.poll(st.job_id, on_outcome=self._on_outcome)
        except FatalError as exc:
            self._complete(st, f"poll_failed:{exc.outcome.reason}", {})
            return
        except CallFailed:
            st.next_poll = time.monotonic() + 60
            return
        usage = data.get("usage") or {}
        self.ctx.events.log("job_status", job=st.job, status=status, processed=usage.get("pages_processed"))
        if status in sarvam_api.TERMINAL:
            self._complete(st, status, data)
        elif time.monotonic() - st.submitted_at > JOB_TIMEOUT_S:
            self._complete(st, "job_timeout", data)
        else:
            st.delay = min(30.0, st.delay * 1.5)
            st.next_poll = time.monotonic() + st.delay

    def _complete(self, st: OpenJob, status: str, data: dict) -> None:
        ctx, m = self.ctx, self.ctx.manifest
        row = m.job(st.job)
        results: dict[int, dict] = {}
        if status in sarvam_api.OK_STATUS:
            zip_path = ctx.path("sarvam", "jobs", f"{st.job_id}.zip")
            try:
                self.api.fetch(st.job_id, zip_path, on_outcome=self._on_outcome)
                results = map_job_pages(zip_path.read_bytes(), st.job, row["files"])
            except (CallFailed, zipfile.BadZipFile, OSError) as exc:
                st.fetch_failures += 1
                ctx.events.log("job_fetch_failed", job=st.job, error=repr(exc)[:200], tries=st.fetch_failures)
                if st.fetch_failures < 5:
                    st.next_poll = time.monotonic() + 30
                    return
        pages = [p for f in row["files"] for p in f]
        for page in pages:
            if page in results:
                self._ingest(page, results[page], cache_hit=False, cost=SARVAM_INR_PER_PAGE, job=st.job)
            elif m.retry_or_fail(page, "sarvam", f"missing:{status}", self.max_attempts) == "failed":
                self.router.on_page_failed(page, "sarvam")
        m.job_finished(st.job, "done" if results else "failed", status=status)
        ctx.budget.settle(st.reserved, len(results) * SARVAM_INR_PER_PAGE)
        self.open.pop(st.job, None)
        ctx.events.log("job_done", job=st.job, status=status, pages=len(pages), ok=len(results))
        print(f"sarvam job {st.job} {status}: {len(results)}/{len(pages)} pages")
