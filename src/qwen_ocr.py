"""OpenRouter Qwen OCR: one PNG per HTTP call. English pages only."""

from __future__ import annotations

import base64
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from openai import APIStatusError, OpenAI

from src import cache
from src.config import (
    OPENROUTER_API_KEY,
    QWEN_MODEL,
    QWEN_PAR_DIR,
    QWEN_SEQ_DIR,
    QWEN_WORKERS,
    has_openrouter_key,
)
from src.pdf_pages import PagePng, shred_pdf
from src.rate_limit import call_with_429_retry, get_limiter

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
MAX_TOKENS = 4000
TEMPERATURE = 0
QWEN_RATE_PER_MIN = 60.0
SLOW_SECONDS = 45.0
QWEN_LANGUAGE = "en-IN"

PROMPT = (
    "OCR this document page.\n"
    "Return only the page text.\n"
    "Keep reading order.\n"
    "Use Markdown tables.\n"
    "Keep the original script. Do not translate.\n"
    "Do not add a title, preface, or closing comment."
)

THINK_MARKERS = ("<think>", "</think>", "<thinking>", "</thinking>")


class QwenModelMissing(RuntimeError):
    """404 on the model slug. Stop this path. Do not switch model."""


@dataclass
class PageOcrResult:
    page_num: int
    markdown: str = ""
    cache_hit: bool = False
    error: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    md_path: Path | None = None
    used_no_reasoning: bool = False


@dataclass
class QwenRun:
    skipped: bool = False
    model_missing: bool = False
    error: str | None = None
    results: list[PageOcrResult] = field(default_factory=list)
    wall_seconds: float = 0.0
    workers: int = 1
    model: str = QWEN_MODEL


def _client() -> OpenAI:
    return OpenAI(base_url=OPENROUTER_BASE, api_key=OPENROUTER_API_KEY)


def _has_think_block(text: str) -> bool:
    lower = text.lower()
    return any(marker in lower for marker in THINK_MARKERS)


def _cache_key(png_bytes: bytes, model: str) -> str:
    extra = f"en|{TEMPERATURE}|{MAX_TOKENS}"
    return cache.make_key(png_bytes, "openrouter-qwen", model, extra)


def _write_page(out_dir: Path, page_num: int, markdown: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"page_{page_num:03d}.md"
    md_path.write_text(markdown, encoding="utf-8")
    return md_path


def _result_from_payload(page_num: int, payload: dict) -> PageOcrResult:
    usage = payload.get("usage") or {}
    return PageOcrResult(
        page_num=page_num,
        markdown=payload.get("markdown") or "",
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        total_tokens=usage.get("total_tokens"),
        used_no_reasoning=bool(payload.get("used_no_reasoning")),
    )


def _usage_dict(completion) -> dict:
    usage = getattr(completion, "usage", None)
    if usage is None:
        return {}
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def _message_text(completion) -> str:
    choices = completion.choices or []
    if not choices:
        return ""
    content = choices[0].message.content
    return content or ""


def _print_404(exc: APIStatusError) -> None:
    body = exc.body
    print("Qwen model 404. Stop this path. Do not switch model.")
    print(f"model={QWEN_MODEL}")
    print(f"status={exc.status_code}")
    print(f"body={body}")


def _create(
    client: OpenAI,
    *,
    b64: str,
    disable_reasoning: bool,
):
    kwargs: dict = {
        "model": QWEN_MODEL,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        ],
    }
    if disable_reasoning:
        kwargs["extra_body"] = {"reasoning": {"enabled": False}}
    limiter = get_limiter(
        "qwen",
        workers=max(1, QWEN_WORKERS),
        rate_per_min=QWEN_RATE_PER_MIN,
    )

    def _send():
        with limiter.slot():
            return client.chat.completions.create(**kwargs)

    try:
        return call_with_429_retry(_send)
    except APIStatusError as exc:
        if exc.status_code == 404:
            _print_404(exc)
            raise QwenModelMissing(str(exc.body)) from exc
        raise


def ocr_one_page(
    page: PagePng,
    *,
    out_dir: Path,
    use_cache: bool = True,
) -> PageOcrResult:
    png_bytes = page.png_path.read_bytes()
    key = _cache_key(png_bytes, QWEN_MODEL)
    cached = cache.get(key) if use_cache else None
    if cached is not None:
        payload = json.loads(cached.decode("utf-8"))
        result = _result_from_payload(page.page_num, payload)
        result.cache_hit = True
        result.md_path = _write_page(out_dir, page.page_num, result.markdown)
        print(f"page {page.page_num:03d} cache hit -> {result.md_path}")
        return result

    client = _client()
    b64 = base64.b64encode(png_bytes).decode("ascii")
    started = time.perf_counter()
    completion = _create(client, b64=b64, disable_reasoning=False)
    elapsed = time.perf_counter() - started
    text = _message_text(completion)
    used_no_reasoning = False
    if elapsed >= SLOW_SECONDS or _has_think_block(text):
        reason = "slow reply" if elapsed >= SLOW_SECONDS else "think block"
        print(f"page {page.page_num:03d} {reason}. Retry with reasoning disabled.")
        completion = _create(client, b64=b64, disable_reasoning=True)
        text = _message_text(completion)
        used_no_reasoning = True

    payload = {
        "markdown": text,
        "usage": _usage_dict(completion),
        "used_no_reasoning": used_no_reasoning,
        "model": QWEN_MODEL,
    }
    cache.put(key, json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    result = _result_from_payload(page.page_num, payload)
    result.md_path = _write_page(out_dir, page.page_num, result.markdown)
    print(f"page {page.page_num:03d} -> {result.md_path}  chars={len(result.markdown)}")
    return result


def ocr_pages_sequential(
    pages: list[PagePng],
    *,
    out_dir: Path | None = None,
    use_cache: bool = True,
) -> QwenRun:
    dest = out_dir or QWEN_SEQ_DIR
    run = QwenRun(workers=1, model=QWEN_MODEL)
    if not has_openrouter_key():
        print("Missing OPENROUTER_API_KEY. Skip Qwen.")
        run.skipped = True
        run.error = "missing_key"
        return run
    start = time.perf_counter()
    try:
        for page in pages:
            run.results.append(ocr_one_page(page, out_dir=dest, use_cache=use_cache))
    except QwenModelMissing:
        run.model_missing = True
        run.error = "404"
    run.wall_seconds = time.perf_counter() - start
    print(f"Qwen sequential wall {run.wall_seconds:.2f}s model={QWEN_MODEL}")
    return run


def ocr_pages_parallel(
    pages: list[PagePng],
    workers: int | None = None,
    *,
    out_dir: Path | None = None,
    use_cache: bool = False,
) -> QwenRun:
    dest = out_dir or QWEN_PAR_DIR
    n_workers = max(1, workers if workers is not None else QWEN_WORKERS)
    run = QwenRun(workers=n_workers, model=QWEN_MODEL)
    if not has_openrouter_key():
        print("Missing OPENROUTER_API_KEY. Skip Qwen.")
        run.skipped = True
        run.error = "missing_key"
        return run
    start = time.perf_counter()
    ordered: dict[int, PageOcrResult] = {}
    try:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(
                    ocr_one_page,
                    page,
                    out_dir=dest,
                    use_cache=use_cache,
                ): page.page_num
                for page in pages
            }
            for future in as_completed(futures):
                page_num = futures[future]
                try:
                    ordered[page_num] = future.result()
                except QwenModelMissing:
                    run.model_missing = True
                    run.error = "404"
                    for pending in futures:
                        pending.cancel()
                    break
    except QwenModelMissing:
        run.model_missing = True
        run.error = "404"
    run.results = [ordered[i] for i in sorted(ordered)]
    run.wall_seconds = time.perf_counter() - start
    print(f"Qwen parallel workers={n_workers} wall {run.wall_seconds:.2f}s model={QWEN_MODEL}")
    return run


def _parse_args(argv: list[str]) -> tuple[Path | None, str]:
    mode = "both"
    positional: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--seq":
            mode = "seq"
            i += 1
            continue
        if token == "--par":
            mode = "par"
            i += 1
            continue
        positional.append(token)
        i += 1
    if len(positional) != 1:
        return None, mode
    return Path(positional[0]), mode


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    pdf_path, mode = _parse_args(args)
    if pdf_path is None:
        print(
            "Usage: python -m src.qwen_ocr <english-file.pdf> [--seq|--par]",
            file=sys.stderr,
        )
        return 2
    print(
        f"Qwen uses English only. model={QWEN_MODEL} language={QWEN_LANGUAGE}. "
        "No Indic language prompt."
    )
    pages = shred_pdf(pdf_path)
    if mode in ("both", "seq"):
        print("Qwen sequential")
        seq = ocr_pages_sequential(pages)
        if seq.model_missing:
            return 1
    if mode in ("both", "par"):
        print("Qwen parallel")
        par = ocr_pages_parallel(pages, workers=QWEN_WORKERS)
        if par.model_missing:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
