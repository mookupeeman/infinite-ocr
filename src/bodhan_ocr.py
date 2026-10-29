"""Bodhan IndicOCR: one page image per HTTP call.

`bodhan_request` is the client used by every caller. It shares one limiter
per process (8 starts per ~60 s, measured 2026-10-10) and handles the API's
quirks: HTTP 200 with a `null` body for an undecodable image, and 502 when a
block overflows `max_tokens`.
"""

from __future__ import annotations

import base64
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pymupdf
import requests

from src import cache
from src.config import (
    BODHAN_API_KEY,
    BODHAN_BACKEND,
    BODHAN_MAX_INFLIGHT,
    BODHAN_MODEL,
    BODHAN_PAR_DIR,
    BODHAN_RATE_PER_MIN,
    BODHAN_SEQ_DIR,
    BODHAN_TABLE_FORMAT,
    BODHAN_WORKERS,
    choose_ocr_language,
    has_bodhan_key,
)
from src.pdf_pages import PagePng, shred_pdf
from src.rate_limit import (
    CallFailed,
    DegradeError,
    FatalError,
    Limiter,
    Outcome,
    Verdict,
    call_with_retry,
    classify_http,
)

BODHAN_URL = "https://api.bodhan.ai/v1/chat/completions"
MAX_TOKENS = 4096  # per block, not per page
WINDOW_S = 62.0  # server window is ~60 s; keep a margin
TIMEOUT = (10, 120)  # connect, read


class BodhanAuthError(RuntimeError):
    """401: stop the Bodhan path."""


_LIMITER: Limiter | None = None
_LIMITER_LOCK = threading.Lock()
_TLS = threading.local()


def bodhan_limiter() -> Limiter:
    """One limiter per process: every Bodhan call shares the same quota."""
    global _LIMITER
    with _LIMITER_LOCK:
        if _LIMITER is None:
            _LIMITER = Limiter(
                max_starts=max(1, int(BODHAN_RATE_PER_MIN)),
                window_s=WINDOW_S,
                max_inflight=max(1, BODHAN_MAX_INFLIGHT),
                min_spacing_s=60.0 / BODHAN_RATE_PER_MIN + 0.5,
            )
        return _LIMITER


def _session() -> requests.Session:
    session = getattr(_TLS, "session", None)
    if session is None:
        session = requests.Session()
        _TLS.session = session
    return session


def _error_detail(response: requests.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:300]
    if not isinstance(body, dict):
        return ""
    detail = body.get("detail")
    if detail is None and isinstance(body.get("error"), dict):
        detail = body["error"].get("detail") or body["error"].get("message")
    return str(detail or "")


def classify_bodhan(result: Any, exc: BaseException | None) -> Outcome:
    if exc is None and isinstance(result, requests.Response):
        response = result
        if response.status_code == 200:
            text = response.text.strip()
            if text in ("", "null"):
                return Outcome(Verdict.DEGRADE, "undecodable")
            try:
                payload = response.json()
            except ValueError:
                return Outcome(Verdict.RETRY, "bad_json")
            if not isinstance(payload, dict) or not payload.get("choices"):
                return Outcome(Verdict.DEGRADE, "undecodable")
            return Outcome(Verdict.OK)
        if response.status_code == 502:
            detail = _error_detail(response)
            if not detail:
                return Outcome(Verdict.RETRY, "http_502")
            if re.search(r"token|length|truncat", detail, re.I):
                return Outcome(Verdict.DEGRADE, "max_tokens")
            if re.search(r"decod|image", detail, re.I):
                return Outcome(Verdict.DEGRADE, "undecodable")
            return Outcome(Verdict.RETRY, f"http_502: {detail[:80]}")
    return classify_http(result, exc)


def to_jpeg(image: bytes, quality: int = 90) -> bytes:
    try:
        pix = pymupdf.Pixmap(image)
    except Exception as exc:  # noqa: BLE001 - local decode failed too
        raise FatalError(Outcome(Verdict.FATAL, "undecodable_locally"), 1) from exc
    if pix.alpha:
        pix = pymupdf.Pixmap(pix, 0)
    return pix.tobytes("jpeg", jpg_quality=quality)


def bodhan_request(
    image: bytes,
    mime: str = "image/png",
    *,
    table_format: str | None = None,
    max_tokens: int = MAX_TOKENS,
    limiter: Limiter | None = None,
    on_outcome: Any = None,
) -> dict:
    """OCR one page image. Returns the API payload plus an `_infinite` meta dict.

    Raises FatalError (bad key, no credit, unfixable image) or
    RetriesExhausted. Degrades instead of failing where it can: an
    undecodable PNG is re-sent as JPEG, and a block that overflows
    `max_tokens` with HTML tables is re-sent with markdown tables.
    """
    fmt = table_format or BODHAN_TABLE_FORMAT
    limiter = limiter or bodhan_limiter()
    notes: list[str] = []
    for _ in range(3):
        body = {
            "model": BODHAN_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,"
                                + base64.b64encode(image).decode("ascii")
                            },
                        }
                    ],
                }
            ],
            "table_format": fmt,
            "max_tokens": max_tokens,
        }

        def _send(body: dict = body) -> requests.Response:
            return _session().post(
                BODHAN_URL,
                headers={"Authorization": f"Bearer {BODHAN_API_KEY}"},
                json=body,
                timeout=TIMEOUT,
            )

        started = time.perf_counter()
        try:
            response = call_with_retry(
                _send,
                classify=classify_bodhan,
                limiter=limiter,
                label="bodhan",
                on_outcome=on_outcome,
            )
        except DegradeError as exc:
            reason = exc.outcome.reason
            if reason == "undecodable" and mime != "image/jpeg":
                image, mime = to_jpeg(image), "image/jpeg"
                notes.append("reencoded_jpeg")
                continue
            if reason == "max_tokens" and fmt == "html":
                fmt = "markdown"
                notes.append("markdown_tables")
                continue
            raise FatalError(exc.outcome, exc.attempts) from exc
        payload = response.json()
        payload["_infinite"] = {
            "table_format": fmt,
            "mime": mime,
            "degraded": notes,
            "latency_s": round(time.perf_counter() - started, 3),
            "ratelimit": {k: v for k, v in response.headers.items() if "ratelimit" in k.lower()},
        }
        return payload
    raise FatalError(Outcome(Verdict.FATAL, "degrade_loop", None), 3)


@dataclass
class PageOcrResult:
    """Result of reading one page with Bodhan (week-1 runners): markdown, blocks, token counts and any error."""
    page_num: int
    markdown: str = ""
    blocks: list[object] = field(default_factory=list)
    cache_hit: bool = False
    error: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    md_path: Path | None = None
    blocks_path: Path | None = None


@dataclass
class BodhanRun:
    """Result of a week-1 Bodhan run over several pages."""
    skipped: bool = False
    auth_failed: bool = False
    error: str | None = None
    results: list[PageOcrResult] = field(default_factory=list)
    wall_seconds: float = 0.0
    workers: int = 1


def _cache_key(png_bytes: bytes, table_format: str, max_tokens: int) -> str:
    if BODHAN_BACKEND == "local":
        from src.indic_local import LOCAL_CACHE_NS, active_model, local_cache_extra

        return cache.make_key(png_bytes, LOCAL_CACHE_NS, active_model(), local_cache_extra(table_format))
    extra = f"{table_format}|{max_tokens}"
    return cache.make_key(png_bytes, "indic-ocr", BODHAN_MODEL, extra)


def _write_page(out_dir: Path, page_num: int, markdown: str, blocks: list[object]) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"page_{page_num:03d}"
    md_path = out_dir / f"{stem}.md"
    blocks_path = out_dir / f"{stem}.blocks.json"
    md_path.write_text(markdown, encoding="utf-8")
    blocks_path.write_text(
        json.dumps(blocks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return md_path, blocks_path


def _result_from_payload(page_num: int, payload: dict) -> PageOcrResult:
    choices = payload.get("choices") or []
    markdown = ""
    if choices:
        message = choices[0].get("message") or {}
        markdown = message.get("content") or ""
    blocks = payload.get("blocks")
    if not isinstance(blocks, list):
        blocks = []
    usage = payload.get("usage") or {}
    return PageOcrResult(
        page_num=page_num,
        markdown=markdown,
        blocks=blocks,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        total_tokens=usage.get("total_tokens"),
    )


def ocr_one_page(
    page: PagePng,
    *,
    out_dir: Path,
    table_format: str | None = None,
    max_tokens: int = MAX_TOKENS,
    use_cache: bool = True,
) -> PageOcrResult:
    fmt = table_format or BODHAN_TABLE_FORMAT
    png_bytes = page.png_path.read_bytes()
    key = _cache_key(png_bytes, fmt, max_tokens)
    cached = cache.get(key) if use_cache else None
    if cached is not None:
        payload = json.loads(cached.decode("utf-8"))
        result = _result_from_payload(page.page_num, payload)
        result.cache_hit = True
        md_path, blocks_path = _write_page(out_dir, page.page_num, result.markdown, result.blocks)
        result.md_path = md_path
        result.blocks_path = blocks_path
        print(f"page {page.page_num:03d} cache hit -> {md_path}")
        return result

    try:
        if BODHAN_BACKEND == "local":
            from src.indic_local import local_ocr

            payload = local_ocr(png_bytes, "image/png", table_format=fmt)
        else:
            payload = bodhan_request(png_bytes, "image/png", table_format=fmt, max_tokens=max_tokens)
    except CallFailed as exc:
        reason = exc.outcome.reason
        if reason in ("http_401", "http_402", "http_403", "credit_exhausted"):
            print(f"Bodhan stopped: {reason}. Check the key and credit on console.bodhan.ai.")
            raise BodhanAuthError(reason) from exc
        print(f"page {page.page_num:03d} failed: {reason}")
        return PageOcrResult(page_num=page.page_num, error=reason)

    cache.put(key, json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    result = _result_from_payload(page.page_num, payload)
    md_path, blocks_path = _write_page(out_dir, page.page_num, result.markdown, result.blocks)
    result.md_path = md_path
    result.blocks_path = blocks_path
    print(f"page {page.page_num:03d} -> {md_path}  blocks={len(result.blocks)}")
    return result


def ocr_pages_sequential(
    pages: list[PagePng],
    *,
    out_dir: Path | None = None,
    use_cache: bool = True,
) -> BodhanRun:
    dest = out_dir or BODHAN_SEQ_DIR
    run = BodhanRun(workers=1)
    if BODHAN_BACKEND != "local" and not has_bodhan_key():
        print("Missing BODHAN_API_KEY. Skip Bodhan.")
        run.skipped = True
        run.error = "missing_key"
        return run
    start = time.perf_counter()
    try:
        for page in pages:
            run.results.append(ocr_one_page(page, out_dir=dest, use_cache=use_cache))
    except BodhanAuthError:
        run.auth_failed = True
        run.error = "401"
    run.wall_seconds = time.perf_counter() - start
    print(f"Bodhan sequential wall {run.wall_seconds:.2f}s")
    return run


def ocr_pages_parallel(
    pages: list[PagePng],
    workers: int | None = None,
    *,
    out_dir: Path | None = None,
    use_cache: bool = False,
) -> BodhanRun:
    dest = out_dir or BODHAN_PAR_DIR
    n_workers = max(1, workers if workers is not None else BODHAN_WORKERS)
    run = BodhanRun(workers=n_workers)
    if BODHAN_BACKEND != "local" and not has_bodhan_key():
        print("Missing BODHAN_API_KEY. Skip Bodhan.")
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
                except BodhanAuthError:
                    run.auth_failed = True
                    run.error = "401"
                    for pending in futures:
                        pending.cancel()
                    break
    except BodhanAuthError:
        run.auth_failed = True
        run.error = "401"
    run.results = [ordered[i] for i in sorted(ordered)]
    run.wall_seconds = time.perf_counter() - start
    print(f"Bodhan parallel workers={n_workers} wall {run.wall_seconds:.2f}s")
    return run


def _parse_args(argv: list[str]) -> tuple[Path | None, str | None, str]:
    language_flag: str | None = None
    mode = "both"
    positional: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token in ("-l", "--language"):
            if i + 1 >= len(argv):
                return None, None, mode
            language_flag = argv[i + 1]
            i += 2
            continue
        if token.startswith("--language="):
            language_flag = token.split("=", 1)[1]
            i += 1
            continue
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
        return None, language_flag, mode
    return Path(positional[0]), language_flag, mode


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    pdf_path, language_flag, mode = _parse_args(args)
    if pdf_path is None:
        print(
            "Usage: python -m src.bodhan_ocr <file.pdf> [--language CODE] [--seq|--par]",
            file=sys.stderr,
        )
        return 2
    language = choose_ocr_language(language_flag)
    print(
        f"Bodhan has no language field. Script is inferred from the image. "
        f"Recorded choice: {language}"
    )
    pages = shred_pdf(pdf_path)
    if mode in ("both", "seq"):
        print("Bodhan sequential")
        seq = ocr_pages_sequential(pages)
        if seq.auth_failed:
            return 1
    if mode in ("both", "par"):
        print("Bodhan parallel")
        par = ocr_pages_parallel(pages, workers=BODHAN_WORKERS)
        if par.auth_failed:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
