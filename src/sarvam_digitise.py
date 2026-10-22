"""Sarvam Vision through the Document AI API (`doc_ai.digitise`).

Job flow: submit -> poll -> fetch (download URL -> ZIP). We never pass
`model`: the server default for digitise is Sarvam Vision, and the API rejects
"sarvam-vision" as a model value.

Every SDK call goes through one Limiter (10 requests/min per account; polls
count) with the SDK's own retries switched off (`max_retries=0`). The SDK
would otherwise resend 429/408/409/5xx twice behind our back, and a resent
`digitise` can start a second paid job.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pymupdf
import requests
from sarvamai import SarvamAI

from src import cache
from src.config import (
    OCR_LANGUAGE,
    SARVAM_API_KEY,
    SARVAM_OUT_DIR,
    SARVAM_RATE_PER_MIN,
    choose_ocr_language,
    has_sarvam_key,
)
from src.rate_limit import (
    Limiter,
    Outcome,
    Verdict,
    call_with_retry,
    classify_http,
)

TERMINAL = {"completed", "partially_completed", "failed", "rejected"}
OK_STATUS = {"completed", "partially_completed"}
OUTPUT_FORMAT = "md"
NO_SDK_RETRY = {"max_retries": 0}
WINDOW_S = 62.0
MAX_WAIT_S = 15 * 60

_LIMITER: Limiter | None = None
_LIMITER_LOCK = threading.Lock()
_CLIENT: SarvamAI | None = None


@dataclass
class DigitiseResult:
    skipped: bool = False
    cache_hit: bool = False
    job_id: str | None = None
    status: str | None = None
    zip_path: Path | None = None
    unzip_dir: Path | None = None
    unzipped_files: list[str] = field(default_factory=list)
    page1_keys: list[str] = field(default_factory=list)
    error: str | None = None
    wall_seconds: float = 0.0
    language: str | None = None


def sarvam_limiter() -> Limiter:
    """One limiter per process for every Sarvam call (submit, poll, download URL)."""
    global _LIMITER
    with _LIMITER_LOCK:
        if _LIMITER is None:
            _LIMITER = Limiter(
                max_starts=max(1, int(SARVAM_RATE_PER_MIN)),
                window_s=WINDOW_S,
                max_inflight=1,
                min_spacing_s=3.0,
            )
        return _LIMITER


def _client() -> SarvamAI:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = SarvamAI(api_subscription_key=SARVAM_API_KEY)
    return _CLIENT


def _as_dict(obj: Any) -> dict:
    for name in ("model_dump", "dict"):
        method = getattr(obj, name, None)
        if callable(method):
            try:
                return method()
            except TypeError:
                continue
    return dict(getattr(obj, "__dict__", {}))


def classify_submit(result: Any, exc: BaseException | None) -> Outcome:
    """Retry a submit only when the job was surely not created (429).

    A 5xx or a timeout may have created the job already, so the caller decides.
    """
    outcome = classify_http(result, exc)
    if outcome.verdict is Verdict.RETRY:
        return Outcome(Verdict.FATAL, f"submit_uncertain:{outcome.reason}")
    return outcome


def submit(
    files: list[tuple[str, bytes]],
    language: str | None,
    *,
    output_format: str = OUTPUT_FORMAT,
    on_outcome: Any = None,
) -> str:
    """Start one digitise job. `files` are (name, PDF bytes); max 10 pages per PDF.

    `language=None` omits the field (Sarvam then reads in Indic mode). The
    API accepts and bills ANY string here, so codes are checked first.
    """
    from src.langid import SARVAM_LANGUAGES

    if language is not None and language not in SARVAM_LANGUAGES:
        raise ValueError(f"not a Sarvam language code: {language!r}")
    parts = [(name, data, "application/pdf") for name, data in files]
    extra = {} if language is None else {"language": language}
    job = call_with_retry(
        lambda: _client().doc_ai.digitise(
            file=parts,
            output_format=output_format,
            request_options=NO_SDK_RETRY,
            **extra,
        ),
        classify=classify_submit,
        limiter=sarvam_limiter(),
        label="sarvam submit",
        on_outcome=on_outcome,
    )
    return str(job.job_id)


def poll(job_id: str, *, on_outcome: Any = None) -> tuple[str, dict]:
    """One status call. Returns (lower-case status, full status dict)."""
    status = call_with_retry(
        lambda: _client().doc_ai.get_status(job_id=job_id, request_options=NO_SDK_RETRY),
        limiter=sarvam_limiter(),
        label="sarvam status",
        on_outcome=on_outcome,
    )
    data = _as_dict(status)
    return str(data.get("status") or getattr(status, "status", "")).lower(), data


def first_poll_delay(pages: int) -> float:
    """An 8-page job took ~4 s server-side; polling early wastes the 10/min budget."""
    return max(5.0, 1.0 * pages)


def wait_for(job_id: str, pages: int, *, max_wait_s: float = MAX_WAIT_S) -> tuple[str, dict]:
    """Poll with a first delay and back-off until a terminal status or timeout."""
    time.sleep(first_poll_delay(pages))
    started = time.monotonic()
    delay = 5.0
    while True:
        status, data = poll(job_id)
        usage = data.get("usage") or {}
        print(
            f"sarvam {job_id[:8]} status={status} "
            f"pages={usage.get('pages_processed', '?')}/{usage.get('pages_total', '?')}"
        )
        if status in TERMINAL:
            return status, data
        if time.monotonic() - started > max_wait_s:
            return "poll_timeout", data
        time.sleep(delay)
        delay = min(30.0, delay * 1.5)


def fetch(job_id: str, dest: Path, *, on_outcome: Any = None) -> Path:
    """Download the job's output ZIP to `dest` (atomic write)."""
    url_info = call_with_retry(
        lambda: _client().doc_ai.get_download_url(job_id=job_id, request_options=NO_SDK_RETRY),
        limiter=sarvam_limiter(),
        label="sarvam download-url",
        on_outcome=on_outcome,
    )
    # Presigned storage URL: not a Sarvam API call, so no limiter.
    response = call_with_retry(
        lambda: requests.get(url_info.url, timeout=(10, 120)),
        label="sarvam zip",
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
    tmp.write_bytes(response.content)
    os.replace(tmp, dest)
    return dest


def unzip(zip_path: Path, unzip_dir: Path) -> list[str]:
    """Extract into a clean folder (old files and folders are removed first)."""
    if unzip_dir.exists():
        shutil.rmtree(unzip_dir)
    unzip_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(unzip_dir)
    return [str(p.relative_to(unzip_dir)) for p in sorted(unzip_dir.rglob("*")) if p.is_file()]


def _print_tree(root: Path) -> None:
    print(f"unzip tree: {root}")
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        mark = "/" if path.is_dir() else ""
        print(f"  {rel}{mark}")


def _metadata_keys(unzip_dir: Path) -> list[str]:
    meta_dir = unzip_dir / "metadata"
    if not meta_dir.is_dir():
        json_files = sorted(unzip_dir.rglob("page_*.json"))
    else:
        json_files = sorted(meta_dir.glob("page_*.json"))
    if not json_files:
        json_files = sorted(unzip_dir.rglob("*.json"))
        json_files = [p for p in json_files if p.name != "manifest.json"]
    if not json_files:
        print("no metadata/page_NNN.json files")
        return []
    keys_out: list[str] = []
    for path in json_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"{path}: not valid JSON ({exc})")
            continue
        if isinstance(payload, dict):
            keys = list(payload.keys())
        else:
            keys = [type(payload).__name__]
        print(f"raw keys {path.name}: {keys}")
        if not keys_out:
            keys_out = keys
    return keys_out


def _page_count(pdf_path: Path) -> int:
    with pymupdf.open(pdf_path) as doc:
        return doc.page_count


def digitise_pdf(pdf_path: Path, language: str | None = None) -> DigitiseResult:
    """Week 1 path: one job on a whole PDF of at most 10 pages."""
    pdf_path = Path(pdf_path)
    lang = language or OCR_LANGUAGE
    result = DigitiseResult(language=lang)
    started = time.perf_counter()

    if not has_sarvam_key():
        print("Missing SARVAM_API_KEY. Skip Digitise.")
        result.skipped = True
        result.error = "missing_key"
        result.wall_seconds = time.perf_counter() - started
        return result

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pdf_bytes = pdf_path.read_bytes()
    extra = f"{lang}|{OUTPUT_FORMAT}"
    key = cache.make_key(pdf_bytes, "sarvam-digitise", "sarvam-vision", extra)
    zip_path = SARVAM_OUT_DIR / "digitise.zip"
    unzip_dir = SARVAM_OUT_DIR / "unzipped"
    SARVAM_OUT_DIR.mkdir(parents=True, exist_ok=True)

    cached = cache.get(key)
    if cached is not None:
        print(f"cache hit {key[:12]}…")
        zip_path.write_bytes(cached)
        result.cache_hit = True
        result.status = "cache"
    else:
        print(f"cache miss {key[:12]}…")
        pages = _page_count(pdf_path)
        job_id = submit([(pdf_path.name, pdf_bytes)], lang)
        result.job_id = job_id
        print(f"job created: {job_id} | language: {lang} | pages: {pages}")
        status, _ = wait_for(job_id, pages)
        result.status = status
        if status not in OK_STATUS:
            result.error = status
            print(f"Digitise ended with status={status}. No download.")
            result.wall_seconds = time.perf_counter() - started
            return result
        fetch(job_id, zip_path)
        cache.put(key, zip_path.read_bytes())

    files = unzip(zip_path, unzip_dir)
    _print_tree(unzip_dir)
    result.zip_path = zip_path
    result.unzip_dir = unzip_dir
    result.unzipped_files = files
    result.page1_keys = _metadata_keys(unzip_dir)
    result.wall_seconds = time.perf_counter() - started
    return result


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    language_flag: str | None = None
    positional: list[str] = []
    i = 0
    while i < len(args):
        token = args[i]
        if token in ("-l", "--language"):
            if i + 1 >= len(args):
                print("Usage: python -m src.sarvam_digitise <file.pdf> [--language CODE]", file=sys.stderr)
                return 2
            language_flag = args[i + 1]
            i += 2
            continue
        if token.startswith("--language="):
            language_flag = token.split("=", 1)[1]
            i += 1
            continue
        positional.append(token)
        i += 1
    if len(positional) != 1:
        print("Usage: python -m src.sarvam_digitise <file.pdf> [--language CODE]", file=sys.stderr)
        return 2
    language = choose_ocr_language(language_flag)
    digitise_pdf(Path(positional[0]), language=language)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
