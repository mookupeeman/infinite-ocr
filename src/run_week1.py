"""Week 1 run: shred, Sarvam, Bodhan seq/par, Qwen seq/par, then out/report.md."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from src.bodhan_ocr import (
    BODHAN_RATE_PER_MIN,
    BodhanRun,
    ocr_pages_parallel as bodhan_par,
)
from src.bodhan_ocr import ocr_pages_sequential as bodhan_seq
from src.compare import write_report
from src.config import (
    BODHAN_WORKERS,
    PAGES_QWEN_DIR,
    QWEN_WORKERS,
    SARVAM_OUT_DIR,
    choose_ocr_language,
)
from src.pdf_pages import shred_pdf
from src.qwen_ocr import QwenRun, ocr_pages_parallel as qwen_par
from src.qwen_ocr import ocr_pages_sequential as qwen_seq
from src.sarvam_digitise import DigitiseResult, digitise_pdf


def _parse_args(
    argv: list[str],
) -> tuple[Path | None, str | None, Path | None, Path | None, bool, bool]:
    language_flag: str | None = None
    qwen_pdf: Path | None = None
    report_path: Path | None = None
    bodhan_only = False
    no_cache = False
    positional: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token in ("-l", "--language"):
            if i + 1 >= len(argv):
                return None, None, None, None, False, False
            language_flag = argv[i + 1]
            i += 2
            continue
        if token.startswith("--language="):
            language_flag = token.split("=", 1)[1]
            i += 1
            continue
        if token == "--qwen-pdf":
            if i + 1 >= len(argv):
                return None, None, None, None, False, False
            qwen_pdf = Path(argv[i + 1])
            i += 2
            continue
        if token.startswith("--qwen-pdf="):
            qwen_pdf = Path(token.split("=", 1)[1])
            i += 1
            continue
        if token == "--report":
            if i + 1 >= len(argv):
                return None, None, None, None, False, False
            report_path = Path(argv[i + 1])
            i += 2
            continue
        if token.startswith("--report="):
            report_path = Path(token.split("=", 1)[1])
            i += 1
            continue
        if token == "--bodhan-only":
            bodhan_only = True
            i += 1
            continue
        if token == "--no-cache":
            no_cache = True
            i += 1
            continue
        positional.append(token)
        i += 1
    if len(positional) != 1:
        return None, language_flag, qwen_pdf, report_path, bodhan_only, no_cache
    return Path(positional[0]), language_flag, qwen_pdf, report_path, bodhan_only, no_cache


def _sarvam_from_disk(language: str) -> DigitiseResult:
    unzip_dir = SARVAM_OUT_DIR / "unzipped"
    result = DigitiseResult(
        skipped=False,
        cache_hit=True,
        status="disk",
        language=language,
        unzip_dir=unzip_dir if unzip_dir.is_dir() else None,
        wall_seconds=0.0,
    )
    if unzip_dir.is_dir():
        result.unzipped_files = [
            str(p.relative_to(unzip_dir))
            for p in sorted(unzip_dir.rglob("*"))
            if p.is_file()
        ]
        meta = sorted(unzip_dir.rglob("page_001.json"))
        if meta:
            import json

            try:
                payload = json.loads(meta[0].read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    result.page1_keys = list(payload.keys())
            except json.JSONDecodeError:
                pass
    else:
        result.skipped = True
        result.error = "no_local_sarvam"
    return result


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    pdf_path, language_flag, qwen_pdf, report_path, bodhan_only, no_cache = _parse_args(
        args
    )
    if pdf_path is None:
        print(
            "Usage: python -m src.run_week1 <file.pdf> [--language CODE] "
            "[--qwen-pdf english.pdf] [--report path/review.md] "
            "[--bodhan-only] [--no-cache]",
            file=sys.stderr,
        )
        return 2

    language = choose_ocr_language(language_flag)
    print(
        "Week 1 order: shred -> Sarvam -> Bodhan seq -> Bodhan par -> "
        "Qwen seq -> Qwen par -> report"
    )
    if bodhan_only:
        print(
            f"Bodhan-only refresh. Rate limit {BODHAN_RATE_PER_MIN:g} req/min. "
            f"no_cache={no_cache}"
        )

    shred_start = time.perf_counter()
    pages = shred_pdf(pdf_path)
    shred_seconds = time.perf_counter() - shred_start

    if bodhan_only:
        sarvam = _sarvam_from_disk(language)
    else:
        try:
            sarvam = digitise_pdf(pdf_path, language=language)
        except Exception as exc:
            print(f"Sarvam failed: {exc}")
            sarvam = DigitiseResult(error=str(exc), language=language)

    # Benchmark: sequential and parallel are both live (no cache) so the
    # speedup compares like with like. Both share one Bodhan limiter.
    print(
        f"Bodhan sequential (rate={BODHAN_RATE_PER_MIN:g}/min, "
        f"workers={BODHAN_WORKERS}, live)"
    )
    try:
        b_seq = bodhan_seq(pages, use_cache=False)
    except Exception as exc:
        print(f"Bodhan sequential failed: {exc}")
        b_seq = BodhanRun(error=str(exc))

    print(
        f"Bodhan parallel (rate={BODHAN_RATE_PER_MIN:g}/min, "
        f"workers={BODHAN_WORKERS}, live)"
    )
    try:
        b_par = bodhan_par(
            pages,
            workers=BODHAN_WORKERS,
            use_cache=False,
        )
    except Exception as exc:
        print(f"Bodhan parallel failed: {exc}")
        b_par = BodhanRun(error=str(exc), workers=BODHAN_WORKERS)

    qwen_pages = pages
    q_seq = QwenRun(skipped=True, error="not_run")
    q_par = QwenRun(skipped=True, error="not_run", workers=QWEN_WORKERS)
    run_qwen = not bodhan_only
    if bodhan_only:
        q_seq = QwenRun(skipped=True, error="bodhan_only")
        q_par = QwenRun(skipped=True, error="bodhan_only", workers=QWEN_WORKERS)
    elif qwen_pdf is not None:
        print(f"Qwen English PDF: {qwen_pdf}")
        qwen_pages = shred_pdf(qwen_pdf, out_dir=PAGES_QWEN_DIR)
    elif language not in ("en-IN", "en"):
        print(
            "Qwen is English only. Pass --qwen-pdf <english.pdf> to run Qwen. Skip Qwen."
        )
        q_seq = QwenRun(skipped=True, error="english_only")
        q_par = QwenRun(skipped=True, error="english_only", workers=QWEN_WORKERS)
        run_qwen = False

    if run_qwen:
        print("Qwen sequential")
        try:
            q_seq = qwen_seq(qwen_pages, use_cache=False)
        except Exception as exc:
            print(f"Qwen sequential failed: {exc}")
            q_seq = QwenRun(error=str(exc))
        print("Qwen parallel")
        try:
            q_par = qwen_par(qwen_pages, workers=QWEN_WORKERS, use_cache=False)
        except Exception as exc:
            print(f"Qwen parallel failed: {exc}")
            q_par = QwenRun(error=str(exc), workers=QWEN_WORKERS)

    write_report(
        pdf_path=pdf_path,
        language=language,
        pages=pages,
        shred_seconds=shred_seconds,
        sarvam=sarvam,
        bodhan_seq=b_seq,
        bodhan_par=b_par,
        qwen_seq=q_seq,
        qwen_par=q_par,
        qwen_pdf=qwen_pdf,
        qwen_pages=qwen_pages if run_qwen else [],
        dest=report_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
