"""Build out/report.md from Week 1 backend results. No CER. No plots."""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.bodhan_ocr import BodhanRun
from src.config import BODHAN_MODEL, BODHAN_PAR_DIR, BODHAN_SEQ_DIR, OUT_DIR, QWEN_MODEL
from src.pdf_pages import PagePng
from src.qwen_ocr import QwenRun
from src.sarvam_digitise import DigitiseResult

from src.config import BODHAN_RATE_PER_MIN

DEVANAGARI = re.compile(r"[\u0900-\u097F]")
TAMIL = re.compile(r"[\u0B80-\u0BFF]")
LATIN = re.compile(r"[A-Za-z]")


def first_n_lines(text: str, n: int = 30) -> str:
    lines = text.splitlines()
    return "\n".join(lines[:n]) if lines else "(empty)"


def _read_text(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def speedup(seq_s: float, par_s: float) -> str:
    if seq_s <= 0 or par_s <= 0:
        return "n/a"
    return f"{seq_s / par_s:.2f}x"


def token_sum(run: BodhanRun | QwenRun, field: str) -> int | None:
    values = [getattr(item, field) for item in run.results if getattr(item, field) is not None]
    if not values:
        return None
    return int(sum(values))


def script_note(text: str) -> str:
    parts: list[str] = []
    if TAMIL.search(text):
        parts.append("Tamil")
    if DEVANAGARI.search(text):
        parts.append("Devanagari")
    if LATIN.search(text):
        parts.append("Latin")
    if not parts:
        return "no Tamil, Devanagari, or Latin letters in the sample"
    return ", ".join(parts)


def has_table_markup(text: str) -> bool:
    return "<table" in text.lower() or bool(re.search(r"^\|.+\|$", text, re.M))


def inspect_bodhan_blocks(blocks_path: Path | None) -> tuple[bool, bool, list[str]]:
    if blocks_path is None or not blocks_path.is_file():
        return False, False, []
    try:
        payload = json.loads(blocks_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False, False, []
    if not isinstance(payload, list):
        return False, False, []
    has_bbox = False
    has_label = False
    labels: list[str] = []
    for block in payload:
        if not isinstance(block, dict):
            continue
        if "bbox_xyxy" in block:
            has_bbox = True
        label = block.get("label")
        if isinstance(label, str) and label:
            has_label = True
            labels.append(label)
    return has_bbox, has_label, labels


def sarvam_page_text(unzip_dir: Path | None, page_num: int) -> str:
    if unzip_dir is None or not unzip_dir.is_dir():
        return ""
    json_files = sorted(unzip_dir.rglob(f"page_{page_num:03d}.json"))
    if json_files:
        try:
            payload = json.loads(json_files[0].read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        blocks = payload.get("blocks") if isinstance(payload, dict) else None
        if isinstance(blocks, list):
            chunks: list[str] = []
            for block in blocks:
                if isinstance(block, dict) and block.get("text"):
                    chunks.append(str(block["text"]))
            if chunks:
                return "\n\n".join(chunks)
    md_files = sorted(unzip_dir.rglob("*.md"))
    if md_files and page_num == 1:
        return md_files[0].read_text(encoding="utf-8")
    return ""


def write_report(
    *,
    pdf_path: Path,
    language: str,
    pages: list[PagePng],
    shred_seconds: float,
    sarvam: DigitiseResult,
    bodhan_seq: BodhanRun,
    bodhan_par: BodhanRun,
    qwen_seq: QwenRun,
    qwen_par: QwenRun,
    qwen_pdf: Path | None,
    qwen_pages: list[PagePng],
    dest: Path | None = None,
) -> Path:
    dest = Path(dest) if dest is not None else (OUT_DIR / "report.md")
    dest.parent.mkdir(parents=True, exist_ok=True)
    png_lines = []
    for page in pages:
        size = page.png_path.stat().st_size if page.png_path.is_file() else 0
        png_lines.append(
            f"- page {page.page_num:03d}: {page.width}x{page.height}, {size} bytes"
        )

    p1_blocks = BODHAN_SEQ_DIR / "page_001.blocks.json"
    if not p1_blocks.is_file():
        p1_blocks = BODHAN_PAR_DIR / "page_001.blocks.json"
    has_bbox, has_label, labels = inspect_bodhan_blocks(p1_blocks)

    sarvam_sample = sarvam_page_text(sarvam.unzip_dir, 1)
    bodhan_sample = _read_text(BODHAN_SEQ_DIR / "page_001.md") or _read_text(
        BODHAN_PAR_DIR / "page_001.md"
    )
    qwen_sample = _read_text(Path(qwen_seq.results[0].md_path) if qwen_seq.results and qwen_seq.results[0].md_path else Path())
    if not qwen_sample:
        qwen_sample = _read_text(OUT_DIR / "qwen_seq" / "page_001.md")

    notes: list[str] = []
    if has_table_markup(sarvam_sample) or has_table_markup(bodhan_sample) or has_table_markup(qwen_sample):
        notes.append("Tables: at least one backend emitted table markup (HTML or Markdown pipes).")
    else:
        notes.append("Tables: no HTML table or Markdown pipe table in the page-1 samples.")
    notes.append(f"Script in Sarvam page 1: {script_note(sarvam_sample)}.")
    notes.append(f"Script in Bodhan page 1: {script_note(bodhan_sample)}.")
    notes.append(f"Script in Qwen page 1: {script_note(qwen_sample)}.")
    notes.append(
        "Missing lines: this report does not compute CER. Compare the 30-line samples by eye."
    )
    notes.append(
        "Parallel vs sequential: on 2 or 3 pages, parallel often does not beat sequential. "
        f"Bodhan allows {BODHAN_RATE_PER_MIN:g} requests per ~60 s, so above that many pages "
        "parallel workers wait on the shared limiter and the speedup is rate-bound."
    )

    qwen_note = "same PDF as Sarvam/Bodhan"
    if qwen_pdf is not None and qwen_pdf.resolve() != pdf_path.resolve():
        qwen_note = f"separate English PDF `{qwen_pdf.name}` ({len(qwen_pages)} pages)"

    lines: list[str] = [
        "# Infinite-OCR Week 1 report",
        "",
        f"- PDF: `{pdf_path.name}`",
        f"- Page count (shred): {len(pages)}",
        f"- Language (prompt / Digitise): `{language}`",
        f"- Sarvam: one Digitise job, `output_format=md`",
        f"- Bodhan model: `{BODHAN_MODEL}`, workers={bodhan_par.workers}, "
        f"rate_limit={BODHAN_RATE_PER_MIN:g} req/min",
        f"- Qwen model: `{QWEN_MODEL}`, workers={qwen_par.workers}, {qwen_note}",
        "",
        "## Shred",
        "",
        f"- Wall: {shred_seconds:.2f} s",
        *png_lines,
        "",
        "## Sarvam Digitise",
        "",
    ]
    if sarvam.skipped:
        lines.append(f"- Skipped ({sarvam.error}).")
    else:
        lines.extend(
            [
                f"- Cache: {'hit' if sarvam.cache_hit else 'miss'}",
                f"- Job id: {sarvam.job_id or 'n/a (cache)'}",
                f"- Status: {sarvam.status}",
                f"- Seconds: {sarvam.wall_seconds:.2f}",
                f"- Unzipped files: {len(sarvam.unzipped_files)}",
            ]
        )
        for name in sarvam.unzipped_files:
            lines.append(f"  - `{name}`")
        lines.append(f"- JSON keys on page 1: {sarvam.page1_keys}")
        if sarvam.error:
            lines.append(f"- Error: {sarvam.error}")
    lines.extend(["", "## Bodhan IndicOCR", ""])
    if bodhan_seq.skipped:
        lines.append(f"- Skipped ({bodhan_seq.error}).")
    else:
        lines.extend(
            [
                f"- Sequential wall: {bodhan_seq.wall_seconds:.2f} s",
                f"- Parallel wall: {bodhan_par.wall_seconds:.2f} s (workers={bodhan_par.workers})",
                f"- Speedup: {speedup(bodhan_seq.wall_seconds, bodhan_par.wall_seconds)}",
                f"- Sequential tokens (sum): prompt={token_sum(bodhan_seq, 'prompt_tokens')} "
                f"completion={token_sum(bodhan_seq, 'completion_tokens')} "
                f"total={token_sum(bodhan_seq, 'total_tokens')}",
                f"- Parallel tokens (sum): prompt={token_sum(bodhan_par, 'prompt_tokens')} "
                f"completion={token_sum(bodhan_par, 'completion_tokens')} "
                f"total={token_sum(bodhan_par, 'total_tokens')}",
            ]
        )
        if bodhan_seq.auth_failed or bodhan_par.auth_failed:
            lines.append("- Stopped: Bodhan API key is bad (401).")
        seq_err = [r.page_num for r in bodhan_seq.results if r.error]
        par_err = [r.page_num for r in bodhan_par.results if r.error]
        if seq_err:
            lines.append(f"- Sequential page errors: {seq_err}")
        if par_err:
            lines.append(f"- Parallel page errors: {par_err}")
    lines.extend(["", "## OpenRouter Qwen", ""])
    if qwen_seq.skipped:
        lines.append(f"- Skipped ({qwen_seq.error}).")
    else:
        lines.extend(
            [
                f"- Sequential wall: {qwen_seq.wall_seconds:.2f} s",
                f"- Parallel wall: {qwen_par.wall_seconds:.2f} s (workers={qwen_par.workers})",
                f"- Speedup: {speedup(qwen_seq.wall_seconds, qwen_par.wall_seconds)}",
                f"- Sequential tokens (sum): prompt={token_sum(qwen_seq, 'prompt_tokens')} "
                f"completion={token_sum(qwen_seq, 'completion_tokens')} "
                f"total={token_sum(qwen_seq, 'total_tokens')}",
                f"- Parallel tokens (sum): prompt={token_sum(qwen_par, 'prompt_tokens')} "
                f"completion={token_sum(qwen_par, 'completion_tokens')} "
                f"total={token_sum(qwen_par, 'total_tokens')}",
            ]
        )
        if qwen_seq.model_missing or qwen_par.model_missing:
            lines.append("- Stopped: model 404. Slug was not changed.")
    lines.extend(
        [
            "",
            "## Bodhan blocks.json (page 1)",
            "",
            f"- File: `{p1_blocks}`" if p1_blocks.is_file() else "- No page_001.blocks.json",
            f"- Has `bbox_xyxy`: {has_bbox}",
            f"- Has `label`: {has_label}",
            f"- Labels on page 1: {labels}",
            "",
            "## Per-page samples (first 30 lines)",
            "",
        ]
    )

    page_nums = sorted({p.page_num for p in pages} | {p.page_num for p in qwen_pages})
    for page_num in page_nums:
        sarvam_md = sarvam_page_text(sarvam.unzip_dir, page_num)
        bodhan_md = _read_text(BODHAN_SEQ_DIR / f"page_{page_num:03d}.md") or _read_text(
            BODHAN_PAR_DIR / f"page_{page_num:03d}.md"
        )
        qwen_md = _read_text(OUT_DIR / "qwen_seq" / f"page_{page_num:03d}.md") or _read_text(
            OUT_DIR / "qwen_par" / f"page_{page_num:03d}.md"
        )
        lines.extend(
            [
                f"### Page {page_num:03d}",
                "",
                "Sarvam:",
                "",
                "```",
                first_n_lines(sarvam_md) if sarvam_md else "(no Sarvam text for this page)",
                "```",
                "",
                "Bodhan:",
                "",
                "```",
                first_n_lines(bodhan_md) if bodhan_md else "(no Bodhan markdown for this page)",
                "```",
                "",
                "Qwen:",
                "",
                "```",
                first_n_lines(qwen_md) if qwen_md else "(no Qwen markdown for this page)",
                "```",
                "",
            ]
        )

    lines.extend(["## Notes", ""])
    for note in notes:
        lines.append(f"- {note}")
    lines.append("")

    dest.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {dest}")
    return dest
