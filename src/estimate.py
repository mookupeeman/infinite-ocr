"""Guess cost and time for all three modes before any API call. No network use.

These numbers are rough. Do not treat them as a bill.

Cost estimate procedure (estimate.preflight, then mode_estimate for each mode)
    Step 1 — Preflight. Open the PDF with PyMuPDF only. Pick up to 40 pages spread through
    the document (every page if the document is short). For each sampled page, record whether
    it is born-digital or scanned and the text-layer type. Mark a page as a likely table page
    if PyMuPDF finds a table, if the page has at least 60 number-like words, or if a small
    scan render shows a ruled grid. Scale those counts up to the full page selection and
    turn sample hits into a low-high range of table pages per class (vector/unicode, raster,
    and so on).

    Step 2 — OCR rupees. Read the mode and lead engine. From lanes routing (modes.policy_for),
    count how many pages Sarvam would bill and how many Bodhan would bill (lead reads every
    page; the other reads table pages only, except think reads both on every page). Multiply
    page counts by sarvam_inr_per_page and bodhan_inr_per_page from estimate_coefficients.json.
    Local Bodhan uses zero rupees per page in this math.

    Step 3 — Vision rupees (normal and think only). For each table-page class, multiply the
    low and high table-page counts by the low and high open-cell counts in
    questions_per_table_page. Price each cell with table_judge.question_cost for the models
    that mode uses (one model on digital tables, more on scanned ruled tables; think adds a
    third on scans). Add that to checks_inr low and high. Fast mode skips this step.

    Step 4 — Total. total_inr is Sarvam plus Bodhan plus checks, each as a low-high pair.
    Time minutes use the same preflight table counts plus pages-per-minute and sheet latency
    from the coefficients (parallel vision calls use AGENT_MAX_INFLIGHT).

"""

from __future__ import annotations

import io
import json
import math
import random
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image

from src import modes, table_geometry, table_judge
from src.config import AGENT_MAX_INFLIGHT, BODHAN_BACKEND, OUT_DIR
from src.page_class import classify
from src.triage import LEGACY_FONT_RE, body_tables, char_script

COEF = json.loads((Path(__file__).with_name("estimate_coefficients.json")).read_text(encoding="utf-8"))
SAMPLE = 40
NUMBER_RE = re.compile(r"^[(\-]?[\d०-९০-৯௦-௯౦-౯]+([.,][\d०-९০-৯௦-௯౦-౯]+)*%?\)?$")


@dataclass
class Preflight:
    """What a free look at up to 40 pages found: page classes, and how many pages have tables."""
    pages: int
    sampled: int
    classes: dict[str, float]  # page class -> pages in the selection (scaled from the sample)
    table_pages: dict[str, tuple[float, float]]  # page class -> (low, high) pages with a table
    seconds: float
    notes: list[str] = field(default_factory=list)

    @property
    def table_range(self) -> tuple[float, float]:
        return sum(v[0] for v in self.table_pages.values()), sum(v[1] for v in self.table_pages.values())


def _legacy(page: pymupdf.Page) -> bool:
    """The triage test for a legacy (non-Unicode) Indic font, without the rest of triage."""
    if any(LEGACY_FONT_RE.search(f[3] or "") for f in page.get_fonts()):
        return True
    scripts = Counter(s for s in map(char_script, page.get_text("text")) if s)
    letters = sum(scripts.values())
    return letters > 0 and scripts.get("Latin1", 0) > 0.2 * letters


def _scan_has_grid(page: pymupdf.Page) -> bool:
    pix = page.get_pixmap(dpi=60, colorspace=pymupdf.csGRAY, alpha=False)
    gray = np.asarray(Image.open(io.BytesIO(pix.tobytes("png"))).convert("L"), dtype=np.uint8)
    dark = gray <= min(table_geometry.otsu(gray), 170)
    v = table_geometry.vertical_rules(dark, min_frac=0.15)
    return len(v) >= 3 and len(table_geometry.horizontal_rules(dark, v)) >= 3


def _interval(hits: int, n: int, total: int) -> tuple[float, float]:
    """Pages with a table among `total`, from `hits` of `n` sampled (exact when all were read; else ~80% Wilson)."""
    if n == 0:
        return 0.0, 0.0
    if n >= total:
        return float(hits), float(hits)
    z, p = 1.28, hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half) * total, min(1.0, centre + half) * total


def preflight(doc: pymupdf.Document, selection: list[int], doc_type: str = "auto") -> Preflight:
    started = time.monotonic()
    n = len(selection)
    step = max(1.0, n / SAMPLE)
    rng = random.Random(n)  # one page from each stretch, at a random place: even spacing can alias (every 3rd page a table)
    sample = sorted({selection[min(n - 1, int(i * step + rng.random() * step))] for i in range(min(n, SAMPLE))})
    if n <= SAMPLE:
        sample = list(selection)
    seen: Counter = Counter()
    hits: Counter = Counter()
    unsure: Counter = Counter()
    for p in sample:
        page = doc[p - 1]
        cls = classify(page, _legacy(page), doc_type)
        key = "raster" if cls.scanned else f"vector/{cls.text_layer if cls.text_layer in ('unicode', 'legacy') else 'other'}"
        seen[key] += 1
        if cls.scanned:
            if _scan_has_grid(page):
                hits[key] += 1
            else:
                unsure[key] += 1  # an unruled scanned table cannot be seen here
            continue
        found = bool(body_tables(page))
        numbers = sum(1 for w in page.get_text("words") if NUMBER_RE.match(w[4]))
        if found or numbers >= COEF["dense_numbers_for_table"]:
            hits[key] += 1
    scale = n / max(1, len(sample))
    classes = {k: v * scale for k, v in seen.items()}
    tables = {}
    for k, pages in classes.items():
        total = round(pages)
        low, high = _interval(hits[k], seen[k], total)
        if unsure[k]:
            high = min(float(total), high + _interval(unsure[k], seen[k], total)[1])
        tables[k] = (low, high)
    notes = []
    if seen.get("raster"):
        notes.append("scanned pages: tables without ruling lines are not counted in the low end")
    return Preflight(n, len(sample), classes, tables, time.monotonic() - started, notes)


def _bodhan_speed() -> tuple[float, float]:
    """(low, high) pages per minute of the configured Bodhan."""
    if BODHAN_BACKEND != "local":
        return (COEF["bodhan_pages_per_min"],) * 2
    from src.indic_local import runtime

    return tuple(COEF["bodhan_vllm_pages_per_min" if runtime() == "vllm" else "bodhan_local_pages_per_min"])


def mode_estimate(pre: Preflight, mode: modes.Mode, cached_pages: int = 0, lead: str = "sarvam") -> dict:
    """Cost (₹) and time (minutes) ranges for one mode, with `lead` reading every page."""
    n = max(0, pre.pages - cached_pages)
    t_low, t_high = pre.table_range
    t_low, t_high = t_low * n / max(1, pre.pages), t_high * n / max(1, pre.pages)
    policy = modes.policy_for(mode, lead)
    every, tables, none = (n, n), (t_low, t_high), (0.0, 0.0)
    sarvam_pages = {"bodhan-primary": tables, "bodhan-only": none}.get(policy, every)
    bodhan_pages = {"sarvam-primary": tables, "sarvam-only": none}.get(policy, every)
    sarvam = tuple(p * COEF["sarvam_inr_per_page"] for p in sarvam_pages)
    b_price = 0.0 if BODHAN_BACKEND == "local" else COEF["bodhan_inr_per_page"]
    bodhan = tuple(p * b_price for p in bodhan_pages)
    checks, questions = [0.0, 0.0], [0.0, 0.0]
    if mode.image_check:
        prof = mode.judge
        for cls, (lo, hi) in pre.table_pages.items():
            q_lo, q_hi = COEF["questions_per_table_page"][cls]
            if cls == "raster":
                factors = [1.0, COEF["scanned_second_model_share"], COEF["think_third_model_share"]]
                models = [(m, f) for m, f in zip(prof.scanned_ruled, factors)]
            else:
                models = [(m, 1.0) for m in prof.born_digital]
            price_lo = sum(f * table_judge.question_cost(m, True, prof) for m, f in models)  # boxed cells where possible
            price_hi = sum(f * table_judge.question_cost(m, False, prof) for m, f in models)
            checks[0] += lo * q_lo * price_lo
            checks[1] += hi * q_hi * price_hi
            questions[0] += lo * q_lo * len(models)
            questions[1] += hi * q_hi * len(models)
    s_lo, s_hi = COEF["sarvam_pages_per_min"]
    o_lo, o_hi = COEF["stage1_overhead_min"]
    b_lo, b_hi = _bodhan_speed()
    stage1 = (o_lo + max(sarvam_pages[0] / s_hi, bodhan_pages[0] / b_hi),
              o_hi + max(sarvam_pages[1] / s_lo, bodhan_pages[1] / b_lo))
    f_lo, f_hi = COEF["stage2_free_s_per_table_page"]
    l_lo, l_hi = COEF["sheet_latency_s"]
    per_sheet = mode.judge.per_sheet
    stage2 = ((t_low * f_lo + math.ceil(questions[0] / per_sheet) * l_lo / AGENT_MAX_INFLIGHT) / 60,
              (t_high * f_hi + math.ceil(questions[1] / per_sheet) * l_hi / AGENT_MAX_INFLIGHT) / 60)
    if not mode.image_check:
        stage2 = (t_low * f_lo / 60, t_high * f_hi / 60)
    return {
        "mode": mode.name,
        "lead": lead,
        "pages": n,
        "table_pages": [round(t_low), round(t_high)],
        "sarvam_inr": [round(sarvam[0], 2), round(sarvam[1], 2)],
        "bodhan_inr": [round(bodhan[0], 2), round(bodhan[1], 2)],
        "checks_inr": [round(checks[0], 2), round(checks[1], 2)],
        "total_inr": [round(sarvam[0] + bodhan[0] + checks[0], 2), round(sarvam[1] + bodhan[1] + checks[1], 2)],
        "minutes": [round(stage1[0] + stage2[0], 1), round(stage1[1] + stage2[1], 1)],
    }


def estimate_all(pre: Preflight, cached_pages: int = 0, lead: str = "sarvam") -> list[dict]:
    return [mode_estimate(pre, m, cached_pages, lead) for m in modes.MODES.values()]


def _money(lo: float, hi: float) -> str:
    def fmt(x: float) -> str:
        return f"₹{x:,.2f}" if hi < 10 else f"₹{x:,.0f}"
    return fmt(lo) if fmt(lo) == fmt(hi) else f"{fmt(lo)}–{fmt(hi)[1:]}"


def _minutes(lo: float, hi: float) -> str:
    return f"{max(1, round(lo))}–{max(1, round(hi))} min" if hi >= 1.5 else "under 2 min"


def render(pre: Preflight, estimates: list[dict], chosen: str) -> str:
    """The table printed before 'Start?'."""
    cls = ", ".join(f"{round(v)} {k.replace('vector/', 'born-digital/').replace('raster', 'scanned')}" for k, v in sorted(pre.classes.items()))
    lo, hi = pre.table_range
    pages = f"{round(lo)}" if round(lo) == round(hi) else f"{round(lo)}–{round(hi)}"
    head = (f"Preflight ({pre.sampled} of {pre.pages} pages read, {pre.seconds:.1f} s): {cls}; "
            f"about {pages} pages with tables")
    rows = [f"  {'mode':8s} {'OCR (Sarvam + Bodhan)':>24s} {'table checks':>18s} {'total':>16s} {'time':>12s}"]
    for e in estimates:
        ocr = _money(e["sarvam_inr"][0] + e["bodhan_inr"][0], e["sarvam_inr"][1] + e["bodhan_inr"][1])
        chk = "₹0" if e["checks_inr"][1] == 0 else _money(*e["checks_inr"])
        mark = " <-" if e["mode"] == chosen else ""
        rows.append(f"  {e['mode']:8s} {ocr:>24s} {chk:>18s} {_money(*e['total_inr']):>16s} {_minutes(*e['minutes']):>12s}{mark}")
    notes = [f"  note: {x}" for x in pre.notes]
    return "\n".join([head] + rows + notes + ["  (cache hits cost nothing; no spending cap unless --max-cost-inr is given)"])


def record(path: Path | None, *, document: str, mode: str, estimate: dict, actual: dict) -> None:
    """Append one JSON line (default out/estimates.jsonl): pre-run estimate and actual rupees, minutes, table pages."""
    path = path or OUT_DIR / "estimates.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"t": round(time.time()), "document": document, "mode": mode,
                             "estimate": estimate, "actual": actual}, ensure_ascii=False) + "\n")
