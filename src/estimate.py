"""Cost and time of a run, for all three modes, before anything is sent. No API calls.

Preflight reads up to 40 pages, one at a random place in each stretch of the selection (all
of them for short documents), a few tens of milliseconds each:

    page class     born digital or scanned, and the kind of text layer (page_class.py)
    table page     born digital: PyMuPDF's table finder, or 60+ numbers on the page;
                   scanned: a ruled grid in a small render (unruled scanned tables are not
                   visible this way, so they only widen the upper end of the range)

The share found in the sample is scaled to the whole selection. Each mode's cost and time
then follow from measured coefficients (estimate_coefficients.json): engine prices and
speeds, open table cells per table page by page class, rupees per question per model, and
sheet latency. Every number is a range. Modes have no spending cap.
After a run, `record` appends estimate and actual to out/estimates.jsonl for refitting.
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


def mode_estimate(pre: Preflight, mode: modes.Mode, cached_pages: int = 0) -> dict:
    """Cost (₹) and time (minutes) ranges for one mode."""
    n = max(0, pre.pages - cached_pages)
    t_low, t_high = pre.table_range
    t_low, t_high = t_low * n / max(1, pre.pages), t_high * n / max(1, pre.pages)
    sarvam = n * COEF["sarvam_inr_per_page"] if mode.policy != "bodhan-only" else 0.0
    local = BODHAN_BACKEND == "local"
    b_price = 0.0 if local else COEF["bodhan_inr_per_page"]
    if mode.policy in ("both", "bodhan-only"):
        bodhan = (n * b_price,) * 2
        bodhan_pages = (n, n)
    else:
        bodhan = (t_low * b_price, t_high * b_price)
        bodhan_pages = (t_low, t_high)
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
    b_hi, b_lo = COEF["bodhan_local_pages_per_min"][::-1] if local else (COEF["bodhan_pages_per_min"],) * 2
    stage1 = (o_lo + max(n / s_hi, bodhan_pages[0] / b_hi), o_hi + max(n / s_lo, bodhan_pages[1] / b_lo))
    f_lo, f_hi = COEF["stage2_free_s_per_table_page"]
    l_lo, l_hi = COEF["sheet_latency_s"]
    per_sheet = mode.judge.per_sheet
    stage2 = ((t_low * f_lo + math.ceil(questions[0] / per_sheet) * l_lo / AGENT_MAX_INFLIGHT) / 60,
              (t_high * f_hi + math.ceil(questions[1] / per_sheet) * l_hi / AGENT_MAX_INFLIGHT) / 60)
    if not mode.image_check:
        stage2 = (t_low * f_lo / 60, t_high * f_hi / 60)
    return {
        "mode": mode.name,
        "pages": n,
        "table_pages": [round(t_low), round(t_high)],
        "sarvam_inr": round(sarvam, 2),
        "bodhan_inr": [round(bodhan[0], 2), round(bodhan[1], 2)],
        "checks_inr": [round(checks[0], 2), round(checks[1], 2)],
        "total_inr": [round(sarvam + bodhan[0] + checks[0], 2), round(sarvam + bodhan[1] + checks[1], 2)],
        "minutes": [round(stage1[0] + stage2[0], 1), round(stage1[1] + stage2[1], 1)],
    }


def estimate_all(pre: Preflight, cached_pages: int = 0) -> list[dict]:
    return [mode_estimate(pre, m, cached_pages) for m in modes.MODES.values()]


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
        ocr = _money(e["sarvam_inr"] + e["bodhan_inr"][0], e["sarvam_inr"] + e["bodhan_inr"][1])
        chk = "₹0" if e["checks_inr"][1] == 0 else _money(*e["checks_inr"])
        mark = " <-" if e["mode"] == chosen else ""
        rows.append(f"  {e['mode']:8s} {ocr:>24s} {chk:>18s} {_money(*e['total_inr']):>16s} {_minutes(*e['minutes']):>12s}{mark}")
    notes = [f"  note: {x}" for x in pre.notes]
    return "\n".join([head] + rows + notes + ["  (cache hits cost nothing; no spending cap unless --max-cost-inr is given)"])


def record(path: Path | None, *, document: str, mode: str, estimate: dict, actual: dict) -> None:
    """Append estimate vs actual (out/estimates.jsonl) so the coefficients can be refitted."""
    path = path or OUT_DIR / "estimates.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"t": round(time.time()), "document": document, "mode": mode,
                             "estimate": estimate, "actual": actual}, ensure_ascii=False) + "\n")
