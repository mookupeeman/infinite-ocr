"""The image check for `standard` table checking: settles open cells from cut-outs of the page.

The rule was chosen on a pilot of 395 real questions with known answers (bench/pilot.py,
2026-10-21); "harm" there means a correct default reading changed to a wrong one.

    born-digital    one model (GPT-5.6 Luna); its reading is accepted only if it equals
                    one engine's reading. Pilot: 24/26 right, harm 0.
    scanned, ruled  two models from different families (Gemini 3.1 Flash-lite and GPT-5.6
                    Luna) must read the same value, and it must equal an engine's reading.
                    Pilot: 45/51 right, harm 0.
    scanned, unruled  no model: row strips are guesses there and no reader was reliable
                    (pilot: about 40% right, with harm). Cells stay open for review.

The models, the vote and the sheet layout come from the run's mode (modes.JudgeProfile):
Think adds a third model on scanned ruled tables and accepts 2 of 3 (pilot: 46 of 56 decided right,
harm 0). Models are asked cheapest first; a later one only sees cells it can still decide.

A model never writes a value no engine produced, and model "confidence" is ignored (it was
"high" on blurred digits it got wrong). Audits (cells the engines agree on but that look
risky) are read blind by one model and can only flag a cell, never change it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pymupdf

from collections import Counter

from src.agent import Agent
from src import modes
from src.config import TABLE_JUDGE_SINGLE
from src.modes import JudgeProfile
from src.table_cutouts import Cutter, sheets
from src.table_qa import NUMERAL_SCRIPTS, QCell, QTable, apply, canon, refresh

# gemini-3.8-flash refuses requests without reasoning ("Reasoning is mandatory"): the lowest effort it takes
EXTRA = {"openai/": {"reasoning": {"effort": "minimal"}}, "google/gemini-3.8": {"reasoning": {"effort": "low"}}}
# Rupees per question per model, measured (pilot of 2026-10-22 and the Union budget runs). Row strips cost
# about the same at 12 or 24 a sheet on dense multi-line rows; cell-only snippets cost about a third.
INR_PER_QUESTION = {"rows": 0.015, "cells": 0.005}
MODEL_PRICE = {"gemini-3.1-flash-lite": 0.8, "gpt-5.6-luna": 1.0, "gemini-3.5-flash-lite": 1.1, "gemini-3.8-flash": 2.3}  # relative to Luna
DEFAULT_PROFILE = modes.get(modes.DEFAULT_MODE).judge


def question_cost(model: str, boxed: bool, profile: JudgeProfile) -> float:
    kind = "cells" if boxed and profile.cell_only else "rows"
    return INR_PER_QUESTION[kind] * MODEL_PRICE.get(model.split("/")[-1], 1.0)
# Cells no model may settle (they stay open for review):
#   rare numerals / look-alikes - every model tried read Tamil numerals as letters (pilot: <= 3 of 45 right);
#   layer_has_neither - a proven text layer contains none of the readings, so the value is probably a third
#   one; a model choosing between two wrong readings was the pilot's main born-digital error.
NOT_FOR_MODELS = {"rare_numerals", "lookalike_numeral", "layer_has_neither"}


def for_models(c: QCell) -> bool:
    if set(c.risk) & NOT_FOR_MODELS:
        return False
    return not any(z <= ord(ch) <= z + 9 for t in c.reads.values() if t for ch in t for z in NUMERAL_SCRIPTS)


def _extra(model: str) -> dict | None:
    return next((v for k, v in EXTRA.items() if model.startswith(k)), None)


def _short(model: str) -> str:
    return model.split("/")[-1]


@dataclass
class JudgeReport:
    asked: int = 0
    decided: int = 0
    cost_inr: float = 0.0
    route: str = ""
    skipped: str | None = None
    calls: int = 0
    audits: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def route(cutter: Cutter, scanned: bool, profile: JudgeProfile = DEFAULT_PROFILE) -> tuple[str, list[str]]:
    if not scanned:
        return "born-digital", list(profile.born_digital)
    if cutter.exact_cols:
        return "scanned-ruled", list(profile.scanned_ruled)
    return "scanned-unruled", []


def boxed(geometry, table) -> bool:
    """Exact cell boxes exist (what Cutter.exact_cols decides): the ruling lines have the table's shape."""
    return geometry is not None and geometry.source == "rules" and (geometry.rows, geometry.cols) == (table.n_rows, table.n_cols)


def estimate_inr(exact: bool, scanned: bool, n_cells: int, profile: JudgeProfile = DEFAULT_PROFILE) -> float:
    """Upper bound for settling n_cells of one table (every model asked about every cell)."""
    models = list(profile.born_digital) if not scanned else list(profile.scanned_ruled) if exact else []
    return sum(n_cells * question_cost(m, exact, profile) for m in models)


def _matches_engine(c: QCell, reading: str) -> bool:
    return any(t is not None and canon(t) == canon(reading) for t in c.reads.values())


def _where(qt: QTable, c: QCell, boxed: bool) -> str:
    if boxed:
        return "inside the red box"
    rows = qt.table.rows_text()
    header = " / ".join(dict.fromkeys(canon(rows[h][c.c]) for h in range(qt.table.header_rows) if rows[h][c.c]))
    others = [canon(x) for j, x in enumerate(rows[c.r]) if j != c.c and x.strip()][:3]
    return f"column headed '{header[:60]}', in the row that also contains: " + " | ".join(o[:30] for o in others)


def _ask(agent: Agent, cutter: Cutter, qt: QTable, cells: list[QCell], style: str, model: str, report: JudgeReport,
         profile: JudgeProfile = DEFAULT_PROFILE) -> dict[str, str]:
    """{question id: reading} from one model over the sheets for these cells."""
    ids = {f"Q{i + 1}": c for i, c in enumerate(cells)}
    readings: dict[str, str] = {}
    cells_xy = [(qid, c.r, c.c, c.rowspan, c.colspan) for qid, c in ids.items()]
    for sheet in sheets(cutter, cells_xy, profile.per_sheet, profile.cell_only):
        qs = []
        for qid, boxed in zip(sheet.qids, sheet.exact):
            c = ids[qid]
            item = {"id": qid, "where": _where(qt, c, boxed)}
            if style == "dispute":
                item["candidates"] = {chr(65 + k): t for k, t in enumerate(dict.fromkeys(t for t in c.reads.values() if t is not None))}
            qs.append(item)
        answer = agent.judge_cells(sheet.png, qs, style, model=model, extra=_extra(model))
        report.calls += 1
        if answer:
            report.cost_inr += float(answer.get("_cost_inr") or 0.0)
            for a in answer.get("answers", []):
                if a.get("id") in ids:
                    readings[a["id"]] = a.get("reading", "")
    return {qid: readings[qid] for qid in readings}


def _agreeing(c: QCell, said: dict[str, str]) -> tuple[str | None, list[str]]:
    """The engine reading most models gave for a cell, and which models gave it."""
    votes = Counter(canon(r) for r in said.values() if _matches_engine(c, r))
    if not votes:
        return None, []
    top = votes.most_common(1)[0][0]
    return top, [m for m, r in said.items() if canon(r) == top]


def settle(agent: Agent, doc: pymupdf.Document, page: int, rotate: int, bbox: tuple, qt: QTable, scanned: bool,
           budget_inr: float | None = None, geometry=None, profile: JudgeProfile = DEFAULT_PROFILE) -> JudgeReport:
    """Ask the image check about this table's open cells and apply the rule."""
    report = JudgeReport()
    open_cells = [c for c in qt.cells if c.open and for_models(c)]
    if not open_cells:
        return report
    cutter = Cutter(doc, page, rotate, bbox, qt.table, geometry)
    report.route, models = route(cutter, scanned, profile)
    if not models:
        report.skipped = "scanned table without ruling lines: left for review"
        return report
    if budget_inr is not None and estimate_inr(cutter.exact_cols, scanned, len(open_cells), profile) > budget_inr:
        report.skipped = "over the table-checking budget: left for review"
        return report
    report.asked = len(open_cells)
    need = min(profile.vote or len(models), len(models))
    # cheapest model first; a later model is only asked about cells it can still help decide (the cell is not
    # settled yet, and enough models are left to reach `need`), so the decisions equal asking every model
    said: list[dict[str, str]] = [{} for _ in open_cells]
    for j, m in enumerate(models):
        left = len(models) - j
        todo = [i for i, c in enumerate(open_cells)
                if len(_agreeing(c, said[i])[1]) < need and len(_agreeing(c, said[i])[1]) + left >= need]
        if not todo:
            break
        got = _ask(agent, cutter, qt, [open_cells[i] for i in todo], "dispute", m, report, profile)
        for k, i in enumerate(todo):
            if f"Q{k + 1}" in got:
                said[i][m] = got[f"Q{k + 1}"]
    for i, c in enumerate(open_cells):
        reading, agree = _agreeing(c, said[i])
        if reading is None or len(agree) < need:
            continue  # the models disagree, one did not answer, or none read an engine's value
        match = next(t for t in c.reads.values() if t is not None and canon(t) == reading)
        if not match.strip() and any((t or "").strip() for t in c.reads.values()):
            continue  # never delete printed text on a model's word: an empty reading is also what a missed box looks like
        apply(qt, c, match, "image:" + "+".join(_short(m) for m in models if m in agree))
        report.decided += 1
    refresh(qt)
    return report


def audit(agent: Agent, doc: pymupdf.Document, page: int, rotate: int, bbox: tuple, qt: QTable, cells: list[QCell],
          scanned: bool, geometry=None, profile: JudgeProfile = DEFAULT_PROFILE) -> JudgeReport:
    """Blind reading of agreed-but-risky cells by one model. A different reading only flags the cell."""
    report = JudgeReport()
    cutter = Cutter(doc, page, rotate, bbox, qt.table, geometry)
    report.route, models = route(cutter, scanned, profile)
    if not models or not cells:
        report.skipped = "no audit on scanned tables without ruling lines" if not models else None
        return report
    model = TABLE_JUDGE_SINGLE
    readings = _ask(agent, cutter, qt, cells, "audit", model, report, profile)
    for i, c in enumerate(cells):
        got = readings.get(f"Q{i + 1}")
        if got is None:
            continue
        same = canon(got) == canon(c.value)
        c.risk.append("audit_confirmed" if same else "audit_disagrees")
        report.audits.append({"page": page, "r": c.r, "c": c.c, "value": c.value, "model_read": got, "same": same})
    refresh(qt)
    return report


SEAM_SCHEMA = {
    "type": "object",
    "properties": {
        "continues": {"type": "boolean"},
        "repeated_header_rows": {"type": "integer", "minimum": 0, "maximum": 6},
        "first_row_continues_previous_row": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["continues", "repeated_header_rows", "first_row_continues_previous_row", "reason"],
    "additionalProperties": False,
}


def seam(agent: Agent, doc_a: pymupdf.Document, a: tuple, qt_a_table, doc_b: pymupdf.Document, b: tuple, qt_b_table,
         geo_a=None, geo_b=None) -> dict | None:
    """One picture of a page break: the end of table A above the start of table B. a/b = (page, rotate, bbox)."""
    import base64
    import io
    import json

    from PIL import Image, ImageDraw

    ca, cb = Cutter(doc_a, a[0], a[1], a[2], qt_a_table, geo_a), Cutter(doc_b, b[0], b[1], b[2], qt_b_table, geo_b)
    parts = []
    head = ca.header()
    if head is not None:
        parts.append(head)
    n_a = qt_a_table.n_rows
    parts.append(ca.strip(max(qt_a_table.header_rows, n_a - 3), n_a, None, label="END").image)
    parts.append(cb.strip(0, min(qt_b_table.n_rows, 4), None, label="NEXT").image)
    width = max(p.width for p in parts)
    sheet = Image.new("RGB", (width, sum(p.height + 14 for p in parts)), "white")
    y = 0
    for i, p in enumerate(parts):
        sheet.paste(p, (0, y))
        y += p.height + 14
        if p is parts[-2]:
            ImageDraw.Draw(sheet).line((0, y - 7, width, y - 7), fill=(220, 0, 0), width=3)  # the page break
    buf = io.BytesIO()
    sheet.save(buf, "PNG", optimize=True)
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    prompt = ("The image shows the end of a table on one page (HEAD = its header, END = its last rows), a red line for the "
              "page break, and the start of the next table on a later page (NEXT). Does NEXT continue the same table? Judge by "
              "structure, not topic: same columns, rows simply carry on, no title of its own. Say how many leading rows of NEXT "
              "repeat the header, and whether NEXT's first row is the rest of END's last row (a row cut by the page break).")
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": url}}]}]
    model = TABLE_JUDGE_SINGLE
    return agent._chat("table_seam", model, messages, SEAM_SCHEMA, 2000, _extra(model))
