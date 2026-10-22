"""Table checking, `standard` mode: the free part (no API calls).

Every engine's reading of a table is lined up into one grid in which each cell keeps
every reading. Then, per cell:

    agree      the readings are the same                          -> accepted
    cosmetic   they differ only in spacing, dashes, quotes, $..$  -> accepted, one form kept
    single     only one reader has the cell (it missed a row/col) -> accepted, flagged for audit
    formula    a printed formula (e.g. "कुल (६+९+१२)") holds for exactly one reading
    text_layer the PDF text layer, proven accurate on this table, contains exactly one reading
    open       nothing free can decide                            -> a question for the image check

A decider only chooses between readings the engines produced; it never writes a value.
Agreement is not taken as proof: agreed cells that look risky (the proven text layer says
something else, a known look-alike confusion, ink where the readers saw nothing, an odd
cell for its column, a failing total) become audit questions, ranked by risk.

Structure is checked against the page image (`table_geometry`): ruling lines give the row
and column counts of a ruled table, so a row that both engines dropped is caught.

Each table ends as `verified` (nothing open, nothing high-risk), `repaired` (something was
decided by a free decider or by the image structure), or `needs_review` (open cells,
high-risk cells or a structure conflict). Questions are what step 4 sends to the model.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from src.table_geometry import Geometry
from src.tables import Cell, Table, _is_numbering_row, check_table, clean, parse_number

ENGINE_ORDER = ("sarvam", "bodhan", "qwen")
FORMULA_KINDS = ("row_formula", "col_formula")
HINT_KINDS = ("row_consensus", "col_consensus", "row_product")

_DASHES = str.maketrans({"–": "-", "—": "-", "−": "-", "‐": "-", "‒": "-", "’": "'", "‘": "'", "“": '"', "”": '"'})
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍﻿­"), None)
_LATEX = re.compile(r"^\$+\s*(.*?)\s*\$+$", re.S)

# Measured in the real-engine runs: both engines (and every vision model tried) read Tamil numerals
# as these letters. A cell made only of them may be a misread numeral, however much the readers
# agree, so it is flagged for review (a mapping rule back to numerals is held for a later decision).
LOOKALIKE_OUTPUTS = set("கஉஙசருசுஎஅகூ")
NUMERAL_SCRIPTS = (0x0BE6, 0x0C66)  # Tamil and Telugu digit zeros: rare in print, often misread


def canon(text: str) -> str:
    """Comparison form: Unicode NFC, no zero-width marks, one kind of dash and quote,
    LaTeX number wrappers removed, whitespace collapsed, a doubled combining mark undoubled."""
    s = unicodedata.normalize("NFC", text or "").translate(_ZERO_WIDTH).translate(_DASHES)
    m = _LATEX.match(s.strip())
    if m:
        s = m.group(1)
    out: list[str] = []
    for ch in s:
        if out and ch == out[-1] and unicodedata.category(ch).startswith("M"):
            continue
        out.append(ch)
    return " ".join("".join(out).split())


def similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class QCell:
    r: int
    c: int
    rowspan: int
    colspan: int
    header: bool
    reads: dict[str, str | None]  # engine -> its text (None: that reader has no such cell)
    value: str
    decided_by: str = "agree"
    risk: list[str] = field(default_factory=list)

    @property
    def open(self) -> bool:
        return self.decided_by == "open"

    def to_dict(self) -> dict:
        return {"r": self.r, "c": self.c, "reads": self.reads, "value": self.value, "by": self.decided_by, "risk": self.risk}


@dataclass
class QTable:
    reference: str
    table: Table
    cells: list[QCell]
    structure: dict
    status: str = "verified"
    questions: list[dict] = field(default_factory=list)
    layer: dict = field(default_factory=dict)

    def counts(self) -> dict:
        by = Counter(c.decided_by for c in self.cells)
        risky = Counter(r for c in self.cells for r in c.risk)
        return {"cells": len(self.cells), "decided_by": dict(by), "risk": dict(risky)}

    def summary(self) -> dict:
        return {"status": self.status, "reference": self.reference, "structure": self.structure, "layer": self.layer,
                **self.counts(), "questions": len(self.questions),
                "questions_by_kind": dict(Counter(q["kind"] for q in self.questions))}


# ---------------------------------------------------------------------------
# Alignment of readings
# ---------------------------------------------------------------------------


def align_rows(ref: list[str], other: list[str], min_sim: float = 0.5) -> dict[int, int]:
    """Reference row -> other row, in order, maximising total similarity (a dropped or extra row is a gap)."""
    n, m = len(ref), len(other)
    s = [[similarity(ref[i], other[j]) for j in range(m)] for i in range(n)]
    best = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            take = s[i][j] + best[i + 1][j + 1] if s[i][j] >= min_sim else -1.0
            best[i][j] = max(take, best[i + 1][j], best[i][j + 1])
    out, i, j = {}, 0, 0
    while i < n and j < m:
        if s[i][j] >= min_sim and abs(best[i][j] - (s[i][j] + best[i + 1][j + 1])) < 1e-9:
            out[i] = j
            i, j = i + 1, j + 1
        elif best[i + 1][j] >= best[i][j + 1]:
            i += 1
        else:
            j += 1
    return out


def align_cols(ref: list[list[str]], other: list[list[str]], rows: dict[int, int]) -> dict[int, int]:
    nr, no = len(ref[0]) if ref else 0, len(other[0]) if other else 0
    if nr == no:
        return {c: c for c in range(nr)}
    pairs = list(rows.items())[:40]
    scores = []
    for c in range(nr):
        for d in range(no):
            vals = [similarity(ref[i][c], other[j][d]) for i, j in pairs]
            scores.append((sum(vals) / max(1, len(vals)), -abs(c - d), c, d))
    mapping: dict[int, int] = {}
    used: set[int] = set()
    for score, _, c, d in sorted(scores, reverse=True):
        if score >= 0.5 and c not in mapping and d not in used:
            mapping[c] = d
            used.add(d)
    return mapping


def _grid(t: Table) -> list[list[str]]:
    return [[canon(x) for x in row] for row in t.rows_text()]


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------


@dataclass
class Evidence:
    """Free evidence about one table (all optional)."""
    geometry: Geometry | None = None
    layer_text: str | None = None  # PDF text layer inside the table box, if the page has a usable one
    scanned: bool = False


_LAYER_COMMON = set("₹–—’‘“”•…×°%")


def odd_glyph(ch: str) -> bool:
    """A glyph outside ASCII and the Indic blocks: in an Indic table, a font's private mapping."""
    o = ord(ch)
    return not (o < 128 or 0x0900 <= o <= 0x0DFF or ch in _LAYER_COMMON or ch.isspace())


@dataclass
class Layer:
    """The PDF text layer of a table box: one string without spaces, and its whitespace tokens.
    `clean`: no unexplained glyphs left (a partly repaired layer may still hold some)."""
    blob: str
    tokens: set[str]
    clean: bool = True

    @classmethod
    def of(cls, text: str | None) -> "Layer":
        c = canon(text or "")
        blob = "".join(c.split())
        return cls(blob, set(c.split()), not any(odd_glyph(ch) for ch in blob))

    def contains(self, value: str) -> bool | None:
        """Numbers must match a whole token (1,234 is inside 11,234.50); text must be 4+ characters
        (shorter strings occur inside other words). None: too short to say."""
        key = canon(value)
        if not key:
            return None
        if parse_number(key) is not None:
            return key in self.tokens
        flat = "".join(key.split())
        return flat in self.blob if len(flat) >= 4 else None


def squeeze(t: Table) -> tuple[Table, list[int]]:
    """Drop phantom columns: empty in every body row and with no header cell of their own
    (a reader that misplaces a colspan invents them)."""
    body = {c.c + k for c in t.cells if c.r >= t.header_rows and c.text.strip() for k in range(c.colspan)}
    heads = {c.c for c in t.cells if c.r < t.header_rows and c.text.strip()}
    empty = [c for c in range(t.n_cols) if c not in body and c not in heads]
    if not empty or len(empty) == t.n_cols:
        return t, []
    keep = {c: i for i, c in enumerate(c for c in range(t.n_cols) if c not in empty)}
    cells = []
    for cell in t.cells:
        cols = [c for c in range(cell.c, cell.c + cell.colspan) if c in keep]
        if cols:
            cells.append(Cell(cell.r, keep[cols[0]], cell.rowspan, len(cols), cell.text, cell.header))
    return Table(cells, t.n_rows, len(keep), t.header_rows, dict(t.source)), empty


def numbering_confirms(t: Table) -> bool:
    """The table prints a column-numbering row 1..n, and n is this reading's column count."""
    for r in range(min(t.n_rows, t.header_rows + 1)):
        if not _is_numbering_row(t, r):
            continue
        nums = []
        for cell in sorted((c for c in t.cells if c.r == r), key=lambda x: x.c):
            m = re.match(r"(\d{1,2})", clean(cell.text))
            if m:
                nums.append(int(m.group(1)))
        return nums == list(range(1, t.n_cols + 1))
    return False


def choose_structure(readings: dict[str, Table], geometry: Geometry | None) -> tuple[str, dict]:
    shapes = {e: (t.n_rows, t.n_cols) for e, t in readings.items()}
    order = [e for e in ENGINE_ORDER if e in readings] + [e for e in readings if e not in ENGINE_ORDER]
    info: dict = {"shapes": {e: list(s) for e, s in shapes.items()}, "geometry": geometry.to_dict() if geometry else None}
    agreed = len(set(shapes.values())) == 1
    if geometry is not None and geometry.confident:
        target = (geometry.rows, geometry.cols)
        matching = [e for e in order if shapes[e] == target]
        if matching:
            info["decision"] = "confirmed" if len(matching) == len(order) else "decided_by_image"
            return matching[0], info
        info["decision"] = "conflict"  # no reader has the shape the ruling lines show
        return order[0], info
    if agreed:
        info["decision"] = "agreed"
        return order[0], info
    numbered = [e for e in order if numbering_confirms(readings[e])]
    if numbered and len({shapes[e] for e in numbered}) == 1:
        info["decision"] = "decided_by_numbering"  # the table's own printed column numbers
        return numbered[0], info
    if geometry is not None and geometry.cols and geometry.cols >= 2:
        by_cols = [e for e in order if shapes[e][1] == geometry.cols]
        if len(by_cols) == 1:
            info["decision"] = "decided_by_gaps"
            return by_cols[0], info
    info["decision"] = "readers_differ"
    return order[0], info


def check(readings: dict[str, Table], evidence: Evidence | None = None) -> QTable:
    evidence = evidence or Evidence()
    squeezed = {}
    for e in list(readings):
        readings[e], removed = squeeze(readings[e])
        if removed:
            squeezed[e] = removed
    ref_engine, structure = choose_structure(readings, evidence.geometry)
    if squeezed:
        structure["phantom_columns_removed"] = squeezed
    ref = readings[ref_engine]
    ref_grid = _grid(ref)
    rows_of = [" | ".join(r) for r in ref_grid]
    maps: dict[str, tuple[dict[int, int], dict[int, int], list[list[str]]]] = {}
    raw_grids = {e: t.rows_text() for e, t in readings.items()}
    for e, t in readings.items():
        if e == ref_engine:
            continue
        g = _grid(t)
        rmap = align_rows(rows_of, [" | ".join(r) for r in g])
        maps[e] = (rmap, align_cols(ref_grid, g, rmap), g)
    structure["rows_matched"] = {e: len(m[0]) for e, m in maps.items()}

    cells: list[QCell] = []
    for cell in sorted(ref.cells, key=lambda x: (x.r, x.c)):
        reads: dict[str, str | None] = {ref_engine: cell.text}
        for e, (rmap, cmap, _) in maps.items():
            j, d = rmap.get(cell.r), cmap.get(cell.c)
            reads[e] = raw_grids[e][j][d] if j is not None and d is not None else None
        cells.append(QCell(cell.r, cell.c, cell.rowspan, cell.colspan, cell.r < ref.header_rows, reads, cell.text))

    final = Table([Cell(c.r, c.c, c.rowspan, c.colspan, c.value, c.header) for c in cells],
                  ref.n_rows, ref.n_cols, ref.header_rows, dict(ref.source))
    by_pos = {(c.r, c.c): c for c in cells}
    final_by_pos = {(c.r, c.c): c for c in final.cells}

    text_layer = Layer.of(evidence.layer_text) if evidence.layer_text else None
    layer = _layer_proof(cells, text_layer) if text_layer else {"usable": False}
    proven = text_layer if layer.get("proven") else None

    def formula_failures(r: int | None = None, c: int | None = None) -> int:
        """Failing printed-formula checks (only those in row r or column c, if given)."""
        return sum(1 for ch in check_table(final) if not ch.ok and ch.kind in FORMULA_KINDS
                   and (r is None or ch.row == r or ch.col == c))

    for qc in cells:
        present = {e: t for e, t in qc.reads.items() if t is not None}
        if len(present) == 1:
            qc.decided_by = "single"
            qc.risk.append("single_reader")
            continue
        forms = {canon(t) for t in present.values()}
        if len(forms) == 1:
            if len(set(present.values())) == 1:
                qc.decided_by = "agree"
            else:  # keep the plainest form (no LaTeX wrapper)
                _set(qc, final_by_pos[(qc.r, qc.c)], next((t for t in present.values() if "$" not in t), qc.value), "cosmetic")
            continue
        candidates = list(dict.fromkeys(present.values()))  # distinct raw readings, reference first
        # printed formula: exactly one reading makes the table's own formulas hold
        if not qc.header and any(parse_number(x) is not None for x in candidates):
            target = final_by_pos[(qc.r, qc.c)]
            fails = {}
            for value in candidates:
                target.text = value
                fails[value] = formula_failures()
            best = min(fails.values())
            winners = [v for v, n in fails.items() if n == best]
            target.text = winners[0] if len(winners) == 1 else qc.value
            # the winner must make every formula in its own row and column hold; "fewer failures"
            # alone would pick the lesser of two wrong readings
            if len(winners) == 1 and best < max(fails.values()) and formula_failures(qc.r, qc.c) == 0:
                _set(qc, target, winners[0], "formula")
                continue
            target.text = qc.value
        # proven text layer: exactly one reading is in it
        if proven:
            hits = [v for v in candidates if proven.contains(v)]
            if len(hits) == 1 and all(proven.contains(v) is not None for v in candidates):
                _set(qc, final_by_pos[(qc.r, qc.c)], hits[0], "text_layer")
                continue
            if not hits and proven.clean and all(proven.contains(v) is False for v in candidates):
                # only a fully clean layer may say "neither": in a partly repaired one the word may simply
                # still be scrambled (measured: 56 cells wrongly held back on the Union budget)
                qc.risk.append("layer_has_neither")  # the printed value is probably a third one
        # readings that differ only in punctuation or spacing (not numbers: a lost decimal point is a real
        # error) are settled free: the one a proven layer contains, else the reference. Measured: 14% of the
        # Union budget's disputes, and comma-vs-full-stop is the image check's known weak spot.
        if minor_difference(candidates):
            pick = next((v for v in candidates if proven and proven.contains(v)), qc.value)
            _set(qc, final_by_pos[(qc.r, qc.c)], pick, "minor")
            continue
        qc.decided_by = "open"

    _risks(cells, final, evidence, proven, ref)
    qt = QTable(ref_engine, final, cells, structure, layer=layer)
    _questions(qt, by_pos)
    qt.status = _status(qt)
    return qt


_PUNCT_SPACE = re.compile(r"[\s.,;:\-–—()\[\]।|'\"/]")


def minor_difference(candidates: list[str]) -> bool:
    """True when the readings differ only in punctuation or spacing and none of them is a number."""
    if len(candidates) < 2 or any(parse_number(canon(c)) is not None for c in candidates):
        return False
    stripped = {_PUNCT_SPACE.sub("", canon(c)) for c in candidates}
    return len(stripped) == 1 and bool(next(iter(stripped)))


def apply(qt: QTable, qc: QCell, value: str, by: str) -> None:
    """Set a cell's value in the checked table (used by the image check)."""
    target = next(c for c in qt.table.cells if (c.r, c.c) == (qc.r, qc.c))
    _set(qc, target, value, by)


def refresh(qt: QTable) -> None:
    """Recompute questions and status after the image check changed cells or risks."""
    _questions(qt, {})
    qt.status = _status(qt)


def _set(qc: QCell, target: Cell, value: str, by: str) -> None:
    qc.value = value
    qc.decided_by = by
    target.text = value


def _layer_proof(cells: list[QCell], layer: Layer) -> dict:
    """The text layer counts as evidence for this table only if it contains the cells the readers agree on."""
    agreed = [c for c in cells if len({canon(t) for t in c.reads.values() if t is not None}) == 1
              and len([t for t in c.reads.values() if t is not None]) > 1]
    tested = [r for r in (layer.contains(c.value) for c in agreed) if r is not None]
    share = sum(tested) / len(tested) if tested else 0.0
    return {"usable": True, "tested": len(tested), "found": round(share, 3), "proven": len(tested) >= 5 and share >= 0.9}


def _risks(cells: list[QCell], final: Table, ev: Evidence, proven: Layer | None, ref: Table) -> None:
    body = [c for c in cells if not c.header]
    # 1. the proven text layer disagrees with what every reader saw. Kept as high risk even though the
    #    layer has defects of its own: relaxing it below 98% proof let wrong tables through (measured
    #    2026-10-21) and saved no reviews, since most reviews come from genuine disputes anyway.
    if proven:
        for c in cells:
            if c.decided_by in ("agree", "cosmetic") and proven.contains(c.value) is False:
                c.risk.append("layer_disagrees")
    # 2. look-alike glyphs that real engines produce for rare numerals
    for c in cells:
        v = canon(c.value).replace(".", "").replace(",", "").replace("-", "").replace(" ", "")
        if v and all(ch in LOOKALIKE_OUTPUTS or unicodedata.category(ch).startswith("M") for ch in v):
            c.risk.append("lookalike_numeral")
        if any(z <= ord(ch) <= z + 9 for ch in c.value for z in NUMERAL_SCRIPTS):
            c.risk.append("rare_numerals")
    # 3. a non-number in a column of numbers
    by_col: dict[int, list[QCell]] = {}
    for c in body:
        by_col.setdefault(c.c, []).append(c)
    for col in by_col.values():
        if len(col) < 5:
            continue
        numeric_col = sum(parse_number(c.value) is not None for c in col) >= 0.7 * len(col)
        for c in col:
            if c.value.strip() and numeric_col and parse_number(c.value) is None and c.value.strip() not in ("-", "–", "NA", ".."):
                c.risk.append("not_a_number_in_number_column")
    # 4. totals that do not add up (hints only)
    for ch in check_table(final):
        if not ch.ok and ch.row is not None and ch.col is not None:
            hit = next((c for c in cells if c.r == ch.row and c.c == ch.col), None)
            if hit is not None:
                hit.risk.append("formula_fails" if ch.kind in FORMULA_KINDS else "total_hint_fails")
    # 5. ink where the readers saw nothing, or text where the image is blank (ruled tables, same shape)
    g = ev.geometry
    if g is not None and g.confident and (g.rows, g.cols) == (ref.n_rows, ref.n_cols):
        for c in cells:
            ink = g.cell_ink(c.r, c.c, c.r + c.rowspan, c.c + c.colspan)
            if ink is None:
                continue
            if not c.value.strip() and ink > 0.02:
                c.risk.append("ink_but_empty")
            elif c.value.strip() and ink < 0.003:
                c.risk.append("text_but_blank")


HIGH_RISK = {"layer_disagrees", "ink_but_empty", "formula_fails", "lookalike_numeral", "audit_disagrees"}
# Weights rank audit questions. Measured precision (share of flagged cells that were really wrong) on the
# real-engine runs and simulation, 2026-10-21: ink_but_empty 4/4, lookalike_numeral 27/30, layer_disagrees
# 3/13, formula_fails 6/19, rare_numerals 34/237, not_a_number 5/37, text_but_blank 0/15.
RISK_WEIGHT = {"layer_has_neither": 5, "layer_disagrees": 5, "ink_but_empty": 4, "lookalike_numeral": 4, "formula_fails": 3, "rare_numerals": 2,
               "not_a_number_in_number_column": 1, "total_hint_fails": 1, "single_reader": 1,
               "text_but_blank": 0.5}


def single_reading(qt: QTable) -> bool:
    """At least half of the cells have only one reader: the other engine did not read this table at all (it was
    not run on the page, its credit ran out, or it filed the table as something else). Nothing was compared,
    so the table cannot be called verified."""
    return bool(qt.cells) and sum(c.decided_by == "single" for c in qt.cells) >= 0.5 * len(qt.cells)


def _questions(qt: QTable, by_pos: dict) -> None:
    """What the image check (step 4) should look at, most valuable first."""
    qs = []
    decision = qt.structure.get("decision")
    if decision in ("conflict", "readers_differ"):
        qs.append({"kind": "structure", "priority": 10, "reason": decision, "shapes": qt.structure["shapes"]})
    elif single_reading(qt):
        qs.append({"kind": "structure", "priority": 9, "reason": "single_reading", "shapes": qt.structure["shapes"]})
    for c in qt.cells:
        weight = sum(RISK_WEIGHT.get(r, 1) for r in c.risk)
        if c.open:
            qs.append({"kind": "cell", "priority": 6 + weight, "r": c.r, "c": c.c,
                       "candidates": list(dict.fromkeys(t for t in c.reads.values() if t is not None)), "risk": c.risk})
        elif c.risk and "audit_confirmed" not in c.risk:
            qs.append({"kind": "audit", "priority": weight, "r": c.r, "c": c.c, "value": c.value, "risk": c.risk})
    qt.questions = sorted(qs, key=lambda q: -q["priority"])


def _status(qt: QTable) -> str:
    if qt.structure.get("decision") in ("conflict", "readers_differ") or single_reading(qt):
        return "needs_review"
    # an audit can add doubt (audit_disagrees) but never removes it: a model's blind "same" does not cancel
    # measured evidence (measured: Luna read Telugu E-౩౩ as E-33 and "confirmed" the engines' error)
    if any(c.open for c in qt.cells) or any(set(c.risk) & HIGH_RISK for c in qt.cells):
        return "needs_review"
    repaired = qt.structure.get("decision") in ("decided_by_image", "decided_by_gaps", "decided_by_numbering") or any(
        c.decided_by in ("formula", "text_layer", "minor") or c.decided_by.startswith("image") for c in qt.cells)
    return "repaired" if repaired else "verified"
