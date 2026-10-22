"""Joining table segments across page breaks (`standard` mode). No API calls except the optional seam check.

Each table segment gets a signature: column positions from the ruling lines, header text,
the kind of content in each column (number, date, text, empty) and its serial numbers.
A segment that ends its page is compared with the first table segment of the next one or
two pages (pages with only a figure or caption in between are skipped). Once segments are
joined, later segments are compared with the table's first segment, so a long table keeps
one identity instead of being decided page pair by page pair.

    for a join      same column positions; repeated header; same content kinds per column;
                    serial numbers continue; a "continued" marker
    against         a different table title; a column count that cannot be mapped

Merging maps columns, drops a repeated header and merges a row cut by the page break: the
first row continues the previous one when the "key" columns (filled in nearly every row,
whatever they contain) are empty in it. After joining, the seam is checked: column kinds
stay the same, serials continue, and the seam row is not a duplicate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from src.table_geometry import Geometry
from src.tables import CONTINUED_RE, Cell, Table, clean, parse_number

DATE_RE = re.compile(r"^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}$")
TITLE_RE = re.compile(r"(table|सारणी|तालिका|तक्ता|அட்டவணை|పట్టిక|সারণি)\s*[\w.\-]*\s*(\d+)", re.I)
JOIN_AT, SEPARATE_AT = 4.0, 1.0


def kind_of(text: str) -> str:
    t = clean(text)
    if not t:
        return "empty"
    if parse_number(t) is not None:
        return "number"
    if DATE_RE.match(t):
        return "date"
    return "text"


@dataclass
class Signature:
    n_cols: int
    header: list[str]  # header text per column (all header rows joined)
    kinds: list[str]  # dominant content kind per column, body rows
    columns: list[float] | None  # inner column boundaries, 0-1 across the table (ruled tables)
    serial_col: int | None
    first_serial: float | None
    last_serial: float | None
    title: str | None  # table number from a caption, e.g. "3"


def _dominant(values: list[str]) -> str:
    kinds = [kind_of(v) for v in values]
    filled = [k for k in kinds if k != "empty"]
    if not filled:
        return "empty"
    top = max(set(filled), key=filled.count)
    return top if filled.count(top) >= 0.6 * len(filled) else "mixed"


def _serial(table: Table) -> tuple[int | None, float | None, float | None]:
    rows = table.rows_text()[table.header_rows:]
    for c in range(min(2, table.n_cols)):
        vals = [parse_number(r[c]) for r in rows]
        nums = [v for v in vals if v is not None]
        if len(nums) >= 2 and all(b - a == 1 for a, b in zip(nums, nums[1:])):
            return c, nums[0], nums[-1]
    return None, None, None


def signature(table: Table, geometry: Geometry | None, caption: str = "") -> Signature:
    rows = table.rows_text()
    body = rows[table.header_rows:]
    header = [" ".join(dict.fromkeys(clean(rows[r][c]) for r in range(table.header_rows) if rows[r][c])) for c in range(table.n_cols)]
    kinds = [_dominant([r[c] for r in body]) for c in range(table.n_cols)]
    columns = None
    if geometry is not None and geometry.source == "rules" and geometry.cols == table.n_cols and len(geometry.v_lines) >= 2:
        v0, v1 = geometry.v_lines[0], geometry.v_lines[-1]
        columns = [round((x - v0) / max(1, v1 - v0), 3) for x in geometry.v_lines[1:-1]]
    sc, first, last = _serial(table)
    m = TITLE_RE.search(caption or "")
    return Signature(table.n_cols, header, kinds, columns, sc, first, last, m.group(2) if m else None)


@dataclass
class JoinDecision:
    score: float
    decision: str  # join | separate | uncertain
    reasons: list[str] = field(default_factory=list)
    drop_rows: int = 0
    column_map: dict[int, int] | None = None  # column of B -> column of A


def _sim(a: str, b: str) -> float:
    return 1.0 if a == b else SequenceMatcher(None, a, b, autojunk=False).ratio()


def repeated_header_rows(head: Table, second: Table) -> int:
    """Leading rows of `second` that repeat `head`'s header rows (fuzzy: OCR reads them differently)."""
    k = head.header_rows
    if not k or second.n_rows < k:
        return 0
    a, b = head.rows_text()[:k], second.rows_text()[:k]
    if len(a[0]) != len(b[0]):
        return 0
    sims = [_sim(clean(" ".join(x)), clean(" ".join(y))) for x, y in zip(a, b)]
    return k if min(sims) >= 0.75 else 0


def map_columns(a: Signature, b: Signature) -> dict[int, int] | None:
    """B's columns onto A's: identity when the counts match; by position when both are ruled."""
    if a.n_cols == b.n_cols:
        return {c: c for c in range(b.n_cols)}
    if a.columns is None or b.columns is None:
        return None
    edges_a, edges_b = [0.0] + a.columns + [1.0], [0.0] + b.columns + [1.0]
    out = {}
    for j in range(b.n_cols):  # a merged column maps to the column it overlaps most
        overlap = [max(0.0, min(edges_a[i + 1], edges_b[j + 1]) - max(edges_a[i], edges_b[j])) for i in range(a.n_cols)]
        out[j] = max(range(a.n_cols), key=overlap.__getitem__)
    return out


def decide(head: Table, head_sig: Signature, second: Table, second_sig: Signature, between: str = "",
           gap_pages: int = 0) -> JoinDecision:
    score, reasons = 0.0, []
    cmap = map_columns(head_sig, second_sig)
    if cmap is None:
        return JoinDecision(-5, "separate", ["column counts differ and cannot be mapped"])
    if head_sig.n_cols == second_sig.n_cols:
        score += 1
        reasons.append("same column count")
    else:
        reasons.append(f"columns mapped by position ({second_sig.n_cols} onto {head_sig.n_cols})")
    if head_sig.columns is not None and second_sig.columns is not None and len(head_sig.columns) == len(second_sig.columns):
        if max((abs(x - y) for x, y in zip(head_sig.columns, second_sig.columns)), default=0) <= 0.03:
            score += 2
            reasons.append("same column positions on the page")
        # different positions are not held against a join: a continuation without a header is often laid out narrower
    drop = repeated_header_rows(head, second) if head_sig.n_cols == second_sig.n_cols else 0
    if drop:
        score += 2
        reasons.append(f"repeats the {drop}-row header")
    elif second.header_rows == 0:
        score += 2  # a table that opens a page without a header is the typical continuation
        reasons.append("no header of its own")
    else:
        header_sim = _sim(" ".join(head_sig.header), " ".join(second_sig.header))
        if header_sim < 0.5:
            score -= 1  # engines sometimes mark a continuation's first row as a header
            reasons.append("a different header")
    pairs = [(head_sig.kinds[cmap[j]], second_sig.kinds[j]) for j in range(second_sig.n_cols)]
    comparable = [(x, y) for x, y in pairs if "empty" not in (x, y) and "mixed" not in (x, y)]
    if comparable:
        same = sum(x == y for x, y in comparable) / len(comparable)
        if same >= 0.8:
            score += 1
            reasons.append("same kind of content per column")
        elif same < 0.5:
            score -= 2
            reasons.append("different kinds of content per column")
    if head_sig.serial_col is not None and second_sig.serial_col == head_sig.serial_col and second_sig.first_serial is not None:
        if second_sig.first_serial == head_sig.last_serial + 1:
            score += 2
            reasons.append("serial numbers continue")
        elif second_sig.first_serial == 1:
            score -= 1
            reasons.append("serial numbers restart at 1")
    if between and CONTINUED_RE.search(between):
        score += 2
        reasons.append("'continued' marker")
    if second_sig.title and head_sig.title and second_sig.title != head_sig.title:
        return JoinDecision(-5, "separate", reasons + [f"a different table title ({head_sig.title} vs {second_sig.title})"])
    if gap_pages:
        reasons.append(f"resumes after {gap_pages} page(s) without a table")
    decision = "join" if score >= JOIN_AT else "separate" if score <= SEPARATE_AT else "uncertain"
    return JoinDecision(score, decision, reasons, drop, cmap)


def key_columns(table: Table) -> list[int]:
    """Columns filled in at least 90% of body rows: a real row has (nearly) all of them, a cut-off tail few."""
    body = table.rows_text()[table.header_rows:]
    if not body:
        return []
    return [c for c in range(table.n_cols) if sum(bool(r[c].strip()) for r in body) >= 0.9 * len(body)]


def merge(head: Table, second: Table, d: JoinDecision) -> tuple[Table, dict]:
    """Append `second` (minus a repeated header) to `head`, mapping columns and mending a cut row."""
    notes: dict = {"cut_row_merged": False, "duplicate_seam_row_dropped": False}
    drop = max(d.drop_rows, second.header_rows)
    cmap = d.column_map or {c: c for c in range(second.n_cols)}
    cells = [Cell(c.r, c.c, c.rowspan, c.colspan, c.text, c.header) for c in head.cells]
    incoming = [c for c in second.cells if c.r >= drop]
    offset = head.n_rows - drop
    if incoming and head.n_rows > head.header_rows:
        first_row = min(c.r for c in incoming)
        row_cells = {cmap.get(c.c, c.c): c for c in incoming if c.r == first_row}
        keys = key_columns(head)
        last = head.n_rows - 1
        head_last = {c.c: c for c in cells if c.r <= last < c.r + c.rowspan}
        tail = {k: v for k, v in row_cells.items() if v.text.strip()}
        empty_keys = [k for k in keys if not row_cells.get(k) or not row_cells[k].text.strip()]
        # a cut-off tail: most key columns empty, and what it does have is text (numbers are never cut)
        if keys and tail and len(empty_keys) >= 0.5 * len(keys) and all(kind_of(c.text) == "text" for c in tail.values()):
            for col, c in tail.items():  # a row cut by the page break
                target = head_last.get(col)
                if target is not None:
                    target.text = f"{target.text} {c.text}".strip()
            incoming = [c for c in incoming if c.r != first_row]
            offset -= 1
            notes["cut_row_merged"] = True
        elif [clean(head_last[k].text) for k in sorted(head_last)] == [clean(row_cells[k].text) for k in sorted(row_cells)]:
            incoming = [c for c in incoming if c.r != first_row]  # the seam row printed on both pages
            offset -= 1
            notes["duplicate_seam_row_dropped"] = True
    for c in incoming:
        cells.append(Cell(c.r + offset, cmap.get(c.c, c.c), c.rowspan, c.colspan, c.text, False))
    n_rows = max((c.r + c.rowspan for c in cells), default=0)
    source = dict(head.source)
    return Table(cells, n_rows, head.n_cols, head.header_rows, source), notes


def seam_checks(joined: Table, head_rows: int, sig: Signature) -> list[str]:
    """Problems at the seam after a join (empty when it looks consistent)."""
    problems = []
    rows = joined.rows_text()
    before, after = rows[joined.header_rows:head_rows], rows[head_rows:]
    if before and after:
        kb = [_dominant([r[c] for r in before]) for c in range(joined.n_cols)]
        ka = [_dominant([r[c] for r in after]) for c in range(joined.n_cols)]
        bad = [c for c, (x, y) in enumerate(zip(kb, ka)) if "empty" not in (x, y) and "mixed" not in (x, y) and x != y]
        if bad:
            problems.append(f"column kinds change at the seam: {bad}")
    if sig.serial_col is not None and before and after:  # only the seam: last serial above it, first below it
        above = [parse_number(r[sig.serial_col]) for r in before]
        below = [parse_number(r[sig.serial_col]) for r in after]
        last = next((v for v in reversed(above) if v is not None), None)
        first = next((v for v in below if v is not None), None)
        if last is not None and first is not None and first != last + 1:
            problems.append(f"serial numbers jump at the seam ({last:g} -> {first:g})")
    return problems
