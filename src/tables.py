"""Tables: parse engine HTML/markdown into grids, check the numbers, join pages.

Checks need no API calls:
- Formula checks. Indian government tables carry a column-numbering row
  (1 2 3 ... or १ २ ३ ...) and formulas in headers, e.g. "एकूण (४ ते ७)"
  (column 8 = columns 4..7) or a total row "एकूण (क+ख+ग+घ)" (sum of the rows
  labelled क, ख, ग, घ). These are read from the table itself.
- Consensus checks, when there is no formula: a "Total" column/row is assumed
  to be a sum only if most rows/columns agree; the ones that do not are flagged.
- Text-layer digits (born-digital PDFs) and engine agreement are computed by
  the caller with `numbers_in` / `compare_numbers`.

`fuse` repairs disagreeing cells between two engines' versions of the same
table by keeping whichever value makes more checks pass.
"""

from __future__ import annotations

import html
import re
from collections import Counter
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any

# ---------------------------------------------------------------------------
# Digits and numbers
# ---------------------------------------------------------------------------

_ZERO_POINTS = (0x0966, 0x09E6, 0x0A66, 0x0AE6, 0x0B66, 0x0BE6, 0x0C66, 0x0CE6, 0x0D66,
                0x0660, 0x06F0, 0x1C50, 0xABF0)
DIGIT_MAP = {zero + i: ord("0") + i for zero in _ZERO_POINTS for i in range(10)}
DIGIT_MAP.update({0x2212: ord("-"), 0x2013: ord("-")})  # minus sign, en dash

NUMBER_RE = re.compile(r"^\(?[-+]?\d[\d,]*(?:\.\d+)?\)?%?$")
# Whole words only: substring matching finds योग inside उपयोग and मोट inside मोटर,
# and \b is unreliable in Indic scripts (vowel signs are not word characters).
TOTAL_TOKENS = {
    "एकूण", "कुल", "योग", "बेरीज", "महायोग", "सर्वयोग", "মোট", "সর্বমোট", "மொத்தம்", "ਕੁੱਲ", "કુલ", "ମୋଟ",
    "మొత్తం", "ಒಟ್ಟು", "ആകെ", "total", "subtotal", "sub-total", "grand",
}
_TOKEN_SPLIT = re.compile(r"[\s,;:()\[\]/।|]+")


class _TotalWords:
    """`TOTAL_WORDS.search(text)`: does the text contain a 'total' word as a whole word?"""

    @staticmethod
    def search(text: str) -> bool:
        return any(tok.lower().strip(".-") in TOTAL_TOKENS for tok in _TOKEN_SPLIT.split(text) if tok)


TOTAL_WORDS = _TotalWords()
RANGE_WORDS = r"(?:ते|से|तक|to|till|-|–)"


def normalize_digits(text: str) -> str:
    return text.translate(DIGIT_MAP)


def parse_number(text: str) -> float | None:
    s = normalize_digits(text).strip().replace(" ", "").replace(" ", "")
    if not s or not NUMBER_RE.match(s):
        return None
    negative = s.startswith("(") and s.endswith(")") or s.startswith("-")
    s = s.strip("()%+-").replace(",", "")
    try:
        value = float(s)
    except ValueError:
        return None
    return -value if negative else value


def number_key(text: str) -> str | None:
    """Canonical string for matching numbers across sources: '1,296.57' -> '1296.57'."""
    if parse_number(text) is None:
        return None
    return normalize_digits(text).strip().strip("()%+").replace(",", "").replace(" ", "").lstrip("-")


def decimals(text: str) -> int:
    s = normalize_digits(text)
    return len(s.split(".", 1)[1].strip("()%")) if "." in s else 0


def clean(text: str) -> str:
    return " ".join(normalize_digits(text).split()).strip()


# ---------------------------------------------------------------------------
# Table model
# ---------------------------------------------------------------------------


@dataclass
class Cell:
    """One table cell: position, span and text."""
    r: int
    c: int
    rowspan: int
    colspan: int
    text: str
    header: bool = False


@dataclass
class Table:
    """A parsed table: cells on a grid, with header rows."""
    cells: list[Cell]
    n_rows: int
    n_cols: int
    header_rows: int = 0
    source: dict[str, Any] = field(default_factory=dict)

    def grid(self) -> list[list[Cell | None]]:
        g: list[list[Cell | None]] = [[None] * self.n_cols for _ in range(self.n_rows)]
        for cell in self.cells:
            for r in range(cell.r, min(cell.r + cell.rowspan, self.n_rows)):
                for c in range(cell.c, min(cell.c + cell.colspan, self.n_cols)):
                    g[r][c] = cell
        return g

    def rows_text(self, expand: bool = True) -> list[list[str]]:
        """Grid of texts. expand=True: a merged cell's text repeats in every slot it
        covers (for labels, CSV). expand=False: only at its top-left slot (for sums,
        so a merged value is counted once)."""
        return [
            [cell.text if cell and (expand or (cell.r == r and cell.c == c)) else "" for c, cell in enumerate(row)]
            for r, row in enumerate(self.grid())
        ]

    def origin(self, r: int, c: int) -> Cell | None:
        cell = self.grid()[r][c]
        return cell if cell is not None and cell.r == r and cell.c == c else None

    def to_html(self) -> str:
        by_row: dict[int, list[Cell]] = {}
        for cell in self.cells:
            by_row.setdefault(cell.r, []).append(cell)
        out = ["<table>"]
        for r in range(self.n_rows):
            if r == 0 and self.header_rows:
                out.append("<thead>")
            if r == self.header_rows and self.header_rows:
                out.append("</thead><tbody>")
            parts = []
            for cell in sorted(by_row.get(r, []), key=lambda x: x.c):
                tag = "th" if cell.header or r < self.header_rows else "td"
                attrs = (f' rowspan="{cell.rowspan}"' if cell.rowspan > 1 else "") + (
                    f' colspan="{cell.colspan}"' if cell.colspan > 1 else ""
                )
                body = "<br/>".join(html.escape(line) for line in cell.text.split("\n"))
                parts.append(f"<{tag}{attrs}>{body}</{tag}>")
            out.append("<tr>" + "".join(parts) + "</tr>")
        if self.header_rows:
            out.append("</tbody>" if self.header_rows < self.n_rows else "</thead>")
        out.append("</table>")
        return "\n".join(out)


class _HTMLTables(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[dict]]] = []
        self.depth = 0
        self.in_thead = False
        self.row: list[dict] | None = None
        self.cell: dict | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if tag == "table":
            self.depth += 1
            if self.depth == 1:
                self.tables.append([])
        elif self.depth != 1:
            if tag == "br" and self.cell is not None:
                self.cell["text"] += "\n"
            return
        elif tag == "thead":
            self.in_thead = True
        elif tag == "tr":
            self.row = []
        elif tag in ("td", "th"):
            if self.row is None:
                self.row = []

            def span(name: str) -> int:
                try:
                    return max(1, int(a.get(name) or 1))
                except ValueError:
                    return 1

            self.cell = {"text": "", "header": tag == "th" or self.in_thead,
                         "rowspan": span("rowspan"), "colspan": span("colspan")}
        elif tag == "br" and self.cell is not None:
            self.cell["text"] += "\n"

    def handle_endtag(self, tag: str) -> None:
        if tag == "table":
            self.depth -= 1
        elif self.depth != 1:
            return
        elif tag == "thead":
            self.in_thead = False
        elif tag in ("td", "th") and self.cell is not None and self.row is not None:
            self.row.append(self.cell)
            self.cell = None
        elif tag == "tr" and self.row is not None:
            if self.cell is not None:
                self.row.append(self.cell)
                self.cell = None
            self.tables[-1].append(self.row)
            self.row = None

    def handle_data(self, data: str) -> None:
        if self.cell is not None:
            self.cell["text"] += data


def _layout(rows: list[list[dict]]) -> Table:
    occupied: set[tuple[int, int]] = set()
    cells: list[Cell] = []
    n_cols = 0
    for r, row in enumerate(rows):
        c = 0
        for raw in row:
            while (r, c) in occupied:
                c += 1
            text = "\n".join(line.strip() for line in raw["text"].strip().split("\n")).strip()
            cell = Cell(r, c, raw["rowspan"], raw["colspan"], text, raw["header"])
            cells.append(cell)
            for dr in range(cell.rowspan):
                for dc in range(cell.colspan):
                    occupied.add((r + dr, c + dc))
            c += cell.colspan
            n_cols = max(n_cols, c)
    n_rows = len(rows)
    for cell in cells:  # clip spans that run past the table
        cell.rowspan = min(cell.rowspan, n_rows - cell.r)
    table = Table(cells, n_rows, n_cols)
    table.header_rows = _header_rows(table)
    return table


def _header_rows(table: Table) -> int:
    """Leading rows that are all header cells, or the column-numbering row."""
    count = 0
    for r in range(table.n_rows):
        starts = [cell for cell in table.cells if cell.r == r]
        if starts and (all(cell.header for cell in starts) or _is_numbering_row(table, r)):
            count = r + 1
            continue
        if any(cell.r < r < cell.r + cell.rowspan for cell in table.cells if cell.header):
            count = r + 1  # covered by a multi-row header cell
            continue
        break
    return count


def _parse_markdown(text: str) -> list[Table]:
    tables: list[Table] = []
    block: list[str] = []
    for line in text.splitlines() + [""]:
        if line.strip().startswith("|"):
            block.append(line.strip())
            continue
        if len(block) >= 2:
            rows: list[list[dict]] = []
            header_rows = 0
            for i, raw in enumerate(block):
                parts = [p.strip() for p in raw.strip("|").split("|")]
                if all(re.fullmatch(r":?-{2,}:?", p) for p in parts if p):
                    header_rows = i
                    continue
                rows.append([{"text": p, "header": False, "rowspan": 1, "colspan": 1} for p in parts])
            table = _layout(rows)
            table.header_rows = max(table.header_rows, header_rows)
            tables.append(table)
        block = []
    return tables


def parse_tables(text: str, source: dict | None = None) -> list[Table]:
    """All tables in a block's text (HTML preferred, markdown pipe tables otherwise)."""
    tables: list[Table] = []
    if "<table" in text.lower():
        parser = _HTMLTables()
        parser.feed(text)
        parser.close()
        tables = [_layout(rows) for rows in parser.tables if rows]
    else:
        tables = _parse_markdown(text)
    for i, table in enumerate(tables):
        table.source = dict(source or {}, index=i)
    return [t for t in tables if t.n_rows and t.n_cols]


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


@dataclass
class Check:
    """One arithmetic check (a printed formula or a consensus rule) and whether it held."""
    kind: str
    row: int | None
    col: int | None
    expected: float | None
    found: float | None
    ok: bool
    detail: str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _is_numbering_row(table: Table, r: int) -> bool:
    """A printed column-numbering row: '1 2 3 ...' or '३ ४अ ४ब ५' (every cell starts with a digit)."""
    texts = [t for t in (clean(cell.text) for cell in table.cells if cell.r == r) if t]
    if len(texts) < 3 or not all(re.match(r"\d{1,2}(?!\d)(?!\.\d)", t) for t in texts):
        return False
    ints = [int(re.match(r"\d{1,2}", t).group(0)) for t in texts]
    return ints == sorted(ints) and ints[0] in (0, 1, 2) and ints[-1] - ints[0] <= len(ints)


def column_numbers(table: Table) -> dict[int, int]:
    """Printed column number -> column index, from the numbering row if any."""
    for r in range(min(table.n_rows, table.header_rows + 1)):
        if _is_numbering_row(table, r):
            mapping: dict[int, int] = {}
            for cell in table.cells:
                if cell.r == r:
                    m = re.match(r"(\d{1,2})", clean(cell.text))
                    if m:
                        mapping.setdefault(int(m.group(1)), cell.c)
            return mapping
    return {i + 1: i for i in range(table.n_cols)}


def _formula_terms(text: str) -> list[str] | None:
    """'(४ ते ७)' -> ['4','5','6','7']; '(क+ख+ग)' -> ['क','ख','ग']; else None."""
    for inner in re.findall(r"\(([^()]{1,40})\)", normalize_digits(text)):
        inner = inner.strip()
        m = re.fullmatch(rf"(\d{{1,2}})\s*{RANGE_WORDS}\s*(\d{{1,2}})", inner)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if 0 < a < b <= 60:
                return [str(i) for i in range(a, b + 1)]
        if "+" in inner:
            terms = [t.strip() for t in inner.split("+") if t.strip()]
            if len(terms) >= 2 and all(len(t) <= 4 for t in terms):
                return terms
    return None


def _tolerance(texts: list[str]) -> float:
    unit = 10 ** -max([decimals(t) for t in texts] or [0])
    return unit * 0.5 * max(1, len(texts)) + 1e-9


def _row_label(rows: list[list[str]], r: int, upto: int | None = None) -> str:
    """The row's text cells (numbers and code columns skipped), in column order."""
    return " ".join(t for t in rows[r][:upto] if t and parse_number(t) is None)


def _is_serial(rows: list[list[str]], body: list[int], c: int) -> bool:
    """A serial-number column (1, 2, 3 ... going down): an identifier, not an amount."""
    values = [parse_number(rows[r][c]) for r in body]
    steps = [b - a for a, b in zip(values, values[1:]) if a is not None and b is not None]
    return len(steps) >= 2 and sum(step == 1 for step in steps) >= 0.6 * len(steps)


def _is_code(text: str) -> bool:
    """Budget heads and serial codes ('००४९', '0230'): identifiers, not amounts."""
    s = clean(text)
    return bool(re.fullmatch(r"0\d{2,}", s))


def check_table(table: Table) -> list[Check]:
    rows = table.rows_text(expand=False)  # numbers: merged cells count once
    labels = table.rows_text()  # labels and headers: merged text everywhere
    body = [r for r in range(table.header_rows, table.n_rows) if not _is_numbering_row(table, r)]
    numbering = column_numbers(table)
    numeric_cols = [
        c for c in range(table.n_cols)
        if sum(parse_number(rows[r][c]) is not None for r in body) >= max(1, len(body) // 3)
        and sum(_is_code(rows[r][c]) for r in body) < max(1, len(body) // 4)
        and not _is_serial(rows, body, c)
    ]
    checks: list[Check] = []

    # Row formulas from header cells: column c = sum of columns given in its header.
    formula_cols: set[int] = set()
    for cell in table.cells:
        if cell.r >= table.header_rows:
            continue
        terms = _formula_terms(cell.text)
        if not terms or not all(t.isdigit() for t in terms):
            continue
        sources = [numbering.get(int(t)) for t in terms]
        if None in sources or cell.c in sources:
            continue
        formula_cols.add(cell.c)
        for r in body:
            target = parse_number(rows[r][cell.c])
            values = [rows[r][c] for c in sources]  # type: ignore[index]
            nums = [parse_number(v) for v in values]
            if target is None or all(n is None for n in nums) or any(n is None and v for n, v in zip(nums, values)):
                continue
            total = sum(n or 0.0 for n in nums)
            ok = abs(total - target) <= _tolerance(values + [rows[r][cell.c]])
            checks.append(Check("row_formula", r, cell.c, round(total, 6), target, ok, clean(cell.text)))

    # Column formulas from total rows: "एकूण (क+ख+ग+घ)" = rows labelled क, ख, ग, घ.
    formula_rows: set[int] = set()
    for r in body:
        label = _row_label(labels, r)
        terms = _formula_terms(label)
        if not terms or all(t.isdigit() for t in terms):
            continue
        members = []
        for term in terms:
            # "(क)" is a section; "क)" may be a sub-item inside one. Prefer full brackets.
            full = re.compile(rf"^\(\s*{re.escape(term)}\s*\)")
            loose = re.compile(rf"^\(?\s*{re.escape(term)}\s*[).]")
            hit = [x for x in body if x != r and full.match(_row_label(labels, x))]
            if len(hit) != 1:
                hit = [x for x in body if x != r and loose.match(_row_label(labels, x))]
            if len(hit) != 1:
                members = []
                break
            members.append(hit[0])
        if not members:
            continue
        formula_rows.add(r)
        for c in numeric_cols:
            target = parse_number(rows[r][c])
            values = [rows[x][c] for x in members]
            if target is None:
                continue
            total = sum(parse_number(v) or 0.0 for v in values)
            ok = abs(total - target) <= _tolerance(values + [rows[r][c]])
            checks.append(Check("col_formula", r, c, round(total, 6), target, ok, label))

    # Consensus: a "total" column with no formula = sum of the numeric run to its left.
    for c in numeric_cols:
        if c in formula_cols:
            continue
        header = " ".join(labels[r][c] for r in range(table.header_rows))
        if not TOTAL_WORDS.search(header):
            continue
        left = [x for x in numeric_cols if x < c]
        run: list[int] = []
        for x in reversed(left):
            if run and x != run[-1] - 1:
                break
            run.append(x)
        if len(run) < 2:
            continue
        candidate = []
        for r in body:
            target = parse_number(rows[r][c])
            values = [rows[r][x] for x in run]
            if target is None or all(parse_number(v) is None for v in values):
                continue
            total = sum(parse_number(v) or 0.0 for v in values)
            candidate.append(Check("row_consensus", r, c, round(total, 6), target,
                                   abs(total - target) <= _tolerance(values + [rows[r][c]]), clean(header)))
        if candidate and sum(ch.ok for ch in candidate) >= 0.6 * len(candidate):
            checks.extend(candidate)

    # Consensus: a "total" row with no formula = sum of body rows since the last total row.
    last_total = None
    for r in body:
        label = _row_label(labels, r)
        if r in formula_rows or not TOTAL_WORDS.search(label):
            if TOTAL_WORDS.search(label):
                last_total = r
            continue
        members = [x for x in body if x < r and (last_total is None or x > last_total)
                   and not TOTAL_WORDS.search(_row_label(labels, x))]
        last_total = r
        if len(members) < 2:
            continue
        candidate = []
        for c in numeric_cols:
            target = parse_number(rows[r][c])
            values = [rows[x][c] for x in members]
            if target is None or all(parse_number(v) is None for v in values):
                continue
            total = sum(parse_number(v) or 0.0 for v in values)
            candidate.append(Check("col_consensus", r, c, round(total, 6), target,
                                   abs(total - target) <= _tolerance(values + [rows[r][c]]), label))
        if candidate and sum(ch.ok for ch in candidate) >= 0.6 * len(candidate):
            checks.extend(candidate)

    # Consensus: a column that is the product of two others (annual x years = total).
    # Many pairs are tried, so the bar is high: >= 5 rows and >= 80% of them agree.
    # A column that mostly copies another one (e.g. "liability remaining" = "total"
    # until payments start) is skipped: its agreement would be borrowed, not real.
    covered = {(ch.kind, ch.col) for ch in checks}

    def copies_another(c: int) -> bool:
        for d in numeric_cols:
            if d == c:
                continue
            both = [r for r in body if parse_number(rows[r][c]) is not None and parse_number(rows[r][d]) is not None]
            if both and sum(parse_number(rows[r][c]) == parse_number(rows[r][d]) for r in both) >= 0.5 * len(both):
                return d < c
        return False

    for c in numeric_cols:
        if any(k.startswith("row_") and col == c for k, col in covered) or copies_another(c):
            continue
        best: list[Check] = []
        for a in numeric_cols:
            for b in numeric_cols:
                if not a < b or c in (a, b):
                    continue
                candidate = []
                for r in body:
                    ta, tb, tc = rows[r][a], rows[r][b], rows[r][c]
                    va, vb, vc = parse_number(ta), parse_number(tb), parse_number(tc)
                    if va is None or vb is None or vc is None or vc == 0 or va * vb == 0:
                        continue
                    tol = (0.5 * 10 ** -decimals(ta) * abs(vb) + 0.5 * 10 ** -decimals(tb) * abs(va)
                           + 0.5 * 10 ** -decimals(tc) + 1e-9)
                    candidate.append(Check("row_product", r, c, round(va * vb, 6), vc,
                                           abs(va * vb - vc) <= tol, f"col{c + 1} = col{a + 1} x col{b + 1}"))
                if len(candidate) >= 5 and sum(ch.ok for ch in candidate) >= 0.8 * len(candidate):
                    if sum(ch.ok for ch in candidate) > sum(ch.ok for ch in best):
                        best = candidate
        checks.extend(best)
    return checks


def failed(checks: list[Check]) -> list[Check]:
    return [ch for ch in checks if not ch.ok]


# ---------------------------------------------------------------------------
# Comparing sources
# ---------------------------------------------------------------------------


def numbers_in(table: Table) -> Counter:
    keys = (number_key(cell.text) for cell in table.cells if cell.r >= table.header_rows)
    return Counter(k for k in keys if k)


def compare_numbers(a: Counter, b: Counter) -> dict:
    common = sum((a & b).values())
    total = max(sum(a.values()), sum(b.values()), 1)
    return {
        "agreement": round(common / total, 4),
        "only_a": sorted((a - b).elements())[:50],
        "only_b": sorted((b - a).elements())[:50],
    }


def text_layer_match(table: Table, layer: Counter) -> dict | None:
    """Share of the table's numbers that also appear in the PDF text layer."""
    mine = numbers_in(table)
    if not mine or sum(layer.values()) < 0.5 * sum(mine.values()):
        return None  # the text layer has too few numbers here to judge
    found = sum((mine & layer).values())
    return {"ratio": round(found / sum(mine.values()), 4), "missing": sorted((mine - layer).elements())[:50]}


@dataclass
class Fusion:
    """The result of merging two engines' readings of one table."""
    table: Table
    replaced: list[dict]
    unresolved: list[dict]


def fuse(primary: Table, other: Table) -> Fusion:
    """Repair `primary` cell by cell from `other` where that makes more checks pass.

    Only for tables with the same shape (same grid). Cells where the engines
    disagree and the checks cannot decide are returned as `unresolved`.
    """
    if (primary.n_rows, primary.n_cols) != (other.n_rows, other.n_cols):
        return Fusion(primary, [], [])
    g_other = other.rows_text()
    replaced: list[dict] = []
    unresolved: list[dict] = []
    baseline = len(failed(check_table(primary)))
    for cell in sorted(primary.cells, key=lambda x: (x.r, x.c)):
        if cell.r < primary.header_rows:
            continue
        theirs = g_other[cell.r][cell.c]
        if parse_number(cell.text) is None and parse_number(theirs) is None:
            continue
        if clean(cell.text) == clean(theirs):
            continue
        mine = cell.text
        cell.text = theirs
        trial = len(failed(check_table(primary)))
        if trial < baseline:
            baseline = trial
            replaced.append({"row": cell.r, "col": cell.c, "from": mine, "to": theirs})
            continue
        cell.text = mine
        if trial == baseline:
            unresolved.append({"row": cell.r, "col": cell.c, "a": mine, "b": theirs})
    return Fusion(primary, replaced, unresolved)


# ---------------------------------------------------------------------------
# Joining tables across pages
# ---------------------------------------------------------------------------

CONTINUED_RE = re.compile(r"contd|continued|cont\.|क्रमशः|क्रमश:|चालू|पुढे|जारी|தொடர்ச்சி|চলবে|ক্রমশ", re.I)


def _header_text(table: Table, rows: int) -> list[str]:
    grid = table.rows_text()
    return [clean(t) for r in range(min(rows, table.n_rows)) for t in grid[r]]


def repeated_header_rows(first: Table, second: Table) -> int:
    """How many leading rows of `second` repeat `first`'s header rows."""
    k = first.header_rows
    if k == 0 or second.n_rows < k or first.n_cols != second.n_cols:
        return 0
    a, b = _header_text(first, k), _header_text(second, k)
    same = sum(x == y for x, y in zip(a, b))
    return k if same >= 0.8 * len(a) else 0


@dataclass
class Continuation:
    """Evidence that a table continues from one page onto the next."""
    score: int
    decision: str  # join | separate | ambiguous
    reasons: list[str]
    drop_rows: int


def continuation(
    first: Table, first_bbox: tuple, second: Table, second_bbox: tuple, between: str = "",
    *, last_on_page: bool = False, first_on_page: bool = False,
) -> Continuation:
    """Does `second` (top of page N+1) continue `first` (bottom of page N)?

    `last_on_page` / `first_on_page`: nothing but footnotes, running heads or
    page numbers come after / before the table, so it ends / starts its page's
    content even when footnotes push it away from the paper's edge.
    """
    score, reasons = 0, []
    if first.n_cols == second.n_cols:
        score += 2
        reasons.append("same column count")
    elif abs(first.n_cols - second.n_cols) == 1:
        reasons.append("column count differs by 1")
    else:
        return Continuation(-5, "separate", ["column counts differ"], 0)
    drop = repeated_header_rows(first, second)
    if drop:
        score += 2
        reasons.append(f"repeats {drop} header row(s)")
    if column_numbers(first) == column_numbers(second) and first.header_rows and second.header_rows:
        score += 1
        reasons.append("same column numbering")
    if first_bbox[3] > 0.75 or last_on_page:
        score += 1
        reasons.append("first ends its page's content" if last_on_page else "first reaches page bottom")
    if second_bbox[1] < 0.3 or first_on_page:
        score += 1
        reasons.append("second starts its page's content" if first_on_page else "second starts at page top")
    if between and CONTINUED_RE.search(between):
        score += 2
        reasons.append("'continued' marker")
    if second.header_rows == 0 and first.n_cols == second.n_cols:
        score += 1
        reasons.append("second has no header of its own")
    if score >= 5:
        decision = "join"
    elif score <= 1:
        decision = "separate"
    else:
        decision = "ambiguous"
    return Continuation(score, decision, reasons, drop)


def join(first: Table, second: Table, drop_rows: int) -> Table:
    """Append `second`'s rows (minus a repeated header) to `first`.

    A first continuation row with no numbers and an empty first cell is a
    description that wrapped over the page break: it is merged into the
    previous row instead of becoming a row of its own.
    """
    # In a continuation, the second part's own header rows (th / thead / numbering
    # row) repeat the first part's header, however differently the OCR read them.
    drop_rows = max(drop_rows, second.header_rows)
    cells = [Cell(c.r, c.c, c.rowspan, c.colspan, c.text, c.header) for c in first.cells]
    offset = first.n_rows
    incoming = [c for c in second.cells if c.r >= drop_rows]
    first_row = min((c.r for c in incoming), default=None)
    if first_row is not None and first.n_rows > first.header_rows:
        row_cells = [c for c in incoming if c.r == first_row]
        texts = {c.c: c.text for c in row_cells}
        if (not texts.get(0) and not any(c.header for c in row_cells)
                and all(parse_number(t) is None for t in texts.values()) and any(texts.values())):
            last = first.n_rows - 1
            for c in row_cells:
                if c.text:
                    target = next((x for x in cells if x.r <= last < x.r + x.rowspan and x.c <= c.c < x.c + x.colspan), None)
                    if target is not None:
                        target.text = f"{target.text} {c.text}".strip()
            incoming = [c for c in incoming if c.r != first_row]
            offset -= 1
    for c in incoming:
        cells.append(Cell(c.r - drop_rows + offset, c.c, c.rowspan, c.colspan, c.text, False))
    n_rows = max((c.r + c.rowspan for c in cells), default=0)
    pages = list(first.source.get("pages", [first.source.get("page")])) + list(
        second.source.get("pages", [second.source.get("page")])
    )
    source = dict(first.source, pages=[p for p in pages if p is not None])
    return Table(cells, n_rows, max(first.n_cols, second.n_cols), first.header_rows, source)
