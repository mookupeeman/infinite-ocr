"""Synthetic benchmark tables with known truth.

A `Case` is one small document: a few pages, each printing a table segment, a
figure or nothing, plus the logical table(s) those pages came from. The truth is
a `src.tables.Table` that the generator wrote, so every cell is known.

    kinds   text      names, categories, descriptions, status words (no arithmetic)
            numeric   amounts with a total column and a total row (arithmetic holds)
            mixed     codes, names, amounts, dates, remarks
            dates     events with start and end dates
            merged    two-level header, a merged group header, a category column with row groups
            sparse    like mixed, with many empty cells and symbols
            wide      budget-style: serial, code, description, 2-3 year groups of three amounts, total,
                      a two-level header and a printed column-numbering row (benchmark v2)
    profiles clean   v1: 10 pt type, roomy cells, mild scans (kept unchanged so v1 results reproduce)
             tough   v2: 7.5 pt type, tight cells, harsh scans; Tamil/Telugu use Latin digits unless
                     native numerals are asked for (they are rare in modern print)
    styles  single        one page
            split         continues on the next page, header not repeated
            split_repeat  continues, header repeated, "(continued)" marker sometimes
            split_cut     as split_repeat, and a description row is cut across the page break
            split_figure  a figure page sits between two parts
            split_blank   a blank page sits between two parts
            two_tables    two different tables on consecutive pages (must stay separate)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from bench import vocab
from src.tables import Cell, Table

KINDS = ("text", "numeric", "mixed", "dates", "merged", "sparse")
KINDS_V2 = KINDS + ("wide",)  # wide: budget-style, 10-13 columns (benchmark v2 only)
STYLES = ("single", "split", "split_repeat", "split_cut", "split_figure", "split_blank", "two_tables")


@dataclass
class PageSpec:
    kind: str  # table | figure | blank
    table: Table | None = None  # what is printed on this page
    logical: int | None = None  # index into Case.truths
    continues: bool = False  # this segment continues the previous table segment
    header_repeated: bool = False
    caption: str | None = None
    note: str | None = None  # a short paragraph printed under the table


@dataclass
class Case:
    id: str
    lang: str
    kind: str
    style: str
    digits: str  # native | latin
    ruled: bool
    pages: list[PageSpec] = field(default_factory=list)
    truths: list[Table] = field(default_factory=list)
    cut: bool = False  # a row really was cut across a page break
    profile: str = "clean"  # clean (v1) | tough (v2)

    def meta(self) -> dict:
        return {"id": self.id, "lang": self.lang, "kind": self.kind, "style": self.style, "digits": self.digits,
                "ruled": self.ruled, "cut": self.cut, "profile": self.profile, "pages": len(self.pages), "page_kinds": [p.kind for p in self.pages],
                "rows": [t.n_rows for t in self.truths], "cols": [t.n_cols for t in self.truths]}


# ---------------------------------------------------------------------------
# Cell content
# ---------------------------------------------------------------------------


def group_indian(digits: str) -> str:
    if len(digits) <= 3:
        return digits
    head, tail, parts = digits[:-3], digits[-3:], []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts + [tail])


def fmt_number(value: float, grouping: str, decimals: int = 2) -> str:
    whole, _, frac = f"{abs(value):.{decimals}f}".partition(".")
    if grouping == "indian":
        whole = group_indian(whole)
    elif grouping == "western":
        whole = f"{int(whole):,}"
    text = whole + ("." + frac if decimals else "")
    return f"({text})" if value < 0 else text


class Builder:
    """Collects cells row by row and tracks the table's grid."""

    def __init__(self) -> None:
        self.cells: list[Cell] = []

    def add(self, r: int, c: int, text: str, *, header: bool = False, rs: int = 1, cs: int = 1) -> None:
        self.cells.append(Cell(r, c, rs, cs, text, header))

    def table(self, n_rows: int, n_cols: int, header_rows: int) -> Table:
        return Table(self.cells, n_rows, n_cols, header_rows)


def _digits(text: str, lang: str, digits: str) -> str:
    return vocab.native_digits(text, lang) if digits == "native" else text


def _amount(rng: random.Random, grouping: str) -> float:
    return round(rng.choice((rng.uniform(5, 900), rng.uniform(900, 99999), rng.uniform(0.5, 60))), 2)


def _date(rng: random.Random) -> str:
    return f"{rng.randint(1, 28):02d}-{rng.randint(1, 12):02d}-{rng.randint(2019, 2027)}"


def build_table(rng: random.Random, lang: str, kind: str, n_rows: int, digits: str, version: str = "v1") -> Table:
    """A logical table of `n_rows` body rows."""
    H = vocab.HEADERS[lang]
    grouping = rng.choice(("indian", "western", "none"))
    d = lambda s: _digits(s, lang, digits)  # noqa: E731
    pool = vocab.words(lang, version)
    word = lambda: rng.choice(pool)  # noqa: E731
    b = Builder()

    def head(labels: list[str]) -> None:
        for c, label in enumerate(labels):
            b.add(0, c, label, header=True)

    if kind == "text":
        head([H[0], H[1], H[2], H[3], H[6]])
        for r in range(1, n_rows + 1):
            b.add(r, 0, d(str(r)))
            b.add(r, 1, " ".join(word() for _ in range(rng.randint(1, 2))))
            b.add(r, 2, word())
            b.add(r, 3, vocab.phrase(rng, lang, 3, 6, version))
            b.add(r, 4, rng.choice(vocab.STATUS[lang]))
        return b.table(n_rows + 1, 5, 1)

    if kind == "numeric":
        n_amt = rng.randint(2, 4)
        head([H[0], H[1]] + [f"{H[4]} {i + 1}" for i in range(n_amt)] + [H[8]])
        sums = [0.0] * n_amt
        for r in range(1, n_rows + 1):
            b.add(r, 0, d(str(r)))
            b.add(r, 1, " ".join(word() for _ in range(rng.randint(1, 2))))
            vals = [_amount(rng, grouping) for _ in range(n_amt)]
            for i, v in enumerate(vals):
                b.add(r, 2 + i, d(fmt_number(v, grouping)))
                sums[i] += v
            b.add(r, 2 + n_amt, d(fmt_number(round(sum(vals), 2), grouping)))
        last = n_rows + 1
        b.add(last, 0, "")
        b.add(last, 1, vocab.TOTAL_WORD[lang])
        for i, s in enumerate(sums):
            b.add(last, 2 + i, d(fmt_number(round(s, 2), grouping)))
        b.add(last, 2 + n_amt, d(fmt_number(round(sum(sums), 2), grouping)))
        return b.table(n_rows + 2, 3 + n_amt, 1)

    if kind in ("mixed", "sparse"):
        head([vocab.CODE[lang], H[1], H[4], H[5], H[6], H[7]])
        sparse = kind == "sparse"
        for r in range(1, n_rows + 1):
            code = rng.choice((f"{rng.randint(0, 9999):04d}", f"{rng.choice('ABCDEFGH')}-{rng.randint(1, 99)}"))
            row = [code, word() + (" " + word() if rng.random() < 0.5 else ""),
                   fmt_number(_amount(rng, grouping), grouping), _date(rng), rng.choice(vocab.STATUS[lang]),
                   vocab.phrase(rng, lang, 2, 4, version) if rng.random() < 0.6 else ""]
            for c, text in enumerate(row):
                if sparse and c > 0 and rng.random() < 0.12:
                    text = rng.choice(("–", "NA", "..", ""))
                if c in (0, 2, 3):
                    text = d(text)
                b.add(r, c, text)
        return b.table(n_rows + 1, 6, 1)

    if kind == "dates":
        head([H[1], H[5] + " 1", H[5] + " 2", H[3]])
        for r in range(1, n_rows + 1):
            b.add(r, 0, " ".join(word() for _ in range(rng.randint(1, 3))))
            b.add(r, 1, d(_date(rng)))
            b.add(r, 2, d(_date(rng)))
            b.add(r, 3, vocab.phrase(rng, lang, 2, 5, version))
        return b.table(n_rows + 1, 4, 1)

    if kind == "merged":
        years = [f"{y}-{str(y + 1)[2:]}" for y in range(rng.randint(2019, 2023), 2027)][:3]
        years = (years + ["2026-27", "2027-28"])[:3]
        b.add(0, 0, H[2], header=True, rs=2)
        b.add(0, 1, H[1], header=True, rs=2)
        b.add(0, 2, f"{H[4]} (₹)", header=True, cs=3)
        b.add(0, 5, H[8], header=True, rs=2)
        for i, y in enumerate(years):
            b.add(1, 2 + i, d(y), header=True)
        r, groups = 2, 0
        while r < n_rows + 2:
            size = min(rng.randint(2, 3), n_rows + 2 - r)
            b.add(r, 0, word(), rs=size)
            for k in range(size):
                b.add(r + k, 1, " ".join(word() for _ in range(rng.randint(1, 2))))
                vals = [_amount(rng, grouping) for _ in range(3)]
                for i, v in enumerate(vals):
                    b.add(r + k, 2 + i, d(fmt_number(v, grouping)))
                b.add(r + k, 5, d(fmt_number(round(sum(vals), 2), grouping)))
            r += size
            groups += 1
        return b.table(n_rows + 2, 6, 2)

    if kind == "wide":
        groups = rng.randint(2, 3)
        first_year = rng.randint(2022, 2024)
        subs = [vocab.HEADERS[lang][4], vocab.STATUS[lang][2], vocab.TOTAL_WORD[lang]]
        n_cols = 3 + 3 * groups + 1
        b.add(0, 0, H[0], header=True, rs=2)
        b.add(0, 1, vocab.CODE[lang], header=True, rs=2)
        b.add(0, 2, H[3], header=True, rs=2)
        for g in range(groups):
            y = first_year + g
            b.add(0, 3 + 3 * g, d(f"{y}-{str(y + 1)[2:]}"), header=True, cs=3)
            for k in range(3):
                b.add(1, 3 + 3 * g + k, subs[k], header=True)
        group_totals = "+".join(d(str(3 + 3 * g + 3)) for g in range(groups))  # printed numbers of the group-total columns
        b.add(0, n_cols - 1, f"{H[8]} ({group_totals})", header=True, rs=2)
        for c in range(n_cols):  # printed column numbers, as in government tables
            b.add(2, c, d(str(c + 1)), header=True)
        for r in range(3, n_rows + 3):
            b.add(r, 0, d(str(r - 2)))
            b.add(r, 1, d(f"{rng.randint(0, 9999):04d}"))
            b.add(r, 2, vocab.phrase(rng, lang, 2, 7, version))
            total = 0.0
            for g in range(groups):
                a1, a2 = _amount(rng, grouping), _amount(rng, grouping)
                if rng.random() < 0.08:
                    a2 = -a2
                dash = rng.random() < 0.08  # "–" printed for nil
                if dash:
                    a1 = 0.0
                vals = (a1, a2, round(a1 + a2, 2))
                for k, v in enumerate(vals):
                    b.add(r, 3 + 3 * g + k, "–" if dash and k == 0 else d(fmt_number(v, grouping)))
                total += vals[2]
            b.add(r, n_cols - 1, d(fmt_number(round(total, 2), grouping)))
        return b.table(n_rows + 3, n_cols, 3)

    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Splitting a logical table over pages
# ---------------------------------------------------------------------------


def _boundaries(t: Table) -> list[int]:
    """Row indexes where a page break may fall (no cell spans across it, enough rows on both sides)."""
    ok = []
    for r in range(t.header_rows + 3, t.n_rows - 2):
        if not any(c.r < r < c.r + c.rowspan for c in t.cells):
            ok.append(r)
    return ok


def _copy(c: Cell, dr: int = 0, rows: int | None = None) -> Cell:
    return Cell(c.r + dr, c.c, c.rowspan if rows is None else min(c.rowspan, rows - c.r), c.colspan, c.text, c.header)


def first_segment(t: Table, end: int) -> Table:
    return Table([_copy(c) for c in t.cells if c.r < end], end, t.n_cols, t.header_rows)


def later_segment(t: Table, start: int, end: int, repeat_header: bool) -> Table:
    hr = t.header_rows if repeat_header else 0
    cells = [_copy(c) for c in t.cells if c.r < hr]
    cells += [_copy(c, hr - start) for c in t.cells if start <= c.r < end]
    return Table(cells, (end - start) + hr, t.n_cols, hr)


def cut_row(rng: random.Random, first: Table, second: Table, desc_col: int) -> bool:
    """Cut the last row's description over the page break: its tail becomes the first row of `second`."""
    last = first.n_rows - 1
    cell = next((c for c in first.cells if c.r == last and c.c == desc_col), None)
    parts = cell.text.split() if cell else []
    if len(parts) < 3:
        return False
    k = rng.randint(1, len(parts) - 1)
    cell.text, tail = " ".join(parts[:k]), " ".join(parts[k:])
    for c in second.cells:  # make room for the tail row
        if c.r >= second.header_rows:
            c.r += 1
    second.n_rows += 1
    second.cells += [Cell(second.header_rows, c, 1, 1, tail if c == desc_col else "", False) for c in range(second.n_cols)]
    return True


def make_case(seed: int, lang: str, kind: str, style: str, *, ruled: bool | None = None, digits: str | None = None,
              profile: str = "clean") -> Case:
    rng = random.Random(seed)
    native_share = 0.7 if profile == "clean" or lang in ("hi", "mr", "bn") else 0.0
    digits = "latin" if lang == "en" else digits or ("native" if rng.random() < native_share else "latin")
    ruled = rng.random() < 0.65 if ruled is None else ruled
    tag = "" if profile == "clean" else "v2-"
    case = Case(f"{tag}{lang}-{kind}-{style}{'-native' if profile != 'clean' and digits == 'native' and lang in ('ta', 'te') else ''}-{seed}",
                lang, kind, style, digits, ruled, profile=profile)
    long_table = style not in ("single", "two_tables")
    if profile == "tough":
        n_rows = (rng.randint(24, 40) if long_table else rng.randint(10, 22)) if kind != "wide" else (rng.randint(20, 30) if long_table else rng.randint(8, 16))
    else:
        n_rows = rng.randint(16, 30) if long_table else rng.randint(5, 14)
    version = "v1" if profile == "clean" else "v2"
    truth = build_table(rng, lang, kind, n_rows, digits, version)
    case.truths.append(truth)

    if style == "single":
        case.pages.append(PageSpec("table", truth, 0, caption=f"Table {rng.randint(1, 20)}"))
        return case

    if style == "two_tables":
        other = build_table(rng, lang, rng.choice([k for k in KINDS if k != kind]), rng.randint(5, 12), digits, version)
        case.truths.append(other)
        note = vocab.phrase(rng, lang, 10, 16, version) if rng.random() < 0.5 else None
        case.pages.append(PageSpec("table", truth, 0, caption="Table 1", note=note))
        case.pages.append(PageSpec("table", other, 1, caption="Table 2"))
        return case

    candidates = _boundaries(truth)
    n_parts = 3 if len(candidates) >= 8 and rng.random() < 0.4 else 2
    cuts = sorted(rng.sample(candidates, n_parts - 1))
    while len(cuts) > 1 and cuts[1] - cuts[0] < 3:  # keep parts at least three rows long
        cuts = sorted(rng.sample(candidates, n_parts - 1))
    edges = [truth.header_rows] + cuts + [truth.n_rows]
    repeat = style != "split" and (style == "split_repeat" or rng.random() < 0.85)
    gap = {"split_figure": "figure", "split_blank": "blank"}.get(style)
    segments = [first_segment(truth, edges[1])]
    segments += [later_segment(truth, a, b, repeat) for a, b in zip(edges[1:], edges[2:])]
    if style == "split_cut":
        case.cut = cut_row(rng, segments[0], segments[1], 3)
    table_no = rng.randint(1, 20)
    for i, seg in enumerate(segments):
        if i and gap:
            case.pages.append(PageSpec(gap))
        contd = i > 0 and repeat and rng.random() < 0.4
        caption = f"Table {table_no}" if i == 0 else f"Table {table_no} (Contd.)" if contd else None
        case.pages.append(PageSpec("table", seg, 0, continues=i > 0, header_repeated=i > 0 and repeat, caption=caption))
    return case
