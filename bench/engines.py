"""Simulated engine readings of a table, with every planted error recorded.

Two readers ("a" ~ Sarvam, "b" ~ Bodhan) each get an independent copy of the truth
with errors planted. A share of the errors (`Profile.shared_frac`) is planted
identically in both: those are the errors that agreement cannot reveal. The real
share is unknown until the real-engine run, so it is a parameter and gets swept.

Character confusions come from real Sarvam-vs-Bodhan differences in the cached runs
(`harvest_confusions`), topped up with common Indic confusions. The two readers also
write merged header cells differently, as the real engines do: "rowspan" keeps
rowspan, "flat" repeats the cell as empty cells.
"""

from __future__ import annotations

import difflib
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from src.config import OUT_DIR
from src.tables import DIGIT_MAP, Cell, Table, parse_number, parse_tables

FALLBACK_CONFUSIONS = [
    ("ण", "न"), ("न", "ण"), ("ि", "ी"), ("ी", "ि"), ("ु", "ू"), ("ू", "ु"), ("ं", ""), ("ा", ""), ("ट", "ठ"),
    ("श", "ष"), ("ष", "श"), ("ब", "व"), ("व", "ब"), ("ढ", "द"), ("த", "ந"), ("ல", "ள"), ("ர", "ற"), ("ன", "ண"),
    ("న", "ణ"), ("ల", "ళ"), ("ে", "ি"), ("ক", "ক্"), ("ো", "ে"), ("ी", ""), ("े", "ै"),
]
# Measured in the real-engine run (2026-10-21): both engines read Tamil numerals as letters and
# Telugu numerals as Latin digits, identically.
SYSTEMATIC_SEEDS = [("௧", "க"), ("௨", "உ"), ("௩", "ங"), ("௪", "ச"), ("௫", "ரு"), ("௬", "சு"), ("௭", "எ"), ("௮", "அ"),
                    ("௯", "கூ")] + [(chr(0x0C66 + i), str(i)) for i in range(10)]
LOOKALIKE = {"3": "8", "8": "3", "1": "7", "7": "1", "5": "6", "6": "5", "0": "9", "9": "0", "2": "7", "4": "9"}


@dataclass
class Profile:
    cell_error: float = 0.04  # per cell: some reader misreads it (before the shared/independent split)
    shared_frac: float = 0.15  # of those, planted identically in both readers
    both_diff_frac: float = 0.0  # of those, both readers wrong in different ways (measured on v2 pages: ~0.2)
    header_error: float = 0.08  # per header cell
    empty_cell: float = 0.005  # a reader leaves a cell empty
    row_drop: float = 0.06  # per table and reader
    row_dup: float = 0.02
    col_merge: float = 0.03
    col_split: float = 0.02
    header_depth: float = 0.03
    systematic_rules: int = 0  # confusion rules drawn per table that hit BOTH readers wherever they apply (v2: 2)
    systematic_hit: float = 0.6  # chance a matching cell is hit by a drawn rule

    def label(self) -> str:
        return f"cell={self.cell_error} shared={self.shared_frac} systematic={self.systematic_rules}"


@dataclass
class Planted:
    kind: str  # char | digit | empty | row_drop | row_dup | col_merge | col_split | header_depth
    row: int | None
    col: int | None
    truth: str
    wrong: str
    engines: str  # a | b | both

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# ---------------------------------------------------------------------------
# Confusions from real disagreements
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def harvest_confusions() -> list[tuple[str, str]]:
    counts: Counter = Counter()
    for doc in sorted((Path(OUT_DIR) / "docs").glob("*/blocks")):
        for sarvam_path in sorted(doc.glob("page_*.sarvam.json")):
            bodhan_path = sarvam_path.with_name(sarvam_path.name.replace(".sarvam.", ".bodhan."))
            if not bodhan_path.is_file():
                continue
            try:
                a = [parse_tables(b["text"])[0] for b in json.loads(sarvam_path.read_text(encoding="utf-8")) if b.get("kind") == "table" and parse_tables(b["text"])]
                b = [parse_tables(x["text"])[0] for x in json.loads(bodhan_path.read_text(encoding="utf-8")) if x.get("kind") == "table" and parse_tables(x["text"])]
            except (OSError, json.JSONDecodeError):
                continue
            for ta, tb in zip(a, b):
                if (ta.n_rows, ta.n_cols) != (tb.n_rows, tb.n_cols):
                    continue
                ga, gb = ta.rows_text(), tb.rows_text()
                for ra, rb in zip(ga, gb):
                    for x, y in zip(ra, rb):
                        if x == y or not x or not y or parse_number(x) is not None or abs(len(x) - len(y)) > 3:
                            continue
                        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, x, y).get_opcodes():
                            u, v = x[i1:i2], y[j1:j2]
                            if tag == "equal" or i2 - i1 > 3 or j2 - j1 > 3 or (not u and not v):
                                continue
                            if any(ch.isspace() for ch in u + v):  # spacing is normalised away, not an OCR error
                                continue
                            counts[(u, v)] += 1
                            counts[(v, u)] += 1
    real = [pair for pair, _ in counts.most_common(80)]
    return real + [p for p in FALLBACK_CONFUSIONS if p not in real]


@lru_cache(maxsize=1)
def systematic_pool() -> list[tuple[str, str]]:
    """Confusions that real engines made against the truth, plus measured shared ones."""
    pool = list(SYSTEMATIC_SEEDS)
    path = Path(__file__).resolve().parent / "results" / "real_confusions.json"
    if path.is_file():
        pool += [(a, b) for a, b, n in json.loads(path.read_text(encoding="utf-8")) if a and n >= 2 and (a, b) not in pool]
    return pool + [p for p in harvest_confusions() if p[0] and p not in pool]


# ---------------------------------------------------------------------------
# Corrupting one cell
# ---------------------------------------------------------------------------


def _digit_value(ch: str) -> str | None:
    mapped = DIGIT_MAP.get(ord(ch))
    return chr(mapped) if mapped is not None and chr(mapped).isdigit() else (ch if ch.isdigit() else None)


def _restore_script(original: str, digit: str) -> str:
    if ord(original) < 128:
        return digit
    zero = ord(original) - int(_digit_value(original) or 0)
    return chr(zero + int(digit))


def corrupt_number(s: str, rng: random.Random) -> str:
    positions = [i for i, ch in enumerate(s) if _digit_value(ch) is not None]
    if not positions:
        return s + "1"
    for _ in range(10):
        roll = rng.random()
        i = rng.choice(positions)
        if roll < 0.6:
            wrong = _restore_script(s[i], LOOKALIKE[_digit_value(s[i])])
            out = s[:i] + wrong + s[i + 1:]
        elif roll < 0.8 and len(positions) > 1:
            j = rng.choice([p for p in positions if p != i])
            lo, hi = sorted((i, j))
            out = s[:lo] + s[hi] + s[lo + 1:hi] + s[lo] + s[hi + 1:]
        else:
            seps = [k for k, ch in enumerate(s) if ch in ",."]
            k = rng.choice(seps) if seps else i
            out = s[:k] + s[k + 1:]
        if out != s:
            return out
    return s + "0"


def corrupt_text(s: str, rng: random.Random) -> str:
    if parse_number(s) is not None:
        return corrupt_number(s, rng)
    pairs = [(a, b) for a, b in harvest_confusions() if a and a in s]
    for _ in range(10):
        roll = rng.random()
        if pairs and roll < 0.7:
            a, b = rng.choice(pairs)
            starts = [i for i in range(len(s)) if s.startswith(a, i)]
            i = rng.choice(starts)
            out = s[:i] + b + s[i + len(a):]
        elif len(s) > 2 and roll < 0.85:
            i = rng.randrange(len(s))
            out = s[:i] + s[i + 1:]
        elif len(s) > 2:
            i = rng.randrange(len(s) - 1)
            out = s[:i] + s[i + 1] + s[i] + s[i + 2:]
        else:
            out = s + s[-1:]
        if out != s and out.strip():
            return out
    return s + "्"


# ---------------------------------------------------------------------------
# Structure edits
# ---------------------------------------------------------------------------


def clone(t: Table) -> Table:
    return Table([Cell(c.r, c.c, c.rowspan, c.colspan, c.text, c.header) for c in t.cells], t.n_rows, t.n_cols, t.header_rows, dict(t.source))


def _plain_rows(t: Table) -> list[int]:
    """Body rows no merged cell starts in or crosses (safe to drop or duplicate)."""
    bad = {r for c in t.cells for r in range(c.r, c.r + c.rowspan) if c.rowspan > 1}
    return [r for r in range(t.header_rows, t.n_rows) if r not in bad]


def _plain_cols(t: Table) -> list[int]:
    bad = {c.c + k for c in t.cells for k in range(c.colspan) if c.colspan > 1 or c.rowspan > 1}
    return [c for c in range(t.n_cols - 1) if c not in bad and c + 1 not in bad]


def drop_row(t: Table, r: int) -> str:
    gone = " | ".join(c.text for c in sorted((c for c in t.cells if c.r == r), key=lambda x: x.c))
    t.cells = [c for c in t.cells if c.r != r]
    for c in t.cells:
        if c.r > r:
            c.r -= 1
    t.n_rows -= 1
    return gone


def dup_row(t: Table, r: int) -> None:
    for c in t.cells:
        if c.r > r:
            c.r += 1
    t.cells += [Cell(r + 1, c.c, 1, c.colspan, c.text, c.header) for c in t.cells if c.r == r]
    t.n_rows += 1


def merge_cols(t: Table, c: int) -> None:
    left = {x.r: x for x in t.cells if x.c == c}
    for x in [x for x in t.cells if x.c == c + 1]:
        if x.r in left:
            left[x.r].text = f"{left[x.r].text} {x.text}".strip()
    t.cells = [x for x in t.cells if x.c != c + 1]
    for x in t.cells:
        if x.c > c + 1:
            x.c -= 1
    t.n_cols -= 1


def split_col(t: Table, c: int) -> None:
    for x in t.cells:
        if x.c > c:
            x.c += 1
    extra = []
    for x in [x for x in t.cells if x.c == c]:
        head, _, tail = x.text.partition(" ")
        x.text = head
        extra.append(Cell(x.r, c + 1, 1, 1, tail, x.header))
    t.cells += extra
    t.n_cols += 1


def to_dialect(t: Table, dialect: str) -> Table:
    """'rowspan' keeps merged cells; 'flat' writes covered slots as empty cells (Sarvam style for headers)."""
    if dialect == "rowspan":
        return t
    out = []
    for c in t.cells:
        out.append(Cell(c.r, c.c, 1, c.colspan, c.text, c.header))
        for dr in range(1, c.rowspan):
            out.append(Cell(c.r + dr, c.c, 1, c.colspan, "", c.header))
    return Table(out, t.n_rows, t.n_cols, t.header_rows, dict(t.source))


# ---------------------------------------------------------------------------
# The simulation
# ---------------------------------------------------------------------------


@dataclass
class Reading:
    a: Table
    b: Table
    planted: list[Planted] = field(default_factory=list)


def simulate(truth: Table, rng: random.Random, profile: Profile) -> Reading:
    a, b = clone(truth), clone(truth)
    log: list[Planted] = []
    for i, cell in enumerate(truth.cells):
        header = cell.r < truth.header_rows
        p = profile.header_error if header else profile.cell_error
        ca, cb = a.cells[i], b.cells[i]
        if cell.text and rng.random() < profile.empty_cell:
            who = rng.choice("ab")
            (ca if who == "a" else cb).text = ""
            log.append(Planted("empty", cell.r, cell.c, cell.text, "", who))
            continue
        if not cell.text or rng.random() >= p:
            continue
        kind = "digit" if parse_number(cell.text) is not None else "char"
        roll = rng.random()
        if roll < profile.shared_frac:
            wrong = corrupt_text(cell.text, rng)
            ca.text = cb.text = wrong
            log.append(Planted(kind, cell.r, cell.c, cell.text, wrong, "both"))
        elif roll < profile.shared_frac + profile.both_diff_frac:
            wa, wb = corrupt_text(cell.text, rng), corrupt_text(cell.text, rng)
            for _ in range(5):
                if wb != wa:
                    break
                wb = corrupt_text(cell.text, rng)
            ca.text, cb.text = wa, wb
            log.append(Planted(kind, cell.r, cell.c, cell.text, f"{wa} | {wb}", "both_different"))
        else:
            who = rng.choice("ab")
            wrong = corrupt_text(cell.text, rng)
            (ca if who == "a" else cb).text = wrong
            log.append(Planted(kind, cell.r, cell.c, cell.text, wrong, who))
    _systematic_errors(truth, a, b, rng, profile, log)
    _structure_errors(a, b, rng, profile, log)
    return Reading(a, b, log)


def _systematic_errors(truth: Table, a: Table, b: Table, rng: random.Random, profile: Profile, log: list[Planted]) -> None:
    """A few confusion rules that both readers apply the same way wherever the pattern occurs."""
    if not profile.systematic_rules:
        return
    texts = [c.text for c in truth.cells if c.text]
    usable = [rule for rule in systematic_pool() if any(rule[0] in t for t in texts)]
    for src, dst in rng.sample(usable, min(profile.systematic_rules, len(usable))):
        for i, cell in enumerate(truth.cells):
            if src in a.cells[i].text and a.cells[i].text == b.cells[i].text and rng.random() < profile.systematic_hit:
                wrong = a.cells[i].text.replace(src, dst)
                a.cells[i].text = b.cells[i].text = wrong
                log.append(Planted("systematic", cell.r, cell.c, cell.text, wrong, "both"))


def _structure_errors(a: Table, b: Table, rng: random.Random, profile: Profile, log: list[Planted]) -> None:
    """Row and column mistakes; a `shared_frac` share hits both readers the same way."""

    def targets() -> list[tuple[str, Table]]:
        if rng.random() < profile.shared_frac:
            return [("a", a), ("b", b)]
        return [rng.choice((("a", a), ("b", b)))]

    def label(chosen: list[tuple[str, Table]]) -> str:
        return "both" if len(chosen) == 2 else chosen[0][0]

    if rng.random() < profile.row_drop:
        chosen = targets()
        rows = set.intersection(*(set(_plain_rows(t)) for _, t in chosen))
        rows -= {max(rows)} if rows else set()
        if len(rows) > 3:
            r = rng.choice(sorted(rows))
            gone = [drop_row(t, r) for _, t in chosen]
            log.append(Planted("row_drop", r, None, gone[0], "", label(chosen)))
    if rng.random() < profile.row_dup:
        chosen = targets()
        rows = set.intersection(*(set(_plain_rows(t)) for _, t in chosen))
        if rows:
            r = rng.choice(sorted(rows))
            for _, t in chosen:
                dup_row(t, r)
            log.append(Planted("row_dup", r, None, "", "duplicated", label(chosen)))
    if rng.random() < profile.col_merge:
        chosen = targets()
        cols = set.intersection(*(set(_plain_cols(t)) for _, t in chosen))
        if cols:
            c = rng.choice(sorted(cols))
            for _, t in chosen:
                merge_cols(t, c)
            log.append(Planted("col_merge", None, c, "", "merged with next", label(chosen)))
    if rng.random() < profile.col_split:
        chosen = targets()
        cols = set.intersection(*(set(_plain_cols(t)) for _, t in chosen))
        if cols:
            c = rng.choice(sorted(cols))
            for _, t in chosen:
                split_col(t, c)
            log.append(Planted("col_split", None, c, "", "split in two", label(chosen)))
    if rng.random() < profile.header_depth:
        chosen = targets()
        delta = rng.choice((-1, 1))
        for _, t in chosen:
            t.header_rows = max(1, t.header_rows + delta)
        log.append(Planted("header_depth", None, None, "", f"header_rows{delta:+d}", label(chosen)))
