"""Scoring a predicted table against the truth, and join decisions against the page structure.

Cell accuracy needs rows and columns to be matched first, because a dropped row or a
merged column shifts everything below it. Rows are matched by a similarity-weighted
alignment (so one wrong digit does not break the match); columns are matched by
content only when the counts differ. TEDS (tree edit distance over rows and cells)
is reported for tables small enough to compute it.
"""

from __future__ import annotations

import unicodedata
from difflib import SequenceMatcher

from apted import APTED, Config

from bench.gen import Case
from src.tables import Table, parse_number

MIN_ROW_SIM = 0.45
TEDS_MAX_NODES = 260


_DASHES = str.maketrans({"–": "-", "—": "-", "−": "-", "‐": "-", "‒": "-", "’": "'", "‘": "'", "“": '"', "”": '"'})


def norm(s: str) -> str:
    """Scoring form: NFC, one kind of dash and quote, whitespace collapsed (a dash read as a hyphen is not an error)."""
    return " ".join(unicodedata.normalize("NFC", s).translate(_DASHES).split())


def sim(a: str, b: str) -> float:
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def align_rows(truth: list[str], pred: list[str]) -> list[tuple[int, int]]:
    """Matched (truth row, predicted row) pairs, in order, maximising total similarity."""
    n, m = len(truth), len(pred)
    s = [[sim(truth[i], pred[j]) for j in range(m)] for i in range(n)]
    best = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            take = s[i][j] + best[i + 1][j + 1] if s[i][j] >= MIN_ROW_SIM else -1.0
            best[i][j] = max(take, best[i + 1][j], best[i][j + 1])
    pairs, i, j = [], 0, 0
    while i < n and j < m:
        if s[i][j] >= MIN_ROW_SIM and abs(best[i][j] - (s[i][j] + best[i + 1][j + 1])) < 1e-9:
            pairs.append((i, j))
            i, j = i + 1, j + 1
        elif best[i + 1][j] >= best[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def align_cols(tg: list[list[str]], pg: list[list[str]], pairs: list[tuple[int, int]]) -> dict[int, int]:
    nt, np_ = len(tg[0]), len(pg[0])
    if nt == np_:
        return {c: c for c in range(nt)}
    scores = []
    for c in range(nt):
        for d in range(np_):
            vals = [sim(norm(tg[i][c]), norm(pg[j][d])) for i, j in pairs[:30]]
            scores.append((sum(vals) / max(1, len(vals)), c, d))
    mapping: dict[int, int] = {}
    used: set[int] = set()
    for score, c, d in sorted(scores, reverse=True):
        if score >= 0.4 and c not in mapping and d not in used:
            mapping[c] = d
            used.add(d)
    return mapping


class _Node:
    __slots__ = ("tag", "cs", "rs", "text", "children")

    def __init__(self, tag: str, cs: int = 1, rs: int = 1, text: str = "") -> None:
        self.tag, self.cs, self.rs, self.text, self.children = tag, cs, rs, text, []


class _Cfg(Config):
    def __init__(self, content: bool) -> None:
        self.content = content

    def rename(self, a: _Node, b: _Node) -> float:
        if a.tag != b.tag:
            return 1.0
        if a.tag == "td":
            if (a.cs, a.rs) != (b.cs, b.rs):
                return 1.0
            return 1.0 - sim(a.text, b.text) if self.content else 0.0
        return 0.0

    def children(self, node: _Node) -> list:
        return node.children


def _tree(t: Table) -> tuple[_Node, int]:
    root, count = _Node("table"), 1
    for r in range(t.n_rows):
        row = _Node("tr")
        for c in sorted((c for c in t.cells if c.r == r), key=lambda x: x.c):
            row.children.append(_Node("td", c.colspan, c.rowspan, norm(c.text)))
            count += 1
        root.children.append(row)
        count += 1
    return root, count


def teds(truth: Table, pred: Table, content: bool = True) -> float | None:
    a, na = _tree(truth)
    b, nb = _tree(pred)
    if max(na, nb) > TEDS_MAX_NODES:
        return None
    return round(1.0 - APTED(a, b, _Cfg(content)).compute_edit_distance() / max(na, nb), 4)


def score_table(truth: Table, pred: Table | None) -> dict:
    """Cell accuracy by cell type, row and column bookkeeping, TEDS."""
    if pred is None or not pred.n_rows:
        return {"missing_table": True, "cells": len(truth.cells), "correct": 0, "exact": False,
                "_ok": {(c.r, c.c): (False, None) for c in truth.cells}}
    tg = [[norm(x) for x in row] for row in truth.rows_text()]
    pg = [[norm(x) for x in row] for row in pred.rows_text()]
    pairs = align_rows([" | ".join(r) for r in tg], [" | ".join(r) for r in pg])
    cols = align_cols(tg, pg, pairs)
    row_of = dict(pairs)
    stats = {k: [0, 0] for k in ("header", "text", "number", "empty")}
    wrong: list[dict] = []
    ok_map: dict[tuple[int, int], tuple[bool, str | None]] = {}
    found = filled = 0  # lenient: the text sits in the matched row, in any column
    for cell in truth.cells:
        kind = ("header" if cell.r < truth.header_rows else "empty" if not cell.text.strip()
                else "number" if parse_number(cell.text) is not None else "text")
        j, d = row_of.get(cell.r), cols.get(cell.c)
        got = pg[j][d] if j is not None and d is not None else None
        ok = got is not None and got == norm(cell.text)
        stats[kind][0] += ok
        stats[kind][1] += 1
        ok_map[(cell.r, cell.c)] = (ok, got)
        if kind != "empty":
            filled += 1
            found += j is not None and norm(cell.text) in pg[j]
        if not ok:
            wrong.append({"row": cell.r, "col": cell.c, "kind": kind, "truth": norm(cell.text), "got": got})
    total = sum(v[1] for v in stats.values())
    correct = sum(v[0] for v in stats.values())
    missing = len(tg) - len(pairs)
    extra = len(pg) - len(pairs)
    cols_ok = len(tg[0]) == len(pg[0]) and len(cols) == len(tg[0])
    return {
        "cells": total, "correct": correct, "filled": filled, "found": found,
        "by_kind": {k: v for k, v in stats.items()},
        "rows_truth": len(tg), "rows_pred": len(pg), "rows_missing": missing, "rows_extra": extra,
        "cols_truth": len(tg[0]), "cols_pred": len(pg[0]), "cols_ok": cols_ok,
        "exact": correct == total and not missing and not extra and cols_ok,
        "teds": teds(truth, pred), "teds_s": teds(truth, pred, content=False),
        "wrong": wrong[:40],
        "_ok": ok_map,
    }


# ---------------------------------------------------------------------------
# Joins
# ---------------------------------------------------------------------------


def expected_joins(case: Case) -> list[dict]:
    """Consecutive table-bearing pages: should they be one table?"""
    table_pages = [(i + 1, p) for i, p in enumerate(case.pages) if p.kind == "table"]
    out = []
    for (pa, a), (pb, b) in zip(table_pages, table_pages[1:]):
        out.append({"pages": [pa, pb], "join": a.logical == b.logical, "gap": pb - pa > 1})
    return out


def score_joins(case: Case, final_pages: list[list[int]]) -> dict:
    """Compare joins the system made (pages that ended up in one table) with the expected ones."""
    joined = set()
    for pages in final_pages:
        for a, b in zip(pages, pages[1:]):
            joined.add((a, b))
    tp = fp = fn = tn = 0
    detail = []
    for e in expected_joins(case):
        got = tuple(e["pages"]) in joined
        tp += e["join"] and got
        fn += e["join"] and not got
        fp += (not e["join"]) and got
        tn += (not e["join"]) and not got
        detail.append({**e, "got": got})
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "pairs": detail}
