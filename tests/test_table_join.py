"""Joining table segments across pages (src/table_join.py). No API calls."""

from __future__ import annotations

import unittest

from src import table_join
from src.tables import Cell, Table


def grid(rows: list[list[str]], header_rows: int = 1) -> Table:
    return Table([Cell(r, c, 1, 1, t, r < header_rows) for r, row in enumerate(rows) for c, t in enumerate(row)],
                 len(rows), len(rows[0]), header_rows)


HEAD = [["No", "District", "Scheme", "Amount"], ["1", "Pune", "Roads and bridges", "10.00"], ["2", "Nashik", "Water", "20.00"],
        ["3", "Latur", "Schools", "30.00"]]


def sig(t: Table, caption: str = "") -> table_join.Signature:
    return table_join.signature(t, None, caption)


class DecideTest(unittest.TestCase):
    def test_continuation_without_header_joins(self):
        a, b = grid(HEAD), grid([["4", "Satara", "Health", "40.00"], ["5", "Sangli", "Water", "50.00"]], 0)
        d = table_join.decide(a, sig(a), b, sig(b))
        self.assertEqual(d.decision, "join")
        self.assertIn("serial numbers continue", d.reasons)

    def test_repeated_header_is_found_and_dropped(self):
        a = grid(HEAD)
        b = grid([HEAD[0], ["4", "Satara", "Health", "40.00"]])
        d = table_join.decide(a, sig(a), b, sig(b))
        self.assertEqual((d.decision, d.drop_rows), ("join", 1))
        joined, _ = table_join.merge(a, b, d)
        self.assertEqual(joined.n_rows, 5)
        self.assertEqual(joined.rows_text()[-1][1], "Satara")

    def test_a_new_table_with_its_own_title_stays_separate(self):
        a = grid(HEAD)
        b = grid([["Year", "Exports", "Imports", "Balance"], ["2024", "10", "12", "-2"]])
        d = table_join.decide(a, sig(a, "Table 3"), b, sig(b, "Table 4"))
        self.assertEqual(d.decision, "separate")
        d2 = table_join.decide(a, sig(a), b, sig(b))
        self.assertNotEqual(d2.decision, "join")  # different header and different content kinds

    def test_unmappable_column_counts_stay_separate(self):
        a, b = grid(HEAD), grid([["4", "Satara", "40.00"]], 0)
        self.assertEqual(table_join.decide(a, sig(a), b, sig(b)).decision, "separate")

    def test_columns_map_by_position_when_ruled(self):
        a = table_join.Signature(4, [""] * 4, ["number", "text", "text", "number"], [0.1, 0.4, 0.8], 0, 1, 3, None)
        b = table_join.Signature(3, [""] * 3, ["number", "text", "number"], [0.1, 0.8], 0, 4, 5, None)
        # B's middle column spans A's columns 1 and 2 (merged): it maps to the one it overlaps most
        self.assertEqual(table_join.map_columns(a, b), {0: 0, 1: 2, 2: 3})


class MergeTest(unittest.TestCase):
    def test_a_text_row_cut_by_the_page_break_is_merged(self):
        a = grid(HEAD)
        b = grid([["", "", "and footpaths", ""], ["4", "Satara", "Health", "40.00"]], 0)
        d = table_join.decide(a, sig(a), b, sig(b))
        joined, notes = table_join.merge(a, b, d)
        self.assertTrue(notes["cut_row_merged"])
        self.assertEqual(joined.rows_text()[3][2], "Schools and footpaths")
        self.assertEqual(joined.n_rows, 5)

    def test_a_seam_row_printed_twice_is_dropped(self):
        a = grid(HEAD)
        b = grid([HEAD[-1], ["4", "Satara", "Health", "40.00"]], 0)
        joined, notes = table_join.merge(a, b, table_join.decide(a, sig(a), b, sig(b)))
        self.assertTrue(notes["duplicate_seam_row_dropped"])
        self.assertEqual([r[0] for r in joined.rows_text()[1:]], ["1", "2", "3", "4"])

    def test_seam_check_flags_a_column_that_changes_kind(self):
        a = grid(HEAD)
        b = grid([["4", "Satara", "Health", "forty"], ["5", "Sangli", "Water", "fifty"]], 0)
        joined, _ = table_join.merge(a, b, table_join.JoinDecision(5, "join", [], 0, {c: c for c in range(4)}))
        self.assertTrue(table_join.seam_checks(joined, a.n_rows, sig(a)))


if __name__ == "__main__":
    unittest.main()
