"""The benchmark's own parts: generator, error planting, metrics. No Chrome, no network."""

from __future__ import annotations

import random
import unittest

from bench import engines, gen, metrics, vocab
from bench.gen import PageSpec
from src.tables import Cell, Table, parse_tables


def grid(rows: list[list[str]], header_rows: int = 1) -> Table:
    cells = [Cell(r, c, 1, 1, text, r < header_rows) for r, row in enumerate(rows) for c, text in enumerate(row)]
    return Table(cells, len(rows), len(rows[0]), header_rows)


ROWS = [["No", "Name", "Amount"], ["1", "Pune", "1,200.50"], ["2", "Nashik", "300.00"], ["3", "Nagpur", "45.10"], ["4", "Latur", "9.00"]]


class GeneratorTest(unittest.TestCase):
    def test_every_style_kind_and_language_builds_a_consistent_case(self):
        for style in gen.STYLES:
            for kind in gen.KINDS:
                for lang in vocab.LANGS:
                    case = gen.make_case(7, lang, kind, style)
                    for page in case.pages:
                        if page.table is not None:
                            self.assertEqual(page.table.to_html().count("<tr>"), page.table.n_rows, case.id)
                            self.assertEqual(len(parse_tables(page.table.to_html())), 1)

    def test_same_seed_gives_the_same_case(self):
        a, b = gen.make_case(11, "hi", "mixed", "split_repeat"), gen.make_case(11, "hi", "mixed", "split_repeat")
        self.assertEqual([p.table.rows_text() for p in a.pages], [p.table.rows_text() for p in b.pages])

    def test_split_parts_add_up_to_the_truth(self):
        case = gen.make_case(5, "en", "numeric", "split_repeat")
        truth = case.truths[0].rows_text()
        header = case.truths[0].header_rows
        body = []
        for i, page in enumerate(case.pages):
            rows = page.table.rows_text()
            body += rows[(page.table.header_rows if i else header):]
        self.assertEqual(body, truth[header:])

    def test_cut_row_tail_rejoins_the_description(self):
        case = gen.make_case(3, "hi", "text", "split_cut")
        self.assertTrue(case.cut)
        first, second = case.pages[0].table.rows_text(), case.pages[1].table.rows_text()
        tail_row = second[case.pages[1].table.header_rows]
        self.assertEqual(tail_row[0], "")
        joined = first[-1][3] + " " + tail_row[3]
        self.assertIn(joined, [r[3] for r in case.truths[0].rows_text()])

    def test_two_tables_are_different_logical_tables(self):
        case = gen.make_case(2, "ta", "text", "two_tables")
        self.assertEqual([p.logical for p in case.pages], [0, 1])
        self.assertEqual(metrics.expected_joins(case), [{"pages": [1, 2], "join": False, "gap": False}])


class SimulationTest(unittest.TestCase):
    def test_shared_errors_are_identical_in_both_readers_and_independent_ones_are_not(self):
        truth = gen.make_case(4, "mr", "mixed", "single").pages[0].table
        reading = engines.simulate(truth, random.Random(1), engines.Profile(cell_error=0.3, shared_frac=0.5, row_drop=0, row_dup=0, col_merge=0, col_split=0, header_depth=0, empty_cell=0))
        kinds = {p.engines for p in reading.planted}
        self.assertEqual(kinds, {"a", "b", "both"})
        for p in reading.planted:
            cell_a = next(c for c in reading.a.cells if (c.r, c.c) == (p.row, p.col))
            cell_b = next(c for c in reading.b.cells if (c.r, c.c) == (p.row, p.col))
            if p.engines == "both":
                self.assertEqual((cell_a.text, cell_b.text), (p.wrong, p.wrong))
            else:
                self.assertNotEqual(cell_a.text, cell_b.text)

    def test_corrupted_values_always_differ(self):
        rng = random.Random(0)
        for text in ("1,23,456.78", "१२३.४५", "घोमन-टांडा की लेन", "Rajasthan", "30-10-2024", "ab"):
            for _ in range(20):
                self.assertNotEqual(engines.corrupt_text(text, rng), text)

    def test_structure_edits_keep_the_table_consistent(self):
        t = engines.clone(grid(ROWS))
        engines.drop_row(t, 2)
        self.assertEqual((t.n_rows, len(parse_tables(t.to_html())[0].rows_text())), (4, 4))
        engines.dup_row(t, 1)
        engines.merge_cols(t, 0)
        self.assertEqual(t.n_cols, 2)
        engines.split_col(t, 0)
        self.assertEqual(t.n_cols, 3)
        self.assertEqual(t.to_html().count("<tr>"), t.n_rows)

    def test_flat_dialect_keeps_the_grid_shape_of_merged_headers(self):
        t = gen.build_table(random.Random(1), "hi", "merged", 6, "native")
        flat = engines.to_dialect(t, "flat")
        self.assertEqual((flat.n_rows, flat.n_cols), (t.n_rows, t.n_cols))
        self.assertTrue(all(c.rowspan == 1 for c in flat.cells))


class MetricsTest(unittest.TestCase):
    def test_perfect_table_scores_one(self):
        s = metrics.score_table(grid(ROWS), grid(ROWS))
        self.assertTrue(s["exact"])
        self.assertEqual((s["correct"], s["cells"]), (15, 15))
        self.assertEqual(s["teds"], 1.0)

    def test_one_wrong_digit_costs_one_cell_not_the_row_match(self):
        wrong = [r[:] for r in ROWS]
        wrong[2][2] = "308.00"
        s = metrics.score_table(grid(ROWS), grid(wrong))
        self.assertEqual((s["correct"], s["rows_missing"], s["rows_extra"]), (14, 0, 0))
        self.assertFalse(s["exact"])
        self.assertEqual(s["wrong"][0]["got"], "308.00")

    def test_dropped_and_extra_rows_are_counted(self):
        s = metrics.score_table(grid(ROWS), grid(ROWS[:2] + ROWS[3:]))
        self.assertEqual((s["rows_missing"], s["rows_extra"]), (1, 0))
        self.assertEqual(s["correct"], 12)
        s = metrics.score_table(grid(ROWS), grid(ROWS + [ROWS[1]]))
        self.assertEqual((s["rows_missing"], s["rows_extra"]), (0, 1))

    def test_merged_column_is_found_by_content_not_scored_as_a_clean_table(self):
        merged = [[r[0] + " " + r[1], r[2]] for r in ROWS]
        s = metrics.score_table(grid(ROWS), grid(merged))
        self.assertFalse(s["exact"])
        self.assertFalse(s["cols_ok"])
        self.assertEqual(s["by_kind"]["number"][0], 4)  # the four amounts were still matched; the serial numbers merged away

    def test_whitespace_and_unicode_form_do_not_count_as_errors(self):
        a = grid([["x"], ["कि"]])
        b = grid([["x"], ["कि "]])  # same text, trailing space
        self.assertTrue(metrics.score_table(a, b)["exact"])

    def test_join_scoring(self):
        case = gen.make_case(9, "en", "text", "split_figure")
        pairs = metrics.expected_joins(case)
        self.assertEqual(len(pairs), 1)
        self.assertTrue(pairs[0]["join"] and pairs[0]["gap"])
        self.assertEqual(metrics.score_joins(case, [[1, 3]])["tp"], 1)
        self.assertEqual(metrics.score_joins(case, [[1], [3]])["fn"], 1)
        two = gen.make_case(2, "ta", "text", "two_tables")
        self.assertEqual(metrics.score_joins(two, [[1, 2]])["fp"], 1)
        self.assertEqual(metrics.score_joins(two, [[1], [2]])["tn"], 1)


if __name__ == "__main__":
    unittest.main()
