"""The free table checker (src/table_qa.py) and image geometry (src/table_geometry.py). No API calls."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src import cache, table_geometry, table_qa
from src.assemble import Assembler
from src.lanes import Settings
from src.run import Run
from src.tables import Cell, Table, parse_tables
from tests.fakes import FakeSarvam, make_pdf


def grid(rows: list[list[str]], header_rows: int = 1) -> Table:
    return Table([Cell(r, c, 1, 1, t, r < header_rows) for r, row in enumerate(rows) for c, t in enumerate(row)],
                 len(rows), len(rows[0]), header_rows)


ROWS = [["No", "District", "Amount"], ["1", "Pune", "1,200.50"], ["2", "Nashik", "300.00"],
        ["3", "Nagpur", "45.10"], ["4", "Latur", "9.00"], ["5", "Satara", "12.25"]]


def edited(rows, r, c, text):
    out = [row[:] for row in rows]
    out[r][c] = text
    return out


class CanonTest(unittest.TestCase):
    def test_cosmetic_differences_vanish(self):
        self.assertEqual(table_qa.canon("–"), table_qa.canon("-"))
        self.assertEqual(table_qa.canon("$-2$"), "-2")
        self.assertEqual(table_qa.canon(" Pune\nCity "), "Pune City")
        self.assertEqual(table_qa.canon("श्रीीक्षेेत्र"), table_qa.canon("श्रीक्षेत्र"))  # doubled matras from a text layer
        self.assertNotEqual(table_qa.canon("1,200.50"), table_qa.canon("1,200.80"))

    def test_layer_matches_numbers_as_whole_tokens(self):
        layer = table_qa.Layer.of("11,234.50 Pune 300.00")
        self.assertFalse(layer.contains("1,234.50"))  # a substring of another number is not a match
        self.assertTrue(layer.contains("300.00"))
        self.assertTrue(layer.contains("Pune"))
        self.assertIsNone(layer.contains("Pu"))  # too short to say


class CheckTest(unittest.TestCase):
    def test_agreement_is_verified(self):
        qt = table_qa.check({"sarvam": grid(ROWS), "bodhan": grid(ROWS)})
        self.assertEqual(qt.status, "verified")
        self.assertEqual(qt.questions, [])

    def test_dash_difference_is_cosmetic(self):
        qt = table_qa.check({"sarvam": grid(edited(ROWS, 2, 2, "–")), "bodhan": grid(edited(ROWS, 2, 2, "-"))})
        self.assertEqual(qt.status, "verified")
        self.assertEqual(qt.counts()["decided_by"].get("cosmetic"), 1)

    def test_disagreement_without_evidence_becomes_a_question(self):
        qt = table_qa.check({"sarvam": grid(edited(ROWS, 3, 1, "Nagpor")), "bodhan": grid(ROWS)})
        self.assertEqual(qt.status, "needs_review")
        q = qt.questions[0]
        self.assertEqual((q["kind"], q["r"], q["c"]), ("cell", 3, 1))
        self.assertEqual(set(q["candidates"]), {"Nagpor", "Nagpur"})

    def test_proven_text_layer_decides_a_text_cell(self):
        layer = "No District Amount 1 Pune 1,200.50 2 Nashik 300.00 3 Nagpur 45.10 4 Latur 9.00 5 Satara 12.25"
        qt = table_qa.check({"sarvam": grid(edited(ROWS, 3, 1, "Nagpor")), "bodhan": grid(ROWS)},
                            table_qa.Evidence(layer_text=layer))
        self.assertTrue(qt.layer["proven"])
        self.assertEqual(qt.table.rows_text()[3][1], "Nagpur")
        self.assertEqual(qt.status, "repaired")

    def test_unproven_text_layer_is_ignored(self):
        garbage = "ŀŀ ¥ ŉ Ď 2 ʡ"  # a legacy-font layer
        qt = table_qa.check({"sarvam": grid(edited(ROWS, 3, 1, "Nagpor")), "bodhan": grid(ROWS)},
                            table_qa.Evidence(layer_text=garbage))
        self.assertFalse(qt.layer["proven"])
        self.assertEqual(qt.status, "needs_review")

    def test_printed_formula_decides_only_when_it_holds(self):
        rows = [["Item", "A", "B", "Total (2+3)"], ["1", "2", "3", "4"], ["x", "10.00", "5.00", "15.00"],
                ["y", "1.00", "2.00", "3.00"], ["z", "4.00", "4.00", "8.00"]]
        qt = table_qa.check({"sarvam": grid(edited(rows, 2, 1, "16.00"), 2), "bodhan": grid(rows, 2)})
        self.assertEqual(qt.table.rows_text()[2][1], "10.00")
        self.assertEqual(next(c for c in qt.cells if (c.r, c.c) == (2, 1)).decided_by, "formula")
        both_wrong = table_qa.check({"sarvam": grid(edited(rows, 2, 1, "16.00"), 2), "bodhan": grid(edited(rows, 2, 1, "11.00"), 2)})
        self.assertTrue(next(c for c in both_wrong.cells if (c.r, c.c) == (2, 1)).open)  # neither makes 15.00 hold

    def test_shared_lookalike_numeral_is_not_verified(self):
        rows = [["எண்", "பெயர்"], ["க", "சென்னை"], ["உ", "மதுரை"], ["ங", "சேலம்"]]  # ௧ ௨ ௩ read as letters by both
        qt = table_qa.check({"sarvam": grid(rows), "bodhan": grid(rows)})
        self.assertEqual(qt.status, "needs_review")
        self.assertTrue(any(q["kind"] == "audit" and "lookalike_numeral" in q["risk"] for q in qt.questions))

    def test_dropped_row_is_aligned_and_flagged_single(self):
        short = ROWS[:3] + ROWS[4:]
        qt = table_qa.check({"sarvam": grid(short), "bodhan": grid(ROWS)})
        self.assertEqual(qt.structure["decision"], "readers_differ")
        self.assertEqual(qt.status, "needs_review")
        qt = table_qa.check({"sarvam": grid(ROWS), "bodhan": grid(short)})
        self.assertEqual(qt.structure["rows_matched"]["bodhan"], 5)

    def test_phantom_columns_are_squeezed_and_numbering_decides(self):
        rows = [["No", "Code", "Name", "Total"], ["1", "2", "3", "4"], ["1", "0956", "Pune", "10.00"],
                ["2", "6625", "Nashik", "20.00"], ["3", "1111", "Latur", "30.00"]]
        good = grid(rows, 2)
        bad_rows = [r[:2] + [""] + r[2:] + [""] for r in rows]  # two empty, headerless columns
        bad = grid(bad_rows, 2)
        qt = table_qa.check({"sarvam": good, "bodhan": bad})
        self.assertEqual(qt.structure["phantom_columns_removed"], {"bodhan": [2, 5]})
        self.assertEqual(qt.structure["decision"], "agreed")
        self.assertEqual(qt.status, "verified")
        shifted = grid([r[:1] + r[2:] for r in rows], 2)  # merged away a column: 3 columns, numbering row 1 3 4
        qt = table_qa.check({"sarvam": shifted, "bodhan": good})
        self.assertEqual((qt.structure["decision"], qt.reference), ("decided_by_numbering", "bodhan"))

    def test_image_rules_decide_structure(self):
        g = table_geometry.Geometry(True, 6, 3, "rules", 0.0)
        qt = table_qa.check({"sarvam": grid(ROWS[:5]), "bodhan": grid(ROWS)}, table_qa.Evidence(geometry=g))
        self.assertEqual((qt.structure["decision"], qt.reference), ("decided_by_image", "bodhan"))
        both_short = table_qa.check({"sarvam": grid(ROWS[:5]), "bodhan": grid(ROWS[:5])}, table_qa.Evidence(geometry=g))
        self.assertEqual(both_short.structure["decision"], "conflict")  # a row both readers dropped
        self.assertEqual(both_short.status, "needs_review")


def drawn_table(rows: int, cols: int, skew: float = 0.0, headlines: bool = True) -> np.ndarray:
    """A white image with a ruled grid and, in every cell, a text-like blob topped by a headline stroke."""
    h, w, cw, rh = 40 + rows * 30, 40 + cols * 120, 120, 30
    img = np.full((h, w), 255, dtype=np.uint8)
    for r in range(rows + 1):
        img[20 + r * rh: 22 + r * rh, 20: 20 + cols * cw + 2] = 0
    for c in range(cols + 1):
        img[20: 20 + rows * rh + 2, 20 + c * cw: 22 + c * cw] = 0
    for r in range(rows):
        for c in range(cols):
            y, x = 20 + r * rh + 9, 20 + c * cw + 12
            if headlines:
                img[y: y + 2, x: x + 90] = 0  # like the Devanagari headline over a word
            img[y + 2: y + 12, x: x + 90: 7] = 0  # stems
    if skew:
        img = table_geometry._rotate(img, skew)
    return img


class GeometryTest(unittest.TestCase):
    def test_ruled_grid_is_counted_and_headlines_are_not_rules(self):
        g = table_geometry.analyse(drawn_table(7, 4))
        self.assertEqual((g.ruled, g.rows, g.cols), (True, 7, 4))
        self.assertGreater(g.cell_ink(0, 0, 1, 1), 0.02)

    def test_skewed_scan_is_straightened(self):
        g = table_geometry.analyse(drawn_table(9, 5, skew=1.5))
        self.assertEqual((g.rows, g.cols), (9, 5))
        self.assertAlmostEqual(abs(g.skew), 1.5, delta=0.25)

    def test_unruled_text_is_not_mistaken_for_a_grid(self):
        img = np.full((300, 600), 255, dtype=np.uint8)
        for y in range(20, 280, 25):
            for x0 in (20, 220, 420):
                img[y: y + 2, x0: x0 + 150] = 0  # headline strokes, columns apart
        self.assertFalse(table_geometry.analyse(img).confident)

    def test_column_line_broken_by_merged_cells_still_counts(self):
        img = drawn_table(12, 4)
        for r in (2, 3, 5, 7, 9):  # rows where columns 1 and 2 are merged: their separator is missing
            img[20 + r * 30 + 3: 20 + (r + 1) * 30 - 1, 260: 262] = 255
        self.assertEqual(len(table_geometry.vertical_rules(img <= 128)), 4)  # too short for the long-line test
        g = table_geometry.analyse(img)
        self.assertEqual((g.rows, g.cols), (12, 4))

    def test_note_row_inside_the_frame_is_not_a_table_row(self):
        img = drawn_table(6, 4)
        y0 = 20 + 5 * 30
        img[y0 + 3: y0 + 30, 22: 20 + 4 * 120] = 255  # the last row becomes one wide cell ...
        img[y0 + 10: y0 + 20, 40: 300: 5] = 0  # ... holding a note
        g = table_geometry.analyse(img)
        self.assertEqual((g.rows, g.cols), (5, 4))

    def test_a_box_that_cuts_off_the_table_side_is_detected(self):
        img = drawn_table(5, 4)[:, 60:]  # the left border and part of the first column cut away
        g = table_geometry.analyse(img)
        self.assertTrue(table_geometry.clipped(g))
        self.assertFalse(table_geometry.clipped(table_geometry.analyse(drawn_table(5, 4))))


class AssembleStandardTest(unittest.TestCase):
    def test_standard_mode_records_status_and_questions(self):
        table_html = ("<table><tr><th>No</th><th>Name</th><th>Amount</th></tr>"
                      "<tr><td>1</td><td>Pune</td><td>10</td></tr><tr><td>2</td><td>Latur</td><td>20</td></tr></table>")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.object(cache, "CACHE_DIR", root / "cache"):
                pdf = make_pdf(root / "one.pdf", 1)
                run = Run(pdf, Settings(language="en-IN", policy="sarvam-only"), out_root=root / "out",
                          sarvam=FakeSarvam(page_tables={1: (table_html, [0.1, 0.3, 0.9, 0.6])}), progress_every=999, verbose=False)
                run.prepare()
                out = Path(run.execute()["out_dir"])
                result = Assembler(out, agent=None, reocr=False, verbose=False, qa_mode="standard").run()
                meta = result["tables"]["list"][0]
                self.assertIn(meta["qa"]["status"], ("verified", "repaired", "needs_review"))
                self.assertEqual(meta["qa"]["reference"], "sarvam")
                saved = json.loads((out / "tables" / f"{meta['id']}.json").read_text())
                self.assertIn("qa", saved)
                self.assertEqual(parse_tables((out / "tables" / f"{meta['id']}.html").read_text())[0].rows_text()[1][1], "Pune")


if __name__ == "__main__":
    unittest.main()


class SecondReadingTest(unittest.TestCase):
    def test_a_table_the_other_engine_wrote_inside_a_text_block_is_found_and_keeps_the_primary_box(self):
        from src.blocks import Block
        from src.assemble import table_in_text
        from src.tables import parse_tables

        html_a = "<table><tr><th>No</th><th>Name</th></tr><tr><td>1</td><td>Pune</td></tr><tr><td>2</td><td>Latur</td></tr></table>"
        html_b = "<table><tr><th>Code</th></tr><tr><td>X1</td></tr></table>"
        mine = parse_tables(html_a)[0]
        primary = Block(1, "sarvam", 0, "table", None, "table", (0.1, 0.1, 0.9, 0.3), html_a)
        page_body = Block(1, "bodhan", 3, "text", None, "List", (0.05, 0.05, 0.95, 0.9), "intro\n" + html_b + "\n" + html_a)
        found = table_in_text([page_body], primary, mine)
        self.assertIsNotNone(found)
        block, table = found
        self.assertEqual(block.bbox, primary.bbox)  # the crop stays on the table, not the page body
        self.assertEqual(table.rows_text()[2], ["2", "Latur"])  # the table that holds the reference's cells, not the other one
        self.assertTrue(table.source.get("in_text_block"))
        small = Block(1, "bodhan", 3, "text", None, "List", (0.0, 0.0, 0.2, 0.2), html_a)
        self.assertIsNone(table_in_text([small], primary, mine))  # a block that does not cover the table

    def test_a_table_with_one_reader_is_never_verified(self):
        rows = [["No", "Name"], ["1", "Pune"], ["2", "Latur"], ["3", "Satara"]]
        qt = table_qa.check({"sarvam": grid(rows)})
        self.assertEqual(qt.status, "needs_review")
        self.assertEqual([q["reason"] for q in qt.questions if q["kind"] == "structure"], ["single_reading"])
        both = table_qa.check({"sarvam": grid(rows), "bodhan": grid(rows)})
        self.assertEqual(both.status, "verified")


class JudgeRuleTest(unittest.TestCase):
    """The image-check rule, with a fake model (no API calls)."""

    def _settle(self, readings_by_model: list[str], a: str, b: str, scanned: bool = False):
        from src import table_judge

        rows = [["No", "Name"], ["1", a], ["2", "Latur"], ["3", "Satara"]]
        other = [r[:] for r in rows]
        other[1][1] = b
        qt = table_qa.check({"sarvam": grid(rows), "bodhan": grid(other)})

        class FakeCutter:
            exact_cols = True

            def __init__(self, *args):
                pass

        answers = iter([{"Q1": r} for r in readings_by_model])
        with mock.patch.object(table_judge, "Cutter", FakeCutter), \
             mock.patch.object(table_judge, "_ask", lambda *args, **kw: next(answers)):
            report = table_judge.settle(None, None, 1, 0, (0, 0, 1, 1), qt, scanned)
        return qt, report

    def test_born_digital_accepts_one_model_matching_an_engine(self):
        qt, report = self._settle(["Pune"], "Pune", "Pone")
        self.assertEqual((report.route, report.decided, qt.table.rows_text()[1][1]), ("born-digital", 1, "Pune"))

    def test_a_reading_no_engine_produced_is_not_accepted(self):
        qt, report = self._settle(["Puna"], "Pune", "Pone")
        self.assertEqual(report.decided, 0)
        self.assertTrue(next(c for c in qt.cells if (c.r, c.c) == (1, 1)).open)

    def test_scanned_ruled_needs_two_models_to_agree(self):
        _, report = self._settle(["Pune", "Pone"], "Pune", "Pone", scanned=True)
        self.assertEqual((report.route, report.decided), ("scanned-ruled", 0))
        _, report = self._settle(["Pune", "Pune"], "Pune", "Pone", scanned=True)
        self.assertEqual(report.decided, 1)

    def test_printed_text_is_never_deleted_on_a_model_word(self):
        _, report = self._settle([""], "", "एकूण (क+ख+ग+घ)")
        self.assertEqual(report.decided, 0)


class LeanStage2Test(unittest.TestCase):
    """Cost and latency changes: minor differences settle free, the cascade asks fewer questions."""

    def test_punctuation_only_difference_is_minor_but_numbers_are_not(self):
        self.assertTrue(table_qa.minor_difference(["21-मई,-2025", "21-मई.-2025"]))
        self.assertTrue(table_qa.minor_difference(["2/4 लेन", "2/4लेन"]))
        self.assertFalse(table_qa.minor_difference(["12.50", "1250"]))  # a lost decimal point is an error
        self.assertFalse(table_qa.minor_difference(["Pune", "Pone"]))
        rows = [["No", "Date"], ["1", "21-मई,-2025"], ["2", "x"], ["3", "y"]]
        other = [r[:] for r in rows]
        other[1][1] = "21-मई.-2025"
        qt = table_qa.check({"sarvam": grid(rows), "bodhan": grid(other)})
        cell = next(c for c in qt.cells if (c.r, c.c) == (1, 1))
        self.assertEqual((cell.decided_by, qt.status), ("minor", "repaired"))

    def test_cascade_skips_the_second_model_when_the_first_cannot_decide(self):
        from src import table_judge

        rows = [["No", "Name"], ["1", "Pune"], ["2", "Latur"], ["3", "Satara"], ["4", "Sangli"]]
        other = [r[:] for r in rows]
        other[1][1], other[2][1] = "Pone", "Lator"
        qt = table_qa.check({"sarvam": grid(rows), "bodhan": grid(other)})

        class FakeCutter:
            exact_cols = True

            def __init__(self, *args):
                pass

        asked: list[int] = []

        def fake_ask(agent, cutter, qt_, cells, style, model, report, profile=None):
            asked.append(len(cells))
            if len(asked) == 1:  # first model: Q1 matches an engine, Q2 matches nothing
                return {"Q1": "Pune", "Q2": "Latoor"}
            return {"Q1": "Pune"}

        with mock.patch.object(table_judge, "Cutter", FakeCutter), mock.patch.object(table_judge, "_ask", fake_ask):
            report = table_judge.settle(None, None, 1, 0, (0, 0, 1, 1), qt, scanned=True)
        self.assertEqual(asked, [2, 1])  # the second model was asked about one cell, not two
        self.assertEqual((report.decided, qt.table.rows_text()[1][1]), (1, "Pune"))

    def test_think_two_of_three_on_scans_asks_the_third_model_only_where_needed(self):
        from src import modes, table_judge

        rows = [["No", "Name"], ["1", "Pune"], ["2", "Latur"], ["3", "Satara"], ["4", "Sangli"]]
        other = [r[:] for r in rows]
        other[1][1], other[2][1] = "Pone", "Lator"
        qt = table_qa.check({"sarvam": grid(rows), "bodhan": grid(other)})

        class FakeCutter:
            exact_cols = True

            def __init__(self, *args):
                pass

        replies = iter([{"Q1": "Pune", "Q2": "Latur"},  # Flash-lite
                        {"Q1": "Pune", "Q2": "Lator"},  # Luna: agrees on Q1 only
                        {"Q1": "Latur"}])  # the third model sees only Q2 (renumbered Q1)
        asked: list[int] = []

        def fake_ask(agent, cutter, qt_, cells, style, model, report, profile=None):
            asked.append(len(cells))
            return next(replies)

        think = modes.get("think").judge
        with mock.patch.object(table_judge, "Cutter", FakeCutter), mock.patch.object(table_judge, "_ask", fake_ask):
            report = table_judge.settle(None, None, 1, 0, (0, 0, 1, 1), qt, scanned=True, profile=think)
        self.assertEqual(asked, [2, 2, 1])
        self.assertEqual(report.decided, 2)
        self.assertEqual(qt.table.rows_text()[1][1:], ["Pune"])
        self.assertEqual(qt.table.rows_text()[2][1], "Latur")
        q2 = next(c for c in qt.cells if (c.r, c.c) == (2, 1))
        self.assertEqual(q2.decided_by, "image:gemini-3.1-flash-lite+gemini-3.5-flash-lite")
