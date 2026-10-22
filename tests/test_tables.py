import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src import cache
from src.assemble import Assembler, format_like
from src.lanes import Settings
from src.run import Run
from src.tables import (
    check_table,
    continuation,
    failed,
    fuse,
    join,
    normalize_digits,
    parse_number,
    parse_tables,
)
from tests.fakes import FakeSarvam, make_pdf

# Header rows, column-numbering row and formulas as printed in Maharashtra's fiscal tables.
MARATHI_TABLE = """<table><thead>
<tr><th>प्रधान शीर्ष</th><th>वर्णन</th><th>० ते १ वर्ष</th><th>१ ते २ वर्ष</th><th>एकूण<br/>(३ ते ४)</th></tr>
<tr><th>१</th><th>२</th><th>३</th><th>४</th><th>५</th></tr></thead>
<tbody>
<tr><td></td><td>(क) कर</td><td>७२.७१</td><td>९५.५४</td><td>१६८.२५</td></tr>
<tr><td rowspan="2">००४९</td><td>(एक) व्याज</td><td>७०.००</td><td>९०.००</td><td>१६०.००</td></tr>
<tr><td>(दोन) लाभांश</td><td>२.७१</td><td>५.५४</td><td>८.२५</td></tr>
<tr><td></td><td>(ख) सेवा</td><td>१०.००</td><td>२०.००</td><td>३०.००</td></tr>
<tr><td></td><td>एकूण (क+ख)</td><td>८२.७१</td><td>११५.५४</td><td>१९८.२५</td></tr>
</tbody></table>"""


def table():
    return parse_tables(MARATHI_TABLE)[0]


class ParseTest(unittest.TestCase):
    def test_digits_and_numbers(self):
        self.assertEqual(normalize_digits("१२९६.५७ ௧௨ ১২"), "1296.57 12 12")
        self.assertEqual(parse_number("14,87,240"), 1487240)
        self.assertEqual(parse_number("(12.5)"), -12.5)
        self.assertIsNone(parse_number("रु. (लाख)"))

    def test_grid_expands_rowspan(self):
        t = table()
        self.assertEqual((t.n_rows, t.n_cols, t.header_rows), (7, 5, 2))
        rows = t.rows_text()
        self.assertEqual(rows[3][0], "००४९")
        self.assertEqual(rows[4][0], "००४९")  # covered by rowspan
        self.assertEqual(rows[4][1], "(दोन) लाभांश")
        self.assertIn('rowspan="2"', t.to_html())
        again = parse_tables(t.to_html())[0]
        self.assertEqual(again.rows_text(), rows)

    def test_markdown_table(self):
        t = parse_tables("| a | b |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |")[0]
        self.assertEqual((t.n_rows, t.n_cols, t.header_rows), (3, 2, 1))


class CheckTest(unittest.TestCase):
    def test_formulas_pass_on_a_correct_table(self):
        checks = check_table(table())
        kinds = {c.kind for c in checks}
        self.assertEqual(kinds, {"row_formula", "col_formula"})
        self.assertEqual(failed(checks), [])

    def test_misread_total_is_flagged_at_the_right_cell(self):
        t = table()
        for cell in t.cells:
            if cell.text == "१६८.२५":
                cell.text = "१६१.२५"
        bad = failed(check_table(t))
        self.assertEqual([(c.kind, c.row, c.col, c.expected) for c in bad][0], ("row_formula", 2, 4, 168.25))

    def test_consensus_total_column(self):
        t = parse_tables(
            "<table><tr><th>Item</th><th>A</th><th>B</th><th>Total</th></tr>"
            "<tr><td>x</td><td>1</td><td>2</td><td>3</td></tr><tr><td>y</td><td>4</td><td>5</td><td>9</td></tr>"
            "<tr><td>z</td><td>2</td><td>2</td><td>7</td></tr></table>"
        )[0]
        checks = check_table(t)
        self.assertEqual(sum(c.ok for c in checks), 2)
        self.assertEqual([(c.row, c.found) for c in failed(checks)], [(3, 7.0)])

    def test_product_rule_skips_serial_and_copied_columns(self):
        # Like the Union budget annuity table: total = annual x years; "remaining"
        # copies "total" until payments start, so it must not get a rule of its own.
        rows = [(1, "24.72", "15.00", "370.80", "370.80"), (2, "57.44", "15.00", "861.60", "861.60"),
                (3, "13.40", "15.00", "201.00", "145.96"), (4, "40.97", "15.00", "614.55", "614.55"),
                (5, "82.67", "15.00", "1240.05", "977.10"), (6, "18.71", "15.00", "280.65", "280.65")]
        html_rows = "".join(f"<tr><td>{s}</td><td>p{s}</td><td>{a}</td><td>{y}</td><td>{t}</td><td>{r}</td></tr>"
                            for s, a, y, t, r in rows)
        t = parse_tables("<table><tr><th>No</th><th>Project</th><th>Annual</th><th>Years</th><th>Total</th>"
                         f"<th>Remaining</th></tr>{html_rows}</table>")[0]
        checks = check_table(t)
        self.assertEqual({(c.kind, c.col) for c in checks}, {("row_product", 4)})
        self.assertEqual(failed(checks), [])
        t.cells[[c.text for c in t.cells].index("861.60")].text = "816.60"  # transposed digits
        self.assertEqual([(c.row, c.expected) for c in failed(check_table(t))], [(2, 861.6)])

    def test_merged_cell_counts_once_and_total_words_are_whole_words(self):
        t = parse_tables(
            "<table><tr><th>No</th><th>Item</th><th>Value</th><th>Rate</th></tr>"
            "<tr><td>1</td><td>मोटर उपयोग</td><td>10</td><td rowspan='3'>7.5</td></tr>"
            "<tr><td>2</td><td>b</td><td>20</td></tr><tr><td>3</td><td>c</td><td>30</td></tr>"
            "<tr><td></td><td>कुल</td><td>60</td><td>7.5</td></tr></table>"
        )[0]
        checks = check_table(t)
        # "मोटर उपयोग" is not a total row (योग inside उपयोग); the rowspan 7.5 is summed once.
        self.assertEqual({(c.row, c.col, c.ok) for c in checks}, {(4, 2, True), (4, 3, True)})

    def test_fuse_repairs_from_the_other_engine(self):
        good, bad = table(), table()
        for cell in bad.cells:
            if cell.text == "१६८.२५":
                cell.text = "१६१.२५"
        fusion = fuse(bad, good)
        self.assertEqual(fusion.replaced, [{"row": 2, "col": 4, "from": "१६१.२५", "to": "१६८.२५"}])
        self.assertEqual(failed(check_table(fusion.table)), [])

    def test_format_like_keeps_script_and_decimals(self):
        self.assertEqual(format_like(168.25, "१६१.२५"), "१६८.२५")


SPLIT_A = ("<table><thead><tr><th>Year</th><th>Item</th><th>Amount</th></tr></thead><tbody>"
           "<tr><td>2019</td><td>Rice</td><td>10</td></tr><tr><td>2020</td><td>Wheat and</td><td>20</td></tr></tbody></table>")
SPLIT_B = ("<table><thead><tr><th>Year</th><th>Item</th><th>Amount</th></tr></thead><tbody>"
           "<tr><td></td><td>pulses</td><td></td></tr><tr><td>2021</td><td>Millets</td><td>12</td></tr>"
           "<tr><td></td><td>Total</td><td>42</td></tr></tbody></table>")


class JoinTest(unittest.TestCase):
    def test_continuation_decision_and_join(self):
        a, b = parse_tables(SPLIT_A)[0], parse_tables(SPLIT_B)[0]
        cont = continuation(a, (0.1, 0.5, 0.9, 0.95), b, (0.1, 0.05, 0.9, 0.5), "")
        self.assertEqual(cont.decision, "join")
        self.assertEqual(cont.drop_rows, 1)
        joined = join(a, b, cont.drop_rows)
        rows = joined.rows_text()
        self.assertEqual(rows[2], ["2020", "Wheat and pulses", "20"])  # wrapped row merged back
        self.assertEqual(len(rows), 5)
        self.assertEqual(failed(check_table(joined)), [])  # Total spans both pages

    def test_table_ending_above_footnotes_still_counts_as_page_end(self):
        a = parse_tables(SPLIT_A)[0]
        b = parse_tables("<table><tr><td>2021</td><td>Millets</td><td>12</td></tr></table>")[0]
        mid_page = continuation(a, (0.1, 0.1, 0.9, 0.72), b, (0.1, 0.09, 0.9, 0.6))
        content_end = continuation(a, (0.1, 0.1, 0.9, 0.72), b, (0.1, 0.09, 0.9, 0.6),
                                   last_on_page=True, first_on_page=True)
        self.assertEqual(mid_page.decision, "ambiguous")
        self.assertEqual(content_end.decision, "join")

    def test_one_column_difference_goes_to_the_agent(self):
        a = parse_tables(SPLIT_A)[0]
        b = parse_tables("<table><tr><td>2021</td><td>x</td><td>Millets</td><td>12</td></tr></table>")[0]
        self.assertEqual(continuation(a, (0, 0.5, 1, 0.95), b, (0, 0.05, 1, 0.5)).decision, "ambiguous")

    def test_different_tables_stay_separate(self):
        a = parse_tables(SPLIT_A)[0]
        other = parse_tables("<table><tr><th>a</th><th>b</th><th>c</th><th>d</th><th>e</th></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td></tr></table>")[0]
        self.assertEqual(continuation(a, (0, 0.5, 1, 0.95), other, (0, 0.05, 1, 0.5)).decision, "separate")


class AssembleEndToEndTest(unittest.TestCase):
    def test_table_split_over_two_pages_is_joined_in_doc_md(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.object(cache, "CACHE_DIR", root / "cache"):
                pdf = make_pdf(root / "two.pdf", 2)
                sarvam = FakeSarvam(page_tables={1: (SPLIT_A, [0.1, 0.5, 0.9, 0.95]), 2: (SPLIT_B, [0.1, 0.05, 0.9, 0.5])})
                run = Run(pdf, Settings(language="en-IN", policy="sarvam-only"), out_root=root / "out",
                          sarvam=sarvam, progress_every=999, verbose=False)
                run.prepare()
                summary = run.execute()
                result = Assembler(Path(summary["out_dir"]), agent=None, reocr=False, verbose=False).run()
                out = Path(summary["out_dir"])
                self.assertEqual(result["tables"]["found"], 2)
                self.assertEqual(result["tables"]["after_joining"], 1)
                self.assertEqual(result["tables"]["list"][0]["pages"], [1, 2])
                self.assertEqual(result["tables"]["checks_failed"], 0)
                doc = (out / "doc.md").read_text()
                self.assertEqual(doc.count("<table>"), 1)
                self.assertIn("Wheat and pulses", doc)
                self.assertTrue((out / "tables" / "t00001_0.csv").is_file())
                self.assertIn("Joins across page breaks", (out / "report.md").read_text())
                self.assertTrue(json.loads((out / "assembly.json").read_text())["joins"][0]["joined"])


if __name__ == "__main__":
    unittest.main()
