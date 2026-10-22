"""Page-text check (src/text_qa.py): the free part, on synthetic blocks. No API calls."""

from __future__ import annotations

import unittest

from src import text_qa
from src.blocks import Block

BASE = ("The external sector remained resilient through the year, with exports of goods holding firm while "
        "services receipts grew faster than expected across most major categories of trade in services. ")


def block(engine: str, text: str, kind: str = "text", order: int = 0) -> Block:
    return Block(1, engine, order, kind, None, "p", (0, 0, 1, 1), text)


class TextQaTest(unittest.TestCase):
    def test_identical_text_has_no_disputes(self):
        r = text_qa.compare(1, [block("sarvam", BASE)], [block("bodhan", BASE.replace(",", " ,").replace("  ", " "))])
        self.assertTrue(r.compared)
        self.assertEqual([d.decided_by for d in r.disputes if d.decided_by != "spacing"], [])

    def test_spacing_and_punctuation_differences_are_settled_free(self):
        a = BASE + "Pune-Mumbai 2/4 lane road opened"
        b = BASE + "Pune-Mumbai 2/4lane road opened"
        r = text_qa.compare(1, [block("sarvam", a)], [block("bodhan", b)])
        self.assertEqual([d.decided_by for d in r.disputes], ["spacing"])
        c = BASE + "Report; final version"
        d_ = BASE + "Report, final version"
        r = text_qa.compare(1, [block("sarvam", c)], [block("bodhan", d_)])
        self.assertEqual([d.decided_by for d in r.disputes], ["minor"])

    def test_a_misread_word_stays_open_without_a_proven_layer(self):
        r = text_qa.compare(1, [block("sarvam", BASE + "Hissar bypass")], [block("bodhan", BASE + "Hamirpur bypass")])
        self.assertEqual([(d.primary, d.other, d.decided_by) for d in r.disputes], [("Hissar", "Hamirpur", "open")])

    def test_a_proven_layer_chooses_and_the_choice_is_applied(self):
        primary = [block("sarvam", BASE + "Hissar bypass opened")]
        other = [block("bodhan", BASE + "Hamirpur bypass opened")]
        layer = BASE + "Hamirpur bypass opened"
        r = text_qa.compare(1, primary, other, layer)
        self.assertTrue(r.layer["proven"])
        self.assertEqual([(d.decided_by, d.chosen) for d in r.disputes], [("text_layer", "other")])
        self.assertEqual(text_qa.apply(primary, r), 1)
        self.assertTrue(primary[0].text.endswith("Hamirpur bypass opened"))

    def test_an_unproven_layer_decides_nothing(self):
        scrambled = "xx yy zz " * 40
        r = text_qa.compare(1, [block("sarvam", BASE + "Hissar bypass")], [block("bodhan", BASE + "Hamirpur bypass")], scrambled)
        self.assertFalse(r.layer["proven"])
        self.assertEqual([d.decided_by for d in r.disputes], ["open"])

    def test_the_layer_never_deletes_printed_words(self):
        primary = [block("sarvam", BASE + "Hissar bypass opened")]
        other = [block("bodhan", BASE + "opened")]  # the other engine lacks two words; the layer has neither
        r = text_qa.compare(1, primary, other, BASE + "opened")
        self.assertEqual([d.decided_by for d in r.disputes], ["open"])
        self.assertEqual(text_qa.apply(primary, r), 0)

    def test_a_different_reading_order_is_not_compared(self):
        a = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa"
        b = "papa oscar november mike lima kilo juliet india hotel golf foxtrot echo delta charlie bravo alpha"
        r = text_qa.compare(1, [block("sarvam", a)], [block("bodhan", b)])
        self.assertFalse(r.compared)

    def test_latex_and_unicode_superscripts_are_the_same_text(self):
        for a, b in (("2024.²²", "2024.$^{22}$"), ("3x²", "3x^2"), ("*libertas*", "\\textit{libertas}"),
                     ("end!&#8221;", "end!#8221;")):
            self.assertEqual(text_qa.key_of(a), text_qa.key_of(b), (a, b))
        r = text_qa.compare(1, [block("sarvam", BASE + "grew by 3x² and fell")], [block("bodhan", BASE + "grew by $\\frac{a}{b}$ and fell")])
        self.assertEqual([d.decided_by for d in r.disputes], ["format"])  # inline maths is a way of writing, not a misread

    def test_words_one_engine_has_as_prose_and_the_other_as_a_header_or_figure_text_are_settled(self):
        for kind in ("furniture", "figure"):
            r = text_qa.compare(1, [block("sarvam", BASE), block("sarvam", "External Sector", kind, 1)],
                                [block("bodhan", BASE + "External Sector")])
            self.assertEqual([d.decided_by for d in r.disputes], ["elsewhere"], kind)
        # and a layer that "confirms" the words must not insert them a second time
        primary = [block("sarvam", BASE), block("sarvam", "Source: FTPA Department of Commerce", "figure", 1)]
        other = [block("bodhan", BASE + "Source: FTPA Department of Commerce")]
        r = text_qa.compare(1, primary, other, BASE + "Source: FTPA Department of Commerce")
        self.assertEqual(text_qa.apply(primary, r), 0)

    def test_furniture_is_ignored(self):
        r = text_qa.compare(1, [block("sarvam", BASE), block("sarvam", "Page 4", "furniture", 1)],
                            [block("bodhan", BASE), block("bodhan", "Annual Report", "furniture", 1)])
        self.assertEqual(r.disputes, [])


if __name__ == "__main__":
    unittest.main()
