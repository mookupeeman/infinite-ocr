"""Learning a scrambled PDF text layer's glyph substitutions (src/layer_repair.py). No API calls."""

from __future__ import annotations

import unittest

from src import layer_repair


def scramble(text: str) -> str:
    """Encode text the way the Union budget's font does: क्त -> Ơ, ी -> ȣ, and the vowel sign ि stored
    as Ǒ before its consonant (कि -> Ǒक)."""
    out = text.replace("क्त", "Ơ").replace("ी", "ȣ")
    import re

    return re.sub(r"([क-ह])ि", lambda m: "Ǒ" + m.group(1), out)


AGREED = ["किमी से", "अक्तूबर", "खजुरी गांव", "विकास योजना", "मुक्त क्षेत्र", "नदी पार", "भक्ति मार्ग", "गिनती",
          "बड़ी सड़क", "नीला पानी", "सीमा तक", "चीनी मिल"]


class LayerRepairTest(unittest.TestCase):
    def test_rules_are_learned_from_agreed_cells(self):
        layer = scramble(" ".join(AGREED))
        rules = layer_repair.learn([(layer, AGREED)])
        self.assertEqual(rules.get("Ơ"), ("sub", "क्त"))
        self.assertEqual(rules.get("ȣ"), ("sub", "ी"))
        self.assertEqual(rules.get("Ǒ"), ("prebase", "ि"))
        self.assertEqual(layer_repair.apply(layer, rules), " ".join(AGREED))

    def test_a_glyph_used_two_ways_is_not_learned(self):
        # Ǒ means ि in half the words and something else in the others: below the 90% bar, so no rule
        good = scramble(" ".join(AGREED[:4]))
        bad = "Ǒअ Ǒआ Ǒइ Ǒई"
        rules = layer_repair.learn([(good + " " + bad, AGREED[:4] + ["अब", "आज", "इस", "ईख"])])
        self.assertNotIn("Ǒ", rules)

    def test_too_few_examples_give_no_rule(self):
        rules = layer_repair.learn([(scramble("अक्तूबर"), ["अक्तूबर"])])
        self.assertEqual(rules, {})

    def test_reph_stored_after_its_cluster(self):
        # "[" after a cluster stands for र् before it: वार्षिक printed as वाषि[क
        words = ["वार्षिकी", "कार्यालय", "सर्वेक्षण", "मार्गदर्शन", "पूर्वोत्तर"]
        layer = "वाषि[की काय[ालय सवे[क्षण माग[दश[न पूव[ोत्तर"
        rules = layer_repair.learn([(layer, words)])
        self.assertEqual(rules.get("["), ("postbase", "र्"))
        self.assertEqual(layer_repair.apply(layer, rules), " ".join(words))


if __name__ == "__main__":
    unittest.main()
