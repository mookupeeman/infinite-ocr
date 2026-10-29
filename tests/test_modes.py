"""Fast / Normal / Think (src/modes.py) and the estimate shown before a run (src/estimate.py). No API calls."""

from __future__ import annotations

import unittest
from unittest import mock
from pathlib import Path

import pymupdf

from src import estimate, modes
from src.assemble import Assembler
from src.lanes import Settings


def _pdf(pages: int, table_pages: set[int]) -> pymupdf.Document:
    doc = pymupdf.open()
    for i in range(1, pages + 1):
        page = doc.new_page(width=595, height=842)
        page.insert_text((50, 80), "The external sector remained resilient through the year. " * 2, fontsize=8)
        if i in table_pages:  # a dense block of numbers, as on a budget table page
            for r in range(16):
                page.insert_text((50, 120 + 14 * r), "  ".join(f"{100 + r * 7 + c}.25" for c in range(6)), fontsize=8)
    return doc


class ModesTest(unittest.TestCase):
    def test_default_is_normal_and_unknown_modes_fail(self):
        self.assertEqual(modes.get(None).name, "normal")
        self.assertEqual(modes.get("THINK").policy, "both")
        with self.assertRaises(ValueError):
            modes.get("turbo")

    def test_think_uses_format_a_and_two_of_three_on_scans(self):
        think, normal = modes.get("think").judge, modes.get("normal").judge
        self.assertEqual((think.per_sheet, think.cell_only, think.vote, len(think.scanned_ruled)), (12, False, 2, 3))
        self.assertEqual((normal.per_sheet, normal.cell_only, normal.vote, len(normal.scanned_ruled)), (24, True, None, 2))

    def test_fast_makes_no_model_calls(self):
        class FakeAgent:
            available = True

        self.assertIsNone(Assembler(Path("."), agent=FakeAgent(), mode="fast").agent)
        self.assertIsNotNone(Assembler(Path("."), agent=FakeAgent(), mode="normal").agent)

    def test_bodhan_leads_only_when_local_on_vllm(self):
        cases = [("api", "vllm", "sarvam"), ("local", "mlx", "sarvam"), ("local", "pytorch", "sarvam"), ("local", "vllm", "bodhan")]
        for backend, runtime, lead in cases:
            with mock.patch.object(modes, "BODHAN_LEAD", "auto"), mock.patch.object(modes, "BODHAN_BACKEND", backend), \
                    mock.patch("src.indic_local.runtime", lambda r=runtime: r):
                self.assertEqual(modes.default_lead(), lead, (backend, runtime))
        with mock.patch.object(modes, "BODHAN_LEAD", "sarvam"):
            self.assertEqual(modes.default_lead(), "sarvam")

    def test_policy_follows_the_lead(self):
        fast, think = modes.get("fast"), modes.get("think")
        self.assertEqual(modes.policy_for(fast, "bodhan"), "bodhan-primary")
        self.assertEqual(modes.policy_for(fast, "sarvam"), "sarvam-primary")
        self.assertEqual(modes.policy_for(think, "bodhan"), "both")
        with self.assertRaises(ValueError):
            modes.policy_for(fast, "qwen")

    def test_document_type_auto_keeps_older_runs_resumable(self):
        self.assertNotIn("doc_type", Settings(language="auto").fingerprint())
        self.assertEqual(Settings(language="auto", doc_type="scanned").fingerprint()["doc_type"], "scanned")


class EstimateTest(unittest.TestCase):
    def test_estimate_from_page_classes_and_table_pages(self):
        doc = _pdf(10, {3, 4, 8, 9})
        pre = estimate.preflight(doc, list(range(1, 11)))
        self.assertEqual((pre.sampled, pre.table_range), (10, (4.0, 4.0)))
        self.assertEqual(round(pre.classes["vector/unicode"]), 10)
        with mock.patch.object(estimate, "BODHAN_BACKEND", "api"):
            by = {e["mode"]: e for e in estimate.estimate_all(pre)}
        self.assertEqual(by["fast"]["total_inr"], [5.8, 5.8])  # Sarvam 10 x 0.50 + Bodhan 4 table pages x 0.20
        self.assertEqual(by["fast"]["checks_inr"], [0.0, 0.0])
        self.assertGreater(by["normal"]["checks_inr"][1], 0.0)  # no cap: the estimate is the cost, not a ceiling
        self.assertEqual(by["think"]["bodhan_inr"], [2.0, 2.0])  # Bodhan on every page
        self.assertIn("normal", estimate.render(pre, list(by.values()), "normal"))
        with mock.patch.object(estimate, "BODHAN_BACKEND", "local"), mock.patch("src.indic_local.runtime", lambda: "mlx"):
            local = {e["mode"]: e for e in estimate.estimate_all(pre)}
        self.assertEqual(local["think"]["bodhan_inr"], [0.0, 0.0])  # local weights cost nothing
        self.assertEqual(local["fast"]["total_inr"], [5.0, 5.0])  # Sarvam only
        self.assertGreater(local["think"]["minutes"][1], by["think"]["minutes"][1])  # MLX reads slower than the API
        with mock.patch.object(estimate, "BODHAN_BACKEND", "local"), mock.patch("src.indic_local.runtime", lambda: "vllm"):
            led = {e["mode"]: e for e in estimate.estimate_all(pre, lead="bodhan")}
        self.assertEqual(led["fast"]["sarvam_inr"], [2.0, 2.0])  # Sarvam reads only the 4 table pages
        self.assertEqual(led["think"]["sarvam_inr"], [5.0, 5.0])  # Think: both engines on every page
        self.assertEqual(led["fast"]["total_inr"], [2.0, 2.0])
        self.assertLess(led["think"]["minutes"][1], local["think"]["minutes"][1])  # vLLM is faster than MLX

    def test_long_documents_are_sampled_with_a_range(self):
        doc = _pdf(120, set(range(1, 121, 3)))
        pre = estimate.preflight(doc, list(range(1, 121)))
        low, high = pre.table_range
        self.assertEqual(pre.sampled, 40)
        self.assertLessEqual(low, 40)
        self.assertGreaterEqual(high, 40)


if __name__ == "__main__":
    unittest.main()
