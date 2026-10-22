"""Page classes (src/page_class.py) on small synthetic PDFs. No API calls."""

from __future__ import annotations

import unittest

import pymupdf

from src import page_class
from src.triage import body_tables, triage_page

TEXT = "Receipts of the Union Government, 2026-27, in crore rupees. " * 3


def _image(page: pymupdf.Page) -> None:
    pix = pymupdf.Pixmap(pymupdf.csGRAY, pymupdf.IRect(0, 0, 60, 80), False)
    pix.set_rect(pix.irect, (200,))
    page.insert_image(page.rect, pixmap=pix)


def _page(*, text: bool = False, image: bool = False, hidden: bool = False) -> pymupdf.Page:
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    if image:
        _image(page)
    if text:
        page.insert_text((50, 100), TEXT, fontsize=8)
    if hidden:
        page.insert_text((50, 200), TEXT, fontsize=8, render_mode=3)  # how OCR software hides its text
    page._keep_doc = doc  # the page must not outlive its document
    return page


class PageClassTest(unittest.TestCase):
    def test_born_digital_text_page(self):
        cls = page_class.classify(_page(text=True))
        self.assertEqual((cls.render, cls.text_layer, cls.scanned, cls.layer_usable), ("vector", "unicode", False, True))

    def test_scan_without_text(self):
        cls = page_class.classify(_page(image=True))
        self.assertEqual((cls.render, cls.text_layer), ("raster", "none"))
        self.assertGreaterEqual(cls.image_cover, 0.9)  # the picture keeps its aspect ratio

    def test_scan_with_hidden_ocr_text_is_a_scan_and_its_text_is_not_evidence(self):
        cls = page_class.classify(_page(image=True, hidden=True))
        self.assertEqual((cls.render, cls.text_layer, cls.layer_usable), ("raster", "hidden_ocr", False))

    def test_text_over_a_full_page_picture_is_born_digital(self):
        cls = page_class.classify(_page(image=True, text=True))
        self.assertEqual(cls.render, "vector")

    def test_legacy_font_page_is_vector_but_its_layer_is_not_used(self):
        cls = page_class.classify(_page(text=True), legacy_font=True)
        self.assertEqual((cls.render, cls.text_layer, cls.layer_usable), ("vector", "legacy", False))

    def test_user_choice_overrides_every_page(self):
        scanned = page_class.classify(_page(text=True), doc_type="scanned")
        self.assertEqual((scanned.render, scanned.text_layer, scanned.source), ("raster", "ignored", "user"))
        digital = page_class.classify(_page(image=True), doc_type="digital")
        self.assertEqual((digital.render, digital.source), ("vector", "user"))
        with self.assertRaises(ValueError):
            page_class.classify(_page(text=True), doc_type="photo")


def _grid(page: pymupdf.Page, x0: float, y0: float, x1: float, y1: float, rows: int, cols: int) -> None:
    for r in range(rows + 1):
        y = y0 + (y1 - y0) * r / rows
        page.draw_line((x0, y), (x1, y), width=0.8)
    for c in range(cols + 1):
        x = x0 + (x1 - x0) * c / cols
        page.draw_line((x, y0), (x, y1), width=0.8)
    for r in range(rows):
        for c in range(cols):
            page.insert_text((x0 + (x1 - x0) * c / cols + 3, y0 + (y1 - y0) * (r + 1) / rows - 4), f"{r}{c}", fontsize=7)


class TableHintTest(unittest.TestCase):
    def test_a_page_number_box_is_not_a_table_page_but_a_body_table_is(self):
        footer_only = _page(text=True)
        _grid(footer_only, 60, 770, 535, 786, 1, 2)  # the box an annual report draws around its page number
        self.assertEqual(body_tables(footer_only), [])
        self.assertFalse(triage_page(footer_only, "en-IN").has_table)
        with_table = _page(text=True)
        _grid(with_table, 60, 770, 535, 786, 1, 2)
        _grid(with_table, 60, 300, 535, 420, 5, 3)
        boxes = body_tables(with_table)
        self.assertEqual(len(boxes), 1)
        self.assertTrue(triage_page(with_table, "en-IN").has_table)


if __name__ == "__main__":
    unittest.main()
