import unittest
from pathlib import Path

import pymupdf

from src.pdf_pages import (
    ROTATIONS,
    RenderSpec,
    build_chunk_pdf,
    pdf_cropbox,
    render_page,
    upright_rotation,
)

ROOT = Path(__file__).resolve().parent.parent
MARATHI = ROOT / "samples" / "marathi-fiscal-report.pdf"
PATRICK = ROOT / "samples" / "Patrick-Deneen-sample.pdf"


def sideways_doc(page_rotation: int) -> pymupdf.Document:
    """A landscape page with one word, then /Rotate applied (like budget books)."""
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=200)
    page.insert_text((60, 80), "WORDMARK", fontsize=28)
    page.set_rotation(page_rotation)
    return doc


def ink(pix: pymupdf.Pixmap, rect: pymupdf.Rect) -> float:
    r = rect.irect & pymupdf.IRect(0, 0, pix.width, pix.height)
    if r.is_empty:
        return 0.0
    dark = total = 0
    for y in range(r.y0, r.y1):
        for x in range(r.x0, r.x1):
            total += 1
            dark += sum(pix.pixel(x, y)[:3]) < 380
    return dark / max(total, 1)


class RenderTransformTest(unittest.TestCase):
    def test_word_box_lands_on_ink_for_every_rotation(self):
        for page_rot in ROTATIONS:
            doc = sideways_doc(page_rot)
            word = pymupdf.Rect(doc[0].get_text("words")[0][:4])
            for extra in ROTATIONS:
                rendered = render_page(doc, 1, RenderSpec(dpi=72, rotate=extra))
                pix = pymupdf.Pixmap(rendered.data)
                with self.subTest(page_rot=page_rot, extra=extra):
                    self.assertGreater(ink(pix, rendered.page_to_px(word)), 0.05)
                    blank = rendered.page_to_px(pymupdf.Rect(300, 150, 380, 190))
                    self.assertEqual(ink(pix, blank), 0.0)
                    back = rendered.px_to_page(rendered.page_to_px(word))
                    for a, b in zip(back, word):
                        self.assertAlmostEqual(a, b, places=3)

    def test_upright_rotation_makes_text_run_left_to_right(self):
        for page_rot in ROTATIONS:
            doc = sideways_doc(page_rot)
            extra = upright_rotation(doc[0])
            rendered = render_page(doc, 1, RenderSpec(dpi=72, rotate=extra))
            m = pymupdf.Matrix(*rendered.transform)
            direction = pymupdf.Point(1, 0) * pymupdf.Matrix(m.a, m.b, m.c, m.d, 0, 0)
            with self.subTest(page_rot=page_rot):
                self.assertGreater(direction.x, 0.9)

    def test_marathi_sideways_pages_detected(self):
        doc = pymupdf.open(MARATHI)
        needs = [i + 1 for i in range(doc.page_count) if upright_rotation(doc[i]) != 0]
        self.assertEqual(needs, [1, 2, 3, 6, 8])

    def test_dpi_and_size_cap(self):
        doc = pymupdf.open(MARATHI)
        r = render_page(doc, 4, RenderSpec(dpi=200, max_long_px=3000))
        self.assertAlmostEqual(r.dpi, 200, delta=0.5)
        capped = render_page(doc, 4, RenderSpec(dpi=600, max_long_px=1000))
        self.assertLessEqual(max(capped.width, capped.height), 1001)


class ChunkTest(unittest.TestCase):
    def test_pages_order_boxes_and_text(self):
        src = pymupdf.open(PATRICK)
        chunk = pymupdf.open("pdf", build_chunk_pdf(src, [2, 1]))
        self.assertEqual(chunk.page_count, 2)
        for chunk_page, src_num in zip(chunk, [2, 1]):
            src_page = src[src_num - 1]
            self.assertEqual(chunk_page.get_text().strip(), src_page.get_text().strip())
            self.assertEqual(chunk_page.mediabox, pdf_cropbox(src_page))
            same = chunk_page.get_pixmap(dpi=40).samples == src_page.get_pixmap(dpi=40).samples
            self.assertTrue(same)

    def test_rotations_make_chunk_pages_upright(self):
        src = pymupdf.open(MARATHI)
        pages = [1, 2, 4]
        rotations = {p: upright_rotation(src[p - 1]) for p in pages}
        chunk = pymupdf.open("pdf", build_chunk_pdf(src, pages, rotations))
        self.assertEqual([upright_rotation(p) for p in chunk], [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
