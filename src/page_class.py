"""What kind of page this is: born digital or a scan, and whether its text layer can be evidence. No API calls.

Documents are often mixed (the Union budget has scanned pages inside a digital PDF), so the
class is decided per page, from two facts PyMuPDF gives in a few milliseconds:

    render      vector  the page is drawn: text and lines (born digital, legacy fonts included)
                raster  one image covers at least 80% of the page and there is (almost) no
                        visible text on it: a scan or a photo
    text_layer  unicode     visible Unicode text: may prove itself and settle table cells
                legacy      a legacy (non-Unicode) font: never used as evidence
                hidden_ocr  invisible text over a page image (someone else's OCR): never used
                none        no text
                ignored     the user said the document is scanned: any text layer is set aside

Before this, one test did both jobs (`text_chars < 20` meant scanned). It counted invisible
OCR text as a text layer, so a scan with hidden OCR took the born-digital route.

`--doc-type digital|scanned` overrides the render class for every page; `scanned` also stops
any text layer from being used. `auto` (the default) decides per page.
"""

from __future__ import annotations

from dataclasses import dataclass

import pymupdf

DOC_TYPES = ("auto", "digital", "scanned")
RASTER_COVER = 0.8  # share of the page one image must cover
MIN_TEXT = 20  # characters; fewer is "no text" (page numbers, stray marks)


@dataclass(frozen=True)
class PageClass:
    """Born digital or scanned, and whether the page's text layer can be used as evidence."""
    render: str  # vector | raster
    text_layer: str  # unicode | legacy | hidden_ocr | none
    image_cover: float
    source: str = "detected"  # detected | user

    @property
    def scanned(self) -> bool:
        return self.render == "raster"

    @property
    def layer_usable(self) -> bool:
        return self.text_layer == "unicode"


def image_cover(page: pymupdf.Page) -> float:
    """Largest share of the page covered by one placed image."""
    area = abs(page.rect) or 1.0
    return max((abs(pymupdf.Rect(i["bbox"]) & page.rect) / area for i in page.get_image_info()), default=0.0)


def text_counts(page: pymupdf.Page) -> tuple[int, int]:
    """(visible, invisible) non-space characters. Invisible = render mode 3 or fully transparent."""
    visible = invisible = 0
    for span in page.get_texttrace():
        n = sum(1 for ch in span["chars"] if not chr(ch[0]).isspace())
        if span.get("type") == 3 or span.get("opacity", 1.0) == 0:
            invisible += n
        else:
            visible += n
    return visible, invisible


def classify(page: pymupdf.Page, legacy_font: bool = False, doc_type: str = "auto") -> PageClass:
    if doc_type not in DOC_TYPES:
        raise ValueError(f"doc_type must be one of {DOC_TYPES}")
    cover = round(image_cover(page), 3)
    visible, invisible = text_counts(page)
    raster = cover >= RASTER_COVER and visible < MIN_TEXT
    if visible >= MIN_TEXT:
        layer = "legacy" if legacy_font else "unicode"
    elif invisible >= MIN_TEXT:
        layer = "hidden_ocr"
    else:
        layer = "none"
    if doc_type == "auto":
        return PageClass("raster" if raster else "vector", layer, cover)
    if doc_type == "scanned":
        return PageClass("raster", layer if layer in ("none", "hidden_ocr") else "ignored", cover, "user")
    return PageClass("vector", layer, cover, "user")
