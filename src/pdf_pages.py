"""PDF pages: render images, find upright rotation, build Sarvam chunk PDFs.

Coordinates: PyMuPDF text extraction (`get_text`) returns boxes in the
*unrotated* page frame. `Rendered.transform` maps that frame to image pixels,
including the page's /Rotate and any extra rotation we applied.
"""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from src.config import PAGES_DIR, PDF_LONG_EDGE, RENDER_DPI, RENDER_MAX_LONG_PX

ROTATIONS = (0, 90, 180, 270)


@dataclass(frozen=True)
class PagePng:
    """A rendered page saved as a PNG file."""
    page_num: int
    png_path: Path
    width: int
    height: int
    sha256: str


@dataclass(frozen=True)
class RenderSpec:
    """How to render a page: dpi, extra rotation, size cap and image format."""
    dpi: int = RENDER_DPI
    max_long_px: int = RENDER_MAX_LONG_PX
    rotate: int = 0  # extra clockwise rotation on top of the displayed page
    fmt: str = "png"  # png | jpeg
    jpg_quality: int = 90


@dataclass(frozen=True)
class Rendered:
    """A rendered page image with the transform from PDF points to pixels."""
    page_num: int
    data: bytes
    mime: str
    width: int
    height: int
    dpi: float
    rotate: int
    transform: tuple[float, float, float, float, float, float]  # unrotated page pt -> px

    def page_to_px(self, rect: pymupdf.Rect | Sequence[float]) -> pymupdf.Rect:
        return pymupdf.Rect(rect) * pymupdf.Matrix(*self.transform)

    def px_to_page(self, rect: pymupdf.Rect | Sequence[float]) -> pymupdf.Rect:
        return pymupdf.Rect(rect) * ~pymupdf.Matrix(*self.transform)


def page_transform(page: pymupdf.Page, rotate: int, zoom: float = 1.0) -> tuple[pymupdf.Matrix, int, int]:
    """Unrotated page pt -> upright image px, without rendering. Returns (matrix, w, h).

    With zoom=1 the image frame is in points; divide by (w, h) to get the 0-1
    frame that `blocks.Block.bbox` uses.
    """
    matrix = pymupdf.Matrix(zoom, zoom).prerotate(rotate)
    irect = (page.rect * matrix).irect
    transform = page.rotation_matrix * matrix * pymupdf.Matrix(1, 0, 0, 1, -irect.x0, -irect.y0)
    return transform, irect.width, irect.height


def render_page(doc: pymupdf.Document, page_num: int, spec: RenderSpec = RenderSpec()) -> Rendered:
    """Render 1-based `page_num`. Returns image bytes plus the pt->px transform."""
    if spec.rotate not in ROTATIONS:
        raise ValueError(f"rotate must be one of {ROTATIONS}")
    page = doc[page_num - 1]
    zoom = spec.dpi / 72.0
    long_pt = max(page.rect.width, page.rect.height)
    if long_pt * zoom > spec.max_long_px:
        zoom = spec.max_long_px / long_pt
    matrix = pymupdf.Matrix(zoom, zoom).prerotate(spec.rotate)
    pix = page.get_pixmap(matrix=matrix, colorspace=pymupdf.csRGB, alpha=False)
    transform = page.rotation_matrix * matrix * pymupdf.Matrix(1, 0, 0, 1, -pix.x, -pix.y)
    if spec.fmt == "jpeg":
        data, mime = pix.tobytes("jpeg", jpg_quality=spec.jpg_quality), "image/jpeg"
    else:
        data, mime = pix.tobytes("png"), "image/png"
    return Rendered(
        page_num=page_num,
        data=data,
        mime=mime,
        width=pix.width,
        height=pix.height,
        dpi=zoom * 72.0,
        rotate=spec.rotate,
        transform=tuple(transform),
    )


def upright_rotation(page: pymupdf.Page) -> int:
    """Extra rotation that makes the page's dominant text run left to right.

    Uses the text layer, so it returns 0 for scanned pages with no text.
    Budget books often print wide tables sideways; this turns them upright.
    """
    weight: dict[tuple[int, int], int] = {}
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            chars = sum(len(span.get("text", "").strip()) for span in line.get("spans", []))
            if chars < 2:
                continue
            key = (round(line["dir"][0]), round(line["dir"][1]))
            weight[key] = weight.get(key, 0) + chars
    if not weight:
        return 0
    dx, dy = max(weight, key=weight.get)
    rot = page.rotation_matrix
    shown = pymupdf.Point(dx, dy) * pymupdf.Matrix(rot.a, rot.b, rot.c, rot.d, 0, 0)
    for extra in ROTATIONS:
        v = shown * pymupdf.Matrix(extra)
        if v.x > 0.9:
            return extra
    return 0


def pdf_cropbox(page: pymupdf.Page) -> pymupdf.Rect:
    """The CropBox in PDF (bottom-left) coordinates, as `set_mediabox` expects."""
    media, crop = page.mediabox, page.cropbox
    return pymupdf.Rect(crop.x0, media.y1 - crop.y1, crop.x1, media.y1 - crop.y0)


def build_chunk_pdf(
    doc: pymupdf.Document,
    pages: Sequence[int],
    rotations: Mapping[int, int] | None = None,
) -> bytes:
    """A new PDF with the given 1-based pages, for one Sarvam job (max 10 pages).

    Each page gets MediaBox = CropBox, so Sarvam renders exactly what we
    render, and the extra rotation from `rotations` so text is upright.
    """
    if doc.needs_pass:
        raise ValueError("encrypted PDF")
    out = pymupdf.open()
    for page_num in pages:
        out.insert_pdf(doc, from_page=page_num - 1, to_page=page_num - 1)
        page = out[-1]
        crop = pdf_cropbox(page)
        if crop != page.mediabox:
            page.set_mediabox(crop)
        extra = (rotations or {}).get(page_num, 0)
        if extra:
            page.set_rotation((page.rotation + extra) % 360)
    data = out.tobytes(garbage=3, deflate=True)
    out.close()
    return data


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def shred_pdf(pdf_path: Path, out_dir: Path | None = None) -> list[PagePng]:
    """Week 1 shredder: every page to a PNG with long edge PDF_LONG_EDGE."""
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    dest = Path(out_dir) if out_dir is not None else PAGES_DIR
    dest.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(pdf_path)
    try:
        page_count = doc.page_count
        if page_count < 1:
            raise SystemExit(f"Abort: {pdf_path} has no pages.")

        pages: list[PagePng] = []
        for i in range(page_count):
            page = doc[i]
            rect = page.rect
            long_edge = max(rect.width, rect.height)
            if long_edge <= 0:
                raise SystemExit(f"Abort: page {i + 1} has invalid size.")
            zoom = PDF_LONG_EDGE / long_edge
            matrix = pymupdf.Matrix(zoom, zoom)
            pix = page.get_pixmap(
                matrix=matrix,
                colorspace=pymupdf.csRGB,
                alpha=False,
            )
            page_num = i + 1
            png_path = dest / f"page_{page_num:03d}.png"
            pix.save(str(png_path))
            record = PagePng(
                page_num=page_num,
                png_path=png_path,
                width=pix.width,
                height=pix.height,
                sha256=_sha256_file(png_path),
            )
            pages.append(record)
            size_bytes = png_path.stat().st_size
            print(
                f"{png_path}  {record.width}x{record.height}  "
                f"{size_bytes} bytes  sha256={record.sha256}"
            )
        return pages
    finally:
        doc.close()


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("Usage: python -m src.pdf_pages <file.pdf>", file=sys.stderr)
        return 2
    shred_pdf(Path(args[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
