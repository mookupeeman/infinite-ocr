"""Cut-out sheets for the image check: small numbered strips of a table, one per question.

A sheet costs far fewer tokens than the whole table, and the model does not have to find
"row 15" by counting: each strip is the row (and its header, shown once on top), with a
red box around the cell when the ruling lines give exact cell boxes. Without rules the
row band is estimated from the text lines (or proportionally) and taken with some
context; the question then names the column header and the row's other cells.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import pymupdf
from PIL import Image, ImageDraw, ImageFont

from src.table_geometry import Geometry, _runs, table_geometry
from src.tables import Table

DPI = 200
MAX_WIDTH = 1200
PER_SHEET = 12
LABEL_FONT = ImageFont.load_default(size=20)


@dataclass
class Strip:
    """A cut-out of one table row or cell for the image check."""
    qid: str
    image: Image.Image
    exact: bool  # the target cell is boxed in red


@dataclass
class Sheet:
    """Several numbered strips laid out in one image, with the ids of their questions."""
    png: bytes
    qids: list[str]
    exact: list[bool]
    size: tuple[int, int]


def _text_lines(dark: np.ndarray, top: int, bottom: int) -> list[tuple[int, int]]:
    """Text line bands between two pixel rows (an inked row run of at least 4 px)."""
    rows = dark[top:bottom].mean(axis=1) > 0.01
    return [(a + top, b + top) for a, b in _runs(rows) if b - a >= 4]


class Cutter:
    """Row bands and cell boxes for one table on one page."""

    def __init__(self, doc: pymupdf.Document | None, page: int, rotate: int, bbox: tuple, table: Table,
                 geometry: Geometry | None = None) -> None:
        """Pass the table's `geometry` (computed at DPI by the free check) to skip a second render and analysis."""
        self.table = table
        self.geo: Geometry = geometry if geometry is not None and geometry.gray is not None else table_geometry(
            doc, page, rotate, bbox, dpi=DPI)
        self.img = Image.fromarray(self.geo.gray)
        g, n = self.geo, table.n_rows
        self.exact_rows = g.source == "rules" and g.rows == n
        self.exact_cols = self.exact_rows and g.cols == table.n_cols
        w, h = self.img.size
        if self.exact_rows:
            self.bands = [(g.h_lines[r], g.h_lines[r + 1]) for r in range(n)]
            return
        top = g.h_lines[0] if g.h_lines else 0
        bottom = g.h_lines[-1] if len(g.h_lines) >= 2 else h
        lines = _text_lines(g.ink, top + 2, bottom - 2) if g.ink is not None else []
        if len(lines) == n:
            self.bands = lines
        else:  # proportional estimate
            step = (bottom - top) / max(1, n)
            self.bands = [(int(top + r * step), int(top + (r + 1) * step)) for r in range(n)]

    def strip(self, r0: int, r1: int, c: int | None, colspan: int = 1, label: str = "") -> Strip:
        w, h = self.img.size
        y0, y1 = self.bands[r0][0], self.bands[min(r1, len(self.bands)) - 1][1]
        if not self.exact_rows:  # take a row of context on each side
            pad = int(0.8 * (y1 - y0) / max(1, r1 - r0))
            y0, y1 = max(0, y0 - pad), min(h, y1 + pad)
        crop = self.img.crop((0, max(0, y0 - 3), w, min(h, y1 + 3))).convert("RGB")
        exact = self.exact_cols and c is not None
        if exact:
            g = self.geo
            x0, x1 = g.v_lines[c], g.v_lines[min(c + colspan, len(g.v_lines) - 1)]
            ImageDraw.Draw(crop).rectangle((x0 - 1, 1, x1 + 1, crop.height - 2), outline=(220, 0, 0), width=3)
        if crop.width > MAX_WIDTH:
            crop = crop.resize((MAX_WIDTH, max(1, int(crop.height * MAX_WIDTH / crop.width))))
        if label:
            canvas = Image.new("RGB", (crop.width + 70, crop.height), "white")
            canvas.paste(crop, (70, 0))
            ImageDraw.Draw(canvas).text((4, max(0, crop.height // 2 - 11)), label, fill=(0, 0, 160), font=LABEL_FONT)
            crop = canvas
        return Strip(label, crop, exact)

    def cell_snippet(self, r: int, c: int, rowspan: int = 1, colspan: int = 1, label: str = "") -> Strip:
        """Just the cell, under its column's header cells (exact boxes only): far fewer pixels than a row strip."""
        g = self.geo
        x0, x1 = g.v_lines[c], g.v_lines[min(c + colspan, len(g.v_lines) - 1)]
        hr = self.table.header_rows
        parts = []
        if hr:
            parts.append(self.img.crop((x0, self.bands[0][0], x1 + 1, self.bands[hr - 1][1] + 1)).convert("RGB"))
        cell = self.img.crop((x0, self.bands[r][0], x1 + 1, self.bands[min(r + rowspan, len(self.bands)) - 1][1] + 1)).convert("RGB")
        ImageDraw.Draw(cell).rectangle((1, 1, cell.width - 2, cell.height - 2), outline=(220, 0, 0), width=3)
        parts.append(cell)
        width = max(p.width for p in parts) + 70
        canvas = Image.new("RGB", (width, sum(p.height + 4 for p in parts)), "white")
        y = 0
        for p in parts:
            canvas.paste(p, (70, y))
            y += p.height + 4
        ImageDraw.Draw(canvas).text((4, max(0, canvas.height - parts[-1].height // 2 - 12)), label, fill=(0, 0, 160), font=LABEL_FONT)
        return Strip(label, canvas, True)

    def header(self) -> Image.Image | None:
        hr = self.table.header_rows
        return self.strip(0, hr, None, label="HEAD").image if hr else None


def _grid_sheet(strips: list[Strip], max_width: int = MAX_WIDTH) -> Sheet:
    """Cell snippets laid out left to right, wrapping at max_width."""
    rows, row, x = [], [], 0
    for s in strips:
        if row and x + s.image.width + 12 > max_width:
            rows.append(row)
            row, x = [], 0
        row.append(s)
        x += s.image.width + 12
    if row:
        rows.append(row)
    width = max(sum(s.image.width + 12 for s in r) for r in rows)
    height = sum(max(s.image.height for s in r) + 12 for r in rows)
    sheet = Image.new("RGB", (width, height), "white")
    y = 0
    for r in rows:
        x = 0
        for s in r:
            sheet.paste(s.image, (x, y))
            x += s.image.width + 12
        y += max(s.image.height for s in r) + 12
    buf = io.BytesIO()
    sheet.save(buf, "PNG", optimize=True)
    return Sheet(buf.getvalue(), [s.qid for s in strips], [s.exact for s in strips], sheet.size)


def sheets(cutter: Cutter, cells: list[tuple[str, int, int, int, int]], per_sheet: int = PER_SHEET,
           cell_only: bool = False) -> list[Sheet]:
    """cells: (question id, row, col, rowspan, colspan). Header strip on top of every sheet, or, with
    `cell_only` on a table whose ruling lines give exact boxes, one small snippet per cell."""
    if cell_only and cutter.exact_cols:
        return [_grid_sheet([cutter.cell_snippet(r, c, rs, cs, label=qid) for qid, r, c, rs, cs in cells[k: k + per_sheet]])
                for k in range(0, len(cells), per_sheet)]
    out = []
    head = cutter.header()
    for k in range(0, len(cells), per_sheet):
        part = cells[k: k + per_sheet]
        strips = [cutter.strip(r, r + rs, c, cs, label=qid) for qid, r, c, rs, cs in part]
        images = ([head] if head is not None else []) + [s.image for s in strips]
        width = max(i.width for i in images)
        height = sum(i.height + 8 for i in images)
        sheet = Image.new("RGB", (width, height), "white")
        y = 0
        for i, img in enumerate(images):
            sheet.paste(img, (0, y))
            y += img.height + 8
            if i == 0 and head is not None:  # separate the header strip from the questions
                ImageDraw.Draw(sheet).line((0, y - 4, width, y - 4), fill=(0, 0, 160), width=2)
        buf = io.BytesIO()
        sheet.save(buf, "PNG", optimize=True)
        out.append(Sheet(buf.getvalue(), [s.qid for s in strips], [s.exact for s in strips], sheet.size))
    return out
