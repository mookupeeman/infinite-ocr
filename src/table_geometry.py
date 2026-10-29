"""Table structure read from the page image: ruling lines, column gaps, ink per cell. No API calls.

This is a reader of structure that is independent of both OCR engines, so it can catch
an error both engines share (a row both dropped, a cell both left empty). It reads:

- vertical rules: long vertical dark runs (text never makes them);
- horizontal rules: in a ruled table, a dark run that crosses at least two vertical rules
  (the Devanagari headline over each word looks like a line, but never crosses a cell
  border); in an unruled table, only near-full-width rules;
- column gaps in unruled tables: vertical bands with no ink in the table body.

Three corrections, each measured on the benchmark and the real samples: a column line broken
by merged cells still counts when it runs rule to rule in two or more rows; a note row at the
bottom of the frame (no inner column line crosses it, e.g. "टीप: ...") is not a table row; and
a table box that cuts off the table's side (a ruling line runs into the crop edge) is widened.

The crop is deskewed first (scans are often a degree or two off). Results are counts
and positions in the deskewed crop; `confident` says whether the counts can decide.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
import pymupdf
from PIL import Image

from src.pdf_pages import RenderSpec, render_page

DPI = 150


@dataclass
class Geometry:
    """Table structure read from the page image: ruling lines, column gaps and the row and column counts they give."""
    ruled: bool
    rows: int | None  # rows between horizontal rules (ruled tables only)
    cols: int | None  # columns between vertical rules, or between whitespace gaps
    source: str  # rules | gaps | none
    skew: float
    h_lines: list[int] = field(default_factory=list)  # pixel rows in the deskewed crop
    v_lines: list[int] = field(default_factory=list)  # pixel columns
    size: tuple[int, int] = (0, 0)  # width, height
    ink: np.ndarray | None = field(default=None, repr=False)  # dark-pixel mask of the deskewed crop
    gray: np.ndarray | None = field(default=None, repr=False)  # the deskewed crop itself (for cut-outs)

    @property
    def confident(self) -> bool:
        return self.source == "rules" and self.rows is not None and self.cols is not None

    def cell_ink(self, r0: int, c0: int, r1: int, c1: int) -> float | None:
        """Share of dark pixels inside grid cells [r0, r1) x [c0, c1), borders excluded."""
        if not self.confident or self.ink is None or r1 > len(self.h_lines) - 1 or c1 > len(self.v_lines) - 1:
            return None
        y0, y1 = self.h_lines[r0] + 3, self.h_lines[r1] - 3
        x0, x1 = self.v_lines[c0] + 3, self.v_lines[c1] - 3
        if y1 <= y0 or x1 <= x0:
            return None
        return float(self.ink[y0:y1, x0:x1].mean())

    def to_dict(self) -> dict:
        return {"ruled": self.ruled, "rows": self.rows, "cols": self.cols, "source": self.source, "skew": self.skew}


def crop_gray(doc: pymupdf.Document, page: int, rotate: int, bbox: tuple, dpi: int = DPI,
              margin: float | tuple[float, float] = 0.012) -> np.ndarray:
    """A 0-1 box of the page, grey, with a margin around it (one value, or (left/right, top/bottom))."""
    mx, my = margin if isinstance(margin, tuple) else (margin, margin)
    rendered = render_page(doc, page, RenderSpec(dpi=dpi, rotate=rotate, max_long_px=6000))
    img = Image.open(io.BytesIO(rendered.data)).convert("L")
    w, h = img.size
    box = (max(0, int((bbox[0] - mx) * w)), max(0, int((bbox[1] - my) * h)),
           min(w, int((bbox[2] + mx) * w) + 1), min(h, int((bbox[3] + my) * h) + 1))
    return np.asarray(img.crop(box), dtype=np.uint8)


def otsu(gray: np.ndarray) -> int:
    """Threshold t splitting pixels into <= t (ink) and > t (paper) with the largest between-class variance."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(float)
    total, cum, cum_mean = hist.sum(), np.cumsum(hist), np.cumsum(hist * np.arange(256))
    mean = cum_mean[-1] / total
    w0 = cum / total
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mean * w0 - cum_mean / total) ** 2 / (w0 * (1 - w0))
    if np.isnan(between).all():  # one grey level: nothing to split, so no ink (a blank page or photo)
        return max(0, int(gray.min()) - 1)
    return int(np.nanargmax(between))


def _rotate(gray: np.ndarray, angle: float) -> np.ndarray:
    return np.asarray(Image.fromarray(gray).rotate(angle, resample=Image.BILINEAR, fillcolor=255), dtype=np.uint8)


def deskew(gray: np.ndarray, limit: float = 2.5, step: float = 0.1) -> tuple[np.ndarray, float]:
    """Angle that makes horizontal structure sharpest (largest variance of the row profile).

    The dark mask is rotated with nearest-neighbour sampling, so every angle (0 included)
    is scored on equally sharp pixels; rotating grey pixels would blur and favour 0.
    """
    scale = max(1, gray.shape[1] // 700)
    mask = Image.fromarray(((gray[::scale, ::scale] <= min(otsu(gray), 170)) * 255).astype(np.uint8))
    best, best_score = 0.0, -1.0
    for angle in np.arange(-limit, limit + 1e-9, step):
        rows = np.asarray(mask.rotate(float(angle), resample=Image.NEAREST, fillcolor=0), dtype=bool).mean(axis=1)
        score = float(np.var(rows))
        if score > best_score:
            best, best_score = round(float(angle), 2), score
    return (_rotate(gray, best) if best else gray), best


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) of True runs in a 1-D mask."""
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2], edges[1::2]))


def _group(positions: list[int], gap: int = 3) -> list[int]:
    """Merge neighbouring pixel rows/columns into one line each (thick or blurred rules)."""
    out: list[list[int]] = []
    for p in sorted(positions):
        if out and p - out[-1][-1] <= gap:
            out[-1].append(p)
        else:
            out.append([p])
    return [int(round(sum(g) / len(g))) for g in out]


def _bridge(mask: np.ndarray, axis: int, gap: int = 1) -> np.ndarray:
    """Close gaps of up to 2*gap pixels along an axis (noise and JPEG break thin rules)."""
    out = mask.copy()
    for k in range(1, gap + 1):
        if axis == 1:
            out[:, k:] |= mask[:, :-k]
            out[:, :-k] |= mask[:, k:]
        else:
            out[k:, :] |= mask[:-k, :]
            out[:-k, :] |= mask[k:, :]
    return out


def vertical_rules(dark: np.ndarray, min_frac: float = 0.3) -> list[int]:
    h = dark.shape[0]
    closed = _bridge(dark, axis=0)
    cols = [x for x in range(dark.shape[1]) if any(b - a >= min_frac * h for a, b in _runs(closed[:, x]))]
    return _group(cols)


def horizontal_rules(dark: np.ndarray, verticals: list[int]) -> list[int]:
    """Rows with a dark run that crosses two vertical rules (ruled), or spans 60% of the width (unruled)."""
    w = dark.shape[1]
    closed = _bridge(dark, axis=1)
    rows = []
    for y in range(dark.shape[0]):
        for a, b in _runs(closed[y]):
            if b - a < 0.05 * w:
                continue
            crossed = sum(1 for x in verticals if a - 2 <= x <= b + 2)
            if (len(verticals) >= 3 and crossed >= 2) or b - a >= 0.6 * w:
                rows.append(y)
                break
    return _group(rows)


def partial_rules(dark: np.ndarray, h: list[int], v: list[int], min_share: float = 0.25) -> list[int]:
    """Column lines broken by merged cells (too short for `vertical_rules`): a thin dark line with paper
    on both sides that reaches from one horizontal rule to the next in at least a quarter of the rows
    (and two). Blurred tall script (Telugu on a poor scan) can fill a row's height, but not with clean
    paper 6-10 px either side; a shaded cell is not thin."""
    closed = _bridge(dark, axis=0)
    count = np.zeros(dark.shape[1], dtype=int)
    bands = [(a, b) for a, b in zip(h, h[1:]) if b - a >= 10]
    for a, b in bands:
        share = closed[a + 3:b - 2].mean(axis=0)
        kernel = np.ones(5) / 5
        side = np.convolve(share, kernel, mode="same")  # mean over x-2..x+2, read at x +- 8
        line = share >= 0.95
        line[8:] &= side[:-8] < 0.35
        line[:-8] &= side[8:] < 0.35
        line[:8] = line[-8:] = False
        count += line
    need = max(2, int(np.ceil(min_share * len(bands))))
    xs = [int(x) for x in np.flatnonzero(count >= need) if all(abs(int(x) - u) > 4 for u in v)]
    return _group(xs)


def note_band(dark: np.ndarray, h: list[int], v: list[int]) -> bool:
    """The bottom row is a note inside the frame (no inner column line crosses it): not a table row."""
    if len(h) < 4 or len(v) < 3:
        return False
    a, b = h[-2] + 3, h[-1] - 2
    if b <= a:
        return True  # a double bottom rule
    band = _bridge(dark, axis=0)[a:b]
    return all(band[:, max(0, x - 1):x + 2].any(axis=1).mean() < 0.9 for x in v[1:-1])  # text over a line position stays well below


def column_gaps(dark: np.ndarray, top: int, bottom: int, min_gap: int = 6) -> int | None:
    """Text columns separated by vertical bands with no ink, between two rows (an unruled body)."""
    body = dark[top:bottom]
    if body.shape[0] < 10:
        return None
    filled = body.mean(axis=0) > 0.004
    runs = [r for r in _runs(filled) if r[1] - r[0] >= 3]
    merged: list[list[int]] = []
    for a, b in runs:  # gaps narrower than min_gap are spaces inside a column
        if merged and a - merged[-1][1] < min_gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return len(merged) or None


def analyse(gray: np.ndarray) -> Geometry:
    straight, angle = deskew(gray)
    thr = min(otsu(straight), 170)
    dark = straight <= thr  # Otsu's class 0 is the values <= the threshold
    v = vertical_rules(dark)
    h = horizontal_rules(dark, v)
    size = (straight.shape[1], straight.shape[0])
    if len(v) >= 3 and len(h) >= 3:
        v = sorted(v + partial_rules(dark, h, v))
        if note_band(dark, h, v):
            h = h[:-1]
        return Geometry(True, len(h) - 1, len(v) - 1, "rules", angle, h, v, size, dark, straight)
    if len(h) >= 2:  # unruled table with top / header / bottom rules: count columns from gaps
        top = h[1] + 3 if len(h) >= 3 else h[0] + 3
        cols = column_gaps(dark, top, h[-1] - 3)
        return Geometry(False, None, cols, "gaps" if cols else "none", angle, h, v, size, dark, straight)
    return Geometry(False, None, None, "none", angle, h, v, size, dark, straight)


def clipped(g: Geometry, edge: int = 3, room: int = 12) -> bool:
    """A ruling line runs into the left or right edge of the crop, beyond the outermost column line:
    the table box cut off part of the table (a border, a column)."""
    if g.ink is None or not g.h_lines or len(g.v_lines) < 2:
        return False
    w = g.size[0]
    for y in g.h_lines:
        row = g.ink[max(0, y - 1):y + 2].any(axis=0)
        if (row[:edge].any() and g.v_lines[0] > room) or (row[-edge:].any() and g.v_lines[-1] < w - room):
            return True
    return False


def table_geometry(doc: pymupdf.Document, page: int, rotate: int, bbox: tuple, dpi: int = DPI) -> Geometry:
    """Geometry of the table inside a 0-1 box; the box is widened once if it cut the table."""
    g = analyse(crop_gray(doc, page, rotate, bbox, dpi=dpi))
    if clipped(g):
        g = analyse(crop_gray(doc, page, rotate, bbox, dpi=dpi, margin=(0.045, 0.012)))
    return g
