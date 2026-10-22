"""Render a benchmark `Case` to a PDF (headless Chrome) and a scan-like copy.

Chrome shapes Indic scripts properly and keeps merged cells, so the page looks like
a real document. Its text layer has realistic defects (doubled matras, split clusters),
which is what the checker will meet. Table boxes are read back from the drawn rules.
"""

from __future__ import annotations

import html
import io
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image, ImageFilter

from bench.gen import Case, PageSpec

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
FONTS = {
    "hi": '"Kohinoor Devanagari"', "mr": '"Kohinoor Devanagari"', "ta": '"Tamil Sangam MN"',
    "te": '"Kohinoor Telugu"', "bn": '"Kohinoor Bangla"', "en": '"Helvetica Neue"',
}
CSS = """
@page { size: A4; margin: 16mm 14mm; }
body { font-family: %(font)s, sans-serif; font-size: %(size)s; margin: 0; color: #111; }
.page { height: 262mm; position: relative; break-after: page; }  /* no overflow:hidden: a clipped table must fail the checks */
.page:last-child { break-after: auto; }
.cap { margin: 0 0 3mm 0; font-weight: 600; }
.note { margin: 4mm 0 0 0; }
table { border-collapse: collapse; width: 100%%; }
table.wide { font-size: 0.88em; }
td, th { padding: %(pad)s; vertical-align: top; text-align: left; line-height: %(lh)s; }
th { font-weight: 600; }
table.ruled td, table.ruled th { border: 0.6pt solid #000; }
table.plain { border-top: 1.2pt solid #000; border-bottom: 1.2pt solid #000; }
table.plain th { border-bottom: 0.8pt solid #000; }
.pn { position: absolute; bottom: 0; left: 0; right: 0; text-align: center; font-size: 9pt; }
.fig { height: 80mm; border: 1pt solid #444; background: #ddd; display: flex; align-items: center; justify-content: center; }
"""


CLEAN = {"size": "10pt", "pad": "1.5mm 2mm", "lh": "1.3"}
TOUGH = {"size": "7.5pt", "pad": "0.5mm 1mm", "lh": "1.12"}
# scan settings: (dpi, skew degrees, blur radius, contrast, noise sigma, JPEG quality)
SCAN = {"clean": (200, 1.2, (0.5, 1.1), (0.82, 0.95), (3, 9), (40, 70)),
        "tough": (150, 2.0, (0.8, 1.5), (0.70, 0.88), (6, 14), (25, 45))}


@dataclass
class Rendered:
    pdf: Path
    scan_pdf: Path | None
    bboxes: dict[int, list[float] | None]  # page -> table box in 0-1 (None for non-table pages)
    ok: bool
    note: str = ""


def page_html(page: PageSpec, n: int, ruled: bool, kind: str = "") -> str:
    body = []
    if page.caption:
        body.append(f'<p class="cap">{html.escape(page.caption)}</p>')
    if page.kind == "table" and page.table is not None:
        body.append(page.table.to_html().replace("<table>", f'<table class="{"ruled" if ruled else "plain"}{" wide" if kind == "wide" else ""}">', 1))
        if page.note:
            body.append(f'<p class="note">{html.escape(page.note)}</p>')
    elif page.kind == "figure":
        body.append('<div class="fig">Figure</div><p class="cap">Figure: chart of the values above</p>')
    return f'<div class="page">{"".join(body)}<div class="pn">{n}</div></div>'


def to_pdf(case: Case, out: Path) -> None:
    css = CSS % ({"font": FONTS[case.lang], **(TOUGH if case.profile == "tough" else CLEAN)})
    pages = "\n".join(page_html(p, i + 1, case.ruled, case.kind) for i, p in enumerate(case.pages))
    doc = f'<!doctype html><meta charset="utf-8"><style>{css}</style><body>{pages}</body>'
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "case.html"
        src.write_text(doc, encoding="utf-8")
        subprocess.run(
            [CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--user-data-dir={tmp}/profile",
             f"--print-to-pdf={out}", src.as_uri()],
            check=True, capture_output=True, timeout=90,
        )


def table_boxes(pdf: Path, case: Case) -> dict[int, list[float] | None]:
    boxes: dict[int, list[float] | None] = {}
    doc = pymupdf.open(pdf)
    for i, spec in enumerate(case.pages):
        if i >= doc.page_count or spec.kind != "table":
            boxes[i + 1] = None
            continue
        page = doc[i]
        rects = [d["rect"] for d in page.get_drawings() if d.get("rect") is not None]
        if not rects:
            boxes[i + 1] = None
            continue
        x0, y0 = min(r.x0 for r in rects), min(r.y0 for r in rects)
        x1, y1 = max(r.x1 for r in rects), max(r.y1 for r in rects)
        w, h = page.rect.width, page.rect.height
        boxes[i + 1] = [round(x0 / w, 4), round(y0 / h, 4), round(x1 / w, 4), round(y1 / h, 4)]
    return boxes


def scan_like(pdf: Path, out: Path, seed: int, profile: str = "clean") -> None:
    """Image-only copy: small skew, blur, grey paper, noise, specks and JPEG compression."""
    dpi, skew, blur, contrast, noise, quality = SCAN[profile]
    rng = np.random.default_rng(seed)
    src, dst = pymupdf.open(pdf), pymupdf.open()
    for page in src:
        pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
        img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
        img = img.rotate(float(rng.uniform(-skew, skew)), resample=Image.BICUBIC, fillcolor=255)
        img = img.filter(ImageFilter.GaussianBlur(float(rng.uniform(*blur))))
        arr = np.asarray(img, dtype=np.float32)
        arr = arr * float(rng.uniform(*contrast)) + float(rng.uniform(5, 25))
        arr += rng.normal(0, float(rng.uniform(*noise)), arr.shape)
        for _ in range(int(rng.integers(20, 80))):  # dust
            y, x = int(rng.integers(0, arr.shape[0] - 3)), int(rng.integers(0, arr.shape[1] - 3))
            arr[y:y + 2, x:x + 2] -= float(rng.uniform(60, 140))
        buf = io.BytesIO()
        Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(buf, "JPEG", quality=int(rng.integers(*quality)))
        new = dst.new_page(width=page.rect.width, height=page.rect.height)
        new.insert_image(new.rect, stream=buf.getvalue())
    dst.save(out)


def render_case(case: Case, work: Path, seed: int = 0, scanned: bool = True) -> Rendered:
    work.mkdir(parents=True, exist_ok=True)
    pdf = work / f"{case.id}.pdf"
    to_pdf(case, pdf)
    n_pdf = pymupdf.open(pdf).page_count
    if n_pdf != len(case.pages):
        return Rendered(pdf, None, {}, False, f"{n_pdf} pages printed, {len(case.pages)} expected (table overflowed)")
    boxes = table_boxes(pdf, case)
    right_edge = 1 - 14 / 210  # A4 width minus the 14 mm margin
    if any(b and b[2] > right_edge + 0.003 for b in boxes.values()):
        return Rendered(pdf, None, boxes, False, "table wider than the page")
    scan = None
    if scanned:
        scan = work / f"{case.id}.scan.pdf"
        scan_like(pdf, scan, seed, case.profile)
    ok = all(boxes[i + 1] for i, p in enumerate(case.pages) if p.kind == "table")
    return Rendered(pdf, scan, boxes, ok, "" if ok else "no table box found")
