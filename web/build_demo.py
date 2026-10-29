"""Build the landing page's demo from a real run: the page image and its blocks (boxes normalised 0-1).

    python web/build_demo.py out/docs/ponniyan-selvan-b7ea5135 3

Reads the saved Bodhan reading of one page and renders the same page upright at the width the page uses.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.manifest import Manifest  # noqa: E402
from src.pdf_pages import RenderSpec, render_page  # noqa: E402

DEMO = Path(__file__).resolve().parent / "static" / "demo"
SHOWN = {"Section-title", "Paragraph", "List", "Folio", "Page-number"}  # labels whose text is shown


def main(run_dir: str, page: int) -> None:
    """Render one page of a finished run and write the demo's image and blocks into web/static/demo."""
    run = Path(run_dir)
    saved = json.loads((run / "bodhan" / "pages" / f"page_{page:05d}.json").read_text(encoding="utf-8"))
    info = saved["render"]
    w, h = info["width"], info["height"]
    manifest = Manifest(run / "manifest.sqlite")
    pdf = Path(manifest.get_meta("source"))
    manifest.close()
    with pymupdf.open(pdf) as doc:
        img = render_page(doc, page, RenderSpec(dpi=info["dpi"], rotate=info["rotate"]))
    DEMO.mkdir(parents=True, exist_ok=True)
    pix = pymupdf.Pixmap(img.data)
    pix.save(DEMO / "page.jpg", jpg_quality=82)
    blocks = []
    for b in saved["payload"]["blocks"]:
        x0, y0, x1, y1 = b["bbox_xyxy"]
        blocks.append({
            "label": b["label"], "text": (b.get("text") or "").strip(), "shown": b["label"] in SHOWN and bool(b.get("text")),
            "box": [round(x0 / w, 4), round(y0 / h, 4), round((x1 - x0) / w, 4), round((y1 - y0) / h, 4)],
        })
    out = {"source": f"{pdf.name}, page {page}", "language": "Tamil", "width": img.width, "height": img.height, "blocks": blocks}
    (DEMO / "page.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{img.width}x{img.height}, {len(blocks)} blocks, {sum(b['shown'] for b in blocks)} with text")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
