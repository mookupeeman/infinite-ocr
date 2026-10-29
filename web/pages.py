"""One page of a finished job, for the inspector: an upright image and its blocks with boxes in 0-1.

Tables are sent as cells (text only); the browser builds them with textContent, so nothing the OCR
engines returned is ever interpreted as HTML.
"""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf

from src.assemble import load_blocks
from src.manifest import Manifest
from src.pdf_pages import RenderSpec, render_page
from src.tables import parse_tables

VIEW_DPI = 130


def _lead_order(manifest: Manifest) -> tuple[str, ...]:
    lead = json.loads(manifest.get_meta("fingerprint") or "{}").get("lead", "sarvam")
    return (lead,) + tuple(e for e in ("sarvam", "bodhan", "qwen") if e != lead)


def _table_cells(text: str) -> dict | None:
    tables = parse_tables(text)
    if not tables:
        return None
    t = tables[0]
    return {"rows": t.n_rows, "cols": t.n_cols,
            "cells": [{"r": c.r, "c": c.c, "rs": c.rowspan, "cs": c.colspan, "t": c.text, "h": c.header} for c in t.cells]}


def page_payload(out_dir: Path, page: int) -> dict:
    """Blocks of one page with boxes in 0-1 and tables as cells, for the inspector."""
    manifest = Manifest(out_dir / "manifest.sqlite")
    try:
        info = manifest.page_info(page)
        order = _lead_order(manifest)
    except KeyError as exc:
        raise LookupError(f"Page {page} was not read.") from exc
    finally:
        manifest.close()
    engine = next((e for e in order if (out_dir / "blocks" / f"page_{page:05d}.{e}.json").is_file()), None)
    if engine is None:
        raise LookupError(f"Page {page} has no result yet.")
    blocks = []
    for b in load_blocks(out_dir / "blocks" / f"page_{page:05d}.{engine}.json"):
        x0, y0, x1, y1 = b.bbox
        item = {"label": b.raw_label, "kind": b.kind, "text": b.text.strip(), "lang": b.lang,
                "box": [round(x0, 4), round(y0, 4), round(x1 - x0, 4), round(y1 - y0, 4)]}
        if b.kind == "table":
            item["table"] = _table_cells(b.text)
        item["shown"] = bool(item["text"]) and b.kind != "figure"
        blocks.append(item)
    return {"page": page, "engine": engine, "blocks": blocks}


def page_image(out_dir: Path, page: int) -> Path:
    """The page as the pipeline saw it (turned upright), cached next to the job's outputs."""
    cache = out_dir / "view" / f"page_{page:05d}.jpg"
    if cache.is_file():
        return cache
    manifest = Manifest(out_dir / "manifest.sqlite")
    try:
        info = manifest.page_info(page)
        source = Path(manifest.get_meta("source") or "")
    except KeyError as exc:
        raise LookupError(f"Page {page} is not available.") from exc
    finally:
        manifest.close()
    if not source.is_file():
        raise LookupError(f"Page {page} is not available.")
    with pymupdf.open(source) as doc:
        rendered = render_page(doc, page, RenderSpec(dpi=VIEW_DPI, rotate=info["rotate"]))
    cache.parent.mkdir(parents=True, exist_ok=True)
    pymupdf.Pixmap(rendered.data).save(cache, jpg_quality=85)
    return cache
