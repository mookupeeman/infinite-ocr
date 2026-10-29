"""One block format for every engine.

Bodhan (37 labels) and Sarvam (`layout_tag`) blocks become `Block`s with a
small set of kinds. Boxes are normalised to 0-1 in the *upright* page frame:
Bodhan sees our upright render, and Sarvam sees our upright chunk PDF with
MediaBox = CropBox, so both land in the same frame.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

KINDS = ("heading", "text", "contents", "table", "caption", "figure", "math", "footnote", "furniture")

# label -> (kind, heading level)
BODHAN_KIND: dict[str, tuple[str, int | None]] = {
    "Title": ("heading", 1),
    "Chapter-title": ("heading", 1),
    "Section-title": ("heading", 2),
    "Sub-section-title": ("heading", 3),
    "Sub-sub-section-title": ("heading", 4),
    "Paragraph": ("text", None),
    "List": ("text", None),
    "Infobox": ("text", None),
    "Question": ("text", None),
    "Answer": ("text", None),
    "MCQ": ("text", None),
    "Solved-example": ("text", None),
    "Code": ("text", None),
    "Reference": ("text", None),
    "Author": ("text", None),
    "Dateline": ("text", None),
    "Contact-info": ("text", None),
    "Website-link": ("text", None),
    "Placeholder-text": ("text", None),
    "Chapter-end-section": ("text", None),
    "Table-of-contents": ("contents", None),
    "Index": ("contents", None),
    "Table": ("table", None),
    "Table-caption": ("caption", None),
    "Image-caption": ("caption", None),
    "Image": ("figure", None),
    "Chart": ("figure", None),
    "Diagram": ("figure", None),
    "Flag": ("figure", None),
    "Equation": ("math", None),
    "Expression": ("math", None),
    "Footnote": ("footnote", None),
    "Header": ("furniture", None),
    "Footer": ("furniture", None),
    "Page-number": ("furniture", None),
    "Folio": ("furniture", None),
    "Advertisement": ("furniture", None),
}

SARVAM_KIND: dict[str, tuple[str, int | None]] = {
    "headline": ("heading", 1),
    "title": ("heading", 1),
    "section-title": ("heading", 2),
    "sub-headline": ("heading", 3),
    "paragraph": ("text", None),
    "list": ("text", None),
    "table-of-contents": ("contents", None),
    "table": ("table", None),
    "image-caption": ("caption", None),
    "table-caption": ("caption", None),
    "image": ("figure", None),
    "photograph": ("figure", None),
    "chart": ("figure", None),
    "diagram": ("figure", None),
    "figure": ("figure", None),
    "equation": ("math", None),
    "formula": ("math", None),
    "footnote": ("footnote", None),
    "header": ("furniture", None),
    "footer": ("furniture", None),
    "page-number": ("furniture", None),
    "folio": ("furniture", None),
    "advertisement": ("furniture", None),
}


@dataclass
class Block:
    """One block of a page in the shared format: kind, text and a 0-1 box in the upright page frame, whichever engine read it."""
    page: int
    engine: str
    order: int
    kind: str
    level: int | None
    raw_label: str
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1 in 0-1, upright frame
    text: str
    conf: float | None = None
    lang: str | None = None  # set from the OCR text by langid.page_languages

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bbox"] = [round(v, 5) for v in self.bbox]
        return d


def _clip(v: float) -> float:
    return min(1.0, max(0.0, float(v)))


def _norm(box: Any, width: float, height: float) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = (float(v) for v in box)
    return (_clip(x0 / width), _clip(y0 / height), _clip(x1 / width), _clip(y1 / height))


def kind_of(engine: str, label: str) -> tuple[str, int | None]:
    table = BODHAN_KIND if engine == "bodhan" else SARVAM_KIND
    key = label if engine == "bodhan" else label.lower()
    return table.get(key, ("text", None))


def from_bodhan(payload: dict, page: int, width: int, height: int) -> list[Block]:
    """`width`/`height`: pixel size of the image we uploaded."""
    out: list[Block] = []
    for i, raw in enumerate(sorted(payload.get("blocks") or [], key=lambda b: b.get("order", 0))):
        label = str(raw.get("label") or "")
        kind, level = kind_of("bodhan", label)
        out.append(
            Block(
                page=page,
                engine="bodhan",
                order=i,
                kind=kind,
                level=level,
                raw_label=label,
                bbox=_norm(raw.get("bbox_xyxy") or (0, 0, width, height), width, height),
                text=str(raw.get("text") or ""),
                conf=raw.get("conf"),
            )
        )
    return merge_furniture(out)


def from_sarvam(page_json: dict, page: int) -> list[Block]:
    width = float(page_json.get("image_width") or 1)
    height = float(page_json.get("image_height") or 1)
    raws = sorted(page_json.get("blocks") or [], key=lambda b: b.get("reading_order", 0))
    out: list[Block] = []
    for i, raw in enumerate(raws):
        label = str(raw.get("layout_tag") or "")
        kind, level = kind_of("sarvam", label)
        if raw.get("bbox_norm"):
            bbox = tuple(_clip(v) for v in raw["bbox_norm"])
        else:
            c = raw.get("coordinates") or {}
            bbox = _norm((c.get("x1", 0), c.get("y1", 0), c.get("x2", width), c.get("y2", height)), width, height)
        out.append(
            Block(
                page=page,
                engine="sarvam",
                order=i,
                kind=kind,
                level=level,
                raw_label=label,
                bbox=bbox,  # type: ignore[arg-type]
                text=str(raw.get("text") or ""),
                conf=raw.get("confidence"),
            )
        )
    return merge_furniture(out)


def from_text_page(markdown: str, page: int, engine: str) -> list[Block]:
    """Engines without layout (Qwen): the whole page is one text block."""
    return [Block(page, engine, 0, "text", None, "page", (0.0, 0.0, 1.0, 1.0), markdown, None)]


def iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def merge_furniture(blocks: list[Block]) -> list[Block]:
    """Drop an empty furniture block that overlaps one with text (Header + Folio)."""
    drop: set[int] = set()
    furniture = [i for i, b in enumerate(blocks) if b.kind == "furniture"]
    for i in furniture:
        for j in furniture:
            if i != j and not blocks[i].text.strip() and blocks[j].text.strip():
                if iou(blocks[i].bbox, blocks[j].bbox) > 0.5:
                    drop.add(i)
    return [b for i, b in enumerate(blocks) if i not in drop]


def page_markdown(blocks: list[Block]) -> str:
    """Reading-order markdown for one page: headings get `#` levels, furniture is dropped."""
    parts: list[str] = []
    for b in blocks:
        text = b.text.strip()
        if b.kind == "furniture":
            continue
        if b.kind == "figure":
            parts.append(f"<!-- figure p{b.page} {b.raw_label} -->")
            continue
        if not text:
            continue
        if b.kind == "heading":
            parts.append(f"{'#' * (b.level or 2)} {' '.join(text.split())}")
        else:
            parts.append(text)
    return "\n\n".join(parts) + ("\n" if parts else "")
