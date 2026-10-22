"""Free per-page checks before any API call (PyMuPDF only).

For each page: the rotation that makes text upright, whether it is blank,
which script the embedded text layer is in (and whether it uses a legacy
font), where vector-ruled tables are, and the OCR mode for Sarvam.

OCR mode. Sarvam's `language` code only switches between Indic and English
reading (measured: every Indic code gives identical output, `en-IN` corrupts
Indic digits). So triage decides just the mode, and leans to Indic:
English only when the text layer is clean Latin with no Indic script. The
actual languages are identified later, per block, from the OCR output
(langid.py), and an English-mode page that comes back Indic is re-read.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import pymupdf

from src.page_class import classify
from src.pdf_pages import upright_rotation

# Unicode block per script, and the script each language code is written in.
SCRIPT_RANGES = {
    "Devanagari": (0x0900, 0x097F),
    "Bengali": (0x0980, 0x09FF),
    "Gurmukhi": (0x0A00, 0x0A7F),
    "Gujarati": (0x0A80, 0x0AFF),
    "Oriya": (0x0B00, 0x0B7F),
    "Tamil": (0x0B80, 0x0BFF),
    "Telugu": (0x0C00, 0x0C7F),
    "Kannada": (0x0C80, 0x0CFF),
    "Malayalam": (0x0D00, 0x0D7F),
    "Arabic": (0x0600, 0x06FF),
    "OlChiki": (0x1C50, 0x1C7F),
    "MeeteiMayek": (0xABC0, 0xABFF),
}
LANGUAGE_SCRIPT = {
    "hi": "Devanagari", "mr": "Devanagari", "ne": "Devanagari", "sa": "Devanagari",
    "kok": "Devanagari", "mai": "Devanagari", "doi": "Devanagari", "brx": "Devanagari",
    "bn": "Bengali", "as": "Bengali", "pa": "Gurmukhi", "gu": "Gujarati",
    "od": "Oriya", "ta": "Tamil", "te": "Telugu", "kn": "Kannada", "ml": "Malayalam",
    "ur": "Arabic", "ks": "Arabic", "sd": "Arabic", "sat": "OlChiki", "mni": "Bengali",
    "en": "Latin",
}
SCRIPT_DEFAULT_LANGUAGE = {
    "Devanagari": "hi-IN", "Bengali": "bn-IN", "Gurmukhi": "pa-IN", "Gujarati": "gu-IN",
    "Oriya": "od-IN", "Tamil": "ta-IN", "Telugu": "te-IN", "Kannada": "kn-IN",
    "Malayalam": "ml-IN", "Arabic": "ur-IN", "OlChiki": "sat-IN", "MeeteiMayek": "mni-IN",
    "Latin": "en-IN",
}
# Legacy (non-Unicode) Indic fonts map glyphs onto Latin-1 code points.
LEGACY_FONT_RE = re.compile(
    r"DV-?TT|Kruti|Shivaji|Walkman|Chanakya|APS-?DV|Akruti|Shree-?Dev|"
    r"Baamini|Bamini|TAB-|TAM-|Vanavil|SunTommy|Nudi",
    re.I,
)


def language_script(language: str) -> str:
    return LANGUAGE_SCRIPT.get(language.split("-")[0].lower(), "Latin")


def char_script(ch: str) -> str | None:
    cp = ord(ch)
    if ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
        return "Latin"
    for name, (lo, hi) in SCRIPT_RANGES.items():
        if lo <= cp <= hi:
            return name
    if 0x00C0 <= cp <= 0x024F:
        return "Latin1"  # accented Latin; in an Indic PDF this means a legacy font
    return None


@dataclass
class PageFlags:
    page: int
    rotate: int
    blank: bool
    text_chars: int
    script_share: float  # share of letters in the expected script; NOT a correctness score
    legacy_font: bool
    script: str | None  # dominant script of the text layer, if any
    language: str  # code sent to Sarvam: "en-IN", an Indic code, or "auto" (field omitted)
    width_pt: float
    height_pt: float
    table_bboxes: list[list[float]] = field(default_factory=list)  # unrotated page pt
    ocr_mode: str = "indic"  # indic | en
    mode_source: str = "default"  # text_layer | user | default
    lang_hint: dict | None = None  # languages in a clean Unicode text layer, before OCR
    render: str = "vector"  # page_class: vector | raster
    text_layer: str = "none"  # page_class: unicode | legacy | hidden_ocr | none | ignored
    image_cover: float = 0.0

    @property
    def has_table(self) -> bool:
        return bool(self.table_bboxes)

    def to_row(self) -> dict:
        row = asdict(self)
        row["blank"] = int(self.blank)
        row["legacy_font"] = int(self.legacy_font)
        return row


def body_tables(page: pymupdf.Page) -> list[list[float]]:
    """PyMuPDF's table boxes, minus the ones that cannot be a body table: a box thinner than 3% of the page
    height, or entirely in the top or bottom 10% (a page-number box, a running header). Annual reports draw
    such a 3 x 2 box on every page, which made every page look like a table page and sent every page to Bodhan
    (an extra ₹0.20 a page, and a queue of 8 requests a minute). Only a routing hint: a table Sarvam finds on
    its own still gets its Bodhan reading (Router.after_sarvam_page)."""
    height = page.rect.height
    boxes: list[list[float]] = []
    try:
        for table in page.find_tables().tables:
            x0, y0, x1, y1 = table.bbox
            if y1 - y0 < 0.03 * height or y1 <= 0.10 * height or y0 >= 0.90 * height:
                continue
            boxes.append([round(v, 2) for v in table.bbox])
    except Exception:  # noqa: BLE001 - table finder is best effort
        pass
    return boxes


def _is_blank(page: pymupdf.Page) -> bool:
    pix = page.get_pixmap(dpi=18, colorspace=pymupdf.csGRAY, alpha=False)
    samples = pix.samples
    dark = sum(1 for v in samples if v < 200)
    return dark / max(len(samples), 1) < 0.001


def triage_page(page: pymupdf.Page, language: str, doc_type: str = "auto") -> PageFlags:
    """`language`: the user's code ("hi-IN", "en-IN", ...) or "auto"; `doc_type`: auto | digital | scanned."""
    from src.langid import SARVAM_LANGUAGES, detect

    user = language if language in SARVAM_LANGUAGES else "auto"
    indic_code = user if user not in ("auto", "en-IN") else "auto"
    text = page.get_text("text")
    counts: dict[str, int] = {}
    for ch in text:
        script = char_script(ch)
        if script:
            counts[script] = counts.get(script, 0) + 1
    letters = sum(counts.values())
    dominant = max(counts, key=counts.get) if counts else None
    expected = language_script(user) if user != "auto" else (dominant or "Latin")
    quality = counts.get(expected, 0) / letters if letters else 0.0

    fonts = [f[3] for f in page.get_fonts()]
    legacy = any(LEGACY_FONT_RE.search(name or "") for name in fonts)
    if expected != "Latin" and counts.get("Latin1", 0) > 0.2 * max(letters, 1):
        legacy = True

    # OCR mode. Wrongly reading Indic text in English mode corrupts it;
    # reading English in Indic mode does not, so doubt means Indic.
    indic_letters = sum(n for s, n in counts.items() if s not in ("Latin", "Latin1"))
    latin = counts.get("Latin", 0)
    if legacy or indic_letters >= 10:
        mode, source = "indic", "text_layer"
    elif latin >= 50 and indic_letters <= 0.01 * latin:
        mode, source = "en", "text_layer"
    elif user == "en-IN":
        mode, source = "en", "user"  # no usable text: trust the user, verify after OCR
    else:
        mode, source = "indic", "user" if user != "auto" else "default"
    page_language = "en-IN" if mode == "en" else indic_code
    hint = None
    if not legacy and letters >= 30:
        found = detect(text, prior=user if user != "auto" else None)
        hint = {"shares": found.shares, "devanagari_confidence": found.devanagari_confidence}

    tables = body_tables(page)

    blank = letters == 0 and not tables and _is_blank(page)
    cls = classify(page, legacy, doc_type)
    return PageFlags(
        page=page.number + 1,
        rotate=upright_rotation(page),
        blank=blank,
        text_chars=letters,
        script_share=round(quality, 3),
        legacy_font=legacy,
        script=dominant,
        language=page_language,
        width_pt=round(page.rect.width, 2),
        height_pt=round(page.rect.height, 2),
        table_bboxes=tables,
        ocr_mode=mode,
        mode_source=source,
        lang_hint=hint,
        render=cls.render,
        text_layer=cls.text_layer,
        image_cover=cls.image_cover,
    )
