"""Language identification for OCR output, per block and per page. Free, offline.

Measured 2026-10-10 on a four-language test page: Sarvam's `language` code is
an Indic/English switch. Every Indic code, and no code at all, gave
byte-identical output for English, Tamil, Hindi and Marathi sections; only
`en-IN` changed the result, and it corrupted Indic digits (२०२१ -> २0२९).
So the language never has to be known before OCR, except to keep Indic pages
out of English mode. It is identified here, from the text the engines return.

How
1. Script from Unicode ranges (each Indic script is its own block).
2. Within a script shared by several languages:
   - Devanagari: function words and suffixes (Hindi है/और/में, Marathi आहे/आणि/च्या,
     Nepali छ/पनि/हरू, Sanskrit अस्ति/इति, Konkani आसा/आनी, Maithili अछि), plus ळ.
   - Bengali script: Assamese uses ৰ and ৱ.
   - Arabic script: Sindhi and Kashmiri letters that Urdu does not use.
3. Short blocks (table cells, headings) have no function words; they take the
   page's decision, and the page takes the document's prior.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

# Sarvam Vision's 23 language codes (docs.sarvam.ai, Sarvam Vision page).
SARVAM_LANGUAGES = {
    "hi-IN", "bn-IN", "ta-IN", "te-IN", "mr-IN", "gu-IN", "kn-IN", "ml-IN", "od-IN", "pa-IN",
    "as-IN", "ur-IN", "sa-IN", "ne-IN", "doi-IN", "brx-IN", "kok-IN", "mai-IN", "sd-IN",
    "ks-IN", "mni-IN", "sat-IN", "en-IN",
}

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
# Scripts that belong to one language only.
SINGLE = {
    "Gurmukhi": "pa-IN", "Gujarati": "gu-IN", "Oriya": "od-IN", "Tamil": "ta-IN",
    "Telugu": "te-IN", "Kannada": "kn-IN", "Malayalam": "ml-IN", "OlChiki": "sat-IN",
    "MeeteiMayek": "mni-IN", "Latin": "en-IN",
}
SCRIPT_OF = {
    **{lang: script for script, lang in SINGLE.items()},
    "hi-IN": "Devanagari", "mr-IN": "Devanagari", "ne-IN": "Devanagari", "sa-IN": "Devanagari",
    "kok-IN": "Devanagari", "mai-IN": "Devanagari", "doi-IN": "Devanagari", "brx-IN": "Devanagari",
    "bn-IN": "Bengali", "as-IN": "Bengali", "ur-IN": "Arabic", "ks-IN": "Arabic", "sd-IN": "Arabic",
}
SCRIPT_DEFAULT = {"Devanagari": "hi-IN", "Bengali": "bn-IN", "Arabic": "ur-IN"}

# Whole-word markers. Chosen for precision: words common in BOTH languages are left out.
DEVANAGARI_MARKERS: dict[str, set[str]] = {
    "hi-IN": {"है", "हैं", "और", "में", "नहीं", "यह", "वह", "था", "थी", "थे", "गया", "गई", "किया",
              "लिए", "भी", "द्वारा", "हुआ", "होता", "होती", "करने", "जाता", "जाती", "इसके", "इस",
              "उसके", "उनके", "किसी", "एवं", "का", "की", "के", "को", "से", "पर", "तक", "कुल"},
    "mr-IN": {"आहे", "आहेत", "आणि", "नाही", "होते", "होती", "होता", "केले", "केली", "केला",
              "करण्यात", "आली", "आले", "आला", "मध्ये", "साठी", "यांच्या", "यांचे", "यांची",
              "त्यांच्या", "येथे", "म्हणून", "पण", "एकूण", "ते", "हे", "या", "व", "करिता", "तसेच"},
    "ne-IN": {"छ", "छन्", "पनि", "गर्न", "भएको", "लागि", "छैन", "थियो", "गरेको", "हुन्छ", "गर्ने"},
    "sa-IN": {"अस्ति", "इति", "च", "एव", "भवति", "सः", "अहम्", "वयम्", "तत्", "कृत्वा", "यत्"},
    "kok-IN": {"आसा", "आनी", "हांगा", "जाल्यार", "नाका", "खातीर", "म्हळ्ळें"},
    "mai-IN": {"अछि", "छल", "सँ", "केँ", "अपन"},
}
DEVANAGARI_SUFFIXES = {"mr-IN": ("च्या", "ांच्या"), "ne-IN": ("हरू",)}
ASSAMESE_LETTERS = {"ৰ", "ৱ"}  # ৰ ৱ
SINDHI_LETTERS = set("ڄڃڇڏڙڪڳڱڻٻٽٿڀڦڊڍڌڎ")
KASHMIRI_LETTERS = set("ۆێٲٳؠ")
_MARKUP = re.compile(r"<[^>]+>|\\[A-Za-z]+|\$\$.*?\$\$|\$[^$\n]{0,80}\$", re.S)
_WORD = re.compile(r"[^\s\d.,;:()\[\]{}<>/\\|\-–—'\"“”‘’!?।॥=+*$%^&_#@`~]+")


def script_of_char(ch: str) -> str | None:
    if ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
        return "Latin"
    cp = ord(ch)
    for name, (lo, hi) in SCRIPT_RANGES.items():
        if lo <= cp <= hi:
            return name
    return None


def strip_markup(text: str) -> str:
    """Remove HTML tags, LaTeX commands and $math$: their letters are not language."""
    return _MARKUP.sub(" ", text)


def _letters(text: str) -> Counter:
    counts: Counter = Counter()
    for ch in strip_markup(text):
        script = script_of_char(ch)
        if script and (script == "Latin" or ch.isalpha() or 0x0900 <= ord(ch) <= 0x0DFF):
            counts[script] += 1
    return counts


def devanagari_scores(text: str) -> Counter:
    scores: Counter = Counter()
    for word in _WORD.findall(text):
        for lang, words in DEVANAGARI_MARKERS.items():
            if word in words:
                scores[lang] += 1
        for lang, suffixes in DEVANAGARI_SUFFIXES.items():
            if word.endswith(suffixes) and word not in DEVANAGARI_MARKERS.get(lang, ()):
                scores[lang] += 1
    if "ळ" in text:  # common in Marathi, rare in Hindi
        scores["mr-IN"] += text.count("ळ") * 0.5
    return scores


def decide_devanagari(scores: Counter, prior: str | None) -> tuple[str, float]:
    """Language and confidence (0-1) from marker scores; the prior breaks weak cases."""
    total = sum(scores.values())
    if total < 2:
        return (prior if prior and SCRIPT_OF.get(prior) == "Devanagari" else "hi-IN"), 0.0
    best, top = scores.most_common(1)[0]
    second = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0
    confidence = (top - second) / total
    if confidence < 0.2 and prior and SCRIPT_OF.get(prior) == "Devanagari":
        return prior, confidence
    return best, round(confidence, 3)


def language_of_script(script: str, text: str, prior: str | None, devanagari_hint: str | None = None) -> str:
    if script in SINGLE:
        return SINGLE[script]
    if script == "Bengali":
        return "as-IN" if any(ch in ASSAMESE_LETTERS for ch in text) else (
            prior if prior in ("bn-IN", "as-IN", "mni-IN") else "bn-IN")
    if script == "Arabic":
        if any(ch in SINDHI_LETTERS for ch in text):
            return "sd-IN"
        if any(ch in KASHMIRI_LETTERS for ch in text):
            return "ks-IN"
        return "ur-IN"
    if script == "Devanagari":
        if devanagari_hint:
            return devanagari_hint
        return decide_devanagari(devanagari_scores(text), prior)[0]
    return prior or "en-IN"


@dataclass
class TextLanguages:
    """Letter shares per language for a piece of text, e.g. {'hi-IN': 0.78, 'en-IN': 0.22}."""

    shares: dict[str, float] = field(default_factory=dict)
    letters: int = 0
    devanagari_confidence: float | None = None

    @property
    def dominant(self) -> str | None:
        return max(self.shares, key=self.shares.get) if self.shares else None

    def present(self, threshold: float = 0.1) -> list[str]:
        """Languages with at least `threshold` of the letters, largest first."""
        return [k for k, v in sorted(self.shares.items(), key=lambda kv: -kv[1]) if v >= threshold]

    def to_dict(self) -> dict:
        return {"shares": self.shares, "letters": self.letters, "dominant": self.dominant,
                "devanagari_confidence": self.devanagari_confidence}


def detect(text: str, prior: str | None = None, devanagari_hint: str | None = None) -> TextLanguages:
    counts = _letters(text)
    letters = sum(counts.values())
    if not letters:
        return TextLanguages()
    shares: Counter = Counter()
    for script, n in counts.items():
        shares[language_of_script(script, text, prior, devanagari_hint)] += n
    conf = None
    if counts.get("Devanagari"):
        conf = decide_devanagari(devanagari_scores(text), prior)[1] if not devanagari_hint else None
    return TextLanguages({k: round(v / letters, 3) for k, v in shares.most_common()}, letters, conf)


def indic_share(text: str) -> float:
    counts = _letters(text)
    total = sum(counts.values())
    return (total - counts.get("Latin", 0)) / total if total else 0.0


SKIP_KINDS = {"math", "figure", "furniture"}


def page_languages(
    texts: list[str], prior: str | None = None, kinds: list[str] | None = None
) -> tuple[list[TextLanguages], TextLanguages]:
    """Per-block languages (in reading order) plus the page total.

    A Devanagari block with clear markers of its own (>= 2, one language well
    ahead) decides for itself. A block with too few markers (a table, a short
    heading) takes the language of the nearest decided block in reading order,
    so a Marathi table under a Marathi caption stays Marathi even when a Hindi
    paragraph sits higher on the same page. With no decided block on the page,
    the page's pooled markers decide, then the document prior.
    """
    kinds = kinds or [""] * len(texts)
    clean = [strip_markup(t) if k not in SKIP_KINDS else "" for t, k in zip(texts, kinds)]
    page_hint, page_conf = decide_devanagari(devanagari_scores("\n".join(clean)), prior)
    decided: dict[int, str] = {}
    for i, text in enumerate(clean):
        own = devanagari_scores(text)
        if sum(own.values()) >= 2:
            lang, conf = decide_devanagari(own, None)
            if conf >= 0.6:
                decided[i] = lang
    per_block: list[TextLanguages] = []
    for i, text in enumerate(clean):
        if not text.strip():
            per_block.append(TextLanguages())
            continue
        hint = decided.get(i)
        if hint is None and decided:
            nearest = min(decided, key=lambda j: (abs(j - i), j > i))  # prefer the block before
            hint = decided[nearest]
        per_block.append(detect(text, prior, devanagari_hint=hint or page_hint))
    total = Counter()
    letters = 0
    for item in per_block:
        for lang, share in item.shares.items():
            total[lang] += share * item.letters
        letters += item.letters
    page = TextLanguages(
        {k: round(v / letters, 3) for k, v in total.most_common()} if letters else {},
        letters,
        page_conf if "Devanagari" in {SCRIPT_OF.get(k) for k in total} else None,
    )
    return per_block, page
