"""Accuracy of a run against a proofread Wikisource book (free; the book's PDF and a finished run are needed).

    python -m bench.wikisource_eval hi "विषयसूची:Ramanama.pdf" out/docs/<run>/doc.md [more doc.md ...]

The reference is the text of every proofread page of one Index on `<lang>.wikisource.org`, with the
layout templates replaced by the text they wrap. Page N of the index is page N of the PDF.

    strict    character and word error rate over the whole page
    covered   the same, but the reference is aligned to its best-matching stretch of the OCR text, so text a
              Wikisource page lacks (a truncated transcription) is not charged. What the OCR wrote beyond the
              reference is reported as overrun = OCR characters / reference characters.

Pages whose reference is under 30 characters (a cover, a plate) are left out. Footnotes go to the end of the
reference page, where they are printed. Scoring is by the page markers `<!-- page N ... -->` in doc.md.
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import requests

HEADERS = {"User-Agent": "infinite-ocr-eval/0.1"}
KEEP_FIRST = {"c", "center", "right", "left", "larger", "x-larger", "xx-larger", "xxx-larger", "smaller", "x-smaller",
              "xx-smaller", "sc", "bold", "b", "i", "u"}   # layout templates whose first argument is the printed text
MIN_REFERENCE_CHARS = 30


def fetch_wikitext(lang: str, index: str) -> dict[int, str]:
    """{page number: wikitext} of every proofread page of one Index."""
    api = f"https://{lang}.wikisource.org/w/api.php"

    def get(**params) -> dict:
        response = requests.get(api, params={**params, "format": "json", "formatversion": 2}, headers=HEADERS, timeout=60)
        response.raise_for_status()
        return response.json()

    listing = get(action="query", list="proofreadpagesinindex", prppiititle=index, prppiiprop="ids|title")
    pages = listing["query"]["proofreadpagesinindex"]
    out: dict[int, str] = {}
    for k in range(0, len(pages), 50):
        ids = "|".join(str(p["pageid"]) for p in pages[k : k + 50])
        got = get(action="query", pageids=ids, prop="revisions", rvprop="content", rvslots="main")
        for page in got["query"]["pages"]:
            out[int(page["title"].rsplit("/", 1)[1])] = page["revisions"][0]["slots"]["main"]["content"]
    return out


def _template(match: re.Match) -> str:
    name, *args = match.group(1).split("|")
    return args[0] if name.strip().lower() in KEEP_FIRST and args else " "


def printed_text(wikitext: str) -> str:
    """The text printed on one page: running header and footer dropped, templates unwrapped, footnotes last."""
    t = re.sub(r"<noinclude>.*?</noinclude>", " ", wikitext, flags=re.S)
    t = re.sub(r"<!--.*?-->", " ", t, flags=re.S)
    notes = re.findall(r"<ref[^>]*>(.*?)</ref>", t, flags=re.S)
    t = re.sub(r"<ref[^>]*>.*?</ref>", " ", t, flags=re.S) + "\n" + "\n".join(notes)
    for _ in range(5):  # innermost templates first
        t = re.sub(r"\{\{([^{}]*)\}\}", _template, t)
    t = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", t)
    t = re.sub(r"'{2,}", "", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"^[:;*#]+", "", t, flags=re.M)
    return re.sub(r"\s+", " ", t).strip()


def normalise(text: str) -> str:
    """Scoring form of OCR output or reference: no figure links, tags or markdown marks; NFC; one space."""
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[#*_`|>•]+", " ", text)
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def edits(ref: list, hyp: list, free_ends: bool) -> int:
    """Levenshtein distance of two sequences; with free_ends, ref may match any stretch of hyp. Row-vectorised."""
    if not ref or not hyp:
        return len(ref)
    ids: dict = {}
    r = np.array([ids.setdefault(c, len(ids)) for c in ref])
    h = np.array([ids.setdefault(c, len(ids)) for c in hyp])
    col = np.arange(len(h) + 1)
    prev = np.zeros(len(h) + 1, dtype=np.int64) if free_ends else col.astype(np.int64)
    for i in range(1, len(r) + 1):
        cur = np.empty_like(prev)
        cur[0] = i
        cur[1:] = np.minimum(prev[1:] + 1, prev[:-1] + (h != r[i - 1]))
        prev = np.minimum.accumulate(cur - col) + col  # the left-neighbour step, for the whole row at once
    return int(prev.min() if free_ends else prev[-1])


def pages_of(markdown: str) -> dict[int, str]:
    parts = re.split(r"<!-- page (\d+) [^>]*-->", markdown)
    return {int(parts[i]): parts[i + 1] for i in range(1, len(parts), 2)}


def score(markdown: str, reference: dict[int, str]) -> dict:
    """Per-page and overall error rates of one doc.md against {page: printed text}."""
    got = pages_of(markdown)
    rows = []
    for page in sorted(reference):
        ref, hyp = normalise(reference[page]), normalise(got.get(page, ""))
        if len(ref) < MIN_REFERENCE_CHARS:
            continue
        ref_words, hyp_words = ref.split(), hyp.split()
        rows.append({
            "page": page, "ref_chars": len(ref), "ref_words": len(ref_words), "overrun": len(hyp) / len(ref),
            "strict_cer": edits(list(ref), list(hyp), False) / len(ref),
            "covered_cer": edits(list(ref), list(hyp), True) / len(ref),
            "strict_wer": edits(ref_words, hyp_words, False) / len(ref_words),
            "covered_wer": edits(ref_words, hyp_words, True) / len(ref_words),
        })

    def overall(key: str, weight: str) -> float:
        return sum(r[key] * r[weight] for r in rows) / sum(r[weight] for r in rows)

    median = sorted(r["covered_cer"] for r in rows)[len(rows) // 2]
    return {
        "pages": len(rows),
        **{k: overall(k, "ref_chars") for k in ("strict_cer", "covered_cer")},
        **{k: overall(k, "ref_words") for k in ("strict_wer", "covered_wer")},
        "median_covered_cer": median,
        "overrun_over_1_5": [r["page"] for r in rows if r["overrun"] > 1.5],
        "covered_cer_over_10pct": [r["page"] for r in rows if r["covered_cer"] > 0.10],
        "rows": rows,
    }


def error_profile(markdown: str, reference: dict[int, str]) -> dict[str, float]:
    """Share of character edits by kind, over pages of 300+ characters whose length is within 20% of the reference.

    Edits longer than 25 characters are layout differences (a moved or missing block), not misreadings, and are
    left out. Kinds: an `अ` plus a vowel sign (this book's older spelling) read as one vowel or the reverse;
    a vowel sign, anusvara or nukta dropped or swapped; digits and punctuation; the rest.
    """
    got = pages_of(markdown)
    digraph = re.compile("अ[ेीिुूैो]")
    signs = set("ंँ़्ुूेैोौिीाॉॊ ")
    counts = {"digraph": 0, "vowel sign": 0, "digits and punctuation": 0, "other": 0}
    for page, text in reference.items():
        ref, hyp = normalise(text), normalise(got.get(page, ""))
        if len(ref) < 300 or not 0.8 < len(hyp) / len(ref) < 1.25:
            continue
        for tag, i1, i2, j1, j2 in SequenceMatcher(None, ref, hyp, autojunk=False).get_opcodes():
            a, b = ref[i1:i2], hyp[j1:j2]
            size = max(len(a), len(b))
            if tag == "equal" or size > 25:
                continue
            if digraph.search(a) or digraph.search(b):
                counts["digraph"] += size
            elif set(a + b) <= signs:
                counts["vowel sign"] += size
            elif re.fullmatch(r"[०-९\-–—'’‘\"“”.,।:;()? ]*", a + b):
                counts["digits and punctuation"] += size
            else:
                counts["other"] += size
    total = sum(counts.values())
    return {kind: n / total for kind, n in counts.items()}


def main(argv: list[str]) -> None:
    if len(argv) < 4:
        sys.exit(__doc__)
    lang, index, *docs = argv[1:]
    reference = {page: printed_text(text) for page, text in fetch_wikitext(lang, index).items()}
    for path in docs:
        result = score(Path(path).read_text(encoding="utf-8"), reference)
        summary = {k: round(v, 4) if isinstance(v, float) else v for k, v in result.items() if k != "rows"}
        print(path, json.dumps(summary, ensure_ascii=False))
        profile = error_profile(Path(path).read_text(encoding="utf-8"), reference)
        print("  errors by kind:", {k: f"{v:.1%}" for k, v in profile.items()})


if __name__ == "__main__":
    main(sys.argv)
