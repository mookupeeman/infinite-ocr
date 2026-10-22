"""Page-text check (Think mode, free part): compare the two engines' prose word by word. No API calls.

Tables get cell-by-cell checking; the text around them was read by one engine unless the other also
read the page. In Think mode Bodhan reads every page, so the two readings of each page's prose
(paragraphs, headings, captions, footnotes) can be compared:

    words      each engine's prose in reading order, as whitespace-separated words (comparison form:
               table_qa.canon, so spacing, dashes and quotes do not count)
    disputes   the stretches where the two word lists differ (difflib), with some words of context
    deciders   in order, each only choosing between the two readings, never writing a value:
                 elsewhere  words one engine has as prose that the other engine has in a block it does
                            not count as prose (a running header, a figure's text, a table) -> primary's
                 format     one side is LaTeX or markup ($\\frac{a}{b}$): a way of writing, not a misread
                            (Bodhan writes inline maths and superscripts as LaTeX)   -> primary's
                 spacing    the readings differ only in spaces                    -> primary's
                 minor      they differ only in punctuation and none is a number  -> primary's
                 text_layer the PDF text layer, proven on the words both engines agree on (at least
                            90% of at least 20 tested words), contains exactly one reading, or contains
                            the words one engine has and the other lacks
                 open       nothing free can decide: listed for review
    comparison words are first stripped of markup: `$^{22}$` and `²²` are the same footnote mark, `\\textit{x}`,
               `*x*` and `x` the same word, `&#8221;` the same quote.
    guards     pages whose word lists barely overlap (a different reading order) are not compared;
               a dispute longer than 25 words is a layout difference, not a misread word; the layer
               never deletes printed words (a missing word is not proof it is not printed).

The primary engine's text is what the document shows; a decision for the other engine's reading
replaces the stretch in the primary's block text (in memory only: block files are not changed).
The image check for the disputes that stay open is not built: it is added only if this free part
leaves enough of them to matter (see bench/text_dispute_report.py).
"""

from __future__ import annotations

import html
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from src.blocks import Block
from src.table_qa import Layer, canon

KINDS = ("text", "heading", "caption", "footnote", "contents")
MIN_PAGE_OVERLAP = 0.6  # share of words in common below which the two readings are not compared
MAX_SPAN = 25  # words; a longer difference is layout, not a misread word
MIN_TESTED = 20  # agreed words that must be testable against the layer before it can be trusted
PROOF_SHARE = 0.9


@dataclass
class Word:
    raw: str
    key: str  # canon(raw)
    block: int  # index into the page's block list
    start: int  # character offsets in that block's text
    end: int


@dataclass
class Dispute:
    page: int
    primary: str  # the primary engine's words in the stretch ("" if it has none)
    other: str
    before: str  # context
    after: str
    decided_by: str = "open"  # spacing | minor | text_layer | open | layout
    chosen: str = "primary"  # primary | other
    block: int | None = None  # primary block holding the stretch, when it is one block
    span: tuple[int, int] | None = None  # character offsets in that block's text (an insertion: start == end)

    def to_dict(self) -> dict:
        return {"page": self.page, "primary": self.primary, "other": self.other, "before": self.before,
                "after": self.after, "by": self.decided_by, "chosen": self.chosen}


@dataclass
class PageText:
    page: int
    compared: bool
    reason: str = ""
    words: tuple[int, int] = (0, 0)
    overlap: float = 0.0
    layer: dict = field(default_factory=dict)
    disputes: list[Dispute] = field(default_factory=list)

    def counts(self) -> Counter:
        return Counter(d.decided_by for d in self.disputes)


_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
_MARKUP_EDGE = "$*_❖•"


def key_of(raw: str) -> str:
    """Comparison form of one word: markup removed, then table_qa.canon."""
    s = html.unescape(re.sub(r"&?#(\d+);", r"&#\1;", raw))  # an engine sometimes drops the & of an entity
    s = re.sub(r"\\text(?:it|bf|rm|sc)\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\$\^\{?([^${}]*)\}?\$", r"\1", s)  # a LaTeX superscript: the digits
    s = re.sub(r"\^\{?(\d+)\}?", r"\1", s)  # x^2, x^{12}: a plain-text superscript
    s = s.translate(_SUPERSCRIPT).replace("\\%", "%")
    return canon(s.strip(_MARKUP_EDGE))


def is_markup(text: str) -> bool:
    """Still LaTeX after normalising: inline maths such as $\\frac{ds}{dt}$ or $y."""
    return "$" in text or "\\" in text


def _letters_only(text: str) -> str:
    """No punctuation, symbols, spaces or marks of format: what is left is the wording."""
    text = html.unescape(re.sub(r"&?#(\d+);", r"&#\1;", text))
    return "".join(ch for ch in canon(text) if not unicodedata.category(ch)[0] in "PSZC")


def words_of(blocks: list[Block]) -> list[Word]:
    out = []
    for i, b in enumerate(blocks):
        if b.kind not in KINDS:
            continue
        for m in re.finditer(r"\S+", b.text or ""):
            key = key_of(m.group())
            if key:
                out.append(Word(m.group(), key, i, m.start(), m.end()))
    return out


def non_prose_of(blocks: list[Block]) -> str:
    """The wording of every block that is not prose: headers and footers, figures, tables, maths."""
    return "".join(_letters_only(b.text or "") for b in blocks if b.kind not in KINDS)


def _is_number(text: str) -> bool:
    return any(ch.isdigit() for ch in text)


def _flat(words: list[str]) -> str:
    return "".join(canon(" ".join(words)).split())


def _layer_proof(layer: Layer, agreed: list[str]) -> dict:
    tested = [r for r in (layer.contains(w) for w in agreed) if r is not None]
    share = sum(tested) / len(tested) if tested else 0.0
    return {"usable": True, "clean": layer.clean, "tested": len(tested), "found": round(share, 3),
            "proven": len(tested) >= MIN_TESTED and share >= PROOF_SHARE}


def compare(page: int, primary: list[Block], other: list[Block], layer_text: str | None = None) -> PageText:
    """Compare one page's prose in the two engines and settle what free evidence can."""
    a, b = words_of(primary), words_of(other)
    if not a or not b:
        return PageText(page, False, "one engine has no prose on this page", (len(a), len(b)))
    sm = SequenceMatcher(None, [w.key for w in a], [w.key for w in b], autojunk=False)
    matched = sum(size for _, _, size in sm.get_matching_blocks())
    overlap = matched / max(len(a), len(b))
    result = PageText(page, True, "", (len(a), len(b)), round(overlap, 3))
    if overlap < MIN_PAGE_OVERLAP:
        result.compared, result.reason = False, "the two readings barely overlap (a different reading order?)"
        return result
    layer = None
    if layer_text:
        lay = Layer.of(layer_text)
        agreed = [a[i + k].key for i, _, size in sm.get_matching_blocks() for k in range(size)]
        result.layer = _layer_proof(lay, agreed)
        layer = lay if result.layer["proven"] else None
    elsewhere = {"primary": non_prose_of(primary), "other": non_prose_of(other)}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        pa, pb = [w.raw for w in a[i1:i2]], [w.raw for w in b[j1:j2]]
        d = Dispute(page, " ".join(pa), " ".join(pb), " ".join(w.raw for w in a[max(0, i1 - 4):i1]),
                    " ".join(w.raw for w in a[i2:i2 + 4]))
        blocks = {w.block for w in a[i1:i2]}
        if len(blocks) == 1:
            d.block = next(iter(blocks))
            d.span = (a[i1].start, a[i2 - 1].end)
        elif not pa and i1 > 0:  # an insertion point: right after the previous primary word
            d.block, d.span = a[i1 - 1].block, (a[i1 - 1].end, a[i1 - 1].end)
        one_sided = (not pa and elsewhere["primary"] and _letters_only(d.other) in elsewhere["primary"]) or \
                    (not pb and elsewhere["other"] and _letters_only(d.primary) in elsewhere["other"])
        if max(len(pa), len(pb)) > MAX_SPAN and not one_sided:
            d.decided_by = "layout"
        elif one_sided and (pa or pb) and _letters_only(d.primary + d.other):
            d.decided_by = "elsewhere"
        elif is_markup(d.primary) or is_markup(d.other):
            d.decided_by = "format"
        elif _flat(pa) == _flat(pb):
            d.decided_by = "spacing"
        elif pa and pb and _letters_only(d.primary) == _letters_only(d.other) and not _is_number(d.primary + d.other):
            d.decided_by = "minor"
        elif layer is not None:
            ca, cb = (layer.contains(d.primary) if pa else None), (layer.contains(d.other) if pb else None)
            if pa and pb and ca is not None and cb is not None and ca != cb:
                d.decided_by, d.chosen = "text_layer", "primary" if ca else "other"
            elif (not pa and cb) or (not pb and ca):  # the layer confirms the words one engine has and the other lacks
                d.decided_by, d.chosen = "text_layer", "other" if cb else "primary"
        if d.chosen == "other" and d.span is None:
            d.decided_by, d.chosen = "open", "primary"  # the stretch cannot be located in one block: leave it
        result.disputes.append(d)
    return result


def apply(blocks: list[Block], page_text: PageText) -> int:
    """Put the decided `other` readings into the primary's block texts (in memory). Returns how many were applied."""
    edits: dict[int, list[tuple[int, int, str]]] = {}
    for d in page_text.disputes:
        if d.chosen == "other" and d.decided_by == "text_layer" and d.block is not None and d.span is not None:
            edits.setdefault(d.block, []).append((d.span[0], d.span[1], d.other))
    n = 0
    for idx, items in edits.items():
        text = blocks[idx].text
        for start, end, new in sorted(items, reverse=True):
            if start == end:  # an insertion
                text = text[:start] + (" " + new if new else "") + text[end:]
            else:
                text = text[:start] + new + text[end:]
            n += 1
        blocks[idx].text = text
    return n


def summarise(pages: list[PageText]) -> dict:
    done = [p for p in pages if p.compared]
    by = Counter(d.decided_by for p in done for d in p.disputes)
    return {
        "pages_compared": len(done),
        "pages_skipped": {p.page: p.reason for p in pages if not p.compared},
        "words": sum(p.words[0] for p in done),
        "disputes": sum(by.values()),
        "by": dict(by),
        "layer_proven_pages": sum(1 for p in done if p.layer.get("proven")),
        "open": [d.to_dict() for p in done for d in p.disputes if d.decided_by == "open"][:200],
    }
