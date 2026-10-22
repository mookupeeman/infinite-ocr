"""Repairing a PDF text layer whose font maps some glyphs to odd code points. No API calls.

Many Indian government PDFs are born digital but their fonts map conjuncts and vowel signs
to unrelated code points, the same way every time. Measured on the Union Receipt Budget:
`Ǒकमी` is `किमी` (Ǒ is the vowel sign ि, stored before its consonant), `Ēामीण` is `ग्रामीण`,
`अƠू.` is `अक्तू.`, `खजुरȣ` is `खजुरी`. Read as is, the layer contains neither engine's
reading of most text cells, so it cannot help settle them.

The mapping is learned from the cells both OCR engines agree on: each such cell is found
in the layer, the two are aligned character by character, and every odd glyph records
what it stood for. A glyph becomes a rule only if it was seen at least 3 times and meant
the same thing at least 90% of the time. Three kinds of rule:

    sub      the glyph stands for a fixed string (Ơ -> क्त)
    prebase  the glyph is a vowel sign stored before its consonant cluster (Ǒ + क -> कि)
    postbase the glyph stands for a sign stored after the cluster it logically precedes:
             the reph (ष + [ -> र्ष, as in वाǒष[की = वार्षिकी)

Learning runs in rounds: each round applies what is known, so a word with two odd glyphs
(एÈसĤेसवे = एक्सप्रेसवे) yields the second one once the first is understood.

The repaired layer is only evidence after it passes the same proof as any layer
(table_qa: it must contain at least 90% of the cells both engines agree on).
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher

from src.table_qa import canon, odd_glyph as odd

SUSPECT_ASCII = set("[]{}|~^`")  # used by some Indic fonts for the reph and similar marks; must still pass the 90% bar


def candidate(ch: str) -> bool:
    return odd(ch) or ch in SUSPECT_ASCII


def _cluster_pattern() -> str:
    """A consonant cluster in any Indic script: letter (+ nukta) (+ virama + letter (+ nukta))*."""
    letters, nukta, virama = [], [], []
    for o in range(0x0900, 0x0E00):
        ch = chr(o)
        name = unicodedata.name(ch, "")
        if "VIRAMA" in name:
            virama.append(ch)
        elif "NUKTA" in name:
            nukta.append(ch)
        elif unicodedata.category(ch) == "Lo":
            letters.append(ch)
    L, N, V = (re.escape("".join(x)) for x in (letters, nukta, virama))
    return rf"((?:[{L}][{N}]?[{V}])*[{L}][{N}]?)"


CLUSTER = _cluster_pattern()
SIGNS = "([" + re.escape("".join(chr(o) for o in range(0x0900, 0x0E00)
                                if unicodedata.category(chr(o)) in ("Mn", "Mc") and "VIRAMA" not in unicodedata.name(chr(o), ""))) + "]*)"


def _best_alignment(blob: str, key: str, anchor: int) -> tuple[list, str]:
    """Align `key` with the layer around `anchor`. A glyph can stand for a longer string, so the layer
    span may start a little before the anchor; try a few starts and keep the one with the fewest
    differences (a wider start can otherwise grab a matching letter from the previous word)."""
    best = None
    for margin in range(0, 4):
        start = max(0, anchor - margin)
        window = blob[start: start + len(key) + 6]
        ops = SequenceMatcher(None, window, key, autojunk=False).get_opcodes()
        inner = ops[:]
        while inner and inner[0][0] == "delete":  # leading and trailing layer text is not part of the cell
            inner.pop(0)
        while inner and inner[-1][0] == "delete":
            inner.pop()
        cost = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in inner if tag != "equal")
        if best is None or cost < best[0]:
            best = (cost, ops, window)
    return best[1], best[2]


def learn(samples: list[tuple[str, list[str]]], min_count: int = 3, min_share: float = 0.9, rounds: int = 4) -> dict[str, tuple[str, str]]:
    """samples: [(layer text of a table, values both engines agreed on)] -> {glyph: (kind, meaning)}."""
    rules: dict[str, tuple[str, str]] = {}
    for _ in range(rounds):
        found = _learn_round([(apply(layer, rules), values) for layer, values in samples], rules, min_count, min_share)
        if not found:
            break
        rules.update(found)
    return rules


def _learn_round(samples: list[tuple[str, list[str]]], known: dict, min_count: int, min_share: float) -> dict[str, tuple[str, str]]:
    outcomes: dict[str, Counter] = defaultdict(Counter)
    for layer, values in samples:
        blob = "".join(canon(layer).split())
        if not any(candidate(ch) for ch in blob):
            continue
        for value in values:
            key = "".join(canon(value).split())
            if len(key) < 4:
                continue
            m = SequenceMatcher(None, blob, key, autojunk=False).find_longest_match(0, len(blob), 0, len(key))
            if m.size < 3:
                continue
            ops, window = _best_alignment(blob, key, m.a - m.b)
            for k, (tag, i1, i2, j1, j2) in enumerate(ops):
                seg = window[i1:i2]
                prebase = tag == "delete" and k + 2 < len(ops) and ops[k + 1][0] == "equal" and ops[k + 2][0] == "insert"
                if (tag == "delete" and not (prebase and len(seg) == 1) and k >= 2 and ops[k - 1][0] == "equal"
                        and ops[k - 2][0] == "insert" and seg and candidate(seg[0]) and seg[0] not in known):
                    # a sign stored after its cluster; at the end of a word it merges with the following text
                    ins = ops[k - 2]
                    outcomes[seg[0]][("postbase", key[ins[3]:ins[4]])] += 1
                    continue
                if len(seg) != 1 or not candidate(seg) or seg in known:
                    continue
                if tag == "replace":
                    outcomes[seg][("sub", key[j1:j2])] += 1
                elif tag == "delete" and k + 2 < len(ops) and ops[k + 1][0] == "equal" and ops[k + 2][0] == "insert":
                    ins = ops[k + 2]
                    outcomes[seg][("prebase", key[ins[3]:ins[4]])] += 1
                elif tag == "delete":
                    outcomes[seg][("other", "")] += 1  # a literal use (e.g. a real bracket) counts against a rule
    rules = {}
    for glyph, counts in outcomes.items():
        n = sum(counts.values())
        (kind, meaning), k = counts.most_common(1)[0]
        if n >= min_count and k / n >= min_share and meaning and kind != "other":
            rules[glyph] = (kind, meaning)
    return rules


def apply(text: str, rules: dict[str, tuple[str, str]]) -> str:
    """The layer with every learned glyph replaced (fixed strings first, then pre-base vowel signs)."""
    for glyph, (kind, meaning) in rules.items():
        if kind == "sub":
            text = text.replace(glyph, meaning)
    for glyph, (kind, meaning) in rules.items():
        if kind == "prebase":
            text = re.sub(re.escape(glyph) + CLUSTER, lambda m, v=meaning: m.group(1) + v, text)
    for glyph, (kind, meaning) in rules.items():
        if kind == "postbase":  # the sign is stored after the cluster and its vowel signs; it belongs before
            text = re.sub(CLUSTER + SIGNS + re.escape(glyph), lambda m, v=meaning: v + m.group(1) + m.group(2), text)
    return text
