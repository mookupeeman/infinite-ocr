"""Word banks for synthetic tables: curated seeds plus words harvested from cached OCR output.

Curated words are ordinary place, department and item names. Harvested words come
from the Sarvam output already on disk (`out/docs/*/blocks`), so the vocabulary has
the shapes real documents have. Both only fill tables with content; the truth of a
benchmark table is always what the generator wrote, never what an OCR engine read.
"""

from __future__ import annotations

import json
import random
import re
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path

from src.config import OUT_DIR

LANGS = ("hi", "mr", "ta", "te", "bn", "en")

# (script-name prefix in unicodedata, languages that use it)
SCRIPT_OF_LANG = {"hi": "DEVANAGARI", "mr": "DEVANAGARI", "ta": "TAMIL", "te": "TELUGU", "bn": "BENGALI", "en": "LATIN"}

SEEDS: dict[str, list[str]] = {
    "hi": "शिक्षा स्वास्थ्य कृषि परिवहन बिजली पानी सड़क अस्पताल विद्यालय जिला गाँव शहर बजट योजना विकास राजस्थान बिहार महाराष्ट्र "
          "उत्तर प्रदेश मध्य प्रदेश गुजरात पंजाब हरियाणा झारखंड ओडिशा केरल कर्नाटक असम सिंचाई पुल बाँध रेलवे बंदरगाह".split(),
    "mr": "शिक्षण आरोग्य शेती वाहतूक वीज पाणी रस्ता रुग्णालय शाळा जिल्हा गाव शहर अर्थसंकल्प योजना विकास महाराष्ट्र पुणे नाशिक "
          "नागपूर कोल्हापूर सातारा सांगली लातूर अकोला जळगाव धुळे पूल धरण बंदर".split(),
    "ta": "தமிழ்நாடு சென்னை கல்வி சுகாதாரம் விவசாயம் போக்குவரத்து மின்சாரம் நீர் சாலை மருத்துவமனை பள்ளி மாவட்டம் கிராமம் நகரம் "
          "திட்டம் வளர்ச்சி மதுரை கோயம்புத்தூர் திருச்சிராப்பள்ளி சேலம் திருநெல்வேலி பாலம் அணை துறைமுகம்".split(),
    "te": "తెలంగాణ హైదరాబాద్ విద్య ఆరోగ్యం వ్యవసాయం రవాణా విద్యుత్ నీరు రోడ్డు ఆసుపత్రి పాఠశాల జిల్లా గ్రామం నగరం పథకం అభివృద్ధి "
          "విజయవాడ విశాఖపట్నం గుంటూరు తిరుపతి వరంగల్ వంతెన ఆనకట్ట ఓడరేవు".split(),
    "bn": "পশ্চিমবঙ্গ কলকাতা শিক্ষা স্বাস্থ্য কৃষি পরিবহন বিদ্যুৎ জল সড়ক হাসপাতাল বিদ্যালয় জেলা গ্রাম শহর বাজেট প্রকল্প উন্নয়ন "
          "হাওড়া দার্জিলিং মালদা বর্ধমান সেতু বাঁধ বন্দর".split(),
    "en": "Education Health Agriculture Transport Electricity Water Roads Hospital School District Village City Budget Scheme "
          "Development Rajasthan Bihar Maharashtra Gujarat Punjab Haryana Odisha Kerala Karnataka Assam Irrigation Bridge Dam "
          "Railways Port Housing Sanitation Fisheries Forests Tourism Police Library Museum".split(),
}
TOTAL_WORD = {"hi": "कुल", "mr": "एकूण", "ta": "மொத்தம்", "te": "మొత్తం", "bn": "মোট", "en": "Total"}
HEADERS = {  # column titles used by the generator (serial, name, category, description, amount, date, status, remark, total)
    "hi": ["क्र.", "नाम", "श्रेणी", "विवरण", "राशि", "दिनांक", "स्थिति", "टिप्पणी", "कुल"],
    "mr": ["अ.क्र.", "नाव", "प्रकार", "तपशील", "रक्कम", "दिनांक", "स्थिती", "शेरा", "एकूण"],
    "ta": ["வ.எண்", "பெயர்", "வகை", "விவரம்", "தொகை", "நாள்", "நிலை", "குறிப்பு", "மொத்தம்"],
    "te": ["క్ర.సం", "పేరు", "వర్గం", "వివరణ", "సొమ్ము", "తేదీ", "స్థితి", "గమనిక", "మొత్తం"],
    "bn": ["ক্রম", "নাম", "বিভাগ", "বিবরণ", "টাকা", "তারিখ", "অবস্থা", "মন্তব্য", "মোট"],
    "en": ["No.", "Name", "Category", "Description", "Amount", "Date", "Status", "Remarks", "Total"],
}
CODE = {"hi": "कोड", "mr": "संकेतांक", "ta": "குறியீடு", "te": "కోడ్", "bn": "কোড", "en": "Code"}
STATUS = {
    "hi": ["पूर्ण", "प्रगति में", "स्वीकृत", "लंबित"], "mr": ["पूर्ण", "प्रगतीपथावर", "मंजूर", "प्रलंबित"],
    "ta": ["முடிந்தது", "நடைபெறுகிறது", "அனுமதி", "நிலுவை"], "te": ["పూర్తి", "జరుగుతోంది", "ఆమోదం", "పెండింగ్"],
    "bn": ["সম্পন্ন", "চলমান", "অনুমোদিত", "মুলতুবি"], "en": ["Complete", "In progress", "Approved", "Pending"],
}

_ZERO = {"hi": 0x0966, "mr": 0x0966, "bn": 0x09E6, "ta": 0x0BE6, "te": 0x0C66}  # native digit zero points


def native_digits(text: str, lang: str) -> str:
    zero = _ZERO.get(lang)
    if zero is None:
        return text
    return "".join(chr(zero + int(ch)) if ch.isdigit() and ord(ch) < 128 else ch for ch in text)


def _script_ok(word: str, lang: str) -> bool:
    """Every character is a letter or a combining mark of the language's script."""
    want = SCRIPT_OF_LANG[lang]
    letters = 0
    for ch in word:
        if unicodedata.category(ch).startswith("M"):
            continue
        if not ch.isalpha() or want not in unicodedata.name(ch, ""):
            return False
        letters += 1
    return letters > 0


BENCH_RUN = re.compile(r"^(v2-)?(hi|mr|ta|te|bn|en)-(text|numeric|mixed|dates|merged|sparse|wide)-")
SNAPSHOT = Path(__file__).resolve().parent / "vocab_snapshot.json"


def harvest(dirs: list[Path]) -> dict[str, list[str]]:
    """Words seen at least twice in the Sarvam output of these run folders, by language (hi and mr share Devanagari)."""
    counts: dict[str, Counter] = {k: Counter() for k in LANGS}
    for path in sorted(p for d in dirs for p in d.glob("blocks/*.sarvam.json")):
        try:
            blocks = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for b in blocks:
            if b.get("kind") not in ("text", "heading"):
                continue
            for word in str(b.get("text", "")).replace("\n", " ").split():
                word = word.strip(".,;:()[]\"'“”‘’—-–|*")
                if not 2 <= len(word) <= 14:
                    continue
                for lang in ("hi", "ta", "te", "bn", "en"):
                    if _script_ok(word, lang):
                        counts[lang][word] += 1
    out = {lang: [w for w, n in c.most_common(400) if n >= 2] for lang, c in counts.items()}
    out["mr"] = out["hi"]
    return out


@lru_cache(maxsize=4)
def harvested(version: str = "v1") -> dict[str, list[str]]:
    """The vocabulary a benchmark version was rendered with, frozen in vocab_snapshot.json.

    Harvesting live would drift: every benchmark run adds folders (with captions like
    "Table 3") to out/docs, which changes the word ranking and so the generated tables.
    v1 was rendered from the real documents only; v2 after the first real-engine run, so
    its vocabulary also saw those 22 benchmark folders.
    """
    if SNAPSHOT.is_file():
        snap = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
        if version in snap:
            return snap[version]
    real = [d for d in sorted((Path(OUT_DIR) / "docs").iterdir()) if d.is_dir() and not BENCH_RUN.match(d.name)]
    return harvest(real)


def words(lang: str, version: str = "v1") -> list[str]:
    return SEEDS[lang] + harvested(version).get(lang, [])


def phrase(rng: random.Random, lang: str, lo: int = 2, hi: int = 4, version: str = "v1") -> str:
    pool = words(lang, version)
    return " ".join(rng.choice(pool) for _ in range(rng.randint(lo, hi)))
