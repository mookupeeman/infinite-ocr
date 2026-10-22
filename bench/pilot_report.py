"""Compare every model in bench/results/pilot/ on the same real questions.

    python -m bench.pilot_report

Decision rule measured everywhere: accept a model's reading only if it equals one of the
engines' readings (two independent readers agree). "harm" = a cell whose default reading
was right and that the rule changed to a wrong value. Pairs: two models from different
families must also agree with each other.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

from bench import metrics

PILOT = Path(__file__).resolve().parent / "results" / "pilot"
OPEN_WEIGHTS = {"qwen", "nvidia", "google/gemma", "deepseek", "meta", "z-ai", "xiaomi"}
FAMILY = {"google/gemini": "Google", "google/gemma": "Google", "anthropic": "Anthropic", "openai": "OpenAI", "qwen": "Qwen",
          "nvidia": "NVIDIA", "deepseek": "DeepSeek", "meta": "Meta", "z-ai": "Zhipu", "xiaomi": "Xiaomi"}
SEGMENTS = {"born-digital": lambda r: not r["scanned"], "scanned ruled": lambda r: r["scanned"] and r["boxed"],
            "scanned unruled": lambda r: r["scanned"] and not r["boxed"]}


def family(m: str) -> str:
    return next(v for k, v in FAMILY.items() if m.startswith(k))


def is_open(m: str) -> bool:
    return any(m.startswith(k) for k in OPEN_WEIGHTS)


def tamil_numeral(r: dict) -> bool:
    return any(0x0BE6 <= ord(ch) <= 0x0BEF for ch in r["truth"])


def N(s: str | None) -> str:
    return metrics.norm(s or "")


def accepted(r: dict) -> bool:
    return bool(r.get("chose_candidate")) and N(r.get("reading")) == N(r.get("final"))


def load() -> tuple[dict, dict]:
    by, cost = {}, {}
    for f in sorted(PILOT.glob("*.json")):
        if f.name == "summary.json" or "-trial" in f.name:
            continue
        data = json.loads(f.read_text())
        m = f.stem.replace("__", "/")
        by[m] = {(r["group"], r["qid"]): r for r in data["rows"]}
        cost[m] = data["summary"]["cost_inr"]
    return by, cost


def main() -> None:
    by, cost = load()
    models = sorted(by, key=lambda m: (not is_open(m), cost[m]))
    first = by[models[0]]
    disputes = [k for k, r in first.items() if r["kind"] == "cell" and not tamil_numeral(r)]
    numerals = [k for k, r in first.items() if r["kind"] == "cell" and tamil_numeral(r)]
    audits = [k for k, r in first.items() if r["kind"] == "audit"]
    print(f"{len(disputes)} disputes (+{len(numerals)} Tamil-numeral disputes), {len(audits)} audits; "
          f"default right: " + ", ".join(f"{s} {sum(first[k]['value_right'] for k in disputes if f(first[k]))}/"
                                         f"{sum(1 for k in disputes if f(first[k]))}" for s, f in SEGMENTS.items()))
    print("\nOne model, rule: accept only if its reading equals an engine reading  (right/decided, harm)")
    print(f"{'model':<36}{'open':>5}{'born-digital':>18}{'scanned ruled':>18}{'scanned unruled':>18}{'Tamil num.':>11}"
          f"{'audit caught/false':>20}{'answered':>10}{'Rs':>8}")
    for m in models:
        row = f"{m:<36}{'yes' if is_open(m) else '':>5}"
        for name, sel in SEGMENTS.items():
            ks = [k for k in disputes if sel(first[k]) and k in by[m]]
            dec = [k for k in ks if accepted(by[m][k])]
            right = sum(by[m][k]["final_right"] for k in dec)
            harm = sum(first[k]["value_right"] and not by[m][k]["final_right"] for k in dec)
            row += f"{f'{right}/{len(dec)} h{harm}':>18}"
        row += f"{sum(by[m][k]['final_right'] for k in numerals if k in by[m]):>8}/{len(numerals):<2}"
        flagged = [k for k in audits if k in by[m] and by[m][k].get("flagged")]
        caught = sum(not first[k]["value_right"] for k in flagged)
        answered = sum(by[m][k]["answered"] for k in by[m])
        row += f"{f'{caught}/{len(flagged) - caught}':>20}{answered:>6}/{len(by[m]):<4}{cost[m]:>8.1f}"
        print(row)

    print("\nBest pairs of different families that must agree (by fewest harms, then most right), scanned ruled / born-digital:")
    pairs = []
    for a, b in itertools.combinations(models, 2):
        if family(a) == family(b):
            continue
        stats = {}
        for name, sel in SEGMENTS.items():
            ks = [k for k in disputes if sel(first[k]) and k in by[a] and k in by[b]]
            dec = [k for k in ks if accepted(by[a][k]) and N(by[a][k].get("reading")) == N(by[b][k].get("reading"))]
            stats[name] = (sum(by[a][k]["final_right"] for k in dec), len(dec),
                           sum(first[k]["value_right"] and not by[a][k]["final_right"] for k in dec))
        num = sum(by[a][k]["final_right"] and N(by[a][k].get("reading")) == N(by[b][k].get("reading"))
                  for k in numerals if k in by[a] and k in by[b])
        pairs.append((a, b, stats, num, cost[a] + cost[b]))
    pairs.sort(key=lambda p: (p[2]["scanned ruled"][2] + p[2]["born-digital"][2], -p[2]["scanned ruled"][0] - p[2]["born-digital"][0]))
    print(f"{'pair':<66}{'born-digital':>16}{'scanned ruled':>16}{'unruled':>16}{'Tamil':>7}{'Rs':>7}")
    for a, b, st, num, rs in pairs[:15]:
        cells = "".join(f"{f'{st[s][0]}/{st[s][1]} h{st[s][2]}':>16}" for s in SEGMENTS)
        print(f"{a.split('/')[1][:30]:<32}+ {b.split('/')[1][:30]:<32}{cells}{num:>7}{rs:>7.1f}")


if __name__ == "__main__":
    main()
