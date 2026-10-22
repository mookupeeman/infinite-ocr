"""What the free page-text check (src/text_qa.py) finds and settles on assembled Think runs. No API calls.

    python -m bench.text_dispute_report <run dir> [<run dir> ...] [--examples 6]

Reads `text_check` from each run's assembly.json (written by `python -m src.assemble <dir> --mode think`).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--examples", type=int, default=6)
    a = p.parse_args()
    total: Counter = Counter()
    for run in a.runs:
        data = json.loads((run / "assembly.json").read_text())
        t = data.get("text_check")
        if not t:
            print(f"{run.name}: no text_check (assemble with --mode think)")
            continue
        total.update(t["by"])
        print(f"{run.name[:34]:34s} pages {t['pages_compared']:2d} compared, {len(t['pages_skipped'])} skipped | words {t['words']:5d} | "
              f"disputes {t['disputes']:3d} {t['by']} | layer proven on {t['layer_proven_pages']} pages")
        for page, why in t["pages_skipped"].items():
            print(f"    page {page} skipped: {why}")
        for d in t["open"][: a.examples]:
            print(f"    open p{d['page']}: ...{d['before'][-30:]} [{d['primary'][:40]!r} | {d['other'][:40]!r}] {d['after'][:25]}")
    print("all runs:", dict(total))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
