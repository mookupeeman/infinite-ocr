"""Compare two checkers on the same benchmark results, table by table.

    python -m bench.compare bench/results/v2_baseline.json bench/results/v2_standard.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def pct(n: float, d: float) -> str:
    return f"{100 * n / d:5.1f}%" if d else "   – "


def headline(path: Path) -> dict:
    out: dict = defaultdict(lambda: defaultdict(int))
    for r in json.loads(path.read_text(encoding="utf-8"))["results"]:
        key = (r.get("cell_error", 0.04), "scanned" if r["scanned"] else "born-digital")
        for t in r["tables"]:
            o = out[key]
            o["tables"] += 1
            o["exact"] += t["exact"]
            o["accepted"] += t["accepted"]
            o["accepted_wrong"] += t["accepted"] and not t["exact"]
            o["wrong"] += not t["exact"]
            o["caught"] += (not t["accepted"]) and not t["exact"]
            o["false_alarm"] += (not t["accepted"]) and t["exact"]
            for k, v in (t.get("questions") or {}).items():
                o[f"q_{k}"] += v
    return out


def main(paths: list[str]) -> None:
    rows = {p: headline(Path(p)) for p in paths}
    keys = sorted({k for h in rows.values() for k in h})
    print(f"{'cell error / input':<24}{'checker':<12}{'tables':>7}{'exact':>8}{'accepted':>9}{'acc&wrong':>10}"
          f"{'wrong caught':>13}{'false alarm':>12}{'cell q/t':>9}{'audit q/t':>10}")
    for key in keys:
        for p in paths:
            o = rows[p].get(key)
            if not o:
                continue
            n = o["tables"]
            name = Path(p).stem.split("_")[-1]
            print(f"{key[0]:<6}{key[1]:<18}{name:<12}{n:>7}{pct(o['exact'], n):>8}{pct(o['accepted'], n):>9}"
                  f"{pct(o['accepted_wrong'], o['accepted']):>10}{pct(o['caught'], o['wrong']):>13}"
                  f"{pct(o['false_alarm'], n - o['wrong']):>12}{o['q_cell'] / n:>9.1f}{o['q_audit'] / n:>10.1f}")


if __name__ == "__main__":
    main(sys.argv[1:])
