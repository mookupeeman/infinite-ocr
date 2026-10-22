"""Summarise a saved benchmark run.

    python -m bench.report bench/results/baseline.json [other.json]

The headline number is the share of tables the system accepted that were still wrong
("verified but wrong"). Second: for cells where the two readers disagreed, how often the
final table has the right value, and for cells where both readers were wrong in the same
way (a shared error) how often it was fixed. Strict accuracy needs the right text in the
right row and column; "content found" only needs the text in the right row.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def pct(num: float, den: float) -> str:
    return f"{100 * num / den:5.1f}%" if den else "   –  "


def table_rows(results: list[dict]):
    for r in results:
        for t in r["tables"]:
            yield r, t


def group(results: list[dict], key) -> dict:
    out: dict = defaultdict(list)
    for r, t in table_rows(results):
        out[key(r, t)].append((r, t))
    return dict(sorted(out.items(), key=lambda kv: str(kv[0])))


def accuracy_block(title: str, groups: dict) -> list[str]:
    lines = [f"\n{title}", f"{'group':<26}{'tables':>7}{'strict':>8}{'content':>9}{'header':>8}{'text':>8}{'number':>8}"
             f"{'exact':>8}{'TEDS':>7}{'TEDS-S':>8}{'accepted':>9}{'acc&wrong':>10}"]
    for name, items in groups.items():
        ts = [t for _, t in items]
        cells = sum(t["cells"] for t in ts)
        k = lambda kind: (sum(t["by_kind"][kind][0] for t in ts if "by_kind" in t), sum(t["by_kind"][kind][1] for t in ts if "by_kind" in t))  # noqa: E731
        te = [t["teds"] for t in ts if t.get("teds") is not None]
        ts_ = [t["teds_s"] for t in ts if t.get("teds_s") is not None]
        accepted = [t for t in ts if t["accepted"]]
        lines.append(
            f"{str(name):<26}{len(ts):>7}{pct(sum(t['correct'] for t in ts), cells):>8}"
            f"{pct(sum(t.get('found', 0) for t in ts), sum(t.get('filled', 0) for t in ts)):>9}"
            f"{pct(*k('header')):>8}{pct(*k('text')):>8}{pct(*k('number')):>8}"
            f"{pct(sum(t['exact'] for t in ts), len(ts)):>8}{(sum(te) / len(te) if te else 0):>7.3f}{(sum(ts_) / len(ts_) if ts_ else 0):>8.3f}"
            f"{pct(len(accepted), len(ts)):>9}{pct(sum(not t['exact'] for t in accepted), len(accepted)):>10}"
        )
    return lines


def status_block(title: str, groups: dict) -> list[str]:
    """Standard checker: what each status claims, and how often the claim was wrong."""
    lines = [f"\n{title}", f"{'group':<26}{'tables':>7}{'verified':>10}{'(wrong)':>8}{'repaired':>10}{'(wrong)':>8}"
             f"{'review':>8}{'(wrong)':>8}{'cell q/t':>9}{'audit q/t':>10}{'struct q/t':>11}"]
    for name, items in groups.items():
        ts = [t for _, t in items if t.get("status")]
        if not ts:
            continue
        n = len(ts)
        row = f"{str(name):<26}{n:>7}"
        for st, width in (("verified", 10), ("repaired", 10), ("needs_review", 8)):
            part = [t for t in ts if t["status"] == st]
            row += f"{len(part):>{width}}{sum(not t['exact'] for t in part):>8}"
        for k, width in (("cell", 9), ("audit", 10), ("structure", 11)):
            row += f"{sum((t.get('questions') or {}).get(k, 0) for t in ts) / n:>{width}.1f}"
        lines.append(row)
    return lines


def cell_states(results: list[dict]) -> list[str]:
    states: dict = defaultdict(lambda: [0, 0])
    for r in results:
        for c in r["cells"]:
            if c["a"] and c["b"]:
                key = "both readers right"
            elif c["a"] != c["b"]:
                key = "readers disagree (one right)"
            elif c["same"]:
                key = "both wrong, same value (shared)"
            else:
                key = "both wrong, different values"
            states[key][0] += c["final"]
            states[key][1] += 1
    total = sum(v[1] for v in states.values())
    lines = ["\nCell by cell (single-page tables): who was right, and is the final cell right?",
             f"{'readers':<34}{'cells':>8}{'share':>8}{'final right':>13}"]
    for key in ("both readers right", "readers disagree (one right)", "both wrong, same value (shared)", "both wrong, different values"):
        n_right, n = states.get(key, [0, 0])
        lines.append(f"{key:<34}{n:>8}{pct(n, total):>8}{pct(n_right, n):>13}")
    return lines


def join_block(results: list[dict]) -> list[str]:
    by_style: dict = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    for r in results:
        for k in ("tp", "fp", "fn", "tn"):
            by_style[r["case"]["style"]][k] += r["joins"][k]
    lines = ["\nJoins across page breaks", f"{'style':<16}{'should join':>12}{'joined':>8}{'should not':>11}{'wrongly joined':>15}"]
    for style, v in sorted(by_style.items()):
        if v["tp"] + v["fn"] + v["fp"] + v["tn"]:
            lines.append(f"{style:<16}{v['tp'] + v['fn']:>12}{pct(v['tp'], v['tp'] + v['fn']):>8}{v['fp'] + v['tn']:>11}{v['fp']:>15}")
    return lines


def render(data: dict) -> str:
    results = data["results"]
    lines = [f"{len(results)} runs, {len(data.get('errors', []))} errors, {len(data.get('skipped', []))} cases skipped"]
    combos = sorted({(r.get("cell_error", 0.04), r["shared_frac"]) for r in results})
    for cell_error, share in combos:
        part = [r for r in results if r.get("cell_error", 0.04) == cell_error and r["shared_frac"] == share]
        lines.append(f"\n===== cell error {cell_error}, shared-error share {share} ({len(part)} runs) =====")
        lines += accuracy_block("By input", group(part, lambda r, t: "scanned" if r["scanned"] else "born-digital"))
        lines += accuracy_block("By table kind", group(part, lambda r, t: r["case"]["kind"]))
        lines += accuracy_block("By language", group(part, lambda r, t: r["case"]["lang"]))
        lines += accuracy_block("By style", group(part, lambda r, t: r["case"]["style"]))
        if any(t.get("status") for _, t in table_rows(part)):
            lines += status_block("Status (standard checker)", group(part, lambda r, t: "scanned" if r["scanned"] else "born-digital"))
            lines += status_block("Status by table kind", group(part, lambda r, t: r["case"]["kind"]))
        lines += cell_states(part)
        lines += join_block(part)
    return "\n".join(lines)


if __name__ == "__main__":
    for path in sys.argv[1:]:
        print(f"# {path}")
        print(render(json.loads(Path(path).read_text(encoding="utf-8"))))
