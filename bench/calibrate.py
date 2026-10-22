"""Measure the real engines on the benchmark pages and compare with what the simulation assumed.

    python -m bench.calibrate                       # reads bench/real/runs.json (written by bench.real)

Free: only reads engine output already on disk and re-assembles it (no agent, no re-read).
Writes bench/results/real_calibration.json and real_confusions.json (used by the simulator).
"""

from __future__ import annotations

import difflib
import json
from collections import Counter, defaultdict
from pathlib import Path

from bench import engines, metrics
from bench.harness import score_case
from bench.run_bench import benchmark_set, benchmark_set_v2
from src.assemble import Assembler, load_blocks
from src.tables import Table, parse_number, parse_tables

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
ENGINES = ("sarvam", "bodhan")


def engine_table(out_dir: Path, page: int, engine: str) -> tuple[Table | None, dict]:
    """The engine's table(s) on a page, stacked top to bottom if it split one table into pieces."""
    path = out_dir / "blocks" / f"page_{page:05d}.{engine}.json"
    if not path.is_file():
        return None, {"read": False}
    blocks = load_blocks(path)
    parts = []
    for b in sorted((b for b in blocks if b.kind == "table"), key=lambda b: b.bbox[1]):
        tables = parse_tables(b.text)
        if tables:
            parts.append(max(tables, key=lambda t: t.n_rows * t.n_cols))
    info = {"read": True, "table_blocks": len(parts), "labels": sorted({b.raw_label for b in blocks})}
    if not parts:
        return None, info
    from bench.harness import stack

    t = stack(parts)
    info["uses_rowspan"] = any(c.rowspan > 1 for c in t.cells)
    info["uses_colspan"] = any(c.colspan > 1 for c in t.cells)
    info["header_rows"] = t.header_rows
    return t, info


def char_pairs(truth: str, got: str) -> list[tuple[str, str]]:
    pairs = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, truth, got, autojunk=False).get_opcodes():
        u, v = truth[i1:i2], got[j1:j2]
        if tag != "equal" and i2 - i1 <= 3 and j2 - j1 <= 3 and (u or v) and not any(ch.isspace() for ch in u + v):
            pairs.append((u, v))
    return pairs


def page_truth(case, page: int) -> Table:
    return case.pages[page - 1].table


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m bench.calibrate")
    ap.add_argument("--set", choices=["v1", "v2"], default="v1")
    ap.add_argument("--image-check", action="store_true", help="also run standard mode WITH the image check (paid, small)")
    a = ap.parse_args(argv)
    checkers = ["legacy", "standard"] + (["standard+image"] if a.image_check else [])
    suffix = "" if a.set == "v1" else f"_{a.set}"
    runs = json.loads((ROOT / "real" / f"runs{suffix}.json").read_text())
    cases = {c.id: c for c in (benchmark_set_v2() if a.set == "v2" else benchmark_set())}
    rows = []  # one per (run, page)
    cells = []  # one per truth cell on a page
    confusions: dict[str, Counter] = {e: Counter() for e in ENGINES}
    assembled = []
    for run in runs:
        case = cases[run["case"]]
        out_dir = Path(run["out_dir"])
        scanned = run["mode"] == "scanned"
        for page, spec in enumerate(case.pages, start=1):
            if spec.kind != "table":
                continue
            truth = page_truth(case, page)
            got, infos, scores = {}, {}, {}
            for e in ENGINES:
                got[e], infos[e] = engine_table(out_dir, page, e)
                scores[e] = metrics.score_table(truth, got[e])
            row = {"case": case.id, "mode": run["mode"], "scanned": scanned, "lang": case.lang, "kind": case.kind,
                   "style": case.style, "page": page, "truth_rows": truth.n_rows, "truth_cols": truth.n_cols,
                   "truth_header_rows": truth.header_rows}
            for e in ENGINES:
                s = scores[e]
                row[e] = {**infos[e], "detected": got[e] is not None, "exact": s.get("exact", False),
                          "rows": got[e].n_rows if got[e] else 0, "cols": got[e].n_cols if got[e] else 0,
                          "rows_missing": s.get("rows_missing"), "rows_extra": s.get("rows_extra"), "cols_ok": s.get("cols_ok"),
                          "teds": s.get("teds"), "teds_s": s.get("teds_s"),
                          "header_depth_off": (got[e].header_rows != truth.header_rows) if got[e] else None}
            rows.append(row)
            if all(got[e] is not None for e in ENGINES):
                oa, ob = scores["sarvam"]["_ok"], scores["bodhan"]["_ok"]
                for cell in truth.cells:
                    key = (cell.r, cell.c)
                    kind = ("header" if cell.r < truth.header_rows else "empty" if not cell.text.strip()
                            else "number" if parse_number(cell.text) is not None else "text")
                    cells.append({"case": case.id, "scanned": scanned, "lang": case.lang, "kind": kind, "table_kind": case.kind,
                                  "a": oa[key][0], "b": ob[key][0], "same": oa[key][1] == ob[key][1],
                                  "empty_a": oa[key][1] in ("", None) and bool(cell.text.strip()),
                                  "empty_b": ob[key][1] in ("", None) and bool(cell.text.strip())})
                    for e, ok in (("sarvam", oa), ("bodhan", ob)):
                        if not ok[key][0] and ok[key][1] and cell.text.strip():
                            for pair in char_pairs(metrics.norm(cell.text), ok[key][1]):
                                confusions[e][pair] += 1
        # stage 2 on the real readings: today's rules (legacy) and the new free checker (standard)
        for checker in checkers:
            from src.agent import Agent

            agent = Agent() if checker == "standard+image" else None
            qa_mode = "legacy" if checker == "legacy" else "standard"
            result = Assembler(out_dir, agent=agent, reocr=False, verbose=False, qa_mode=qa_mode).run()
            finals = []
            for meta in result["tables"]["list"]:
                parsed = parse_tables((out_dir / "tables" / f"{meta['id']}.html").read_text(encoding="utf-8"))
                finals.append((meta, parsed[0] if parsed else None))
            scored = score_case(case, {}, finals, result)
            scored.update(scanned=scanned, mode=run["mode"], checker=checker)
            assembled.append(scored)

    summary = summarise(rows, cells, assembled, checkers)
    summary["confusions"] = {e: [[a, b, n] for (a, b), n in c.most_common(60)] for e, c in confusions.items()}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"real_calibration{suffix}.json").write_text(json.dumps({"summary": summary, "pages": rows, "assembled": assembled}, ensure_ascii=False, indent=1), encoding="utf-8")
    merged = Counter()
    for c in confusions.values():
        merged.update(c)
    (RESULTS / f"real_confusions{suffix}.json").write_text(json.dumps([[a, b, n] for (a, b), n in merged.most_common(150)], ensure_ascii=False), encoding="utf-8")
    print(render(summary))
    return 0


def pct(n: float, d: float) -> str:
    return f"{100 * n / d:.1f}%" if d else "–"


def summarise(rows: list[dict], cells: list[dict], assembled: list[dict], checkers: list[str]) -> dict:
    out: dict = {"pages": len(rows), "cells": len(cells)}
    out["detected"] = {e: sum(r[e]["detected"] for r in rows) for e in ENGINES}
    out["exact_pages"] = {e: sum(r[e]["exact"] for r in rows) for e in ENGINES}

    def block(sel) -> dict:
        part = [c for c in cells if sel(c) and c["kind"] != "empty"]
        n = len(part)
        both_right = sum(c["a"] and c["b"] for c in part)
        one = sum(c["a"] != c["b"] for c in part)
        shared = sum((not c["a"]) and (not c["b"]) and c["same"] for c in part)
        diff = sum((not c["a"]) and (not c["b"]) and not c["same"] for c in part)
        union = n - both_right
        return {"cells": n, "sarvam_wrong": sum(not c["a"] for c in part), "bodhan_wrong": sum(not c["b"] for c in part),
                "union_error_rate": round(union / n, 4) if n else None, "disagree": one, "shared_same": shared,
                "both_wrong_different": diff, "shared_share_of_errors": round(shared / union, 4) if union else None,
                "empty_when_truth_filled": sum(c["empty_a"] or c["empty_b"] for c in part)}

    out["cells_all"] = block(lambda c: True)
    out["cells_body"] = block(lambda c: c["kind"] != "header")
    out["cells_header"] = block(lambda c: c["kind"] == "header")
    for key in ("number", "text"):
        out[f"cells_{key}"] = block(lambda c, k=key: c["kind"] == k)
    out["by_mode"] = {m: block(lambda c, m=m: c["scanned"] == (m == "scanned")) for m in ("born-digital", "scanned")}
    out["by_lang"] = {lang: block(lambda c, lang=lang: c["lang"] == lang) for lang in sorted({r["lang"] for r in rows})}
    out["structure"] = {e: {"tables": len(rows), "rows_missing": sum(1 for r in rows if (r[e]["rows_missing"] or 0) > 0),
                            "rows_extra": sum(1 for r in rows if (r[e]["rows_extra"] or 0) > 0),
                            "cols_not_ok": sum(1 for r in rows if r[e]["detected"] and not r[e]["cols_ok"]),
                            "header_depth_off": sum(1 for r in rows if r[e]["header_depth_off"]),
                            "split_into_pieces": sum(1 for r in rows if (r[e].get("table_blocks") or 0) > 1),
                            "uses_rowspan": sum(1 for r in rows if r[e].get("uses_rowspan")),
                            "uses_colspan": sum(1 for r in rows if r[e].get("uses_colspan"))} for e in ENGINES}
    out["stage2_on_real"] = {}
    for checker in checkers:
        for mode in ("born-digital", "scanned"):
            part = [t for a in assembled if a["mode"] == mode and a["checker"] == checker for t in a["tables"]]
            accepted = [t for t in part if t["accepted"]]
            out["stage2_on_real"][f"{checker}/{mode}"] = {
                "tables": len(part), "exact": sum(t["exact"] for t in part), "accepted": len(accepted),
                "accepted_and_wrong": sum(not t["exact"] for t in accepted),
                "needs_review": sum(t.get("status") == "needs_review" for t in part),
                "needs_review_and_wrong": sum(t.get("status") == "needs_review" and not t["exact"] for t in part),
                "cells": sum(t["cells"] for t in part), "correct": sum(t["correct"] for t in part),
                "questions": {k: sum((t.get("questions") or {}).get(k, 0) for t in part) for k in ("cell", "audit", "structure")}}
        out[f"joins_on_real/{checker}"] = {k: sum(a["joins"][k] for a in assembled if a["checker"] == checker) for k in ("tp", "fn", "fp")}
    return out


def render(s: dict) -> str:
    L = [f"{s['pages']} table pages, {s['cells']} cells (both engines detected a table)",
         f"table detected: {s['detected']}; exactly right: {s['exact_pages']}", ""]
    L.append(f"{'cells':<22}{'n':>6}{'sarvam':>8}{'bodhan':>8}{'union err':>10}{'disagree':>9}{'shared':>7}{'differ':>7}{'shared/err':>11}")
    for name in ("cells_all", "cells_body", "cells_header", "cells_number", "cells_text"):
        b = s[name]
        L.append(f"{name:<22}{b['cells']:>6}{pct(b['sarvam_wrong'], b['cells']):>8}{pct(b['bodhan_wrong'], b['cells']):>8}"
                 f"{(b['union_error_rate'] or 0):>10.3f}{b['disagree']:>9}{b['shared_same']:>7}{b['both_wrong_different']:>7}"
                 f"{(b['shared_share_of_errors'] if b['shared_share_of_errors'] is not None else float('nan')):>11.2f}")
    for grp in ("by_mode", "by_lang"):
        for k, b in s[grp].items():
            L.append(f"  {k:<20}{b['cells']:>6}{pct(b['sarvam_wrong'], b['cells']):>8}{pct(b['bodhan_wrong'], b['cells']):>8}"
                     f"{(b['union_error_rate'] or 0):>10.3f}{b['disagree']:>9}{b['shared_same']:>7}{b['both_wrong_different']:>7}")
    L.append("\nstructure per engine (out of %d table pages): %s" % (s["pages"], json.dumps(s["structure"])))
    L.append("\nstage 2 on the real readings (exact = every cell right):")
    for key, v in s["stage2_on_real"].items():
        L.append(f"  {key:<30} tables {v['tables']:>3}  exact {v['exact']:>3}  accepted {v['accepted']:>3} (wrong {v['accepted_and_wrong']})"
                 f"  needs review {v['needs_review']:>3} (wrong {v['needs_review_and_wrong']})  cells {v['correct']}/{v['cells']}  questions {v['questions']}")
    L.append("\ntop real confusions (truth -> read):")
    for e in ENGINES:
        L.append(f"  {e}: " + ", ".join(f"{a or '∅'}→{b or '∅'}×{n}" for a, b, n in s["confusions"][e][:15]))
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
