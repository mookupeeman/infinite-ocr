"""How often the image geometry reads a table's shape exactly, on benchmark pages with known truth. Free.

    python -m bench.geometry_eval            # v1 and v2, born-digital and scanned copies

For ruled tables the line count must equal the true rows x columns; for unruled tables the
geometry must not claim a grid (a false grid would decide structure wrongly).
"""

from __future__ import annotations

import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pymupdf

from bench.render import render_case
from bench.run_bench import benchmark_set, benchmark_set_v2
from src.table_cutouts import DPI
from src.table_geometry import table_geometry

ROOT = Path(__file__).resolve().parent


def evaluate(cases, work: Path) -> tuple[Counter, list[dict]]:
    tally, misses = Counter(), []
    for case in cases:
        rendered = render_case(case, work, scanned=True)
        if not rendered.ok:
            continue
        for kind, path in (("digital", rendered.pdf), ("scan", rendered.scan_pdf)):
            doc = pymupdf.open(path)
            for i, spec in enumerate(case.pages, start=1):
                if spec.kind != "table" or not rendered.bboxes.get(i):
                    continue
                g = table_geometry(doc, i, 0, tuple(rendered.bboxes[i]), dpi=DPI)
                truth = (spec.table.n_rows, spec.table.n_cols)
                if case.ruled:
                    ok = g.confident and (g.rows, g.cols) == truth
                    tally[f"ruled {kind} {'exact' if ok else 'wrong'}"] += 1
                    if not ok:
                        misses.append({"case": case.id, "page": i, "kind": kind, "truth": truth, "got": [g.source, g.rows, g.cols]})
                else:
                    tally[f"unruled {kind} {'false grid' if g.confident else 'no grid'}"] += 1
                    if g.confident:
                        misses.append({"case": case.id, "page": i, "kind": kind, "truth": truth, "got": [g.source, g.rows, g.cols]})
    return tally, misses


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        out = {}
        for name, cases in (("v1", benchmark_set()), ("v2", benchmark_set_v2())):
            tally, misses = evaluate(cases, Path(tmp) / name)
            out[name] = {"tally": dict(sorted(tally.items())), "misses": misses}
            print(name, json.dumps(out[name]["tally"]))
            for m in misses:
                print("  miss", json.dumps(m))
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
