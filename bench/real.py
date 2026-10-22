"""Real-engine run on benchmark pages (PAID: Sarvam Rs 0.50 + Bodhan Rs 0.20 per page).

    python -m bench.real --dry-run            # plan and cost estimate, no API call
    python -m bench.real --cap-inr 17         # run; resumes a half-finished run, cached pages cost nothing

Each chosen benchmark case becomes its own run under out/docs/ (so stage 2 can later be
re-assembled per case at no cost). Single-page tables are read twice, born-digital and as a
scan-like copy, so the effect of the scan can be separated from the effect of the table.
Only stage 1 runs here: the raw engine readings are what the calibration needs.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from bench import gen, render
from bench.harness import LANG_CODE
from bench.run_bench import WORK, benchmark_set, benchmark_set_v2, stable
from src.lanes import Settings
from src.manifest import Manifest
from src.run import Run
from src.config import BODHAN_INR_PER_PAGE, SARVAM_INR_PER_PAGE

REAL = Path(__file__).resolve().parent / "real"
PER_PAGE_INR = SARVAM_INR_PER_PAGE + BODHAN_INR_PER_PAGE

# single-page cases read in both modes: (language, kind) -> case id looked up in the benchmark set
SINGLES = [("hi", "mixed"), ("mr", "text"), ("ta", "text"), ("te", "mixed"), ("bn", "dates"),
           ("en", "numeric"), ("hi", "merged"), ("mr", "numeric"), ("ta", "sparse"), ("bn", "text")]
# multi-page cases read born-digital only (two pages each): chosen by (style, language)
MULTI = [("split_repeat", "hi"), ("split_cut", "ta")]
# benchmark v2 (tough pages): budget-style wide tables in four scripts, dense text tables, Latin digits for ta/te
SINGLES_V2 = [("hi", "wide"), ("mr", "wide"), ("en", "wide"), ("bn", "wide"), ("hi", "text"), ("mr", "mixed"),
              ("ta", "text"), ("te", "numeric"), ("bn", "merged")]
MULTI_V2 = [("split_repeat", "hi"), ("split", "mr")]


def select(version: str = "v1") -> list[tuple[gen.Case, list[str]]]:
    """[(case, modes)] where modes is a subset of ['born-digital', 'scanned']."""
    cases = benchmark_set_v2() if version == "v2" else benchmark_set()
    singles, multis = (SINGLES_V2, MULTI_V2) if version == "v2" else (SINGLES, MULTI)
    by_key = {(c.lang, c.kind, c.style): c for c in cases if "-native-" not in c.id}
    chosen = [(by_key[(lang, kind, "single")], ["born-digital", "scanned"]) for lang, kind in singles]
    for style, lang in multis:
        pool = [c for c in cases if c.style == style and c.lang == lang]
        chosen.append((next((c for c in pool if len(c.pages) == 2), pool[0]), ["born-digital"]))  # prefer two pages
    return chosen


def plan(version: str = "v1") -> list[dict]:
    items = []
    for case, modes in select(version):
        for mode in modes:
            items.append({"case": case, "mode": mode, "pages": len(case.pages)})
    return items


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m bench.real")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cap-inr", type=float, default=17.0, help="stop before a run that would pass this total")
    ap.add_argument("--set", choices=["v1", "v2"], default="v1")
    a = ap.parse_args(argv)

    items = plan(a.set)
    runs_file = REAL / ("runs.json" if a.set == "v1" else f"runs_{a.set}.json")
    pages = sum(i["pages"] for i in items)
    print(f"{len(items)} runs, {pages} pages, estimate Rs {pages * PER_PAGE_INR:.2f} "
          f"(Sarvam {pages} x {SARVAM_INR_PER_PAGE} + Bodhan {pages} x {BODHAN_INR_PER_PAGE}); cap Rs {a.cap_inr}")
    for i in items:
        print(f"  {i['case'].id:<32} {i['mode']:<13} {i['pages']} page(s)")
    if a.dry_run:
        return 0
    if pages * PER_PAGE_INR > a.cap_inr + 1e-9:
        print("estimate is over the cap; not running", file=sys.stderr)
        return 2

    REAL.mkdir(parents=True, exist_ok=True)
    started, spent, record = time.time(), 0.0, []
    for n, item in enumerate(items, start=1):
        case, mode = item["case"], item["mode"]
        rendered = render.render_case(case, WORK, seed=stable(case.id))
        if not rendered.ok:  # an overflowing table would print extra (paid) pages
            print(f"skipping {case.id}: {rendered.note}")
            continue
        src = rendered.scan_pdf if mode == "scanned" else rendered.pdf
        pdf = REAL / (f"{case.id}.scan.pdf" if mode == "scanned" else f"{case.id}.pdf")
        shutil.copyfile(src, pdf)
        if spent + item["pages"] * PER_PAGE_INR > a.cap_inr + 1e-9:
            print(f"cap reached before {pdf.name}; stopping")
            break
        run = Run(pdf, Settings(language=LANG_CODE[case.lang], policy="both", max_cost_inr=round(a.cap_inr - spent, 2)), resume=True, progress_every=999, verbose=False)
        run.prepare()
        summary = run.execute()
        out_dir = Path(summary["out_dir"])
        m = Manifest(out_dir / "manifest.sqlite")
        run_spent = m.spent_inr()
        m.close()
        spent += run_spent
        record.append({"case": case.id, "mode": mode, "out_dir": str(out_dir), "pages": item["pages"], "spent_inr": round(run_spent, 4)})
        print(f"[{n}/{len(items)}] {pdf.name}: Rs {run_spent:.2f} (running total Rs {spent:.2f}, {time.time() - started:.0f}s)", flush=True)
        runs_file.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"done: {len(record)} runs, Rs {spent:.2f}, {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
