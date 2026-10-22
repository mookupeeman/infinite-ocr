"""Build the benchmark set, run a checker over it, save the results.

    python -m bench.run_bench --out bench/results/baseline.json            # born-digital and scanned
    python -m bench.run_bench --shared 0.0 0.15 0.4 --limit 20             # sweep the shared-error share

Rendering is cached under bench/work/; simulated engines are re-drawn per run from the seed.
No paid API is called.
"""

from __future__ import annotations

import argparse
import json
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from bench import engines, gen, harness, render, vocab

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
LANGS = vocab.LANGS


def stable(*parts: object) -> int:
    """Reproducible seed (str hash() changes between processes)."""
    return zlib.crc32("|".join(map(str, parts)).encode()) % 100_000


def benchmark_set_v2() -> list[gen.Case]:
    """Tough profile: every language x kind (incl. wide) on one page, every multi-page style, rare numerals."""
    cases = []
    for li, lang in enumerate(LANGS):
        for ki, kind in enumerate(gen.KINDS_V2):
            cases.append(gen.make_case(5000 + li * 10 + ki, lang, kind, "single", profile="tough"))
    multi = [s for s in gen.STYLES if s != "single"]
    for si, style in enumerate(multi):
        for li, lang in enumerate(LANGS):
            kinds = ("text", "dates") if style == "split_cut" else gen.KINDS_V2
            cases.append(gen.make_case(6000 + si * 10 + li, lang, kinds[(si + li) % len(kinds)], style, profile="tough"))
    for i, (lang, kind) in enumerate([("ta", "text"), ("ta", "mixed"), ("te", "numeric"), ("te", "sparse")]):
        cases.append(gen.make_case(7000 + i, lang, kind, "single", profile="tough", digits="native"))
    return cases


def benchmark_set() -> list[gen.Case]:
    """Single-page tables (every language x kind) and multi-page styles (every language)."""
    cases = []
    for li, lang in enumerate(LANGS):
        for ki, kind in enumerate(gen.KINDS):
            cases.append(gen.make_case(1000 + li * 10 + ki, lang, kind, "single"))
    multi = [s for s in gen.STYLES if s != "single"]
    for si, style in enumerate(multi):
        for li, lang in enumerate(LANGS):
            kinds = ("text", "dates") if style == "split_cut" else gen.KINDS
            kind = kinds[(si + li) % len(kinds)]
            cases.append(gen.make_case(2000 + si * 10 + li, lang, kind, style))
    return cases


def _work(args: tuple) -> dict:
    case, rendered, scanned, profile, seed, checker = args
    try:
        res = harness.run_case(case, rendered, scanned=scanned, profile=profile, seed=seed, checker=checker)
    except Exception as exc:  # noqa: BLE001 - one bad case must not stop the sweep
        return {"case": case.meta(), "error": f"{type(exc).__name__}: {exc}", "scanned": scanned, "shared_frac": profile.shared_frac}
    res["scanned"], res["shared_frac"], res["cell_error"], res["checker"] = scanned, profile.shared_frac, profile.cell_error, checker
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m bench.run_bench")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "baseline.json")
    ap.add_argument("--limit", type=int, help="only the first N cases of the set")
    ap.add_argument("--shared", type=float, nargs="+", default=[0.15], help="shares of errors planted in both readers")
    ap.add_argument("--modes", nargs="+", default=["born-digital", "scanned"], choices=["born-digital", "scanned"])
    ap.add_argument("--cell-error", type=float, nargs="+", default=[engines.Profile.cell_error])
    ap.add_argument("--set", choices=["v1", "v2"], default="v1")
    ap.add_argument("--checker", choices=["baseline", "standard"], default="baseline")
    ap.add_argument("--styles", nargs="+", help="only cases of these styles (e.g. split split_repeat two_tables)")
    ap.add_argument("--both-diff", type=float, help="share of errors where both readers are wrong differently (default 0 v1, 0.2 v2)")
    ap.add_argument("--systematic", type=int, help="systematic shared rules per table (default: 0 for v1, 2 for v2)")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)

    cases = (benchmark_set_v2() if a.set == "v2" else benchmark_set())
    cases = [c for c in cases if not a.styles or c.style in a.styles][: a.limit]
    systematic = a.systematic if a.systematic is not None else (2 if a.set == "v2" else 0)
    both_diff = a.both_diff if a.both_diff is not None else (0.2 if a.set == "v2" else 0.0)
    started = time.time()
    rendered, skipped = {}, []
    for case in cases:
        r = render.render_case(case, WORK, seed=stable(case.id))
        if r.ok:
            rendered[case.id] = r
        else:
            skipped.append({"case": case.id, "why": r.note})
    print(f"rendered {len(rendered)}/{len(cases)} cases in {time.time() - started:.0f}s; skipped {skipped or 'none'}")

    jobs = []
    for case in cases:
        if case.id not in rendered:
            continue
        for cell_error in a.cell_error:
            for shared in a.shared:
                profile = engines.Profile(cell_error=cell_error, shared_frac=shared, systematic_rules=systematic, both_diff_frac=both_diff)
                for mode in a.modes:
                    jobs.append((case, rendered[case.id], mode == "scanned", profile, stable(case.id, shared, mode, *([cell_error] if cell_error != engines.Profile.cell_error else [])), a.checker))
    print(f"running {len(jobs)} simulated runs on {a.workers} workers")
    results = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i, res in enumerate(pool.map(_work, jobs), start=1):
            results.append(res)
            if i % 20 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    errors = [r for r in results if "error" in r]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"skipped": skipped, "errors": errors, "results": [r for r in results if "error" not in r]},
                                ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {a.out}; {len(errors)} errors; wall {time.time() - started:.0f}s")
    for e in errors[:5]:
        print("  error:", e["case"]["id"], e["error"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
