"""Infinite-OCR job engine: OCR a PDF of any length, resumably.

    python -m src.run file.pdf --language mr-IN
    python -m src.run file.pdf --resume              # continue after a crash or Ctrl-C
    python -m src.run file.pdf --max-pages 20 --max-cost-inr 15
    python -m src.run file.pdf --mode think --doc-type scanned

Outputs go to out/docs/<name>-<sha8>/: manifest.sqlite, events.jsonl,
<engine>/pages/*.json (raw results), blocks/*.json (one block format),
md/*.md (page previews), summary.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import signal
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

import pymupdf

from src import sarvam_digitise as sarvam_api
from src.config import (
    BODHAN_BACKEND,
    BODHAN_MAX_INFLIGHT,
    BODHAN_MLX_PAGES,
    BODHAN_VLLM_PAGES,
    BODHAN_TABLE_FORMAT,
    DOCS_OUT_DIR,
    RENDER_DPI,
    has_bodhan_key,
    has_openrouter_key,
    has_sarvam_key,
    resolve_ocr_language,
)
from src.lanes import (
    POLICIES,
    BodhanAPIBackend,
    BodhanLocalBackend,
    Budget,
    Context,
    PageBackend,
    PageLane,
    QwenBackend,
    Router,
    SarvamLane,
    Settings,
)
from src import estimate, modes
from src.manifest import EventLog, Manifest
from src.page_class import DOC_TYPES
from src.triage import triage_page


class RunError(RuntimeError):
    """A run cannot start; the message says why."""
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slug(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower()
    return (s or "document")[:40]


def _fmt_states(states: dict[str, int]) -> str:
    order = ("done", "inflight", "queued", "pending", "failed", "stolen", "skipped")
    return " ".join(f"{k}={states[k]}" for k in order if states.get(k))


class Run:
    """One document run (stage 1): triage, the engine lanes, the manifest and the summary."""
    def __init__(
        self,
        pdf_path: Path,
        settings: Settings,
        *,
        resume: bool = False,
        fresh: bool = False,
        out_root: Path = DOCS_OUT_DIR,
        sarvam: ModuleType | Any = sarvam_api,
        bodhan_backend: PageBackend | None = None,
        qwen_backend: PageBackend | None = None,
        progress_every: float = 10.0,
        verbose: bool = True,
    ) -> None:
        self.pdf_path = Path(pdf_path).resolve()
        self.settings = settings
        self.resume = resume
        self.fresh = fresh
        self.out_root = out_root
        self.sarvam = sarvam
        self.bodhan_backend = bodhan_backend
        self.qwen_backend = qwen_backend
        self.progress_every = progress_every
        self.verbose = verbose
        self.ctx: Context | None = None

    def say(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    # -- setup ------------------------------------------------------------

    def prepare(self) -> Context:
        if not self.pdf_path.is_file():
            raise RunError(f"PDF not found: {self.pdf_path}")
        if self.settings.policy not in POLICIES:
            raise RunError(f"policy must be one of {POLICIES}")
        sha = file_sha256(self.pdf_path)
        out_dir = self.out_root / f"{slug(self.pdf_path.stem)}-{sha[:8]}"
        db_path = out_dir / "manifest.sqlite"
        if db_path.exists() and not (self.resume or self.fresh):
            raise RunError(f"{out_dir} has a previous run. Use --resume to continue or --fresh to start over.")
        if self.fresh and out_dir.exists():
            for name in ("manifest.sqlite", "manifest.sqlite-wal", "manifest.sqlite-shm", "events.jsonl"):
                (out_dir / name).unlink(missing_ok=True)
            for sub in ("sarvam", "bodhan", "qwen", "blocks", "md"):
                shutil.rmtree(out_dir / sub, ignore_errors=True)

        manifest = Manifest(db_path)
        stored = manifest.get_meta("fingerprint")
        current = self.settings.fingerprint()
        if stored is not None and json.loads(stored) != current:
            old = json.loads(stored)
            diff = {k: (old.get(k), current.get(k)) for k in current if old.get(k) != current.get(k)}
            raise RunError(f"settings differ from the previous run {diff}. Use --fresh, or the same settings.")
        manifest.set_meta("fingerprint", json.dumps(current))
        manifest.set_meta("source", str(self.pdf_path))
        manifest.set_meta("sha256", sha)

        with pymupdf.open(self.pdf_path) as doc:
            if doc.needs_pass:
                raise RunError("encrypted PDF")
            total = doc.page_count
        selection = self.settings.pages or list(range(1, total + 1))
        bad = [p for p in selection if not 1 <= p <= total]
        if bad:
            raise RunError(f"pages out of range 1-{total}: {bad[:5]}")
        if self.settings.max_pages:
            selection = selection[: self.settings.max_pages]
        self.selection = selection
        self.n_pages = len(selection)
        self.total_pages = total

        events = EventLog(out_dir / "events.jsonl")
        budget = Budget(self.settings.max_cost_inr, manifest.spent_inr())
        ctx = Context(self.pdf_path, out_dir, self.settings, manifest, events, budget, sha)
        self.ctx = ctx
        self.router = Router(ctx)
        engines = self.router.engines()
        missing = []
        if "sarvam" in engines and self.sarvam is sarvam_api and not has_sarvam_key():
            missing.append("SARVAM_API_KEY")
        if (
            "bodhan" in engines
            and self.bodhan_backend is None
            and BODHAN_BACKEND != "local"
            and not has_bodhan_key()
        ):
            missing.append("BODHAN_API_KEY")
        if "qwen" in engines and self.qwen_backend is None and not has_openrouter_key():
            missing.append("OPENROUTER_API_KEY")
        if missing:
            raise RunError(f"missing keys in .env: {', '.join(missing)}")
        if "sarvam" in engines and self.sarvam is sarvam_api:
            try:
                sarvam_api.check_sdk()
            except RuntimeError as exc:
                raise RunError(str(exc)) from exc
        return ctx

    # -- run --------------------------------------------------------------

    def _triage(self) -> None:
        ctx = self.ctx
        assert ctx is not None
        try:
            done = ctx.manifest.triaged_pages()
            doc = ctx.doc()
            for page in self.selection:
                if ctx.stop.is_set() or ctx.hard_stop.is_set():
                    return
                if page in done:
                    continue
                flags = triage_page(doc[page - 1], self.settings.language, self.settings.doc_type)
                ctx.manifest.add_page(flags.to_row(), self.router.initial_tasks(flags))
            ctx.triage_done.set()
        except Exception as exc:  # noqa: BLE001
            ctx.events.log("triage_crash", error=repr(exc)[:300])
            print(f"triage crashed: {exc!r}")
            ctx.stop.set()

    def execute(self) -> dict:
        ctx = self.ctx or self.prepare()
        m, s = ctx.manifest, self.settings
        engines = self.router.engines()
        requeued = m.requeue_inflight()
        ctx.events.log("run_start", pages=self.n_pages, policy=s.policy, resume=self.resume, requeued=requeued)
        if requeued:
            self.say(f"resume: {requeued} in-flight page tasks back to pending")

        lanes: list[Any] = []
        sarvam_lane: SarvamLane | None = None
        if "sarvam" in engines:
            sarvam_lane = SarvamLane(ctx, self.router, api=self.sarvam)
            lanes.append(sarvam_lane)
        if "bodhan" in engines:
            backend = self.bodhan_backend
            if backend is None:
                backend = (
                    BodhanLocalBackend(s.table_format)
                    if BODHAN_BACKEND == "local"
                    else BodhanAPIBackend(s.table_format)
                )
            from src.indic_local import runtime

            # PyTorch local stays one page at a time. MLX and vLLM batch crops across pages.
            workers = 1 if isinstance(backend, BodhanLocalBackend) and runtime() == "pytorch" else s.bodhan_workers
            steal_from = "sarvam" if (s.policy == "sarvam-primary" and s.steal) else None
            backlog = sarvam_lane.per_job * s.max_open_jobs if sarvam_lane else 0
            lanes.append(PageLane(ctx, self.router, backend, workers, steal_from=steal_from, steal_min_backlog=backlog))
        if "qwen" in engines:
            lanes.append(PageLane(ctx, self.router, self.qwen_backend or QwenBackend(), 4))

        triage = threading.Thread(target=self._triage, name="triage", daemon=True)
        started = time.monotonic()
        done_at_start = m.pages_done_by_any(self.selection)
        triage.start()
        for lane in lanes:
            lane.start()

        last_print = 0.0
        stopped_reason = None
        while True:
            time.sleep(0.2)
            if ctx.hard_stop.is_set():
                stopped_reason = "hard_stop"
                break
            open_jobs = len(sarvam_lane.open) if sarvam_lane else 0
            active = m.count()
            idle = not any(getattr(lane, "busy", False) for lane in lanes)
            if ctx.triage_done.is_set() and active == 0 and open_jobs == 0 and idle:
                break
            in_flight = m.count(states=("inflight", "queued"))
            if not ctx.accepting() and in_flight == 0 and open_jobs == 0:
                stopped_reason = "budget" if ctx.budget_exhausted.is_set() else "stopped"
                break
            if not any(ctx.alive[e] for e in engines):
                stopped_reason = "all_lanes_dead"
                break
            now = time.monotonic()
            if now - last_print >= self.progress_every:
                last_print = now
                self._progress(started, done_at_start)

        ctx.done.set()
        for lane in lanes:
            lane.join(timeout=30)
        triage.join(timeout=5)
        summary = self._summary(started, stopped_reason)
        (ctx.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
        ctx.events.log("run_end", complete=summary["complete"], stopped=stopped_reason, cost_inr=summary["cost_inr"])
        m.close()
        return summary

    def _progress(self, started: float, done_at_start: int) -> None:
        ctx = self.ctx
        assert ctx is not None
        counts = ctx.manifest.counts()
        done = ctx.manifest.pages_done_by_any(self.selection)
        elapsed = time.monotonic() - started
        rate = (done - done_at_start) / (elapsed / 60) if elapsed > 1 else 0.0
        remaining = self.n_pages - done
        eta = f"{remaining / rate:.0f} min" if rate > 0 and remaining > 0 else "-"
        lanes = " | ".join(f"{e}: {_fmt_states(c)}" for e, c in sorted(counts.items()))
        self.say(
            f"[{elapsed:5.0f}s] pages {done}/{self.n_pages} | {lanes} | "
            f"₹{ctx.budget.spent:.2f} | {rate:.1f} pages/min | ETA {eta}"
        )

    def _summary(self, started: float, stopped_reason: str | None) -> dict:
        ctx = self.ctx
        assert ctx is not None
        m = ctx.manifest
        wall = time.monotonic() - started
        events = ctx.events.read()
        kinds = Counter(e["kind"] for e in events)
        outcomes = Counter(
            (e.get("engine"), e.get("verdict")) for e in events if e["kind"] == "outcome"
        )
        counts = m.counts()
        done = m.pages_done_by_any(self.selection)
        complete = stopped_reason is None and m.count() == 0
        jobs = m.jobs()
        return {
            "document": str(self.pdf_path),
            "out_dir": str(ctx.out_dir),
            "pages_in_pdf": self.total_pages,
            "pages_in_run": self.n_pages,
            "pages_done": done,
            "complete": complete,
            "stopped": stopped_reason,
            "policy": self.settings.policy,
            "language": self.settings.language,
            "wall_s": round(wall, 1),
            "pages_per_min": round(done / (wall / 60), 1) if wall > 0 else None,
            "cost_inr": round(m.spent_inr(), 2),
            "engines": {
                e: {
                    "states": c,
                    "cost_inr": round(m.spent_inr(e), 2),
                    "cache_hits": m.cache_hits(e),
                    "rate_limited": outcomes.get((e, "rate_limited"), 0),
                    "retries": outcomes.get((e, "retry"), 0),
                }
                for e, c in counts.items()
            },
            "sarvam_jobs": {
                "total": len(jobs),
                "by_state": dict(Counter(j["state"] for j in jobs)),
                "submissions": kinds.get("job_submitted", 0),
            },
            "events": dict(kinds),
        }


def print_summary(summary: dict) -> None:
    print()
    print(f"Done: {summary['pages_done']}/{summary['pages_in_run']} pages"
          f"{' (complete)' if summary['complete'] else ' (stopped: ' + str(summary['stopped']) + ')'}")
    print(f"Wall {summary['wall_s']} s, {summary['pages_per_min']} pages/min, cost ₹{summary['cost_inr']}")
    for engine, info in summary["engines"].items():
        print(f"  {engine:7s} {_fmt_states(info['states'])} | ₹{info['cost_inr']} | cache hits {info['cache_hits']}"
              f" | 429s {info['rate_limited']} | retries {info['retries']}")
    jobs = summary["sarvam_jobs"]
    if jobs["total"]:
        print(f"  sarvam jobs: {jobs['total']} {jobs['by_state']}")
    print(f"Outputs: {summary['out_dir']}")


def parse_pages(spec: str) -> list[int]:
    """'1-3,7,10-12' -> [1, 2, 3, 7, 10, 11, 12] (sorted, unique)."""
    pages: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = (int(x) for x in part.split("-", 1))
            pages.update(range(min(a, b), max(a, b) + 1))
        else:
            pages.add(int(part))
    return sorted(pages)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m src.run", description="OCR a PDF of any length with Sarvam + Bodhan.")
    p.add_argument("pdf", type=Path)
    p.add_argument("--out-root", type=Path, default=DOCS_OUT_DIR,
                   help="folder that gets this run's <name>-<hash> output folder (default out/docs); the web app gives every job its own")
    p.add_argument("-l", "--language", default="auto",
                   help="auto (default; detected per page and per block), or a hint such as hi-IN, mr-IN, en-IN")
    p.add_argument("--mode", choices=tuple(modes.MODES), default=modes.DEFAULT_MODE,
                   help="fast: engines + free checks, no model calls; normal (default): + image check on open table "
                        "cells; think: + Bodhan on every page, more accurate image check (src/modes.py)")
    p.add_argument("--doc-type", choices=DOC_TYPES, default="auto",
                   help="auto (default): born digital or scanned decided per page; digital or scanned: force every page")
    p.add_argument("--lead", choices=modes.LEADS,
                   help="the engine that reads every page and supplies the document (default: bodhan when it runs locally "
                        "on vLLM, else sarvam; BODHAN_LEAD in .env)")
    p.add_argument("--policy", choices=POLICIES,
                   help="override the mode's engine routing (fast/normal: <lead>-primary, think: both)")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--resume", action="store_true", help="continue a previous run of this PDF")
    group.add_argument("--fresh", action="store_true", help="discard the previous run's state (cache is kept)")
    p.add_argument("--max-pages", type=int, help="only the first N pages (of the selection)")
    p.add_argument("--pages", help="page selection, e.g. 120-135,200-210 (default: all)")
    p.add_argument("--max-cost-inr", type=float, help="stop taking new work at this spend")
    p.add_argument("--dpi", type=int, default=RENDER_DPI)
    p.add_argument("--table-format", choices=("html", "markdown"), default=BODHAN_TABLE_FORMAT)
    p.add_argument("--chunk-pages", type=int, default=10, help="pages per PDF inside a Sarvam job (max 10)")
    p.add_argument("--files-per-job", type=int, default=5)
    p.add_argument("--max-open-jobs", type=int, default=3)
    p.add_argument("--bodhan-workers", type=int,
                   help=f"Bodhan pages in flight (default: API {BODHAN_MAX_INFLIGHT}, local MLX {BODHAN_MLX_PAGES}, local vLLM {BODHAN_VLLM_PAGES})")
    p.add_argument("--no-steal", action="store_true", help="Bodhan never takes pages from Sarvam's queue")
    p.add_argument("--qwen", action="store_true", help="also read English pages with Qwen (OpenRouter)")
    p.add_argument("--no-assemble", action="store_true", help="skip building doc.md, tables and report.md")
    p.add_argument("--no-agent", action="store_true", help="assembly uses rules only (no LLM / vision calls)")
    p.add_argument("--no-reocr", action="store_true", help="assembly never re-reads failing tables with Bodhan")
    p.add_argument("--table-qa", choices=("legacy", "standard"), default="standard",
                   help="table checking: standard (aligned readings, free evidence, image check by the measured rule, new joins; default) or legacy")
    p.add_argument("-y", "--yes", action="store_true", help="do not ask to confirm the cost estimate")
    return p.parse_args(argv)


def default_bodhan_workers() -> int:
    """API: its parallel limit. Local MLX and vLLM: enough pages to keep the batch full."""
    if BODHAN_BACKEND == "local":
        from src.indic_local import runtime

        return {"mlx": BODHAN_MLX_PAGES, "vllm": BODHAN_VLLM_PAGES}.get(runtime(), 1)
    return BODHAN_MAX_INFLIGHT


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    from src.langid import SARVAM_LANGUAGES

    language = "auto" if args.language.strip().lower() == "auto" else resolve_ocr_language(args.language)
    if language != "auto" and language not in SARVAM_LANGUAGES:
        print(f"error: unknown language {args.language!r}; use auto or one of {sorted(SARVAM_LANGUAGES)}", file=sys.stderr)
        return 2
    mode = modes.get(args.mode)
    lead = args.lead or modes.default_lead()
    settings = Settings(
        language=language,
        policy=args.policy or modes.policy_for(mode, lead),
        lead=lead,
        dpi=args.dpi,
        table_format=args.table_format,
        chunk_pages=max(1, min(args.chunk_pages, 10)),
        files_per_job=max(1, args.files_per_job),
        max_open_jobs=max(1, args.max_open_jobs),
        bodhan_workers=max(1, args.bodhan_workers or default_bodhan_workers()),
        steal=not args.no_steal,
        qwen=args.qwen,
        max_pages=args.max_pages,
        max_cost_inr=args.max_cost_inr,
        pages=parse_pages(args.pages) if args.pages else None,
        doc_type=args.doc_type,
    )
    run = Run(args.pdf, settings, resume=args.resume, fresh=args.fresh, out_root=args.out_root)
    try:
        ctx = run.prepare()
    except RunError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"{run.pdf_path.name}: {run.n_pages} of {run.total_pages} pages, mode {mode.name}, lead {settings.lead}, policy {settings.policy}, "
          f"language {language}, document type {settings.doc_type}")
    with pymupdf.open(run.pdf_path) as doc:
        pre = estimate.preflight(doc, run.selection, settings.doc_type)
    estimates = estimate.estimate_all(pre, lead=settings.lead)
    print(estimate.render(pre, estimates, mode.name)
          + (f"\n  spending cap for this run: ₹{settings.max_cost_inr}" if settings.max_cost_inr is not None else ""))
    if settings.policy != modes.policy_for(mode, settings.lead):
        print(f"  note: --policy {settings.policy} replaces the mode's {modes.policy_for(mode, settings.lead)}; "
              f"the OCR estimate assumes the mode's")
    if not args.yes and sys.stdin.isatty():
        if input("Start? [Y/n] ").strip().lower() in ("n", "no"):
            return 0

    def on_sigint(signum: int, frame: Any) -> None:
        if ctx.stop.is_set():
            print("\nsecond Ctrl-C: leaving now (state is saved; use --resume)")
            ctx.hard_stop.set()
        else:
            print("\nCtrl-C: finishing requests in flight, then stopping (Ctrl-C again to leave now)")
            ctx.stop.set()

    signal.signal(signal.SIGINT, on_sigint)
    summary = run.execute()
    print_summary(summary)
    if not args.no_assemble and summary["pages_done"] and not ctx.hard_stop.is_set():
        from src.agent import Agent
        from src.assemble import Assembler

        print()
        result = Assembler(
            Path(summary["out_dir"]),
            agent=None if args.no_agent else Agent(),
            reocr=not args.no_reocr,
            max_cost_inr=settings.max_cost_inr,
            qa_mode=args.table_qa,
            mode=mode.name,
        ).run()
        t = result["tables"]
        if t.get("qa_status"):
            print(f"Tables: {t['found']} found, {t['after_joining']} after joining across pages; status {t['qa_status']}; "
                  f"cells settled by {t['cells_by']}; left for review {t['questions'] or 'nothing'}")
        else:
            print(f"Tables: {t['found']} found, {t['after_joining']} after joining across pages; "
                  f"checks {t['checks_total'] - t['checks_failed']}/{t['checks_total']} passed; "
                  f"cells changed {t['cells_fixed'] or 0}, confirmed {t['cells_confirmed'] or 0}; unresolved {t['unresolved_cells']}")
        print(f"Assembly cost ₹{result['cost']['assembly_inr']}; document total ₹{result['cost']['total_inr']}")
        chosen = next(e for e in estimates if e["mode"] == mode.name)
        actual = {"total_inr": result["cost"]["total_inr"], "assembly_inr": result["cost"]["assembly_inr"],
                  "minutes": round((summary["wall_s"] + (result.get("assembly_wall_s") or 0)) / 60, 1),
                  "table_pages": len({r["page"] for r in result["tables"].get("list", [])})}
        print(f"Estimate was {chosen['total_inr'][0]}–{chosen['total_inr'][1]} ₹ and {chosen['minutes'][0]}–"
              f"{chosen['minutes'][1]} min; actual ₹{actual['total_inr']} and {actual['minutes']} min (cached pages cost nothing)")
        estimate.record(None, document=str(run.pdf_path), mode=mode.name, estimate=chosen, actual=actual)
        print(f"Read: {Path(summary['out_dir']) / 'doc.md'}  and  report.md")
    if summary["complete"]:
        return 0
    return 3 if summary["stopped"] == "budget" else 1


if __name__ == "__main__":
    raise SystemExit(main())
