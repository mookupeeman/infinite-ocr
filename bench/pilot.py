"""Step 4 pilot: which vision model answers the checker's questions best per rupee?

    python -m bench.pilot --dry-run                      # build sheets, count questions, estimate cost (free)
    python -m bench.pilot --models google/gemini-3.8-flash anthropic/claude-haiku-5.5 --cap-usd 3

The questions are the real ones: the standard checker run on the real Sarvam and Bodhan
readings of the benchmark pages (bench/real), where the truth is known. Disputes are
asked with the candidates shown; audits (agreed but risky cells) are asked blind.
Answers are cached, so a re-run costs nothing; spend goes to out/openrouter_ledger.jsonl.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from bench import metrics
from bench.run_bench import benchmark_set, benchmark_set_v2
from src.agent import Agent
from src.assemble import Assembler
from src.config import INR_PER_USD
from src.table_cutouts import Cutter, sheets
from src.table_qa import canon

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "pilot"
DEFAULT_MODELS = ["google/gemini-3.1-flash-lite", "anthropic/claude-haiku-5.5", "openai/gpt-5.6-luna",
                  "google/gemini-3.8-flash", "anthropic/claude-sonnet-5.5"]
# per-model request extras (measured 2026-10-21): OpenAI's models take an effort, not an on/off switch;
# Gemini 3.8 Flash and Claude Sonnet 5.5 refuse requests with reasoning disabled ("Reasoning is mandatory")
EXTRA = {"openai/": {"reasoning": {"effort": "minimal"}},
         "google/gemini-3.8": {"reasoning": {"effort": "low"}},
         "anthropic/claude-sonnet": {"reasoning": {"effort": "low"}}}


@dataclass
class Question:
    qid: str
    kind: str  # cell (dispute) | audit
    r: int
    c: int
    rowspan: int
    colspan: int
    truth: str
    value: str  # what the checker would output without the image check
    candidates: list[str] = field(default_factory=list)
    risk: list[str] = field(default_factory=list)
    where: str = ""


@dataclass
class Group:
    """Questions of one kind on one table page: one or more sheets."""
    key: str
    lang: str
    scanned: bool
    kind: str
    questions: list[Question]
    sheets: list = field(default_factory=list)


def _where(table, r: int, c: int) -> str:
    rows = table.rows_text()
    header = " / ".join(dict.fromkeys(canon(rows[h][c]) for h in range(table.header_rows) if rows[h][c]))
    others = [canon(x) for j, x in enumerate(rows[r]) if j != c and x.strip()][:3]
    return f"column headed '{header[:60]}', in the row that also contains: " + " | ".join(o[:30] for o in others)


def collect(per_sheet: int = 12, cell_only: bool = False) -> list[Group]:
    groups: list[Group] = []
    for version, runs_file, cases in (("v1", "runs.json", benchmark_set()), ("v2", "runs_v2.json", benchmark_set_v2())):
        cases = {c.id: c for c in cases}
        for run in json.loads((ROOT / "real" / runs_file).read_text()):
            case = cases[run["case"]]
            asm = Assembler(Path(run["out_dir"]), agent=None, reocr=False, verbose=False, qa_mode="standard")
            asm._load()
            for page in asm.pages:
                spec = case.pages[page - 1]
                if spec.table is None:
                    continue
                blocks = [b for b in asm.blocks[page][asm.primary[page]] if b.kind == "table"]
                for k, block in enumerate(blocks):
                    rec = asm._build_table_qa(page, block, f"t{page}_{k}")
                    if rec is None or (rec.table.n_rows, rec.table.n_cols) != (spec.table.n_rows, spec.table.n_cols):
                        continue  # truth positions only line up when the shape is right
                    truth = {(c.r, c.c): c.text for c in spec.table.cells}
                    spans = {(c.r, c.c): (c.rowspan, c.colspan) for c in rec.table.cells}
                    by_kind = defaultdict(list)
                    for i, q in enumerate(rec.questions):
                        if q["kind"] not in ("cell", "audit") or (q["r"], q["c"]) not in truth:
                            continue
                        rs, cs = spans.get((q["r"], q["c"]), (1, 1))
                        value = q.get("value") if q["kind"] == "audit" else rec.table.rows_text()[q["r"]][q["c"]]
                        by_kind[q["kind"]].append(Question(
                            f"Q{len(by_kind[q['kind']]) + 1}", q["kind"], q["r"], q["c"], rs, cs, truth[(q["r"], q["c"])], value,
                            q.get("candidates", []), q.get("risk", []), _where(rec.table, q["r"], q["c"])))
                    if not by_kind:
                        continue
                    cutter = Cutter(asm.doc, page, asm.info[page]["rotate"], rec.bbox, rec.table)
                    for kind, qs in by_kind.items():
                        g = Group(f"{version}/{case.id}/{run['mode']}/p{page}/{kind}", case.lang, run["mode"] == "scanned", kind, qs)
                        g.sheets = sheets(cutter, [(q.qid, q.r, q.c, q.rowspan, q.colspan) for q in qs], per_sheet, cell_only)
                        exact = {qid: e for s in g.sheets for qid, e in zip(s.qids, s.exact)}
                        for q in qs:
                            if not exact[q.qid]:
                                q.where = _where(rec.table, q.r, q.c)
                            else:
                                q.where = "inside the red box"
                        groups.append(g)
    return groups


def ask(agent: Agent, model: str, g: Group, reasoning: str | None = None, think_tokens: int = 3000,
        style: str = "dispute") -> list[dict]:
    by_id = {q.qid: q for q in g.questions}
    out = []
    extra = {"reasoning": {"effort": reasoning}} if reasoning else next((v for k, v in EXTRA.items() if model.startswith(k)), None)
    for sheet in g.sheets:
        qs = []
        for qid in sheet.qids:
            q = by_id[qid]
            item = {"id": qid, "where": q.where}
            if g.kind == "cell":
                item["candidates"] = {chr(65 + i): t for i, t in enumerate(q.candidates)}
            qs.append(item)
        started = time.time()
        answer = agent.judge_cells(sheet.png, qs, style if g.kind == "cell" else "audit", model=model, extra=extra,
                                   think_tokens=think_tokens)
        if answer and style == "choose":  # a chosen letter stands for that candidate's text
            for a in answer.get("answers", []):
                q = by_id.get(a.get("id"))
                letter = (a.get("match") or "").strip().upper()[:1]
                if q is not None and q.kind == "cell" and letter and ord(letter) - 65 < len(q.candidates):
                    a["reading"] = q.candidates[ord(letter) - 65]
        seconds = time.time() - started
        got = {a.get("id"): a for a in (answer or {}).get("answers", [])}
        for i, qid in enumerate(sheet.qids):
            q, a = by_id[qid], got.get(qid)
            row = score(q, a, g, seconds / max(1, len(sheet.qids)))
            row["cost_inr"] = (answer or {}).get("_cost_inr", 0.0) if i == 0 else 0.0  # one call per sheet
            out.append(row)
    return out


def score(q: Question, a: dict | None, g: Group, seconds: float) -> dict:
    truth, value = metrics.norm(q.truth), metrics.norm(q.value)
    row = {"group": g.key, "lang": g.lang, "scanned": g.scanned, "kind": q.kind, "qid": q.qid, "boxed": q.where == "inside the red box",
           "truth": q.truth, "value": q.value, "value_right": value == truth, "answered": a is not None, "seconds": seconds}
    if a is None:  # no answer: counted as unanswered, never as right
        return {**row, "final": q.value, "final_right": False, "flagged": False}
    reading = metrics.norm(a.get("reading", ""))
    row.update(reading=a.get("reading", ""), reading_right=reading == truth, confidence=a.get("confidence"), match=a.get("match"))
    if q.kind == "cell":
        letters = {chr(65 + i): t for i, t in enumerate(q.candidates)}
        chosen = letters.get((a.get("match") or "").strip().upper()[:1])
        final = chosen if chosen is not None else a.get("reading", "")
        row.update(final=final, final_right=metrics.norm(final) == truth, chose_candidate=chosen is not None,
                   truth_among_candidates=truth in {metrics.norm(t) for t in q.candidates})
    else:  # audit: does the model's blind reading disagree with the agreed value?
        flagged = reading != value
        row.update(final=a.get("reading", "") if flagged else q.value, flagged=flagged,
                   final_right=(reading == truth) if flagged else value == truth)
    return row


def summarise(rows: list[dict]) -> dict:
    s: dict = {}
    for kind in ("cell", "audit"):
        part = [r for r in rows if r["kind"] == kind]
        if not part:
            continue
        n = len(part)
        d = {"questions": n, "answered": sum(r["answered"] for r in part),
             "reading_right": sum(r.get("reading_right", False) for r in part),
             "final_right": sum(r["final_right"] for r in part), "before_right": sum(r["value_right"] for r in part),
             "made_worse": sum(r["value_right"] and not r["final_right"] for r in part),
             "fixed": sum((not r["value_right"]) and r["final_right"] for r in part)}
        if kind == "audit":
            wrong = [r for r in part if not r["value_right"]]
            d.update(wrong_agreed=len(wrong), caught=sum(r.get("flagged", False) for r in wrong),
                     false_flags=sum(r.get("flagged", False) for r in part if r["value_right"]))
        for name, sel in (("scanned", lambda r: r["scanned"]), ("born_digital", lambda r: not r["scanned"]),
                          ("boxed", lambda r: r["boxed"]), ("not_boxed", lambda r: not r["boxed"])):
            sub = [r for r in part if sel(r)]
            d[f"final_right_{name}"] = [sum(r["final_right"] for r in sub), len(sub)]
        s[kind] = d
    s["cost_inr"] = round(sum(r["cost_inr"] for r in rows), 3)
    s["by_lang"] = {lang: [sum(r["final_right"] for r in rows if r["lang"] == lang), sum(r["lang"] == lang for r in rows)]
                    for lang in sorted({r["lang"] for r in rows})}
    return s


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m bench.pilot")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--cap-usd", type=float, default=3.0, help="stop when this pilot has spent this much")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, help="only the first N groups (for a quick try)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--reasoning", choices=["minimal", "low", "medium", "high"], help="reasoning effort for every model given")
    ap.add_argument("--think-tokens", type=int, default=3000, help="extra max_tokens for reasoning")
    ap.add_argument("--per-sheet", type=int, default=12, help="questions per sheet")
    ap.add_argument("--cell-only", action="store_true", help="one small snippet per cell where ruling lines give exact boxes")
    ap.add_argument("--style", choices=["dispute", "choose"], default="dispute", help="dispute: read then match; choose: letter only")
    ap.add_argument("--tag", default="", help="suffix for the result files (e.g. -high)")
    a = ap.parse_args(argv)

    groups = collect(a.per_sheet, a.cell_only)[: a.limit]
    n_sheets = sum(len(g.sheets) for g in groups)
    n_q = sum(len(g.questions) for g in groups)
    px = sum(s.size[0] * s.size[1] for g in groups for s in g.sheets)
    print(f"{len(groups)} groups, {n_q} questions ({sum(len(g.questions) for g in groups if g.kind == 'cell')} disputes, "
          f"{sum(len(g.questions) for g in groups if g.kind == 'audit')} audits), {n_sheets} sheets, "
          f"{px / max(1, n_sheets) / 1e6:.2f} Mpx per sheet on average")
    OUT.mkdir(parents=True, exist_ok=True)
    if a.dry_run:
        for i, g in enumerate(groups[:3]):
            (OUT / f"sample_{i}.png").write_bytes(g.sheets[0].png)
        # rough tokens: ~1 token per 750 px (Anthropic's rule; others are similar or cheaper) + 90 out per question
        tok_in, tok_out = px / 750 + 600 * n_sheets, 90 * n_q
        from src.agent import _openrouter_prices
        prices = _openrouter_prices()
        for m in a.models:
            pin, pout = prices.get(m, (0, 0))
            print(f"  {m:<36} estimate US${tok_in * pin + tok_out * pout:.3f}")
        return 0

    agent = Agent(budget_usd=10.0)
    start_spend = agent.spent_usd()
    results = {}
    for model in a.models:
        if agent.spent_usd() - start_spend >= a.cap_usd:
            print(f"pilot cap US${a.cap_usd} reached; skipping {model}")
            break
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            rows = [r for part in pool.map(lambda g: ask(agent, model, g, a.reasoning, a.think_tokens, a.style), groups) for r in part]
        summary = summarise(rows)
        summary["wall_s"] = round(time.time() - t0, 1)
        results[model] = summary
        (OUT / f"{model.replace('/', '__')}{a.tag}.json").write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1))
        c, au = summary.get("cell", {}), summary.get("audit", {})
        print(f"{model:<36} answered {c.get('answered', 0) + au.get('answered', 0)}/{c.get('questions', 0) + au.get('questions', 0)} | "
              f"disputes right {c.get('final_right')}/{c.get('questions')} (before {c.get('before_right')}, "
              f"made worse {c.get('made_worse')}) | audits caught {au.get('caught')}/{au.get('wrong_agreed')} "
              f"false flags {au.get('false_flags')} | Rs {summary['cost_inr']} | {summary['wall_s']}s", flush=True)
    spent = agent.spent_usd() - start_spend
    print(f"pilot spend US${spent:.3f} (Rs {spent * INR_PER_USD:.2f})")
    summary_file = OUT / "summary.json"
    old = json.loads(summary_file.read_text()) if summary_file.is_file() else {}
    summary_file.write_text(json.dumps({**old, **{m + a.tag: v for m, v in results.items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
