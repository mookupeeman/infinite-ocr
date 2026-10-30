"""Fast / Normal / Think: one switch for how much each document run reads and checks.

Every mode keeps the validator rule: a verifier model never writes a value no engine produced, never
empties a cell over printed text, and never settles rare numerals or cells a proven text layer
contradicts. Cells nothing could settle stay "needs review" with the primary engine's reading.

    fast    Two OCR engines only. The lead engine reads every page; the other reads pages with
            tables. Stage 2 lines up both readings and runs free checks: table formulas, ruling
            lines, trusted PDF text, and joins across pages. No vision or LLM calls.

    Note — free checks (Stage 2: src/table_qa.py and related modules; no API calls; every mode):
            Line up both engine readings of each table in one grid, cell by cell. Accept a cell
            when the readings match, when they differ only in spaces, dashes, or quotes, or when
            only one engine returned that cell.

            Printed formulas. When readings disagree on a number, pick the reading for which the
            table's own formulas hold (for example column numbers १ २ ३ or a header like
            एकूण (४ ते ७), "sum columns 4 through 7"). Every formula in that cell's row and
            column must pass, not just fewer failures overall.

            PDF text (born-digital only). Selectable text sits under the print. On cells both
            engines already agree on, check that the PDF text contains the same value. The layer
            is proven only after at least five such cells were tested and at least 90% matched.
            A disputed value cell settles when the proven layer contains exactly one of the two
            engine readings (look up both; only one may match). layer_repair.py learns legacy
            font mappings from agreed cells, then retries the check.

            Totals without a printed formula. Rows or columns or Total with no
            printed formula: check whether the numbers add up. Flag cells where one engine's
            totals work for most rows and the other engine's do not.

            Grid shape. table_geometry.py reads the rendered page image (scans included, after
            deskew) for ruling lines. When detection is confident, compare that row and column
            count to each engine (choose_structure in table_qa.py). Flag a structure problem when
            no grid matches the lines or both grids are too small. Unruled tables or faint lines
            give weak geometry; fall back to matching engine shapes or the table's column-number
            row.

            Two engine versions. Sarvam and Bodhan may return the same table as two HTML grids.
            tables.fuse keeps each cell from whichever engine passes more formulas, totals, and
            text-layer checks.

            Tables across pages. table_join.py compares the last table on page N with the first
            on page N+1. Merge when columns line up, headers repeat or continue, serial numbers
            follow on, or the PDF marks the table as continued.

    normal  Same page routing as fast (lead engine on every page, the other on table pages).
            Vision verification (table_judge.py through agent.py, usually OpenRouter): runs only
            on cells still marked open after free checks. The model sees a crop of the printed
            table plus the two engine readings as candidates A and B. It may pick A, B, neither,
            or unreadable. The pipeline accepts an answer only if it equals a value Sarvam or
            Bodhan already returned.

            By page type. Born-digital tables: one vision model per open cell. 
            For scanned tables with clear ruling lines: two models from different families must read the same value,
            and that value must match an engine reading. Scanned tables without ruling lines: no
            vision call; those cells stay for review.

            How images are sent. Up to 24 open cells per API call. When the grid matches the
            ruling lines, each question uses a small crop around that cell. Otherwise the model
            sees a horizontal strip of whole rows. Up to 5 blind audits per document on agreed
            cells tagged risky in table_qa._risks (for example proven PDF text disagrees with
            the reading, rare or look-alike numerals, text in an empty cell or blank where text
            was read, a non-number in a mostly numeric column, or a total or formula hint that
            still fails). assemble.py sums RISK_WEIGHT per cell and audits the highest scores
            document-wide; an audit can flag a cell but never change its value. When table_join
            is unsure whether two segments continue across a page break, one seam picture goes
            to the model. Default.

    think   Stage 1 runs both engines on every page, not only table pages. Open cells use the
            same image tie-break as normal, with a different layout: wider row strips and at most
            12 cells per API call (no per-cell-only crops). On scanned ruled tables a third
            vision model joins; two of the three must agree on the same engine reading. Up to 15
            audits per document, chosen the same way as normal (RISK_WEIGHT in table_qa.py).
            After tables, src/text_qa.py compares the two engines' prose on each page word by
            word (no API calls). Disputes the free prose rules cannot settle stay for review.

There is no built-in spending cap: a capped run leaves tables unchecked, which defeats the mode.
`--max-cost-inr` on a run remains an explicit opt-in cap.

The lead engine reads every page and supplies the document (reading order, prose, headings, each
table's default reading); the other engine reads the table pages as the second reading. Sarvam leads
by default. Bodhan leads when it runs locally on vLLM: free and faster than Sarvam. On the real-table
benchmark neither engine is better overall (cells wrong, Sarvam / Bodhan: v1 85 / 107 of 1345,
v2 225 / 174 of 2544; Sarvam better on headers and Tamil, Bodhan on scanned numbers).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from src.config import BODHAN_BACKEND, BODHAN_LEAD, TABLE_JUDGE_MODELS, TABLE_JUDGE_SINGLE

# Measured on the 63 scanned ruled disputes of the 2026-10-21 pilot: (3.1, Luna, 3.5) 2 of 3 gives 46 right, harm 0,
# against 41 right with Gemini 3.8 Flash as the third (which alone harmed 13 cells and costs twice as much).
# Used with the provider's default reasoning, as measured: it cannot be switched off (HTTP 400).
THIRD_JUDGE = os.environ.get("TABLE_JUDGE_THIRD", "google/gemini-3.5-flash-lite")


@dataclass(frozen=True)
class JudgeProfile:
    """Which models settle which tables, and how cut-outs are laid out."""

    born_digital: tuple[str, ...] = (TABLE_JUDGE_SINGLE,)
    scanned_ruled: tuple[str, ...] = TABLE_JUDGE_MODELS
    vote: int | None = None  # how many must give the same engine reading; None = all of them
    per_sheet: int = 24
    cell_only: bool = True


@dataclass(frozen=True)
class Mode:
    """One effort level: how pages are routed and how much checking Stage 2 does."""
    name: str
    policy: str  # Stage 1 routing (lanes.POLICIES) when Sarvam leads; see policy_for
    image_check: bool  # any model calls in Stage 2
    judge: JudgeProfile
    audits: int  # per document
    summary: str
    text_check: bool = False  # compare the two engines' prose on every page (src/text_qa.py)


MODES = {
    "fast": Mode("fast", "sarvam-primary", False, JudgeProfile(), 0,
                 "engines + free checks; no model calls"),
    "normal": Mode("normal", "sarvam-primary", True, JudgeProfile(per_sheet=24, cell_only=True), 5,
                   "free checks + vision tie-break on open cells (24/sheet, per-cell crops when possible)"),
    "think": Mode("think", "both", True,
                  JudgeProfile(scanned_ruled=TABLE_JUDGE_MODELS + (THIRD_JUDGE,), vote=2, per_sheet=12, cell_only=False), 15,
                  "both engines on every page + vision tie-break (12/sheet, row strips) + third vote on scans + prose compare", text_check=True),
}
DEFAULT_MODE = "normal"
LEADS = ("sarvam", "bodhan")


def default_lead() -> str:
    """BODHAN_LEAD, or for auto: Bodhan when it runs locally on vLLM, else Sarvam."""
    if BODHAN_LEAD in LEADS:
        return BODHAN_LEAD
    if BODHAN_BACKEND != "local":
        return "sarvam"
    from src.indic_local import runtime

    return "bodhan" if runtime() == "vllm" else "sarvam"


def policy_for(mode: Mode, lead: str) -> str:
    """The mode's Stage 1 routing with this lead engine. Think reads both engines on every page either way."""
    if lead not in LEADS:
        raise ValueError(f"lead must be one of {LEADS}")
    return "bodhan-primary" if lead == "bodhan" and mode.policy == "sarvam-primary" else mode.policy


def get(name: str | None) -> Mode:
    name = (name or DEFAULT_MODE).lower()
    if name not in MODES:
        raise ValueError(f"mode must be one of {tuple(MODES)}")
    return MODES[name]
