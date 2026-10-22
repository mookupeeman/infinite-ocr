"""Fast / Normal / Think: one switch for how much each document run reads and checks.

Every mode keeps the validator rule: a model never writes a value no engine produced, never
empties a cell over printed text, and never settles rare numerals or cells a proven text layer
contradicts. Cells nothing could settle stay "needs review" with the primary engine's reading.

    fast    engines only. Sarvam on every page, Bodhan on table pages, the free table check
            (aligned readings, ruling lines, formulas, proven text layer) and joins. No model calls.
    normal  fast + the image check by the measured rule, in format C (24 questions a sheet,
            cell-only snippets where the ruling lines give exact boxes, row strips elsewhere),
            up to 5 audits, a seam picture for uncertain joins. Default.
    think   more readings: Bodhan on every page; the image check in format A (12 row strips a
            sheet, slightly more accurate in the pilot); scanned ruled tables get a third model
            (2 of 3 must agree with an engine reading: 46 of 56 decided right, harm 0, against 38 of 44
            for the pair, on the pilot's 63 scanned ruled disputes); up to 15 audits; the free
            page-text check (src/text_qa.py): the two engines' prose compared word by word.

There is no built-in spending cap: a capped run leaves tables unchecked, which defeats the mode.
`--max-cost-inr` on a run remains an explicit opt-in cap.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from src.config import TABLE_JUDGE_MODELS, TABLE_JUDGE_SINGLE

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
    name: str
    policy: str  # Stage 1 routing (lanes.POLICIES)
    image_check: bool  # any model calls in Stage 2
    judge: JudgeProfile
    audits: int  # per document
    summary: str
    text_check: bool = False  # compare the two engines' prose on every page (src/text_qa.py)


MODES = {
    "fast": Mode("fast", "sarvam-primary", False, JudgeProfile(), 0,
                 "engines + free checks; no model calls"),
    "normal": Mode("normal", "sarvam-primary", True, JudgeProfile(per_sheet=24, cell_only=True), 5,
                   "free checks + image check (format C) on open cells"),
    "think": Mode("think", "both", True,
                  JudgeProfile(scanned_ruled=TABLE_JUDGE_MODELS + (THIRD_JUDGE,), vote=2, per_sheet=12, cell_only=False), 15,
                  "Bodhan on every page + image check (format A, 2 of 3 on scans) + page-text check", text_check=True),
}
DEFAULT_MODE = "normal"


def get(name: str | None) -> Mode:
    name = (name or DEFAULT_MODE).lower()
    if name not in MODES:
        raise ValueError(f"mode must be one of {tuple(MODES)}")
    return MODES[name]
