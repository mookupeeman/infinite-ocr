"""Assemble one document from the per-page engine outputs of a run.

    python -m src.assemble out/docs/<name>-<sha8> [--no-agent] [--no-reocr]

For every table: pick the best engine version, then repair disputed cells in
order of cost (free checks -> PDF text-layer digits -> a Bodhan re-read of the
table crop -> the vision tie-breaker). Then join tables that continue across
pages, crop figures, and write:

    doc.md            the document: headings, text, HTML tables, figures
    tables/<id>.html|csv|json   every (joined) table with its checks
    figures/*.png     cropped figures
    report.md         what happened: coverage, checks, fixes, cost
    assembly.json     the same, machine-readable
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import csv
import io
import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pymupdf

from src import cache
from src.agent import Agent, same_number
from src.blocks import Block, from_bodhan
from src.langid import page_languages
from src.bodhan_ocr import BODHAN_MODEL, MAX_TOKENS, bodhan_request
from src.config import AGENT_MAX_INFLIGHT, BODHAN_BACKEND, BODHAN_INR_PER_PAGE, has_bodhan_key
from src.manifest import EventLog, Manifest
from src.pdf_pages import RenderSpec, page_transform, render_page
from src.rate_limit import CallFailed
import threading
from concurrent.futures import ThreadPoolExecutor

from src import layer_repair, modes, page_class, table_cutouts, table_join, table_judge, table_qa, text_qa
from src.table_geometry import table_geometry
from src.tables import (
    CONTINUED_RE,
    Table,
    check_table,
    clean,
    compare_numbers,
    continuation,
    failed,
    fuse,
    join,
    number_key,
    numbers_in,
    parse_number,
    parse_tables,
    text_layer_match,
)

ENGINE_ORDER = ("sarvam", "bodhan", "qwen")
TERMINAL_PUNCT = tuple(".?!।॥:;\"'”’)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_blocks(path: Path) -> list[Block]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Block(**{**d, "bbox": tuple(d["bbox"])}) for d in data]


def crop_png(doc: pymupdf.Document, page: int, rotate: int, bbox: tuple, dpi: int, margin: float = 0.01) -> bytes:
    """Crop a 0-1 box (upright frame) out of a page render."""
    rendered = render_page(doc, page, RenderSpec(dpi=dpi, rotate=rotate, max_long_px=6000))
    pix = pymupdf.Pixmap(rendered.data)
    x0 = max(0, int((bbox[0] - margin) * pix.width))
    y0 = max(0, int((bbox[1] - margin) * pix.height))
    x1 = min(pix.width, int((bbox[2] + margin) * pix.width) + 1)
    y1 = min(pix.height, int((bbox[3] + margin) * pix.height) + 1)
    irect = pymupdf.IRect(x0, y0, max(x1, x0 + 1), max(y1, y0 + 1))
    out = pymupdf.Pixmap(pix.colorspace, irect, pix.alpha)
    out.copy(pix, irect)
    out.set_origin(0, 0)
    return out.tobytes("png")


def layer_numbers(page: pymupdf.Page, rotate: int, bbox: tuple, margin: float = 0.01) -> Counter:
    """Numbers in the PDF text layer inside a 0-1 box of the upright frame."""
    transform, w, h = page_transform(page, rotate)
    upright = pymupdf.Rect((bbox[0] - margin) * w, (bbox[1] - margin) * h, (bbox[2] + margin) * w, (bbox[3] + margin) * h)
    clip = upright * ~transform
    keys = (number_key(word[4]) for word in page.get_text("words", clip=clip))
    return Counter(k for k in keys if k)


def format_like(value: float, like: str) -> str:
    """Format a computed number like an existing cell: same decimals, same digit script."""
    from src.tables import DIGIT_MAP, decimals

    s = f"{value:.{decimals(like)}f}"
    for ch in like:
        cp = ord(ch)
        if cp in DIGIT_MAP and cp > 127 and chr(DIGIT_MAP[cp]).isdigit():
            zero = cp - (DIGIT_MAP[cp] - ord("0"))
            return "".join(chr(zero + int(c)) if c.isdigit() else c for c in s)
    return s


def cleanup_text(text: str) -> str:
    """Light LaTeX-to-markdown for prose blocks (Bodhan writes italics as LaTeX)."""
    text = re.sub(r"\\textit\{([^{}]*)\}", r"*\1*", text)
    text = re.sub(r"\\textbf\{([^{}]*)\}", r"**\1**", text)
    text = re.sub(r"\$\^\{?([^${}]{1,6})\}?\$", r"<sup>\1</sup>", text)
    text = re.sub(r"\$([A-Za-z][A-Za-z ]{0,40})\$", r"*\1*", text)
    return text.strip()


def table_in_text(blocks: list[Block], block: Block, reference: Table | None) -> tuple[Block, Table] | None:
    """A table another engine wrote inside a text block. Bodhan sometimes files a whole page body as one `List`
    block with the tables as HTML inside (measured on the annual report, page 317: three tables in one block).
    The block must cover at least 80% of `block`; among the tables parsed from its text, the one that holds
    most of the reference table's cells is taken (a merged table holds them all, plus its neighbours' rows).
    The block returned has `block`'s box, so the table's crop is not stretched to the whole page."""
    x0, y0, x1, y1 = block.bbox
    area = max(1e-9, (x1 - x0) * (y1 - y0))
    wanted = {" ".join(c.text.split()) for c in reference.cells if c.text.strip()} if reference is not None else set()
    best: tuple[float, Block, Table] | None = None
    for b in blocks:
        if b.kind not in ("text", "contents") or "<table" not in b.text.lower():
            continue
        ix = max(0.0, min(x1, b.bbox[2]) - max(x0, b.bbox[0]))
        iy = max(0.0, min(y1, b.bbox[3]) - max(y0, b.bbox[1]))
        if ix * iy < 0.8 * area:
            continue
        for table in parse_tables(b.text, {"page": b.page, "engine": b.engine, "in_text_block": True}):
            have = {" ".join(c.text.split()) for c in table.cells}
            score = len(wanted & have) / len(wanted) if wanted else float(table.n_rows * table.n_cols)
            if best is None or score > best[0]:
                best = (score, dataclasses.replace(b, bbox=block.bbox), table)
    return (best[1], best[2]) if best and (not wanted or best[0] >= 0.5) else None


@dataclass
class TableRec:
    id: str
    page: int
    order: int
    bbox: tuple
    table: Table
    engine: str
    checks_total: int = 0
    checks_failed: int = 0
    text_layer: dict | None = None
    agreement: dict | None = None
    versions: dict = field(default_factory=dict)
    fixes: list[dict] = field(default_factory=list)  # cells whose value changed
    confirmed: list[dict] = field(default_factory=list)  # disputed cells that kept their value
    unresolved: list[dict] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    joined_into: str | None = None
    join_decisions: list[dict] = field(default_factory=list)
    alts: dict = field(default_factory=dict)  # other engines' versions, for join-aware choice
    qa: dict | None = None  # table_qa summary (standard mode)
    qa_cells: list[dict] = field(default_factory=list)  # cells that were not plain agreement, or carry a risk
    questions: list[dict] = field(default_factory=list)  # left for the image check, or for review
    qt: Any = field(default=None, repr=False)  # the table_qa.QTable behind this table (standard mode)
    geometry: Any = field(default=None, repr=False)  # table_geometry.Geometry of this segment
    caption: str = ""  # caption or heading printed just above the table
    layer_raw: str | None = field(default=None, repr=False)  # PDF text layer in the table box, as printed
    readings: Any = field(default=None, repr=False)  # each engine's table, before the check (for a re-check)
    scanned: bool = False
    image_check: dict | None = None  # what the image check did (standard mode)

    def meta(self) -> dict:
        return {
            "id": self.id, "page": self.page, "pages": self.pages or [self.page], "engine": self.engine,
            "rows": self.table.n_rows, "cols": self.table.n_cols, "header_rows": self.table.header_rows,
            "checks_total": self.checks_total, "checks_failed": self.checks_failed,
            "text_layer": self.text_layer, "agreement": self.agreement, "versions": self.versions,
            "fixes": self.fixes, "confirmed": self.confirmed, "unresolved": self.unresolved, "flags": self.flags,
            "joined_into": self.joined_into, "join_decisions": self.join_decisions,
            "qa": self.qa, "qa_cells": self.qa_cells, "questions": self.questions, "image_check": self.image_check,
        }


# ---------------------------------------------------------------------------
# Assembler
# ---------------------------------------------------------------------------


class Assembler:
    def __init__(
        self,
        out_dir: Path,
        *,
        agent: Agent | None = None,
        reocr: bool = True,
        max_reocr: int = 10,
        max_cost_inr: float | None = None,
        verbose: bool = True,
        qa_mode: str = "standard",
        mode: str = modes.DEFAULT_MODE,
    ) -> None:
        if qa_mode not in ("legacy", "standard"):
            raise ValueError("qa_mode must be legacy or standard")
        self.qa_mode = qa_mode
        self.mode = modes.get(mode)
        self.out_dir = Path(out_dir)
        self.agent = agent if agent is not None and agent.available and self.mode.image_check else None
        self.reocr = reocr and (BODHAN_BACKEND == "local" or has_bodhan_key())
        self.max_reocr = max_reocr
        self.max_cost_inr = max_cost_inr
        self.verbose = verbose
        self.reocr_done = 0
        self.stats: Counter = Counter()

    def say(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def _can_spend(self, amount: float) -> bool:
        if self.max_cost_inr is None:
            return True
        return self.m.spent_inr() + amount <= self.max_cost_inr + 1e-9

    # -- loading ----------------------------------------------------------

    def _load(self) -> None:
        self.m = Manifest(self.out_dir / "manifest.sqlite")
        self.events = EventLog(self.out_dir / "events.jsonl")
        self.pdf_path = Path(self.m.get_meta("source") or "")
        self.doc = pymupdf.open(self.pdf_path)
        self.pages: list[int] = []
        self.info: dict[int, dict] = {}
        self.blocks: dict[int, dict[str, list[Block]]] = {}
        self.primary: dict[int, str] = {}
        for page in sorted(self.m.triaged_pages()):
            tasks = self.m.tasks_for_page(page)
            per_engine: dict[str, list[Block]] = {}
            for engine in ENGINE_ORDER:
                path = self.out_dir / "blocks" / f"page_{page:05d}.{engine}.json"
                if tasks.get(engine, {}).get("state") == "done" and path.is_file():
                    per_engine[engine] = load_blocks(path)
            if not per_engine:
                continue
            self.pages.append(page)
            self.info[page] = info = self.m.page_info(page)
            if not info.get("render"):  # a run triaged before page classes: classify now (free, a few ms)
                cls = page_class.classify(self.doc[page - 1], bool(info.get("legacy_font")))
                info.update(render=cls.render, text_layer=cls.text_layer, image_cover=cls.image_cover)
            self.blocks[page] = per_engine
            self.primary[page] = next(e for e in ENGINE_ORDER if e in per_engine)
        self._tag_languages()

    def _tag_languages(self) -> None:
        """Language of every block, from its text. Two passes: pages decide
        Devanagari languages from their own words first; pages with too few
        words then fall back to the document's most common decision."""
        fp = json.loads(self.m.get_meta("fingerprint") or "{}")
        user = fp.get("language")
        prior = user if user not in (None, "auto", "en-IN") else None
        self.page_langs: dict[int, dict[str, Any]] = {}

        def run(prior_lang: str | None) -> None:
            for page in self.pages:
                self.page_langs[page] = {}
                for engine, blocks in self.blocks[page].items():
                    per, total = page_languages([b.text for b in blocks], prior=prior_lang, kinds=[b.kind for b in blocks])
                    for b, found in zip(blocks, per):
                        b.lang = found.dominant
                    self.page_langs[page][engine] = total

        run(prior)
        if prior is None:
            votes = Counter()
            for page in self.pages:
                total = self.page_langs[page][self.primary[page]]
                if total.devanagari_confidence and total.devanagari_confidence >= 0.5:
                    votes[next((k for k in total.shares if k not in ("en-IN",)), None)] += 1
            votes.pop(None, None)
            if votes:
                run(votes.most_common(1)[0][0])

    # -- tables -----------------------------------------------------------

    def _versions(self, page: int, block: Block) -> list[tuple[str, Block, Table]]:
        """This table from every engine that read the page (matched by box overlap)."""
        from src.blocks import iou

        out: list[tuple[str, Block, Table]] = []
        for engine, blocks in self.blocks[page].items():
            candidates = [b for b in blocks if b.kind == "table"]
            best = max(candidates, key=lambda b: iou(b.bbox, block.bbox)) if candidates else None
            if engine == block.engine:
                best = block
            elif best is None or iou(best.bbox, block.bbox) < 0.3:
                mine = next((t for e, _, t in out if e == block.engine), None)
                found = table_in_text(blocks, block, mine)
                if found is not None:
                    out.append((engine, found[0], found[1]))
                continue
            tables = parse_tables(best.text, {"page": page, "engine": engine})
            if tables:
                out.append((engine, best, max(tables, key=lambda t: t.n_rows * t.n_cols)))
        return out

    def _score(self, table: Table, layer: Counter | None) -> tuple:
        checks = check_table(table)
        tl = text_layer_match(table, layer) if layer else None
        return (tl["ratio"] if tl else -1.0, -len(failed(checks)), len(checks)), checks, tl

    def _resolve_with_layer(self, table: Table, unresolved: list[dict], layer: Counter | None) -> tuple[list, list]:
        fixes, left = [], []
        for u in unresolved:
            ka, kb = number_key(u["a"]), number_key(u["b"])
            a_in, b_in = bool(layer and ka and layer[ka]), bool(layer and kb and layer[kb])
            if b_in and not a_in:
                cell = table.grid()[u["row"]][u["col"]]
                if cell is not None:
                    cell.text = u["b"]
                    fixes.append({**u, "to": u["b"], "by": "text_layer"})
            elif a_in and not b_in:
                fixes.append({**u, "to": u["a"], "by": "text_layer", "kept": True})
            else:
                left.append(u)
        return fixes, left

    def _reocr(self, page: int, bbox: tuple) -> Table | None:
        if not self.reocr or self.reocr_done >= self.max_reocr or not self._can_spend(BODHAN_INR_PER_PAGE):
            return None
        crop = crop_png(self.doc, page, self.info[page]["rotate"], bbox, dpi=300, margin=0.015)
        local = BODHAN_BACKEND == "local"
        if local:
            from src.indic_local import LOCAL_CACHE_NS, active_model, local_cache_extra

            key = cache.make_key(crop, LOCAL_CACHE_NS, active_model(), local_cache_extra("html"))
        else:
            key = cache.make_key(crop, "indic-ocr", BODHAN_MODEL, f"html|{MAX_TOKENS}")
        raw = cache.get(key)
        if raw is not None:
            payload = json.loads(raw.decode("utf-8"))
        else:
            try:
                if local:
                    from src.indic_local import local_ocr

                    payload = local_ocr(crop, "image/png", table_format="html")
                else:
                    payload = bodhan_request(crop, "image/png")
            except CallFailed as exc:
                self.say(f"  re-read of table on page {page} failed: {exc.outcome.reason}")
                return None
            except Exception as exc:  # noqa: BLE001 - a failed re-read must not stop assembly
                self.say(f"  re-read of table on page {page} failed: {type(exc).__name__}")
                return None
            cache.put(key, json.dumps(payload, ensure_ascii=False).encode("utf-8"))
            self.m.add_extra("bodhan_table_reread", 0.0 if local else BODHAN_INR_PER_PAGE, page)
            self.reocr_done += 1
        self.stats["table_rereads"] += 1
        pix = pymupdf.Pixmap(crop)
        tables = [
            t for b in from_bodhan(payload, page, pix.width, pix.height) if b.kind == "table"
            for t in parse_tables(b.text, {"page": page, "engine": "bodhan_crop"})
        ]
        return max(tables, key=lambda t: t.n_rows * t.n_cols) if tables else None

    def _tiebreak(self, rec: TableRec) -> None:
        """Vision model reads disputed cells. Two kinds of dispute:
        engines disagree (a vs b), or a total fails its check (printed vs computed)."""
        if not self.agent:
            return
        disputes = [dict(u, kind="engines") for u in rec.unresolved]
        seen = {(u["row"], u["col"]) for u in disputes}
        for ch in failed(check_table(rec.table)):
            cell = rec.table.grid()[ch.row][ch.col] if ch.row is not None and ch.col is not None else None
            if cell is None or (ch.row, ch.col) in seen or ch.expected is None:
                continue
            seen.add((ch.row, ch.col))
            disputes.append({"row": ch.row, "col": ch.col, "a": cell.text,
                             "b": format_like(ch.expected, cell.text), "kind": "check"})
        if not disputes:
            return
        rows = rec.table.rows_text()
        header = [" / ".join(dict.fromkeys(clean(rows[r][c]) for r in range(rec.table.header_rows) if rows[r][c]))
                  for c in range(rec.table.n_cols)]
        questions = []
        for i, u in enumerate(disputes[:20]):
            label = " ".join(t for t in rows[u["row"]] if t and parse_number(t) is None)[:80]
            questions.append({"id": f"q{i}", "row_label": label or f"row {u['row'] + 1}",
                              "column_header": header[u["col"]][:80] or f"column {u['col'] + 1}",
                              "a": u["a"], "b": u["b"]})
        image = crop_png(self.doc, rec.page, self.info[rec.page]["rotate"], rec.bbox, dpi=200)
        answer = self.agent.read_cells(image, questions)
        if not answer:
            return
        self.m.add_extra("vlm_tiebreak", answer.get("_cost_inr", 0.0), rec.page, {"questions": len(questions)})
        self.stats["vlm_calls"] += 1
        by_id = {a["id"]: a for a in answer.get("answers", [])}
        left = []
        for q, u in zip(questions, disputes[:20]):
            a = by_id.get(q["id"])
            choice = a.get("choice") if a else None
            if a and choice not in ("a", "b") and same_number(a.get("value", ""), u["b"]):
                choice = "b"
            # For a failed check, only "the printed total is really the computed sum" is a fix.
            value = u["b"] if choice == "b" else (u["a"] if choice == "a" and u["kind"] == "engines" else None)
            if value is None:
                if u["kind"] == "engines":
                    left.append({k: u[k] for k in ("row", "col", "a", "b")})
                continue
            cell = rec.table.grid()[u["row"]][u["col"]]
            entry = {"row": u["row"], "col": u["col"], "from": u["a"], "to": value,
                     "by": f"vision:{answer.get('_model')}", "dispute": u["kind"]}
            if cell is not None and cell.text != value:
                cell.text = value
                rec.fixes.append(entry)
            else:
                rec.confirmed.append(entry)
        rec.unresolved = left + [u for u in rec.unresolved if (u["row"], u["col"]) not in
                                 {(d["row"], d["col"]) for d in disputes[:20]}]

    def _scanned(self, page: int) -> bool:
        return self.info[page].get("render") == "raster"

    def _layer_text(self, page: int, bbox: tuple, margin: float = 0.01) -> str | None:
        """The PDF text layer inside a table box, when the page has a usable one (visible Unicode text)."""
        info = self.info[page]
        if (info.get("text_chars") or 0) < 50 or info.get("text_layer") != "unicode":
            return None
        pdf_page = self.doc[page - 1]
        transform, w, h = page_transform(pdf_page, info["rotate"])
        upright = pymupdf.Rect((bbox[0] - margin) * w, (bbox[1] - margin) * h, (bbox[2] + margin) * w, (bbox[3] + margin) * h)
        return pdf_page.get_text("text", clip=upright * ~transform)

    def _build_table_qa(self, page: int, block: Block, tid: str) -> TableRec | None:
        """Standard mode: align every reading, decide cells by free evidence, queue the rest."""
        versions = self._versions(page, block)
        if not versions:
            return None
        readings = {e: t for e, _, t in versions}
        boxes = [b.bbox for _, b, _ in versions]
        bbox = (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
        info = self.info[page]
        try:
            # one render at the cut-out resolution; the image check, audits and seam checks reuse it
            geometry = table_geometry(self.doc, page, info["rotate"], bbox, dpi=table_cutouts.DPI)
        except Exception as exc:  # noqa: BLE001 - geometry is evidence, never a reason to lose the table
            self.say(f"  geometry failed on page {page}: {exc}")
            geometry = None
        layer_raw = self._layer_text(page, bbox)
        originals = {e: copy.deepcopy(t) for e, t in readings.items()}
        qt = table_qa.check(readings, table_qa.Evidence(geometry, layer_raw, scanned=self._scanned(page)))
        rec = TableRec(tid, page, block.order, bbox, qt.table, qt.reference, pages=[page])
        # a table read out of a text block can hold the neighbouring tables too: fine as a second reading of
        # cells, wrong as a stand-in for the page's table when joining, so it is not offered to the joiner
        rec.alts = {e: copy.deepcopy(t) for e, t in readings.items() if e != qt.reference and not t.source.get("in_text_block")}
        rec.versions = {e: {"rows": t.n_rows, "cols": t.n_cols} for e, t in readings.items()}
        rec.qt, rec.scanned, rec.geometry = qt, self._scanned(page), geometry
        rec.layer_raw, rec.readings = layer_raw, originals
        primary = self.blocks[page][self.primary[page]]
        above = [b for b in primary if b.order < block.order and b.kind in ("caption", "heading")]
        rec.caption = above[-1].text if above else ""
        self._sync(rec)
        return rec

    def _sync(self, rec: TableRec) -> None:
        """Copy the QTable's state (after the free checks or the image check) onto the record."""
        qt = rec.qt
        rec.table = qt.table
        rec.qa = qt.summary()
        rec.qa_cells = [{**c.to_dict(), "page": rec.page} for c in qt.cells if c.decided_by != "agree" or c.risk]
        rec.questions = [{**q, "page": rec.page, "table": rec.id} for q in qt.questions]
        checks = check_table(qt.table)
        rec.checks_total, rec.checks_failed = len(checks), len(failed(checks))
        rec.flags = [f for f in rec.flags if f not in ("verified", "repaired", "needs_review")]
        if qt.status != "verified":
            rec.flags.append(qt.status)

    # -- page text (Think mode) -------------------------------------------

    def _check_text(self) -> None:
        """Compare the two engines' prose on every page both read; decisions go into the primary's blocks."""
        applied = 0
        for page in self.pages:
            engines = self.blocks[page]
            primary = self.primary[page]
            other = next((e for e in ENGINE_ORDER if e in engines and e != primary), None)
            if other is None:
                continue
            info = self.info[page]
            layer = self.doc[page - 1].get_text("text") if info.get("text_layer") == "unicode" else None
            result = text_qa.compare(page, engines[primary], engines[other], layer)
            applied += text_qa.apply(engines[primary], result)
            self.text_checks.append(result)
        self.stats["text_edits"] = applied

    # -- text-layer repair (standard mode) --------------------------------

    def _repair_layers(self) -> None:
        """Learn the document's glyph substitutions from all tables' agreed cells, then re-check the tables
        whose layer changed. Free; the repaired layer still has to prove itself before it decides anything."""
        recs = [r for r in self.recs.values() if r.qt is not None and r.layer_raw]
        samples = [(r.layer_raw, [c.value for c in r.qt.cells if c.decided_by == "agree"]) for r in recs]
        self.layer_rules = layer_repair.learn(samples)
        if not self.layer_rules:
            return
        for r in recs:
            repaired = layer_repair.apply(r.layer_raw, self.layer_rules)
            if repaired == r.layer_raw:
                continue
            qt = table_qa.check({e: copy.deepcopy(t) for e, t in r.readings.items()},
                                table_qa.Evidence(r.geometry, repaired, scanned=r.scanned))
            qt.layer["repaired_glyphs"] = len(self.layer_rules)
            if qt.layer.get("proven") or not r.qt.layer.get("proven"):  # never trade a proven layer for an unproven one
                r.qt = qt
                self._sync(r)
                self.stats["tables_with_repaired_layer"] += 1

    # -- image check (standard mode) --------------------------------------

    def _image_check(self) -> None:
        """Settle open cells from page cut-outs (table_judge), then audit the riskiest agreed cells. Only an explicit
        --max-cost-inr limits the spending; the modes have no cap of their own."""
        recs = [r for r in self.recs.values() if r.qt is not None]
        if not self.agent or not recs:
            return
        profile = self.mode.judge
        lock, local = threading.Lock(), threading.local()
        reserved = {"inr": 0.0}  # worst case of the tables being checked right now (matters only with --max-cost-inr)

        def doc() -> pymupdf.Document:  # PyMuPDF documents must not be shared between threads
            if not hasattr(local, "doc"):
                local.doc = pymupdf.open(self.pdf_path)
            return local.doc

        def one(rec: TableRec) -> bool:
            """Settle one table; False if the explicit --max-cost-inr had no room for it (yet)."""
            n_open = sum(c.open and table_judge.for_models(c) for c in rec.qt.cells)
            if not n_open:
                return True
            estimate = table_judge.estimate_inr(table_judge.boxed(rec.geometry, rec.qt.table), rec.scanned, n_open, profile)
            with lock:
                allowed = self._can_spend(estimate + reserved["inr"])
                if allowed:
                    reserved["inr"] += estimate
            if not allowed:
                rec.image_check = {"skipped": "over --max-cost-inr: left for review"}
                return False
            report = table_judge.settle(self.agent, doc(), rec.page, self.info[rec.page]["rotate"], rec.bbox, rec.qt, rec.scanned,
                                        geometry=rec.geometry, profile=profile)
            with lock:
                reserved["inr"] -= estimate
                rec.image_check = report.to_dict()
                if report.calls:
                    self.m.add_extra("table_image_check", report.cost_inr, rec.page,
                                     {"asked": report.asked, "decided": report.decided, "route": report.route})
                    self.stats["image_check_calls"] += report.calls
            return True

        # With --max-cost-inr, tables with fewer open cells go first (more tables finished). Tables run in parallel
        # and each reserves its worst case, so one turned away may fit once the others have settled at their actual
        # cost (often far less, or nothing on a cache hit): retry until none fits.
        waiting = sorted(recs, key=lambda r: sum(c.open and table_judge.for_models(c) for c in r.qt.cells))
        while waiting:
            with ThreadPoolExecutor(max_workers=AGENT_MAX_INFLIGHT) as pool:
                done = list(pool.map(one, waiting))
            left = [r for r, ok in zip(waiting, done) if not ok]
            if len(left) == len(waiting):
                break
            waiting = left

        # audits: the riskiest agreed cells in the whole document, at most the mode's number
        ranked = []
        for rec in recs:
            for c in rec.qt.cells:
                # only cells not already headed for review: there an audit can still add information
                if (not c.open and c.risk and not set(c.risk) & table_qa.HIGH_RISK and "audit_confirmed" not in c.risk
                        and table_judge.for_models(c)):
                    ranked.append((sum(table_qa.RISK_WEIGHT.get(k, 1) for k in c.risk), rec.id, c))
        ranked.sort(key=lambda x: -x[0])
        chosen: dict[str, list] = {}
        for _, rid, c in ranked[:self.mode.audits]:
            chosen.setdefault(rid, []).append(c)
        for rid, cells in chosen.items():
            rec = self.recs[rid]
            report = table_judge.audit(self.agent, self.doc, rec.page, self.info[rec.page]["rotate"], rec.bbox, rec.qt, cells, rec.scanned,
                                       geometry=rec.geometry, profile=profile)
            if report.calls:
                self.m.add_extra("table_audit", report.cost_inr, rec.page, {"audits": report.audits})
            rec.image_check = {**(rec.image_check or {}), "audits": report.audits, "audit_skipped": report.skipped}
        for rec in recs:
            self._sync(rec)
            self.stats[f"tables_{rec.qt.status}"] += 1

    def _build_table(self, page: int, block: Block, tid: str) -> TableRec | None:
        if self.qa_mode == "standard":
            return self._build_table_qa(page, block, tid)
        versions = self._versions(page, block)
        if not versions:
            return None
        pdf_page = self.doc[page - 1]
        layer = layer_numbers(pdf_page, self.info[page]["rotate"], block.bbox)
        layer = layer if sum(layer.values()) >= 3 else None
        scored = []
        for engine, _, table in versions:
            score, checks, tl = self._score(table, layer)
            scored.append((score, engine == block.engine, engine, table, checks, tl))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        _, _, engine, best, checks, tl = scored[0]
        best = copy.deepcopy(best)
        rec = TableRec(tid, page, block.order, block.bbox, best, engine, pages=[page])
        rec.alts = {e: t for (_, _, e, t, _, _) in scored[1:]}
        rec.versions = {
            e: {"rows": t.n_rows, "cols": t.n_cols, "checks": len(c), "failed": len(failed(c)),
                "text_layer": (tl_["ratio"] if tl_ else None)}
            for (_, _, e, t, c, tl_) in scored
        }
        if len(scored) > 1:
            other = scored[1][3]
            rec.agreement = compare_numbers(numbers_in(best), numbers_in(other))
            fusion = fuse(best, other)
            rec.fixes += [{**f, "by": "checks"} for f in fusion.replaced]
            layer_fixes, rec.unresolved = self._resolve_with_layer(best, fusion.unresolved, layer)
            rec.fixes += [f for f in layer_fixes if not f.get("kept")]
            rec.confirmed += [f for f in layer_fixes if f.get("kept")]
            if (best.n_rows, best.n_cols) != (other.n_rows, other.n_cols):
                rec.flags.append("engines_disagree_on_shape")
        final_checks = check_table(best)
        needs_help = bool(failed(final_checks)) or (bool(rec.unresolved) and layer is None)
        if needs_help:
            crop_table = self._reocr(page, block.bbox)
            if crop_table is not None and (crop_table.n_rows, crop_table.n_cols) == (best.n_rows, best.n_cols):
                g = crop_table.rows_text()
                left = []
                for u in rec.unresolved:  # majority of three readings
                    third = g[u["row"]][u["col"]]
                    pick = u["a"] if same_number(third, u["a"]) else u["b"] if same_number(third, u["b"]) else None
                    if pick is None:
                        left.append(u)
                        continue
                    cell = best.grid()[u["row"]][u["col"]]
                    entry = {**u, "to": pick, "by": "bodhan_crop_majority"}
                    if cell is not None and pick != cell.text:
                        cell.text = pick
                        rec.fixes.append(entry)
                    else:
                        rec.confirmed.append(entry)
                rec.unresolved = left
                fusion = fuse(best, crop_table)  # may also repair failing checks
                rec.fixes += [{**f, "by": "bodhan_crop+checks"} for f in fusion.replaced]
            elif crop_table is not None:
                rec.flags.append("reread_shape_differs")
        self._tiebreak(rec)
        final_checks = check_table(best)
        rec.checks_total, rec.checks_failed = len(final_checks), len(failed(final_checks))
        rec.text_layer = text_layer_match(best, layer) if layer else None
        if rec.checks_failed:
            rec.flags.append("checks_failed")
        if rec.unresolved:
            rec.flags.append("unresolved_cells")
        return rec

    # -- joining across pages ---------------------------------------------

    def _body(self, page: int) -> list[Block]:
        return [b for b in self.blocks[page][self.primary[page]] if b.kind not in ("furniture", "footnote")]

    def _join_tables(self) -> None:
        self.joins: list[dict] = []
        for prev, nxt in zip(self.pages, self.pages[1:]):
            if nxt != prev + 1:
                continue
            body_prev, body_next = self._body(prev), self._body(nxt)
            if not body_prev or not body_next or body_prev[-1].kind != "table":
                continue
            between: list[str] = []
            first = None
            for b in body_next:
                if b.kind == "table":
                    first = b
                    break
                if b.kind in ("caption", "heading") and len(between) < 2:
                    between.append(b.text)
                    continue
                break
            if first is None:
                continue
            a = self.table_at.get((prev, body_prev[-1].order))
            b = self.table_at.get((nxt, first.order))
            if a is None or b is None:
                continue
            head = a
            while head.joined_into:
                head = self.recs[head.joined_into]
            ctx_flags = {"last_on_page": True, "first_on_page": True}  # guaranteed by _body() above
            cont = continuation(head.table, a.bbox, b.table, b.bbox, " ".join(between), **ctx_flags)
            if head.table.n_cols != b.table.n_cols:
                # Engines read column structure differently; try the other engine's version of page N+1.
                for alt_engine, alt in b.alts.items():
                    if alt.n_cols == head.table.n_cols:
                        alt_cont = continuation(head.table, a.bbox, alt, b.bbox, " ".join(between), **ctx_flags)
                        if alt_cont.score > cont.score:
                            b.table, b.engine, cont = copy.deepcopy(alt), alt_engine, alt_cont
                            b.flags.append("version_switched_for_join")
            decision = {"from": a.id, "to": b.id, "pages": [prev, nxt], "rule_score": cont.score,
                        "rule": cont.decision, "reasons": cont.reasons}
            joined, drop = cont.decision == "join", cont.drop_rows
            if cont.decision == "ambiguous":
                verdict = self.agent.decide_continuation(self._evidence(head, b, between, cont)) if self.agent else None
                if verdict:
                    self.m.add_extra("agent_continuation", verdict.get("_cost_inr", 0.0), nxt)
                    self.stats["agent_calls"] += 1
                    joined = bool(verdict["continues"]) and verdict.get("confidence", 0) >= 0.6
                    drop = int(verdict.get("repeated_header_rows", 0)) if joined else 0
                    decision["agent"] = {k: verdict[k] for k in ("continues", "repeated_header_rows", "confidence", "reason")}
                else:
                    decision["agent"] = None
            decision["joined"] = joined
            self.joins.append(decision)
            head.join_decisions.append(decision)
            if joined:
                head.table = join(head.table, b.table, drop)
                head.pages.append(nxt)
                head.fixes += b.fixes
                head.confirmed += b.confirmed
                head.unresolved += b.unresolved
                head.flags = sorted(set(head.flags) | set(b.flags))
                if head.qa is not None and b.qa is not None:  # a joined table is only as good as its worst part
                    rank = {"verified": 0, "repaired": 1, "needs_review": 2}
                    head.qa = {**head.qa, "status": max(head.qa["status"], b.qa["status"], key=rank.get),
                               "parts": head.qa.get("parts", 1) + 1}
                    head.qa_cells += b.qa_cells
                    head.questions += b.questions
                b.joined_into = head.id
                checks = check_table(head.table)  # totals may now span pages
                head.checks_total, head.checks_failed = len(checks), len(failed(checks))

    # -- joining, standard mode ------------------------------------------

    def _first_table(self, page: int) -> tuple[Block | None, list[str], bool]:
        """(first table block that starts the page's body, captions/headings before it, page_has_only_figures)."""
        between: list[str] = []
        body = self._body(page)
        for b in body:
            if b.kind == "table":
                return b, between, False
            if b.kind in ("caption", "heading") and len(between) < 2:
                between.append(b.text)
                continue
            break
        text = sum(len(b.text) for b in body if b.kind in ("text", "contents"))
        has_table = any(b.kind == "table" for b in body)
        return None, between, not has_table and text < 200

    def _join_tables_standard(self) -> None:
        self.joins = []
        order = {p: i for i, p in enumerate(self.pages)}
        for page in self.pages:
            body = self._body(page)
            if not body or body[-1].kind != "table":
                continue
            a = self.table_at.get((page, body[-1].order))
            if a is None:
                continue
            head = a
            while head.joined_into:
                head = self.recs[head.joined_into]
            # the head is the table's first segment (joined segments point to it), so its signature is the table's
            head_sig = table_join.signature(head.table, head.geometry, head.caption)
            for nxt in self.pages[order[page] + 1: order[page] + 3]:
                if nxt - page > 3:
                    break
                first, between, only_figures = self._first_table(nxt)
                if first is None:
                    if only_figures:  # a figure page (or caption only) between two parts of a table
                        continue
                    break
                b = self.table_at.get((nxt, first.order))
                if b is None:
                    break
                b_sig = table_join.signature(b.table, b.geometry, " ".join(between) or b.caption)
                d = table_join.decide(head.table, head_sig, b.table, b_sig, " ".join(between), gap_pages=nxt - page - 1)
                if d.column_map is None:  # one engine dropped or split a column on this page: try the other reading
                    for engine, alt in b.alts.items():
                        if alt.n_cols == head.table.n_cols:
                            alt_sig = table_join.signature(alt, None, " ".join(between) or b.caption)
                            alt_d = table_join.decide(head.table, head_sig, alt, alt_sig, " ".join(between), gap_pages=nxt - page - 1)
                            if alt_d.score > d.score:
                                b.table, b.engine, d = copy.deepcopy(alt), engine, alt_d
                                b.flags.append("version_switched_for_join")
                                d.reasons.append(f"used {engine}'s reading of page {nxt}")
                                break
                record = {"from": a.id, "to": b.id, "pages": [page, nxt], "rule_score": d.score, "rule": d.decision,
                          "reasons": d.reasons, "agent": None}
                joined = d.decision == "join"
                if d.decision == "uncertain" and self.agent:
                    verdict = table_judge.seam(self.agent, self.doc, (a.page, self.info[a.page]["rotate"], a.bbox), a.table,
                                               self.doc, (b.page, self.info[b.page]["rotate"], b.bbox), b.table,
                                               geo_a=a.geometry, geo_b=b.geometry)
                    if verdict:
                        self.m.add_extra("table_seam_check", verdict.get("_cost_inr", 0.0), nxt)
                        record["agent"] = {k: verdict[k] for k in ("continues", "repeated_header_rows", "first_row_continues_previous_row", "reason")}
                        joined = bool(verdict["continues"])
                        d.drop_rows = max(d.drop_rows, int(verdict.get("repeated_header_rows") or 0)) if joined else d.drop_rows
                if joined:
                    before = head.table.n_rows
                    head.table, notes = table_join.merge(head.table, b.table, d)
                    problems = table_join.seam_checks(head.table, before, head_sig)
                    record.update(notes, seam_problems=problems)
                    self._absorb(head, b, nxt)
                    if problems:
                        head.flags = sorted((set(head.flags) - {"repaired"}) | {"seam_check_failed", "needs_review"})
                        if head.qa:
                            head.qa = {**head.qa, "status": "needs_review"}
                elif d.decision == "uncertain":
                    for r in (head, b):
                        r.questions.append({"kind": "join", "priority": 8, "pages": [page, nxt], "score": d.score, "reasons": d.reasons})
                        r.flags = sorted(set(r.flags) | {"join_uncertain"})
                        if r.qa:
                            r.qa = {**r.qa, "status": "needs_review"}
                record["joined"] = joined
                self.joins.append(record)
                head.join_decisions.append(record)
                break

    def _absorb(self, head: TableRec, b: TableRec, page: int) -> None:
        head.pages.append(page)
        head.fixes += b.fixes
        head.confirmed += b.confirmed
        head.unresolved += b.unresolved
        head.flags = sorted(set(head.flags) | set(b.flags))
        if head.qa is not None and b.qa is not None:
            rank = {"verified": 0, "repaired": 1, "needs_review": 2}
            head.qa = {**head.qa, "status": max(head.qa["status"], b.qa["status"], key=rank.get), "parts": head.qa.get("parts", 1) + 1}
            head.qa_cells += b.qa_cells
            head.questions += b.questions
            statuses = {"verified", "repaired", "needs_review"}  # one status flag: the joined table's
            head.flags = sorted({f for f in head.flags if f not in statuses} | ({head.qa["status"]} - {"verified"}))
        b.joined_into = head.id
        checks = check_table(head.table)
        head.checks_total, head.checks_failed = len(checks), len(failed(checks))

    def _evidence(self, head: TableRec, nxt: TableRec, between: list[str], cont: Any) -> dict:
        def rows(t: Table, which: slice) -> list[str]:
            return [" | ".join(clean(x) for x in r) for r in t.rows_text()[which]]

        return {
            "page_n": {"n_cols": head.table.n_cols, "header_rows": rows(head.table, slice(0, head.table.header_rows)),
                       "last_rows": rows(head.table, slice(-3, None)), "bbox_bottom": round(head.bbox[3], 3)},
            "page_n_plus_1": {"n_cols": nxt.table.n_cols, "first_rows": rows(nxt.table, slice(0, 4)),
                              "bbox_top": round(nxt.bbox[1], 3)},
            "text_between": " ".join(between)[:300],
            "rule_score": cont.score,
            "rule_reasons": cont.reasons,
        }

    # -- composition ------------------------------------------------------

    def _figure(self, page: int, block: Block, k: int) -> str:
        name = f"p{page:05d}_{k}.png"
        path = self.out_dir / "figures" / name
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(crop_png(self.doc, page, self.info[page]["rotate"], block.bbox, dpi=150, margin=0.005))
        self.stats["figures"] += 1
        return f"figures/{name}"

    def _compose(self) -> str:
        title = self.pdf_path.stem.replace("-", " ").replace("_", " ")
        out: list[str] = [f"# {title}", ""]
        prev_text_idx: int | None = None  # index in `out` of the previous page's last prose block
        self.printed_numbers: dict[int, str] = {}
        for page in self.pages:
            engine = self.primary[page]
            blocks = self.blocks[page][engine]
            langs = self.page_langs[page][engine]
            mix = ", ".join(f"{k} {round(v * 100)}%" for k, v in langs.shares.items() if v >= 0.05)
            out.append(f"<!-- page {page} ({engine}){' · languages: ' + mix if mix else ''} -->")
            page_lang = langs.dominant
            footnotes: list[str] = []
            first_body = True
            figure_k = 0
            for b in blocks:
                text = b.text.strip()
                if b.lang and page_lang and b.lang != page_lang and text and b.kind not in ("furniture", "figure", "math"):
                    out.append(f"<!-- lang: {b.lang} -->")
                if b.kind == "furniture":
                    if b.raw_label.lower() == "page-number" and text:
                        self.printed_numbers[page] = text
                    continue
                if b.kind == "footnote":
                    if text:
                        footnotes.append(cleanup_text(text))
                    continue
                if b.kind == "table":
                    rec = self.table_at.get((page, b.order))
                    if rec is None:
                        out.append(text)
                    elif rec.joined_into is None:
                        span = f"pages {rec.pages[0]}–{rec.pages[-1]}" if len(rec.pages) > 1 else f"page {page}"
                        out.append(f"<!-- table {rec.id}: {span}, {rec.engine}, checks {rec.checks_total - rec.checks_failed}/{rec.checks_total} passed -->")
                        out.append(rec.table.to_html())
                    first_body = False
                    prev_text_idx = None
                    continue
                if b.kind == "figure":
                    figure_k += 1
                    alt = " ".join(text.split())[:200] or b.raw_label
                    out.append(f"![{alt}]({self._figure(page, b, figure_k)})")
                    first_body = False
                    prev_text_idx = None
                    continue
                if not text:
                    continue
                if b.kind == "heading":
                    out.append(f"{'#' * min(6, (b.level or 2) + 1)} {' '.join(text.split())}")
                    prev_text_idx = None
                elif b.kind == "caption":
                    out.append(f"*{' '.join(text.split())}*")
                    prev_text_idx = None
                elif b.kind == "math":
                    out.append(text if text.startswith("$") else f"$$\n{text}\n$$")
                    prev_text_idx = None
                else:
                    prose = cleanup_text(text)
                    # A paragraph split by a page break, a column break or a drop cap.
                    if prev_text_idx is not None and self._continues(out[prev_text_idx], prose):
                        out[prev_text_idx] = self._join_prose(out[prev_text_idx], prose)
                        key = "paragraphs_joined_across_pages" if first_body else "paragraphs_joined_in_page"
                        self.stats[key] += 1
                    else:
                        out.append(prose)
                    prev_text_idx = len(out) - 1
                first_body = False
            if footnotes:
                out.append("\n".join(f"> {f}" for f in footnotes))
                prev_text_idx = None
            out.append("")
        return "\n\n".join(x for x in out if x is not None).replace("\n\n\n", "\n\n") + "\n"

    @staticmethod
    def _continues(previous: str, current: str) -> bool:
        if not previous or not current:
            return False
        if previous.endswith("-") and current[:1].isalpha():
            return True
        return not previous.endswith(TERMINAL_PUNCT) and current[:1].islower()

    @staticmethod
    def _join_prose(previous: str, current: str) -> str:
        if previous.endswith("-") and current[:1].islower():
            return previous[:-1] + current
        return previous + " " + current

    # -- outputs ----------------------------------------------------------

    def _write_tables(self) -> None:
        tdir = self.out_dir / "tables"
        tdir.mkdir(parents=True, exist_ok=True)
        for rec in self.recs.values():
            if rec.joined_into:
                continue
            (tdir / f"{rec.id}.html").write_text(
                f"<!doctype html><meta charset='utf-8'><title>{rec.id}</title>\n{rec.table.to_html()}\n", encoding="utf-8"
            )
            buf = io.StringIO()
            csv.writer(buf).writerows(rec.table.rows_text())
            (tdir / f"{rec.id}.csv").write_text(buf.getvalue(), encoding="utf-8")
            (tdir / f"{rec.id}.json").write_text(
                json.dumps({**rec.meta(), "rows_text": rec.table.rows_text()}, ensure_ascii=False, indent=1),
                encoding="utf-8",
            )

    def _page_number_gaps(self) -> list[dict]:
        gaps = []
        nums = [(p, self.printed_numbers[p]) for p in self.pages if p in self.printed_numbers]
        for (p1, n1), (p2, n2) in zip(nums, nums[1:]):
            a, b = parse_number(n1), parse_number(n2)
            if a is None or b is None:
                gaps.append({"pages": [p1, p2], "printed": [n1, n2], "issue": "not a number"})
            elif b - a != p2 - p1:
                gaps.append({"pages": [p1, p2], "printed": [n1, n2], "issue": "sequence break"})
        return gaps

    def run(self) -> dict:
        started = time.monotonic()
        self._load()
        spent_before = self.m.spent_inr()
        self.say(f"assembling {len(self.pages)} pages from {self.out_dir.name}")
        self.recs: dict[str, TableRec] = {}
        self.table_at: dict[tuple[int, int], TableRec] = {}
        for page in self.pages:
            primary = self.blocks[page][self.primary[page]]
            for k, block in enumerate(b for b in primary if b.kind == "table"):
                rec = self._build_table(page, block, f"t{page:05d}_{k}")
                if rec is not None:
                    self.recs[rec.id] = rec
                    self.table_at[(page, block.order)] = rec
        self.layer_rules = {}
        self.text_checks: list[text_qa.PageText] = []
        if self.qa_mode == "standard" and self.mode.text_check:
            self._check_text()
        if self.qa_mode == "standard":
            self._repair_layers()
            self._image_check()
            self._join_tables_standard()
        else:
            self._join_tables()
        doc_md = self._compose()
        (self.out_dir / "doc.md").write_text(doc_md, encoding="utf-8")
        self._write_tables()
        result = self._result(time.monotonic() - started, spent_before)
        (self.out_dir / "assembly.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        (self.out_dir / "report.md").write_text(render_report(result), encoding="utf-8")
        self.events.log("assembled", tables=len(self.recs), cost_inr=result["cost"]["assembly_inr"])
        self.m.close()
        self.say(f"wrote {self.out_dir / 'doc.md'} and report.md")
        return result

    def _language_summary(self) -> dict:
        doc_letters: Counter = Counter()
        mixed, devanagari, agree, disagree = [], [], 0, []
        for page in self.pages:
            primary = self.page_langs[page][self.primary[page]]
            for lang, share in primary.shares.items():
                doc_letters[lang] += share * primary.letters
            present = primary.present(0.1)
            if len(present) > 1:
                mixed.append({"page": page, "languages": {k: primary.shares[k] for k in present}})
            if primary.devanagari_confidence is not None:
                deva = next((k for k in primary.shares if k != "en-IN"), None)
                devanagari.append({"page": page, "language": deva, "confidence": primary.devanagari_confidence})
            engines = self.page_langs[page]
            if "sarvam" in engines and "bodhan" in engines:
                a, b = engines["sarvam"].dominant, engines["bodhan"].dominant
                if a == b:
                    agree += 1
                else:
                    disagree.append({"page": page, "sarvam": a, "bodhan": b})
        letters = sum(doc_letters.values()) or 1
        pages = self.m.all_pages()
        return {
            "document": {k: round(v / letters, 3) for k, v in doc_letters.most_common()},
            "pages_with_several_languages": mixed,
            "devanagari_pages": devanagari,
            "engine_agreement": {"same_dominant_language": agree, "different": disagree},
            "ocr_mode": dict(Counter(p.get("ocr_mode") or "unknown" for p in pages if p["page"] in self.info)),
            "mode_source": dict(Counter(p.get("mode_source") or "unknown" for p in pages if p["page"] in self.info)),
            "mode_retries": [p["page"] for p in pages if p.get("mode_retried")],
        }

    def _result(self, wall: float, spent_before: float) -> dict:
        summary_path = self.out_dir / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
        labels: dict[str, Counter] = {}
        kinds: dict[str, Counter] = {}
        for page in self.pages:
            for engine, blocks in self.blocks[page].items():
                labels.setdefault(engine, Counter()).update(b.raw_label for b in blocks)
                kinds.setdefault(engine, Counter()).update(b.kind for b in blocks)
        heads = [r for r in self.recs.values() if not r.joined_into]
        triage = [self.info[p] for p in self.pages]
        fixes = Counter(f["by"].split(":")[0] for r in heads for f in r.fixes)
        confirmed = Counter(f["by"].split(":")[0] for r in heads for f in r.confirmed)
        return {
            "document": str(self.pdf_path),
            "pages": len(self.pages),
            "page_range": [self.pages[0], self.pages[-1]] if self.pages else None,
            "primary_engine": dict(Counter(self.primary.values())),
            "run": {k: summary.get(k) for k in ("wall_s", "pages_per_min", "policy", "language", "engines", "sarvam_jobs")},
            "text_check": text_qa.summarise(self.text_checks) if self.mode.text_check else None,
            "mode": {"name": self.mode.name, "image_check": self.agent is not None,
                     "audits": self.mode.audits,
                     "per_sheet": self.mode.judge.per_sheet, "cell_only": self.mode.judge.cell_only},
            "triage": {
                "rotated_pages": sum(1 for t in triage if t["rotate"]),
                "legacy_font_pages": sum(1 for t in triage if t["legacy_font"]),
                "blank_pages": sum(1 for t in triage if t["blank"]),
                "pages_pymupdf_found_tables": sum(1 for t in triage if t["table_bboxes"]),  # find_tables: a hint only
                "page_classes": dict(Counter(f"{t.get('render')}/{t.get('text_layer')}" for t in triage)),
            },
            "layout": {e: {"kinds": dict(kinds[e]), "labels": dict(labels[e].most_common())} for e in labels},
            "tables": {
                "found": len(self.recs),
                "after_joining": len(heads),
                "joined_across_pages": sum(1 for r in heads if len(r.pages) > 1),
                "checks_total": sum(r.checks_total for r in heads),
                "checks_failed": sum(r.checks_failed for r in heads),
                "cells_fixed": dict(fixes),
                "cells_confirmed": dict(confirmed),
                "unresolved_cells": sum(len(r.unresolved) for r in heads),
                "engine_agreement_mean": round(
                    sum(r.agreement["agreement"] for r in self.recs.values() if r.agreement) /
                    max(1, sum(1 for r in self.recs.values() if r.agreement)), 4),
                "chosen_engine": dict(Counter(r.engine for r in self.recs.values())),
                "qa_status": dict(Counter(r.qa["status"] for r in heads if r.qa)),
                "questions": dict(Counter(q["kind"] for r in heads for q in r.questions)),
                "cells_by": dict(Counter(c["by"].split(":")[0] for r in heads for c in r.qa_cells)),
                "layer_repair": {g: list(rule) for g, rule in getattr(self, "layer_rules", {}).items()},
                "list": [r.meta() for r in heads],
            },
            "joins": self.joins,
            "figures": self.stats["figures"],
            "paragraphs_joined_across_pages": self.stats["paragraphs_joined_across_pages"],
            "paragraphs_joined_in_page": self.stats["paragraphs_joined_in_page"],
            "printed_page_number_issues": self._page_number_gaps(),
            "languages": self._language_summary(),
            "cost": {
                "pages_inr": round(self.m.spent_inr() - sum(v["cost_inr"] for v in self.m.extras_by_kind().values()), 4),
                "assembly_inr": round(self.m.spent_inr() - spent_before, 4),
                "total_inr": round(self.m.spent_inr(), 4),
                "extras": self.m.extras_by_kind(),
            },
            "assembly_wall_s": round(wall, 1),
        }


def render_report(r: dict) -> str:
    t = r["tables"]
    run = r.get("run") or {}
    lines = [
        f"# Report: {Path(r['document']).name}",
        "",
        f"- Pages: {r['pages']} (range {r['page_range']}); primary engine per page: {r['primary_engine']}",
        f"- Run: policy `{run.get('policy')}`, language `{run.get('language')}`, "
        f"wall {run.get('wall_s')} s, {run.get('pages_per_min')} pages/min",
        f"- Cost: ₹{r['cost']['total_inr']} total (this assembly ₹{r['cost']['assembly_inr']}); extras {r['cost']['extras']}",
        "",
        "## Triage (free, before any API call)",
        "",
        *(f"- {k.replace('_', ' ')}: {v}" for k, v in r["triage"].items()),
        "",
        "## Tables",
        "",
        f"- Found {t['found']}; {t['after_joining']} after joining; {t['joined_across_pages']} span more than one page",
        f"- Number checks (hints only in standard mode): {t['checks_total'] - t['checks_failed']}/{t['checks_total']} passed",
        *([f"- Table status: {t['qa_status']}",
           f"- How cells were settled: {t.get('cells_by') or 'none'}",
           f"- Left for review: {t['questions'] or 'nothing'}"] if t.get("qa_status") else [
           f"- Disputed cells: changed {t['cells_fixed'] or 'none'}; confirmed as read {t['cells_confirmed'] or 'none'}; "
           f"unresolved {t['unresolved_cells']}",
           f"- Engine agreement on table numbers (mean): {t['engine_agreement_mean']}"]),
        f"- Version chosen per table: {t['chosen_engine']}",
        "",
        "| table | pages | engine | size | checks passed | text layer | fixes | flags |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for x in t["list"]:
        tl = x["text_layer"]["ratio"] if x["text_layer"] else "–"
        pages = f"{x['pages'][0]}–{x['pages'][-1]}" if len(x["pages"]) > 1 else str(x["page"])
        lines.append(
            f"| {x['id']} | {pages} | {x['engine']} | {x['rows']}×{x['cols']} | "
            f"{x['checks_total'] - x['checks_failed']}/{x['checks_total']} | {tl} | {len(x['fixes'])} | {', '.join(x['flags']) or '–'} |"
        )
    lines += ["", "## Joins across page breaks", ""]
    if not r["joins"]:
        lines.append("- No table ended one page and started the next.")
    for j in r["joins"]:
        agent = j.get("agent")
        who = (f"agent ({agent['confidence']:.2f}): {agent['reason']}" if agent and "confidence" in agent  # legacy join judge
               else f"seam check: {agent['reason']}" if agent else f"rules ({j['rule']})")
        lines.append(f"- pages {j['pages'][0]}→{j['pages'][1]}: {'joined' if j['joined'] else 'kept separate'}; "
                     f"score {j['rule_score']} [{'; '.join(j['reasons'])}]; decided by {who}")
    lang = r.get("languages") or {}
    if lang:
        doc_mix = ", ".join(f"{k} {round(v * 100)}%" for k, v in lang["document"].items() if v >= 0.01)
        low = [d for d in lang["devanagari_pages"] if d["confidence"] < 0.5]
        lines += [
            "", "## Languages", "",
            f"- Document (share of letters): {doc_mix or 'none'}",
            f"- Pages with more than one language (each ≥ 10%): {len(lang['pages_with_several_languages'])}",
            *(f"  - page {m['page']}: " + ", ".join(f"{k} {round(v * 100)}%" for k, v in m['languages'].items())
              for m in lang["pages_with_several_languages"][:30]),
            f"- Devanagari pages told apart by their words (Hindi/Marathi/Nepali/...): {len(lang['devanagari_pages'])}; "
            f"low confidence: {[d['page'] for d in low] or 'none'}",
            f"- Sarvam and Bodhan agree on the main language: {lang['engine_agreement']['same_dominant_language']} pages; "
            f"disagree: {lang['engine_agreement']['different'] or 'none'}",
            f"- OCR mode per page: {lang['ocr_mode']} (decided by {lang['mode_source']}); "
            f"re-read in Indic mode after English mode: {lang['mode_retries'] or 'none'}",
        ]
    tc = r.get("text_check")
    if tc:
        lines += ["", "## Page text (both engines compared, Think mode)", "",
                  f"- Pages compared: {tc['pages_compared']}; skipped: {tc['pages_skipped'] or 'none'}; words {tc['words']}",
                  f"- Differences between the engines: {tc['disputes']}; settled by {tc['by']} (layer proven on {tc['layer_proven_pages']} pages)",
                  f"- Left for review: {sum(1 for _ in tc['open'])}"]
        lines += [f"  - p{d['page']}: …{d['before'][-30:]} **[{d['primary'][:40]} | {d['other'][:40]}]** {d['after'][:25]}" for d in tc["open"][:30]]
    lines += ["", "## Layout labels seen (taxonomy coverage)", ""]
    for engine, info in r["layout"].items():
        labels = ", ".join(f"{k} {v}" for k, v in info["labels"].items())
        lines.append(f"- **{engine}**: {labels}")
    lines += ["", "## Other", "",
              f"- Figures cropped: {r['figures']}",
              f"- Paragraphs joined across page breaks: {r['paragraphs_joined_across_pages']}",
              f"- Printed page-number issues: {r['printed_page_number_issues'] or 'none'}", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.assemble", description="Assemble doc.md, tables and report from a run.")
    p.add_argument("out_dir", type=Path)
    p.add_argument("--no-agent", action="store_true", help="rules only; no OpenRouter/local LLM calls")
    p.add_argument("--no-reocr", action="store_true", help="never re-read failing tables with Bodhan")
    p.add_argument("--max-reocr", type=int, default=10)
    p.add_argument("--max-cost-inr", type=float)
    p.add_argument("--table-qa", choices=("legacy", "standard"), default="standard",
                   help="table checking: standard (default) or legacy")
    p.add_argument("--mode", choices=tuple(modes.MODES), default=modes.DEFAULT_MODE,
                   help="fast (no model calls), normal (default) or think (see src/modes.py)")
    a = p.parse_args(sys.argv[1:] if argv is None else argv)
    if not (a.out_dir / "manifest.sqlite").is_file():
        print(f"error: no run in {a.out_dir}", file=sys.stderr)
        return 2
    result = Assembler(
        a.out_dir, agent=None if a.no_agent else Agent(), reocr=not a.no_reocr,
        max_reocr=a.max_reocr, max_cost_inr=a.max_cost_inr, qa_mode=a.table_qa, mode=a.mode,
    ).run()
    t = result["tables"]
    if t.get("qa_status"):
        print(f"tables {t['found']} -> {t['after_joining']} after joining; status {t['qa_status']}; cells settled by "
              f"{t['cells_by']}; left for review {t['questions'] or 'nothing'}; cost ₹{result['cost']['assembly_inr']}")
    else:
        print(f"tables {t['found']} -> {t['after_joining']} after joining; checks {t['checks_total'] - t['checks_failed']}/"
              f"{t['checks_total']}; fixed {t['cells_fixed']}; unresolved {t['unresolved_cells']}; cost ₹{result['cost']['assembly_inr']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
