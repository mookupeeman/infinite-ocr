"""Run one benchmark case through the real stage 1 and stage 2 with simulated engines.

The simulated engines return the planted readings of `bench.engines`; everything else
(triage, manifest, lanes, assembly, joining) is the project's own code. Nothing here
calls a paid API. The checker under test is chosen by `checker`:

    "baseline"   today's stage 2 with no agent and no Bodhan re-read (rules only)
"""

from __future__ import annotations

import hashlib
import io
import json
import random
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import pymupdf

from bench import engines, metrics
from bench.gen import Case
from bench.render import Rendered
from src import cache
from src.assemble import Assembler
from src.lanes import Settings
from src.pdf_pages import RenderSpec, render_page
from src.run import Run
from src.tables import Cell, Table, parse_tables
from tests.fakes import FakeBodhan, FakeSarvam

LANG_CODE = {"hi": "hi-IN", "mr": "mr-IN", "ta": "ta-IN", "te": "te-IN", "bn": "bn-IN", "en": "en-IN"}


@dataclass
class PageReading:
    caption: str | None
    html_a: str | None
    html_b: str | None
    bbox: list[float] | None
    kind: str
    note: str | None = None


def _jitter(rng: random.Random, bbox: list[float]) -> list[float]:
    return [round(min(1.0, max(0.0, v + rng.uniform(-0.004, 0.004))), 4) for v in bbox]


class SimSarvam(FakeSarvam):
    """Answers each chunk page with the reading of the original page the manifest says it holds
    (blank pages never reach Sarvam, so positions in a chunk are not page numbers)."""

    def __init__(self, readings: dict[int, PageReading], rng: random.Random) -> None:
        super().__init__()
        self.readings, self.rng = readings, rng
        self.run: Run | None = None  # set once the run is prepared

    def _blocks(self, page: int) -> list[dict]:
        rd = self.readings[page]
        blocks: list[dict] = []

        def add(tag: str, bbox: list[float], text: str) -> None:
            blocks.append({"layout_tag": tag, "bbox_norm": bbox, "reading_order": len(blocks) + 1, "confidence": 0.95, "text": text})

        if rd.caption:
            add("section-title", [0.07, 0.04, 0.5, 0.075], rd.caption)
        if rd.kind == "table":
            add("table", _jitter(self.rng, rd.bbox), rd.html_a)
        elif rd.kind == "figure":
            add("image", [0.07, 0.07, 0.93, 0.3], "")
        if rd.note:
            add("paragraph", [0.07, rd.bbox[3] + 0.015, 0.93, rd.bbox[3] + 0.06], rd.note)
        add("page-number", [0.47, 0.95, 0.53, 0.97], str(page))
        return blocks

    def fetch(self, job_id, dest: Path, on_outcome=None):
        job = next(j for j in self.run.ctx.manifest.jobs() if j["job_id"] == job_id)
        plan = json.loads(job["files"])  # per file: the original page numbers, in chunk order
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            for (name, _), pages in zip(self.files[job_id], plan):
                for position, page in enumerate(pages, start=1):
                    z.writestr(f"{name}/metadata/page_{position:03d}.json",
                               json.dumps({"page_num": position, "image_width": 1000, "image_height": 1400,
                                           "blocks": self._blocks(page)}))
            z.writestr("manifest.json", json.dumps({"status": "completed"}))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(buf.getvalue())
        return dest


class SimBodhan(FakeBodhan):
    """Finds the page from the rendered image the lane sends (same render call, same bytes)."""

    def __init__(self, pdf: Path, readings: dict[int, PageReading], rng: random.Random, dpi: int = 200) -> None:
        super().__init__()
        self.rng = rng
        self.by_hash: dict[str, tuple[int, int, int]] = {}
        doc = pymupdf.open(pdf)
        self.readings = readings
        for page in range(1, doc.page_count + 1):
            img = render_page(doc, page, RenderSpec(dpi=dpi, rotate=0))
            self.by_hash[hashlib.sha256(img.data).hexdigest()] = (page, img.width, img.height)

    def ocr(self, image, mime, on_outcome):
        page, w, h = self.by_hash[hashlib.sha256(image).hexdigest()]
        rd = self.readings[page]
        blocks, order = [], 0

        def add(label: str, bbox: list[float], text: str) -> None:
            nonlocal order
            order += 1
            blocks.append({"order": order, "label": label, "bbox_xyxy": [bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h],
                           "conf": 0.97, "text": text})

        if rd.caption:
            add("Table-caption", [0.07, 0.04, 0.5, 0.075], rd.caption)
        if rd.kind == "table":
            add("Table", _jitter(self.rng, rd.bbox), rd.html_b)
        elif rd.kind == "figure":
            add("Image", [0.07, 0.07, 0.93, 0.3], "")
        if rd.note:
            add("Paragraph", [0.07, rd.bbox[3] + 0.015, 0.93, rd.bbox[3] + 0.06], rd.note)
        add("Page-number", [0.47, 0.95, 0.53, 0.97], str(page))
        return {"choices": [{"message": {"content": ""}}], "blocks": blocks, "usage": {"total_tokens": 10},
                "_infinite": {"degraded": []}}


# ---------------------------------------------------------------------------


def stack(tables: list[Table]) -> Table | None:
    """Final tables of one logical table, one under the other (what a reader of doc.md sees)."""
    if not tables:
        return None
    cells, offset = [], 0
    for t in tables:
        cells += [Cell(c.r + offset, c.c, c.rowspan, c.colspan, c.text, c.header) for c in t.cells]
        offset += t.n_rows
    return Table(cells, offset, max(t.n_cols for t in tables), tables[0].header_rows)


def readings_for(case: Case, rendered: Rendered, rng: random.Random, profile: engines.Profile):
    readings: dict[int, PageReading] = {}
    sims: dict[int, engines.Reading] = {}
    for i, spec in enumerate(case.pages, start=1):
        if spec.kind != "table":
            readings[i] = PageReading(spec.caption, None, None, None, spec.kind)
            continue
        sim = engines.simulate(spec.table, rng, profile)
        sims[i] = sim
        readings[i] = PageReading(
            spec.caption, engines.to_dialect(sim.a, "flat").to_html(), engines.to_dialect(sim.b, "rowspan").to_html(),
            rendered.bboxes[i], "table", spec.note,
        )
    return readings, sims


def run_case(case: Case, rendered: Rendered, *, scanned: bool, profile: engines.Profile, seed: int,
             checker: str = "baseline") -> dict:
    rng = random.Random(seed)
    pdf = rendered.scan_pdf if scanned else rendered.pdf
    readings, sims = readings_for(case, rendered, rng, profile)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with mock.patch.object(cache, "CACHE_DIR", root / "cache"):
            sarvam = SimSarvam(readings, rng)
            run = Run(pdf, Settings(language=LANG_CODE[case.lang], policy="both"), out_root=root / "out",
                      sarvam=sarvam, bodhan_backend=SimBodhan(pdf, readings, rng),
                      progress_every=999, verbose=False)
            run.prepare()
            sarvam.run = run
            summary = run.execute()
            out = Path(summary["out_dir"])
            result = Assembler(out, agent=None, reocr=False, verbose=False,
                               qa_mode="standard" if checker == "standard" else "legacy").run()
            finals = []
            for meta in result["tables"]["list"]:
                html_file = out / "tables" / f"{meta['id']}.html"
                parsed = parse_tables(html_file.read_text(encoding="utf-8"))
                finals.append((meta, parsed[0] if parsed else None))
    return score_case(case, sims, finals, result)


def accepted_by_system(meta: dict) -> bool:
    """Standard mode: the table's status claims it is right. Legacy: nothing was flagged or left unresolved."""
    if meta.get("qa"):
        return meta["qa"]["status"] in ("verified", "repaired")
    return not meta["flags"] and not meta["unresolved"]


def worst_status(metas) -> str | None:
    rank = {"verified": 0, "repaired": 1, "needs_review": 2}
    found = [m["qa"]["status"] for m in metas if m.get("qa")]
    return max(found, key=rank.get) if found else None


def score_case(case: Case, sims: dict[int, engines.Reading], finals: list, result: dict) -> dict:
    table_pages = {i + 1: p for i, p in enumerate(case.pages) if p.kind == "table"}
    tables_out, cells = [], []
    for li, truth in enumerate(case.truths):
        pages = [i for i, p in table_pages.items() if p.logical == li]
        mine = [(m, t) for m, t in finals if m["pages"][0] in pages and t is not None]
        pred = stack([t for _, t in sorted(mine, key=lambda x: x[0]["pages"][0])])
        score = metrics.score_table(truth, pred)
        ok_final = score.pop("_ok", {})
        accepted = bool(mine) and all(accepted_by_system(m) for m, _ in mine)
        questions = [q for m, _ in mine for q in m.get("questions") or []]
        entry = {"logical": li, "pages": pages, "final_tables": len(mine), "accepted": accepted,
                 "status": worst_status(m for m, _ in mine),
                 "questions": {k: sum(q["kind"] == k for q in questions) for k in ("cell", "audit", "structure")}, **score}
        # who was right, cell by cell (single-page logical tables only)
        if len(pages) == 1 and pages[0] in sims:
            sa = metrics.score_table(truth, sims[pages[0]].a)
            sb = metrics.score_table(truth, sims[pages[0]].b)
            oa, ob = sa.pop("_ok"), sb.pop("_ok")
            entry["engine_a_exact"], entry["engine_b_exact"] = sa["exact"], sb["exact"]
            truth_text = {(c.r, c.c): metrics.norm(c.text) for c in truth.cells}
            for key, (okf, gotf) in ok_final.items():
                entry_c = {"a": oa[key][0], "b": ob[key][0], "same": oa[key][1] == ob[key][1], "final": okf}
                if not (entry_c["a"] and entry_c["b"] and okf):  # keep the detail of every interesting cell
                    entry_c.update(key=list(key), truth=truth_text[key], got_a=oa[key][1], got_b=ob[key][1], got_final=gotf,
                                   header=key[0] < truth.header_rows)
                cells.append(entry_c)
        tables_out.append(entry)
    final_pages = [m["pages"] for m, _ in finals]
    planted = [p.to_dict() for sim in sims.values() for p in sim.planted]
    return {"case": case.meta(), "tables": tables_out, "joins": metrics.score_joins(case, final_pages), "cells": cells,
            "planted": planted, "flags": [m["flags"] for m, _ in finals], "cost_inr": result["cost"]["assembly_inr"]}
