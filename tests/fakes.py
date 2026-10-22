"""Fake engines for engine tests: no network, no cost."""

from __future__ import annotations

import io
import json
import threading
import time
import zipfile
from collections import Counter
from pathlib import Path

import pymupdf

from src.lanes import FATAL_LANE_REASONS
from src.rate_limit import FatalError, Outcome, Verdict


class FakeSarvam:
    """Stands in for src.sarvam_digitise: submit / poll / fetch / first_poll_delay."""

    def __init__(self, *, complete_after_polls: int = 1, missing: set[tuple[int, int]] | None = None,
                 table_every_page: bool = False, delay_s: float = 0.0, fail_submit: str | None = None,
                 page_tables: dict[int, tuple[str, list[float]]] | None = None,
                 body_text: str | None = None) -> None:
        self.complete_after_polls = complete_after_polls
        self.missing = missing or set()  # (job number, position in job) pairs to drop
        self.table_every_page = table_every_page
        self.delay_s = delay_s
        self.fail_submit = fail_submit
        self.page_tables = page_tables or {}  # position in job -> (table html, bbox_norm)
        self.body_text = body_text  # replaces each page's paragraph text (e.g. Hindi)
        self.submissions: list[dict] = []
        self.files: dict[str, list[tuple[str, bytes]]] = {}
        self.polls: Counter = Counter()
        self.lock = threading.Lock()

    def first_poll_delay(self, pages: int) -> float:
        return 0.0

    def submit(self, files, language, on_outcome=None):
        if self.fail_submit:
            raise FatalError(Outcome(Verdict.FATAL, self.fail_submit), 1)
        with self.lock:
            job_id = f"job-{len(self.submissions) + 1}"
            pages = [pymupdf.open("pdf", data).page_count for _, data in files]
            self.submissions.append({"job_id": job_id, "names": [n for n, _ in files], "pages": pages, "language": language})
            self.files[job_id] = files
        return job_id

    def poll(self, job_id, on_outcome=None):
        time.sleep(self.delay_s)
        self.polls[job_id] += 1
        if self.polls[job_id] >= self.complete_after_polls:
            return "completed", {"usage": {"pages_processed": 1}}
        return "running", {"usage": {}}

    def fetch(self, job_id, dest: Path, on_outcome=None):
        job_no = int(job_id.split("-")[1])
        buf = io.BytesIO()
        position = 0
        with zipfile.ZipFile(buf, "w") as z:
            for name, data in self.files[job_id]:
                doc = pymupdf.open("pdf", data)
                for i in range(1, doc.page_count + 1):
                    position += 1
                    if (job_no, position) in self.missing:
                        continue
                    blocks = [
                        {"layout_tag": "section-title", "bbox_norm": [0.1, 0.05, 0.9, 0.1], "reading_order": 1,
                         "confidence": 0.9, "text": f"{name} page {i}"},
                        {"layout_tag": "paragraph", "bbox_norm": [0.1, 0.2, 0.9, 0.5], "reading_order": 2,
                         "confidence": 0.9, "text": self.body_text or doc[i - 1].get_text()[:80]},
                        {"layout_tag": "page-number", "bbox_norm": [0.45, 0.95, 0.55, 0.98], "reading_order": 3,
                         "confidence": 0.9, "text": str(i)},
                    ]
                    if (job_no == 1) and position in self.page_tables:
                        html_, bbox = self.page_tables[position]
                        blocks = [blocks[0], {"layout_tag": "table", "bbox_norm": bbox, "reading_order": 2,
                                              "confidence": 0.9, "text": html_}, blocks[2]]
                    if self.table_every_page:
                        blocks.append({"layout_tag": "table", "bbox_norm": [0.1, 0.5, 0.9, 0.9], "reading_order": 4,
                                       "confidence": 0.9, "text": "<table><tr><td>1</td></tr></table>"})
                    z.writestr(f"{name}/metadata/page_{i:03d}.json",
                               json.dumps({"page_num": i, "image_width": 1000, "image_height": 1400, "blocks": blocks}))
            z.writestr("manifest.json", json.dumps({"status": "completed"}))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(buf.getvalue())
        return dest


class FakeBodhan:
    engine = "bodhan"
    cache_ns = "fake-bodhan"
    model = "fake"
    price_inr = 0.2
    fatal_reasons = FATAL_LANE_REASONS

    def __init__(self, *, delay_s: float = 0.0, fatal_reason: str | None = None) -> None:
        self.delay_s = delay_s
        self.fatal_reason = fatal_reason
        self.calls = 0
        self.lock = threading.Lock()

    def cache_extra(self) -> str:
        return "x"

    def ocr(self, image, mime, on_outcome):
        with self.lock:
            self.calls += 1
        if self.fatal_reason:
            raise FatalError(Outcome(Verdict.FATAL, self.fatal_reason), 1)
        time.sleep(self.delay_s)
        return {
            "choices": [{"message": {"content": "hello"}}],
            "blocks": [
                {"order": 0, "label": "Header", "bbox_xyxy": [0, 0, 100, 20], "conf": 0.9, "text": ""},
                {"order": 1, "label": "Folio", "bbox_xyxy": [0, 0, 100, 21], "conf": 0.9, "text": "RUNNING HEAD"},
                {"order": 2, "label": "Section-title", "bbox_xyxy": [10, 30, 90, 40], "conf": 0.9, "text": "Title"},
                {"order": 3, "label": "Paragraph", "bbox_xyxy": [10, 50, 90, 90], "conf": 0.9, "text": "hello"},
            ],
            "usage": {"total_tokens": 10},
            "_infinite": {"degraded": []},
        }

    def to_blocks(self, payload, page, width, height):
        from src.blocks import from_bodhan

        return from_bodhan(payload, page, width, height)

    def cost_inr(self, payload):
        return self.price_inr


def make_pdf(path: Path, pages: int) -> Path:
    doc = pymupdf.open()
    for i in range(pages):
        page = doc.new_page(width=300, height=400)
        page.insert_text((40, 80), f"Synthetic page {i + 1}", fontsize=14)
    doc.save(path)
    return path
