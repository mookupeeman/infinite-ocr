import tempfile
import threading
import unittest
from pathlib import Path

from src.blocks import BODHAN_KIND, KINDS, SARVAM_KIND, from_bodhan, from_sarvam, page_markdown
from src.manifest import Manifest

DOCUMENTED_BODHAN_LABELS = """Advertisement Answer Author Chapter-end-section Chapter-title Chart Code
Contact-info Dateline Diagram Equation Expression Flag Folio Footer Footnote Header Image Image-caption
Index Infobox List MCQ Page-number Paragraph Placeholder-text Question Reference Section-title
Solved-example Sub-section-title Sub-sub-section-title Table Table-caption Table-of-contents Title
Website-link""".split()
SEEN_SARVAM_TAGS = ["header", "section-title", "paragraph", "page-number", "image", "folio",
                    "footnote", "image-caption", "table"]


class BlocksTest(unittest.TestCase):
    def test_every_documented_bodhan_label_is_mapped(self):
        self.assertEqual(len(DOCUMENTED_BODHAN_LABELS), 37)
        self.assertEqual(set(BODHAN_KIND), set(DOCUMENTED_BODHAN_LABELS))
        self.assertTrue(all(kind in KINDS for kind, _ in BODHAN_KIND.values()))

    def test_every_seen_sarvam_tag_is_mapped(self):
        for tag in SEEN_SARVAM_TAGS:
            self.assertIn(tag, SARVAM_KIND)

    def test_bodhan_normalises_and_merges_furniture(self):
        payload = {"blocks": [
            {"order": 0, "label": "Header", "bbox_xyxy": [0, 0, 200, 40], "conf": 0.9, "text": ""},
            {"order": 1, "label": "Folio", "bbox_xyxy": [0, 0, 200, 42], "conf": 0.9, "text": "HEAD"},
            {"order": 2, "label": "Chapter-title", "bbox_xyxy": [20, 60, 180, 80], "conf": 0.9, "text": "One"},
            {"order": 3, "label": "Table", "bbox_xyxy": [20, 100, 180, 380], "conf": 0.9, "text": "<table></table>"},
        ]}
        blocks = from_bodhan(payload, 3, 200, 400)
        self.assertEqual([b.raw_label for b in blocks], ["Folio", "Chapter-title", "Table"])
        self.assertEqual(blocks[2].bbox, (0.1, 0.25, 0.9, 0.95))
        md = page_markdown(blocks)
        self.assertTrue(md.startswith("# One"))
        self.assertNotIn("HEAD", md)

    def test_sarvam_uses_bbox_norm_and_reading_order(self):
        page_json = {"image_width": 1000, "image_height": 2000, "blocks": [
            {"layout_tag": "paragraph", "bbox_norm": [0.1, 0.5, 0.9, 0.6], "reading_order": 2, "text": "body"},
            {"layout_tag": "section-title", "coordinates": {"x1": 100, "y1": 100, "x2": 900, "y2": 200},
             "reading_order": 1, "text": "Heading"},
        ]}
        blocks = from_sarvam(page_json, 1)
        self.assertEqual([b.kind for b in blocks], ["heading", "text"])
        self.assertEqual(blocks[0].bbox, (0.1, 0.05, 0.9, 0.1))


class ManifestTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.m = Manifest(Path(self.tmp.name) / "m.sqlite")

    def tearDown(self):
        self.m.close()
        self.tmp.cleanup()

    def add(self, page, tasks, language="mr-IN"):
        self.m.add_page({"page": page, "rotate": 0, "language": language}, tasks)

    def test_threads_never_claim_the_same_page(self):
        for p in range(1, 101):
            self.add(p, {"bodhan": ("pending", 0)})
        claimed: list[int] = []
        lock = threading.Lock()

        def worker():
            while True:
                c = self.m.claim_page_task("bodhan")
                if c is None:
                    return
                with lock:
                    claimed.append(c[0])

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(sorted(claimed), list(range(1, 101)))

    def test_priority_then_steal_from_tail(self):
        for p in range(1, 11):
            self.add(p, {"sarvam": ("pending", 0)})
        self.m.ensure_task(4, "bodhan", priority=1)
        self.assertEqual(self.m.claim_page_task("bodhan", steal_from="sarvam", steal_min_backlog=5), (4, False))
        self.assertEqual(self.m.claim_page_task("bodhan", steal_from="sarvam", steal_min_backlog=5), (10, True))
        self.assertEqual(self.m.task(10, "sarvam")["state"], "stolen")
        # 9 Sarvam pages still pending: a threshold of 9 forbids another steal
        self.assertIsNone(self.m.claim_page_task("bodhan", steal_from="sarvam", steal_min_backlog=9))
        self.assertTrue(self.m.reopen_stolen(10))
        self.assertEqual(self.m.task(10, "sarvam")["state"], "pending")

    def test_sarvam_claims_one_language_in_page_order(self):
        self.add(1, {"sarvam": ("pending", 0)}, "mr-IN")
        self.add(2, {"sarvam": ("pending", 0)}, "en-IN")
        self.add(3, {"sarvam": ("pending", 0)}, "mr-IN")
        self.assertEqual(self.m.claim_sarvam(10), ("mr-IN", [1, 3]))
        self.assertEqual(self.m.claim_sarvam(10), ("en-IN", [2]))

    def test_retry_then_fail_and_requeue_inflight(self):
        self.add(1, {"bodhan": ("pending", 0)})
        self.m.claim_page_task("bodhan")
        self.assertEqual(self.m.requeue_inflight(), 1)
        self.assertEqual(self.m.retry_or_fail(1, "bodhan", "x", 2), "pending")
        self.assertEqual(self.m.retry_or_fail(1, "bodhan", "x", 2), "failed")

    def test_job_lifecycle(self):
        for p in (1, 2, 3):
            self.add(p, {"sarvam": ("pending", 0)})
        _, pages = self.m.claim_sarvam(10)
        job = self.m.new_job([pages[:2], pages[2:]], "mr-IN")
        self.assertEqual(self.m.open_jobs()[0]["state"], "submitting")
        self.m.job_submitted(job, "abc")
        self.assertEqual(self.m.queued("sarvam"), {1: job, 2: job, 3: job})
        self.m.job_finished(job, "done", status="completed")
        self.assertEqual(self.m.open_jobs(), [])


if __name__ == "__main__":
    unittest.main()
