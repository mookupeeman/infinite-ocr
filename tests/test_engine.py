"""End-to-end engine runs with fake engines and an isolated cache (no cost)."""

import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from src import cache
from src.lanes import Settings
from src.manifest import EventLog, Manifest
from src.run import Run
from tests.fakes import FakeBodhan, FakeSarvam, make_pdf

ROOT = Path(__file__).resolve().parent.parent
MARATHI = ROOT / "samples" / "marathi-fiscal-report.pdf"
PATRICK = ROOT / "samples" / "Patrick-Deneen-sample.pdf"


class EngineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        patcher = mock.patch.object(cache, "CACHE_DIR", self.root / "cache")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)

    def make_run(self, pdf, settings, **kw) -> Run:
        run = Run(pdf, settings, out_root=self.root / "out", progress_every=999, verbose=False, **kw)
        run.prepare()
        return run

    def states(self, summary, engine):
        return summary["engines"].get(engine, {}).get("states", {})

    def test_full_run_then_free_replay_from_cache(self):
        sarvam, bodhan = FakeSarvam(), FakeBodhan()
        s = Settings(language="mr-IN", chunk_pages=3, files_per_job=5)
        summary = self.make_run(MARATHI, s, sarvam=sarvam, bodhan_backend=bodhan).execute()
        self.assertTrue(summary["complete"], summary)
        self.assertEqual(self.states(summary, "sarvam"), {"done": 8})
        self.assertEqual(self.states(summary, "bodhan"), {"done": 8})  # every page has a table
        self.assertEqual(len(sarvam.submissions), 1)
        self.assertEqual(sarvam.submissions[0]["pages"], [3, 3, 2])
        self.assertAlmostEqual(summary["cost_inr"], 8 * 0.5 + 8 * 0.2)
        out = Path(summary["out_dir"])
        self.assertTrue((out / "blocks" / "page_00002.sarvam.json").is_file())
        self.assertTrue((out / "md" / "page_00002.bodhan.md").read_text().startswith("## Title"))

        sarvam2, bodhan2 = FakeSarvam(), FakeBodhan()
        replay = self.make_run(MARATHI, s, sarvam=sarvam2, bodhan_backend=bodhan2, fresh=True).execute()
        self.assertTrue(replay["complete"])
        self.assertEqual(replay["cost_inr"], 0)
        self.assertEqual((sarvam2.submissions, bodhan2.calls), ([], 0))
        self.assertEqual(replay["engines"]["sarvam"]["cache_hits"], 8)
        self.assertEqual(replay["engines"]["bodhan"]["cache_hits"], 8)

    def test_missing_page_is_retried_in_a_new_job(self):
        sarvam = FakeSarvam(missing={(1, 2)})
        summary = self.make_run(PATRICK, Settings(language="en-IN"), sarvam=sarvam, bodhan_backend=FakeBodhan()).execute()
        self.assertTrue(summary["complete"])
        self.assertEqual(self.states(summary, "sarvam"), {"done": 8})
        self.assertEqual(len(sarvam.submissions), 2)
        self.assertEqual(sarvam.submissions[1]["pages"], [1])

    def test_page_missing_twice_falls_back_to_bodhan(self):
        sarvam = FakeSarvam(missing={(1, 2), (2, 1)})
        summary = self.make_run(PATRICK, Settings(language="en-IN"), sarvam=sarvam, bodhan_backend=FakeBodhan()).execute()
        self.assertTrue(summary["complete"])
        self.assertEqual(self.states(summary, "sarvam"), {"done": 7, "failed": 1})
        self.assertEqual(self.states(summary, "bodhan"), {"done": 1})

    def test_hard_stop_then_resume_polls_the_same_job(self):
        sarvam = FakeSarvam(complete_after_polls=10**9)
        s = Settings(language="en-IN")
        run = self.make_run(PATRICK, s, sarvam=sarvam, bodhan_backend=FakeBodhan())
        result: dict = {}
        t = threading.Thread(target=lambda: result.update(run.execute()))
        t.start()
        deadline = time.time() + 20
        while not sarvam.submissions and time.time() < deadline:
            time.sleep(0.05)
        time.sleep(0.3)
        run.ctx.hard_stop.set()
        t.join(30)
        self.assertFalse(result["complete"])
        self.assertEqual(result["stopped"], "hard_stop")

        sarvam.complete_after_polls = 1
        summary = self.make_run(PATRICK, s, sarvam=sarvam, bodhan_backend=FakeBodhan(), resume=True).execute()
        self.assertTrue(summary["complete"])
        self.assertEqual(len(sarvam.submissions), 1)  # polled again, never resubmitted
        self.assertEqual(summary["sarvam_jobs"]["total"], 1)
        self.assertIn("job_resumed", summary["events"])

    def test_crash_while_submitting_resubmits_and_logs(self):
        s = Settings(language="en-IN")
        first = self.make_run(PATRICK, s, sarvam=FakeSarvam(), bodhan_backend=FakeBodhan())
        m = first.ctx.manifest
        for p in range(1, 9):
            m.add_page({"page": p, "rotate": 0, "language": "en-IN"}, {"sarvam": ("pending", 0)})
        _, pages = m.claim_sarvam(10)
        m.new_job([pages], "en-IN")  # recorded, then "crashed" before submit returned
        m.close()

        sarvam = FakeSarvam()
        summary = self.make_run(PATRICK, s, sarvam=sarvam, bodhan_backend=FakeBodhan(), resume=True).execute()
        self.assertTrue(summary["complete"])
        self.assertEqual(len(sarvam.submissions), 1)
        self.assertEqual(summary["sarvam_jobs"]["by_state"], {"abandoned": 1, "done": 1})
        events = EventLog(Path(summary["out_dir"]) / "events.jsonl").read()
        self.assertTrue(any(e["kind"] == "job_abandoned" and e["possible_duplicate"] for e in events))

    def test_budget_cap_stops_cleanly(self):
        s = Settings(language="en-IN", policy="bodhan-only", max_cost_inr=1.0)
        summary = self.make_run(PATRICK, s, bodhan_backend=FakeBodhan(delay_s=0.02)).execute()
        self.assertEqual(summary["stopped"], "budget")
        self.assertLessEqual(summary["cost_inr"], 1.0 + 1e-9)
        self.assertEqual(self.states(summary, "bodhan").get("done"), 5)

    def test_bodhan_steals_from_sarvam_backlog_without_double_work(self):
        pdf = make_pdf(self.root / "thirty.pdf", 30)
        sarvam = FakeSarvam(delay_s=0.3)
        bodhan = FakeBodhan(delay_s=0.05)
        s = Settings(language="en-IN", chunk_pages=2, files_per_job=1, max_open_jobs=1)
        summary = self.make_run(pdf, s, sarvam=sarvam, bodhan_backend=bodhan).execute()
        self.assertTrue(summary["complete"], summary)
        sv, bd = self.states(summary, "sarvam"), self.states(summary, "bodhan")
        self.assertGreater(bd.get("done", 0), 0)
        self.assertEqual(sv.get("done", 0) + bd.get("done", 0), 30)
        self.assertEqual(sv.get("stolen", 0), bd.get("done", 0))
        self.assertEqual(summary["pages_done"], 30)

    def test_fatal_bodhan_error_stops_only_that_lane(self):
        summary = self.make_run(
            MARATHI, Settings(language="mr-IN"), sarvam=FakeSarvam(), bodhan_backend=FakeBodhan(fatal_reason="credit_exhausted")
        ).execute()
        self.assertTrue(summary["complete"])
        self.assertEqual(self.states(summary, "sarvam"), {"done": 8})
        bodhan = self.states(summary, "bodhan")
        # Pages triaged after the lane died get no Bodhan task at all; the rest fail.
        self.assertEqual(set(bodhan), {"failed"})
        self.assertTrue(1 <= bodhan["failed"] <= 8)
        self.assertIn("lane_dead", summary["events"])

    # -- Bodhan leads (local vLLM) ----------------------------------------

    def test_bodhan_lead_reads_every_page_and_sarvam_only_the_table_pages(self):
        sarvam, bodhan = FakeSarvam(), FakeBodhan()
        pdf = make_pdf(self.root / "plain.pdf", 4)  # no tables: Sarvam has nothing to read
        s = Settings(language="en-IN", policy="bodhan-primary")
        summary = self.make_run(pdf, s, sarvam=sarvam, bodhan_backend=bodhan).execute()
        self.assertTrue(summary["complete"], summary)
        self.assertEqual(self.states(summary, "bodhan"), {"done": 4})
        self.assertEqual(self.states(summary, "sarvam"), {})
        self.assertEqual(sarvam.submissions, [])

        tables = self.make_run(MARATHI, Settings(language="mr-IN", policy="bodhan-primary"),
                               sarvam=FakeSarvam(), bodhan_backend=FakeBodhan()).execute()
        self.assertEqual(self.states(tables, "bodhan"), {"done": 8})
        self.assertEqual(self.states(tables, "sarvam"), {"done": 8})  # every page has a ruled table

    def test_bodhan_lead_sends_a_table_triage_missed_to_sarvam(self):
        pdf = make_pdf(self.root / "plain.pdf", 3)
        summary = self.make_run(pdf, Settings(language="en-IN", policy="bodhan-primary"),
                                sarvam=FakeSarvam(), bodhan_backend=FakeBodhan(table=True)).execute()
        self.assertTrue(summary["complete"], summary)
        self.assertEqual(self.states(summary, "sarvam"), {"done": 3})
        self.assertEqual(summary["events"].get("verify_table"), 3)

    def test_bodhan_lead_falls_back_to_sarvam_when_bodhan_is_down(self):
        pdf = make_pdf(self.root / "plain.pdf", 5)
        summary = self.make_run(pdf, Settings(language="en-IN", policy="bodhan-primary"), sarvam=FakeSarvam(),
                                bodhan_backend=FakeBodhan(fatal_reason="missing_runtime")).execute()
        self.assertTrue(summary["complete"], summary)
        self.assertEqual(summary["pages_done"], 5)
        self.assertEqual(self.states(summary, "sarvam"), {"done": 5})

    def test_assembly_takes_the_lead_engines_reading(self):
        from src.assemble import Assembler

        pdf = make_pdf(self.root / "plain.pdf", 2)
        for lead, first in (("sarvam", "sarvam"), ("bodhan", "bodhan")):
            summary = self.make_run(pdf, Settings(language="en-IN", policy="both", lead=lead), sarvam=FakeSarvam(),
                                    bodhan_backend=FakeBodhan(), fresh=True).execute()
            a = Assembler(Path(summary["out_dir"]), verbose=False, mode="fast")
            a._load()
            self.assertEqual(set(a.primary.values()), {first}, lead)
            a.m.close()

    def test_lead_is_recorded_only_when_bodhan_leads(self):
        self.assertNotIn("lead", Settings(language="auto").fingerprint())
        self.assertEqual(Settings(language="auto", policy="bodhan-primary").lead, "bodhan")
        self.assertEqual(Settings(language="auto", policy="both", lead="bodhan").fingerprint()["lead"], "bodhan")
        self.assertEqual(Settings(language="auto", policy="sarvam-primary", lead="bodhan").lead, "sarvam")


if __name__ == "__main__":
    unittest.main()
