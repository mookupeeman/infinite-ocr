"""Web app: file intake, progress parsing, the cost safety cap, and the model manager's idle state.
No GPU, no model server, no API calls."""

import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pymupdf
from PIL import Image

from tests.fakes import make_pdf
from web import intake
from web.gpu import ModelServer
from web.jobs import PROGRESS, cost_cap


class IntakeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch.object(intake, "UPLOADS", Path(self.tmp.name) / "uploads")
        patcher.start()
        self.addCleanup(patcher.stop)

    def save(self, data: bytes, name: str = "doc.pdf") -> dict:
        return intake.save_upload(io.BytesIO(data), name)

    def pdf_bytes(self, pages: int = 3) -> bytes:
        return make_pdf(Path(self.tmp.name) / "src.pdf", pages).read_bytes()

    def test_a_pdf_is_stored_with_its_pages_and_an_estimate_for_every_mode(self):
        rec = self.save(self.pdf_bytes(3), "My Report (final).pdf")
        self.assertEqual(rec["pages"], 3)
        self.assertEqual(set(rec["estimates"]["modes"]), {"fast", "normal", "think"})
        self.assertTrue(Path(rec["path"]).is_file())
        self.assertEqual(intake.load_record(rec["id"])["name"], "My Report (final).pdf")

    def test_a_photo_becomes_a_one_page_pdf(self):
        for fmt in ("JPEG", "PNG"):
            buf = io.BytesIO()
            Image.new("RGB", (300, 400), "white").save(buf, fmt)
            rec = self.save(buf.getvalue(), "photo")
            with pymupdf.open(rec["path"]) as doc:
                self.assertEqual(doc.page_count, 1, fmt)

    def test_a_blank_page_has_no_ink_and_can_be_estimated(self):
        import numpy as np

        from src.table_geometry import otsu

        for level in (255, 128, 0):
            threshold = otsu(np.full((20, 20), level, dtype=np.uint8))
            self.assertEqual(int((np.full((20, 20), level) <= threshold).sum()), 400 if level == 0 else 0, level)

    def test_a_sideways_photo_is_turned_upright(self):
        img = Image.new("RGB", (400, 200), "white")
        exif = img.getexif()
        exif[0x0112] = 6  # stored rotated: shown turned 90 degrees clockwise
        buf = io.BytesIO()
        img.save(buf, "JPEG", exif=exif)
        rec = self.save(buf.getvalue(), "phone")
        with pymupdf.open(rec["path"]) as doc:
            self.assertGreater(doc[0].rect.height, doc[0].rect.width)

    def test_files_that_cannot_be_read_get_a_plain_message(self):
        encrypted = io.BytesIO()
        with pymupdf.open() as doc:
            doc.new_page()
            doc.save(encrypted, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="owner")
        cases = {
            "not a document": (b"hello world", "not a PDF, JPEG or PNG"),
            "empty": (b"", "empty"),
            "damaged": (b"%PDF-1.4 garbage", "could not be opened"),
            "password": (encrypted.getvalue(), "password"),
        }
        for label, (data, expected) in cases.items():
            with self.subTest(label), self.assertRaises(intake.IntakeError) as caught:
                self.save(data)
            self.assertIn(expected, str(caught.exception))
        self.assertEqual(list((Path(self.tmp.name) / "uploads").glob("*/*")), [])  # nothing left behind

    def test_a_file_id_is_not_a_path(self):
        for bad in ("../../etc/passwd", "x" * 32, "", "A" * 32):
            with self.subTest(bad), self.assertRaises(intake.IntakeError):
                intake.load_record(bad)


class JobPiecesTest(unittest.TestCase):
    def test_progress_line_of_the_pipeline_is_parsed(self):
        line = "[   61s] pages 7/20 | bodhan: done=7 inflight=13 | ₹1.50 | 6.9 pages/min | ETA 2 min"
        m = PROGRESS.match(line)
        self.assertEqual(m.groups(), ("61", "7", "20", "1.50", "6.9", "2 min"))
        self.assertIsNone(PROGRESS.match("Done: 20/20 pages (complete)"))

    def test_cost_cap_sits_above_the_estimate_with_a_floor(self):
        self.assertEqual(cost_cap({"cost_inr": [0.0, 0.0]}), 5.0)
        self.assertEqual(cost_cap({"cost_inr": [20.0, 40.0]}), 60.0)


class FakeProc:
    pid = 4242

    def poll(self):
        return None


class SleepingModelTest(unittest.TestCase):
    """The model is put to sleep after a job and woken for the next, with no GPU and no vLLM."""

    def owned(self, state: str = "ready", after_job: str = "sleep") -> ModelServer:
        server = ModelServer(url="http://127.0.0.1:9/v1", after_job=after_job)
        server._proc, server._state = FakeProc(), state
        return server

    def test_after_a_job_the_model_sleeps_and_the_next_job_wakes_it(self):
        server = self.owned()
        with mock.patch("web.gpu.requests.post") as post, mock.patch("web.gpu.gpu_memory_mb", return_value=(500, 8000)), \
                mock.patch("web.gpu.requests.get") as get, mock.patch.object(ModelServer, "stop") as stop:
            self.assertEqual(server.release(), "sleeping")
            self.assertEqual(server.state, "sleeping")
            self.assertEqual(post.call_args.args[0], "http://127.0.0.1:9/sleep")  # not under /v1
            self.assertEqual(post.call_args.kwargs["params"], {"level": 1})
            get.return_value.json.return_value = {"is_sleeping": False}
            server.start()
            self.assertEqual(post.call_args.args[0], "http://127.0.0.1:9/wake_up")
            self.assertEqual(server.state, "ready")
        stop.assert_not_called()

    def test_if_it_cannot_sleep_the_gpu_is_freed_the_hard_way(self):
        import requests

        server = self.owned()
        with mock.patch("web.gpu.requests.post", side_effect=requests.ConnectionError()), mock.patch.object(ModelServer, "stop") as stop:
            self.assertEqual(server.release(), "stopped")
        stop.assert_called_once()

    def test_if_it_will_not_wake_it_is_stopped_and_started_over(self):
        server = self.owned("sleeping")
        with mock.patch("web.gpu.requests.post"), mock.patch("web.gpu.requests.get") as get, \
                mock.patch.object(ModelServer, "stop") as stop, mock.patch("web.gpu.gpu_memory_mb", return_value=(100, 8000)), \
                mock.patch("web.gpu.subprocess.Popen", side_effect=RuntimeError("cold start attempted")):
            get.return_value.json.return_value = {"is_sleeping": True}  # still asleep after the wake request
            with self.assertRaises(RuntimeError) as caught:
                server.start()
        self.assertIn("cold start", str(caught.exception))
        stop.assert_called_once()

    def test_stop_after_job_setting_and_foreign_servers_are_left_alone(self):
        with mock.patch("web.gpu.requests.post") as post, mock.patch.object(ModelServer, "stop") as stop:
            self.assertEqual(self.owned(after_job="stop").release(), "stopped")
            stop.assert_called_once()
            foreign = ModelServer(url="http://127.0.0.1:9/v1")  # no process of ours
            self.assertEqual(foreign.release(), "kept")
        post.assert_not_called()


class ModelServerTest(unittest.TestCase):
    def test_nothing_is_started_by_asking_for_the_state(self):
        server = ModelServer(url="http://127.0.0.1:9/v1")  # nothing listens here
        self.assertEqual(server.state, "off")
        self.assertFalse(server.owned)
        server.stop()  # a no-op: it never started anything

    def test_a_server_someone_else_started_is_used_and_never_stopped(self):
        server = ModelServer()
        with mock.patch.object(ModelServer, "_serving", return_value=True), mock.patch("web.gpu.subprocess.Popen") as popen:
            server.start()
            server.stop()
        popen.assert_not_called()
        self.assertEqual(server.state, "ready" if server._serving() else "off")


if __name__ == "__main__":
    unittest.main()
