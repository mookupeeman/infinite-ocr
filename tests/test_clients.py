import base64
import unittest
from unittest import mock

import pymupdf
from sarvamai.core.api_error import ApiError

from src import bodhan_ocr, sarvam_digitise
from src.rate_limit import FatalError, Limiter, Verdict, call_with_retry
from tests.helpers import OK_PAYLOAD, THROTTLE_BODY, FakeClock, response


def tiny_png() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page(width=120, height=40)
    page.insert_text((10, 25), "test", fontsize=14)
    return page.get_pixmap().tobytes("png")


class FakeSession:
    def __init__(self, replies):
        self.replies = list(replies)
        self.bodies: list[dict] = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.bodies.append(json)
        return self.replies.pop(0)


def free_limiter() -> Limiter:
    clock = FakeClock()
    return Limiter(1000, 60.0, 4, clock=clock, sleep=clock.sleep)


class ClassifyBodhanTest(unittest.TestCase):
    def check(self, resp, verdict, reason=None):
        out = bodhan_ocr.classify_bodhan(resp, None)
        self.assertIs(out.verdict, verdict)
        if reason is not None:
            self.assertEqual(out.reason, reason)
        return out

    def test_cases(self):
        self.check(response(200, OK_PAYLOAD), Verdict.OK)
        # Measured: an undecodable image returns HTTP 200 with body `null`.
        self.check(response(200, "null"), Verdict.DEGRADE, "undecodable")
        out = self.check(response(429, THROTTLE_BODY, {"Retry-After": "42"}), Verdict.RATE_LIMITED)
        self.assertEqual(out.wait_s, 42.0)
        self.check(response(429, {"error": {"message": "Insufficient credit"}}), Verdict.FATAL, "credit_exhausted")
        self.check(response(401, {"error": {"type": "auth_error", "code": "401"}}), Verdict.FATAL, "http_401")
        self.check(response(502, ""), Verdict.RETRY, "http_502")
        self.check(response(502, {"detail": "Block exceeded max_tokens"}), Verdict.DEGRADE, "max_tokens")
        self.check(response(502, {"detail": "image could not be decoded"}), Verdict.DEGRADE, "undecodable")


class BodhanRequestTest(unittest.TestCase):
    def run_with(self, replies):
        session = FakeSession(replies)
        with mock.patch.object(bodhan_ocr, "_session", return_value=session):
            payload = bodhan_ocr.bodhan_request(tiny_png(), "image/png", limiter=free_limiter())
        return payload, session

    def test_null_body_is_resent_as_jpeg(self):
        payload, session = self.run_with([response(200, "null"), response(200, OK_PAYLOAD)])
        self.assertEqual(payload["_infinite"]["degraded"], ["reencoded_jpeg"])
        url = session.bodies[1]["messages"][0]["content"][0]["image_url"]["url"]
        self.assertTrue(url.startswith("data:image/jpeg;base64,"))
        jpeg = base64.b64decode(url.split(",", 1)[1])
        self.assertEqual(jpeg[:2], b"\xff\xd8")

    def test_max_tokens_falls_back_to_markdown(self):
        payload, session = self.run_with(
            [response(502, {"detail": "block exceeded max_tokens"}), response(200, OK_PAYLOAD)]
        )
        self.assertEqual(session.bodies[0]["table_format"], "html")
        self.assertEqual(session.bodies[1]["table_format"], "markdown")
        self.assertEqual(payload["_infinite"]["degraded"], ["markdown_tables"])

    def test_default_table_format_is_html(self):
        _, session = self.run_with([response(200, OK_PAYLOAD)])
        self.assertEqual(session.bodies[0]["table_format"], "html")

    def test_auth_error_is_fatal(self):
        with self.assertRaises(FatalError):
            self.run_with([response(401, {"error": {"type": "auth_error"}})])


class SarvamClassifyTest(unittest.TestCase):
    def test_submit_never_retries_uncertain_errors(self):
        calls = []

        def fn():
            calls.append(1)
            raise ApiError(status_code=500, body="internal error")

        with self.assertRaises(FatalError) as ctx:
            call_with_retry(fn, classify=sarvam_digitise.classify_submit, sleep=lambda s: None)
        self.assertTrue(ctx.exception.outcome.reason.startswith("submit_uncertain"))
        self.assertEqual(len(calls), 1)

    def test_submit_retries_429_with_retry_after(self):
        exc = ApiError(status_code=429, headers={"retry-after": "7"}, body="rate limit")
        out = sarvam_digitise.classify_submit(None, exc)
        self.assertIs(out.verdict, Verdict.RATE_LIMITED)
        self.assertEqual(out.wait_s, 7.0)

    def test_poll_retries_5xx(self):
        from src.rate_limit import classify_http

        out = classify_http(None, ApiError(status_code=503, body="busy"))
        self.assertIs(out.verdict, Verdict.RETRY)

    def test_first_poll_delay(self):
        self.assertEqual(sarvam_digitise.first_poll_delay(2), 5.0)
        self.assertEqual(sarvam_digitise.first_poll_delay(10), 10.0)

    def test_sdk_without_the_digitise_call_is_refused_up_front(self):
        class New:  # sarvamai 0.1.40: digitise became parse
            def parse(self, *, file): ...

        class Old:  # sarvamai 0.1.36
            def digitise(self, *, file, output_format): ...

        for doc_ai, ok in ((New(), False), (Old(), True)):
            with mock.patch.object(sarvam_digitise, "_client", return_value=mock.Mock(doc_ai=doc_ai)):
                if ok:
                    sarvam_digitise.check_sdk()
                else:
                    with self.assertRaisesRegex(RuntimeError, "sarvamai==0.1.36"):
                        sarvam_digitise.check_sdk()


if __name__ == "__main__":
    unittest.main()
