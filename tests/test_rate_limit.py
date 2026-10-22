import threading
import time
import unittest

from src.rate_limit import (
    FatalError,
    Limiter,
    RetriesExhausted,
    Verdict,
    call_with_retry,
    classify_status,
)
from tests.helpers import THROTTLE_BODY, FakeClock, response


def max_in_window(times: list[float], window: float) -> int:
    best = 0
    for i, t in enumerate(times):
        best = max(best, sum(1 for u in times[i:] if u - t < window))
    return best


class LimiterTest(unittest.TestCase):
    def test_bodhan_settings_never_exceed_8_per_62s(self):
        clock = FakeClock()
        lim = Limiter(8, 62.0, 4, min_spacing_s=60 / 8 + 0.5, clock=clock, sleep=clock.sleep)
        times = [lim.acquire_start() for _ in range(40)]
        self.assertLessEqual(max_in_window(times, 62.0), 8)
        gaps = [b - a for a, b in zip(times, times[1:])]
        self.assertGreaterEqual(min(gaps), 8.0 - 1e-9)

    def test_window_without_spacing_allows_burst_then_waits(self):
        clock = FakeClock()
        lim = Limiter(8, 62.0, 8, clock=clock, sleep=clock.sleep)
        times = [lim.acquire_start() for _ in range(9)]
        self.assertEqual(times[:8], [0.0] * 8)
        self.assertAlmostEqual(times[8], 62.0)

    def test_pause_delays_next_start(self):
        clock = FakeClock()
        lim = Limiter(100, 60.0, 4, clock=clock, sleep=clock.sleep)
        lim.acquire_start()
        lim.pause_until(42.0)
        self.assertGreaterEqual(lim.acquire_start(), 42.0)

    def test_inflight_cap_with_threads(self):
        lim = Limiter(1000, 1.0, 2)
        active = 0
        peak = 0
        lock = threading.Lock()

        def work():
            nonlocal active, peak
            with lim.slot():
                with lock:
                    active += 1
                    peak = max(peak, active)
                time.sleep(0.05)
                with lock:
                    active -= 1

        threads = [threading.Thread(target=work) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(peak, 2)

    def test_pause_blocks_every_thread(self):
        lim = Limiter(1000, 1.0, 4)
        lim.pause_for(0.3)
        until = time.monotonic() + 0.29
        starts: list[float] = []
        lock = threading.Lock()

        def work():
            with lim.slot():
                with lock:
                    starts.append(time.monotonic())

        threads = [threading.Thread(target=work) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(starts), 3)
        self.assertTrue(all(s >= until for s in starts))


class ClassifyStatusTest(unittest.TestCase):
    def test_codes(self):
        self.assertIs(classify_status(200).verdict, Verdict.OK)
        rl = classify_status(429, str(THROTTLE_BODY), 42.0)
        self.assertIs(rl.verdict, Verdict.RATE_LIMITED)
        self.assertEqual(rl.wait_s, 42.0)
        credit = classify_status(429, "Insufficient credit balance")
        self.assertEqual((credit.verdict, credit.reason), (Verdict.FATAL, "credit_exhausted"))
        self.assertEqual(classify_status(401).reason, "http_401")
        self.assertIs(classify_status(401).verdict, Verdict.FATAL)
        self.assertIs(classify_status(503).verdict, Verdict.RETRY)
        self.assertIs(classify_status(400).verdict, Verdict.FATAL)


class CallWithRetryTest(unittest.TestCase):
    def test_429_pauses_lane_for_retry_after(self):
        clock = FakeClock()
        lim = Limiter(100, 60.0, 4, clock=clock, sleep=clock.sleep)
        replies = iter([response(429, THROTTLE_BODY, {"Retry-After": "42"}), response(200, "{}")])
        out = call_with_retry(lambda: next(replies), limiter=lim, sleep=clock.sleep)
        self.assertEqual(out.status_code, 200)
        starts = list(lim.history)
        self.assertGreaterEqual(starts[1] - starts[0], 42.0)

    def test_5xx_retries_then_succeeds(self):
        clock = FakeClock()
        replies = iter([response(503), response(502), response(200, "{}")])
        out = call_with_retry(lambda: next(replies), sleep=clock.sleep)
        self.assertEqual(out.status_code, 200)
        self.assertEqual(len(clock.slept), 2)

    def test_fatal_stops_at_once(self):
        calls = []

        def fn():
            calls.append(1)
            return response(401, {"error": {"type": "auth_error"}})

        with self.assertRaises(FatalError) as ctx:
            call_with_retry(fn, sleep=lambda s: None)
        self.assertEqual(ctx.exception.outcome.reason, "http_401")
        self.assertEqual(len(calls), 1)

    def test_exhausted(self):
        with self.assertRaises(RetriesExhausted):
            call_with_retry(lambda: response(500), max_tries=3, sleep=lambda s: None)


if __name__ == "__main__":
    unittest.main()
