"""Rate limiting and retries.

New code uses `Limiter` (sliding window + even spacing + in-flight cap +
lane-wide pause) with `call_with_retry` and a `classify` function that turns
each response or exception into a `Verdict`.

The older `TokenBucket` / `RateLimiter` / `call_with_429_retry` helpers are
kept for the Week 1 Qwen path.
"""

from __future__ import annotations

import random
import re
import threading
import time
from collections import deque
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterator, TypeVar

import requests

T = TypeVar("T")

MAX_RETRIES = 5

# A 429 whose body mentions money is not a rate limit. Retrying cannot help.
CREDIT_RE = re.compile(r"credit|balance|insufficient|billing|payment|recharge|top.?up", re.I)


class Limiter:
    """At most `max_starts` request starts in any `window_s` seconds.

    Starts are also spaced by at least `min_spacing_s`, so a server with a
    fixed window never sees a burst. `max_inflight` caps concurrent calls.
    `pause_until` blocks every caller of this limiter (one lane), which is how
    a 429 `Retry-After` is honoured lane-wide.
    """

    def __init__(
        self,
        max_starts: int,
        window_s: float,
        max_inflight: int,
        min_spacing_s: float = 0.0,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_starts < 1 or window_s <= 0 or max_inflight < 1:
            raise ValueError("max_starts, window_s and max_inflight must be positive")
        self.max_starts = max_starts
        self.window_s = window_s
        self.max_inflight = max_inflight
        self.min_spacing_s = min_spacing_s
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._sema = threading.BoundedSemaphore(max_inflight)
        self._starts: deque[float] = deque()
        self._last_start: float | None = None
        self._paused_until = 0.0
        self.history: deque[float] = deque(maxlen=10_000)

    def _wait_needed(self, now: float) -> float:
        while self._starts and now - self._starts[0] >= self.window_s:
            self._starts.popleft()
        wait = self._paused_until - now
        if self._last_start is not None:
            wait = max(wait, self._last_start + self.min_spacing_s - now)
        if len(self._starts) >= self.max_starts:
            wait = max(wait, self._starts[0] + self.window_s - now)
        return wait

    def acquire_start(self) -> float:
        """Block until a request may start. Returns the start time."""
        while True:
            with self._lock:
                now = self._clock()
                wait = self._wait_needed(now)
                if wait <= 0:
                    self._starts.append(now)
                    self._last_start = now
                    self.history.append(now)
                    return now
            self._sleep(wait)

    @contextmanager
    def slot(self) -> Iterator[None]:
        self._sema.acquire()
        try:
            self.acquire_start()
            yield
        finally:
            self._sema.release()

    def pause_until(self, t: float) -> None:
        with self._lock:
            self._paused_until = max(self._paused_until, t)

    def pause_for(self, seconds: float) -> None:
        self.pause_until(self._clock() + seconds)


class Verdict(Enum):
    """How a response is judged: ok, retry, rate limited, degrade or fatal."""
    OK = "ok"
    RETRY = "retry"
    RATE_LIMITED = "rate_limited"
    DEGRADE = "degrade"
    FATAL = "fatal"


@dataclass(frozen=True)
class Outcome:
    """A response's verdict, the reason, and how long to wait before the next call."""
    verdict: Verdict
    reason: str = ""
    wait_s: float | None = None


class CallFailed(RuntimeError):
    """A call that could not be completed; carries its outcome and the number of attempts."""
    def __init__(self, outcome: Outcome, attempts: int) -> None:
        super().__init__(f"{outcome.verdict.value}: {outcome.reason} (attempts={attempts})")
        self.outcome = outcome
        self.attempts = attempts


class FatalError(CallFailed):
    """Do not retry: bad key, no credit, bad request."""


class DegradeError(CallFailed):
    """The caller should change the request (format, encoding) and try again."""


class RetriesExhausted(CallFailed):
    """Retryable failures used up every attempt."""


def _headers_of(obj: Any) -> Any:
    headers = getattr(obj, "headers", None)
    if headers:
        return headers
    response = getattr(obj, "response", None)
    return getattr(response, "headers", None) if response is not None else None


def _status_code(exc: BaseException) -> int | None:
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code
    response = getattr(exc, "response", None)
    if response is not None:
        resp_code = getattr(response, "status_code", None)
        if isinstance(resp_code, int):
            return resp_code
    return None


def _retry_after_seconds(obj: Any) -> float | None:
    headers = _headers_of(obj)
    if not headers:
        return None
    raw = headers.get("Retry-After") or headers.get("retry-after")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _body_text(obj: Any) -> str:
    if isinstance(obj, requests.Response):
        return obj.text[:1000]
    body = getattr(obj, "body", None)
    if body is not None:
        return str(body)[:1000]
    response = getattr(obj, "response", None)
    text = getattr(response, "text", None) if response is not None else None
    return str(text or obj)[:1000]


def classify_status(code: int, body: str = "", retry_after: float | None = None) -> Outcome:
    if 200 <= code < 300:
        return Outcome(Verdict.OK)
    if code == 429:
        if CREDIT_RE.search(body):
            return Outcome(Verdict.FATAL, "credit_exhausted")
        return Outcome(Verdict.RATE_LIMITED, "http_429", retry_after)
    if code in (401, 402, 403):
        return Outcome(Verdict.FATAL, f"http_{code}")
    if code in (408, 425) or code >= 500:
        return Outcome(Verdict.RETRY, f"http_{code}", retry_after)
    return Outcome(Verdict.FATAL, f"http_{code}")


def _is_transport_error(exc: BaseException) -> bool:
    if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
        return True
    if type(exc).__name__ in ("APIConnectionError", "APITimeoutError"):  # openai SDK
        return True
    try:
        import httpx
    except ImportError:
        return False
    return isinstance(exc, (httpx.TimeoutException, httpx.TransportError))


def classify_http(result: Any, exc: BaseException | None) -> Outcome:
    """Default classifier for `requests` responses and SDK exceptions."""
    if exc is not None:
        if _is_transport_error(exc):
            return Outcome(Verdict.RETRY, type(exc).__name__)
        code = _status_code(exc)
        if code is None:
            return Outcome(Verdict.FATAL, repr(exc)[:200])
        return classify_status(code, _body_text(exc), _retry_after_seconds(exc))
    if isinstance(result, requests.Response):
        return classify_status(
            result.status_code, _body_text(result), _retry_after_seconds(result)
        )
    return Outcome(Verdict.OK)


def call_with_retry(
    fn: Callable[[], T],
    *,
    classify: Callable[[Any, BaseException | None], Outcome] = classify_http,
    limiter: Limiter | None = None,
    max_tries: int = MAX_RETRIES,
    base_s: float = 2.0,
    cap_s: float = 120.0,
    default_429_wait: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
    label: str = "",
    on_outcome: Callable[[Outcome], None] | None = None,
) -> T:
    """Call `fn` (inside `limiter.slot()` if given) until `classify` says OK.

    RATE_LIMITED pauses the whole limiter, so other threads on the same lane
    wait too. FATAL raises FatalError and DEGRADE raises DegradeError at once.
    `on_outcome` sees every non-OK outcome (for event logs).
    """
    tries = max(1, max_tries)
    outcome = Outcome(Verdict.FATAL, "not_called")
    for attempt in range(1, tries + 1):
        exc: BaseException | None = None
        result: Any = None
        try:
            if limiter is not None:
                with limiter.slot():
                    result = fn()
            else:
                result = fn()
        except Exception as caught:  # noqa: BLE001 - classified below
            exc = caught
        outcome = classify(result, exc)
        if outcome.verdict is Verdict.OK:
            return result
        if on_outcome is not None:
            on_outcome(outcome)
        if outcome.verdict is Verdict.FATAL:
            raise FatalError(outcome, attempt) from exc
        if outcome.verdict is Verdict.DEGRADE:
            raise DegradeError(outcome, attempt) from exc
        if attempt == tries:
            break
        if outcome.verdict is Verdict.RATE_LIMITED:
            wait = outcome.wait_s if outcome.wait_s is not None else default_429_wait
            wait += random.uniform(0.5, 2.0)
            print(f"{label or 'call'}: 429, lane paused {wait:.1f}s (try {attempt + 1}/{tries})")
            if limiter is not None:
                limiter.pause_for(wait)
            else:
                sleep(wait)
        else:
            wait = outcome.wait_s or min(cap_s, base_s * (2 ** (attempt - 1)))
            wait *= random.uniform(0.8, 1.2)
            print(f"{label or 'call'}: {outcome.reason}, retry in {wait:.1f}s (try {attempt + 1}/{tries})")
            sleep(wait)
    raise RetriesExhausted(outcome, tries)


# ---------------------------------------------------------------------------
# Week 1 helpers (still used by qwen_ocr.py)
# ---------------------------------------------------------------------------


class TokenBucket:
    """Token-bucket limiter for the older week-1 runners: blocks until a request may start."""
    def __init__(self, rate_per_min: float, capacity: float | None = None) -> None:
        if rate_per_min <= 0:
            raise ValueError("rate_per_min must be > 0")
        self.rate_per_sec = rate_per_min / 60.0
        self.capacity = float(rate_per_min if capacity is None else capacity)
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: float = 1.0) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last
                self._last = now
                self._tokens = min(
                    self.capacity,
                    self._tokens + elapsed * self.rate_per_sec,
                )
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                deficit = tokens - self._tokens
                wait = deficit / self.rate_per_sec
            time.sleep(wait)


class RateLimiter:
    """A token bucket plus a cap on requests in flight, for the older week-1 runners."""
    def __init__(
        self,
        rate_per_min: float,
        max_inflight: int,
        capacity: float | None = None,
    ) -> None:
        if max_inflight < 1:
            raise ValueError("max_inflight must be >= 1")
        self.bucket = TokenBucket(rate_per_min, capacity=capacity)
        self._sema = threading.BoundedSemaphore(max_inflight)
        self.max_inflight = max_inflight

    @contextmanager
    def slot(self) -> Iterator[None]:
        self._sema.acquire()
        try:
            self.bucket.acquire()
            yield
        finally:
            self._sema.release()


def is_rate_limited(exc: BaseException) -> bool:
    return _status_code(exc) == 429


def call_with_429_retry(
    func: Callable[[], T],
    *,
    max_tries: int = MAX_RETRIES,
    base_sleep: float = 1.0,
) -> T:
    last_exc: BaseException | None = None
    tries = max(1, max_tries)
    for attempt in range(tries):
        try:
            return func()
        except Exception as exc:
            if not is_rate_limited(exc):
                raise
            last_exc = exc
            if attempt >= tries - 1:
                break
            retry_after = _retry_after_seconds(exc)
            sleep_s = retry_after if retry_after is not None else base_sleep * (2**attempt)
            sleep_s += random.uniform(0, 0.25)
            print(f"429 rate limit. Sleep {sleep_s:.1f}s then retry {attempt + 2}/{tries}.")
            time.sleep(sleep_s)
    assert last_exc is not None
    raise last_exc


_LIMITERS: dict[str, RateLimiter] = {}
_LIMITERS_LOCK = threading.Lock()


def get_limiter(
    backend: str,
    *,
    workers: int,
    rate_per_min: float,
    capacity: float | None = None,
) -> RateLimiter:
    with _LIMITERS_LOCK:
        limiter = _LIMITERS.get(backend)
        want_capacity = float(rate_per_min if capacity is None else capacity)
        if limiter is None:
            limiter = RateLimiter(
                rate_per_min=rate_per_min,
                max_inflight=workers,
                capacity=capacity,
            )
            _LIMITERS[backend] = limiter
            return limiter
        if (
            limiter.max_inflight != workers
            or abs(limiter.bucket.rate_per_sec * 60.0 - rate_per_min) > 1e-9
            or abs(limiter.bucket.capacity - want_capacity) > 1e-9
        ):
            limiter = RateLimiter(
                rate_per_min=rate_per_min,
                max_inflight=workers,
                capacity=capacity,
            )
            _LIMITERS[backend] = limiter
        return limiter


def reset_limiter(backend: str | None = None) -> None:
    with _LIMITERS_LOCK:
        if backend is None:
            _LIMITERS.clear()
        else:
            _LIMITERS.pop(backend, None)
