"""Shared test helpers: a fake clock and hand-built HTTP responses."""

from __future__ import annotations

import json

import requests


class FakeClock:
    """`clock()` returns fake time; `sleep(s)` advances it instantly."""

    def __init__(self, start: float = 0.0) -> None:
        self.t = start
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.t += max(0.0, seconds)


def response(status: int, body: object = "", headers: dict | None = None) -> requests.Response:
    r = requests.Response()
    r.status_code = status
    text = body if isinstance(body, str) else json.dumps(body)
    r._content = text.encode("utf-8")
    r.headers.update(headers or {})
    r.encoding = "utf-8"
    return r


OK_PAYLOAD = {
    "id": "x",
    "object": "chat.completion",
    "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "hi"}}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    "blocks": [],
}
THROTTLE_BODY = {
    "error": {
        "message": "Maximum rate limit reached for the model",
        "type": "throttling_error",
        "param": None,
        "code": "429",
    }
}
