"""Stage 2 recognizer for the RTX 5070: crops to a local vLLM server.

vLLM runs in its own environment (`.venv-vllm`: torch 2.13 + CUDA 13, which would
break this one) and serves the recognizer weights over the OpenAI chat API. Layout
stays in this process. Every crop of every page in flight is one request, so crops
from all pages share vLLM's continuous batch; a long table never holds up short
blocks.

    ./serve-bodhan-vllm.sh        # once per session; ~2 min to compile and capture graphs

Measured on the RTX 5070 Laptop (12 pages: 6 Hindi budget tables, en/mr/ta/te
prose and tables; 4096 tokens per block):
    HF transformers, one page at a time    2.75 pages/min    69 tok/s   6 tables cut at 2048
    vLLM, 8 pages in flight                25.4 pages/min   775 tok/s   none cut
    vLLM, 32 pages in flight (36 pages)    42.3 pages/min  1291 tok/s
Of 62 crops, 53 read the same as HF, 6 are the tables HF cut, and 3 differ by a
near-tie (HF was closer on the ones checked). vLLM output was identical at 8, 16
and 32 pages in flight.

Text is decoded here, not by the server. vLLM's detokenizer turns characters that the tokenizer spells as
several byte tokens (x, c-cedilla, curly quotes) into U+FFFD, although the tokens are right: `8x A100` came back as
`8?A100`. So each request asks for the token ids and they are decoded with the model's own tokenizer file, which
also gives real spaces instead of the word-start marker.

A paragraph that continues on the next page is cropped mid-sentence, and then the model never finds a place to
stop: it invents a continuation until the token cap (measured on a paper's last paragraph of a page: 7 visible lines
became 4,651 characters). Copying visible text is near-certain for the model (mean logprob 0.00, worst token -0.02),
while the first invented tokens are not (windows of -0.4 to -1.4). For a text crop that ran to the cap, the text is
cut at the first token of that drop. Nothing else is trimmed.

Each crop's token cap follows its size. Over 153 crops (32 pages) no real block
wrote more than 2.66 tokens per image token (a 39-row budget table: 6402 tokens);
a 45 px page number looped on Odia digits to the cap. TOKENS_PER_IMAGE_TOKEN
leaves 50% over the densest table and stops such a loop at MIN_TOKENS.
"""

from __future__ import annotations

import base64
import io
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import requests

from src.config import BODHAN_LOCAL_DIR, BODHAN_VLLM_MODEL, BODHAN_VLLM_URL, ROOT

logger = logging.getLogger("vllm_recognizer")

# The processor turns 32 x 32 px into one image token (16 px patches, 2 x 2 merge).
PX_PER_IMAGE_TOKEN = 32 * 32
TOKENS_PER_IMAGE_TOKEN = 4
MIN_TOKENS = 512

# Cutting invented text (see above). Only text prompts: a table or an equation legitimately ends at the cap.
TEXT_PROMPT_START = "Transcribe the text"
TRIM_WINDOW = 6     # tokens
TRIM_MEAN = -0.3    # a window this unsure is the model inventing, not copying
TRIM_DIP = -0.15   # the first token below this inside that window is the first invented one


def token_cap(width: int, height: int, ceiling: int) -> int:
    """Output budget for one crop: proportional to its image tokens, within [MIN_TOKENS, ceiling]."""
    image_tokens = width * height // PX_PER_IMAGE_TOKEN
    return max(min(MIN_TOKENS, ceiling), min(ceiling, TOKENS_PER_IMAGE_TOKEN * image_tokens))


def load_tokenizer():
    """The recognizer's own tokenizer file (tokenizers library only: no torch, no GPU), or None if unavailable."""
    path = ROOT / BODHAN_LOCAL_DIR / "weights" / "ocr" / "tokenizer.json"
    try:
        from tokenizers import Tokenizer

        return Tokenizer.from_file(str(path))
    except Exception as exc:  # noqa: BLE001 - fall back to the server's text rather than stop
        logger.warning("tokenizer %s not usable (%s); using the server's text, which may show U+FFFD", path, exc)
        return None


def invention_starts(logprobs: list[float]) -> int | None:
    """Index of the first invented token in a block that ran to its cap, or None if the model stayed sure of itself."""
    for i in range(len(logprobs) - TRIM_WINDOW + 1):
        window = logprobs[i : i + TRIM_WINDOW]
        if sum(window) / TRIM_WINDOW < TRIM_MEAN:
            return next(i + k for k, lp in enumerate(window) if lp < TRIM_DIP)
    return None


class VllmUnavailable(RuntimeError):
    """The server is not running or does not serve the recognizer."""


class VllmRecognizer:
    """Transcribe block crops through a local vLLM server."""

    def __init__(
        self,
        url: str = BODHAN_VLLM_URL,
        model: str = BODHAN_VLLM_MODEL,
        max_tokens: int = 8192,
        max_parallel: int = 256,
        timeout_s: float = 900.0,
        tokenizer=None,
    ) -> None:
        self.url = url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self._tls = threading.local()
        self._layout = threading.Lock()
        self._tokenizer = tokenizer if tokenizer is not None else load_tokenizer()
        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=max_parallel)
        self._session.mount("http://", adapter)
        # One pool for every page: the server batches whatever arrives.
        self._pool = ThreadPoolExecutor(max_parallel, thread_name_prefix="vllm-crop")
        self._check()

    def _check(self) -> None:
        try:
            response = self._session.get(f"{self.url}/models", timeout=5)
            response.raise_for_status()
            served = [m.get("id") for m in response.json().get("data", [])]
        except (requests.RequestException, ValueError) as exc:
            raise VllmUnavailable(f"no vLLM server at {self.url} ({exc}); run ./serve-bodhan-vllm.sh") from exc
        if self.model not in served:
            raise VllmUnavailable(f"vLLM at {self.url} serves {served}, not {self.model!r}")

    def begin_layout(self) -> None:
        # Layouts take turns. vLLM is another process, so they never wait for decode.
        self._layout.acquire()

    def end_layout(self) -> None:
        self._layout.release()

    def _body(self, req) -> dict:
        buf = io.BytesIO()
        req.image.save(buf, format="PNG")
        data = base64.b64encode(buf.getvalue()).decode("ascii")
        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}},
                        {"type": "text", "text": req.prompt},
                    ],
                }
            ],
            "max_tokens": token_cap(req.image.width, req.image.height, self.max_tokens),
            "temperature": 0.0,
            "return_token_ids": True,
        }
        if req.prompt.startswith(TEXT_PROMPT_START):
            body["logprobs"] = True  # for cutting invented text; tables do not need them
        return body

    def _one(self, req) -> str:
        response = self._session.post(f"{self.url}/chat/completions", json=self._body(req), timeout=self.timeout_s)
        response.raise_for_status()
        choice = response.json()["choices"][0]
        ids = choice.get("token_ids")
        if choice.get("finish_reason") == "length":
            logprobs = [t["logprob"] for t in (choice.get("logprobs") or {}).get("content") or []]
            trimmable = req.prompt.startswith(TEXT_PROMPT_START) and ids and len(logprobs) == len(ids)
            cut = invention_starts(logprobs) if trimmable else None
            if cut:
                logger.warning("block (%dx%d px) ran past its text; cut %d invented tokens", req.image.width, req.image.height, len(ids) - cut)
                ids = ids[:cut]
            else:
                logger.warning("block hit its token cap (%dx%d px); its text is cut", req.image.width, req.image.height)
        if self._tokenizer is not None and ids:
            return self._tokenizer.decode(ids, skip_special_tokens=True)
        return choice["message"]["content"] or ""

    def transcribe(self, requests: list) -> list[str]:
        self._tls.n_crops = len(requests)
        return [text.strip() for text in self._pool.map(self._one, requests)]

    def last_n_crops(self) -> int:
        return int(getattr(self._tls, "n_crops", 0))

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
        self._session.close()
