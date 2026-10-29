"""Local bodhan-ai/indic-ocr. One page image in, API-shaped payload out.

The Hub snapshot ships the loader (`indic_ocr.py`) next to the weights.
`IndicOCR.parse` returns reading-ordered blocks. This module turns that
record into the payload `from_bodhan` already reads.
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from src.config import (
    BODHAN_LOCAL_DIR,
    BODHAN_MLX_DIR,
    BODHAN_MLX_REPO,
    BODHAN_REPO,
    BODHAN_RUNTIME,
)

LOCAL_CACHE_NS = "indic-ocr-local"
# MLX ceiling per block. Each crop gets a smaller limit from its size and class
# (src.mlx_recognizer.token_cap). 2048 cut 5 of 6 Union budget tables mid-row.
MAX_NEW_TOKENS = 8192
# PyTorch generates a batch to one limit, so a looping block runs all the way to it.
# This is the official local loader's cap; the HTTP API allows 4096.
PYTORCH_MAX_NEW_TOKENS = 2048
# vLLM ceiling (each crop's cap scales with its size). 2048 cut every full-page budget table;
# 4096 (the API's) still cut 6 of 20 Union receipt-budget pages; the longest measured is 6402.
VLLM_MAX_NEW_TOKENS = 8192

_LOCK = threading.RLock()
_PARSER: Any = None
_PARSER_FMT: str | None = None


def runtime() -> str:
    return BODHAN_RUNTIME if BODHAN_RUNTIME in ("mlx", "pytorch", "vllm") else "pytorch"


def active_model() -> str:
    return BODHAN_MLX_REPO if runtime() == "mlx" else BODHAN_REPO


def local_cache_extra(table_format: str) -> str:
    """MLX names its weights (8-bit and BF16 read differently). p3: per-crop token limits."""
    if runtime() == "mlx":
        return f"{table_format}|mlx|{BODHAN_MLX_REPO}|{MAX_NEW_TOKENS}|p3"
    if runtime() == "vllm":
        return f"{table_format}|vllm|{VLLM_MAX_NEW_TOKENS}|d3"  # d3: invented text cut from runaway blocks (d2: decoded here; d1 had U+FFFD for x, c-cedilla, quotes)
    return f"{table_format}|{runtime()}|{PYTORCH_MAX_NEW_TOKENS}"


def _readable(text: str) -> str:
    """Turn a leftover SentencePiece word-boundary mark into a space."""
    return text.replace("\u2581", " ")


def payload_from_page(page: dict, *, table_format: str, mime: str) -> dict:
    """Map a local parse record onto the Bodhan chat-completion payload."""
    markdown = _readable(page.get("markdown") or "")
    raw_blocks = page.get("blocks") if isinstance(page.get("blocks"), list) else []
    blocks = []
    for block in raw_blocks:
        if isinstance(block, dict) and isinstance(block.get("text"), str):
            block = {**block, "text": _readable(block["text"])}
        blocks.append(block)
    return {
        "model": active_model(),
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": markdown},
            }
        ],
        "blocks": blocks,
        "usage": {},
        "_infinite": {
            "table_format": table_format,
            "mime": mime,
            "backend": "local",
            "runtime": runtime(),
            "degraded": [],
            **(page.get("_timing") or {}),
        },
    }


def pick_device() -> str:
    """cuda, then Metal, then CPU. `BODHAN_DEVICE` overrides the choice."""
    chosen = os.environ.get("BODHAN_DEVICE", "").strip()
    if chosen:
        return chosen
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def ensure_snapshot(local_dir: Path | None = None) -> Path:
    """Return the snapshot directory. Download it when the weights are absent."""
    dest = Path(local_dir or BODHAN_LOCAL_DIR)
    weights = dest / "weights" / "ocr"
    if (dest / "indic_ocr.py").is_file() and any(weights.glob("*.safetensors")):
        return dest
    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(BODHAN_REPO, local_dir=str(dest))
    return dest


def ensure_mlx(local_dir: Path | None = None) -> Path:
    """Return the BF16 MLX snapshot. Download it when the weights are absent."""
    dest = Path(local_dir or BODHAN_MLX_DIR)
    if (dest / "config.json").is_file() and any(dest.glob("*.safetensors")):
        return dest
    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(BODHAN_MLX_REPO, local_dir=str(dest))
    return dest


class _LocalEngine:
    """Layout from the official snapshot, recognizer from MLX or PyTorch."""

    def __init__(self, layout: Any, ocr: Any, name: str) -> None:
        self.layout = layout
        self.ocr = ocr
        self.name = name

    def parse(self, image_path: str) -> dict:
        backend = self.ocr.backend
        begin = getattr(backend, "begin_layout", None)
        end = getattr(backend, "end_layout", None)
        started = time.perf_counter()
        if begin is not None:
            begin()
        try:
            layout = self.layout.detect(image_path)
        finally:
            if end is not None:
                end()
        layout_s = time.perf_counter() - started
        started = time.perf_counter()
        page = self.ocr.run(image_path, layout)
        ocr_s = time.perf_counter() - started
        n_crops = backend.last_n_crops() if hasattr(backend, "last_n_crops") else 0
        record = {**page.as_record(), "markdown": page.markdown or ""}
        record["_timing"] = {
            "runtime": self.name,
            "layout_s": round(layout_s, 3),
            "ocr_s": round(ocr_s, 3),
            "total_s": round(layout_s + ocr_s, 3),
            "crops": n_crops,
        }
        return record


def parser(table_format: str = "html") -> _LocalEngine:
    """Load both stages once."""
    global _PARSER, _PARSER_FMT
    with _LOCK:
        if _PARSER is not None and _PARSER_FMT == table_format:
            return _PARSER
        return _load_parser(table_format)


def _load_parser(table_format: str) -> _LocalEngine:
    """Build the engine. The caller holds `_LOCK`."""
    global _PARSER, _PARSER_FMT
    repo = ensure_snapshot()
    repo_s = str(repo)
    if repo_s not in sys.path:
        sys.path.insert(0, repo_s)
    from idp_offline import IndicBlockOCR, IndicDocLayout
    from idp_types import CropConfig, DedupConfig, LayoutConfig, RecognizerConfig, TableFormat

    device = pick_device()
    layout = IndicDocLayout(
        ckpt=str(repo / "weights" / "layout"),
        config=LayoutConfig(device=device),
        dedup=DedupConfig(),
    )
    name = runtime()
    max_tokens = {"vllm": VLLM_MAX_NEW_TOKENS, "mlx": MAX_NEW_TOKENS}.get(name, PYTORCH_MAX_NEW_TOKENS)
    rec_cfg = RecognizerConfig(max_tokens=max_tokens, table_format=TableFormat(table_format))
    if name == "vllm":
        from src.vllm_recognizer import VllmRecognizer

        backend = VllmRecognizer(max_tokens=max_tokens)
    elif name == "mlx":
        from src.mlx_recognizer import MlxRecognizer

        backend = MlxRecognizer(ensure_mlx(), max_tokens=max_tokens)
    else:
        from idp_recognizer import HfRecognizer

        backend = HfRecognizer(
            ckpt=str(repo / "weights" / "ocr"),
            config=rec_cfg,
            device=device,
        )
    ocr = IndicBlockOCR(
        backend=backend,
        config=rec_cfg,
        dedup=DedupConfig(),
        crop=CropConfig(),
    )
    _PARSER = _LocalEngine(layout, ocr, name)
    _PARSER_FMT = table_format
    return _PARSER


def local_ocr(image: bytes, mime: str = "image/png", *, table_format: str = "html") -> dict:
    """Run both stages on one page image.

    MLX and vLLM pages share one model. Layouts take turns. Crops from ready
    pages share one batch. PyTorch stays one page at a time.
    """
    suffix = ".jpg" if mime in ("image/jpeg", "image/jpg") else ".png"
    engine = parser(table_format)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(image)
        path = handle.name
    try:
        if runtime() in ("mlx", "vllm"):
            page = engine.parse(path)
        else:
            with _LOCK:
                page = engine.parse(path)
    finally:
        Path(path).unlink(missing_ok=True)
    return payload_from_page(page, table_format=table_format, mime=mime)
