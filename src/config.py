"""Load environment for Week 1. Do not print secret values."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

SAMPLES_DIR = ROOT / "samples"
CACHE_DIR = ROOT / "cache"
OUT_DIR = ROOT / "out"
PAGES_DIR = OUT_DIR / "pages"
PAGES_QWEN_DIR = OUT_DIR / "pages_qwen"
SARVAM_OUT_DIR = OUT_DIR / "sarvam"
BODHAN_SEQ_DIR = OUT_DIR / "bodhan_seq"
BODHAN_PAR_DIR = OUT_DIR / "bodhan_par"
QWEN_SEQ_DIR = OUT_DIR / "qwen_seq"
QWEN_PAR_DIR = OUT_DIR / "qwen_par"

# User-chosen OCR language. Default is English (Sarvam code en-IN).
LANGUAGE_ALIASES = {
    "en": "en-IN",
    "eng": "en-IN",
    "english": "en-IN",
    "hi": "hi-IN",
    "hindi": "hi-IN",
    "bn": "bn-IN",
    "bengali": "bn-IN",
    "ta": "ta-IN",
    "tamil": "ta-IN",
    "te": "te-IN",
    "telugu": "te-IN",
    "mr": "mr-IN",
    "marathi": "mr-IN",
    "gu": "gu-IN",
    "gujarati": "gu-IN",
    "kn": "kn-IN",
    "kannada": "kn-IN",
    "ml": "ml-IN",
    "malayalam": "ml-IN",
    "od": "od-IN",
    "or": "od-IN",
    "odia": "od-IN",
    "oriya": "od-IN",
    "pa": "pa-IN",
    "punjabi": "pa-IN",
    "as": "as-IN",
    "assamese": "as-IN",
    "ur": "ur-IN",
    "urdu": "ur-IN",
    "sa": "sa-IN",
    "sanskrit": "sa-IN",
    "ne": "ne-IN",
    "nepali": "ne-IN",
}


LANGUAGE_MENU = (
    "en-IN English",
    "ta-IN Tamil",
    "mr-IN Marathi",
    "hi-IN Hindi",
    "bn-IN Bengali",
    "te-IN Telugu",
    "kn-IN Kannada",
    "ml-IN Malayalam",
    "gu-IN Gujarati",
    "pa-IN Punjabi",
)


def resolve_ocr_language(raw: str | None) -> str:
    """Return a language code. Empty input is English. Unknown values pass through."""
    value = (raw or "").strip()
    if not value:
        return "en-IN"
    alias = LANGUAGE_ALIASES.get(value.lower())
    if alias:
        return alias
    return value


def choose_ocr_language(raw: str | None = None) -> str:
    """Ask for a language on a TTY. Skip (empty Enter) is English."""
    if raw is not None and str(raw).strip():
        code = resolve_ocr_language(raw)
        print(f"OCR language: {code}")
        return code
    if not sys.stdin.isatty():
        print("OCR language: en-IN (no prompt; stdin is not a terminal)")
        return "en-IN"
    print("Choose OCR language. Press Enter to skip (English).")
    print("Examples: english, tamil, marathi, hindi, or a code such as ta-IN")
    for item in LANGUAGE_MENU:
        print(f"  {item}")
    try:
        answer = input("Language [English]: ").strip()
    except EOFError:
        answer = ""
    code = resolve_ocr_language(answer)
    print(f"OCR language: {code}")
    return code


OCR_LANGUAGE = "en-IN"
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen/qwen3.8-flash")
BODHAN_MODEL = os.environ.get("BODHAN_MODEL", "indic-ocr")
# html keeps merged cells (rowspan/colspan). markdown flattens them.
BODHAN_TABLE_FORMAT = os.environ.get("BODHAN_TABLE_FORMAT", "html")
QWEN_WORKERS = int(os.environ.get("QWEN_WORKERS", "3"))
BODHAN_WORKERS = int(os.environ.get("BODHAN_WORKERS", "4"))
PDF_LONG_EDGE = int(os.environ.get("PDF_LONG_EDGE", "1600"))

# Measured 2026-10-10: 8 requests per ~60 s per model; 10 parallel per key.
BODHAN_RATE_PER_MIN = float(os.environ.get("BODHAN_RATE_PER_MIN", "8"))
BODHAN_MAX_INFLIGHT = int(os.environ.get("BODHAN_MAX_INFLIGHT", "4"))
BODHAN_BACKEND = os.environ.get("BODHAN_BACKEND", "api")  # api | local
BODHAN_REPO = os.environ.get("BODHAN_REPO", "bodhan-ai/indic-ocr")
BODHAN_LOCAL_DIR = Path(os.environ.get("BODHAN_LOCAL_DIR", str(ROOT / "models" / "indic-ocr")))
# mlx: Apple Metal recognizer. pytorch: official stack (CUDA kernels on the RTX 5070).
# vllm: recognizer on a local vLLM server (./serve-bodhan-vllm.sh), layout in this process.
BODHAN_RUNTIME = os.environ.get("BODHAN_RUNTIME", "mlx" if sys.platform == "darwin" else "pytorch")
# 8-bit: same text as BF16 on 56 of 57 measured crops, 0.7 GB less GPU memory, as fast.
# BF16 is hari31416/indic-ocr-mlx-bf16 in models/indic-ocr-mlx-bf16.
BODHAN_MLX_REPO = os.environ.get("BODHAN_MLX_REPO", "hari31416/indic-ocr-mlx-8bit")
BODHAN_MLX_DIR = Path(os.environ.get("BODHAN_MLX_DIR", str(ROOT / "models" / BODHAN_MLX_REPO.split("/")[-1])))
# Crops from every page in flight share one continuous Metal batch: slots, then pages
# in flight to keep them full. One model, not one copy per page. M4: 16 slots ~3 GB.
BODHAN_MLX_BATCH = int(os.environ.get("BODHAN_MLX_BATCH", "16"))
BODHAN_MLX_PAGES = int(os.environ.get("BODHAN_MLX_PAGES", "6"))
# vLLM: pages in flight. RTX 5070 Laptop: 8 -> 25, 16 -> 36, 32 -> 42 pages/min.
BODHAN_VLLM_URL = os.environ.get("BODHAN_VLLM_URL", "http://127.0.0.1:8000/v1")
BODHAN_VLLM_MODEL = os.environ.get("BODHAN_VLLM_MODEL", "indic-ocr")
BODHAN_VLLM_PAGES = int(os.environ.get("BODHAN_VLLM_PAGES", "32"))
# Which engine reads every page and supplies the document: sarvam | bodhan | auto. auto = bodhan when it runs
# locally on vLLM (free, 23-42 pages/min), sarvam otherwise (Bodhan API 8/min, MLX 3-6/min; Sarvam 15-30).
BODHAN_LEAD = os.environ.get("BODHAN_LEAD", "auto").strip().lower()
# Sarvam Vision: 10 requests/min per account. Status polls count.
SARVAM_RATE_PER_MIN = float(os.environ.get("SARVAM_RATE_PER_MIN", "10"))
SARVAM_MAX_PAGES_PER_PDF = 10

# Page images sent to Bodhan and used for crops.
RENDER_DPI = int(os.environ.get("RENDER_DPI", "200"))
RENDER_MAX_LONG_PX = int(os.environ.get("RENDER_MAX_LONG_PX", "3000"))

# Agent (decision points) and vision tie-breaker. OpenRouter now, local later.
AGENT_BASE_URL = os.environ.get("AGENT_BASE_URL", "https://openrouter.ai/api/v1")
AGENT_MODEL = os.environ.get("AGENT_MODEL", "qwen/qwen3.5-9b")
VERIFIER_VLM_MODEL = os.environ.get("VERIFIER_VLM_MODEL", "google/gemini-3.1-flash-lite")
OPENROUTER_BUDGET_USD = float(os.environ.get("OPENROUTER_BUDGET_USD", "10"))
# Model calls in flight and per minute. Was 2 / 30: the image check waited on itself (Union budget:
# ~36 sheets x 28 s / 2 in flight = 8-9 min). OpenRouter allows far more.
AGENT_MAX_INFLIGHT = int(os.environ.get("AGENT_MAX_INFLIGHT", "8"))
AGENT_STARTS_PER_MIN = int(os.environ.get("AGENT_STARTS_PER_MIN", "120"))
# Image check for standard table checking (chosen on the 2026-10-21 pilot, bench/pilot.py):
# born-digital tables use one model; scanned ruled tables need two families to agree.
TABLE_JUDGE_SINGLE = os.environ.get("TABLE_JUDGE_SINGLE", "openai/gpt-5.6-luna")
TABLE_JUDGE_MODELS = tuple(os.environ.get("TABLE_JUDGE_MODELS", "google/gemini-3.1-flash-lite,openai/gpt-5.6-luna").split(","))
# Per-mode table checking (ceiling, audits, sheet layout, third model) lives in src/modes.py.

# Cost accounting in rupees. INR_PER_USD is approximate; set it in .env.
SARVAM_INR_PER_PAGE = float(os.environ.get("SARVAM_INR_PER_PAGE", "0.5"))
BODHAN_INR_PER_PAGE = float(os.environ.get("BODHAN_INR_PER_PAGE", "0.2"))
INR_PER_USD = float(os.environ.get("INR_PER_USD", "85"))
# OpenRouter qwen/qwen3.8-flash list price (2026-10-10), US$ per million tokens.
QWEN_USD_PER_M_IN = float(os.environ.get("QWEN_USD_PER_M_IN", "0.15"))
QWEN_USD_PER_M_OUT = float(os.environ.get("QWEN_USD_PER_M_OUT", "0.47"))
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DOCS_OUT_DIR = OUT_DIR / "docs"

SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "").strip()
BODHAN_API_KEY = os.environ.get("BODHAN_API_KEY", "").strip()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()


def has_sarvam_key() -> bool:
    return bool(SARVAM_API_KEY)


def has_bodhan_key() -> bool:
    return bool(BODHAN_API_KEY)


def has_openrouter_key() -> bool:
    return bool(OPENROUTER_API_KEY)
