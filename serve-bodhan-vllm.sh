#!/usr/bin/env bash
# Serve the Bodhan recognizer (models/indic-ocr/weights/ocr) on a local vLLM, for BODHAN_RUNTIME=vllm.
#
#     ./serve-bodhan-vllm.sh            # foreground; ready at "Application startup complete" (~2 min)
#
# vLLM lives in .venv-vllm (torch 2.13 + CUDA 13); installing it into .venv would replace that
# env's torch 2.11 + cu128 and break causal_conv1d. One-time setup:
#
#     /usr/bin/python3.12 -m venv .venv-vllm && .venv-vllm/bin/pip install --no-cache-dir vllm==0.32.0
#
# 0.65 of the 8 GB leaves room for the layout model in the pipeline process and the desktop;
# it still gives ~97k tokens of cache. 16384 = an 8192-token table plus its largest crop (~2.4k).
#
# Sleep mode: the web app puts the model to sleep between jobs (weights kept in RAM, GPU memory freed, 0.7 s)
# and wakes it for the next one (0.1 s) instead of paying the ~70 s cold start again. Measured on the RTX 5070:
# 4.0 GB used awake, 0.5 GB asleep, identical output after waking, no change in speed (24 pages/min).
# VLLM_SERVER_DEV_MODE exposes the /sleep and /wake_up routes; the server listens on 127.0.0.1 only.
set -euo pipefail
cd "$(dirname "$0")"
export VLLM_SERVER_DEV_MODE=1

exec .venv-vllm/bin/vllm serve "${BODHAN_LOCAL_DIR:-models/indic-ocr}/weights/ocr" \
    --served-model-name "${BODHAN_VLLM_MODEL:-indic-ocr}" \
    --host 127.0.0.1 --port "${BODHAN_VLLM_PORT:-8000}" \
    --dtype bfloat16 \
    --max-model-len 16384 \
    --gpu-memory-utilization "${BODHAN_VLLM_GPU_UTIL:-0.65}" \
    --max-num-seqs 64 \
    --limit-mm-per-prompt '{"image":1,"video":0}' \
    --generation-config vllm \
    --enable-sleep-mode
