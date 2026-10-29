#!/usr/bin/env bash
# Start the local web app and open it in the browser. The GPU stays idle until a document is read.
#
#     ./start-web.sh              # http://127.0.0.1:8765
#     ./start-web.sh --port 9000 --no-browser
set -euo pipefail
cd "$(dirname "$0")"
exec .venv/bin/python -m web "$@"
