"""Start the web app: `python -m web`. Opens the page in the browser; the GPU stays idle until a job starts."""

from __future__ import annotations

import argparse
import threading
import time
import webbrowser

import requests
import uvicorn

from .server import app


def _open_when_ready(url: str) -> None:
    for _ in range(100):
        try:
            if requests.get(f"{url}/api/status", timeout=0.5).ok:
                webbrowser.open_new_tab(url)
                return
        except requests.RequestException:
            time.sleep(0.2)


def main() -> None:
    """Read the command line, open the page in the browser and serve the app on 127.0.0.1 only."""
    parser = argparse.ArgumentParser(prog="python -m web", description="Infinite-OCR local web app")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="do not open the page")
    args = parser.parse_args()
    url = f"http://127.0.0.1:{args.port}"
    if not args.no_browser:
        threading.Thread(target=_open_when_ready, args=(url,), daemon=True).start()
    print(f"Infinite-OCR at {url}  (the GPU stays idle until a document is read; Ctrl-C to stop)")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
