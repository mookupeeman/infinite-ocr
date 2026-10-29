"""The Bodhan recognizer's vLLM server: loaded when the first job needs it, asleep between jobs.

Until a job starts, nothing here touches the GPU. Loading takes about 70 s, so after a job the model is not
killed but put to sleep: vLLM keeps the weights in RAM and frees the GPU (4.0 GB awake, 0.5 GB asleep), and waking
it takes 0.1 s. WEB_MODEL_AFTER_JOB=stop restores the old behaviour (kill it, reload next time). A server someone
else already started on the port is reused and never put to sleep or stopped by us.
"""

from __future__ import annotations

import atexit
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

import requests

from src.config import BODHAN_VLLM_MODEL, BODHAN_VLLM_URL

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "out" / "web" / "vllm.log"
START_TIMEOUT_S = 420
MIN_FREE_MB = 5500  # weights plus cache at 0.65 utilisation
ASLEEP_MB = 1500    # a sleeping model should leave the GPU below this
AFTER_JOB = os.environ.get("WEB_MODEL_AFTER_JOB", "sleep").strip().lower()  # sleep | stop


class ModelError(RuntimeError):
    """The model server could not be started."""


class Cancelled(RuntimeError):
    """The user cancelled while the model was starting."""


def gpu_memory_mb() -> tuple[int, int] | None:
    """(used, total) in MB, or None without an NVIDIA GPU."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout
        used, total = (int(x) for x in out.strip().splitlines()[0].split(","))
        return used, total
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def _tail(path: Path, lines: int = 12) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


class ModelServer:
    """off -> starting -> ready <-> sleeping, and stopping -> off."""

    def __init__(self, url: str = BODHAN_VLLM_URL, model: str = BODHAN_VLLM_MODEL, after_job: str = AFTER_JOB) -> None:
        self.url = url.rstrip("/")
        self.base = self.url.removesuffix("/v1")  # the sleep and wake routes are not under /v1
        self.model = model
        self.after_job = after_job
        self._proc: subprocess.Popen | None = None
        self._state = "off"
        self._lock = threading.Lock()
        atexit.register(self.stop)

    def _serving(self) -> bool:
        try:
            r = requests.get(f"{self.url}/models", timeout=0.5)
            return r.ok and self.model in [m.get("id") for m in r.json().get("data", [])]
        except (requests.RequestException, ValueError):
            return False

    @property
    def owned(self) -> bool:
        """True if this app started the vLLM process, and so may put it to sleep or stop it."""
        return self._proc is not None

    @property
    def state(self) -> str:
        """off, starting, ready, sleeping or stopping. A server someone else started shows as ready."""
        if self._proc is not None:
            if self._proc.poll() is not None:  # it died on its own
                self._proc, self._state = None, "off"
                return "off"
            return self._state
        return "ready" if self._serving() else "off"

    def _wake(self) -> bool:
        """Bring a sleeping model back onto the GPU. False if it would not wake."""
        try:
            requests.post(f"{self.base}/wake_up", timeout=120).raise_for_status()
            asleep = requests.get(f"{self.base}/is_sleeping", timeout=5).json().get("is_sleeping", True)
        except (requests.RequestException, ValueError):
            return False
        if asleep:
            return False
        self._state = "ready"
        return True

    def start(self, cancelled: Callable[[], bool] = lambda: False) -> None:
        """Block until the model answers. Raises ModelError, or Cancelled if `cancelled()` turns true."""
        with self._lock:
            if self._proc is None and self._serving():
                return  # someone else's server: use it, never stop it
            if self.state == "ready" and self._proc is not None:
                return
            if self.state == "sleeping":
                if self._wake():
                    return
                self.stop()  # it would not wake: start over
            mem = gpu_memory_mb()
            if mem and mem[1] - mem[0] < MIN_FREE_MB:
                raise ModelError(
                    f"Only {(mem[1] - mem[0]) / 1024:.1f} GB of GPU memory is free and the model needs about "
                    f"{MIN_FREE_MB / 1024:.1f} GB. Close other programs that use the GPU and try again."
                )
            LOG.parent.mkdir(parents=True, exist_ok=True)
            log = LOG.open("wb")
            self._state = "starting"
            self._proc = subprocess.Popen(
                ["bash", str(ROOT / "serve-bodhan-vllm.sh")], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            deadline = time.monotonic() + START_TIMEOUT_S
            while time.monotonic() < deadline:
                if cancelled():
                    self.stop()
                    raise Cancelled()
                if self._proc.poll() is not None:
                    detail = _tail(LOG)
                    self._proc, self._state = None, "off"
                    raise ModelError("The model server stopped while starting.\n" + detail)
                if self._serving():
                    self._state = "ready"
                    return
                time.sleep(1.0)
            self.stop()
            raise ModelError(f"The model did not answer within {START_TIMEOUT_S // 60} minutes.\n" + _tail(LOG))

    def release(self) -> str:
        """After a job: put the model to sleep (or stop it). Returns "sleeping", "stopped" or "kept" (not ours)."""
        if self._proc is None or self.state != "ready":
            return "kept" if self._proc is None else self.state
        if self.after_job == "sleep":
            try:
                requests.post(f"{self.base}/sleep", params={"level": 1}, timeout=60).raise_for_status()
                self._state = "sleeping"
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:  # the driver frees the memory a moment later
                    mem = gpu_memory_mb()
                    if mem is None or mem[0] < ASLEEP_MB:
                        break
                    time.sleep(0.3)
                return "sleeping"
            except requests.RequestException:
                pass  # could not sleep: free the GPU the hard way
        self.stop()
        return "stopped"

    def stop(self) -> None:
        """Stop a server we started, even if asleep, and wait until the GPU memory is released. No-op for someone else's."""
        proc = self._proc
        if proc is None:
            return
        self._state = "stopping"
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=40)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=15)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:  # the driver frees memory a moment after the process exits
            mem = gpu_memory_mb()
            if mem is None or mem[0] < 1500:
                break
            time.sleep(0.5)
        self._proc, self._state = None, "off"
