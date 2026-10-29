"""One OCR job at a time: start the model, run the pipeline in its own process, report progress, free the GPU.

The pipeline (`python -m src.run`) is a child process, so its GPU memory (layout model) is returned when it
exits, a cancel is a signal, and a crash cannot take the web server down.
"""

from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from src.config import has_openrouter_key, has_sarvam_key

from .gpu import Cancelled, ModelError, ModelServer
from .intake import IntakeError, load_record

ROOT = Path(__file__).resolve().parent.parent
JOBS = ROOT / "out" / "web" / "jobs"
RUNS = ROOT / "out" / "web" / "runs"  # one folder per job, so a re-run or a cancelled run never wipes an earlier result
MODES = ("fast", "normal", "think")
TERMINAL = ("done", "failed", "cancelled")

PROGRESS = re.compile(r"^\[\s*(\d+)s\] pages (\d+)/(\d+) \| .*?\| ₹([\d.]+) \| ([\d.]+) pages/min \| ETA (.*)$")


class JobError(RuntimeError):
    """A job cannot start; the message is shown to the user."""


@dataclass
class Job:
    """One OCR job: its progress, result, log and the event that cancels it."""
    id: str
    file_id: str
    name: str
    mode: str
    pages_total: int
    estimate: dict
    cost_cap_inr: float
    state: str = "queued"  # queued, starting_model, reading, checking, done, failed, cancelled
    pages_done: int = 0
    cost_inr: float = 0.0
    pages_per_min: float = 0.0
    eta: str = ""
    error: str = ""
    out_dir: str = ""
    result: dict = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    reading_at: float = 0.0
    finished_at: float = 0.0
    model_action: str = ""  # "start" (loading it, about a minute) or "wake" (it was asleep, a moment)
    log: deque = field(default_factory=lambda: deque(maxlen=40))
    cancel: threading.Event = field(default_factory=threading.Event)

    def snapshot(self) -> dict:
        """The job as plain data for the API."""
        end = self.finished_at or time.time()
        return {
            "id": self.id, "name": self.name, "mode": self.mode, "state": self.state,
            "pages_done": self.pages_done, "pages_total": self.pages_total, "cost_inr": self.cost_inr,
            "pages_per_min": self.pages_per_min, "eta": self.eta, "error": self.error,
            "elapsed_s": round(end - self.started_at), "cost_cap_inr": self.cost_cap_inr, "model_action": self.model_action,
            "estimate": self.estimate, "result": self.result, "log": list(self.log),
        }


def cost_cap(estimate: dict) -> float:
    """A safety net a little above the estimate's top end, so a surprise cannot spend without limit."""
    return float(max(5, math.ceil(estimate["cost_inr"][1] * 1.5)))


class JobManager:
    """Runs one job at a time: wakes the model, runs the pipeline in a child process, reports progress and releases the GPU."""
    def __init__(self, model: ModelServer) -> None:
        self.model = model
        self._jobs: dict[str, Job] = {}
        self._current: Job | None = None
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    # -- public -----------------------------------------------------------

    @property
    def current(self) -> Job | None:
        """The job in progress, if any."""
        return self._current if self._current and self._current.state not in TERMINAL else None

    def start(self, file_id: str, mode: str) -> Job:
        """Start a job for an uploaded file. Raises JobError if it cannot start."""
        if mode not in MODES:
            raise JobError(f"Mode must be one of {', '.join(MODES)}.")
        try:
            record = load_record(file_id)
        except IntakeError as exc:
            raise JobError(str(exc)) from exc
        if not has_sarvam_key():
            raise JobError("The Sarvam API key is missing from .env. It is needed for the second reading of table pages.")
        with self._lock:
            if self.current is not None:
                raise JobError("Another document is being read. Wait for it to finish or cancel it.")
            estimate = record["estimates"]["modes"][mode]
            job = Job(uuid.uuid4().hex, file_id, record["name"], mode, record["pages"], estimate, cost_cap(estimate))
            self._jobs[job.id] = job
            self._current = job
        threading.Thread(target=self._run, args=(job, Path(record["path"])), name=f"job-{job.id[:6]}", daemon=True).start()
        return job

    def get(self, job_id: str) -> Job | None:
        """A job by id, including finished ones from earlier sessions."""
        if job_id in self._jobs:
            return self._jobs[job_id]
        saved = JOBS / f"{job_id}.json"
        if re.fullmatch(r"[0-9a-f]{32}", job_id) and saved.is_file():
            data = json.loads(saved.read_text(encoding="utf-8"))
            job = Job(data["id"], "", data["name"], data["mode"], data["pages_total"], data["estimate"], data["cost_cap_inr"])
            job.state, job.pages_done, job.cost_inr = data["state"], data["pages_done"], data["cost_inr"]
            job.out_dir, job.result, job.error = data["out_dir"], data["result"], data["error"]
            self._jobs[job_id] = job
            return job
        return None

    def cancel(self) -> bool:
        """Ask the running job to stop. Returns False if none is running."""
        job = self.current
        if job is None:
            return False
        job.cancel.set()
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.send_signal(signal.SIGINT)  # first: finish what is in flight
            threading.Timer(1.5, lambda: proc.poll() is None and proc.send_signal(signal.SIGINT)).start()  # second: leave now
        return True

    def shutdown(self) -> None:
        """Stop the job and the model. Called when the server exits."""
        self.cancel()
        proc = self._proc
        if proc is not None:
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
        self.model.stop()

    # -- the job ----------------------------------------------------------

    def _run(self, job: Job, pdf: Path) -> None:
        try:
            job.state = "starting_model"
            job.model_action = "wake" if self.model.state == "sleeping" else "start"
            job.log.append("Waking the model." if job.model_action == "wake" else "Starting the model on the GPU.")
            self.model.start(cancelled=job.cancel.is_set)
            if job.cancel.is_set():
                raise Cancelled()
            self._pipeline(job, pdf)
        except Cancelled:
            job.state = "cancelled"
        except ModelError as exc:
            job.state, job.error = "failed", str(exc)
        except Exception as exc:  # noqa: BLE001 - the job must always end in a terminal state
            job.state, job.error = "failed", f"{type(exc).__name__}: {exc}"
        finally:
            self._proc = None
            self.model.release()  # the GPU is free (model asleep) before the page says "done"
            job.finished_at = time.time()
            if job.state not in TERMINAL:
                job.state = "failed"
            self._save(job)

    def _pipeline(self, job: Job, pdf: Path) -> None:
        cmd = [sys.executable, "-u", "-m", "src.run", str(pdf), "--mode", job.mode, "--fresh", "-y",
               "--max-cost-inr", str(job.cost_cap_inr), "--out-root", str(RUNS / job.id)]
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        job.log.append("Reading the document.")
        self._proc = proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                             text=True, bufsize=1)
        job.state, job.reading_at = "reading", time.time()
        for line in proc.stdout:
            self._on_line(job, line.rstrip())
        code = proc.wait()
        if job.cancel.is_set():
            raise Cancelled()
        if code not in (0, 3):  # 3: stopped at the cost cap, a partial but valid result
            tail = "\n".join(list(job.log)[-6:])
            raise RuntimeError(f"The reading stopped with an error (exit code {code}).\n{tail}")
        self._collect(job, code)

    def _on_line(self, job: Job, line: str) -> None:
        if not line.strip():
            return
        m = PROGRESS.match(line)
        if m:
            _, done, total, cost, rate, eta = m.groups()
            job.pages_done, job.cost_inr, job.pages_per_min, job.eta = int(done), float(cost), float(rate), eta
            return
        job.log.append(line[:300])
        if line.startswith("assembling"):
            job.state = "checking"
        elif line.startswith("Outputs:"):
            job.out_dir = line.split(":", 1)[1].strip()

    def _collect(self, job: Job, code: int) -> None:
        out = Path(job.out_dir)
        summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        assembly = {}
        if (out / "assembly.json").is_file():
            assembly = json.loads((out / "assembly.json").read_text(encoding="utf-8"))
        tables = assembly.get("tables", {})
        total_inr = (assembly.get("cost") or {}).get("total_inr", summary["cost_inr"])
        seconds = round(summary["wall_s"] + (assembly.get("assembly_wall_s") or 0))
        job.pages_done, job.cost_inr = summary["pages_done"], total_inr
        job.result = {
            "pages_done": summary["pages_done"], "pages_total": summary["pages_in_run"], "complete": summary["complete"],
            "stopped": summary["stopped"], "cost_inr": total_inr, "seconds": seconds,
            "pages_per_min": summary["pages_per_min"],
            "tables": {"found": tables.get("found", 0), "after_joining": tables.get("after_joining", 0),
                       "status": tables.get("qa_status", {})},
            "has_doc": (out / "doc.md").is_file(), "has_report": (out / "report.md").is_file(),
        }
        job.state = "done"

    def _save(self, job: Job) -> None:
        JOBS.mkdir(parents=True, exist_ok=True)
        data = {k: v for k, v in job.snapshot().items() if k not in ("log", "elapsed_s")}
        data["out_dir"] = job.out_dir
        (JOBS / f"{job.id}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def available() -> dict:
    """Which of the API keys the pipeline needs are set."""
    return {"sarvam_key": has_sarvam_key(), "openrouter_key": has_openrouter_key()}
