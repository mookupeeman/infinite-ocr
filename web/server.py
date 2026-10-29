"""HTTP API for the local web app. Serves web/static and drives one job at a time.

    GET  /api/status                       model and GPU state, which keys are set, the current job
    POST /api/files                        upload a PDF or photo; returns pages and a cost/time estimate per mode
    POST /api/jobs                         {file_id, mode}: start the model, read the document
    GET  /api/jobs/{id}                    progress, then the result
    POST /api/jobs/{id}/cancel
    GET  /api/jobs/{id}/pages/{n}          blocks of one page (inspector)
    GET  /api/jobs/{id}/pages/{n}/image    the page, upright
    GET  /api/jobs/{id}/download/{doc|report}
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src import modes
from src.config import BODHAN_BACKEND

from . import intake, pages
from .gpu import ModelServer, gpu_memory_mb
from .jobs import JobError, JobManager, available

STATIC = Path(__file__).resolve().parent / "static"

model = ModelServer()
manager = JobManager(model)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    manager.shutdown()  # leave the GPU free when the server stops


app = FastAPI(title="Infinite-OCR", lifespan=lifespan)


class StartJob(BaseModel):
    """Body of POST /api/jobs."""
    file_id: str
    mode: Literal["fast", "normal", "think"]


def _job(job_id: str):
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job.")
    return job


@app.get("/api/status")
def status() -> dict:
    """Model and GPU state, which keys are set, and the current job."""
    mem = gpu_memory_mb()
    current = manager.current
    return {
        "model": {"state": model.state, "managed": BODHAN_BACKEND == "local", "lead": modes.default_lead()},
        "gpu_used_mb": mem[0] if mem else None,
        "gpu_total_mb": mem[1] if mem else None,
        "keys": available(),
        "job": current.id if current else None,
    }


@app.post("/api/files")
def upload(file: UploadFile = File(...)) -> dict:
    """Check an uploaded PDF or photo and return its page count and estimates."""
    try:
        record = intake.save_upload(file.file, file.filename or "document")
    except intake.IntakeError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {k: record[k] for k in ("id", "name", "pages", "bytes", "estimates")} | {
        "model_state": model.state}


@app.post("/api/jobs")
def start(body: StartJob) -> dict:
    """Start reading an uploaded file in the chosen mode."""
    try:
        return manager.start(body.file_id, body.mode).snapshot()
    except JobError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    """Progress of a job, then its result."""
    return _job(job_id).snapshot()


@app.post("/api/jobs/{job_id}/cancel")
def cancel(job_id: str) -> dict:
    """Cancel the running job."""
    _job(job_id)
    return {"cancelled": manager.cancel()}


def _done(job_id: str) -> Path:
    job = _job(job_id)
    if job.state != "done" or not job.out_dir:
        raise HTTPException(409, "The job has not finished.")
    return Path(job.out_dir)


@app.get("/api/jobs/{job_id}/pages/{page}")
def page_blocks(job_id: str, page: int) -> dict:
    """Blocks of one page of a finished job."""
    try:
        return pages.page_payload(_done(job_id), page)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/api/jobs/{job_id}/pages/{page}/image")
def page_picture(job_id: str, page: int) -> FileResponse:
    """The upright page image of a finished job."""
    try:
        return FileResponse(pages.page_image(_done(job_id), page), media_type="image/jpeg")
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/api/jobs/{job_id}/download/{which}")
def download(job_id: str, which: Literal["doc", "report"]) -> FileResponse:
    """The document text or the report of a finished job."""
    name = {"doc": "doc.md", "report": "report.md"}[which]
    path = _done(job_id) / name
    if not path.is_file():
        raise HTTPException(404, "Not available.")
    return FileResponse(path, media_type="text/markdown", filename=f"{_job(job_id).name}-{name}")


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
