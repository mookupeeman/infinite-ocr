"""Uploaded files: check them, turn a photo into a one-page PDF, and estimate cost and time. No GPU, no API calls."""

from __future__ import annotations

import io
import json
import re
import uuid
from pathlib import Path
from typing import BinaryIO

import pymupdf
from PIL import Image, ImageOps

from src import estimate, modes
from src.run import slug

ROOT = Path(__file__).resolve().parent.parent
UPLOADS = ROOT / "out" / "web" / "uploads"
MAX_BYTES = 300 * 1024 * 1024
FILE_ID = re.compile(r"^[0-9a-f]{32}$")
IMAGE_KINDS = {b"\xff\xd8\xff": "JPEG", b"\x89PNG": "PNG"}


class IntakeError(ValueError):
    """The file cannot be read; the message says what to do."""


def _read_limited(stream: BinaryIO) -> bytes:
    data = stream.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise IntakeError(f"The file is larger than {MAX_BYTES // 2**20} MB.")
    if not data:
        raise IntakeError("The file is empty.")
    return data


def _image_to_pdf(data: bytes) -> bytes:
    """One page the size of the photo, turned upright if the camera stored it sideways."""
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except Exception as exc:  # noqa: BLE001 - PIL raises many types for a bad image
        raise IntakeError("The image could not be opened. Try saving it again as JPEG or PNG.") from exc
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)
    with pymupdf.open(stream=buf.getvalue(), filetype="jpeg") as image_doc:
        return image_doc.convert_to_pdf()


def check_pdf(path: Path) -> int:
    """Page count of a readable, unprotected PDF."""
    try:
        with pymupdf.open(path) as doc:
            if doc.needs_pass:
                raise IntakeError("The PDF is password protected. Remove the password and upload it again.")
            if doc.page_count == 0:
                raise IntakeError("The PDF has no pages.")
            doc[0].get_pixmap(dpi=20)  # a broken first page shows up here, not in the middle of a job
            return doc.page_count
    except IntakeError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise IntakeError("The PDF could not be opened. It may be damaged.") from exc


def estimates_for(path: Path, pages: int) -> dict:
    """Cost and time estimates for each mode, from a free look at up to 40 pages."""
    with pymupdf.open(path) as doc:
        pre = estimate.preflight(doc, list(range(1, pages + 1)))
    by_mode = {e["mode"]: e for e in estimate.estimate_all(pre, lead=modes.default_lead())}
    return {
        "lead": modes.default_lead(),
        "classes": {k: round(v) for k, v in pre.classes.items()},
        "table_pages": [round(v) for v in pre.table_range],
        "modes": {m: {"cost_inr": e["total_inr"], "minutes": e["minutes"]} for m, e in by_mode.items()},
        "notes": pre.notes,
    }


def save_upload(stream: BinaryIO, filename: str) -> dict:
    """Store the upload as a PDF under out/web/uploads/<id>/ and return its record."""
    data = _read_limited(stream)
    head = data[:1024]
    if b"%PDF-" in head:
        pdf = data
    elif any(data.startswith(magic) for magic in IMAGE_KINDS):
        pdf = _image_to_pdf(data)
    else:
        raise IntakeError("This is not a PDF, JPEG or PNG file.")
    file_id = uuid.uuid4().hex
    stem = slug(Path(filename or "document").stem)
    folder = UPLOADS / file_id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stem}.pdf"
    path.write_bytes(pdf)
    try:
        pages = check_pdf(path)
        record = {"id": file_id, "name": filename or path.name, "path": str(path), "pages": pages,
                  "bytes": len(data), "estimates": estimates_for(path, pages)}
    except Exception:
        path.unlink(missing_ok=True)
        folder.rmdir()
        raise
    (folder / "file.json").write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    return record


def load_record(file_id: str) -> dict:
    """The saved record of an upload. Raises IntakeError for an unknown id."""
    if not FILE_ID.match(file_id):
        raise IntakeError("Unknown file.")
    meta = UPLOADS / file_id / "file.json"
    if not meta.is_file():
        raise IntakeError("That file is no longer on the server. Upload it again.")
    return json.loads(meta.read_text(encoding="utf-8"))
