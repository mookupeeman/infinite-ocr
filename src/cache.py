"""Disk cache. Key is sha256(bytes + backend + model + extra)."""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path

from src.config import CACHE_DIR


def make_key(payload: bytes, backend: str, model: str, extra: str = "") -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    digest.update(b"\0")
    digest.update(backend.encode("utf-8"))
    digest.update(b"\0")
    digest.update(model.encode("utf-8"))
    digest.update(b"\0")
    digest.update(extra.encode("utf-8"))
    return digest.hexdigest()


def cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / key


def get(key: str) -> bytes | None:
    path = cache_path(key)
    if not path.is_file():
        return None
    return path.read_bytes()


def put(key: str, data: bytes) -> Path:
    """Write atomically: a crash never leaves a half-written entry."""
    path = cache_path(key)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return path


def hit(key: str) -> bool:
    return cache_path(key).is_file()
