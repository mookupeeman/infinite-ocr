"""Run state for one document: SQLite (WAL) plus an append-only events log.

Tables
- pages: triage flags per page.
- tasks: one row per (page, engine).
  States: pending -> inflight (page engines) or queued (Sarvam, inside a
  job) -> done | failed. Also skipped (blank page) and stolen (a Sarvam task
  that Bodhan took over).
- jobs: Sarvam jobs. States: submitting -> submitted (job_id saved) ->
  done | failed | abandoned.

Rules: one shared connection behind a lock; no network call inside a
transaction; result files are written before the row that points to them.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS pages(
    page INTEGER PRIMARY KEY, rotate INTEGER, blank INTEGER, text_chars INTEGER,
    script_share REAL, legacy_font INTEGER, script TEXT, language TEXT,
    width_pt REAL, height_pt REAL, table_bboxes TEXT,
    ocr_mode TEXT, mode_source TEXT, lang_hint TEXT, detected TEXT,
    mode_retried INTEGER NOT NULL DEFAULT 0, render TEXT, text_layer TEXT, image_cover REAL);
CREATE TABLE IF NOT EXISTS tasks(
    page INTEGER NOT NULL, engine TEXT NOT NULL, state TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 0, attempts INTEGER NOT NULL DEFAULT 0,
    reason TEXT, job INTEGER, result_path TEXT, latency_s REAL, tokens INTEGER,
    cost_inr REAL NOT NULL DEFAULT 0, cache_hit INTEGER NOT NULL DEFAULT 0,
    degraded TEXT, updated_at REAL, PRIMARY KEY(page, engine));
CREATE INDEX IF NOT EXISTS tasks_by_state ON tasks(engine, state, priority, page);
CREATE TABLE IF NOT EXISTS extras(
    id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, page INTEGER,
    cost_inr REAL NOT NULL DEFAULT 0, detail TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS jobs(
    id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT, state TEXT NOT NULL,
    language TEXT, files TEXT NOT NULL, n_pages INTEGER, status TEXT, error TEXT,
    created_at REAL, submitted_at REAL, finished_at REAL);
"""

ACTIVE = ("pending", "inflight", "queued")

# Columns added after the first schema: (table, column, type). Older run
# databases are upgraded in place when opened.
MIGRATIONS = [
    ("pages", "ocr_mode", "TEXT"),
    ("pages", "mode_source", "TEXT"),
    ("pages", "lang_hint", "TEXT"),
    ("pages", "detected", "TEXT"),
    ("pages", "mode_retried", "INTEGER NOT NULL DEFAULT 0"),
    ("pages", "render", "TEXT"),  # page_class: vector | raster
    ("pages", "text_layer", "TEXT"),  # page_class: unicode | legacy | hidden_ocr | none | ignored
    ("pages", "image_cover", "REAL"),
]


class EventLog:
    """events.jsonl: one JSON object per line, for metrics and the report."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, kind: str, **fields: Any) -> None:
        line = json.dumps({"t": round(time.time(), 3), "kind": kind, **fields}, ensure_ascii=False)
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def read(self) -> list[dict]:
        if not self.path.is_file():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line]


class Manifest:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None, timeout=30)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute("PRAGMA busy_timeout=30000")
        self._db.executescript(SCHEMA)
        for table, column, kind in MIGRATIONS:
            have = {r["name"] for r in self._db.execute(f"PRAGMA table_info({table})")}
            if column not in have:
                self._db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")

    def close(self) -> None:
        with self._lock:
            self._db.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield self._db
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
            self._db.execute("COMMIT")

    def _rows(self, sql: str, args: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._db.execute(sql, args).fetchall()

    # -- meta -------------------------------------------------------------

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        rows = self._rows("SELECT value FROM meta WHERE key=?", (key,))
        return rows[0]["value"] if rows else default

    def set_meta(self, key: str, value: str) -> None:
        with self._tx() as db:
            db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)", (key, value))

    # -- pages ------------------------------------------------------------

    def triaged_pages(self) -> set[int]:
        return {r["page"] for r in self._rows("SELECT page FROM pages")}

    def add_page(self, row: dict, tasks: dict[str, tuple[str, int]]) -> None:
        """Insert a triaged page and its tasks: engine -> (state, priority)."""
        cols = {k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v) for k, v in row.items()}
        cols.setdefault("table_bboxes", "[]")
        now = time.time()
        with self._tx() as db:
            names = ", ".join(cols)
            marks = ", ".join("?" for _ in cols)
            db.execute(f"INSERT OR REPLACE INTO pages({names}) VALUES({marks})", list(cols.values()))
            for engine, (state, priority) in tasks.items():
                db.execute(
                    "INSERT OR IGNORE INTO tasks(page, engine, state, priority, updated_at) VALUES(?,?,?,?,?)",
                    (row["page"], engine, state, priority, now),
                )

    def page_info(self, page: int) -> dict:
        rows = self._rows("SELECT * FROM pages WHERE page=?", (page,))
        if not rows:
            raise KeyError(page)
        info = dict(rows[0])
        info["table_bboxes"] = json.loads(info.get("table_bboxes") or "[]")
        for key in ("lang_hint", "detected"):
            info[key] = json.loads(info[key]) if info.get(key) else None
        return info

    def update_detected(self, page: int, engine: str, languages: dict) -> None:
        """Languages found in one engine's output for a page (merged per engine)."""
        with self._tx() as db:
            row = db.execute("SELECT detected FROM pages WHERE page=?", (page,)).fetchone()
            current = json.loads(row["detected"]) if row and row["detected"] else {}
            current[engine] = languages
            db.execute("UPDATE pages SET detected=? WHERE page=?", (json.dumps(current, ensure_ascii=False), page))

    def set_page_mode(self, page: int, ocr_mode: str, language: str, source: str, retried: bool = False) -> None:
        with self._tx() as db:
            db.execute(
                "UPDATE pages SET ocr_mode=?, language=?, mode_source=?, mode_retried=? WHERE page=?",
                (ocr_mode, language, source, int(retried), page),
            )

    def all_pages(self) -> list[dict]:
        out = []
        for r in self._rows("SELECT * FROM pages ORDER BY page"):
            d = dict(r)
            for key in ("lang_hint", "detected"):
                d[key] = json.loads(d[key]) if d.get(key) else None
            out.append(d)
        return out

    # -- tasks ------------------------------------------------------------

    def task(self, page: int, engine: str) -> dict | None:
        rows = self._rows("SELECT * FROM tasks WHERE page=? AND engine=?", (page, engine))
        return dict(rows[0]) if rows else None

    def ensure_task(self, page: int, engine: str, priority: int = 0) -> bool:
        """Create a pending task if the page has none for this engine."""
        with self._tx() as db:
            cur = db.execute(
                "INSERT OR IGNORE INTO tasks(page, engine, state, priority, updated_at) VALUES(?,?, 'pending', ?, ?)",
                (page, engine, priority, time.time()),
            )
            return cur.rowcount == 1

    def count(self, engine: str | None = None, states: Sequence[str] = ACTIVE) -> int:
        marks = ",".join("?" for _ in states)
        sql = f"SELECT COUNT(*) AS n FROM tasks WHERE state IN ({marks})"
        args: list[Any] = list(states)
        if engine is not None:
            sql += " AND engine=?"
            args.append(engine)
        return self._rows(sql, args)[0]["n"]

    def claim_sarvam(self, max_pages: int) -> tuple[str | None, list[int]]:
        """Head of the Sarvam queue, all one language. Marks them queued."""
        with self._tx() as db:
            first = db.execute(
                "SELECT p.language FROM tasks t JOIN pages p USING(page) "
                "WHERE t.engine='sarvam' AND t.state='pending' ORDER BY t.priority DESC, t.page LIMIT 1"
            ).fetchone()
            if first is None:
                return None, []
            language = first["language"]
            pages = [
                r["page"]
                for r in db.execute(
                    "SELECT t.page FROM tasks t JOIN pages p USING(page) "
                    "WHERE t.engine='sarvam' AND t.state='pending' AND p.language=? "
                    "ORDER BY t.priority DESC, t.page LIMIT ?",
                    (language, max_pages),
                )
            ]
            db.executemany(
                "UPDATE tasks SET state='queued', updated_at=? WHERE page=? AND engine='sarvam'",
                [(time.time(), p) for p in pages],
            )
            return language, pages

    def claim_page_task(
        self, engine: str, *, steal_from: str | None = None, steal_min_backlog: int = 0
    ) -> tuple[int, bool] | None:
        """Next page for a per-page engine. Returns (page, stolen) or None.

        Own pending tasks come first (highest priority, lowest page). When
        there are none, and `steal_from` has a backlog larger than
        `steal_min_backlog`, take that engine's *last* pending page.
        """
        now = time.time()
        with self._tx() as db:
            row = db.execute(
                "SELECT page FROM tasks WHERE engine=? AND state='pending' ORDER BY priority DESC, page LIMIT 1",
                (engine,),
            ).fetchone()
            if row is not None:
                db.execute(
                    "UPDATE tasks SET state='inflight', updated_at=? WHERE page=? AND engine=?",
                    (now, row["page"], engine),
                )
                return row["page"], False
            if steal_from is None:
                return None
            backlog = db.execute(
                "SELECT COUNT(*) AS n FROM tasks WHERE engine=? AND state='pending'", (steal_from,)
            ).fetchone()["n"]
            if backlog <= steal_min_backlog:
                return None
            victim = db.execute(
                "SELECT page FROM tasks WHERE engine=? AND state='pending' ORDER BY page DESC LIMIT 1",
                (steal_from,),
            ).fetchone()
            page = victim["page"]
            db.execute(
                "UPDATE tasks SET state='stolen', reason=?, updated_at=? WHERE page=? AND engine=?",
                (f"taken by {engine}", now, page, steal_from),
            )
            db.execute(
                "INSERT INTO tasks(page, engine, state, priority, updated_at) VALUES(?,?, 'inflight', 0, ?) "
                "ON CONFLICT(page, engine) DO UPDATE SET state='inflight', updated_at=excluded.updated_at",
                (page, engine, now),
            )
            return page, True

    def finish(self, page: int, engine: str, *, state: str = "done", **fields: Any) -> None:
        allowed = {"result_path", "latency_s", "tokens", "cost_inr", "cache_hit", "degraded", "reason", "job"}
        sets = ["state=?", "updated_at=?"]
        args: list[Any] = [state, time.time()]
        for key, value in fields.items():
            if key not in allowed:
                raise KeyError(key)
            sets.append(f"{key}=?")
            args.append(json.dumps(value) if isinstance(value, (list, dict)) else value)
        args += [page, engine]
        with self._tx() as db:
            db.execute(f"UPDATE tasks SET {', '.join(sets)} WHERE page=? AND engine=?", args)

    def retry_or_fail(self, page: int, engine: str, reason: str, max_attempts: int) -> str:
        """Count a failed attempt. Back to pending, or failed after `max_attempts`."""
        with self._tx() as db:
            row = db.execute(
                "SELECT attempts FROM tasks WHERE page=? AND engine=?", (page, engine)
            ).fetchone()
            attempts = (row["attempts"] if row else 0) + 1
            state = "failed" if attempts >= max_attempts else "pending"
            db.execute(
                "UPDATE tasks SET state=?, attempts=?, reason=?, job=NULL, updated_at=? WHERE page=? AND engine=?",
                (state, attempts, reason, time.time(), page, engine),
            )
            return state

    def release(self, pages: Sequence[int], engine: str, reason: str = "") -> None:
        """Back to pending without counting an attempt (budget stop, 429 storm)."""
        with self._tx() as db:
            db.executemany(
                "UPDATE tasks SET state='pending', job=NULL, reason=?, updated_at=? WHERE page=? AND engine=?",
                [(reason, time.time(), p, engine) for p in pages],
            )

    def reopen_stolen(self, page: int, owner: str = "sarvam") -> bool:
        with self._tx() as db:
            cur = db.execute(
                "UPDATE tasks SET state='pending', reason='reopened', updated_at=? "
                "WHERE page=? AND engine=? AND state='stolen'",
                (time.time(), page, owner),
            )
            return cur.rowcount == 1

    def fail_lane(self, engine: str, reason: str) -> list[int]:
        """An engine is unusable (bad key, no credit): fail its open tasks."""
        with self._tx() as db:
            pages = [
                r["page"]
                for r in db.execute(
                    "SELECT page FROM tasks WHERE engine=? AND state IN ('pending','inflight','queued')",
                    (engine,),
                )
            ]
            db.execute(
                "UPDATE tasks SET state='failed', reason=?, updated_at=? "
                "WHERE engine=? AND state IN ('pending','inflight','queued')",
                (reason, time.time(), engine),
            )
            return pages

    def requeue_inflight(self) -> int:
        """On resume: page tasks that were in flight go back to pending."""
        with self._tx() as db:
            cur = db.execute(
                "UPDATE tasks SET state='pending', updated_at=? WHERE state='inflight'", (time.time(),)
            )
            return cur.rowcount

    def queued(self, engine: str) -> dict[int, int | None]:
        """Queued tasks: page -> job row id."""
        rows = self._rows("SELECT page, job FROM tasks WHERE engine=? AND state='queued'", (engine,))
        return {r["page"]: r["job"] for r in rows}

    def pages_done_by_any(self, pages: Sequence[int] | None = None) -> int:
        rows = self._rows("SELECT DISTINCT page FROM tasks WHERE state='done'")
        done = {r["page"] for r in rows}
        return len(done if pages is None else done & set(pages))

    def counts(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for r in self._rows("SELECT engine, state, COUNT(*) AS n FROM tasks GROUP BY engine, state"):
            out.setdefault(r["engine"], {})[r["state"]] = r["n"]
        return out

    def add_extra(self, kind: str, cost_inr: float, page: int | None = None, detail: Any = None) -> None:
        """Paid calls outside page tasks: table re-reads, agent and vision calls."""
        with self._tx() as db:
            db.execute(
                "INSERT INTO extras(kind, page, cost_inr, detail, created_at) VALUES(?,?,?,?,?)",
                (kind, page, cost_inr, json.dumps(detail, ensure_ascii=False) if detail is not None else None, time.time()),
            )

    def extras_by_kind(self) -> dict[str, dict[str, float]]:
        rows = self._rows("SELECT kind, COUNT(*) AS n, COALESCE(SUM(cost_inr),0) AS s FROM extras GROUP BY kind")
        return {r["kind"]: {"calls": r["n"], "cost_inr": round(r["s"], 4)} for r in rows}

    def spent_inr(self, engine: str | None = None) -> float:
        if engine is None:
            tasks = float(self._rows("SELECT COALESCE(SUM(cost_inr),0) AS s FROM tasks")[0]["s"])
            extras = float(self._rows("SELECT COALESCE(SUM(cost_inr),0) AS s FROM extras")[0]["s"])
            return tasks + extras
        return float(
            self._rows("SELECT COALESCE(SUM(cost_inr),0) AS s FROM tasks WHERE engine=?", (engine,))[0]["s"]
        )

    def cache_hits(self, engine: str) -> int:
        return self._rows(
            "SELECT COUNT(*) AS n FROM tasks WHERE engine=? AND cache_hit=1 AND state='done'", (engine,)
        )[0]["n"]

    def tasks_for_page(self, page: int) -> dict[str, dict]:
        return {r["engine"]: dict(r) for r in self._rows("SELECT * FROM tasks WHERE page=?", (page,))}

    # -- Sarvam jobs ------------------------------------------------------

    def new_job(self, files: list[list[int]], language: str) -> int:
        """Record a job *before* submitting it, so a crash can never lose one."""
        n_pages = sum(len(f) for f in files)
        with self._tx() as db:
            cur = db.execute(
                "INSERT INTO jobs(state, language, files, n_pages, created_at) VALUES('submitting',?,?,?,?)",
                (language, json.dumps(files), n_pages, time.time()),
            )
            job = int(cur.lastrowid)
            db.executemany(
                "UPDATE tasks SET job=? WHERE page=? AND engine='sarvam'",
                [(job, p) for f in files for p in f],
            )
            return job

    def job_submitted(self, job: int, job_id: str) -> None:
        with self._tx() as db:
            db.execute(
                "UPDATE jobs SET job_id=?, state='submitted', submitted_at=? WHERE id=?",
                (job_id, time.time(), job),
            )

    def job_finished(self, job: int, state: str, status: str | None = None, error: str | None = None) -> None:
        with self._tx() as db:
            db.execute(
                "UPDATE jobs SET state=?, status=?, error=?, finished_at=? WHERE id=?",
                (state, status, error, time.time(), job),
            )

    def job(self, job: int) -> dict:
        row = dict(self._rows("SELECT * FROM jobs WHERE id=?", (job,))[0])
        row["files"] = json.loads(row["files"])
        return row

    def open_jobs(self) -> list[dict]:
        rows = self._rows("SELECT * FROM jobs WHERE state IN ('submitting','submitted') ORDER BY id")
        out = []
        for r in rows:
            d = dict(r)
            d["files"] = json.loads(d["files"])
            out.append(d)
        return out

    def jobs(self) -> list[dict]:
        return [dict(r) for r in self._rows("SELECT * FROM jobs ORDER BY id")]
