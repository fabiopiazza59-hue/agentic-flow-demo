"""
Run / Job / Event persistence.

`RunStore` is the interface the rest of the harness talks to. Two backends:

- InMemoryRunStore : fast, lost on restart (default, used by tests)
- SqliteRunStore   : stdlib sqlite3, durable. In-flight runs survive a
                     restart because the reconciler re-polls RUNNING jobs
                     at startup.

Both store whole entities as JSON so the schema never drifts from the
Pydantic models.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from typing import Iterable, Protocol

from .models import Job, JobStatus, Run, RunEvent, RunStatus, utcnow


class RunStore(Protocol):
    async def create_run(self, run: Run) -> Run: ...
    async def get_run(self, run_id: str) -> Run | None: ...
    async def save_run(self, run: Run) -> Run: ...
    async def list_runs(self, limit: int = 50) -> list[Run]: ...

    async def add_job(self, job: Job) -> Job: ...
    async def get_job(self, job_id: str) -> Job | None: ...
    async def save_job(self, job: Job) -> Job: ...
    async def jobs_for_run(self, run_id: str) -> list[Job]: ...
    async def jobs_with_status(self, statuses: Iterable[JobStatus]) -> list[Job]: ...

    async def append_event(self, event: RunEvent) -> RunEvent: ...
    async def events_for_run(self, run_id: str) -> list[RunEvent]: ...

    async def close(self) -> None: ...


# ---------------------------------------------------------------------------
# In-memory
# ---------------------------------------------------------------------------

class InMemoryRunStore:
    def __init__(self) -> None:
        self._runs: dict[str, Run] = {}
        self._jobs: dict[str, Job] = {}
        self._events: dict[str, list[RunEvent]] = {}
        self._lock = asyncio.Lock()

    async def create_run(self, run: Run) -> Run:
        async with self._lock:
            self._runs[run.run_id] = run.model_copy(deep=True)
            self._events.setdefault(run.run_id, [])
        return run

    async def get_run(self, run_id: str) -> Run | None:
        run = self._runs.get(run_id)
        return run.model_copy(deep=True) if run else None

    async def save_run(self, run: Run) -> Run:
        run.updated_at = utcnow()
        async with self._lock:
            self._runs[run.run_id] = run.model_copy(deep=True)
        return run

    async def list_runs(self, limit: int = 50) -> list[Run]:
        runs = sorted(self._runs.values(), key=lambda r: r.created_at, reverse=True)
        return [r.model_copy(deep=True) for r in runs[:limit]]

    async def add_job(self, job: Job) -> Job:
        async with self._lock:
            self._jobs[job.job_id] = job.model_copy(deep=True)
            run = self._runs.get(job.run_id)
            if run and job.job_id not in run.job_ids:
                run.job_ids.append(job.job_id)
        return job

    async def get_job(self, job_id: str) -> Job | None:
        job = self._jobs.get(job_id)
        return job.model_copy(deep=True) if job else None

    async def save_job(self, job: Job) -> Job:
        async with self._lock:
            self._jobs[job.job_id] = job.model_copy(deep=True)
        return job

    async def jobs_for_run(self, run_id: str) -> list[Job]:
        return [j.model_copy(deep=True) for j in self._jobs.values() if j.run_id == run_id]

    async def jobs_with_status(self, statuses: Iterable[JobStatus]) -> list[Job]:
        wanted = set(statuses)
        return [j.model_copy(deep=True) for j in self._jobs.values() if j.status in wanted]

    async def append_event(self, event: RunEvent) -> RunEvent:
        async with self._lock:
            events = self._events.setdefault(event.run_id, [])
            event.seq = len(events) + 1
            events.append(event.model_copy(deep=True))
        return event

    async def events_for_run(self, run_id: str) -> list[RunEvent]:
        return [e.model_copy(deep=True) for e in self._events.get(run_id, [])]

    async def close(self) -> None:
        return None


# ---------------------------------------------------------------------------
# SQLite (stdlib)
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    body TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    status TEXT NOT NULL,
    body TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_run_idx ON jobs(run_id);
CREATE INDEX IF NOT EXISTS jobs_status_idx ON jobs(status);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    body TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_run_idx ON events(run_id);
"""


class SqliteRunStore:
    """Durable store. All sqlite work runs in a worker thread behind one lock."""

    def __init__(self, path: str = "harness.db") -> None:
        self._path = path
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._lock = asyncio.Lock()

    async def _run(self, fn, *args):
        async with self._lock:
            return await asyncio.to_thread(fn, *args)

    # -- runs -------------------------------------------------------------
    async def create_run(self, run: Run) -> Run:
        return await self.save_run(run)

    async def save_run(self, run: Run) -> Run:
        run.updated_at = utcnow()

        def _save():
            self._conn.execute(
                "INSERT OR REPLACE INTO runs(run_id, status, created_at, body) VALUES (?,?,?,?)",
                (run.run_id, run.status.value, run.created_at.isoformat(), run.model_dump_json()),
            )
            self._conn.commit()

        await self._run(_save)
        return run

    async def get_run(self, run_id: str) -> Run | None:
        def _get():
            row = self._conn.execute("SELECT body FROM runs WHERE run_id=?", (run_id,)).fetchone()
            return Run.model_validate_json(row[0]) if row else None

        return await self._run(_get)

    async def list_runs(self, limit: int = 50) -> list[Run]:
        def _list():
            rows = self._conn.execute(
                "SELECT body FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [Run.model_validate_json(r[0]) for r in rows]

        return await self._run(_list)

    # -- jobs -------------------------------------------------------------
    async def add_job(self, job: Job) -> Job:
        def _add():
            self._conn.execute(
                "INSERT OR REPLACE INTO jobs(job_id, run_id, status, body) VALUES (?,?,?,?)",
                (job.job_id, job.run_id, job.status.value, job.model_dump_json()),
            )
            row = self._conn.execute("SELECT body FROM runs WHERE run_id=?", (job.run_id,)).fetchone()
            if row:
                run = Run.model_validate_json(row[0])
                if job.job_id not in run.job_ids:
                    run.job_ids.append(job.job_id)
                    self._conn.execute(
                        "UPDATE runs SET body=? WHERE run_id=?", (run.model_dump_json(), run.run_id)
                    )
            self._conn.commit()

        await self._run(_add)
        return job

    async def save_job(self, job: Job) -> Job:
        def _save():
            self._conn.execute(
                "INSERT OR REPLACE INTO jobs(job_id, run_id, status, body) VALUES (?,?,?,?)",
                (job.job_id, job.run_id, job.status.value, job.model_dump_json()),
            )
            self._conn.commit()

        await self._run(_save)
        return job

    async def get_job(self, job_id: str) -> Job | None:
        def _get():
            row = self._conn.execute("SELECT body FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            return Job.model_validate_json(row[0]) if row else None

        return await self._run(_get)

    async def jobs_for_run(self, run_id: str) -> list[Job]:
        def _list():
            rows = self._conn.execute("SELECT body FROM jobs WHERE run_id=?", (run_id,)).fetchall()
            return [Job.model_validate_json(r[0]) for r in rows]

        return await self._run(_list)

    async def jobs_with_status(self, statuses: Iterable[JobStatus]) -> list[Job]:
        wanted = [s.value for s in statuses]
        if not wanted:
            return []

        def _list():
            marks = ",".join("?" for _ in wanted)
            rows = self._conn.execute(f"SELECT body FROM jobs WHERE status IN ({marks})", wanted).fetchall()
            return [Job.model_validate_json(r[0]) for r in rows]

        return await self._run(_list)

    # -- events -----------------------------------------------------------
    async def append_event(self, event: RunEvent) -> RunEvent:
        def _append():
            row = self._conn.execute(
                "SELECT COALESCE(MAX(seq), 0) FROM events WHERE run_id=?", (event.run_id,)
            ).fetchone()
            event.seq = int(row[0]) + 1
            self._conn.execute(
                "INSERT INTO events(run_id, seq, body) VALUES (?,?,?)",
                (event.run_id, event.seq, event.model_dump_json()),
            )
            self._conn.commit()

        await self._run(_append)
        return event

    async def events_for_run(self, run_id: str) -> list[RunEvent]:
        def _list():
            rows = self._conn.execute(
                "SELECT body FROM events WHERE run_id=? ORDER BY seq", (run_id,)
            ).fetchall()
            return [RunEvent.model_validate_json(r[0]) for r in rows]

        return await self._run(_list)

    async def close(self) -> None:
        await self._run(self._conn.close)


def build_store(backend: str, sqlite_path: str = "harness.db") -> RunStore:
    if backend == "sqlite":
        return SqliteRunStore(sqlite_path)
    return InMemoryRunStore()


__all__ = ["RunStore", "InMemoryRunStore", "SqliteRunStore", "build_store", "RunStatus", "json"]
