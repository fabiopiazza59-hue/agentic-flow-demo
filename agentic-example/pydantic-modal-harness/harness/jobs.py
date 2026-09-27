"""
Job lifecycle: dispatch -> (webhook | poll) -> idempotent completion -> resume.

Delivery strategy ("webhook first, poll fallback"):

1. The worker POSTs a signed `JobResultEnvelope` to `/webhooks/modal` the
   moment it finishes. This is the fast path (sub-second).
2. The reconciler periodically polls the dispatcher for every job still
   RUNNING that is older than a grace period. This catches lost webhooks,
   workers that could not reach the harness (local dev without a public
   URL), and harness restarts (SQLite store).

Both paths call `handle_result`, which is idempotent: the first terminal
result wins, later duplicates are acknowledged and ignored. When every job
of a run is terminal, the run is moved to RESUMING exactly once (per-run
lock) and the orchestrator's resume callback is scheduled.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Awaitable, Callable

from .config import Settings
from .dispatch import JobDispatcher
from .events import EventBus
from .models import Job, JobKind, JobResultEnvelope, JobStatus, RunEvent, RunStatus, utcnow
from .store import RunStore

log = logging.getLogger("harness.jobs")

ResumeCallback = Callable[[str], Awaitable[None]]


class JobManager:
    def __init__(self, store: RunStore, bus: EventBus, dispatcher: JobDispatcher, settings: Settings) -> None:
        self.store = store
        self.bus = bus
        self.dispatcher = dispatcher
        self.settings = settings
        self._resume_callback: ResumeCallback | None = None
        self._run_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._background: set[asyncio.Task[Any]] = set()
        self._reconciler: asyncio.Task[None] | None = None

    # -- wiring -----------------------------------------------------------
    def set_resume_callback(self, callback: ResumeCallback) -> None:
        self._resume_callback = callback

    async def emit(self, run_id: str, event_type: str, **data: Any) -> RunEvent:
        event = await self.store.append_event(RunEvent(run_id=run_id, type=event_type, data=data))
        self.bus.publish(event)
        return event

    # -- dispatch ---------------------------------------------------------
    async def dispatch(self, *, run_id: str, tool_call_id: str, tool_name: str,
                       kind: JobKind, payload: dict[str, Any]) -> Job:
        job = Job(run_id=run_id, tool_call_id=tool_call_id, tool_name=tool_name, kind=kind,
                  payload=payload, backend=self.dispatcher.backend)
        await self.store.add_job(job)
        await self.emit(run_id, "job.created", job_id=job.job_id, kind=kind.value, tool=tool_name,
                        backend=job.backend)
        try:
            job.external_id = await self.dispatcher.submit(job)
            job.status = JobStatus.RUNNING
            job.dispatched_at = utcnow()
            await self.store.save_job(job)
            await self.emit(run_id, "job.dispatched", job_id=job.job_id, kind=kind.value,
                            backend=job.backend, external_id=job.external_id)
        except Exception as exc:  # noqa: BLE001 - surface as a failed job, not a crash
            log.exception("dispatch failed for job %s", job.job_id)
            job.status = JobStatus.FAILED
            job.error = f"dispatch failed: {type(exc).__name__}: {exc}"
            job.completed_at = utcnow()
            await self.store.save_job(job)
            await self.emit(run_id, "job.failed", job_id=job.job_id, error=job.error, delivered_via="dispatch")
        return job

    # -- result handling (idempotent) -------------------------------------
    async def handle_result(self, envelope: JobResultEnvelope, via: str) -> dict[str, Any]:
        job = await self.store.get_job(envelope.job_id)
        if job is None:
            return {"accepted": False, "reason": "unknown job"}
        if job.run_id != envelope.run_id:
            return {"accepted": False, "reason": "run id mismatch"}
        if job.status.is_terminal:
            return {"accepted": True, "duplicate": True, "status": job.status.value}

        job.status = JobStatus.SUCCEEDED if envelope.status == "succeeded" else JobStatus.FAILED
        job.result = envelope.result
        job.error = envelope.error
        job.worker_meta = envelope.worker
        job.completed_at = utcnow()
        job.delivered_via = via  # type: ignore[assignment]
        if envelope.external_id and not job.external_id:
            job.external_id = envelope.external_id
        await self.store.save_job(job)

        event_type = "job.completed" if job.status == JobStatus.SUCCEEDED else "job.failed"
        await self.emit(job.run_id, event_type, job_id=job.job_id, kind=job.kind.value,
                        delivered_via=via, error=job.error,
                        summary=_summarise(job.result) if job.status == JobStatus.SUCCEEDED else None,
                        worker=job.worker_meta)
        await self.check_resume(job.run_id)
        return {"accepted": True, "duplicate": False, "status": job.status.value}

    async def check_resume(self, run_id: str) -> bool:
        """Resume the run if it is suspended and all of its jobs are terminal. Safe to call often."""
        async with self._run_locks[run_id]:
            run = await self.store.get_run(run_id)
            if run is None or run.status != RunStatus.WAITING_FOR_JOBS:
                return False
            jobs = await self.store.jobs_for_run(run_id)
            pending = [j for j in jobs if not j.status.is_terminal]
            if pending:
                return False
            run.status = RunStatus.RESUMING
            await self.store.save_run(run)
        if self._resume_callback is None:
            log.warning("no resume callback configured; run %s stays in RESUMING", run_id)
            return False
        task = asyncio.create_task(self._resume_callback(run_id), name=f"resume-{run_id}")
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        return True

    # -- cancellation -----------------------------------------------------
    async def cancel_jobs_for_run(self, run_id: str) -> int:
        cancelled = 0
        for job in await self.store.jobs_for_run(run_id):
            if job.status.is_terminal:
                continue
            try:
                await self.dispatcher.cancel(job)
            except Exception as exc:  # noqa: BLE001
                log.warning("cancel failed for job %s: %s", job.job_id, exc)
            job.status = JobStatus.CANCELLED
            job.completed_at = utcnow()
            await self.store.save_job(job)
            await self.emit(run_id, "job.cancelled", job_id=job.job_id)
            cancelled += 1
        return cancelled

    # -- reconciliation (poll fallback) -----------------------------------
    async def reconcile_once(self, *, ignore_grace: bool = False) -> int:
        """Poll every RUNNING job older than the grace period. Returns number of jobs resolved."""
        resolved = 0
        now = utcnow()
        for job in await self.store.jobs_with_status([JobStatus.RUNNING]):
            if not ignore_grace and job.age_seconds(now) < self.settings.reconcile_grace_seconds:
                continue
            try:
                outcome = await self.dispatcher.poll(job)
            except Exception as exc:  # noqa: BLE001
                log.warning("poll failed for job %s: %s", job.job_id, exc)
                continue
            if outcome.state == "pending":
                continue
            envelope = JobResultEnvelope(
                job_id=job.job_id, run_id=job.run_id,
                status="succeeded" if outcome.state == "succeeded" else "failed",
                result=outcome.result,
                error=outcome.error if outcome.state != "succeeded" else None,
                external_id=job.external_id, worker=outcome.worker,
            )
            if outcome.state == "expired":
                envelope.error = outcome.error or "output expired"
            response = await self.handle_result(envelope, via="poll")
            if response.get("accepted") and not response.get("duplicate"):
                resolved += 1
        return resolved

    async def _reconcile_loop(self) -> None:
        while True:
            await asyncio.sleep(self.settings.reconcile_interval_seconds)
            try:
                n = await self.reconcile_once()
                if n:
                    log.info("reconciler resolved %d job(s) by polling", n)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                log.exception("reconciler iteration failed")

    def start(self) -> None:
        if self._reconciler is None:
            self._reconciler = asyncio.create_task(self._reconcile_loop(), name="job-reconciler")

    async def stop(self) -> None:
        if self._reconciler is not None:
            self._reconciler.cancel()
            try:
                await self._reconciler
            except asyncio.CancelledError:
                pass
            self._reconciler = None
        for task in list(self._background):
            if not task.done():
                task.cancel()


def _summarise(result: Any) -> dict[str, Any] | None:
    """Small, UI-friendly excerpt of a job result for the event stream."""
    if not isinstance(result, dict):
        return None
    keys = ("workload", "device", "elapsed_ms", "price", "median_final_value",
            "probability_reaching_target", "success")
    return {k: result[k] for k in keys if k in result}
