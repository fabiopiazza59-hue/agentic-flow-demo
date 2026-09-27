"""
Job dispatch backends.

A dispatcher knows how to *submit* a job, *poll* its state (used by the
reconciler as the fallback delivery path) and *cancel* it.

- ModalDispatcher : spawns the deployed Modal function for the job kind and
                    keeps the `FunctionCall` id so results can be fetched
                    even if the webhook never arrives.
- LocalDispatcher : runs the same workload in-process (thread pool) after an
                    optional delay. Used for development and tests. It also
                    reports through the same `on_result` path the webhook
                    uses, so the resume logic is exercised identically.
"""

from __future__ import annotations

import asyncio
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, Protocol

from .models import Job, JobKind, JobResultEnvelope
from .workloads import execute_workload

PollState = Literal["pending", "succeeded", "failed", "expired"]


@dataclass
class PollOutcome:
    state: PollState
    result: Any = None
    error: str | None = None
    worker: dict[str, Any] = field(default_factory=dict)


class JobDispatcher(Protocol):
    backend: Literal["local", "modal"]

    async def submit(self, job: Job) -> str | None:
        """Start the job. Returns an external id (e.g. Modal FunctionCall id) or None."""
        ...

    async def poll(self, job: Job) -> PollOutcome: ...

    async def cancel(self, job: Job) -> None: ...


# ---------------------------------------------------------------------------
# Modal
# ---------------------------------------------------------------------------

FUNCTION_FOR_KIND: dict[JobKind, str] = {
    JobKind.CPU_MONTE_CARLO: "run_cpu_job",
    JobKind.GPU_PRICING: "run_gpu_job",
    JobKind.SANDBOX_CODE: "run_sandbox_job",
}


class ModalDispatcher:
    backend: Literal["local", "modal"] = "modal"

    def __init__(self, app_name: str, callback_url: str | None, environment_name: str | None = None) -> None:
        self.app_name = app_name
        self.callback_url = callback_url
        self.environment_name = environment_name
        self._functions: dict[str, Any] = {}

    def _function(self, kind: JobKind):
        import modal

        name = FUNCTION_FOR_KIND[kind]
        if name not in self._functions:
            self._functions[name] = modal.Function.from_name(
                self.app_name, name, environment_name=self.environment_name
            )
        return self._functions[name]

    async def submit(self, job: Job) -> str | None:
        fn = self._function(job.kind)
        request = {
            "job_id": job.job_id,
            "run_id": job.run_id,
            "kind": job.kind.value,
            "payload": job.payload,
            "callback_url": self.callback_url,   # None -> worker skips the webhook, reconciler polls
        }
        call = await fn.spawn.aio(request)
        return call.object_id

    async def poll(self, job: Job) -> PollOutcome:
        import modal
        from modal.exception import OutputExpiredError

        if not job.external_id:
            return PollOutcome(state="failed", error="job has no Modal call id")
        call = modal.FunctionCall.from_id(job.external_id)
        try:
            envelope = await call.get.aio(timeout=0)
        except TimeoutError:
            return PollOutcome(state="pending")
        except OutputExpiredError:
            return PollOutcome(state="expired", error="Modal output expired before it was collected")
        except Exception as exc:  # noqa: BLE001 - remote exception surfaced by Modal
            return PollOutcome(state="failed", error=f"{type(exc).__name__}: {exc}")

        # Workers return the same envelope they POST to the webhook.
        if isinstance(envelope, dict) and envelope.get("status") == "failed":
            return PollOutcome(state="failed", error=envelope.get("error"), worker=envelope.get("worker", {}))
        if isinstance(envelope, dict) and "result" in envelope:
            return PollOutcome(state="succeeded", result=envelope["result"], worker=envelope.get("worker", {}))
        return PollOutcome(state="succeeded", result=envelope)

    async def cancel(self, job: Job) -> None:
        import modal

        if job.external_id:
            await modal.FunctionCall.from_id(job.external_id).cancel.aio()


# ---------------------------------------------------------------------------
# Local (dev + tests)
# ---------------------------------------------------------------------------

ResultCallback = Callable[[JobResultEnvelope], Awaitable[None]]


class LocalDispatcher:
    """Runs workloads in-process. Mirrors the Modal worker contract exactly."""

    backend: Literal["local", "modal"] = "local"

    def __init__(self, on_result: ResultCallback | None = None, delay_seconds: float = 0.0) -> None:
        self.on_result = on_result
        self.delay_seconds = delay_seconds
        self._tasks: dict[str, asyncio.Task[JobResultEnvelope]] = {}

    async def submit(self, job: Job) -> str | None:
        task = asyncio.create_task(self._execute(job), name=f"local-job-{job.job_id}")
        self._tasks[job.job_id] = task
        return f"local-{job.job_id}"

    async def _execute(self, job: Job) -> JobResultEnvelope:
        started = time.perf_counter()
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        try:
            result = await asyncio.to_thread(execute_workload, job.kind.value, job.payload)
            envelope = JobResultEnvelope(
                job_id=job.job_id, run_id=job.run_id, status="succeeded", result=result,
                external_id=f"local-{job.job_id}",
                worker={"backend": "local", "elapsed_ms": round((time.perf_counter() - started) * 1000, 1)},
            )
        except Exception as exc:  # noqa: BLE001 - report failure to the run
            envelope = JobResultEnvelope(
                job_id=job.job_id, run_id=job.run_id, status="failed",
                error=f"{type(exc).__name__}: {exc}", external_id=f"local-{job.job_id}",
                worker={"backend": "local", "traceback": traceback.format_exc()[-2000:]},
            )
        if self.on_result is not None:
            await self.on_result(envelope)
        return envelope

    async def poll(self, job: Job) -> PollOutcome:
        task = self._tasks.get(job.job_id)
        if task is None:
            return PollOutcome(state="expired", error="no local task for job")
        if not task.done():
            return PollOutcome(state="pending")
        if task.cancelled():
            return PollOutcome(state="failed", error="local task cancelled")
        envelope = task.result()
        if envelope.status == "failed":
            return PollOutcome(state="failed", error=envelope.error, worker=envelope.worker)
        return PollOutcome(state="succeeded", result=envelope.result, worker=envelope.worker)

    async def cancel(self, job: Job) -> None:
        task = self._tasks.get(job.job_id)
        if task and not task.done():
            task.cancel()

    async def wait_all(self) -> None:
        """Test helper: wait for every in-flight local job."""
        pending = [t for t in self._tasks.values() if not t.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


def build_dispatcher(mode: str, *, app_name: str, callback_url: str | None,
                     on_result: ResultCallback | None, local_delay: float) -> JobDispatcher:
    if mode == "modal":
        return ModalDispatcher(app_name=app_name, callback_url=callback_url)
    return LocalDispatcher(on_result=on_result, delay_seconds=local_delay)
