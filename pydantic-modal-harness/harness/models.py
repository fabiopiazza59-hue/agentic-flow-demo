"""
Data models for runs, jobs and events.

A *Run* is one user query handled by the orchestrator agent.
A *Job* is one long-running unit of work the agent deferred to a compute
backend (Modal or local). A run can own several jobs and can suspend/resume
several times.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"                    # agent is talking to the model
    WAITING_FOR_JOBS = "waiting_for_jobs"  # agent suspended, jobs in flight
    RESUMING = "resuming"                  # all jobs done, agent being resumed
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED)


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.EXPIRED, JobStatus.CANCELLED)


class JobKind(StrEnum):
    """Workload types. Each maps to a Modal function with the right resources."""
    CPU_MONTE_CARLO = "cpu_monte_carlo"   # many-core CPU container
    GPU_PRICING = "gpu_pricing"           # GPU container (torch)
    SANDBOX_CODE = "sandbox_code"         # isolated modal.Sandbox running untrusted code


class Job(BaseModel):
    job_id: str = Field(default_factory=lambda: new_id("job"))
    run_id: str
    tool_call_id: str
    tool_name: str
    kind: JobKind
    payload: dict[str, Any] = Field(default_factory=dict)
    status: JobStatus = JobStatus.PENDING
    backend: Literal["local", "modal"] = "local"
    external_id: str | None = None        # Modal FunctionCall id
    result: Any = None
    error: str | None = None
    worker_meta: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)
    dispatched_at: datetime | None = None
    completed_at: datetime | None = None
    delivered_via: Literal["webhook", "poll", "local"] | None = None
    consumed: bool = False                # result already fed back into the agent run

    def age_seconds(self, now: datetime | None = None) -> float:
        start = self.dispatched_at or self.created_at
        return ((now or utcnow()) - start).total_seconds()


class Run(BaseModel):
    run_id: str = Field(default_factory=lambda: new_id("run"))
    query: str
    status: RunStatus = RunStatus.QUEUED
    output: str | None = None
    error: str | None = None
    messages: list[Any] = Field(default_factory=list)   # serialized Pydantic AI message history
    job_ids: list[str] = Field(default_factory=list)
    callback_url: str | None = None                     # optional client webhook
    resume_count: int = 0
    usage: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class RunEvent(BaseModel):
    run_id: str
    seq: int = 0
    type: str
    data: dict[str, Any] = Field(default_factory=dict)
    ts: datetime = Field(default_factory=utcnow)


class JobResultEnvelope(BaseModel):
    """
    Payload a worker (Modal or local) sends back to the harness.

    This is the single format both delivery paths (webhook push and
    reconciler poll) are normalised into before touching the store.
    """
    job_id: str
    run_id: str
    status: Literal["succeeded", "failed"]
    result: Any = None
    error: str | None = None
    external_id: str | None = None
    worker: dict[str, Any] = Field(default_factory=dict)


class RunCreateRequest(BaseModel):
    query: str = Field(..., min_length=1, examples=[
        "Price an Asian call on NVDA: spot 140, strike 150, 1y, vol 45% on the GPU",
        "Run a 1M-path Monte Carlo for $100k initial, $12k/yr, 30 years",
        "What is the Sharpe ratio?",
    ])
    callback_url: str | None = Field(default=None, description="Optional URL to POST the final result to")


class RunSummary(BaseModel):
    run_id: str
    status: RunStatus
    query: str
    output: str | None = None
    error: str | None = None
    jobs: list[Job] = Field(default_factory=list)
    resume_count: int = 0
    usage: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    events_url: str
