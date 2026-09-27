"""
Test fixtures.

The model is a Pydantic AI `FunctionModel` scripted to call one heavy tool
first and then answer with the tool result, so no LLM API key is needed.
Compute runs through the LocalDispatcher (or a fake) with zero delay.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Callable

import pytest
from httpx import ASGITransport, AsyncClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic_ai import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart  # noqa: E402
from pydantic_ai.models.function import AgentInfo, FunctionModel  # noqa: E402

from harness.config import Settings  # noqa: E402
from harness.dispatch import PollOutcome  # noqa: E402
from harness.models import Job  # noqa: E402
from main import create_app  # noqa: E402


def scripted_model(tool_name: str, args: dict[str, Any],
                   final: Callable[[Any], str] | None = None) -> FunctionModel:
    """First request -> call `tool_name`; once a tool return exists -> final text."""

    def respond(messages, info: AgentInfo) -> ModelResponse:
        returns = [p for m in messages if isinstance(m, ModelRequest) for p in m.parts if isinstance(p, ToolReturnPart)]
        if returns:
            content = returns[-1].content
            text = final(content) if final else f"DONE: {content}"
            return ModelResponse(parts=[TextPart(text)])
        return ModelResponse(parts=[ToolCallPart(tool_name, args)])

    return FunctionModel(respond)


def text_model(text: str = "hello") -> FunctionModel:
    return FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart(text)]))


class NeverFinishesDispatcher:
    """Submits fine, never completes on its own. Used to exercise the webhook path."""

    backend = "modal"

    def __init__(self) -> None:
        self.submitted: list[Job] = []
        self.cancelled: list[str] = []

    async def submit(self, job: Job) -> str | None:
        self.submitted.append(job)
        return f"fc-{job.job_id}"

    async def poll(self, job: Job) -> PollOutcome:
        return PollOutcome(state="pending")

    async def cancel(self, job: Job) -> None:
        self.cancelled.append(job.job_id)


class PollSucceedsDispatcher(NeverFinishesDispatcher):
    """Never calls back, but polling returns a result. Exercises the reconciler path."""

    async def poll(self, job: Job) -> PollOutcome:
        return PollOutcome(state="succeeded", result={"workload": job.kind.value, "price": 12.34, "device": "torch:cuda"},
                           worker={"backend": "modal"})


class SubmitFailsDispatcher(NeverFinishesDispatcher):
    async def submit(self, job: Job) -> str | None:
        raise RuntimeError("modal is down")


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        modal_mode="local",
        local_job_delay_seconds=0.0,
        store_backend="memory",
        harness_webhook_secret="test-secret",
        reconcile_grace_seconds=0,
        reconcile_interval_seconds=0.05,
        orchestrator_model="test:function",
    )


@pytest.fixture
async def make_client(settings):
    """Factory: build an app around a model/dispatcher and return (app, client)."""
    created: list[AsyncClient] = []

    async def _make(model, dispatcher=None, override: Settings | None = None):
        app = create_app(settings=override or settings, model=model, dispatcher=dispatcher, start_reconciler=False)
        client = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
        created.append(client)
        return app, client

    yield _make
    for c in created:
        await c.aclose()


async def wait_for_status(client: AsyncClient, run_id: str, statuses: set[str], timeout: float = 5.0) -> dict:
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        run = (await client.get(f"/runs/{run_id}")).json()
        if run["status"] in statuses:
            return run
        if asyncio.get_event_loop().time() > deadline:
            raise AssertionError(f"run {run_id} stuck in {run['status']}: {run.get('error')}")
        await asyncio.sleep(0.02)
