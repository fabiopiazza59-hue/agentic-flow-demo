"""
Pydantic AI + Modal Harness - FastAPI server

Run:      python main.py
API:      http://localhost:8000          (demo UI)
Docs:     http://localhost:8000/docs
Phoenix:  http://localhost:6006          (when PHOENIX_ENABLED=true)

Endpoints
  POST /runs                 submit a query -> 202 + run_id (work continues in the background)
  GET  /runs/{id}            status, jobs, final output
  GET  /runs/{id}/events     Server-Sent Events: live timeline of the run
  POST /runs/{id}/cancel     cancel in-flight jobs and the run
  POST /webhooks/modal       signed callback from Modal workers (fast delivery path)
  GET  /health
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from sse_starlette.sse import EventSourceResponse  # noqa: E402

from harness.config import Settings, get_settings  # noqa: E402
from harness.dispatch import JobDispatcher, LocalDispatcher, ModalDispatcher  # noqa: E402
from harness.events import EventBus  # noqa: E402
from harness.jobs import JobManager  # noqa: E402
from harness.models import JobResultEnvelope, RunCreateRequest, RunStatus, RunSummary  # noqa: E402
from harness.orchestrator import Orchestrator, build_agent  # noqa: E402
from harness.security import SIGNATURE_HEADER, TIMESTAMP_HEADER, verify_signature  # noqa: E402
from harness.store import RunStore, build_store  # noqa: E402
from harness.tracing import build_instrumentation  # noqa: E402

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(name)s %(levelname)s %(message)s")
for _noisy in ("httpx", "httpx2", "httpcore"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
log = logging.getLogger("harness.main")

STATIC_DIR = Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# Runtime wiring
# ---------------------------------------------------------------------------

@dataclass
class HarnessRuntime:
    settings: Settings
    store: RunStore
    bus: EventBus
    dispatcher: JobDispatcher
    jobs: JobManager
    orchestrator: Orchestrator


def resolve_model(settings: Settings):
    """Build the Pydantic AI model from settings. Anthropic gets an explicit provider."""
    name = settings.orchestrator_model
    if name.startswith("anthropic:"):
        api_key = settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Add it to .env (see .env.example) or export it."
            )
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider

        return AnthropicModel(name.split(":", 1)[1], provider=AnthropicProvider(api_key=api_key))
    return name


def build_runtime(settings: Settings, model: Any = None, dispatcher: JobDispatcher | None = None) -> HarnessRuntime:
    store = build_store(settings.store_backend, settings.sqlite_path)
    bus = EventBus()

    jobs_ref: list[JobManager] = []

    async def on_local_result(envelope: JobResultEnvelope) -> None:
        await jobs_ref[0].handle_result(envelope, via="local")

    if dispatcher is None:
        if settings.modal_mode == "modal":
            dispatcher = ModalDispatcher(app_name=settings.modal_app_name, callback_url=settings.callback_url)
        else:
            dispatcher = LocalDispatcher(on_result=on_local_result, delay_seconds=settings.local_job_delay_seconds)

    jobs = JobManager(store, bus, dispatcher, settings)
    jobs_ref.append(jobs)

    agent = build_agent(model or resolve_model(settings), settings=settings,
                        instrumentation=build_instrumentation(settings))
    orchestrator = Orchestrator(agent, store, bus, jobs, settings)
    return HarnessRuntime(settings, store, bus, dispatcher, jobs, orchestrator)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(settings: Settings | None = None, model: Any = None,
               dispatcher: JobDispatcher | None = None, start_reconciler: bool = True) -> FastAPI:
    settings = settings or get_settings()
    runtime = build_runtime(settings, model=model, dispatcher=dispatcher)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        print("\n" + "=" * 64)
        print("  PYDANTIC AI + MODAL HARNESS")
        print("=" * 64)
        print(f"  model      : {settings.orchestrator_model}")
        print(f"  backend    : {settings.modal_mode}" + (f"  (app={settings.modal_app_name})" if settings.modal_mode == "modal" else ""))
        print(f"  delivery   : webhook={'on -> ' + settings.callback_url if settings.callback_url else 'off (no HARNESS_PUBLIC_URL)'}, "
              f"poll every {settings.reconcile_interval_seconds:.0f}s")
        print(f"  store      : {settings.store_backend}")
        print(f"  UI / docs  : http://localhost:{settings.port}  |  http://localhost:{settings.port}/docs")
        print("=" * 64 + "\n")
        if start_reconciler:
            runtime.jobs.start()
            # Runs that were mid-flight when the server last stopped get polled immediately.
            asyncio.create_task(runtime.jobs.reconcile_once(ignore_grace=True))
        yield
        await runtime.jobs.stop()
        await runtime.store.close()

    app = FastAPI(
        title="Pydantic AI + Modal Harness",
        description="Async orchestration: Pydantic AI agent + long-running Modal jobs (CPU/GPU/Sandbox)",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.runtime = runtime
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                       allow_methods=["*"], allow_headers=["*"])

    # -- helpers ----------------------------------------------------------
    async def summary_for(run_id: str) -> RunSummary:
        run = await runtime.store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        jobs = await runtime.store.jobs_for_run(run_id)
        return RunSummary(
            run_id=run.run_id, status=run.status, query=run.query, output=run.output, error=run.error,
            jobs=sorted(jobs, key=lambda j: j.created_at), resume_count=run.resume_count, usage=run.usage,
            created_at=run.created_at, updated_at=run.updated_at, events_url=f"/runs/{run.run_id}/events",
        )

    # -- routes -----------------------------------------------------------
    @app.get("/", include_in_schema=False)
    async def root():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "model": settings.orchestrator_model,
            "backend": settings.modal_mode,
            "modal_app": settings.modal_app_name if settings.modal_mode == "modal" else None,
            "delivery": {"webhook_url": settings.callback_url,
                         "poll_interval_seconds": settings.reconcile_interval_seconds,
                         "poll_grace_seconds": settings.reconcile_grace_seconds},
            "store": settings.store_backend,
            "tracing": "phoenix" if settings.phoenix_enabled else "off",
            "anthropic_key": "configured" if (settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")) else "missing",
        }

    @app.post("/runs", status_code=202, response_model=RunSummary)
    async def create_run(body: RunCreateRequest):
        run = await runtime.orchestrator.start_run(body.query, callback_url=body.callback_url)
        return await summary_for(run.run_id)

    @app.get("/runs", response_model=list[RunSummary])
    async def list_runs(limit: int = 20):
        runs = await runtime.store.list_runs(limit=limit)
        return [await summary_for(r.run_id) for r in runs]

    @app.get("/runs/{run_id}", response_model=RunSummary)
    async def get_run(run_id: str):
        return await summary_for(run_id)

    @app.post("/runs/{run_id}/cancel", response_model=RunSummary)
    async def cancel_run(run_id: str):
        run = await runtime.orchestrator.cancel_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        return await summary_for(run_id)

    @app.get("/runs/{run_id}/events")
    async def run_events(run_id: str, request: Request):
        if await runtime.store.get_run(run_id) is None:
            raise HTTPException(status_code=404, detail="run not found")
        return EventSourceResponse(_event_stream(runtime, run_id, request))

    @app.post("/webhooks/modal")
    async def modal_webhook(request: Request):
        body = await request.body()
        ok, reason = verify_signature(
            settings.harness_webhook_secret, body,
            request.headers.get(TIMESTAMP_HEADER), request.headers.get(SIGNATURE_HEADER),
            max_skew_seconds=settings.webhook_max_skew_seconds,
        )
        if not ok:
            raise HTTPException(status_code=401, detail=f"invalid signature: {reason}")
        try:
            envelope = JobResultEnvelope.model_validate_json(body)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=422, detail=f"invalid envelope: {exc}") from exc
        outcome = await runtime.jobs.handle_result(envelope, via="webhook")
        if not outcome.get("accepted"):
            return JSONResponse(status_code=404, content=outcome)
        return outcome

    return app


async def _event_stream(runtime: HarnessRuntime, run_id: str, request: Request) -> AsyncIterator[dict[str, Any]]:
    """Replay persisted events, then follow live ones until the run is terminal."""
    queue = runtime.bus.subscribe(run_id)   # subscribe *before* replay so nothing is missed
    terminal = {"run.completed", "run.failed", "run.cancelled"}
    try:
        last_seq = 0
        for event in await runtime.store.events_for_run(run_id):
            yield _sse(event)
            last_seq = event.seq
            if event.type in terminal:
                yield {"event": "end", "data": json.dumps({"run_id": run_id})}
                return
        run = await runtime.store.get_run(run_id)
        if run is not None and run.status.is_terminal:
            yield {"event": "end", "data": json.dumps({"run_id": run_id})}
            return
        while True:
            if await request.is_disconnected():
                return
            try:
                event = await asyncio.wait_for(queue.get(), timeout=15)
            except asyncio.TimeoutError:
                yield {"comment": "keepalive"}
                continue
            if event is None:
                yield {"event": "end", "data": json.dumps({"run_id": run_id})}
                return
            if event.seq <= last_seq:
                continue
            last_seq = event.seq
            yield _sse(event)
            if event.type in terminal:
                yield {"event": "end", "data": json.dumps({"run_id": run_id})}
                return
    finally:
        runtime.bus.unsubscribe(run_id, queue)


def _sse(event) -> dict[str, Any]:
    return {"id": str(event.seq), "event": event.type,
            "data": json.dumps({"seq": event.seq, "type": event.type, "ts": event.ts.isoformat(), **event.data}, default=str)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:create_app", factory=True, host="0.0.0.0", port=get_settings().port, reload=False, log_level="info")
