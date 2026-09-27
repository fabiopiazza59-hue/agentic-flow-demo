# Pydantic AI + Modal Harness

An orchestration harness where a **Pydantic AI** agent hands long-running work (GPU inference, large
simulations, sandboxed code) to **Modal**, *suspends* its run, and *resumes automatically* once Modal
reports back. Clients get answers asynchronously through Server-Sent Events, polling, or a webhook.

It is the async counterpart of [`simple-MVP`](../simple-MVP): same financial-assistant domain, but the
orchestrator is Pydantic AI instead of LangGraph, and heavy tools run on Modal CPUs/GPUs instead of
in-process.

## Table of Contents

- [Why this exists](#why-this-exists)
- [Architecture](#architecture)
- [How a run flows](#how-a-run-flows)
- [Design decision: getting results back](#design-decision-getting-results-back)
- [Quick start (local mode, no Modal account)](#quick-start-local-mode-no-modal-account)
- [Running on Modal](#running-on-modal)
- [API reference](#api-reference)
- [Configuration](#configuration)
- [Adding a new heavy tool](#adding-a-new-heavy-tool)
- [Tests](#tests)
- [Observability](#observability)
- [Production notes](#production-notes)
- [Folder structure](#folder-structure)

---

## Why this exists

An LLM agent loop is request/response: the model asks for a tool, the tool returns, the model continues.
That breaks when a tool takes minutes (GPU model inference, a 10M-path Monte Carlo, a backtest):

- the HTTP request to your API times out,
- the agent process holds a connection open doing nothing,
- if the process restarts, the whole conversation is lost.

This harness makes long-running tools first-class:

| Problem | Mechanism |
|---|---|
| Tool takes minutes | Pydantic AI **deferred tools**: the tool dispatches a job and raises `CallDeferred`; the run ends with `DeferredToolRequests` and its message history is persisted |
| Compute needs CPUs/GPUs | Jobs are `spawn`ed on **Modal** functions with the right resources (8-core CPU, A10G GPU, isolated `modal.Sandbox`) |
| Getting the answer back | **Webhook first, poll fallback**: Modal POSTs a signed result; a reconciler polls `FunctionCall.get(timeout=0)` for anything that never called back |
| Resuming the agent | `agent.run(message_history=..., deferred_tool_results=...)` feeds every job result back as its tool return, and the model finishes the answer |
| Telling the client | `GET /runs/{id}/events` (SSE), `GET /runs/{id}` (poll), optional `callback_url` (signed POST) |
| Restarts | SQLite store + startup reconciliation: in-flight runs survive a process restart |

## Architecture

```
 Client (UI / cli.py / any HTTP client)
   │  POST /runs {query}            ──▶ 202 {run_id, events_url}
   │  GET  /runs/{id}/events        ◀── SSE: run.started, job.dispatched, run.suspended, job.completed, run.resumed, run.completed
   ▼
┌───────────────────────────────────── HARNESS (FastAPI, main.py) ─────────────────────────────────────┐
│                                                                                                        │
│   ┌────────────────────────── Orchestrator (Pydantic AI Agent, Claude Opus 5) ───────────────────────┐ │
│   │  fast tools  : get_stock_quote · calculate · explain_term          → answered inline             │ │
│   │  heavy tools : run_monte_carlo · price_option_gpu · run_python_sandbox                            │ │
│   │                 └─ JobManager.dispatch() → raise CallDeferred  ⇒ run WAITING_FOR_JOBS             │ │
│   └────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                     │ spawn                                        ▲ resume(message_history +          │
│                     ▼                                              │        deferred_tool_results)     │
│   ┌── JobManager ─────────────────────────────────────────────────┴─────────────────────────────────┐ │
│   │  handle_result()  ← POST /webhooks/modal (HMAC)   ← fast path                                    │ │
│   │  handle_result()  ← reconciler: FunctionCall.get(timeout=0) every 15s  ← fallback path           │ │
│   │  idempotent: first terminal result wins; all jobs done ⇒ RESUMING (once, per-run lock)           │ │
│   └────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│   RunStore (memory | sqlite) · EventBus (SSE fan-out) · Phoenix tracing (optional)                     │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                      │ modal.Function.from_name(app, fn).spawn(job)
                      ▼
┌──────────────────────────────── MODAL (modal_app.py) ─────────────────────────────────┐
│  run_cpu_job      cpu=8, 8 GB        numpy Monte Carlo                                 │
│  run_gpu_job      gpu="A10G"         torch Asian-option pricing (CUDA)                 │
│  run_sandbox_job  → modal.Sandbox    model-written Python, block_network=True          │
│  each: execute → envelope → signed POST to harness webhook → return envelope           │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

## How a run flows

```
POST /runs ─▶ QUEUED ─▶ RUNNING ──(text answer)──────────────────────────────▶ COMPLETED
                          │
                          │ heavy tool called → job spawned on Modal → CallDeferred
                          ▼
                    WAITING_FOR_JOBS ◀─────────────────────────┐
                          │ every job terminal (webhook or poll) │ model defers again
                          ▼                                      │ (bounded by MAX_RUN_RESUMES)
                       RESUMING ─▶ RUNNING ──────────────────────┘
                                      │
                                      └──(text answer)──▶ COMPLETED   (or FAILED / CANCELLED)
```

Concretely, for *"Price an Asian call on NVDA, spot 140, strike 150, vol 45%, 1 year, on the GPU"*:

1. `run.started`: the agent reads the instructions and calls `price_option_gpu(...)`.
2. `job.created` / `job.dispatched`: the tool asks `JobManager` to spawn `run_gpu_job` on Modal and raises
   `CallDeferred(metadata={"job_id": ...})`. Pydantic AI returns `DeferredToolRequests`.
3. `run.suspended`: message history is stored, status is `waiting_for_jobs`. No HTTP connection is held.
4. Modal finishes on an A10G, signs the result and POSTs `/webhooks/modal` ⇒ `job.completed (via webhook)`.
   If the webhook never arrives, the reconciler fetches the same envelope with `FunctionCall.get(timeout=0)`
   ⇒ `job.completed (via poll)`.
5. `run.resumed`: the harness calls `agent.run(message_history=..., deferred_tool_results={tool_call_id: result})`.
   The model sees the price, the device (`torch:cuda:NVIDIA A10G`) and the elapsed time, and writes the answer.
6. `run.completed`: SSE subscribers get the output; the optional client `callback_url` is POSTed.

## Design decision: getting results back

Three ways exist for the orchestrator to learn that a Modal job finished. The harness uses the first two
together and exposes the third for clients.

| Option | Latency | Reliability | Needs | Verdict |
|---|---|---|---|---|
| **Webhook** (Modal → harness) | sub-second | lost if the harness is unreachable/restarting | a public URL (tunnel in dev), HMAC secret | **Primary path** |
| **Polling** (`FunctionCall.from_id(id).get(timeout=0)`) | up to one interval (15 s default) | very high: Modal keeps outputs for hours | only the call id (stored on the job) | **Fallback + recovery after restart** |
| Blocking `FunctionCall.get()` in the tool | immediate | ties a worker/connection to every job; nothing survives a restart | nothing | rejected as the harness design; fine for sub-minute jobs |

Why both: the webhook gives instant resumption; polling makes correctness independent of network luck.
`handle_result` is idempotent (first terminal result wins, duplicates are acknowledged and dropped), so
receiving the same result twice is harmless. In local development without a public URL, simply leave
`HARNESS_PUBLIC_URL` empty: the workers skip the webhook and every job is picked up by polling.

Notifying the *client* follows the same idea: SSE for live UIs, `GET /runs/{id}` for anything else, and an
optional signed `callback_url` for machine-to-machine consumers.

## Quick start (local mode, no Modal account)

Local mode runs the exact same workloads in-process (thread pool) after a small artificial delay, so the
whole suspend → callback → resume flow is visible without deploying anything.

```bash
cd agentic-example/pydantic-modal-harness
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # set ANTHROPIC_API_KEY; keep MODAL_MODE=local
python main.py            # http://localhost:8000
```

Try it:

```bash
python cli.py "Run a Monte Carlo: \$100k initial, \$12k per year, 30 years, 1,000,000 paths. Chance of reaching \$1M?"
```

You will see the event timeline (`run.suspended` → `job.completed via local` → `run.resumed` → `run.completed`)
followed by the final answer.

## Running on Modal

```bash
pip install modal && modal setup                     # one-time auth

# 1. Shared secret used to sign webhooks (same value as HARNESS_WEBHOOK_SECRET in .env)
modal secret create agentic-harness-secret HARNESS_WEBHOOK_SECRET=<long-random-string>

# 2. Deploy the workers (builds the CPU image, the GPU image with torch, and the sandbox image)
modal deploy modal_app.py

# 3. Optional smoke tests straight on Modal
modal run modal_app.py --kind cpu_monte_carlo
modal run modal_app.py --kind gpu_pricing
modal run modal_app.py --kind sandbox_code
```

Then in `.env`:

```env
MODAL_MODE=modal
MODAL_APP_NAME=agentic-modal-harness
# Fast path. Expose the harness, e.g. `ngrok http 8000` or `cloudflared tunnel --url http://localhost:8000`
HARNESS_PUBLIC_URL=https://<your-tunnel>.ngrok.app
# Without a public URL everything still works: results arrive via the polling reconciler.
```

Restart `python main.py`. Jobs now show `backend=modal`, and `delivered_via` tells you which path won.

Resource profiles live in [`modal_app.py`](modal_app.py): change `gpu="A10G"` to `"H100"`, raise `cpu=`,
add volumes for model weights, etc. Workers never retry on their own (`retries=0`) so a failure is reported
once and the model decides what to do.

## API reference

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/runs` | `{query, callback_url?}` → **202** `{run_id, status, events_url, ...}`. Work continues in the background. |
| `GET` | `/runs/{id}` | Status, jobs (kind, backend, `delivered_via`, result), final `output`, `resume_count`, usage. |
| `GET` | `/runs/{id}/events` | Server-Sent Events. Replays history, then streams live until a terminal event, then `event: end`. |
| `POST` | `/runs/{id}/cancel` | Cancel in-flight Modal calls and mark the run `cancelled`. |
| `POST` | `/webhooks/modal` | Worker callback. Body = `JobResultEnvelope`; headers `X-Harness-Timestamp`, `X-Harness-Signature: sha256=<hmac>`. 200 `{accepted, duplicate}`, 401 on bad signature, 404 unknown job. |
| `GET` | `/runs` | Recent runs. |
| `GET` | `/health` | Backend, model, delivery config. |

Event types: `run.created`, `run.started`, `job.created`, `job.dispatched`, `run.suspended`, `job.completed`,
`job.failed`, `job.cancelled`, `run.resumed`, `run.completed`, `run.failed`, `run.cancelled`,
`run.callback_sent`, `run.callback_failed`.

Result envelope (what workers send and what polling returns):

```json
{
  "job_id": "job_1a2b3c", "run_id": "run_9f8e7d", "status": "succeeded",
  "result": {"workload": "gpu_pricing", "price": 12.41, "std_error": 0.021, "device": "torch:cuda:NVIDIA A10G", "elapsed_ms": 4180},
  "external_id": "fc-01ABC…", "worker": {"backend": "modal", "function": "run_gpu_job", "webhook": {"status_code": 200}}
}
```

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | – | Required for real runs |
| `ORCHESTRATOR_MODEL` | `anthropic:claude-opus-5` | Any Pydantic AI model string (e.g. `anthropic:claude-sonnet-5`) |
| `MODAL_MODE` | `local` | `local` (in-process) or `modal` |
| `MODAL_APP_NAME` | `agentic-modal-harness` | Must match `modal_app.py` |
| `LOCAL_JOB_DELAY_SECONDS` | `2` | Artificial latency in local mode |
| `HARNESS_PUBLIC_URL` | – | Public base URL for the webhook fast path; empty = polling only |
| `HARNESS_WEBHOOK_SECRET` | – | HMAC secret shared with the Modal secret and with client callbacks |
| `RECONCILE_INTERVAL_SECONDS` | `15` | Poll cadence |
| `RECONCILE_GRACE_SECONDS` | `20` | Do not poll a job younger than this (let the webhook win) |
| `STORE_BACKEND` | `memory` | `memory` or `sqlite` |
| `SQLITE_PATH` | `harness.db` | SQLite file |
| `MAX_RUN_RESUMES` | `5` | Suspend/resume cycles allowed per run |
| `MAX_MODEL_REQUESTS_PER_RUN` | `12` | Pydantic AI `UsageLimits(request_limit=...)` |
| `PHOENIX_ENABLED` | `false` | Send Pydantic AI spans to Phoenix |
| `PORT` | `8000` | HTTP port |

Model notes: Claude Opus 5 runs adaptive thinking by default, so no `thinking` setting is sent. Prompt
caching is enabled (`anthropic_cache=True`) so the stable instructions + tool schema prefix is cached across
runs and resumes.

## Adding a new heavy tool

Three steps, mirroring the "add an agent" recipe in `simple-MVP`:

1. **Workload** – add a pure function to `harness/workloads.py` and register it in `WORKLOADS`.
2. **Modal function** – in `modal_app.py`, either map the new kind to an existing worker or add one with
   the resources it needs; add the kind to `harness/models.py::JobKind` and `harness/dispatch.py::FUNCTION_FOR_KIND`.
3. **Tool** – in `harness/orchestrator.py`:

```python
@agent.tool
async def run_backtest(ctx: RunContext[HarnessDeps], symbol: str, years: int = 5) -> dict[str, Any]:
    """LONG-RUNNING (CPU container on Modal). Backtest the V2.1 scalp strategy."""
    return await _defer(ctx, JobKind.BACKTEST, {"symbol": symbol, "years": years})
```

`_defer` dispatches the job and raises `CallDeferred`; everything else (suspend, callback, resume, events) is
generic.

## Tests

```bash
pytest -q
```

24 tests, no API key or Modal account needed. The model is a scripted Pydantic AI `FunctionModel`; fake
dispatchers exercise every delivery path:

- `test_flow_local.py` – fast-tool-only run, full suspend/resume cycle, SSE replay, inline dispatch failure
- `test_delivery_paths.py` – webhook delivery, HMAC rejection, idempotent duplicates, polling fallback,
  failed jobs reported to the model, cancellation
- `test_sqlite_recovery.py` – a suspended run survives a process restart and is finished by the reconciler
- `test_security.py`, `test_store.py`, `test_workloads.py` – units

## Observability

Set `PHOENIX_ENABLED=true` and start Phoenix (`python -m phoenix.server.main serve`, or the compose file in
`simple-MVP`). Pydantic AI's `Instrumentation` capability exports every model request, tool call, deferral
and resume as OpenTelemetry spans to the `pydantic-modal-harness` project at http://localhost:6006.

## Production notes

- **State**: swap `SqliteRunStore` for Postgres/Redis by implementing the `RunStore` protocol (`harness/store.py`).
  Everything is stored as JSON blobs, so the migration is mechanical.
- **Webhook exposure**: put `/webhooks/modal` behind your ingress; HMAC + timestamp window already prevent
  forgery and replay. Rotate `HARNESS_WEBHOOK_SECRET` by updating the Modal secret and redeploying.
- **Scaling the harness**: several harness replicas can share one store, but only one should run the
  reconciler (or add a lease). The per-run resume lock is in-process, so pin a run to a replica or move the
  lock into the store.
- **Cost control**: `UsageLimits` caps model requests per run; `MAX_RUN_RESUMES` caps how many times the
  model may defer; Modal functions carry hard `timeout`s.
- **Sandbox**: `run_sandbox_job` uses `modal.Sandbox(block_network=True)` and a throwaway image. The local
  fallback (`sandbox_code_local`) is a demo-grade restricted `exec` and must not be used with untrusted
  code in production.

## Folder structure

```
pydantic-modal-harness/
├── main.py                 # FastAPI: /runs, SSE events, /webhooks/modal, health
├── modal_app.py            # Modal App: run_cpu_job, run_gpu_job, run_sandbox_job (+ signed callback)
├── cli.py                  # Submit a query and follow the event stream
├── harness/
│   ├── orchestrator.py     # Pydantic AI Agent, tools, suspend/resume loop
│   ├── jobs.py             # JobManager: dispatch, idempotent results, reconciler
│   ├── dispatch.py         # ModalDispatcher (spawn/poll/cancel) and LocalDispatcher
│   ├── store.py            # RunStore protocol, InMemoryRunStore, SqliteRunStore
│   ├── events.py           # Per-run pub/sub for SSE
│   ├── security.py         # HMAC sign / verify
│   ├── workloads.py        # numpy / torch workloads shared by Modal and local mode
│   ├── tracing.py          # Phoenix / OpenTelemetry
│   ├── models.py           # Run, Job, JobResultEnvelope, events
│   └── config.py           # Settings (.env)
├── static/index.html       # Demo UI with live SSE timeline
├── tests/                  # 24 tests, no external services
├── requirements.txt
└── .env.example
```
