# Results & Learnings — Pydantic AI + Modal Harness

First real end-to-end run on Modal, 2026-09-06. This document captures what we
actually observed (not what the README predicts), the configuration that worked,
and the practical lessons for the next iteration.

## What we ran

- Backend flipped from `MODAL_MODE=local` to `MODAL_MODE=modal`.
- Modal workspace `fabio-piazza59`, app `agentic-modal-harness`, three functions
  live: `run_cpu_job`, `run_gpu_job`, `run_sandbox_job`.
- Result delivery: **polling only** (`HARNESS_PUBLIC_URL` left empty; no ngrok /
  cloudflared tunnel). Reconciler interval = 15 s, grace = 20 s.
- Store: `sqlite` (`harness.db`). Server on `PORT=8010`.
- Orchestrator: `anthropic:claude-opus-5` via Pydantic AI.

Query:

> Run a Monte Carlo simulation: $100k initial, $12k per year contributions,
> 30 years, 500,000 paths. What is the chance of reaching $1M?

## Observed event timeline

```
[ 1] run.created
[ 2] run.started          model=anthropic:claude-opus-5
[ 3] job.created          job_79fc8e85f505  kind=cpu_monte_carlo  backend=modal
[ 4] job.dispatched       external_id=fc-01M1W8B2JSVEGG46QSCR2QPTCE
[ 5] run.suspended        pending=[toolu_01LMUECvhWKmWcDnHSsH1RtG]
[ 6] job.completed        delivered_via=poll   elapsed_ms=358
[ 7] run.resumed          resume_count=1
[ 8] run.completed        usage: 4011 in / 543 out tokens, 2 model requests
```

Final result (produced by the LLM from the Modal output):

> **Probability of reaching $1M: 76.5%** — p10 $722k · median $1.55M ·
> mean $1.89M · p90 $3.45M.

## Latency & cost snapshot

| Phase | Wall time | Notes |
|---|---|---|
| Modal `modal deploy modal_app.py` | **212 s** total; **183 s** of that was the GPU image build (torch 2.14 + full CUDA 13 stack, ~1.4 GB of wheels) | One-time cost, cached for later deploys |
| Monte Carlo workload on Modal CPU | **358 ms** | 500k paths × 30 yrs, numpy on `cpu=8` |
| Suspend → resume gap | ≤ 15 s (one poll cycle) | Governed by `RECONCILE_INTERVAL_SECONDS`, not the workload |
| Model tokens per run | 4,011 in / 543 out, 2 requests | Prompt cache hit expected on subsequent runs |

**Interpretation:** with polling-only delivery, resume latency is dominated by
the 15 s poll interval, not by Modal or the model. A webhook tunnel would drop
end-to-end wall time below ~2 s for a job like this.

## Configuration that worked

Only two things had to change vs. the local defaults:

1. `.env`: `MODAL_MODE=modal` (everything else already set — key, port, sqlite,
   secret).
2. Modal: `agentic-harness-secret` created with the **same**
   `HARNESS_WEBHOOK_SECRET` value that is in `.env` so HMAC signatures match on
   both sides (even though the webhook path is unused today, the secret still
   has to be provisioned so worker code can sign).

Everything else (`MODAL_APP_NAME`, `RECONCILE_*`, `MAX_RUN_RESUMES`, etc.) worked
at defaults.

## Learnings

**1. Polling-only is a fine dev/staging mode.**
No public tunnel, no ingress, no dev-loop pain. `delivered_via=poll` shows up
cleanly in `job.completed`. The only cost is up-to-`RECONCILE_INTERVAL_SECONDS`
of extra latency per suspend cycle.

**2. Modal CLI lives inside the venv on this Mac.**
`modal` is not on the system `PATH`. Use `./venv/bin/modal ...` or
`./venv/bin/python -m modal ...`. Modal itself warns about this on `modal setup`,
and it does not break anything, but scripts that assume `modal` on PATH will
fail. Any wrapper we add later should call `python -m modal`.

**3. The GPU image build is the slow part of `modal deploy`.**
The CPU image and app registration are fast. Building the GPU image pulled
~1.4 GB of NVIDIA CUDA 13 wheels + torch 2.14, ~3 minutes on this network. Do
not put `modal deploy` in an inner dev loop; deploy once, then iterate the
harness locally.

**4. Suspend / resume worked on the first try with SQLite.**
`resume_count=1` after a single deferred call, message history round-tripped
through the store, no manual replay needed. This validates the deferred-tool
mechanism end-to-end against real Modal (not just the local fake dispatcher used
in tests).

**5. Model output is grounded in the Modal job result.**
Claude Opus 5 cited the exact `elapsed_ms=358` from the result envelope and
formatted the percentiles rather than hallucinating a fresh simulation. That is
the pattern we want: heavy compute produces numbers, the model narrates and
frames them.

**6. `.env` currently holds live secrets.**
`ANTHROPIC_API_KEY`, `HARNESS_WEBHOOK_SECRET`, and `FINNHUB_API_KEY` are all in
plaintext. `.gitignore` needs to be verified (or `.env` moved to a secret store)
before this directory is pushed anywhere public.

## Real use cases the harness now unlocks

Each of these was designed for but only the first has been proven end-to-end
against real Modal.

- **Long-horizon Monte Carlo** (proven). Portfolio glide-path simulations,
  retirement success probability, VaR estimation with hundreds of thousands of
  paths. `cpu=8`, sub-second on Modal.
- **GPU option pricing** (deployed, not yet invoked in prod). Asian / barrier
  options on an A10G with torch; cold-start expected ~30 s the first time, then
  warm. Same suspend/resume flow; only the workload changes.
- **Sandboxed code execution** (deployed). Model writes Python, harness runs it
  in `modal.Sandbox(block_network=True)` and returns stdout/last expression.
  Useful for "explore this CSV" or "prototype this indicator" without giving the
  model shell access to the harness host.
- **Backtests / batch jobs** (recipe only). Add a workload to
  `harness/workloads.py`, register it in `JobKind` + `FUNCTION_FOR_KIND`, expose
  it as an `@agent.tool` calling `_defer(...)`. No changes to suspend/resume.

## Follow-ups worth doing

- Add the webhook fast path: run `cloudflared tunnel --url http://localhost:8010`,
  set `HARNESS_PUBLIC_URL`, re-run the same query, and confirm
  `delivered_via=webhook` in the log. That closes the last unproven path.
- Invoke `run_gpu_job` at least once so the A10G image is warm and cold-start
  numbers are measured, not assumed.
- Point `PHOENIX_ENABLED=true` at a local Phoenix collector and capture a full
  span tree for one suspend/resume cycle — useful for the write-up alongside
  `simple-MVP`.
- Move `.env` secrets out of the repo or verify `.gitignore` before any push.
