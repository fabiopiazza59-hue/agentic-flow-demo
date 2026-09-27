"""
Modal side of the harness: the workers the orchestrator spawns.

Three functions, one per resource profile:

    run_cpu_job      many-core CPU container   (Monte Carlo simulations)
    run_gpu_job      GPU container with torch  (option pricing)
    run_sandbox_job  creates a modal.Sandbox   (untrusted, model-written code)

Every worker follows the same contract:
1. execute the workload for `job["kind"]` with `job["payload"]`
2. build a `JobResultEnvelope`-shaped dict
3. POST it, HMAC-signed, to the harness webhook (fast path) - skipped when
   no callback URL is known
4. return the same envelope, so the harness reconciler can fetch it with
   `FunctionCall.get()` if the webhook was lost (fallback path)

Deploy:
    modal secret create agentic-harness-secret HARNESS_WEBHOOK_SECRET=<same value as the harness .env>
    modal deploy modal_app.py

Smoke test (runs a small CPU job remotely and prints the envelope):
    modal run modal_app.py
"""

from __future__ import annotations

import json
import os
import time
import traceback
from typing import Any, Callable

import modal

APP_NAME = os.environ.get("MODAL_APP_NAME", "agentic-modal-harness")
MINUTES = 60

app = modal.App(APP_NAME)

harness_secret = modal.Secret.from_name("agentic-harness-secret")  # HARNESS_WEBHOOK_SECRET [, HARNESS_CALLBACK_URL]

_base = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy>=1.26", "httpx>=0.27")
cpu_image = _base.add_local_python_source("harness")
gpu_image = _base.uv_pip_install("torch>=2.4").add_local_python_source("harness")
# Image used *inside* the sandbox. No harness code, no network.
sandbox_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy>=1.26", "pandas>=2.2")


# ---------------------------------------------------------------------------
# Shared worker plumbing
# ---------------------------------------------------------------------------

def _notify(envelope: dict[str, Any], callback_url: str | None) -> dict[str, Any]:
    """POST the envelope to the harness. Returns delivery metadata (never raises)."""
    url = callback_url or os.environ.get("HARNESS_CALLBACK_URL")
    if not url:
        return {"attempted": False, "reason": "no callback url; harness will poll"}
    secret = os.environ.get("HARNESS_WEBHOOK_SECRET")
    if not secret:
        return {"attempted": False, "reason": "HARNESS_WEBHOOK_SECRET missing in Modal secret"}

    import httpx
    from harness.security import signed_headers

    body = json.dumps(envelope, default=str).encode("utf-8")
    last_error = None
    for attempt in range(1, 4):
        try:
            resp = httpx.post(url, content=body, headers=signed_headers(secret, body), timeout=15)
            if resp.status_code < 500:
                return {"attempted": True, "status_code": resp.status_code, "attempts": attempt}
            last_error = f"HTTP {resp.status_code}"
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(2 ** attempt)
    return {"attempted": True, "delivered": False, "error": last_error, "attempts": 3}


def _run(job: dict[str, Any], executor: Callable[[str, dict[str, Any]], dict[str, Any]], function_name: str) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        call_id = modal.current_function_call_id()
    except Exception:  # noqa: BLE001 - not inside a Modal container (e.g. `modal run` local entrypoint)
        call_id = None
    worker: dict[str, Any] = {"backend": "modal", "function": function_name, "call_id": call_id}
    try:
        result = executor(job["kind"], job.get("payload", {}))
        envelope = {"job_id": job["job_id"], "run_id": job["run_id"], "status": "succeeded",
                    "result": result, "external_id": call_id}
    except Exception as exc:  # noqa: BLE001 - always report, never lose the job
        envelope = {"job_id": job["job_id"], "run_id": job["run_id"], "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}", "external_id": call_id}
        worker["traceback"] = traceback.format_exc()[-3000:]
    worker["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
    envelope["worker"] = worker
    worker["webhook"] = _notify(envelope, job.get("callback_url"))
    return envelope


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

@app.function(image=cpu_image, cpu=8.0, memory=8192, timeout=60 * MINUTES, retries=0, secrets=[harness_secret])
def run_cpu_job(job: dict[str, Any]) -> dict[str, Any]:
    from harness.workloads import execute_workload

    return _run(job, execute_workload, "run_cpu_job")


@app.function(image=gpu_image, gpu="A10G", timeout=60 * MINUTES, retries=0, secrets=[harness_secret])
def run_gpu_job(job: dict[str, Any]) -> dict[str, Any]:
    from harness.workloads import execute_workload

    return _run(job, execute_workload, "run_gpu_job")


@app.function(image=cpu_image, timeout=30 * MINUTES, retries=0, secrets=[harness_secret])
def run_sandbox_job(job: dict[str, Any]) -> dict[str, Any]:
    """Execute model-written code inside an isolated `modal.Sandbox` (no network, own filesystem)."""
    from harness.workloads import SANDBOX_DRIVER

    def _in_sandbox(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        code = str(payload.get("code", ""))
        if not code.strip():
            raise ValueError("empty code")
        timeout = int(payload.get("timeout_seconds", 5 * MINUTES))
        sb = modal.Sandbox.create(
            app=app, image=sandbox_image, timeout=timeout, cpu=2.0, memory=2048, block_network=True,
        )
        try:
            proc = sb.exec("python", "-c", SANDBOX_DRIVER, timeout=timeout)
            proc.stdin.write(code)
            proc.stdin.write_eof()
            proc.stdin.drain()
            proc.wait()
            stdout = proc.stdout.read()
            stderr = proc.stderr.read()
        finally:
            sb.terminate()

        marker = "__HARNESS_RESULT__"
        for line in reversed(stdout.splitlines()):
            if line.startswith(marker):
                parsed = json.loads(line[len(marker):])
                return {"workload": "sandbox_code", "device": "modal.Sandbox", "sandbox_id": sb.object_id,
                        "purpose": payload.get("purpose"), **parsed}
        return {"workload": "sandbox_code", "device": "modal.Sandbox", "sandbox_id": sb.object_id,
                "success": False, "error": "sandbox produced no result",
                "stdout": stdout[-4000:], "stderr": stderr[-4000:], "returncode": proc.returncode}

    return _run(job, _in_sandbox, "run_sandbox_job")


# ---------------------------------------------------------------------------
# Smoke test: `modal run modal_app.py --kind cpu_monte_carlo`
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(kind: str = "cpu_monte_carlo"):
    payloads = {
        "cpu_monte_carlo": {"initial_value": 100_000, "annual_contribution": 12_000, "years": 30, "n_paths": 200_000, "seed": 7},
        "gpu_pricing": {"spot": 140, "strike": 150, "volatility": 0.45, "maturity_years": 1.0, "n_paths": 2_000_000, "seed": 7},
        "sandbox_code": {"code": "import numpy as np\nresult = {'mean': float(np.arange(10).mean())}\nprint('hi')", "purpose": "smoke"},
    }
    fn = {"cpu_monte_carlo": run_cpu_job, "gpu_pricing": run_gpu_job, "sandbox_code": run_sandbox_job}[kind]
    job = {"job_id": "job_smoke", "run_id": "run_smoke", "kind": kind, "payload": payloads[kind], "callback_url": None}
    print(json.dumps(fn.remote(job), indent=2, default=str))
