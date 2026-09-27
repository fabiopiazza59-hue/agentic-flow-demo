"""Webhook (push) and reconciler (poll) delivery paths, idempotency and cancellation."""

import json

from tests.conftest import NeverFinishesDispatcher, PollSucceedsDispatcher, scripted_model, wait_for_status
from harness.security import signed_headers

GPU_ARGS = {"spot": 140, "strike": 150, "volatility": 0.45}


def _final(content):
    return f"PRICE={content['result']['price']} VIA={content['delivered_via']}"


async def _suspended_run(make_client, dispatcher):
    model = scripted_model("price_option_gpu", GPU_ARGS, final=_final)
    app, client = await make_client(model, dispatcher=dispatcher)
    run_id = (await client.post("/runs", json={"query": "price it"})).json()["run_id"]
    run = await wait_for_status(client, run_id, {"waiting_for_jobs", "completed", "failed"})
    assert run["status"] == "waiting_for_jobs", run
    assert run["jobs"][0]["status"] == "running" and run["jobs"][0]["external_id"].startswith("fc-")
    return app, client, run_id, run["jobs"][0]


async def test_webhook_delivers_result_and_resumes(make_client):
    dispatcher = NeverFinishesDispatcher()
    app, client, run_id, job = await _suspended_run(make_client, dispatcher)

    envelope = {"job_id": job["job_id"], "run_id": run_id, "status": "succeeded",
                "result": {"workload": "gpu_pricing", "price": 9.87, "device": "torch:cuda:A10G", "elapsed_ms": 1234},
                "external_id": job["external_id"], "worker": {"backend": "modal", "function": "run_gpu_job"}}
    body = json.dumps(envelope).encode()
    resp = await client.post("/webhooks/modal", content=body, headers=signed_headers("test-secret", body))
    assert resp.status_code == 200 and resp.json() == {"accepted": True, "duplicate": False, "status": "succeeded"}

    run = await wait_for_status(client, run_id, {"completed", "failed"})
    assert run["status"] == "completed" and run["output"] == "PRICE=9.87 VIA=webhook"
    assert run["jobs"][0]["delivered_via"] == "webhook"

    # Idempotent: a replayed webhook (or a later poll) is acknowledged but ignored.
    dup = await client.post("/webhooks/modal", content=body, headers=signed_headers("test-secret", body))
    assert dup.status_code == 200 and dup.json()["duplicate"] is True
    assert (await client.get(f"/runs/{run_id}")).json()["resume_count"] == 1


async def test_webhook_rejects_bad_signature_and_unknown_job(make_client):
    dispatcher = NeverFinishesDispatcher()
    app, client, run_id, job = await _suspended_run(make_client, dispatcher)
    body = json.dumps({"job_id": job["job_id"], "run_id": run_id, "status": "succeeded", "result": {}}).encode()

    bad = await client.post("/webhooks/modal", content=body, headers=signed_headers("wrong-secret", body))
    assert bad.status_code == 401

    unknown_body = json.dumps({"job_id": "job_nope", "run_id": run_id, "status": "succeeded", "result": {}}).encode()
    unknown = await client.post("/webhooks/modal", content=unknown_body, headers=signed_headers("test-secret", unknown_body))
    assert unknown.status_code == 404

    # Still waiting: nothing above should have touched the run.
    assert (await client.get(f"/runs/{run_id}")).json()["status"] == "waiting_for_jobs"


async def test_reconciler_polls_when_webhook_never_arrives(make_client):
    dispatcher = PollSucceedsDispatcher()
    app, client, run_id, job = await _suspended_run(make_client, dispatcher)

    resolved = await app.state.runtime.jobs.reconcile_once(ignore_grace=True)
    assert resolved == 1
    run = await wait_for_status(client, run_id, {"completed", "failed"})
    assert run["status"] == "completed" and run["output"] == "PRICE=12.34 VIA=poll"
    assert run["jobs"][0]["delivered_via"] == "poll"

    # A second pass finds nothing to do.
    assert await app.state.runtime.jobs.reconcile_once(ignore_grace=True) == 0


async def test_failed_job_is_reported_to_the_model(make_client):
    dispatcher = NeverFinishesDispatcher()
    model = scripted_model("price_option_gpu", GPU_ARGS, final=lambda c: f"STATUS={c['status']} ERR={c['error']}")
    app, client = await make_client(model, dispatcher=dispatcher)
    run_id = (await client.post("/runs", json={"query": "price it"})).json()["run_id"]
    run = await wait_for_status(client, run_id, {"waiting_for_jobs"})
    job = run["jobs"][0]

    body = json.dumps({"job_id": job["job_id"], "run_id": run_id, "status": "failed", "error": "CUDA out of memory"}).encode()
    assert (await client.post("/webhooks/modal", content=body, headers=signed_headers("test-secret", body))).status_code == 200
    run = await wait_for_status(client, run_id, {"completed", "failed"})
    assert run["status"] == "completed" and run["output"] == "STATUS=failed ERR=CUDA out of memory"


async def test_cancel_run_cancels_jobs(make_client):
    dispatcher = NeverFinishesDispatcher()
    app, client, run_id, job = await _suspended_run(make_client, dispatcher)
    resp = await client.post(f"/runs/{run_id}/cancel")
    assert resp.status_code == 200 and resp.json()["status"] == "cancelled"
    assert dispatcher.cancelled == [job["job_id"]]
    assert resp.json()["jobs"][0]["status"] == "cancelled"

    # Late result for a cancelled job is a no-op duplicate.
    body = json.dumps({"job_id": job["job_id"], "run_id": run_id, "status": "succeeded", "result": {"price": 1}}).encode()
    late = await client.post("/webhooks/modal", content=body, headers=signed_headers("test-secret", body))
    assert late.json()["duplicate"] is True
    assert (await client.get(f"/runs/{run_id}")).json()["status"] == "cancelled"
