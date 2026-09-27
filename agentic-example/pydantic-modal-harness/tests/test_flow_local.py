"""End-to-end: agent defers -> local job completes -> run resumes -> final answer."""

import json

from tests.conftest import scripted_model, text_model, wait_for_status


async def test_fast_tool_only_run_completes_without_suspension(make_client):
    model = scripted_model("calculate", {"expression": "15 * 3"})
    app, client = await make_client(model)
    resp = await client.post("/runs", json={"query": "what is 15*3"})
    assert resp.status_code == 202
    run = await wait_for_status(client, resp.json()["run_id"], {"completed", "failed"})
    assert run["status"] == "completed", run
    assert "45" in run["output"]
    assert run["resume_count"] == 0 and run["jobs"] == []


async def test_deferred_job_suspends_and_resumes(make_client):
    model = scripted_model("run_monte_carlo", {
        "initial_value": 1000, "annual_contribution": 100, "years": 3, "n_paths": 5000,
    }, final=lambda content: f"MEDIAN={content['result']['median_final_value']:.0f} VIA={content['delivered_via']}")
    app, client = await make_client(model)

    resp = await client.post("/runs", json={"query": "simulate"})
    run_id = resp.json()["run_id"]
    run = await wait_for_status(client, run_id, {"completed", "failed"})
    assert run["status"] == "completed", run
    assert run["output"].startswith("MEDIAN=") and "VIA=local" in run["output"]
    assert run["resume_count"] == 1
    assert len(run["jobs"]) == 1
    job = run["jobs"][0]
    assert job["status"] == "succeeded" and job["kind"] == "cpu_monte_carlo" and job["delivered_via"] == "local"
    assert job["result"]["workload"] == "cpu_monte_carlo"

    # Event timeline covers the whole lifecycle in order.
    events = await app.state.runtime.store.events_for_run(run_id)
    types = [e.type for e in events]
    for expected in ("run.created", "run.started", "job.created", "job.dispatched", "run.suspended",
                     "job.completed", "run.resumed", "run.completed"):
        assert expected in types, types
    assert types.index("run.suspended") < types.index("run.resumed") < types.index("run.completed")

    # The job result was consumed exactly once.
    stored = await app.state.runtime.store.get_job(job["job_id"])
    assert stored.consumed is True


async def test_sse_stream_replays_history_and_ends(make_client):
    model = text_model("plain answer")
    app, client = await make_client(model)
    run_id = (await client.post("/runs", json={"query": "hi"})).json()["run_id"]
    await wait_for_status(client, run_id, {"completed"})

    names, payloads = [], []
    async with client.stream("GET", f"/runs/{run_id}/events") as stream:
        assert stream.status_code == 200
        async for line in stream.aiter_lines():
            if line.startswith("event:"):
                names.append(line.split(":", 1)[1].strip())
            elif line.startswith("data:"):
                payloads.append(json.loads(line.split(":", 1)[1].strip()))
    assert names[0] == "run.created" and names[-2] == "run.completed" and names[-1] == "end"
    assert payloads[-2]["output"] == "plain answer"


async def test_dispatch_failure_returns_inline_tool_error(make_client):
    from tests.conftest import SubmitFailsDispatcher

    model = scripted_model("price_option_gpu", {"spot": 100, "strike": 100},
                           final=lambda content: f"STATUS={content['status']} ERR={content['error']}")
    app, client = await make_client(model, dispatcher=SubmitFailsDispatcher())
    run_id = (await client.post("/runs", json={"query": "price"})).json()["run_id"]
    run = await wait_for_status(client, run_id, {"completed", "failed"})
    assert run["status"] == "completed"
    assert run["output"].startswith("STATUS=failed") and "modal is down" in run["output"]
    assert run["resume_count"] == 0          # never suspended: failure came back inline
    assert run["jobs"][0]["status"] == "failed"


async def test_unknown_run_404(make_client):
    app, client = await make_client(text_model())
    assert (await client.get("/runs/run_missing")).status_code == 404
    assert (await client.get("/runs/run_missing/events")).status_code == 404


async def test_health(make_client):
    app, client = await make_client(text_model())
    body = (await client.get("/health")).json()
    assert body["status"] == "healthy" and body["backend"] == "local"
