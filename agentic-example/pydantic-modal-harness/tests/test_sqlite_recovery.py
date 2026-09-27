"""A restart mid-flight: the SQLite store keeps the suspended run and the reconciler finishes it."""

from tests.conftest import PollSucceedsDispatcher, scripted_model, wait_for_status
from harness.config import Settings


async def test_suspended_run_survives_restart(make_client, settings, tmp_path):
    durable = settings.model_copy(update={"store_backend": "sqlite", "sqlite_path": str(tmp_path / "h.db")})
    model = scripted_model("price_option_gpu", {"spot": 1, "strike": 1},
                           final=lambda c: f"PRICE={c['result']['price']} VIA={c['delivered_via']}")

    # Process 1: run suspends, then "crashes" before any result arrives.
    app1, client1 = await make_client(model, dispatcher=PollSucceedsDispatcher(), override=durable)
    run_id = (await client1.post("/runs", json={"query": "price"})).json()["run_id"]
    await wait_for_status(client1, run_id, {"waiting_for_jobs"})
    await app1.state.runtime.store.close()

    # Process 2: same database, fresh runtime. Startup reconciliation polls Modal and resumes.
    app2, client2 = await make_client(model, dispatcher=PollSucceedsDispatcher(), override=durable)
    assert (await client2.get(f"/runs/{run_id}")).json()["status"] == "waiting_for_jobs"
    assert await app2.state.runtime.jobs.reconcile_once(ignore_grace=True) == 1
    run = await wait_for_status(client2, run_id, {"completed", "failed"})
    assert run["status"] == "completed" and run["output"] == "PRICE=12.34 VIA=poll"
    await app2.state.runtime.store.close()
