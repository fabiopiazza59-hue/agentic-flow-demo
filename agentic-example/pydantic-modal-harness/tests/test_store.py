import pytest

from harness.models import Job, JobKind, JobStatus, Run, RunEvent, RunStatus
from harness.store import InMemoryRunStore, SqliteRunStore


@pytest.fixture(params=["memory", "sqlite"])
async def store(request, tmp_path):
    s = InMemoryRunStore() if request.param == "memory" else SqliteRunStore(str(tmp_path / "t.db"))
    yield s
    await s.close()


async def test_run_job_event_roundtrip(store):
    run = Run(query="q")
    await store.create_run(run)
    job = Job(run_id=run.run_id, tool_call_id="tc1", tool_name="run_monte_carlo", kind=JobKind.CPU_MONTE_CARLO, payload={"a": 1})
    await store.add_job(job)

    loaded = await store.get_run(run.run_id)
    assert loaded is not None and loaded.job_ids == [job.job_id]

    job.status = JobStatus.RUNNING
    await store.save_job(job)
    assert [j.job_id for j in await store.jobs_with_status([JobStatus.RUNNING])] == [job.job_id]
    assert await store.jobs_with_status([JobStatus.SUCCEEDED]) == []

    run.status = RunStatus.WAITING_FOR_JOBS
    run.messages = [{"kind": "request", "parts": []}]
    await store.save_run(run)
    assert (await store.get_run(run.run_id)).messages == [{"kind": "request", "parts": []}]

    e1 = await store.append_event(RunEvent(run_id=run.run_id, type="run.started"))
    e2 = await store.append_event(RunEvent(run_id=run.run_id, type="run.suspended", data={"x": 1}))
    assert (e1.seq, e2.seq) == (1, 2)
    events = await store.events_for_run(run.run_id)
    assert [e.type for e in events] == ["run.started", "run.suspended"]

    assert [r.run_id for r in await store.list_runs()] == [run.run_id]
    assert await store.get_run("nope") is None
    assert await store.get_job("nope") is None
