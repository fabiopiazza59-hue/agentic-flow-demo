from harness.workloads import cpu_monte_carlo, execute_workload, gpu_pricing, sandbox_code_local


def test_cpu_monte_carlo_is_deterministic_with_seed():
    a = cpu_monte_carlo({"n_paths": 5_000, "years": 5, "seed": 3})
    b = cpu_monte_carlo({"n_paths": 5_000, "years": 5, "seed": 3})
    assert a["median_final_value"] == b["median_final_value"]
    assert 0.0 <= a["probability_reaching_target"] <= 1.0
    assert a["p10"] <= a["median_final_value"] <= a["p90"]


def test_gpu_pricing_runs_on_cpu_fallback():
    out = gpu_pricing({"spot": 100, "strike": 100, "n_paths": 20_000, "steps": 12, "seed": 1})
    assert out["price"] > 0
    assert out["confidence_95"][0] <= out["price"] <= out["confidence_95"][1]
    assert out["device"].endswith("cpu")


def test_sandbox_local_executes_and_blocks():
    ok = sandbox_code_local({"code": "x = np.arange(4)\nresult = {'sum': int(x.sum())}\nprint('done')"})
    assert ok["success"] and ok["result"] == {"sum": 6} and "done" in ok["stdout"]
    blocked = sandbox_code_local({"code": "import os\nresult = os.listdir('.')"})
    assert not blocked["success"] and "blocked" in blocked["error"]


def test_unknown_kind():
    try:
        execute_workload("nope", {})
    except ValueError as exc:
        assert "unknown workload" in str(exc)
    else:
        raise AssertionError("expected ValueError")
