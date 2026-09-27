"""Headline verdict is judged on all-time data, not the rolling window."""
from src import report
from src.config import settings


def _row(i: int, ape: float, baseline_ape: float) -> dict:
    return {"date": f"2024-{1 + i // 28:02d}-{1 + i % 28:02d}", "status": "scored",
            "actual_close": 100.0, "ape": ape, "baseline_ape": baseline_ape,
            "pass": ape <= 0.01, "directional_hit": True, "brier": 0.2}


def test_verdict_uses_all_time_not_rolling():
    w = settings.ROLLING_WINDOW
    # Early days badly lose to baseline; the recent rolling window beats it.
    rows = [_row(i, 0.03, 0.01) for i in range(w)] + [_row(w + i, 0.009, 0.01) for i in range(w)]
    m = report.build_metrics(rows, {})
    assert m["rolling"]["edge"] > 0 and m["all_time"]["edge"] < 0
    md = report.render_results_md(m, rows)
    assert "No edge yet" in md and "Edge confirmed" not in md
    assert "all-time MAPE" in md


def test_verdict_confirms_edge_on_all_time():
    rows = [_row(i, 0.005, 0.01) for i in range(30)]
    md = report.render_results_md(report.build_metrics(rows, {}), rows)
    assert "Edge confirmed" in md and "over 30 scored days" in md
