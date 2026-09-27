"""Eval metric tests — boundary correctness is the contract the loop optimizes."""

from src.evals import metrics


def test_ape():
    assert metrics.ape(100.0, 100.0) == 0.0
    assert abs(metrics.ape(101.0, 100.0) - 0.01) < 1e-9


def test_pass_boundary():
    # within ±1% passes; just over fails
    assert metrics.is_pass(100.99, 100.0) is True   # 0.99%
    assert metrics.is_pass(101.00, 100.0) is True   # exactly 1.0% -> PASS (<=)
    assert metrics.is_pass(101.01, 100.0) is False  # 1.01%


def test_direction_and_hit():
    assert metrics.direction(105, 100) == "up"
    assert metrics.direction(100, 100) == "down"  # flat -> down
    assert metrics.directional_hit(predicted=106, actual=104, prior_close=100) is True
    assert metrics.directional_hit(predicted=106, actual=98, prior_close=100) is False


def test_baseline_and_beats():
    # prediction closer than prior-close baseline => beats baseline
    assert metrics.beats_baseline(predicted=101.5, actual=101, prior_close=100) is True
    # prediction further than baseline => does not beat
    assert metrics.beats_baseline(predicted=90, actual=101, prior_close=100) is False


def test_brier():
    assert metrics.brier_component(1.0, True) == 0.0
    assert metrics.brier_component(0.0, True) == 1.0
    assert abs(metrics.brier_component(0.5, False) - 0.25) < 1e-9


def test_score_row_and_aggregate():
    row = {"prior_close": 100.0, "predicted_close": 101.0, "confidence": 0.8}
    fields = metrics.score_row(row, actual=100.5)
    assert fields["pass"] is True
    assert fields["status"] == "scored"
    rows = [
        {"date": "2024-01-02", "status": "scored", "actual_close": 100.5,
         "ape": 0.005, "baseline_ape": 0.004, "pass": True, "directional_hit": True, "brier": 0.04},
        {"date": "2024-01-03", "status": "scored", "actual_close": 99.0,
         "ape": 0.02, "baseline_ape": 0.03, "pass": False, "directional_hit": False, "brier": 0.5},
    ]
    agg = metrics.aggregate(rows)
    assert agg["n"] == 2
    assert agg["pass_rate"] == 0.5
    assert agg["edge"] is not None
