"""Guardrail gates: consensus math, shrink behavior, calibrated confidence."""

from src.agents.meta_judge import _user_prompt
from src.config import settings
from src.evals.gates import apply_gates, calibrated_confidence, directional_consensus

FEATURES = {"prev_close": 200.0, "premarket_gap_pct": 0.0}


def _preds(directions):
    return {
        f"a{i}": {"predicted_close": 201.0, "direction": d, "confidence": 0.5, "rationale": ""}
        for i, d in enumerate(directions)
    }


def _final(predicted=202.0, direction="up"):
    return {"predicted_close": predicted, "direction": direction,
            "confidence": 0.5, "weights": {}, "rationale": "test"}


def _scored_row(date, predicted, actual, prior, seed=False):
    row = {
        "date": date, "status": "scored", "prior_close": prior,
        "predicted_close": predicted, "actual_close": actual,
        "ape": abs(predicted - actual) / actual,
        "baseline_ape": abs(prior - actual) / actual,
        "pass": abs(predicted - actual) / actual <= 0.01,
        "directional_hit": (predicted > prior) == (actual > prior),
        "brier": 0.25,
    }
    if seed:
        row["seed"] = True
    return row


def _losing_rows(n=6):
    """Model APE consistently worse than baseline -> negative rolling edge."""
    return [_scored_row(f"2026-01-{i+1:02d}", predicted=206.0, actual=200.0, prior=201.0)
            for i in range(n)]


def test_directional_consensus():
    assert directional_consensus(_preds(["up"] * 5), "up") == 1.0
    assert directional_consensus(_preds(["up", "up", "down", "down", "down"]), "up") == 0.4
    assert directional_consensus({}, "up") == 1.0  # unknowable -> permissive


def test_no_gates_with_consensus_and_no_history():
    final, gates = apply_gates(_final(), _preds(["up"] * 5), rows=[], features=FEATURES)
    assert gates == []
    assert final["predicted_close"] == 202.0
    assert 0.1 <= final["confidence"] <= 0.9  # calibrated even when no gate fires


def test_low_consensus_shrinks_move():
    final, gates = apply_gates(_final(), _preds(["up", "up", "down", "down", "down"]),
                               rows=[], features=FEATURES)
    assert gates == ["low_consensus_shrink"]
    assert final["predicted_close"] == 201.0  # +2.0 move halved
    assert final["direction"] == "up"


def test_negative_edge_shrinks_move():
    final, gates = apply_gates(_final(), _preds(["up"] * 5),
                               rows=_losing_rows(), features=FEATURES)
    assert gates == ["negative_edge_shrink"]
    assert final["predicted_close"] == 201.0


def test_both_gates_compound():
    final, gates = apply_gates(_final(), _preds(["up", "up", "down", "down", "down"]),
                               rows=_losing_rows(), features=FEATURES)
    assert set(gates) == {"low_consensus_shrink", "negative_edge_shrink"}
    assert final["predicted_close"] == 200.5  # +2.0 move quartered
    assert "gates:" in final["rationale"]


def test_seed_rows_do_not_trigger_edge_gate():
    rows = [_scored_row(f"2026-01-{i+1:02d}", 206.0, 200.0, 201.0, seed=True) for i in range(6)]
    _, gates = apply_gates(_final(), _preds(["up"] * 5), rows=rows, features=FEATURES)
    assert gates == []


def test_calibrated_confidence_bounds_and_discrimination():
    stats = {"n": 10, "pass_rate": 0.35}
    high = calibrated_confidence(stats, consensus=1.0)
    low = calibrated_confidence(stats, consensus=0.4)
    assert high > low  # varies with agreement instead of sitting flat
    assert 0.1 <= low <= high <= 0.9
    # cold start: falls back to 0.4 base
    assert calibrated_confidence({"n": 0, "pass_rate": None}, 0.75) == 0.4


def test_original_final_not_mutated():
    final = _final()
    apply_gates(final, _preds(["up", "down", "down", "down", "down"]), rows=[], features=FEATURES)
    assert final["predicted_close"] == 202.0 and final["confidence"] == 0.5


def test_judge_prompt_keeps_newest_strategy_notes():
    """STRATEGY.md is append-only (newest last); truncation must drop the oldest, not the newest."""
    strategy_md = "OLDEST-NOTE\n" + ("x" * 5000) + "\nNEWEST-NOTE"
    prompt = _user_prompt({}, {}, strategy_md, [], FEATURES)
    assert "NEWEST-NOTE" in prompt
    assert "OLDEST-NOTE" not in prompt


def test_settings_gate_defaults():
    assert 0 < settings.GATE_SHRINK < 1
    assert 0.5 <= settings.GATE_CONSENSUS_MIN <= 1
