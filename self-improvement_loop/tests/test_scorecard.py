"""Per-strategy scorecard updates + weight hints (the self-improvement signal)."""

from src.evals.scorecard import update_scorecards, weight_hints


def test_winner_gets_win_and_weights_favor_accuracy():
    sc = {}
    preds = {
        "technical": {"predicted_close": 100.5},
        "momentum": {"predicted_close": 103.0},
    }
    # Run several days where technical is consistently closer to actual=100.4
    for _ in range(4):
        sc = update_scorecards(sc, preds, actual=100.4)
    assert sc["technical"]["wins"] == 4
    assert sc["momentum"]["wins"] == 0
    sc = weight_hints(sc)
    # technical (lower MAPE) should carry more weight than momentum
    assert sc["technical"]["weight_hint"] > sc["momentum"]["weight_hint"]


def test_empty_predictions_noop():
    sc = {"x": {"n": 1}}
    assert update_scorecards(sc, {}, actual=100.0) == sc
