"""Variant B: statistical prior, bounded decider caps, offline behavior, B runner."""

import numpy as np
import pandas as pd

from src.config import settings
from src.variant_b.decider import _capped, _evidence_agreement, decide
from src.variant_b.prior import build_prior


def _history(n=80, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n).date
    prices = 200 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, n)))
    return pd.DataFrame(
        {"open": prices, "high": prices + 1, "low": prices - 1, "close": prices,
         "volume": rng.integers(1e6, 5e6, n)},
        index=list(dates),
    )


def test_prior_shape_and_determinism():
    hist = _history()
    features = {"prev_close": float(hist["close"].iloc[-1]), "premarket_last": None}
    p1 = build_prior(hist, features)
    p2 = build_prior(hist, features)
    assert p1 == p2  # fixed MC seed -> deterministic
    assert p1["sigma_pct"] > 0
    assert p1["p10"] < p1["p50"] < p1["p90"]
    assert p1["centered_on"] == "prev_close_drift"
    assert abs(p1["center"] - p1["prev_close"]) / p1["prev_close"] < 0.01


def test_prior_centers_on_premarket_quote():
    hist = _history()
    prev = float(hist["close"].iloc[-1])
    prior = build_prior(hist, {"prev_close": prev, "premarket_last": prev * 1.02})
    assert prior["centered_on"] == "premarket_quote"
    assert prior["center"] == round(prev * 1.02, 2)


def _prior(center=200.0, sigma=0.02, prev=200.0):
    return {"center": center, "sigma_pct": sigma, "prev_close": prev,
            "drift_pct": 0.0, "p10": 195.0, "p50": 200.0, "p90": 205.0,
            "centered_on": "prev_close_drift"}


def test_adjustment_hard_cap():
    predicted, caps = _capped(_prior(), adj_sigma=3.0)  # way past ±0.8σ
    assert "adjustment_capped" in caps
    assert predicted == round(200.0 * (1 + settings.B_MAX_ADJ_SIGMA * 0.02), 2)


def test_move_cap_from_prev_close():
    # premarket gap put center 5% above prev; even adj=0 exceeds 1.5σ move -> capped
    predicted, caps = _capped(_prior(center=210.0, prev=200.0), adj_sigma=0.0)
    assert "move_capped" in caps
    assert predicted == round(200.0 + settings.B_MAX_MOVE_SIGMA * 0.02 * 200.0, 2)


def test_no_caps_within_bounds():
    predicted, caps = _capped(_prior(), adj_sigma=0.5)
    assert caps == []
    assert predicted == round(200.0 * 1.01, 2)


def test_evidence_agreement():
    assert _evidence_agreement({"items": []}) == 1.0
    items = [{"direction": "up", "strength": 0.5}, {"direction": "up", "strength": 0.5},
             {"direction": "down", "strength": 0.5}, {"direction": "neutral", "strength": 0.5}]
    assert _evidence_agreement({"items": items}) == 2 / 3


def test_offline_decide_abstains_to_prior_center():
    final = decide(_prior(), {"items": [], "summary": ""}, rows_b=[], client=None)
    assert final["predicted_close"] == 200.0
    assert final["adjustment_sigma"] == 0.0
    assert final["caps_applied"] == []
    assert 0.1 <= final["confidence"] <= 0.9
    assert final["direction"] in ("up", "down")
