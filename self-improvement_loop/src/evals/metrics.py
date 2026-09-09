"""Pure, unit-testable evaluation metrics.

Eval contract (see spec/spec.md §3):
  PASS              : ape <= PASS_THRESHOLD (±1%)
  direction         : 'up' if close > prior_close else 'down'
  baseline          : random walk -> baseline_pred = prior_close
  edge              : baseline_mape - mape  (positive => real skill)  [headline verdict]
  brier_component   : (confidence - outcome)^2
"""

from __future__ import annotations

from ..config import settings
from . import integrity


def ape(predicted: float, actual: float) -> float:
    """Absolute percent error as a fraction (0.01 == 1%)."""
    if actual == 0:
        raise ValueError("actual close cannot be zero")
    return abs(predicted - actual) / abs(actual)


def is_pass(predicted: float, actual: float, threshold: float | None = None) -> bool:
    threshold = settings.PASS_THRESHOLD if threshold is None else threshold
    return ape(predicted, actual) <= threshold


def direction(close: float, prior_close: float) -> str:
    return "up" if close > prior_close else "down"


def directional_hit(predicted: float, actual: float, prior_close: float) -> bool:
    return direction(predicted, prior_close) == direction(actual, prior_close)


def baseline_ape(prior_close: float, actual: float) -> float:
    """Random-walk baseline: predict that today's close == prior close."""
    return ape(prior_close, actual)


def beats_baseline(predicted: float, actual: float, prior_close: float) -> bool:
    return ape(predicted, actual) < baseline_ape(prior_close, actual)


def brier_component(confidence: float, outcome_pass: bool) -> float:
    o = 1.0 if outcome_pass else 0.0
    c = max(0.0, min(1.0, float(confidence)))
    return (c - o) ** 2


def score_row(row: dict, actual: float) -> dict:
    """Compute all eval fields for a pending ledger row given the actual close.

    Returns a dict of fields to merge into the row (does not mutate input).
    """
    prior = float(row["prior_close"])
    predicted = float(row["predicted_close"])
    conf = float(row.get("confidence", 0.5))
    _ape = ape(predicted, actual)
    passed = _ape <= settings.PASS_THRESHOLD
    return {
        "actual_close": float(actual),
        "ape": _ape,
        "pass": passed,
        "directional_hit": directional_hit(predicted, actual, prior),
        "baseline_ape": baseline_ape(prior, actual),
        "beats_baseline": beats_baseline(predicted, actual, prior),
        "brier": brier_component(conf, passed),
        "status": "scored",
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def aggregate(rows: list[dict], window: int | None = None,
              pre_open_only: bool = False) -> dict:
    """Aggregate model metrics over scored, non-seed rows.

    Backfill seed rows (`seed=True`) are excluded so headline metrics reflect only real ensemble
    predictions. With `pre_open_only`, rows created at or after their own session's open are
    excluded too — those saw part of the tape they were forecasting (see evals/integrity.py).
    If window is set, use the most recent N qualifying rows.
    """
    scored = [
        r for r in rows
        if r.get("status") == "scored" and r.get("actual_close") is not None and not r.get("seed")
        and not (pre_open_only and integrity.is_late(r))
    ]
    scored.sort(key=lambda r: r.get("date", ""))
    windowed = scored[-window:] if window else scored

    def vals(key: str) -> list[float]:
        return [float(r[key]) for r in windowed if r.get(key) is not None]

    passes = [1.0 if r.get("pass") else 0.0 for r in windowed]
    hits = [1.0 if r.get("directional_hit") else 0.0 for r in windowed]
    mape = _mean(vals("ape"))
    base_mape = _mean(vals("baseline_ape"))
    edge = (base_mape - mape) if (mape is not None and base_mape is not None) else None

    return {
        "n": len(windowed),
        "n_all_time": len(scored),
        "pass_rate": _mean(passes),
        "directional_accuracy": _mean(hits),
        "mape": mape,
        "baseline_mape": base_mape,
        "edge": edge,
        "beats_baseline_overall": (edge is not None and edge > 0),
        "brier": _mean(vals("brier")),
    }
