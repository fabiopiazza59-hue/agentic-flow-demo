"""Deterministic guardrails applied to the meta-judge's final prediction.

The failure diagnosis (learnings/WHATS_NOT_WORKING.md) kept prescribing the same fixes —
pull back when the desk has no directional conviction, defer to the random-walk baseline
while the model is losing to it, make confidence discriminative — but as prompt-level
advice they never stuck. These gates enforce them in code, after synthesis:

  low_consensus_shrink : analyst directional agreement with the blend < GATE_CONSENSUS_MIN
                         -> shrink the predicted move toward prior close
  negative_edge_shrink : rolling edge vs baseline (last GATE_EDGE_WINDOW scored days) < 0
                         -> shrink the predicted move toward prior close

Confidence is always replaced by a calibrated value: rolling pass rate, nudged by analyst
consensus, so it finally varies with conditions instead of sitting flat at ~0.5.

Gates shrink toward the prior close (the baseline) rather than hard-abstaining so the model
still expresses a view and per-analyst scorecards keep accruing learning signal.
"""

from __future__ import annotations

from ..config import settings
from .metrics import aggregate


def directional_consensus(analyst_predictions: dict, final_direction: str | None) -> float:
    """Share of analysts whose direction agrees with the blend's direction (1.0 if unknowable)."""
    dirs = [p.get("direction") for p in analyst_predictions.values()
            if p.get("direction") in ("up", "down")]
    if not dirs or final_direction not in ("up", "down"):
        return 1.0
    return dirs.count(final_direction) / len(dirs)


def calibrated_confidence(stats: dict, consensus: float) -> float:
    """Anchor confidence to the realized rolling pass rate, +/- an analyst-agreement nudge.

    Full agreement (1.0) adds 0.1; a split desk (0.5) subtracts 0.1. Falls back to a 0.4
    base until enough real predictions have been scored.
    """
    enough = stats.get("n", 0) >= settings.GATE_MIN_SCORED
    base = stats["pass_rate"] if enough and stats.get("pass_rate") is not None else 0.4
    conf = base + (consensus - 0.75) * 0.4
    return round(min(0.9, max(0.1, conf)), 3)


def apply_gates(final: dict, analyst_predictions: dict, rows: list[dict],
                features: dict) -> tuple[dict, list[str]]:
    """Return (gated final prediction, list of gates that fired). Does not mutate `final`."""
    prev_close = float(features.get("prev_close") or 0.0)
    gates: list[str] = []
    if prev_close <= 0:
        return final, gates

    final = dict(final)
    move = float(final["predicted_close"]) - prev_close
    consensus = directional_consensus(analyst_predictions, final.get("direction"))
    stats = aggregate(rows, window=settings.GATE_EDGE_WINDOW)

    if len(analyst_predictions) >= 3 and consensus < settings.GATE_CONSENSUS_MIN:
        move *= settings.GATE_SHRINK
        gates.append("low_consensus_shrink")
    edge = stats.get("edge")
    if stats.get("n", 0) >= settings.GATE_MIN_SCORED and edge is not None and edge < 0:
        move *= settings.GATE_SHRINK
        gates.append("negative_edge_shrink")

    if gates:
        final["predicted_close"] = round(prev_close + move, 2)
        if final["predicted_close"] != prev_close:
            final["direction"] = "up" if final["predicted_close"] > prev_close else "down"

    final["confidence"] = calibrated_confidence(stats, consensus)
    edge_txt = f"{edge * 100:.2f}%" if edge is not None else "n/a"
    note = (f"[gates: {', '.join(gates) if gates else 'none'}; "
            f"consensus {consensus:.2f}; rolling edge {edge_txt}]")
    final["rationale"] = (str(final.get("rationale", "")) + " " + note).strip()
    return final, gates
