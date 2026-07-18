"""Bounded decision layer for variant B — evidence adjusts the prior, code enforces the caps.

predict-raven's core discipline transplanted: the World Cup forecaster lets evidence move its
statistical prior by at most ±8pp, and risk limits live in the service layer, not in prompts.
Here the decider proposes an adjustment in units of the prior's sigma; the code clamps it to
±B_MAX_ADJ_SIGMA, then clamps the total move from prev close to ±B_MAX_MOVE_SIGMA·σ. With no
evidence and no client, the prediction IS the prior center (abstain).
"""

from __future__ import annotations

import json

from ..config import settings
from ..evals.gates import calibrated_confidence
from ..evals.metrics import aggregate
from ..utils import extract_json, safe_float

_SYSTEM = (
    "You are the decision agent of an AMZN close-prediction desk that works Bayesian-style: a "
    "statistical prior for today's close is given, plus today's gathered evidence and the desk's "
    "recent failure log. Your job is ONLY to size how far the evidence justifies moving off the "
    "prior, in units of the prior's daily sigma. Positive = above prior center, negative = below. "
    "Weak/contradictory evidence deserves ~0. Your adjustment is hard-capped at "
    f"±{0}σ by the system, so spend your range wisely.\n"
    "Respond with ONLY JSON:\n"
    '{"adjustment_sigma": <number>, "rationale": "<2-3 sentences>"}'
)


def _evidence_agreement(pulse: dict) -> float:
    """Share of directional evidence agreeing with its own majority direction (1.0 if none)."""
    dirs = [it["direction"] for it in pulse.get("items", []) if it["direction"] != "neutral"]
    if not dirs:
        return 1.0
    majority = max(set(dirs), key=dirs.count)
    return dirs.count(majority) / len(dirs)


def _capped(prior: dict, adj_sigma: float) -> tuple[float, list[str]]:
    """Apply both hard caps; return (predicted_close, caps_applied)."""
    caps: list[str] = []
    sigma, center, prev = prior["sigma_pct"], prior["center"], prior["prev_close"]

    max_adj = settings.B_MAX_ADJ_SIGMA
    if abs(adj_sigma) > max_adj:
        adj_sigma = max_adj if adj_sigma > 0 else -max_adj
        caps.append("adjustment_capped")

    predicted = center * (1.0 + adj_sigma * sigma)

    max_move = settings.B_MAX_MOVE_SIGMA * sigma * prev
    if abs(predicted - prev) > max_move:
        predicted = prev + (max_move if predicted > prev else -max_move)
        caps.append("move_capped")

    return round(predicted, 2), caps


def _failures_tail() -> str:
    p = settings.LEARNINGS_B_DIR / "FAILURES.md"
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")[-settings.B_FAILURES_TAIL_CHARS:]


def _propose(prior: dict, pulse: dict, client) -> tuple[float, str]:
    """Return (raw adjustment_sigma, rationale). Offline or on failure: (0, abstain note)."""
    if client is None:
        return 0.0, "[offline] no evidence — prediction is the statistical prior center."
    if not pulse.get("items"):
        return 0.0, "No usable evidence gathered — abstaining to the prior center."
    system = _SYSTEM.replace("±0σ", f"±{settings.B_MAX_ADJ_SIGMA}σ")
    user = (
        f"Prior for today's {settings.SYMBOL} close (JSON):\n{json.dumps(prior, indent=2)}\n\n"
        f"Evidence pulse (JSON):\n{json.dumps(pulse, indent=2)}\n\n"
        f"Recent failures of THIS strategy (learn from them):\n{_failures_tail() or '(none yet)'}"
    )
    try:
        resp = client.messages.create(
            model=settings.B_DECIDER_MODEL,
            max_tokens=settings.B_DECIDER_MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        raw = extract_json(text)
        adj = safe_float(raw.get("adjustment_sigma"), 0.0) or 0.0
        return adj, str(raw.get("rationale", ""))[:1000]
    except Exception:
        return 0.0, "Decider call failed — abstaining to the prior center."


def decide(prior: dict, pulse: dict, rows_b: list[dict], client=None) -> dict:
    """Produce variant B's final prediction dict (same shape as A's `final` + prior/caps)."""
    adj_raw, rationale = _propose(prior, pulse, client)
    predicted, caps = _capped(prior, adj_raw)
    prev = prior["prev_close"]
    agreement = _evidence_agreement(pulse)
    confidence = calibrated_confidence(
        aggregate(rows_b, window=settings.GATE_EDGE_WINDOW), agreement)
    note = (f"[prior {prior['center']} ± {prior['sigma_pct'] * 100:.2f}%σ ({prior['centered_on']}); "
            f"adj {adj_raw:+.2f}σ; caps: {', '.join(caps) if caps else 'none'}]")
    return {
        "predicted_close": predicted,
        "direction": "up" if predicted > prev else "down",
        "confidence": confidence,
        "adjustment_sigma": round(adj_raw, 3),
        "caps_applied": caps,
        "rationale": (rationale + " " + note).strip(),
    }
