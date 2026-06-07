"""Meta-judge — synthesizes analyst predictions into one final forecast.

The judge sees every analyst's prediction, the per-strategy accuracy scorecards (weight hints),
the living STRATEGY.md, and recent post-mortems. It outputs the final prediction plus the weights
it assigned to each strategy. Up-weighting historically accurate strategies is the loop's core
self-improvement mechanism.

Offline (no client) it computes a transparent weighted blend using scorecard weight hints, so the
full pipeline is testable without an API key.
"""

from __future__ import annotations

import json

from ..config import settings
from ..utils import extract_json, safe_float

_SYSTEM = (
    "You are the head of a quantitative research desk. Several analysts have each predicted today's "
    "AMZN regular-session CLOSE. You also have each strategy's historical accuracy scorecard, a "
    "living strategy note, and recent post-mortems. Produce ONE final close prediction.\n"
    "Rules:\n"
    "- Weight strategies by their demonstrated accuracy (lower MAPE / higher hit_rate = more weight).\n"
    "- Be skeptical of overconfident outliers; the prior close is a strong anchor.\n"
    "- Your weights should reflect how much each analyst influenced your final number.\n"
    "Respond with ONLY JSON:\n"
    '{"predicted_close": <number>, "direction": "up"|"down", "confidence": <0..1>, '
    '"weights": {"<strategy>": <0..1>, ...}, "rationale": "<2-4 sentences>"}'
)


def _offline_blend(analyst_predictions: dict, scorecards: dict, prev_close: float) -> dict:
    """Weighted average using scorecard weight_hints (fallback to equal weights)."""
    names = list(analyst_predictions.keys())
    hints = {n: float((scorecards.get(n) or {}).get("weight_hint", 0.2)) for n in names}
    total = sum(hints.values()) or float(len(names) or 1)
    weights = {n: round(hints.get(n, 0.0) / total, 4) for n in names}
    if not names:
        return {
            "predicted_close": round(prev_close, 2),
            "direction": "down",
            "confidence": 0.3,
            "weights": {},
            "rationale": "No analyst predictions available; defaulting to prior close.",
        }
    blended = sum(weights[n] * float(analyst_predictions[n]["predicted_close"]) for n in names)
    blended = round(blended, 2)
    avg_conf = sum(float(analyst_predictions[n]["confidence"]) for n in names) / len(names)
    return {
        "predicted_close": blended,
        "direction": "up" if blended > prev_close else "down",
        "confidence": round(avg_conf, 3),
        "weights": weights,
        "rationale": "[offline] scorecard-weighted blend of analyst predictions.",
    }


def _user_prompt(analyst_predictions: dict, scorecards: dict, strategy_md: str,
                 recent_learnings: list[str], features: dict) -> str:
    return (
        f"Symbol: {settings.SYMBOL}\n"
        f"Prior close: {features.get('prev_close')}\n"
        f"Pre-market gap %: {features.get('premarket_gap_pct')}\n\n"
        f"Analyst predictions (JSON):\n{json.dumps(analyst_predictions, indent=2)}\n\n"
        f"Per-strategy scorecards (JSON):\n{json.dumps(scorecards, indent=2)}\n\n"
        f"Living strategy notes (STRATEGY.md):\n{strategy_md[:4000]}\n\n"
        f"Recent post-mortems:\n" + "\n---\n".join(recent_learnings[-settings.LEARNINGS_CONTEXT_N:])
    )


def _normalize(raw: dict, analyst_predictions: dict, scorecards: dict, prev_close: float) -> dict:
    pred = safe_float(raw.get("predicted_close"))
    if pred is None or pred <= 0:
        return _offline_blend(analyst_predictions, scorecards, prev_close)
    direction = raw.get("direction")
    if direction not in ("up", "down"):
        direction = "up" if pred > prev_close else "down"
    conf = safe_float(raw.get("confidence"), 0.5) or 0.5
    weights = raw.get("weights") if isinstance(raw.get("weights"), dict) else {}
    weights = {k: round(float(v), 4) for k, v in weights.items() if safe_float(v) is not None}
    return {
        "predicted_close": round(pred, 2),
        "direction": direction,
        "confidence": round(max(0.0, min(1.0, conf)), 3),
        "weights": weights,
        "rationale": str(raw.get("rationale", ""))[:1000],
    }


def synthesize(analyst_predictions: dict, scorecards: dict, strategy_md: str,
               recent_learnings: list[str], features: dict, client=None) -> dict:
    prev_close = float(features.get("prev_close") or 0.0)
    if client is None or not analyst_predictions:
        return _offline_blend(analyst_predictions, scorecards, prev_close)
    try:
        resp = client.messages.create(
            model=settings.JUDGE_MODEL,
            max_tokens=settings.JUDGE_MAX_TOKENS,
            system=_SYSTEM,
            messages=[{
                "role": "user",
                "content": _user_prompt(analyst_predictions, scorecards, strategy_md,
                                        recent_learnings, features),
            }],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        return _normalize(extract_json(text), analyst_predictions, scorecards, prev_close)
    except Exception:
        return _offline_blend(analyst_predictions, scorecards, prev_close)
