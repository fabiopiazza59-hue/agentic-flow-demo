"""Meta-judge (v2) — a bounded adjuster on top of a code-computed blend.

v1 let the judge write the final number. Over 60 paired days that cost 0.14% MAPE against a plain
mean of its own analysts (review F1, spec/improvements-from-2609.05663.md), and its "weights"
never moved (F4). v2 applies the pattern arm B already uses — evidence may move a prior only
within a hard-capped band, and the cap lives in code:

  blend      = the lab's champion aggregation rule over the analysts (evals/lab.py; equal mean
               until a challenger proves better)
  adjustment = the judge's proposal in units of daily σ, clamped to ±A_JUDGE_MAX_SIGMA in code
  forecast   = blend · (1 + adjustment · σ)

The judge's job shrinks to "size a deviation", which is the only part it can be scored on: the
lab measures APE(blend) − APE(blend + adjustment) every day and switches the judge off once that
is shown to hurt. Offline (no client) the adjustment is 0.
"""

from __future__ import annotations

import json

from ..config import settings
from ..utils import extract_json, safe_float

_SYSTEM = (
    "You are the head of a quantitative research desk. Several analysts have each predicted the "
    "target session's AMZN regular-session CLOSE, and code has already combined them into a base "
    "forecast using the desk's measured best aggregation rule. Your ONLY job is to decide whether "
    "anything the blend cannot see justifies moving off it, in units of the daily volatility σ: "
    "positive = above the base forecast, negative = below. Weak, stale or conflicting evidence "
    "deserves 0 — most days should be 0. The system clamps your answer to ±{cap}σ.\n"
    "Respond with ONLY JSON:\n"
    '{{"adjustment_sigma": <number>, "rationale": "<2-4 sentences>"}}'
)


def clamp_adjustment(raw: float) -> float:
    cap = settings.A_JUDGE_MAX_SIGMA
    return max(-cap, min(cap, float(raw)))


def _user_prompt(analyst_predictions: dict, scorecards: dict, strategy_md: str,
                 recent_learnings: list[str], features: dict, whats_not_working: str = "") -> str:
    wnw = (f"KNOWN FAILURE PATTERNS — actively avoid repeating these:\n{whats_not_working[:3000]}\n\n"
           if whats_not_working.strip() else "")
    return (
        f"Symbol: {settings.SYMBOL}\n"
        f"Prior close: {features.get('prev_close')}\n"
        f"Pre-market gap %: {features.get('premarket_gap_pct')}\n\n"
        f"{wnw}"
        f"Analyst predictions (JSON):\n{json.dumps(analyst_predictions, indent=2)}\n\n"
        f"Per-strategy scorecards (JSON):\n{json.dumps(scorecards, indent=2)}\n\n"
        f"Living strategy notes (STRATEGY.md, newest last):\n{strategy_md[-4000:]}\n\n"
        f"Recent post-mortems:\n" + "\n---\n".join(recent_learnings[-settings.LEARNINGS_CONTEXT_N:])
    )


def adjust(base_close: float, rule: str, sigma: float, analyst_predictions: dict,
           scorecards: dict, strategy_md: str, recent_learnings: list[str], features: dict,
           client=None, whats_not_working: str = "") -> tuple[float, str]:
    """Return (raw adjustment in σ units, rationale). Offline or on failure: (0, note)."""
    if client is None or not analyst_predictions:
        return 0.0, "[offline] no judge call — the champion blend ships unadjusted."
    user = (
        f"Target session: {features.get('target_session')}\n"
        f"Base forecast (code, rule '{rule}'): {base_close}\n"
        f"Daily σ (EWMA of log returns): {sigma * 100:.2f}%\n"
        f"Latest pre-open price (anchor): {features.get('premarket_last') or 'none — no extended-hours trade yet'}\n\n"
        + _user_prompt(analyst_predictions, scorecards, strategy_md, recent_learnings, features,
                       whats_not_working)
    )
    try:
        resp = client.messages.create(
            model=settings.JUDGE_MODEL,
            max_tokens=settings.JUDGE_MAX_TOKENS,
            system=_SYSTEM.format(cap=settings.A_JUDGE_MAX_SIGMA),
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        raw = extract_json(text)
        adj = safe_float(raw.get("adjustment_sigma"), 0.0) or 0.0
        return adj, str(raw.get("rationale", ""))[:1000]
    except Exception:  # noqa: BLE001 - any failure abstains to the blend
        return 0.0, "Judge call failed — the champion blend ships unadjusted."
