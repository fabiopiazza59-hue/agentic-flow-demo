"""Reflector — turns a scored result into a learning, with explicit focus on failures.

Two jobs:
  reflect(row, ...)            -> per-day post-mortem + strategy note + failure reflection.
                                  On a FAIL it does a real root-cause analysis (what misled the
                                  blend, direction vs magnitude, the one change to try).
  diagnose_failures(rows, ...) -> a rolling "what's not working" synthesis across ALL scored days,
                                  surfacing recurring failure patterns (not one-offs).

Both have deterministic offline fallbacks so the pipeline runs without an API key.
"""

from __future__ import annotations

import json

from ..config import settings
from ..utils import extract_json

_BASE_RULES = (
    "You are a trading-research post-mortem analyst for a daily AMZN close-prediction desk. "
    "Be specific and brutally honest; never generic. Do not overfit to a single day — frame every "
    "lesson as a testable hypothesis."
)

_PASS_TASK = (
    "This prediction PASSED (within ±1%). Note briefly what worked and one way it could still be "
    "more robust. Keep failure_reflection short."
)

_FAIL_TASK = (
    "This prediction FAILED (error > 1%). Do a real failure analysis: WHAT specifically went wrong "
    "(which analyst(s) dragged the blend off, which signal was over- or under-weighted, was the miss "
    "directional or magnitude), the most likely ROOT CAUSE, and the ONE concrete change to try next "
    "time. This is the lesson the desk rereads before its next shot — make it actionable, not vague."
)

_OUTPUT = (
    "Respond with ONLY JSON:\n"
    '{"postmortem_md": "<markdown, 4-8 lines>", '
    '"strategy_note": "<one-sentence testable insight to append to STRATEGY.md>", '
    '"failure_reflection": "<markdown: what went wrong, root cause, the one change to try — '
    'rich and concrete on FAIL, one line on PASS>"}'
)

_DIAGNOSE_SYSTEM = (
    "You are the head of research running a standing review of WHAT IS NOT WORKING in a daily AMZN "
    "close-prediction system. You get every scored prediction so far plus per-strategy scorecards. "
    "Find RECURRING failure patterns, not one-offs: market conditions where the model keeps missing, "
    "which strategies are unreliable and when, whether errors are directional or magnitude, and "
    "whether confidence is miscalibrated (high confidence on misses). Be blunt and specific.\n"
    "Respond with ONLY markdown (no JSON), at most ~25 lines, with exactly these sections:\n"
    "## What keeps going wrong\n## Unreliable under these conditions\n## Fixes to try next"
)


# --------------------------------------------------------------------------- per-day reflection

def _context(row: dict, scorecards: dict) -> str:
    keys = ["prior_close", "predicted_close", "actual_close", "ape", "pass", "directional_hit",
            "baseline_ape", "beats_baseline", "confidence", "winning_strategy"]
    return (
        f"Scored prediction for {row.get('date')} (symbol {settings.SYMBOL}):\n"
        f"{json.dumps({k: row.get(k) for k in keys}, indent=2)}\n\n"
        f"Per-analyst predictions:\n{json.dumps(row.get('analyst_predictions', {}), indent=2)}\n\n"
        f"Meta-judge weights used:\n{json.dumps(row.get('weights', {}), indent=2)}\n\n"
        f"Updated scorecards:\n{json.dumps(scorecards, indent=2)}"
    )


def _offline(row: dict) -> dict:
    passed = bool(row.get("pass"))
    verdict = "PASS" if passed else "FAIL"
    beat = "beat" if row.get("beats_baseline") else "did not beat"
    ape_pct = (row.get("ape") or 0) * 100
    preds = row.get("analyst_predictions", {})
    actual = row.get("actual_close")
    worst = None
    if preds and actual:
        worst = max(preds, key=lambda n: abs(float(preds[n].get("predicted_close", actual)) - actual))
    md = (
        f"### {row.get('date')} — {verdict}\n"
        f"- Predicted **{row.get('predicted_close')}** vs actual **{row.get('actual_close')}** "
        f"(APE {ape_pct:.2f}%).\n"
        f"- Directional hit: {row.get('directional_hit')}; {beat} the random-walk baseline.\n"
        f"- Closest analyst: **{row.get('winning_strategy')}**; worst: **{worst}**.\n"
    )
    if passed:
        failure_reflection = "PASS — no failure to analyze."
    else:
        failure_reflection = (
            f"- **What went wrong:** {'wrong direction' if not row.get('directional_hit') else 'right direction, magnitude off'}; "
            f"final blend {row.get('predicted_close')} missed by {ape_pct:.2f}%.\n"
            f"- **Likely culprit:** analyst `{worst}` was furthest from actual and pulled the blend.\n"
            f"- **Try next:** reduce weight on `{worst}` under today's conditions and lean on `{row.get('winning_strategy')}`."
        )
    return {
        "postmortem_md": md,
        "strategy_note": (
            f"{row.get('date')}: {verdict}, {beat} baseline; closest {row.get('winning_strategy')}, "
            f"worst {worst}."
        ),
        "failure_reflection": failure_reflection,
    }


def reflect(row: dict, scorecards: dict, client=None) -> dict:
    """Return {'postmortem_md', 'strategy_note', 'failure_reflection'} for a scored row."""
    if client is None:
        return _offline(row)
    failed = not row.get("pass")
    system = "\n".join([_BASE_RULES, _FAIL_TASK if failed else _PASS_TASK, _OUTPUT])
    try:
        resp = client.messages.create(
            model=settings.REFLECTOR_MODEL,
            max_tokens=settings.REFLECTOR_MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": _context(row, scorecards)}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        out = extract_json(text)
        if "postmortem_md" not in out:
            return _offline(row)
        out.setdefault("strategy_note", "")
        out.setdefault("failure_reflection", "")
        return out
    except Exception:
        return _offline(row)


# ----------------------------------------------------------------- rolling failure self-diagnosis

def _failure_stats(scored: list[dict]) -> dict:
    fails = [r for r in scored if r.get("pass") is False]
    dir_misses = [r for r in fails if r.get("directional_hit") is False]
    mag_misses = [r for r in fails if r.get("directional_hit") is True]
    overconfident = [r for r in fails if (r.get("confidence") or 0) >= 0.6]
    mean_fail_ape = (sum(r.get("ape", 0) for r in fails) / len(fails)) if fails else 0.0
    return {
        "n_scored": len(scored),
        "n_fail": len(fails),
        "dir_misses": len(dir_misses),
        "mag_misses": len(mag_misses),
        "overconfident_misses": len(overconfident),
        "mean_fail_ape_pct": round(mean_fail_ape * 100, 2),
    }


def _diagnose_context(scored: list[dict], scorecards: dict) -> str:
    compact = [
        {k: r.get(k) for k in ["date", "pass", "ape", "baseline_ape", "directional_hit",
                                "beats_baseline", "confidence", "winning_strategy", "weights"]}
        for r in scored
    ]
    return (
        f"Failure stats: {json.dumps(_failure_stats(scored))}\n\n"
        f"All scored predictions (newest last):\n{json.dumps(compact, indent=2)}\n\n"
        f"Per-strategy scorecards:\n{json.dumps(scorecards, indent=2)}"
    )


def _offline_diagnose(scored: list[dict], scorecards: dict) -> str:
    s = _failure_stats(scored)
    worst_strategy = None
    if scorecards:
        worst_strategy = max(scorecards, key=lambda n: scorecards[n].get("mape", 0))
    return (
        "# What's Not Working — AMZN Close Predictor\n\n"
        "_Auto-generated each scoring run from the full scored history._\n\n"
        "## What keeps going wrong\n"
        f"- {s['n_fail']} of {s['n_scored']} scored predictions FAILED (>1% error); "
        f"mean error on misses {s['mean_fail_ape_pct']}%.\n"
        f"- Directional misses: {s['dir_misses']}; magnitude-only misses: {s['mag_misses']}.\n"
        f"- Overconfident misses (conf ≥ 0.6): {s['overconfident_misses']}.\n\n"
        "## Unreliable under these conditions\n"
        f"- Highest-MAPE strategy so far: `{worst_strategy}`.\n"
        "- (Offline summary — richer pattern analysis runs when an API key is set.)\n\n"
        "## Fixes to try next\n"
        f"- {'Tighten direction calls; ' if s['dir_misses'] >= s['mag_misses'] else 'Tighten magnitude/sizing; '}"
        f"down-weight `{worst_strategy}` until its MAPE improves.\n"
        f"- {'Lower confidence on uncertain setups (calibration).' if s['overconfident_misses'] else 'Maintain confidence calibration.'}\n"
    )


def diagnose_failures(scored: list[dict], scorecards: dict, client=None) -> str:
    """Return a markdown 'what's not working' synthesis across all scored predictions."""
    if not scored:
        return "# What's Not Working\n\n_No scored predictions yet._\n"
    if client is None:
        return _offline_diagnose(scored, scorecards)
    try:
        resp = client.messages.create(
            model=settings.REFLECTOR_MODEL,
            max_tokens=settings.REFLECTOR_MAX_TOKENS,
            system=_DIAGNOSE_SYSTEM,
            messages=[{"role": "user", "content": _diagnose_context(scored, scorecards)}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
        header = "# What's Not Working — AMZN Close Predictor\n\n_Auto-generated each scoring run._\n\n"
        return header + text if text else _offline_diagnose(scored, scorecards)
    except Exception:
        return _offline_diagnose(scored, scorecards)
