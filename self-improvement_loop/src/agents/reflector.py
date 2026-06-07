"""Reflector — turns a scored result into a learning.

After a prediction is scored, the reflector (Opus) writes a short post-mortem and proposes a single
concrete, bounded note to append to STRATEGY.md. It NEVER rewrites or deletes existing strategy
history — it only appends one dated insight. Offline it produces a deterministic factual summary.
"""

from __future__ import annotations

import json

from ..config import settings
from ..utils import extract_json

_SYSTEM = (
    "You are a trading-research post-mortem analyst. Given a scored AMZN close prediction and how "
    "each strategy did, write a concise, honest post-mortem and propose ONE concrete, testable "
    "adjustment to the desk's living strategy. Be specific (e.g. 'down-weight momentum on high-RSI "
    "gap-up days'), not generic. Do not overfit to a single day.\n"
    "Respond with ONLY JSON:\n"
    '{"postmortem_md": "<markdown, 4-8 lines>", "strategy_note": "<one sentence insight to append>"}'
)


def _context(row: dict, scorecards: dict) -> str:
    return (
        f"Scored prediction for {row.get('date')} (symbol {settings.SYMBOL}):\n"
        f"{json.dumps({k: row.get(k) for k in ['prior_close','predicted_close','actual_close','ape','pass','directional_hit','baseline_ape','beats_baseline','confidence','winning_strategy']}, indent=2)}\n\n"
        f"Per-analyst predictions:\n{json.dumps(row.get('analyst_predictions', {}), indent=2)}\n\n"
        f"Meta-judge weights used:\n{json.dumps(row.get('weights', {}), indent=2)}\n\n"
        f"Updated scorecards:\n{json.dumps(scorecards, indent=2)}"
    )


def _offline(row: dict) -> dict:
    verdict = "PASS" if row.get("pass") else "FAIL"
    beat = "beat" if row.get("beats_baseline") else "did not beat"
    ape_pct = (row.get("ape") or 0) * 100
    md = (
        f"### {row.get('date')} — {verdict}\n"
        f"- Predicted **{row.get('predicted_close')}** vs actual **{row.get('actual_close')}** "
        f"(APE {ape_pct:.2f}%).\n"
        f"- Directional hit: {row.get('directional_hit')}; {beat} the random-walk baseline.\n"
        f"- Closest analyst: **{row.get('winning_strategy')}**.\n"
    )
    return {
        "postmortem_md": md,
        "strategy_note": (
            f"{row.get('date')}: closest strategy was {row.get('winning_strategy')}; "
            f"result {verdict}, {beat} baseline."
        ),
    }


def reflect(row: dict, scorecards: dict, client=None) -> dict:
    """Return {'postmortem_md': str, 'strategy_note': str}."""
    if client is None:
        return _offline(row)
    try:
        resp = client.messages.create(
            model=settings.REFLECTOR_MODEL,
            max_tokens=settings.REFLECTOR_MAX_TOKENS,
            system=_SYSTEM,
            messages=[{"role": "user", "content": _context(row, scorecards)}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        out = extract_json(text)
        if "postmortem_md" not in out:
            return _offline(row)
        out.setdefault("strategy_note", "")
        return out
    except Exception:
        return _offline(row)
