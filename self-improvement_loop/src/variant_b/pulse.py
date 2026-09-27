"""Evidence pulse for variant B — predict-raven's "Market Pulse" adapted to one stock.

One web-search-enabled call gathers today's evidence as structured items; the decider consumes
them and the raw pulse is archived to learnings_b/pulse/YYYY-MM-DD.md so every decision is
auditable (raven's runtime-artifacts idea). Offline (no client) the pulse is empty — the prior
then stands unadjusted.
"""

from __future__ import annotations

import json

from ..agents.analysts import count_web_results
from ..config import settings
from ..utils import extract_json

_SYSTEM = (
    "You are an evidence-gathering agent for an AMZN close-prediction desk. Use AT MOST 3 web "
    "searches to collect decision-relevant evidence for the target session: AMZN news and "
    "catalysts, index futures, notable analyst actions, macro events. Report FINDINGS ONLY — do "
    "not predict a price. Your FINAL message must be ONLY JSON:\n"
    '{"items": [{"finding": "<one sentence>", "direction": "up"|"down"|"neutral", '
    '"strength": <0..1>}, ...], "summary": "<2-3 sentences>"}\n'
    "3-8 items; direction is the finding's implied pressure on the target session's AMZN close; "
    "strength is how material and well-confirmed it is."
)


def _empty(reason: str) -> dict:
    return {"items": [], "summary": f"[offline] no evidence gathered ({reason})."}


def _normalize(raw: dict) -> dict:
    items = []
    for it in raw.get("items") or []:
        if not isinstance(it, dict) or not it.get("finding"):
            continue
        direction = it.get("direction")
        if direction not in ("up", "down", "neutral"):
            direction = "neutral"
        try:
            strength = max(0.0, min(1.0, float(it.get("strength", 0.5))))
        except (TypeError, ValueError):
            strength = 0.5
        items.append({"finding": str(it["finding"])[:300], "direction": direction,
                      "strength": round(strength, 2)})
    return {"items": items[:8], "summary": str(raw.get("summary", ""))[:600]}


def gather_pulse(features: dict, client=None) -> dict:
    """Return {"items": [...], "summary": str}. Never raises."""
    if client is None:
        return _empty("no client")
    try:
        resp = client.messages.create(
            model=settings.B_PULSE_MODEL,
            max_tokens=settings.B_PULSE_MAX_TOKENS,
            system=_SYSTEM,
            messages=[{
                "role": "user",
                "content": (
                    f"Symbol: {settings.SYMBOL}. Target session: "
                    f"{features.get('target_session') or 'the next regular session'} (forecast "
                    f"made at {features.get('forecast_made_at') or 'now'} UTC). Prior close "
                    f"{features.get('prev_close')}; latest pre-open price "
                    f"{features.get('premarket_last') or 'none yet'} (gap "
                    f"{features.get('premarket_gap_pct')}). Gather evidence relevant to the "
                    f"target session's close."
                ),
            }],
            tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}],
        )
        blocks = [b.text for b in resp.content
                  if getattr(b, "type", None) == "text" and b.text.strip()]
        n_results = count_web_results(resp)
        for text in reversed(blocks):
            try:
                return {**_normalize(extract_json(text)), "web_results": n_results}
            except ValueError:
                continue
        return _empty("unparseable pulse response")
    except Exception:
        return _empty("pulse call failed")


def write_pulse_artifact(d: str, pulse: dict, prior: dict) -> None:
    """Archive the day's pulse + prior to learnings_b/pulse/<date>.md for auditability."""
    pulse_dir = settings.LEARNINGS_B_DIR / "pulse"
    pulse_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# Pulse — {d}", "",
             f"Prior: center {prior['center']} (±{prior['sigma_pct'] * 100:.2f}% σ, "
             f"centered on {prior['centered_on']}); p10/p50/p90 = "
             f"{prior['p10']}/{prior['p50']}/{prior['p90']}.", "",
             pulse.get("summary", ""), ""]
    for it in pulse.get("items", []):
        lines.append(f"- [{it['direction']} · {it['strength']:.2f}] {it['finding']}")
    (pulse_dir / f"{d}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
