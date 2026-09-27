"""Panel analyst — one LLM call per batch of names, each name sized in σ off its anchor.

Same contract as arm B's decider and arm A's v2 judge: the free baseline (the latest pre-open
trade) is the forecast unless evidence the market has not priced in justifies a move, expressed
in the name's own daily σ and clamped in code. Batching keeps the cost to a few calls a day for
thirty names. Offline (no client) every adjustment is 0 — the panel is then the anchor baseline.
"""

from __future__ import annotations

import json

from ..agents.analysts import count_web_results
from ..config import settings
from ..utils import extract_json, safe_float

# The evolvable part (src/panel/evolve.py may replace it with a proven challenger) ...
SEED_STRATEGY = (
    "You are an equity analyst covering a panel of US large caps for the target session. Each "
    "name comes with a free baseline: its latest pre-open price (the anchor), which already "
    "reflects everything the market has priced in. For each name, decide whether evidence the "
    "anchor does NOT yet reflect justifies expecting the regular-session close away from it, in "
    "units of that name's daily volatility σ (positive = above the anchor). Most names on most "
    "days deserve 0; a scheduled catalyst during the session (earnings, guidance, a macro print, "
    "an investor day) is the typical reason not to. Search only for the most material catalysts."
)

# ... and the fixed contract, which no variant can change: tool budget, cap, output format.
CONTRACT = (
    "\n\nRules the system enforces: use at most {searches} web searches; every answer is "
    "clamped to ±{cap}σ.\n"
    "Your FINAL message must be ONLY JSON:\n"
    '{{"adjustments": {{"<TICKER>": {{"sigma": <number>, "reason": "<one short sentence>"}}}}}}\n'
    "Include every ticker you were given."
)


def system_prompt(strategy: str | None = None) -> str:
    return (strategy or SEED_STRATEGY) + CONTRACT.format(
        searches=settings.PANEL_WEB_SEARCHES, cap=settings.PANEL_MAX_ADJ_SIGMA)


def clamp(raw: float) -> float:
    cap = settings.PANEL_MAX_ADJ_SIGMA
    return max(-cap, min(cap, float(raw)))


def _table(names: dict[str, dict]) -> str:
    lines = ["ticker | prior close | anchor (latest pre-open) | gap % | daily σ % | 5d % | 20d %"]
    for s, f in names.items():
        gap = f.get("gap")
        lines.append(
            f"{s} | {f['prev_close']:.2f} | {f['anchor']:.2f}"
            f"{'' if f.get('anchor_live') else ' (no trade since close)'} | "
            f"{'—' if gap is None else f'{gap * 100:+.2f}'} | {f['sigma'] * 100:.2f} | "
            f"{f.get('ret_5d', 0) * 100:+.1f} | {f.get('ret_20d', 0) * 100:+.1f}")
    return "\n".join(lines)


def _call(client, target: str, made_at: str, names: dict[str, dict],
          strategy: str | None = None) -> tuple[dict, int]:
    resp = client.messages.create(
        model=settings.PANEL_MODEL,
        max_tokens=settings.PANEL_MAX_TOKENS,
        system=system_prompt(strategy),
        messages=[{"role": "user", "content": (
            f"Target session: {target} (forecast made at {made_at} UTC).\n\n{_table(names)}")}],
        tools=[{"type": "web_search_20250305", "name": "web_search",
                "max_uses": settings.PANEL_WEB_SEARCHES}],
    )
    blocks = [b.text for b in resp.content if getattr(b, "type", None) == "text" and b.text.strip()]
    for text in reversed(blocks):
        try:
            return extract_json(text), count_web_results(resp)
        except ValueError:
            continue
    return {}, count_web_results(resp)


def adjustments(names: dict[str, dict], target: str, made_at: str,
                client=None, strategy: str | None = None) -> dict[str, dict]:
    """{ticker: {"raw": float, "sigma": clamped float, "reason": str, "web_results": int}}."""
    out = {s: {"raw": 0.0, "sigma": 0.0, "reason": "[offline] anchor only", "web_results": 0}
           for s in names}
    if client is None:
        return out
    tickers = list(names)
    size = max(1, settings.PANEL_BATCH_SIZE)
    for i in range(0, len(tickers), size):
        batch = {s: names[s] for s in tickers[i:i + size]}
        try:
            parsed, n_web = _call(client, target, made_at, batch, strategy)
        except Exception:  # noqa: BLE001 - a failed batch abstains to the anchors
            parsed, n_web = {}, 0
            for s in batch:
                out[s]["reason"] = "LLM call failed — anchor only"
        adj = parsed.get("adjustments") if isinstance(parsed, dict) else None
        for s in batch:
            item = (adj or {}).get(s) or {}
            raw = safe_float(item.get("sigma"), 0.0) if isinstance(item, dict) else 0.0
            raw = raw or 0.0
            out[s] = {"raw": round(raw, 3), "sigma": round(clamp(raw), 3),
                      "reason": str(item.get("reason", "") if isinstance(item, dict) else "")[:240]
                      or out[s]["reason"], "web_results": n_web}
    return out


def prompt_preview(names: dict[str, dict]) -> str:
    """The table the model sees (for tests and audits)."""
    return json.dumps({"system": system_prompt()[:80], "table": _table(names)})
