"""Strategy analysts — the creative core of the ensemble.

Five distinct lenses each produce an independent prediction of today's AMZN close as strict JSON:
  - technical   : moving averages, RSI, support/resistance
  - momentum    : trend continuation from recent returns
  - contrarian  : mean-reversion / fade-the-move
  - news        : catalysts via the Anthropic server-side web_search tool (earnings, macro, sentiment)
  - macro       : index/sector correlation, rates, broad-market regime

Each analyst returns: {predicted_close, direction, confidence, rationale}.
A failing analyst is dropped (None) so the meta-judge proceeds with the survivors.
When no ANTHROPIC_API_KEY is set (dry-run/tests), a deterministic feature-driven stub is used.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..config import settings
from ..utils import extract_json, safe_float

# strategy name -> (description, persona/system framing, uses_web_search)
ANALYSTS: dict[str, dict] = {
    "technical": {
        "uses_web": False,
        "system": (
            "You are a disciplined technical analyst. Predict today's AMZN regular-session CLOSE "
            "using moving averages, RSI, support/resistance, and the pre-market gap. Weight the most "
            "recent price action. Be precise and avoid round numbers unless justified."
        ),
    },
    "momentum": {
        "uses_web": False,
        "system": (
            "You are a momentum/trend-following analyst. Predict today's AMZN CLOSE assuming the "
            "prevailing short-term trend (1d/5d/20d returns) tends to persist intraday. Extrapolate "
            "carefully from recent returns and the pre-market gap."
        ),
    },
    "contrarian": {
        "uses_web": False,
        "system": (
            "You are a mean-reversion/contrarian analyst. Predict today's AMZN CLOSE assuming "
            "overextended moves (high RSI, large distance from SMA20/50, big gaps) tend to revert. "
            "Fade stretched conditions; respect the trend when conditions are neutral."
        ),
    },
    "news": {
        "uses_web": True,
        "max_tokens": 4096,  # web_search reasoning is token-heavy; avoid truncating the final JSON
        "system": (
            "You are a catalyst-driven equity analyst. Use AT MOST 2 web searches to find the latest "
            "AMZN news, earnings timing, analyst actions, and macro events for today, plus index "
            "futures sentiment. Then predict today's AMZN regular-session CLOSE. Anchor on the prior "
            "close and the pre-market gap; adjust for catalysts you find. Keep reasoning brief. Your "
            "FINAL message must be ONLY the JSON object — no prose after it."
        ),
    },
    "macro": {
        "uses_web": False,
        "system": (
            "You are a macro/cross-asset analyst. Predict today's AMZN CLOSE by reasoning about the "
            "broad-market regime (S&P 500 / Nasdaq direction, rates, risk sentiment) and AMZN's "
            "typical beta to tech. Use the pre-market gap as the day's opening risk signal."
        ),
    },
}

_JSON_INSTRUCTION = (
    "Respond with ONLY a JSON object, no prose, of the form:\n"
    '{"predicted_close": <number>, "direction": "up"|"down", '
    '"confidence": <0..1>, "rationale": "<one or two sentences>"}\n'
    'where direction is relative to the prior close.'
)


def _features_prompt(features: dict) -> str:
    payload = {k: v for k, v in features.items() if k != "recent_ohlc"}
    return (
        f"Symbol: {settings.SYMBOL}\n"
        f"Prior close: {features.get('prev_close')}\n"
        f"Technical features (JSON):\n{json.dumps(payload, indent=2)}\n\n"
        f"Recent daily OHLCV:\n{features.get('recent_ohlc')}\n\n"
        f"{_JSON_INSTRUCTION}"
    )


def _stub_prediction(name: str, features: dict) -> dict:
    """Deterministic offline prediction so the pipeline runs without an API key."""
    prev = float(features.get("prev_close") or 0.0)
    gap = float(features.get("premarket_gap_pct") or 0.0)
    ret5 = float(features.get("ret_5d") or 0.0)
    rsi = float(features.get("rsi_14") or 50.0)
    nudges = {
        "technical": gap * 0.5,
        "momentum": (ret5 / 5.0) + gap * 0.6,
        "contrarian": -(rsi - 50.0) / 5000.0 - gap * 0.3,
        "news": gap * 0.4,
        "macro": gap * 0.5 + ret5 * 0.05,
    }
    pred = round(prev * (1.0 + nudges.get(name, 0.0)), 2)
    return {
        "predicted_close": pred,
        "direction": "up" if pred > prev else "down",
        "confidence": 0.5,
        "rationale": f"[stub:{name}] prev_close adjusted by feature heuristic (offline mode).",
    }


def _normalize(raw: dict, prev_close: float) -> dict | None:
    pred = safe_float(raw.get("predicted_close"))
    if pred is None or pred <= 0:
        return None
    direction = raw.get("direction")
    if direction not in ("up", "down"):
        direction = "up" if pred > prev_close else "down"
    conf = safe_float(raw.get("confidence"), 0.5)
    conf = max(0.0, min(1.0, conf if conf is not None else 0.5))
    return {
        "predicted_close": round(pred, 2),
        "direction": direction,
        "confidence": round(conf, 3),
        "rationale": str(raw.get("rationale", ""))[:600],
    }


def _extract_from_blocks(resp) -> dict:
    """Parse the analyst JSON from a response that may contain many text blocks.

    With web_search the model emits interleaved reasoning + tool-result summaries as separate text
    blocks; the JSON answer is the final one. Try blocks newest-first, then the concatenation.
    """
    blocks = [b.text for b in resp.content if getattr(b, "type", None) == "text" and b.text.strip()]
    for text in reversed(blocks):
        try:
            return extract_json(text)
        except ValueError:
            continue
    return extract_json("".join(blocks))


def _run_one(client, name: str, spec: dict, features: dict) -> dict | None:
    prev = float(features.get("prev_close") or 0.0)
    if client is None:
        return _normalize(_stub_prediction(name, features), prev)
    try:
        kwargs = dict(
            model=settings.ANALYST_MODEL,
            max_tokens=spec.get("max_tokens", settings.ANALYST_MAX_TOKENS),
            system=spec["system"],
            messages=[{"role": "user", "content": _features_prompt(features)}],
        )
        if spec.get("uses_web"):
            kwargs["tools"] = [
                {"type": "web_search_20250305", "name": "web_search", "max_uses": 2}
            ]
        resp = client.messages.create(**kwargs)
        return _normalize(_extract_from_blocks(resp), prev)
    except Exception:
        return None


def run_analysts(features: dict, client=None) -> dict:
    """Run all analysts (in parallel when a client is provided). Returns {name: prediction}."""
    if client is None:
        return {
            name: pred
            for name, spec in ANALYSTS.items()
            if (pred := _run_one(None, name, spec, features)) is not None
        }

    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=len(ANALYSTS)) as pool:
        futures = {
            pool.submit(_run_one, client, name, spec, features): name
            for name, spec in ANALYSTS.items()
        }
        for fut in as_completed(futures):
            name = futures[fut]
            pred = fut.result()
            if pred is not None:
                results[name] = pred
    return results


def get_client():
    """Return an Anthropic client if a key is set, else None (dry-run/stub mode)."""
    if not settings.anthropic_api_key:
        return None
    import anthropic

    return anthropic.Anthropic()
