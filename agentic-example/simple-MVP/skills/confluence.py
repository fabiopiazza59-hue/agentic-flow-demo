"""
V2.1 Confluence Scoring Tool

Implements the scalp trading methodology with 6-factor confluence scoring:
- Volume (0-2 points)
- RSI (0-2 points)
- VWAP Position (0-2 points)
- Support Proximity (0-2 points)
- Market Context/SPY (0-1 point)
- Catalyst (0-1 point)

Total: 10 points max
"""

from langchain_core.tools import tool
from typing import Literal

# High-beta stocks requiring 50% position size and tighter stops
HIGH_BETA_STOCKS = ["TSLA", "PLTR", "SMCI", "AMD", "NVDA", "COIN", "MSTR", "RIVN"]


@tool
def calculate_confluence(
    ticker: str,
    price: float,
    high: float,
    low: float,
    rsi: int,
    volume_ratio: float,
    vwap_position: Literal["above", "below", "touching"],
    trend: Literal["up", "down", "sideways"],
    support: float,
    spy_change: float,
    has_catalyst: bool = False
) -> dict:
    """
    Calculate V2.1 confluence score and generate trade decision.

    Args:
        ticker: Stock symbol (e.g., "NVDA")
        price: Current stock price
        high: Day high price
        low: Day low price
        rsi: RSI value (0-100)
        volume_ratio: Volume relative to average (e.g., 1.2 = 120%)
        vwap_position: Price position relative to VWAP
        trend: Current trend direction
        support: Support level price
        spy_change: SPY % change for market context
        has_catalyst: Whether there's a positive catalyst (upgrade, earnings beat, etc.)

    Returns:
        Dictionary with score, verdict, trade levels, and reasoning
    """
    score = 0.0
    factors = []

    # 1. Volume (0-2 points)
    if volume_ratio >= 1.2:
        score += 2
        factors.append({"name": "Volume", "score": 2, "note": f"{volume_ratio}x - Strong"})
    elif volume_ratio >= 0.75:
        score += 1
        factors.append({"name": "Volume", "score": 1, "note": f"{volume_ratio}x - OK (V2.1)"})
    else:
        factors.append({"name": "Volume", "score": 0, "note": f"{volume_ratio}x - Too thin"})

    # 2. RSI (0-2 points)
    if rsi <= 30:
        score += 2
        factors.append({"name": "RSI", "score": 2, "note": f"{rsi} - Oversold"})
    elif rsi <= 40:
        score += 1
        factors.append({"name": "RSI", "score": 1, "note": f"{rsi} - Low zone"})
    elif rsi >= 68:
        factors.append({"name": "RSI", "score": 0, "note": f"{rsi} - Overbought (WAIT)"})
    else:
        score += 1
        factors.append({"name": "RSI", "score": 1, "note": f"{rsi} - Neutral"})

    # 3. VWAP Position (0-2 points)
    if vwap_position == "above" and trend == "up":
        score += 2
        factors.append({"name": "VWAP", "score": 2, "note": "Above + Uptrend"})
    elif vwap_position == "touching":
        score += 2
        factors.append({"name": "VWAP", "score": 2, "note": "At VWAP entry"})
    elif vwap_position == "below" and trend == "down":
        score += 1
        factors.append({"name": "VWAP", "score": 1, "note": "Bounce setup"})
    else:
        factors.append({"name": "VWAP", "score": 0, "note": "Misaligned"})

    # 4. Support Proximity (0-2 points)
    if support > 0 and price > 0:
        dist_to_support = ((price - support) / price) * 100
        if 0 <= dist_to_support <= 1:
            score += 2
            factors.append({"name": "Support", "score": 2, "note": f"At ${support:.2f}"})
        elif dist_to_support <= 2.5:
            score += 1
            factors.append({"name": "Support", "score": 1, "note": f"Near ${support:.2f}"})
        else:
            factors.append({"name": "Support", "score": 0, "note": f"{dist_to_support:.1f}% away"})
    else:
        factors.append({"name": "Support", "score": 0, "note": "Not defined"})

    # 5. Market Context - SPY (0-1 point)
    if spy_change > 0.5:
        score += 1
        factors.append({"name": "SPY", "score": 1, "note": f"+{spy_change}% Risk-on"})
    elif spy_change < -1:
        factors.append({"name": "SPY", "score": 0, "note": f"{spy_change}% Risk-off"})
    else:
        score += 0.5
        factors.append({"name": "SPY", "score": 0.5, "note": f"{spy_change}% Neutral"})

    # 6. Catalyst (0-1 point)
    if has_catalyst:
        score += 1
        factors.append({"name": "Catalyst", "score": 1, "note": "Positive catalyst"})

    # Cap score at 10
    final_score = min(score, 10)

    # Calculate ATR estimate and trade levels
    atr = (high - low) * 0.3
    is_high_beta = ticker.upper() in HIGH_BETA_STOCKS
    stop_mult = 0.5 if is_high_beta else 0.75

    entry = price
    stop = price - (atr * stop_mult)
    target1 = price + (atr * 1.5)
    target2 = price + (atr * 2.5)
    risk_reward = round((atr * 1.5) / (atr * stop_mult), 2)

    # Decision logic
    if rsi >= 68:
        verdict = "WAIT"
        confidence = "LOW"
        reasoning = "V2.1: RSI >= 68 - Wait for pullback"
    elif volume_ratio < 0.75:
        verdict = "NO_TRADE"
        confidence = "LOW"
        reasoning = "V2.1: Volume < 0.75x - Insufficient liquidity"
    elif final_score >= 6:
        verdict = "LONG_SCALP"
        confidence = "HIGH"
        reasoning = f"Strong confluence {final_score}/10"
    elif final_score >= 5 and has_catalyst:
        verdict = "LONG_SCALP"
        confidence = "MEDIUM"
        reasoning = "Score 5+ with catalyst support"
    elif final_score >= 5:
        verdict = "WAIT"
        confidence = "MEDIUM"
        reasoning = f"Borderline {final_score}/10 - Need catalyst"
    else:
        verdict = "NO_TRADE"
        confidence = "HIGH"
        reasoning = f"Low confluence {final_score}/10"

    return {
        "ticker": ticker.upper(),
        "score": final_score,
        "verdict": verdict,
        "confidence": confidence,
        "reasoning": reasoning,
        "factors": factors,
        "trade_setup": {
            "entry": round(entry, 2),
            "stop": round(stop, 2),
            "target1": round(target1, 2),
            "target2": round(target2, 2),
            "risk_reward": f"{risk_reward}:1",
            "position_size": "50%" if is_high_beta else "100%",
            "is_high_beta": is_high_beta
        }
    }
