"""Technical feature engineering from daily OHLCV history + a live quote.

Produces a compact, model-friendly dict of numeric features plus a short recent-OHLC text table.
All features are computed from completed sessions; the live quote supplies pre-market gap signal.
"""

from __future__ import annotations

import pandas as pd


def _rsi(close: pd.Series, period: int = 14) -> float | None:
    if len(close) < period + 1:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return round(float(val), 2) if pd.notna(val) else None


def _ret(close: pd.Series, n: int) -> float | None:
    if len(close) < n + 1:
        return None
    return round(float(close.iloc[-1] / close.iloc[-1 - n] - 1.0), 5)


def _sma(close: pd.Series, n: int) -> float | None:
    if len(close) < n:
        return None
    return round(float(close.tail(n).mean()), 2)


def build_features(history: pd.DataFrame, quote: dict | None = None) -> dict:
    """Return a dict of technical features. `history` is ascending daily OHLCV."""
    close = history["close"].astype(float)
    prev_close = float(close.iloc[-1])
    vol20 = close.pct_change().tail(20).std()
    sma20 = _sma(close, 20)
    sma50 = _sma(close, 50)
    hi_252 = float(close.tail(252).max()) if len(close) >= 5 else prev_close
    lo_252 = float(close.tail(252).min()) if len(close) >= 5 else prev_close

    premarket = quote.get("last") if quote else None
    gap = None
    if premarket and prev_close:
        gap = round(premarket / prev_close - 1.0, 5)

    recent = history.tail(7)[["open", "high", "low", "close", "volume"]]
    recent_table = recent.reset_index().to_string(index=False)

    return {
        "prev_close": round(prev_close, 2),
        "premarket_last": round(float(premarket), 2) if premarket else None,
        "premarket_gap_pct": gap,
        "ret_1d": _ret(close, 1),
        "ret_5d": _ret(close, 5),
        "ret_20d": _ret(close, 20),
        "realized_vol_20d": round(float(vol20), 5) if pd.notna(vol20) else None,
        "rsi_14": _rsi(close),
        "sma_5": _sma(close, 5),
        "sma_20": sma20,
        "sma_50": sma50,
        "dist_to_sma20_pct": round(prev_close / sma20 - 1.0, 5) if sma20 else None,
        "dist_to_sma50_pct": round(prev_close / sma50 - 1.0, 5) if sma50 else None,
        "high_252": round(hi_252, 2),
        "low_252": round(lo_252, 2),
        "pct_from_252_high": round(prev_close / hi_252 - 1.0, 5) if hi_252 else None,
        "recent_ohlc": recent_table,
        "quote_source": (quote or {}).get("source"),
    }
