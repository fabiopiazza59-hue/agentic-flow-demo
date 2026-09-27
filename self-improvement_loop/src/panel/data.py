"""Batched market data for the panel — two Yahoo calls cover every name.

Alpha Vantage's free tier (25 requests/day) cannot serve 30 names, so the panel is keyless:
one daily-bar download and one 1-minute pre/post-market download for all symbols. A name whose
data is missing or stale is dropped for that session, never guessed.
"""

from __future__ import annotations

import time
from datetime import date, datetime

import pandas as pd

from ..utils import to_date


def _download(symbols: list[str], attempts: int = 3, **kwargs) -> pd.DataFrame | None:
    import yfinance as yf

    for i in range(attempts):
        try:
            df = yf.download(symbols, group_by="ticker", auto_adjust=False, progress=False,
                             threads=True, **kwargs)
            if df is not None and not df.empty:
                return df
        except Exception:  # noqa: BLE001 - rate limits and transient network errors
            pass
        time.sleep(2 * (i + 1))
    return None


def _per_symbol(df: pd.DataFrame, symbol: str) -> pd.DataFrame | None:
    try:
        sub = df[symbol] if isinstance(df.columns, pd.MultiIndex) else df
    except KeyError:
        return None
    sub = sub.rename(columns=str.lower)
    if "close" not in sub:
        return None
    sub = sub[sub["close"].notna()]
    return sub if len(sub) else None


def daily_history(symbols: list[str], through: str | date,
                  period: str = "1y") -> dict[str, pd.DataFrame]:
    """Daily OHLCV per symbol, capped at `through`; only symbols that have `through`'s bar."""
    day = to_date(through)
    df = _download(symbols, period=period, interval="1d")
    out: dict[str, pd.DataFrame] = {}
    if df is None:
        return out
    for s in symbols:
        sub = _per_symbol(df, s)
        if sub is None:
            continue
        sub = sub.copy()
        sub.index = [to_date(ts) for ts in sub.index]
        sub = sub[[i <= day for i in sub.index]]
        if len(sub) and sub.index[-1] == day:
            out[s] = sub[["open", "high", "low", "close", "volume"]]
    return out


def extended_last(symbols: list[str], since: datetime, until: datetime) -> dict[str, dict]:
    """Latest completed 1-minute bar per symbol in [since, until], pre/post-market included."""
    df = _download(symbols, period="5d", interval="1m", prepost=True)
    out: dict[str, dict] = {}
    if df is None:
        return out
    idx = df.index.tz_convert("UTC") if df.index.tz is not None else df.index.tz_localize("UTC")
    window = (idx >= since) & (idx + pd.Timedelta(minutes=1) <= until)
    for s in symbols:
        sub = _per_symbol(df[window], s)
        if sub is None:
            continue
        ts = sub.index[-1]
        ts = ts.tz_convert("UTC") if ts.tzinfo is not None else ts.tz_localize("UTC")
        out[s] = {"price": float(sub["close"].iloc[-1]), "time": ts.to_pydatetime().isoformat(),
                  "source": "yfinance_ext", "live": True}
    return out
