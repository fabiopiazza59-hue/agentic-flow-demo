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


def log(msg: str) -> None:
    print(f"[panel.data] {msg}", flush=True)


def _download(symbols: list[str], attempts: int = 3, **kwargs) -> pd.DataFrame | None:
    import yfinance as yf

    for i in range(attempts):
        try:
            df = yf.download(symbols, group_by="ticker", auto_adjust=False, progress=False,
                             threads=True, **kwargs)
            if df is not None and not df.empty:
                return df
            log(f"batch download attempt {i + 1} returned no rows ({kwargs.get('interval')}).")
        except Exception as e:  # noqa: BLE001 - rate limits and transient network errors
            log(f"batch download attempt {i + 1} failed: {type(e).__name__}: {e}")
        time.sleep(2 * (i + 1))
    return None


def _single(symbol: str, **kwargs) -> pd.DataFrame | None:
    """One ticker via Ticker.history — the path that keeps working when batches are refused."""
    import yfinance as yf

    for i in range(2):
        try:
            df = yf.Ticker(symbol).history(auto_adjust=False, **kwargs)
            if df is not None and not df.empty:
                return df
        except Exception as e:  # noqa: BLE001
            log(f"{symbol}: single download failed: {type(e).__name__}: {e}")
        time.sleep(1 + i)
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

    def _take(s: str, sub: pd.DataFrame | None) -> None:
        if sub is None:
            return
        sub = sub.copy()
        sub.index = [to_date(ts) for ts in sub.index]
        sub = sub[[i <= day for i in sub.index]]
        if len(sub) and sub.index[-1] == day:
            out[s] = sub[["open", "high", "low", "close", "volume"]]

    if df is not None:
        for s in symbols:
            _take(s, _per_symbol(df, s))
    missing = [s for s in symbols if s not in out]
    if missing:
        log(f"daily bars through {day} missing for {len(missing)}/{len(symbols)} names after the "
            f"batch download; fetching them one by one.")
        for s in missing:
            _take(s, _per_symbol(_single(s, period=period, interval="1d"), s))
        still = [s for s in symbols if s not in out]
        if still:
            log(f"still missing after single fetches: {', '.join(still)}")
    return out


def extended_last(symbols: list[str], since: datetime, until: datetime) -> dict[str, dict]:
    """Latest completed 1-minute bar per symbol in [since, until], pre/post-market included."""
    out: dict[str, dict] = {}

    def _take(s: str, frame: pd.DataFrame | None) -> None:
        if frame is None or frame.empty:
            return
        idx = (frame.index.tz_convert("UTC") if frame.index.tz is not None
               else frame.index.tz_localize("UTC"))
        sub = _per_symbol(frame[(idx >= since) & (idx + pd.Timedelta(minutes=1) <= until)], s)
        if sub is None:
            return
        ts = sub.index[-1]
        ts = ts.tz_convert("UTC") if ts.tzinfo is not None else ts.tz_localize("UTC")
        out[s] = {"price": float(sub["close"].iloc[-1]), "time": ts.to_pydatetime().isoformat(),
                  "source": "yfinance_ext", "live": True}

    df = _download(symbols, period="5d", interval="1m", prepost=True)
    if df is not None:
        for s in symbols:
            _take(s, df)
    missing = [s for s in symbols if s not in out]
    if df is None and missing:
        log(f"batch minute bars unavailable; fetching {len(missing)} names one by one.")
        for s in missing:
            _take(s, _single(s, period="5d", interval="1m", prepost=True))
    return out
