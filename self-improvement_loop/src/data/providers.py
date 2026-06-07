"""Market-data providers with a layered fallback chain (robust in CI and locally).

Historical daily data & official past closes (used for features, backfill, validation):
    Alpha Vantage TIME_SERIES_DAILY (keyed, reliable from CI IPs)
      -> yfinance (keyless)
      -> Stooq CSV with proof-of-work solver (keyless, last resort)

Live / pre-market quote (today's signal):
    Finnhub /quote (keyed)
      -> Alpha Vantage GLOBAL_QUOTE (keyed)
      -> yfinance fast_info
      -> derived from the latest completed session in history

Public API:
    get_history(symbol, lookback)  -> DataFrame [open, high, low, close, volume] (date index, asc)
    get_actual_close(symbol, date) -> float official close for a past session (or None)
    get_quote(symbol)              -> Quote dict {last, prev_close, open, high, low, asof, source, symbol}
"""

from __future__ import annotations

import hashlib
import io
import re
from datetime import date

import pandas as pd
import requests

from ..config import settings
from ..utils import safe_float, to_date

HTTP_TIMEOUT = 25
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/csv,text/plain,*/*",
}


# ============================================================= Alpha Vantage (keyed, authoritative)

def alphavantage_history(symbol: str, api_key: str) -> pd.DataFrame:
    resp = requests.get(
        "https://www.alphavantage.co/query",
        params={"function": "TIME_SERIES_DAILY", "symbol": symbol,
                "outputsize": "compact", "apikey": api_key},
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json() or {}
    series = data.get("Time Series (Daily)")
    if not series:
        raise RuntimeError(f"Alpha Vantage history unavailable: {str(data)[:160]}")
    recs = {
        to_date(d): {
            "open": safe_float(v["1. open"]), "high": safe_float(v["2. high"]),
            "low": safe_float(v["3. low"]), "close": safe_float(v["4. close"]),
            "volume": safe_float(v["5. volume"]),
        }
        for d, v in series.items()
    }
    df = pd.DataFrame.from_dict(recs, orient="index").sort_index()
    return df[["open", "high", "low", "close", "volume"]]


def alphavantage_quote(symbol: str, api_key: str) -> dict:
    resp = requests.get(
        "https://www.alphavantage.co/query",
        params={"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": api_key},
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    q = (resp.json() or {}).get("Global Quote") or {}
    prev = safe_float(q.get("08. previous close"))
    if prev in (None, 0.0):
        raise RuntimeError(f"Alpha Vantage empty quote: {q}")
    return {
        "last": safe_float(q.get("05. price")), "prev_close": prev,
        "open": safe_float(q.get("02. open")), "high": safe_float(q.get("03. high")),
        "low": safe_float(q.get("04. low")), "source": "alphavantage",
    }


# ============================================================================== Finnhub (keyed)

def finnhub_quote(symbol: str, api_key: str) -> dict:
    resp = requests.get("https://finnhub.io/api/v1/quote",
                        params={"symbol": symbol, "token": api_key}, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    d = resp.json() or {}
    if safe_float(d.get("pc")) in (None, 0.0):
        raise RuntimeError(f"Finnhub empty quote: {d}")
    return {
        "last": safe_float(d.get("c")), "prev_close": safe_float(d.get("pc")),
        "open": safe_float(d.get("o")), "high": safe_float(d.get("h")),
        "low": safe_float(d.get("l")), "source": "finnhub",
    }


# ============================================================================ yfinance (keyless)

def yfinance_history(symbol: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.Ticker(symbol).history(period="1y", interval="1d", auto_adjust=False)
    if df is None or df.empty:
        raise RuntimeError("yfinance returned no data")
    df = df.rename(columns=str.lower)
    df.index = [to_date(ts) for ts in df.index]
    df.index.name = "date"
    return df[["open", "high", "low", "close", "volume"]].sort_index()


# ============================================== Stooq (keyless, proof-of-work gated, last resort)

def _solve_stooq_pow(c: str, difficulty: int) -> int:
    target = "0" * difficulty
    n = 0
    while True:
        h = hashlib.sha256(f"{c}{n}".encode()).hexdigest()
        if h.startswith(target):
            return n
        n += 1


def stooq_history(symbol: str) -> pd.DataFrame:
    url = f"https://stooq.com/q/d/l/?s={symbol.lower()}.us&i=d"
    session = requests.Session()
    session.headers.update(_HEADERS)
    text = session.get(url, timeout=HTTP_TIMEOUT).text.strip()

    if text.lower().startswith("<"):  # JS proof-of-work challenge
        m = re.search(r'const c="([^"]+)",d=(\d+)', text)
        if not m:
            raise RuntimeError(f"Stooq challenge not parseable: {text[:120]!r}")
        c, difficulty = m.group(1), int(m.group(2))
        n = _solve_stooq_pow(c, difficulty)
        session.post("https://stooq.com/__verify",
                     data={"c": c, "n": n},
                     headers={"Content-Type": "application/x-www-form-urlencoded"},
                     timeout=HTTP_TIMEOUT)
        text = session.get(url, timeout=HTTP_TIMEOUT).text.strip()

    if not text or text.lower().startswith("<") or "Date" not in text.splitlines()[0]:
        raise RuntimeError(f"Stooq returned no data for {symbol}: {text[:120]!r}")
    df = pd.read_csv(io.StringIO(text))
    df.columns = [col.strip().lower() for col in df.columns]
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.set_index("date").sort_index()
    return df[["open", "high", "low", "close", "volume"]]


# ============================================================================ fallback chains

def _fetch_history(symbol: str) -> pd.DataFrame:
    """Daily OHLCV with provider fallback. Raises only if every source fails."""
    errors = []
    if settings.alphavantage_api_key:
        try:
            return alphavantage_history(symbol, settings.alphavantage_api_key)
        except Exception as e:  # noqa: BLE001
            errors.append(f"alphavantage: {e}")
    for name, fn in (("yfinance", yfinance_history), ("stooq", stooq_history)):
        try:
            return fn(symbol)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name}: {e}")
    raise RuntimeError("All history providers failed -> " + " | ".join(errors))


def _fetch_quote(symbol: str) -> dict | None:
    if settings.finnhub_api_key:
        try:
            return finnhub_quote(symbol, settings.finnhub_api_key)
        except Exception:  # noqa: BLE001
            pass
    if settings.alphavantage_api_key:
        try:
            return alphavantage_quote(symbol, settings.alphavantage_api_key)
        except Exception:  # noqa: BLE001
            pass
    try:
        import yfinance as yf

        fi = yf.Ticker(symbol).fast_info
        last, prev = safe_float(fi.get("last_price")), safe_float(fi.get("previous_close"))
        if prev:
            return {"last": last, "prev_close": prev, "open": safe_float(fi.get("open")),
                    "high": safe_float(fi.get("day_high")), "low": safe_float(fi.get("day_low")),
                    "source": "yfinance"}
    except Exception:  # noqa: BLE001
        pass
    return None


# ================================================================================== public API

def get_history(symbol: str | None = None, lookback_days: int | None = None) -> pd.DataFrame:
    symbol = symbol or settings.SYMBOL
    df = _fetch_history(symbol)
    return df.tail(lookback_days) if lookback_days else df


def get_actual_close(symbol: str, when: str | date) -> float | None:
    target = to_date(when)
    df = _fetch_history(symbol)
    if target in df.index:
        return float(df.loc[target, "close"])
    return None


def get_quote(symbol: str | None = None) -> dict:
    symbol = symbol or settings.SYMBOL
    quote = _fetch_quote(symbol)
    df = _fetch_history(symbol)
    asof = df.index[-1].isoformat()
    if quote is None:
        last_row = df.iloc[-1]
        quote = {"last": float(last_row["close"]), "prev_close": float(last_row["close"]),
                 "open": float(last_row["open"]), "high": float(last_row["high"]),
                 "low": float(last_row["low"]), "source": "history"}
    elif not quote.get("prev_close"):
        quote["prev_close"] = float(df.iloc[-1]["close"])
    quote["symbol"] = symbol
    quote["asof"] = asof
    return quote
