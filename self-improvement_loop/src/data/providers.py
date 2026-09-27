"""Market-data providers with a layered fallback chain (robust in CI and locally).

Historical daily data & official past closes (used for features, backfill, validation):
    Alpha Vantage TIME_SERIES_DAILY (keyed, reliable from CI IPs)
      -> yfinance (keyless)
      -> Stooq CSV with proof-of-work solver (keyless, last resort)

Pre-open anchor (the freshest price information before the next open):
    Yahoo 1-minute bars incl. pre/post-market (keyless) -> Finnhub /quote (keyed, timestamped)
      -> the previous official close
    Alpha Vantage GLOBAL_QUOTE is not an anchor source: on the free tier it returns the previous
    session's close before the open, which the loop used to present as a flat pre-market gap.

Legacy quote chain (kept for back-compat): Finnhub -> Alpha Vantage -> yfinance -> history.

Public API:
    get_history(symbol, lookback)            -> DataFrame [open, high, low, close, volume]
    get_history_through(symbol, day, lookback) -> history whose last session is >= day, or None
    get_actual_close(symbol, date)           -> official close for a finished session (or None)
    get_anchor(symbol, prev_close, since)    -> {price, time, source, live}
    get_quote(symbol)                        -> Quote dict {last, prev_close, ..., source, stale}
"""

from __future__ import annotations

import hashlib
import io
import re
import time
from datetime import date, datetime, timezone
from functools import lru_cache

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

_AV_MIN_INTERVAL = 1.5  # seconds; free tier wants <= ~1 request/second
_av_last_call = 0.0


def _av_throttle() -> None:
    """Sleep so consecutive Alpha Vantage requests respect the free-tier rate limit."""
    global _av_last_call
    wait = _AV_MIN_INTERVAL - (time.monotonic() - _av_last_call)
    if wait > 0:
        time.sleep(wait)
    _av_last_call = time.monotonic()


def alphavantage_history(symbol: str, api_key: str) -> pd.DataFrame:
    _av_throttle()
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
    _av_throttle()
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
        "trading_day": q.get("07. latest trading day"),
    }


# ============================================================================== Finnhub (keyed)

def finnhub_quote(symbol: str, api_key: str) -> dict:
    resp = requests.get("https://finnhub.io/api/v1/quote",
                        params={"symbol": symbol, "token": api_key}, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    d = resp.json() or {}
    if safe_float(d.get("pc")) in (None, 0.0):
        raise RuntimeError(f"Finnhub empty quote: {d}")
    ts = safe_float(d.get("t"))
    return {
        "last": safe_float(d.get("c")), "prev_close": safe_float(d.get("pc")),
        "open": safe_float(d.get("o")), "high": safe_float(d.get("h")),
        "low": safe_float(d.get("l")), "source": "finnhub",
        "time": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else None,
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


def yfinance_extended_last(symbol: str, since: datetime, until: datetime) -> dict | None:
    """Latest Yahoo 1-minute bar close in [since, until], pre/post-market included.

    With `since` = the previous session's close this is the most recent after-hours or
    pre-market trade: information the market has already priced in before the next open.
    """
    import yfinance as yf

    df = yf.Ticker(symbol).history(period="5d", interval="1m", prepost=True)
    if df is None or df.empty or "Close" not in df:
        return None
    idx = df.index.tz_convert("UTC") if df.index.tz is not None else df.index.tz_localize("UTC")
    # A bar stamped hh:mm closes at hh:mm+1 — only bars completed by `until` are known then.
    mask = ((idx >= since) & (idx + pd.Timedelta(minutes=1) <= until)
            & df["Close"].notna().to_numpy())
    if not mask.any():
        return None
    sub = df[mask]
    ts = sub.index[-1]
    ts = ts.tz_convert("UTC") if ts.tzinfo is not None else ts.tz_localize("UTC")
    return {"price": float(sub["Close"].iloc[-1]), "time": ts.to_pydatetime().isoformat(),
            "source": "yfinance_ext"}


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

@lru_cache(maxsize=8)
def _fetch_history(symbol: str) -> pd.DataFrame:
    """Daily OHLCV with provider fallback. Raises only if every source fails.

    Cached per process so a single daily run hits the (rate-limited) data provider once per symbol
    instead of re-fetching for score/quote/predict separately.
    """
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


_FALLBACK_HISTORY = (("yfinance", yfinance_history), ("stooq", stooq_history))


def get_history_through(symbol: str, through: str | date,
                        lookback_days: int | None = None) -> pd.DataFrame | None:
    """Daily history whose last completed session is at least `through`, capped at `through`.

    Right after a close the primary provider can lag by a session; a forecast built on that
    would use the wrong prior close. Each provider is tried until one has caught up; None means
    none has yet (the caller should wait for the next run rather than guess).
    """
    day = to_date(through)
    candidates = []
    try:
        candidates.append(_fetch_history(symbol))
    except RuntimeError:
        pass
    for df in candidates:
        if len(df) and df.index[-1] >= day:
            df = df[df.index <= day]
            return df.tail(lookback_days) if lookback_days else df
    for _name, fn in _FALLBACK_HISTORY:
        try:
            df = fn(symbol)
        except Exception:  # noqa: BLE001
            continue
        if len(df) and df.index[-1] >= day:
            df = df[df.index <= day]
            return df.tail(lookback_days) if lookback_days else df
    return None


def get_actual_close(symbol: str, when: str | date) -> float | None:
    """Official close of a finished session, or None if no provider has it yet."""
    target = to_date(when)
    df = _fetch_history(symbol)
    if target in df.index:
        return float(df.loc[target, "close"])
    if len(df) and df.index[-1] >= target:
        return None  # the provider is past this date and has no bar: not a session
    df = get_history_through(symbol, target)
    if df is not None and target in df.index:
        return float(df.loc[target, "close"])
    return None


def get_anchor(symbol: str, prev_close: float, since: datetime,
               now: datetime | None = None) -> dict:
    """The freshest price information available before the next open.

    Returns {price, time, source, live}: the latest extended-hours trade after `since` (the
    previous session's close) when one exists, else the previous close itself (`live` False).
    A live anchor is the free baseline every forecast has to beat — the market's own
    pre-open estimate — and moves more than 35% from the prior close are rejected as bad ticks
    (a real earnings gap on this name has exceeded 15%, so the band must stay wide).
    """
    now = now or datetime.now(timezone.utc)

    def _sane(price) -> bool:
        return bool(price) and prev_close > 0 and abs(float(price) / prev_close - 1.0) < 0.35

    try:
        ext = yfinance_extended_last(symbol, since, now)
        if ext and _sane(ext["price"]):
            return {**ext, "price": round(float(ext["price"]), 4), "live": True}
    except Exception:  # noqa: BLE001 - keyless source, best effort
        pass
    if settings.finnhub_api_key:
        try:
            q = finnhub_quote(symbol, settings.finnhub_api_key)
            t = q.get("time")
            if t and datetime.fromisoformat(t) > since and _sane(q.get("last")):
                return {"price": float(q["last"]), "time": t, "source": "finnhub", "live": True}
        except Exception:  # noqa: BLE001
            pass
    return {"price": float(prev_close), "time": since.isoformat(), "source": "prior_close",
            "live": False}


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
    # A quote whose trading day is not after the latest completed session carries no new
    # information: pre-open, Alpha Vantage's "price" is just the previous close.
    trading_day = quote.get("trading_day")
    quote["stale"] = quote.get("source") == "history" or bool(trading_day and trading_day <= asof)
    quote["symbol"] = symbol
    quote["asof"] = asof
    return quote
