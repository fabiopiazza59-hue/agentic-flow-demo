"""NYSE trading-day calendar helpers.

Uses pandas_market_calendars when available; otherwise falls back to a weekday-only
approximation (still correct for weekends, but not US market holidays).
"""

from __future__ import annotations

from datetime import date, timedelta

from ..utils import to_date

try:
    import pandas_market_calendars as mcal

    _NYSE = mcal.get_calendar("NYSE")
    _HAS_MCAL = True
except Exception:  # pragma: no cover - exercised only when dep missing
    _NYSE = None
    _HAS_MCAL = False


def _sessions(start: date, end: date) -> set[date]:
    if _HAS_MCAL:
        sched = _NYSE.schedule(start_date=start.isoformat(), end_date=end.isoformat())
        return {d.date() for d in sched.index}
    # weekday-only fallback
    out: set[date] = set()
    cur = start
    while cur <= end:
        if cur.weekday() < 5:
            out.add(cur)
        cur += timedelta(days=1)
    return out


def is_trading_day(when: str | date) -> bool:
    d = to_date(when)
    return d in _sessions(d, d)


def previous_trading_day(when: str | date) -> date:
    d = to_date(when)
    cur = d - timedelta(days=1)
    for _ in range(15):  # generous window to clear long holiday weekends
        if is_trading_day(cur):
            return cur
        cur -= timedelta(days=1)
    raise RuntimeError(f"No trading day found before {d}")


def last_n_trading_days(n: int, end: str | date) -> list[date]:
    """Most recent n trading days up to and including `end` if it is a session."""
    end_d = to_date(end)
    start = end_d - timedelta(days=n * 3 + 15)
    sessions = sorted(s for s in _sessions(start, end_d) if s <= end_d)
    return sessions[-n:]
