"""NYSE trading-day calendar helpers.

Uses pandas_market_calendars when available (holidays, early closes, DST); otherwise falls back
to a weekday-only approximation with the regular 09:30-16:00 America/New_York session.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from functools import lru_cache

from ..utils import to_date

try:
    import pandas_market_calendars as mcal

    _NYSE = mcal.get_calendar("NYSE")
    _HAS_MCAL = True
except Exception:  # pragma: no cover - exercised only when dep missing
    _NYSE = None
    _HAS_MCAL = False

try:
    from zoneinfo import ZoneInfo

    _NY = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover - no tz database
    _NY = None


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


def sessions_between(start: str | date, end: str | date) -> list[date]:
    """NYSE sessions from `start` to `end`, inclusive, oldest first."""
    return sorted(_sessions(to_date(start), to_date(end)))


def last_n_trading_days(n: int, end: str | date) -> list[date]:
    """Most recent n trading days up to and including `end` if it is a session."""
    end_d = to_date(end)
    start = end_d - timedelta(days=n * 3 + 15)
    sessions = sorted(s for s in _sessions(start, end_d) if s <= end_d)
    return sessions[-n:]


# ------------------------------------------------------------------ session open/close times

def _regular_hours(d: date) -> tuple[datetime, datetime]:
    """09:30-16:00 America/New_York in UTC (DST-aware); fixed EDT offsets without a tz db."""
    if _NY is not None:
        opened = datetime.combine(d, time(9, 30), tzinfo=_NY).astimezone(timezone.utc)
        closed = datetime.combine(d, time(16, 0), tzinfo=_NY).astimezone(timezone.utc)
        return opened, closed
    base = datetime.combine(d, time(0, 0), tzinfo=timezone.utc)
    return base + timedelta(hours=13, minutes=30), base + timedelta(hours=20)


@lru_cache(maxsize=64)
def _month_times(year: int, month: int) -> dict[date, tuple[datetime, datetime]]:
    start = date(year, month, 1)
    end = (date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1))
    if _HAS_MCAL:
        sched = _NYSE.schedule(start_date=start.isoformat(), end_date=end.isoformat())
        return {
            idx.date(): (row["market_open"].to_pydatetime(), row["market_close"].to_pydatetime())
            for idx, row in sched.iterrows()
        }
    return {d: _regular_hours(d) for d in _sessions(start, end)}


def session_times(when: str | date) -> tuple[datetime, datetime] | None:
    """(open, close) of the regular session on `when` as UTC datetimes, or None if no session.

    Holiday-, early-close- and DST-aware: the open is 13:30 UTC in summer and 14:30 UTC in
    winter, and e.g. the day after Thanksgiving closes at 18:00 UTC.
    """
    d = to_date(when)
    return _month_times(d.year, d.month).get(d)


def _sessions_around(now: datetime, back_days: int = 10, fwd_days: int = 12) -> list[date]:
    d = now.astimezone(timezone.utc).date()
    return sorted(_sessions(d - timedelta(days=back_days), d + timedelta(days=fwd_days)))


def next_session_after(now: datetime) -> date | None:
    """The first session whose regular open is still in the future at `now`."""
    for s in _sessions_around(now):
        times = session_times(s)
        if times and now < times[0]:
            return s
    return None


def forecast_target(now: datetime, settle_minutes: int = 30) -> date | None:
    """The session a forecast made at `now` should target, or None while one is in progress.

    A forecast for session T is legitimate from `settle_minutes` after the previous session's
    close (its prior close is then final) until T opens. During a session — and in the short
    window after its close while the official close settles — there is nothing to forecast.
    """
    target = next_session_after(now)
    if target is None:
        return None
    earlier = [s for s in _sessions_around(now) if s < target]
    if earlier:
        times = session_times(earlier[-1])
        if times and now < times[1] + timedelta(minutes=settle_minutes):
            return None
    return target
