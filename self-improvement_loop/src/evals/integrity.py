"""Data-integrity checks on ledger rows — is this row actually a prediction?

A row only counts as a pre-open forecast if it was created before the session it forecasts
opened. Rows created after the open saw part (or all) of the tape they were predicting, so
they carry lookahead: the `news` analyst quotes live intraday prices, and a row created after
the close is scored against a session whose result was already visible.

We do not delete such rows — the ledger is the experiment log. We label them (`late_minutes`)
and let `aggregate()` report the record both ways, so the honest pre-open number is always
visible next to the full one. Paper canon rules 9 and 17 (arXiv:2609.05663): check the
calendar footprint, and know your timing-luck floor.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from ..config import settings
from ..data.market_calendar import session_times


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def session_open(date_iso: str) -> datetime | None:
    """UTC datetime of the regular-session open for a YYYY-MM-DD session date.

    Read off the NYSE calendar, so it is DST-aware (13:30 UTC in summer, 14:30 UTC in winter).
    Dates that are not sessions fall back to the configured summer open, which is never later
    than the real one — the conservative side for a "was this made before the open?" check.
    """
    try:
        times = session_times(date_iso)
    except Exception:  # noqa: BLE001 - a malformed date falls through to the fixed open
        times = None
    if times:
        return times[0]
    day = _parse_ts(f"{date_iso}T00:00:00+00:00")
    if day is None:
        return None
    return day.replace(hour=settings.SESSION_OPEN_UTC_HOUR,
                       minute=settings.SESSION_OPEN_UTC_MINUTE)


def session_close(date_iso: str) -> datetime | None:
    """UTC datetime of the regular-session close (early-close aware); 16:00 ET fallback."""
    try:
        times = session_times(date_iso)
    except Exception:  # noqa: BLE001
        times = None
    if times:
        return times[1]
    opened = session_open(date_iso)
    return opened + timedelta(hours=6, minutes=30) if opened else None


def minutes_after_open(created_at: str | None, date_iso: str | None) -> int | None:
    """Minutes between the session open and when the row was created.

    Negative (or zero) means the row was created before the open — a real forecast.
    None when either timestamp is unusable (e.g. backfill seed rows).
    """
    created = _parse_ts(created_at)
    opened = session_open(date_iso) if date_iso else None
    if created is None or opened is None:
        return None
    return int((created - opened).total_seconds() // 60)


def last_write(row: dict) -> str | None:
    """When the forecast last changed: its creation, or its latest pre-open anchor refresh."""
    stamps = [s for s in (row.get("created_at"), row.get("anchored_at")) if s]
    return max(stamps, key=lambda s: _parse_ts(s) or datetime.min.replace(tzinfo=timezone.utc)) \
        if stamps else None


def is_late(row: dict) -> bool:
    """True when the row was created (or last re-anchored) at or after its session's open.

    Seed rows are never "late" — they are backfilled history, not forecasts.
    """
    if row.get("seed"):
        return False
    late = row.get("late_minutes")
    if late is None:
        late = minutes_after_open(last_write(row), row.get("date"))
    return late is not None and late >= 0


def annotate(row: dict) -> dict:
    """Stamp `late_minutes` on a row (in place) and return it.

    Measured from the row's last write, so an anchor refresh that slipped past the open would
    be caught. Backfill seed rows are skipped: they all carry the timestamp of the backfill run,
    not of a forecast, so a "minutes after open" figure would be meaningless for them.
    """
    if row.get("seed"):
        return row
    late = minutes_after_open(last_write(row), row.get("date"))
    if late is not None:
        row["late_minutes"] = late
    return row


def is_pre_open_now(date_iso: str, now: datetime | None = None, margin_minutes: int = 0) -> bool:
    """Would a prediction created right now still be a pre-open forecast for this session?

    `margin_minutes` keeps a safety gap before the open: a run that starts a minute before
    the bell and spends two minutes on LLM calls would otherwise stamp a post-open row.
    """
    opened = session_open(date_iso)
    if opened is None:
        return True
    return (now or datetime.now(timezone.utc)) < opened - timedelta(minutes=margin_minutes)


def session_has_closed(date_iso: str, now: datetime | None = None,
                       settle_minutes: int = 30) -> bool:
    """True once the session's official close is final (close + a settling buffer).

    Scoring before this would grade a forecast against a partial intraday bar, which some data
    providers return as the day's "close" while the session is still trading.
    """
    closed = session_close(date_iso)
    if closed is None:
        return True
    return (now or datetime.now(timezone.utc)) >= closed + timedelta(minutes=settle_minutes)
