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

from datetime import datetime, timezone

from ..config import settings


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def session_open(date_iso: str) -> datetime | None:
    """UTC datetime of the regular-session open for a YYYY-MM-DD session date."""
    day = _parse_ts(f"{date_iso}T00:00:00+00:00")
    if day is None:
        return None
    return day.replace(hour=settings.SESSION_OPEN_UTC_HOUR,
                       minute=settings.SESSION_OPEN_UTC_MINUTE)


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


def is_late(row: dict) -> bool:
    """True when the row was created at or after its own session's open.

    Seed rows are never "late" — they are backfilled history, not forecasts.
    """
    if row.get("seed"):
        return False
    late = row.get("late_minutes")
    if late is None:
        late = minutes_after_open(row.get("created_at"), row.get("date"))
    return late is not None and late >= 0


def annotate(row: dict) -> dict:
    """Stamp `late_minutes` on a row (in place) and return it.

    Backfill seed rows are skipped: they all carry the timestamp of the backfill run, not of a
    forecast, so a "minutes after open" figure would be meaningless for them.
    """
    if row.get("seed"):
        return row
    late = minutes_after_open(row.get("created_at"), row.get("date"))
    if late is not None:
        row["late_minutes"] = late
    return row


def is_pre_open_now(date_iso: str, now: datetime | None = None) -> bool:
    """Would a prediction created right now still be a pre-open forecast for this session?"""
    opened = session_open(date_iso)
    if opened is None:
        return True
    return (now or datetime.now(timezone.utc)) < opened
