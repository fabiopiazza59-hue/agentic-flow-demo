"""Session plumbing shared by both arms: which session to forecast, the pre-open snapshot, and
the code-only anchor refresh.

Timing model (v2). GitHub starts scheduled runs late — 3.5–5.5 hours late through September 2026,
which put every prediction since 2026-08-27 after the open. So the work is split by how
time-critical it is:

  * Research (LLM calls) for session T may run any time between the previous session's close
    (+ settling time) and T's open minus PREDICT_CUTOFF_MINUTES. The evening run after the close
    does it, far from the bell; the morning runs are a fallback.
  * Every later run that still lands before T's open refreshes the forecast's anchor — the latest
    after-hours / pre-market trade — in code, with no LLM cost. The forecast keeps its view
    relative to the anchor (a multiplicative residual), so fresher price information flows in
    without re-running the models. Measured over 60 sessions, the pre-market price alone cuts the
    random walk's MAPE from 1.56% (prior close) to 1.27% (04:30 ET) and 1.11% (09:20 ET).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from ..config import settings
from ..data.market_calendar import forecast_target, previous_trading_day
from ..data.providers import get_anchor, get_history_through
from ..evals import integrity
from ..evals.probabilistic import ewma_sigma, log_returns
from ..features import build_features


def resolve_target(explicit: str | None, now: datetime) -> date | None:
    """The session to work on: an explicit --date, else the next session in its forecast window."""
    if explicit:
        from ..utils import to_date
        return to_date(explicit)
    return forecast_target(now, settings.SCORE_SETTLE_MINUTES)


def _information_cutoff(target: date, now: datetime) -> datetime:
    """For a live run, now. For a replay of a past session, just before that session opened."""
    opened = integrity.session_open(target.isoformat())
    if opened is not None and now >= opened:
        return opened - timedelta(minutes=1)
    return now


def take_snapshot(target: date, now: datetime | None = None) -> dict | None:
    """History through the session before `target`, the pre-open anchor, and model features.

    Returns None when no data provider has the previous session's official close yet — a
    forecast built on a stale prior close would be wrong in a way no model can fix.
    """
    now = now or datetime.now(timezone.utc)
    prev = previous_trading_day(target)
    history = get_history_through(settings.SYMBOL, prev, settings.LOOKBACK_DAYS)
    if history is None or not len(history):
        return None
    prev_close = float(history["close"].iloc[-1])
    since = integrity.session_close(history.index[-1].isoformat())
    anchor = get_anchor(settings.SYMBOL, prev_close, since, _information_cutoff(target, now))
    features = build_features(history, anchor=anchor)
    sigma = ewma_sigma(log_returns([float(c) for c in history["close"]]))
    features.update({
        "target_session": target.isoformat(),
        "forecast_made_at": now.isoformat(timespec="minutes"),
        "sigma_ewma": round(sigma, 6) if sigma else None,
        "anchor": anchor,
    })
    return {"history": history, "features": features, "anchor": anchor}


def anchor_fields(anchor: dict, prior_close: float, when_iso: str) -> dict:
    """Ledger fields describing the price the forecast is anchored to."""
    return {
        "anchor": round(float(anchor.get("price") or prior_close), 4),
        "anchor_source": anchor.get("source"),
        "anchor_time": anchor.get("time"),
        "anchor_live": bool(anchor.get("live")),
        "anchored_at": when_iso,
        "anchor_updates": 0,
    }


_PRICE_KEYS = ("predicted_close", "predicted_close_raw", "predicted_close_blend",
               "predicted_close_pre_gates")
_PRIOR_KEYS = ("center", "p10", "p50", "p90")


def reanchor_row(row: dict, anchor: dict, when_iso: str) -> dict | None:
    """Move a pending forecast onto a fresher pre-open anchor; None if nothing would change.

    Every price level on the row is scaled by new_anchor / old_anchor, so the forecast keeps its
    view relative to the anchor (the analysts' and deciders' residual) while absorbing the
    price information that arrived since. Only a live anchor newer than the current one counts.
    """
    if row.get("status") != "pending" or not anchor.get("live"):
        return None
    old = float(row.get("anchor") or row.get("prior_close") or 0)
    new = float(anchor.get("price") or 0)
    if old <= 0 or new <= 0:
        return None
    if row.get("anchor_live") and str(anchor.get("time") or "") <= str(row.get("anchor_time") or ""):
        return None
    k = new / old
    out = dict(row)
    for key in _PRICE_KEYS:
        if out.get(key) is not None:
            out[key] = round(float(out[key]) * k, 2)
    if isinstance(out.get("quantiles"), list):
        out["quantiles"] = [round(float(q) * k, 4) for q in out["quantiles"]]
    if isinstance(out.get("prior"), dict):
        prior = dict(out["prior"])
        for key in _PRIOR_KEYS:
            if prior.get(key) is not None:
                prior[key] = round(float(prior[key]) * k, 2)
        out["prior"] = prior
    prior_close = float(out.get("prior_close") or 0)
    if prior_close:
        out["predicted_direction"] = "up" if out["predicted_close"] > prior_close else "down"
    if isinstance(out.get("quantiles"), list):
        from ..evals.probabilistic import prob_above, prob_pass
        out["p_up"] = round(prob_above(out["quantiles"], prior_close), 4)
        out["confidence"] = round(prob_pass(out["quantiles"], out["predicted_close"]), 4)
    out.update({
        "anchor": round(new, 4),
        "anchor_source": anchor.get("source"),
        "anchor_time": anchor.get("time"),
        "anchor_live": True,
        "anchored_at": when_iso,
        "anchor_updates": int(row.get("anchor_updates") or 0) + 1,
    })
    return out
