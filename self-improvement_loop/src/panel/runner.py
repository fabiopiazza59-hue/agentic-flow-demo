"""Panel runner — research, re-anchor and score many names per session.

Why a panel: at one forecast a day the daily APE gain has an SD of ~0.3%, so a real 0.05% edge
needs ~1.4 years of clean forecasts to show (spec/v2-sota-upgrade.md §0). Averaging the gain over
N names each session shrinks that noise by up to √N; the day stays the inferential unit, so
names that move together are not double-counted.

Per name and session the row carries three forecasts of the close:
  anchor              the latest pre-open trade — the free baseline
  predicted_close_llm anchor · (1 + adj · σ), adj = the LLM's clamped σ-move (the counterfactual)
  predicted_close     what ships: the LLM forecast, or the anchor if the kill switch is off
The LLM forecast is always logged, so evidence keeps accruing after a switch-off.
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime, timezone

from ..config import settings
from ..data.market_calendar import previous_trading_day
from ..evals import integrity
from ..evals import probabilistic as pb
from ..evals.metrics import ape, score_row
from ..loop.session import reanchor_row
from ..utils import now_iso, read_jsonl, write_jsonl
from . import analyst, data, evolve

VERSION = "panel-v1"


def log(msg: str) -> None:
    print(f"[panel] {msg}", flush=True)


# ------------------------------------------------------------------------------ ledger & state

def load_rows() -> list[dict]:
    return read_jsonl(settings.PANEL_LEDGER_PATH)


def save_rows(rows: list[dict]) -> None:
    write_jsonl(settings.PANEL_LEDGER_PATH,
                sorted(rows, key=lambda r: (r.get("date", ""), r.get("symbol", ""))))


def load_state() -> dict:
    default = {"llm_enabled": True}
    p = settings.PANEL_STATE_PATH
    if not p.exists():
        return default
    try:
        return {**default, **json.loads(p.read_text(encoding="utf-8"))}
    except json.JSONDecodeError:
        return default


def save_state(state: dict) -> None:
    settings.PANEL_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    settings.PANEL_STATE_PATH.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


# ------------------------------------------------------------------------------ calibration

def _pooled_residuals(rows: list[dict], before: str, key: str) -> list[float]:
    """Standardized residuals ln(actual/center)/σ of clean scored rows before `before`."""
    out = []
    for r in sorted(rows, key=lambda r: r["date"]):
        if r["date"] >= before or r.get("status") != "scored" or integrity.is_late(r):
            continue
        c, a, s = r.get(key), r.get("actual_close"), r.get("sigma_pct")
        if c and a and s:
            out.append(math.log(float(a) / float(c)) / float(s))
    return out[-pb.CALIBRATION_WINDOW * 10:]


def _distribution(rows: list[dict], target: str, point: float, sigma: float,
                  prior_close: float) -> dict:
    zq, method = pb.z_quantiles(_pooled_residuals(rows, target, "predicted_close"))
    q = pb.predictive_quantiles(point, sigma, zq)
    return {"quantiles": q, "p_up": round(pb.prob_above(q, prior_close), 4),
            "confidence": round(pb.prob_pass(q, point, settings.PASS_THRESHOLD), 4),
            "calibration": method}


# ---------------------------------------------------------------------------------- research

def _features(hist) -> dict:
    closes = [float(c) for c in hist["close"]]
    sigma = pb.ewma_sigma(pb.log_returns(closes)) or settings.DEFAULT_SIGMA

    def ret(n):
        return closes[-1] / closes[-1 - n] - 1.0 if len(closes) > n else 0.0

    return {"prev_close": closes[-1], "sigma": sigma, "ret_5d": ret(5), "ret_20d": ret(20)}


def research(rows: list[dict], target: date, client, now: datetime | None = None) -> list[dict]:
    """Forecast every panel name for `target` from one snapshot; returns the updated ledger."""
    now = now or datetime.now(timezone.utc)
    iso = target.isoformat()
    if not integrity.is_pre_open_now(iso, now, settings.PREDICT_CUTOFF_MINUTES):
        log(f"{iso}'s research window has closed — no panel forecast for this session.")
        return rows
    prev = previous_trading_day(target)
    symbols = list(settings.PANEL_SYMBOLS)
    hist = data.daily_history(symbols, prev)
    if len(hist) < settings.PANEL_MIN_NAMES:
        log(f"only {len(hist)} names have {prev}'s close yet — retrying next run.")
        return rows
    since = integrity.session_close(prev.isoformat())
    anchors = data.extended_last(list(hist), since, now)

    names: dict[str, dict] = {}
    for s, h in hist.items():
        f = _features(h)
        a = anchors.get(s)
        if a and f["prev_close"] > 0 and abs(a["price"] / f["prev_close"] - 1) < 0.35:
            f.update(anchor=a["price"], anchor_live=True, anchor_time=a["time"],
                     anchor_source=a["source"], gap=a["price"] / f["prev_close"] - 1)
        else:
            f.update(anchor=f["prev_close"], anchor_live=False, anchor_time=since.isoformat(),
                     anchor_source="prior_close", gap=None)
        names[s] = f

    made_at = now.isoformat(timespec="minutes")
    reg = evolve.load_registry()
    champ = evolve.champion(reg)
    adj = analyst.adjustments(names, iso, made_at, client, strategy=champ["strategy"])
    # Challenger prompts run in shadow on the same snapshot; nothing they say ships.
    shadows = {v["id"]: analyst.adjustments(names, iso, made_at, client, strategy=v["strategy"])
               for v in evolve.challengers(reg)} if client is not None else {}
    state = load_state()
    created = now_iso()
    new_rows = []
    for s, f in names.items():
        a = adj[s]
        llm = round(f["anchor"] * (1.0 + a["sigma"] * f["sigma"]), 4)
        shipped = llm if state["llm_enabled"] else round(f["anchor"], 4)
        row = {
            "date": iso, "symbol": s, "created_at": created, "pipeline_version": VERSION,
            "prior_close": round(f["prev_close"], 4),
            "anchor": round(f["anchor"], 4), "anchor_live": f["anchor_live"],
            "anchor_time": f["anchor_time"], "anchor_source": f["anchor_source"],
            "anchored_at": created, "anchor_updates": 0,
            "sigma_pct": round(f["sigma"], 6),
            "adj_sigma_raw": a["raw"], "adj_sigma": a["sigma"], "reason": a["reason"],
            "web_results": a["web_results"], "llm_enabled": state["llm_enabled"],
            "prompt_variant": champ["id"],
            "shadow_adj": {vid: sh[s]["sigma"] for vid, sh in shadows.items()},
            "predicted_close_llm": llm, "predicted_close": shipped,
            "predicted_direction": "up" if shipped > f["prev_close"] else "down",
            **_distribution(rows, iso, shipped, f["sigma"], f["prev_close"]),
            "status": "pending", "actual_close": None,
        }
        integrity.annotate(row)
        new_rows.append(row)
    moved = sum(1 for r in new_rows if r["adj_sigma"] != 0)
    live = sum(1 for r in new_rows if r["anchor_live"])
    log(f"researched {iso}: {len(new_rows)} names, {live} live anchors, prompt {champ['id']} "
        f"moved {moved} off their anchor"
        f"{'' if state['llm_enabled'] else ' (shadow only: LLM switched off)'}"
        f"{'; shadow prompts: ' + ', '.join(shadows) if shadows else ''}.")
    return [r for r in rows if r.get("date") != iso] + new_rows


def refresh(rows: list[dict], target: date, now: datetime | None = None) -> list[dict]:
    """Re-anchor the pending rows for `target` on fresher pre-open trades (code only)."""
    now = now or datetime.now(timezone.utc)
    iso = target.isoformat()
    if not integrity.is_pre_open_now(iso, now, settings.REFRESH_CUTOFF_MINUTES):
        return rows
    pending = [r for r in rows if r.get("date") == iso and r.get("status") == "pending"]
    if not pending:
        return rows
    since = integrity.session_close(previous_trading_day(target).isoformat())
    anchors = data.extended_last([r["symbol"] for r in pending], since, now)
    stamp, changed, out = now_iso(), 0, []
    for r in rows:
        a = anchors.get(r.get("symbol")) if r.get("date") == iso else None
        new = reanchor_row(r, a, stamp) if a else None
        if new is not None:
            integrity.annotate(new)
            changed += 1
        out.append(new or r)
    log(f"{iso}: re-anchored {changed} of {len(pending)} pending names.")
    return out


# ------------------------------------------------------------------------------------ scoring

def score(rows: list[dict], now: datetime | None = None) -> list[dict]:
    """Score pending rows whose session has closed and settled (one download per session)."""
    by_date: dict[str, list[dict]] = {}
    for r in rows:
        if r.get("status") == "pending" and integrity.session_has_closed(
                r["date"], now, settings.SCORE_SETTLE_MINUTES):
            by_date.setdefault(r["date"], []).append(r)
    for d, pend in sorted(by_date.items()):
        hist = data.daily_history([r["symbol"] for r in pend], d, period="1mo")
        n = 0
        for r in pend:
            h = hist.get(r["symbol"])
            if h is None:
                continue
            actual = float(h["close"].iloc[-1])
            r.update(score_row(r, actual))
            r["llm_ape"] = ape(float(r["predicted_close_llm"]), actual)
            if r.get("shadow_adj"):
                r["shadow_ape"] = evolve.shadow_ape(r, actual)
            r["scored_at"] = now_iso()
            n += 1
        log(f"scored {n} of {len(pend)} names for {d}.")
    return rows
