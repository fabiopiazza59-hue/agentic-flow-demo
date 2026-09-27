"""Daily orchestrator for arm A (the analyst ensemble) — and the machinery run_ab reuses.

Modes:
  daily     score finished sessions -> research the next session, or refresh its anchor
            -> report -> commit                                                     (default)
  score     score every pending prediction whose session has closed
  predict   research the next session (or refresh the anchor of its pending forecast)
  report    regenerate results/* artifacts from the ledger
  backfill  seed the ledger with recent scored days from history (baseline-only, no API)

Which session: with --date, that one (a past date is a replay, flagged late); otherwise the next
session whose forecast window is open — from the previous close (+ settling time) until
PREDICT_CUTOFF_MINUTES before its open. Scheduled runs are therefore idempotent: the first run in
the window researches, later pre-open runs only refresh the anchor (see loop/session.py).

Flags:
  --date YYYY-MM-DD   work on this session instead of the next one
  --dry-run           no git commit; uses offline stubs if no ANTHROPIC_API_KEY
  --no-commit         run fully but skip the git commit step
  --allow-late        write a prediction even after the session opened (row is flagged)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import date, datetime, timezone

from ..agents.analysts import get_client, run_analysts
from ..agents.meta_judge import adjust, clamp_adjustment
from ..agents.reflector import diagnose_failures, reflect
from ..config import PROJECT_ROOT, settings
from ..data.market_calendar import is_trading_day
from ..data.providers import get_actual_close, get_history
from ..evals import integrity, lab
from ..evals.gates import apply_gates
from ..evals.metrics import score_row
from ..evals.probabilistic import forecast_distribution
from ..evals.scorecard import load_scorecards, save_scorecards, update_scorecards
from ..utils import iso_today, now_iso, read_jsonl, upsert_ledger_row, write_jsonl
from .. import report
from . import session

# Relative to the project root (self-improvement_loop/).
COMMIT_PATHS = [
    "data",
    "results",
    "learnings",
    "learnings_b",
    "RESULTS.md",
    "README.md",
]


def log(msg: str) -> None:
    print(f"[run_daily] {msg}", flush=True)


# ---------------------------------------------------------------------------------- scoring

def _closest_strategy(analyst_predictions: dict, actual: float) -> str | None:
    best, best_ape = None, None
    for name, pred in analyst_predictions.items():
        p = pred.get("predicted_close")
        if p is None:
            continue
        a = abs(float(p) - actual) / abs(actual)
        if best_ape is None or a < best_ape:
            best, best_ape = name, a
    return best


def do_score(rows: list[dict], client, now: datetime | None = None) -> tuple[list[dict], dict | None]:
    """Score every pending prediction whose session has closed, oldest first.

    A session only counts as closed SCORE_SETTLE_MINUTES after its official close, so a partial
    intraday bar can never be graded as the close. Returns (rows, last row scored or None).
    """
    pending = sorted((r for r in rows if r.get("status") == "pending"), key=lambda r: r["date"])
    if not pending:
        log("no pending predictions to score.")
        return rows, None
    scorecards = load_scorecards()
    scored: list[dict] = []
    for target in pending:
        if not integrity.session_has_closed(target["date"], now, settings.SCORE_SETTLE_MINUTES):
            log(f"{target['date']} has not closed (and settled) yet; leaving it pending.")
            continue
        actual = get_actual_close(settings.SYMBOL, target["date"])
        if actual is None:
            log(f"actual close for {target['date']} not available yet; skipping score.")
            continue

        target.update(score_row(target, actual))
        target["winning_strategy"] = _closest_strategy(target.get("analyst_predictions", {}), actual)
        target["scored_at"] = now_iso()
        scorecards = update_scorecards(scorecards, target.get("analyst_predictions", {}), actual)

        # reflect -> post-mortem + bounded STRATEGY.md note + (on FAIL) failure analysis
        reflection = reflect(target, scorecards, client=client)
        _write_learning(target["date"], reflection)
        rows = upsert_ledger_row(rows, target)
        if target.get("pass") is False:
            _append_failure_log(target, reflection)
        log(f"scored {target['date']}: predicted {target['predicted_close']} vs actual {actual} "
            f"(APE {target['ape']*100:.2f}%, {'PASS' if target['pass'] else 'FAIL'}).")
        scored.append(target)

    if scored:
        save_scorecards(scorecards)
        # Rolling "what's not working" self-diagnosis, once per run rather than per row.
        _refresh_whats_not_working(rows, scorecards, client)
    return rows, (scored[-1] if scored else None)


def _write_learning(d: str, reflection: dict) -> None:
    settings.ensure_dirs()
    (settings.LEARNINGS_DIR / f"{d}.md").write_text(
        reflection.get("postmortem_md", ""), encoding="utf-8"
    )
    note = (reflection.get("strategy_note") or "").strip()
    if note:
        if not settings.STRATEGY_PATH.exists():
            settings.STRATEGY_PATH.write_text(
                "# Living Strategy — AMZN Close Predictor\n\n"
                "Append-only insights. Newest at the bottom.\n", encoding="utf-8"
            )
        with settings.STRATEGY_PATH.open("a", encoding="utf-8") as f:
            f.write(f"\n- ({d}) {note}")


# Failure-focused learnings. Paths derive from LEARNINGS_DIR at call time so tests can isolate them.

def _failures_path():
    return settings.LEARNINGS_DIR / "FAILURES.md"


def _whats_not_working_path():
    return settings.LEARNINGS_DIR / "WHATS_NOT_WORKING.md"


def _append_failure_log(row: dict, reflection: dict) -> None:
    """Append a concentrated entry for every missed prediction (>1% error)."""
    settings.ensure_dirs()
    p = _failures_path()
    if not p.exists():
        p.write_text(
            "# Failure Log — AMZN Close Predictor\n\n"
            "Every missed prediction (>1% error), newest at the bottom. "
            "Reread before each shot.\n", encoding="utf-8"
        )
    ape_pct = (row.get("ape") or 0) * 100
    entry = (
        f"\n## {row.get('date')} — FAIL (APE {ape_pct:.2f}%)\n"
        f"- Predicted {row.get('predicted_close')} vs actual {row.get('actual_close')} "
        f"(prior {row.get('prior_close')}); dir hit: {row.get('directional_hit')}; "
        f"beat baseline: {row.get('beats_baseline')}; closest analyst: {row.get('winning_strategy')}.\n"
        f"{(reflection.get('failure_reflection') or '').strip()}\n"
    )
    with p.open("a", encoding="utf-8") as f:
        f.write(entry)


def _refresh_whats_not_working(rows: list[dict], scorecards: dict, client) -> None:
    """Regenerate the rolling failure self-diagnosis once at least one real prediction has failed."""
    scored = [r for r in rows if r.get("status") == "scored" and not r.get("seed")]
    if not any(r.get("pass") is False for r in scored):
        return
    md = diagnose_failures(scored, scorecards, client=client)
    _whats_not_working_path().write_text(md, encoding="utf-8")


# --------------------------------------------------------------------------------- predicting

def _past_clean(rows: list[dict], before_iso: str) -> list[dict]:
    return sorted((r for r in rows if r.get("status") == "scored" and not r.get("seed")
                   and r.get("date", "") < before_iso and not integrity.is_late(r)),
                  key=lambda r: r["date"])


def do_predict(rows: list[dict], target_date: date, client,
               features: dict | None = None,
               allow_late: bool = False) -> tuple[list[dict], dict | None]:
    """Research `target_date` for arm A and append a pending row (v2 pipeline).

    analysts -> the lab's champion blend (code) -> the judge's σ-sized adjustment (clamped in
    code) -> guardrail gates -> a calibrated predictive distribution. Every intermediate value is
    logged so the lab can measure each stage. Pass a pre-built `features` snapshot for a fair
    A/B run.

    Refuses to write a live row once the session's research window has closed (less than
    PREDICT_CUTOFF_MINUTES before the open): 40 of the first 74 live rows in this ledger were
    written after the open, two after the close. A past-dated `--date` is a replay, not a live
    forecast, so the guard stays out of its way; replays and `--allow-late` rows are stamped
    `late_minutes` and so stay out of the pre-open headline.
    """
    target_iso = target_date.isoformat()
    live = target_iso >= iso_today()
    if settings.ENFORCE_PRE_OPEN and live and not allow_late and not integrity.is_pre_open_now(
            target_iso, margin_minutes=settings.PREDICT_CUTOFF_MINUTES):
        log(f"{target_date} opens in under {settings.PREDICT_CUTOFF_MINUTES} min or has opened — "
            f"refusing to write a post-open 'prediction'. Use --allow-late to override (the row "
            f"is then flagged late_minutes).")
        return rows, None

    if features is None:
        snap = session.take_snapshot(target_date)
        if snap is None:
            log("no data provider has the previous session's close yet — retrying next run.")
            return rows, None
        features = snap["features"]
    prior_close = float(features["prev_close"])
    anchor = features.get("anchor") or {"price": prior_close, "source": "prior_close",
                                        "time": None, "live": False}
    anchor_price = float(anchor.get("price") or prior_close)
    sigma = float(features.get("sigma_ewma") or features.get("realized_vol_20d")
                  or settings.DEFAULT_SIGMA)

    state = lab.load_state()
    scorecards = load_scorecards()
    strategy_md = settings.STRATEGY_PATH.read_text(encoding="utf-8") if settings.STRATEGY_PATH.exists() else ""
    recent_learnings = _recent_learnings(settings.LEARNINGS_CONTEXT_N)
    wnw_path = _whats_not_working_path()
    whats_not_working = wnw_path.read_text(encoding="utf-8") if wnw_path.exists() else ""

    analyst_predictions = run_analysts(
        features, client=client,
        context={"scorecards": scorecards, "whats_not_working": whats_not_working},
    )

    blend = lab.aggregate_analysts(state["champion"], analyst_predictions,
                                   _past_clean(rows, target_iso), anchor_price)
    if blend is None:
        blend = round(anchor_price, 2)      # no analyst survived: ship the free baseline
    if state["judge_enabled"]:
        adj_raw, judge_note = adjust(blend, state["champion"], sigma, analyst_predictions,
                                     scorecards, strategy_md, recent_learnings, features,
                                     client=client, whats_not_working=whats_not_working)
    else:
        adj_raw, judge_note = 0.0, "Judge off: the lab measured its adjustments as harmful."
    adj = clamp_adjustment(adj_raw)
    pre_gates = round(blend * (1.0 + adj * sigma), 2)

    final = {"predicted_close": pre_gates,
             "direction": "up" if pre_gates > prior_close else "down",
             "confidence": 0.5, "rationale": judge_note}
    if state["gates_enabled"]:
        final, gates = apply_gates(final, analyst_predictions, rows, features)
    else:
        final, gates = {**final, "predicted_close_raw": pre_gates}, []

    dist = forecast_distribution(rows, target_iso, float(final["predicted_close"]), sigma,
                                 prior_close, is_clean=lambda r: not integrity.is_late(r),
                                 threshold=settings.PASS_THRESHOLD)
    created = now_iso()
    row = {
        "date": target_iso,
        "created_at": created,
        "pipeline_version": settings.PIPELINE_VERSION,
        "prior_close": prior_close,
        "predicted_close": final["predicted_close"],
        "predicted_close_raw": final.get("predicted_close_raw"),
        "predicted_close_pre_gates": pre_gates,
        "predicted_close_blend": blend,
        "aggregator": state["champion"],
        "judge_adjustment_sigma_raw": round(float(adj_raw), 3),
        "judge_adjustment_sigma": round(adj, 3),
        "predicted_direction": final["direction"],
        "confidence": dist["p_pass"],
        "p_up": dist["p_up"],
        "sigma_pct": round(sigma, 6),
        "quantiles": dist["quantiles"],
        "calibration": dist["calibration"],
        "analyst_predictions": analyst_predictions,
        "analyst_anchor": round(anchor_price, 4),
        "rationale": final.get("rationale", ""),
        "gates_applied": gates,
        "quote_source": features.get("quote_source"),
        **session.anchor_fields(anchor, prior_close, created),
        "status": "pending",
        "actual_close": None,
    }
    integrity.annotate(row)
    rows = upsert_ledger_row(rows, row)
    log(f"predicted {target_iso}: close ≈ {row['predicted_close']} ({row['predicted_direction']}, "
        f"P(up) {row['p_up']}, P(PASS) {row['confidence']}) — {state['champion']} blend {blend} "
        f"of {len(analyst_predictions)} analysts, judge {adj:+.2f}σ, anchor {row['anchor']} "
        f"({row['anchor_source']}){'; gates: ' + ', '.join(gates) if gates else ''}.")
    return rows, row


def refresh_anchor(rows: list[dict], target: date, anchor: dict) -> tuple[list[dict], bool]:
    """Re-anchor the pending row for `target` on a fresher pre-open trade (code only)."""
    target_iso = target.isoformat()
    existing = next((r for r in rows if r.get("date") == target_iso), None)
    if existing is None or existing.get("status") != "pending":
        return rows, False
    if not integrity.is_pre_open_now(target_iso, margin_minutes=settings.REFRESH_CUTOFF_MINUTES):
        return rows, False
    updated = session.reanchor_row(existing, anchor, now_iso())
    if updated is None:
        return rows, False
    integrity.annotate(updated)
    return upsert_ledger_row(rows, updated), True


def forecast_or_refresh(rows: list[dict], target: date, client,
                        allow_late: bool = False) -> list[dict]:
    """Research `target` if it has no row yet, else refresh the pending row's anchor."""
    existing = next((r for r in rows if r.get("date") == target.isoformat()), None)
    if existing is None:
        rows, _ = do_predict(rows, target, client, allow_late=allow_late)
        return rows
    if existing.get("status") != "pending" or not integrity.is_pre_open_now(
            target.isoformat(), margin_minutes=settings.REFRESH_CUTOFF_MINUTES):
        log(f"{target} already has a forecast and is past its refresh window; nothing to do.")
        return rows
    snap = session.take_snapshot(target)
    if snap is None:
        return rows
    rows, changed = refresh_anchor(rows, target, snap["anchor"])
    log(f"{target}: anchor {'refreshed to ' + str(snap['anchor']['price']) if changed else 'unchanged (no fresher pre-open trade)'}.")
    return rows


def _recent_learnings(n: int) -> list[str]:
    files = sorted(settings.LEARNINGS_DIR.glob("20*.md"))
    return [f.read_text(encoding="utf-8") for f in files[-n:]]


# ----------------------------------------------------------------------------------- backfill

def do_backfill(rows: list[dict], n: int = 25) -> list[dict]:
    """Seed scored rows from history so metrics/dashboard have content (baseline-only predictions)."""
    history = get_history(settings.SYMBOL, settings.LOOKBACK_DAYS)
    sessions = [d for d in history.index]
    existing = {r["date"] for r in rows}
    seeded = 0
    for i in range(len(sessions) - n, len(sessions)):
        if i <= 0:
            continue
        d = sessions[i]
        if d.isoformat() in existing:
            continue
        prior = float(history.iloc[i - 1]["close"])
        actual = float(history.iloc[i]["close"])
        # naive seed prediction = prior close (random walk) so early dashboard isn't empty
        row = {
            "date": d.isoformat(),
            "created_at": now_iso(),
            "prior_close": prior,
            "predicted_close": prior,
            "predicted_direction": "down",
            "confidence": 0.3,
            "weights": {},
            "analyst_predictions": {},
            "rationale": "[backfill seed] random-walk baseline prediction.",
            "status": "pending",
            "actual_close": None,
            "seed": True,  # excluded from model metrics; kept as price-history context
        }
        row.update(score_row(row, actual))
        row["winning_strategy"] = None
        rows = upsert_ledger_row(rows, row)
        seeded += 1
    log(f"backfilled {seeded} seed rows.")
    return rows


# ------------------------------------------------------------------------------------- commit

def git_commit(target_date: str) -> bool:
    try:
        root = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        paths = [str(PROJECT_ROOT / p) for p in COMMIT_PATHS if (PROJECT_ROOT / p).exists()]
        subprocess.run(["git", "-C", root, "add", *paths], check=True)
        status = subprocess.run(["git", "-C", root, "status", "--porcelain", *paths],
                                capture_output=True, text=True)
        if not status.stdout.strip():
            log("nothing to commit.")
            return True
        subprocess.run(
            ["git", "-C", root, "commit", "-m", f"chore: daily AMZN run {target_date}"],
            check=True,
        )
        subprocess.run(["git", "-C", root, "pull", "--rebase"], check=False)
        subprocess.run(["git", "-C", root, "push", "origin", "HEAD"], check=True)
        log("committed and pushed results.")
        return True
    except subprocess.CalledProcessError as e:
        log(f"git commit/push failed: {e}")
        return False


# ---------------------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AMZN daily close predictor")
    parser.add_argument("--mode", default="daily",
                        choices=["daily", "score", "predict", "report", "backfill"])
    parser.add_argument("--date", default=None,
                        help="session YYYY-MM-DD (default: the next session in its forecast window)")
    parser.add_argument("--dry-run", action="store_true", help="no git commit; offline if no key")
    parser.add_argument("--no-commit", action="store_true")
    parser.add_argument("--allow-late", action="store_true",
                        help="write a prediction even after the session has opened (flagged)")
    args = parser.parse_args(argv)

    settings.ensure_dirs()
    rows = read_jsonl(settings.LEDGER_PATH)
    client = None if args.dry_run else get_client()
    now = datetime.now(timezone.utc)

    if args.mode == "report":
        report.generate()
        log("report regenerated.")
        return 0

    if args.mode == "backfill":
        rows = do_backfill(rows)
        write_jsonl(settings.LEDGER_PATH, rows)
        report.generate()
        return 0

    if args.mode in ("daily", "score", "predict"):
        if args.mode in ("daily", "score"):
            rows, _ = do_score(rows, client)
            write_jsonl(settings.LEDGER_PATH, rows)

        if args.mode in ("daily", "predict"):
            target = session.resolve_target(args.date, now)
            if target is None:
                log("a session is in progress (or its close is settling) — nothing to forecast.")
            elif not is_trading_day(target):
                log(f"{target} is not an NYSE trading day; nothing to forecast.")
            else:
                rows = forecast_or_refresh(rows, target, client, allow_late=args.allow_late)
                write_jsonl(settings.LEDGER_PATH, rows)

        target_label = (args.date or iso_today())
        metrics = report.generate()
        _emit_ci_summary(metrics)

        if not args.dry_run and not args.no_commit:
            committed = git_commit(target_label)
            # In CI a failed commit means the day's prediction is lost — fail loudly.
            if not committed and os.getenv("GITHUB_ACTIONS"):
                return 1
        return 0

    return 1


def _emit_ci_summary(metrics: dict) -> None:
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    r = metrics.get("rolling", {})
    text = (
        f"### {settings.SYMBOL} run — rolling metrics\n"
        f"- Scored days: {r.get('n', 0)}\n"
        f"- PASS rate (±1%): {r.get('pass_rate')}\n"
        f"- Directional accuracy: {r.get('directional_accuracy')}\n"
        f"- MAPE: {r.get('mape')} vs baseline {r.get('baseline_mape')} (edge {r.get('edge')})\n"
    )
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    log(text)


if __name__ == "__main__":
    sys.exit(main())
