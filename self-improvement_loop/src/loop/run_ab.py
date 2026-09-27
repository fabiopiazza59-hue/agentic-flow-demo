"""A/B orchestrator — runs arm A (ensemble) and arm B (raven prior+pulse) on one shared pre-open
snapshot, then reports the paired comparison. One process, one commit.

Modes mirror run_daily:
  daily     score finished sessions (both arms) -> research the next session from one snapshot,
            or refresh both pending rows onto a fresher pre-open anchor -> report -> commit
  score     score both arms only
  predict   research / refresh only
  report    regenerate all artifacts incl. the A/B comparison
  backfill  arm A seed backfill (B intentionally starts clean; pairing handles the offset)

Every scheduled run calls `daily`: the first run inside a session's forecast window researches
it, later pre-open runs only re-anchor (see loop/session.py), and runs during a session score and
report. Both arms predict or neither does — a paired test needs both sides of every day.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime, timezone

from .. import report, report_ab
from ..config import settings
from ..data.market_calendar import is_trading_day
from ..evals import integrity
from ..utils import iso_today, read_jsonl, write_jsonl
from ..variant_b.runner import do_predict_b, do_score_b
from . import session
from .run_daily import (_emit_ci_summary, do_backfill, do_predict, do_score, get_client,
                        git_commit, refresh_anchor)


def log(msg: str) -> None:
    print(f"[run_ab] {msg}", flush=True)


def forecast_both(rows_a: list[dict], rows_b: list[dict], target: date, client,
                  allow_late: bool = False) -> tuple[list[dict], list[dict]]:
    """Research `target` for both arms from one snapshot, or re-anchor both pending rows."""
    iso = target.isoformat()
    if not any(r.get("date") == iso for r in rows_a):
        live = iso >= iso_today()
        if settings.ENFORCE_PRE_OPEN and live and not allow_late and not integrity.is_pre_open_now(
                iso, margin_minutes=settings.PREDICT_CUTOFF_MINUTES):
            log(f"{iso}'s research window has closed (open in < {settings.PREDICT_CUTOFF_MINUTES} "
                f"min or already open) — no forecast for this session.")
            return rows_a, rows_b
        snap = session.take_snapshot(target)
        if snap is None:
            log("no data provider has the previous session's close yet — retrying next run.")
            return rows_a, rows_b
        features = snap["features"]
        log(f"shared snapshot for {iso}: prev_close {features['prev_close']}, anchor "
            f"{snap['anchor']['price']} ({snap['anchor']['source']}), gap "
            f"{features.get('premarket_gap_pct')}, σ {features.get('sigma_ewma')}.")
        rows_a, row_a = do_predict(rows_a, target, client, features=features,
                                   allow_late=allow_late)
        if row_a is None:
            log("arm A declined to predict; skipping arm B to keep the pairing clean.")
            return rows_a, rows_b
        rows_b, _ = do_predict_b(rows_b, target, client, features, snap["history"])
        return rows_a, rows_b

    if not integrity.is_pre_open_now(iso, margin_minutes=settings.REFRESH_CUTOFF_MINUTES):
        log(f"{iso} is already forecast and past its refresh window; nothing to do.")
        return rows_a, rows_b
    snap = session.take_snapshot(target)
    if snap is None:
        return rows_a, rows_b
    rows_a, changed_a = refresh_anchor(rows_a, target, snap["anchor"])
    rows_b, changed_b = refresh_anchor(rows_b, target, snap["anchor"])
    if changed_a or changed_b:
        log(f"{iso}: re-anchored to {snap['anchor']['price']} ({snap['anchor']['source']} @ "
            f"{snap['anchor']['time']}) — A {'✓' if changed_a else '–'}, B {'✓' if changed_b else '–'}.")
    else:
        log(f"{iso}: no fresher pre-open trade than the current anchor.")
    return rows_a, rows_b


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AMZN A/B daily predictor (arms A and B)")
    parser.add_argument("--mode", default="daily",
                        choices=["daily", "score", "predict", "report", "backfill"])
    parser.add_argument("--date", default=None,
                        help="session YYYY-MM-DD (default: the next session in its forecast window)")
    parser.add_argument("--dry-run", action="store_true", help="no git commit; offline if no key")
    parser.add_argument("--no-commit", action="store_true")
    parser.add_argument("--allow-late", action="store_true",
                        help="write predictions even after the session has opened (flagged)")
    args = parser.parse_args(argv)

    settings.ensure_dirs()
    client = None if args.dry_run else get_client()
    now = datetime.now(timezone.utc)

    if args.mode == "report":
        report.generate()
        report_ab.generate()
        log("reports regenerated (A + A/B comparison).")
        return 0

    if args.mode == "backfill":
        rows_a = do_backfill(read_jsonl(settings.LEDGER_PATH))
        write_jsonl(settings.LEDGER_PATH, rows_a)
        report.generate()
        report_ab.generate()
        return 0

    rows_a = read_jsonl(settings.LEDGER_PATH)
    rows_b = read_jsonl(settings.LEDGER_B_PATH)

    if args.mode in ("daily", "score"):
        rows_a, _ = do_score(rows_a, client)
        write_jsonl(settings.LEDGER_PATH, rows_a)
        rows_b, _ = do_score_b(rows_b)
        write_jsonl(settings.LEDGER_B_PATH, rows_b)

    if args.mode in ("daily", "predict"):
        target = session.resolve_target(args.date, now)
        if target is None:
            log("a session is in progress (or its close is settling) — nothing to forecast.")
        elif not is_trading_day(target):
            log(f"{target} is not an NYSE trading day; nothing to forecast.")
        else:
            rows_a, rows_b = forecast_both(rows_a, rows_b, target, client,
                                           allow_late=args.allow_late)
            write_jsonl(settings.LEDGER_PATH, rows_a)
            write_jsonl(settings.LEDGER_B_PATH, rows_b)

    metrics = report.generate()
    _emit_ci_summary(metrics)
    cmp = report_ab.generate()
    log(cmp["verdict"])
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(f"### A/B\n{cmp['verdict']}\n")

    if not args.dry_run and not args.no_commit:
        committed = git_commit(args.date or iso_today())
        if not committed and os.getenv("GITHUB_ACTIONS"):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
