"""A/B orchestrator — runs arm A (ensemble+gates) and arm B (raven prior+pulse) on one
shared pre-open snapshot, then reports the paired comparison. One process, one commit.

Modes mirror run_daily:
  daily     guard -> score both arms -> predict both arms (same snapshot) -> report -> commit
  score     score both arms only
  predict   predict both arms only
  report    regenerate all artifacts incl. the A/B comparison
  backfill  arm A seed backfill (B intentionally starts clean; pairing handles the offset)
"""

from __future__ import annotations

import argparse
import os
import sys

from .. import report, report_ab
from ..config import settings
from ..data.market_calendar import is_trading_day
from ..data.providers import get_history, get_quote
from ..features import build_features
from ..utils import iso_today, read_jsonl, to_date, write_jsonl
from ..variant_b.runner import do_predict_b, do_score_b
from .run_daily import _emit_ci_summary, do_backfill, do_predict, do_score, get_client, git_commit


def log(msg: str) -> None:
    print(f"[run_ab] {msg}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AMZN A/B daily predictor (arms A and B)")
    parser.add_argument("--mode", default="daily",
                        choices=["daily", "score", "predict", "report", "backfill"])
    parser.add_argument("--date", default=None, help="target session YYYY-MM-DD (default today UTC)")
    parser.add_argument("--dry-run", action="store_true", help="no git commit; offline if no key")
    parser.add_argument("--no-commit", action="store_true")
    parser.add_argument("--allow-late", action="store_true",
                        help="write predictions even after the session has opened (flagged)")
    args = parser.parse_args(argv)

    settings.ensure_dirs()
    target_date = to_date(args.date) if args.date else to_date(iso_today())
    client = None if args.dry_run else get_client()

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

    if args.mode == "daily" and not is_trading_day(target_date):
        log(f"{target_date} is not an NYSE trading day; exiting cleanly.")
        return 0

    rows_a = read_jsonl(settings.LEDGER_PATH)
    rows_b = read_jsonl(settings.LEDGER_B_PATH)

    if args.mode in ("daily", "score"):
        rows_a, _ = do_score(rows_a, client)
        write_jsonl(settings.LEDGER_PATH, rows_a)
        rows_b, _ = do_score_b(rows_b)
        write_jsonl(settings.LEDGER_B_PATH, rows_b)

    if args.mode in ("daily", "predict"):
        # One snapshot for both arms — this is what makes the A/B comparison paired.
        history = get_history(settings.SYMBOL, settings.LOOKBACK_DAYS)
        quote = get_quote(settings.SYMBOL)
        features = build_features(history, quote)
        log(f"shared snapshot: prev_close {features['prev_close']}, "
            f"gap {features.get('premarket_gap_pct')}, quote via {features.get('quote_source')}.")

        rows_a, row_a = do_predict(rows_a, target_date, client, features=features,
                                   allow_late=args.allow_late)
        write_jsonl(settings.LEDGER_PATH, rows_a)
        # Both arms predict or neither does — a paired test needs both sides of every day.
        if row_a is None:
            log("arm A declined to predict (post-open); skipping arm B to keep the pairing clean.")
        else:
            rows_b, _ = do_predict_b(rows_b, target_date, client, features, history)
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
        committed = git_commit(target_date.isoformat())
        if not committed and os.getenv("GITHUB_ACTIONS"):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
