"""Panel orchestrator — the multi-name experiment (src/panel/), one process, one commit.

Modes:
  daily     score finished sessions -> research the next session, or re-anchor it -> report
  score     score only
  predict   research / re-anchor only
  report    regenerate RESULTS_PANEL.md and results/panel_metrics.json

Same timing model as the AMZN arms (loop/session.py): the first run inside a session's forecast
window researches it, later pre-open runs only re-anchor, runs during a session score and report.
Live only: a past --date is refused, because the panel's anchors cannot be replayed honestly.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

from ..agents.analysts import get_client
from ..config import settings
from ..data.market_calendar import is_trading_day
from ..panel import report, runner
from ..utils import iso_today
from . import session
from .run_daily import git_commit

COMMIT_PATHS = ["data/panel.jsonl", "results/panel_metrics.json", "RESULTS_PANEL.md",
                "learnings/panel_state.json"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Panel: LLM vs the pre-open price, many names")
    parser.add_argument("--mode", default="daily", choices=["daily", "score", "predict", "report"])
    parser.add_argument("--date", default=None, help="session YYYY-MM-DD (default: the next one)")
    parser.add_argument("--dry-run", action="store_true", help="no git commit; offline if no key")
    parser.add_argument("--no-commit", action="store_true")
    args = parser.parse_args(argv)

    settings.ensure_dirs()
    now = datetime.now(timezone.utc)
    rows = runner.load_rows()

    if args.mode in ("daily", "score"):
        rows = runner.score(rows, now)
        runner.save_rows(rows)

    if args.mode in ("daily", "predict"):
        target = session.resolve_target(args.date, now)
        if target is None:
            runner.log("a session is in progress (or its close is settling) — nothing to forecast.")
        elif not is_trading_day(target):
            runner.log(f"{target} is not an NYSE trading day; nothing to forecast.")
        elif any(r.get("date") == target.isoformat() for r in rows):
            rows = runner.refresh(rows, target, now)
        else:
            client = None if args.dry_run else get_client()
            rows = runner.research(rows, target, client, now)
        runner.save_rows(rows)

    m = report.generate()
    runner.log(f"{m['n_sessions']} clean sessions; APE gain vs free baseline: "
               f"{m['cs']['ape_gain_vs_anchor']['decision']}.")

    if args.mode != "report" and not args.dry_run and not args.no_commit:
        label = args.date or iso_today()
        if not git_commit(label, COMMIT_PATHS, f"chore: daily panel run {label}") \
                and os.getenv("GITHUB_ACTIONS"):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
