"""Daily orchestrator for the AMZN close predictor.

Modes:
  daily     guard -> score(T-1) -> reflect -> predict(T) -> report -> commit   (default)
  score     score the most recent pending prediction against its actual close
  predict   generate today's prediction only
  report    regenerate results/* artifacts from the ledger
  backfill  seed the ledger with recent scored days from history (baseline-only, no API)

Flags:
  --date YYYY-MM-DD   override the target session (default: today, UTC)
  --dry-run           no git commit; uses offline stubs if no ANTHROPIC_API_KEY
  --no-commit         run fully but skip the git commit step
  --allow-late        write a prediction even after the session opened (row is flagged)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import date

from ..agents.analysts import get_client, run_analysts
from ..agents.meta_judge import synthesize
from ..agents.reflector import diagnose_failures, reflect
from ..config import PROJECT_ROOT, settings
from ..data.market_calendar import is_trading_day, last_n_trading_days, previous_trading_day
from ..data.providers import get_actual_close, get_history, get_quote
from ..evals import integrity
from ..evals.gates import apply_gates
from ..evals.metrics import score_row
from ..evals.scorecard import load_scorecards, save_scorecards, update_scorecards
from ..features import build_features
from ..utils import iso_today, now_iso, read_jsonl, to_date, upsert_ledger_row, write_jsonl
from .. import report

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


def do_score(rows: list[dict], client) -> tuple[list[dict], dict | None]:
    """Score the most recent unscored prediction whose session has completed."""
    pending = [r for r in rows if r.get("status") == "pending"]
    if not pending:
        log("no pending predictions to score.")
        return rows, None
    target = sorted(pending, key=lambda r: r["date"])[-1]
    actual = get_actual_close(settings.SYMBOL, target["date"])
    if actual is None:
        log(f"actual close for {target['date']} not available yet; skipping score.")
        return rows, None

    fields = score_row(target, actual)
    target.update(fields)
    target["winning_strategy"] = _closest_strategy(target.get("analyst_predictions", {}), actual)
    target["scored_at"] = now_iso()

    scorecards = load_scorecards()
    scorecards = update_scorecards(scorecards, target.get("analyst_predictions", {}), actual)
    save_scorecards(scorecards)

    # reflect -> post-mortem + bounded STRATEGY.md note + (on FAIL) failure analysis
    reflection = reflect(target, scorecards, client=client)
    _write_learning(target["date"], reflection)

    rows = upsert_ledger_row(rows, target)

    # Concentrated failure log + rolling "what's not working" self-diagnosis.
    if target.get("pass") is False:
        _append_failure_log(target, reflection)
    _refresh_whats_not_working(rows, scorecards, client)

    log(f"scored {target['date']}: predicted {target['predicted_close']} vs actual {actual} "
        f"(APE {target['ape']*100:.2f}%, {'PASS' if target['pass'] else 'FAIL'}).")
    return rows, target


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

def do_predict(rows: list[dict], target_date: date, client,
               features: dict | None = None,
               allow_late: bool = False) -> tuple[list[dict], dict | None]:
    """Predict today's close. Pass a pre-built `features` snapshot for a fair A/B run.

    Refuses to write a row for *today's* session once that session has opened: a forecast made
    after the open has seen part of the tape it is forecasting, and 26 of the first 60 rows in
    this ledger were written that way (two of them after the close). Pass `allow_late` to
    override deliberately.

    A past-dated `--date` is a replay, not a live forecast, so the guard stays out of its way;
    such rows are still stamped `late_minutes` and so are still excluded from the pre-open
    headline by `aggregate(pre_open_only=True)`.
    """
    targets_today = target_date.isoformat() == iso_today()
    if settings.ENFORCE_PRE_OPEN and targets_today and not allow_late \
            and not integrity.is_pre_open_now(target_date.isoformat()):
        log(f"{target_date} has already opened — refusing to write a post-open 'prediction'. "
            f"Use --allow-late to override (the row is then flagged late_minutes).")
        return rows, None

    if features is None:
        history = get_history(settings.SYMBOL, settings.LOOKBACK_DAYS)
        quote = get_quote(settings.SYMBOL)
        features = build_features(history, quote)
    prior_close = features["prev_close"]

    scorecards = load_scorecards()
    strategy_md = settings.STRATEGY_PATH.read_text(encoding="utf-8") if settings.STRATEGY_PATH.exists() else ""
    recent_learnings = _recent_learnings(settings.LEARNINGS_CONTEXT_N)
    wnw_path = _whats_not_working_path()
    whats_not_working = wnw_path.read_text(encoding="utf-8") if wnw_path.exists() else ""

    analyst_predictions = run_analysts(
        features, client=client,
        context={"scorecards": scorecards, "whats_not_working": whats_not_working},
    )

    final = synthesize(analyst_predictions, scorecards, strategy_md, recent_learnings,
                       features, client=client, whats_not_working=whats_not_working)
    final, gates = apply_gates(final, analyst_predictions, rows, features)

    row = {
        "date": target_date.isoformat(),
        "created_at": now_iso(),
        "prior_close": prior_close,
        "predicted_close": final["predicted_close"],
        "predicted_close_raw": final.get("predicted_close_raw"),
        "predicted_direction": final["direction"],
        "confidence": final["confidence"],
        "weights": final.get("weights", {}),
        "analyst_predictions": analyst_predictions,
        "rationale": final.get("rationale", ""),
        "gates_applied": gates,
        "quote_source": features.get("quote_source"),
        "status": "pending",
        "actual_close": None,
    }
    integrity.annotate(row)
    rows = upsert_ledger_row(rows, row)
    log(f"predicted {target_date.isoformat()}: close ≈ {row['predicted_close']} "
        f"({row['predicted_direction']}, conf {row['confidence']}) from {len(analyst_predictions)} analysts"
        f"{'; gates: ' + ', '.join(gates) if gates else ''}.")
    return rows, row


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
    parser.add_argument("--date", default=None, help="target session YYYY-MM-DD (default today UTC)")
    parser.add_argument("--dry-run", action="store_true", help="no git commit; offline if no key")
    parser.add_argument("--no-commit", action="store_true")
    parser.add_argument("--allow-late", action="store_true",
                        help="write a prediction even after the session has opened (flagged)")
    args = parser.parse_args(argv)

    settings.ensure_dirs()
    target_date = to_date(args.date) if args.date else to_date(iso_today())
    rows = read_jsonl(settings.LEDGER_PATH)
    client = None if args.dry_run else get_client()

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
        if args.mode == "daily" and not is_trading_day(target_date):
            log(f"{target_date} is not an NYSE trading day; exiting cleanly.")
            return 0

        if args.mode in ("daily", "score"):
            rows, _ = do_score(rows, client)
            write_jsonl(settings.LEDGER_PATH, rows)

        if args.mode in ("daily", "predict"):
            rows, _ = do_predict(rows, target_date, client, allow_late=args.allow_late)
            write_jsonl(settings.LEDGER_PATH, rows)

        metrics = report.generate()
        _emit_ci_summary(metrics)

        if not args.dry_run and not args.no_commit:
            committed = git_commit(target_date.isoformat())
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
