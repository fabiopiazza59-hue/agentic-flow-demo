"""Reporting — regenerate all experiment-tracking artifacts from the ledger.

Outputs (all under the project, committed each run):
  results/results.csv      flat, spreadsheet-friendly export
  results/metrics.json     rolling + all-time aggregates, per-strategy scorecards, time series
  RESULTS.md               human dashboard rendered on the repo
  results/site/data.json   data feed for the GitHub Pages dashboard
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import settings
from .evals.metrics import aggregate
from .evals.scorecard import load_scorecards
from .utils import read_jsonl

CSV_FIELDS = [
    "date", "prior_close", "predicted_close", "actual_close",
    "predicted_direction", "directional_hit", "ape", "pass",
    "baseline_ape", "beats_baseline", "confidence", "winning_strategy", "status",
]


def _pct(x: float | None) -> str:
    return f"{x * 100:.2f}%" if isinstance(x, (int, float)) else "—"


def _num(x: float | None) -> str:
    return f"{x:.2f}" if isinstance(x, (int, float)) else "—"


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in sorted(rows, key=lambda r: r.get("date", "")):
            writer.writerow({k: r.get(k) for k in CSV_FIELDS})


def build_metrics(rows: list[dict], scorecards: dict) -> dict:
    rolling = aggregate(rows, window=settings.ROLLING_WINDOW)
    alltime = aggregate(rows, window=None)
    scored = [r for r in rows if r.get("status") == "scored"]
    series = [
        {
            "date": r["date"],
            "predicted_close": r.get("predicted_close"),
            "actual_close": r.get("actual_close"),
            "prior_close": r.get("prior_close"),
            "ape": r.get("ape"),
            "baseline_ape": r.get("baseline_ape"),
            "pass": r.get("pass"),
        }
        for r in sorted(scored, key=lambda r: r.get("date", ""))
    ]
    pending = [r for r in rows if r.get("status") == "pending"]
    return {
        "symbol": settings.SYMBOL,
        "pass_threshold": settings.PASS_THRESHOLD,
        "rolling_window": settings.ROLLING_WINDOW,
        "rolling": rolling,
        "all_time": alltime,
        "scorecards": scorecards,
        "series": series,
        "pending": pending[-1] if pending else None,
        "updated_count": len(scored),
    }


def render_results_md(metrics: dict, rows: list[dict]) -> str:
    r = metrics["rolling"]
    a = metrics["all_time"]
    edge = r.get("edge")
    if edge is None:
        verdict = "⏳ **Not enough scored days yet** to judge edge vs the random-walk baseline."
    elif edge > 0:
        verdict = (f"✅ **Edge confirmed** — rolling MAPE {_pct(r['mape'])} beats the random-walk "
                   f"baseline {_pct(r['baseline_mape'])} by {_pct(edge)}.")
    else:
        verdict = (f"❌ **No edge yet** — rolling MAPE {_pct(r['mape'])} does not beat the "
                   f"random-walk baseline {_pct(r['baseline_mape'])}. Keep learning.")

    pending = metrics.get("pending")
    pending_line = ""
    if pending:
        pending_line = (
            f"\n**Today's open prediction ({pending['date']}):** "
            f"close ≈ **{_num(pending.get('predicted_close'))}** "
            f"({pending.get('predicted_direction')}, confidence "
            f"{_num((pending.get('confidence') or 0) * 100)}%) vs prior close "
            f"{_num(pending.get('prior_close'))}.\n"
        )

    lines = [
        f"# 📈 {metrics['symbol']} Daily Close Predictor — Results",
        "",
        "_Auto-generated each trading day. PASS = predicted close within ±1% of actual._",
        "",
        f"## Verdict",
        verdict,
        pending_line,
        "## Rolling metrics "
        f"(last {min(r.get('n', 0), metrics['rolling_window'])} scored days)",
        "",
        "| Metric | Rolling | All-time |",
        "|---|---|---|",
        f"| Scored days | {r.get('n', 0)} | {a.get('n_all_time', 0)} |",
        f"| PASS rate (±1%) | {_pct(r.get('pass_rate'))} | {_pct(a.get('pass_rate'))} |",
        f"| Directional accuracy | {_pct(r.get('directional_accuracy'))} | {_pct(a.get('directional_accuracy'))} |",
        f"| MAPE | {_pct(r.get('mape'))} | {_pct(a.get('mape'))} |",
        f"| Baseline MAPE (random walk) | {_pct(r.get('baseline_mape'))} | {_pct(a.get('baseline_mape'))} |",
        f"| Edge (baseline − model) | {_pct(r.get('edge'))} | {_pct(a.get('edge'))} |",
        f"| Brier (confidence calib.) | {_num(r.get('brier'))} | {_num(a.get('brier'))} |",
        "",
        "## Per-strategy scorecards",
        "",
        "| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |",
        "|---|---|---|---|---|",
    ]
    for name, c in sorted(metrics["scorecards"].items(),
                          key=lambda kv: kv[1].get("mape", 9e9)):
        lines.append(
            f"| {name} | {c.get('n', 0)} | {_pct(c.get('hit_rate'))} | "
            f"{_pct(c.get('mape'))} | {_num(c.get('weight_hint'))} |"
        )

    lines += ["", "## Last 20 scored model predictions",
              "_(backfill seed rows are excluded from metrics and this table; they appear only as "
              "price-history context on the dashboard chart)_", "",
              "| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest |",
              "|---|---|---|---|---|---|---|---|"]
    scored = [r for r in rows if r.get("status") == "scored" and not r.get("seed")]
    if not scored:
        lines.append("| _no model predictions scored yet_ | | | | | | | |")
    for row in sorted(scored, key=lambda r: r.get("date", ""))[-20:][::-1]:
        lines.append(
            f"| {row['date']} | {_num(row.get('predicted_close'))} | {_num(row.get('actual_close'))} "
            f"| {_pct(row.get('ape'))} | {'✅' if row.get('pass') else '❌'} "
            f"| {'✅' if row.get('directional_hit') else '❌'} "
            f"| {'✅' if row.get('beats_baseline') else '❌'} | {row.get('winning_strategy') or '—'} |"
        )
    lines += ["", "_This is a research experiment, not financial advice._", ""]
    return "\n".join(lines)


def generate(ledger_path: Path | None = None) -> dict:
    """Regenerate all artifacts. Returns the metrics dict."""
    settings.ensure_dirs()
    rows = read_jsonl(ledger_path or settings.LEDGER_PATH)
    scorecards = load_scorecards()
    metrics = build_metrics(rows, scorecards)

    write_csv(rows, settings.RESULTS_CSV)
    settings.METRICS_JSON.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    settings.RESULTS_MD.write_text(render_results_md(metrics, rows), encoding="utf-8")
    settings.SITE_DATA.parent.mkdir(parents=True, exist_ok=True)
    settings.SITE_DATA.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    return metrics
