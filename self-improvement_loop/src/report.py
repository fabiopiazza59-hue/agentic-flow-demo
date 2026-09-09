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
from .evals import integrity
from .evals.metrics import aggregate, ape
from .evals.paired import paired_effect
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


def gate_effect(rows: list[dict]) -> dict:
    """Paired day-level effect of the guardrail gates: ungated APE minus gated APE.

    Positive means the gates helped. Only rows carrying a `predicted_close_raw` counterfactual
    can contribute, so this fills in from the first run after that field shipped.
    """
    deltas, fired = [], 0
    for r in rows:
        raw, actual = r.get("predicted_close_raw"), r.get("actual_close")
        if raw is None or actual in (None, 0) or r.get("status") != "scored" or r.get("seed"):
            continue
        deltas.append(ape(float(raw), float(actual)) - float(r["ape"]))
        if r.get("gates_applied"):
            fired += 1
    effect = paired_effect(deltas)
    effect["n_gates_fired"] = fired
    return effect


def baseline_effect(rows: list[dict], window: int | None = None) -> dict:
    """Paired day-level advantage over the random-walk baseline, on pre-open rows only.

    This is the headline verdict's evidence: an `edge` of 1e-4 in a series whose daily APE swings
    by whole percent is not skill, it is noise with a sign. Canon rule 1 makes the day the
    inferential unit; the FIRM bar (interval excludes zero AND the sign test agrees) decides
    whether the dashboard is allowed to claim an edge at all.
    """
    scored = sorted(
        (r for r in rows
         if r.get("status") == "scored" and not r.get("seed") and not integrity.is_late(r)
         and r.get("ape") is not None and r.get("baseline_ape") is not None),
        key=lambda r: r.get("date", ""),
    )
    if window:
        scored = scored[-window:]
    return paired_effect([float(r["baseline_ape"]) - float(r["ape"]) for r in scored])


def build_metrics(rows: list[dict], scorecards: dict) -> dict:
    rolling = aggregate(rows, window=settings.ROLLING_WINDOW)
    alltime = aggregate(rows, window=None)
    # The same aggregates restricted to rows created before their session opened. Rows written
    # after the open saw part of the tape they forecast, so this is the honest record.
    rolling_pre_open = aggregate(rows, window=settings.ROLLING_WINDOW, pre_open_only=True)
    alltime_pre_open = aggregate(rows, window=None, pre_open_only=True)
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
        "rolling_pre_open": rolling_pre_open,
        "all_time_pre_open": alltime_pre_open,
        "n_late_rows": sum(1 for r in scored if not r.get("seed") and integrity.is_late(r)),
        "gate_effect": gate_effect(rows),
        "baseline_effect": baseline_effect(rows, window=settings.ROLLING_WINDOW),
        "scorecards": scorecards,
        "series": series,
        "pending": pending[-1] if pending else None,
        "updated_count": len(scored),
    }


def _render_integrity(metrics: dict) -> str:
    """A short, always-visible note on what the headline excludes and what the gates are worth."""
    late = metrics.get("n_late_rows", 0)
    g = metrics.get("gate_effect") or {}
    lines = ["### Integrity & mechanism", ""]
    if late:
        lines.append(
            f"- **{late} scored row(s) were created after their session opened** and are excluded "
            f"from the pre-open column. They saw part of the tape they forecast, so they cannot "
            f"carry the verdict. New post-open rows are refused (`--allow-late` to override)."
        )
    else:
        lines.append("- All scored rows were created before their session opened. ✅")
    if g.get("n"):
        ci = f"[{_pct(g.get('ci_low'))}, {_pct(g.get('ci_high'))}]"
        lines.append(
            f"- **Gate effect** (ungated − gated APE, paired over {g['n']} days): "
            f"{_pct(g.get('mean'))} 90% CI {ci}, gates helped on {g.get('wins', 0)}/"
            f"{g.get('n_decisive', 0)} decisive days (sign test p={g.get('sign_test_p')}). "
            f"Gates fired on {g.get('n_gates_fired', 0)} of them."
        )
    else:
        lines.append(
            "- **Gate effect**: not measurable yet — accrues from the first run that records an "
            "ungated counterfactual (`predicted_close_raw`) on each row."
        )
    return "\n".join(lines)


def render_results_md(metrics: dict, rows: list[dict]) -> str:
    r = metrics["rolling"]
    a = metrics["all_time"]
    # The verdict is read off pre-open rows only: a row written after the open saw part of the
    # tape it was forecasting, so including it would let lookahead carry the headline.
    p = metrics.get("rolling_pre_open") or r
    # An edge is only claimed when the paired day-level test clears the FIRM bar: the bootstrap
    # interval excludes zero AND the sign test agrees. A positive `edge` of 1e-4 is noise.
    eff = metrics.get("baseline_effect") or {}
    edge = p.get("edge")
    evidence = (f"paired over {eff.get('n', 0)} pre-open days: mean {_pct(eff.get('mean'))}, "
                f"90% CI [{_pct(eff.get('ci_low'))}, {_pct(eff.get('ci_high'))}], "
                f"beat baseline on {eff.get('wins', 0)}/{eff.get('n_decisive', 0)} decisive days, "
                f"sign test p={eff.get('sign_test_p')}")
    if edge is None or not eff.get("n"):
        verdict = "⏳ **Not enough scored days yet** to judge edge vs the random-walk baseline."
    elif eff.get("significant"):
        verdict = (f"✅ **Edge confirmed** — rolling MAPE {_pct(p['mape'])} beats the random-walk "
                   f"baseline {_pct(p['baseline_mape'])} ({evidence}).")
    elif edge > 0:
        verdict = (f"❌ **No edge yet** — rolling MAPE {_pct(p['mape'])} is nominally ahead of the "
                   f"random-walk baseline {_pct(p['baseline_mape'])} by {_pct(edge)}, but the "
                   f"difference is not distinguishable from noise ({evidence}).")
    else:
        verdict = (f"❌ **No edge yet** — rolling MAPE {_pct(p['mape'])} does not beat the "
                   f"random-walk baseline {_pct(p['baseline_mape'])} ({evidence}). Keep learning.")

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
        "| Metric | Rolling (pre-open) | Rolling (all rows) | All-time |",
        "|---|---|---|---|",
        f"| Scored days | {p.get('n', 0)} | {r.get('n', 0)} | {a.get('n_all_time', 0)} |",
        f"| PASS rate (±1%) | {_pct(p.get('pass_rate'))} | {_pct(r.get('pass_rate'))} | {_pct(a.get('pass_rate'))} |",
        f"| Directional accuracy | {_pct(p.get('directional_accuracy'))} | {_pct(r.get('directional_accuracy'))} | {_pct(a.get('directional_accuracy'))} |",
        f"| MAPE | {_pct(p.get('mape'))} | {_pct(r.get('mape'))} | {_pct(a.get('mape'))} |",
        f"| Baseline MAPE (random walk) | {_pct(p.get('baseline_mape'))} | {_pct(r.get('baseline_mape'))} | {_pct(a.get('baseline_mape'))} |",
        f"| Edge (baseline − model) | {_pct(p.get('edge'))} | {_pct(r.get('edge'))} | {_pct(a.get('edge'))} |",
        f"| Brier (confidence calib.) | {_num(p.get('brier'))} | {_num(r.get('brier'))} | {_num(a.get('brier'))} |",
        "",
        _render_integrity(metrics),
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
