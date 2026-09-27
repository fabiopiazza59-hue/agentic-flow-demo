"""Reporting — regenerate all experiment-tracking artifacts from the ledger.

Outputs (all under the project, committed each run):
  results/results.csv      flat, spreadsheet-friendly export
  results/metrics.json     rolling + all-time aggregates, per-strategy scorecards, time series
  RESULTS.md               human dashboard rendered on the repo
  results/site/data.json   data feed for the GitHub Pages dashboard
  learnings/lab_state.json the lab's current mechanism settings (read by the next forecast)

Every verdict reads pre-open forecasts only, all-time, through an anytime-valid 95% confidence
sequence (evals/sequential.py) — valid although this page is regenerated and re-read daily.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import settings
from .data.market_calendar import sessions_between
from .evals import integrity, lab
from .evals import probabilistic as pb
from .evals.metrics import aggregate, ape
from .evals.paired import paired_effect
from .evals.scorecard import load_scorecards
from .evals.sequential import confidence_sequence
from .utils import read_jsonl

CSV_FIELDS = [
    "date", "prior_close", "predicted_close", "actual_close",
    "predicted_direction", "directional_hit", "ape", "pass",
    "baseline_ape", "beats_baseline", "confidence", "winning_strategy", "status",
    "anchor", "anchor_source", "anchor_ape", "p_up", "late_minutes", "pipeline_version",
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


def _clean_scored(rows: list[dict]) -> list[dict]:
    return sorted((r for r in rows if r.get("status") == "scored" and not r.get("seed")
                   and not integrity.is_late(r) and r.get("ape") is not None
                   and r.get("baseline_ape") is not None), key=lambda r: r["date"])


def value_added(rows: list[dict], evals: list[dict]) -> dict:
    """Anytime-valid 95% CSs on paired daily gains over the free baselines (pre-open rows only).

    Positive = the model did better. APE gains are in fractions of price; CRPS gains likewise;
    direction gains are Brier-score points of P(close > prior close).
    """
    clean = _clean_scored(rows)
    free = [float(r.get("anchor_ape") if r.get("anchor_ape") is not None else r["baseline_ape"])
            - float(r["ape"]) for r in clean]
    rw = [float(r["baseline_ape"]) - float(r["ape"]) for r in clean]
    e = [x for x in evals if x["clean"]]
    return {
        "vs_free_baseline": confidence_sequence(free),
        "vs_random_walk": confidence_sequence(rw),
        "crps_vs_random_walk": confidence_sequence(
            [x["random_walk"]["crps"] - x["model"]["crps"] for x in e]),
        "crps_vs_free_baseline": confidence_sequence(
            [x["anchor"]["crps"] - x["model"]["crps"] for x in e]),
        "direction_vs_random_walk": confidence_sequence(
            [x["random_walk"]["brier_up"] - x["model"]["brier_up"] for x in e]),
    }


def coverage(rows: list[dict], last_n: int = 20) -> dict:
    """How many of the most recent sessions got a genuine pre-open forecast."""
    live = sorted(r["date"] for r in rows if not r.get("seed") and r.get("created_at"))
    if not live:
        return {"n_sessions": 0, "n_pre_open": 0}
    sessions = [d.isoformat() for d in sessions_between(live[0], live[-1])][-last_n:]
    by_date = {r["date"]: r for r in rows if not r.get("seed")}
    pre_open = sum(1 for d in sessions if d in by_date and not integrity.is_late(by_date[d]))
    return {"n_sessions": len(sessions), "n_pre_open": pre_open, "since": sessions[0]}


def build_metrics(rows: list[dict], scorecards: dict) -> dict:
    rolling = aggregate(rows, window=settings.ROLLING_WINDOW)
    alltime = aggregate(rows, window=None)
    # The same aggregates restricted to rows created before their session opened. Rows written
    # after the open saw part of the tape they forecast, so this is the honest record.
    rolling_pre_open = aggregate(rows, window=settings.ROLLING_WINDOW, pre_open_only=True)
    alltime_pre_open = aggregate(rows, window=None, pre_open_only=True)
    scored = [r for r in rows if r.get("status") == "scored"]
    evals = pb.evaluate(rows, pb._returns_by_date(rows), is_clean=lambda r: not integrity.is_late(r))
    series = [
        {
            "date": r["date"],
            "predicted_close": r.get("predicted_close"),
            "actual_close": r.get("actual_close"),
            "prior_close": r.get("prior_close"),
            "anchor": r.get("anchor"),
            "ape": r.get("ape"),
            "baseline_ape": r.get("baseline_ape"),
            "pass": r.get("pass"),
            "pre_open": not integrity.is_late(r),
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
        "coverage": coverage(rows),
        "gate_effect": gate_effect(rows),
        "baseline_effect": baseline_effect(rows),
        "probabilistic": pb.summarize(evals),
        "value_added": value_added(rows, evals),
        "lab": lab.evaluate(rows),
        "scorecards": scorecards,
        "series": series,
        "pending": pending[-1] if pending else None,
        "updated_count": len(scored),
    }


def _cs(cs: dict, scale: float = 100.0, unit: str = "%") -> str:
    if not cs or cs.get("lo") is None:
        return f"n={cs.get('n', 0) if cs else 0} (too few)"
    return f"[{cs['lo'] * scale:+.3f}{unit}, {cs['hi'] * scale:+.3f}{unit}]"


def _cs_verdict(cs: dict) -> str:
    return {"positive": "✅ better", "negative": "❌ worse", "undecided": "≈ not distinguishable",
            "insufficient": "⏳ too few days"}.get(cs.get("decision", "insufficient"), "—")


def render_verdict(metrics: dict) -> str:
    """Headline: does the model beat the free baseline, all-time, on pre-open forecasts?"""
    p = metrics.get("all_time_pre_open") or metrics["all_time"]
    va = (metrics.get("value_added") or {}).get("vs_free_baseline") or {}
    fx = metrics.get("baseline_effect") or {}
    n = va.get("n", 0)
    evidence = (f"paired over {n} pre-open days: mean {_pct(va.get('mean'))}, anytime-valid 95% "
                f"CS {_cs(va)}; fixed-sample 90% CI [{_pct(fx.get('ci_low'))}, "
                f"{_pct(fx.get('ci_high'))}], sign test p={fx.get('sign_test_p')}")
    mape, base, edge = p.get("mape"), p.get("anchor_mape"), p.get("anchor_edge")
    decision = va.get("decision", "insufficient")
    if not n or decision == "insufficient" or edge is None:
        return "⏳ **Not enough scored days yet** to judge the model against the free baseline."
    if decision == "positive":
        return (f"✅ **Edge confirmed** — all-time MAPE {_pct(mape)} beats the free baseline "
                f"{_pct(base)} ({evidence}).")
    if decision == "negative":
        return (f"❌ **No edge yet — worse than the free baseline**: all-time MAPE {_pct(mape)} vs "
                f"{_pct(base)} ({evidence}).")
    if edge > 0:
        return (f"❌ **No edge yet** — all-time MAPE {_pct(mape)} is nominally ahead of the free "
                f"baseline {_pct(base)} by {_pct(edge)}, but the difference is not "
                f"distinguishable from noise ({evidence}).")
    return (f"❌ **No edge yet** — all-time MAPE {_pct(mape)} does not beat the free baseline "
            f"{_pct(base)} ({evidence}). Keep learning.")


def _render_pending(pending: dict | None) -> str:
    if not pending:
        return ""
    q = pending.get("quantiles")
    band = (f", 80% interval {_num(q[1])}–{_num(q[17])}" if isinstance(q, list) and len(q) == 19
            else "")
    p_up = pending.get("p_up")
    anchor = (f"; anchored to {_num(pending.get('anchor'))} ({pending.get('anchor_source')}"
              f"{', live' if pending.get('anchor_live') else ''}, refreshed "
              f"{pending.get('anchor_updates', 0)}×)" if pending.get("anchor") else "")
    late = " ⚠️ written after the open — excluded from the verdict" if integrity.is_late(pending) else ""
    label = ("Awaiting score" if integrity.session_has_closed(pending["date"], None, 0)
             else "Next forecast")
    return (f"\n**{label} ({pending['date']}):** close ≈ "
            f"**{_num(pending.get('predicted_close'))}** ({pending.get('predicted_direction')}"
            f"{f', P(up) {p_up:.0%}' if isinstance(p_up, (int, float)) else ''}{band}) vs prior "
            f"close {_num(pending.get('prior_close'))}{anchor}.{late}\n")


def _render_scoreboard(metrics: dict) -> list[str]:
    pr = metrics.get("probabilistic") or {}
    p = metrics.get("all_time_pre_open") or {}
    va = metrics.get("value_added") or {}
    if not pr.get("n"):
        return []
    m, rw, an = pr["model"], pr["random_walk"], pr["anchor"]
    return [
        "## Scoreboard — all-time, pre-open forecasts only",
        "",
        "_Every forecaster is scored as a calibrated predictive distribution (CRPS: lower is better; "
        "coverage should match its nominal level). The **free baseline** is the latest pre-open "
        "trade the forecast was anchored to — the prior close when there was none._",
        "",
        "| Metric | Model (arm A) | Random walk (prior close) | Free baseline (pre-open price) |",
        "|---|---|---|---|",
        f"| Days | {pr['n']} | {pr['n']} | {pr['n']} |",
        f"| MAPE | {_pct(p.get('mape'))} | {_pct(p.get('baseline_mape'))} | {_pct(p.get('anchor_mape'))} |",
        f"| CRPS | {_pct(m['crps'])} | {_pct(rw['crps'])} | {_pct(an['crps'])} |",
        f"| 80% interval coverage | {_pct(m['cover80'])} | {_pct(rw['cover80'])} | {_pct(an['cover80'])} |",
        f"| 50% interval coverage | {_pct(m['cover50'])} | {_pct(rw['cover50'])} | {_pct(an['cover50'])} |",
        f"| Brier of P(up) | {_num3(m['brier_up'])} | {_num3(rw['brier_up'])} | {_num3(an['brier_up'])} |",
        "",
        "**Does the model add value?** Paired daily gains, anytime-valid 95% confidence sequences "
        "(valid although this page is re-read every day):",
        "",
        "| Comparison | Mean daily gain | 95% CS | Verdict |",
        "|---|---|---|---|",
        f"| APE vs free baseline | {_pct(va.get('vs_free_baseline', {}).get('mean'))} | {_cs(va.get('vs_free_baseline'))} | {_cs_verdict(va.get('vs_free_baseline', {}))} |",
        f"| APE vs random walk | {_pct(va.get('vs_random_walk', {}).get('mean'))} | {_cs(va.get('vs_random_walk'))} | {_cs_verdict(va.get('vs_random_walk', {}))} |",
        f"| CRPS vs free baseline | {_pct(va.get('crps_vs_free_baseline', {}).get('mean'))} | {_cs(va.get('crps_vs_free_baseline'))} | {_cs_verdict(va.get('crps_vs_free_baseline', {}))} |",
        f"| CRPS vs random walk | {_pct(va.get('crps_vs_random_walk', {}).get('mean'))} | {_cs(va.get('crps_vs_random_walk'))} | {_cs_verdict(va.get('crps_vs_random_walk', {}))} |",
        f"| Direction (Brier of P(up)) vs random walk | {_num3(va.get('direction_vs_random_walk', {}).get('mean'))} | {_cs(va.get('direction_vs_random_walk'), 1.0, '')} | {_cs_verdict(va.get('direction_vs_random_walk', {}))} |",
        "",
    ]


def _num3(x) -> str:
    return f"{x:.3f}" if isinstance(x, (int, float)) else "—"


def _render_lab(metrics: dict) -> list[str]:
    lb = metrics.get("lab") or {}
    if not lb:
        return []
    champ = lb.get("champion", lab.REFERENCE)
    k = len(lb.get("challengers", {}))
    status = ("reference — no challenger has proven better yet" if champ == lab.REFERENCE
              else "promoted: proven better than the equal-weight mean")
    lines = [
        "## Self-improvement lab — measured, not narrated",
        "",
        f"Aggregation rules are replayed over every pre-open day using only earlier data. A rule "
        f"replaces the equal-weight mean only when its anytime-valid confidence sequence "
        f"(Bonferroni over {k} challengers, α={lb.get('alpha')}) shows it is better; the judge and "
        f"the gates are switched off the same way once shown to hurt "
        f"([`evals/lab.py`](src/evals/lab.py)).",
        "",
        f"**Champion: `{champ}`** ({status}; {lb.get('n_days', 0)} days replayed).",
        "",
        "| Rule | MAPE | Gain vs mean | CS (α/K) | Status |",
        "|---|---|---|---|---|",
        f"| mean (reference) | {_pct(lb.get('reference_mape'))} | — | — | "
        f"{'🏆 champion' if champ == lab.REFERENCE else 'replaced'} |",
    ]
    for name, cs in lb.get("challengers", {}).items():
        lines.append(f"| {name} | {_pct(cs.get('mape'))} | {_pct(cs.get('mean'))} | {_cs(cs)} | "
                     f"{'🏆 champion' if name == champ else _cs_verdict(cs)} |")
    j, g, legacy = lb.get("judge_effect", {}), lb.get("gates_effect", {}), lb.get("legacy_shipped_vs_mean", {})
    state = lb.get("state", {})
    lines += [
        "",
        f"- **Judge adjustment** (APE of blend − APE after the judge's clamped σ-move; v2 rows): "
        f"n={j.get('n', 0)}, mean {_pct(j.get('mean'))}, CS {_cs(j)} → judge "
        f"{'**on**' if state.get('judge_enabled', True) else '**switched off**'}.",
        f"- **Guardrail gates** (APE before − after the gates): n={g.get('n', 0)}, mean "
        f"{_pct(g.get('mean'))}, CS {_cs(g)} → gates "
        f"{'**on**' if state.get('gates_enabled', True) else '**switched off**'}.",
        f"- **Legacy v1 pipeline** (judge wrote the number) vs the plain mean of its own analysts: "
        f"mean {_pct(legacy.get('mean'))} over {legacy.get('n', 0)} pre-open days, CS {_cs(legacy)} "
        f"— {_cs_verdict(legacy)} (negative = the judge cost accuracy).",
        "",
    ]
    return lines


def _render_integrity(metrics: dict) -> str:
    """A short, always-visible note on what the headline excludes and what the gates are worth."""
    late = metrics.get("n_late_rows", 0)
    g = metrics.get("gate_effect") or {}
    cov = metrics.get("coverage") or {}
    p = metrics.get("all_time_pre_open") or {}
    lines = ["### Integrity & mechanism", ""]
    if late:
        lines.append(
            f"- **{late} scored row(s) were created after their session opened** and are excluded "
            f"from every verdict. They saw part of the tape they forecast. New post-open rows are "
            f"refused (`--allow-late` to override), and the schedule now researches the evening "
            f"before and only re-anchors in the morning."
        )
    else:
        lines.append("- All scored rows were created before their session opened. ✅")
    if cov.get("n_sessions"):
        lines.append(f"- **Pre-open coverage:** {cov['n_pre_open']} of the last "
                     f"{cov['n_sessions']} sessions (since {cov.get('since')}) got a forecast "
                     f"written before the open.")
    lines.append(f"- **Live anchors:** {p.get('n_live_anchor', 0)} of {p.get('n', 0)} pre-open "
                 f"forecasts were anchored on a real after-hours/pre-market trade (the rest on the "
                 f"prior close — before v2 the quote feed silently echoed it).")
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
    # The verdict is read off ALL-TIME pre-open rows: a row written after the open saw part of the
    # tape it was forecasting, and a short rolling window flips on noise.
    p = metrics.get("all_time_pre_open") or a

    lines = [
        f"# 📈 {metrics['symbol']} Daily Close Predictor — Results",
        "",
        "_Auto-generated after every run. Verdicts use pre-open forecasts only and anytime-valid "
        "95% confidence sequences, so they stay valid although this page is re-read daily. "
        "PASS = predicted close within ±1% of actual._",
        "",
        "## Verdict",
        render_verdict(metrics),
        _render_pending(metrics.get("pending")),
        *_render_scoreboard(metrics),
        *_render_lab(metrics),
        _render_integrity(metrics),
        "",
        "## Metrics",
        "",
        f"| Metric | All-time (pre-open) — verdict basis | Last {metrics['rolling_window']} (all rows) "
        f"| All-time (all rows) |",
        "|---|---|---|---|",
        f"| Scored days | {p.get('n', 0)} | {r.get('n', 0)} | {a.get('n_all_time', 0)} |",
        f"| PASS rate (±1%) | {_pct(p.get('pass_rate'))} | {_pct(r.get('pass_rate'))} | {_pct(a.get('pass_rate'))} |",
        f"| Directional accuracy | {_pct(p.get('directional_accuracy'))} | {_pct(r.get('directional_accuracy'))} | {_pct(a.get('directional_accuracy'))} |",
        f"| MAPE | {_pct(p.get('mape'))} | {_pct(r.get('mape'))} | {_pct(a.get('mape'))} |",
        f"| Baseline MAPE (random walk) | {_pct(p.get('baseline_mape'))} | {_pct(r.get('baseline_mape'))} | {_pct(a.get('baseline_mape'))} |",
        f"| Edge (baseline − model) | {_pct(p.get('edge'))} | {_pct(r.get('edge'))} | {_pct(a.get('edge'))} |",
        f"| Brier (confidence calib.) | {_num(p.get('brier'))} | {_num(r.get('brier'))} | {_num(a.get('brier'))} |",
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
              "price-history context on the dashboard chart. ⚠️ = written after the open, excluded "
              "from verdicts)_", "",
              "| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest | Pre-open |",
              "|---|---|---|---|---|---|---|---|---|"]
    scored = [r for r in rows if r.get("status") == "scored" and not r.get("seed")]
    if not scored:
        lines.append("| _no model predictions scored yet_ | | | | | | | | |")
    for row in sorted(scored, key=lambda r: r.get("date", ""))[-20:][::-1]:
        lines.append(
            f"| {row['date']} | {_num(row.get('predicted_close'))} | {_num(row.get('actual_close'))} "
            f"| {_pct(row.get('ape'))} | {'✅' if row.get('pass') else '❌'} "
            f"| {'✅' if row.get('directional_hit') else '❌'} "
            f"| {'✅' if row.get('beats_baseline') else '❌'} | {row.get('winning_strategy') or '—'} "
            f"| {'⚠️' if integrity.is_late(row) else '✅'} |"
        )
    lines += ["", "_This is a research experiment, not financial advice._", ""]
    return "\n".join(lines)


def generate(ledger_path: Path | None = None) -> dict:
    """Regenerate all artifacts (and the lab's mechanism state). Returns the metrics dict."""
    settings.ensure_dirs()
    rows = read_jsonl(ledger_path or settings.LEDGER_PATH)
    scorecards = load_scorecards()
    metrics = build_metrics(rows, scorecards)

    write_csv(rows, settings.RESULTS_CSV)
    lab.save_state(metrics["lab"])
    settings.METRICS_JSON.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    settings.RESULTS_MD.write_text(render_results_md(metrics, rows), encoding="utf-8")
    settings.SITE_DATA.parent.mkdir(parents=True, exist_ok=True)
    settings.SITE_DATA.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    return metrics
