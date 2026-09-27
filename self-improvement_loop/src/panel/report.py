"""Panel report — does the LLM add value over the free baseline, across many names?

The inferential unit is the session, never the name: each metric is first averaged over the
names forecast that session, and the anytime-valid confidence sequence (evals/sequential.py)
runs on that daily series. Names that move together are therefore not counted as independent
evidence, while idiosyncratic noise still averages out — which is where the power comes from.

Metrics (pre-open rows only; LLM = the logged counterfactual, valid even when switched off):
  APE gain      anchor APE − LLM APE, mean over names
  CRPS gain     same, for calibrated predictive distributions (pooled symmetric conformal)
  Rank IC       Spearman(LLM adjustment, realized ln(close/anchor)) across names — the standard
                cross-sectional skill measure; days where the LLM moved no name are skipped
  Direction     Brier of P(close > prior close), LLM vs anchor distribution
"""

from __future__ import annotations

import json
import math
import statistics as st

from ..config import settings
from ..evals import integrity
from ..evals import probabilistic as pb
from ..evals.sequential import confidence_sequence
from .runner import _pooled_residuals, load_rows, save_state

CENTERS = {"llm": "predicted_close_llm", "anchor": "anchor", "random_walk": "prior_close"}


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    rx, ry = _ranks(x), _ranks(y)
    mx, my = st.mean(rx), st.mean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return cov / (sx * sy) if sx and sy else None


def _clean_scored(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("status") == "scored" and not integrity.is_late(r)
            and r.get("actual_close") and r.get("sigma_pct")]


def daily_series(rows: list[dict]) -> list[dict]:
    """One entry per clean session: name-averaged gains, rank IC, coverage."""
    clean = _clean_scored(rows)
    dates = sorted({r["date"] for r in clean})
    zcache: dict[tuple[str, str], list[float]] = {}
    out = []
    for d in dates:
        day = [r for r in clean if r["date"] == d]
        crps = {k: [] for k in CENTERS}
        brier = {k: [] for k in CENTERS}
        cover80 = []
        for r in day:
            actual, prior, sigma = float(r["actual_close"]), float(r["prior_close"]), float(r["sigma_pct"])
            for name, key in CENTERS.items():
                if (d, key) not in zcache:
                    zcache[(d, key)] = _pooled_residuals(rows, d, key)
                zq, _ = pb.z_quantiles(zcache[(d, key)])
                q = pb.predictive_quantiles(float(r[key]), sigma, zq)
                s = pb.score_distribution(q, actual, prior)
                crps[name].append(s["crps"])
                brier[name].append(s["brier_up"])
                if name == "llm":
                    cover80.append(1.0 if s["cover80"] else 0.0)
        ic = spearman([float(r.get("adj_sigma") or 0) for r in day],
                      [math.log(float(r["actual_close"]) / float(r["anchor"])) for r in day])
        out.append({
            "date": d, "n_names": len(day),
            "ape_gain_vs_anchor": st.mean(r["anchor_ape"] - r["llm_ape"] for r in day),
            "ape_gain_vs_rw": st.mean(r["baseline_ape"] - r["llm_ape"] for r in day),
            "crps_gain_vs_anchor": st.mean(crps["anchor"]) - st.mean(crps["llm"]),
            "direction_gain_vs_anchor": st.mean(brier["anchor"]) - st.mean(brier["llm"]),
            "anchor_gain_vs_rw": st.mean(r["baseline_ape"] - r["anchor_ape"] for r in day),
            "rank_ic": ic, "llm_cover80": st.mean(cover80),
            "llm_mape": st.mean(r["llm_ape"] for r in day),
            "anchor_mape": st.mean(r["anchor_ape"] for r in day),
            "rw_mape": st.mean(r["baseline_ape"] for r in day),
            "n_moved": sum(1 for r in day if r.get("adj_sigma")),
        })
    return out


def build(rows: list[dict]) -> dict:
    days = daily_series(rows)
    col = lambda k: [d[k] for d in days if d.get(k) is not None]  # noqa: E731
    cs = {k: confidence_sequence(col(k)) for k in
          ("ape_gain_vs_anchor", "ape_gain_vs_rw", "crps_gain_vs_anchor",
           "direction_gain_vs_anchor", "rank_ic", "anchor_gain_vs_rw")}
    late = sum(1 for r in rows if r.get("status") == "scored" and integrity.is_late(r))
    per_name: dict[str, dict] = {}
    for r in _clean_scored(rows):
        p = per_name.setdefault(r["symbol"], {"n": 0, "llm": 0.0, "anchor": 0.0, "moved": 0})
        p["n"] += 1
        p["llm"] += r["llm_ape"]
        p["anchor"] += r["anchor_ape"]
        p["moved"] += 1 if r.get("adj_sigma") else 0
    for p in per_name.values():
        p["llm"] /= p["n"]
        p["anchor"] /= p["n"]
    pending = [r for r in rows if r.get("status") == "pending"]
    llm_enabled = cs["ape_gain_vs_anchor"]["decision"] != "negative"
    return {
        "n_sessions": len(days), "n_forecasts": sum(d["n_names"] for d in days),
        "n_late_rows": late, "cs": cs, "days": days, "per_name": per_name,
        "mean": {k: (st.mean(col(k)) if col(k) else None) for k in
                 ("llm_mape", "anchor_mape", "rw_mape", "llm_cover80", "n_moved")},
        "pending": {"date": pending[0]["date"], "n": len(pending),
                    "n_moved": sum(1 for r in pending if r.get("adj_sigma")),
                    "n_live": sum(1 for r in pending if r.get("anchor_live"))} if pending else None,
        "state": {"llm_enabled": llm_enabled,
                  "why": cs["ape_gain_vs_anchor"]["decision"],
                  "n_sessions": len(days)},
    }


def _pct(x, digits=2) -> str:
    return f"{x * 100:.{digits}f}%" if isinstance(x, (int, float)) else "—"


def _cs(cs: dict, pct: bool = True) -> str:
    if cs.get("lo") is None:
        return f"n={cs.get('n', 0)} sessions (too few)"
    f = (lambda v: f"{v * 100:+.3f}%") if pct else (lambda v: f"{v:+.3f}")
    return f"[{f(cs['lo'])}, {f(cs['hi'])}]"


def _verdict(cs: dict) -> str:
    return {"positive": "✅ better", "negative": "❌ worse", "undecided": "≈ not distinguishable",
            "insufficient": "⏳ too few sessions"}.get(cs.get("decision"), "—")


def render(m: dict) -> str:
    cs, mean = m["cs"], m["mean"]
    head = cs["ape_gain_vs_anchor"]
    if head["decision"] == "positive":
        verdict = "✅ **The LLM adds value over the free baseline** across the panel"
    elif head["decision"] == "negative":
        verdict = ("❌ **The LLM makes forecasts worse than the free baseline** — it is switched "
                   "off (the panel ships the anchor) and keeps running in shadow")
    elif head["decision"] == "insufficient":
        verdict = "⏳ **Not enough clean sessions yet** to judge the LLM against the free baseline"
    else:
        verdict = "❌ **No edge yet** — the LLM's gain over the free baseline is not distinguishable from noise"
    pend = m.get("pending")
    late_note = (f" {m['n_late_rows']} rows written after the open are excluded."
                 if m["n_late_rows"] else "")
    lines = [
        "# 📊 Panel — does an LLM add value over the pre-open price?",
        "",
        f"_{len(settings.PANEL_SYMBOLS)} US large caps a session. Every name's free baseline is its "
        "latest pre-open trade (the anchor); the LLM may move it by at most "
        f"±{settings.PANEL_MAX_ADJ_SIGMA}σ. Verdicts average each metric over the names of a "
        "session and read an anytime-valid 95% confidence sequence on that daily series, so they "
        "stay valid although this page is regenerated daily. Pre-open forecasts only. "
        "Design: [spec/panel.md](spec/panel.md)._",
        "",
        "## Verdict",
        f"{verdict} ({head.get('n', 0)} sessions, {m['n_forecasts']} name-forecasts; mean daily "
        f"APE gain {_pct(head.get('mean'), 3)}, CS {_cs(head)}).",
        "",
    ]
    if pend:
        lines += [f"**Pending ({pend['date']}):** {pend['n']} names, {pend['n_live']} anchored on "
                  f"a live pre-open trade, the LLM moved {pend['n_moved']} off their anchor.", ""]
    lines += [
        "| Daily metric (mean over names) | Mean | 95% CS | Verdict |",
        "|---|---|---|---|",
        f"| APE gain: LLM vs free baseline | {_pct(head.get('mean'), 3)} | {_cs(head)} | {_verdict(head)} |",
        f"| CRPS gain: LLM vs free baseline | {_pct(cs['crps_gain_vs_anchor'].get('mean'), 3)} | {_cs(cs['crps_gain_vs_anchor'])} | {_verdict(cs['crps_gain_vs_anchor'])} |",
        f"| Rank IC (adjustment vs realized move off the anchor) | {cs['rank_ic'].get('mean') if cs['rank_ic'].get('mean') is None else round(cs['rank_ic']['mean'], 3)} | {_cs(cs['rank_ic'], pct=False)} | {_verdict(cs['rank_ic'])} |",
        f"| Direction (Brier of P(up)) vs free baseline | {cs['direction_gain_vs_anchor'].get('mean') if cs['direction_gain_vs_anchor'].get('mean') is None else round(cs['direction_gain_vs_anchor']['mean'], 4)} | {_cs(cs['direction_gain_vs_anchor'], pct=False)} | {_verdict(cs['direction_gain_vs_anchor'])} |",
        f"| APE gain: LLM vs prior close | {_pct(cs['ape_gain_vs_rw'].get('mean'), 3)} | {_cs(cs['ape_gain_vs_rw'])} | {_verdict(cs['ape_gain_vs_rw'])} |",
        f"| APE gain: anchor vs prior close (the free information itself) | {_pct(cs['anchor_gain_vs_rw'].get('mean'), 3)} | {_cs(cs['anchor_gain_vs_rw'])} | {_verdict(cs['anchor_gain_vs_rw'])} |",
        "",
        f"MAPE — LLM {_pct(mean['llm_mape'])}, free baseline {_pct(mean['anchor_mape'])}, prior "
        f"close {_pct(mean['rw_mape'])}; LLM 80% interval coverage {_pct(mean['llm_cover80'])}; "
        f"names moved per session {mean['n_moved'] if mean['n_moved'] is None else round(mean['n_moved'], 1)}. "
        f"LLM is **{'on' if m['state']['llm_enabled'] else 'off (shadow)'}**.{late_note}",
        "",
        "## Per name (pre-open forecasts)",
        "",
        "| Name | Sessions | LLM MAPE | Free-baseline MAPE | Days moved |",
        "|---|---|---|---|---|",
    ]
    for s, p in sorted(m["per_name"].items()):
        lines.append(f"| {s} | {p['n']} | {_pct(p['llm'])} | {_pct(p['anchor'])} | {p['moved']} |")
    if not m["per_name"]:
        lines.append("| _no scored sessions yet_ | | | | |")
    lines += ["", "_Research experiment, not financial advice._", ""]
    return "\n".join(lines)


def generate() -> dict:
    rows = load_rows()
    m = build(rows)
    save_state(m["state"])
    settings.PANEL_METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    settings.PANEL_METRICS_JSON.write_text(json.dumps(m, indent=2, default=str), encoding="utf-8")
    settings.PANEL_RESULTS_MD.write_text(render(m), encoding="utf-8")
    return m
