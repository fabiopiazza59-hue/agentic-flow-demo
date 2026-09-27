"""Self-improvement by measured selection — the loop's mechanism lab.

The original self-improvement channel was text: post-mortems and strategy notes fed back into
the judge's prompt. The review of this loop against arXiv:2609.05663 found that channel inert or
harmful ("mechanism over exhortation"): the meta-judge cost 0.14% MAPE against a plain mean of its
own analysts, and the scorecard reweighting never moved. This module makes improvement a
measured mechanism instead:

1. Challengers — alternative rules for combining the analysts' closes — are replayed over every
   logged day using only information available before that day. No API calls, no lookahead.
2. A challenger replaces the reference rule (the equal-weight mean: the "forecast combination
   puzzle" — Stock & Watson 2004, Smith & Wallis 2009 — is that it is very hard to beat) only when
   an anytime-valid confidence sequence on the paired daily APE differences, Bonferroni-corrected
   for the number of challengers, excludes zero (see evals/sequential.py). The ledger is re-read
   every run, so the champion is a pure function of the record.
3. The same test runs on the loop's other two mechanisms — the judge's bounded adjustment and the
   guardrail gates — and switches either one off once it has been shown to hurt.

Only pre-open rows count: late rows saw part of the tape they forecast.
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

from ..config import settings
from . import integrity
from .metrics import ape
from .sequential import confidence_sequence

REFERENCE = "mean"


# ----------------------------------------------------------------------------- aggregators

def _closes(preds: dict, scale: float = 1.0) -> dict[str, float]:
    out = {}
    for name, p in (preds or {}).items():
        v = p.get("predicted_close") if isinstance(p, dict) else None
        if v is not None and float(v) > 0:
            out[name] = float(v) * scale
    return out


def _trailing(past: list[dict], name: str, window: int = 20) -> list[float]:
    """This analyst's APEs over the most recent `window` clean scored days before now."""
    errs = []
    for r in reversed(past):
        p = (r.get("analyst_predictions") or {}).get(name) or {}
        if p.get("predicted_close") and r.get("actual_close"):
            errs.append(ape(float(p["predicted_close"]), float(r["actual_close"])))
            if len(errs) >= window:
                break
    return errs


def agg_mean(c: dict, past: list[dict], anchor: float) -> float:
    return st.mean(c.values())


def agg_median(c: dict, past: list[dict], anchor: float) -> float:
    return st.median(c.values())


def agg_trimmed_mean(c: dict, past: list[dict], anchor: float) -> float:
    vals = sorted(c.values())
    return st.mean(vals[1:-1]) if len(vals) >= 4 else st.mean(vals)


def agg_inverse_mse(c: dict, past: list[dict], anchor: float) -> float:
    """Weights ∝ 1/MSE over each analyst's last 20 clean days (equal weights until 5 obs)."""
    w = {}
    for name in c:
        errs = _trailing(past, name)
        if len(errs) < 5:
            return agg_mean(c, past, anchor)
        w[name] = 1.0 / max(st.mean(e * e for e in errs), 1e-10)
    total = sum(w.values())
    return sum(w[n] * c[n] for n in c) / total


def agg_best_recent(c: dict, past: list[dict], anchor: float) -> float:
    """The single analyst with the lowest trailing-20 MAPE (mean until 5 obs each)."""
    scores = {}
    for name in c:
        errs = _trailing(past, name)
        if len(errs) < 5:
            return agg_mean(c, past, anchor)
        scores[name] = st.mean(errs)
    return c[min(scores, key=scores.get)]


def agg_shrink_half(c: dict, past: list[dict], anchor: float) -> float:
    """Half-way between the anchor and the mean: a 50% shrinkage toward the free baseline."""
    return anchor + 0.5 * (st.mean(c.values()) - anchor)


AGGREGATORS = {
    "mean": agg_mean,
    "median": agg_median,
    "trimmed_mean": agg_trimmed_mean,
    "inverse_mse": agg_inverse_mse,
    "best_recent": agg_best_recent,
    "shrink_half": agg_shrink_half,
}


def aggregate_analysts(rule: str, analyst_predictions: dict, past: list[dict],
                       anchor: float) -> float | None:
    """Combine analyst closes with the named rule (unknown rule -> the reference)."""
    c = _closes(analyst_predictions)
    if not c:
        return None
    fn = AGGREGATORS.get(rule) or AGGREGATORS[REFERENCE]
    return round(float(fn(c, past, anchor)), 2)


# ------------------------------------------------------------------------------- replay

def _clean_scored(rows: list[dict]) -> list[dict]:
    return sorted((r for r in rows if r.get("status") == "scored" and not r.get("seed")
                   and r.get("actual_close") and not integrity.is_late(r)),
                  key=lambda r: r["date"])


def replay(rows: list[dict]) -> list[dict]:
    """Per clean day: the APE every aggregation rule would have scored, using only prior days.

    Analyst closes are rescaled from the anchor they saw to the row's final anchor, so a
    re-anchored forecast is replayed on the same information it was finally shipped with.
    """
    clean = _clean_scored(rows)
    out = []
    for i, r in enumerate(clean):
        preds = r.get("analyst_predictions") or {}
        if len(_closes(preds)) < 3:
            continue
        anchor = float(r.get("anchor") or r["prior_close"])
        seen = float(r.get("analyst_anchor") or anchor)
        c = _closes(preds, scale=anchor / seen if seen > 0 else 1.0)
        actual = float(r["actual_close"])
        apes = {name: ape(fn(c, clean[:i], anchor), actual) for name, fn in AGGREGATORS.items()}
        out.append({"date": r["date"], "apes": apes, "shipped": float(r["ape"]),
                    "pipeline_version": r.get("pipeline_version", "v1")})
    return out


def _mechanism_effect(rows: list[dict], before_key: str, after_key: str) -> dict:
    """Paired effect of one pipeline stage: APE(before) − APE(after); positive = it helped."""
    deltas = []
    for r in _clean_scored(rows):
        b, a = r.get(before_key), r.get(after_key)
        if b is None or a is None:
            continue
        actual = float(r["actual_close"])
        deltas.append(ape(float(b), actual) - ape(float(a), actual))
    return confidence_sequence(deltas, alpha=settings.LAB_ALPHA)


def evaluate(rows: list[dict]) -> dict:
    """Full lab verdict: champion aggregator + kill-switch evidence for judge and gates."""
    days = replay(rows)
    challengers = [n for n in AGGREGATORS if n != REFERENCE]
    alpha_each = settings.LAB_ALPHA / max(len(challengers), 1)
    results = {}
    for name in challengers:
        deltas = [d["apes"][REFERENCE] - d["apes"][name] for d in days]
        cs = confidence_sequence(deltas, alpha=alpha_each)
        cs["mape"] = st.mean(d["apes"][name] for d in days) if days else None
        results[name] = cs
    promoted = [n for n, cs in results.items() if cs["decision"] == "positive"]
    champion = max(promoted, key=lambda n: results[n]["mean"]) if promoted else REFERENCE

    legacy = [d for d in days if d["pipeline_version"] == "v1"]
    shipped_vs_mean = confidence_sequence(
        [d["apes"][REFERENCE] - d["shipped"] for d in legacy], alpha=settings.LAB_ALPHA)

    judge = _mechanism_effect(rows, "predicted_close_blend", "predicted_close_pre_gates")
    gates = _mechanism_effect(rows, "predicted_close_raw", "predicted_close")
    return {
        "n_days": len(days),
        "reference": REFERENCE,
        "reference_mape": st.mean(d["apes"][REFERENCE] for d in days) if days else None,
        "champion": champion,
        "alpha": settings.LAB_ALPHA,
        "alpha_per_challenger": alpha_each,
        "challengers": results,
        "legacy_shipped_vs_mean": shipped_vs_mean,
        "judge_effect": judge,
        "gates_effect": gates,
        "state": {
            "champion": champion,
            "judge_enabled": judge["decision"] != "negative",
            "gates_enabled": gates["decision"] != "negative",
        },
    }


# ----------------------------------------------------------------------------- state file

def _state_path() -> Path:
    return settings.LEARNINGS_DIR / "lab_state.json"


def load_state() -> dict:
    """Current mechanism settings for arm A (defaults until the lab has written any)."""
    default = {"champion": REFERENCE, "judge_enabled": True, "gates_enabled": True}
    p = _state_path()
    if not p.exists():
        return default
    try:
        state = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default
    return {**default, **{k: state[k] for k in default if k in state}}


def save_state(lab: dict) -> None:
    p = _state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {**lab["state"], "n_days": lab["n_days"],
               "why": {"champion": lab["challengers"].get(lab["champion"], {}).get("decision",
                                                                                   "reference"),
                       "judge_effect": lab["judge_effect"]["decision"],
                       "gates_effect": lab["gates_effect"]["decision"]}}
    p.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
