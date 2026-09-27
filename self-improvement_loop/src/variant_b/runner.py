"""Variant B score/predict — mirrors run_daily's do_score/do_predict for the raven-style arm.

State lives in its own ledger (data/predictions_b.jsonl) and learnings dir (learnings_b/).
B has no reflector: its failure log is a compact deterministic entry per miss, and that log's
tail is fed straight back into the decider — the whole learning loop in one hop.
"""

from __future__ import annotations

from datetime import date

from ..config import settings
from ..data.providers import get_actual_close
from ..evals import integrity
from ..evals.metrics import score_row
from ..evals.probabilistic import forecast_distribution
from ..loop.session import anchor_fields
from ..utils import now_iso, upsert_ledger_row
from .decider import decide
from .prior import build_prior
from .pulse import gather_pulse, write_pulse_artifact


def log(msg: str) -> None:
    print(f"[variant_b] {msg}", flush=True)


def _append_failure(row: dict) -> None:
    settings.LEARNINGS_B_DIR.mkdir(parents=True, exist_ok=True)
    p = settings.LEARNINGS_B_DIR / "FAILURES.md"
    if not p.exists():
        p.write_text(
            "# Failure Log — Variant B (raven-style prior + pulse)\n\n"
            "Every missed prediction (>1% error), newest at the bottom. "
            "The tail of this file is fed to the decider before each shot.\n", encoding="utf-8"
        )
    ape_pct = (row.get("ape") or 0) * 100
    prior = row.get("prior") or {}
    entry = (
        f"\n## {row.get('date')} — FAIL (APE {ape_pct:.2f}%)\n"
        f"- Predicted {row.get('predicted_close')} vs actual {row.get('actual_close')} "
        f"(prior center {prior.get('center')}, σ {(prior.get('sigma_pct') or 0) * 100:.2f}%); "
        f"adjustment {row.get('adjustment_sigma')}σ; caps: {row.get('caps_applied')}; "
        f"dir hit: {row.get('directional_hit')}; beat baseline: {row.get('beats_baseline')}.\n"
        f"- Miss was {'directional' if not row.get('directional_hit') else 'magnitude-only'}; "
        f"{'the adjustment moved the wrong way or was oversized' if not row.get('directional_hit') else 'the adjustment was too timid or the prior center was off'}.\n"
    )
    with p.open("a", encoding="utf-8") as f:
        f.write(entry)


def do_score_b(rows_b: list[dict]) -> tuple[list[dict], dict | None]:
    """Score every pending B prediction whose session has closed (and settled), oldest first."""
    pending = sorted((r for r in rows_b if r.get("status") == "pending"), key=lambda r: r["date"])
    if not pending:
        log("no pending predictions to score.")
        return rows_b, None
    last = None
    for target in pending:
        if not integrity.session_has_closed(target["date"], None, settings.SCORE_SETTLE_MINUTES):
            continue
        actual = get_actual_close(settings.SYMBOL, target["date"])
        if actual is None:
            log(f"actual close for {target['date']} not available yet; skipping score.")
            continue
        target.update(score_row(target, actual))
        target["scored_at"] = now_iso()
        rows_b = upsert_ledger_row(rows_b, target)
        if target.get("pass") is False:
            _append_failure(target)
        log(f"scored {target['date']}: predicted {target['predicted_close']} vs actual {actual} "
            f"(APE {target['ape'] * 100:.2f}%, {'PASS' if target['pass'] else 'FAIL'}).")
        last = target
    return rows_b, last


def do_predict_b(rows_b: list[dict], target_date: date, client, features: dict,
                 history) -> tuple[list[dict], dict]:
    """Prior -> pulse -> bounded decision -> pending ledger row (shared features snapshot)."""
    prior = build_prior(history, features)
    pulse = gather_pulse(features, client=client)
    write_pulse_artifact(target_date.isoformat(), pulse, prior)
    final = decide(prior, pulse, rows_b, client=client)

    prior_close = float(features["prev_close"])
    sigma = float(features.get("sigma_ewma") or prior["sigma_pct"] or settings.DEFAULT_SIGMA)
    dist = forecast_distribution(rows_b, target_date.isoformat(), float(final["predicted_close"]),
                                 sigma, prior_close, is_clean=lambda r: not integrity.is_late(r),
                                 threshold=settings.PASS_THRESHOLD)
    anchor = features.get("anchor") or {"price": prior_close, "source": "prior_close",
                                        "time": None, "live": False}
    created = now_iso()
    row = {
        "date": target_date.isoformat(),
        "created_at": created,
        "variant": "b",
        "pipeline_version": settings.PIPELINE_VERSION,
        "prior_close": prior_close,
        "predicted_close": final["predicted_close"],
        "predicted_direction": final["direction"],
        "confidence": dist["p_pass"],
        "p_up": dist["p_up"],
        "sigma_pct": round(sigma, 6),
        "quantiles": dist["quantiles"],
        "calibration": dist["calibration"],
        "prior": prior,
        "pulse_summary": pulse.get("summary", ""),
        "n_evidence": len(pulse.get("items", [])),
        "pulse_web_results": pulse.get("web_results"),
        "adjustment_sigma": final["adjustment_sigma"],
        "caps_applied": final["caps_applied"],
        "rationale": final["rationale"],
        "quote_source": features.get("quote_source"),
        **anchor_fields(anchor, prior_close, created),
        "status": "pending",
        "actual_close": None,
    }
    integrity.annotate(row)
    rows_b = upsert_ledger_row(rows_b, row)
    log(f"predicted {target_date.isoformat()}: close ≈ {row['predicted_close']} "
        f"({row['predicted_direction']}, conf {row['confidence']}, adj {row['adjustment_sigma']}σ, "
        f"{row['n_evidence']} evidence items).")
    return rows_b, row
