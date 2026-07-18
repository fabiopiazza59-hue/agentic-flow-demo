"""Variant B score/predict — mirrors run_daily's do_score/do_predict for the raven-style arm.

State lives in its own ledger (data/predictions_b.jsonl) and learnings dir (learnings_b/).
B has no reflector: its failure log is a compact deterministic entry per miss, and that log's
tail is fed straight back into the decider — the whole learning loop in one hop.
"""

from __future__ import annotations

from datetime import date

from ..config import settings
from ..data.providers import get_actual_close
from ..evals.metrics import score_row
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
    """Score the most recent pending B prediction whose session has completed."""
    pending = [r for r in rows_b if r.get("status") == "pending"]
    if not pending:
        log("no pending predictions to score.")
        return rows_b, None
    target = sorted(pending, key=lambda r: r["date"])[-1]
    actual = get_actual_close(settings.SYMBOL, target["date"])
    if actual is None:
        log(f"actual close for {target['date']} not available yet; skipping score.")
        return rows_b, None

    target.update(score_row(target, actual))
    target["scored_at"] = now_iso()
    rows_b = upsert_ledger_row(rows_b, target)
    if target.get("pass") is False:
        _append_failure(target)
    log(f"scored {target['date']}: predicted {target['predicted_close']} vs actual {actual} "
        f"(APE {target['ape'] * 100:.2f}%, {'PASS' if target['pass'] else 'FAIL'}).")
    return rows_b, target


def do_predict_b(rows_b: list[dict], target_date: date, client, features: dict,
                 history) -> tuple[list[dict], dict]:
    """Prior -> pulse -> bounded decision -> pending ledger row (shared features snapshot)."""
    prior = build_prior(history, features)
    pulse = gather_pulse(features, client=client)
    write_pulse_artifact(target_date.isoformat(), pulse, prior)
    final = decide(prior, pulse, rows_b, client=client)

    row = {
        "date": target_date.isoformat(),
        "created_at": now_iso(),
        "variant": "b",
        "prior_close": features["prev_close"],
        "predicted_close": final["predicted_close"],
        "predicted_direction": final["direction"],
        "confidence": final["confidence"],
        "prior": prior,
        "pulse_summary": pulse.get("summary", ""),
        "n_evidence": len(pulse.get("items", [])),
        "adjustment_sigma": final["adjustment_sigma"],
        "caps_applied": final["caps_applied"],
        "rationale": final["rationale"],
        "quote_source": features.get("quote_source"),
        "status": "pending",
        "actual_close": None,
    }
    rows_b = upsert_ledger_row(rows_b, row)
    log(f"predicted {target_date.isoformat()}: close ≈ {row['predicted_close']} "
        f"({row['predicted_direction']}, conf {row['confidence']}, adj {row['adjustment_sigma']}σ, "
        f"{row['n_evidence']} evidence items).")
    return rows_b, row
