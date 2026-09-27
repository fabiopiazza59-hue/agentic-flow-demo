"""A/B comparison report — paired evaluation of arm A (ensemble+gates) vs arm B (raven-style).

Both arms predict the same sessions from the same features snapshot, so the comparison is a
paired test: per-day APE deltas, B-wins rate, and a two-sided binomial sign test. Outputs:
  results/ab_compare.json      full comparison payload
  RESULTS.md                   an appended "A/B test" section
  results/site/data.json       gains "variant_b" and "ab" keys for the dashboard
"""

from __future__ import annotations

import json

from .config import settings
from .evals.metrics import aggregate
from .evals.paired import sign_test_p
from .utils import read_jsonl

ARM_LABELS = {"a": "A — ensemble + gates", "b": "B — raven prior + pulse"}


def paired_days(rows_a: list[dict], rows_b: list[dict]) -> list[dict]:
    """Days scored in BOTH ledgers (non-seed), oldest first."""
    a = {r["date"]: r for r in rows_a
         if r.get("status") == "scored" and not r.get("seed") and r.get("ape") is not None}
    b = {r["date"]: r for r in rows_b
         if r.get("status") == "scored" and r.get("ape") is not None}
    out = []
    for d in sorted(a.keys() & b.keys()):
        ra, rb = a[d], b[d]
        out.append({
            "date": d,
            "ape_a": ra["ape"], "ape_b": rb["ape"],
            "delta": round(ra["ape"] - rb["ape"], 6),   # positive => B better
            "b_wins": rb["ape"] < ra["ape"],
            "dir_hit_a": ra.get("directional_hit"), "dir_hit_b": rb.get("directional_hit"),
            "predicted_a": ra.get("predicted_close"), "predicted_b": rb.get("predicted_close"),
            "actual": ra.get("actual_close"),
        })
    return out


def build_comparison(rows_a: list[dict], rows_b: list[dict]) -> dict:
    pairs = paired_days(rows_a, rows_b)
    decisive = [p for p in pairs if p["ape_a"] != p["ape_b"]]
    b_wins = sum(1 for p in decisive if p["b_wins"])
    n_pairs = len(pairs)
    mean_delta = round(sum(p["delta"] for p in pairs) / n_pairs, 6) if pairs else None

    if n_pairs < settings.AB_MIN_PAIRED_DAYS:
        verdict = (f"⏳ Only {n_pairs} paired scored day(s) — no verdict before "
                   f"{settings.AB_MIN_PAIRED_DAYS}.")
    else:
        p_val = sign_test_p(b_wins, len(decisive))
        better = "B" if mean_delta and mean_delta > 0 else "A"
        sig = "statistically significant" if p_val is not None and p_val < 0.05 else "not significant"
        verdict = (f"Arm {better} leads on paired MAPE (mean daily delta "
                   f"{abs(mean_delta) * 100:.2f}% in {better}'s favor); B wins "
                   f"{b_wins}/{len(decisive)} decisive days (sign test p={p_val}, {sig}).")

    return {
        "labels": ARM_LABELS,
        "n_paired": n_pairs,
        "b_wins": b_wins,
        "n_decisive": len(decisive),
        "mean_delta_a_minus_b": mean_delta,
        "sign_test_p": sign_test_p(b_wins, len(decisive)),
        "verdict": verdict,
        "arm_a": {"rolling": aggregate(rows_a, window=settings.ROLLING_WINDOW),
                  "all_time": aggregate(rows_a, window=None)},
        "arm_b": {"rolling": aggregate(rows_b, window=settings.ROLLING_WINDOW),
                  "all_time": aggregate(rows_b, window=None)},
        "pairs": pairs,
    }


def _pct(x) -> str:
    return f"{x * 100:.2f}%" if isinstance(x, (int, float)) else "—"


def render_ab_md(cmp: dict, pending_b: dict | None) -> str:
    ra, rb = cmp["arm_a"]["rolling"], cmp["arm_b"]["rolling"]
    lines = [
        "## 🅰️/🅱️ A/B test — ensemble+gates vs raven-style prior+pulse",
        "",
        "_Both arms predict the same sessions from the same pre-open snapshot (paired test)._",
        "",
        f"**Verdict:** {cmp['verdict']}",
        "",
    ]
    if pending_b:
        lines += [f"**B's open prediction ({pending_b['date']}):** close ≈ "
                  f"**{pending_b.get('predicted_close')}** ({pending_b.get('predicted_direction')}, "
                  f"adj {pending_b.get('adjustment_sigma')}σ, "
                  f"{pending_b.get('n_evidence')} evidence items).", ""]
    lines += [
        "| Rolling metric | A — ensemble+gates | B — raven prior+pulse |",
        "|---|---|---|",
        f"| Scored days | {ra.get('n', 0)} | {rb.get('n', 0)} |",
        f"| PASS rate (±1%) | {_pct(ra.get('pass_rate'))} | {_pct(rb.get('pass_rate'))} |",
        f"| Directional accuracy | {_pct(ra.get('directional_accuracy'))} | {_pct(rb.get('directional_accuracy'))} |",
        f"| MAPE | {_pct(ra.get('mape'))} | {_pct(rb.get('mape'))} |",
        f"| Edge vs baseline | {_pct(ra.get('edge'))} | {_pct(rb.get('edge'))} |",
        "",
        f"Paired days: {cmp['n_paired']}; B wins {cmp['b_wins']}/{cmp['n_decisive']} decisive; "
        f"mean daily APE delta (A−B) {_pct(cmp['mean_delta_a_minus_b'])}; "
        f"sign test p = {cmp['sign_test_p'] if cmp['sign_test_p'] is not None else '—'}.",
        "",
    ]
    return "\n".join(lines)


def generate() -> dict:
    """Build the comparison, write ab_compare.json, extend RESULTS.md + site data.json."""
    rows_a = read_jsonl(settings.LEDGER_PATH)
    rows_b = read_jsonl(settings.LEDGER_B_PATH)
    cmp = build_comparison(rows_a, rows_b)

    settings.AB_COMPARE_JSON.parent.mkdir(parents=True, exist_ok=True)
    settings.AB_COMPARE_JSON.write_text(json.dumps(cmp, indent=2, default=str), encoding="utf-8")

    pending_b = next((r for r in sorted(rows_b, key=lambda r: r.get("date", ""), reverse=True)
                      if r.get("status") == "pending"), None)

    # Append the A/B section to RESULTS.md (generated fresh by report.generate() each run).
    md = settings.RESULTS_MD.read_text(encoding="utf-8") if settings.RESULTS_MD.exists() else ""
    footer = "_This is a research experiment, not financial advice._"
    section = render_ab_md(cmp, pending_b)
    if footer in md:
        md = md.replace(footer, section + footer)
    else:
        md += "\n" + section
    settings.RESULTS_MD.write_text(md, encoding="utf-8")

    # Extend the dashboard data feed.
    if settings.SITE_DATA.exists():
        site = json.loads(settings.SITE_DATA.read_text(encoding="utf-8"))
        scored_b = [r for r in rows_b if r.get("status") == "scored"]
        site["variant_b"] = {
            "rolling": cmp["arm_b"]["rolling"],
            "all_time": cmp["arm_b"]["all_time"],
            "series": [{"date": r["date"], "predicted_close": r.get("predicted_close"),
                        "actual_close": r.get("actual_close"), "ape": r.get("ape"),
                        "pass": r.get("pass")}
                       for r in sorted(scored_b, key=lambda r: r.get("date", ""))],
            "pending": pending_b,
        }
        site["ab"] = {k: cmp[k] for k in
                      ["labels", "n_paired", "b_wins", "n_decisive",
                       "mean_delta_a_minus_b", "sign_test_p", "verdict"]}
        settings.SITE_DATA.write_text(json.dumps(site, indent=2, default=str), encoding="utf-8")

    return cmp
