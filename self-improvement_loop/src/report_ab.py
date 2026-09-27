"""A/B comparison report — paired evaluation of arm A (ensemble+gates) vs arm B (raven-style).

Both arms predict the same sessions from the same features snapshot, so the comparison is a
paired test over days. Only days on which BOTH forecasts were written before the open count
(late rows saw part of the tape). The verdict reads an anytime-valid 95% confidence sequence on
the daily APE differences (evals/sequential.py), so re-rendering it every day does not inflate
its error rate; the sign test is kept as a descriptive number. Outputs:
  results/ab_compare.json      full comparison payload
  RESULTS.md                   an appended "A/B test" section
  results/site/data.json       gains "variant_b" and "ab" keys for the dashboard
"""

from __future__ import annotations

import json

from .config import settings
from .evals import integrity
from .evals import probabilistic as pb
from .evals.metrics import aggregate
from .evals.paired import sign_test_p
from .evals.sequential import confidence_sequence
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
            "clean": not integrity.is_late(ra) and not integrity.is_late(rb),
        })
    return out


def _crps_pairs(rows_a: list[dict], rows_b: list[dict]) -> list[float]:
    """Daily CRPS(A) − CRPS(B) on clean paired days (positive => B's distribution was better)."""
    rets = pb._returns_by_date(rows_a, rows_b)
    clean = lambda r: not integrity.is_late(r)  # noqa: E731
    ea = {e["date"]: e for e in pb.evaluate(rows_a, rets, clean) if e["clean"]}
    eb = {e["date"]: e for e in pb.evaluate(rows_b, rets, clean) if e["clean"]}
    return [ea[d]["model"]["crps"] - eb[d]["model"]["crps"] for d in sorted(ea.keys() & eb.keys())]


def build_comparison(rows_a: list[dict], rows_b: list[dict]) -> dict:
    pairs = paired_days(rows_a, rows_b)
    clean = [p for p in pairs if p["clean"]]
    decisive = [p for p in clean if p["ape_a"] != p["ape_b"]]
    b_wins = sum(1 for p in decisive if p["b_wins"])
    n_clean = len(clean)
    mean_delta = round(sum(p["delta"] for p in clean) / n_clean, 6) if clean else None
    cs = confidence_sequence([p["delta"] for p in clean])
    crps_cs = confidence_sequence(_crps_pairs(rows_a, rows_b))
    p_val = sign_test_p(b_wins, len(decisive))

    if n_clean < settings.AB_MIN_PAIRED_DAYS:
        verdict = (f"⏳ Only {n_clean} clean paired day(s) (both forecasts written before the "
                   f"open) — no verdict before {settings.AB_MIN_PAIRED_DAYS}.")
    else:
        better = "B" if mean_delta and mean_delta > 0 else "A"
        lo, hi = cs.get("lo"), cs.get("hi")
        interval = (f"[{lo * 100:+.3f}%, {hi * 100:+.3f}%]" if lo is not None else "n/a")
        if cs["decision"] in ("positive", "negative"):
            sig = f"anytime-valid 95% CS {interval} excludes zero — **{better} is better**"
        else:
            sig = (f"anytime-valid 95% CS {interval} includes zero — not distinguishable "
                   f"from noise")
        verdict = (f"Arm {better} leads on paired MAPE (mean daily delta "
                   f"{abs(mean_delta) * 100:.2f}% in {better}'s favor over {n_clean} pre-open "
                   f"days); B wins {b_wins}/{len(decisive)} decisive days (sign test p={p_val}, "
                   f"descriptive); {sig}.")

    return {
        "labels": ARM_LABELS,
        "n_paired": len(pairs),
        "n_clean": n_clean,
        "b_wins": b_wins,
        "n_decisive": len(decisive),
        "mean_delta_a_minus_b": mean_delta,
        "sign_test_p": p_val,
        "cs": cs,
        "crps_cs": crps_cs,
        "verdict": verdict,
        "arm_a": {"rolling": aggregate(rows_a, window=settings.ROLLING_WINDOW),
                  "all_time": aggregate(rows_a, window=None),
                  "all_time_pre_open": aggregate(rows_a, window=None, pre_open_only=True)},
        "arm_b": {"rolling": aggregate(rows_b, window=settings.ROLLING_WINDOW),
                  "all_time": aggregate(rows_b, window=None),
                  "all_time_pre_open": aggregate(rows_b, window=None, pre_open_only=True)},
        "pairs": pairs,
    }


def _pct(x) -> str:
    return f"{x * 100:.2f}%" if isinstance(x, (int, float)) else "—"


def render_ab_md(cmp: dict, pending_b: dict | None) -> str:
    ra, rb = cmp["arm_a"]["all_time_pre_open"], cmp["arm_b"]["all_time_pre_open"]
    lines = [
        "## 🅰️/🅱️ A/B test — ensemble+gates vs raven-style prior+pulse",
        "",
        "_Both arms forecast the same sessions from the same pre-open snapshot (paired test). Only "
        "days on which both forecasts were written before the open count._",
        "",
        f"**Verdict:** {cmp['verdict']}",
        "",
    ]
    if pending_b:
        label = ("awaiting score" if integrity.session_has_closed(pending_b["date"], None, 0)
                 else "next forecast")
        lines += [f"**B's {label} ({pending_b['date']}):** close ≈ "
                  f"**{pending_b.get('predicted_close')}** ({pending_b.get('predicted_direction')}, "
                  f"adj {pending_b.get('adjustment_sigma')}σ, "
                  f"{pending_b.get('n_evidence')} evidence items).", ""]
    c = cmp.get("crps_cs") or {}
    crps_line = (f"[{c['lo'] * 100:+.3f}%, {c['hi'] * 100:+.3f}%] ({c['decision']})"
                 if c.get("lo") is not None else f"n={c.get('n', 0)} (too few)")
    lines += [
        "| All-time, pre-open | A — ensemble+gates | B — raven prior+pulse |",
        "|---|---|---|",
        f"| Scored days | {ra.get('n', 0)} | {rb.get('n', 0)} |",
        f"| PASS rate (±1%) | {_pct(ra.get('pass_rate'))} | {_pct(rb.get('pass_rate'))} |",
        f"| Directional accuracy | {_pct(ra.get('directional_accuracy'))} | {_pct(rb.get('directional_accuracy'))} |",
        f"| MAPE | {_pct(ra.get('mape'))} | {_pct(rb.get('mape'))} |",
        f"| Edge vs free baseline | {_pct(ra.get('anchor_edge'))} | {_pct(rb.get('anchor_edge'))} |",
        "",
        f"Clean paired days: {cmp['n_clean']} (of {cmp['n_paired']} paired); B wins "
        f"{cmp['b_wins']}/{cmp['n_decisive']} decisive; mean daily APE delta (A−B) "
        f"{_pct(cmp['mean_delta_a_minus_b'])}; CRPS delta (A−B) anytime-valid 95% CS {crps_line}.",
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
            "all_time_pre_open": cmp["arm_b"]["all_time_pre_open"],
            "series": [{"date": r["date"], "predicted_close": r.get("predicted_close"),
                        "actual_close": r.get("actual_close"), "ape": r.get("ape"),
                        "pass": r.get("pass")}
                       for r in sorted(scored_b, key=lambda r: r.get("date", ""))],
            "pending": pending_b,
        }
        site["ab"] = {k: cmp[k] for k in
                      ["labels", "n_paired", "n_clean", "b_wins", "n_decisive",
                       "mean_delta_a_minus_b", "sign_test_p", "cs", "crps_cs", "verdict"]}
        settings.SITE_DATA.write_text(json.dumps(site, indent=2, default=str), encoding="utf-8")

    return cmp
