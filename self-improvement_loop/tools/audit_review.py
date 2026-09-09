"""Reproduce every measured claim in spec/improvements-from-2609.05663.md.

Read-only: reads data/predictions.jsonl (+ predictions_b.jsonl) and prints the audit table.
Run:  python -m tools.audit_review    (from the project root)
"""

from __future__ import annotations

import itertools
import json
import random
import statistics as st
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evals.integrity import is_late  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
NAMES = ["technical", "momentum", "contrarian", "news", "macro"]


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def scored(rows: list[dict]) -> list[dict]:
    return sorted(
        (r for r in rows
         if r.get("status") == "scored" and not r.get("seed") and r.get("ape") is not None),
        key=lambda r: r["date"],
    )


def spearman(x: list[float], y: list[float]) -> float:
    def rank(z):
        order = sorted(range(len(z)), key=lambda i: z[i])
        out = [0] * len(z)
        for pos, idx in enumerate(order):
            out[idx] = pos
        return out
    rx, ry = rank(x), rank(y)
    n = len(x)
    d2 = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1 - 6 * d2 / (n * (n * n - 1))


def analyst_closes(row: dict) -> list[float]:
    ap = row.get("analyst_predictions") or {}
    return [float(ap[k]["predicted_close"]) for k in NAMES
            if k in ap and ap[k].get("predicted_close")]


def sign_test(deltas: list[float]) -> tuple[int, int, float]:
    """Two-sided exact binomial sign test on paired day deltas (ties dropped)."""
    d = [x for x in deltas if x != 0]
    n = len(d)
    wins = sum(1 for x in d if x > 0)
    k = max(wins, n - wins)
    p = sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n * 2
    return wins, n, min(1.0, p)


def bootstrap_ci(deltas: list[float], draws: int = 20000) -> tuple[float, float]:
    """90% percentile bootstrap over days (the day is the inferential unit — canon rule 1)."""
    random.seed(7)
    means = []
    for _ in range(draws):
        sample = [deltas[random.randrange(len(deltas))] for _ in range(len(deltas))]
        means.append(st.mean(sample))
    means.sort()
    return means[int(0.05 * draws)], means[int(0.95 * draws)]


def head(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def finding_1_capture_gap(sc: list[dict]) -> None:
    head("F1  Capture gap — the pipeline loses what its own analysts already had")
    variants = {
        "meta-judge blend (shipped)": lambda ps, r: r["predicted_close"],
        "equal-weight mean of analysts": lambda ps, r: sum(ps) / len(ps),
        "median of analysts": lambda ps, r: st.median(ps),
        "random-walk baseline": lambda ps, r: r["prior_close"],
        "oracle best analyst (ex-post)": lambda ps, r: min(ps, key=lambda p: abs(p - r["actual_close"])),
    }
    usable = [r for r in sc if analyst_closes(r)]
    for label, fn in variants.items():
        apes, passes = [], []
        for r in usable:
            pred = fn(analyst_closes(r), r)
            a = float(r["actual_close"])
            apes.append(abs(pred - a) / a)
            passes.append(1.0 if abs(pred - a) / a <= 0.01 else 0.0)
        print(f"  {label:32s} MAPE {100 * st.mean(apes):.3f}%   PASS {100 * st.mean(passes):3.0f}%")
    both = sum(1 for r in usable
               if min(abs(p - r["actual_close"]) / r["actual_close"] for p in analyst_closes(r)) <= 0.01
               and r["ape"] > 0.01)
    print(f"  days an analyst was inside +/-1% but the blend FAILED: "
          f"{both}/{len(usable)} = {100 * both / len(usable):.0f}%")


def finding_1b_paired(sc: list[dict]) -> None:
    head("F1b Paired day-level test: drop the meta-judge, keep a plain mean")
    usable = [r for r in sc if analyst_closes(r)]
    eq = [abs(sum(analyst_closes(r)) / len(analyst_closes(r)) - r["actual_close"]) / r["actual_close"]
          for r in usable]
    shipped = [r["ape"] for r in usable]
    baseline = [r["baseline_ape"] for r in usable]
    comparisons = [
        ("shipped - equal-mean  (>0 => equal-mean better)", [a - b for a, b in zip(shipped, eq)]),
        ("baseline - equal-mean (>0 => equal-mean beats RW)", [a - b for a, b in zip(baseline, eq)]),
        ("baseline - shipped    (>0 => shipped beats RW)", [a - b for a, b in zip(baseline, shipped)]),
    ]
    for label, deltas in comparisons:
        wins, n, p = sign_test(deltas)
        lo, hi = bootstrap_ci(deltas)
        print(f"  {label}\n      mean {100 * st.mean(deltas):+.4f}%  "
              f"90% CI [{100 * lo:+.4f}%, {100 * hi:+.4f}%]  wins {wins}/{n}  sign-test p={p:.3f}")


def finding_2_conviction_decay(sc: list[dict]) -> None:
    head("F2  Conviction decay — every aggregation layer shrinks the move toward the baseline")
    per = {k: [] for k in NAMES}
    ens, blend = [], []
    for r in sc:
        ap = r.get("analyst_predictions") or {}
        prev = float(r["prior_close"])
        for k in NAMES:
            if k in ap and ap[k].get("predicted_close"):
                per[k].append(abs(float(ap[k]["predicted_close"]) - prev) / prev)
        ps = analyst_closes(r)
        if ps:
            ens.append(abs(sum(ps) / len(ps) - prev) / prev)
        blend.append(abs(float(r["predicted_close"]) - prev) / prev)
    actual = [abs(r["actual_close"] - r["prior_close"]) / r["prior_close"] for r in sc]
    clean = [a for a in actual if a < 0.10]  # drop the 2026-07-31 gap artifact
    for k in NAMES:
        print(f"  median |move| {k:12s} {100 * st.median(per[k]):.3f}%")
    print(f"  median |move| {'equal-mean':12s} {100 * st.median(ens):.3f}%")
    print(f"  median |move| {'SHIPPED':12s} {100 * st.median(blend):.3f}%")
    print(f"  median |move| {'ACTUAL day':12s} {100 * st.median(clean):.3f}%  "
          f"(shipped is {st.median(blend) / st.median(clean):.2f}x a typical day)")
    gated = [abs(r["predicted_close"] - r["prior_close"]) / r["prior_close"]
             for r in sc if r.get("gates_applied")]
    ungated = [abs(r["predicted_close"] - r["prior_close"]) / r["prior_close"]
               for r in sc if not r.get("gates_applied")]
    print(f"  gated days   median |move| {100 * st.median(gated):.3f}%  (n={len(gated)})")
    print(f"  ungated days median |move| {100 * st.median(ungated):.3f}%  (n={len(ungated)})")


def finding_3_vol_blind(sc: list[dict]) -> None:
    head("F3  Sizing is scale-blind (paper section 5.1 analogue, partial)")
    acts = [abs(r["actual_close"] - r["prior_close"]) / r["prior_close"] for r in sc]
    vols, moves = [], []
    for i, r in enumerate(sc):
        if i < 20:
            continue
        window = [a for a in acts[i - 20:i] if a < 0.10]
        if len(window) < 10:
            continue
        vols.append(st.median(window))
        moves.append(abs(r["predicted_close"] - r["prior_close"]) / r["prior_close"])
    print(f"  n={len(vols)}  Spearman(trailing vol, |predicted move|) = {spearman(vols, moves):+.3f}")
    med = st.median(vols)
    calm = [m for m, v in zip(moves, vols) if v < med]
    wild = [m for m, v in zip(moves, vols) if v >= med]
    print(f"  median |predicted move|  calm half {100 * st.median(calm):.3f}%  "
          f"vs wild half {100 * st.median(wild):.3f}%")


def finding_4_dead_weights(sc: list[dict]) -> None:
    head("F4  The self-improvement signal is flat — meta-judge weights barely move")
    third = max(1, len(sc) // 3)
    for label, seg in [("first", sc[:third]), ("mid", sc[third:2 * third]), ("last", sc[2 * third:])]:
        means = {k: st.mean([float((r.get("weights") or {}).get(k, 0)) for r in seg]) for k in NAMES}
        spread = max(means.values()) - min(means.values())
        print(f"  {label:5s} {len(seg):2d}d  " + " ".join(f"{k[:4]}={means[k]:.2f}" for k in NAMES)
              + f"   spread {spread:.2f}")


def finding_5_confidence(sc: list[dict]) -> None:
    head("F5  Confidence is worse than a constant (paper: calibrate before you read the table)")
    conf = [float(r["confidence"]) for r in sc]
    out = [1.0 if r["pass"] else 0.0 for r in sc]
    base = st.mean(out)
    print(f"  shipped Brier                 {st.mean([(c - o) ** 2 for c, o in zip(conf, out)]):.4f}")
    print(f"  constant base rate ({base:.2f})      {st.mean([(base - o) ** 2 for o in out]):.4f}")
    print(f"  constant 0.50                 {st.mean([(0.5 - o) ** 2 for o in out]):.4f}")
    med = st.median(conf)
    hi = [o for c, o in zip(conf, out) if c >= med]
    lo = [o for c, o in zip(conf, out) if c < med]
    print(f"  PASS | high-conf half {st.mean(hi):.2f} (n={len(hi)})  "
          f"vs low-conf half {st.mean(lo):.2f} (n={len(lo)})  -> ranks, does not calibrate")


def finding_6_timing(sc: list[dict]) -> None:
    head("F6  Pre-open integrity (paper canon 9 and 17)")
    late = [r for r in sc if is_late(r)]
    print(f"  scored rows created at/after the 13:30 UTC open: {len(late)}/{len(sc)}")
    post_close = [r for r in late if (r.get("late_minutes") or 0) > 390]
    for r in sorted(post_close, key=lambda r: r["date"]):
        flat = abs(r["prior_close"] - r["actual_close"]) < 1e-9
        print(f"    {r['date']} created {r['created_at'][11:16]}Z  "
              f"+{r['late_minutes']} min after open (after the close)"
              + ("  prior_close == actual_close" if flat else ""))
    clean = [r for r in sc if not is_late(r)]
    def edge(rs):
        return st.mean([r["baseline_ape"] for r in rs]) - st.mean([r["ape"] for r in rs])
    print(f"  all-time edge, every row      {100 * edge(sc):+.4f}%  (n={len(sc)})")
    print(f"  all-time edge, pre-open only  {100 * edge(clean):+.4f}%  (n={len(clean)})")
    print(f"  rolling-20 edge, every row     {100 * edge(sc[-20:]):+.4f}%")
    print(f"  rolling-20 edge, pre-open only {100 * edge(clean[-20:]):+.4f}%")
    print("  Both effects are ~1e-4 of MAPE — far under this series' day-to-day noise, so")
    print("  neither sign is readable. The record has to be clean before the number means anything.")


def finding_7_redundancy(sc: list[dict]) -> None:
    head("F7  Analyst redundancy (lower than the post-mortems assume)")
    series = {k: [] for k in NAMES}
    for r in sc:
        ap = r.get("analyst_predictions") or {}
        if not all(k in ap and ap[k].get("predicted_close") for k in NAMES):
            continue
        prev = float(r["prior_close"])
        for k in NAMES:
            series[k].append((float(ap[k]["predicted_close"]) - prev) / prev)
    n = len(series[NAMES[0]])
    cs = []
    for a, b in itertools.combinations(NAMES, 2):
        c = st.correlation(series[a], series[b])
        cs.append(c)
    print(f"  n={n} days, mean pairwise correlation of predicted move: {st.mean(cs):+.3f}")
    unanimous = sum(1 for i in range(n) if len({series[k][i] > 0 for k in NAMES}) == 1)
    print(f"  all-five-same-direction days: {unanimous}/{n} = {100 * unanimous / n:.0f}%")


def main() -> int:
    rows_a = scored(load(ROOT / "data" / "predictions.jsonl"))
    rows_b = scored(load(ROOT / "data" / "predictions_b.jsonl"))
    print(f"arm A scored non-seed days: {len(rows_a)}  ({rows_a[0]['date']} -> {rows_a[-1]['date']})")
    print(f"arm B scored days:          {len(rows_b)}")
    finding_1_capture_gap(rows_a)
    finding_1b_paired(rows_a)
    finding_2_conviction_decay(rows_a)
    finding_3_vol_blind(rows_a)
    finding_4_dead_weights(rows_a)
    finding_5_confidence(rows_a)
    finding_6_timing(rows_a)
    finding_7_redundancy(rows_a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
