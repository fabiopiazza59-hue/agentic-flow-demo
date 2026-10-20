# 📈 AMZN Daily Close Predictor — Results

_Auto-generated after every run. Verdicts use pre-open forecasts only and anytime-valid 95% confidence sequences, so they stay valid although this page is re-read daily. PASS = predicted close within ±1% of actual._

## Verdict
❌ **No edge yet** — all-time MAPE 1.58% is nominally ahead of the free baseline 1.61% by 0.02%, but the difference is not distinguishable from noise (paired over 50 pre-open days: mean 0.02%, anytime-valid 95% CS [-0.095%, +0.140%]; fixed-sample 90% CI [-0.07%, 0.11%], sign test p=0.6718).

**Next forecast (2026-10-20):** close ≈ **262.53** (up, P(up) 51%, 80% interval 257.66–267.49) vs prior close 262.45; anchored to 263.20 (yfinance_ext, live, refreshed 0×).

## Scoreboard — all-time, pre-open forecasts only

_Every forecaster is scored as a calibrated predictive distribution (CRPS: lower is better; coverage should match its nominal level). The **free baseline** is the latest pre-open trade the forecast was anchored to — the prior close when there was none._

| Metric | Model (arm A) | Random walk (prior close) | Free baseline (pre-open price) |
|---|---|---|---|
| Days | 50 | 50 | 50 |
| MAPE | 1.58% | 1.60% | 1.61% |
| CRPS | 1.30% | 1.31% | 1.31% |
| 80% interval coverage | 84.00% | 80.00% | 82.00% |
| 50% interval coverage | 52.00% | 52.00% | 52.00% |
| Brier of P(up) | 0.245 | 0.250 | 0.253 |

**Does the model add value?** Paired daily gains, anytime-valid 95% confidence sequences (valid although this page is re-read every day):

| Comparison | Mean daily gain | 95% CS | Verdict |
|---|---|---|---|
| APE vs free baseline | 0.02% | [-0.095%, +0.140%] | ≈ not distinguishable |
| APE vs random walk | 0.02% | [-0.147%, +0.186%] | ≈ not distinguishable |
| CRPS vs free baseline | 0.01% | [-0.051%, +0.077%] | ≈ not distinguishable |
| CRPS vs random walk | 0.02% | [-0.087%, +0.122%] | ≈ not distinguishable |
| Direction (Brier of P(up)) vs random walk | 0.005 | [-0.045, +0.054] | ≈ not distinguishable |

## Self-improvement lab — measured, not narrated

Aggregation rules are replayed over every pre-open day using only earlier data. A rule replaces the equal-weight mean only when its anytime-valid confidence sequence (Bonferroni over 5 challengers, α=0.05) shows it is better; the judge and the gates are switched off the same way once shown to hurt ([`evals/lab.py`](src/evals/lab.py)).

**Champion: `mean`** (reference — no challenger has proven better yet; 50 days replayed).

| Rule | MAPE | Gain vs mean | CS (α/K) | Status |
|---|---|---|---|---|
| mean (reference) | 1.46% | — | — | 🏆 champion |
| median | 1.49% | -0.03% | [-0.191%, +0.127%] | ≈ not distinguishable |
| trimmed_mean | 1.49% | -0.04% | [-0.154%, +0.082%] | ≈ not distinguishable |
| inverse_mse | 1.49% | -0.03% | [-0.081%, +0.028%] | ≈ not distinguishable |
| best_recent | 1.70% | -0.24% | [-0.529%, +0.046%] | ≈ not distinguishable |
| shrink_half | 1.53% | -0.07% | [-0.184%, +0.049%] | ≈ not distinguishable |

- **Judge adjustment** (APE of blend − APE after the judge's clamped σ-move; v2 rows): n=16, mean -0.02%, CS [-0.070%, +0.040%] → judge **on**.
- **Guardrail gates** (APE before − after the gates): n=16, mean -0.01%, CS [-0.063%, +0.045%] → gates **on**.
- **Legacy v1 pipeline** (judge wrote the number) vs the plain mean of its own analysts: mean -0.17% over 34 pre-open days, CS [-0.407%, +0.059%] — ≈ not distinguishable (negative = the judge cost accuracy).

### Integrity & mechanism

- **40 scored row(s) were created after their session opened** and are excluded from every verdict. They saw part of the tape they forecast. New post-open rows are refused (`--allow-late` to override), and the schedule now researches the evening before and only re-anchors in the morning.
- **Pre-open coverage:** 17 of the last 20 sessions (since 2026-09-23) got a forecast written before the open.
- **Live anchors:** 16 of 50 pre-open forecasts were anchored on a real after-hours/pre-market trade (the rest on the prior close — before v2 the quote feed silently echoed it).
- **Gate effect** (ungated − gated APE, paired over 16 days): -0.01% 90% CI [-0.03%, 0.01%], gates helped on 4/8 decisive days (sign test p=1.0). Gates fired on 8 of them.

## Metrics

| Metric | All-time (pre-open) — verdict basis | Last 20 (all rows) | All-time (all rows) |
|---|---|---|---|
| Scored days | 50 | 20 | 90 |
| PASS rate (±1%) | 40.00% | 40.00% | 43.33% |
| Directional accuracy | 52.00% | 55.00% | 53.33% |
| MAPE | 1.58% | 1.31% | 1.50% |
| Baseline MAPE (random walk) | 1.60% | 1.36% | 1.50% |
| Edge (baseline − model) | 0.02% | 0.04% | 0.00% |
| Brier (confidence calib.) | 0.24 | 0.29 | 0.26 |

## Per-strategy scorecards

| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |
|---|---|---|---|---|
| news | 90 | 30.00% | 1.33% | 0.23 |
| technical | 90 | 13.33% | 1.46% | 0.21 |
| contrarian | 90 | 25.56% | 1.52% | 0.20 |
| momentum | 90 | 18.89% | 1.61% | 0.19 |
| macro | 90 | 12.22% | 1.67% | 0.18 |

## Last 20 scored model predictions
_(backfill seed rows are excluded from metrics and this table; they appear only as price-history context on the dashboard chart. ⚠️ = written after the open, excluded from verdicts)_

| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest | Pre-open |
|---|---|---|---|---|---|---|---|---|
| 2026-10-19 | 248.10 | 262.45 | 5.47% | ❌ | ❌ | ❌ | contrarian | ✅ |
| 2026-10-16 | 252.88 | 250.00 | 1.15% | ❌ | ✅ | ✅ | technical | ✅ |
| 2026-10-15 | 250.82 | 254.76 | 1.55% | ❌ | ❌ | ❌ | contrarian | ✅ |
| 2026-10-14 | 257.27 | 254.08 | 1.26% | ❌ | ✅ | ✅ | contrarian | ✅ |
| 2026-10-13 | 258.70 | 258.58 | 0.05% | ✅ | ✅ | ✅ | macro | ✅ |
| 2026-10-12 | 262.39 | 259.53 | 1.10% | ❌ | ✅ | ✅ | contrarian | ✅ |
| 2026-10-09 | 254.94 | 262.43 | 2.85% | ❌ | ✅ | ✅ | macro | ✅ |
| 2026-10-08 | 260.24 | 254.06 | 2.43% | ❌ | ❌ | ❌ | contrarian | ✅ |
| 2026-10-07 | 257.26 | 259.92 | 1.02% | ❌ | ✅ | ✅ | momentum | ✅ |
| 2026-10-06 | 252.40 | 256.29 | 1.52% | ❌ | ✅ | ✅ | news | ✅ |
| 2026-10-05 | 251.67 | 251.40 | 0.11% | ✅ | ❌ | ❌ | contrarian | ✅ |
| 2026-10-02 | 249.13 | 251.52 | 0.95% | ✅ | ✅ | ✅ | news | ✅ |
| 2026-10-01 | 250.26 | 248.23 | 0.82% | ✅ | ❌ | ❌ | contrarian | ✅ |
| 2026-09-30 | 247.28 | 249.15 | 0.75% | ✅ | ✅ | ✅ | contrarian | ✅ |
| 2026-09-29 | 246.02 | 246.67 | 0.26% | ✅ | ❌ | ❌ | news | ✅ |
| 2026-09-28 | 249.91 | 246.15 | 1.53% | ❌ | ❌ | ❌ | news | ✅ |
| 2026-09-25 | 249.05 | 249.67 | 0.25% | ✅ | ❌ | ❌ | contrarian | ⚠️ |
| 2026-09-24 | 249.36 | 249.38 | 0.01% | ✅ | ✅ | ✅ | contrarian | ⚠️ |
| 2026-09-23 | 252.85 | 249.27 | 1.44% | ❌ | ✅ | ✅ | news | ⚠️ |
| 2026-09-22 | 259.42 | 254.98 | 1.74% | ❌ | ❌ | ❌ | contrarian | ⚠️ |

## 🅰️/🅱️ A/B test — ensemble+gates vs raven-style prior+pulse

_Both arms forecast the same sessions from the same pre-open snapshot (paired test). Only days on which both forecasts were written before the open count._

**Verdict:** Arm B leads on paired MAPE (mean daily delta 0.03% in B's favor over 42 pre-open days); B wins 21/42 decisive days (sign test p=1.0, descriptive); anytime-valid 95% CS [-0.191%, +0.244%] includes zero — not distinguishable from noise.

**B's next forecast (2026-10-20):** close ≈ **264.17** (up, adj 0.2σ, 8 evidence items).

| All-time, pre-open | A — ensemble+gates | B — raven prior+pulse |
|---|---|---|
| Scored days | 50 | 42 |
| PASS rate (±1%) | 40.00% | 50.00% |
| Directional accuracy | 52.00% | 59.52% |
| MAPE | 1.58% | 1.68% |
| Edge vs free baseline | 0.02% | 0.03% |

Clean paired days: 42 (of 65 paired); B wins 21/42 decisive; mean daily APE delta (A−B) 0.03%; CRPS delta (A−B) anytime-valid 95% CS [-0.130%, +0.150%] (undecided).
_This is a research experiment, not financial advice._
