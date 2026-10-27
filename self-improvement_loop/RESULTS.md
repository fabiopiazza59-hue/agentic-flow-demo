# 📈 AMZN Daily Close Predictor — Results

_Auto-generated after every run. Verdicts use pre-open forecasts only and anytime-valid 95% confidence sequences, so they stay valid although this page is re-read daily. PASS = predicted close within ±1% of actual._

## Verdict
❌ **No edge yet** — all-time MAPE 1.55% is nominally ahead of the free baseline 1.58% by 0.03%, but the difference is not distinguishable from noise (paired over 55 pre-open days: mean 0.03%, anytime-valid 95% CS [-0.078%, +0.138%]; fixed-sample 90% CI [-0.05%, 0.11%], sign test p=1.0).

**Next forecast (2026-10-27):** close ≈ **250.33** (up, P(up) 65%, 80% interval 245.62–255.13) vs prior close 249.09; anchored to 249.45 (yfinance_ext, live, refreshed 0×).

## Scoreboard — all-time, pre-open forecasts only

_Every forecaster is scored as a calibrated predictive distribution (CRPS: lower is better; coverage should match its nominal level). The **free baseline** is the latest pre-open trade the forecast was anchored to — the prior close when there was none._

| Metric | Model (arm A) | Random walk (prior close) | Free baseline (pre-open price) |
|---|---|---|---|
| Days | 55 | 55 | 55 |
| MAPE | 1.55% | 1.57% | 1.58% |
| CRPS | 1.28% | 1.30% | 1.29% |
| 80% interval coverage | 83.64% | 80.00% | 81.82% |
| 50% interval coverage | 54.55% | 54.55% | 54.55% |
| Brier of P(up) | 0.242 | 0.250 | 0.251 |

**Does the model add value?** Paired daily gains, anytime-valid 95% confidence sequences (valid although this page is re-read every day):

| Comparison | Mean daily gain | 95% CS | Verdict |
|---|---|---|---|
| APE vs free baseline | 0.03% | [-0.078%, +0.138%] | ≈ not distinguishable |
| APE vs random walk | 0.03% | [-0.123%, +0.180%] | ≈ not distinguishable |
| CRPS vs free baseline | 0.01% | [-0.044%, +0.071%] | ≈ not distinguishable |
| CRPS vs random walk | 0.02% | [-0.075%, +0.119%] | ≈ not distinguishable |
| Direction (Brier of P(up)) vs random walk | 0.008 | [-0.038, +0.053] | ≈ not distinguishable |

## Self-improvement lab — measured, not narrated

Aggregation rules are replayed over every pre-open day using only earlier data. A rule replaces the equal-weight mean only when its anytime-valid confidence sequence (Bonferroni over 5 challengers, α=0.05) shows it is better; the judge and the gates are switched off the same way once shown to hurt ([`evals/lab.py`](src/evals/lab.py)).

**Champion: `mean`** (reference — no challenger has proven better yet; 55 days replayed).

| Rule | MAPE | Gain vs mean | CS (α/K) | Status |
|---|---|---|---|---|
| mean (reference) | 1.44% | — | — | 🏆 champion |
| median | 1.48% | -0.04% | [-0.197%, +0.108%] | ≈ not distinguishable |
| trimmed_mean | 1.48% | -0.05% | [-0.155%, +0.064%] | ≈ not distinguishable |
| inverse_mse | 1.46% | -0.02% | [-0.071%, +0.030%] | ≈ not distinguishable |
| best_recent | 1.68% | -0.24% | [-0.515%, +0.032%] | ≈ not distinguishable |
| shrink_half | 1.50% | -0.06% | [-0.168%, +0.043%] | ≈ not distinguishable |

- **Judge adjustment** (APE of blend − APE after the judge's clamped σ-move; v2 rows): n=21, mean 0.00%, CS [-0.053%, +0.061%] → judge **on**.
- **Guardrail gates** (APE before − after the gates): n=21, mean -0.00%, CS [-0.044%, +0.037%] → gates **on**.
- **Legacy v1 pipeline** (judge wrote the number) vs the plain mean of its own analysts: mean -0.17% over 34 pre-open days, CS [-0.407%, +0.059%] — ≈ not distinguishable (negative = the judge cost accuracy).

### Integrity & mechanism

- **40 scored row(s) were created after their session opened** and are excluded from every verdict. They saw part of the tape they forecast. New post-open rows are refused (`--allow-late` to override), and the schedule now researches the evening before and only re-anchors in the morning.
- **Pre-open coverage:** 20 of the last 20 sessions (since 2026-09-30) got a forecast written before the open.
- **Live anchors:** 21 of 55 pre-open forecasts were anchored on a real after-hours/pre-market trade (the rest on the prior close — before v2 the quote feed silently echoed it).
- **Gate effect** (ungated − gated APE, paired over 21 days): -0.00% 90% CI [-0.02%, 0.01%], gates helped on 5/9 decisive days (sign test p=1.0). Gates fired on 9 of them.

## Metrics

| Metric | All-time (pre-open) — verdict basis | Last 20 (all rows) | All-time (all rows) |
|---|---|---|---|
| Scored days | 55 | 20 | 95 |
| PASS rate (±1%) | 43.64% | 50.00% | 45.26% |
| Directional accuracy | 54.55% | 65.00% | 54.74% |
| MAPE | 1.55% | 1.35% | 1.48% |
| Baseline MAPE (random walk) | 1.57% | 1.41% | 1.49% |
| Edge (baseline − model) | 0.03% | 0.06% | 0.01% |
| Brier (confidence calib.) | 0.25 | 0.28 | 0.27 |

## Per-strategy scorecards

| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |
|---|---|---|---|---|
| news | 95 | 32.63% | 1.32% | 0.23 |
| technical | 95 | 12.63% | 1.47% | 0.21 |
| contrarian | 95 | 25.26% | 1.53% | 0.20 |
| momentum | 95 | 17.89% | 1.60% | 0.19 |
| macro | 95 | 11.58% | 1.66% | 0.18 |

## Last 20 scored model predictions
_(backfill seed rows are excluded from metrics and this table; they appear only as price-history context on the dashboard chart. ⚠️ = written after the open, excluded from verdicts)_

| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest | Pre-open |
|---|---|---|---|---|---|---|---|---|
| 2026-10-26 | 248.61 | 249.09 | 0.19% | ✅ | ✅ | ✅ | news | ✅ |
| 2026-10-23 | 250.35 | 248.00 | 0.95% | ✅ | ✅ | ✅ | news | ✅ |
| 2026-10-22 | 261.61 | 250.38 | 4.49% | ❌ | ✅ | ✅ | contrarian | ✅ |
| 2026-10-21 | 262.64 | 262.37 | 0.10% | ✅ | ❌ | ❌ | news | ✅ |
| 2026-10-20 | 262.53 | 262.63 | 0.04% | ✅ | ✅ | ✅ | news | ✅ |
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

## 🅰️/🅱️ A/B test — ensemble+gates vs raven-style prior+pulse

_Both arms forecast the same sessions from the same pre-open snapshot (paired test). Only days on which both forecasts were written before the open count._

**Verdict:** Arm B leads on paired MAPE (mean daily delta 0.02% in B's favor over 47 pre-open days); B wins 24/47 decisive days (sign test p=1.0, descriptive); anytime-valid 95% CS [-0.187%, +0.219%] includes zero — not distinguishable from noise.

**B's next forecast (2026-10-27):** close ≈ **250.21** (up, adj 0.15σ, 8 evidence items).

| All-time, pre-open | A — ensemble+gates | B — raven prior+pulse |
|---|---|---|
| Scored days | 55 | 47 |
| PASS rate (±1%) | 43.64% | 53.19% |
| Directional accuracy | 54.55% | 61.70% |
| MAPE | 1.55% | 1.63% |
| Edge vs free baseline | 0.03% | 0.03% |

Clean paired days: 47 (of 70 paired); B wins 24/47 decisive; mean daily APE delta (A−B) 0.02%; CRPS delta (A−B) anytime-valid 95% CS [-0.111%, +0.139%] (undecided).
_This is a research experiment, not financial advice._
