# 📈 AMZN Daily Close Predictor — Results

_Auto-generated after every run. Verdicts use pre-open forecasts only and anytime-valid 95% confidence sequences, so they stay valid although this page is re-read daily. PASS = predicted close within ±1% of actual._

## Verdict
❌ **No edge yet** — all-time MAPE 1.71% is nominally ahead of the free baseline 1.78% by 0.07%, but the difference is not distinguishable from noise (paired over 78 pre-open days: mean 0.07%, anytime-valid 95% CS [-0.036%, +0.180%]; fixed-sample 90% CI [-0.04%, 0.24%], sign test p=0.4282).

**Next forecast (2026-11-30):** close ≈ **255.05** (up, P(up) 58%, 80% interval 247.70–262.62) vs prior close 254.12; anchored to 255.05 (yfinance_ext, live, refreshed 4×).

## Scoreboard — all-time, pre-open forecasts only

_Every forecaster is scored as a calibrated predictive distribution (CRPS: lower is better; coverage should match its nominal level). The **free baseline** is the latest pre-open trade the forecast was anchored to — the prior close when there was none._

| Metric | Model (arm A) | Random walk (prior close) | Free baseline (pre-open price) |
|---|---|---|---|
| Days | 78 | 78 | 78 |
| MAPE | 1.71% | 1.81% | 1.78% |
| CRPS | 1.39% | 1.44% | 1.43% |
| 80% interval coverage | 76.92% | 71.79% | 76.92% |
| 50% interval coverage | 51.28% | 47.44% | 50.00% |
| Brier of P(up) | 0.224 | 0.250 | 0.236 |

**Does the model add value?** Paired daily gains, anytime-valid 95% confidence sequences (valid although this page is re-read every day):

| Comparison | Mean daily gain | 95% CS | Verdict |
|---|---|---|---|
| APE vs free baseline | 0.07% | [-0.036%, +0.180%] | ≈ not distinguishable |
| APE vs random walk | 0.10% | [-0.168%, +0.371%] | ≈ not distinguishable |
| CRPS vs free baseline | 0.05% | [-0.025%, +0.119%] | ≈ not distinguishable |
| CRPS vs random walk | 0.05% | [-0.148%, +0.253%] | ≈ not distinguishable |
| Direction (Brier of P(up)) vs random walk | 0.026 | [-0.019, +0.072] | ≈ not distinguishable |

## Self-improvement lab — measured, not narrated

Aggregation rules are replayed over every pre-open day using only earlier data. A rule replaces the equal-weight mean only when its anytime-valid confidence sequence (Bonferroni over 5 challengers, α=0.05) shows it is better; the judge and the gates are switched off the same way once shown to hurt ([`evals/lab.py`](src/evals/lab.py)).

**Champion: `mean`** (reference — no challenger has proven better yet; 70 days replayed).

| Rule | MAPE | Gain vs mean | CS (α/K) | Status |
|---|---|---|---|---|
| mean (reference) | 1.57% | — | — | 🏆 champion |
| median | 1.60% | -0.03% | [-0.157%, +0.097%] | ≈ not distinguishable |
| trimmed_mean | 1.60% | -0.03% | [-0.124%, +0.059%] | ≈ not distinguishable |
| inverse_mse | 1.59% | -0.02% | [-0.056%, +0.024%] | ≈ not distinguishable |
| best_recent | 1.80% | -0.22% | [-0.457%, +0.008%] | ≈ not distinguishable |
| shrink_half | 1.64% | -0.07% | [-0.160%, +0.017%] | ≈ not distinguishable |

- **Judge adjustment** (APE of blend − APE after the judge's clamped σ-move; v2 rows): n=44, mean 0.02%, CS [-0.038%, +0.079%] → judge **on**.
- **Guardrail gates** (APE before − after the gates): n=44, mean -0.00%, CS [-0.019%, +0.016%] → gates **on**.
- **Legacy v1 pipeline** (judge wrote the number) vs the plain mean of its own analysts: mean -0.17% over 34 pre-open days, CS [-0.407%, +0.059%] — ≈ not distinguishable (negative = the judge cost accuracy).

### Integrity & mechanism

- **40 scored row(s) were created after their session opened** and are excluded from every verdict. They saw part of the tape they forecast. New post-open rows are refused (`--allow-late` to override), and the schedule now researches the evening before and only re-anchors in the morning.
- **Pre-open coverage:** 20 of the last 20 sessions (since 2026-11-02) got a forecast written before the open.
- **Live anchors:** 44 of 78 pre-open forecasts were anchored on a real after-hours/pre-market trade (the rest on the prior close — before v2 the quote feed silently echoed it).
- **Gate effect** (ungated − gated APE, paired over 44 days): -0.00% 90% CI [-0.01%, 0.01%], gates helped on 5/9 decisive days (sign test p=1.0). Gates fired on 13 of them.

## Metrics

| Metric | All-time (pre-open) — verdict basis | Last 20 (all rows) | All-time (all rows) |
|---|---|---|---|
| Scored days | 78 | 20 | 118 |
| PASS rate (±1%) | 41.03% | 35.00% | 43.22% |
| Directional accuracy | 61.54% | 75.00% | 59.32% |
| MAPE | 1.71% | 2.16% | 1.60% |
| Baseline MAPE (random walk) | 1.81% | 2.40% | 1.66% |
| Edge (baseline − model) | 0.10% | 0.23% | 0.06% |
| Brier (confidence calib.) | 0.24 | 0.24 | 0.26 |

## Per-strategy scorecards

| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |
|---|---|---|---|---|
| news | 110 | 30.91% | 1.41% | 0.22 |
| technical | 110 | 10.91% | 1.55% | 0.20 |
| contrarian | 110 | 27.27% | 1.61% | 0.20 |
| momentum | 110 | 17.27% | 1.68% | 0.19 |
| macro | 110 | 13.64% | 1.71% | 0.19 |

## Last 20 scored model predictions
_(backfill seed rows are excluded from metrics and this table; they appear only as price-history context on the dashboard chart. ⚠️ = written after the open, excluded from verdicts)_

| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest | Pre-open |
|---|---|---|---|---|---|---|---|---|
| 2026-11-27 | 246.77 | 254.12 | 2.89% | ❌ | ✅ | ✅ | — | ✅ |
| 2026-11-25 | 244.77 | 244.43 | 0.14% | ✅ | ✅ | ✅ | — | ✅ |
| 2026-11-24 | 253.64 | 252.29 | 0.54% | ✅ | ✅ | ✅ | — | ✅ |
| 2026-11-23 | 251.00 | 255.39 | 1.72% | ❌ | ✅ | ✅ | — | ✅ |
| 2026-11-20 | 256.37 | 246.82 | 3.87% | ❌ | ✅ | ✅ | — | ✅ |
| 2026-11-19 | 271.32 | 257.37 | 5.42% | ❌ | ❌ | ❌ | — | ✅ |
| 2026-11-18 | 260.20 | 267.74 | 2.82% | ❌ | ✅ | ✅ | — | ✅ |
| 2026-11-17 | 256.01 | 257.89 | 0.73% | ✅ | ❌ | ❌ | — | ✅ |
| 2026-11-16 | 263.56 | 257.75 | 2.25% | ❌ | ✅ | ✅ | macro | ✅ |
| 2026-11-13 | 268.46 | 266.66 | 0.68% | ✅ | ❌ | ❌ | contrarian | ✅ |
| 2026-11-12 | 267.03 | 267.44 | 0.15% | ✅ | ✅ | ✅ | macro | ✅ |
| 2026-11-11 | 273.14 | 266.17 | 2.62% | ❌ | ❌ | ❌ | contrarian | ✅ |
| 2026-11-10 | 267.13 | 270.87 | 1.38% | ❌ | ✅ | ✅ | momentum | ✅ |
| 2026-11-09 | 260.31 | 266.93 | 2.48% | ❌ | ✅ | ✅ | momentum | ✅ |
| 2026-11-06 | 257.15 | 259.30 | 0.83% | ✅ | ✅ | ❌ | macro | ✅ |
| 2026-11-05 | 262.97 | 260.21 | 1.06% | ❌ | ✅ | ✅ | news | ✅ |
| 2026-11-04 | 248.22 | 266.03 | 6.69% | ❌ | ❌ | ❌ | contrarian | ✅ |
| 2026-11-03 | 248.00 | 250.69 | 1.07% | ❌ | ✅ | ✅ | contrarian | ✅ |
| 2026-11-02 | 243.96 | 245.63 | 0.68% | ✅ | ✅ | ✅ | contrarian | ✅ |
| 2026-10-30 | 227.77 | 240.35 | 5.23% | ❌ | ✅ | ❌ | macro | ✅ |

## 🅰️/🅱️ A/B test — ensemble+gates vs raven-style prior+pulse

_Both arms forecast the same sessions from the same pre-open snapshot (paired test). Only days on which both forecasts were written before the open count._

**Verdict:** Arm A leads on paired MAPE (mean daily delta 0.07% in A's favor over 70 pre-open days); B wins 29/62 decisive days (sign test p=0.7035, descriptive); anytime-valid 95% CS [-0.243%, +0.103%] includes zero — not distinguishable from noise.

**B's next forecast (2026-11-30):** close ≈ **255.05** (up, adj 0.0σ, 0 evidence items).

| All-time, pre-open | A — ensemble+gates | B — raven prior+pulse |
|---|---|---|
| Scored days | 78 | 70 |
| PASS rate (±1%) | 41.03% | 42.86% |
| Directional accuracy | 61.54% | 62.86% |
| MAPE | 1.71% | 1.86% |
| Edge vs free baseline | 0.07% | -0.00% |

Clean paired days: 70 (of 93 paired); B wins 29/62 decisive; mean daily APE delta (A−B) -0.07%; CRPS delta (A−B) anytime-valid 95% CS [-0.163%, +0.072%] (undecided).
_This is a research experiment, not financial advice._
