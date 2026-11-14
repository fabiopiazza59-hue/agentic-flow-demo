# 📈 AMZN Daily Close Predictor — Results

_Auto-generated after every run. Verdicts use pre-open forecasts only and anytime-valid 95% confidence sequences, so they stay valid although this page is re-read daily. PASS = predicted close within ±1% of actual._

## Verdict
❌ **No edge yet** — all-time MAPE 1.63% is nominally ahead of the free baseline 1.71% by 0.08%, but the difference is not distinguishable from noise (paired over 69 pre-open days: mean 0.08%, anytime-valid 95% CS [-0.045%, +0.199%]; fixed-sample 90% CI [-0.12%, 0.14%], sign test p=0.8099).

**Next forecast (2026-11-16):** close ≈ **263.56** (down, P(up) 26%, 80% interval 257.54–269.72) vs prior close 266.66; anchored to 264.33 (yfinance_ext, live, refreshed 0×).

## Scoreboard — all-time, pre-open forecasts only

_Every forecaster is scored as a calibrated predictive distribution (CRPS: lower is better; coverage should match its nominal level). The **free baseline** is the latest pre-open trade the forecast was anchored to — the prior close when there was none._

| Metric | Model (arm A) | Random walk (prior close) | Free baseline (pre-open price) |
|---|---|---|---|
| Days | 69 | 69 | 69 |
| MAPE | 1.63% | 1.65% | 1.71% |
| CRPS | 1.34% | 1.34% | 1.39% |
| 80% interval coverage | 79.71% | 78.26% | 79.71% |
| 50% interval coverage | 53.62% | 50.72% | 50.72% |
| Brier of P(up) | 0.229 | 0.250 | 0.242 |

**Does the model add value?** Paired daily gains, anytime-valid 95% confidence sequences (valid although this page is re-read every day):

| Comparison | Mean daily gain | 95% CS | Verdict |
|---|---|---|---|
| APE vs free baseline | 0.08% | [-0.045%, +0.199%] | ≈ not distinguishable |
| APE vs random walk | 0.02% | [-0.225%, +0.259%] | ≈ not distinguishable |
| CRPS vs free baseline | 0.05% | [-0.028%, +0.132%] | ≈ not distinguishable |
| CRPS vs random walk | -0.00% | [-0.192%, +0.190%] | ≈ not distinguishable |
| Direction (Brier of P(up)) vs random walk | 0.021 | [-0.024, +0.065] | ≈ not distinguishable |

## Self-improvement lab — measured, not narrated

Aggregation rules are replayed over every pre-open day using only earlier data. A rule replaces the equal-weight mean only when its anytime-valid confidence sequence (Bonferroni over 5 challengers, α=0.05) shows it is better; the judge and the gates are switched off the same way once shown to hurt ([`evals/lab.py`](src/evals/lab.py)).

**Champion: `mean`** (reference — no challenger has proven better yet; 69 days replayed).

| Rule | MAPE | Gain vs mean | CS (α/K) | Status |
|---|---|---|---|---|
| mean (reference) | 1.56% | — | — | 🏆 champion |
| median | 1.59% | -0.03% | [-0.160%, +0.098%] | ≈ not distinguishable |
| trimmed_mean | 1.59% | -0.03% | [-0.126%, +0.060%] | ≈ not distinguishable |
| inverse_mse | 1.58% | -0.02% | [-0.057%, +0.024%] | ≈ not distinguishable |
| best_recent | 1.79% | -0.23% | [-0.464%, +0.008%] | ≈ not distinguishable |
| shrink_half | 1.63% | -0.07% | [-0.160%, +0.020%] | ≈ not distinguishable |

- **Judge adjustment** (APE of blend − APE after the judge's clamped σ-move; v2 rows): n=35, mean 0.03%, CS [-0.049%, +0.101%] → judge **on**.
- **Guardrail gates** (APE before − after the gates): n=35, mean -0.00%, CS [-0.025%, +0.021%] → gates **on**.
- **Legacy v1 pipeline** (judge wrote the number) vs the plain mean of its own analysts: mean -0.17% over 34 pre-open days, CS [-0.407%, +0.059%] — ≈ not distinguishable (negative = the judge cost accuracy).

### Integrity & mechanism

- **40 scored row(s) were created after their session opened** and are excluded from every verdict. They saw part of the tape they forecast. New post-open rows are refused (`--allow-late` to override), and the schedule now researches the evening before and only re-anchors in the morning.
- **Pre-open coverage:** 20 of the last 20 sessions (since 2026-10-20) got a forecast written before the open.
- **Live anchors:** 35 of 69 pre-open forecasts were anchored on a real after-hours/pre-market trade (the rest on the prior close — before v2 the quote feed silently echoed it).
- **Gate effect** (ungated − gated APE, paired over 35 days): -0.00% 90% CI [-0.01%, 0.01%], gates helped on 5/9 decisive days (sign test p=1.0). Gates fired on 9 of them.

## Metrics

| Metric | All-time (pre-open) — verdict basis | Last 20 (all rows) | All-time (all rows) |
|---|---|---|---|
| Scored days | 69 | 20 | 109 |
| PASS rate (±1%) | 42.03% | 45.00% | 44.04% |
| Directional accuracy | 59.42% | 75.00% | 57.80% |
| MAPE | 1.63% | 1.95% | 1.55% |
| Baseline MAPE (random walk) | 1.65% | 1.93% | 1.55% |
| Edge (baseline − model) | 0.02% | -0.03% | 0.00% |
| Brier (confidence calib.) | 0.24 | 0.25 | 0.26 |

## Per-strategy scorecards

| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |
|---|---|---|---|---|
| news | 109 | 31.19% | 1.41% | 0.22 |
| technical | 109 | 11.01% | 1.54% | 0.20 |
| contrarian | 109 | 27.52% | 1.60% | 0.20 |
| momentum | 109 | 17.43% | 1.67% | 0.19 |
| macro | 109 | 12.84% | 1.70% | 0.19 |

## Last 20 scored model predictions
_(backfill seed rows are excluded from metrics and this table; they appear only as price-history context on the dashboard chart. ⚠️ = written after the open, excluded from verdicts)_

| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest | Pre-open |
|---|---|---|---|---|---|---|---|---|
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
| 2026-10-29 | 252.81 | 243.79 | 3.70% | ❌ | ✅ | ✅ | news | ✅ |
| 2026-10-28 | 254.90 | 254.60 | 0.12% | ✅ | ✅ | ✅ | news | ✅ |
| 2026-10-27 | 250.33 | 253.10 | 1.09% | ❌ | ✅ | ✅ | contrarian | ✅ |
| 2026-10-26 | 248.61 | 249.09 | 0.19% | ✅ | ✅ | ✅ | news | ✅ |
| 2026-10-23 | 250.35 | 248.00 | 0.95% | ✅ | ✅ | ✅ | news | ✅ |
| 2026-10-22 | 261.61 | 250.38 | 4.49% | ❌ | ✅ | ✅ | contrarian | ✅ |
| 2026-10-21 | 262.64 | 262.37 | 0.10% | ✅ | ❌ | ❌ | news | ✅ |
| 2026-10-20 | 262.53 | 262.63 | 0.04% | ✅ | ✅ | ✅ | news | ✅ |
| 2026-10-19 | 248.10 | 262.45 | 5.47% | ❌ | ❌ | ❌ | contrarian | ✅ |

## 🅰️/🅱️ A/B test — ensemble+gates vs raven-style prior+pulse

_Both arms forecast the same sessions from the same pre-open snapshot (paired test). Only days on which both forecasts were written before the open count._

**Verdict:** Arm A leads on paired MAPE (mean daily delta 0.07% in A's favor over 61 pre-open days); B wins 29/61 decisive days (sign test p=0.7982, descriptive); anytime-valid 95% CS [-0.269%, +0.127%] includes zero — not distinguishable from noise.

**B's next forecast (2026-11-16):** close ≈ **265.03** (down, adj 0.1σ, 8 evidence items).

| All-time, pre-open | A — ensemble+gates | B — raven prior+pulse |
|---|---|---|
| Scored days | 69 | 61 |
| PASS rate (±1%) | 42.03% | 44.26% |
| Directional accuracy | 59.42% | 60.66% |
| MAPE | 1.63% | 1.79% |
| Edge vs free baseline | 0.08% | 0.00% |

Clean paired days: 61 (of 84 paired); B wins 29/61 decisive; mean daily APE delta (A−B) -0.07%; CRPS delta (A−B) anytime-valid 95% CS [-0.178%, +0.088%] (undecided).
_This is a research experiment, not financial advice._
