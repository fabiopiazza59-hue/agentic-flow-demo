# 📈 AMZN Daily Close Predictor — Results

_Auto-generated after every run. Verdicts use pre-open forecasts only and anytime-valid 95% confidence sequences, so they stay valid although this page is re-read daily. PASS = predicted close within ±1% of actual._

## Verdict
❌ **No edge yet** — all-time MAPE 1.61% is nominally ahead of the free baseline 1.67% by 0.06%, but the difference is not distinguishable from noise (paired over 59 pre-open days: mean 0.06%, anytime-valid 95% CS [-0.053%, +0.167%]; fixed-sample 90% CI [-0.16%, 0.11%], sign test p=1.0).

**Next forecast (2026-11-02):** close ≈ **243.96** (up, P(up) 83%, 80% interval 239.16–248.85) vs prior close 240.35; anchored to 240.80 (yfinance_ext, live, refreshed 0×).

## Scoreboard — all-time, pre-open forecasts only

_Every forecaster is scored as a calibrated predictive distribution (CRPS: lower is better; coverage should match its nominal level). The **free baseline** is the latest pre-open trade the forecast was anchored to — the prior close when there was none._

| Metric | Model (arm A) | Random walk (prior close) | Free baseline (pre-open price) |
|---|---|---|---|
| Days | 59 | 59 | 59 |
| MAPE | 1.61% | 1.60% | 1.67% |
| CRPS | 1.33% | 1.32% | 1.37% |
| 80% interval coverage | 81.36% | 79.66% | 79.66% |
| 50% interval coverage | 52.54% | 52.54% | 52.54% |
| Brier of P(up) | 0.232 | 0.250 | 0.243 |

**Does the model add value?** Paired daily gains, anytime-valid 95% confidence sequences (valid although this page is re-read every day):

| Comparison | Mean daily gain | 95% CS | Verdict |
|---|---|---|---|
| APE vs free baseline | 0.06% | [-0.053%, +0.167%] | ≈ not distinguishable |
| APE vs random walk | -0.01% | [-0.260%, +0.242%] | ≈ not distinguishable |
| CRPS vs free baseline | 0.04% | [-0.036%, +0.118%] | ≈ not distinguishable |
| CRPS vs random walk | -0.02% | [-0.223%, +0.190%] | ≈ not distinguishable |
| Direction (Brier of P(up)) vs random walk | 0.018 | [-0.027, +0.064] | ≈ not distinguishable |

## Self-improvement lab — measured, not narrated

Aggregation rules are replayed over every pre-open day using only earlier data. A rule replaces the equal-weight mean only when its anytime-valid confidence sequence (Bonferroni over 5 challengers, α=0.05) shows it is better; the judge and the gates are switched off the same way once shown to hurt ([`evals/lab.py`](src/evals/lab.py)).

**Champion: `mean`** (reference — no challenger has proven better yet; 59 days replayed).

| Rule | MAPE | Gain vs mean | CS (α/K) | Status |
|---|---|---|---|---|
| mean (reference) | 1.52% | — | — | 🏆 champion |
| median | 1.56% | -0.05% | [-0.187%, +0.097%] | ≈ not distinguishable |
| trimmed_mean | 1.56% | -0.04% | [-0.143%, +0.062%] | ≈ not distinguishable |
| inverse_mse | 1.54% | -0.02% | [-0.065%, +0.029%] | ≈ not distinguishable |
| best_recent | 1.73% | -0.21% | [-0.473%, +0.050%] | ≈ not distinguishable |
| shrink_half | 1.59% | -0.07% | [-0.169%, +0.031%] | ≈ not distinguishable |

- **Judge adjustment** (APE of blend − APE after the judge's clamped σ-move; v2 rows): n=25, mean 0.02%, CS [-0.040%, +0.080%] → judge **on**.
- **Guardrail gates** (APE before − after the gates): n=25, mean -0.00%, CS [-0.036%, +0.030%] → gates **on**.
- **Legacy v1 pipeline** (judge wrote the number) vs the plain mean of its own analysts: mean -0.17% over 34 pre-open days, CS [-0.407%, +0.059%] — ≈ not distinguishable (negative = the judge cost accuracy).

### Integrity & mechanism

- **40 scored row(s) were created after their session opened** and are excluded from every verdict. They saw part of the tape they forecast. New post-open rows are refused (`--allow-late` to override), and the schedule now researches the evening before and only re-anchors in the morning.
- **Pre-open coverage:** 20 of the last 20 sessions (since 2026-10-06) got a forecast written before the open.
- **Live anchors:** 25 of 59 pre-open forecasts were anchored on a real after-hours/pre-market trade (the rest on the prior close — before v2 the quote feed silently echoed it).
- **Gate effect** (ungated − gated APE, paired over 25 days): -0.00% 90% CI [-0.02%, 0.01%], gates helped on 5/9 decisive days (sign test p=1.0). Gates fired on 9 of them.

## Metrics

| Metric | All-time (pre-open) — verdict basis | Last 20 (all rows) | All-time (all rows) |
|---|---|---|---|
| Scored days | 59 | 20 | 99 |
| PASS rate (±1%) | 42.37% | 35.00% | 44.44% |
| Directional accuracy | 57.63% | 75.00% | 56.57% |
| MAPE | 1.61% | 1.72% | 1.52% |
| Baseline MAPE (random walk) | 1.60% | 1.67% | 1.51% |
| Edge (baseline − model) | -0.01% | -0.05% | -0.01% |
| Brier (confidence calib.) | 0.24 | 0.29 | 0.26 |

## Per-strategy scorecards

| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |
|---|---|---|---|---|
| news | 99 | 33.33% | 1.37% | 0.23 |
| technical | 99 | 12.12% | 1.52% | 0.21 |
| contrarian | 99 | 25.25% | 1.58% | 0.20 |
| momentum | 99 | 17.17% | 1.65% | 0.19 |
| macro | 99 | 12.12% | 1.70% | 0.18 |

## Last 20 scored model predictions
_(backfill seed rows are excluded from metrics and this table; they appear only as price-history context on the dashboard chart. ⚠️ = written after the open, excluded from verdicts)_

| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest | Pre-open |
|---|---|---|---|---|---|---|---|---|
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

## 🅰️/🅱️ A/B test — ensemble+gates vs raven-style prior+pulse

_Both arms forecast the same sessions from the same pre-open snapshot (paired test). Only days on which both forecasts were written before the open count._

**Verdict:** Arm A leads on paired MAPE (mean daily delta 0.01% in A's favor over 51 pre-open days); B wins 26/51 decisive days (sign test p=1.0, descriptive); anytime-valid 95% CS [-0.208%, +0.191%] includes zero — not distinguishable from noise.

**B's next forecast (2026-11-02):** close ≈ **241.35** (up, adj 0.1σ, 8 evidence items).

| All-time, pre-open | A — ensemble+gates | B — raven prior+pulse |
|---|---|---|
| Scored days | 59 | 51 |
| PASS rate (±1%) | 42.37% | 50.98% |
| Directional accuracy | 57.63% | 64.71% |
| MAPE | 1.61% | 1.72% |
| Edge vs free baseline | 0.06% | 0.04% |

Clean paired days: 51 (of 74 paired); B wins 26/51 decisive; mean daily APE delta (A−B) -0.01%; CRPS delta (A−B) anytime-valid 95% CS [-0.155%, +0.125%] (undecided).
_This is a research experiment, not financial advice._
