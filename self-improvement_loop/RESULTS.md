# 📈 AMZN Daily Close Predictor — Results

_Auto-generated after every run. Verdicts use pre-open forecasts only and anytime-valid 95% confidence sequences, so they stay valid although this page is re-read daily. PASS = predicted close within ±1% of actual._

## Verdict
❌ **No edge yet** — all-time MAPE 1.53% is nominally ahead of the free baseline 1.54% by 0.01%, but the difference is not distinguishable from noise (paired over 43 pre-open days: mean 0.01%, anytime-valid 95% CS [-0.120%, +0.143%]; fixed-sample 90% CI [-0.05%, 0.10%], sign test p=0.3604).

**Next forecast (2026-10-09):** close ≈ **254.94** (up, P(up) 62%, 80% interval 250.97–258.97) vs prior close 254.06; anchored to 254.88 (yfinance_ext, live, refreshed 0×).

## Scoreboard — all-time, pre-open forecasts only

_Every forecaster is scored as a calibrated predictive distribution (CRPS: lower is better; coverage should match its nominal level). The **free baseline** is the latest pre-open trade the forecast was anchored to — the prior close when there was none._

| Metric | Model (arm A) | Random walk (prior close) | Free baseline (pre-open price) |
|---|---|---|---|
| Days | 43 | 43 | 43 |
| MAPE | 1.53% | 1.55% | 1.54% |
| CRPS | 1.26% | 1.28% | 1.27% |
| 80% interval coverage | 86.05% | 86.05% | 86.05% |
| 50% interval coverage | 58.14% | 55.81% | 58.14% |
| Brier of P(up) | 0.239 | 0.250 | 0.245 |

**Does the model add value?** Paired daily gains, anytime-valid 95% confidence sequences (valid although this page is re-read every day):

| Comparison | Mean daily gain | 95% CS | Verdict |
|---|---|---|---|
| APE vs free baseline | 0.01% | [-0.120%, +0.143%] | ≈ not distinguishable |
| APE vs random walk | 0.02% | [-0.120%, +0.168%] | ≈ not distinguishable |
| CRPS vs free baseline | 0.01% | [-0.062%, +0.079%] | ≈ not distinguishable |
| CRPS vs random walk | 0.02% | [-0.062%, +0.098%] | ≈ not distinguishable |
| Direction (Brier of P(up)) vs random walk | 0.011 | [-0.026, +0.048] | ≈ not distinguishable |

## Self-improvement lab — measured, not narrated

Aggregation rules are replayed over every pre-open day using only earlier data. A rule replaces the equal-weight mean only when its anytime-valid confidence sequence (Bonferroni over 5 challengers, α=0.05) shows it is better; the judge and the gates are switched off the same way once shown to hurt ([`evals/lab.py`](src/evals/lab.py)).

**Champion: `mean`** (reference — no challenger has proven better yet; 43 days replayed).

| Rule | MAPE | Gain vs mean | CS (α/K) | Status |
|---|---|---|---|---|
| mean (reference) | 1.39% | — | — | 🏆 champion |
| median | 1.42% | -0.03% | [-0.215%, +0.145%] | ≈ not distinguishable |
| trimmed_mean | 1.43% | -0.04% | [-0.177%, +0.096%] | ≈ not distinguishable |
| inverse_mse | 1.42% | -0.03% | [-0.095%, +0.031%] | ≈ not distinguishable |
| best_recent | 1.66% | -0.27% | [-0.602%, +0.061%] | ≈ not distinguishable |
| shrink_half | 1.46% | -0.07% | [-0.204%, +0.067%] | ≈ not distinguishable |

- **Judge adjustment** (APE of blend − APE after the judge's clamped σ-move; v2 rows): n=9, mean 0.00%, CS n=9 (too few) → judge **on**.
- **Guardrail gates** (APE before − after the gates): n=9, mean -0.02%, CS n=9 (too few) → gates **on**.
- **Legacy v1 pipeline** (judge wrote the number) vs the plain mean of its own analysts: mean -0.17% over 34 pre-open days, CS [-0.407%, +0.059%] — ≈ not distinguishable (negative = the judge cost accuracy).

### Integrity & mechanism

- **40 scored row(s) were created after their session opened** and are excluded from every verdict. They saw part of the tape they forecast. New post-open rows are refused (`--allow-late` to override), and the schedule now researches the evening before and only re-anchors in the morning.
- **Pre-open coverage:** 10 of the last 20 sessions (since 2026-09-14) got a forecast written before the open.
- **Live anchors:** 9 of 43 pre-open forecasts were anchored on a real after-hours/pre-market trade (the rest on the prior close — before v2 the quote feed silently echoed it).
- **Gate effect** (ungated − gated APE, paired over 9 days): -0.02% 90% CI [-0.06%, 0.03%], gates helped on 4/8 decisive days (sign test p=1.0). Gates fired on 8 of them.

## Metrics

| Metric | All-time (pre-open) — verdict basis | Last 20 (all rows) | All-time (all rows) |
|---|---|---|---|
| Scored days | 43 | 20 | 83 |
| PASS rate (±1%) | 44.19% | 50.00% | 45.78% |
| Directional accuracy | 48.84% | 55.00% | 51.81% |
| MAPE | 1.53% | 1.12% | 1.46% |
| Baseline MAPE (random walk) | 1.55% | 1.25% | 1.47% |
| Edge (baseline − model) | 0.02% | 0.13% | 0.00% |
| Brier (confidence calib.) | 0.23 | 0.28 | 0.26 |

## Per-strategy scorecards

| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |
|---|---|---|---|---|
| news | 83 | 32.53% | 1.29% | 0.23 |
| technical | 83 | 13.25% | 1.42% | 0.21 |
| contrarian | 83 | 22.89% | 1.51% | 0.19 |
| momentum | 83 | 20.48% | 1.56% | 0.19 |
| macro | 83 | 10.84% | 1.63% | 0.18 |

## Last 20 scored model predictions
_(backfill seed rows are excluded from metrics and this table; they appear only as price-history context on the dashboard chart. ⚠️ = written after the open, excluded from verdicts)_

| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest | Pre-open |
|---|---|---|---|---|---|---|---|---|
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
| 2026-09-21 | 255.15 | 258.45 | 1.28% | ❌ | ✅ | ✅ | news | ⚠️ |
| 2026-09-18 | 251.27 | 253.71 | 0.96% | ✅ | ✅ | ✅ | news | ⚠️ |
| 2026-09-17 | 245.10 | 251.19 | 2.42% | ❌ | ❌ | ❌ | news | ⚠️ |
| 2026-09-16 | 247.05 | 245.96 | 0.44% | ✅ | ✅ | ✅ | macro | ⚠️ |
| 2026-09-15 | 252.45 | 248.42 | 1.62% | ❌ | ✅ | ✅ | momentum | ⚠️ |
| 2026-09-14 | 255.60 | 253.54 | 0.81% | ✅ | ✅ | ✅ | news | ⚠️ |
| 2026-09-11 | 251.72 | 256.78 | 1.97% | ❌ | ❌ | ❌ | news | ⚠️ |

## 🅰️/🅱️ A/B test — ensemble+gates vs raven-style prior+pulse

_Both arms forecast the same sessions from the same pre-open snapshot (paired test). Only days on which both forecasts were written before the open count._

**Verdict:** Arm B leads on paired MAPE (mean daily delta 0.07% in B's favor over 35 pre-open days); B wins 19/35 decisive days (sign test p=0.7359, descriptive); anytime-valid 95% CS [-0.173%, +0.315%] includes zero — not distinguishable from noise.

**B's next forecast (2026-10-09):** close ≈ **254.31** (up, adj -0.15σ, 8 evidence items).

| All-time, pre-open | A — ensemble+gates | B — raven prior+pulse |
|---|---|---|
| Scored days | 43 | 35 |
| PASS rate (±1%) | 44.19% | 51.43% |
| Directional accuracy | 48.84% | 60.00% |
| MAPE | 1.53% | 1.59% |
| Edge vs free baseline | 0.01% | 0.06% |

Clean paired days: 35 (of 58 paired); B wins 19/35 decisive; mean daily APE delta (A−B) 0.07%; CRPS delta (A−B) anytime-valid 95% CS [-0.099%, +0.198%] (undecided).
_This is a research experiment, not financial advice._
