# 📈 AMZN Daily Close Predictor — Results

_Auto-generated after every run. Verdicts use pre-open forecasts only and anytime-valid 95% confidence sequences, so they stay valid although this page is re-read daily. PASS = predicted close within ±1% of actual._

## Verdict
❌ **No edge yet** — all-time MAPE 1.66% is nominally ahead of the free baseline 1.67% by 0.01%, but the difference is not distinguishable from noise (paired over 34 pre-open days: mean 0.01%, anytime-valid 95% CS [-0.155%, +0.182%]; fixed-sample 90% CI [-0.07%, 0.10%], sign test p=0.3915).

**Awaiting score (2026-09-28):** close ≈ **249.91** (up, P(up) 55%, 80% interval 245.85–254.04) vs prior close 249.67; anchored to 249.98 (yfinance_ext, live, refreshed 0×).

## Scoreboard — all-time, pre-open forecasts only

_Every forecaster is scored as a calibrated predictive distribution (CRPS: lower is better; coverage should match its nominal level). The **free baseline** is the latest pre-open trade the forecast was anchored to — the prior close when there was none._

| Metric | Model (arm A) | Random walk (prior close) | Free baseline (pre-open price) |
|---|---|---|---|
| Days | 34 | 34 | 34 |
| MAPE | 1.66% | 1.67% | 1.67% |
| CRPS | 1.39% | 1.40% | 1.40% |
| 80% interval coverage | 88.24% | 88.24% | 88.24% |
| 50% interval coverage | 61.76% | 61.76% | 61.76% |
| Brier of P(up) | 0.238 | 0.250 | 0.250 |

**Does the model add value?** Paired daily gains, anytime-valid 95% confidence sequences (valid although this page is re-read every day):

| Comparison | Mean daily gain | 95% CS | Verdict |
|---|---|---|---|
| APE vs free baseline | 0.01% | [-0.155%, +0.182%] | ≈ not distinguishable |
| APE vs random walk | 0.01% | [-0.155%, +0.182%] | ≈ not distinguishable |
| CRPS vs free baseline | 0.01% | [-0.081%, +0.100%] | ≈ not distinguishable |
| CRPS vs random walk | 0.01% | [-0.081%, +0.100%] | ≈ not distinguishable |
| Direction (Brier of P(up)) vs random walk | 0.012 | [-0.027, +0.050] | ≈ not distinguishable |

## Self-improvement lab — measured, not narrated

Aggregation rules are replayed over every pre-open day using only earlier data. A rule replaces the equal-weight mean only when its anytime-valid confidence sequence (Bonferroni over 5 challengers, α=0.05) shows it is better; the judge and the gates are switched off the same way once shown to hurt ([`evals/lab.py`](src/evals/lab.py)).

**Champion: `mean`** (reference — no challenger has proven better yet; 34 days replayed).

| Rule | MAPE | Gain vs mean | CS (α/K) | Status |
|---|---|---|---|---|
| mean (reference) | 1.49% | — | — | 🏆 champion |
| median | 1.51% | -0.02% | [-0.240%, +0.200%] | ≈ not distinguishable |
| trimmed_mean | 1.53% | -0.04% | [-0.216%, +0.127%] | ≈ not distinguishable |
| inverse_mse | 1.53% | -0.04% | [-0.117%, +0.033%] | ≈ not distinguishable |
| best_recent | 1.80% | -0.31% | [-0.728%, +0.100%] | ≈ not distinguishable |
| shrink_half | 1.57% | -0.08% | [-0.255%, +0.087%] | ≈ not distinguishable |

- **Judge adjustment** (APE of blend − APE after the judge's clamped σ-move; v2 rows): n=0, mean —, CS n=0 (too few) → judge **on**.
- **Guardrail gates** (APE before − after the gates): n=0, mean —, CS n=0 (too few) → gates **on**.
- **Legacy v1 pipeline** (judge wrote the number) vs the plain mean of its own analysts: mean -0.17% over 34 pre-open days, CS [-0.407%, +0.059%] — ≈ not distinguishable (negative = the judge cost accuracy).

### Integrity & mechanism

- **40 scored row(s) were created after their session opened** and are excluded from every verdict. They saw part of the tape they forecast. New post-open rows are refused (`--allow-late` to override), and the schedule now researches the evening before and only re-anchors in the morning.
- **Pre-open coverage:** 1 of the last 20 sessions (since 2026-08-31) got a forecast written before the open.
- **Live anchors:** 0 of 34 pre-open forecasts were anchored on a real after-hours/pre-market trade (the rest on the prior close — before v2 the quote feed silently echoed it).
- **Gate effect**: not measurable yet — accrues from the first run that records an ungated counterfactual (`predicted_close_raw`) on each row.

## Metrics

| Metric | All-time (pre-open) — verdict basis | Last 20 (all rows) | All-time (all rows) |
|---|---|---|---|
| Scored days | 34 | 20 | 74 |
| PASS rate (±1%) | 41.18% | 50.00% | 44.59% |
| Directional accuracy | 50.00% | 70.00% | 52.70% |
| MAPE | 1.66% | 1.08% | 1.51% |
| Baseline MAPE (random walk) | 1.67% | 1.19% | 1.51% |
| Edge (baseline − model) | 0.01% | 0.11% | -0.00% |
| Brier (confidence calib.) | 0.22 | 0.29 | 0.26 |

## Per-strategy scorecards

| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |
|---|---|---|---|---|
| news | 74 | 31.08% | 1.33% | 0.23 |
| technical | 74 | 14.86% | 1.45% | 0.21 |
| contrarian | 74 | 20.27% | 1.57% | 0.19 |
| momentum | 74 | 21.62% | 1.61% | 0.19 |
| macro | 74 | 12.16% | 1.68% | 0.18 |

## Last 20 scored model predictions
_(backfill seed rows are excluded from metrics and this table; they appear only as price-history context on the dashboard chart. ⚠️ = written after the open, excluded from verdicts)_

| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest | Pre-open |
|---|---|---|---|---|---|---|---|---|
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
| 2026-09-10 | 251.35 | 251.89 | 0.21% | ✅ | ✅ | ❌ | technical | ⚠️ |
| 2026-09-09 | 256.44 | 252.40 | 1.60% | ❌ | ✅ | ✅ | news | ⚠️ |
| 2026-09-08 | 258.28 | 256.97 | 0.51% | ✅ | ✅ | ✅ | macro | ⚠️ |
| 2026-09-04 | 258.82 | 258.51 | 0.12% | ✅ | ✅ | ✅ | technical | ⚠️ |
| 2026-09-03 | 254.70 | 258.90 | 1.62% | ❌ | ❌ | ❌ | contrarian | ⚠️ |
| 2026-09-02 | 253.69 | 254.98 | 0.51% | ✅ | ❌ | ❌ | momentum | ⚠️ |
| 2026-09-01 | 258.91 | 254.92 | 1.57% | ❌ | ✅ | ✅ | news | ⚠️ |
| 2026-08-31 | 265.87 | 259.77 | 2.35% | ❌ | ✅ | ✅ | news | ⚠️ |
| 2026-08-28 | 265.99 | 266.43 | 0.17% | ✅ | ✅ | ❌ | news | ⚠️ |

## 🅰️/🅱️ A/B test — ensemble+gates vs raven-style prior+pulse

_Both arms forecast the same sessions from the same pre-open snapshot (paired test). Only days on which both forecasts were written before the open count._

**Verdict:** Arm B leads on paired MAPE (mean daily delta 0.08% in B's favor over 26 pre-open days); B wins 14/26 decisive days (sign test p=0.845, descriptive); anytime-valid 95% CS [-0.245%, +0.408%] includes zero — not distinguishable from noise.

**B's awaiting score (2026-09-28):** close ≈ **250.2** (up, adj 0.05σ, 8 evidence items).

| All-time, pre-open | A — ensemble+gates | B — raven prior+pulse |
|---|---|---|
| Scored days | 34 | 26 |
| PASS rate (±1%) | 41.18% | 46.15% |
| Directional accuracy | 50.00% | 61.54% |
| MAPE | 1.66% | 1.79% |
| Edge vs free baseline | 0.01% | 0.07% |

Clean paired days: 26 (of 49 paired); B wins 14/26 decisive; mean daily APE delta (A−B) 0.08%; CRPS delta (A−B) anytime-valid 95% CS [-0.146%, +0.252%] (undecided).
_This is a research experiment, not financial advice._
