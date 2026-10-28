# 📈 AMZN Daily Close Predictor — Results

_Auto-generated after every run. Verdicts use pre-open forecasts only and anytime-valid 95% confidence sequences, so they stay valid although this page is re-read daily. PASS = predicted close within ±1% of actual._

## Verdict
❌ **No edge yet** — all-time MAPE 1.54% is nominally ahead of the free baseline 1.57% by 0.04%, but the difference is not distinguishable from noise (paired over 56 pre-open days: mean 0.04%, anytime-valid 95% CS [-0.072%, +0.143%]; fixed-sample 90% CI [-0.05%, 0.12%], sign test p=1.0).

**Next forecast (2026-10-28):** close ≈ **254.90** (up, P(up) 68%, 80% interval 250.17–259.72) vs prior close 253.10; anchored to 253.76 (yfinance_ext, live, refreshed 0×).

## Scoreboard — all-time, pre-open forecasts only

_Every forecaster is scored as a calibrated predictive distribution (CRPS: lower is better; coverage should match its nominal level). The **free baseline** is the latest pre-open trade the forecast was anchored to — the prior close when there was none._

| Metric | Model (arm A) | Random walk (prior close) | Free baseline (pre-open price) |
|---|---|---|---|
| Days | 56 | 56 | 56 |
| MAPE | 1.54% | 1.57% | 1.57% |
| CRPS | 1.27% | 1.29% | 1.28% |
| 80% interval coverage | 83.93% | 80.36% | 82.14% |
| 50% interval coverage | 53.57% | 53.57% | 53.57% |
| Brier of P(up) | 0.240 | 0.250 | 0.250 |

**Does the model add value?** Paired daily gains, anytime-valid 95% confidence sequences (valid although this page is re-read every day):

| Comparison | Mean daily gain | 95% CS | Verdict |
|---|---|---|---|
| APE vs free baseline | 0.04% | [-0.072%, +0.143%] | ≈ not distinguishable |
| APE vs random walk | 0.04% | [-0.114%, +0.187%] | ≈ not distinguishable |
| CRPS vs free baseline | 0.02% | [-0.041%, +0.074%] | ≈ not distinguishable |
| CRPS vs random walk | 0.03% | [-0.069%, +0.124%] | ≈ not distinguishable |
| Direction (Brier of P(up)) vs random walk | 0.010 | [-0.035, +0.054] | ≈ not distinguishable |

## Self-improvement lab — measured, not narrated

Aggregation rules are replayed over every pre-open day using only earlier data. A rule replaces the equal-weight mean only when its anytime-valid confidence sequence (Bonferroni over 5 challengers, α=0.05) shows it is better; the judge and the gates are switched off the same way once shown to hurt ([`evals/lab.py`](src/evals/lab.py)).

**Champion: `mean`** (reference — no challenger has proven better yet; 56 days replayed).

| Rule | MAPE | Gain vs mean | CS (α/K) | Status |
|---|---|---|---|---|
| mean (reference) | 1.44% | — | — | 🏆 champion |
| median | 1.48% | -0.05% | [-0.198%, +0.102%] | ≈ not distinguishable |
| trimmed_mean | 1.48% | -0.05% | [-0.152%, +0.062%] | ≈ not distinguishable |
| inverse_mse | 1.45% | -0.02% | [-0.069%, +0.031%] | ≈ not distinguishable |
| best_recent | 1.66% | -0.23% | [-0.501%, +0.043%] | ≈ not distinguishable |
| shrink_half | 1.50% | -0.06% | [-0.167%, +0.041%] | ≈ not distinguishable |

- **Judge adjustment** (APE of blend − APE after the judge's clamped σ-move; v2 rows): n=22, mean 0.01%, CS [-0.048%, +0.070%] → judge **on**.
- **Guardrail gates** (APE before − after the gates): n=22, mean -0.00%, CS [-0.042%, +0.035%] → gates **on**.
- **Legacy v1 pipeline** (judge wrote the number) vs the plain mean of its own analysts: mean -0.17% over 34 pre-open days, CS [-0.407%, +0.059%] — ≈ not distinguishable (negative = the judge cost accuracy).

### Integrity & mechanism

- **40 scored row(s) were created after their session opened** and are excluded from every verdict. They saw part of the tape they forecast. New post-open rows are refused (`--allow-late` to override), and the schedule now researches the evening before and only re-anchors in the morning.
- **Pre-open coverage:** 20 of the last 20 sessions (since 2026-10-01) got a forecast written before the open.
- **Live anchors:** 22 of 56 pre-open forecasts were anchored on a real after-hours/pre-market trade (the rest on the prior close — before v2 the quote feed silently echoed it).
- **Gate effect** (ungated − gated APE, paired over 22 days): -0.00% 90% CI [-0.02%, 0.01%], gates helped on 5/9 decisive days (sign test p=1.0). Gates fired on 9 of them.

## Metrics

| Metric | All-time (pre-open) — verdict basis | Last 20 (all rows) | All-time (all rows) |
|---|---|---|---|
| Scored days | 56 | 20 | 96 |
| PASS rate (±1%) | 42.86% | 45.00% | 44.79% |
| Directional accuracy | 55.36% | 70.00% | 55.21% |
| MAPE | 1.54% | 1.39% | 1.48% |
| Baseline MAPE (random walk) | 1.57% | 1.48% | 1.49% |
| Edge (baseline − model) | 0.04% | 0.09% | 0.01% |
| Brier (confidence calib.) | 0.25 | 0.28 | 0.26 |

## Per-strategy scorecards

| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |
|---|---|---|---|---|
| news | 96 | 32.29% | 1.32% | 0.23 |
| technical | 96 | 12.50% | 1.47% | 0.21 |
| contrarian | 96 | 26.04% | 1.52% | 0.20 |
| momentum | 96 | 17.71% | 1.60% | 0.19 |
| macro | 96 | 11.46% | 1.66% | 0.18 |

## Last 20 scored model predictions
_(backfill seed rows are excluded from metrics and this table; they appear only as price-history context on the dashboard chart. ⚠️ = written after the open, excluded from verdicts)_

| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest | Pre-open |
|---|---|---|---|---|---|---|---|---|
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
| 2026-10-02 | 249.13 | 251.52 | 0.95% | ✅ | ✅ | ✅ | news | ✅ |
| 2026-10-01 | 250.26 | 248.23 | 0.82% | ✅ | ❌ | ❌ | contrarian | ✅ |
| 2026-09-30 | 247.28 | 249.15 | 0.75% | ✅ | ✅ | ✅ | contrarian | ✅ |

## 🅰️/🅱️ A/B test — ensemble+gates vs raven-style prior+pulse

_Both arms forecast the same sessions from the same pre-open snapshot (paired test). Only days on which both forecasts were written before the open count._

**Verdict:** Arm B leads on paired MAPE (mean daily delta 0.01% in B's favor over 48 pre-open days); B wins 24/48 decisive days (sign test p=1.0, descriptive); anytime-valid 95% CS [-0.184%, +0.213%] includes zero — not distinguishable from noise.

**B's next forecast (2026-10-28):** close ≈ **254.39** (up, adj 0.12σ, 7 evidence items).

| All-time, pre-open | A — ensemble+gates | B — raven prior+pulse |
|---|---|---|
| Scored days | 56 | 48 |
| PASS rate (±1%) | 42.86% | 52.08% |
| Directional accuracy | 55.36% | 62.50% |
| MAPE | 1.54% | 1.62% |
| Edge vs free baseline | 0.04% | 0.04% |

Clean paired days: 48 (of 71 paired); B wins 24/48 decisive; mean daily APE delta (A−B) 0.01%; CRPS delta (A−B) anytime-valid 95% CS [-0.109%, +0.135%] (undecided).
_This is a research experiment, not financial advice._
