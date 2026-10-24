# 📈 AMZN Daily Close Predictor — Results

_Auto-generated after every run. Verdicts use pre-open forecasts only and anytime-valid 95% confidence sequences, so they stay valid although this page is re-read daily. PASS = predicted close within ±1% of actual._

## Verdict
❌ **No edge yet** — all-time MAPE 1.57% is nominally ahead of the free baseline 1.60% by 0.03%, but the difference is not distinguishable from noise (paired over 54 pre-open days: mean 0.03%, anytime-valid 95% CS [-0.078%, +0.142%]; fixed-sample 90% CI [-0.06%, 0.10%], sign test p=0.8919).

**Next forecast (2026-10-26):** close ≈ **248.61** (up, P(up) 58%, 80% interval 243.80–253.52) vs prior close 248.00; anchored to 248.78 (yfinance_ext, live, refreshed 1×).

## Scoreboard — all-time, pre-open forecasts only

_Every forecaster is scored as a calibrated predictive distribution (CRPS: lower is better; coverage should match its nominal level). The **free baseline** is the latest pre-open trade the forecast was anchored to — the prior close when there was none._

| Metric | Model (arm A) | Random walk (prior close) | Free baseline (pre-open price) |
|---|---|---|---|
| Days | 54 | 54 | 54 |
| MAPE | 1.57% | 1.60% | 1.60% |
| CRPS | 1.29% | 1.32% | 1.31% |
| 80% interval coverage | 83.33% | 79.63% | 81.48% |
| 50% interval coverage | 53.70% | 53.70% | 53.70% |
| Brier of P(up) | 0.244 | 0.250 | 0.253 |

**Does the model add value?** Paired daily gains, anytime-valid 95% confidence sequences (valid although this page is re-read every day):

| Comparison | Mean daily gain | 95% CS | Verdict |
|---|---|---|---|
| APE vs free baseline | 0.03% | [-0.078%, +0.142%] | ≈ not distinguishable |
| APE vs random walk | 0.02% | [-0.130%, +0.178%] | ≈ not distinguishable |
| CRPS vs free baseline | 0.01% | [-0.044%, +0.073%] | ≈ not distinguishable |
| CRPS vs random walk | 0.02% | [-0.077%, +0.121%] | ≈ not distinguishable |
| Direction (Brier of P(up)) vs random walk | 0.006 | [-0.040, +0.052] | ≈ not distinguishable |

## Self-improvement lab — measured, not narrated

Aggregation rules are replayed over every pre-open day using only earlier data. A rule replaces the equal-weight mean only when its anytime-valid confidence sequence (Bonferroni over 5 challengers, α=0.05) shows it is better; the judge and the gates are switched off the same way once shown to hurt ([`evals/lab.py`](src/evals/lab.py)).

**Champion: `mean`** (reference — no challenger has proven better yet; 54 days replayed).

| Rule | MAPE | Gain vs mean | CS (α/K) | Status |
|---|---|---|---|---|
| mean (reference) | 1.46% | — | — | 🏆 champion |
| median | 1.49% | -0.03% | [-0.184%, +0.116%] | ≈ not distinguishable |
| trimmed_mean | 1.50% | -0.04% | [-0.154%, +0.068%] | ≈ not distinguishable |
| inverse_mse | 1.48% | -0.02% | [-0.073%, +0.028%] | ≈ not distinguishable |
| best_recent | 1.70% | -0.24% | [-0.519%, +0.039%] | ≈ not distinguishable |
| shrink_half | 1.53% | -0.06% | [-0.172%, +0.042%] | ≈ not distinguishable |

- **Judge adjustment** (APE of blend − APE after the judge's clamped σ-move; v2 rows): n=20, mean 0.00%, CS [-0.056%, +0.064%] → judge **on**.
- **Guardrail gates** (APE before − after the gates): n=20, mean -0.01%, CS [-0.048%, +0.034%] → gates **on**.
- **Legacy v1 pipeline** (judge wrote the number) vs the plain mean of its own analysts: mean -0.17% over 34 pre-open days, CS [-0.407%, +0.059%] — ≈ not distinguishable (negative = the judge cost accuracy).

### Integrity & mechanism

- **40 scored row(s) were created after their session opened** and are excluded from every verdict. They saw part of the tape they forecast. New post-open rows are refused (`--allow-late` to override), and the schedule now researches the evening before and only re-anchors in the morning.
- **Pre-open coverage:** 20 of the last 20 sessions (since 2026-09-29) got a forecast written before the open.
- **Live anchors:** 20 of 54 pre-open forecasts were anchored on a real after-hours/pre-market trade (the rest on the prior close — before v2 the quote feed silently echoed it).
- **Gate effect** (ungated − gated APE, paired over 20 days): -0.01% 90% CI [-0.03%, 0.01%], gates helped on 4/8 decisive days (sign test p=1.0). Gates fired on 8 of them.

## Metrics

| Metric | All-time (pre-open) — verdict basis | Last 20 (all rows) | All-time (all rows) |
|---|---|---|---|
| Scored days | 54 | 20 | 94 |
| PASS rate (±1%) | 42.59% | 45.00% | 44.68% |
| Directional accuracy | 53.70% | 60.00% | 54.26% |
| MAPE | 1.57% | 1.42% | 1.49% |
| Baseline MAPE (random walk) | 1.60% | 1.46% | 1.50% |
| Edge (baseline − model) | 0.02% | 0.04% | 0.01% |
| Brier (confidence calib.) | 0.24 | 0.28 | 0.26 |

## Per-strategy scorecards

| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |
|---|---|---|---|---|
| news | 94 | 31.91% | 1.33% | 0.23 |
| technical | 94 | 12.77% | 1.47% | 0.21 |
| contrarian | 94 | 25.53% | 1.54% | 0.20 |
| momentum | 94 | 18.09% | 1.61% | 0.19 |
| macro | 94 | 11.70% | 1.67% | 0.18 |

## Last 20 scored model predictions
_(backfill seed rows are excluded from metrics and this table; they appear only as price-history context on the dashboard chart. ⚠️ = written after the open, excluded from verdicts)_

| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest | Pre-open |
|---|---|---|---|---|---|---|---|---|
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
| 2026-09-28 | 249.91 | 246.15 | 1.53% | ❌ | ❌ | ❌ | news | ✅ |

## 🅰️/🅱️ A/B test — ensemble+gates vs raven-style prior+pulse

_Both arms forecast the same sessions from the same pre-open snapshot (paired test). Only days on which both forecasts were written before the open count._

**Verdict:** Arm B leads on paired MAPE (mean daily delta 0.01% in B's favor over 46 pre-open days); B wins 23/46 decisive days (sign test p=1.0, descriptive); anytime-valid 95% CS [-0.193%, +0.222%] includes zero — not distinguishable from noise.

**B's next forecast (2026-10-26):** close ≈ **248.78** (up, adj 0.0σ, 0 evidence items).

| All-time, pre-open | A — ensemble+gates | B — raven prior+pulse |
|---|---|---|
| Scored days | 54 | 46 |
| PASS rate (±1%) | 42.59% | 52.17% |
| Directional accuracy | 53.70% | 60.87% |
| MAPE | 1.57% | 1.66% |
| Edge vs free baseline | 0.03% | 0.03% |

Clean paired days: 46 (of 69 paired); B wins 23/46 decisive; mean daily APE delta (A−B) 0.01%; CRPS delta (A−B) anytime-valid 95% CS [-0.115%, +0.141%] (undecided).
_This is a research experiment, not financial advice._
