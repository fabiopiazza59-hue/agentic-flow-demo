# 📈 AMZN Daily Close Predictor — Results

_Auto-generated after every run. Verdicts use pre-open forecasts only and anytime-valid 95% confidence sequences, so they stay valid although this page is re-read daily. PASS = predicted close within ±1% of actual._

## Verdict
❌ **No edge yet** — all-time MAPE 1.59% is nominally ahead of the free baseline 1.67% by 0.09%, but the difference is not distinguishable from noise (paired over 61 pre-open days: mean 0.09%, anytime-valid 95% CS [-0.040%, +0.212%]; fixed-sample 90% CI [-0.11%, 0.16%], sign test p=0.7982).

**Next forecast (2026-11-04):** close ≈ **248.22** (down, P(up) 29%, 80% interval 243.35–253.18) vs prior close 250.69; anchored to 248.80 (yfinance_ext, live, refreshed 0×).

## Scoreboard — all-time, pre-open forecasts only

_Every forecaster is scored as a calibrated predictive distribution (CRPS: lower is better; coverage should match its nominal level). The **free baseline** is the latest pre-open trade the forecast was anchored to — the prior close when there was none._

| Metric | Model (arm A) | Random walk (prior close) | Free baseline (pre-open price) |
|---|---|---|---|
| Days | 61 | 61 | 61 |
| MAPE | 1.59% | 1.62% | 1.67% |
| CRPS | 1.31% | 1.32% | 1.37% |
| 80% interval coverage | 81.97% | 78.69% | 80.33% |
| 50% interval coverage | 54.10% | 50.82% | 50.82% |
| Brier of P(up) | 0.226 | 0.250 | 0.240 |

**Does the model add value?** Paired daily gains, anytime-valid 95% confidence sequences (valid although this page is re-read every day):

| Comparison | Mean daily gain | 95% CS | Verdict |
|---|---|---|---|
| APE vs free baseline | 0.09% | [-0.040%, +0.212%] | ≈ not distinguishable |
| APE vs random walk | 0.03% | [-0.227%, +0.288%] | ≈ not distinguishable |
| CRPS vs free baseline | 0.06% | [-0.026%, +0.141%] | ≈ not distinguishable |
| CRPS vs random walk | 0.01% | [-0.198%, +0.217%] | ≈ not distinguishable |
| Direction (Brier of P(up)) vs random walk | 0.024 | [-0.021, +0.070] | ≈ not distinguishable |

## Self-improvement lab — measured, not narrated

Aggregation rules are replayed over every pre-open day using only earlier data. A rule replaces the equal-weight mean only when its anytime-valid confidence sequence (Bonferroni over 5 challengers, α=0.05) shows it is better; the judge and the gates are switched off the same way once shown to hurt ([`evals/lab.py`](src/evals/lab.py)).

**Champion: `mean`** (reference — no challenger has proven better yet; 61 days replayed).

| Rule | MAPE | Gain vs mean | CS (α/K) | Status |
|---|---|---|---|---|
| mean (reference) | 1.51% | — | — | 🏆 champion |
| median | 1.55% | -0.04% | [-0.181%, +0.094%] | ≈ not distinguishable |
| trimmed_mean | 1.55% | -0.04% | [-0.138%, +0.060%] | ≈ not distinguishable |
| inverse_mse | 1.53% | -0.02% | [-0.063%, +0.029%] | ≈ not distinguishable |
| best_recent | 1.72% | -0.21% | [-0.464%, +0.042%] | ≈ not distinguishable |
| shrink_half | 1.59% | -0.08% | [-0.177%, +0.022%] | ≈ not distinguishable |

- **Judge adjustment** (APE of blend − APE after the judge's clamped σ-move; v2 rows): n=27, mean 0.04%, CS [-0.032%, +0.112%] → judge **on**.
- **Guardrail gates** (APE before − after the gates): n=27, mean -0.00%, CS [-0.033%, +0.028%] → gates **on**.
- **Legacy v1 pipeline** (judge wrote the number) vs the plain mean of its own analysts: mean -0.17% over 34 pre-open days, CS [-0.407%, +0.059%] — ≈ not distinguishable (negative = the judge cost accuracy).

### Integrity & mechanism

- **40 scored row(s) were created after their session opened** and are excluded from every verdict. They saw part of the tape they forecast. New post-open rows are refused (`--allow-late` to override), and the schedule now researches the evening before and only re-anchors in the morning.
- **Pre-open coverage:** 20 of the last 20 sessions (since 2026-10-08) got a forecast written before the open.
- **Live anchors:** 27 of 61 pre-open forecasts were anchored on a real after-hours/pre-market trade (the rest on the prior close — before v2 the quote feed silently echoed it).
- **Gate effect** (ungated − gated APE, paired over 27 days): -0.00% 90% CI [-0.02%, 0.01%], gates helped on 5/9 decisive days (sign test p=1.0). Gates fired on 9 of them.

## Metrics

| Metric | All-time (pre-open) — verdict basis | Last 20 (all rows) | All-time (all rows) |
|---|---|---|---|
| Scored days | 61 | 20 | 101 |
| PASS rate (±1%) | 42.62% | 35.00% | 44.55% |
| Directional accuracy | 59.02% | 80.00% | 57.43% |
| MAPE | 1.59% | 1.73% | 1.51% |
| Baseline MAPE (random walk) | 1.62% | 1.78% | 1.52% |
| Edge (baseline − model) | 0.03% | 0.05% | 0.01% |
| Brier (confidence calib.) | 0.24 | 0.29 | 0.26 |

## Per-strategy scorecards

| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |
|---|---|---|---|---|
| news | 101 | 32.67% | 1.37% | 0.23 |
| technical | 101 | 11.88% | 1.51% | 0.20 |
| contrarian | 101 | 26.73% | 1.56% | 0.20 |
| momentum | 101 | 16.83% | 1.65% | 0.19 |
| macro | 101 | 11.88% | 1.69% | 0.18 |

## Last 20 scored model predictions
_(backfill seed rows are excluded from metrics and this table; they appear only as price-history context on the dashboard chart. ⚠️ = written after the open, excluded from verdicts)_

| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest | Pre-open |
|---|---|---|---|---|---|---|---|---|
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
| 2026-10-16 | 252.88 | 250.00 | 1.15% | ❌ | ✅ | ✅ | technical | ✅ |
| 2026-10-15 | 250.82 | 254.76 | 1.55% | ❌ | ❌ | ❌ | contrarian | ✅ |
| 2026-10-14 | 257.27 | 254.08 | 1.26% | ❌ | ✅ | ✅ | contrarian | ✅ |
| 2026-10-13 | 258.70 | 258.58 | 0.05% | ✅ | ✅ | ✅ | macro | ✅ |
| 2026-10-12 | 262.39 | 259.53 | 1.10% | ❌ | ✅ | ✅ | contrarian | ✅ |
| 2026-10-09 | 254.94 | 262.43 | 2.85% | ❌ | ✅ | ✅ | macro | ✅ |
| 2026-10-08 | 260.24 | 254.06 | 2.43% | ❌ | ❌ | ❌ | contrarian | ✅ |
| 2026-10-07 | 257.26 | 259.92 | 1.02% | ❌ | ✅ | ✅ | momentum | ✅ |

## 🅰️/🅱️ A/B test — ensemble+gates vs raven-style prior+pulse

_Both arms forecast the same sessions from the same pre-open snapshot (paired test). Only days on which both forecasts were written before the open count._

**Verdict:** Arm A leads on paired MAPE (mean daily delta 0.03% in A's favor over 53 pre-open days); B wins 26/53 decisive days (sign test p=1.0, descriptive); anytime-valid 95% CS [-0.234%, +0.169%] includes zero — not distinguishable from noise.

**B's next forecast (2026-11-04):** close ≈ **249.68** (down, adj 0.15σ, 8 evidence items).

| All-time, pre-open | A — ensemble+gates | B — raven prior+pulse |
|---|---|---|
| Scored days | 61 | 53 |
| PASS rate (±1%) | 42.62% | 49.06% |
| Directional accuracy | 59.02% | 66.04% |
| MAPE | 1.59% | 1.72% |
| Edge vs free baseline | 0.09% | 0.05% |

Clean paired days: 53 (of 76 paired); B wins 26/53 decisive; mean daily APE delta (A−B) -0.03%; CRPS delta (A−B) anytime-valid 95% CS [-0.168%, +0.110%] (undecided).
_This is a research experiment, not financial advice._
