# 📈 AMZN Daily Close Predictor — Results

_Auto-generated after every run. Verdicts use pre-open forecasts only and anytime-valid 95% confidence sequences, so they stay valid although this page is re-read daily. PASS = predicted close within ±1% of actual._

## Verdict
❌ **No edge yet** — all-time MAPE 1.52% is nominally ahead of the free baseline 1.53% by 0.01%, but the difference is not distinguishable from noise (paired over 41 pre-open days: mean 0.01%, anytime-valid 95% CS [-0.126%, +0.150%]; fixed-sample 90% CI [-0.06%, 0.10%], sign test p=0.3489).

**Next forecast (2026-10-07):** close ≈ **257.26** (up, P(up) 64%, 80% interval 253.58–260.99) vs prior close 256.29; anchored to 257.23 (yfinance_ext, live, refreshed 0×).

## Scoreboard — all-time, pre-open forecasts only

_Every forecaster is scored as a calibrated predictive distribution (CRPS: lower is better; coverage should match its nominal level). The **free baseline** is the latest pre-open trade the forecast was anchored to — the prior close when there was none._

| Metric | Model (arm A) | Random walk (prior close) | Free baseline (pre-open price) |
|---|---|---|---|
| Days | 41 | 41 | 41 |
| MAPE | 1.52% | 1.54% | 1.53% |
| CRPS | 1.26% | 1.27% | 1.27% |
| 80% interval coverage | 87.80% | 87.80% | 87.80% |
| 50% interval coverage | 60.98% | 58.54% | 60.98% |
| Brier of P(up) | 0.240 | 0.250 | 0.246 |

**Does the model add value?** Paired daily gains, anytime-valid 95% confidence sequences (valid although this page is re-read every day):

| Comparison | Mean daily gain | 95% CS | Verdict |
|---|---|---|---|
| APE vs free baseline | 0.01% | [-0.126%, +0.150%] | ≈ not distinguishable |
| APE vs random walk | 0.02% | [-0.129%, +0.168%] | ≈ not distinguishable |
| CRPS vs free baseline | 0.01% | [-0.066%, +0.083%] | ≈ not distinguishable |
| CRPS vs random walk | 0.02% | [-0.066%, +0.097%] | ≈ not distinguishable |
| Direction (Brier of P(up)) vs random walk | 0.010 | [-0.027, +0.048] | ≈ not distinguishable |

## Self-improvement lab — measured, not narrated

Aggregation rules are replayed over every pre-open day using only earlier data. A rule replaces the equal-weight mean only when its anytime-valid confidence sequence (Bonferroni over 5 challengers, α=0.05) shows it is better; the judge and the gates are switched off the same way once shown to hurt ([`evals/lab.py`](src/evals/lab.py)).

**Champion: `mean`** (reference — no challenger has proven better yet; 41 days replayed).

| Rule | MAPE | Gain vs mean | CS (α/K) | Status |
|---|---|---|---|---|
| mean (reference) | 1.37% | — | — | 🏆 champion |
| median | 1.41% | -0.04% | [-0.228%, +0.149%] | ≈ not distinguishable |
| trimmed_mean | 1.42% | -0.05% | [-0.189%, +0.095%] | ≈ not distinguishable |
| inverse_mse | 1.41% | -0.03% | [-0.099%, +0.034%] | ≈ not distinguishable |
| best_recent | 1.66% | -0.28% | [-0.631%, +0.062%] | ≈ not distinguishable |
| shrink_half | 1.44% | -0.07% | [-0.214%, +0.071%] | ≈ not distinguishable |

- **Judge adjustment** (APE of blend − APE after the judge's clamped σ-move; v2 rows): n=7, mean 0.00%, CS n=7 (too few) → judge **on**.
- **Guardrail gates** (APE before − after the gates): n=7, mean -0.02%, CS n=7 (too few) → gates **on**.
- **Legacy v1 pipeline** (judge wrote the number) vs the plain mean of its own analysts: mean -0.17% over 34 pre-open days, CS [-0.407%, +0.059%] — ≈ not distinguishable (negative = the judge cost accuracy).

### Integrity & mechanism

- **40 scored row(s) were created after their session opened** and are excluded from every verdict. They saw part of the tape they forecast. New post-open rows are refused (`--allow-late` to override), and the schedule now researches the evening before and only re-anchors in the morning.
- **Pre-open coverage:** 8 of the last 20 sessions (since 2026-09-10) got a forecast written before the open.
- **Live anchors:** 7 of 41 pre-open forecasts were anchored on a real after-hours/pre-market trade (the rest on the prior close — before v2 the quote feed silently echoed it).
- **Gate effect** (ungated − gated APE, paired over 7 days): -0.02% 90% CI [-0.07%, 0.03%], gates helped on 3/6 decisive days (sign test p=1.0). Gates fired on 6 of them.

## Metrics

| Metric | All-time (pre-open) — verdict basis | Last 20 (all rows) | All-time (all rows) |
|---|---|---|---|
| Scored days | 41 | 20 | 81 |
| PASS rate (±1%) | 46.34% | 55.00% | 46.91% |
| Directional accuracy | 48.78% | 60.00% | 51.85% |
| MAPE | 1.52% | 1.03% | 1.46% |
| Baseline MAPE (random walk) | 1.54% | 1.16% | 1.46% |
| Edge (baseline − model) | 0.02% | 0.13% | 0.00% |
| Brier (confidence calib.) | 0.22 | 0.28 | 0.26 |

## Per-strategy scorecards

| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |
|---|---|---|---|---|
| news | 81 | 33.33% | 1.29% | 0.23 |
| technical | 81 | 13.58% | 1.41% | 0.21 |
| contrarian | 81 | 22.22% | 1.50% | 0.20 |
| momentum | 81 | 19.75% | 1.55% | 0.19 |
| macro | 81 | 11.11% | 1.63% | 0.18 |

## Last 20 scored model predictions
_(backfill seed rows are excluded from metrics and this table; they appear only as price-history context on the dashboard chart. ⚠️ = written after the open, excluded from verdicts)_

| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest | Pre-open |
|---|---|---|---|---|---|---|---|---|
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
| 2026-09-10 | 251.35 | 251.89 | 0.21% | ✅ | ✅ | ❌ | technical | ⚠️ |
| 2026-09-09 | 256.44 | 252.40 | 1.60% | ❌ | ✅ | ✅ | news | ⚠️ |

## 🅰️/🅱️ A/B test — ensemble+gates vs raven-style prior+pulse

_Both arms forecast the same sessions from the same pre-open snapshot (paired test). Only days on which both forecasts were written before the open count._

**Verdict:** Arm B leads on paired MAPE (mean daily delta 0.06% in B's favor over 33 pre-open days); B wins 17/33 decisive days (sign test p=1.0, descriptive); anytime-valid 95% CS [-0.200%, +0.316%] includes zero — not distinguishable from noise.

**B's next forecast (2026-10-07):** close ≈ **257.59** (up, adj 0.1σ, 8 evidence items).

| All-time, pre-open | A — ensemble+gates | B — raven prior+pulse |
|---|---|---|
| Scored days | 41 | 33 |
| PASS rate (±1%) | 46.34% | 51.52% |
| Directional accuracy | 48.78% | 57.58% |
| MAPE | 1.52% | 1.60% |
| Edge vs free baseline | 0.01% | 0.05% |

Clean paired days: 33 (of 56 paired); B wins 17/33 decisive; mean daily APE delta (A−B) 0.06%; CRPS delta (A−B) anytime-valid 95% CS [-0.115%, +0.198%] (undecided).
_This is a research experiment, not financial advice._
