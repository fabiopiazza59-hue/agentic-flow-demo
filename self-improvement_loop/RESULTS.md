# 📈 AMZN Daily Close Predictor — Results

_Auto-generated after every run. Verdicts use pre-open forecasts only and anytime-valid 95% confidence sequences, so they stay valid although this page is re-read daily. PASS = predicted close within ±1% of actual._

## Verdict
❌ **No edge yet** — all-time MAPE 1.62% is nominally ahead of the free baseline 1.63% by 0.01%, but the difference is not distinguishable from noise (paired over 36 pre-open days: mean 0.01%, anytime-valid 95% CS [-0.148%, +0.168%]; fixed-sample 90% CI [-0.07%, 0.09%], sign test p=0.243).

**Next forecast (2026-09-30):** close ≈ **247.28** (up, P(up) 59%, 80% interval 243.44–251.18) vs prior close 246.67; anchored to 247.30 (yfinance_ext, live, refreshed 0×).

## Scoreboard — all-time, pre-open forecasts only

_Every forecaster is scored as a calibrated predictive distribution (CRPS: lower is better; coverage should match its nominal level). The **free baseline** is the latest pre-open trade the forecast was anchored to — the prior close when there was none._

| Metric | Model (arm A) | Random walk (prior close) | Free baseline (pre-open price) |
|---|---|---|---|
| Days | 36 | 36 | 36 |
| MAPE | 1.62% | 1.63% | 1.63% |
| CRPS | 1.35% | 1.36% | 1.36% |
| 80% interval coverage | 88.89% | 88.89% | 88.89% |
| 50% interval coverage | 61.11% | 61.11% | 61.11% |
| Brier of P(up) | 0.241 | 0.250 | 0.251 |

**Does the model add value?** Paired daily gains, anytime-valid 95% confidence sequences (valid although this page is re-read every day):

| Comparison | Mean daily gain | 95% CS | Verdict |
|---|---|---|---|
| APE vs free baseline | 0.01% | [-0.148%, +0.168%] | ≈ not distinguishable |
| APE vs random walk | 0.01% | [-0.150%, +0.167%] | ≈ not distinguishable |
| CRPS vs free baseline | 0.01% | [-0.078%, +0.093%] | ≈ not distinguishable |
| CRPS vs random walk | 0.01% | [-0.081%, +0.091%] | ≈ not distinguishable |
| Direction (Brier of P(up)) vs random walk | 0.009 | [-0.028, +0.045] | ≈ not distinguishable |

## Self-improvement lab — measured, not narrated

Aggregation rules are replayed over every pre-open day using only earlier data. A rule replaces the equal-weight mean only when its anytime-valid confidence sequence (Bonferroni over 5 challengers, α=0.05) shows it is better; the judge and the gates are switched off the same way once shown to hurt ([`evals/lab.py`](src/evals/lab.py)).

**Champion: `mean`** (reference — no challenger has proven better yet; 36 days replayed).

| Rule | MAPE | Gain vs mean | CS (α/K) | Status |
|---|---|---|---|---|
| mean (reference) | 1.45% | — | — | 🏆 champion |
| median | 1.48% | -0.02% | [-0.239%, +0.189%] | ≈ not distinguishable |
| trimmed_mean | 1.50% | -0.04% | [-0.207%, +0.117%] | ≈ not distinguishable |
| inverse_mse | 1.48% | -0.03% | [-0.107%, +0.043%] | ≈ not distinguishable |
| best_recent | 1.74% | -0.29% | [-0.683%, +0.113%] | ≈ not distinguishable |
| shrink_half | 1.53% | -0.08% | [-0.240%, +0.083%] | ≈ not distinguishable |

- **Judge adjustment** (APE of blend − APE after the judge's clamped σ-move; v2 rows): n=2, mean 0.00%, CS n=2 (too few) → judge **on**.
- **Guardrail gates** (APE before − after the gates): n=2, mean -0.02%, CS n=2 (too few) → gates **on**.
- **Legacy v1 pipeline** (judge wrote the number) vs the plain mean of its own analysts: mean -0.17% over 34 pre-open days, CS [-0.407%, +0.059%] — ≈ not distinguishable (negative = the judge cost accuracy).

### Integrity & mechanism

- **40 scored row(s) were created after their session opened** and are excluded from every verdict. They saw part of the tape they forecast. New post-open rows are refused (`--allow-late` to override), and the schedule now researches the evening before and only re-anchors in the morning.
- **Pre-open coverage:** 3 of the last 20 sessions (since 2026-09-02) got a forecast written before the open.
- **Live anchors:** 2 of 36 pre-open forecasts were anchored on a real after-hours/pre-market trade (the rest on the prior close — before v2 the quote feed silently echoed it).
- **Gate effect** (ungated − gated APE, paired over 2 days): -0.02% 90% CI [-0.03%, 0.00%], gates helped on 0/1 decisive days (sign test p=1.0). Gates fired on 1 of them.

## Metrics

| Metric | All-time (pre-open) — verdict basis | Last 20 (all rows) | All-time (all rows) |
|---|---|---|---|
| Scored days | 36 | 20 | 76 |
| PASS rate (±1%) | 41.67% | 50.00% | 44.74% |
| Directional accuracy | 47.22% | 60.00% | 51.32% |
| MAPE | 1.62% | 1.04% | 1.50% |
| Baseline MAPE (random walk) | 1.63% | 1.14% | 1.49% |
| Edge (baseline − model) | 0.01% | 0.10% | -0.01% |
| Brier (confidence calib.) | 0.22 | 0.28 | 0.26 |

## Per-strategy scorecards

| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |
|---|---|---|---|---|
| news | 76 | 32.89% | 1.31% | 0.23 |
| technical | 76 | 14.47% | 1.44% | 0.21 |
| contrarian | 76 | 19.74% | 1.56% | 0.19 |
| momentum | 76 | 21.05% | 1.59% | 0.19 |
| macro | 76 | 11.84% | 1.67% | 0.18 |

## Last 20 scored model predictions
_(backfill seed rows are excluded from metrics and this table; they appear only as price-history context on the dashboard chart. ⚠️ = written after the open, excluded from verdicts)_

| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest | Pre-open |
|---|---|---|---|---|---|---|---|---|
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
| 2026-09-08 | 258.28 | 256.97 | 0.51% | ✅ | ✅ | ✅ | macro | ⚠️ |
| 2026-09-04 | 258.82 | 258.51 | 0.12% | ✅ | ✅ | ✅ | technical | ⚠️ |
| 2026-09-03 | 254.70 | 258.90 | 1.62% | ❌ | ❌ | ❌ | contrarian | ⚠️ |
| 2026-09-02 | 253.69 | 254.98 | 0.51% | ✅ | ❌ | ❌ | momentum | ⚠️ |
| 2026-09-01 | 258.91 | 254.92 | 1.57% | ❌ | ✅ | ✅ | news | ⚠️ |

## 🅰️/🅱️ A/B test — ensemble+gates vs raven-style prior+pulse

_Both arms forecast the same sessions from the same pre-open snapshot (paired test). Only days on which both forecasts were written before the open count._

**Verdict:** Arm B leads on paired MAPE (mean daily delta 0.06% in B's favor over 28 pre-open days); B wins 14/28 decisive days (sign test p=1.0, descriptive); anytime-valid 95% CS [-0.245%, +0.364%] includes zero — not distinguishable from noise.

**B's next forecast (2026-09-30):** close ≈ **247.64** (up, adj 0.1σ, 8 evidence items).

| All-time, pre-open | A — ensemble+gates | B — raven prior+pulse |
|---|---|---|
| Scored days | 36 | 28 |
| PASS rate (±1%) | 41.67% | 46.43% |
| Directional accuracy | 47.22% | 57.14% |
| MAPE | 1.62% | 1.74% |
| Edge vs free baseline | 0.01% | 0.04% |

Clean paired days: 28 (of 51 paired); B wins 14/28 decisive; mean daily APE delta (A−B) 0.06%; CRPS delta (A−B) anytime-valid 95% CS [-0.143%, +0.227%] (undecided).
_This is a research experiment, not financial advice._
