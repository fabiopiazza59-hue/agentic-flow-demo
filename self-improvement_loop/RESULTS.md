# 📈 AMZN Daily Close Predictor — Results

_Auto-generated after every run. Verdicts use pre-open forecasts only and anytime-valid 95% confidence sequences, so they stay valid although this page is re-read daily. PASS = predicted close within ±1% of actual._

## Verdict
❌ **No edge yet** — all-time MAPE 1.55% is nominally ahead of the free baseline 1.58% by 0.03%, but the difference is not distinguishable from noise (paired over 51 pre-open days: mean 0.03%, anytime-valid 95% CS [-0.090%, +0.141%]; fixed-sample 90% CI [-0.07%, 0.10%], sign test p=0.7798).

**Next forecast (2026-10-21):** close ≈ **262.64** (up, P(up) 50%, 80% interval 257.91–267.45) vs prior close 262.63; anchored to 263.46 (yfinance_ext, live, refreshed 0×).

## Scoreboard — all-time, pre-open forecasts only

_Every forecaster is scored as a calibrated predictive distribution (CRPS: lower is better; coverage should match its nominal level). The **free baseline** is the latest pre-open trade the forecast was anchored to — the prior close when there was none._

| Metric | Model (arm A) | Random walk (prior close) | Free baseline (pre-open price) |
|---|---|---|---|
| Days | 51 | 51 | 51 |
| MAPE | 1.55% | 1.57% | 1.58% |
| CRPS | 1.28% | 1.29% | 1.29% |
| 80% interval coverage | 84.31% | 80.39% | 82.35% |
| 50% interval coverage | 52.94% | 52.94% | 52.94% |
| Brier of P(up) | 0.245 | 0.250 | 0.251 |

**Does the model add value?** Paired daily gains, anytime-valid 95% confidence sequences (valid although this page is re-read every day):

| Comparison | Mean daily gain | 95% CS | Verdict |
|---|---|---|---|
| APE vs free baseline | 0.03% | [-0.090%, +0.141%] | ≈ not distinguishable |
| APE vs random walk | 0.02% | [-0.143%, +0.182%] | ≈ not distinguishable |
| CRPS vs free baseline | 0.01% | [-0.050%, +0.075%] | ≈ not distinguishable |
| CRPS vs random walk | 0.02% | [-0.086%, +0.119%] | ≈ not distinguishable |
| Direction (Brier of P(up)) vs random walk | 0.005 | [-0.044, +0.053] | ≈ not distinguishable |

## Self-improvement lab — measured, not narrated

Aggregation rules are replayed over every pre-open day using only earlier data. A rule replaces the equal-weight mean only when its anytime-valid confidence sequence (Bonferroni over 5 challengers, α=0.05) shows it is better; the judge and the gates are switched off the same way once shown to hurt ([`evals/lab.py`](src/evals/lab.py)).

**Champion: `mean`** (reference — no challenger has proven better yet; 51 days replayed).

| Rule | MAPE | Gain vs mean | CS (α/K) | Status |
|---|---|---|---|---|
| mean (reference) | 1.43% | — | — | 🏆 champion |
| median | 1.47% | -0.04% | [-0.194%, +0.120%] | ≈ not distinguishable |
| trimmed_mean | 1.48% | -0.04% | [-0.159%, +0.076%] | ≈ not distinguishable |
| inverse_mse | 1.46% | -0.02% | [-0.078%, +0.029%] | ≈ not distinguishable |
| best_recent | 1.69% | -0.26% | [-0.544%, +0.030%] | ≈ not distinguishable |
| shrink_half | 1.50% | -0.07% | [-0.180%, +0.047%] | ≈ not distinguishable |

- **Judge adjustment** (APE of blend − APE after the judge's clamped σ-move; v2 rows): n=17, mean -0.01%, CS [-0.066%, +0.053%] → judge **on**.
- **Guardrail gates** (APE before − after the gates): n=17, mean -0.01%, CS [-0.058%, +0.042%] → gates **on**.
- **Legacy v1 pipeline** (judge wrote the number) vs the plain mean of its own analysts: mean -0.17% over 34 pre-open days, CS [-0.407%, +0.059%] — ≈ not distinguishable (negative = the judge cost accuracy).

### Integrity & mechanism

- **40 scored row(s) were created after their session opened** and are excluded from every verdict. They saw part of the tape they forecast. New post-open rows are refused (`--allow-late` to override), and the schedule now researches the evening before and only re-anchors in the morning.
- **Pre-open coverage:** 18 of the last 20 sessions (since 2026-09-24) got a forecast written before the open.
- **Live anchors:** 17 of 51 pre-open forecasts were anchored on a real after-hours/pre-market trade (the rest on the prior close — before v2 the quote feed silently echoed it).
- **Gate effect** (ungated − gated APE, paired over 17 days): -0.01% 90% CI [-0.03%, 0.01%], gates helped on 4/8 decisive days (sign test p=1.0). Gates fired on 8 of them.

## Metrics

| Metric | All-time (pre-open) — verdict basis | Last 20 (all rows) | All-time (all rows) |
|---|---|---|---|
| Scored days | 51 | 20 | 91 |
| PASS rate (±1%) | 41.18% | 45.00% | 43.96% |
| Directional accuracy | 52.94% | 60.00% | 53.85% |
| MAPE | 1.55% | 1.23% | 1.48% |
| Baseline MAPE (random walk) | 1.57% | 1.29% | 1.49% |
| Edge (baseline − model) | 0.02% | 0.07% | 0.00% |
| Brier (confidence calib.) | 0.24 | 0.29 | 0.26 |

## Per-strategy scorecards

| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |
|---|---|---|---|---|
| news | 91 | 30.77% | 1.32% | 0.23 |
| technical | 91 | 13.19% | 1.45% | 0.21 |
| contrarian | 91 | 25.27% | 1.52% | 0.20 |
| momentum | 91 | 18.68% | 1.60% | 0.19 |
| macro | 91 | 12.09% | 1.66% | 0.18 |

## Last 20 scored model predictions
_(backfill seed rows are excluded from metrics and this table; they appear only as price-history context on the dashboard chart. ⚠️ = written after the open, excluded from verdicts)_

| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest | Pre-open |
|---|---|---|---|---|---|---|---|---|
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
| 2026-09-25 | 249.05 | 249.67 | 0.25% | ✅ | ❌ | ❌ | contrarian | ⚠️ |
| 2026-09-24 | 249.36 | 249.38 | 0.01% | ✅ | ✅ | ✅ | contrarian | ⚠️ |
| 2026-09-23 | 252.85 | 249.27 | 1.44% | ❌ | ✅ | ✅ | news | ⚠️ |

## 🅰️/🅱️ A/B test — ensemble+gates vs raven-style prior+pulse

_Both arms forecast the same sessions from the same pre-open snapshot (paired test). Only days on which both forecasts were written before the open count._

**Verdict:** Arm B leads on paired MAPE (mean daily delta 0.01% in B's favor over 43 pre-open days); B wins 21/43 decisive days (sign test p=1.0, descriptive); anytime-valid 95% CS [-0.203%, +0.229%] includes zero — not distinguishable from noise.

**B's next forecast (2026-10-21):** close ≈ **264.18** (up, adj 0.15σ, 7 evidence items).

| All-time, pre-open | A — ensemble+gates | B — raven prior+pulse |
|---|---|---|
| Scored days | 51 | 43 |
| PASS rate (±1%) | 41.18% | 51.16% |
| Directional accuracy | 52.94% | 60.47% |
| MAPE | 1.55% | 1.65% |
| Edge vs free baseline | 0.03% | 0.02% |

Clean paired days: 43 (of 66 paired); B wins 21/43 decisive; mean daily APE delta (A−B) 0.01%; CRPS delta (A−B) anytime-valid 95% CS [-0.127%, +0.145%] (undecided).
_This is a research experiment, not financial advice._
