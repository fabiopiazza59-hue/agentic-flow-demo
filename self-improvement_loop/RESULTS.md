# 📈 AMZN Daily Close Predictor — Results

_Auto-generated after every run. Verdicts use pre-open forecasts only and anytime-valid 95% confidence sequences, so they stay valid although this page is re-read daily. PASS = predicted close within ±1% of actual._

## Verdict
❌ **No edge yet** — all-time MAPE 1.56% is nominally ahead of the free baseline 1.57% by 0.01%, but the difference is not distinguishable from noise (paired over 44 pre-open days: mean 0.01%, anytime-valid 95% CS [-0.116%, +0.140%]; fixed-sample 90% CI [-0.04%, 0.11%], sign test p=0.4514).

**Next forecast (2026-10-12):** close ≈ **262.39** (down, P(up) 49%, 80% interval 258.00–266.85) vs prior close 262.43; anchored to 262.53 (yfinance_ext, live, refreshed 0×).

## Scoreboard — all-time, pre-open forecasts only

_Every forecaster is scored as a calibrated predictive distribution (CRPS: lower is better; coverage should match its nominal level). The **free baseline** is the latest pre-open trade the forecast was anchored to — the prior close when there was none._

| Metric | Model (arm A) | Random walk (prior close) | Free baseline (pre-open price) |
|---|---|---|---|
| Days | 44 | 44 | 44 |
| MAPE | 1.56% | 1.59% | 1.57% |
| CRPS | 1.28% | 1.31% | 1.29% |
| 80% interval coverage | 84.09% | 84.09% | 84.09% |
| 50% interval coverage | 56.82% | 54.55% | 56.82% |
| Brier of P(up) | 0.237 | 0.250 | 0.243 |

**Does the model add value?** Paired daily gains, anytime-valid 95% confidence sequences (valid although this page is re-read every day):

| Comparison | Mean daily gain | 95% CS | Verdict |
|---|---|---|---|
| APE vs free baseline | 0.01% | [-0.116%, +0.140%] | ≈ not distinguishable |
| APE vs random walk | 0.03% | [-0.111%, +0.173%] | ≈ not distinguishable |
| CRPS vs free baseline | 0.01% | [-0.059%, +0.080%] | ≈ not distinguishable |
| CRPS vs random walk | 0.03% | [-0.056%, +0.110%] | ≈ not distinguishable |
| Direction (Brier of P(up)) vs random walk | 0.013 | [-0.023, +0.050] | ≈ not distinguishable |

## Self-improvement lab — measured, not narrated

Aggregation rules are replayed over every pre-open day using only earlier data. A rule replaces the equal-weight mean only when its anytime-valid confidence sequence (Bonferroni over 5 challengers, α=0.05) shows it is better; the judge and the gates are switched off the same way once shown to hurt ([`evals/lab.py`](src/evals/lab.py)).

**Champion: `mean`** (reference — no challenger has proven better yet; 44 days replayed).

| Rule | MAPE | Gain vs mean | CS (α/K) | Status |
|---|---|---|---|---|
| mean (reference) | 1.42% | — | — | 🏆 champion |
| median | 1.45% | -0.03% | [-0.206%, +0.148%] | ≈ not distinguishable |
| trimmed_mean | 1.46% | -0.04% | [-0.171%, +0.097%] | ≈ not distinguishable |
| inverse_mse | 1.45% | -0.03% | [-0.093%, +0.030%] | ≈ not distinguishable |
| best_recent | 1.69% | -0.27% | [-0.590%, +0.058%] | ≈ not distinguishable |
| shrink_half | 1.49% | -0.07% | [-0.199%, +0.065%] | ≈ not distinguishable |

- **Judge adjustment** (APE of blend − APE after the judge's clamped σ-move; v2 rows): n=10, mean 0.00%, CS [-0.000%, +0.000%] → judge **on**.
- **Guardrail gates** (APE before − after the gates): n=10, mean -0.01%, CS [-0.112%, +0.084%] → gates **on**.
- **Legacy v1 pipeline** (judge wrote the number) vs the plain mean of its own analysts: mean -0.17% over 34 pre-open days, CS [-0.407%, +0.059%] — ≈ not distinguishable (negative = the judge cost accuracy).

### Integrity & mechanism

- **40 scored row(s) were created after their session opened** and are excluded from every verdict. They saw part of the tape they forecast. New post-open rows are refused (`--allow-late` to override), and the schedule now researches the evening before and only re-anchors in the morning.
- **Pre-open coverage:** 11 of the last 20 sessions (since 2026-09-15) got a forecast written before the open.
- **Live anchors:** 10 of 44 pre-open forecasts were anchored on a real after-hours/pre-market trade (the rest on the prior close — before v2 the quote feed silently echoed it).
- **Gate effect** (ungated − gated APE, paired over 10 days): -0.01% 90% CI [-0.05%, 0.02%], gates helped on 4/8 decisive days (sign test p=1.0). Gates fired on 8 of them.

## Metrics

| Metric | All-time (pre-open) — verdict basis | Last 20 (all rows) | All-time (all rows) |
|---|---|---|---|
| Scored days | 44 | 20 | 84 |
| PASS rate (±1%) | 43.18% | 50.00% | 45.24% |
| Directional accuracy | 50.00% | 60.00% | 52.38% |
| MAPE | 1.56% | 1.16% | 1.48% |
| Baseline MAPE (random walk) | 1.59% | 1.31% | 1.49% |
| Edge (baseline − model) | 0.03% | 0.15% | 0.01% |
| Brier (confidence calib.) | 0.23 | 0.29 | 0.26 |

## Per-strategy scorecards

| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |
|---|---|---|---|---|
| news | 84 | 32.14% | 1.30% | 0.23 |
| technical | 84 | 13.10% | 1.44% | 0.21 |
| contrarian | 84 | 22.62% | 1.54% | 0.19 |
| momentum | 84 | 20.24% | 1.57% | 0.19 |
| macro | 84 | 11.90% | 1.65% | 0.18 |

## Last 20 scored model predictions
_(backfill seed rows are excluded from metrics and this table; they appear only as price-history context on the dashboard chart. ⚠️ = written after the open, excluded from verdicts)_

| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest | Pre-open |
|---|---|---|---|---|---|---|---|---|
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
| 2026-09-22 | 259.42 | 254.98 | 1.74% | ❌ | ❌ | ❌ | contrarian | ⚠️ |
| 2026-09-21 | 255.15 | 258.45 | 1.28% | ❌ | ✅ | ✅ | news | ⚠️ |
| 2026-09-18 | 251.27 | 253.71 | 0.96% | ✅ | ✅ | ✅ | news | ⚠️ |
| 2026-09-17 | 245.10 | 251.19 | 2.42% | ❌ | ❌ | ❌ | news | ⚠️ |
| 2026-09-16 | 247.05 | 245.96 | 0.44% | ✅ | ✅ | ✅ | macro | ⚠️ |
| 2026-09-15 | 252.45 | 248.42 | 1.62% | ❌ | ✅ | ✅ | momentum | ⚠️ |
| 2026-09-14 | 255.60 | 253.54 | 0.81% | ✅ | ✅ | ✅ | news | ⚠️ |

## 🅰️/🅱️ A/B test — ensemble+gates vs raven-style prior+pulse

_Both arms forecast the same sessions from the same pre-open snapshot (paired test). Only days on which both forecasts were written before the open count._

**Verdict:** Arm B leads on paired MAPE (mean daily delta 0.06% in B's favor over 36 pre-open days); B wins 19/36 decisive days (sign test p=0.8679, descriptive); anytime-valid 95% CS [-0.176%, +0.300%] includes zero — not distinguishable from noise.

**B's next forecast (2026-10-12):** close ≈ **261.69** (down, adj -0.2σ, 8 evidence items).

| All-time, pre-open | A — ensemble+gates | B — raven prior+pulse |
|---|---|---|
| Scored days | 44 | 36 |
| PASS rate (±1%) | 43.18% | 50.00% |
| Directional accuracy | 50.00% | 61.11% |
| MAPE | 1.56% | 1.63% |
| Edge vs free baseline | 0.01% | 0.05% |

Clean paired days: 36 (of 59 paired); B wins 19/36 decisive; mean daily APE delta (A−B) 0.06%; CRPS delta (A−B) anytime-valid 95% CS [-0.112%, +0.187%] (undecided).
_This is a research experiment, not financial advice._
