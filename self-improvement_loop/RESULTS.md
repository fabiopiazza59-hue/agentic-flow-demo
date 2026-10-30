# 📈 AMZN Daily Close Predictor — Results

_Auto-generated after every run. Verdicts use pre-open forecasts only and anytime-valid 95% confidence sequences, so they stay valid although this page is re-read daily. PASS = predicted close within ±1% of actual._

## Verdict
❌ **No edge yet** — all-time MAPE 1.55% is nominally ahead of the free baseline 1.60% by 0.05%, but the difference is not distinguishable from noise (paired over 58 pre-open days: mean 0.05%, anytime-valid 95% CS [-0.061%, +0.151%]; fixed-sample 90% CI [-0.03%, 0.13%], sign test p=0.8957).

**Next forecast (2026-10-30):** close ≈ **227.77** (down, P(up) 2%, 80% interval 223.21–232.42) vs prior close 243.79; anchored to 225.99 (yfinance_ext, live, refreshed 0×).

## Scoreboard — all-time, pre-open forecasts only

_Every forecaster is scored as a calibrated predictive distribution (CRPS: lower is better; coverage should match its nominal level). The **free baseline** is the latest pre-open trade the forecast was anchored to — the prior close when there was none._

| Metric | Model (arm A) | Random walk (prior close) | Free baseline (pre-open price) |
|---|---|---|---|
| Days | 58 | 58 | 58 |
| MAPE | 1.55% | 1.61% | 1.60% |
| CRPS | 1.28% | 1.32% | 1.31% |
| 80% interval coverage | 82.76% | 79.31% | 81.03% |
| 50% interval coverage | 53.45% | 53.45% | 53.45% |
| Brier of P(up) | 0.236 | 0.250 | 0.247 |

**Does the model add value?** Paired daily gains, anytime-valid 95% confidence sequences (valid although this page is re-read every day):

| Comparison | Mean daily gain | 95% CS | Verdict |
|---|---|---|---|
| APE vs free baseline | 0.05% | [-0.061%, +0.151%] | ≈ not distinguishable |
| APE vs random walk | 0.06% | [-0.096%, +0.208%] | ≈ not distinguishable |
| CRPS vs free baseline | 0.03% | [-0.036%, +0.088%] | ≈ not distinguishable |
| CRPS vs random walk | 0.04% | [-0.060%, +0.145%] | ≈ not distinguishable |
| Direction (Brier of P(up)) vs random walk | 0.014 | [-0.030, +0.059] | ≈ not distinguishable |

## Self-improvement lab — measured, not narrated

Aggregation rules are replayed over every pre-open day using only earlier data. A rule replaces the equal-weight mean only when its anytime-valid confidence sequence (Bonferroni over 5 challengers, α=0.05) shows it is better; the judge and the gates are switched off the same way once shown to hurt ([`evals/lab.py`](src/evals/lab.py)).

**Champion: `mean`** (reference — no challenger has proven better yet; 58 days replayed).

| Rule | MAPE | Gain vs mean | CS (α/K) | Status |
|---|---|---|---|---|
| mean (reference) | 1.46% | — | — | 🏆 champion |
| median | 1.50% | -0.05% | [-0.191%, +0.098%] | ≈ not distinguishable |
| trimmed_mean | 1.50% | -0.04% | [-0.146%, +0.062%] | ≈ not distinguishable |
| inverse_mse | 1.47% | -0.02% | [-0.067%, +0.030%] | ≈ not distinguishable |
| best_recent | 1.67% | -0.21% | [-0.479%, +0.053%] | ≈ not distinguishable |
| shrink_half | 1.52% | -0.06% | [-0.164%, +0.036%] | ≈ not distinguishable |

- **Judge adjustment** (APE of blend − APE after the judge's clamped σ-move; v2 rows): n=24, mean 0.02%, CS [-0.042%, +0.084%] → judge **on**.
- **Guardrail gates** (APE before − after the gates): n=24, mean -0.00%, CS [-0.038%, +0.032%] → gates **on**.
- **Legacy v1 pipeline** (judge wrote the number) vs the plain mean of its own analysts: mean -0.17% over 34 pre-open days, CS [-0.407%, +0.059%] — ≈ not distinguishable (negative = the judge cost accuracy).

### Integrity & mechanism

- **40 scored row(s) were created after their session opened** and are excluded from every verdict. They saw part of the tape they forecast. New post-open rows are refused (`--allow-late` to override), and the schedule now researches the evening before and only re-anchors in the morning.
- **Pre-open coverage:** 20 of the last 20 sessions (since 2026-10-05) got a forecast written before the open.
- **Live anchors:** 24 of 58 pre-open forecasts were anchored on a real after-hours/pre-market trade (the rest on the prior close — before v2 the quote feed silently echoed it).
- **Gate effect** (ungated − gated APE, paired over 24 days): -0.00% 90% CI [-0.02%, 0.01%], gates helped on 5/9 decisive days (sign test p=1.0). Gates fired on 9 of them.

## Metrics

| Metric | All-time (pre-open) — verdict basis | Last 20 (all rows) | All-time (all rows) |
|---|---|---|---|
| Scored days | 58 | 20 | 98 |
| PASS rate (±1%) | 43.10% | 40.00% | 44.90% |
| Directional accuracy | 56.90% | 75.00% | 56.12% |
| MAPE | 1.55% | 1.51% | 1.49% |
| Baseline MAPE (random walk) | 1.61% | 1.66% | 1.51% |
| Edge (baseline − model) | 0.06% | 0.16% | 0.03% |
| Brier (confidence calib.) | 0.25 | 0.29 | 0.26 |

## Per-strategy scorecards

| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |
|---|---|---|---|---|
| news | 98 | 33.67% | 1.33% | 0.23 |
| technical | 98 | 12.24% | 1.48% | 0.21 |
| contrarian | 98 | 25.51% | 1.54% | 0.20 |
| momentum | 98 | 17.35% | 1.61% | 0.19 |
| macro | 98 | 11.22% | 1.67% | 0.18 |

## Last 20 scored model predictions
_(backfill seed rows are excluded from metrics and this table; they appear only as price-history context on the dashboard chart. ⚠️ = written after the open, excluded from verdicts)_

| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest | Pre-open |
|---|---|---|---|---|---|---|---|---|
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
| 2026-10-02 | 249.13 | 251.52 | 0.95% | ✅ | ✅ | ✅ | news | ✅ |

## 🅰️/🅱️ A/B test — ensemble+gates vs raven-style prior+pulse

_Both arms forecast the same sessions from the same pre-open snapshot (paired test). Only days on which both forecasts were written before the open count._

**Verdict:** Arm B leads on paired MAPE (mean daily delta 0.01% in B's favor over 50 pre-open days); B wins 26/50 decisive days (sign test p=0.8877, descriptive); anytime-valid 95% CS [-0.175%, +0.205%] includes zero — not distinguishable from noise.

**B's next forecast (2026-10-30):** close ≈ **224.96** (down, adj -0.2σ, 8 evidence items).

| All-time, pre-open | A — ensemble+gates | B — raven prior+pulse |
|---|---|---|
| Scored days | 58 | 50 |
| PASS rate (±1%) | 43.10% | 52.00% |
| Directional accuracy | 56.90% | 64.00% |
| MAPE | 1.55% | 1.63% |
| Edge vs free baseline | 0.05% | 0.05% |

Clean paired days: 50 (of 73 paired); B wins 26/50 decisive; mean daily APE delta (A−B) 0.01%; CRPS delta (A−B) anytime-valid 95% CS [-0.107%, +0.128%] (undecided).
_This is a research experiment, not financial advice._
