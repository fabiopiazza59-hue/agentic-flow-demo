# 📈 AMZN Daily Close Predictor — Results

_Auto-generated after every run. Verdicts use pre-open forecasts only and anytime-valid 95% confidence sequences, so they stay valid although this page is re-read daily. PASS = predicted close within ±1% of actual._

## Verdict
❌ **No edge yet** — all-time MAPE 1.67% is nominally ahead of the free baseline 1.75% by 0.08%, but the difference is not distinguishable from noise (paired over 62 pre-open days: mean 0.08%, anytime-valid 95% CS [-0.044%, +0.205%]; fixed-sample 90% CI [-0.13%, 0.14%], sign test p=0.8991).

**Next forecast (2026-11-05):** close ≈ **262.97** (down, P(up) 29%, 80% interval 256.31–269.80) vs prior close 266.03; anchored to 265.46 (yfinance_ext, live, refreshed 0×).

## Scoreboard — all-time, pre-open forecasts only

_Every forecaster is scored as a calibrated predictive distribution (CRPS: lower is better; coverage should match its nominal level). The **free baseline** is the latest pre-open trade the forecast was anchored to — the prior close when there was none._

| Metric | Model (arm A) | Random walk (prior close) | Free baseline (pre-open price) |
|---|---|---|---|
| Days | 62 | 62 | 62 |
| MAPE | 1.67% | 1.69% | 1.75% |
| CRPS | 1.38% | 1.38% | 1.44% |
| 80% interval coverage | 80.65% | 77.42% | 79.03% |
| 50% interval coverage | 53.23% | 50.00% | 50.00% |
| Brier of P(up) | 0.230 | 0.250 | 0.244 |

**Does the model add value?** Paired daily gains, anytime-valid 95% confidence sequences (valid although this page is re-read every day):

| Comparison | Mean daily gain | 95% CS | Verdict |
|---|---|---|---|
| APE vs free baseline | 0.08% | [-0.044%, +0.205%] | ≈ not distinguishable |
| APE vs random walk | 0.02% | [-0.243%, +0.273%] | ≈ not distinguishable |
| CRPS vs free baseline | 0.05% | [-0.028%, +0.138%] | ≈ not distinguishable |
| CRPS vs random walk | -0.00% | [-0.211%, +0.204%] | ≈ not distinguishable |
| Direction (Brier of P(up)) vs random walk | 0.020 | [-0.027, +0.067] | ≈ not distinguishable |

## Self-improvement lab — measured, not narrated

Aggregation rules are replayed over every pre-open day using only earlier data. A rule replaces the equal-weight mean only when its anytime-valid confidence sequence (Bonferroni over 5 challengers, α=0.05) shows it is better; the judge and the gates are switched off the same way once shown to hurt ([`evals/lab.py`](src/evals/lab.py)).

**Champion: `mean`** (reference — no challenger has proven better yet; 62 days replayed).

| Rule | MAPE | Gain vs mean | CS (α/K) | Status |
|---|---|---|---|---|
| mean (reference) | 1.59% | — | — | 🏆 champion |
| median | 1.63% | -0.04% | [-0.178%, +0.092%] | ≈ not distinguishable |
| trimmed_mean | 1.63% | -0.04% | [-0.137%, +0.057%] | ≈ not distinguishable |
| inverse_mse | 1.61% | -0.02% | [-0.062%, +0.028%] | ≈ not distinguishable |
| best_recent | 1.80% | -0.21% | [-0.461%, +0.037%] | ≈ not distinguishable |
| shrink_half | 1.67% | -0.07% | [-0.173%, +0.024%] | ≈ not distinguishable |

- **Judge adjustment** (APE of blend − APE after the judge's clamped σ-move; v2 rows): n=28, mean 0.04%, CS [-0.030%, +0.108%] → judge **on**.
- **Guardrail gates** (APE before − after the gates): n=28, mean -0.00%, CS [-0.032%, +0.027%] → gates **on**.
- **Legacy v1 pipeline** (judge wrote the number) vs the plain mean of its own analysts: mean -0.17% over 34 pre-open days, CS [-0.407%, +0.059%] — ≈ not distinguishable (negative = the judge cost accuracy).

### Integrity & mechanism

- **40 scored row(s) were created after their session opened** and are excluded from every verdict. They saw part of the tape they forecast. New post-open rows are refused (`--allow-late` to override), and the schedule now researches the evening before and only re-anchors in the morning.
- **Pre-open coverage:** 20 of the last 20 sessions (since 2026-10-09) got a forecast written before the open.
- **Live anchors:** 28 of 62 pre-open forecasts were anchored on a real after-hours/pre-market trade (the rest on the prior close — before v2 the quote feed silently echoed it).
- **Gate effect** (ungated − gated APE, paired over 28 days): -0.00% 90% CI [-0.02%, 0.01%], gates helped on 5/9 decisive days (sign test p=1.0). Gates fired on 9 of them.

## Metrics

| Metric | All-time (pre-open) — verdict basis | Last 20 (all rows) | All-time (all rows) |
|---|---|---|---|
| Scored days | 62 | 20 | 102 |
| PASS rate (±1%) | 41.94% | 35.00% | 44.12% |
| Directional accuracy | 58.06% | 75.00% | 56.86% |
| MAPE | 1.67% | 2.01% | 1.56% |
| Baseline MAPE (random walk) | 1.69% | 2.00% | 1.56% |
| Edge (baseline − model) | 0.02% | -0.01% | 0.00% |
| Brier (confidence calib.) | 0.24 | 0.28 | 0.26 |

## Per-strategy scorecards

| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |
|---|---|---|---|---|
| news | 102 | 32.35% | 1.42% | 0.22 |
| technical | 102 | 11.76% | 1.56% | 0.20 |
| contrarian | 102 | 27.45% | 1.60% | 0.20 |
| momentum | 102 | 16.67% | 1.70% | 0.19 |
| macro | 102 | 11.76% | 1.74% | 0.18 |

## Last 20 scored model predictions
_(backfill seed rows are excluded from metrics and this table; they appear only as price-history context on the dashboard chart. ⚠️ = written after the open, excluded from verdicts)_

| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest | Pre-open |
|---|---|---|---|---|---|---|---|---|
| 2026-11-04 | 248.22 | 266.03 | 6.69% | ❌ | ❌ | ❌ | contrarian | ✅ |
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

## 🅰️/🅱️ A/B test — ensemble+gates vs raven-style prior+pulse

_Both arms forecast the same sessions from the same pre-open snapshot (paired test). Only days on which both forecasts were written before the open count._

**Verdict:** Arm A leads on paired MAPE (mean daily delta 0.02% in A's favor over 54 pre-open days); B wins 27/54 decisive days (sign test p=1.0, descriptive); anytime-valid 95% CS [-0.222%, +0.179%] includes zero — not distinguishable from noise.

**B's next forecast (2026-11-05):** close ≈ **266.53** (up, adj 0.15σ, 7 evidence items).

| All-time, pre-open | A — ensemble+gates | B — raven prior+pulse |
|---|---|---|
| Scored days | 62 | 54 |
| PASS rate (±1%) | 41.94% | 48.15% |
| Directional accuracy | 58.06% | 64.81% |
| MAPE | 1.67% | 1.80% |
| Edge vs free baseline | 0.08% | 0.06% |

Clean paired days: 54 (of 77 paired); B wins 27/54 decisive; mean daily APE delta (A−B) -0.02%; CRPS delta (A−B) anytime-valid 95% CS [-0.159%, +0.120%] (undecided).
_This is a research experiment, not financial advice._
