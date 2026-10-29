# 📈 AMZN Daily Close Predictor — Results

_Auto-generated after every run. Verdicts use pre-open forecasts only and anytime-valid 95% confidence sequences, so they stay valid although this page is re-read daily. PASS = predicted close within ±1% of actual._

## Verdict
❌ **No edge yet** — all-time MAPE 1.51% is nominally ahead of the free baseline 1.55% by 0.04%, but the difference is not distinguishable from noise (paired over 57 pre-open days: mean 0.04%, anytime-valid 95% CS [-0.067%, +0.145%]; fixed-sample 90% CI [-0.04%, 0.12%], sign test p=1.0).

**Next forecast (2026-10-29):** close ≈ **252.81** (down, P(up) 32%, 80% interval 248.24–257.46) vs prior close 254.60; anchored to 253.82 (yfinance_ext, live, refreshed 0×).

## Scoreboard — all-time, pre-open forecasts only

_Every forecaster is scored as a calibrated predictive distribution (CRPS: lower is better; coverage should match its nominal level). The **free baseline** is the latest pre-open trade the forecast was anchored to — the prior close when there was none._

| Metric | Model (arm A) | Random walk (prior close) | Free baseline (pre-open price) |
|---|---|---|---|
| Days | 57 | 57 | 57 |
| MAPE | 1.51% | 1.56% | 1.55% |
| CRPS | 1.25% | 1.28% | 1.27% |
| 80% interval coverage | 84.21% | 80.70% | 82.46% |
| 50% interval coverage | 54.39% | 54.39% | 54.39% |
| Brier of P(up) | 0.238 | 0.250 | 0.248 |

**Does the model add value?** Paired daily gains, anytime-valid 95% confidence sequences (valid although this page is re-read every day):

| Comparison | Mean daily gain | 95% CS | Verdict |
|---|---|---|---|
| APE vs free baseline | 0.04% | [-0.067%, +0.145%] | ≈ not distinguishable |
| APE vs random walk | 0.04% | [-0.106%, +0.194%] | ≈ not distinguishable |
| CRPS vs free baseline | 0.02% | [-0.040%, +0.073%] | ≈ not distinguishable |
| CRPS vs random walk | 0.03% | [-0.067%, +0.123%] | ≈ not distinguishable |
| Direction (Brier of P(up)) vs random walk | 0.012 | [-0.032, +0.057] | ≈ not distinguishable |

## Self-improvement lab — measured, not narrated

Aggregation rules are replayed over every pre-open day using only earlier data. A rule replaces the equal-weight mean only when its anytime-valid confidence sequence (Bonferroni over 5 challengers, α=0.05) shows it is better; the judge and the gates are switched off the same way once shown to hurt ([`evals/lab.py`](src/evals/lab.py)).

**Champion: `mean`** (reference — no challenger has proven better yet; 57 days replayed).

| Rule | MAPE | Gain vs mean | CS (α/K) | Status |
|---|---|---|---|---|
| mean (reference) | 1.41% | — | — | 🏆 champion |
| median | 1.46% | -0.05% | [-0.194%, +0.101%] | ≈ not distinguishable |
| trimmed_mean | 1.45% | -0.04% | [-0.149%, +0.063%] | ≈ not distinguishable |
| inverse_mse | 1.43% | -0.02% | [-0.068%, +0.030%] | ≈ not distinguishable |
| best_recent | 1.64% | -0.22% | [-0.492%, +0.045%] | ≈ not distinguishable |
| shrink_half | 1.48% | -0.06% | [-0.166%, +0.038%] | ≈ not distinguishable |

- **Judge adjustment** (APE of blend − APE after the judge's clamped σ-move; v2 rows): n=23, mean 0.01%, CS [-0.045%, +0.067%] → judge **on**.
- **Guardrail gates** (APE before − after the gates): n=23, mean -0.00%, CS [-0.040%, +0.033%] → gates **on**.
- **Legacy v1 pipeline** (judge wrote the number) vs the plain mean of its own analysts: mean -0.17% over 34 pre-open days, CS [-0.407%, +0.059%] — ≈ not distinguishable (negative = the judge cost accuracy).

### Integrity & mechanism

- **40 scored row(s) were created after their session opened** and are excluded from every verdict. They saw part of the tape they forecast. New post-open rows are refused (`--allow-late` to override), and the schedule now researches the evening before and only re-anchors in the morning.
- **Pre-open coverage:** 20 of the last 20 sessions (since 2026-10-02) got a forecast written before the open.
- **Live anchors:** 23 of 57 pre-open forecasts were anchored on a real after-hours/pre-market trade (the rest on the prior close — before v2 the quote feed silently echoed it).
- **Gate effect** (ungated − gated APE, paired over 23 days): -0.00% 90% CI [-0.02%, 0.01%], gates helped on 5/9 decisive days (sign test p=1.0). Gates fired on 9 of them.

## Metrics

| Metric | All-time (pre-open) — verdict basis | Last 20 (all rows) | All-time (all rows) |
|---|---|---|---|
| Scored days | 57 | 20 | 97 |
| PASS rate (±1%) | 43.86% | 45.00% | 45.36% |
| Directional accuracy | 56.14% | 70.00% | 55.67% |
| MAPE | 1.51% | 1.36% | 1.46% |
| Baseline MAPE (random walk) | 1.56% | 1.46% | 1.48% |
| Edge (baseline − model) | 0.04% | 0.10% | 0.02% |
| Brier (confidence calib.) | 0.25 | 0.29 | 0.26 |

## Per-strategy scorecards

| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |
|---|---|---|---|---|
| news | 97 | 32.99% | 1.30% | 0.23 |
| technical | 97 | 12.37% | 1.45% | 0.21 |
| contrarian | 97 | 25.77% | 1.52% | 0.20 |
| momentum | 97 | 17.53% | 1.59% | 0.19 |
| macro | 97 | 11.34% | 1.65% | 0.18 |

## Last 20 scored model predictions
_(backfill seed rows are excluded from metrics and this table; they appear only as price-history context on the dashboard chart. ⚠️ = written after the open, excluded from verdicts)_

| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest | Pre-open |
|---|---|---|---|---|---|---|---|---|
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
| 2026-10-01 | 250.26 | 248.23 | 0.82% | ✅ | ❌ | ❌ | contrarian | ✅ |

## 🅰️/🅱️ A/B test — ensemble+gates vs raven-style prior+pulse

_Both arms forecast the same sessions from the same pre-open snapshot (paired test). Only days on which both forecasts were written before the open count._

**Verdict:** Arm B leads on paired MAPE (mean daily delta 0.01% in B's favor over 49 pre-open days); B wins 25/49 decisive days (sign test p=1.0, descriptive); anytime-valid 95% CS [-0.179%, +0.209%] includes zero — not distinguishable from noise.

**B's next forecast (2026-10-29):** close ≈ **252.77** (down, adj -0.2σ, 8 evidence items).

| All-time, pre-open | A — ensemble+gates | B — raven prior+pulse |
|---|---|---|
| Scored days | 57 | 49 |
| PASS rate (±1%) | 43.86% | 53.06% |
| Directional accuracy | 56.14% | 63.27% |
| MAPE | 1.51% | 1.59% |
| Edge vs free baseline | 0.04% | 0.04% |

Clean paired days: 49 (of 72 paired); B wins 25/49 decisive; mean daily APE delta (A−B) 0.01%; CRPS delta (A−B) anytime-valid 95% CS [-0.106%, +0.133%] (undecided).
_This is a research experiment, not financial advice._
