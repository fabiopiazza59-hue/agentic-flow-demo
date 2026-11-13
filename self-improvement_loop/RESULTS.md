# 📈 AMZN Daily Close Predictor — Results

_Auto-generated after every run. Verdicts use pre-open forecasts only and anytime-valid 95% confidence sequences, so they stay valid although this page is re-read daily. PASS = predicted close within ±1% of actual._

## Verdict
❌ **No edge yet** — all-time MAPE 1.65% is nominally ahead of the free baseline 1.73% by 0.08%, but the difference is not distinguishable from noise (paired over 68 pre-open days: mean 0.08%, anytime-valid 95% CS [-0.044%, +0.203%]; fixed-sample 90% CI [-0.11%, 0.15%], sign test p=0.7163).

**Next forecast (2026-11-13):** close ≈ **268.46** (up, P(up) 59%, 80% interval 262.14–274.93) vs prior close 267.44; anchored to 268.28 (yfinance_ext, live, refreshed 0×).

## Scoreboard — all-time, pre-open forecasts only

_Every forecaster is scored as a calibrated predictive distribution (CRPS: lower is better; coverage should match its nominal level). The **free baseline** is the latest pre-open trade the forecast was anchored to — the prior close when there was none._

| Metric | Model (arm A) | Random walk (prior close) | Free baseline (pre-open price) |
|---|---|---|---|
| Days | 68 | 68 | 68 |
| MAPE | 1.65% | 1.67% | 1.73% |
| CRPS | 1.35% | 1.35% | 1.40% |
| 80% interval coverage | 79.41% | 77.94% | 79.41% |
| 50% interval coverage | 52.94% | 50.00% | 50.00% |
| Brier of P(up) | 0.227 | 0.250 | 0.240 |

**Does the model add value?** Paired daily gains, anytime-valid 95% confidence sequences (valid although this page is re-read every day):

| Comparison | Mean daily gain | 95% CS | Verdict |
|---|---|---|---|
| APE vs free baseline | 0.08% | [-0.044%, +0.203%] | ≈ not distinguishable |
| APE vs random walk | 0.02% | [-0.222%, +0.268%] | ≈ not distinguishable |
| CRPS vs free baseline | 0.05% | [-0.029%, +0.134%] | ≈ not distinguishable |
| CRPS vs random walk | 0.00% | [-0.194%, +0.194%] | ≈ not distinguishable |
| Direction (Brier of P(up)) vs random walk | 0.023 | [-0.022, +0.067] | ≈ not distinguishable |

## Self-improvement lab — measured, not narrated

Aggregation rules are replayed over every pre-open day using only earlier data. A rule replaces the equal-weight mean only when its anytime-valid confidence sequence (Bonferroni over 5 challengers, α=0.05) shows it is better; the judge and the gates are switched off the same way once shown to hurt ([`evals/lab.py`](src/evals/lab.py)).

**Champion: `mean`** (reference — no challenger has proven better yet; 68 days replayed).

| Rule | MAPE | Gain vs mean | CS (α/K) | Status |
|---|---|---|---|---|
| mean (reference) | 1.57% | — | — | 🏆 champion |
| median | 1.60% | -0.03% | [-0.158%, +0.103%] | ≈ not distinguishable |
| trimmed_mean | 1.61% | -0.03% | [-0.125%, +0.063%] | ≈ not distinguishable |
| inverse_mse | 1.59% | -0.02% | [-0.058%, +0.024%] | ≈ not distinguishable |
| best_recent | 1.80% | -0.23% | [-0.465%, +0.014%] | ≈ not distinguishable |
| shrink_half | 1.65% | -0.07% | [-0.163%, +0.019%] | ≈ not distinguishable |

- **Judge adjustment** (APE of blend − APE after the judge's clamped σ-move; v2 rows): n=34, mean 0.03%, CS [-0.050%, +0.105%] → judge **on**.
- **Guardrail gates** (APE before − after the gates): n=34, mean -0.00%, CS [-0.025%, +0.021%] → gates **on**.
- **Legacy v1 pipeline** (judge wrote the number) vs the plain mean of its own analysts: mean -0.17% over 34 pre-open days, CS [-0.407%, +0.059%] — ≈ not distinguishable (negative = the judge cost accuracy).

### Integrity & mechanism

- **40 scored row(s) were created after their session opened** and are excluded from every verdict. They saw part of the tape they forecast. New post-open rows are refused (`--allow-late` to override), and the schedule now researches the evening before and only re-anchors in the morning.
- **Pre-open coverage:** 20 of the last 20 sessions (since 2026-10-19) got a forecast written before the open.
- **Live anchors:** 34 of 68 pre-open forecasts were anchored on a real after-hours/pre-market trade (the rest on the prior close — before v2 the quote feed silently echoed it).
- **Gate effect** (ungated − gated APE, paired over 34 days): -0.00% 90% CI [-0.01%, 0.01%], gates helped on 5/9 decisive days (sign test p=1.0). Gates fired on 9 of them.

## Metrics

| Metric | All-time (pre-open) — verdict basis | Last 20 (all rows) | All-time (all rows) |
|---|---|---|---|
| Scored days | 68 | 20 | 108 |
| PASS rate (±1%) | 41.18% | 40.00% | 43.52% |
| Directional accuracy | 60.29% | 80.00% | 58.33% |
| MAPE | 1.65% | 1.98% | 1.55% |
| Baseline MAPE (random walk) | 1.67% | 2.01% | 1.56% |
| Edge (baseline − model) | 0.02% | 0.03% | 0.01% |
| Brier (confidence calib.) | 0.24 | 0.25 | 0.26 |

## Per-strategy scorecards

| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |
|---|---|---|---|---|
| news | 108 | 31.48% | 1.41% | 0.22 |
| technical | 108 | 11.11% | 1.55% | 0.20 |
| contrarian | 108 | 26.85% | 1.61% | 0.20 |
| momentum | 108 | 17.59% | 1.68% | 0.19 |
| macro | 108 | 12.96% | 1.71% | 0.19 |

## Last 20 scored model predictions
_(backfill seed rows are excluded from metrics and this table; they appear only as price-history context on the dashboard chart. ⚠️ = written after the open, excluded from verdicts)_

| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest | Pre-open |
|---|---|---|---|---|---|---|---|---|
| 2026-11-12 | 267.03 | 267.44 | 0.15% | ✅ | ✅ | ✅ | macro | ✅ |
| 2026-11-11 | 273.14 | 266.17 | 2.62% | ❌ | ❌ | ❌ | contrarian | ✅ |
| 2026-11-10 | 267.13 | 270.87 | 1.38% | ❌ | ✅ | ✅ | momentum | ✅ |
| 2026-11-09 | 260.31 | 266.93 | 2.48% | ❌ | ✅ | ✅ | momentum | ✅ |
| 2026-11-06 | 257.15 | 259.30 | 0.83% | ✅ | ✅ | ❌ | macro | ✅ |
| 2026-11-05 | 262.97 | 260.21 | 1.06% | ❌ | ✅ | ✅ | news | ✅ |
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

## 🅰️/🅱️ A/B test — ensemble+gates vs raven-style prior+pulse

_Both arms forecast the same sessions from the same pre-open snapshot (paired test). Only days on which both forecasts were written before the open count._

**Verdict:** Arm A leads on paired MAPE (mean daily delta 0.07% in A's favor over 60 pre-open days); B wins 29/60 decisive days (sign test p=0.8974, descriptive); anytime-valid 95% CS [-0.269%, +0.133%] includes zero — not distinguishable from noise.

**B's next forecast (2026-11-13):** close ≈ **269.14** (up, adj 0.12σ, 8 evidence items).

| All-time, pre-open | A — ensemble+gates | B — raven prior+pulse |
|---|---|---|
| Scored days | 68 | 60 |
| PASS rate (±1%) | 41.18% | 43.33% |
| Directional accuracy | 60.29% | 61.67% |
| MAPE | 1.65% | 1.81% |
| Edge vs free baseline | 0.08% | 0.01% |

Clean paired days: 60 (of 83 paired); B wins 29/60 decisive; mean daily APE delta (A−B) -0.07%; CRPS delta (A−B) anytime-valid 95% CS [-0.180%, +0.091%] (undecided).
_This is a research experiment, not financial advice._
