# 📈 AMZN Daily Close Predictor — Results

_Auto-generated after every run. Verdicts use pre-open forecasts only and anytime-valid 95% confidence sequences, so they stay valid although this page is re-read daily. PASS = predicted close within ±1% of actual._

## Verdict
❌ **No edge yet** — all-time MAPE 1.53% is nominally ahead of the free baseline 1.56% by 0.03%, but the difference is not distinguishable from noise (paired over 52 pre-open days: mean 0.03%, anytime-valid 95% CS [-0.083%, +0.145%]; fixed-sample 90% CI [-0.07%, 0.10%], sign test p=0.6778).

**Next forecast (2026-10-22):** close ≈ **261.61** (down, P(up) 41%, 80% interval 257.04–266.26) vs prior close 262.37; anchored to 261.40 (yfinance_ext, live, refreshed 0×).

## Scoreboard — all-time, pre-open forecasts only

_Every forecaster is scored as a calibrated predictive distribution (CRPS: lower is better; coverage should match its nominal level). The **free baseline** is the latest pre-open trade the forecast was anchored to — the prior close when there was none._

| Metric | Model (arm A) | Random walk (prior close) | Free baseline (pre-open price) |
|---|---|---|---|
| Days | 52 | 52 | 52 |
| MAPE | 1.53% | 1.55% | 1.56% |
| CRPS | 1.26% | 1.28% | 1.27% |
| 80% interval coverage | 84.62% | 80.77% | 82.69% |
| 50% interval coverage | 53.85% | 53.85% | 53.85% |
| Brier of P(up) | 0.245 | 0.250 | 0.254 |

**Does the model add value?** Paired daily gains, anytime-valid 95% confidence sequences (valid although this page is re-read every day):

| Comparison | Mean daily gain | 95% CS | Verdict |
|---|---|---|---|
| APE vs free baseline | 0.03% | [-0.083%, +0.145%] | ≈ not distinguishable |
| APE vs random walk | 0.02% | [-0.140%, +0.179%] | ≈ not distinguishable |
| CRPS vs free baseline | 0.01% | [-0.048%, +0.075%] | ≈ not distinguishable |
| CRPS vs random walk | 0.02% | [-0.085%, +0.116%] | ≈ not distinguishable |
| Direction (Brier of P(up)) vs random walk | 0.005 | [-0.043, +0.052] | ≈ not distinguishable |

## Self-improvement lab — measured, not narrated

Aggregation rules are replayed over every pre-open day using only earlier data. A rule replaces the equal-weight mean only when its anytime-valid confidence sequence (Bonferroni over 5 challengers, α=0.05) shows it is better; the judge and the gates are switched off the same way once shown to hurt ([`evals/lab.py`](src/evals/lab.py)).

**Champion: `mean`** (reference — no challenger has proven better yet; 52 days replayed).

| Rule | MAPE | Gain vs mean | CS (α/K) | Status |
|---|---|---|---|---|
| mean (reference) | 1.41% | — | — | 🏆 champion |
| median | 1.45% | -0.04% | [-0.194%, +0.114%] | ≈ not distinguishable |
| trimmed_mean | 1.46% | -0.04% | [-0.160%, +0.071%] | ≈ not distinguishable |
| inverse_mse | 1.43% | -0.02% | [-0.076%, +0.030%] | ≈ not distinguishable |
| best_recent | 1.67% | -0.26% | [-0.541%, +0.022%] | ≈ not distinguishable |
| shrink_half | 1.48% | -0.07% | [-0.178%, +0.045%] | ≈ not distinguishable |

- **Judge adjustment** (APE of blend − APE after the judge's clamped σ-move; v2 rows): n=18, mean 0.00%, CS [-0.064%, +0.073%] → judge **on**.
- **Guardrail gates** (APE before − after the gates): n=18, mean -0.01%, CS [-0.054%, +0.039%] → gates **on**.
- **Legacy v1 pipeline** (judge wrote the number) vs the plain mean of its own analysts: mean -0.17% over 34 pre-open days, CS [-0.407%, +0.059%] — ≈ not distinguishable (negative = the judge cost accuracy).

### Integrity & mechanism

- **40 scored row(s) were created after their session opened** and are excluded from every verdict. They saw part of the tape they forecast. New post-open rows are refused (`--allow-late` to override), and the schedule now researches the evening before and only re-anchors in the morning.
- **Pre-open coverage:** 19 of the last 20 sessions (since 2026-09-25) got a forecast written before the open.
- **Live anchors:** 18 of 52 pre-open forecasts were anchored on a real after-hours/pre-market trade (the rest on the prior close — before v2 the quote feed silently echoed it).
- **Gate effect** (ungated − gated APE, paired over 18 days): -0.01% 90% CI [-0.03%, 0.01%], gates helped on 4/8 decisive days (sign test p=1.0). Gates fired on 8 of them.

## Metrics

| Metric | All-time (pre-open) — verdict basis | Last 20 (all rows) | All-time (all rows) |
|---|---|---|---|
| Scored days | 52 | 20 | 92 |
| PASS rate (±1%) | 42.31% | 50.00% | 44.57% |
| Directional accuracy | 51.92% | 55.00% | 53.26% |
| MAPE | 1.53% | 1.16% | 1.47% |
| Baseline MAPE (random walk) | 1.55% | 1.18% | 1.47% |
| Edge (baseline − model) | 0.02% | 0.02% | 0.00% |
| Brier (confidence calib.) | 0.24 | 0.30 | 0.26 |

## Per-strategy scorecards

| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |
|---|---|---|---|---|
| news | 92 | 31.52% | 1.31% | 0.23 |
| technical | 92 | 13.04% | 1.44% | 0.21 |
| contrarian | 92 | 25.00% | 1.51% | 0.20 |
| momentum | 92 | 18.48% | 1.59% | 0.19 |
| macro | 92 | 11.96% | 1.65% | 0.18 |

## Last 20 scored model predictions
_(backfill seed rows are excluded from metrics and this table; they appear only as price-history context on the dashboard chart. ⚠️ = written after the open, excluded from verdicts)_

| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest | Pre-open |
|---|---|---|---|---|---|---|---|---|
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
| 2026-09-25 | 249.05 | 249.67 | 0.25% | ✅ | ❌ | ❌ | contrarian | ⚠️ |
| 2026-09-24 | 249.36 | 249.38 | 0.01% | ✅ | ✅ | ✅ | contrarian | ⚠️ |

## 🅰️/🅱️ A/B test — ensemble+gates vs raven-style prior+pulse

_Both arms forecast the same sessions from the same pre-open snapshot (paired test). Only days on which both forecasts were written before the open count._

**Verdict:** Arm A leads on paired MAPE (mean daily delta 0.00% in A's favor over 44 pre-open days); B wins 21/44 decisive days (sign test p=0.8804, descriptive); anytime-valid 95% CS [-0.215%, +0.214%] includes zero — not distinguishable from noise.

**B's next forecast (2026-10-22):** close ≈ **260.95** (down, adj -0.1σ, 8 evidence items).

| All-time, pre-open | A — ensemble+gates | B — raven prior+pulse |
|---|---|---|
| Scored days | 52 | 44 |
| PASS rate (±1%) | 42.31% | 52.27% |
| Directional accuracy | 51.92% | 59.09% |
| MAPE | 1.53% | 1.63% |
| Edge vs free baseline | 0.03% | 0.02% |

Clean paired days: 44 (of 67 paired); B wins 21/44 decisive; mean daily APE delta (A−B) -0.00%; CRPS delta (A−B) anytime-valid 95% CS [-0.127%, +0.139%] (undecided).
_This is a research experiment, not financial advice._
