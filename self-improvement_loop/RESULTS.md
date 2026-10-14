# 📈 AMZN Daily Close Predictor — Results

_Auto-generated after every run. Verdicts use pre-open forecasts only and anytime-valid 95% confidence sequences, so they stay valid although this page is re-read daily. PASS = predicted close within ±1% of actual._

## Verdict
❌ **No edge yet** — all-time MAPE 1.52% is nominally ahead of the free baseline 1.54% by 0.02%, but the difference is not distinguishable from noise (paired over 46 pre-open days: mean 0.02%, anytime-valid 95% CS [-0.104%, +0.146%]; fixed-sample 90% CI [-0.03%, 0.11%], sign test p=0.6587).

**Next forecast (2026-10-14):** close ≈ **257.27** (down, P(up) 34%, 80% interval 253.17–261.44) vs prior close 258.58; anchored to 257.77 (yfinance_ext, live, refreshed 0×).

## Scoreboard — all-time, pre-open forecasts only

_Every forecaster is scored as a calibrated predictive distribution (CRPS: lower is better; coverage should match its nominal level). The **free baseline** is the latest pre-open trade the forecast was anchored to — the prior close when there was none._

| Metric | Model (arm A) | Random walk (prior close) | Free baseline (pre-open price) |
|---|---|---|---|
| Days | 46 | 46 | 46 |
| MAPE | 1.52% | 1.56% | 1.54% |
| CRPS | 1.25% | 1.27% | 1.26% |
| 80% interval coverage | 84.78% | 84.78% | 84.78% |
| 50% interval coverage | 56.52% | 54.35% | 56.52% |
| Brier of P(up) | 0.235 | 0.250 | 0.245 |

**Does the model add value?** Paired daily gains, anytime-valid 95% confidence sequences (valid although this page is re-read every day):

| Comparison | Mean daily gain | 95% CS | Verdict |
|---|---|---|---|
| APE vs free baseline | 0.02% | [-0.104%, +0.146%] | ≈ not distinguishable |
| APE vs random walk | 0.04% | [-0.100%, +0.174%] | ≈ not distinguishable |
| CRPS vs free baseline | 0.01% | [-0.054%, +0.078%] | ≈ not distinguishable |
| CRPS vs random walk | 0.03% | [-0.052%, +0.106%] | ≈ not distinguishable |
| Direction (Brier of P(up)) vs random walk | 0.015 | [-0.020, +0.050] | ≈ not distinguishable |

## Self-improvement lab — measured, not narrated

Aggregation rules are replayed over every pre-open day using only earlier data. A rule replaces the equal-weight mean only when its anytime-valid confidence sequence (Bonferroni over 5 challengers, α=0.05) shows it is better; the judge and the gates are switched off the same way once shown to hurt ([`evals/lab.py`](src/evals/lab.py)).

**Champion: `mean`** (reference — no challenger has proven better yet; 46 days replayed).

| Rule | MAPE | Gain vs mean | CS (α/K) | Status |
|---|---|---|---|---|
| mean (reference) | 1.39% | — | — | 🏆 champion |
| median | 1.41% | -0.03% | [-0.196%, +0.144%] | ≈ not distinguishable |
| trimmed_mean | 1.42% | -0.04% | [-0.165%, +0.090%] | ≈ not distinguishable |
| inverse_mse | 1.42% | -0.03% | [-0.088%, +0.029%] | ≈ not distinguishable |
| best_recent | 1.64% | -0.25% | [-0.563%, +0.059%] | ≈ not distinguishable |
| shrink_half | 1.46% | -0.07% | [-0.196%, +0.058%] | ≈ not distinguishable |

- **Judge adjustment** (APE of blend − APE after the judge's clamped σ-move; v2 rows): n=12, mean 0.00%, CS [-0.000%, +0.000%] → judge **on**.
- **Guardrail gates** (APE before − after the gates): n=12, mean -0.01%, CS [-0.089%, +0.065%] → gates **on**.
- **Legacy v1 pipeline** (judge wrote the number) vs the plain mean of its own analysts: mean -0.17% over 34 pre-open days, CS [-0.407%, +0.059%] — ≈ not distinguishable (negative = the judge cost accuracy).

### Integrity & mechanism

- **40 scored row(s) were created after their session opened** and are excluded from every verdict. They saw part of the tape they forecast. New post-open rows are refused (`--allow-late` to override), and the schedule now researches the evening before and only re-anchors in the morning.
- **Pre-open coverage:** 13 of the last 20 sessions (since 2026-09-17) got a forecast written before the open.
- **Live anchors:** 12 of 46 pre-open forecasts were anchored on a real after-hours/pre-market trade (the rest on the prior close — before v2 the quote feed silently echoed it).
- **Gate effect** (ungated − gated APE, paired over 12 days): -0.01% 90% CI [-0.04%, 0.02%], gates helped on 4/8 decisive days (sign test p=1.0). Gates fired on 8 of them.

## Metrics

| Metric | All-time (pre-open) — verdict basis | Last 20 (all rows) | All-time (all rows) |
|---|---|---|---|
| Scored days | 46 | 20 | 86 |
| PASS rate (±1%) | 43.48% | 50.00% | 45.35% |
| Directional accuracy | 52.17% | 60.00% | 53.49% |
| MAPE | 1.52% | 1.10% | 1.46% |
| Baseline MAPE (random walk) | 1.56% | 1.22% | 1.47% |
| Edge (baseline − model) | 0.04% | 0.12% | 0.01% |
| Brier (confidence calib.) | 0.23 | 0.29 | 0.26 |

## Per-strategy scorecards

| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |
|---|---|---|---|---|
| news | 86 | 31.40% | 1.28% | 0.23 |
| technical | 86 | 12.79% | 1.42% | 0.21 |
| contrarian | 86 | 23.26% | 1.50% | 0.20 |
| momentum | 86 | 19.77% | 1.57% | 0.19 |
| macro | 86 | 12.79% | 1.63% | 0.18 |

## Last 20 scored model predictions
_(backfill seed rows are excluded from metrics and this table; they appear only as price-history context on the dashboard chart. ⚠️ = written after the open, excluded from verdicts)_

| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest | Pre-open |
|---|---|---|---|---|---|---|---|---|
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
| 2026-09-22 | 259.42 | 254.98 | 1.74% | ❌ | ❌ | ❌ | contrarian | ⚠️ |
| 2026-09-21 | 255.15 | 258.45 | 1.28% | ❌ | ✅ | ✅ | news | ⚠️ |
| 2026-09-18 | 251.27 | 253.71 | 0.96% | ✅ | ✅ | ✅ | news | ⚠️ |
| 2026-09-17 | 245.10 | 251.19 | 2.42% | ❌ | ❌ | ❌ | news | ⚠️ |
| 2026-09-16 | 247.05 | 245.96 | 0.44% | ✅ | ✅ | ✅ | macro | ⚠️ |

## 🅰️/🅱️ A/B test — ensemble+gates vs raven-style prior+pulse

_Both arms forecast the same sessions from the same pre-open snapshot (paired test). Only days on which both forecasts were written before the open count._

**Verdict:** Arm B leads on paired MAPE (mean daily delta 0.05% in B's favor over 38 pre-open days); B wins 20/38 decisive days (sign test p=0.8714, descriptive); anytime-valid 95% CS [-0.187%, +0.281%] includes zero — not distinguishable from noise.

**B's next forecast (2026-10-14):** close ≈ **256.6** (down, adj -0.3σ, 8 evidence items).

| All-time, pre-open | A — ensemble+gates | B — raven prior+pulse |
|---|---|---|
| Scored days | 46 | 38 |
| PASS rate (±1%) | 43.48% | 52.63% |
| Directional accuracy | 52.17% | 60.53% |
| MAPE | 1.52% | 1.59% |
| Edge vs free baseline | 0.02% | 0.05% |

Clean paired days: 38 (of 61 paired); B wins 20/38 decisive; mean daily APE delta (A−B) 0.05%; CRPS delta (A−B) anytime-valid 95% CS [-0.107%, +0.177%] (undecided).
_This is a research experiment, not financial advice._
