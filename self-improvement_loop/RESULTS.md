# 📈 AMZN Daily Close Predictor — Results

_Auto-generated after every run. Verdicts use pre-open forecasts only and anytime-valid 95% confidence sequences, so they stay valid although this page is re-read daily. PASS = predicted close within ±1% of actual._

## Verdict
❌ **No edge yet** — all-time MAPE 1.66% is nominally ahead of the free baseline 1.67% by 0.01%, but the difference is not distinguishable from noise (paired over 35 pre-open days: mean 0.01%, anytime-valid 95% CS [-0.149%, +0.177%]; fixed-sample 90% CI [-0.07%, 0.09%], sign test p=0.3105).

**Next forecast (2026-09-29):** close ≈ **246.02** (down, P(up) 47%, 80% interval 242.08–250.03) vs prior close 246.15; anchored to 246.32 (yfinance_ext, live, refreshed 0×).

## Scoreboard — all-time, pre-open forecasts only

_Every forecaster is scored as a calibrated predictive distribution (CRPS: lower is better; coverage should match its nominal level). The **free baseline** is the latest pre-open trade the forecast was anchored to — the prior close when there was none._

| Metric | Model (arm A) | Random walk (prior close) | Free baseline (pre-open price) |
|---|---|---|---|
| Days | 35 | 35 | 35 |
| MAPE | 1.66% | 1.67% | 1.67% |
| CRPS | 1.38% | 1.38% | 1.39% |
| 80% interval coverage | 88.57% | 88.57% | 88.57% |
| 50% interval coverage | 60.00% | 60.00% | 60.00% |
| Brier of P(up) | 0.240 | 0.250 | 0.252 |

**Does the model add value?** Paired daily gains, anytime-valid 95% confidence sequences (valid although this page is re-read every day):

| Comparison | Mean daily gain | 95% CS | Verdict |
|---|---|---|---|
| APE vs free baseline | 0.01% | [-0.149%, +0.177%] | ≈ not distinguishable |
| APE vs random walk | 0.01% | [-0.153%, +0.173%] | ≈ not distinguishable |
| CRPS vs free baseline | 0.01% | [-0.079%, +0.097%] | ≈ not distinguishable |
| CRPS vs random walk | 0.01% | [-0.082%, +0.095%] | ≈ not distinguishable |
| Direction (Brier of P(up)) vs random walk | 0.010 | [-0.028, +0.047] | ≈ not distinguishable |

## Self-improvement lab — measured, not narrated

Aggregation rules are replayed over every pre-open day using only earlier data. A rule replaces the equal-weight mean only when its anytime-valid confidence sequence (Bonferroni over 5 challengers, α=0.05) shows it is better; the judge and the gates are switched off the same way once shown to hurt ([`evals/lab.py`](src/evals/lab.py)).

**Champion: `mean`** (reference — no challenger has proven better yet; 35 days replayed).

| Rule | MAPE | Gain vs mean | CS (α/K) | Status |
|---|---|---|---|---|
| mean (reference) | 1.49% | — | — | 🏆 champion |
| median | 1.50% | -0.01% | [-0.227%, +0.203%] | ≈ not distinguishable |
| trimmed_mean | 1.53% | -0.04% | [-0.209%, +0.124%] | ≈ not distinguishable |
| inverse_mse | 1.52% | -0.04% | [-0.112%, +0.036%] | ≈ not distinguishable |
| best_recent | 1.78% | -0.29% | [-0.701%, +0.118%] | ≈ not distinguishable |
| shrink_half | 1.57% | -0.08% | [-0.248%, +0.083%] | ≈ not distinguishable |

- **Judge adjustment** (APE of blend − APE after the judge's clamped σ-move; v2 rows): n=1, mean 0.00%, CS n=1 (too few) → judge **on**.
- **Guardrail gates** (APE before − after the gates): n=1, mean -0.03%, CS n=1 (too few) → gates **on**.
- **Legacy v1 pipeline** (judge wrote the number) vs the plain mean of its own analysts: mean -0.17% over 34 pre-open days, CS [-0.407%, +0.059%] — ≈ not distinguishable (negative = the judge cost accuracy).

### Integrity & mechanism

- **40 scored row(s) were created after their session opened** and are excluded from every verdict. They saw part of the tape they forecast. New post-open rows are refused (`--allow-late` to override), and the schedule now researches the evening before and only re-anchors in the morning.
- **Pre-open coverage:** 2 of the last 20 sessions (since 2026-09-01) got a forecast written before the open.
- **Live anchors:** 1 of 35 pre-open forecasts were anchored on a real after-hours/pre-market trade (the rest on the prior close — before v2 the quote feed silently echoed it).
- **Gate effect** (ungated − gated APE, paired over 1 days): -0.03% 90% CI [-0.03%, -0.03%], gates helped on 0/1 decisive days (sign test p=1.0). Gates fired on 1 of them.

## Metrics

| Metric | All-time (pre-open) — verdict basis | Last 20 (all rows) | All-time (all rows) |
|---|---|---|---|
| Scored days | 35 | 20 | 75 |
| PASS rate (±1%) | 40.00% | 45.00% | 44.00% |
| Directional accuracy | 48.57% | 65.00% | 52.00% |
| MAPE | 1.66% | 1.15% | 1.51% |
| Baseline MAPE (random walk) | 1.67% | 1.26% | 1.51% |
| Edge (baseline − model) | 0.01% | 0.11% | -0.01% |
| Brier (confidence calib.) | 0.22 | 0.30 | 0.26 |

## Per-strategy scorecards

| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |
|---|---|---|---|---|
| news | 75 | 32.00% | 1.33% | 0.23 |
| technical | 75 | 14.67% | 1.45% | 0.21 |
| contrarian | 75 | 20.00% | 1.57% | 0.19 |
| momentum | 75 | 21.33% | 1.60% | 0.19 |
| macro | 75 | 12.00% | 1.68% | 0.18 |

## Last 20 scored model predictions
_(backfill seed rows are excluded from metrics and this table; they appear only as price-history context on the dashboard chart. ⚠️ = written after the open, excluded from verdicts)_

| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest | Pre-open |
|---|---|---|---|---|---|---|---|---|
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
| 2026-08-31 | 265.87 | 259.77 | 2.35% | ❌ | ✅ | ✅ | news | ⚠️ |

## 🅰️/🅱️ A/B test — ensemble+gates vs raven-style prior+pulse

_Both arms forecast the same sessions from the same pre-open snapshot (paired test). Only days on which both forecasts were written before the open count._

**Verdict:** Arm B leads on paired MAPE (mean daily delta 0.07% in B's favor over 27 pre-open days); B wins 14/27 decisive days (sign test p=1.0, descriptive); anytime-valid 95% CS [-0.240%, +0.387%] includes zero — not distinguishable from noise.

**B's next forecast (2026-09-29):** close ≈ **245.23** (down, adj -0.3σ, 8 evidence items).

| All-time, pre-open | A — ensemble+gates | B — raven prior+pulse |
|---|---|---|
| Scored days | 35 | 27 |
| PASS rate (±1%) | 40.00% | 44.44% |
| Directional accuracy | 48.57% | 59.26% |
| MAPE | 1.66% | 1.79% |
| Edge vs free baseline | 0.01% | 0.06% |

Clean paired days: 27 (of 50 paired); B wins 14/27 decisive; mean daily APE delta (A−B) 0.07%; CRPS delta (A−B) anytime-valid 95% CS [-0.147%, +0.238%] (undecided).
_This is a research experiment, not financial advice._
