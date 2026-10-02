# 📈 AMZN Daily Close Predictor — Results

_Auto-generated after every run. Verdicts use pre-open forecasts only and anytime-valid 95% confidence sequences, so they stay valid although this page is re-read daily. PASS = predicted close within ±1% of actual._

## Verdict
❌ **No edge yet** — all-time MAPE 1.57% is nominally ahead of the free baseline 1.58% by 0.01%, but the difference is not distinguishable from noise (paired over 38 pre-open days: mean 0.01%, anytime-valid 95% CS [-0.141%, +0.158%]; fixed-sample 90% CI [-0.08%, 0.08%], sign test p=0.2559).

**Next forecast (2026-10-02):** close ≈ **249.13** (up, P(up) 63%, 80% interval 245.45–252.87) vs prior close 248.23; anchored to 248.90 (yfinance_ext, live, refreshed 0×).

## Scoreboard — all-time, pre-open forecasts only

_Every forecaster is scored as a calibrated predictive distribution (CRPS: lower is better; coverage should match its nominal level). The **free baseline** is the latest pre-open trade the forecast was anchored to — the prior close when there was none._

| Metric | Model (arm A) | Random walk (prior close) | Free baseline (pre-open price) |
|---|---|---|---|
| Days | 38 | 38 | 38 |
| MAPE | 1.57% | 1.58% | 1.58% |
| CRPS | 1.31% | 1.31% | 1.31% |
| 80% interval coverage | 89.47% | 89.47% | 89.47% |
| 50% interval coverage | 63.16% | 60.53% | 63.16% |
| Brier of P(up) | 0.244 | 0.250 | 0.253 |

**Does the model add value?** Paired daily gains, anytime-valid 95% confidence sequences (valid although this page is re-read every day):

| Comparison | Mean daily gain | 95% CS | Verdict |
|---|---|---|---|
| APE vs free baseline | 0.01% | [-0.141%, +0.158%] | ≈ not distinguishable |
| APE vs random walk | 0.00% | [-0.153%, +0.158%] | ≈ not distinguishable |
| CRPS vs free baseline | 0.01% | [-0.074%, +0.087%] | ≈ not distinguishable |
| CRPS vs random walk | 0.00% | [-0.080%, +0.086%] | ≈ not distinguishable |
| Direction (Brier of P(up)) vs random walk | 0.006 | [-0.033, +0.044] | ≈ not distinguishable |

## Self-improvement lab — measured, not narrated

Aggregation rules are replayed over every pre-open day using only earlier data. A rule replaces the equal-weight mean only when its anytime-valid confidence sequence (Bonferroni over 5 challengers, α=0.05) shows it is better; the judge and the gates are switched off the same way once shown to hurt ([`evals/lab.py`](src/evals/lab.py)).

**Champion: `mean`** (reference — no challenger has proven better yet; 38 days replayed).

| Rule | MAPE | Gain vs mean | CS (α/K) | Status |
|---|---|---|---|---|
| mean (reference) | 1.42% | — | — | 🏆 champion |
| median | 1.45% | -0.03% | [-0.233%, +0.172%] | ≈ not distinguishable |
| trimmed_mean | 1.47% | -0.05% | [-0.202%, +0.106%] | ≈ not distinguishable |
| inverse_mse | 1.45% | -0.03% | [-0.105%, +0.039%] | ≈ not distinguishable |
| best_recent | 1.71% | -0.29% | [-0.663%, +0.088%] | ≈ not distinguishable |
| shrink_half | 1.49% | -0.07% | [-0.226%, +0.079%] | ≈ not distinguishable |

- **Judge adjustment** (APE of blend − APE after the judge's clamped σ-move; v2 rows): n=4, mean 0.00%, CS n=4 (too few) → judge **on**.
- **Guardrail gates** (APE before − after the gates): n=4, mean 0.00%, CS n=4 (too few) → gates **on**.
- **Legacy v1 pipeline** (judge wrote the number) vs the plain mean of its own analysts: mean -0.17% over 34 pre-open days, CS [-0.407%, +0.059%] — ≈ not distinguishable (negative = the judge cost accuracy).

### Integrity & mechanism

- **40 scored row(s) were created after their session opened** and are excluded from every verdict. They saw part of the tape they forecast. New post-open rows are refused (`--allow-late` to override), and the schedule now researches the evening before and only re-anchors in the morning.
- **Pre-open coverage:** 5 of the last 20 sessions (since 2026-09-04) got a forecast written before the open.
- **Live anchors:** 4 of 38 pre-open forecasts were anchored on a real after-hours/pre-market trade (the rest on the prior close — before v2 the quote feed silently echoed it).
- **Gate effect** (ungated − gated APE, paired over 4 days): 0.00% 90% CI [-0.02%, 0.02%], gates helped on 2/3 decisive days (sign test p=1.0). Gates fired on 3 of them.

## Metrics

| Metric | All-time (pre-open) — verdict basis | Last 20 (all rows) | All-time (all rows) |
|---|---|---|---|
| Scored days | 38 | 20 | 78 |
| PASS rate (±1%) | 44.74% | 55.00% | 46.15% |
| Directional accuracy | 47.37% | 60.00% | 51.28% |
| MAPE | 1.57% | 1.02% | 1.48% |
| Baseline MAPE (random walk) | 1.58% | 1.11% | 1.47% |
| Edge (baseline − model) | 0.00% | 0.09% | -0.01% |
| Brier (confidence calib.) | 0.22 | 0.27 | 0.26 |

## Per-strategy scorecards

| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |
|---|---|---|---|---|
| news | 78 | 32.05% | 1.31% | 0.23 |
| technical | 78 | 14.10% | 1.42% | 0.21 |
| contrarian | 78 | 21.79% | 1.53% | 0.19 |
| momentum | 78 | 20.51% | 1.57% | 0.19 |
| macro | 78 | 11.54% | 1.65% | 0.18 |

## Last 20 scored model predictions
_(backfill seed rows are excluded from metrics and this table; they appear only as price-history context on the dashboard chart. ⚠️ = written after the open, excluded from verdicts)_

| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest | Pre-open |
|---|---|---|---|---|---|---|---|---|
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
| 2026-09-11 | 251.72 | 256.78 | 1.97% | ❌ | ❌ | ❌ | news | ⚠️ |
| 2026-09-10 | 251.35 | 251.89 | 0.21% | ✅ | ✅ | ❌ | technical | ⚠️ |
| 2026-09-09 | 256.44 | 252.40 | 1.60% | ❌ | ✅ | ✅ | news | ⚠️ |
| 2026-09-08 | 258.28 | 256.97 | 0.51% | ✅ | ✅ | ✅ | macro | ⚠️ |
| 2026-09-04 | 258.82 | 258.51 | 0.12% | ✅ | ✅ | ✅ | technical | ⚠️ |
| 2026-09-03 | 254.70 | 258.90 | 1.62% | ❌ | ❌ | ❌ | contrarian | ⚠️ |

## 🅰️/🅱️ A/B test — ensemble+gates vs raven-style prior+pulse

_Both arms forecast the same sessions from the same pre-open snapshot (paired test). Only days on which both forecasts were written before the open count._

**Verdict:** Arm B leads on paired MAPE (mean daily delta 0.08% in B's favor over 30 pre-open days); B wins 16/30 decisive days (sign test p=0.8555, descriptive); anytime-valid 95% CS [-0.209%, +0.359%] includes zero — not distinguishable from noise.

**B's next forecast (2026-10-02):** close ≈ **249.24** (up, adj 0.1σ, 7 evidence items).

| All-time, pre-open | A — ensemble+gates | B — raven prior+pulse |
|---|---|---|
| Scored days | 38 | 30 |
| PASS rate (±1%) | 44.74% | 50.00% |
| Directional accuracy | 47.37% | 56.67% |
| MAPE | 1.57% | 1.66% |
| Edge vs free baseline | 0.01% | 0.06% |

Clean paired days: 30 (of 53 paired); B wins 16/30 decisive; mean daily APE delta (A−B) 0.08%; CRPS delta (A−B) anytime-valid 95% CS [-0.122%, +0.222%] (undecided).
_This is a research experiment, not financial advice._
