# 📈 AMZN Daily Close Predictor — Results

_Auto-generated after every run. Verdicts use pre-open forecasts only and anytime-valid 95% confidence sequences, so they stay valid although this page is re-read daily. PASS = predicted close within ±1% of actual._

## Verdict
❌ **No edge yet** — all-time MAPE 1.65% is nominally ahead of the free baseline 1.74% by 0.09%, but the difference is not distinguishable from noise (paired over 64 pre-open days: mean 0.09%, anytime-valid 95% CS [-0.039%, +0.218%]; fixed-sample 90% CI [-0.12%, 0.15%], sign test p=0.9007).

**Awaiting score (2026-11-09):** close ≈ **260.31** (up, P(up) 59%, 80% interval 253.96–266.82) vs prior close 259.30; anchored to 259.84 (yfinance_ext, live, refreshed 0×).

## Scoreboard — all-time, pre-open forecasts only

_Every forecaster is scored as a calibrated predictive distribution (CRPS: lower is better; coverage should match its nominal level). The **free baseline** is the latest pre-open trade the forecast was anchored to — the prior close when there was none._

| Metric | Model (arm A) | Random walk (prior close) | Free baseline (pre-open price) |
|---|---|---|---|
| Days | 64 | 64 | 64 |
| MAPE | 1.65% | 1.67% | 1.74% |
| CRPS | 1.36% | 1.36% | 1.42% |
| 80% interval coverage | 81.25% | 78.12% | 79.69% |
| 50% interval coverage | 54.69% | 50.00% | 50.00% |
| Brier of P(up) | 0.226 | 0.250 | 0.241 |

**Does the model add value?** Paired daily gains, anytime-valid 95% confidence sequences (valid although this page is re-read every day):

| Comparison | Mean daily gain | 95% CS | Verdict |
|---|---|---|---|
| APE vs free baseline | 0.09% | [-0.039%, +0.218%] | ≈ not distinguishable |
| APE vs random walk | 0.03% | [-0.231%, +0.282%] | ≈ not distinguishable |
| CRPS vs free baseline | 0.06% | [-0.023%, +0.144%] | ≈ not distinguishable |
| CRPS vs random walk | 0.01% | [-0.198%, +0.209%] | ≈ not distinguishable |
| Direction (Brier of P(up)) vs random walk | 0.024 | [-0.022, +0.071] | ≈ not distinguishable |

## Self-improvement lab — measured, not narrated

Aggregation rules are replayed over every pre-open day using only earlier data. A rule replaces the equal-weight mean only when its anytime-valid confidence sequence (Bonferroni over 5 challengers, α=0.05) shows it is better; the judge and the gates are switched off the same way once shown to hurt ([`evals/lab.py`](src/evals/lab.py)).

**Champion: `mean`** (reference — no challenger has proven better yet; 64 days replayed).

| Rule | MAPE | Gain vs mean | CS (α/K) | Status |
|---|---|---|---|---|
| mean (reference) | 1.57% | — | — | 🏆 champion |
| median | 1.62% | -0.04% | [-0.176%, +0.090%] | ≈ not distinguishable |
| trimmed_mean | 1.61% | -0.04% | [-0.136%, +0.055%] | ≈ not distinguishable |
| inverse_mse | 1.59% | -0.02% | [-0.059%, +0.028%] | ≈ not distinguishable |
| best_recent | 1.79% | -0.22% | [-0.461%, +0.023%] | ≈ not distinguishable |
| shrink_half | 1.65% | -0.08% | [-0.173%, +0.019%] | ≈ not distinguishable |

- **Judge adjustment** (APE of blend − APE after the judge's clamped σ-move; v2 rows): n=30, mean 0.04%, CS [-0.040%, +0.122%] → judge **on**.
- **Guardrail gates** (APE before − after the gates): n=30, mean -0.00%, CS [-0.029%, +0.024%] → gates **on**.
- **Legacy v1 pipeline** (judge wrote the number) vs the plain mean of its own analysts: mean -0.17% over 34 pre-open days, CS [-0.407%, +0.059%] — ≈ not distinguishable (negative = the judge cost accuracy).

### Integrity & mechanism

- **40 scored row(s) were created after their session opened** and are excluded from every verdict. They saw part of the tape they forecast. New post-open rows are refused (`--allow-late` to override), and the schedule now researches the evening before and only re-anchors in the morning.
- **Pre-open coverage:** 20 of the last 20 sessions (since 2026-10-13) got a forecast written before the open.
- **Live anchors:** 30 of 64 pre-open forecasts were anchored on a real after-hours/pre-market trade (the rest on the prior close — before v2 the quote feed silently echoed it).
- **Gate effect** (ungated − gated APE, paired over 30 days): -0.00% 90% CI [-0.02%, 0.01%], gates helped on 5/9 decisive days (sign test p=1.0). Gates fired on 9 of them.

## Metrics

| Metric | All-time (pre-open) — verdict basis | Last 20 (all rows) | All-time (all rows) |
|---|---|---|---|
| Scored days | 64 | 20 | 104 |
| PASS rate (±1%) | 42.19% | 40.00% | 44.23% |
| Directional accuracy | 59.38% | 80.00% | 57.69% |
| MAPE | 1.65% | 1.84% | 1.55% |
| Baseline MAPE (random walk) | 1.67% | 1.85% | 1.56% |
| Edge (baseline − model) | 0.03% | 0.01% | 0.01% |
| Brier (confidence calib.) | 0.24 | 0.27 | 0.26 |

## Per-strategy scorecards

| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |
|---|---|---|---|---|
| news | 104 | 32.69% | 1.41% | 0.23 |
| technical | 104 | 11.54% | 1.55% | 0.20 |
| contrarian | 104 | 26.92% | 1.59% | 0.20 |
| momentum | 104 | 16.35% | 1.69% | 0.19 |
| macro | 104 | 12.50% | 1.73% | 0.18 |

## Last 20 scored model predictions
_(backfill seed rows are excluded from metrics and this table; they appear only as price-history context on the dashboard chart. ⚠️ = written after the open, excluded from verdicts)_

| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest | Pre-open |
|---|---|---|---|---|---|---|---|---|
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
| 2026-10-15 | 250.82 | 254.76 | 1.55% | ❌ | ❌ | ❌ | contrarian | ✅ |
| 2026-10-14 | 257.27 | 254.08 | 1.26% | ❌ | ✅ | ✅ | contrarian | ✅ |
| 2026-10-13 | 258.70 | 258.58 | 0.05% | ✅ | ✅ | ✅ | macro | ✅ |
| 2026-10-12 | 262.39 | 259.53 | 1.10% | ❌ | ✅ | ✅ | contrarian | ✅ |

## 🅰️/🅱️ A/B test — ensemble+gates vs raven-style prior+pulse

_Both arms forecast the same sessions from the same pre-open snapshot (paired test). Only days on which both forecasts were written before the open count._

**Verdict:** Arm A leads on paired MAPE (mean daily delta 0.06% in A's favor over 56 pre-open days); B wins 27/56 decisive days (sign test p=0.8939, descriptive); anytime-valid 95% CS [-0.264%, +0.153%] includes zero — not distinguishable from noise.

**B's awaiting score (2026-11-09):** close ≈ **259.17** (down, adj -0.1σ, 8 evidence items).

| All-time, pre-open | A — ensemble+gates | B — raven prior+pulse |
|---|---|---|
| Scored days | 64 | 56 |
| PASS rate (±1%) | 42.19% | 46.43% |
| Directional accuracy | 59.38% | 64.29% |
| MAPE | 1.65% | 1.80% |
| Edge vs free baseline | 0.09% | 0.03% |

Clean paired days: 56 (of 79 paired); B wins 27/56 decisive; mean daily APE delta (A−B) -0.06%; CRPS delta (A−B) anytime-valid 95% CS [-0.183%, +0.104%] (undecided).
_This is a research experiment, not financial advice._
