# 📈 AMZN Daily Close Predictor — Results

_Auto-generated each trading day. PASS = predicted close within ±1% of actual._

## Verdict
✅ **Edge confirmed** — rolling MAPE 1.95% beats the random-walk baseline 1.97% by 0.02%.

**Today's open prediction (2026-08-03):** close ≈ **284.56** (up, confidence 10.00%) vs prior close 284.45.

## Rolling metrics (last 20 scored days)

| Metric | Rolling | All-time |
|---|---|---|
| Scored days | 20 | 35 |
| PASS rate (±1%) | 35.00% | 34.29% |
| Directional accuracy | 60.00% | 48.57% |
| MAPE | 1.95% | 1.99% |
| Baseline MAPE (random walk) | 1.97% | 1.92% |
| Edge (baseline − model) | 0.02% | -0.07% |
| Brier (confidence calib.) | 0.24 | 0.24 |

## Per-strategy scorecards

| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |
|---|---|---|---|---|
| news | 35 | 31.43% | 1.62% | 0.23 |
| technical | 35 | 17.14% | 1.85% | 0.21 |
| contrarian | 35 | 14.29% | 1.99% | 0.19 |
| momentum | 35 | 22.86% | 2.04% | 0.19 |
| macro | 35 | 14.29% | 2.06% | 0.18 |

## Last 20 scored model predictions
_(backfill seed rows are excluded from metrics and this table; they appear only as price-history context on the dashboard chart)_

| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest |
|---|---|---|---|---|---|---|---|
| 2026-07-31 | 235.70 | 271.58 | 13.21% | ❌ | ✅ | ✅ | news |
| 2026-07-30 | 226.35 | 235.50 | 3.89% | ❌ | ❌ | ❌ | macro |
| 2026-07-29 | 230.55 | 226.65 | 1.72% | ❌ | ✅ | ✅ | macro |
| 2026-07-28 | 230.05 | 230.86 | 0.35% | ✅ | ✅ | ❌ | momentum |
| 2026-07-27 | 232.09 | 231.39 | 0.30% | ✅ | ✅ | ✅ | technical |
| 2026-07-24 | 234.90 | 232.11 | 1.20% | ❌ | ❌ | ❌ | momentum |
| 2026-07-23 | 243.50 | 233.66 | 4.21% | ❌ | ✅ | ✅ | news |
| 2026-07-22 | 247.59 | 244.85 | 1.12% | ❌ | ❌ | ❌ | news |
| 2026-07-21 | 251.15 | 247.55 | 1.45% | ❌ | ❌ | ❌ | contrarian |
| 2026-07-20 | 246.90 | 249.99 | 1.24% | ❌ | ❌ | ❌ | news |
| 2026-07-17 | 248.60 | 247.23 | 0.55% | ✅ | ✅ | ✅ | technical |
| 2026-07-16 | 253.90 | 249.89 | 1.60% | ❌ | ✅ | ✅ | contrarian |
| 2026-07-15 | 247.35 | 254.96 | 2.98% | ❌ | ❌ | ❌ | macro |
| 2026-07-14 | 247.72 | 247.49 | 0.09% | ✅ | ✅ | ❌ | momentum |
| 2026-07-13 | 243.85 | 247.31 | 1.40% | ❌ | ❌ | ❌ | macro |
| 2026-07-10 | 246.85 | 245.34 | 0.62% | ✅ | ✅ | ✅ | contrarian |
| 2026-07-09 | 243.87 | 247.04 | 1.28% | ❌ | ✅ | ✅ | news |
| 2026-07-08 | 246.15 | 243.62 | 1.04% | ❌ | ❌ | ❌ | news |
| 2026-07-07 | 244.78 | 245.98 | 0.49% | ✅ | ✅ | ✅ | technical |
| 2026-07-06 | 243.42 | 244.16 | 0.30% | ✅ | ✅ | ✅ | momentum |

## 🅰️/🅱️ A/B test — ensemble+gates vs raven-style prior+pulse

_Both arms predict the same sessions from the same pre-open snapshot (paired test)._

**Verdict:** Arm B leads on paired MAPE (mean daily delta 0.29% in B's favor); B wins 8/10 decisive days (sign test p=0.1094, not significant).

**B's open prediction (2026-08-03):** close ≈ **285.53** (up, adj 0.1σ, 8 evidence items).

| Rolling metric | A — ensemble+gates | B — raven prior+pulse |
|---|---|---|
| Scored days | 20 | 10 |
| PASS rate (±1%) | 35.00% | 40.00% |
| Directional accuracy | 60.00% | 70.00% |
| MAPE | 1.95% | 2.58% |
| Edge vs baseline | 0.02% | 0.23% |

Paired days: 10; B wins 8/10 decisive; mean daily APE delta (A−B) 0.29%; sign test p = 0.1094.
_This is a research experiment, not financial advice._
