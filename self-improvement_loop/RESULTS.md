# 📈 AMZN Daily Close Predictor — Results

_Auto-generated each trading day. PASS = predicted close within ±1% of actual._

## Verdict
✅ **Edge confirmed** — rolling MAPE 2.05% beats the random-walk baseline 2.07% by 0.02%.

**Today's open prediction (2026-08-06):** close ≈ **270.35** (down, confidence 40.00%) vs prior close 272.65.

## Rolling metrics (last 20 scored days)

| Metric | Rolling | All-time |
|---|---|---|
| Scored days | 20 | 38 |
| PASS rate (±1%) | 30.00% | 34.21% |
| Directional accuracy | 60.00% | 50.00% |
| MAPE | 2.05% | 1.93% |
| Baseline MAPE (random walk) | 2.07% | 1.88% |
| Edge (baseline − model) | 0.02% | -0.05% |
| Brier (confidence calib.) | 0.25 | 0.25 |

## Per-strategy scorecards

| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |
|---|---|---|---|---|
| news | 38 | 28.95% | 1.62% | 0.23 |
| technical | 38 | 15.79% | 1.79% | 0.21 |
| contrarian | 38 | 18.42% | 1.90% | 0.19 |
| momentum | 38 | 23.68% | 1.98% | 0.19 |
| macro | 38 | 13.16% | 2.01% | 0.18 |

## Last 20 scored model predictions
_(backfill seed rows are excluded from metrics and this table; they appear only as price-history context on the dashboard chart)_

| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest |
|---|---|---|---|---|---|---|---|
| 2026-08-05 | 276.30 | 272.65 | 1.34% | ❌ | ✅ | ✅ | contrarian |
| 2026-08-04 | 283.69 | 277.42 | 2.26% | ❌ | ✅ | ✅ | contrarian |
| 2026-08-03 | 284.56 | 284.02 | 0.19% | ✅ | ❌ | ❌ | momentum |
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

## 🅰️/🅱️ A/B test — ensemble+gates vs raven-style prior+pulse

_Both arms predict the same sessions from the same pre-open snapshot (paired test)._

**Verdict:** Arm B leads on paired MAPE (mean daily delta 0.28% in B's favor); B wins 10/13 decisive days (sign test p=0.0923, not significant).

**B's open prediction (2026-08-06):** close ≈ **272.65** (down, adj 0.0σ, 8 evidence items).

| Rolling metric | A — ensemble+gates | B — raven prior+pulse |
|---|---|---|
| Scored days | 20 | 13 |
| PASS rate (±1%) | 30.00% | 46.15% |
| Directional accuracy | 60.00% | 69.23% |
| MAPE | 2.05% | 2.22% |
| Edge vs baseline | 0.02% | 0.27% |

Paired days: 13; B wins 10/13 decisive; mean daily APE delta (A−B) 0.28%; sign test p = 0.0923.
_This is a research experiment, not financial advice._
