# 📈 AMZN Daily Close Predictor — Results

_Auto-generated each trading day. PASS = predicted close within ±1% of actual._

## Verdict
❌ **No edge yet** — rolling MAPE 1.95% does not beat the random-walk baseline 1.95%. Keep learning.

**Today's open prediction (2026-08-04):** close ≈ **283.69** (down, confidence 32.00%) vs prior close 284.02.

## Rolling metrics (last 20 scored days)

| Metric | Rolling | All-time |
|---|---|---|
| Scored days | 20 | 36 |
| PASS rate (±1%) | 35.00% | 36.11% |
| Directional accuracy | 55.00% | 47.22% |
| MAPE | 1.95% | 1.94% |
| Baseline MAPE (random walk) | 1.95% | 1.87% |
| Edge (baseline − model) | -0.00% | -0.07% |
| Brier (confidence calib.) | 0.27 | 0.26 |

## Per-strategy scorecards

| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |
|---|---|---|---|---|
| news | 36 | 30.56% | 1.61% | 0.23 |
| technical | 36 | 16.67% | 1.82% | 0.20 |
| contrarian | 36 | 13.89% | 1.96% | 0.19 |
| momentum | 36 | 25.00% | 1.99% | 0.19 |
| macro | 36 | 13.89% | 2.01% | 0.19 |

## Last 20 scored model predictions
_(backfill seed rows are excluded from metrics and this table; they appear only as price-history context on the dashboard chart)_

| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest |
|---|---|---|---|---|---|---|---|
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
| 2026-07-08 | 246.15 | 243.62 | 1.04% | ❌ | ❌ | ❌ | news |
| 2026-07-07 | 244.78 | 245.98 | 0.49% | ✅ | ✅ | ✅ | technical |

## 🅰️/🅱️ A/B test — ensemble+gates vs raven-style prior+pulse

_Both arms predict the same sessions from the same pre-open snapshot (paired test)._

**Verdict:** Arm B leads on paired MAPE (mean daily delta 0.23% in B's favor); B wins 8/11 decisive days (sign test p=0.2266, not significant).

**B's open prediction (2026-08-04):** close ≈ **281.86** (down, adj -0.2σ, 8 evidence items).

| Rolling metric | A — ensemble+gates | B — raven prior+pulse |
|---|---|---|
| Scored days | 20 | 11 |
| PASS rate (±1%) | 35.00% | 45.45% |
| Directional accuracy | 55.00% | 63.64% |
| MAPE | 1.95% | 2.39% |
| Edge vs baseline | -0.00% | 0.17% |

Paired days: 11; B wins 8/11 decisive; mean daily APE delta (A−B) 0.23%; sign test p = 0.2266.
_This is a research experiment, not financial advice._
