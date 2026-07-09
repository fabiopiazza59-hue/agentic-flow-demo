# 📈 AMZN Daily Close Predictor — Results

_Auto-generated each trading day. PASS = predicted close within ±1% of actual._

## Verdict
❌ **No edge yet** — rolling MAPE 1.81% does not beat the random-walk baseline 1.68%. Keep learning.

**Today's open prediction (2026-07-09):** close ≈ **243.87** (up, confidence 40.00%) vs prior close 243.62.

## Rolling metrics (last 18 scored days)

| Metric | Rolling | All-time |
|---|---|---|
| Scored days | 18 | 18 |
| PASS rate (±1%) | 38.89% | 38.89% |
| Directional accuracy | 38.89% | 38.89% |
| MAPE | 1.81% | 1.81% |
| Baseline MAPE (random walk) | 1.68% | 1.68% |
| Edge (baseline − model) | -0.13% | -0.13% |
| Brier (confidence calib.) | 0.25 | 0.25 |

## Per-strategy scorecards

| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |
|---|---|---|---|---|
| technical | 18 | 22.22% | 1.55% | 0.23 |
| news | 18 | 33.33% | 1.59% | 0.22 |
| contrarian | 18 | 11.11% | 1.78% | 0.20 |
| momentum | 18 | 27.78% | 1.91% | 0.18 |
| macro | 18 | 5.56% | 1.97% | 0.18 |

## Last 20 scored model predictions
_(backfill seed rows are excluded from metrics and this table; they appear only as price-history context on the dashboard chart)_

| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest |
|---|---|---|---|---|---|---|---|
| 2026-07-08 | 246.15 | 243.62 | 1.04% | ❌ | ❌ | ❌ | news |
| 2026-07-07 | 244.78 | 245.98 | 0.49% | ✅ | ✅ | ✅ | technical |
| 2026-07-06 | 243.42 | 244.16 | 0.30% | ✅ | ✅ | ✅ | momentum |
| 2026-07-02 | 241.85 | 242.67 | 0.34% | ✅ | ✅ | ✅ | momentum |
| 2026-07-01 | 238.15 | 241.70 | 1.47% | ❌ | ❌ | ❌ | news |
| 2026-06-30 | 240.85 | 238.34 | 1.05% | ❌ | ❌ | ❌ | technical |
| 2026-06-29 | 232.30 | 240.14 | 3.26% | ❌ | ❌ | ❌ | news |
| 2026-06-26 | 225.30 | 232.69 | 3.18% | ❌ | ❌ | ❌ | contrarian |
| 2026-06-25 | 235.10 | 227.01 | 3.56% | ❌ | ❌ | ❌ | momentum |
| 2026-06-24 | 234.74 | 234.27 | 0.20% | ✅ | ✅ | ❌ | technical |
| 2026-06-23 | 232.95 | 234.11 | 0.50% | ✅ | ✅ | ✅ | news |
| 2026-06-22 | 244.62 | 232.79 | 5.08% | ❌ | ❌ | ❌ | news |
| 2026-06-18 | 237.90 | 244.39 | 2.66% | ❌ | ✅ | ✅ | contrarian |
| 2026-06-17 | 246.35 | 237.50 | 3.73% | ❌ | ❌ | ❌ | technical |
| 2026-06-16 | 246.42 | 246.00 | 0.17% | ✅ | ❌ | ❌ | momentum |
| 2026-06-15 | 237.65 | 246.02 | 3.40% | ❌ | ❌ | ❌ | news |
| 2026-06-12 | 243.35 | 238.55 | 2.01% | ❌ | ❌ | ❌ | momentum |
| 2026-06-08 | 245.05 | 245.22 | 0.07% | ✅ | ✅ | ✅ | macro |

_This is a research experiment, not financial advice._
