# 📈 AMZN Daily Close Predictor — Results

_Auto-generated each trading day. PASS = predicted close within ±1% of actual._

## Verdict
❌ **No edge yet** — rolling MAPE 2.45% does not beat the random-walk baseline 2.29%. Keep learning.

**Today's open prediction (2026-06-23):** close ≈ **232.95** (up, confidence 40.00%) vs prior close 232.79.

## Rolling metrics (last 7 scored days)

| Metric | Rolling | All-time |
|---|---|---|
| Scored days | 7 | 7 |
| PASS rate (±1%) | 28.57% | 28.57% |
| Directional accuracy | 28.57% | 28.57% |
| MAPE | 2.45% | 2.45% |
| Baseline MAPE (random walk) | 2.29% | 2.29% |
| Edge (baseline − model) | -0.16% | -0.16% |
| Brier (confidence calib.) | 0.22 | 0.22 |

## Per-strategy scorecards

| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |
|---|---|---|---|---|
| news | 7 | 28.57% | 1.98% | 0.24 |
| technical | 7 | 14.29% | 2.36% | 0.20 |
| contrarian | 7 | 14.29% | 2.51% | 0.19 |
| momentum | 7 | 28.57% | 2.57% | 0.19 |
| macro | 7 | 14.29% | 2.78% | 0.17 |

## Last 20 scored model predictions
_(backfill seed rows are excluded from metrics and this table; they appear only as price-history context on the dashboard chart)_

| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest |
|---|---|---|---|---|---|---|---|
| 2026-06-22 | 244.62 | 232.79 | 5.08% | ❌ | ❌ | ❌ | news |
| 2026-06-18 | 237.90 | 244.39 | 2.66% | ❌ | ✅ | ✅ | contrarian |
| 2026-06-17 | 246.35 | 237.50 | 3.73% | ❌ | ❌ | ❌ | technical |
| 2026-06-16 | 246.42 | 246.00 | 0.17% | ✅ | ❌ | ❌ | momentum |
| 2026-06-15 | 237.65 | 246.02 | 3.40% | ❌ | ❌ | ❌ | news |
| 2026-06-12 | 243.35 | 238.55 | 2.01% | ❌ | ❌ | ❌ | momentum |
| 2026-06-08 | 245.05 | 245.22 | 0.07% | ✅ | ✅ | ✅ | macro |

_This is a research experiment, not financial advice._
