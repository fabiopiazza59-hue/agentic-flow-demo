# 📈 AMZN Daily Close Predictor — Results

_Auto-generated each trading day. PASS = predicted close within ±1% of actual._

## Verdict
❌ **No edge yet** — rolling MAPE 1.98% does not beat the random-walk baseline 1.85%. Keep learning.

**Today's open prediction (2026-06-25):** close ≈ **235.10** (up, confidence 45.00%) vs prior close 234.27.

## Rolling metrics (last 9 scored days)

| Metric | Rolling | All-time |
|---|---|---|
| Scored days | 9 | 9 |
| PASS rate (±1%) | 44.44% | 44.44% |
| Directional accuracy | 44.44% | 44.44% |
| MAPE | 1.98% | 1.98% |
| Baseline MAPE (random walk) | 1.85% | 1.85% |
| Edge (baseline − model) | -0.13% | -0.13% |
| Brier (confidence calib.) | 0.25 | 0.25 |

## Per-strategy scorecards

| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |
|---|---|---|---|---|
| news | 9 | 33.33% | 1.63% | 0.24 |
| technical | 9 | 22.22% | 1.88% | 0.21 |
| contrarian | 9 | 11.11% | 2.12% | 0.19 |
| momentum | 9 | 22.22% | 2.18% | 0.18 |
| macro | 9 | 11.11% | 2.27% | 0.18 |

## Last 20 scored model predictions
_(backfill seed rows are excluded from metrics and this table; they appear only as price-history context on the dashboard chart)_

| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest |
|---|---|---|---|---|---|---|---|
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
