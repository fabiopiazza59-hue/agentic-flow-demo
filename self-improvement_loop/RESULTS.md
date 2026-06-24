# 📈 AMZN Daily Close Predictor — Results

_Auto-generated each trading day. PASS = predicted close within ±1% of actual._

## Verdict
❌ **No edge yet** — rolling MAPE 2.20% does not beat the random-walk baseline 2.07%. Keep learning.

**Today's open prediction (2026-06-24):** close ≈ **234.74** (up, confidence 40.00%) vs prior close 234.11.

## Rolling metrics (last 8 scored days)

| Metric | Rolling | All-time |
|---|---|---|
| Scored days | 8 | 8 |
| PASS rate (±1%) | 37.50% | 37.50% |
| Directional accuracy | 37.50% | 37.50% |
| MAPE | 2.20% | 2.20% |
| Baseline MAPE (random walk) | 2.07% | 2.07% |
| Edge (baseline − model) | -0.13% | -0.13% |
| Brier (confidence calib.) | 0.24 | 0.24 |

## Per-strategy scorecards

| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |
|---|---|---|---|---|
| news | 8 | 37.50% | 1.75% | 0.25 |
| technical | 8 | 12.50% | 2.08% | 0.21 |
| contrarian | 8 | 12.50% | 2.27% | 0.19 |
| momentum | 8 | 25.00% | 2.39% | 0.18 |
| macro | 8 | 12.50% | 2.45% | 0.18 |

## Last 20 scored model predictions
_(backfill seed rows are excluded from metrics and this table; they appear only as price-history context on the dashboard chart)_

| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest |
|---|---|---|---|---|---|---|---|
| 2026-06-23 | 232.95 | 234.11 | 0.50% | ✅ | ✅ | ✅ | news |
| 2026-06-22 | 244.62 | 232.79 | 5.08% | ❌ | ❌ | ❌ | news |
| 2026-06-18 | 237.90 | 244.39 | 2.66% | ❌ | ✅ | ✅ | contrarian |
| 2026-06-17 | 246.35 | 237.50 | 3.73% | ❌ | ❌ | ❌ | technical |
| 2026-06-16 | 246.42 | 246.00 | 0.17% | ✅ | ❌ | ❌ | momentum |
| 2026-06-15 | 237.65 | 246.02 | 3.40% | ❌ | ❌ | ❌ | news |
| 2026-06-12 | 243.35 | 238.55 | 2.01% | ❌ | ❌ | ❌ | momentum |
| 2026-06-08 | 245.05 | 245.22 | 0.07% | ✅ | ✅ | ✅ | macro |

_This is a research experiment, not financial advice._
