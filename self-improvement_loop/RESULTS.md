# 📈 AMZN Daily Close Predictor — Results

_Auto-generated each trading day. PASS = predicted close within ±1% of actual._

## Verdict
❌ **No edge yet** — rolling MAPE 2.22% does not beat the random-walk baseline 2.01%. Keep learning.

**Today's open prediction (2026-07-01):** close ≈ **238.15** (down, confidence 40.00%) vs prior close 238.34.

## Rolling metrics (last 13 scored days)

| Metric | Rolling | All-time |
|---|---|---|
| Scored days | 13 | 13 |
| PASS rate (±1%) | 30.77% | 30.77% |
| Directional accuracy | 30.77% | 30.77% |
| MAPE | 2.22% | 2.22% |
| Baseline MAPE (random walk) | 2.01% | 2.01% |
| Edge (baseline − model) | -0.21% | -0.21% |
| Brier (confidence calib.) | 0.24 | 0.24 |

## Per-strategy scorecards

| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |
|---|---|---|---|---|
| news | 13 | 30.77% | 1.90% | 0.22 |
| technical | 13 | 23.08% | 1.93% | 0.22 |
| contrarian | 13 | 15.38% | 2.02% | 0.21 |
| momentum | 13 | 23.08% | 2.36% | 0.18 |
| macro | 13 | 7.69% | 2.42% | 0.17 |

## Last 20 scored model predictions
_(backfill seed rows are excluded from metrics and this table; they appear only as price-history context on the dashboard chart)_

| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest |
|---|---|---|---|---|---|---|---|
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
