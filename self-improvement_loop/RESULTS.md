# 📈 AMZN Daily Close Predictor — Results

_Auto-generated each trading day. PASS = predicted close within ±1% of actual._

## Verdict
❌ **No edge yet** — rolling MAPE 1.83% does not beat the random-walk baseline 1.54%. Keep learning.

**Today's open prediction (2026-06-16):** close ≈ **246.42** (up, confidence 42.00%) vs prior close 246.02.

## Rolling metrics (last 3 scored days)

| Metric | Rolling | All-time |
|---|---|---|
| Scored days | 3 | 3 |
| PASS rate (±1%) | 33.33% | 33.33% |
| Directional accuracy | 33.33% | 33.33% |
| MAPE | 1.83% | 1.83% |
| Baseline MAPE (random walk) | 1.54% | 1.54% |
| Edge (baseline − model) | -0.29% | -0.29% |
| Brier (confidence calib.) | 0.25 | 0.25 |

## Per-strategy scorecards

| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |
|---|---|---|---|---|
| news | 3 | 33.33% | 1.06% | 0.31 |
| contrarian | 3 | 0.00% | 1.61% | 0.20 |
| technical | 3 | 0.00% | 1.96% | 0.17 |
| momentum | 3 | 33.33% | 1.96% | 0.16 |
| macro | 3 | 33.33% | 1.98% | 0.16 |

## Last 20 scored model predictions
_(backfill seed rows are excluded from metrics and this table; they appear only as price-history context on the dashboard chart)_

| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest |
|---|---|---|---|---|---|---|---|
| 2026-06-15 | 237.65 | 246.02 | 3.40% | ❌ | ❌ | ❌ | news |
| 2026-06-12 | 243.35 | 238.55 | 2.01% | ❌ | ❌ | ❌ | momentum |
| 2026-06-08 | 245.05 | 245.22 | 0.07% | ✅ | ✅ | ✅ | macro |

_This is a research experiment, not financial advice._
