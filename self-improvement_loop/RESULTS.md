# 📈 AMZN Daily Close Predictor — Results

_Auto-generated each trading day. PASS = predicted close within ±1% of actual._

## Verdict
❌ **No edge yet** — rolling MAPE 2.01% does not beat the random-walk baseline 1.84%. Keep learning.

**Today's open prediction (2026-06-22):** close ≈ **244.62** (up, confidence 40.00%) vs prior close 244.39.

## Rolling metrics (last 6 scored days)

| Metric | Rolling | All-time |
|---|---|---|
| Scored days | 6 | 6 |
| PASS rate (±1%) | 33.33% | 33.33% |
| Directional accuracy | 33.33% | 33.33% |
| MAPE | 2.01% | 2.01% |
| Baseline MAPE (random walk) | 1.84% | 1.84% |
| Edge (baseline − model) | -0.17% | -0.17% |
| Brier (confidence calib.) | 0.23 | 0.23 |

## Per-strategy scorecards

| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |
|---|---|---|---|---|
| news | 6 | 16.67% | 1.67% | 0.24 |
| contrarian | 6 | 16.67% | 1.94% | 0.20 |
| technical | 6 | 16.67% | 2.03% | 0.20 |
| momentum | 6 | 33.33% | 2.11% | 0.19 |
| macro | 6 | 16.67% | 2.31% | 0.17 |

## Last 20 scored model predictions
_(backfill seed rows are excluded from metrics and this table; they appear only as price-history context on the dashboard chart)_

| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest |
|---|---|---|---|---|---|---|---|
| 2026-06-18 | 237.90 | 244.39 | 2.66% | ❌ | ✅ | ✅ | contrarian |
| 2026-06-17 | 246.35 | 237.50 | 3.73% | ❌ | ❌ | ❌ | technical |
| 2026-06-16 | 246.42 | 246.00 | 0.17% | ✅ | ❌ | ❌ | momentum |
| 2026-06-15 | 237.65 | 246.02 | 3.40% | ❌ | ❌ | ❌ | news |
| 2026-06-12 | 243.35 | 238.55 | 2.01% | ❌ | ❌ | ❌ | momentum |
| 2026-06-08 | 245.05 | 245.22 | 0.07% | ✅ | ✅ | ✅ | macro |

_This is a research experiment, not financial advice._
