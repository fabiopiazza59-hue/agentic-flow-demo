# 📈 AMZN Daily Close Predictor — Results

_Auto-generated each trading day. PASS = predicted close within ±1% of actual._

## Verdict
❌ **No edge yet** — rolling MAPE 2.14% does not beat the random-walk baseline 1.98%. Keep learning.

**Today's open prediction (2026-06-26):** close ≈ **225.30** (down, confidence 48.00%) vs prior close 227.01.

## Rolling metrics (last 10 scored days)

| Metric | Rolling | All-time |
|---|---|---|
| Scored days | 10 | 10 |
| PASS rate (±1%) | 40.00% | 40.00% |
| Directional accuracy | 40.00% | 40.00% |
| MAPE | 2.14% | 2.14% |
| Baseline MAPE (random walk) | 1.98% | 1.98% |
| Edge (baseline − model) | -0.16% | -0.16% |
| Brier (confidence calib.) | 0.25 | 0.25 |

## Per-strategy scorecards

| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |
|---|---|---|---|---|
| news | 10 | 30.00% | 1.98% | 0.22 |
| technical | 10 | 20.00% | 1.98% | 0.22 |
| momentum | 10 | 30.00% | 2.21% | 0.20 |
| contrarian | 10 | 10.00% | 2.33% | 0.19 |
| macro | 10 | 10.00% | 2.43% | 0.18 |

## Last 20 scored model predictions
_(backfill seed rows are excluded from metrics and this table; they appear only as price-history context on the dashboard chart)_

| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest |
|---|---|---|---|---|---|---|---|
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
