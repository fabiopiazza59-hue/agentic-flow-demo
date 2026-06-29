# 📈 AMZN Daily Close Predictor — Results

_Auto-generated each trading day. PASS = predicted close within ±1% of actual._

## Verdict
❌ **No edge yet** — rolling MAPE 2.23% does not beat the random-walk baseline 2.02%. Keep learning.

**Today's open prediction (2026-06-29):** close ≈ **232.30** (down, confidence 42.00%) vs prior close 232.69.

## Rolling metrics (last 11 scored days)

| Metric | Rolling | All-time |
|---|---|---|
| Scored days | 11 | 11 |
| PASS rate (±1%) | 36.36% | 36.36% |
| Directional accuracy | 36.36% | 36.36% |
| MAPE | 2.23% | 2.23% |
| Baseline MAPE (random walk) | 2.02% | 2.02% |
| Edge (baseline − model) | -0.21% | -0.21% |
| Brier (confidence calib.) | 0.25 | 0.25 |

## Per-strategy scorecards

| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |
|---|---|---|---|---|
| technical | 11 | 18.18% | 1.92% | 0.23 |
| news | 11 | 27.27% | 2.12% | 0.21 |
| contrarian | 11 | 18.18% | 2.16% | 0.20 |
| momentum | 11 | 27.27% | 2.32% | 0.19 |
| macro | 11 | 9.09% | 2.53% | 0.17 |

## Last 20 scored model predictions
_(backfill seed rows are excluded from metrics and this table; they appear only as price-history context on the dashboard chart)_

| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest |
|---|---|---|---|---|---|---|---|
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
