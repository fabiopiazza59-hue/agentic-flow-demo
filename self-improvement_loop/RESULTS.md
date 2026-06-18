# 📈 AMZN Daily Close Predictor — Results

_Auto-generated each trading day. PASS = predicted close within ±1% of actual._

## Verdict
❌ **No edge yet** — rolling MAPE 1.88% does not beat the random-walk baseline 1.64%. Keep learning.

**Today's open prediction (2026-06-18):** close ≈ **237.90** (up, confidence 40.00%) vs prior close 237.50.

## Rolling metrics (last 5 scored days)

| Metric | Rolling | All-time |
|---|---|---|
| Scored days | 5 | 5 |
| PASS rate (±1%) | 40.00% | 40.00% |
| Directional accuracy | 20.00% | 20.00% |
| MAPE | 1.88% | 1.88% |
| Baseline MAPE (random walk) | 1.64% | 1.64% |
| Edge (baseline − model) | -0.24% | -0.24% |
| Brier (confidence calib.) | 0.25 | 0.25 |

## Per-strategy scorecards

| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |
|---|---|---|---|---|
| news | 5 | 20.00% | 1.67% | 0.23 |
| momentum | 5 | 40.00% | 1.92% | 0.20 |
| technical | 5 | 20.00% | 1.97% | 0.20 |
| contrarian | 5 | 0.00% | 2.01% | 0.19 |
| macro | 5 | 20.00% | 2.15% | 0.18 |

## Last 20 scored model predictions
_(backfill seed rows are excluded from metrics and this table; they appear only as price-history context on the dashboard chart)_

| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest |
|---|---|---|---|---|---|---|---|
| 2026-06-17 | 246.35 | 237.50 | 3.73% | ❌ | ❌ | ❌ | technical |
| 2026-06-16 | 246.42 | 246.00 | 0.17% | ✅ | ❌ | ❌ | momentum |
| 2026-06-15 | 237.65 | 246.02 | 3.40% | ❌ | ❌ | ❌ | news |
| 2026-06-12 | 243.35 | 238.55 | 2.01% | ❌ | ❌ | ❌ | momentum |
| 2026-06-08 | 245.05 | 245.22 | 0.07% | ✅ | ✅ | ✅ | macro |

_This is a research experiment, not financial advice._
