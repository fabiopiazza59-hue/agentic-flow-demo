# 📈 AMZN Daily Close Predictor — Results

_Auto-generated each trading day. PASS = predicted close within ±1% of actual._

## Verdict
❌ **No edge yet** — rolling MAPE 1.41% does not beat the random-walk baseline 1.15%. Keep learning.

**Today's open prediction (2026-06-17):** close ≈ **246.35** (up, confidence 38.00%) vs prior close 246.00.

## Rolling metrics (last 4 scored days)

| Metric | Rolling | All-time |
|---|---|---|
| Scored days | 4 | 4 |
| PASS rate (±1%) | 50.00% | 50.00% |
| Directional accuracy | 25.00% | 25.00% |
| MAPE | 1.41% | 1.41% |
| Baseline MAPE (random walk) | 1.15% | 1.15% |
| Edge (baseline − model) | -0.26% | -0.26% |
| Brier (confidence calib.) | 0.27 | 0.27 |

## Per-strategy scorecards

| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |
|---|---|---|---|---|
| news | 4 | 25.00% | 1.01% | 0.28 |
| contrarian | 4 | 0.00% | 1.36% | 0.20 |
| momentum | 4 | 50.00% | 1.55% | 0.18 |
| technical | 4 | 0.00% | 1.64% | 0.17 |
| macro | 4 | 25.00% | 1.64% | 0.17 |

## Last 20 scored model predictions
_(backfill seed rows are excluded from metrics and this table; they appear only as price-history context on the dashboard chart)_

| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest |
|---|---|---|---|---|---|---|---|
| 2026-06-16 | 246.42 | 246.00 | 0.17% | ✅ | ❌ | ❌ | momentum |
| 2026-06-15 | 237.65 | 246.02 | 3.40% | ❌ | ❌ | ❌ | news |
| 2026-06-12 | 243.35 | 238.55 | 2.01% | ❌ | ❌ | ❌ | momentum |
| 2026-06-08 | 245.05 | 245.22 | 0.07% | ✅ | ✅ | ✅ | macro |

_This is a research experiment, not financial advice._
