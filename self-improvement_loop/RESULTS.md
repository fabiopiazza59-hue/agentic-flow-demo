# 📈 AMZN Daily Close Predictor — Results

_Auto-generated each trading day. PASS = predicted close within ±1% of actual._

## Verdict
✅ **Edge confirmed** — rolling MAPE 0.07% beats the random-walk baseline 0.33% by 0.26%.

**Today's open prediction (2026-06-12):** close ≈ **243.35** (up, confidence 58.00%) vs prior close 241.51.

## Rolling metrics (last 1 scored days)

| Metric | Rolling | All-time |
|---|---|---|
| Scored days | 1 | 1 |
| PASS rate (±1%) | 100.00% | 100.00% |
| Directional accuracy | 100.00% | 100.00% |
| MAPE | 0.07% | 0.07% |
| Baseline MAPE (random walk) | 0.33% | 0.33% |
| Edge (baseline − model) | 0.26% | 0.26% |
| Brier (confidence calib.) | 0.25 | 0.25 |

## Per-strategy scorecards

| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |
|---|---|---|---|---|
| macro | 1 | 100.00% | 0.13% | 0.20 |
| technical | 1 | 0.00% | 0.33% | 0.20 |
| news | 1 | 0.00% | 0.33% | 0.20 |
| contrarian | 1 | 0.00% | 0.62% | 0.20 |
| momentum | 1 | 0.00% | 1.49% | 0.20 |

## Last 20 scored model predictions
_(backfill seed rows are excluded from metrics and this table; they appear only as price-history context on the dashboard chart)_

| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest |
|---|---|---|---|---|---|---|---|
| 2026-06-08 | 245.05 | 245.22 | 0.07% | ✅ | ✅ | ✅ | macro |

_This is a research experiment, not financial advice._
