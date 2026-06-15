# 📈 AMZN Daily Close Predictor — Results

_Auto-generated each trading day. PASS = predicted close within ±1% of actual._

## Verdict
❌ **No edge yet** — rolling MAPE 1.04% does not beat the random-walk baseline 0.79%. Keep learning.

**Today's open prediction (2026-06-15):** close ≈ **237.65** (down, confidence 40.00%) vs prior close 238.55.

## Rolling metrics (last 2 scored days)

| Metric | Rolling | All-time |
|---|---|---|
| Scored days | 2 | 2 |
| PASS rate (±1%) | 50.00% | 50.00% |
| Directional accuracy | 50.00% | 50.00% |
| MAPE | 1.04% | 1.04% |
| Baseline MAPE (random walk) | 0.79% | 0.79% |
| Edge (baseline − model) | -0.26% | -0.26% |
| Brier (confidence calib.) | 0.29 | 0.29 |

## Per-strategy scorecards

| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |
|---|---|---|---|---|
| momentum | 2 | 50.00% | 1.07% | 0.20 |
| macro | 2 | 50.00% | 1.10% | 0.20 |
| technical | 2 | 0.00% | 1.28% | 0.20 |
| news | 2 | 0.00% | 1.35% | 0.20 |
| contrarian | 2 | 0.00% | 1.56% | 0.20 |

## Last 20 scored model predictions
_(backfill seed rows are excluded from metrics and this table; they appear only as price-history context on the dashboard chart)_

| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest |
|---|---|---|---|---|---|---|---|
| 2026-06-12 | 243.35 | 238.55 | 2.01% | ❌ | ❌ | ❌ | momentum |
| 2026-06-08 | 245.05 | 245.22 | 0.07% | ✅ | ✅ | ✅ | macro |

_This is a research experiment, not financial advice._
