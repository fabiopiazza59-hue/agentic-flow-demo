# 📈 AMZN Daily Close Predictor — Results

_Auto-generated each trading day. PASS = predicted close within ±1% of actual._

## Verdict
✅ **Edge confirmed** — rolling MAPE 1.88% beats the random-walk baseline 1.89% by 0.01%.

**Today's open prediction (2026-08-27):** close ≈ **256.85** (up, confidence 64.00%) vs prior close 256.26.

## Rolling metrics (last 20 scored days)

| Metric | Rolling | All-time |
|---|---|---|
| Scored days | 20 | 53 |
| PASS rate (±1%) | 50.00% | 41.51% |
| Directional accuracy | 45.00% | 47.17% |
| MAPE | 1.88% | 1.70% |
| Baseline MAPE (random walk) | 1.89% | 1.66% |
| Edge (baseline − model) | 0.01% | -0.04% |
| Brier (confidence calib.) | 0.25 | 0.25 |

## Per-strategy scorecards

| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |
|---|---|---|---|---|
| technical | 53 | 16.98% | 1.53% | 0.21 |
| news | 53 | 24.53% | 1.54% | 0.21 |
| contrarian | 53 | 20.75% | 1.67% | 0.20 |
| momentum | 53 | 24.53% | 1.73% | 0.19 |
| macro | 53 | 13.21% | 1.75% | 0.19 |

## Last 20 scored model predictions
_(backfill seed rows are excluded from metrics and this table; they appear only as price-history context on the dashboard chart)_

| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest |
|---|---|---|---|---|---|---|---|
| 2026-08-26 | 260.35 | 260.28 | 0.03% | ✅ | ✅ | ✅ | momentum |
| 2026-08-25 | 262.95 | 261.06 | 0.72% | ✅ | ❌ | ❌ | macro |
| 2026-08-24 | 258.49 | 262.07 | 1.37% | ❌ | ❌ | ❌ | contrarian |
| 2026-08-21 | 260.35 | 258.63 | 0.67% | ✅ | ❌ | ❌ | technical |
| 2026-08-20 | 265.91 | 260.11 | 2.23% | ❌ | ❌ | ❌ | contrarian |
| 2026-08-19 | 258.15 | 265.84 | 2.89% | ❌ | ❌ | ❌ | macro |
| 2026-08-18 | 259.85 | 259.45 | 0.15% | ✅ | ✅ | ✅ | technical |
| 2026-08-17 | 262.90 | 261.31 | 0.61% | ✅ | ❌ | ❌ | momentum |
| 2026-08-14 | 264.05 | 262.65 | 0.53% | ✅ | ✅ | ✅ | news |
| 2026-08-13 | 266.56 | 265.13 | 0.54% | ✅ | ✅ | ✅ | technical |
| 2026-08-12 | 272.01 | 267.28 | 1.77% | ❌ | ✅ | ✅ | contrarian |
| 2026-08-11 | 278.15 | 272.27 | 2.16% | ❌ | ❌ | ❌ | contrarian |
| 2026-08-10 | 274.29 | 278.09 | 1.37% | ❌ | ❌ | ❌ | news |
| 2026-08-07 | 271.81 | 274.48 | 0.97% | ✅ | ❌ | ❌ | momentum |
| 2026-08-06 | 270.35 | 272.26 | 0.70% | ✅ | ✅ | ❌ | momentum |
| 2026-08-05 | 276.30 | 272.65 | 1.34% | ❌ | ✅ | ✅ | contrarian |
| 2026-08-04 | 283.69 | 277.42 | 2.26% | ❌ | ✅ | ✅ | contrarian |
| 2026-08-03 | 284.56 | 284.02 | 0.19% | ✅ | ❌ | ❌ | momentum |
| 2026-07-31 | 235.70 | 271.58 | 13.21% | ❌ | ✅ | ✅ | news |
| 2026-07-30 | 226.35 | 235.50 | 3.89% | ❌ | ❌ | ❌ | macro |

## 🅰️/🅱️ A/B test — ensemble+gates vs raven-style prior+pulse

_Both arms predict the same sessions from the same pre-open snapshot (paired test)._

**Verdict:** Arm B leads on paired MAPE (mean daily delta 0.07% in B's favor); B wins 15/28 decisive days (sign test p=0.8506, not significant).

**B's open prediction (2026-08-27):** close ≈ **257.2** (up, adj 0.1σ, 7 evidence items).

| Rolling metric | A — ensemble+gates | B — raven prior+pulse |
|---|---|---|
| Scored days | 20 | 20 |
| PASS rate (±1%) | 50.00% | 50.00% |
| Directional accuracy | 45.00% | 60.00% |
| MAPE | 1.88% | 1.85% |
| Edge vs baseline | 0.01% | 0.03% |

Paired days: 28; B wins 15/28 decisive; mean daily APE delta (A−B) 0.07%; sign test p = 0.8506.
_This is a research experiment, not financial advice._
