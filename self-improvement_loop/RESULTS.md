# 📈 AMZN Daily Close Predictor — Results

_Auto-generated each trading day. PASS = predicted close within ±1% of actual._

## Verdict
❌ **No edge yet** — rolling MAPE 1.08% does not beat the random-walk baseline 1.05%. Keep learning.

**Today's open prediction (2026-09-03):** close ≈ **254.70** (down, confidence 62.00%) vs prior close 254.98.

## Rolling metrics (last 20 scored days)

| Metric | Rolling | All-time |
|---|---|---|
| Scored days | 20 | 58 |
| PASS rate (±1%) | 60.00% | 43.10% |
| Directional accuracy | 45.00% | 48.28% |
| MAPE | 1.08% | 1.64% |
| Baseline MAPE (random walk) | 1.05% | 1.59% |
| Edge (baseline − model) | -0.03% | -0.04% |
| Brier (confidence calib.) | 0.26 | 0.25 |

## Per-strategy scorecards

| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |
|---|---|---|---|---|
| news | 58 | 27.59% | 1.46% | 0.22 |
| technical | 58 | 15.52% | 1.48% | 0.21 |
| contrarian | 58 | 18.97% | 1.65% | 0.19 |
| momentum | 58 | 25.86% | 1.68% | 0.19 |
| macro | 58 | 12.07% | 1.72% | 0.19 |

## Last 20 scored model predictions
_(backfill seed rows are excluded from metrics and this table; they appear only as price-history context on the dashboard chart)_

| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest |
|---|---|---|---|---|---|---|---|
| 2026-09-02 | 253.69 | 254.98 | 0.51% | ✅ | ❌ | ❌ | momentum |
| 2026-09-01 | 258.91 | 254.92 | 1.57% | ❌ | ✅ | ✅ | news |
| 2026-08-31 | 265.87 | 259.77 | 2.35% | ❌ | ✅ | ✅ | news |
| 2026-08-28 | 265.99 | 266.43 | 0.17% | ✅ | ✅ | ❌ | news |
| 2026-08-27 | 256.85 | 256.26 | 0.23% | ✅ | ❌ | ❌ | momentum |
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

## 🅰️/🅱️ A/B test — ensemble+gates vs raven-style prior+pulse

_Both arms predict the same sessions from the same pre-open snapshot (paired test)._

**Verdict:** Arm B leads on paired MAPE (mean daily delta 0.07% in B's favor); B wins 18/33 decisive days (sign test p=0.7283, not significant).

**B's open prediction (2026-09-03):** close ≈ **255.4** (up, adj 0.1σ, 8 evidence items).

| Rolling metric | A — ensemble+gates | B — raven prior+pulse |
|---|---|---|
| Scored days | 20 | 20 |
| PASS rate (±1%) | 60.00% | 55.00% |
| Directional accuracy | 45.00% | 50.00% |
| MAPE | 1.08% | 1.13% |
| Edge vs baseline | -0.03% | -0.09% |

Paired days: 33; B wins 18/33 decisive; mean daily APE delta (A−B) 0.07%; sign test p = 0.7283.
_This is a research experiment, not financial advice._
