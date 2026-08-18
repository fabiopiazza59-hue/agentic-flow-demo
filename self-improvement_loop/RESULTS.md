# 📈 AMZN Daily Close Predictor — Results

_Auto-generated each trading day. PASS = predicted close within ±1% of actual._

## Verdict
❌ **No edge yet** — rolling MAPE 1.99% does not beat the random-walk baseline 1.99%. Keep learning.

**Today's open prediction (2026-08-18):** close ≈ **259.85** (down, confidence 60.00%) vs prior close 261.31.

## Rolling metrics (last 20 scored days)

| Metric | Rolling | All-time |
|---|---|---|
| Scored days | 20 | 46 |
| PASS rate (±1%) | 40.00% | 39.13% |
| Directional accuracy | 55.00% | 50.00% |
| MAPE | 1.99% | 1.79% |
| Baseline MAPE (random walk) | 1.99% | 1.74% |
| Edge (baseline − model) | -0.01% | -0.04% |
| Brier (confidence calib.) | 0.25 | 0.25 |

## Per-strategy scorecards

| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |
|---|---|---|---|---|
| news | 46 | 28.26% | 1.55% | 0.22 |
| technical | 46 | 15.22% | 1.64% | 0.21 |
| contrarian | 46 | 19.57% | 1.74% | 0.20 |
| momentum | 46 | 26.09% | 1.81% | 0.19 |
| macro | 46 | 10.87% | 1.85% | 0.19 |

## Last 20 scored model predictions
_(backfill seed rows are excluded from metrics and this table; they appear only as price-history context on the dashboard chart)_

| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest |
|---|---|---|---|---|---|---|---|
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
| 2026-07-29 | 230.55 | 226.65 | 1.72% | ❌ | ✅ | ✅ | macro |
| 2026-07-28 | 230.05 | 230.86 | 0.35% | ✅ | ✅ | ❌ | momentum |
| 2026-07-27 | 232.09 | 231.39 | 0.30% | ✅ | ✅ | ✅ | technical |
| 2026-07-24 | 234.90 | 232.11 | 1.20% | ❌ | ❌ | ❌ | momentum |
| 2026-07-23 | 243.50 | 233.66 | 4.21% | ❌ | ✅ | ✅ | news |
| 2026-07-22 | 247.59 | 244.85 | 1.12% | ❌ | ❌ | ❌ | news |
| 2026-07-21 | 251.15 | 247.55 | 1.45% | ❌ | ❌ | ❌ | contrarian |

## 🅰️/🅱️ A/B test — ensemble+gates vs raven-style prior+pulse

_Both arms predict the same sessions from the same pre-open snapshot (paired test)._

**Verdict:** Arm B leads on paired MAPE (mean daily delta 0.21% in B's favor); B wins 14/21 decisive days (sign test p=0.1892, not significant).

**B's open prediction (2026-08-18):** close ≈ **258.27** (down, adj -0.3σ, 8 evidence items).

| Rolling metric | A — ensemble+gates | B — raven prior+pulse |
|---|---|---|
| Scored days | 20 | 20 |
| PASS rate (±1%) | 40.00% | 50.00% |
| Directional accuracy | 55.00% | 75.00% |
| MAPE | 1.99% | 1.78% |
| Edge vs baseline | -0.01% | 0.21% |

Paired days: 21; B wins 14/21 decisive; mean daily APE delta (A−B) 0.21%; sign test p = 0.1892.
_This is a research experiment, not financial advice._
