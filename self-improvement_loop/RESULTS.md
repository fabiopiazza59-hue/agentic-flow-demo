# 📈 AMZN Daily Close Predictor — Results

_Auto-generated each trading day. PASS = predicted close within ±1% of actual._

## Verdict
✅ **Edge confirmed** — rolling MAPE 1.89% beats the random-walk baseline 1.91% by 0.01%.

**Today's open prediction (2026-08-24):** close ≈ **258.49** (down, confidence 36.00%) vs prior close 258.63.

## Rolling metrics (last 20 scored days)

| Metric | Rolling | All-time |
|---|---|---|
| Scored days | 20 | 50 |
| PASS rate (±1%) | 50.00% | 40.00% |
| Directional accuracy | 55.00% | 48.00% |
| MAPE | 1.89% | 1.76% |
| Baseline MAPE (random walk) | 1.91% | 1.72% |
| Edge (baseline − model) | 0.01% | -0.04% |
| Brier (confidence calib.) | 0.28 | 0.25 |

## Per-strategy scorecards

| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |
|---|---|---|---|---|
| news | 50 | 26.00% | 1.57% | 0.22 |
| technical | 50 | 18.00% | 1.61% | 0.21 |
| contrarian | 50 | 20.00% | 1.72% | 0.20 |
| momentum | 50 | 24.00% | 1.78% | 0.19 |
| macro | 50 | 12.00% | 1.82% | 0.19 |

## Last 20 scored model predictions
_(backfill seed rows are excluded from metrics and this table; they appear only as price-history context on the dashboard chart)_

| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest |
|---|---|---|---|---|---|---|---|
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
| 2026-07-29 | 230.55 | 226.65 | 1.72% | ❌ | ✅ | ✅ | macro |
| 2026-07-28 | 230.05 | 230.86 | 0.35% | ✅ | ✅ | ❌ | momentum |
| 2026-07-27 | 232.09 | 231.39 | 0.30% | ✅ | ✅ | ✅ | technical |

## 🅰️/🅱️ A/B test — ensemble+gates vs raven-style prior+pulse

_Both arms predict the same sessions from the same pre-open snapshot (paired test)._

**Verdict:** Arm B leads on paired MAPE (mean daily delta 0.12% in B's favor); B wins 15/25 decisive days (sign test p=0.4244, not significant).

**B's open prediction (2026-08-24):** close ≈ **257.17** (down, adj -0.15σ, 7 evidence items).

| Rolling metric | A — ensemble+gates | B — raven prior+pulse |
|---|---|---|
| Scored days | 20 | 20 |
| PASS rate (±1%) | 50.00% | 50.00% |
| Directional accuracy | 55.00% | 70.00% |
| MAPE | 1.89% | 1.81% |
| Edge vs baseline | 0.01% | 0.10% |

Paired days: 25; B wins 15/25 decisive; mean daily APE delta (A−B) 0.12%; sign test p = 0.4244.
_This is a research experiment, not financial advice._
