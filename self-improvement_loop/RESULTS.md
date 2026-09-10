# 📈 AMZN Daily Close Predictor — Results

_Auto-generated each trading day. PASS = predicted close within ±1% of actual._

## Verdict
✅ **Edge confirmed** — rolling MAPE 1.01% beats the random-walk baseline 1.03% by 0.02%.

**Today's open prediction (2026-09-10):** close ≈ **251.35** (down, confidence 62.00%) vs prior close 252.40.

## Rolling metrics (last 20 scored days)

| Metric | Rolling | All-time |
|---|---|---|
| Scored days | 20 | 62 |
| PASS rate (±1%) | 60.00% | 43.55% |
| Directional accuracy | 55.00% | 50.00% |
| MAPE | 1.01% | 1.60% |
| Baseline MAPE (random walk) | 1.03% | 1.56% |
| Edge (baseline − model) | 0.02% | -0.04% |
| Brier (confidence calib.) | 0.27 | 0.25 |

## Per-strategy scorecards

| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |
|---|---|---|---|---|
| news | 62 | 27.42% | 1.41% | 0.22 |
| technical | 62 | 16.13% | 1.45% | 0.21 |
| contrarian | 62 | 19.35% | 1.62% | 0.19 |
| momentum | 62 | 24.19% | 1.64% | 0.19 |
| macro | 62 | 12.90% | 1.66% | 0.19 |

## Last 20 scored model predictions
_(backfill seed rows are excluded from metrics and this table; they appear only as price-history context on the dashboard chart)_

| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest |
|---|---|---|---|---|---|---|---|
| 2026-09-09 | 256.44 | 252.40 | 1.60% | ❌ | ✅ | ✅ | news |
| 2026-09-08 | 258.28 | 256.97 | 0.51% | ✅ | ✅ | ✅ | macro |
| 2026-09-04 | 258.82 | 258.51 | 0.12% | ✅ | ✅ | ✅ | technical |
| 2026-09-03 | 254.70 | 258.90 | 1.62% | ❌ | ❌ | ❌ | contrarian |
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

## 🅰️/🅱️ A/B test — ensemble+gates vs raven-style prior+pulse

_Both arms predict the same sessions from the same pre-open snapshot (paired test)._

**Verdict:** Arm B leads on paired MAPE (mean daily delta 0.08% in B's favor); B wins 20/37 decisive days (sign test p=0.7428, not significant).

**B's open prediction (2026-09-10):** close ≈ **251.79** (down, adj -0.15σ, 8 evidence items).

| Rolling metric | A — ensemble+gates | B — raven prior+pulse |
|---|---|---|
| Scored days | 20 | 20 |
| PASS rate (±1%) | 60.00% | 55.00% |
| Directional accuracy | 55.00% | 55.00% |
| MAPE | 1.01% | 1.09% |
| Edge vs baseline | 0.02% | -0.06% |

Paired days: 37; B wins 20/37 decisive; mean daily APE delta (A−B) 0.08%; sign test p = 0.7428.
_This is a research experiment, not financial advice._
