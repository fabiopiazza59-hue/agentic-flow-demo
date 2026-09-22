# 📈 AMZN Daily Close Predictor — Results

_Auto-generated each trading day. PASS = predicted close within ±1% of actual._

## Verdict
✅ **Edge confirmed** — rolling MAPE 1.03% beats the random-walk baseline 1.10% by 0.07%.

**Today's open prediction (2026-09-22):** close ≈ **259.42** (up, confidence 52.00%) vs prior close 258.45.

## Rolling metrics (last 20 scored days)

| Metric | Rolling | All-time |
|---|---|---|
| Scored days | 20 | 70 |
| PASS rate (±1%) | 55.00% | 44.29% |
| Directional accuracy | 65.00% | 52.86% |
| MAPE | 1.03% | 1.55% |
| Baseline MAPE (random walk) | 1.10% | 1.54% |
| Edge (baseline − model) | 0.07% | -0.01% |
| Brier (confidence calib.) | 0.27 | 0.26 |

## Per-strategy scorecards

| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |
|---|---|---|---|---|
| news | 70 | 31.43% | 1.35% | 0.23 |
| technical | 70 | 15.71% | 1.46% | 0.21 |
| contrarian | 70 | 17.14% | 1.60% | 0.19 |
| momentum | 70 | 22.86% | 1.61% | 0.19 |
| macro | 70 | 12.86% | 1.68% | 0.18 |

## Last 20 scored model predictions
_(backfill seed rows are excluded from metrics and this table; they appear only as price-history context on the dashboard chart)_

| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest |
|---|---|---|---|---|---|---|---|
| 2026-09-21 | 255.15 | 258.45 | 1.28% | ❌ | ✅ | ✅ | news |
| 2026-09-18 | 251.27 | 253.71 | 0.96% | ✅ | ✅ | ✅ | news |
| 2026-09-17 | 245.10 | 251.19 | 2.42% | ❌ | ❌ | ❌ | news |
| 2026-09-16 | 247.05 | 245.96 | 0.44% | ✅ | ✅ | ✅ | macro |
| 2026-09-15 | 252.45 | 248.42 | 1.62% | ❌ | ✅ | ✅ | momentum |
| 2026-09-14 | 255.60 | 253.54 | 0.81% | ✅ | ✅ | ✅ | news |
| 2026-09-11 | 251.72 | 256.78 | 1.97% | ❌ | ❌ | ❌ | news |
| 2026-09-10 | 251.35 | 251.89 | 0.21% | ✅ | ✅ | ❌ | technical |
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

## 🅰️/🅱️ A/B test — ensemble+gates vs raven-style prior+pulse

_Both arms predict the same sessions from the same pre-open snapshot (paired test)._

**Verdict:** Arm B leads on paired MAPE (mean daily delta 0.08% in B's favor); B wins 26/45 decisive days (sign test p=0.3713, not significant).

**B's open prediction (2026-09-22):** close ≈ **259.11** (up, adj 0.15σ, 8 evidence items).

| Rolling metric | A — ensemble+gates | B — raven prior+pulse |
|---|---|---|
| Scored days | 20 | 20 |
| PASS rate (±1%) | 55.00% | 55.00% |
| Directional accuracy | 65.00% | 65.00% |
| MAPE | 1.03% | 0.99% |
| Edge vs baseline | 0.07% | 0.10% |

Paired days: 45; B wins 26/45 decisive; mean daily APE delta (A−B) 0.08%; sign test p = 0.3713.
_This is a research experiment, not financial advice._
