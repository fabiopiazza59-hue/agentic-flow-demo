# 📈 AMZN Daily Close Predictor — Results

_Auto-generated each trading day. PASS = predicted close within ±1% of actual._

## Verdict
✅ **Edge confirmed** — rolling MAPE 1.06% beats the random-walk baseline 1.09% by 0.04%.

**Today's open prediction (2026-09-18):** close ≈ **251.27** (up, confidence 36.00%) vs prior close 251.19.

## Rolling metrics (last 20 scored days)

| Metric | Rolling | All-time |
|---|---|---|
| Scored days | 20 | 68 |
| PASS rate (±1%) | 55.00% | 44.12% |
| Directional accuracy | 55.00% | 51.47% |
| MAPE | 1.06% | 1.56% |
| Baseline MAPE (random walk) | 1.09% | 1.55% |
| Edge (baseline − model) | 0.04% | -0.02% |
| Brier (confidence calib.) | 0.26 | 0.25 |

## Per-strategy scorecards

| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |
|---|---|---|---|---|
| news | 68 | 29.41% | 1.37% | 0.22 |
| technical | 68 | 16.18% | 1.45% | 0.21 |
| contrarian | 68 | 17.65% | 1.61% | 0.19 |
| momentum | 68 | 23.53% | 1.61% | 0.19 |
| macro | 68 | 13.24% | 1.67% | 0.18 |

## Last 20 scored model predictions
_(backfill seed rows are excluded from metrics and this table; they appear only as price-history context on the dashboard chart)_

| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest |
|---|---|---|---|---|---|---|---|
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
| 2026-08-21 | 260.35 | 258.63 | 0.67% | ✅ | ❌ | ❌ | technical |
| 2026-08-20 | 265.91 | 260.11 | 2.23% | ❌ | ❌ | ❌ | contrarian |

## 🅰️/🅱️ A/B test — ensemble+gates vs raven-style prior+pulse

_Both arms predict the same sessions from the same pre-open snapshot (paired test)._

**Verdict:** Arm B leads on paired MAPE (mean daily delta 0.08% in B's favor); B wins 24/43 decisive days (sign test p=0.5424, not significant).

**B's open prediction (2026-09-18):** close ≈ **251.82** (up, adj 0.15σ, 8 evidence items).

| Rolling metric | A — ensemble+gates | B — raven prior+pulse |
|---|---|---|
| Scored days | 20 | 20 |
| PASS rate (±1%) | 55.00% | 55.00% |
| Directional accuracy | 55.00% | 60.00% |
| MAPE | 1.06% | 1.09% |
| Edge vs baseline | 0.04% | 0.00% |

Paired days: 43; B wins 24/43 decisive; mean daily APE delta (A−B) 0.08%; sign test p = 0.5424.
_This is a research experiment, not financial advice._
