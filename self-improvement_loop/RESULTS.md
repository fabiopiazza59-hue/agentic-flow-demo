# 📈 AMZN Daily Close Predictor — Results

_Auto-generated each trading day. PASS = predicted close within ±1% of actual._

## Verdict
❌ **No edge yet** — rolling MAPE 2.02% does not beat the random-walk baseline 2.01%. Keep learning.

**Today's open prediction (2026-08-07):** close ≈ **271.81** (down, confidence 50.00%) vs prior close 272.26.

## Rolling metrics (last 20 scored days)

| Metric | Rolling | All-time |
|---|---|---|
| Scored days | 20 | 39 |
| PASS rate (±1%) | 35.00% | 35.90% |
| Directional accuracy | 60.00% | 51.28% |
| MAPE | 2.02% | 1.90% |
| Baseline MAPE (random walk) | 2.01% | 1.84% |
| Edge (baseline − model) | -0.02% | -0.06% |
| Brier (confidence calib.) | 0.26 | 0.25 |

## Per-strategy scorecards

| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |
|---|---|---|---|---|
| news | 39 | 28.21% | 1.60% | 0.23 |
| technical | 39 | 15.38% | 1.75% | 0.21 |
| contrarian | 39 | 17.95% | 1.88% | 0.19 |
| momentum | 39 | 25.64% | 1.94% | 0.19 |
| macro | 39 | 12.82% | 1.97% | 0.18 |

## Last 20 scored model predictions
_(backfill seed rows are excluded from metrics and this table; they appear only as price-history context on the dashboard chart)_

| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest |
|---|---|---|---|---|---|---|---|
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
| 2026-07-20 | 246.90 | 249.99 | 1.24% | ❌ | ❌ | ❌ | news |
| 2026-07-17 | 248.60 | 247.23 | 0.55% | ✅ | ✅ | ✅ | technical |
| 2026-07-16 | 253.90 | 249.89 | 1.60% | ❌ | ✅ | ✅ | contrarian |
| 2026-07-15 | 247.35 | 254.96 | 2.98% | ❌ | ❌ | ❌ | macro |
| 2026-07-14 | 247.72 | 247.49 | 0.09% | ✅ | ✅ | ❌ | momentum |
| 2026-07-13 | 243.85 | 247.31 | 1.40% | ❌ | ❌ | ❌ | macro |
| 2026-07-10 | 246.85 | 245.34 | 0.62% | ✅ | ✅ | ✅ | contrarian |

## 🅰️/🅱️ A/B test — ensemble+gates vs raven-style prior+pulse

_Both arms predict the same sessions from the same pre-open snapshot (paired test)._

**Verdict:** Arm B leads on paired MAPE (mean daily delta 0.30% in B's favor); B wins 11/14 decisive days (sign test p=0.0574, not significant).

**B's open prediction (2026-08-07):** close ≈ **273.32** (up, adj 0.1σ, 8 evidence items).

| Rolling metric | A — ensemble+gates | B — raven prior+pulse |
|---|---|---|
| Scored days | 20 | 14 |
| PASS rate (±1%) | 35.00% | 50.00% |
| Directional accuracy | 60.00% | 71.43% |
| MAPE | 2.02% | 2.07% |
| Edge vs baseline | -0.02% | 0.25% |

Paired days: 14; B wins 11/14 decisive; mean daily APE delta (A−B) 0.30%; sign test p = 0.0574.
_This is a research experiment, not financial advice._
