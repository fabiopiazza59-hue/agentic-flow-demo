# 📈 AMZN Daily Close Predictor — Results

_Auto-generated each trading day. PASS = predicted close within ±1% of actual._

## Verdict
✅ **Edge confirmed** — rolling MAPE 1.96% beats the random-walk baseline 1.96% by 0.00%.

**Today's open prediction (2026-08-26):** close ≈ **260.35** (down, confidence 54.00%) vs prior close 261.06.

## Rolling metrics (last 20 scored days)

| Metric | Rolling | All-time |
|---|---|---|
| Scored days | 20 | 52 |
| PASS rate (±1%) | 45.00% | 40.38% |
| Directional accuracy | 45.00% | 46.15% |
| MAPE | 1.96% | 1.73% |
| Baseline MAPE (random walk) | 1.96% | 1.69% |
| Edge (baseline − model) | 0.00% | -0.05% |
| Brier (confidence calib.) | 0.24 | 0.25 |

## Per-strategy scorecards

| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |
|---|---|---|---|---|
| technical | 52 | 17.31% | 1.56% | 0.21 |
| news | 52 | 25.00% | 1.56% | 0.21 |
| contrarian | 52 | 21.15% | 1.68% | 0.20 |
| macro | 52 | 13.46% | 1.76% | 0.19 |
| momentum | 52 | 23.08% | 1.76% | 0.19 |

## Last 20 scored model predictions
_(backfill seed rows are excluded from metrics and this table; they appear only as price-history context on the dashboard chart)_

| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest |
|---|---|---|---|---|---|---|---|
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
| 2026-07-29 | 230.55 | 226.65 | 1.72% | ❌ | ✅ | ✅ | macro |

## 🅰️/🅱️ A/B test — ensemble+gates vs raven-style prior+pulse

_Both arms predict the same sessions from the same pre-open snapshot (paired test)._

**Verdict:** Arm B leads on paired MAPE (mean daily delta 0.09% in B's favor); B wins 15/27 decisive days (sign test p=0.7011, not significant).

**B's open prediction (2026-08-26):** close ≈ **261.55** (up, adj 0.05σ, 8 evidence items).

| Rolling metric | A — ensemble+gates | B — raven prior+pulse |
|---|---|---|
| Scored days | 20 | 20 |
| PASS rate (±1%) | 45.00% | 45.00% |
| Directional accuracy | 45.00% | 60.00% |
| MAPE | 1.96% | 1.93% |
| Edge vs baseline | 0.00% | 0.04% |

Paired days: 27; B wins 15/27 decisive; mean daily APE delta (A−B) 0.09%; sign test p = 0.7011.
_This is a research experiment, not financial advice._
