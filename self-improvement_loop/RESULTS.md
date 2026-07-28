# 📈 AMZN Daily Close Predictor — Results

_Auto-generated each trading day. PASS = predicted close within ±1% of actual._

## Verdict
❌ **No edge yet** — rolling MAPE 1.30% does not beat the random-walk baseline 1.30%. Keep learning.

**Today's open prediction (2026-07-28):** close ≈ **230.05** (down, confidence 32.00%) vs prior close 231.39.

## Rolling metrics (last 20 scored days)

| Metric | Rolling | All-time |
|---|---|---|
| Scored days | 20 | 31 |
| PASS rate (±1%) | 35.00% | 35.48% |
| Directional accuracy | 50.00% | 45.16% |
| MAPE | 1.30% | 1.63% |
| Baseline MAPE (random walk) | 1.30% | 1.55% |
| Edge (baseline − model) | -0.01% | -0.08% |
| Brier (confidence calib.) | 0.26 | 0.25 |

## Per-strategy scorecards

| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |
|---|---|---|---|---|
| technical | 31 | 19.35% | 1.49% | 0.22 |
| news | 31 | 32.26% | 1.50% | 0.22 |
| contrarian | 31 | 16.13% | 1.63% | 0.20 |
| momentum | 31 | 22.58% | 1.70% | 0.19 |
| macro | 31 | 9.68% | 1.79% | 0.18 |

## Last 20 scored model predictions
_(backfill seed rows are excluded from metrics and this table; they appear only as price-history context on the dashboard chart)_

| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest |
|---|---|---|---|---|---|---|---|
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
| 2026-07-09 | 243.87 | 247.04 | 1.28% | ❌ | ✅ | ✅ | news |
| 2026-07-08 | 246.15 | 243.62 | 1.04% | ❌ | ❌ | ❌ | news |
| 2026-07-07 | 244.78 | 245.98 | 0.49% | ✅ | ✅ | ✅ | technical |
| 2026-07-06 | 243.42 | 244.16 | 0.30% | ✅ | ✅ | ✅ | momentum |
| 2026-07-02 | 241.85 | 242.67 | 0.34% | ✅ | ✅ | ✅ | momentum |
| 2026-07-01 | 238.15 | 241.70 | 1.47% | ❌ | ❌ | ❌ | news |
| 2026-06-30 | 240.85 | 238.34 | 1.05% | ❌ | ❌ | ❌ | technical |
| 2026-06-29 | 232.30 | 240.14 | 3.26% | ❌ | ❌ | ❌ | news |

## 🅰️/🅱️ A/B test — ensemble+gates vs raven-style prior+pulse

_Both arms predict the same sessions from the same pre-open snapshot (paired test)._

**Verdict:** ⏳ Only 6 paired scored day(s) — no verdict before 10.

**B's open prediction (2026-07-28):** close ≈ **230.38** (down, adj -0.25σ, 8 evidence items).

| Rolling metric | A — ensemble+gates | B — raven prior+pulse |
|---|---|---|
| Scored days | 20 | 6 |
| PASS rate (±1%) | 35.00% | 50.00% |
| Directional accuracy | 50.00% | 66.67% |
| MAPE | 1.30% | 1.34% |
| Edge vs baseline | -0.01% | 0.15% |

Paired days: 6; B wins 5/6 decisive; mean daily APE delta (A−B) 0.25%; sign test p = 0.2188.
_This is a research experiment, not financial advice._
