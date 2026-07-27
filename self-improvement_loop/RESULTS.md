# 📈 AMZN Daily Close Predictor — Results

_Auto-generated each trading day. PASS = predicted close within ±1% of actual._

## Verdict
❌ **No edge yet** — rolling MAPE 1.44% does not beat the random-walk baseline 1.40%. Keep learning.

**Today's open prediction (2026-07-27):** close ≈ **232.09** (down, confidence 10.00%) vs prior close 232.11.

## Rolling metrics (last 20 scored days)

| Metric | Rolling | All-time |
|---|---|---|
| Scored days | 20 | 30 |
| PASS rate (±1%) | 30.00% | 33.33% |
| Directional accuracy | 45.00% | 43.33% |
| MAPE | 1.44% | 1.68% |
| Baseline MAPE (random walk) | 1.40% | 1.60% |
| Edge (baseline − model) | -0.04% | -0.08% |
| Brier (confidence calib.) | 0.23 | 0.23 |

## Per-strategy scorecards

| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |
|---|---|---|---|---|
| news | 30 | 33.33% | 1.52% | 0.22 |
| technical | 30 | 16.67% | 1.53% | 0.21 |
| contrarian | 30 | 16.67% | 1.64% | 0.20 |
| momentum | 30 | 23.33% | 1.75% | 0.19 |
| macro | 30 | 10.00% | 1.82% | 0.18 |

## Last 20 scored model predictions
_(backfill seed rows are excluded from metrics and this table; they appear only as price-history context on the dashboard chart)_

| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest |
|---|---|---|---|---|---|---|---|
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
| 2026-06-26 | 225.30 | 232.69 | 3.18% | ❌ | ❌ | ❌ | contrarian |

## 🅰️/🅱️ A/B test — ensemble+gates vs raven-style prior+pulse

_Both arms predict the same sessions from the same pre-open snapshot (paired test)._

**Verdict:** ⏳ Only 5 paired scored day(s) — no verdict before 10.

**B's open prediction (2026-07-27):** close ≈ **231.69** (down, adj -0.1σ, 7 evidence items).

| Rolling metric | A — ensemble+gates | B — raven prior+pulse |
|---|---|---|
| Scored days | 20 | 5 |
| PASS rate (±1%) | 30.00% | 40.00% |
| Directional accuracy | 45.00% | 60.00% |
| MAPE | 1.44% | 1.58% |
| Edge vs baseline | -0.04% | 0.15% |

Paired days: 5; B wins 4/5 decisive; mean daily APE delta (A−B) 0.26%; sign test p = 0.375.
_This is a research experiment, not financial advice._
