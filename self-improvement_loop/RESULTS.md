# 📈 AMZN Daily Close Predictor — Results

_Auto-generated each trading day. PASS = predicted close within ±1% of actual._

## Verdict
❌ **No edge yet** — rolling MAPE 1.51% does not beat the random-walk baseline 1.47%. Keep learning.

**Today's open prediction (2026-07-21):** close ≈ **251.15** (up, confidence 42.00%) vs prior close 249.99.

## Rolling metrics (last 20 scored days)

| Metric | Rolling | All-time |
|---|---|---|
| Scored days | 20 | 26 |
| PASS rate (±1%) | 40.00% | 38.46% |
| Directional accuracy | 50.00% | 46.15% |
| MAPE | 1.51% | 1.63% |
| Baseline MAPE (random walk) | 1.47% | 1.55% |
| Edge (baseline − model) | -0.05% | -0.08% |
| Brier (confidence calib.) | 0.26 | 0.25 |

## Per-strategy scorecards

| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |
|---|---|---|---|---|
| news | 26 | 30.77% | 1.44% | 0.22 |
| technical | 26 | 19.23% | 1.45% | 0.22 |
| contrarian | 26 | 15.38% | 1.63% | 0.20 |
| momentum | 26 | 23.08% | 1.75% | 0.18 |
| macro | 26 | 11.54% | 1.80% | 0.18 |

## Last 20 scored model predictions
_(backfill seed rows are excluded from metrics and this table; they appear only as price-history context on the dashboard chart)_

| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest |
|---|---|---|---|---|---|---|---|
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
| 2026-06-25 | 235.10 | 227.01 | 3.56% | ❌ | ❌ | ❌ | momentum |
| 2026-06-24 | 234.74 | 234.27 | 0.20% | ✅ | ✅ | ❌ | technical |
| 2026-06-23 | 232.95 | 234.11 | 0.50% | ✅ | ✅ | ✅ | news |
| 2026-06-22 | 244.62 | 232.79 | 5.08% | ❌ | ❌ | ❌ | news |

## 🅰️/🅱️ A/B test — ensemble+gates vs raven-style prior+pulse

_Both arms predict the same sessions from the same pre-open snapshot (paired test)._

**Verdict:** ⏳ Only 1 paired scored day(s) — no verdict before 10.

**B's open prediction (2026-07-21):** close ≈ **250.72** (up, adj 0.15σ, 8 evidence items).

| Rolling metric | A — ensemble+gates | B — raven prior+pulse |
|---|---|---|
| Scored days | 20 | 1 |
| PASS rate (±1%) | 40.00% | 0.00% |
| Directional accuracy | 50.00% | 0.00% |
| MAPE | 1.51% | 1.10% |
| Edge vs baseline | -0.05% | 0.00% |

Paired days: 1; B wins 1/1 decisive; mean daily APE delta (A−B) 0.13%; sign test p = 1.0.
_This is a research experiment, not financial advice._
