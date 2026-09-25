# 📈 AMZN Daily Close Predictor — Results

_Auto-generated each trading day. PASS = predicted close within ±1% of actual._

## Verdict
✅ **Edge confirmed** — rolling MAPE 1.08% beats the random-walk baseline 1.18% by 0.10%.

**Today's open prediction (2026-09-25):** close ≈ **249.05** (down, confidence 42.00%) vs prior close 249.38.

## Rolling metrics (last 20 scored days)

| Metric | Rolling | All-time |
|---|---|---|
| Scored days | 20 | 73 |
| PASS rate (±1%) | 50.00% | 43.84% |
| Directional accuracy | 70.00% | 53.42% |
| MAPE | 1.08% | 1.53% |
| Baseline MAPE (random walk) | 1.18% | 1.53% |
| Edge (baseline − model) | 0.10% | -0.00% |
| Brier (confidence calib.) | 0.28 | 0.26 |

## Per-strategy scorecards

| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |
|---|---|---|---|---|
| news | 73 | 31.51% | 1.34% | 0.23 |
| technical | 73 | 15.07% | 1.46% | 0.21 |
| contrarian | 73 | 19.18% | 1.58% | 0.19 |
| momentum | 73 | 21.92% | 1.62% | 0.19 |
| macro | 73 | 12.33% | 1.69% | 0.18 |

## Last 20 scored model predictions
_(backfill seed rows are excluded from metrics and this table; they appear only as price-history context on the dashboard chart)_

| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest |
|---|---|---|---|---|---|---|---|
| 2026-09-24 | 249.36 | 249.38 | 0.01% | ✅ | ✅ | ✅ | contrarian |
| 2026-09-23 | 252.85 | 249.27 | 1.44% | ❌ | ✅ | ✅ | news |
| 2026-09-22 | 259.42 | 254.98 | 1.74% | ❌ | ❌ | ❌ | contrarian |
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

## 🅰️/🅱️ A/B test — ensemble+gates vs raven-style prior+pulse

_Both arms predict the same sessions from the same pre-open snapshot (paired test)._

**Verdict:** Arm B leads on paired MAPE (mean daily delta 0.05% in B's favor); B wins 27/48 decisive days (sign test p=0.4709, not significant).

**B's open prediction (2026-09-25):** close ≈ **250.26** (up, adj 0.2σ, 8 evidence items).

| Rolling metric | A — ensemble+gates | B — raven prior+pulse |
|---|---|---|
| Scored days | 20 | 20 |
| PASS rate (±1%) | 50.00% | 45.00% |
| Directional accuracy | 70.00% | 70.00% |
| MAPE | 1.08% | 1.05% |
| Edge vs baseline | 0.10% | 0.13% |

Paired days: 48; B wins 27/48 decisive; mean daily APE delta (A−B) 0.05%; sign test p = 0.4709.
_This is a research experiment, not financial advice._
