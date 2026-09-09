# 📈 AMZN Daily Close Predictor — Results

_Auto-generated each trading day. PASS = predicted close within ±1% of actual._

## Verdict
❌ **No edge yet** — rolling MAPE 1.96% is nominally ahead of the random-walk baseline 1.97% by 0.02%, but the difference is not distinguishable from noise (paired over 20 pre-open days: mean 0.02%, 90% CI [-0.09%, 0.12%], beat baseline on 9/20 decisive days, sign test p=0.8238).

**Today's open prediction (2026-09-08):** close ≈ **258.28** (down, confidence 62.00%) vs prior close 258.51.

## Rolling metrics (last 20 scored days)

| Metric | Rolling (pre-open) | Rolling (all rows) | All-time |
|---|---|---|---|
| Scored days | 20 | 20 | 60 |
| PASS rate (±1%) | 45.00% | 55.00% | 43.33% |
| Directional accuracy | 50.00% | 45.00% | 48.33% |
| MAPE | 1.96% | 1.08% | 1.61% |
| Baseline MAPE (random walk) | 1.97% | 1.08% | 1.57% |
| Edge (baseline − model) | 0.02% | 0.00% | -0.04% |
| Brier (confidence calib.) | 0.21 | 0.25 | 0.25 |

### Integrity & mechanism

- **26 scored row(s) were created after their session opened** and are excluded from the pre-open column. They saw part of the tape they forecast, so they cannot carry the verdict. New post-open rows are refused (`--allow-late` to override).
- **Gate effect**: not measurable yet — accrues from the first run that records an ungated counterfactual (`predicted_close_raw`) on each row.

## Per-strategy scorecards

| Strategy | Obs | Win rate (closest) | MAPE | Weight hint |
|---|---|---|---|---|
| news | 60 | 26.67% | 1.45% | 0.22 |
| technical | 60 | 16.67% | 1.47% | 0.21 |
| contrarian | 60 | 20.00% | 1.61% | 0.20 |
| momentum | 60 | 25.00% | 1.66% | 0.19 |
| macro | 60 | 11.67% | 1.70% | 0.19 |

## Last 20 scored model predictions
_(backfill seed rows are excluded from metrics and this table; they appear only as price-history context on the dashboard chart)_

| Date | Predicted | Actual | APE | PASS | Dir hit | Beat baseline | Closest |
|---|---|---|---|---|---|---|---|
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
| 2026-08-11 | 278.15 | 272.27 | 2.16% | ❌ | ❌ | ❌ | contrarian |
| 2026-08-10 | 274.29 | 278.09 | 1.37% | ❌ | ❌ | ❌ | news |

## 🅰️/🅱️ A/B test — ensemble+gates vs raven-style prior+pulse

_Both arms predict the same sessions from the same pre-open snapshot (paired test)._

**Verdict:** Arm B leads on paired MAPE (mean daily delta 0.08% in B's favor); B wins 19/35 decisive days (sign test p=0.7359, not significant).

**B's open prediction (2026-09-08):** close ≈ **258.51** (down, adj 0.0σ, 0 evidence items).

| Rolling metric | A — ensemble+gates | B — raven prior+pulse |
|---|---|---|
| Scored days | 20 | 20 |
| PASS rate (±1%) | 55.00% | 50.00% |
| Directional accuracy | 45.00% | 50.00% |
| MAPE | 1.08% | 1.18% |
| Edge vs baseline | 0.00% | -0.10% |

Paired days: 35; B wins 19/35 decisive; mean daily APE delta (A−B) 0.08%; sign test p = 0.7359.
_This is a research experiment, not financial advice._
