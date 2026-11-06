# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- 56% fail rate (58/103) with mean fail APE 2.45% — misses are large, not marginal. Direction errors (31) slightly outnumber magnitude errors (27), so the model is roughly as bad at *where* as *how much*.
- Tail blowups dominate the damage: APEs of 13.2% (07-31), 6.7% (11-04), 5.5% (06-22), 5.4% (10-30), 5.5% (10-19). These single-day gaps are where the system hemorrhages and several still "beat baseline" (baseline also blew up) — a false comfort.
- Beating baseline ≠ passing. Many fails beat baseline on big-move days (e.g. 10-29, 11-05) — the model tracks direction but undershoots magnitude on large moves.
- **macro** and **technical** are near-useless as winners: 11.6% hit rate each, highest MAPE (1.74% / 1.56%). Yet macro keeps getting selected on the worst days (06-12, 07-13, 07-30, 10-30, 10-09) — it wins the vote precisely when it's wrong.

## Unreliable under these conditions
- **High-volatility / gap days**: whenever the true move is large (baseline_ape >2%), the model systematically underpredicts magnitude. All the big APEs cluster here.
- **macro-led regimes**: macro-winning predictions fail disproportionately and carry the worst errors — it fires on turbulent days and misdirects.
- **technical- and momentum-led calls**: lowest hit rates; momentum notably whipsaws in choppy stretches (06-12, 07-08, 07-25, 09-15).
- **Confidence is miscalibrated upward in the recent run**: late-period confidence drifted to 0.55–0.66 while fail rate stayed high (08-19 conf 0.62 fail, 09-09 conf 0.72 fail, 10-06/10-08 conf ~0.62 fail). Only 7 flagged overconfident misses understates this — confidence rose without accuracy rising.

## Fixes to try next
- Down-weight or gate **macro** and **technical** as *winning* strategies; cap their vote unless corroborated. Lean on **news** (33% hit, lowest MAPE) and **contrarian** (27%).
- Add a volatility regime detector; on high-expected-move days widen magnitude and stop trusting point estimates — this is where losses concentrate.
- Recalibrate confidence against realized hit rate (isotonic/Platt); current 0.5–0.66 band shows no accuracy edge over 0.4 band.
- Stop rewarding "beats_baseline" when both miss; track absolute pass separately so baseline co-blowups don't mask failures.
- Investigate the 07-31 13% outlier (earnings/event?) — likely an unmodeled scheduled catalyst class.