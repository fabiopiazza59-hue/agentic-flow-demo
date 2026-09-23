# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Directional errors dominate**: 26 of 40 fails (65%) are wrong-direction, not just magnitude. The system is bad at calling *which way*, not merely by how much. Sign prediction is the core defect.
- **Barely beating baseline**: on most fails APE ≈ baseline_ape (e.g. 07-15, 07-20, 08-20, 09-11). The blended model adds little edge — it's tracking a naive baseline and inheriting its misses.
- **Fat-tail blowups**: occasional catastrophic days (07-31 APE 13.2%, 06-22 5.1%, 07-23 4.2%, 07-30 3.9%) — likely earnings/gap/macro-shock days the model has no mechanism to flag.
- **macro** is the single worst strategy (hit_rate 0.127, highest MAPE 0.0169) yet still wins days (06-08, 07-13, 07-15, 08-19) — when macro "wins," it usually loses.

## Unreliable under these conditions
- **High-volatility / large-move days**: every large baseline_ape day is also a fail. The model compresses toward small moves and misses direction on big ones.
- **macro- and momentum-led days**: macro 12.7% hit, momentum 22.5%, technical 15.5% — all weak. Only **news** (31% hit, lowest MAPE) is marginally reliable.
- **Confidence is miscalibrated on the upside**: high-confidence fails cluster late — 08-19 (conf 0.62), 08-31 (0.72), 09-03 (0.62), 09-09 (0.72), 09-11 (0.54), 09-17 (0.54). Rising confidence since August is NOT matched by accuracy; recent stretch (09-11→09-22) is 4 fails of 5.
- Note many misses are still directional_hit=true but fail on magnitude — magnitude sizing is systematically off on volatile days.

## Fixes to try next
- **Cut macro weight to near-zero**; it drags MAPE and hit rate. Lean toward news; stress-test whether news edge is real or lucky.
- **Add a volatility/event gate**: on high-expected-move days (earnings, macro prints), widen intervals or abstain rather than predict a small move.
- **Recalibrate confidence** — current high-confidence buckets (0.6–0.72) show no accuracy lift; force confidence to track realized hit-rate, penalize recent overconfidence.
- **Attack directional error directly**: add a dedicated sign classifier; current magnitude blend guesses direction poorly.
- **Investigate the recent Sept regime shift** — accuracy degraded while confidence rose; check for drift/stale weights.