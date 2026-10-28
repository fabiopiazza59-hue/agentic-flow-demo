# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Direction is the bigger problem than magnitude**: 30 of 53 fails are directional misses (57%). The model is frequently picking the wrong sign, not just the wrong size.
- **Baseline beats us on failures**: mean fail APE 2.34% vs frequent cases where baseline_ape is lower (e.g. 2026-10-15, 2026-09-02, 2026-08-06). When we miss, we often also lose to the naive baseline — no edge on hard days.
- **Macro is the weakest winning strategy** (hit_rate 11%, highest MAPE 1.66%) yet keeps getting selected as "winner" on fails (2026-07-13, 07-15, 08-19, 10-09). It wins when it shouldn't.
- **Technical also unreliable** (hit_rate 12.5%) but still carries ~20% weight and the 2nd-highest weight_hint.
- **Big-move days are blind spots**: largest APEs cluster on volatile sessions (07-31 at 13.2%, 10-19 at 5.5%, 06-22 at 5.1%, 10-22 at 4.5%) — model stays anchored near prior close and misses regime shifts.

## Unreliable under these conditions
- **High-volatility / gap days**: errors blow out 3–13% when the real move is large; model systematically under-reacts in magnitude AND gets direction wrong.
- **Macro- or technical-led days**: whenever macro or technical is the winning strategy, failure rate is high — these should rarely dominate the blend.
- **Rising confidence, flat accuracy (miscalibration)**: later window confidence drifted up to 0.6–0.72 while still failing (08-19 conf .62 fail, 08-31 .72 fail, 09-09 .72 fail, 10-06 .66 fail, 10-08 .62 fail). 7 overconfident misses — confidence is not predictive of correctness.
- **Contrarian in trending stretches**: contrarian is selected on many fails (10-08, 10-15, 10-19) — it fades moves that keep going.

## Fixes to try next
- **Down-weight or gate macro and technical**; cut their blend weight hard (both <0.1) until hit-rate recovers — lean on news (32% hit) and contrarian (26%).
- **Add a volatility regime filter**: when expected/realized range is high, widen predicted magnitude and suppress contrarian's mean-reversion bias.
- **Recalibrate confidence**: current scores above 0.5 carry no accuracy signal. Refit confidence against realized hit-rate and cap it until calibration (Brier/reliability) improves.
- **Build a direction-first sub-model**: since 57% of fails are sign errors, score direction separately and only trust magnitude when direction confidence is high.
- **Add a baseline-fallback guard**: if the blend disagrees sharply with naive persistence on low-signal days, defer toward baseline to stop losing to it.