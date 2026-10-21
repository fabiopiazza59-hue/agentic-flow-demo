# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Directional misses dominate**: 30 of 51 fails are wrong-direction, not just magnitude. The model isn't sized wrong — it's pointed wrong. This is the core problem.
- **Baseline-matching behavior**: on large-move days (ape 3-13%), predictions track baseline and miss together (06-15, 06-22, 07-15, 07-31, 10-19). The ensemble adds no edge when it matters most — it just mirrors naive persistence.
- **macro strategy is dead weight**: hit_rate 0.12, worst MAPE (0.0166), yet still carries ~0.18 weight. When macro "wins" the vote it frequently fails (06-12 region, 07-13, 07-15, 07-30, 08-19).
- **technical also weak**: hit_rate 0.13 but still ~0.21 weight — second-highest allocation to a near-coinflip-loser.
- **news is the only real signal**: hit_rate 0.31, lowest MAPE. Everything else is below 0.26.

## Unreliable under these conditions
- **High-volatility / gap days** (>2.5% true moves): consistent directional failure and APE blowouts. Mean fail APE 2.33% vs tiny win APEs shows fails cluster on big days.
- **Confidence is miscalibrated upward over time**: recent block (09-28→10-19) runs confidence 0.53-0.66 while failing repeatedly (10-06, 10-08, 10-15, 10-19). Confidence rose but accuracy didn't — 7 overconfident misses concentrated late.
- **contrarian in trending moves**: wins on some reversals but fails hard when the move continues (10-15, 10-19 at conf ~0.57).
- **macro-led days in any regime**: unreliable regardless of market condition.

## Fixes to try next
- **Cut macro and technical weights hard** (toward ~0.08 floor); reallocate to news (proven best) and contrarian.
- **Decouple confidence from recency drift** — recalibrate against realized hit-rate; current high-conf late predictions are not earning it.
- **Add a volatility regime gate**: on expected large-move/gap days, stop mirroring baseline — widen and lean on news only.
- **Diagnose directional sign logic** — 59% of fails are direction, so a sign-flip bias (likely from contrarian/technical) needs isolation.
- **Track strategy hit-rate conditioned on volatility bucket**, not just overall, before re-weighting.