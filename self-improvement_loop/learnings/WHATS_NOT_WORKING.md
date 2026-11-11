# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- Nearly even split: 31 directional misses vs 29 magnitude misses — the model fails on *both* axes, not one. It can't be patched by a single sign-flip fix.
- Failures cluster around large-move days. The biggest APEs (0.13 on 07-31, 0.067 on 11-04, 0.055 on 10-19, 0.052 on 10-30) are all days where the move far exceeded the model's muted, mean-reverting estimate. Magnitude is chronically under-sized on volatile days.
- On many fails the model still beat baseline (e.g. 10-29, 11-05, 11-09) — i.e. it got direction right but undershot the magnitude. The pass/fail threshold is being tripped by magnitude error on correct-direction calls.
- `macro` and `technical` are dead weight: macro 12% hit rate / 0.0172 MAPE, technical 11% / 0.0155. Yet macro wins the blend on several of the worst days (07-30, 08-19, 10-30) — it surfaces exactly when it's least reliable.

## Unreliable under these conditions
- High-volatility / event days (earnings-like gaps): every strategy under-predicts magnitude; macro and contrarian lead the blend into the biggest misses (10-19, 11-04, 10-08, 06-22).
- `contrarian` when fading a real trend: it logs good hit-rate overall (26%) but owns several large-APE fails (10-19 0.055, 11-04 0.067, 10-15, 06-26) — it's catching falling knives.
- Confidence is modestly miscalibrated, not catastrophically: only 7 overconfident misses, but note 08-19 (conf 0.62, fail), 08-31/09-09 (conf 0.72, fail), 09-03 (0.62, fail). Rising recent confidence (0.5–0.66) does NOT track accuracy — calibration is flat/noisy above ~0.5.
- `news` is the single best strategy (32% hit, lowest MAPE) yet still only wins ~1/3 of the time it's chosen.

## Fixes to try next
- Drop or heavily down-weight macro and technical; shift that mass to news/contrarian. They add noise, not signal.
- Add a volatility regime detector; on high-vol days widen magnitude and suppress contrarian fading.
- Scale predicted magnitude up when recent realized vol is high — systematic undershoot is the dominant error.
- Recalibrate confidence against realized hit-rate; current 0.6–0.7 confidence shows no accuracy edge.
- Gate contrarian: block it when a strong multi-day trend is intact.