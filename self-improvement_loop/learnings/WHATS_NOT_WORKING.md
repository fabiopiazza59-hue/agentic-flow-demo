# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Direction is the core failure**: 19 of 28 misses are directional (68%), not magnitude. The model gets the price *level* close but the *sign* of the move wrong. It's essentially predicting "flat/persistence" and getting run over on real moves.
- Fails **beat baseline barely or not at all** on most losses — on many misses (e.g., 06-25, 06-26, 07-13, 07-20, 07-21, 07-24) APE is worse than the naive baseline. We're adding negative value on those days.
- **News-led calls dominate the loss column**: news is the winning strategy on ~9 of 28 misses despite being the highest-weighted strategy (0.226 hint). Highest weight, mediocre 28% hit rate.
- The 07-31 blowup (APE 13%) is a real event/gap the model completely under-scoped — magnitude collapse on a large gap.

## Unreliable under these conditions
- **High-magnitude / gap days** (baseline_ape >3%): model consistently misses direction AND magnitude (06-15, 06-22, 06-25, 07-15, 07-23, 07-30, 07-31). It cannot handle regime breaks or news gaps.
- **When macro or technical is the winning strategy**: macro hit rate 11.6%, technical 14% — both near-useless. Macro-led days (07-13, 07-15, 07-30) are almost all fails.
- **Momentum during reversals**: momentum-led calls fail when the trend flips (06-12, 06-25, 07-08, 07-24) — classic momentum-chasing into turns.
- **Confidence is NOT the problem**: 0 overconfident misses, and confidence is generally low (0.1–0.55). If anything the model is *under*-confident on wins (07-27 hit at conf 0.1). Calibration is flat/uninformative — confidence carries no signal.

## Fixes to try next
- Treat this as a **directional problem first**: add a dedicated up/down classifier and stop optimizing pure APE, which rewards the flat-prediction bias.
- **Down-weight or gate macro and technical** (hit rates 12–14%); redistribute toward news/momentum, but cap news since it leads many big misses.
- **Regime detector for high-volatility/gap days**: widen intervals and suppress momentum on detected reversals; don't let momentum extrapolate into turns.
- **Rebuild confidence**: current scores are noise (uncorrelated with outcomes). Recalibrate against realized direction; a flat 0.4 signal is useless for sizing.
- Add explicit **event/earnings flag** to avoid 07-31-style magnitude blowups.