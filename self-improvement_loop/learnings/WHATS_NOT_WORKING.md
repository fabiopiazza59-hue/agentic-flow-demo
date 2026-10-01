# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Direction is the core failure, not magnitude.** 27 of 42 misses (64%) are directional. The model gets the move size roughly right but points the wrong way — a sign routing problem, not a calibration-of-size problem.
- **Baseline is beating us too often.** Many fails also have `beats_baseline=false` (e.g. 06-12, 06-15, 06-25, 07-13, 08-19, 09-22). On a meaningful chunk of days a naive persistence baseline would have done as well or better.
- **Macro is a net drag.** 11.7% hit rate, highest MAPE (0.0166). Every day macro is the winning strategy it tends to coincide with a fail (06-12 context, 07-13, 07-15, 07-30, 08-19). It is actively hurting.
- **Technical is unreliable despite high weight.** 14% hit rate but carries ~0.21 weight — second-highest allocation to a near-worst performer.
- **Big-move days blow up.** The large APEs (07-31 at 0.13, 06-22, 07-23, 06-17 ~0.037) cluster around apparent gap/event days the model can't anticipate.

## Unreliable under these conditions
- **High-volatility / event days:** APE spikes 3–13% when the actual move is large; the ensemble reverts to small moves and misses both direction and size.
- **When macro or technical is the deciding strategy:** disproportionately correlated with fails and baseline losses.
- **Mild overconfidence creeping in late:** overconfident misses are only 4 total, but recent fails carry rising confidence (08-19 conf 0.62, 08-31 conf 0.72, 09-09 conf 0.72 all failed). High-confidence days are no longer reliably better.
- **Choppy sideways days:** repeated ~1–2% directional misses (09-11, 09-22, 09-28) where the model picks a side on noise.

## Fixes to try next
- **Cut macro weight toward zero** and reallocate to news (only strategy >0.30 hit rate); demote technical weight to match its hit rate.
- **Add a directional gate:** when strategies disagree on sign, widen toward baseline/flat instead of committing — most losses are sign errors.
- **Volatility regime detector:** on high-expected-move days, stop damping toward small moves; size predictions up or abstain.
- **Recalibrate confidence** on recent window — current high-confidence predictions (0.6–0.72) are failing; confidence should track realized hit rate, not grow unanchored.
- **Track beats_baseline as a first-class gate**; if the ensemble can't beat persistence in backtest for a regime, default to baseline there.