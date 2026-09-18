# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Direction is the core failure**: 25 of 38 misses are directional, not magnitude. The model gets the size roughly right but bets the wrong way — a sign-prediction problem, not a scaling problem.
- **Baseline dominance**: on failure days the model loses to a naive baseline far too often. Blending 5 mediocre strategies (all hit rates 13–29%) produces mush that can't beat persistence.
- **News-led days are a trap**: `news` is the most-frequent winning strategy on losing days (e.g. 06-15, 06-22, 06-25, 06-29, 07-15, 09-11, 09-17). It has the best relative hit rate but still misses direction constantly on big-move days.
- **Big moves are catastrophic**: 07-31 (13.2% APE) and clusters of 3–5% APE days (06-17, 06-22, 06-25, 07-15, 07-30, 08-19) show the model has no handle on gap/event days.
- **Confidence is creeping up while accuracy isn't**: late-window predictions carry 0.5–0.72 confidence but still fail (08-19 @0.62, 08-31 @0.72, 09-09 @0.72, 09-17 @0.54). Overconfident misses are rising, not falling.

## Unreliable under these conditions
- **High-volatility / large-move days** (>2.5% baseline move): near-universal directional miss.
- **`macro` as winning strategy**: 13% hit rate, worst MAPE — actively harmful, especially when macro weight is tiny (07-13, 07-15, 08-19 all failed with macro "winning" at ~7% weight, meaning it won by default among losers).
- **`technical` and `contrarian`**: 16–18% hit rates; unreliable standalone, no clear regime where they earn their weight.
- **Confidence ≥0.5 regime**: cluster of misses at elevated confidence (07-13, 08-07, 08-19, 08-20, 09-03, 09-11, 09-17) — calibration is broken on the upper end.

## Fixes to try next
- Reframe as **direction-first**: separate a sign classifier from a magnitude regressor; most losses are sign errors.
- **Cut or heavily down-weight macro and contrarian**; they drag MAPE and rarely hit.
- **Detect high-vol/event days** (earnings, gaps, wide baseline moves) and either widen intervals or defer to persistence.
- **Recalibrate confidence**: current 0.5–0.72 band has no edge; cap confidence and re-fit against realized hit rate.
- **Stop treating "winning_strategy" as signal** when it wins by default among losers — require the winner to beat baseline before trusting it.