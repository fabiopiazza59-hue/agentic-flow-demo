# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Direction is the core failure**, not magnitude: 26 of 41 misses are directional (63%). The model gets the size roughly right but the sign wrong — it's chasing/lagging rather than anticipating turns.
- **Fails don't beat the baseline.** On almost every miss, ape ≈ or > baseline_ape (e.g. 06-22, 06-25, 06-29, 07-15, 08-19, 09-17). When wrong, we add no value over naive persistence — the ensemble degenerates to "yesterday's price plus noise."
- **Macro is the worst engine**: 12% hit rate, highest MAPE (0.0169), yet it "wins" on several large-error days (07-13, 07-15, 08-19, 08-25). Its weight_hint (0.18) is still too high for that record.
- **Technical is a low-hit, well-calibrated-magnitude strategy** (15% hit) but keeps getting ~0.21 weight — over-trusted.
- **News is the only strategy carrying the system** (31% hit, lowest MAPE) and correctly weighted highest — but it's not enough to offset the rest.

## Unreliable under these conditions
- **High-volatility / gap days.** The big-APE clusters (06-22 ~5%, 07-31 ~13%, 07-30 ~3.9%, 08-19 ~2.9%) are all directional misses under large moves. The model systematically under-reacts to regime breaks.
- **Late-confidence overconfidence.** From late Aug onward confidence rose to 0.54–0.72, but misses persisted at high confidence (08-19 conf 0.62, 09-03 conf 0.62, 09-11 conf 0.54, 08-31 conf 0.72). The 4 flagged overconfident misses understate this — calibration drifted worse as confidence climbed.
- **Macro-led and contrarian-led days** are coin-flip-to-bad; contrarian misses direction repeatedly (06-26, 08-20, 08-24, 09-03, 09-22).

## Fixes to try next
- **Cut macro weight to near-zero** or gate it to explicit macro-event days only; redistribute to news.
- **Add a directional-agreement gate**: when strategies disagree on sign, shrink toward baseline and lower confidence instead of committing.
- **Recalibrate confidence** against realized hit rate — current high-confidence band (0.6+) is not earning its confidence; cap or penalize confidence on high-volatility days.
- **Regime detector for gap/vol days**: widen expected move and de-weight lagging technical/momentum signals during breaks.