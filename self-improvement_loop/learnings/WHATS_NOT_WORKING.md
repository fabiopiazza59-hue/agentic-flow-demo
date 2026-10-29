# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Directional misses dominate failures: 30 of 53 (57%).** The model is getting the *sign* of the move wrong more than the magnitude. This is the core problem — it's not a calibration-of-size issue, it's that we can't call direction.
- **Fails barely beat baseline.** On most fail days baseline_ape ≈ model_ape (e.g. 06-17, 06-22, 06-29, 09-11, 10-08). The ensemble adds little over naive persistence on hard days; it mostly loses where the naive guess also loses.
- **Big-move days are a bloodbath.** Every large-APE event (07-31 at 13.2%, 10-19 at 5.5%, 06-22 at 5.1%, 10-22 at 4.5%, 07-23 at 4.2%) is a miss. Magnitude errors cluster entirely on gap/high-volatility days — the model systematically under-shoots range.
- **macro and technical are dead weight.** macro hit_rate 11%, technical 12% — both worse than contrarian (26%) and news (33%), yet they still carry ~18-20% weight each and win days they shouldn't (06-12, 07-13, 08-19 all macro/momentum fails).

## Unreliable under these conditions
- **High-volatility / earnings-adjacent days** (late July, mid-Oct): magnitude collapses, directional coin-flip.
- **When macro or technical is the winning strategy:** disproportionately present on fails (06-12, 07-13, 07-15, 08-19, 10-09).
- **Recent overconfidence creep.** Since ~08-19 confidence sits 0.5–0.72 while fail rate stays high; 08-19 (conf 0.62), 08-20 (0.54), 09-03 (0.62), 10-08 (0.62), 10-19 (0.56) are confident directional misses. 7 overconfident misses — confidence is rising faster than accuracy.

## Fixes to try next
- **Cut macro + technical weight hard** (toward 0.05–0.10 each); reallocate to news/contrarian which actually carry the signal.
- **Add a volatility regime gate:** when expected range is large, widen magnitude and drop confidence — don't under-predict the move.
- **Recalibrate confidence** against realized hit rate; current 0.5–0.7 band is not earned. Shrink toward 0.4–0.5 until accuracy supports it.
- **Attack the directional problem directly:** the ensemble's sign is near-random on fails — test a separate direction classifier rather than averaging five weak point-estimates.