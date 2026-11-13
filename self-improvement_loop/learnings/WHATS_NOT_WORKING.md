# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Magnitude, not just direction, is the bigger problem lately.** Of 61 fails, 32 are directional misses but many "directional hits" still fail on APE (e.g. 10-29, 11-05, 11-09, 10-22) — we get the sign right and still blow the size by 2-5%.
- **Fat-tail blowups cluster.** The worst fails (APE 5-13%: 07-31, 06-22, 10-19, 11-04, 10-30) are large-move days the model systematically under-shoots. Mean fail APE 2.43% vs typical pass ~0.5% — misses are regime events, not noise.
- **We barely beat baseline on fails.** On many fails baseline_ape ≈ our ape (06-15, 06-17, 06-22, 06-29, 08-20, 09-11). On those days the ensemble adds nothing — it's just tracking a bad baseline.
- **Every strategy is sub-33% hit rate.** Best (news, 31%) is still a coin-flip loser. Technical (11%) and macro (13%) are near-useless as *winning* strategies yet still carry ~0.19-0.20 weight.

## Unreliable under these conditions
- **High-volatility / large-gap days:** model compresses the predicted move toward the mean and eats huge APE. This is the dominant failure mode.
- **Macro as winning strategy = red flag:** macro "wins" on several of the worst days (06-08 aside: 07-13, 07-15, 07-30, 08-19, 10-30) — it surfaces precisely when nothing else fits and then misses big.
- **Contrarian in trending/continuation moves:** contrarian wins frequently but fails on sustained directional runs (10-08, 10-15, 10-19, 11-04, 11-11) — classic "fade the trend" error.
- **Confidence is weakly calibrated, newer runs worse.** Only 7 flagged overconfident misses, but recent fails sit at 0.56-0.66 confidence (10-06, 10-08, 10-19, 09-03, 08-19 @0.62) — confidence rose while hit rate didn't. The 0.1-0.3 confidence band actually contains several passes. Confidence is roughly noise.

## Fixes to try next
- **Add a volatility-regime gate:** when realized/implied vol is elevated, widen predicted move and down-weight mean-reverting strategies; stop compressing toward baseline.
- **Cut technical and macro weight hard** (both ~12% hit rate); let news + contrarian dominate, but gate contrarian off during confirmed trends (e.g. 3-day momentum filter).
- **Re-calibrate confidence against outcomes** — current scores don't predict passes; recalibrate so high confidence requires low-vol + strategy agreement.
- **Separate magnitude model from direction model;** most value left on the table is sizing, not sign.
- **Flag "baseline-tie" days** where ensemble ≈ baseline and treat as low-information rather than scoring them as conviction calls.