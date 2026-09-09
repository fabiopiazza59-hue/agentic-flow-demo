# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Direction is the core problem**: 23 of 34 fails are directional misses (68%). The model gets the sign wrong more than the magnitude. Overall directional hit rate across strategies is dismal (13–26%).
- **Baseline routinely beats us on fails**: on most fail days `beats_baseline` is false — we're not just wrong, we're worse than naive persistence. The model adds negative value on the days it matters.
- **Magnitude blowups cluster in high-vol days**: fails carry mean APE 2.55%, with tail events (07-31 at 13.2%, 06-22 at 5.1%, 07-23 at 4.2%) where we're directionally lucky but grossly mis-sized.
- **Confidence is essentially noise**: passes and fails both span 0.1–0.72. High-confidence fails (08-19 @0.62, 08-20 @0.54, 09-03 @0.62, 08-31 @0.72) show confidence isn't tracking accuracy — mild inverse if anything.

## Unreliable under these conditions
- **macro as winning strategy = red flag**: macro hit rate 13%, worst MAPE (0.0167). Nearly every macro-led day (06-12, 07-13, 07-15, 07-30, 08-19) failed. When macro wins the blend, expect a miss.
- **contrarian-led days are coin flips at best** (19.7% hit): 06-26, 07-21, 08-11, 08-20, 09-03 all failed directionally.
- **Large-move / gap days**: whenever baseline_ape > ~2.5% the model fails to keep up (06-15, 06-22, 06-25, 07-15, 07-30, 08-19). We systematically underreact to big directional days.
- **news wins most (16) but still only 26% hit** — best of a weak field, not reliable.

## Fixes to try next
- **Recalibrate or scrap confidence**: current scores don't separate hits from misses. Fit confidence against realized directional accuracy; suppress trades when calibrated confidence is low.
- **Down-weight/gate macro and contrarian**: cut macro weight sharply; only let it lead with corroborating signals. Their weight_hints (0.186/0.194) are still too high given 13–20% hit rates.
- **Add a volatility/gap regime filter**: on high-expected-move days, widen magnitude and defer to baseline unless a strong confirmed signal exists.
- **Attack directional accuracy directly**: model is barely better than a coin on sign — retrain the sign classifier separately from magnitude; a 13–26% hit rate suggests systematic sign inversion worth investigating.