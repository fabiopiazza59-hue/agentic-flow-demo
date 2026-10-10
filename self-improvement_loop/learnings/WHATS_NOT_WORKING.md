# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Direction is the core problem, not magnitude.** 28 of 46 fails (61%) are directional misses. The model is often close in size but on the wrong side — a sign flip issue, not a scaling issue.
- **Large-move days blow up.** Fails cluster on high-APE days (2026-06-22 ~5%, 2026-07-31 ~13%, 2026-07-30 ~3.9%, 2026-06-17 ~3.7%). On big gaps the model anchors to prior close and gets run over.
- **macro is the weakest strategy** (11.9% hit rate, highest MAPE 0.0165) yet still wins the ensemble on fail days (07-13, 07-15, 08-19, 10-09). When macro "wins," it tends to lose.
- **technical is nearly as bad** (13.1% hit rate) and routinely gets heavy weight (0.2+). Only **news** is respectable (32% hit, lowest MAPE).
- Beating baseline is coin-flip at best; many fails barely differ from baseline APE — the ensemble adds little edge on hard days.

## Unreliable under these conditions
- **High-volatility / gap days:** any day where baseline_ape > ~0.02, both direction and magnitude collapse regardless of winning strategy.
- **When macro or technical is the deciding strategy:** disproportionately present in misses.
- **Confidence miscalibration is emerging in the recent window.** Earlier high-confidence days were fine, but late-stage confident calls miss: 08-19 (0.62), 08-20 (0.54), 09-03 (0.62), 09-17 (0.54), 09-22 (0.52), 10-06/10-07/10-08 (0.60–0.66 all fail). The recent rising-confidence regime (0.5–0.66) is NOT earning its confidence — 7 flagged overconfident misses understates it.
- **Null/empty weights rows** (late Sept onward) coincide with a confident-but-missing streak — check whether the weighting pipeline is silently broken.

## Fixes to try next
- **Down-weight or gate macro and technical** hard; lean the ensemble toward news on normal days. Current weight_hints over-reward losers.
- **Add a volatility/gap regime detector**: when expected move is large, widen the band and stop anchoring to prior close — magnitude fails concentrate here.
- **Recalibrate confidence** against the last ~40 days specifically; the 0.5–0.66 band is currently anti-predictive. Cap confidence until direction hit-rate recovers.
- **Fix/verify the weights pipeline** (null weights since 09-28) — confident misses started right around the same time.
- **Target the sign, not the size**: add a directional-only classifier and penalize wrong-side predictions separately from APE.