# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Direction, not magnitude, is the core problem**: 17 of 25 fails are directional misses (68%). The model gets the sign wrong far more than it gets the size wrong. On small-move days the APE is fine but the coin-flip on direction sinks the score.
- **Systematically overweighting `news`**: news carries the highest weight_hint (0.227) and highest single-strategy influence, yet only hits 27.5% of the time. Many fails (06-15, 06-22, 06-29, 07-01, 07-20, 07-23) have news as the winning strategy while direction is wrong.
- **Big misses cluster on high-volatility / event days**: 07-31 (13.2% APE), 06-22 (5.1%), 07-23 (4.2%), 07-30 (3.9%), 07-15 (3.0%) — the model is blind to large gaps and cannot even beat baseline meaningfully on them.
- **Weakly beats baseline on losses**: most fails have ape ≈ baseline_ape, meaning the model adds no edge exactly when it's wrong.
- Confidence is NOT the issue: 0 overconfident misses, and passing days often had low confidence (0.1). Confidence signal is essentially noise/uninformative, not miscalibrated-high.

## Unreliable under these conditions
- **Large-move / event days** (APE > 3%): every strategy fails, macro and news worst.
- **`macro`-led predictions**: 12.5% hit rate, highest MAPE (0.0196) — near-useless as a lead strategy.
- **`technical`-led**: 15% hit rate despite 2nd-highest weight; drop its influence.
- **`contrarian`-led**: 17.5% hit; contrarian consistently gets deprioritized in weights (often ~0.05–0.15) yet still leads losses.
- Empty-weights entries (07-22, 07-24, 07-28, 08-05) coincide with fails/low-conf — pipeline dropouts.

## Fixes to try next
- Cut `news` and `macro` weight; promote `momentum`/`news` only on their higher-hit regimes; demote `technical` and `macro` to tie-breakers.
- Build a **direction-first classifier** separate from magnitude — the ensemble optimizes size but flips sign.
- Add a **volatility/event gate**: on high-expected-move days, widen intervals or abstain rather than commit a direction.
- Recalibrate or retire the confidence field — it currently has no predictive relationship to passing.
- Fix empty-`weights` dropouts; they correlate with degraded outcomes.