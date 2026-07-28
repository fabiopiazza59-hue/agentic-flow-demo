# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Direction is the core problem, not magnitude.** 16 of 20 fails are directional misses (80%); only 4 are magnitude-driven. The system is barely better than a coin flip on which way AMZN moves.
- **Errors are worst exactly when moves are large.** Every big-APE fail (0.03–0.05) is also a directional miss (06-15, 06-22, 06-25, 06-29, 07-15, 07-23). The model gets small drift days roughly right and violently wrong-foots on big-move days.
- **Beats-baseline tracks pass/fail nearly 1:1.** When we fail we almost always also lose to the naive baseline — we add negative value on miss days, not just underperform.
- **No overconfidence flag firing, but confidence is essentially flat/uninformative.** Most predictions sit at 0.40–0.55 regardless of outcome; confidence carries almost no signal separating hits from misses. Miscalibration here is *under-discrimination*, not overconfidence.
- **Macro is the worst strategy** (9.7% hit rate, highest MAPE) yet still carries ~18% weight and "wins" on 3 days — all of which failed (06-12 setup, 07-13, 07-15).

## Unreliable under these conditions
- **High-volatility / large-move days:** consistently directionally wrong and worst APE.
- **When `macro` or `contrarian` is the winning strategy:** contrarian 16% hit, macro 10% hit — both drag. Contrarian-led days (06-26, 07-21) miss badly.
- **News-led days are the *only* relative bright spot** (32% hit) but still lose more than half.
- **Days with empty `weights` (06-30, 07-22)** all failed — fallback/degenerate config path is broken.

## Fixes to try next
- Cut or floor **macro weight toward ~0**; redistribute to news/technical. It is net-harmful.
- Add a **volatility regime filter**: on high-expected-range days, widen intervals and de-weight trend-following (momentum/contrarian) which whipsaw.
- **Recalibrate confidence** to actually separate outcomes (isotonic/Platt on realized hit rate); current 0.4–0.55 band is noise.
- Investigate/patch the **empty-weights fallback** — it produces guaranteed misses.
- Focus modeling effort on **sign prediction** (directional loss), since magnitude is already acceptable on non-fail days.