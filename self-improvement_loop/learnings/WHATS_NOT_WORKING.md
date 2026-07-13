# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Direction, not magnitude, is the core problem.** 10 of 12 fails are directional misses (only 2 pure magnitude). The model gets the sign wrong repeatedly — it isn't a calibration-of-size issue, it's a can't-call-up/down issue.
- **Beating baseline is a coin flip at best.** Most fails also lose to baseline (e.g. 06-12, 06-15, 06-22, 06-25, 06-26, 06-29). On misses the ensemble adds nothing over naive persistence.
- **Sustained losing streak mid-to-late June.** From 06-12 through 06-29 there were 8 fails in 9 days with APEs of 2–5%. The model broke during a directional regime and stayed broken — no adaptation.
- **Confidence is flat and uninformative, not overconfident.** 0 overconfident misses only because confidence sits in a dead 0.4–0.58 band regardless of outcome. It carries no signal; it's not calibrated so much as constant.

## Unreliable under these conditions
- **Trending / regime-shift days.** The high-APE cluster (3–5%) is where price moved decisively and the ensemble faded or lagged the move — classic directional whiplash.
- **`macro` is nearly useless: 5% hit rate, worst MAPE.** It should not be carrying ~0.18 weight. `contrarian` is also weak (15% hit) and tends to lose exactly on the trending days.
- **`news`-led days are hit-or-miss.** News is the best strategy (35% hit) yet led many of the worst fails (06-15, 06-22, 06-29, 07-01) — it wins when it wins big but drags on high-vol news days.
- **07-08 fail ran on empty/blank weights** — pipeline integrity issue on 06-30 (empty weights dict) too.

## Fixes to try next
- Add an explicit **regime/trend filter**; in trending regimes cut `contrarian` and `macro` toward zero.
- **Down-weight or drop `macro`** (5% hit rate); redistribute to `news`/`technical`.
- Rebuild **confidence as a real calibrated probability** tied to directional hit rate — current values are noise.
- Fix the **empty-weights bug** and enforce a schema check before scoring.
- Since best hit rate is only 35%, treat **directional prediction itself as the priority metric**, not APE.