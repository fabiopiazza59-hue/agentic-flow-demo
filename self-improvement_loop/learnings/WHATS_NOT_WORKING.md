# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Direction, not magnitude, is the core failure**: 17 of 22 misses are directional. The model calls the sign wrong far more often than it sizes the move wrong. Fixing APE won't help; we're guessing up/down.
- **Overall directional hit rate is poor** (~50% at best across strategies; individual strategy hit rates 15–29%). We are near coin-flip and losing to baseline on most fail days (beats_baseline false on nearly every miss).
- **Big-move days are systematically blown**: the worst APEs (3–5%: 06-22, 06-25, 06-17, 06-15, 07-23, 07-30) cluster on large realized moves, and we almost always get the sign wrong on them. The model reverts to a small predicted move and gets run over.
- **No strategy is reliable.** News is "best" at 29% hit / 0.153 MAPE but still loses more than it wins. Contrarian and macro are the worst (15% hit) yet still carry ~19% weight each — dead weight.

## Unreliable under these conditions
- **High-volatility / gap days**: whenever the true move exceeds ~2%, directional accuracy collapses. All 3–5% APE days are directional misses.
- **When macro or contrarian is the winning strategy**: 06-15, 07-13, 07-15, 07-30 (macro) and 06-26, 07-21 (contrarian) are almost all fails. These strategies win the vote but lose reality.
- **Confidence is uninformative, not overconfident**: overconfident_misses = 0, but that's because confidence is uniformly low (0.1–0.58). Some passes came at conf 0.1 (07-27) and some fails at 0.55 (06-30, 07-13). Confidence carries near-zero signal — it's flat noise, not calibrated.
- **Empty-weights days** (06-30, 07-22, 07-24, 07-28) are a config/plumbing red flag — weights aren't being populated, yet predictions still ship.

## Fixes to try next
- Add a **volatility regime gate**: on expected-high-move days, widen predicted magnitude and stop defaulting to small mean-reversion.
- **Cut or down-weight macro and contrarian** (15% hit); rebalance toward news/momentum and require directional agreement before committing.
- **Rebuild confidence as a calibrated, directional probability** — current values are meaningless; suppress trades when true directional confidence is low.
- Fix the **empty-weights bug**; audit those days separately.
- Track and report a **directional-accuracy-by-volatility-bucket** metric; APE is hiding the real problem.