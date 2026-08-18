# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Directional misses dominate**: 19 of 28 fails (68%) are wrong-direction, not just off-magnitude. The system can't call the sign, which is worse than a coin flip on fail days.
- **We rarely beat baseline on fails**: most fails also lose to baseline, meaning the model adds noise, not signal, in the exact regimes it struggles.
- **News is the most-trusted strategy (weight_hint 0.22, top winner) yet its hit rate is only 28%** — we're leaning hardest on a strategy that's right barely 1-in-4 times.
- **Macro is dead weight**: 10.9% hit rate, highest MAPE (0.0185). Every day it "wins" the ensemble (6/12/15, 7/13, 7/15, 7/30) it fails. It's a reliable loss signal.
- **Tail blowups**: 7/31 APE 13.2%, 6/22 5.1%, 6/17/25/26/29 all ~3-5% — clustered magnitude errors on high-volatility days the model doesn't widen for.

## Unreliable under these conditions
- **High-volatility / gap days** (baseline_ape >3%): model tracks the miss rather than correcting it — 6/15, 6/17, 6/22, 6/25, 6/29, 7/15, 7/30, 7/31 all fail directionally.
- **Whenever macro or news is the winning strategy** in choppy tape: near-automatic fail.
- **Mid-confidence band (0.4–0.55)**: this is where most fails live. Confidence is essentially flat/uninformative — overconfident_misses=0 only because confidence never gets high, not because it's calibrated. Low conf (0.1–0.32) days pass about as often as they fail, so confidence carries no discriminating power.
- **Late-June cluster (6/12–6/29): 8 straight fails** — a sustained regime the model never adapted to.

## Fixes to try next
- **Cut or floor macro weight to ~0** and redistribute; it's a net-negative contributor.
- **Stop trusting news's high weight**: cap it until its directional hit rate improves; require corroboration from a second strategy before acting on news-driven signals.
- **Add a volatility regime gate**: when recent baseline_ape or realized vol is elevated, widen intervals and lower position/confidence — the tail blowups all came from not respecting vol.
- **Rebuild confidence calibration**: current scores don't separate wins from losses. Tie confidence to cross-strategy agreement and recent regime stability.
- **Attack the sign problem directly**: 68% of fails are directional — add a dedicated up/down classifier and only commit magnitude when direction agreement is strong.