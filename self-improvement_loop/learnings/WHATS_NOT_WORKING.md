# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Direction is the core failure**, not magnitude. 15 of 18 misses are directional (83%); only 3 are magnitude-driven. The model is often close in size but points the wrong way.
- **We barely beat the coin flip.** Overall directional hit rate is roughly 46% (13/28), and on fail days we're routinely on the wrong side while also failing to beat baseline (most fails have `beats_baseline: false`).
- **Big-error days are clustered and directional.** The worst APEs (0.051, 0.037, 0.036, 0.034, 0.032, 0.032, 0.030) are almost all directional misses concentrated in mid-to-late June — a sustained regime where every strategy read the tape wrong.
- **No overconfidence problem — the opposite.** 0 overconfident misses; confidence sits in a dead band of 0.38–0.58. Confidence carries almost no discriminative signal; it neither flags misses nor marks the good calls. It's effectively noise.

## Unreliable under these conditions
- **Macro-led days are the worst:** 10.7% hit rate, highest MAPE (0.0179). Every macro-winning day shown (06-08 aside) failed (07-13, 07-15). Do not trust macro as the driver.
- **Momentum and technical are weak in choppy/reversal tape:** hit rates 21% and 18%. Momentum leads several large directional whiffs (06-12, 06-25). These strategies chase and get caught on turns.
- **Contrarian is consistently down-weighted (0.05–0.16) yet wins when it does — it's starved.** The blend suppresses the one strategy that catches reversals.
- **The mid/late-June high-volatility regime** broke everything at once — a systemic regime-detection gap, not a single-strategy issue.

## Fixes to try next
- **Attack direction directly:** add a sign-accuracy objective / directional loss; separate the up/down decision from the magnitude estimate.
- **Down-weight macro hard** (near floor) and cap momentum/technical when recent volatility is elevated; **raise contrarian's floor** so it can express on reversal days.
- **Add a regime filter** (realized vol / trend-vs-chop) and switch blend weights by regime; the June cluster shows one blend can't cover both.
- **Recalibrate confidence** — current range is uninformative. Fit confidence to historical hit rate so low-conviction days can be flagged or sat out.
- **Lean on news** as the relative best (32% hit, lowest MAPE) but note it's still sub-coinflip — it's least-bad, not good.