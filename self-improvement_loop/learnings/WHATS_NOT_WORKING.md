# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Directional errors dominate**: 20 of 29 fails are wrong-direction, only 9 are magnitude-only. This is a sign/timing problem, not a scaling problem. The model is coin-flipping direction.
- **The ensemble rarely beats baseline on fails**: most failing days have ape ≈ or worse than baseline_ape, meaning the blend adds noise, not edge. On clean days it barely edges baseline (e.g. 08-06, 07-14 lose to baseline despite passing).
- **No strategy is reliable**: best hit-rate is news at 27% and macro at 12%. Every strategy loses direction more than 70% of the time — the "winning strategy" label is retrospective luck, not predictive.
- **Big-move days blow up**: fails cluster at high APE (0.03–0.13; e.g. 07-31 at 13%, 06-22 at 5%). The model cannot handle gap/large-range days and gets both direction and magnitude wrong.

## Unreliable under these conditions
- **macro-led days**: 6 wins/48, mostly on losing fails (06-12, 07-13, 07-15, 07-30, 08-19). Whenever macro is the winning strategy it's usually a miss.
- **High-volatility / large-move regimes**: all worst APE days are directional misses — model has no volatility awareness.
- **Higher-confidence calls are NOT safer**: 08-19 (conf 0.62) and 06-12 (0.58) both failed; passes occur at conf 0.1–0.32 as often as at 0.5+. Confidence is essentially uncorrelated with correctness — flat/miscalibrated even if only 1 flagged "overconfident."
- **news-heavy weighting on quiet days**: news carries top weight_hint (0.22) but still misses direction 73% of the time.

## Fixes to try next
- Attack **direction first**: add a separate directional classifier / sign-vote gate; magnitude is not the bottleneck.
- **Down-weight or benchmark-out macro and technical** (hit rates 12.5% / 16.7%); test a news+momentum-only blend.
- **Recalibrate confidence** against realized hit-rate; current confidence is decorative. Suppress/abstain when strategies disagree.
- Add a **volatility/gap detector** and widen intervals or abstain on high-range days — that's where the worst APEs live.
- Since ensemble barely beats baseline, **prove edge vs baseline explicitly** before trusting any blend; consider defaulting to baseline when confidence is low.