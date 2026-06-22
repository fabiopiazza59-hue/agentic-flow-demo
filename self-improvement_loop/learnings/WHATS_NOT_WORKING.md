# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Directional errors dominate**: 3 of 4 fails missed direction. We're not just sizing moves wrong, we're calling up/down wrong. That's the core problem, not magnitude.
- **Losing to the naive baseline**: 4 of 6 predictions failed to beat baseline (incl. one "pass" on 06-16 that was still a directional miss and worse than baseline). The ensemble is adding noise, not signal.
- **Mean fail APE 2.95%** — these aren't near-misses; they're meaningfully off on a stock that doesn't move 3%/day typically.
- **No strategy is reliable**: hit rates are 17–33% across the board. Momentum is the "best" at 33% and still loses two-thirds of the time. Macro is the worst (MAPE 2.31%, 17% hit).

## Unreliable under these conditions
- **Mid-June consecutive losing streak (06-12 → 06-18)** suggests a regime shift the model never adapted to — likely trending/volatile conditions where it kept fading the wrong way.
- **Whenever news/momentum carry the top weight** (06-12, 06-15, 06-17), results are bad. The 06-17 fail had news at 0.34 and technical — the "winning" strategy — buried at 0.12. Winning-strategy attribution is disconnected from weighting.
- **Low-confidence days are also losing** (conf 0.38–0.42 fails). Note: 0 overconfident misses, so confidence isn't inflated — but it's also not discriminating winners from losers. Confidence is flat and uninformative.

## Fixes to try next
- **Add a regime filter**: detect trending vs. mean-reverting and gate contrarian/momentum accordingly; the 06-12→06-18 streak screams unhandled regime.
- **Demote macro and news in the blend** — worst MAPE and 17% hit rates; current weight_hints (news 0.24, macro 0.17) over-reward news.
- **Fix weight/winner mismatch**: stop letting a strategy "win" attribution while weighted near-zero (contrarian 0.05 on 06-18 "won").
- **Calibrate confidence to outcomes** — right now it's noise; either tie it to ensemble agreement or suppress trades when strategies disagree.
- **Sample size caveat**: n=6 is tiny. Treat these as hypotheses; keep scoring before re-weighting hard.