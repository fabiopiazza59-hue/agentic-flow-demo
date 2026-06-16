# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- Sample is tiny (n=3), so treat everything below as early signal, not proven law.
- Both failures are **directional misses, not magnitude blowups** — the model gets the *direction* wrong, then the error compounds (APE 2.0% and 3.4%). Magnitude itself isn't the core problem.
- Both losses also **failed to beat baseline** — on miss days we'd have been better off doing nothing/persistence.
- Errors are **escalating**: 0.07% → 2.0% → 3.4%. Whatever regime started ~6/12 is getting worse, not mean-reverting.
- The only win came when weights were **flat/equal (all 0.2)**. Both losses came after the ensemble **tilted hard** into one or two strategies (macro/momentum heavy). Concentration is correlating with failure.

## Unreliable under these conditions
- **momentum** as winning strategy on 6/12 → directional miss. Momentum-heavy weighting appears to chase the wrong way during turns.
- **news** as winning strategy on 6/15 → worst miss (3.4%), yet news carries the **highest weight hint (0.31)**. We're over-trusting our weakest live performer.
- **technical and contrarian: 0% hit rate** across all 3. No demonstrated edge so far; contrarian still gets a 0.20 weight.
- Confidence is **mildly miscalibrated downward, not overconfident**: 0.58 conf on a miss, 0.40 on the worst miss — at least confidence dropped on the hardest day. No overconfident-miss flag triggered. Conf isn't the headline problem; direction is.

## Fixes to try next
- **Don't let any single strategy exceed ~0.25 weight** until it earns it; the flat-weight day was the only pass.
- **Cap/penalize news weight** — it leads weighting but only 33% hit rate and drove the largest error.
- Add a **persistence/baseline floor**: if ensemble disagrees with baseline direction, shrink position size — both misses lost to baseline.
- Build a **directional-confidence gate**: when strategies disagree on sign, widen interval and lower conviction rather than committing.
- Investigate the **post-6/12 regime** (vol spike? trend break?) — track whether momentum/technical degrade specifically in choppy/reversal conditions.
- Keep scoring; **n=3 is far too small** to retire strategies — flag, don't kill.