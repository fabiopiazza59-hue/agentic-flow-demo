# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Direction is the core problem, not magnitude.** All 3 directional misses (incl. the "passing" 06-16) come with zero magnitude misses. The model nails the level but bets the wrong sign — useless for trading.
- **The two outright fails (06-12, 06-15) also lost to baseline.** Mean fail APE 2.71% and both worse than just-guessing-prior. When it's wrong, it's actively worse than doing nothing.
- **06-16 "pass" is misleading**: APE 0.17% but directional_hit=false and beats_baseline=false. Counting it as a win flatters the scorecard.
- Sample is tiny (n=4); treat all of this as directional signal, not proven law.

## Unreliable under these conditions
- **momentum as winning strategy is a coin flip on direction** (hit_rate 0.5) and was the "winner" on both 06-12 and 06-16 misses — high MAPE on macro-skewed weight days.
- **macro-heavy weighting precedes failure**: 06-12 (macro 0.34) and 06-15 (momentum+macro 0.60) are both fails. macro strategy is the worst performer (MAPE 1.64%, hit 0.25).
- **technical and contrarian are dead weight**: 0 wins across all 4, yet contrarian carries the 2nd-highest weight hint (0.20). It's getting weight it hasn't earned.
- Confidence is **not the issue** — overconfident_misses=0, and confidence was actually *lower* (0.40, 0.42) on the bad-direction days. If anything the model knows when it's shaky but still ships wrong-sign calls.

## Fixes to try next
- Add a **directional gate**: when strategies disagree on sign, suppress/flatten the trade rather than committing — magnitude accuracy is wasted if sign is wrong.
- **Cut technical and contrarian weight toward zero** until they show any directional hits; reallocate to news (best MAPE + only consistent contributor).
- **Cap macro weight** and stop letting macro+momentum jointly dominate (>0.5) — that combo owns both failures.
- Re-score 06-16 as a directional miss; stop crediting passes that fail direction AND baseline.
- Get more samples before trusting any weight hint; n=4 can't distinguish skill from noise.