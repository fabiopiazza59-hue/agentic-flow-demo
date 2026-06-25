# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Direction, not magnitude, is the killer**: 4 of 5 fails are directional misses; only 1 is pure magnitude. The model gets the size roughly right but bets the wrong way.
- **Baseline is beating us on fails**: on 4 of 5 fails we also failed `beats_baseline`. We're adding negative value precisely when we miss — worse than a naive carry-forward.
- **Error is escalating in late June**: APE climbs 2.0% → 3.4% → 3.7% → 5.1% (06-12 through 06-22). This is a trending regime the model fights.
- **Confidence is flat and uninformative** (mostly 0.40). 0 overconfident misses only because confidence never rises — calibration is dead, not good. No signal value.

## Unreliable under these conditions
- **Trending/momentum regimes**: directional misses cluster when price moves persistently; the ensemble keeps reverting against the trend (contrarian hit_rate 0.11, momentum only 0.22).
- **News-led days are coin-flips on direction**: `news` "wins" most often (3) yet still drove the 06-15 and 06-22 misses — it wins the ensemble but not the call.
- **Low-confidence days (≤0.40) dominate the fail set** — every big-APE miss sits at 0.38–0.42. The model knows it's unsure but still commits.
- **Contrarian and macro are dead weight**: contrarian 0.11 hit / 0.0212 MAPE, macro 0.11 hit / 0.0227 MAPE — worst two on both axes.

## Fixes to try next
- **Add a trend/regime filter**: when a multi-day directional move is in force, suppress contrarian and lean directional; stop fading trends.
- **Cut contrarian and macro weight hard** (both 0.11 hit rate); reallocate toward news/technical, the only sub-0.018 MAPE strategies.
- **Gate on baseline**: when ensemble direction disagrees with carry-forward AND confidence ≤0.45, default toward baseline rather than overriding it.
- **Rebuild confidence calibration**: current 0.40-flat output is useless; force spread and penalize directional disagreement across strategies.