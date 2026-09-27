# Panel — many names per session, so an edge can be seen in weeks, not years

**Status:** implemented 2026-09-27 (roadmap item R1 of [`v2-sota-upgrade.md`](v2-sota-upgrade.md)).
**Code:** `src/panel/` (data, analyst, runner, report), `src/loop/run_panel.py`.
**Results:** [`RESULTS_PANEL.md`](../RESULTS_PANEL.md), `results/panel_metrics.json`.

## Why

At one forecast a day the loop cannot see realistic edges: its daily gain is too noisy. Averaging
the gain over the names forecast in a session removes the idiosyncratic part of that noise.
Measured on the last 60 sessions with real 09:20 ET anchors and a no-skill ±0.1σ adjuster:

| | SD of daily APE gain | Sessions to confirm a true 0.02% gain | … a 0.05% gain |
|---|---|---|---|
| AMZN alone | 0.158% | ~620 (2.5 years) | ~93 (4–5 months) |
| 30-name panel average | 0.026% (**6.0× lower**) | ~21 (**1 month**) | ~10 (2 weeks) |

Co-movement did not eat the gain (6.0× vs √30 = 5.5× if independent): because every forecast is
judged against its own pre-open price, market-wide moves largely cancel. The session stays the
inferential unit, so correlated names are never counted as independent evidence.

## Design

* **Universe:** 30 liquid US large caps across sectors (`settings.PANEL_SYMBOLS`), AMZN included.
* **Data:** keyless, two Yahoo downloads per run for all names (daily bars; 1-minute bars with
  pre/post-market). A name without the previous session's close is dropped for that session;
  with fewer than `PANEL_MIN_NAMES` names the run waits for the next slot.
* **Forecast per name:** anchor = latest pre-open trade (else the prior close);
  `predicted_close_llm = anchor · (1 + adj · σ)`, where `adj` is the LLM's move in the name's own
  daily σ (EWMA), clamped in code to ±0.5σ. Most names on most days should get 0.
* **LLM:** one call per 10 names (`PANEL_MODEL`, ≤3 web searches each) — three calls a day for
  the whole panel. Offline, every `adj` is 0 and the panel is the anchor baseline.
* **Timing:** identical to the AMZN arms — research inside the session's forecast window (the
  evening run), code-only re-anchoring on later pre-open runs, nothing written after the open.
* **Distributions:** symmetric split-conformal quantiles pooled over all names' past clean
  residuals (Student-t(5) until 30 exist); rows carry `quantiles`, `p_up`, `confidence`.
* **Kill switch:** if the anytime-valid CS on the daily APE gain vs the anchor turns negative,
  `learnings/panel_state.json` switches the LLM off: the panel then ships the anchor, but still
  logs `predicted_close_llm`, so the evidence keeps accruing and the switch can turn back on.

## Metrics (pre-open rows only; each averaged over names per session, then a CS over sessions)

| Metric | Question it answers |
|---|---|
| APE gain vs the anchor | Does the LLM's number beat the free baseline? (**headline**) |
| CRPS gain vs the anchor | …as a calibrated distribution? |
| Rank IC | Do the names it pushes up outperform their anchors? (cross-sectional skill) |
| Brier of P(up) vs the anchor | Does it add directional information? |
| Anchor vs prior close | How much the free information itself is worth (sanity check) |

## Cost and limits

* Three LLM calls with web search per session, plus the two AMZN arms.
* Keyless Yahoo data can be rate-limited from CI IPs; the run then retries twice and otherwise
  skips the session rather than forecasting on stale data.
* No replays: past sessions cannot be re-forecast honestly (the anchors and news are gone).

## Prompt evolution (`src/panel/evolve.py`)

The panel's analyst prompt now evolves, under the same rule as every other mechanism: nothing
changes what ships until it is proven better.

* **What can evolve:** only the *strategy* text (what to look for, when to move a name, when to
  stay at 0). The *contract* — tool budget, the ±0.5σ clamp, the JSON format — is appended in
  code to every variant, so no prompt can break parsing or widen the cap.
* **Shadow runs:** each session the champion ships and up to `EVOLVE_MAX_CHALLENGERS` (2)
  challengers run on the same snapshot; rows log `prompt_variant` and `shadow_adj`, scoring adds
  `shadow_ape`. Cost: 3 extra batched calls per challenger per session.
* **Promotion:** per session, average (champion APE − challenger APE) over the names; a
  challenger is promoted when the anytime-valid CS on that series excludes zero. The k-th
  challenger ever created is tested at α/(k(k+1)) — the budgets sum to α, so the probability of
  *ever* promoting a prompt that is not better stays below α however long the loop runs.
  Shown worse → retired; still undecided after `EVOLVE_MAX_SESSIONS` (60) → retired.
* **Proposals:** when a slot is free and the champion has ≥ 5 clean sessions, the reflector model
  (`REFLECTOR_MODEL`, one call) reads the champion's strategy, numbers-only failure cases (where
  it hurt most vs the anchor, the largest moves it faced) and every variant tried so far with its
  result, and writes one new strategy (reflective mutation, in the spirit of GEPA). The analysts'
  free-text reasons are never passed on: they were shaped by web pages and could carry injected
  instructions into future prompts. Proposals outside 200–3,000 characters or identical to an
  earlier variant are discarded.
* **Audit:** every prompt that ever ran, its parent, rationale, statistics and fate are in
  `learnings/prompt_variants.json` (committed each run); RESULTS_PANEL.md shows the table.

Expected pace: a challenger needs roughly the same number of sessions as the edge test above —
about 2–4 weeks if it is materially better, longer (then retired at 60) if not. The first
proposal comes after the champion's fifth clean session.
