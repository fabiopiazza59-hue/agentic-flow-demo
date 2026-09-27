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
