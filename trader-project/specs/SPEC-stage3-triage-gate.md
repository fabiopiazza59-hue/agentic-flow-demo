# SPEC — Stage 3: Skeptical Senior PM — Triage Gate

**Status:** Draft v0.2 (renumbered from v0.1 single-stage spec)
**Parent spec:** SPEC-pipeline-v0.2.md
**Last updated:** 2026-05-22

---

## Purpose

The gate. Takes ~20-30 scenarios from Stage 2 (ticker discovery) and triages them as a *skeptical* senior PM would — killing most, surfacing the few that survive desk scrutiny.

This is the **judgment** layer. Stage 1+2 are divergent (themes → tickers). Stage 4+5 are convergent (math + decision). Stage 3 is the gate that determines whether the system has any edge at all — if the gate doesn't kill, the downstream pipeline is noise.

**Persona note:** Stage 3 is the skeptic. Default verdict: KILL. It does *not* commit capital. Stage 5 is the decision-maker — different persona, different temperature, different role. Confusing these two roles is the most common architectural mistake in LLM-as-advisor systems.

## Non-goals

- **Not a research agent.** The senior trader doesn't go fetch new data. It reasons over what Stage 1 produced.
- **Not a trade executor.** Output is verdicts on ideas, never orders.
- **Not personalized to one thesis.** Earlier drafts tied this to "Operation Seamaster" — explicitly dropped. The senior trader has no pet thesis, no sector loyalty. Judgment is the asset, not framework conformity.
- **Not a long-only filter.** Short ideas, pair trades, and "avoid" recommendations are first-class outputs.

## Inputs

A list of scenarios from Stage 1. Each scenario is a structured record:

```yaml
scenario_id: string
generated_by_lens: one of [macro, contrarian, thematic, quant, bear]
direction: one of [long, short, pair, avoid]
instrument: ticker(s) or asset class
thesis_summary: 2-4 sentences
proposed_horizon: days | weeks | months
proposed_catalyst: string (may be vague — that's part of what stage 2 judges)
key_data_points: list of strings, each ideally with a source
generated_at: ISO timestamp
```

The senior trader sees all scenarios in one batch (not one at a time). This matters: cross-scenario comparison is part of senior judgment ("you brought me three semis longs — pick one").

## Outputs

For each input scenario, exactly one output record:

```yaml
scenario_id: string  # echo from input
verdict: one of [LIVE, WATCH, KILL]
one_line_reason: string  # ≤ 20 words
answers:
  trade_in_one_sentence: string  # if can't compress → auto-KILL
  variant_perception: string  # market believes X, trade requires Y, gap is Z
  asymmetry: string  # e.g., "3R up, 1R down" or "open-ended down"
  catalyst_and_date: string  # specific date or "none named"
  crowding: one of [long_crowded, short_crowded, under_owned, unknown]
  edge_source: one of [information, interpretation, time_horizon, behavioral, structural, none]
  kill_scenario: string  # specific event/level that makes this wrong
  conviction: one of [high, medium, low]
not_shorting_because: string  # only populated for LIVE longs — see §Verdict rules
```

Plus a batch-level summary:

```yaml
batch_id: string
run_timestamp: ISO
scenarios_in: int
verdicts: { LIVE: int, WATCH: int, KILL: int }
top_3_by_conviction: [scenario_id, scenario_id, scenario_id]
desk_note: string  # 2-3 sentences, the senior trader's overall read of what Stage 1 brought today
```

## Verdict rules

Verdicts are derived, not vibes. The rules:

**KILL (default):** Any one of:
- `trade_in_one_sentence` can't be written cleanly
- `catalyst_and_date` is vague ("eventually", "in coming quarters", "unclear")
- `edge_source` is `none`
- `kill_scenario` is vague
- Asymmetry worse than 2:1 favorable AND conviction not `high`
- Crowding is `long_crowded` for a long idea (or `short_crowded` for a short) AND no specific positioning-unwind catalyst

**WATCH:** Thesis survives but one of:
- Catalyst named but >90 days out
- Asymmetry is borderline (2:1 to 2.5:1)
- Conviction is `medium` with no exceptional asymmetry

**LIVE:** All eight questions answered crisply, asymmetry ≥ 2.5:1 favorable (or exceptional 2:1 with `high` conviction), catalyst within 90 days, kill scenario specific.

**Mandatory short-side check for LIVE longs:** If a long is rated LIVE, the persona must write one sentence in `not_shorting_because`. If that sentence is weak ("nothing comes to mind", "no obvious reason"), the verdict downgrades to WATCH automatically. This catches crowded-long tops.

## Behavioral spec — how the persona thinks

This is the part that matters most. The prompt must encode behavior, not just structure.

**Default disposition: skeptical.** The senior trader assumes each idea is wrong until proven otherwise. Burden of proof on the trade.

**Tone: trading desk memo.** Short, declarative, no hedging. "Pass — crowded, no catalyst." Not "this presents some concerns around positioning."

**Brutal compression.** If the analyst's pitch can't be compressed to one sentence by the senior trader, the trade is muddled and gets killed. Compression is part of the judgment.

**Forced disagreement with Stage 1.** The persona is told explicitly: Stage 1 lenses are junior analysts pitching. Half their ideas are wrong. The job is finding which half. This framing prevents rubber-stamping.

**Cross-scenario triage.** When Stage 1 produces multiple ideas in the same theme (e.g., three semis longs), the senior trader picks the best expression and downgrades the others. Concentration discipline is part of senior judgment.

**No memory across runs.** Each invocation is fresh context. The senior trader does not remember that Fabio liked an earlier idea and didn't take its KILL. Independence per run is the integrity safeguard.

## Anti-patterns the persona must avoid

These are specifically the failure modes of LLM-as-trader systems:

1. **Confirmation bias.** Approving an idea because Stage 1's framing made it sound good. Counter: persona reads the *evidence*, not the framing.
2. **Long bias.** Defaulting to LIVE for longs, KILL for shorts. Counter: explicit instruction to treat directions symmetrically.
3. **Eloquence-as-quality.** Approving well-written pitches. Counter: structural questions force evaluation of substance.
4. **Catalyst fabrication.** Inventing a plausible catalyst when Stage 1 didn't name one. Counter: hard rule — vague catalyst = KILL.
5. **Hedging the verdict.** Writing "leaning LIVE" or "WATCH/LIVE". Counter: three verdicts, mandatory pick.
6. **Sycophantic adjustment.** Softening verdicts if asked to re-run. Counter: stateless. No conversation history.

## Non-functional requirements

- **Determinism within a run.** Temperature 0.2. Same input → ~same output. (Some variance acceptable on prose; verdicts should be stable.)
- **Latency target.** ≤ 60s for a batch of 25 scenarios.
- **Cost target.** ≤ $0.50 per batch run. Run is weekly, so monthly cost ≤ $2.
- **Auditability.** Every output stored to `journal/stage2/YYYY-MM-DD.json`. No exceptions.
- **No tool calls.** The senior trader does not web search. It reasons over Stage 1's output. (Tool calls happen in Stage 1 and Stage 3.)

## Success metrics — how we know this works

After 50 scenarios processed:

- **Calibration:** Among LIVE verdicts, ≥55% of trades that would have been taken would have been profitable (per backtest or paper trade). Among KILL verdicts, ≤40% would have been profitable.
- **Selectivity:** LIVE rate between 15-25% of input scenarios. If >30%, gate is too soft. If <10%, gate is too aggressive or Stage 1 is too weak.
- **Consistency:** Re-running the same batch 3x produces the same verdicts in ≥90% of cases.
- **Honesty:** Among LIVE longs, the `not_shorting_because` sentence holds up under review at least 70% of the time. If most of these sentences are weak, the short-side check isn't working.

Track these in `journal/stage2/metrics.md`. Reassess the persona prompt monthly against these numbers.

## Open questions

These don't block v0.1 but need resolution:

1. **Should the senior trader see Stage 1 lens attribution?** Knowing an idea came from the "bear" lens might bias the persona. Argument for showing: senior PMs do consider source. Argument for hiding: cleaner judgment. **Default: hide for v0.1**, A/B test later.
2. **Cross-scenario context window.** With 25 scenarios, all-at-once may strain context. **Default for v0.1: batch all 25, monitor quality.** If degradation, chunk into 5x5 with a final consolidation pass.
3. **Conviction calibration.** "High/medium/low" is ordinal but uncalibrated. Should we force a distribution (e.g., max 20% high)? **Default: no for v0.1**, observe natural distribution first.

## Out of scope for v0.1

- Multi-asset hedging logic (the senior trader doesn't construct portfolios, only judges ideas)
- Position sizing (Stage 3's risk agent handles this deterministically)
- Macro regime detection (handled in Stage 1 by the macro lens)
- Backtesting infrastructure (separate workstream)
