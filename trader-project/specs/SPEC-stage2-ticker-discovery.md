# SPEC — Stage 2: Ticker Discovery

**Status:** Draft v0.1
**Parent spec:** SPEC-pipeline-v0.2.md
**Last updated:** 2026-05-22

---

## Purpose

Translate each Stage 1 theme into 3-5 specific tradeable instruments with a per-instrument thesis. This is the *creative* step in the funnel — given "nuclear renaissance accelerates," propose the picks-and-shovels expressions (CCJ, CEG, NNE, OKLO, GEV, etc.) with reasoning.

The output is not yet a trade. It's a *scenario*: ticker + direction + thesis + proposed catalyst + proposed kill condition. Stage 3 decides whether each scenario survives.

## Non-goals

- Not making the trade decision (Stage 5).
- Not sizing (Stage 4a).
- Not screening a universe by quantitative criteria (that would be a separate Stage 4 sub-task if ever needed).
- Not exotic instruments. Limit v0.1 to: liquid US/EU equities, major ETFs, occasional commodity proxies (e.g., URA for uranium). No options, no futures, no individual bonds. CFD-tradeable on Revolut is the practical universe.

## Inputs

Stage 1 output (all themes from the run).

## Outputs

```yaml
run_id: string
scenarios:
  - scenario_id: string  # e.g., S-001
    parent_theme_id: string  # T-001
    direction: long | short | pair | avoid
    instrument: string  # ticker or ETF
    instrument_class: equity | etf | commodity_etf
    thesis_summary: string  # 2-4 sentences
    proposed_horizon: days | weeks | months
    proposed_catalyst: string  # named event + approximate date
    proposed_kill: string  # specific price level or event that invalidates
    key_data_points:
      - claim: string
        source: string
        url: string
    why_this_expression: string  # 1-2 sentences — why this instrument vs. alternatives
    alternative_expressions: string[]  # other tickers considered but not chosen
```

## Behavioral spec

**Multiple expressions per theme — but ranked.** For each theme, propose 3-5 ticker candidates. Pick the best as primary scenario. List the others in `alternative_expressions` of that primary record. Don't emit 5 scenarios for the same theme — that floods Stage 3 with near-duplicates.

**Direction is theme-derived, not freelanced.** If a theme is `long_bias`, scenarios should be long; if `short_bias`, scenarios should be short; if `mixed`, agent can produce both. Don't invent direction.

**Catalyst and kill are mandatory.** A scenario without a named catalyst date and a specific kill condition fails the contract and is dropped (logged as `dropped: missing_catalyst_or_kill`). This pre-empts Stage 3's auto-KILL for the same reason; better to drop here than spend Stage 3 compute on it.

**Liquidity check.** Each proposed instrument must clear a liquidity floor (median daily volume ≥ $5M USD equivalent). If unsure, the agent web-searches and verifies. Microcaps dropped.

**No theme without ≥1 scenario.** If a theme produces zero viable scenarios, log it with reason — this means either the theme was un-actionable (rare) or the agent failed (more common, prompt issue).

**Symmetric direction continues.** If Stage 1 produced a `short_bias` theme, Stage 2 must produce at least one short scenario. No quiet conversion to longs.

## Reasoning quality — the "why this expression" field

The most valuable field for downstream stages is `why_this_expression`. It must answer: "of the alternative tickers, why is this one the cleanest play?" Acceptable reasons:

- Pure-play exposure (highest revenue concentration in the thematic driver)
- Liquidity advantage (vs. otherwise-cleaner microcap)
- Asymmetric setup (one cycle ahead or behind peers)
- Quality/balance sheet (less levered for short scenarios in the same theme)

Unacceptable (and the agent will flag and drop them):
- "It's the most well-known" (lazy)
- "It's trending" (recency bias)
- "Best performing recently" (chasing)

## Anti-patterns

1. **Ticker dump.** Producing 8 scenarios per theme to look exhaustive. Cap at the primary + alternatives list.
2. **Re-using the same ticker across themes** without flagging it. If MU appears as the expression for two themes, the agent must note this and Stage 3 will see the cross-link.
3. **Inventing catalysts.** If the agent doesn't find a real, dated catalyst via web search, the scenario is dropped — never make up an earnings date.
4. **ETF cop-out.** Defaulting to the broad sector ETF for every theme. ETFs are fine when the theme is genuinely broad-based (e.g., "credit stress"), wrong when there's a clean single-name expression (e.g., "AI memory ramp" → MU is cleaner than SOXX).
5. **Liquidity violations.** Proposing illiquid names because they're "purer." Pure + untradeable = useless on a $2k CFD account.

## Tool use

- Web search: yes, for verifying catalyst dates, recent earnings, liquidity.
- File read/write: writes its output YAML, reads Stage 1 output.
- No real-time pricing dependency (Stage 4 handles current prices).

## Success metrics

- **Theme coverage:** ≥90% of Stage 1 themes produce at least 1 scenario. Below 80% = Stage 2 prompt failing.
- **Catalyst specificity:** ≥80% of scenarios have a catalyst with a specific date (not "later this year"). Below 60% = prompt too lax.
- **Direction balance:** matches Stage 1's direction distribution within ±10pp.
- **Re-use detection:** when a ticker appears in 2+ scenarios in a run, agent flags it. Manual review monthly to verify.

## Out of scope for v0.1

- Options strategies (defined-risk overlays).
- Pair construction (mentioned in direction enum but not implemented — long-only + short-only for v0.1).
- Sector-relative trades.
- Position sizing (Stage 4a).
