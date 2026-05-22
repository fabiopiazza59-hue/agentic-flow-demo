# SPEC — Stage 1: Market Scan

**Status:** Draft v0.1
**Parent spec:** SPEC-pipeline-v0.2.md
**Last updated:** 2026-05-22

---

## Purpose

Read the world. Generate 5-8 themes that matter for the coming 4-12 weeks. Themes are *not* trade ideas. Themes are observations about how the world is shifting that *might* contain trades.

Two parallel lenses (journalist, macro analyst). They run independently — no shared context — and Stage 1 emits the union of their themes with attribution.

## Non-goals

- Not generating tickers (Stage 2's job).
- Not making forecasts. Themes are observations + supporting evidence, not predictions.
- Not personalized to any thesis or sector.
- Not trying to be original. The journalist lens should reflect what consensus is saying; the macro lens reflects positioning and flows. Originality emerges from the combination, not from either lens alone.

## The two lenses

**Lens A — Journalist.** Reads broadly across financial press, central bank communications, regulatory news, earnings transcripts, geopolitical events. Produces themes that describe *what's happening in the world*. Tone: descriptive, source-anchored. Bias to flag: recency (over-weighting last week's news).

**Lens B — Macro Analyst.** Looks at positioning, flows, rate expectations, dollar, credit spreads, volatility, factor performance, sector rotation. Produces themes that describe *how markets are positioned vs. what's happening*. Tone: structural, numbers-anchored. Bias to flag: contrarianism-for-its-own-sake.

Why two and not five: two distinct lenses with strong, opposed biases create useful tension. Five risks monoculture (LLM convergence) and noise (too many themes downstream).

## Inputs

A run trigger with a date range (default: last 7 days for journalist, last 30 days for macro). Both lenses have web search access.

```yaml
run_id: string
run_date: ISO date
journalist_window_days: 7
macro_window_days: 30
```

No prior-week context. Each Stage 1 run is fresh — see pipeline spec Open Question #1.

## Outputs

```yaml
run_id: string
run_date: ISO date
themes:
  - theme_id: string  # e.g., T-001
    lens: journalist | macro_analyst
    title: string  # ≤10 words
    summary: string  # 3-5 sentences
    direction_implication: long_bias | short_bias | mixed | neutral
    horizon: weeks | months
    evidence:
      - claim: string
        source: string  # publication / data provider
        url: string
        date: ISO date
    contradicting_evidence: string  # 1-2 sentences — what would invalidate this theme
    confidence: high | medium | low
lens_metadata:
  journalist:
    articles_consulted: int
    sources_diversity: int  # unique publications
  macro_analyst:
    data_points_consulted: int
    indicators_referenced: string[]
desk_note: string  # 2-3 sentences — what's the overall feel of the week
```

## Behavioral spec — both lenses

**Source diversity required.** Journalist lens: ≥4 distinct publications. Macro lens: ≥3 distinct data providers/indicators. Single-source themes get auto-downgraded to `low` confidence.

**Contradicting evidence is mandatory.** Every theme must include a `contradicting_evidence` field. If the lens can't write one, the theme is too weak to forward.

**No prediction language.** "X will happen" → reject. "X is happening / has happened / is being priced" → accept. Themes describe state, not forecasts.

**Numbers anchor claims.** A theme without at least 2 specific numbers (rates, flows, prices, growth %) gets downgraded.

**Symmetric direction.** Each lens must produce at least one `short_bias` or defensive theme per run if it produces ≥3 themes total. Pure long output is a red flag.

## Behavioral spec — journalist lens specifically

- Reads central bank statements, major financial press (FT, WSJ, Reuters, Bloomberg, Nikkei), regulatory filings, and earnings transcripts in the window.
- Themes describe shifts in *narrative* and *facts*, not opinions of pundits.
- Quotes editorial pages or strategists only as positioning indicators, never as evidence.
- Skews to themes with verifiable, recent triggers (a speech, a filing, a release).

## Behavioral spec — macro analyst lens specifically

- Looks at: rate curve (US, EU, JP), dollar index, credit spreads (HY, IG), VIX/MOVE, sector relative performance, factor returns, CFTC positioning, fund flows.
- Themes describe *divergences* between positioning and price, or *regime shifts* (e.g., correlation breakdown, volatility regime change).
- Distrusts narrative-only themes. Every macro theme cites at least one quantitative indicator with a current vs. baseline number.
- Comfortable with "nothing material this week" outputs. Macro shifts are rare.

## Verdict on the run

After both lenses run, an aggregator step produces the `desk_note`. The aggregator's only job is description: "Journalist found 4 themes, all long-biased, weighted toward AI infrastructure. Macro found 2 themes, both defensive, flagging credit spread widening." No interpretation. No "this is interesting."

## Anti-patterns the lenses must avoid

1. **Recency bias dominance** (journalist) — three themes all from the past 48 hours. Mitigation: lens prompt explicitly asks for at least one theme rooted in a multi-week shift.
2. **Contrarian-for-its-own-sake** (macro) — declaring every consensus theme wrong. Mitigation: lens prompt explicitly allows agreement with consensus when positioning supports it.
3. **Theme inflation** — producing 12 themes because more sounds better. Cap: 5 per lens. If a lens produces >5, it must pick its top 5.
4. **Source padding** — citing 10 sources for one theme to look thorough. Each source must add a distinct fact.
5. **Forecasting in disguise** — "rates will continue lower" reframed as "the market is pricing rates lower." Only valid if actual market-implied rate change is cited.

## Success metrics

- **Source diversity:** ≥4 unique pubs for journalist, ≥3 unique indicators for macro. Tracked weekly.
- **Direction balance:** over rolling 4 runs, share of long-bias themes ≤70%. If higher, lenses have drifted.
- **Theme survival to Stage 6:** of themes generated, what % eventually drive an action row? Target: 20-40%. Too low = themes too vague. Too high = funnel not filtering.
- **Contradiction quality:** monthly manual review — does the `contradicting_evidence` field actually contain real disconfirmation, or is it boilerplate? If boilerplate >30% of the time, the lens prompts need hardening.

## Out of scope for v0.1

- Sentiment scoring (subjective, low signal).
- Multi-language source coverage (English only for v0.1; French press later if useful).
- Real-time / intra-week runs. Weekly cadence only.
