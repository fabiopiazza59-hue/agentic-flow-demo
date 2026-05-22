# SPEC — Stage 6: Action Table

**Status:** Draft v0.1
**Parent spec:** SPEC-pipeline-v0.2.md
**Last updated:** 2026-05-22

---

## Purpose

The user-facing output. Takes Stage 5's decisions and renders the single artifact Fabio actually reads: a table of trades to enter, plus a section of actions on open positions.

**This stage does no judgment.** It is a deterministic formatter. If Stage 6 changes a recommendation, the pipeline is broken — Stage 5 is the last decision point.

## Non-goals

- Not adding analysis.
- Not summarizing the bear case in a "softer" way to make trades look better.
- Not generating new ideas.
- Not running searches or fetches.

## Inputs

Stage 5 output (`decisions[]`, `portfolio_after`, `ic_note`, and the open-position checks).

## Outputs

Two artifacts:

### Artifact 1 — `actions.md` (the human read)

A single markdown file. One screen on mobile. The format Fabio reads on Sunday evening and uses on Monday.

```markdown
# Weekly Action Table — <YYYY-MM-DD>

## New trades

| Ticker | Action | Entry | Stop | Target | Size $ | R | Reason |
|---|---|---|---|---|---|---|---|
| MU | BUY | $268.50 | $245.00 | $325.00 | $400 | 2.4 | HBM ramp + Q4 catalyst June 24; bear: hyperscaler capex normalization (priced) |
| KRE | SELL (short) | $48.20 | $51.50 | $39.00 | $250 | 2.8 | CRE refi wall Q3; bear: yields collapse on Fed cut (low prob within horizon) |

## Position management

| Ticker | Held since | Current P&L | Action | Reason |
|---|---|---|---|---|
| CCJ | 2026-04-12 | +$45 | HOLD | NRC catalyst still ahead, thesis intact |
| TSM | 2026-03-08 | +$120 | TRIM (50%) | At target band; bank profit, let half run |

## Portfolio after

- Heat: 4.8% of capital (was 3.1%)
- New positions: 2
- Net direction: balanced (1 long, 1 short)

## IC note

<copied verbatim from Stage 5 — the senior PM's logic for the week>

---

*Run ID: <id> | Pipeline version: v0.2 | Generated <timestamp>*
```

### Artifact 2 — `actions.yaml` (the machine read)

Same content, structured. For future automation (alert setting in Revolut, journal logging, performance tracking).

```yaml
run_id: string
generated_at: ISO timestamp
pipeline_version: "0.2"
new_trades:
  - ticker: string
    action: BUY | SELL_SHORT
    entry_price: number
    stop_loss: number
    target_price: number
    size_usd: number
    r_multiple: number
    reason: string  # ≤150 chars, includes catalyst + bear
    scenario_id: string  # backlink
position_management:
  - ticker: string
    held_since: ISO date
    current_pnl_usd: number
    action: HOLD | TRIM | EXIT | ADD
    action_size_pct: number | null  # only for TRIM or ADD
    reason: string  # ≤100 chars
portfolio_after:
  heat_pct: number
  new_positions_count: int
  net_direction: long_skew | short_skew | balanced
ic_note: string
```

## Behavioral spec — formatting rules

**One row per decision.** No grouping by theme, no nesting. Flat table.

**Numbers, not adjectives.** Entry / Stop / Target are explicit prices. Size is dollars. R is a number. No "moderate" / "small" / "tight."

**Reason field is ≤150 characters.** Forced compression. Must include: catalyst (when), bear case (one phrase). Format: "<catalyst>; bear: <bear point> (<prob phrase>)".

**Sort order:** new trades sorted by conviction descending (highest R × Stage 5 confidence first). Position management sorted by held-since ascending (oldest first).

**Mobile-readable.** Max 4 new trades, max 6 position-management rows. If Stage 5 produces more, Stage 6 truncates by conviction and adds a note. (In practice this won't happen often — the pipeline narrows aggressively.)

**No emoji, no color coding.** Plain markdown table. Renders identically in Notion, Slack, Apple Notes.

**No prose preamble in the artifact.** The table is the artifact. The IC note is the only prose, and it comes after the tables, not before.

## What Stage 6 may not do

1. **Re-phrase a `take` as a `reduce_size`** (or vice versa). Reproduce Stage 5's decision exactly.
2. **Add caveats** ("this is high risk" / "consider waiting for confirmation"). Caveats belong in the `reason` field, not as new content.
3. **Suggest position sizes** different from Stage 5. Size is set upstream.
4. **Omit a `take` decision** because it looks unflattering. All Stage 5 decisions render.
5. **Add disclaimers** ("not financial advice", etc.) — those are project-level, not per-run.

## When the table is empty

If Stage 5 produces zero `take` decisions, Stage 6 produces:

```markdown
# Weekly Action Table — <YYYY-MM-DD>

**No new trades this week.**

## Position management

<table if any open positions need action; otherwise "All open positions: HOLD">

## IC note

<Stage 5's ic_note, which will explain why nothing was taken>
```

Zero-trade weeks are a valid output and the pipeline should be comfortable producing them.

## File outputs

- `journal/stage6/<YYYY-MM-DD>_<run_id>.md` — human read
- `journal/stage6/<YYYY-MM-DD>_<run_id>.yaml` — machine read
- Optionally copied to: `latest.md` (a stable filename pointing to the most recent action table)

## Tool use

- File read (Stage 5 output) and file write (the two artifacts).
- No LLM judgment, no web search, no math.

## Success metrics

- **Fidelity:** every Stage 5 decision appears in Stage 6, byte-identical in numbers. Tested with diff.
- **Length:** action table renders in one mobile screen ≥95% of weeks. If chronically over, narrow Stage 5 output.
- **Read latency:** Fabio confirms after 4 runs that the table is the format he actually uses. If he ends up reformatting or excerpting it, this spec needs revision.

## Out of scope for v0.1

- Push notifications when the table is generated.
- Direct integration with Revolut (alert setting, order pre-staging).
- Multi-format output (PDF, image card for messaging).
- Per-trade audit links back to upstream stage outputs (would need a UI layer; v0.2 leaves the journal as plain files for now).
