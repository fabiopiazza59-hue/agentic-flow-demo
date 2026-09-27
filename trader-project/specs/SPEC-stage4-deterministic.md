# SPEC — Stage 4: Deterministic Checks (4a Math + 4b Bear Challenger)

**Status:** Draft v0.1
**Parent spec:** SPEC-pipeline-v0.2.md
**Last updated:** 2026-05-22

---

## Purpose

Two sub-stages running in parallel on each LIVE/WATCH scenario from Stage 3:

- **Stage 4a — Risk math.** Pure Python. No LLM. Position sizing, stop distance, R-multiples, correlation, max-loss per trade and at portfolio level. Numbers in, numbers out, deterministic.
- **Stage 4b — Bear challenger.** LLM with adversarial prompt. Receives the original scenario + 4a numbers. Job: make the strongest possible case *against* the trade.

Both run **without seeing Stage 3's reasoning**. Stage 3 said "LIVE" — Stage 4 doesn't care why. This isolation prevents 4b from being a rubber stamp.

## Non-goals

- Not making the decision (Stage 5).
- Not generating ideas (Stage 1+2).
- Not soft-judging the trade ("seems risky") — Stage 4a is numbers, Stage 4b is adversarial argument with cited evidence.

## Inputs

Stage 3 output: LIVE + WATCH scenarios only (KILL drops out of the funnel here, never re-enters). Plus a fetched current price per ticker (real-time at run time).

## Outputs

```yaml
run_id: string
enriched_scenarios:
  - scenario_id: string  # echo from Stage 3
    stage3_verdict: LIVE | WATCH
    current_price: number
    risk_math:                          # from Stage 4a — deterministic
      proposed_entry: number
      proposed_stop: number
      proposed_target: number
      stop_distance_pct: number
      target_distance_pct: number
      r_multiple: number                # (target - entry) / (entry - stop) for longs
      position_size_usd: number         # given account & risk-per-trade
      position_size_units: number       # shares/contracts
      leverage_required: number         # for CFD
      max_loss_usd: number              # at stop
      max_loss_pct_account: number
      correlation_with_open_positions: { ticker: float } | null
      portfolio_heat_after: number      # % of account at risk across all open positions
      cfd_overnight_cost_per_day: number  # estimated financing
      cfd_breakeven_days: int           # how many days until financing exceeds target gain
      passes_risk_floor: bool
      risk_floor_failures: string[]     # e.g., ["leverage_too_high", "r_multiple_below_2"]
    bear_case:                          # from Stage 4b — adversarial LLM
      thesis_against: string            # 3-5 sentences, strongest case against
      specific_risks:
        - risk: string
          evidence: string              # cited
          probability_assessment: low | medium | high
      historical_precedent: string      # last time this kind of trade failed, with date
      what_market_knows_that_we_dont: string
      recommended_size_haircut: number  # 0.0-1.0, fraction of proposed size to actually take
      bear_verdict: dangerous | acceptable_risk | strong_objection
```

## Stage 4a — Risk math behavioral spec

**Pure code, never LLM.** Implemented as Python functions called by Claude Code. No reasoning, no judgment. Given inputs, output is fully determined.

**Risk-per-trade floor.** `max_loss_pct_account` must be ≤ 2% of account. If position sizing required to hit a meaningful R-multiple would breach 2%, the trade fails risk floor.

**R-multiple floor.** R ≥ 2.0 to clear floor (set by user config). R < 2.0 → `passes_risk_floor: false`.

**CFD financing reality check.** This is the part retail systems skip. Compute:
- Daily financing cost = (notional × annual_rate) / 365
- Breakeven days = (target_distance_pct × notional) / daily_financing_cost
- If `proposed_horizon` (from Stage 2) exceeds breakeven_days × 0.6, flag — financing is eating the trade.

**Correlation check.** If account holds other open positions, compute pairwise correlation (90-day, daily returns) of proposed trade vs. open. If max correlation > 0.7 with an existing same-direction position, flag — concentration risk.

**Portfolio heat.** Sum of `max_loss_pct_account` across all open positions + the proposed one. Floor: 6% total heat. Above → `passes_risk_floor: false`.

**No LLM-judged inputs.** Entry/stop/target proposals come from Stage 2's `proposed_catalyst` + `proposed_kill` + a rule-based target calc (default: target = entry + 2.5 × stop_distance for long, symmetric for short, unless Stage 2 specified a price-level target).

## Stage 4b — Bear challenger behavioral spec

**Adversarial framing — explicit.** Prompt opens: "You are a short-selling research analyst tasked with making the strongest possible case against this trade. Your reputation is built on calling out bad longs (and bad shorts when you're given a short to attack). Do not be balanced. Do not hedge. Make the strongest argument the other side has."

**Web search mandatory.** The bear challenger must cite at least 3 distinct sources for its bear case. Pure-reasoning bear cases without evidence get flagged as `low quality` and the agent re-runs.

**Historical precedent required.** "Last time this kind of trade failed" — must be a real, dated, citable instance. If no precedent found, the bear says so explicitly ("no recent precedent — this may be a novel setup or I missed it") rather than fabricating one.

**No knowledge of Stage 3 reasoning.** The prompt receives Stage 2's scenario as if it were a fresh pitch, plus Stage 4a's numbers. It does *not* know that Stage 3 graded this LIVE or WATCH. This is the integrity safeguard.

**Three verdicts only.**
- `dangerous`: bear case is so strong the trade should be killed entirely (Stage 5 will weigh this heavily).
- `acceptable_risk`: bear case is real but priced/known; trade can proceed with normal sizing.
- `strong_objection`: bear case is significant but not killer; recommend sizing down.

**Size haircut.** If `bear_verdict` is `strong_objection`, the bear suggests a haircut (e.g., 0.5 = take half size). Stage 5 may override but must justify.

## How 4a and 4b interact

They don't. They run in parallel, both produce structured output, both go to Stage 5 as separate fields. Stage 5 reconciles. The reason for the separation: a quantitative system flagging a problem and an adversarial argument flagging the same problem are *two independent signals*. Combining them into one agent destroys that independence.

## Anti-patterns

**Stage 4a:**
1. Letting Claude "reason" about the numbers. The 4a output should be reproducible byte-for-byte from inputs. Pure function.
2. Hard-coding risk floors per ticker. Universal floors only (set in user config).
3. Approximating correlation when real data is available. If price history is fetchable, fetch it.

**Stage 4b:**
1. Diplomatic hedging. "There are some concerns…" → reject, re-run with stronger adversarial framing.
2. Fabricated precedents. Every historical precedent must have a verifiable date and citation.
3. Recycling generic risks (e.g., "valuation could compress" on every long). Each risk must be specific to the scenario.
4. Bearish-on-everything bias. The challenger must distinguish between "this trade has real risks" and "every trade has risks." `acceptable_risk` should be the modal verdict — if every scenario gets `dangerous` or `strong_objection`, the challenger is over-tuned.

## Tool use

- 4a: Python execution only. No web search, no LLM call.
- 4b: web search (mandatory ≥3 sources), price history fetch, no file writes except output.

## Success metrics

- **4a determinism:** same inputs → identical outputs, byte-for-byte. Tested via unit tests.
- **4a coverage:** 100% of LIVE+WATCH scenarios get risk math. No exceptions.
- **4b modal verdict:** `acceptable_risk` should be 40-60% of cases. If `dangerous` >40%, challenger is over-tuned. If `acceptable_risk` >75%, challenger is rubber-stamping.
- **4b precedent rate:** ≥70% of bear cases include a verifiable historical precedent. Below 50% means the challenger is hand-waving.

## Out of scope for v0.1

- Multi-asset hedging logic (Stage 5 may suggest hedges; 4 doesn't construct them).
- Greeks / options risk.
- Monte Carlo simulation of position outcomes (overkill at $2k account).
- Tax implications (CFD in France is BIC if active — handled outside the pipeline).
