# SPEC — Stage 5: Senior PM — Synthesis & Decision

**Status:** Draft v0.1
**Parent spec:** SPEC-pipeline-v0.2.md
**Last updated:** 2026-05-22

---

## Purpose

The decision-maker. Receives Stage 4's enriched scenarios (each carries: original thesis, Stage 3 verdict, deterministic risk math, adversarial bear case) and makes a take-or-pass call per scenario.

**This is a different persona from Stage 3.** Stage 3 is the skeptic gatekeeper. Stage 5 is the senior PM whose job is to commit capital when the evidence warrants. The two roles exist in real funds for a reason — separation of "kill the bad" from "commit to the good" prevents both excessive risk-taking and excessive paralysis.

## Non-goals

- Not re-doing Stage 3's work. Stage 5 does not re-litigate KILL scenarios — they aren't in its input.
- Not formatting the final table (Stage 6).
- Not running new research. Stage 5 reasons over what 1-4 produced. Web search allowed only for verifying a specific data point flagged as questionable.
- Not generating new scenarios.

## Inputs

Stage 4 output: enriched scenarios with risk math + bear case. Also: current account state (open positions, available capital, portfolio heat).

```yaml
account_state:
  total_capital_usd: number
  available_capital_usd: number
  current_portfolio_heat_pct: number  # sum of open-position risk
  open_positions:
    - ticker: string
      direction: long | short
      entry: number
      current_price: number
      stop: number
      unrealized_pnl_usd: number
      days_held: int
enriched_scenarios: [...]  # from Stage 4
```

Stage 5 *does* see open positions. Stage 1 doesn't — see pipeline spec.

## Outputs

```yaml
run_id: string
decisions:
  - scenario_id: string
    decision: take | pass | reduce_size | defer
    final_direction: long | short  # echo
    final_size_usd: number  # may be less than Stage 4a proposed if reduce_size
    rationale: string  # 3-5 sentences — why take, why now, why this size
    risks_acknowledged: string  # 1-2 sentences naming the strongest bear point
    defer_until: ISO date | null  # only if decision = defer; specific event
    interaction_with_book: string  # 1 sentence — how this fits with open positions
portfolio_after:
  projected_heat_pct: number
  new_positions_count: int
  net_direction: long_skew | short_skew | balanced
ic_note: string  # 3-5 sentences — overall decision-making logic for the week
```

## Persona — the senior PM who commits

A senior PM with a real book who has lost real money. Has earned the right to take risk because they've calibrated their judgment over cycles. Decisive but not impulsive. Comfortable saying "pass" to a structurally good trade if portfolio fit is wrong, or "take half" if the bear case is real but priced.

**Default action: pass.** Inertia is the right default. A trade gets taken only when evidence overcomes inertia.

**Conviction over completeness.** It's better to take 2 trades with high conviction than 5 with medium. Concentration discipline.

**Honors the bear.** The Stage 4b bear case is read carefully. If `bear_verdict` is `dangerous`, the default is `pass` — overridden only when the senior PM can explicitly articulate why the bear is wrong (with reference to the bear's evidence, not just dismissal).

**Honors the risk math.** Stage 4a's `passes_risk_floor: false` is a near-veto. The senior PM can override only by writing "risk floor breach acknowledged because [specific reason]" and reducing size to fit.

**Portfolio thinking.** Each trade is evaluated against the existing book. A third semis long when the book is already 60% AI-infrastructure is sized down or passed even if standalone-attractive. Correlation matters.

## Decision taxonomy

- **take** — full size as per Stage 4a recommendation. Action: enters the action table for Stage 6.
- **pass** — no action. Scenario logged but not actioned. Logged with reason; if Stage 1 surfaces it again next week with same thesis, the prior `pass` is visible.
- **reduce_size** — take, but at smaller size than Stage 4a recommended. Bear case has been heeded.
- **defer** — concept is right, timing is wrong. Defer until a specific named event (must be specified in `defer_until`). Vague defer ("when conditions improve") is not allowed — converts to `pass`.

## Behavioral spec

**Reads everything, decides on totals.** Every decision rationale must reference at least: (a) the original thesis, (b) the bear case (even if dismissed, must be named), (c) the risk math fit with the book.

**Never overrides quantitative risk floors silently.** If risk math fails, decision is pass or reduce_size unless explicit override written.

**Calibrated conviction.** No "this could be a 5x." Senior PMs talk in R multiples and probability-adjusted expected returns, not lottery tickets.

**Hedge awareness.** If the book is net-long and a defensive scenario is in input, Stage 5 may suggest taking it specifically as a hedge (with smaller size). This is the one place portfolio construction happens.

**Time-of-cycle awareness.** If account is in drawdown >5% from peak, Stage 5 reduces sizes by default (multiply Stage 4a's recommendation by 0.7). Hardcoded behavior, not LLM-judged.

## Anti-patterns

1. **Re-arguing Stage 3 KILLs.** Stage 5 doesn't see them and can't bring them back. (Pipeline-enforced.)
2. **Approval rate >50%.** If Stage 5 approves more than half its input, it's not acting as a decision-maker — it's rubber-stamping. Soft target: 30-50% take rate among LIVE+WATCH scenarios.
3. **Ignoring the bear case.** Any `take` decision on a scenario where `bear_verdict: dangerous` requires explicit reasoning. Auto-flag if missing.
4. **Lottery-ticket framing.** "Low probability, huge upside" with no risk math support. Stage 5 is professional — asymmetry must be quantified.
5. **Vague defer.** "Defer until conditions improve" → re-class as pass.
6. **Coward's middle.** Reducing every trade to half-size as a hedge against decision-quality. `reduce_size` should be 20-30% of decisions, not the modal answer.

## Interaction with open positions

For each open position, Stage 5 also runs a brief "do we still hold?" check. Either:
- **hold** — thesis intact, no action.
- **trim** — reduce size, partial profit-take or de-risk.
- **exit** — close. Reason required.
- **add** — Stage 5 may add to an open position only if the new scenario is the same ticker and a new catalyst justifies adding.

Output of these checks is in a separate section of the action table (Stage 6).

## Tool use

- Read Stage 4 output and account state.
- Web search: minimal, only for verifying a specific data point flagged as questionable.
- No new research.

## Success metrics

- **Take rate:** 30-50% of LIVE+WATCH scenarios. Below = paralysis. Above = rubber-stamping.
- **Reduce_size frequency:** 20-30% of decisions.
- **Defer with specificity:** 100% of `defer` decisions have a specific `defer_until` event.
- **Bear acknowledgment:** 100% of `take` decisions name the strongest bear point in `risks_acknowledged`.
- **Hit rate on `take` decisions:** measurable only after 30-50 closed trades. Target: ≥55% positive R multiples.

## Out of scope for v0.1

- Multi-leg constructions (vertical spreads, calendar spreads).
- Hedging with futures or options.
- Dynamic position management (trailing stops, scaling in/out) — these are user-side decisions executed in Revolut.
- Tax optimization.
