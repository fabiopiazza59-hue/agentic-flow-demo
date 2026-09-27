# SPEC — Pipeline v0.2: Six-Stage Trading Advisor

**Status:** Draft v0.2 (supersedes v0.1 single-stage spec)
**Owner:** Fabio
**Last updated:** 2026-05-22

---

## What changed from v0.1

v0.1 specified only Stage 2 (a senior trader gate) in isolation. v0.2 specifies the full six-stage funnel. The old senior-trader spec is preserved as the **Stage 3** spec; everything else is new.

## The funnel — at a glance

| # | Stage | Role | Output |
|---|---|---|---|
| 1 | Market scan | Journalist + macro analyst lenses — read the world | Top-down themes with evidence |
| 2 | Ticker discovery | Map themes → tradeable instruments | Candidate scenarios per theme |
| 3 | Triage gate | Skeptical senior PM — KILL most | LIVE/WATCH/KILL verdicts |
| 4 | Deterministic checks | Pure math, no LLM judgment | Risk numbers + adversarial bear pass |
| 5 | Synthesis call | Senior PM — decision-maker, not skeptic | Take / pass + rationale |
| 6 | Action table | Format only — what to do | Entry / Stop / Target / Reason |

**Why six and not three:** each stage has *one job*. The fastest way to lose trust in an LLM pipeline is to ask one agent to do two jobs (e.g., research + judge). Separation of concerns is the whole point.

**Why this order:** divergent → convergent → deterministic → convergent → format. Each filter narrows the funnel. Information flows one way; later stages cannot pollute earlier ones.

## Cardinality through the funnel — what to expect per weekly run

```
Stage 1:  ~5-8 themes
Stage 2:  ~20-30 ticker-scenarios (3-5 per theme)
Stage 3:  ~4-7 LIVE + ~3-5 WATCH (rest KILL)
Stage 4:  same as Stage 3 LIVE+WATCH, annotated with risk math + bear case
Stage 5:  ~2-4 take / ~3-5 pass with rationale
Stage 6:  ~2-4 action rows
```

If a run produces zero action rows, that's a valid output. Quiet weeks exist. The system must be allowed to say "nothing today."

## The three design questions resolved in v0.2

These were flagged but not resolved in the v0.1 draft. v0.2 commits to a position; revisit after 50 scenarios of data.

**Q1: Two "senior trader" stages — what's the difference?**

*Stage 3 is the skeptic.* Its job is to kill. It defaults to KILL. It does not commit capital. It is the desk's gatekeeper.

*Stage 5 is the decision-maker.* It receives only what survived Stage 3 plus the Stage 4 numerical work. Its job is to commit. It defaults to *pass* (do nothing), but when it acts, it acts with conviction and writes a position-level rationale.

Same seniority, different role. Different personas, different prompts, different temperature.

**Q2: Where does the bear/short challenge live?**

Inside Stage 4, as a parallel sub-agent to the risk math. Stage 4 is "deterministic" overall but contains two sub-stages: (4a) pure math (Python, no LLM) and (4b) bear challenger (LLM with adversarial prompt). The bear challenger runs *without* seeing Stage 3's reasoning — it gets the original Stage 2 scenario and the Stage 4a numbers, and is told to make the strongest case against the trade. This isolation prevents it from being a rubber-stamp.

**Q3: Stage 2 — is it ticker generation or stock scanning?**

Ticker generation. Stage 2 takes Stage 1's themes and proposes specific tradeable instruments. "Scanning" implies screening a universe by criteria, which is closer to a Stage 4 risk task. Stage 2 is creative: given the theme "nuclear renaissance accelerates," propose the picks-and-shovels expressions (CCJ, CEG, NNE, OKLO, etc.) with reasoning per pick.

## Information contract between stages

Every stage has a strict YAML I/O contract. Stages don't pass prose. Stages don't pass conversation. They pass structured records.

```
Stage 1 → Stage 2:  themes[]
Stage 2 → Stage 3:  scenarios[]
Stage 3 → Stage 4:  verdicts[] (LIVE + WATCH only forwarded)
Stage 4 → Stage 5:  enriched_scenarios[] (verdict + risk_math + bear_case)
Stage 5 → Stage 6:  decisions[]
Stage 6:             actions[] (final user-facing table)
```

A stage that can't produce its contract from its input emits an empty list and logs why. Stages do not invent data to fill the contract. This is the #1 anti-pattern.

## Auditability

Every stage writes its output to `journal/stage<N>/<YYYY-MM-DD>_<run_id>.yaml`. Never overwrites. The full run is reproducible from the journal alone. If something looks wrong in Stage 6, you can `diff` last week's Stage 3 output against this week's and see exactly where the funnel diverged.

## Failure modes the architecture prevents (or doesn't)

**Prevented by design:**
- *Confirmation bias propagation:* Stage 5 doesn't see Stage 1's themes, only Stage 4's enriched scenarios. The decision-maker doesn't know which theme generated the trade. Reduces theme-loyalty bias.
- *Senior-trader sycophancy:* Stage 3 is stateless. No memory of past pushback. Each run independent.
- *Long bias:* Each stage gets an explicit symmetric-direction instruction. Stage 1 macro lens must produce at least one defensive/short theme per run. Stage 4 bear challenger is mandatory.
- *Catalyst fabrication:* Stage 3 hard-rule: vague catalyst → KILL. Stage 5 cannot un-KILL.

**Not prevented (known risks):**
- *Stage 1 monoculture:* if both Stage 1 lenses converge on the same themes, the funnel narrows too early. Mitigation: track Stage 1 diversity metric over time. Lenses must be redesigned if they consistently agree.
- *Risk math gaming:* if Stage 5 learns that high R-multiples auto-approve, Stage 1/2 may inflate R. Mitigation: Stage 4a uses fixed methodology; Stage 5 prompt explicitly distrusts R >5:1.
- *Pipeline complexity:* six stages × auditability × eventual UI = real engineering. Realistic build cost: 40-80 hours including tuning. Worth being honest about this before starting.

## Build order

1. **Stage 3 first** (already drafted in v0.1) — the gate. Until the gate works, nothing else matters.
2. **Stage 4a** (deterministic risk math) — pure Python, easiest to validate.
3. **Stage 6** (action table) — pure formatter, no LLM judgment. Decouple format from decision early.
4. **Stage 1 lenses** — parallel, two lenses for v0.2 (journalist, macro analyst). Add more later if needed.
5. **Stage 2** (ticker discovery) — single agent, depends on Stage 1 working.
6. **Stage 4b** (bear challenger) — adversarial agent.
7. **Stage 5** (synthesis decision-maker) — last because it depends on everything upstream being trustworthy.

Tempting order would be 1→2→3→4→5→6 (left to right). That order is wrong because if Stage 3 (the gate) is broken, everything upstream is wasted effort. Build the highest-leverage stage first, then build the others to feed it.

## Open questions (v0.2)

1. **Should Stage 1 see the prior week's Stage 6 actions?** Arguments for: continuity, "are open trades still valid given new themes?" Arguments against: contaminates fresh-eyes reading of the world. Default: **no** for v0.2. Stage 5 may see open positions; Stage 1 may not.
2. **Stage 5 — one persona or a panel?** A real IC has 3-5 voices. A panel here means more API cost and longer latency. Default: **one senior PM** for v0.2; add a co-PM if v0.2 outputs feel one-dimensional.
3. **How often does the funnel run?** Default: **weekly (Sunday evening).** Stage 4 alone could run daily as a position-check, but full funnel weekly. Costs scale linearly; weekly = ~$10/month all-in.
