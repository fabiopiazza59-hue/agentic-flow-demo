# Seamaster — Trading Advisor

A Claude Code project that operates as a six-stage trade advisor funnel. Spec-driven: every stage has a SPEC.md in `specs/` defining purpose, I/O, behavior, success metrics, and anti-patterns before any prompt code is written.

## The pipeline

| # | Stage | One-line role | Spec |
|---|---|---|---|
| 1 | Market scan | Journalist + macro analyst lenses generate themes | [`specs/SPEC-stage1-market-scan.md`](specs/SPEC-stage1-market-scan.md) |
| 2 | Ticker discovery | Translate themes into 3-5 ticker scenarios each | [`specs/SPEC-stage2-ticker-discovery.md`](specs/SPEC-stage2-ticker-discovery.md) |
| 3 | Triage gate | Skeptical senior PM kills most scenarios | [`specs/SPEC-stage3-triage-gate.md`](specs/SPEC-stage3-triage-gate.md) |
| 4 | Deterministic checks | 4a risk math (Python) + 4b adversarial bear (LLM) | [`specs/SPEC-stage4-deterministic.md`](specs/SPEC-stage4-deterministic.md) |
| 5 | Synthesis decision | Senior PM (decision-maker, not skeptic) commits | [`specs/SPEC-stage5-synthesis-decision.md`](specs/SPEC-stage5-synthesis-decision.md) |
| 6 | Action table | Formatter — Entry/Stop/Target/Reason | [`specs/SPEC-stage6-action-table.md`](specs/SPEC-stage6-action-table.md) |

Master spec: [`specs/SPEC-pipeline-v0.2.md`](specs/SPEC-pipeline-v0.2.md).

## Project structure

```
trader-project/
├── CLAUDE.md
├── specs/           # Spec files — contracts before code
├── src/
│   ├── config/      # User config (account size, risk floors, etc.)
│   ├── models/      # YAML data models (Pydantic)
│   ├── stages/      # One module per pipeline stage
│   └── pipeline.py  # Orchestrator — runs stages in sequence
├── fixtures/        # Sample data for testing stages independently
├── journal/         # Audit trail — stage<N>/<date>_<run_id>.yaml
└── requirements.txt
```

## Cardinality through the funnel

```
Stage 1:  ~5-8 themes
Stage 2:  ~20-30 ticker-scenarios
Stage 3:  ~4-7 LIVE + ~3-5 WATCH
Stage 4:  same as Stage 3 LIVE+WATCH (enriched)
Stage 5:  ~2-4 take + ~3-5 pass
Stage 6:  ~2-4 action rows (often zero — that's valid)
```

## Core conventions

- **Specs before prompts.** Every agent has a SPEC.md that defines the contract. Spec changes drive prompt changes, not the reverse.
- **All outputs auditable.** Agent outputs go to `journal/stage<N>/<date>_<id>.yaml`. Never overwrite.
- **Stateless agents.** No agent remembers past runs. Persistence through the file system. Independence per invocation is the integrity safeguard.
- **Two distinct senior personas.** Stage 3 = skeptic gatekeeper, defaults to KILL. Stage 5 = decision-maker, defaults to pass but commits with conviction. Different prompts, different temperatures, different roles.
- **Deterministic where possible.** Stage 4a is pure Python — no LLM. Stage 6 is pure formatter — no LLM. Reserve LLM for genuine judgment (1, 2, 3, 4b, 5).
- **Information flows one way.** Earlier stages cannot see later outputs. Stage 5 cannot un-KILL a Stage 3 decision. Stage 1 cannot see open positions.

## Build order

Build the highest-leverage stage first, not left-to-right.

1. **Stage 3** — the gate (drafted).
2. **Stage 4a** — pure Python risk math; easiest to validate.
3. **Stage 6** — pure formatter; decouple format from decision early.
4. **Stage 1** lenses.
5. **Stage 2** — depends on Stage 1.
6. **Stage 4b** — adversarial bear.
7. **Stage 5** — last; depends on everything upstream being trustworthy.

## Validation plan

Each stage validated independently before chaining. Stage 3 against the sample fixture; 4a and 6 against unit tests; 1, 2, 4b, 5 against real market state with 4 weeks of manual review.

## What this is not

- Not a signal service.
- Not personalized to one thesis (Operation Seamaster was an earlier framing — dropped).
- Not real-time.
- Not a P&L generator. Realistic payoff is fewer bad trades.
- Not financial advice. Output is decision-support; accountability rests with Fabio.
