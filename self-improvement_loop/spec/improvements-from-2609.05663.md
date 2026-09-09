# Improvements from arXiv:2609.05663

**Paper:** *What LLM Trading Agents Actually Do in Production: A Six-Month, Population-Scale Record
from Two Fleets* — Barton et al., DX Research Group, September 2026.
**Reviewed against:** this loop at commit time, arm A `n=60` scored non-seed days
(2026-06-08 → 2026-09-04), arm B `n=35`, 35 paired days.
**Reproduce every number below:** `python -m tools.audit_review`

---

## 0. Why this paper applies to this repo

The paper is the closest published thing to what this folder is: LLM agents making
market decisions on a schedule, measured honestly, with the null results reported. It studies
3,505 real-capital vaults and a 500–599-agent fleet over ~6 months, and its central claim
transfers directly:

> *"The operating layer (slider constraints, rendered candidate lists, order-path mechanics)
> determines behavior more than anything written in strategy text."*

and its working principle:

> **"Mechanism over exhortation.** Every prompt-side attempt to fix a behavior that lives in the
> operating layer (forced restatement, added context, stronger wording) underperforms a one-line
> change to the order path or the render."*

This loop's improvement mechanism is almost entirely exhortation. `learnings/FAILURES.md` contains
~15 post-mortems, each ending in "the one change to try" — and essentially all of them are prompt
instructions to the meta-judge ("boost contrarian weight to 0.30 when RSI<25", "cap the crowd's
weight", "force the dissenting view to at least 30%"). Only two ever became code
(`src/evals/gates.py`). The paper predicts, with a measured precedent, that the prompt-side ones do
not work: its strongest prompt lever — forcing the model to compute and state its liquidation
distance before entry — not only failed to change sizing, the agents that stated it liquidated
*more* often (5.8% vs 1.2%), "because stating marks aggressive intent rather than restraining it."

The rest of this document applies the paper's findings and its 17-rule methodology canon to
measurements of this repo's own ledger. Findings carry the paper's evidence classes: **FIRM**
(day-level paired test, interval excludes zero, sign test agrees) or **PROVISIONAL**.

---

## 1. Findings

### F1 — The meta-judge is destroying accuracy. **FIRM**

Paper §6.1, *the capture gap*: 43.2% of positions reached ≥+300 bps of favorable excursion, and
49.3% of those closed negative — "the market state predicts what the position will *do*, and it
does not predict what the agent will *keep*." The analogue here is the gap between what the
analysts produce and what the pipeline ships.

| Aggregation over the same 5 analyst numbers | MAPE | PASS |
|---|---|---|
| **meta-judge blend (shipped)** | **1.613%** | **43%** |
| equal-weight mean of the analysts | 1.472% | 48% |
| median of the analysts | 1.569% | 45% |
| random-walk baseline | 1.569% | 50% |
| oracle best analyst (ex-post, unachievable) | 0.763% | 80% |

Paired over days (the day is the inferential unit — canon rule 1):

- **shipped − equal-mean: +0.1413% MAPE, 90% bootstrap CI [+0.0649%, +0.2254%], equal-mean wins
  41/60 days, sign test p = 0.006.** Replacing the Opus meta-judge with `sum(preds)/len(preds)`
  improves MAPE by ~9% relative and PASS by 5 points.
- On 23 of 60 days (38%) at least one analyst was inside ±1% and the blend still FAILED.
- The blend beat its own best analyst on 9 of 60 days.

Honest caveat: equal-mean vs the random walk is +0.097% but CI [−0.0004%, +0.2006%], 31/60 days,
p = 0.897 — **PROVISIONAL, and not an edge.** Removing the meta-judge does not create skill; it
stops destroying it. That is exactly the paper's §8.4 result — decision quality across frontier
models was statistically indistinguishable (min p = 0.46) while cost differed 25× — pointed at
this repo's most expensive component.

### F2 — Conviction decay: each layer shrinks the move toward the baseline. **FIRM**

| Source | median \|predicted move\| |
|---|---|
| contrarian | 0.834% |
| news | 0.798% |
| macro | 0.652% |
| technical | 0.545% |
| momentum | 0.460% |
| → equal-weight mean of all five | 0.349% |
| → **meta-judge blend (shipped)** | **0.167%** |
| → shipped, on the 20 days a gate fired | 0.103% |
| **actual median daily move** | **0.976%** |

The individual analysts predict moves of roughly the right scale. Averaging cancels them to ~35%
of a typical day; the meta-judge halves that again; the gates halve it again when they fire. **The
shipped prediction is 0.17× a typical day's move.** A forecast that is structurally a 17%-scaled
random walk *cannot* beat the random walk by more than rounding, which is precisely what the
scoreboard shows (all-time edge −0.044%). The architecture, not the prompts, is the binding
constraint.

### F3 — Sizing is scale-blind. **PROVISIONAL** (paper §5.1 analogue, partial)

The paper's largest single defect was volatility-blind sizing: median leverage 5.0× in *every*
volatility sextile, Spearman(vol, leverage) = −0.001. Here the relationship is weakly positive —
Spearman(trailing vol, |predicted move|) = **+0.214**, median move 0.133% in the calm half vs
0.271% in the wild half — so this loop is *not* volatility-blind in the paper's strict sense. But
the level is wrong by ~5×: 0.13–0.27% predicted against ~1% realized. And the calm/wild ratio is a
side effect of the gates, not of any explicit σ term; `src/evals/gates.py` shrinks by a **fixed**
`GATE_SHRINK = 0.5` regardless of volatility, and nothing in the pipeline ever scales the move
*up*. The paper's prescription — "leverage scaled to realized volatility at the tool rather than
requested in text" — maps onto: scale the predicted move to realized σ in code.

### F4 — The self-improvement mechanism produces no measurable reweighting. **FIRM**

`src/agents/meta_judge.py` calls scorecard reweighting "the loop's core self-improvement
mechanism," and `README.md` bills the scorecards as the loop's learning signal. Over 60 days the
meta-judge's assigned weights are effectively frozen and near-uniform:

| period | technical | momentum | contrarian | news | macro | spread |
|---|---|---|---|---|---|---|
| first 20d | 0.21 | 0.20 | 0.14 | 0.25 | 0.15 | 0.11 |
| mid 20d | 0.19 | 0.17 | 0.13 | 0.25 | 0.10 | 0.15 |
| last 20d | 0.20 | 0.19 | 0.17 | 0.25 | 0.12 | 0.13 |

`news` sits at 0.25 for all three periods; the max−min spread never exceeds 0.15 and does not
trend. The cause is in `src/evals/scorecard.py:70-83`: `weight_hints` normalizes **inverse
cumulative all-time MAPE**. After 60 observations of five strategies whose MAPEs span
1.45%–1.70%, inverse-MAPE weights mathematically converge to ~0.19–0.22 and stop moving. The
tracked `hit_rate` (16%–27%, a 1.6× spread — the only discriminating statistic in the file) is
computed and never used. This mirrors paper §4.1: "agent fixed effects absorb 60% of variance" —
the configured identity, not the learning, is doing the work.

### F5 — Confidence is worse than a constant. **FIRM**

Brier 0.2524 shipped, vs 0.2456 for a constant equal to the base rate (0.43) and 0.2500 for a
constant 0.5. It does have discrimination (PASS 0.52 in the high-confidence half vs 0.33 in the
low half), so the ranking is real and only the *scale* is wrong — `calibrated_confidence`
(`src/evals/gates.py:35-44`) anchors on the rolling PASS rate but adds a ±0.1 consensus nudge that
overshoots. Paper canon 13: calibrate against a null arm before reading the table.

### F6 — Same-session contamination, and the headline verdict rests on it. **FIRM**

Canon 9 ("check the calendar footprint before believing any group difference" — this overturned
two of the lab's own claims within an hour) and canon 17 ("know your timing-luck floor").

- **18 of 60 predictions were stamped after the 13:30 UTC open**, i.e. not pre-open at all. Two
  (2026-08-27 at 21:08Z, 2026-08-28 at 21:29Z) were created *after the 20:00 UTC close*.
- On those two days `prior_close == actual_close` exactly and `baseline_ape == 0.0`: the "previous
  close" the row was scored against was the target session's own close. Both rows are in the
  headline metrics and are not flagged as seeds.
- Effect on the repo's one stated success criterion:

  | rolling-20 edge vs baseline | value |
  |---|---|
  | as reported in `RESULTS.md` / `metrics.json` | **+0.0014%** |
  | with the two contaminated rows removed | **−0.0149%** |

  The published "beats_baseline_overall: true" is carried by two rows that could not have been
  predictions. The true effect size (+1.4e-5) is also ~2 orders of magnitude below the daily noise
  in this series, which is canon 17's point about most compared effect sizes sitting under the
  timing-luck floor.

Related: `learnings/FAILURES.md` for 2026-08-31 and 2026-09-01 diagnoses that "news anchors to a
real-time intraday quote already >1.5% off prior close" and prescribes **raising** news's weight to
≥0.45 on that basis. That is a prescription to lean harder on lookahead. Canon 8: agent text never
promotes an experiment.

### F7 — The analysts are less redundant than the post-mortems assume. **FIRM**

Several post-mortems blame "false consensus… one trend signal counted four times" and prescribe a
correlation penalty. Measured: mean pairwise correlation of predicted *move* across the five
analysts is **+0.093**, and all five agreed on direction on only **6 of 60 days (10%)**. The
crowding diagnosis is mostly wrong — the analysts disagree, the aggregation cancels them (F2), and
then a post-mortem reads the flat output as groupthink. Building the prescribed correlation-penalty
gate would be optimizing a phantom.

---

## 2. Recommendations, in the paper's own priority order

The paper orders its development program by effect-size class: harness levers first (FIRM, large,
cheap), then deployable checks, then model work, with benchmarking throughout. Same ordering here.

### P0 — Harness fixes. Cheap, mechanical, measured.

**P0.1 Log the counterfactual so the mechanism becomes measurable.**
The gates are the only true mechanism in this system and their effect is currently *unknowable*:
zero of 60 rows record the pre-gate prediction. The paper could state "the bracket earns +39.0 bps
per position [+21.3, +56.5]" only because both arms of every entry were recoverable.
→ In `src/evals/gates.py:apply_gates`, return the ungated value too, and persist
`predicted_close_raw` on every ledger row. Then report the gate's paired day-level effect in
`report.py` next to the headline. Any future gate must clear this bar before it stays.
*Verify:* a `gate_effect` block in `metrics.json` with a bootstrap CI over days.

**P0.2 Make the pre-open guarantee real, and quarantine the contaminated rows.**
→ In `run_daily.do_predict` / `run_ab`, refuse to write a prediction row whose timestamp is at or
after the session open (13:30 UTC), exiting cleanly the way the non-trading-day guard does.
→ Add an assertion at score time that `prior_close != actual_close` for the target session, and
mark 2026-08-27 and 2026-08-28 `contaminated: true` so `aggregate()` excludes them the way it
excludes seeds. Restate the headline honestly — the rolling edge is −0.015%, not +0.001%. This is
the paper's own retraction discipline (§2.4), and doing it in public is the point of the repo's
"honesty over vanity" principle.
*Verify:* `metrics.json` edge changes sign; `RESULTS.md` says "no edge yet."

**P0.3 Demote the meta-judge to a bounded adjuster; make the mean the default.**
F1 is a measured, day-clustered, sign-test-confirmed result that the LLM synthesis layer costs
0.14% of MAPE. Do not delete it — bound it, the way arm B already bounds its decider (that pattern
is already in this repo and is the right one):
→ compute `blend = mean(analyst_closes)` in code; let the judge propose an adjustment in units of
realized σ; clamp it in code to ±0.5σ; record both. The judge's job stops being "produce a number"
and becomes "size a deviation," which is the only part it can be scored on.
*Verify:* the clamped arm should reproduce equal-mean's 1.472% MAPE as a floor; the paired
judge-vs-mean delta is now a first-class metric in `ab_compare.json`.

**P0.4 Scale the move to realized σ at the code layer.**
→ After blending, rescale the predicted move so its expected magnitude matches `k · σ_20d` with
`k` a registered constant (start `k = 0.5`, pre-register the sweep), instead of leaving it at 0.17×
a typical day. Shrink toward the prior close only through the existing gates, and make
`GATE_SHRINK` σ-relative rather than a flat 0.5.
*Note:* this will likely **raise** MAPE while making direction meaningful. Decide which metric is
primary *before* running it (canon 2), or the result is unreadable.

**P0.5 Fix the dead weighting signal.**
→ In `scorecard.py`, window the MAPE (last 20 obs, not cumulative), use the discriminating
`hit_rate`, and give the weights a temperature so they can actually separate. Or delete
`weight_hints` and admit it is decorative — either is better than shipping an inert mechanism
described as the core of the loop.
*Verify:* the F4 spread table should show a widening, trending spread; if it does not, the
mechanism is null and should be removed (paper §8.2's utility bar: "a tool surface stays in the
manifest only if it demonstrably changes behavior").

### P1 — Measurement discipline. Nothing above is readable without it.

**P1.1 Add a null arm (canon 13).**
"Byte-identical templates produced ~$3.5K of spread; anything smaller than your null arm's spread
is noise." Arm B currently leads arm A by 0.08% mean daily APE with sign-test p = 0.736 — and
nobody knows whether 0.08% is even distinguishable from running the same arm twice.
→ Add **arm A′**: arm A's exact pipeline on the same snapshot with a different sampling seed. Its
A-vs-A′ spread is the noise floor. Print it in `report_ab.py` above the A/B verdict and refuse to
render any A/B verdict whose effect is smaller than it.

**P1.2 Stop reading a rolling p-value (canon 15).**
`report_ab.generate()` recomputes and re-renders the sign-test verdict every single day. "One of
ours ran 0.0067 → 0.19 → 0.0277." `AB_MIN_PAIRED_DAYS = 10` is a floor, not a stopping rule.
→ Pre-register a horizon (e.g. 60 paired days) in `spec/`, show the daily number as descriptive
only, and render a verdict once at the horizon.

**P1.3 Measure the timing-luck floor (canon 17).**
The paper found ±15-minute schedule offsets producing −88 vs +21 dollars on identical contracts.
→ Record the quote's own timestamp (not just `quote_source`) on every row, and run the same
snapshot at two offsets for a week to bound how much of the A/B delta is clock noise.

**P1.4 Audit tool success at the result level, not the transport level (canon 14).**
14.9% of tool calls in the paper's league failed at result level while transport reported success.
The `news` analyst is the highest-weighted strategy in this ensemble and its `web_search` calls are
never audited: `_run_one` catches exceptions, but a response containing zero search results and a
hallucinated catalyst returns clean JSON and full weight.
→ Record `n_search_results` per analyst run; treat a zero-result news call as a dropped analyst.

**P1.5 Report what the primary metric throws away (canon 12).**
PASS(±1%) is largely a volatility thermometer: it measures P(|move| ≤ 1%) more than skill, which is
why the shipped system can post a 55% rolling PASS rate with a negative edge.
→ Keep `edge` as the sole headline (the spec already says so — enforce it in `RESULTS.md`'s
ordering), and publish PASS alongside the realized share of days with |move| ≤ 1% so the reader can
see the confound.

### P2 — Model and cost work. Last, because the paper says it buys the least.

**P2.1 Run a cheap model league before spending more on models.**
§8.4: three frontier models spanned 263.27–264.37 bps of replay regret, every interval overlapping,
Holm-adjusted min p = 0.46 — at a 25× cost difference. This repo runs Opus for the judge *and* the
reflector *and* arm B's decider.
→ Replay the 60 stored `analyst_predictions` snapshots through a cheaper judge model and compare
paired MAPE. If the deltas overlap (they probably will), downgrade the judge and spend the savings
on more paired days, which is what the statistics are actually starved of.

**P2.2 Treat `STRATEGY.md` / `WHATS_NOT_WORKING.md` as a contamination channel, not memory.**
§7: memory-write frequency correlated *negatively* with P&L (ρ = −0.200); one agent ran 31 of 32
positions under strategy text that had been deleted; 40% of compactions *grew* the file. This loop
appends one insight per day forever and feeds the tail to the judge; `_user_prompt` truncates
`strategy_md[-4000:]`, so which advice survives is decided by a character count.
→ Cap the living strategy at N rules, require each rule to carry the date it was added and the
measured effect that justified it, and **expire any rule that has not been validated** — the way
`FAILURES.md`'s prescriptions should have been expired once F7 showed the crowding diagnosis was
phantom.

**P2.3 Apply §8.2's utility bar to the analyst roster.**
`macro` has a 12% hit rate and the worst MAPE, and post-mortems repeatedly name it as the drag. The
paper removed its `market research` sub-agent from the informational bar when it came back null at
n=120.
→ Register the test: does dropping `macro` change the paired MAPE over 20 days? If not, drop it —
it is a fifth of the API spend for no measured behavior change.

---

## 3. What *not* to build

The paper's null results predict which of `FAILURES.md`'s standing prescriptions will not pay:

- **The correlation-penalty gate** ("cap combined weight when analysts share inputs"). F7 measures
  mean pairwise correlation at +0.093 and 10% unanimity. The diagnosis is phantom.
- **Any further prompt-level rule aimed at the meta-judge** ("when RSI<25 boost contrarian to
  0.30", "when news quotes a live intraday price set a floor"). §5.3 is the direct precedent: the
  strongest prompt-side lever the lab found not only failed, it inverted. The blend is set by the
  aggregation code, so fix it there.
- **More context.** §7: stuffing 770K tokens into a paired comparison scored −0.46 points
  [−2.93, +1.75]; a world-context arm went 0/84; the strongest positive template finding was a
  *restraint* arm, with intervention rate correlating to score at −0.72. The instinct to feed the
  judge more post-mortems is the one the record says to resist.
- **Raising news's weight because it quotes live intraday prices.** That is lookahead (F6), and
  acting on it would manufacture an edge that evaporates the moment the schedule is enforced.

---

## 4. Suggested sequencing

1. P0.2 (pre-open guard + quarantine) and P0.1 (log the counterfactual) — both are small, and
   nothing else is trustworthy until they land.
2. P1.1 (null arm) — establishes the noise floor that every later comparison is read against.
3. P0.3 + P0.4 (bounded judge, σ-scaled move) as one registered change with the primary metric
   declared in advance.
4. P0.5, then P1.2–P1.5.
5. P2 only after the above has produced 20+ clean paired days.

The paper's closing line is the right standard for this repo: *"finish the measured operating-layer
fixes, deploy the post-order risk check and the decision-provenance ledger, train on the
environment the record itself provides, and let versioned benchmarks decide whether any of it
worked."*
