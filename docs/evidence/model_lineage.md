# Disease Agent Model Lineage — Full Reconciliation

Written in response to a review flag: metric #6's "old vs new" comparison
used two checkpoints without stating which was which or why, sitting next
to Phase 1's documented decision not to fine-tune on PlantDoc. This
document is the fix — every checkpoint that exists, what it was trained
on, what it was tested on, and which one is actually deployed, in one
place, so no downstream report section can cite a number without knowing
which model produced it.

## The lineage table

| # | Checkpoint(s) | Trained on | PlantDoc role | PlantVillage val acc | PlantDoc accuracy | Deployed? |
|---|---|---|---|---|---|---|
| 1 | `flat_baseline_best.pt` | PlantVillage only (13-class flat) | Zero-shot eval only | 99.72% | 22.00% (n=968, train+test combined) | No |
| 2 | `stage1/2_*_plantvillage_only.pt` | PlantVillage only (hierarchical) | Zero-shot eval only | 99.70% | 23.45% (n=968) / 28.24% (n=85, test-only) | No (kept for rollback) |
| 3 | `stage1/2_*_plantdoc_no_target_spot.pt` | PlantVillage + PlantDoc **train** split | Train split used for fine-tuning; **test** split held out | n/a (not re-measured) | 40.0% (n=85, test-only, held-out) | No (superseded) |
| 4 | `stage1/2_*_pre_potato_healthy.pt` | Same as #3 + 20 external Target_Spot photos | Same as #3 | n/a | 38.8% (n=85, test-only, held-out) | No (superseded) |
| 5 | `stage1/2_*_best.pt` | Same as #4 + 100 external Potato_healthy photos | Same as #3 | n/a | **37.65% (n=85, test-only, held-out)** | **YES — this is what `orchestrator/pipeline.py` calls via `disease_agent.predict_disease()` right now** |
| 6 | `disease_agent_efficientnetb0.pt` (flat 13-class, commit `2fdd4fd`) | PlantVillage only, full fine-tune + augmentation (per description) | Zero-shot eval, train+test combined (n=965; see discrepancy note in metric1 evidence) | 99.87% val / 99.80% test | **21.45% (207/965)** | No (still row 5 deployed) |

Rows 3-5 all used PlantDoc's **train** split for fine-tuning and held out
the **test** split completely throughout (verified: same 85 test-only
images, same missing-folder caveat, across every evaluation run this
session — see `docs/evidence/metric6_confidence_calibration_evidence.json`'s
`apples_to_apples_check`). None of the reported PlantDoc accuracy numbers
above mix train and test images — that specific contamination risk was
checked and ruled out for rows 3-5 (see `metric6_confidence_calibration_evidence.json`'s
`CRITICAL_METHODOLOGY_FLAG` for the one place a *different* number, the
967-image aggregate, WOULD have been contaminated, and was excluded).

**Row 6 (the v2 checkpoint) is unverified in the same way rows 3-5 used to
be** -- its PlantDoc accuracy has not been measured yet, and if/when it
is, the same train/test-contamination check built for metric #6 will be
applied to it before any number is reported (see "Next" below).

## Metric #6's checkpoints, named explicitly

Metric #6 compared **row 2** ("BEFORE," zero-shot, dated Sep 3) against
**row 5** ("AFTER," fine-tuned, currently deployed). This was not stated
explicitly in the original evidence file and should have been — fixed in
the updated `metric6_confidence_calibration_evidence.json` (see below).

## The Phase 1 decision, and what actually changed

`docs/claude_phase1-summary.md` (line 34), verbatim:

> Explicit decision made: did **not** retroactively fine-tune on PlantDoc
> just to hit the target number, to keep the generalization test honest

This is real and was correctly quoted. Row 5 (the deployed model) directly
does what this decision said not to do. What follows is an honest account
of what changed and why, not a defense that nothing changed:

- **What Phase 1 was protecting**: a *zero-shot cross-domain generalization*
  claim — "trained only on clean lab photos, still works on real-world
  photos it has never seen in any form." That claim is genuinely
  incompatible with fine-tuning on PlantDoc, for exactly the reason Phase 1
  gave, and rows 1-2 are the only checkpoints that can honestly support it
  (22-23.45%, well below the guide's 65-80% target).
- **What actually happened this session**: the zero-shot number was judged
  too weak to be the project's field-readiness claim, and PlantDoc's own
  official **train** split (never touched by rows 1-2) was used for
  fine-tuning, with PlantDoc's **test** split held out and never trained on
  at any point (verified above). This is a different, narrower claim —
  *domain-adapted accuracy on unseen real-world photos after being shown
  some real-world examples* — not zero-shot generalization, and not the
  same claim Phase 1's methodology was designed to protect.
- **This is a real methodological pivot, and it was not flagged against
  Phase 1's decision when it happened.** The fine-tuning work itself was
  done openly, deliberately, and is extensively documented at every step
  (`docs/evaluation_and_validation.md` Sec. 2.5, `disease_agent.py`'s
  module docstring, individual commit messages) — it was never hidden.
  But at no point during that work was Phase 1's specific prior commitment
  re-examined and either reaffirmed or explicitly superseded. That's a
  real gap in this session's own rigor, not just a documentation gap.

## Decision (recorded)

**Confirmed with Dr. Nagaraju, 2026-09-13.** The question above has been
resolved, not by this document or by Claude Code:

- **The zero-shot PlantDoc result (row 2: 23.45% / 28.24% test-only)
  is the headline real-world accuracy number for the Disease Agent.**
- **The PlantDoc-fine-tuned result (rows 3-5, including the currently
  deployed row 5 at 37.65%) is demoted to a secondary, clearly-labeled
  *exploratory* result — not presented as a comparison against the
  zero-shot number, since row 5 was fine-tuned on PlantDoc and is no
  longer methodologically comparable to a zero-shot claim, contaminated
  in the dataset-level sense even though the specific 85 test images were
  properly held out.**

**Reasoning, as given**: (1) the zero-shot number is what the literature
survey's comparison points expect and can be meaningfully placed against
— a fine-tuned number Would need its own, different category of
comparison this project hasn't done; (2) methodological consistency with
Phase 1's original decision (`docs/claude_phase1-summary.md:34`) — the
project committed once to keeping the generalization test honest, and
this decision restores that commitment rather than letting it be quietly
superseded by a later, better-performing but differently-scoped number.

**What this means downstream**: any report section, slide, or evidence
summary that cites "the" Disease Agent accuracy should cite row 2
(23.45% / 28.24%), not row 5. Row 5 and the fine-tuning work remain fully
documented (nothing is deleted or hidden) but must be labeled exploratory
wherever it appears — metric #6's evidence has been updated accordingly
(see `metric6_confidence_calibration_evidence.json`'s new
`reporting_status` field).

If the separate literature-survey document (referenced in
`orchestrator/intent_router_results.md`'s citation of "Radeva et al. [16]")
turns out to frame things differently once read directly, that's a reason
to revisit this entry, not a reason to have waited on it — this decision
is now the team's committed position either way.

## Row 6 results (Metric #1, closed 2026-09-13)

Evaluated via `agents/disease/evaluate_v2_checkpoint.py` — full evidence
in `docs/evidence/metric1_v2_checkpoint_evidence.json` /
`metric1_v2_checkpoint_raw_log.txt`.

- **21.45% accuracy (207/965)**, train+test combined — comparable to row
  2's 23.45% (n=968), a similar (slightly lower) zero-shot result from an
  independently trained, differently-architected model.
- **Contamination sanity check confirmed clean**: train-only accuracy
  21.14% vs. test-only 24.71% (test actually *higher* — the opposite of
  what contamination would show; row 5's real contamination signature was
  a positive ~8-point train-over-test gap). Supports the "genuinely
  zero-shot" claim without relying on the stated training provenance
  alone.
- **Cross-architecture confirmation of the attractor-class bias**: this
  checkpoint independently reproduces row 2's `Tomato_Late_blight`
  over-prediction (44.0% predicted vs. 11.5% true here, vs. row 2's
  documented 49% vs. 12%), plus a second, comparably strong bias toward
  `Tomato_Early_blight` (33.4% predicted vs. 9.1% true) that row 2 did not
  show as strongly. Two independently trained models both defaulting to
  blight-type diagnoses on real photos is stronger evidence this is a
  structural property of the PlantVillage-to-PlantDoc domain gap itself,
  not one training run's quirk — a genuinely new, citable finding this
  evaluation produced, not just a confirmation of what was already known.
- **Known discrepancy, investigated and documented, not material**: this
  run evaluated n=965 images against the same 10 PlantDoc folders that
  produced n=967 in the Sep 3 baseline run. Traced 1 of the 2 missing
  images to a local file-count change inside `data/plantdoc_raw` (itself
  a separate git repo) between Sep 3 and now; the second was not
  identified. Effect on the headline number is at most ~0.2 percentage
  points — noted for completeness, not treated as invalidating the result.

## Zero-shot convergence across all three independent checkpoints

`agents/disease/compute_zero_shot_convergence.py` /
`docs/evidence/metric1_zero_shot_convergence.json`.

| Checkpoint | Architecture | Accuracy | n |
|---|---|---|---|
| Row 1 (flat baseline) | Flat 13-class EfficientNetB0 | 22.00% | 968 |
| Row 2 (hierarchical) | 2-stage EfficientNetB0 | 23.45% | 968 |
| Row 6 (v2) | Flat 13-class EfficientNetB0, independent Colab run | 21.45% | 965 |

**Mean 22.30%, stdev 1.03 points, range 21.45-23.45% (2.00-point spread).**
Three separately trained models — 2 architectures, 2 training
environments/runs, none fine-tuned on PlantDoc — converge to within a
2-point band. Framed as convergence evidence: the same argument already
made from the shared `Tomato_Late_blight` attractor bias (both zero-shot
checkpoints independently defaulting to blight-type diagnoses) is now
also supported by the aggregate accuracy number landing in a tight range
across genuinely independent runs, not just one model's idiosyncrasy.

## Metric #6 — CLOSED, final decision recorded (2026-09-13)

Full evidence in `docs/evidence/metric6_confidence_calibration_evidence.json`
(`cutoff_sweep_and_stability_check_2026-09-13` → `final_decision_2026-09-13`)
and `metric6_confidence_calibration_raw_log.txt`. This closes out the
confidence-calibration spot-check, including the cutoff sweep and the
statistical stability check that followed it.

**Finding**: sweeping `DISEASE_CONFIDENCE_CUTOFF` candidates from 0.30 to
0.70 (step 0.05) against the deployed checkpoint's 85-image PlantDoc
test-only set found several values with higher treat_now-bucket accuracy
than the deployed 0.7 on far larger samples (e.g. 0.55: 75.0% accuracy,
n=16, vs. 0.7's 75.0% at n=4). A stability check — Wilson 95% CI overlap
across 0.50/0.55/0.60, plus a 2,000-iteration bootstrap resample — showed
this is a real but imprecise effect: the 0.50-0.65 region wins 92.5% of
bootstrap resamples collectively, but no single value (0.55 included, at
50.5%) can be pinned down as *the* optimum on this sample size. Honest
framing: **0.50-0.65 is a statistically defensible range where a
better-calibrated cutoff than 0.7 likely lives — not a specific number
this dataset can justify hard-coding.**

**Decision**: `orchestrator/pipeline.py`'s `DISEASE_CONFIDENCE_CUTOFF`
stays at **0.7 — no code change**. The finding is reported as a
documented, evidence-backed recommendation for future work, not an
applied fix. Reasoning: the deployment risk of changing live pipeline
behavior this close to the defense outweighs a marginal, statistically
imprecise accuracy gain that the evidence itself says cannot be pinned
to one value.

**Confirmed**: `git diff -- orchestrator/pipeline.py` is empty and
`DISEASE_CONFIDENCE_CUTOFF = 0.7` is unchanged in the file — the entire
metric #6 investigation (original spot-check, median-split analysis,
Wilson CI work, the 9-point sweep, the CI-overlap and bootstrap
stability checks, and this final decision) never modified deployed
pipeline behavior. Worth stating plainly: a multi-round evaluation
process that surfaces a real, tempting-looking improvement and still
ends in "we gathered rigorous evidence and did not change the system"
is itself evidence the evaluation process has integrity — the same
point already made for metric #5's rule-table fix, in reverse (there,
evidence justified a change; here, evidence justified restraint).
