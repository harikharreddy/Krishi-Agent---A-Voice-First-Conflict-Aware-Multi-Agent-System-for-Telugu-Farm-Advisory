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
| 6 | "v2 EfficientNetB0" (Colab, `/content/disease_agent_efficientnetb0.pt`) | PlantVillage only, full fine-tune + augmentation (per description) | **Not yet evaluated** | 99.87% | **Not yet measured** | No (not in this repo yet) |

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

## What this document does NOT decide

Whether the deployed system's headline claim should be the zero-shot
number (22-23%, methodologically pure, matches Phase 1's original
commitment) or the fine-tuned number (37.65%, better performance, a
narrower and different claim) is **a decision for the team and Dr.
Nagaraju, not something resolved by this evidence document or by Claude
Code unilaterally**. Both numbers are real, both are honestly measured,
and both are now clearly attributed to a specific, named checkpoint in
the table above. What the project's report claims as *the* result should
follow from that conversation, not precede it.

If there is a separate literature-survey document (referenced in
`orchestrator/intent_router_results.md`'s citation of "Radeva et al. [16]",
and in `docs/krish_agent_project.md`'s instruction to run the literature
survey as a standalone parallel thread) that frames "strict zero-shot
methodology" as part of the project's novelty claim, that document is
**not in this repository and Claude Code has not read it** — its exact
framing could not be verified before writing this reconciliation. If it
exists and makes that claim, it needs to be reconciled against row 5
being the deployed system, as part of the same conversation above.

## Next

Once the v2 checkpoint (row 6) is available, the same three checks applied
to rows 3-5 this session will be applied to it before its PlantDoc number
is reported anywhere:
1. Confirm exactly what it was trained on (PlantVillage only, per its
   description — to be verified against the actual checkpoint/training
   log if available, not assumed from the filename).
2. Evaluate on PlantDoc's test split only for any headline number; if an
   aggregate over train+test is computed for any reason, label it
   explicitly and separately, the same way this row's honest_gaps do for
   the other checkpoints.
3. State plainly in whatever evidence file is produced which row of this
   table it becomes, so it can't be compared against row 5 without a
   reader knowing whether it's a zero-shot or fine-tuned number.
