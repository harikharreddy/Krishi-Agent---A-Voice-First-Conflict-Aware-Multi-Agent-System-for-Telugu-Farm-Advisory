# Disease Agent Model Lineage — Full Reconciliation

Written in response to a review flag: metric #6's "old vs new" comparison
used two checkpoints without stating which was which or why, sitting next
to Phase 1's documented decision not to fine-tune on PlantDoc. This
document is the fix — every checkpoint that exists, what it was trained
on, what it was tested on, and which one is actually deployed, in one
place, so no downstream report section can cite a number without knowing
which model produced it.

## The lineage table

Compute footprint column added 2026-09-16 (params/FLOPs via `ptflops`,
`agents/disease/compute_model_footprint.py`; latency/throughput pulled
in from the existing real measurement — `agents/disease/
compute_full_metrics.py` — rather than reported separately, per
architecture-sharing notes below the table).

| # | Checkpoint(s) | Trained on | PlantDoc role | PlantVillage val acc | PlantDoc accuracy | Deployed? | Compute footprint* |
|---|---|---|---|---|---|---|---|
| 1 | `flat_baseline_best.pt` | PlantVillage only (13-class flat) | Zero-shot eval only | 99.72% | 22.00% (n=968, train+test combined) | No | 4.02M params, 0.82 GFLOPs |
| 2 | `stage1/2_*_plantvillage_only.pt` | PlantVillage only (hierarchical) | Zero-shot eval only | 99.70% | 23.45% (n=968) / 28.24% (n=85, test-only) | No (kept for rollback) | 12.04M params (3 models), 1.64 GFLOPs/inference — architecture shared with row 5 |
| 3 | `stage1/2_*_plantdoc_no_target_spot.pt` | PlantVillage + PlantDoc **train** split | Train split used for fine-tuning; **test** split held out | n/a (not re-measured) | 40.0% (n=85, test-only, held-out) | No (superseded) | same as row 2/5 (architecture unchanged by fine-tuning) |
| 4 | `stage1/2_*_pre_potato_healthy.pt` | Same as #3 + 20 external Target_Spot photos | Same as #3 | n/a | 38.8% (n=85, test-only, held-out) | No (superseded) | same as row 2/5 |
| 5 | `stage1/2_*_best.pt` | Same as #4 + 100 external Potato_healthy photos | Same as #3 | n/a | **37.65% (n=85, test-only, held-out)** | **YES — this is what `orchestrator/pipeline.py` calls via `disease_agent.predict_disease()` right now** | 12.04M params, 1.64 GFLOPs/inference, **240.2ms mean latency (std 20.7ms, n=85), 4.16 img/s throughput** (measured, not inherited) |
| 6 | `disease_agent_efficientnetb0.pt` (flat 13-class, commit `2fdd4fd`) | PlantVillage only, full fine-tune + augmentation (per description) | Zero-shot eval, train+test combined (n=965; see discrepancy note in metric1 evidence) | 99.87% val / 99.80% test | **21.45% (207/965)** | No (still row 5 deployed) | 4.02M params, 0.82 GFLOPs — no real-world latency measured for this single-model architecture |
| 7 | `disease_agent_plantdoc_finetuned_best.pt` | Row 6 (v2, flat 13-class) + PlantDoc native train split, content-hash deduped, best-checkpoint tracked | Train split fine-tuning; test split held out throughout | n/a (not re-measured) | ~~32.39% best / 26.76% final~~ **INVALID FOR HEADLINE REPORTING — bug-confirmed, n=71 on a test set silently missing 2 of 13 classes; kept for transparency only, see below** | No — not deployed, and not citable as a comparison point | same as row 6 (architecture unchanged by fine-tuning) |

**\*Compute footprint notes**: params/FLOPs depend on architecture and
class count, not trained weight values, so rows sharing an architecture
share identical figures by construction — not independently
re-measured per row. Row 2's family (rows 2/3/4/5) is the 3-model
hierarchical split (stage1 crop-ID + stage2 tomato + stage2 potato);
FLOPs/inference reflects that a real call only ever runs 2 of the 3
models (stage1 + whichever stage2 branch matches the detected crop), not
all 3 simultaneously, while the 12.04M param count is all 3 models
loaded in memory (matches the already-published 12,041,859 figure in
`evaluation_and_validation.md`'s precision/recall/latency subsection
exactly — cross-validates this fresh computation against that prior
independent measurement). Row 5's latency/throughput is the only row
with a REAL measurement (via live inference timing, not FLOPs-derived)
— restated here from `compute_full_metrics.py` rather than re-measured,
and is architecturally representative of rows 2/3/4 too since they share
the identical compute graph, though those specific checkpoints were
never independently latency-profiled. Row 6/7's single-model architecture
has never been latency-profiled at all — stated honestly as a gap, not
estimated from FLOPs alone. Full evidence:
`docs/evidence/model_footprint_row2_row6_evidence.json`.

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

**Formally tested, not just eyeballed (2026-09-16)**: the descriptive
convergence above was pulled forward from the Post-Defense Journal
Extension Roadmap and actually run —
`agents/disease/mcnemar_test_zero_shot_checkpoints.py` /
`docs/evidence/mcnemar_zero_shot_checkpoint_comparison_evidence.json`.
McNemar's test (the correct tool for paired binary outcomes — same
images, correct/incorrect per checkpoint — not independent-sample
comparison) on each pairwise combination:

| Pair | n (shared images) | Discordant pairs | Test | Statistic | p-value | Significant? |
|---|---|---|---|---|---|---|
| Row 1 vs Row 2 | 967 | 168 | chi-square, continuity-corrected | 1.006 | 0.316 | No |
| Row 1 vs Row 6 | 965 | 163 | chi-square, continuity-corrected | 0.098 | 0.754 | No |
| Row 2 vs Row 6 | 965 | 179 | chi-square, continuity-corrected | 1.810 | 0.179 | No |

**All three pairwise comparisons are non-significant (p ≥ 0.05).** This
is genuine statistical support, not a stronger-worded restatement of the
descriptive finding — the null hypothesis (these two checkpoints have
the same underlying accuracy) cannot be rejected for any pair on this
paired sample. Image-set identity was verified before testing, not
assumed: row 1 and row 2 use the exact same 967-image set (checked via
set equality on the image path, not just matching counts); row 6 is a
strict subset missing exactly the same 2 images already flagged in
`metric1_v2_checkpoint_evidence.json`'s discrepancy note — each pair
involving row 6 is tested on the actual 965-image intersection, not
assumed to align with rows 1/2's 967. Test variant (exact binomial vs.
chi-square) was checked per pair via the discordant-pair count (all
three comfortably exceeded the 25-pair threshold for a reliable
chi-square approximation, so chi-square with continuity correction was
used throughout, not defaulted to without checking). Not corrected for
multiple comparisons (3 tests at α=0.05) — noted as a caveat in the
evidence file, not hidden.

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

## Cross-reference: Metric #4 (voice vs. text ablation) — CLOSED

Out of scope for this document (no Disease Agent checkpoint involved —
metric #4's test set carries no image, so the Disease Agent never
fires), included here only as a pointer since this file has become the
place readers check for closed Phase 8.1 metric status. Full writeup
lives in `docs/evaluation_and_validation.md` Sec. 2.9 and
`docs/evidence/metric4_voice_vs_text_ablation_evidence.json`. One-line
summary (updated 2026-09-16, expanded to FINAL n=64/4 speakers): 60/64
stage-by-stage matches between the voice and text paths; the one
genuine finding -- ASR transliterating "pH" and flipping the intent
router's classification -- is now CONFIRMED STRUCTURAL, reproducing
independently across all 4 speakers' recordings, not a single-speaker
fluke. The remaining 2 mismatches were investigated and settled as
voice-path live-API failures (speaker4 only), not a pipeline-logic
effect. Zero unexplained mismatches remain across the full run.

## Row 7 — extended clean PlantDoc fine-tune, best-checkpoint tracked (2026-09-14)

**INVALID FOR HEADLINE REPORTING — bug-confirmed, kept for transparency
only.** Same treatment as the earlier train/test-contamination finding
(`docs/evidence/metric6_confidence_calibration_evidence.json`'s
`CRITICAL_METHODOLOGY_FLAG`): this run's 32.39%/26.76% numbers are
documented below in full, but must **not** be cited as a comparison
point against the deployed model's 37.65% or against any zero-shot
baseline. The root-cause investigation later in this section confirms a
data-loading bug silently dropped 2 of 13 classes from this run's test
set entirely — the numbers below measure something structurally
different from every other accuracy figure in this project's evaluation,
not a harder or easier version of the same test.

Full evidence: `docs/evidence/plantdoc_finetune_evidence.json`. Checkpoint:
`agents/disease/checkpoints/disease_agent_plantdoc_finetuned_best.pt`.
Base: row 6 (the independently-trained flat "v2" checkpoint), fine-tuned
on PlantDoc's native train split (content-hash deduped, 6 duplicates
removed) with best-checkpoint tracking and early stopping added this
round — a methodology improvement over rows 3-5's final-epoch-only
selection.

**Overfitting pattern, and why best-checkpoint tracking exists**: test
accuracy peaked at epoch 11 (32.39%) and *declined* over the remaining 8
epochs to 26.76% at epoch 19 (early-stopped), while train accuracy kept
climbing the whole time (25.5% -> 28.7% across those same epochs) — a
textbook overfitting-after-peak signature, visible directly in
`training_history_full`. This is exactly the failure mode best-checkpoint
tracking is meant to catch: a final-epoch-only evaluation on this run
would have reported 26.76%, 5.6 points below what the model actually
achieved at its best point. Both numbers are reported together
deliberately, per the evidence file's own `honest_note`: with only 71 test
images (~1.4 points per image), best-epoch selection out of 19 epochs
carries real risk of picking a lucky noise peak rather than a true
improvement — reporting best alone would overstate confidence in 32.39%
specifically.

**Class coverage differs from the deployed model's — NOT "the same 4
classes," investigated and characterized precisely.** This run's
`classes_with_zero_train_images` lists 4 classes (`Tomato__Target_Spot`,
`Tomato__Tomato_YellowLeaf__Curl_Virus`, `Tomato_healthy`,
`Potato___healthy`). That is a **different, larger set** than the
previously-established "2 classes with zero real-world data anywhere"
(`Potato___healthy` and `Tomato__Target_Spot` — see
`docs/evaluation_and_validation.md`'s Disease Agent section) for the
*deployed* model. Two of these four are genuinely structural and
consistent with prior findings: `Potato___healthy` and `Target_Spot` have
no PlantDoc images anywhere (train or test) in *any* run this project has
evaluated, deployed model included — this part of the "4" is real and
consistent.

The other two are not a continuation of anything previously documented,
and were verified, not assumed: `Tomato_healthy` and
`Tomato__Tomato_YellowLeaf__Curl_Virus` **do** have real PlantDoc images
-- the deployed model's own per-class table
(`docs/evaluation_and_validation.md` Sec. 2.5) reports real accuracy
numbers for both (`Tomato_healthy`: 37.5%, n=8; `Tomato_YellowLeaf_Curl_Virus`:
66.7%, n=6), and this checkpoint's own base model (row 6) was evaluated
zero-shot against these same PlantDoc folders earlier this project. Cross-
checked the arithmetic directly: this run's test set totals n=71, and
`established n=85 test-only set minus (YellowLeafCurl_Virus n=6 + healthy
n=8) = 71` exactly, with every other overlapping class count identical
between the two sets (Bacterial_spot 9=9, Early_blight 9=9, Late_blight
10=10, Leaf_Mold 6=6, Septoria 11=11, mosaic_virus 10=10,
Potato_Early_blight 8=8, Potato_Late_blight 8=8). This is conclusive, not
coincidental: this run's data pipeline (most likely the "content-hash
dedup" step, though the evidence file doesn't say which stage) dropped
`Tomato_healthy` and `Tomato_YellowLeaf_Curl_Virus` entirely from both
train and test, on top of the 2 genuinely-structural gaps.

**Practical consequence, stated plainly**: 32.39%/26.76% on this run's
n=71 test set is **not a valid comparison number against the deployed
model's 37.65% on n=85, or against any zero-shot baseline** -- the test
set is missing 2 of 13 classes entirely, not measuring a harder or
easier version of the same benchmark. Per the root-cause investigation
below (bug confirmed, not correct-but-unlucky dedup), this row is
**INVALID FOR HEADLINE REPORTING** and exists in this document for
transparency and reproducibility only.

**Root cause, investigated and confirmed (2026-09-14) -- NOT genuine
content-hash dedup, a mapping/loading bug.** Checked directly against
`data/plantdoc_raw` rather than assumed either way, per the two
hypotheses worth distinguishing (correct-but-unlucky dedup vs. a bug):

- **Ruled out by arithmetic alone, before any file was even opened**:
  the evidence file's own `duplicates_removed_from_train` is **6**,
  dataset-wide, across all 13 classes. The raw PlantDoc folders for
  these two classes hold **55** (`Tomato leaf`, i.e. `Tomato_healthy`)
  and **70** (`Tomato leaf yellow virus`) train images respectively --
  125 combined. A dedup budget of 6 cannot possibly explain 125 images
  vanishing from two classes; every other class's post-dedup count in
  the evidence file (101, 79, 101, 85, 140, 44, 109, 97) already matches
  its raw folder's image count almost exactly, confirming dedup only
  ever removed a small handful, consistent with the reported 6.
- **Ruled out by content hash**: SHA-256'd every image in both raw train
  folders. `Tomato_healthy`: 55/55 hashes unique (0 internal duplicates),
  0 matches against its own test folder, 0 matches against any other
  class's train images. `Tomato_YellowLeaf_Curl_Virus`: 70/70 unique, 1
  single match against its own test folder (a legitimate, minor dedup
  candidate -- consistent with the small reported total), 0 cross-class
  matches. No wholesale duplication exists anywhere for either class.
- **Ruled out file corruption as a silent-skip cause**: opened and
  verified all 125 images (`PIL.Image.verify()`) -- 0 corrupt or
  unopenable files, all valid JPEGs.

With genuine deduplication, and corruption, both directly ruled out by
evidence rather than assumed, the remaining explanation is a **class-
inclusion or folder-to-class mapping bug specific to this run's data-
loading script** (not present in this repo -- it ran in a separate
session) that excluded these two folders from being loaded at all,
upstream of wherever content-hash dedup runs. This is not a "some data
was noisy and got cleaned up correctly" story -- it is a concrete,
fixable bug: these two folders' 125 real, unique, valid training images
should have been available to this fine-tune and were not. Recommend
checking that script's class list / folder-name mapping for these two
specific entries before the next fine-tuning round -- this is now a
confirmed defect to fix, not an open question.
