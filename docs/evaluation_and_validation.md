# Krishi-Agent: Evaluation & Validation

Compiled Sep 2026, for use as a research-paper Evaluation/Results section draft.
All numbers below are either (a) freshly re-run against the current codebase
on this date, or (b) pulled from dated result files already in this repo
(`agents/disease/results/`, `latency_proof/`, `docs/claude_phase*.md`), with
that provenance noted per metric. Nothing here is estimated or asserted
without a script, dataset, or log backing it — where a number is weak
(small n, single speaker, etc.), that is stated explicitly rather than
smoothed over.

## 1. System overview

Krishi-Agent: voice-first, conflict-aware multi-agent farm advisory for
Telugu-speaking farmers (Tomato/Potato). Pipeline:

```
Question (voice/text) + Farm Profile (+ optional leaf photo)
  -> Intent Router (Ollama, qwen2.5:7b-instruct)
  -> [Disease Agent | Weather Agent | Price Agent]  (conditional)
  -> state mapping -> Conflict Resolver -> Phrasing Templates
  -> TTS (indic-parler-tts) -> spoken Telugu answer
```

Each pipeline stage was evaluated independently (below), plus the two
component-level rewrites that motivated the current architecture (TTS
latency, Disease Agent domain shift).

**This evaluation substantiates three system-level contributions, not a
single component-level accuracy claim** (full evidence and exact wording
in Sec. 3.1): (1) explicit, independently-tested conflict arbitration
across competing advice signals, not present in any of the three
comparable published systems surveyed; (2) confidence calibration
validated empirically, not asserted -- a five-signal convergence
(accuracy band, formal confusion matrix, Expected Calibration Error, a
formal McNemar's-test significance check confirming all three
same-family checkpoints are statistically indistinguishable, and now a
fourth, independently-architected checkpoint (ResNet18, not
EfficientNetB0) reproducing the same accuracy band, the same
attractor-class bias, and formal statistical indistinguishability from
all three) across four independently-trained checkpoints spanning two
distinct architecture families (Sec. 2.5), plus statistically-grounded
work on the deployed confidence cutoff (Wilson CIs and bootstrap
stability checks, Sec. 2.5); (3) a confirmed, cross-speaker
structural failure mode at the ASR-to-intent-router boundary -- a
domain-specific technical term (soil pH) transliterated by ASR flips the
intent router's classification, reproduced independently across all 4
recorded speakers (Sec. 2.9), not a single-speaker artifact. The Disease
Agent's own accuracy (~22% real-world, structurally bounded by domain
shift -- Sec. 2.5) is reported honestly as a measured limitation, not
positioned as the project's novelty claim.

## 2. Component-level results

### 2.0 Full logic/correctness test suite -- run today, all passing

Every deterministic-logic test in the repo, re-run together for one
consolidated snapshot (excludes the percentage-style accuracy metrics in
2.1-2.8, which are reported separately since "pass/fail" doesn't apply to
them):

| Test file | Result | What it covers |
|---|---|---|
| `test_conflict_resolver.py` | 13/13 (100%) | Conflict Resolver rule logic, synthetic scenarios (13th added as a regression test for the Sec. 2.7 rule-table gap, now fixed) |
| `test_phrasing_templates.py` | 13/13 (100%) | Confidence-calibrated template selection (shares `tests/conflict_scenarios.json` with `test_conflict_resolver.py` and `test_pipeline.py` -- grew from 12 to 13 cases when the 13th regression scenario was added to that shared fixture) |
| `test_state_mapping.py` | 13/13 (100%) | Boundary-value tests for the Weather/Price state-mapping thresholds (exact 0.10/-0.10 cutoffs, missing-data fallbacks) -- previously only comfortably-inside values were tested |
| `test_pipeline.py` | 13/13 (100%) | End-to-end conflict scenarios through the real pipeline |
| `test_pipeline_edge_cases.py` | 5/5 (100%) | No photo, missing profile fields, agent API failures |
| `test_weather_agent.py` | 9/9 (100%) | Schema validity, graceful failure, drought-signal unit tests |
| `test_price_agent.py` | 5/5 (100%) | Schema validity, graceful failure, market lookup |

**71/71 (100%)** across all deterministic-logic tests. **Corrected
2026-09-17** -- this table previously listed `test_phrasing_templates.py`
at 12/12 and the total at 70/70, both stale by 1. Root cause:
`test_conflict_resolver.py`, `test_phrasing_templates.py`, and
`test_pipeline.py` are script-style modules with a `main()` function, not
pytest-native `test_*` functions -- `pytest` silently collects zero
tests from all three (confirmed directly: `pytest tests/test_phrasing_templates.py`
reports "no tests ran", not a failure), so getting a real count requires
running each file directly rather than trusting a pytest summary. All
three share `tests/conflict_scenarios.json` (13 entries); when the 13th
regression scenario was added for the Sec. 2.7 fix,
`test_conflict_resolver.py` and `test_pipeline.py` were correctly
re-counted at the time, but `test_phrasing_templates.py`'s row was
missed. Caught and fixed by direct re-execution of all 7 files, not by
re-deriving the number from memory. This is the "nothing is silently
broken" layer underneath the accuracy metrics below -- a paper/evaluator
claim like "37.65% disease accuracy" only means something if the
surrounding pipeline logic (conflict resolution, phrasing, state
thresholds) is itself verified correct, which this table establishes.

### 2.1 Intent Router (LLM-based question routing)

- **Dataset**: `tests/intent_router_test_set.json` -- 16 hand-written Telugu
  farmer questions covering single-intent, multi-intent, and no-intent
  (greeting) cases, with ground-truth `{wants_disease, wants_weather,
  wants_price, wants_soil}` labels.
- **Method**: `tests/test_intent_router.py`, live calls to `route_intent()`
  against the running Ollama model. Exact-match scoring (all 4 boolean
  fields must match).
- **Result (re-run today)**: **14/16 correct, 87.5%** on the full set.
  Matches the originally-measured Phase 2.2 number exactly, confirming
  reproducibility. **But 5 of these 16 questions are verbatim few-shot
  examples inside the router's own `SYSTEM_PROMPT`** (see External
  validity below) -- recomputing on only the 11 genuinely held-out
  questions gives **9/11, 81.8%**, which is the more honest estimate of
  real-world performance. **This is worse news than 87.5% suggests, not
  just a smaller, noisier sample**: both of the full set's 2 failures
  (#12, #14) fall inside the 11 held-out questions, so neither was
  "propped up" by the few-shot-exposed subset scoring artificially well --
  removing that subset concentrates the same 2 errors onto a smaller base
  rather than diluting them. Both numbers are reported side by side going
  forward; **81.8% (held-out), not 87.5% (full set), is the number that
  should be cited as this component's real-world accuracy estimate.**
- **Failure modes** (both n=1 cases, illustrative not statistical): #12
  under-routed a broad "give me full advice" request (missed weather/price
  intents); #14 over-routed a pesticide-availability question as a disease
  question.
- **Headline limitation: compound-intent questions are where this breaks.**
  Question #12 ("నా టమాటా పంట గురించి పూర్తి సలహా కావాలి" / "I want full
  advice on my tomato crop") is the *only* question in the 16-question set
  whose ground truth expects three intents simultaneously
  (`wants_disease=wants_weather=wants_price=True`), and it's exactly the
  one the router does worst on -- recovering only 1 of the 3 expected
  labels (`wants_disease` alone; `wants_weather` and `wants_price` both
  missed). This is a **specific, useful limitation to name plainly**, not
  just a table row: the model appears able to hold one clear intent
  judgment reliably, but its ability to affirm *multiple, simultaneous*
  intents degrades on a maximal, "tell me everything" style question --
  precisely the kind of open-ended request a real farmer is likely to
  actually ask ("give me full advice on my crop" is a completely natural
  compound question, not an edge case constructed to break the system).
  A farmer asking that exact kind of question today would silently lose
  the weather and price portions of the answer they expected, with no
  error or partial-answer indication surfaced anywhere in the pipeline.
  Flagged as a concrete direction for future work (e.g. a routing prompt
  that explicitly enumerates each intent as an independent yes/no check
  rather than one combined judgment), not applied as a fix here --
  observation only, `orchestrator/intent_router.py` untouched.
- **Per-intent breakdown** (added per external review, 2026-09-13 -- are
  the 2 failures concentrated in one intent category, or spread evenly?):
  scored each of the 4 target labels as its own binary classification task
  across all 16 questions.

  | Intent | Accuracy | Precision | Recall | F1 | n positive |
  |---|---|---|---|---|---|
  | `wants_disease` | 93.8% | 85.7% | 100% | 0.923 | 6 |
  | `wants_weather` | 93.8% | 100% | 83.3% | 0.909 | 6 |
  | `wants_price` | 93.8% | 100% | 80.0% | 0.889 | 5 |
  | `wants_soil` | 100% | 100% | 100% | 1.000 | 3 |

  **Verdict: SPREAD, not concentrated** -- the 2 failures touch 3 different
  intent labels (`wants_disease` x1, `wants_weather` x1, `wants_price` x1;
  `wants_soil` is perfect), so this isn't one weak intent category dragging
  the average down; the more informative axis is compound- vs. single-
  intent questions (see headline limitation above), not which specific
  intent category. Full breakdown, including the held-out-only recompute:
  `docs/evidence/metric_intent_router_per_intent_evidence.json`.
- **Ablation -- model size**: qwen2.5:3b-instruct was tested on the same set
  in Phase 2.2 and scored **8/16, 50%** -- the 7B model was kept specifically
  because of this gap, at a documented latency cost (Sec. 2.6).
- **Known limitation**: n=16 (11 held-out) is a hand-curated smoke-test
  set, not a statistically powered benchmark. A real paper submission
  needs a larger, more diverse question set (see Sec. 5).
- **External validity limitation** (per external review, 2026-09-13):
  questions #3, #4, #9, #10, #13 appear verbatim as worked few-shot
  examples inside `orchestrator/intent_router.py`'s own `SYSTEM_PROMPT` --
  the model is being tested partly on phrasing it was directly shown for
  those 5, so 87.5% should not be read as an unbiased estimate of field
  performance on genuinely novel farmer phrasing (see the 81.8% held-out
  recompute above, which is why this isn't just a theoretical concern).
  This is not hypothetical either way: Phase 6 already found a real-world
  misrouting failure on unvalidated phrasing outside this set
  (`"ఈ ఆకుకు ఏమి జబ్బు?"`, see `orchestrator/intent_router_results.md`).

### 2.2 Conflict Resolver

- **Dataset**: 12 synthetic scenarios (`tests/test_conflict_resolver.py`)
  covering single-agent, dual-agent, and triple-agent conflict/agreement
  cases at varying confidence levels.
- **Result (re-run today)**: **13/13 correct, 100%** (13th scenario added as a regression test for the Sec. 2.7 rule-table gap, now fixed).
- **Limitation, denominator-honest**: rule-based logic over synthetic
  inputs, not live agent output -- validates the *decision logic*, not
  real-world scenario coverage. 13/13 is a true, well-evidenced pass
  rate, but it is not the same claim as full state-space coverage: **7 of
  the 13 designed scenarios cover the full 3-agent-active state grid (3
  disease states x 4 weather states x 3 price states = 36 combinations),
  i.e. 7/36 of that grid is covered** (the other 6 scenarios cover 1- and
  2-agent combinations, a separate, larger denominator not counted here).
  Reported together deliberately -- "13/13 designed scenarios pass; 7/36
  of the full 3-agent grid is covered" preempts the obvious next question
  rather than waiting to be asked it. Full detail:
  `docs/evidence/metric5_conflict_rule_table_evidence.json`.

### 2.3 ASR (speech-to-text) -- FINAL, 4/4 speakers

- **Model**: `ai4bharat/indic-conformer-600m-multilingual`.
- **Dataset**: 64 real microphone recordings (`tests/audio/wer_eval/speaker{1,2,3,4}/`,
  16 questions x 4 speakers), same 16-question set as Sec. 2.1.
- **Method**: `tests/check_all_wer.py`, word error rate via `jiwer`, text
  normalized (punctuation/whitespace stripped) before scoring. Auto-discovers
  and requires all 4 planned speakers before marking `reporting_status: FINAL`
  -- this run has all 4, so the result below is the project's final ASR
  number, not preliminary.
- **Result**: **overall mean WER 11.19%** across 64 utterances (n=64, 4
  speakers x 16 questions), full per-question table in
  `tests/audio/wer_eval/wer_results.json`.

  | Speaker | Mean WER | n |
  |---|---|---|
  | speaker1 | 10.96% | 16 |
  | speaker2 | 13.62% | 16 |
  | speaker3 | 8.09% | 16 |
  | speaker4 | 12.11% | 16 |

  All 4 speakers land within a ~5.5-point band (8.09%-13.62%) -- no
  speaker is a wild outlier, which is itself informative: the original
  1/4-speaker preliminary number (10.96%, speaker1) turned out to sit
  almost exactly at the middle of the eventual 4-speaker range, not at
  either extreme.

- **Per-question breakdown** (`tests/wer_breakdown_by_question.py`,
  `tests/audio/wer_eval/wer_breakdown_by_question.json`) -- which
  vocabulary is hardest, across all 4 speakers per question (n=4 each):

  | Question | Mean WER | Range | Vocabulary |
  |---|---|---|---|
  | Q3 "ఈ రోజు టమాటా ధర ఎంత" (today's tomato price) | 45.0% | 40-60% | Hardest -- short question, `ఈ రోజు`/`ఈరోజు` spacing variance dominates |
  | Q7 "మార్కెట్‌లో బంగాళదుంప ధర పెరుగుతుందా" (market potato price rising) | 37.5% | 25-50% | ZWNJ in `మార్కెట్‌లో` + `బంగాళదుంప` compound noun |
  | Q8 "నేల pH ఎంత ఉండాలి టమాటా కోసం" (soil pH for tomato) | 20.8% | 16.7-33.3% | The Latin-script "pH" token itself -- same technical-term-crossing-ASR issue metric #4 (Sec. 2.9) already found downstream in the intent router |
  | Q10 (compound sell/hold weather+price question) | 17.5% | 10-20% | Longest question in the set -- more words, more chances for a small slip |
  | Q2, Q4, Q6, Q13, Q14 | 0.0% | 0.0% | Perfect across all 4 speakers -- short, common-vocabulary questions |

  **Pattern, not noise**: the 4 hardest questions (Q3, Q7, Q8, Q10) are
  hard for *every* speaker, not just one -- consistent min-to-max ranges
  (e.g. Q3: 40-60%, never near 0%) rather than one speaker dragging up an
  otherwise-easy question. This points to specific vocabulary/phrasing
  properties (price-related compound nouns, ZWNJ-heavy words, embedded
  Latin-script technical terms, longer sentences) being the actual
  difficulty driver, not individual speaker variation -- the opposite
  pattern from what a small, single-speaker sample could have shown.

- Most individual "errors" remain spacing/ZWNJ artifacts (e.g. `ఈరోజు`
  vs `ఈ రోజు`), not content mistakes; genuine substitutions exist but are
  a minority of the error mass.
- **No longer a limitation**: 4 speakers, 64 utterances -- this is now
  the statistically-intended design (see Sec. 5's prior item on this,
  now closed), though 4 speakers is still a modest n by publication
  standards for claiming broad accent/dialect robustness.

### 2.4 TTS (text-to-speech) latency -- the original motivating bug

- **Model**: `ai4bharat/indic-parler-tts`.
- **Method**: `latency_proof/standalone_tts.py` (bare script) vs.
  `latency_proof/api_tts.py` (identical `synthesize_speech()` call through
  FastAPI) vs. the original Streamlit integration, all on the same machine.
- **Results** (dated in-repo, `backend/voice.py`/`tts_worker.py` docstrings):
  - Standalone baseline: **~47-53s**
  - Inside Streamlit (original architecture): **200-280s** (one measured
    case: 191s), a **5-7x** slowdown attributed to Streamlit's threading
    model interfering with generation
  - FastAPI backend (current architecture): matches the standalone baseline,
    since `api_tts.py` calls the identical `synthesize_speech()` function
- **Secondary finding**: a memory collision between the Intent Router LLM
  (Ollama, resident for 5 min post-call by default) and TTS on 8GB RAM
  caused an additional multi-minute stall independent of the
  Streamlit-vs-FastAPI difference (see Sec. 2.6).

**Formal N-trial re-measurement** (`latency_proof/formal_benchmark.py`,
closes the "single-run, no variance" gap): 8 trials of `synthesize_speech()`
on 8 different real sentences (actual agent/template outputs, not one
fixed sentence), model warm after the first call, on the same 8GB Apple
Silicon Mac (macOS 26.6.2, arm64):

| | n | mean | std | min | max |
|---|---|---|---|---|---|
| TTS generation (all 8 trials) | 8 | 26.16s | 8.67s | 17.59s | 39.30s |
| TTS generation (excluding first/cold-load call) | 7 | 24.28s | 7.41s | 17.59s | 35.21s |

This is **lower** than the historical "~47-53s" standalone baseline above,
most likely because this run used 8 different, generally shorter real
sentences rather than one fixed (possibly longer) test sentence -- flagged
here rather than silently reconciled, since the two numbers were never
meant to measure identical inputs. Either way, both are two to three
orders of magnitude below the 200-280s Streamlit figure, which is the
claim that actually matters and isn't sensitive to this difference. The
Streamlit path itself cannot be formally re-benchmarked with N trials --
that code was deliberately removed from the repo this session (see git
history) once the FastAPI migration was validated; the original
measurement stands as a historical, single-run data point, already
caveated as such when it was taken.

### 2.5 Disease Agent -- domain-shift and fine-tuning study

This is the most extensively evaluated component; full detail in
`agents/disease/results/*.json` and `agents/disease/disease_agent.py`'s
docstring.

**Training curves**: `agents/disease/results/plantdoc_finetuning_curves.png`
(generated by `agents/disease/plot_finetuning_curves.py` from a
reproduced re-run of the exact training that produced the currently
deployed checkpoints -- same data, same result: 37.65% end-to-end, 1/5
Target_Spot, 14/15 Potato_healthy, confirming determinism). Deliberately
plots the **PlantDoc (real-world) validation curve**, not a PlantVillage
one -- a PlantVillage curve converges to ~99%+ within a few epochs almost
trivially (Sec. 2.5's own baseline row shows why) and would visually
suggest a far stronger result than what the model actually achieves on
real photos. The honest curves plateau around 41-77% validation accuracy
per stage, matching the accuracy numbers reported throughout this
document -- shown as three loss curves (train vs. held-out PlantDoc val)
and three accuracy curves, one per stage model.

**Baseline (PlantVillage-trained, no real-world fine-tuning)**:
| Eval set | Domain | n | Accuracy |
|---|---|---|---|
| PlantVillage validation | studio/lab photos (same distribution as training) | 3,628 | 99.7% |
| PlantDoc (zero-shot) | real-world field photos, same 85-image test split used below | 85 | 28.2% |

The 71.5-point drop is a textbook train/test domain-shift result, and came
with a specific, diagnosable failure mode: the model over-predicted
`Tomato_Late_blight` for **49% of test images** (true rate: 12%) -- an
attractor-class bias, not just noise.

**Is this one model's fluke, or a structural property of the domain gap?
Independent-checkpoint convergence, now confirmed across two
architecture families.** The single-checkpoint baseline above (28.2%,
85-image test-only subset) could, on its own, be one model's
idiosyncrasy -- undertrained, an unlucky initialization, a quirk of this
specific run. It isn't: two more independently-trained EfficientNetB0
checkpoints -- a flat 13-class baseline, and a separately-trained flat
"v2" checkpoint built from a fresh Colab session (full provenance in
`docs/evidence/model_lineage.md`) -- were evaluated the same way, and a
fourth checkpoint built on a genuinely different backbone architecture
(ResNet18, not EfficientNetB0) confirms the same pattern again (signal 5
below). All four converge on **five independent signals**, not accuracy
alone. (Signals 1-4 were established across the three EfficientNetB0
checkpoints -- note that "flat vs. hierarchical" there describes a
*head-structure* variation within the same EfficientNetB0 backbone, not
a different architecture family; signal 5 is the first checkpoint with
an actually different backbone.)

1. **Accuracy band convergence**: all three checkpoints' zero-shot
   PlantDoc accuracy (full train+test combined, n=965-968 -- the valid
   comparison here since none of these three were ever fine-tuned on
   PlantDoc) lands in a 2-point band: 22.00% (flat baseline) / 23.45%
   (hierarchical) / 21.45% (v2 independent) -- **mean 22.30%, stdev 1.03
   points**. `docs/evidence/metric1_zero_shot_convergence.json`.
2. **Attractor-class convergence**, now formalized as an actual confusion
   matrix (previously an informal observation) for **all three**
   checkpoints (row 1's confusion matrix closed 2026-09-16, via a fresh
   inference run against the actual `flat_baseline_best.pt` checkpoint --
   no per-image predictions existed for it before, verified recoverable
   rather than assumed unrecoverable): all three independently default to
   predicting `Tomato_Late_blight`/`Tomato_Early_blight` regardless of
   true label -- flat baseline: `Tomato_Late_blight` predicted 373/967
   times (38.6% of all predictions, 59.5% recall but only 17.7%
   precision); hierarchical: `Tomato_Late_blight` predicted 522/967 times
   (54% of ALL predictions); v2 checkpoint: `Tomato_Late_blight` (425x) +
   `Tomato_Early_blight` (322x) dominate the same way. Per-class recall on
   those classes' own true instances stays far below their predicted
   share in every checkpoint (hierarchical: 80.2% recall but only 17.0%
   *precision* on `Tomato_Late_blight` -- the model calls almost
   everything blight, so it's "right" whenever the true label happens to
   be blight and wrong almost everywhere else).
   `docs/evidence/metric1_confusion_matrix_calibration_evidence.json`;
   matrices in
   `agents/disease/results/confusion_matrix_row{1,2,6}_*_zero_shot.png`.
3. **Calibration-shape convergence**: Expected Calibration Error is close
   across all three checkpoints (0.4892 flat baseline / 0.4725
   hierarchical / 0.4606 v2), and all three reliability diagrams show the
   same shape -- confidence bins above ~0.5 sit well below the diagonal
   throughout, i.e. the model is systematically *overconfident*, not just
   inaccurate, and this overconfidence pattern itself replicates
   independently across three, not two, runs.
   `agents/disease/results/reliability_diagram_row{1,2,6}_*_zero_shot.png`.
4. **Formal statistical confirmation (McNemar's test, 2026-09-16)** --
   pulled forward from the Post-Defense Journal Extension Roadmap and
   actually run, not left as a future-work gesture: the accuracy-band
   convergence above is a descriptive observation ("these numbers are
   close"); this tests it. All three pairwise comparisons (row 1 vs. row
   2, row 1 vs. row 6, row 2 vs. row 6), using McNemar's test on the
   paired per-image predictions (image-set identity verified before
   testing -- rows 1/2 share the exact same 967 images, row 6 is a
   verified 965-image subset, each pair tested on its actual shared
   image set, not assumed to align), come back **non-significant
   (p=0.316 / 0.754 / 0.179, all ≥ 0.05)**. This is genuine statistical
   support that these three independently-trained checkpoints perform
   indistinguishably on PlantDoc, not just a stronger-sounding
   restatement of "the numbers are close." Full contingency tables,
   test-variant selection reasoning (chi-square vs. exact binomial,
   checked per pair via discordant-pair count, not defaulted), and the
   multiple-comparisons caveat:
   `docs/evidence/mcnemar_zero_shot_checkpoint_comparison_evidence.json`.
5. **Cross-architecture-family confirmation (ResNet18, 2026-09-16)** --
   a fourth checkpoint, trained externally under a protocol matched to
   the three above (same seed, split, LR schedule, epoch/patience
   budget) but on a **different backbone architecture** (ResNet18, not
   EfficientNetB0), reproduces every signal above independently:
   22.51% zero-shot accuracy (185/822; 22.41-22.47% on the exact
   image subset shared with each of the three EfficientNetB0
   checkpoints, the fair apples-to-apples figure -- this checkpoint's
   own evaluation set is missing 2 of 13 classes relative to the
   others', documented honestly in `docs/evidence/model_lineage.md`'s
   Row 8 section rather than glossed over), landing inside the existing
   21.45-23.45% band; the same `Tomato_Late_blight`/`Tomato_Early_blight`
   attractor bias (44.0%/28.8% of all predictions, 72.9% combined,
   against 13.5%/10.7% true prevalence); and formal statistical
   indistinguishability from all three EfficientNetB0 checkpoints via
   McNemar's test (p=0.369 / 0.156 / 0.083, all non-significant).
   `agents/disease/mcnemar_test_row8_resnet18_vs_zeroshot.py` /
   `docs/evidence/mcnemar_row8_resnet18_vs_zeroshot_evidence.json`.

*(Signals 1-3 use each checkpoint's full combined image set, n=965-968,
matching how the accuracy-band convergence was computed -- not the
85-image test-only restriction used above and below for the fine-tuned-
model comparison. Directionally consistent with, not contradicting, the
49%-predicted/12%-true `Tomato_Late_blight` figure already quoted for the
single-checkpoint 85-image baseline. Signal 4 uses the same full-image
sets per pair, restricted to each pair's actual intersection where row 6
is involved -- see above. Signal 5's own image set (n=822) is smaller
than the others' (missing 2 of 13 classes, documented in
`model_lineage.md`'s Row 8 section) -- its headline 22.51% is on its own
822-image set, while its McNemar's comparisons and the 22.41-22.47%
range quoted above are restricted to the actual per-pair shared-image
intersection, the same discipline used for row 6 in signal 4.)*

**Read together, these are not five separate findings -- they are the
same underlying phenomenon viewed at five resolutions of the same
confusion matrix, three descriptive and now two formally tested**,
replicated across four independently-trained models, at least three
training environments, and -- critically -- **two genuinely distinct
backbone architecture families (EfficientNetB0 and ResNet18), not just a
head-structure variation within one family**. The evidence points to a
decision boundary shaped almost entirely by PlantVillage's uniform lab
backgrounds and framing -- a "blotchy texture on a leaf-shaped object,
pick the nearest common label" heuristic -- rather than lesion-specific
morphology, and signal 5 rules out the remaining loophole in that
argument: that this heuristic might be something EfficientNetB0
specifically learns (e.g. an inductive bias from compound scaling),
rather than a property of the PlantVillage-to-PlantDoc domain shift
itself. A structurally different architecture (residual blocks, no
compound scaling) reproducing the identical failure mode closes that
gap. This is strong evidence the ~22% real-world accuracy ceiling is a
**structural covariate-shift problem inherent to PlantVillage-only
training data, not a fixable modeling error specific to one run or one
architecture**, and it is this project's most scientifically interesting
result. Stated plainly, without hedging: **the Disease Agent, as
currently trained, should not be presented as reliable for field
deployment** -- this five-signal, architecture-independent convergence
(three descriptive, two formally tested) is the evidence for that
limitation and the honest framing of it, not a hidden weakness.

**A second, complementary form of rigor: the estimate is stable, not
just cross-architecture-convergent (k-fold CV, 2026-09-17).** The
five-signal convergence above answers "do independently-trained models
agree with each other" (yes, formally). A separate question -- does the
~22-23% number itself move around if you retrain the same recipe on a
different partition of the same training data -- is answered by 5-fold
stratified cross-validation on EfficientNetB0 (row 6's recipe; CV pool
= train+val only, the original held-out test set untouched throughout):
**PlantDoc zero-shot 23.38% ± 0.83%** across 5 folds (range
22.02-24.33%), a tight band. **Put together, this is a materially
stronger combined claim than either piece alone**: not only do
independent architectures converge to ~21-23%, but the estimate itself
is stable across different training-data partitions, not a fragile
point estimate that happened to land in a convenient range once. Scoped
honestly (Sec. 5 item 3): run on one architecture only, at a reduced
15-epoch/fold budget, and its PlantDoc image set (n=822) most likely
carries the same class-coverage gap as row 8's rather than matching
rows 1/2/6's fuller 967/968-image benchmark exactly -- so the ±0.83pp
*spread* is a clean same-benchmark comparison across the 5 folds, but
the 23.38% mean is not asserted as identical-benchmark to the other
rows' headline figures. Full detail and the fold-by-fold table:
`docs/evidence/kfold_cv_row6_variance_evidence.json`,
`docs/evidence/model_lineage.md`.

**After fine-tuning on PlantDoc's own `train` split** (conservative
transfer learning: frozen backbone except the last block + classifier
head, low LR, few epochs; `agents/disease/finetune_plantdoc.py`; PlantDoc
`test` split held out and untouched throughout):

| Metric | Before | After | n |
|---|---|---|---|
| End-to-end accuracy (PlantDoc test) | 28.2% | **37.65%** (later measurement: 37.65%, see note) | 85 |
| Stage-1 crop ID (Potato vs Tomato) | 81.2% | 82.4% | 85 |
| `Tomato_Late_blight` prediction rate | 49% | ~11% (true rate: 12%) | 85 |

*Note: end-to-end accuracy on the 85-image test set moved in small steps
(28.2% -> 40.0% -> 38.8% -> 37.65%) across successive fine-tuning rounds as
new classes (Target_Spot, then Potato_healthy) were added to the shared
softmax heads -- each addition traded a little accuracy on already-covered
classes for coverage of a previously-unaddressed class. The 37.65% figure
is the final, currently-deployed state; all intermediate numbers are
preserved in the checkpoint lineage and result files for full
reproducibility of the ablation.*

**A scale gap worth stating plainly, separate from the train/test-split
question above**: 37.65% is measured on **n=85**, while the zero-shot
convergence numbers in the signal table above (rows 1/2/6/8, "Accuracy
band" and "Cross-architecture-family confirmation") are measured on
**n=822-968** -- roughly an **11x larger sample**. This is not a
contamination risk (PlantDoc's test split was held out and never trained
on, verified above) but a genuinely different scale of statistical
evidence: on n=85, one misclassified image moves the headline number by
**1.18 percentage points**; on n=822-968, one image moves it by roughly
**0.10-0.12 points**. Reading 37.65% against the zero-shot rows' 21.45-
23.45% as if both carried equal precision would overstate how tightly the
37.65% figure specifically is pinned down -- part of why this project's
own model-lineage decision (`docs/evidence/model_lineage.md`, "Decision
(recorded)") treats it as a separately-labeled *exploratory* result
rather than a head-to-head comparison point against the zero-shot
checkpoints, and why the Sec. 5 roadmap's McNemar's-testing item notes
this same n=85-vs-n=965-968 mismatch as the reason a formal significance
test was not attempted between them.

**Per-class accuracy, current deployed model** (PlantDoc test, n=85):

| Class | Accuracy | n |
|---|---|---|
| `Tomato__Tomato_YellowLeaf__Curl_Virus` | 66.7% | 6 |
| `Tomato_Septoria_leaf_spot` | 54.5% | 11 |
| `Tomato_Early_blight` | 55.6% | 9 |
| `Tomato_healthy` | 37.5% | 8 |
| `Potato___Late_blight` | 37.5%\* | 8 |
| `Tomato_Leaf_Mold` | 33.3% | 6 |
| `Tomato__Tomato_mosaic_virus` | 30.0% | 10 |
| `Tomato_Late_blight` | 20-30% | 10 |
| `Potato___Early_blight` | 25-37.5%\* | 8 |
| `Tomato_Bacterial_spot` | 22.2% | 9 |

*\*Potato classes shifted slightly (-1 image net) after `Potato___healthy`
data was added; see the two-attempt breakdown below.*

**Classes with zero real-world data (the harder problem)**:

Two classes had **no real-world images anywhere** (PlantDoc train or test):
`Potato___healthy` and `Tomato__Target_Spot`. Real photos were sourced
externally for both -- one succeeded cleanly, one did not, and both
outcomes are reported (a negative result is still a result):

| Class | Source | n (train) | Held-out check | Result |
|---|---|---|---|---|
| `Potato___healthy` | Mendeley CC BY 4.0, Ethiopia farm photos (DOI 10.17632/v4w72bsts5.1) | 100 | **n=263** (all unused images) | **93.54% accuracy, 95% Wilson CI [89.89%, 95.93%]** |
| `Tomato__Target_Spot` | UF/IFAS extension (5 photos, cropped from a multi-panel figure) + Mendeley CC BY 4.0 Bangladesh field photos (DOI 10.17632/bpfd9cns5g.2, 15 photos) | 20 | n=5 | **1/5 correct (20%)**, and cost ~1.2 percentage points of accuracy on other classes (shared softmax head) |

A first Target_Spot attempt (5 photos only) was tried, scored 0/2, and was
**not promoted** -- documented in
`agents/disease/results/plantdoc_target_spot_experiment.json` alongside the
20-photo attempt that *was* promoted, as an explicit before/after
comparison of "not enough data" vs. "somewhat more data, still not enough."

### Row 7 fine-tune (extended, best-checkpoint tracked) -- INVALID FOR HEADLINE REPORTING, bug-confirmed

**Do not cite the numbers in this subsection as a comparison against the
deployed model's 37.65% or any zero-shot baseline.** Kept here for
transparency and reproducibility only, same treatment as the earlier
train/test-contamination finding
(`docs/evidence/metric6_confidence_calibration_evidence.json`'s
`CRITICAL_METHODOLOGY_FLAG`). A confirmed data-loading bug (not
correct-but-unlucky deduplication -- investigated and ruled out below)
silently dropped 2 of 13 classes from this run's test set entirely, so
its accuracy numbers measure something structurally different from
every other figure in this document, not a harder or easier version of
the same benchmark.

A separate fine-tune, built on model lineage row 6 (the independently-
trained flat "v2" checkpoint), not on the deployed hierarchical row 5 --
full provenance and the class-coverage discrepancy below in
`docs/evidence/model_lineage.md`'s "Row 7" section. Evidence:
`docs/evidence/plantdoc_finetune_evidence.json`. Checkpoint:
`agents/disease/checkpoints/disease_agent_plantdoc_finetuned_best.pt`
(not currently loaded by `orchestrator/pipeline.py`).

**Best vs. final epoch, and why both are reported**: test accuracy peaked
at **epoch 11 (32.39%)**, then declined over 8 more epochs to **epoch 19
(26.76%, early-stopped)** while train accuracy kept climbing the entire
time -- a clean overfitting-after-peak signature, the specific reason
best-checkpoint tracking and early stopping were added this round (rows
3-5 selected on final epoch only). Reporting final alone would have
understated the model's best achieved result by 5.6 points; reporting
best alone would overstate confidence in a single epoch out of 19 on a
small test set. Both are reported together, per the evidence file's own
caveat.

**Per-class recall at the best checkpoint (epoch 11)**:

| Class | Recall | n (test) |
|---|---|---|
| `Tomato_Leaf_Mold` | 50.0% | 6 |
| `Tomato_Septoria_leaf_spot` | 54.5% | 11 |
| `Tomato_Early_blight` | 44.4% | 9 |
| `Tomato_Late_blight` | 40.0% | 10 |
| `Potato___Early_blight` | 37.5% | 8 |
| `Potato___Late_blight` | 25.0% | 8 |
| `Tomato_Mosaic_virus` | 10.0% | 10 |
| `Tomato_Bacterial_spot` | 0.0% | 9 |

**Same small-sample-size caveat as every other per-class table in this
document**: n=6-11 per class -- each individual test image is worth
roughly 9-17 percentage points of that class's own recall. `0.0%` on
`Tomato_Bacterial_spot` (0/9) and `10.0%` on `Tomato_Mosaic_virus` (1/10)
are not statistically distinguishable from several-times-better results
at this n; read the ranking as directional, not precise.

**Class coverage: NOT "the same classes as the deployed model," verified
via direct arithmetic, not assumed.** Four classes have zero training
images in this run (`Potato___healthy`, `Tomato__Target_Spot`,
`Tomato_healthy`, `Tomato__Tomato_YellowLeaf__Curl_Virus`) and are
therefore structurally untestable here -- the model had no opportunity to
learn them. Only **two** of these four match the previously-established
"classes with zero real-world data anywhere" finding above
(`Potato___healthy`, `Target_Spot`) -- consistent with prior work. The
other two, `Tomato_healthy` and `Tomato_YellowLeaf_Curl_Virus`, are **not**
a continuation of anything previously documented: this document's own
per-class table above (current deployed model) reports real accuracy for
both (`Tomato_healthy` 37.5% n=8, `Tomato_YellowLeaf_Curl_Virus` 66.7%
n=6), meaning PlantDoc genuinely has images for these classes. Checked
directly: this run's test set totals n=71, and 85 (the established
test-only set used everywhere else in this document) minus 6
(`YellowLeaf_Curl_Virus`) minus 8 (`healthy`) equals exactly 71, with
every other overlapping class count identical between the two sets. This
is conclusive, not coincidental -- this run's data pipeline dropped these
two classes entirely from both train and test, on top of the two
genuinely-structural gaps.

**Root cause investigated and confirmed (2026-09-14), not assumed either
way**: checked directly against `data/plantdoc_raw` whether this was
correct-but-unlucky content-hash dedup or a bug. Content-hash deduplication
is ruled out on arithmetic alone -- the evidence file's own
`duplicates_removed_from_train` is 6, dataset-wide, while the raw
`Tomato leaf` and `Tomato leaf yellow virus` folders hold 55 and 70 train
images respectively (125 combined); a budget of 6 cannot explain 125
images vanishing. Confirmed directly by SHA-256 hashing every image in
both folders: 0 internal duplicates, 0 cross-class duplicates, only 1
single match against `Tomato_YellowLeaf_Curl_Virus`'s own test split (a
minor, legitimate dedup candidate, consistent with the small reported
total) -- no wholesale duplication anywhere. File corruption was also
ruled out (`PIL.Image.verify()` on all 125 images: 0 corrupt). The actual
cause is a **class-inclusion or folder-to-class mapping bug** in this
run's data-loading script (not present in this repo) that excluded these
two folders from loading at all -- a concrete, fixable defect, not an
open question. Full investigation: `docs/evidence/model_lineage.md`'s
"Row 7" section. **Practical consequence**: 32.39%/26.76% here is not
directly comparable to the deployed model's 37.65% -- the test sets
differ in composition (2 fewer classes represented, n=71 vs. n=85), not
just the checkpoint.

### Precision / recall / F1, latency, throughput, and model footprint

Accuracy alone doesn't show false-positive/false-negative balance, and
nothing above reports inference cost or model size. Computed by
`agents/disease/compute_full_metrics.py` against the current deployed
checkpoints on the same 85-image PlantDoc test set (per-image predictions
recorded this time, not just aggregate correct/incorrect):

| Metric | Value |
|---|---|
| Macro precision / recall / F1 | 0.334 / 0.294 / 0.299 |
| Weighted precision / recall / F1 | 0.432 / 0.377 / 0.385 |
| Mean inference latency | 240.2ms (std 20.7ms, n=85) |
| Throughput | 4.16 images/sec (single-sample, CPU/MPS, no batching) |
| Total parameters (3 stage models) | 12,041,859 |
| Total checkpoint size | 46.79MB |
| Peak RSS during inference | 517.5MB |

**A specific, useful finding from the per-class breakdown**:
`Tomato_healthy` has **precision 1.0, recall 0.375** -- the model almost
never mislabels a diseased leaf as healthy (no false "all clear"), but
does mislabel some healthy leaves as diseased (false alarms). For a farm
advisory tool this is the safer direction to be wrong in -- worth stating
explicitly rather than leaving buried in a table, since "high precision,
low recall on the healthy class" reads very differently once its
real-world implication is spelled out.

**IoU / Dice coefficient**: not applicable and not reported -- this is an
image classification task (one whole-image label per photo), not
segmentation or detection, so there's no predicted region to compare
against a ground-truth mask or box. Precision/recall/F1 above serve the
equivalent purpose for a classifier (proving low false-positive/
false-negative rates), which is why they're reported instead.

**Architecture ablation** (from the original training run,
`agents/disease/results/hierarchical_results.json` /
`flat_baseline_results.json`): a hierarchical model (crop classifier ->
per-crop disease classifier) scored 99.70% PlantVillage validation
accuracy vs. a flat 13-class classifier's 99.72% -- statistically
indistinguishable on the training distribution; the hierarchical
architecture was kept for its cleaner separation of crop-ID error from
disease-ID error (visible in the per-class PlantDoc breakdown above) rather
than for a raw accuracy gain.

### 2.6 Intent Router / TTS memory-collision fix -- cost-benefit

- **Root cause**: Ollama keeps a model resident in RAM for 5 minutes
  post-call by default; on an 8GB machine, the 7B Intent Router model and
  TTS collided for memory, producing the 191s+ stalls in Sec. 2.4.
- **Fix**: force-unload the Intent Router model immediately after each use
  (`backend/main.py`'s `_unload_intent_router_model()`).
- **Measured cost** (original 3-trial spot check): **5.6s / 12.1s / 10.7s**
  per call, vs. **~3.6s** for a warm (non-unloaded) call -- roughly a 2-3x
  per-question latency cost, traded for eliminating the ~191s collision
  entirely.
- **Formal 8-trial re-measurement** (`latency_proof/formal_benchmark.py`):
  **mean 13.58s, std 3.18s, min 11.45s, max 20.99s** (n=8) -- notably
  higher than the original 3-trial spot check. Reported honestly rather
  than reconciled: this run's 8 router trials immediately followed 8 TTS
  trials in the same process (Sec. 2.4), so residual memory/cache pressure
  from those TTS calls is a plausible confound that wasn't isolated here.
  The original 3-trial number and this 8-trial number were also measured
  in different sessions on the same physical machine, so ordinary
  session-to-session system variance (background processes, thermal
  state) can't be ruled out either. What both measurements agree on: the
  fix costs single-digit-to-low-double-digit seconds per question, several
  orders of magnitude below the ~191s collision it prevents -- that
  conclusion is not sensitive to which of the two numbers is more
  representative.
- This is reported as an explicit, accepted tradeoff, not a fully solved
  problem -- see Sec. 4 (Limitations).

**Full-request, per-stage latency** (Phase 8.1 metric #3,
`docs/evidence/metric3_latency_evidence.json`): 6 real end-to-end
`/api/ask` HTTP trials, covering different request shapes (weather-only,
price-only, disease-with-photo, multi-agent, voice). Per external review
(2026-09-13, Sec 2.4): median is now reported alongside mean, since
mean-vs-median gap is itself informative about skew --

| Stage | n | mean | median | min | max |
|---|---|---|---|---|---|
| `request_total_s` (full request) | 6 | 55.76s | 50.14s | 38.39s | 80.88s |
| `tts_s` | 6 | 38.72s | 35.13s | 23.08s | 67.93s |
| `intent_router_s` | 6 | 14.37s | 12.19s | 11.74s | 23.73s |

Mean sitting well above median on both `tts_s` and `request_total_s`
confirms the right-skew already visible in the min/max spread -- a small
number of slow requests pull the mean up, consistent with the
already-documented TTS text-length sensitivity and the intent-router
memory-collision confound (both above). **p95 is deliberately not
reported** -- at n=6 total trials (n=1-2 for several individual stages),
a 95th-percentile estimate would just be the max relabeled with false
statistical precision; roughly 20+ independent trials would be needed
before a real p95 claim is defensible. The min/max/std spread already
shown is the honest signal about tail latency at this sample size.

### 2.7 End-to-end pipeline characterization (text-only, no photo)

Every metric above is per-component. This section runs the **full,
live** pipeline (`tests/end_to_end_evaluation.py`, re-run today) across
the same 16-question set, capturing not just crash/no-crash (already
100% reliable, Sec. 4.2c's original result reproduced) but *what kind of
answer the farmer actually gets*.

- **Substantive-answer rate: 43.8% (7/16)** -- the rest fall back to a
  generic "no clear answer, consult a local officer" sentence. Root-cause
  breakdown of the 9 fallbacks (not a bare number -- each is individually
  classified):

| Cause | Count | Assessment |
|---|---|---|
| No Soil Agent exists at all (Q4, Q8, Q16) | 3 | Structural gap -- `wants_soil` is a real Intent Router category with zero implementation behind it |
| Disease intent, no photo attached (Q1, Q5, Q12, Q14) | 4 | Expected/correct -- this test set is intentionally text-only; the real app requires a photo for disease questions |
| No intent detected -- a greeting (Q13) | 1 | Expected/correct -- there is nothing to answer |
| **Conflict Resolver rule-table gap (Q10)** | 1 | **Genuine bug-adjacent finding -- FIXED 2026-09-13**, see below |

- **The Conflict Resolver gap, precisely diagnosed and now fixed**: Q10
  ("Should I sell or hold, considering price and weather?") returns
  `weather_state=rain_risk`, `price_state=sell_now`, `disease_state=None`.
  `orchestrator/conflict_resolver.py`'s rule table had an entry for
  `(None, "rain_risk", "hold")` but **no entry for
  `(None, "rain_risk", "sell_now")`** -- an asymmetric gap (one price
  state covered, the other not) that fell through to the generic
  low-confidence fallback instead of a real answer. This was a small,
  precisely-located, fixable gap in the rule table, not a systemic design
  flaw -- flagged here rather than fixed silently when first found, since
  a paper's evaluation section is exactly where this kind of finding
  belongs. **Fixed as Phase 8.1 metric #5**: added to the rule table
  (weather and price agree here -- both point to harvesting now -- so
  it's resolved as a synergy case, `is_conflict=False`, `harvest_now`,
  High confidence, not a conflict needing arbitration) and added as
  scenario 13 in `tests/conflict_scenarios.json`, a permanent regression
  test (now part of the 13/13 pass rate in Sec. 2.0/2.2). See
  `docs/evidence/metric5_conflict_rule_table_evidence.json` for the
  updated evidence.
- **Important scope caveat**: the 43.8% figure is an artifact of this
  test's text-only, no-Soil-Agent setup, not a general "56% of farmer
  questions get no answer" claim -- 7 of the 9 fallbacks are correct,
  expected behavior for this specific test condition. Only 1 of 16 (6.25%)
  is a genuine, fixable coverage gap.

### 2.8 Disease-with-photo end-to-end accuracy (new: full pipeline, not just the classifier)

Every Disease Agent number in Sec. 2.5 measures `predict_disease()` in
isolation. This measures the **full pipeline's final spoken-answer text**
against real PlantDoc photos with known ground truth, run through Intent
Router -> Disease Agent -> Conflict Resolver -> Phrasing Templates.

- **Method**: 30 real PlantDoc test images (3 per class, 10 classes),
  question "ఈ ఆకుకు ఏమైంది?" ("What happened to this leaf?"), full
  `run_pipeline()` call per image. Correctness requires the final Telugu
  answer to name **both the correct crop and the correct disease** --
  an earlier draft of this check matched on disease name text alone and
  was caught overcounting, since several Telugu disease labels are reused
  across crops (e.g. `Tomato_Early_blight` and `Potato___Early_blight`
  both render as "ఎర్లీ బ్లైట్"); fixed before reporting.
- **Result**: **n=30, 36.7% (11/30) end-to-end accuracy**, identical to
  the component-level accuracy on this same 30-image subset (also 36.7%).
- **Wiring integrity: 100%** -- every case where `predict_disease()` got
  the right answer, the final spoken answer also got it right. This is
  the more important number for validating the *pipeline*, as opposed to
  the *classifier*: it confirms Conflict Resolver and Phrasing Templates
  never drop or corrupt a correct diagnosis on the way to the farmer --
  the end-to-end accuracy ceiling is set entirely by the Disease Agent's
  own accuracy (Sec. 2.5), not by anything downstream of it.
- **Note on reproducibility**: a first run of this same check (before the
  crop+disease fix above) measured component-level accuracy at 33.3%
  (10/30) vs. this run's 36.7% (11/30) on the identical 30 images and
  model weights -- a 1-image difference attributable to Apple MPS
  backend's known minor run-to-run floating-point nondeterminism, not
  model or methodology instability. Flagged rather than silently using
  whichever run looked better.
- Full per-image results: `docs/end_to_end_eval_results.json`.

### 2.9 Ablation Study: Voice vs. text (Phase 8.1, metric #4) -- FINAL, n=64, 4 speakers

**Formal framing**: this is an ablation study in the standard sense --
one component of the pipeline (ASR) is removed/bypassed while every
other component (intent router, agents, conflict resolver, phrasing) is
held fixed, isolating that single component's causal contribution to
downstream behavior and failures. Where a typical model ablation removes
a layer or a training signal and measures the accuracy delta, this
ablation removes the ASR step and measures the *behavioral* delta --
whether the pipeline's decisions (which intents fire, which agents run,
what answer is phrased) change when voice is swapped for text carrying
the identical semantic content. The two confirmed findings below (Sec.
2.9) are exactly this kind of ablation result: one isolates the ASR
component's specific causal contribution to a real failure (the pH
transliteration flip), the other rules the ASR/voice path in as the
locus of a separate failure (the live-API timing issue) by showing it
does not reproduce on the text-only control path.

Does going through ASR change what the pipeline does, compared to typing
the exact same question? Each of the 16 questions in
`tests/intent_router_test_set.json` run through the real, unmocked
`run_pipeline()` twice per speaker, across all 4 speakers (expanded
2026-09-16 from the original n=16 single-speaker preliminary run, once
metric #2's 4-speaker recordings closed): once via that speaker's
recorded WAV -> `backend.voice.transcribe_audio()` (the exact production
ASR call path), once as the reference text typed directly, run fresh per
speaker iteration to stay time-paired with its voice call. Both paths use
live, unmocked agent calls (Weather API, Price API, Ollama intent
router). No image attached (this test set has none), so the Disease Agent
never fires. Stage-by-stage comparison, not just final-answer pass/fail,
so an ASR mishearing and a downstream (intent-router/state-mapping/
conflict-resolver) difference stay distinguishable rather than collapsed
into one number.

- **Result**: n=64 (16 questions x 4 speakers). Intent-router output
  matched **60/64**; agents-fired matched **60/64**; final answer matched
  **58/64**. Of the 6 apparent mismatches, every single one is now fully
  explained -- **4 are the confirmed structural ASR-transliteration
  finding below (one per speaker, same root cause), and 2 were
  investigated and ruled out as live-API failures unrelated to
  voice-vs-text pipeline logic (see below). After investigation, zero
  mismatches in this 64-pair run are attributable to genuine
  state-mapping/conflict-resolver sensitivity.**

  | Speaker | Intent match | Agents-fired match | Final answer match |
  |---|---|---|---|
  | speaker1 | 15/16 | 15/16 | 15/16 |
  | speaker2 | 15/16 | 15/16 | 15/16 |
  | speaker3 | 15/16 | 15/16 | 15/16 |
  | speaker4 | 15/16 | 15/16 | 13/16 |

- **Headline finding, now CONFIRMED STRUCTURAL across all 4 speakers**
  (question 8, "నేల pH ఎంత ఉండాలి టమాటా కోసం?" / "What soil pH is needed
  for tomato?"): every one of the 4 speakers' independent recordings had
  ASR transcribe "pH" as its Telugu transliteration ("పీహెచ్"/"పియచ్",
  spelled slightly differently per speaker but always transliterated)
  instead of the reference text's Latin-script "pH" -- and **all 4**
  triggered the same intent-router flip, from `wants_soil=True` to
  `wants_price=True`. This is no longer a single-speaker curiosity: **4
  independent speakers, 4 independent recordings, 4 identical behavior
  changes** is strong evidence this is a structural property of how the
  intent router weighs Latin-script vs. transliterated technical terms,
  not an artifact of one person's pronunciation. A farmer asking about
  soil pH by voice would reliably receive a tomato price quote instead,
  regardless of who is speaking. Flagged as a concrete direction for
  future work (e.g. normalizing known technical terms before intent
  routing), not applied as a fix here -- this metric is observation only,
  `orchestrator/pipeline.py` was not touched.

- **Secondary finding: 2 voice-path live-API failures, investigated and
  distinguished from pipeline-logic mismatches.** Both of the run's
  remaining 2 mismatches (speaker4, question 2 "Will it rain tomorrow?"
  and question 7 "Will potato market price rise?") initially looked like
  generic answer-text divergence. Checked, not assumed: both voice-path
  final answers exactly matched the source-code fallback strings for a
  live API call outright failing (`requests.exceptions.RequestException`
  in `agents/weather/weather_agent.py`/`agents/price/price_agent.py`),
  not the "no data reported today" branches. Both occurred on the
  **voice** path specifically -- zero such failures anywhere on the text
  path across all 64 pairs. A plausible mechanism, flagged as plausible
  and not proven: every voice-path call is immediately preceded by a
  20-30s ASR transcription (a heavy model-inference call with no text-path
  equivalent), and this project's own Sec. 2.6 already documents a related,
  independently-confirmed resource-contention pattern on this same 8GB
  machine (Intent Router latency increasing when multiple models are
  concurrently resident). Consistent with that prior finding, not
  isolated as a controlled variable here. Reclassified from
  `DOWNSTREAM_ANSWER_DIVERGED` to `LIVE_API_FAILURE_VOICE_PATH`, original
  classification kept in the evidence file for transparency.

- **FINAL, 4/4 speakers, n=64** -- no longer preliminary. Cross-speaker
  robustness of the ASR-to-intent-router pathway is now measured, not
  just single-speaker behavior. Remaining gaps: 4 speakers is still a
  modest n for claiming the exact population frequency of the Q8 finding
  (though its structural mechanism, not just its existence, is now
  confirmed); the Disease Agent path remains unexercised (no images in
  this test set); and the voice-path API-failure mechanism is plausible
  but not causally isolated.
- Full evidence, raw log, and reusable generator script:
  `docs/evidence/metric4_voice_vs_text_ablation_evidence.json`,
  `docs/evidence/metric4_voice_vs_text_raw_log.txt`,
  `tests/generate_voice_vs_text_ablation_evidence.py`.

## 3. Related work (starting point for literature review)

*A conference/journal submission needs a fuller literature search than this
one session can responsibly do -- treat this as a starting bibliography,
not a complete related-work section.*

**Real-world plant disease classification**: PlantDoc is an established
benchmark specifically because it exposes the PlantVillage-to-field
accuracy gap this project also measured -- published results vary widely
by model scale and training data (e.g., large hybrid transformer models
report 95-99% on PlantDoc when trained substantially on in-domain data;
smaller/lighter models trained mostly on PlantVillage report much lower
numbers, consistent with the 28-40% range found here for a
similarly-scoped small-data fine-tune). A paper draft should cite specific
comparable-scale baselines rather than the largest SOTA systems, to keep
the comparison fair.

### 3.1 Direct comparison against comparable published systems

Three systems were read in enough depth to compare honestly (not just
named), all pulled directly from `krishiagent-literaturesurvey.docx`
(our own 25-paper survey), not sourced separately -- **Farmer.Chat**
([15], arXiv:2409.08916, Digital Green & Microsoft Research -- the most
detailed and most comparable: a large-scale, publicly documented
deployment), **Kisaan Margadarshak** ([13], Trimukhe et al. 2025,
Springer, DOI:10.1007/978-3-031-74440-2_1 -- rated ★ HIGH in our survey,
and already the base paper the rest of this report treats as the closest
structural precedent), and **CropCare Companion** ([14], Sable et al.
2025, IJRASET -- rated ⚠ CAUTION in our survey; its own >91% accuracy
figure is reported here with that caveat attached, not at face value).

| Dimension | Krishi-Agent | Farmer.Chat [15] | Kisaan Margadarshak [13] | CropCare Companion [14] |
|---|---|---|---|---|
| Architecture | Multi-agent + explicit rule-based Conflict Resolver | RAG + multi-agent orchestration (Planning/Execution/Tooling agents) | Android app -- on-device MobileNetV2 CNN (disease) + Django backend + Flutter frontend, weather + mandi price API integration | Naive Bayes classifier + DeepSeek LLM fallback; NLP pipeline (tokenization/stemming/stop-word removal) + Google Translate + Web Speech API |
| Explicit conflict resolution across advice types (weather/price/disease) | **Yes** -- rule table, 13/13 on synthetic tests (7/36 of the full 3-agent state grid); 1 real gap found via live-data testing (Sec. 2.7) and fixed | Not described in the paper | Not described in the survey entry | Not described in the survey entry |
| Disease/pest diagnosis | Own trained, fine-tuned CV model, in-pipeline (Sec. 2.5, honestly measured at 37.65%) | Delegated to a third-party service (Plantix) | On-device MobileNetV2 CNN, but trained on a **single-crop (cotton)** dataset (Kaggle) -- no quantified accuracy/F1 reported | **No image-based disease detection** -- explicitly named as future work, not yet implemented |
| Confidence/uncertainty shown to the user | **Yes** -- hedged phrasing templates + a UI confidence indicator, both keyed to the real underlying confidence level | Not described (feedback is retrospective thumbs-up/down, not prospective confidence) | Not described in the survey entry | Not described in the survey entry |
| Languages | Telugu (bilingual UI) | 6 languages incl. Telugu, deployed across 4 countries | Multilingual UI (specific languages not itemized in the survey entry) | Hindi, Marathi, Gujarati, English |
| Evaluation scale | Component-level (n=12-263) + two fresh-clone reproducibility tests; **no live user study yet** | **15,000+ real users, 300,000+ queries**, formal focus groups + bi-weekly satisfaction surveys | **No quantified accuracy/F1 or user-scale reported** -- the survey's own stated limitation, not an omission on our part | **>91% accuracy** across multilingual inputs -- but on a **proprietary Q&A dataset of undisclosed size/source**; ⚠ CAUTION-rated in our survey, not cited here as full-confidence |
| Response latency (reported) | ~13.6s intent routing + ~24-26s TTS for a full voice answer (Sec. 2.4/2.6) | 9.05s average (text response; not confirmed whether this includes voice synthesis) | Not reported in the survey entry | ~1.7s average response time (text; multilingual NLP + LLM fallback, no voice synthesis stage to account for) |

**Honest reading of this table, not a one-sided one**: Farmer.Chat
massively outscales this project on real-world deployment and user-study
rigor -- that is a genuine, uncontested strength of theirs, not something
to argue around. CropCare Companion's own reported >91% accuracy is also
higher than any Krishi-Agent number in this report, and is stated here
rather than omitted -- but it comes with a caveat that matters: that
figure is measured on a proprietary, undisclosed-size Q&A dataset with no
independent evaluation, which is exactly why our own survey rated that
paper ⚠ CAUTION rather than ★ HIGH, and why it isn't treated here as a
like-for-like comparison against this report's own accuracy figures
(each of which traces to a named, sized, and in most cases
publicly-sourced dataset). What this comparison actually supports is a
narrower, verifiable, **system-level** claim, not a component-level one
(the Disease Agent's own ~22% real-world accuracy is not the novelty
claim -- see Sec. 2.5's honest framing of that as a structural,
domain-shift-bound limitation):

1. **Conflict arbitration between competing advice signals is explicitly
   engineered and independently tested** -- a deterministic rule table
   (13/13 on synthetic tests, 7/36 of the full 3-agent-active state grid,
   Sec. 2.2), one real gap found via live-data testing and fixed
   (Sec. 2.7), with denominator-honest coverage reporting built into the
   claim itself rather than left implicit (Sec. 2.2) -- and is not
   described as present in any of the three comparable systems surveyed
   above.
2. **Confidence calibration is validated empirically, not merely
   asserted.** A five-signal convergence (accuracy band, formal confusion
   matrix, Expected Calibration Error, a formal McNemar's-test
   significance check across the three same-family checkpoints --
   p=0.316/0.754/0.179, all non-significant -- and now a fourth,
   independently-architected checkpoint (ResNet18) reproducing the same
   accuracy band, attractor bias, and formal indistinguishability,
   p=0.369/0.156/0.083) is now confirmed across four independently-
   trained zero-shot checkpoints spanning two distinct architecture
   families, not one (Sec. 2.5) -- **and, complementing that
   cross-architecture agreement, the estimate itself is stable under
   resampling of the training data**: 5-fold cross-validation on
   EfficientNetB0 gives PlantDoc zero-shot 23.38% ± 0.83% across folds
   (Sec. 2.5), so the ~22-23% real-world ceiling is not just something
   independent architectures happen to agree on once, but a number that
   holds up when the same recipe is retrained on different partitions of
   the same data -- and the deployed
   confidence cutoff was stress-tested with Wilson
   95% confidence intervals and a 2,000-iteration bootstrap stability
   check before any change was even considered, ultimately supporting a
   documented "hold, don't change" decision rather than a headline-
   chasing point estimate (Sec. 2.5). None of the three comparable
   systems describe prospective confidence communication to the user at
   all (Farmer.Chat's feedback is retrospective thumbs-up/down, not
   prospective).
3. **A confirmed, cross-speaker structural failure mode at the
   ASR-to-intent-router boundary was found, characterized, and
   reproduced -- not just observed once.** A domain-specific technical
   term (soil pH) transliterated by ASR into Telugu script flips the
   intent router's classification from `wants_soil` to `wants_price`;
   this reproduced independently across all 4 recorded speakers (Sec.
   2.9's ablation study), demonstrating the failure is structural to how
   the intent router weighs Latin-script vs. transliterated technical
   vocabulary, not one recording's fluke. This class of failure -- voice
   input silently changing system behavior in ways a text-only test
   suite would never catch -- is exactly the kind of gap a voice-vs-text
   ablation exists to find, and none of the three comparable systems
   report running one.

That is the project's specific, evidence-backed novelty contribution --
not a claim of being "better overall," which the evaluation scale gap
above would not support. Separately, and not part of the novelty claim
itself: the Intent Router's own model-size finding (Sec. 2.1 -- the 3B
model scored 50%, the deployed 7B model 87.5%/81.8% held-out) is
consistent with the literature survey's documented finding that small
(7-8B and below) LLMs can underperform on domain-specific coordination
tasks (Radeva et al. [16], per `orchestrator/intent_router_results.md`)
-- that citation supports the model-size decision specifically, not the
conflict-arbitration claim above, which rests on this project's own
rule-table evidence, not an external literature anchor.

**Real-world plant disease classification**: PlantDoc is an established
benchmark specifically because it exposes the PlantVillage-to-field
accuracy gap this project also measured -- published results vary widely
by model scale and training data (e.g., large hybrid transformer models
report 95-99% on PlantDoc when trained substantially on in-domain data;
smaller/lighter models trained mostly on PlantVillage report much lower
numbers, consistent with the 28-40% range found here for a
similarly-scoped small-data fine-tune). A paper draft should cite specific
comparable-scale baselines rather than the largest SOTA systems, to keep
the comparison fair.

## 4. Limitations (for a paper's Limitations section)

- **Disease Agent real-world accuracy remains modest** (37.65% end-to-end)
  for 8 of 10 tomato/potato disease classes with real data; genuinely
  reliable only for crop identification (82%) and `Potato___healthy` (94%).
  `Tomato__Target_Spot` (20%) and `Potato___healthy`'s absence from
  PlantDoc entirely are both root-caused to real-world training data
  scarcity, not an architecture problem.
- **Weather drought signal is a short-range proxy** (5-day forecast
  uniformly dry+hot), not true multi-week drought detection, which a
  5-day forecast cannot observe.
- **The 0.7 disease confidence cutoff** (treat_now vs. monitor) is
  described in-code as "a draft threshold, not yet calibrated" -- a Phase 8
  spot-check found weak but real separation (26.5% vs 19.9% accuracy
  above/below the cutoff), not a rigorously calibrated threshold.
- **The Price Agent's ±10% sell/hold thresholds** (`SELL_THRESHOLD`/
  `HOLD_THRESHOLD`, `agents/price/price_agent.py`) are unvalidated
  constants, the same category of gap the disease confidence cutoff was
  before that investigation. A descriptive check (2026-09-13,
  `docs/evidence/price_threshold_distribution_evidence.json`) confirmed
  the thresholds are at least a non-degenerate operating point on the
  real 2002-2026 historical distribution (roughly a 34%/56%/10%
  sell/hold/neutral split for Telangana Tomato, not near-0% or near-100%
  on either side) -- but this is descriptive only, not a validation that
  the values are well-calibrated to real farmer outcomes. A rigorous
  outcome-based backtest (does triggering `sell_now` at a given pct_diff
  actually correlate with a better outcome than waiting?) was
  deliberately **not** attempted: it requires an assumed holding horizon
  (tomatoes are perishable, so "wait N days" needs a specific N), and
  this project has no solid agronomic grounding for that assumption --
  picking one arbitrarily would introduce more uncertainty than it
  resolves. Held as explicit future work, not a time-constraint
  shortcut (see Sec. 5 item 8).
- **The Price Agent's seasonal baseline is an arithmetic mean over a
  right-skewed price distribution, which mechanically biases it toward
  triggering `hold` far more often than `sell_now`** -- a distinct
  finding from the threshold-value question above, not a restatement of
  it. The same 2026-09-13 distribution check
  (`docs/evidence/price_threshold_distribution_evidence.json`) found
  `hold` firing ~1.7x more often than `sell_now` across every
  state/commodity combination tested (e.g. Telangana Tomato: 56.5% hold
  vs. 33.8% sell_now). Root cause, verified directly: raw Telangana
  Tomato modal price has skew 2.23 (mean ₹1,454 vs. median ₹1,000) --
  a classic right-skewed commodity-price distribution where occasional
  price-spike days pull the mean well above the price a "typical" day
  actually sees. Since `_seasonal_baseline()` (`agents/price/
  price_agent.py`) uses that inflated mean as the reference point, a
  typical day's price sits below baseline more often than above it by
  construction -- `hold` fires more often not because market conditions
  favor holding more often, but because the baseline itself is skewed
  upward by rare high-price days. This is independent of whatever the
  "correct" ±10% threshold value turns out to be: even a perfectly
  chosen threshold applied to a systematically skewed baseline would
  inherit this same asymmetry. Not fixed here -- see Sec. 5 item 8 for
  the specific proposed change.
- **Intent Router latency tradeoff** is accepted, not eliminated (Sec. 2.6).
- **All evaluation sets in this document are small** by publication
  standards (n=12-85 depending on component) except the newly-expanded
  Potato_healthy check (n=263). This is explicitly flagged, not hidden --
  see Sec. 5 for what would close this gap.
- **No end-to-end human evaluation of final spoken answers exists yet** --
  every number above is per-component (ASR alone, disease classifier alone,
  etc.), not a holistic judgment of whether a full pipeline answer is
  correct, helpful, and appropriately hedged.
- **No user study with real farmers/Telugu speakers has been conducted.**

## 5. What's still needed for a competitive submission

In priority order:

1. ~~**End-to-end answer-quality evaluation.**~~ **Done for the
   automatable parts** -- Sec. 2.7 (text-only pipeline characterization,
   with root-caused fallback analysis) and Sec. 2.8 (disease-with-photo
   full-pipeline accuracy + wiring integrity) were built and run. Still
   open: factual correctness for live Weather/Price answers isn't
   automatable at all (the "right answer" changes day to day with real
   API data), and actionability genuinely needs a human judge (see the
   rubric below, updated to reflect what's done vs. still open).
2. **Statistical power.** Most component evals above are still n<20
   (Intent Router, Conflict Resolver, ASR, the disease-with-photo
   end-to-end check). Potato_healthy is the one exception, now n=263.
   Where more real-world data can be sourced (more PlantDoc-style photos,
   more speakers for ASR, a larger hand-written question set), doing so
   would materially strengthen every remaining small-n claim.
3. **A small human/user study.** Even 5-10 Telugu-speaking test users doing
   a handful of realistic tasks, plus a short usability survey (e.g.
   System Usability Scale), would be the single highest-value addition for
   reviewer credibility on a "voice-first for farmers" claim. Needs real
   people -- can't be simulated.
4. ~~**Multi-speaker ASR evaluation**~~ **Done** -- Sec. 2.3 now reports
   the full 4-speaker, 64-utterance FINAL result (11.19% overall,
   8.09-13.62% per-speaker range) plus a per-question breakdown showing
   the difficulty pattern is vocabulary-driven, not speaker-driven.
   Background-noise robustness specifically remains unmeasured (all 4
   speakers recorded in similar quiet conditions) -- a narrower residual
   gap than "single-speaker," not the same gap.
5. **A proper literature review** -- Sec. 3 is a starting point, not a
   finished related-work section.
6. ~~**Formal latency benchmarking methodology**~~ **Done** -- Sec. 2.4/2.6
   now report real n=8-trial mean/std/min/max with stated hardware spec,
   not single-run numbers. It also surfaced a genuine, honestly-reported
   discrepancy (the Intent Router's 8-trial mean came in higher than the
   original 3-trial spot check, with a plausible confound noted rather
   than hidden) -- exactly the kind of thing formal benchmarking is
   supposed to catch.
7. **Price Agent threshold outcome validation.** The descriptive
   distribution check above (Sec. 4) confirms the ±10% thresholds aren't
   a degenerate operating point, but doesn't validate them against real
   farmer outcomes. A proper version needs an outcome-based backtest
   (does `sell_now` at a given pct_diff actually beat waiting?) against
   an explicitly-justified holding-horizon assumption -- tomato
   perishability makes this a real agronomic question, not just a
   modeling detail, and deserves domain input before being attempted
   rather than an arbitrarily chosen horizon. Data already available and
   sufficient (`data/price_history_ap_telangana.csv`, 2002-2026,
   116K+/39K+ rows for Tomato/Potato) -- this is a scoping/domain-input
   gap, not a data-availability one.
8. **Switch the Price Agent's seasonal baseline from arithmetic mean to
   median.** Specific, named fix for the mean-vs-skew asymmetry in Sec.
   4: `_seasonal_baseline()` (`agents/price/price_agent.py`) currently
   returns `subset["Modal_Price"].mean()`; switching this one line to
   `.median()` would make the baseline robust to the rare high-price
   spike days that currently pull it upward, and should materially
   reduce (though not by itself validate) the ~1.7x `hold`-over-`sell_now`
   trigger asymmetry documented in Sec. 4 -- an untested prediction, not
   a guarantee, since the confidence-bucket thresholds (0.08/0.20) and
   sell/hold thresholds (±0.10) were never chosen with a median baseline
   in mind and would need re-characterizing against it (rerun
   `agents/price/generate_threshold_distribution_evidence.py` against
   the changed formula) rather than assumed to still be well-placed.
   Small, mechanical, low-risk change in isolation; explicitly not made
   in this pass because it changes live pipeline behavior (every
   `sell_now`/`hold`/`neutral` decision's boundary would shift) and
   needs its own sign-off and before/after comparison, same discipline
   as every other pipeline.py change this evaluation phase.

### Post-Defense Journal Extension Roadmap

The items below are deliberately **out of scope for the defense
submission** -- not unfinished work, but rigor that belongs to a
different venue tier than this project targets right now. Per the
project's own guiding scope document (Publication Plan section):
the realistic target is a student research track, workshop paper, or
regional/national journal, explicitly *not* a top-tier international
conference. The items here are the specific additional rigor a future
journal-tier extension would need that a workshop/regional-journal-tier
defense submission does not -- listed explicitly and by name so "what's
deliberately deferred" is as documented as everything else in this
report, not left implicit.

1. **Baseline reproduction under controlled conditions.** Every
   accuracy figure in this report comes from a single training/eval run
   per checkpoint (with honest exceptions already noted where
   nondeterminism was caught and reported, e.g. Sec. 2.8's 33.3% vs.
   36.7% MPS-backend floating-point difference). A journal-tier
   submission would re-run each checkpoint's training from scratch
   multiple times (different random seeds, same hyperparameters) and
   report variance, not a single point estimate -- distinguishing
   "this specific run's number" from "this architecture's expected
   performance band." **Still open as originally scoped** -- a related
   but different piece of evidence was added 2026-09-16 (Sec. 2.5 signal
   5, `model_lineage.md` Row 8): a *different architecture*
   (ResNet18) trained once under a matched protocol, not the *same*
   architecture retrained with different seeds. That checkpoint answers
   "does a different backbone reproduce this finding" (yes, formally
   confirmed by McNemar's test), which is stronger evidence for the
   domain-gap claim than seed variance would have been, but it does not
   answer the seed-variance question this item specifically names --
   stated honestly rather than counted as covering it.
2. ~~**Formal statistical significance testing (McNemar's test) across
   model comparisons.**~~ **Done for the zero-shot triad, and extended to
   a fourth, cross-architecture checkpoint** -- pulled forward and run
   2026-09-16 (Sec. 2.5's five-signal convergence,
   `docs/evidence/mcnemar_zero_shot_checkpoint_comparison_evidence.json`
   and `docs/evidence/mcnemar_row8_resnet18_vs_zeroshot_evidence.json`):
   all three pairwise comparisons among rows 1/2/6 (p=0.316/0.754/0.179)
   are non-significant, formally confirming the accuracy-band convergence
   rather than leaving it as an eyeballed gap; the same test between the
   ResNet18 checkpoint (row 8) and each of rows 1/2/6 (p=0.369/0.156/
   0.083) is also non-significant, extending the formal confirmation
   across architecture families. **Still open**: the
   deployed model's 37.65% vs. the zero-shot 23.45% headline is a
   different comparison (different checkpoints, different test-set sizes
   -- n=85 test-only vs. n=965-968 train+test combined, not a like-for-
   like paired sample the way the zero-shot triad is) and was not tested
   here; a McNemar comparison for that pair would need the zero-shot
   checkpoint's predictions restricted to the same 85 test-only images
   first, not attempted in this pass.
3. ~~**k-fold cross-validation.** Every PlantDoc evaluation in this
   report uses PlantDoc's own fixed train/test split (the benchmark's
   standard split, used so results stay comparable to other published
   PlantDoc numbers). A journal-tier extension would additionally run
   k-fold cross-validation within the training data to characterize
   variance from the specific train/test partition itself, separate
   from architecture or checkpoint variance -- a different, complementary
   question to the one this report answers.~~ **Done, explicitly scoped**
   -- pulled forward and run 2026-09-17
   (`docs/evidence/kfold_cv_row6_variance_evidence.json`,
   `docs/evidence/model_lineage.md`'s "K-fold cross-validation variance
   estimate" section): 5-fold stratified CV on EfficientNetB0 (row 6/v2's
   recipe), CV pool = train+val only (original held-out test set
   untouched throughout), gives **PlantDoc zero-shot 23.38% ± 0.83%**
   (fold range 22.02-24.33%) -- a tight variance band directly answering
   this item's question of partition-specific variance. **Scoped
   honestly, same as item 1's compute-cost scoping note**: run on **one
   architecture only** (EfficientNetB0; does not establish ResNet18's or
   the hierarchical split's fold-to-fold variance), at a **reduced
   15-epoch/fold budget** (vs. the original checkpoints' 30), and on a
   PlantDoc image set (n=822) that matches row 8's coverage-gap-affected
   benchmark rather than rows 1/2/6's fuller 967/968 -- so the fold
   spread itself is a clean apples-to-apples comparison (all 5 folds
   share the same 822 images), but the 23.38% mean isn't directly
   comparable number-for-number to rows 1/2/6's headline figures. A
   strictly monotonic downward drift across the 5 folds is visible in
   the raw data and reported as an observed pattern, not investigated or
   claimed as a trend at n=5.

These are named explicitly, with the reasoning for deferring each,
rather than left as a vague "more rigor would be nice" gesture --
consistent with this report's standing rule that a gap gets stated
honestly, not smoothed over, whether the gap is in evidence already
gathered or in evidence deliberately not yet attempted.

### End-to-end answer-quality rubric -- status

- **Intent correctness** (0/1) -- **done**, Sec. 2.1/2.7 (81.8%
  held-out / 87.5% full-set on the 16-question set -- see Sec. 2.1 for
  why the held-out number is the one to cite; Sec. 2.7 additionally
  root-causes every fallback).
- **Wiring/consistency correctness** (0/1: did the final answer faithfully
  reflect what the underlying agent(s) actually said, with nothing lost
  or corrupted in Conflict Resolver + Phrasing?) -- **done** for the
  disease path (Sec. 2.8, 100% wiring integrity, n=30). Not yet done for
  Weather/Price, though the mechanism (compare agent's raw `action` text
  against the final answer) is the same and cheap to extend.
- **Factual correctness against ground truth** (0/1/2) -- **done** for
  Disease (Sec. 2.8, since PlantDoc photos have known labels). **Not
  automatable** for Weather/Price -- there is no fixed ground truth for
  "will it rain tomorrow" on a given real day; this needs either frozen
  historical API responses (buildable) or a human check against the raw
  agent output (not the API itself).
- **Confidence calibration** (0/1: does the hedging language match the
  underlying confidence level?) -- **mechanism already verified**:
  `tests/test_phrasing_templates.py` confirms `phrase_resolution()`
  selects the exact `TEMPLATES[(resolution, confidence)]` entry for all
  13 synthetic (resolution, confidence) pairs (13/13, re-run today --
  corrected 2026-09-17 from a stale 12/12, see Sec. 2.0). What
  Sec. 2.7 additionally shows is that on live data, most real questions in
  the 16-question set never reach a *multi-agent* resolved-conflict
  template at all (`phrase_resolution()` is only called when >1 agent
  fires; single-agent answers pass the agent's raw text through
  unmodified) -- so confirming this on live, multi-agent, disease+weather
  or disease+price combinations (which needs a photo to give disease_state
  a non-None value) is the one piece still worth doing, not the template
  logic itself.
- **Actionability** (0/1: does a farmer come away knowing what to do?) --
  **needs a human judge**, or an explicitly-caveated LLM-as-judge proxy
  (usable as a preliminary signal, not a substitute for human eval in the
  paper's actual reported numbers).
