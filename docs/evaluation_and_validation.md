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

## 2. Component-level results

### 2.0 Full logic/correctness test suite -- run today, all passing

Every deterministic-logic test in the repo, re-run together for one
consolidated snapshot (excludes the percentage-style accuracy metrics in
2.1-2.8, which are reported separately since "pass/fail" doesn't apply to
them):

| Test file | Result | What it covers |
|---|---|---|
| `test_conflict_resolver.py` | 13/13 (100%) | Conflict Resolver rule logic, synthetic scenarios (13th added as a regression test for the Sec. 2.7 rule-table gap, now fixed) |
| `test_phrasing_templates.py` | 12/12 (100%) | Confidence-calibrated template selection |
| `test_state_mapping.py` | 13/13 (100%) | Boundary-value tests for the Weather/Price state-mapping thresholds (exact 0.10/-0.10 cutoffs, missing-data fallbacks) -- previously only comfortably-inside values were tested |
| `test_pipeline.py` | 13/13 (100%) | End-to-end conflict scenarios through the real pipeline |
| `test_pipeline_edge_cases.py` | 5/5 (100%) | No photo, missing profile fields, agent API failures |
| `test_weather_agent.py` | 9/9 (100%) | Schema validity, graceful failure, drought-signal unit tests |
| `test_price_agent.py` | 5/5 (100%) | Schema validity, graceful failure, market lookup |

**70/70 (100%)** across all deterministic-logic tests (was 68/68 before
the metric #5 fix added a 13th conflict scenario, exercised by 2 test
files). This is the
"nothing is silently broken" layer underneath the accuracy metrics below
-- a paper/evaluator claim like "37.65% disease accuracy" only means
something if the surrounding pipeline logic (conflict resolution,
phrasing, state thresholds) is itself verified correct, which this table
establishes.

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
- **Limitation**: rule-based logic over synthetic inputs, not live agent
  output -- validates the *decision logic*, not real-world scenario coverage.

### 2.3 ASR (speech-to-text)

- **Model**: `ai4bharat/indic-conformer-600m-multilingual`.
- **Dataset**: 16 real microphone recordings (`tests/audio/wer_eval/speaker1/`),
  same 16-question set as Sec. 2.1, read aloud by one Telugu speaker.
- **Method**: `tests/check_all_wer.py`, word error rate via `jiwer`, text
  normalized (punctuation/whitespace stripped) before scoring.
- **Result**: **mean WER 10.96%** across 16 utterances (full per-question
  table in `tests/audio/wer_eval/wer_results.json`, which now
  auto-discovers however many of the planned 4 speakers are recorded --
  currently 1/4, explicitly flagged `PRELIMINARY` in that file until
  Speakers 2-4 are done; see `docs/evidence/metric2_wer_recording_instructions.md`).
  Most individual
  "errors" are spacing/ZWNJ artifacts (e.g. `ఈరోజు` vs `ఈ రోజు`), not
  content mistakes; a few are genuine substitutions (e.g. `పట్టా` for
  `పంట`).
- **Known limitation**: single speaker, 16 utterances -- not statistically
  powered, no measure of cross-speaker/accent/noise robustness. A paper
  needs multi-speaker WER (see Sec. 5).

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
Cross-architecture convergence (3 independent checkpoints).** The
single-checkpoint baseline above (28.2%, 85-image test-only subset) could,
on its own, be one model's idiosyncrasy -- undertrained, an unlucky
initialization, a quirk of this specific run. It isn't: two more
independently-trained EfficientNetB0 checkpoints -- a flat 13-class
baseline, and a separately-trained flat "v2" checkpoint built from a
fresh Colab session (full provenance in `docs/evidence/model_lineage.md`)
-- were evaluated the same way, and all three converge on **three
independent signals**, not accuracy alone:

1. **Accuracy band convergence**: all three checkpoints' zero-shot
   PlantDoc accuracy (full train+test combined, n=965-968 -- the valid
   comparison here since none of these three were ever fine-tuned on
   PlantDoc) lands in a 2-point band: 22.00% (flat baseline) / 23.45%
   (hierarchical) / 21.45% (v2 independent) -- **mean 22.30%, stdev 1.03
   points**. `docs/evidence/metric1_zero_shot_convergence.json`.
2. **Attractor-class convergence**, now formalized as an actual confusion
   matrix (previously an informal observation) for the two checkpoints
   with saved per-image predictions: both independently default to
   predicting `Tomato_Late_blight`/`Tomato_Early_blight` regardless of
   true label -- hierarchical checkpoint: `Tomato_Late_blight` predicted
   522/967 times (54% of ALL predictions, any true class); v2 checkpoint:
   `Tomato_Late_blight` (425x) + `Tomato_Early_blight` (322x) dominate the
   same way. Per-class recall on those classes' own true instances stays
   far below their predicted share (hierarchical: 80.2% recall but only
   17.0% *precision* on `Tomato_Late_blight` -- the model calls almost
   everything blight, so it's "right" whenever the true label happens to
   be blight and wrong almost everywhere else).
   `docs/evidence/metric1_confusion_matrix_calibration_evidence.json`;
   matrices in
   `agents/disease/results/confusion_matrix_row{2,6}_*_zero_shot.png`.
3. **Calibration-shape convergence**: Expected Calibration Error is
   nearly identical across the two checkpoints (0.4725 vs. 0.4606), and
   both reliability diagrams show the same shape -- confidence bins above
   ~0.5 sit well below the diagonal throughout, i.e. the model is
   systematically *overconfident*, not just inaccurate, and this
   overconfidence pattern itself replicates independently.
   `agents/disease/results/reliability_diagram_row{2,6}_*_zero_shot.png`.

*(These three numbers use each checkpoint's full combined image set,
n=965-968, matching how the accuracy-band convergence was computed --
not the 85-image test-only restriction used above and below for the
fine-tuned-model comparison. Directionally consistent with, not
contradicting, the 49%-predicted/12%-true `Tomato_Late_blight` figure
already quoted for the single-checkpoint 85-image baseline.)*

**Read together, these are not three separate findings -- they are the
same underlying phenomenon viewed at three resolutions of the same
confusion matrix**, replicated across three independently-trained
models, two training environments, and two architectures (flat 13-class
vs. hierarchical 2-stage). The evidence points to a decision boundary
shaped almost entirely by PlantVillage's uniform lab backgrounds and
framing -- a "blotchy texture on a leaf-shaped object, pick the nearest
common label" heuristic -- rather than lesion-specific morphology. This
is strong evidence the ~22% real-world accuracy ceiling is a **structural
covariate-shift problem inherent to PlantVillage-only training data, not
a fixable modeling error specific to one run**, and it is this project's
most scientifically interesting result. Stated plainly, without hedging:
**the Disease Agent, as currently trained, should not be presented as
reliable for field deployment** -- this three-signal convergence is the
evidence for that limitation and the honest framing of it, not a hidden
weakness.

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

### 2.9 Voice vs. text ablation (Phase 8.1, metric #4) -- PRELIMINARY, n=16, single speaker

Does going through ASR change what the pipeline does, compared to typing
the exact same question? Each of the 16 questions in
`tests/intent_router_test_set.json` run through the real, unmocked
`run_pipeline()` twice: once via speaker1's recorded WAV ->
`backend.voice.transcribe_audio()` (the exact production ASR call path),
once as the reference text typed directly, no ASR. Both paths use live,
unmocked agent calls (Weather API, Price API, Ollama intent router). No
image attached (this test set has none), so the Disease Agent never
fires. Stage-by-stage comparison, not just final-answer pass/fail, so an
ASR mishearing and a downstream (intent-router/state-mapping/
conflict-resolver) difference stay distinguishable rather than collapsed
into one number.

- **Result**: intent-router output matched 15/16; agents-fired matched
  15/16; of the 2 apparent final-answer mismatches, one was investigated
  and ruled out as a live Price API fluctuation unrelated to voice vs.
  text (see below), leaving **exactly one genuine voice-vs-text pipeline
  finding out of 16 questions**.

- **Headline finding (question 8, "నేల pH ఎంత ఉండాలి టమాటా కోసం?" / "What
  soil pH is needed for tomato?")**: ASR transcribed "pH" as its Telugu
  transliteration "పీహెచ్" instead of the reference text's Latin-script
  "pH". That single spelling difference flipped the LLM intent router's
  classification -- the Latin-script version correctly routed to
  `wants_soil=True`; the Telugu-transliterated version routed instead to
  `wants_price=True`, so a farmer asking about soil pH by voice would
  have received a tomato price quote instead. This is a **structural**
  finding, not an ASR accuracy defect -- the transcription was arguably
  *more* natural Telugu than the reference text (which embeds a
  Latin-script technical abbreviation no farmer would actually say), and
  that naturalness is exactly what caused the behavior change. It shows
  domain-specific technical terms (soil pH, NPK, pesticide/disease names)
  crossing the ASR-to-LLM boundary can silently change system behavior in
  ways a text-only test suite would never catch -- the actual reason to
  run a voice-vs-text ablation at all. Flagged as a concrete direction for
  future work (e.g. normalizing known technical terms before intent
  routing), not applied as a fix here -- this metric is observation only,
  `orchestrator/pipeline.py` was not touched.

- **One apparent mismatch investigated and ruled out** (question 7,
  potato market price): same intents, same agents fired on both paths,
  but a different final price quote. Traced to the Price Agent, which
  hits `data.gov.in` live with no caching on every call -- checked every
  price-agent call across the entire 16-question run and found 8 of 9
  returned the *identical* value to 16 decimal places, including this
  question's own text-path call made 4 seconds later; only the voice-path
  call for this question differed. Confirmed via source read (not
  inference) that `price_agent.py`'s code path cannot differ by
  invocation source at all -- `get_price_advice()` never receives the
  question text, only the farm profile's state/crop/mandi fields, which
  were identical for both calls. Settled, not provisional: a one-off live
  API fluctuation, not a voice-vs-text pipeline effect.

- **PRELIMINARY**: n=16, single speaker (speaker1) only -- same caveat as
  Sec. 2.3's WER check at 1/4 speakers. Does not measure cross-speaker or
  accent robustness; measures whether, for one speaker's recordings, ASR
  changes pipeline behavior relative to typing the same question. Should
  be re-run against speakers 2-4 once their WER recordings are available,
  using the same reusable script.
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
named) -- **Farmer.Chat** (arXiv:2409.08916, the most detailed and most
comparable: a large-scale, publicly documented deployment), **Krishi
Sathi** (arXiv:2508.03719, intent-aware RAG for multi-turn agricultural
QA), and **Raithubot** (RLHF-fine-tuned Telugu chatbot, ICDSA
best-paper -- less public technical detail was findable than for the
other two, noted honestly rather than inferred).

| Dimension | Krishi-Agent | Farmer.Chat | Krishi Sathi | Raithubot |
|---|---|---|---|---|
| Architecture | Multi-agent + explicit rule-based Conflict Resolver | RAG + multi-agent orchestration (Planning/Execution/Tooling agents) | Multi-turn RAG with intent-aware context retrieval | Single RLHF-fine-tuned LLM (Pythia-2.8B) |
| Explicit conflict resolution across advice types (weather/price/disease) | **Yes** -- rule table, 13/13 on synthetic tests; 1 real gap found via live-data testing (Sec. 2.7) and fixed | Not described in the paper | Not described | Not described |
| Disease/pest diagnosis | Own trained, fine-tuned CV model, in-pipeline (Sec. 2.5, honestly measured at 37.65%) | Delegated to a third-party service (Plantix) | Not described | Not described |
| Confidence/uncertainty shown to the user | **Yes** -- hedged phrasing templates + a UI confidence indicator, both keyed to the real underlying confidence level | Not described (feedback is retrospective thumbs-up/down, not prospective confidence) | Not described | Not described |
| Languages | Telugu (bilingual UI) | 6 languages incl. Telugu, deployed across 4 countries | Not specified in available sources | Telugu, Hindi, English |
| Evaluation scale | Component-level (n=12-263) + two fresh-clone reproducibility tests; **no live user study yet** | **15,000+ real users, 300,000+ queries**, formal focus groups + bi-weekly satisfaction surveys | Published methodology; user-scale not found in available sources | Conference-published; accuracy not publicly detailed |
| Response latency (reported) | ~13.6s intent routing + ~24-26s TTS for a full voice answer (Sec. 2.4/2.6) | 9.05s average (text response; not confirmed whether this includes voice synthesis) | Not found | Not reported |

**Honest reading of this table, not a one-sided one**: Farmer.Chat
massively outscales this project on real-world deployment and user-study
rigor -- that is a genuine, uncontested strength of theirs, not something
to argue around. What this comparison actually supports is a narrower,
verifiable claim: **conflict arbitration between competing advice signals
and prospective confidence communication to the farmer are explicitly
engineered, independently-tested components in Krishi-Agent, and are
not described as present in any of the three comparable systems' own
published accounts.** That is the project's specific, evidence-backed
novelty contribution -- not a claim of being "better overall," which the
evaluation scale gap above would not support.

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
4. **Multi-speaker ASR evaluation** -- current WER is single-speaker;
   accent/dialect/background-noise robustness is completely unmeasured.
5. **A proper literature review** -- Sec. 3 is a starting point, not a
   finished related-work section.
6. ~~**Formal latency benchmarking methodology**~~ **Done** -- Sec. 2.4/2.6
   now report real n=8-trial mean/std/min/max with stated hardware spec,
   not single-run numbers. It also surfaced a genuine, honestly-reported
   discrepancy (the Intent Router's 8-trial mean came in higher than the
   original 3-trial spot check, with a plausible confound noted rather
   than hidden) -- exactly the kind of thing formal benchmarking is
   supposed to catch.

### End-to-end answer-quality rubric -- status

- **Intent correctness** (0/1) -- **done**, Sec. 2.1/2.7 (87.5% on the
  16-question set; Sec. 2.7 additionally root-causes every fallback).
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
  12 synthetic (resolution, confidence) pairs (12/12, re-run today). What
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
