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

### 2.1 Intent Router (LLM-based question routing)

- **Dataset**: `tests/intent_router_test_set.json` -- 16 hand-written Telugu
  farmer questions covering single-intent, multi-intent, and no-intent
  (greeting) cases, with ground-truth `{wants_disease, wants_weather,
  wants_price, wants_soil}` labels.
- **Method**: `tests/test_intent_router.py`, live calls to `route_intent()`
  against the running Ollama model. Exact-match scoring (all 4 boolean
  fields must match).
- **Result (re-run today)**: **14/16 correct, 87.5%**. Matches the
  originally-measured Phase 2.2 number exactly, confirming reproducibility.
- **Failure modes** (both n=1 cases, illustrative not statistical): #12
  under-routed a broad "give me full advice" request (missed weather/price
  intents); #14 over-routed a pesticide-availability question as a disease
  question.
- **Ablation -- model size**: qwen2.5:3b-instruct was tested on the same set
  in Phase 2.2 and scored **8/16, 50%** -- the 7B model was kept specifically
  because of this gap, at a documented latency cost (Sec. 2.6).
- **Known limitation**: n=16 is a hand-curated smoke-test set, not a
  statistically powered benchmark. A real paper submission needs a larger,
  more diverse question set (see Sec. 5).

### 2.2 Conflict Resolver

- **Dataset**: 12 synthetic scenarios (`tests/test_conflict_resolver.py`)
  covering single-agent, dual-agent, and triple-agent conflict/agreement
  cases at varying confidence levels.
- **Result (re-run today)**: **12/12 correct, 100%**.
- **Limitation**: rule-based logic over synthetic inputs, not live agent
  output -- validates the *decision logic*, not real-world scenario coverage.

### 2.3 ASR (speech-to-text)

- **Model**: `ai4bharat/indic-conformer-600m-multilingual`.
- **Dataset**: 16 real microphone recordings (`tests/audio/wer_eval/speaker1/`),
  same 16-question set as Sec. 2.1, read aloud by one Telugu speaker.
- **Method**: `tests/check_all_wer.py`, word error rate via `jiwer`, text
  normalized (punctuation/whitespace stripped) before scoring.
- **Result**: **mean WER 10.96%** across 16 utterances (full per-question
  table in `tests/audio/wer_eval/speaker1_results.json`). Most individual
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

### 2.5 Disease Agent -- domain-shift and fine-tuning study

This is the most extensively evaluated component; full detail in
`agents/disease/results/*.json` and `agents/disease/disease_agent.py`'s
docstring.

**Baseline (PlantVillage-trained, no real-world fine-tuning)**:
| Eval set | Domain | n | Accuracy |
|---|---|---|---|
| PlantVillage validation | studio/lab photos (same distribution as training) | 3,628 | 99.7% |
| PlantDoc (zero-shot) | real-world field photos, same 85-image test split used below | 85 | 28.2% |

The 71.5-point drop is a textbook train/test domain-shift result, and came
with a specific, diagnosable failure mode: the model over-predicted
`Tomato_Late_blight` for **49% of test images** (true rate: 12%) -- an
attractor-class bias, not just noise.

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
- **Measured cost** (3 consecutive real `route_intent()` calls, with the
  fix active): **5.6s / 12.1s / 10.7s** per call, vs. **~3.6s** for a warm
  (non-unloaded) call -- roughly a 2-3x per-question latency cost, traded
  for eliminating the ~191s collision entirely.
- This is reported as an explicit, accepted tradeoff, not a fully solved
  problem -- see Sec. 4 (Limitations).

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

**Telugu/regional-language agricultural voice assistants**: several
directly comparable systems exist and should be discussed as related work
-- "Raithubot" (RLHF-fine-tuned Telugu farmer chatbot), "Farmer.Chat"
(multilingual incl. Telugu, multimodal agricultural advisory at scale),
and RAG-based regional-language agri-LLM systems. Krishi-Agent's specific
differentiators worth foregrounding against these: (a) explicit
multi-agent conflict resolution rather than a single LLM call, (b)
deliberately hand-written/reviewed phrasing templates over free LLM
generation (a locked decision after qwen2.5:7b-instruct produced garbled
Telugu in testing), and (c) an explicit confidence-honesty design (visible
uncertainty in both text and UI) rather than a single confident-sounding
answer.

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

1. **End-to-end answer-quality evaluation.** Run the full pipeline
   (question -> spoken answer) across a larger question set than 16, score
   each final answer against a rubric (correctness, appropriate hedging,
   actionability). This is buildable without external people -- see
   proposed rubric below.
2. **Statistical power.** Most component evals above are n<20. Where more
   real-world data can be sourced (more PlantDoc-style photos, more
   speakers for ASR, a larger hand-written question set for Intent
   Router/Conflict Resolver), doing so would materially strengthen every
   claim in this document.
3. **A small human/user study.** Even 5-10 Telugu-speaking test users doing
   a handful of realistic tasks, plus a short usability survey (e.g.
   System Usability Scale), would be the single highest-value addition for
   reviewer credibility on a "voice-first for farmers" claim. Needs real
   people -- can't be simulated.
4. **Multi-speaker ASR evaluation** -- current WER is single-speaker;
   accent/dialect/background-noise robustness is completely unmeasured.
5. **A proper literature review** -- Sec. 3 is a starting point, not a
   finished related-work section.
6. **Formal latency benchmarking methodology** -- current latency numbers
   are single-run or few-run measurements without reported variance; a
   paper wants n-trial means with standard deviation and stated hardware
   spec.

### Proposed end-to-end answer-quality rubric (for item 1)

For each of N test questions, score the full pipeline's final spoken
answer on:
- **Intent correctness** (0/1): did it engage the right agent(s)? (already
  measurable via Sec. 2.1's ground truth)
- **Factual correctness** (0/1/2): wrong / partially right / right, judged
  against the known agent outputs for that question's farm profile
- **Confidence calibration** (0/1): does the hedging language match the
  actual underlying confidence level (e.g., a Low-confidence Disease Agent
  result shouldn't produce an unhedged, certain-sounding sentence)?
- **Actionability** (0/1): does a farmer come away knowing what to actually
  do?

This can be scored programmatically for intent correctness and confidence
calibration (both are derivable from the pipeline's trace output without a
human), and needs a rubric-following human (not necessarily a domain
expert) for factual correctness and actionability.
