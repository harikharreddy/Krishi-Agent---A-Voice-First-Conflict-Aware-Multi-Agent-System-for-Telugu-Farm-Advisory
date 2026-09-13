# Krishi-Agent: Outcome Matrix, Realistic Constraints, and Engineering Standards

Drafted for direct insertion into the capstone report template. Every claim
below cites a real file, decision, or measured number from this repository
-- nothing is generic template language standing in for actual project
content.

## Outcome Matrix

### (a) Ability to apply knowledge of mathematics, science, and engineering

**What was applied, concretely:**
- **Loss function**: class-weighted cross-entropy for the Disease Agent's
  fine-tuning (`agents/disease/finetune_plantdoc.py`'s `class_weights_for()`)
  -- weights computed as `total / (num_classes * class_count)`, standard
  inverse-frequency weighting to counteract class imbalance (Tomato
  Septoria: 140 training images vs. Tomato Mosaic Virus: 44).
- **Optimization**: Adam with discriminative learning rates -- classifier
  head at 3e-4, unfrozen backbone block at 3e-5 -- a standard transfer-
  learning schedule (faster adaptation for new layers, gentler updates
  for pretrained features) applied deliberately, not a default left
  untouched.
- **Probability/statistics**: joint confidence computed as
  `stage1_confidence * stage2_confidence` (independent-event probability
  multiplication across the two-stage classifier); a Wilson score 95%
  confidence interval computed for the Potato_healthy evaluation (n=263,
  93.54% accuracy, CI [89.89%, 95.93%]) rather than reporting a bare
  percentage; precision/recall/F1 derived from the confusion matrix
  (`agents/disease/compute_full_metrics.py`).
- **Signal processing**: 16kHz mono PCM audio capture via Web Audio API
  (`static/js/app.js`'s `WavRecorder`), matching the exact sample rate
  the ASR model requires -- avoiding a resampling step by getting the
  math right at capture time.
- **Matrix/tensor transforms**: standard ImageNet normalization
  (per-channel mean/std) and `RandomResizedCrop` augmentation
  (`finetune_plantdoc.py`'s `train_transform`) for the vision model input
  pipeline.

### (b) Ability to design and conduct experiments, and analyze/interpret data

- Every Disease Agent fine-tuning attempt was run as a controlled
  before/after experiment: PlantDoc's `train` split used only for
  fine-tuning, `test` split held out and never touched, enabling an honest
  before/after comparison (28.2% -> 37.65% end-to-end accuracy).
- Two experiments (`Tomato__Target_Spot` fine-tuning attempts) were
  analyzed and **rejected/qualified based on the data**, not assumed to
  work: attempt 1 (5 training images) scored 0/2 on a held-out check and
  was not promoted; attempt 2 (20 images) scored 1/5 and was promoted only
  after quantifying its cost to other classes (-1.2 percentage points).
- A methodology bug was caught mid-analysis and corrected before
  reporting: an early end-to-end accuracy check matched on disease-name
  text alone and overcounted, since several Telugu disease labels are
  reused across crops (`Tomato_Early_blight` and `Potato___Early_blight`
  both render as "ఎర్లీ బ్లైట్"). Re-derived the check to require both
  crop and disease name before accepting the result.

### (c) Ability to design a system, component, or process under realistic constraints

The dominant realistic constraint on this project was **8GB of RAM on the
target machine, no dedicated GPU** -- this single constraint drove several
concrete design decisions:
- The Disease Agent uses 3 small EfficientNetB0-based models (12.04M
  parameters total, 46.79MB combined checkpoint size) rather than a larger
  architecture, keeping inference feasible on CPU/Apple MPS at
  4.16 images/sec with no batching.
- Fine-tuning used a frozen backbone (only the last block + classifier
  head trainable) specifically to keep training time and memory bounded
  on consumer hardware, not because it was the theoretically optimal
  choice.
- The Intent Router (Ollama, 7B parameter LLM) and TTS model were found to
  collide for RAM, producing 191+ second stalls -- fixed by force-unloading
  the LLM after each use, a direct engineering response to the hardware
  ceiling, explicitly documented as a tradeoff (adds 5-14s of reload cost
  per question) rather than a free fix.
- The system is modular by design: `agents/` (Disease/Weather/Price,
  independently swappable), `orchestrator/` (Intent Router, Conflict
  Resolver, Phrasing Templates -- each independently unit-tested, 68/68
  passing), `backend/` (FastAPI service layer), `static/` (frontend) --
  allowing each component to be modified without touching the others, and
  verified by a fresh git-clone test that the whole system still installs
  and runs end-to-end from nothing.

### (d) Ability to function on multidisciplinary teams

Four-person team spanning: ML/data work (dataset sourcing and licensing
verification for disease fine-tuning data -- UF/IFAS, Mendeley CC BY 4.0
sources with documented provenance), backend/systems engineering (FastAPI
migration, latency debugging), frontend/UX (bilingual Telugu/English UI,
voice recording, confidence-honesty interface design), and domain/language
validation (Telugu phrasing templates and disease-name translations
reviewed by fluent-Telugu team members before deployment, per
`orchestrator/phrasing_templates.py` and `agents/disease/disease_agent.py`
docstrings).

### (e) Ability to identify, formulate, and solve engineering problems

This project's strongest evidence category -- a partial list of problems
independently diagnosed and solved, each with a documented root cause,
not just a patched symptom:

| Problem | Root cause found | Fix |
|---|---|---|
| TTS taking 200-280s | Streamlit's threading model interfering with generation | Migrated to FastAPI; verified same `synthesize_speech()` call runs in ~24-26s |
| Still slow after migration | Ollama LLM + TTS colliding for RAM on 8GB machine | Force-unload LLM after each use (accepted latency tradeoff, documented) |
| Disease model 99.7% in testing, unreliable in practice | PlantVillage (studio photos) to PlantDoc (real-world photos) domain shift, plus a systematic Late_Blight over-prediction bias | Fine-tuned on real PlantDoc photos; bias measured and resolved (49% -> 11% prediction rate, true rate 12%) |
| Two disease classes never predicted correctly | Zero real-world training images existed for them anywhere | Sourced real photos from 2 external CC-licensed datasets; one fix worked cleanly (93.5%), one partially (20%, reported honestly) |
| A fresh clone of the repo couldn't run at all | Trained model checkpoints were never committed to git, in the project's entire history | Migrated to Git LFS; verified with two independent from-scratch clone tests |
| `requirements-voice.txt` failed to install from scratch | A genuine dependency conflict (Streamlit needs `protobuf>=5.26`, the TTS stack needs `protobuf<5.0.0`) | Removed the now-unused legacy Streamlit code entirely rather than fighting the conflict |
| A specific farmer question always got a generic fallback answer | Conflict Resolver's rule table had an entry for `(None, rain_risk, hold)` but not `(None, rain_risk, sell_now)` -- an asymmetric gap | Precisely diagnosed and documented in the evaluation report (fix is a one-line addition, not yet applied) |

### (g) Ability to communicate effectively

- `docs/evaluation_and_validation.md` and `docs/evaluation_report.html` --
  a full methodology-and-results writeup for every measured component,
  written for an audience that did not build the system.
- Every non-obvious code decision carries an in-repo explanation of *why*,
  not just *what* (e.g. `backend/main.py`'s `_unload_intent_router_model()`
  docstring documents the measured cost/benefit tradeoff inline, not in a
  separate document that can drift out of sync with the code).
- A system architecture diagram and a formal related-work comparison
  against 3 published comparable systems (Farmer.Chat, Raithubot, Krishi
  Sathi) are included in the evaluation report, positioning this project's
  specific contribution rather than asserting novelty without evidence.

### (k) Ability to use techniques, skills, and modern engineering tools

PyTorch, Hugging Face Transformers, scikit-learn (precision/recall/F1),
FastAPI, Ollama (local LLM serving), Git + Git LFS (large-file model
versioning), pytest, `jiwer` (WER scoring), Apple MPS GPU acceleration,
matplotlib (training-curve visualization), and a RESTful API architecture
-- all open-source, all chosen and configured deliberately (e.g. Git LFS
specifically adopted this session after discovering plain git couldn't
handle the model checkpoints).

## Realistic Constraints

**Economic constraints**: every tool in the stack is free or open-source
-- AI4Bharat's ASR/TTS models, PlantVillage/PlantDoc (open disease
datasets), Ollama (local, no per-token API cost), OpenWeatherMap and
data.gov.in (free-tier APIs). The two external disease-photo datasets
used for fine-tuning were both CC BY 4.0 licensed and used with
attribution, at zero cost.

**Computational & hardware constraints**: the entire project ran on a
single 8GB-RAM machine with no dedicated GPU. This is not incidental --
it's the specific constraint that produced the two central engineering
findings of the project (the Streamlit/FastAPI latency difference and the
Ollama/TTS memory collision), and it shaped the Disease Agent's
architecture choice (12M total parameters, 46.79MB, 4.16 images/sec on
CPU/MPS -- deliberately small enough to run without a GPU).

**Sustainability & ethical constraints**: model sizes were kept small by
design (no large from-scratch training runs, fine-tuning only, frozen
backbones), keeping compute and energy use low. Dataset licensing was
verified and documented for every external data source used, with
attribution preserved in code comments and evaluation documentation.
Real-world accuracy limitations are disclosed honestly rather than
overstated (e.g. the Disease Agent's per-class accuracy table is
published in full, including classes where it performs poorly) -- a
farm-advisory tool that overclaims diagnostic confidence has a real
downstream harm risk (a farmer acting on a wrong "certain" diagnosis),
which is the direct motivation behind the confidence-honesty UI design.

## Engineering Standards

- **IEEE 830 / IEEE 12207 (requirements & software lifecycle)**: phased
  development is documented end-to-end in `docs/claude_phase*.md`
  (Phase 0 through Phase 8), each phase recording what was built, what was
  tested, and what limitation was carried forward -- a running
  requirements-and-progress record rather than a single retrospective
  writeup.
- **ISO/IEC 25010 (software quality model)**: functionality verified via
  68/68 passing deterministic logic tests across every orchestration
  component; performance efficiency measured directly (latency,
  throughput, memory footprint, Sec. "Precision/recall/F1..." above);
  usability addressed via the bilingual UI and confidence-honesty design;
  reliability verified via 16/16 pipeline survival on a real question set
  and two independent fresh-clone reproducibility tests; maintainability
  supported by a 4-tier checkpoint lineage (`*_plantvillage_only.pt` ->
  `*_plantdoc_no_target_spot.pt` -> `*_pre_potato_healthy.pt` ->
  `*_best.pt`) that keeps every model version auditable and reversible.
- **Reproducibility standards**: fixed random seed (`SEED = 42` in
  `finetune_plantdoc.py`) for every fine-tuning run; PlantDoc's own
  official train/test split used throughout, never mixed or re-shuffled;
  a repeat fine-tuning run reproduced identical reported results (37.65%
  end-to-end, 1/5 Target_Spot, 14/15 Potato_healthy), and the one place it
  didn't match exactly (a 1-image difference in a 30-image subset check)
  was traced to Apple MPS's documented floating-point nondeterminism and
  reported rather than hidden; the entire system was verified installable
  and runnable from a completely fresh `git clone`, twice, on two separate
  occasions in this project's history.
