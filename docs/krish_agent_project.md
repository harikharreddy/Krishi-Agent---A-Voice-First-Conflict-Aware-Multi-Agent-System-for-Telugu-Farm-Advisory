# KRISHI-AGENT — Complete Project Guide (Final)

A single, self-contained reference for the whole project: what it does, the final architecture, the tools and models, what to build at each step, and everything else your team needs (roles, evaluation, risks, ethics, publication, cost). This is the only document you need to upload to your Claude Project. Prompts aren't pre-written here — see Section 8 for how to get the right prompt for whatever you're starting next.

---

## 1. Project Summary

**Title (official, approved by guide):** Krishi-Agent: Hierarchical Crop Disease Classification and Conflict-Aware Multi-Agent System for Telugu Farm Advisory

**What it does:** A farmer speaks a question in Telugu, optionally attaching a photo of a crop leaf. The system checks whichever of crop disease, weather, and market price are relevant, merges the results through a conflict-aware Orchestrator, and speaks back one honest, confidence-calibrated answer in Telugu — resolving contradicting advice explicitly instead of ignoring it.

- **Team:** 4 people, working together each session
- **Language:** Telugu
- **Connectivity:** Normal internet required (not offline) — a deliberate scoping choice to reduce risk, not a novelty claim
- **Total cost:** Zero — every dataset, model, and tool is free and open-source

---

## 2. Locked Scope

**Core Crops and Classes (13 total, 2 crops):**

| Crop | Classes |
|---|---|
| Tomato (10) | Bacterial spot, Early blight, Late blight, Leaf mold, Septoria leaf spot, Spider mites, Target spot, Yellow leaf curl virus, Mosaic virus, Healthy |
| Potato (3) | Early blight, Late blight, Healthy |

**Stretch Crop (Phase 7, only after Phase 4 is solid):**

| Crop | Classes |
|---|---|
| Pepper, Bell (2) | Bacterial spot, Healthy |

Pepper is low-effort to add since it's already in PlantVillage alongside tomato and potato — no new dataset to source. The one thing to verify before committing to it: whether PlantDoc has enough real-world pepper images to give a meaningful accuracy number (Section 11 flags this as a risk to check early in Phase 7, not discover late).

**Core scope:** Disease Agent (2 crops), Weather Agent, Price Agent, Orchestrator, voice in/out.
**Stretch scope (only after the core pipeline works):** Soil Agent, Pepper as a 3rd crop.
**Deliberately excluded (future work only):** true offline/edge deployment.

---

## 3. Final Architecture

```
 Farmer (voice, Telugu; optional leaf photo)
        │
        ▼
 [1] Speech-to-Text  ──────────────────────  AI4Bharat IndicConformer
        │
        ▼
 [2] Intent Router (small LLM)
      parses: which agents are needed (disease / weather / price / soil)
      pulls the question type only — NOT location or mandi from speech
        │
        ▼
 [3] Farm Profile lookup (structured, not voice-extracted)
      district / mandi / crop, set once via dropdown or saved profile
        │
   ┌────┼──────────────┬─────────────────┐
   ▼    ▼               ▼                 ▼
[4a] Disease Agent  [4b] Weather Agent [4c] Price Agent   [4d] Soil Agent (stretch)
 hierarchical CV     API + rules        Agmarknet + rules  Soil Health Card lookup + rules
 (only if photo       │                  │                  │
  attached)            │                  │                  │
 emits: label,         emits: forecast,   emits: trend,      emits: nutrient/pH
 confidence (0-1)      confidence tag     confidence tag     note, confidence tag
   │                  │                  │                  │
   └──────────────────┴──────────────────┴──────────────────┘
                       │  (only the agents Step 2 flagged as relevant run)
                       ▼
        [5] Orchestrator
            a. Deterministic conflict-rule table resolves disagreements
               (e.g. disease says spray, weather says rain → wait-then-spray)
            b. LLM (English-side reasoning over structured fields) drafts
               the merged, confidence-aware message
            c. Telugu phrasing: template library first, LLM fallback with
               guardrails, always reviewed by a fluent Telugu speaker
                       │
                       ▼
        [5.5] Numeral Text Normalization (added Phase 5.2)
               Converts embedded digits (prices, percentages, hour counts)
               into Telugu words before TTS -- raw digits were found to be
               mispronounced/unintelligible when spoken directly.
                       │
                       ▼
        [6] Text-to-Speech ── AI4Bharat TTS (IndicF5 / Indic-Parler-TTS)
                       │
                       ▼
              Farmer hears the answer
```

**Shared agent schema** — every agent (including the stretch Soil Agent) must return output in this exact shape, so the Orchestrator can merge them fairly regardless of whether the agent is ML-based or rule-based:

```
{
  answer: string,
  confidence: "High" | "Medium" | "Low"  (or 0-1 for the Disease Agent's softmax score),
  reason_for_confidence: string,
  source_freshness: string
}
```

**Why this differs from a naive design (the fixes baked in here):**
- The photo has an explicit entry point (Disease Agent only runs if one is attached) — the earlier draft had no place for it.
- The Intent Router has one real job: deciding which agents to call, not decorative routing.
- Location and mandi come from a structured Farm Profile, not from parsing spontaneous Telugu speech — that NLU problem is hard and unnecessary to take on.
- Conflict resolution is a hand-written, testable rule table first; the LLM only phrases the already-made decision. This is what makes "conflict-aware" something you can report a pass rate on, not just a demo that "seems to work."
- Small local LLMs (2-3B class) aren't trusted to reason natively in Telugu — structured reasoning happens over English fields, with a template fallback for the final sentence.

---

## 4. Key Design Improvements (Novelty)

**Core novelty claims** (use these in your report):
1. **Hierarchical disease classification** (crop then disease) instead of flat — higher accuracy, cleaner reasoning, validated against your own flat baseline.
2. **Conflict-aware Orchestrator** — explicitly detects and resolves contradicting agent advice via a testable rule table, which existing systems don't do.
3. **Confidence-aware responses** — shifts to cautious phrasing when confidence is low, using a shared confidence schema across ML-based and rule-based agents.

**Methodology and scoping strengths** (mention for rigor, don't list as novelty — a panel will push back if framed as a contribution):
- Own controlled baseline comparison (flat classifier before hierarchical).
- Normal internet connectivity (not offline) — a risk-reducing scope decision.

**Novelty statement for your report:** Existing systems combine disease detection, weather, and market advisory through multilingual chatbots, but voice interaction is usually left as future work, and no existing system explicitly resolves contradictory agent recommendations or communicates model confidence to the farmer. Krishi-Agent addresses this gap directly.

---

## 5. Tools, Models, and Datasets

| Purpose | Tool / Model | Note |
|---|---|---|
| Telugu ASR | `ai4bharat/indic-conformer-600m-multilingual` (Hugging Face) | Verify it loads first — highest-risk component. Whisper is your fallback. |
| Telugu TTS | AI4Bharat `IndicF5` or `Indic-Parler-TTS` (Hugging Face) | Naming has shifted from the older generic "Indic-TTS" — check current listings. |
| Intent Router / Orchestrator LLM | Smallest current instruct model that runs on your laptop | Don't hard-lock a model name — check what's current (Gemma, Llama, Qwen small variants, or Indic-tuned options like Sarvam AI) when you get there. Run quantized (GGUF via Ollama/llama.cpp) for demo-day latency. |
| Disease training | Google Colab (free GPU) | |
| Disease datasets | PlantVillage (train), PlantDoc (real-world test) | Expect and report a real accuracy drop between them — targets ~95-98% on PlantVillage, ~65-80% on PlantDoc. |
| Weather | OpenWeatherMap free API | |
| Price | Agmarknet (Indian government portal) | Data is messy to extract — budget extra time. |
| Soil (stretch) | Soil Health Card data via data.gov.in / IndiaDataPortal / `google-research-datasets/india-soil-health-card` | Lookup-only, no sensors needed. |
| Demo UI | Streamlit, using native `st.audio_input` for mic capture | Test on the actual demo phone early, not at the end. |
| Version control | GitHub, pushed daily | Don't lose a week's work to a laptop issue. |
| Local editing | VS Code (optional) | If not using Colab exclusively. |
| Text normalization | `num2words` (Python, `lang="te"`) | Added Phase 5.2 — converts digits to Telugu words before TTS synthesis; raw digits (prices, percentages, hour counts) were found to be unintelligible when spoken directly. |

**Environment note (added Phase 5):** ASR/TTS dependencies (`transformers==4.46.1`, `parler-tts`, `tokenizers`) require Python ≤3.12 — they failed to build on Python 3.14 (used by the main project `.venv`). A separate local environment, `.venv-voice` (Python 3.12), was created specifically for voice work. Activate it with `source .venv-voice/bin/activate` for any ASR/TTS/voice-pipeline work; the rest of the project continues to use the original `.venv`.

**Claude features to use throughout:** Claude Projects (this document as Project Knowledge), Claude Code for actually writing/running/debugging code, file upload for error logs and screenshots, web search for anything that might be outdated (model names, API tiers), and Artifacts for architecture diagrams and result charts.

---

## 6. Build Order (dependency-based, no calendar)

Work through these phases in order. Each is gated on the one before it — don't start Phase N+1 until Phase N's "done" condition is met.

- **Phase 0** — Prove the risky parts work (ASR, TTS, mic capture on the demo phone).
- **Literature Survey — complete before Phase 1.** Runs alongside Phase 0, but should be finished (or at least a solid first pass through your candidate papers) before you start Phase 1 — your methodology decisions there, especially the Disease Agent's architecture, should be informed by what you've read, not decided blind.
- **Phase 1** — Build each data agent standalone (Weather, Price, Disease baseline, Disease hierarchical, PlantDoc validation).
- **Phase 2** — Structured input layer (Farm Profile, Intent Router).
- **Phase 3** — Orchestrator (conflict-rule table, then LLM phrasing layer).
- **Phase 4** — Text-only pipeline integration. **This is the checkpoint that matters most — protect it above everything else.**
- **Phase 5** — Voice layer (STT/TTS wired onto the proven text pipeline).
- **Phase 6** — Polish (confidence tone, mobile testing, latency).
- **Phase 7** — Stretch scope: Soil Agent, and Pepper as a 3rd crop (both only after Phase 4 is solid).
  - **Update (Sep 2026): Phase 7 evaluated and deliberately skipped.** See Section 7's Phase 7 entry for the reasoning.
- **Phase 8** — Evaluation and report.

If you fall behind: cut Phase 6 polish or delay Phase 7 — never skip validating Phase 4 before moving to Phase 5. A working, honest, text-only pipeline is worth more than a half-working voice demo.

---

## 7. Step-by-Step: What to Do in Each Phase

Work through these in order — each step lists what needs to get done, not exact wording to paste. For how to actually turn each step into a prompt (and how to move from one phase's chat to the next), see Section 8.

### Phase 0 — Prove the risky parts work first

- **0.1 — Repo and environment.** Create a GitHub repo with a sensible folder structure for a multi-agent system (separate folders per agent, the orchestrator, the UI). Set up a shared Google Colab notebook with GPU access. Get all 4 of you access to both.
- **0.2 — Telugu speech-to-text.** Load `ai4bharat/indic-conformer-600m-multilingual` from Hugging Face in Colab and confirm it transcribes a short Telugu audio clip. This is the highest-risk component in the project — if it's difficult to get running within an hour or two, fall back to Whisper rather than losing the day to it.
- **0.3 — Telugu text-to-speech.** Find and load AI4Bharat's current Telugu TTS model (verify whether IndicF5 or Indic-Parler-TTS is the current right choice, don't assume). Confirm it turns a short Telugu text string into playable audio.
- **0.4 — Mic capture on the actual demo phone.** Test Streamlit's native `st.audio_input` widget, saving a recording to a file, on the specific phone and browser you'll demo on. Mobile mic permissions and audio formats are inconsistent — find out now, not at Phase 6.

**Done when:** you can record a Telugu sentence on the demo phone, transcribe it, and hear a synthesized reply.

### Phase 1 — Build each data agent standalone

- **1.1 — Weather Agent.** Call the OpenWeatherMap free API for a given location and apply simple advice rules (e.g. flag rain expected in the next 24-48 hours). Output must match the shared schema: `{answer, confidence: High/Medium/Low, reason_for_confidence, source_freshness}`. Standalone module, no UI, a few test cases.
- **1.2 — Price Agent.** Fetch trend data from Agmarknet for a given commodity/mandi and apply simple sell/hold logic, same shared schema. Inspect a real sample response before writing the parser — Agmarknet's data is messy.
- **1.3 — Disease Agent: flat baseline.** Train a flat (non-hierarchical) CNN classifier on PlantVillage covering all 13 classes across the 2 crops, using a pretrained backbone for a free-GPU budget. Log accuracy — this is your controlled baseline.
- **1.4 — Disease Agent: hierarchical.** Stage 1 classifies crop (Tomato/Potato), Stage 2 classifies disease within that crop. Train on PlantVillage, output softmax confidence mapped into the shared schema.
- **1.5 — Validate on PlantDoc.** Evaluate both classifiers on PlantDoc (real-world images), not just PlantVillage. Produce a confusion matrix for each and report the accuracy drop honestly — it's expected.

**Done when:** each agent, called standalone with test inputs, returns sane schema-consistent output.

### Phase 2 — Structured input layer

- **2.1 — Farm Profile.** A simple form/dropdown (not voice) capturing district, nearest mandi, and crop, saved for the session.
- **2.2 — Intent Router.** Given a Telugu question, output which of `wants_disease`, `wants_weather`, `wants_price`, `wants_soil` are relevant — no entity extraction. Generate 15-20 realistic sample Telugu farmer questions (single- and multi-intent) as a test set first, then build and test against it.

**Done when:** a typed question reliably produces correct agent flags, and the Farm Profile reliably supplies context.

### Phase 3 — Orchestrator

- **3.1 — Conflict-rule table (deterministic, no LLM yet).** Generate at least 10-15 realistic conflict scenarios between agents (e.g. disease says spray vs weather says rain coming). Write these as explicit if/then rules in code — not an LLM prompt — producing one resolved decision. Keep the rules and the test scenarios as separate, reusable code; this becomes your conflict-handling test suite.
- **3.2 — LLM phrasing layer.** Given the resolved decision and confidence tags, use your chosen small LLM to produce one fluent, appropriately-hedged Telugu sentence — reasoning over the structured fields in English internally if that's more reliable. Build a template-based fallback for malformed output.

**Done when:** feeding the Orchestrator fabricated conflicting outputs produces a sensible, correctly-hedged text answer.

### Phase 4 — Text-only pipeline integration (protect this above everything)

- **4.1 — Wire it together.** Connect the Intent Router, Farm Profile, the three data agents, and the Orchestrator into one pipeline: typed question in, one combined text answer out. Add logging at each stage.
- **4.2 — End-to-end testing.** Test with the Phase 2 sample questions and Phase 3 conflict scenarios, plus edge cases (no photo attached, missing farm profile, agent API failure).

**Done when:** typed question → one correct combined text answer, reliably, across your test set.

### Phase 5 — Voice layer ✅ COMPLETE

- **5.1 — Wire STT/TTS onto the proven pipeline.** ✅ Done. Integrated the Phase 0 ASR (`indic-conformer-600m-multilingual`) and TTS (`indic-parler-tts`) models onto the front and back of the Phase 4 text pipeline (`orchestrator/pipeline.py`), without touching its logic. Verified on two real cases: an unresolved-fallback question and a genuine, data-backed weather answer. Required a separate `.venv-voice` (Python 3.12) environment — see Section 5's environment note.
- **5.2 — Validate with a fluent Telugu speaker.** ✅ Done. Team (fluent Telugu speakers) reviewed all 10 Phrasing Template sentences (Phase 3.2) as spoken audio — clear and natural, no issues (these templates contain no numeric content).
  - **Real finding:** the TTS model mispronounces multi-digit currency amounts and mid-range percentages when given raw digits (e.g. "2450" and "65%" came out garbled/unintelligible; simple two-digit numbers like "16%" and round hour counts were fine). Confirmed two ways: team listening, and an automated ASR round-trip cross-check (feeding TTS output back through the ASR model and comparing the transcription to the intended text).
  - **Fix built and verified:** added a numeral-to-Telugu-words normalization step (`shared/text_normalization.py`, using `num2words`) before TTS synthesis. Re-tested via the same ASR round-trip method — all numeral test cases now transcribe back word-for-word correctly — and the team confirmed the normalized audio sounds natural.
  - Worth citing in the Phase 8 report as a genuine discovered-and-solved TTS limitation, with before/after evidence (see `tests/audio/numeral_review/` vs `tests/audio/numeral_review_normalized/`).

**Done when:** you can record a Telugu sentence, transcribe it, run it through the pipeline, and hear a synthesized reply. **Met**, and additionally stress-tested against numeral-heavy agent answers (Weather/Price), not just a single demo case.

### Phase 6 — Polish
Refine the Orchestrator's phrasing so confidence differences actually sound different in tone. Test the full app on the demo phone and fix mobile-specific issues. Measure end-to-end latency and quantize the local LLM (Ollama/llama.cpp) if it's slow.

### Phase 7 — Stretch scope (only after Phase 4 is solid)

**Decision (Sep 2026): Phase 7 was evaluated and deliberately not pursued.** Both 7a (Soil Agent) and 7b (Pepper) were confirmed independent of the project's three core novelty claims (Section 4 — hierarchical disease classification, conflict-aware Orchestrator, confidence-aware responses), none of which depend on a third agent or third crop existing. Section 11's own risk table already flagged both as "optional upside, never something that delays the core checkpoint." Time was redirected instead toward Phase 8 evaluation depth (confidence-calibration spot-check, drought-detection gap closure, TTS numeral fix confirmed on live data — see `claude_phase8prep-summary.md`), which directly strengthens the existing three claims rather than adding a fourth, more lightly-tested one under time pressure.

Original scope, for reference (not built):
- **7a — Soil Agent.** Same pattern as Weather/Price — load Soil Health Card data for your target districts, write simple nutrient/pH advice rules, output in the shared schema, and add relevant new conflict scenarios (e.g. soil says irrigate vs weather says rain coming) to the Phase 3 rule table and test set.
- **7b — Pepper as a 3rd crop.** Retrain both the flat baseline and hierarchical Disease Agent with pepper's 2 classes added (15 classes total, 3-way crop stage instead of 2-way). Before investing time here, check PlantDoc actually has enough real-world pepper images for a meaningful accuracy number — if it's too thin, note that limitation honestly rather than reporting an unreliable figure. Add pepper as a commodity to the Price Agent's Agmarknet lookup, verify it has consistent mandi data for your target districts, and add a few pepper-specific conflict scenarios to the Phase 3 test set.

### Phase 8 — Evaluation and report

- **8.1 — Run the full evaluation.** Disease accuracy on PlantVillage vs PlantDoc for both classifiers, Telugu ASR Word Error Rate on farming vocabulary, end-to-end latency, a voice-vs-text-only ablation, the conflict-rule test suite pass rate, and a confidence-calibration spot-check (Section 9 has the full list). Produce clear tables/charts.
- **8.2 — Write the report.** Draft it from your real results and the literature survey. Use the novelty framing from Section 4 as the core claims, with baseline methodology and scope decisions mentioned as rigor, not novelty. Report the PlantVillage-to-PlantDoc accuracy drop honestly.
- **8.3 — Rehearse the demo.** Draft likely panel questions from your architecture and results, prepare honest answers for your weakest results, and record a backup demo video.

*8.1, 8.2, and 8.3 happen in the same Phase 8 chat, in that order — don't write the report before you have real evaluation numbers.*

### Separate Chat: Literature Survey — complete before Phase 1

Start this immediately, alongside Phase 0 — but unlike the rest of the build, this one has a checkpoint of its own: get a solid first pass through your candidate papers done before you start Phase 1, since your Phase 1 methodology decisions (especially the Disease Agent's architecture) should be informed by what you've read, not decided blind. Feed it one paper PDF at a time, extracted into a table (Author/Year, Title, Journal/DOI, Method, Dataset, Results, Key Contribution, Limitations, plus an honest quality/relevance assessment), tracked across four categories: Crop Disease Detection, Multilingual/Voice Advisory, Multi-Agent Coordination, and Price Prediction. You can keep adding papers to it after Phase 1 starts if you're building toward a larger total count — it just shouldn't still be empty by the time Phase 1 begins.

---

## 8. Using Claude Efficiently

**How to actually start each phase — the core workflow:**

1. Open a fresh chat for the new phase, with this document uploaded/available as context.
2. Tell it plainly: which phase you're starting (name it, e.g. "Phase 2"), that Section 7 of this guide has what needs to get done, that you want ONE step at a time, and that it should wait for you to confirm or report back before giving the next step — you're doing the actual typing and running, it's checking your work, not building for you.
3. If you're coming from a previous phase's chat, ask that chat, before you close it, to summarize in a few bullet points what was actually done/decided — then paste that summary into your opening message in the new chat, so it's not starting from the generic plan alone.

Do this the same way at the start of every phase — you don't need pre-written wording; describing what you want (phase number, one-step-at-a-time, check-before-moving-on, plus the carried-over summary) is enough for Claude to generate the right opening prompt itself, and it'll be more accurate than anything fixed in this document since it reflects what you actually did.

**Other habits worth keeping:**

- **One Project, this document as Project Knowledge.** Every chat starts with full context.
- **Start a new chat per phase, not on a fixed schedule.** Reuse the same chat within a phase — switching mid-phase loses debugging context. The literature survey is the one exception — it's a standalone chat that runs the whole time, alongside every phase.
- **Have Claude generate test cases before it generates logic** — sample questions for the Intent Router, conflict scenarios for the Orchestrator — before writing the code that handles them.
- **Keep Claude on a short leash for domain logic.** The weather→spray and price→sell rules are drafts Claude produces, not final answers — have someone with real agricultural knowledge sanity-check them.
- **Use Artifacts for diagrams and charts** once you have real Phase 8 numbers.
- **Update this document as real decisions get made** — final model IDs, the shared schema, the conflict-rule table — and re-upload so later chats work from what you actually built, not the original plan.
- **Web search anything that might be outdated** — model names, API tiers — rather than trusting memory.

---

## 9. Evaluation Plan

1. Disease accuracy: clean (PlantVillage) versus real-world (PlantDoc), flat baseline versus hierarchical model.
2. Telugu speech recognition Word Error Rate, specifically on farming vocabulary.
3. End-to-end response latency, question to spoken answer.
4. Simple ablation: system performance with voice versus without.
5. Conflict-handling test cases: the Phase 3 rule table's pass rate against deliberately constructed disagreement scenarios.
6. Confidence-calibration spot-check: is the system less often wrong when it says it's uncertain.

**Accuracy targets:** 95-98% on PlantVillage, 65-80% on PlantDoc — this drop is expected and should be reported honestly as a genuine finding, not hidden.

---

## 10. Team Roles (Reporting Purposes Only)

The actual build happens with all 4 of you together on one laptop, working through the phases in order — that's the real workflow, and it's a good one, since everyone stays on the same page and no one has to sync up scattered work later. These roles exist purely so you have a clean answer when your guide or a review panel asks "who owns which part":

- **Person 1:** Disease Agent
- **Person 2:** Weather and Price Agents (and the Soil Agent if you build it)
- **Person 3:** Orchestrator
- **Person 4:** Voice Integration

Since you're building together, use the roles as the answer to "who can explain this component" during your defense — pick whoever actually did the most talking/typing for that phase's steps in the Step-by-Step section, or just divide them however feels natural in hindsight. No need to force it during the actual build.

---

## 11. Risks and Mitigation

| Risk | Mitigation |
|---|---|
| Agmarknet data is messy to extract | Budget extra time; inspect real sample responses before parsing; scrape or download manually if needed |
| Voice integration is the hardest technical piece | The Phase 4 text-only pipeline is your guaranteed fallback if voice runs late |
| Location/mandi can't be reliably extracted from Telugu speech | Solved by design — the Farm Profile is structured input, not speech-parsed |
| Small local LLM struggles with Telugu reasoning or contradiction-resolution | Solved by design — conflict decisions are a deterministic rule table; the LLM only phrases the result, with a template fallback |
| Conflict-rule table is incomplete on demo day | Keep it as a living test suite from Phase 3 onward; add new scenarios (including Soil Agent ones) as you find them, don't treat it as "done" after one pass |
| Real-world accuracy is lower than lab accuracy | Expected and should be reported honestly as a finding, not a failure |
| Live demo fails on the day | Always have a recorded backup video of the working system; keep a hosted inference fallback since offline isn't in scope anyway |
| Mobile interface has issues | Test on an actual phone browser starting Phase 0, not only at Phase 6 |
| TTS mispronounces numbers (prices, percentages) in agent answers | **Resolved in Phase 5.2** — found via team listening + an automated ASR round-trip cross-check, fixed with a numeral-to-Telugu-words normalization step (`num2words`) before TTS; re-verified via the same round-trip method. Worth citing as a discovered-and-solved limitation, with before/after audio evidence. |
| Soil Agent scope creep threatens the core checkpoint | Strictly Phase 7, gated behind a solid Phase 4 — treat it as optional upside, never as something that delays the text-only milestone |
| PlantDoc may not have enough real-world pepper images for a meaningful accuracy number | Check this first thing in Phase 7 before investing time in the pepper classifier; if coverage is thin, report that limitation honestly rather than presenting an unreliable figure |
| Pepper scope creep threatens the core checkpoint | Same rule as Soil Agent — strictly Phase 7, gated behind a solid Phase 4, optional upside only |

---

## 12. Ethical Notes

- The system is advisory only, never diagnostic-certain. It should recommend confirming serious cases with a local agriculture officer.
- The weather-to-spray and price-to-sell advice logic should be sanity-checked by someone with real agricultural knowledge if possible.
- Any voice samples collected during testing need verbal consent and should not be retained longer than necessary.

---

## 13. Publication Plan

**Realistic target:** an IEEE Xplore-indexed conference (mid-tier or India-based — e.g. student research/social-innovation/applied-AI conferences), a workshop track at a major venue, or a solid regional/national journal. Main-track top-tier venues (NeurIPS, ICML, ACL, IEEE Transactions journals) are not realistic given this build's scope, team size, and timeline — being honest about that now avoids months spent on a submission unlikely to succeed. IEEE conference submissions need the IEEE double-column template and typically a per-paper registration/publication fee — confirm with your guide whether the department covers this, and check the specific conference's topic scope and deadline before targeting it.

**Novelty statement:** Existing systems combine disease detection, weather, and market advisory through multilingual chatbots, but voice interaction is usually left as future work, and no existing system explicitly resolves contradictory agent recommendations or communicates model confidence to the farmer. Krishi-Agent addresses this gap directly.

**Patent:** Not applicable — the project combines existing open techniques in a new application and doesn't meet the bar for a technical invention.

---

## 14. Cost Summary

**Total cost: zero.** Every dataset, pretrained model, and development tool used in this project is free and open-source, which also supports reproducibility if you pursue publication.

---

## 15. Final Checklist

These aren't a Phase 8 wrap-up list — most of them need to happen at the very start, or continuously, so they don't quietly cost you later. When to act on each:

- **Confirm your department's expected report format** — *do this before you write anything, ideally in Phase 0 alongside repo setup.* Page limits, required sections, and citation count expectations vary by department; know them before Phase 8, not during it.
- **Agree on your citation style** (IEEE or APA) — *also Phase 0*, and definitely before you open the literature survey chat (which starts in parallel from day one), so references never need reformatting.
- **Decide your demo device** — *Phase 0*, because Step 0.4 already has you testing mic capture on "the actual demo phone." Lock in which phone and laptop right then, don't leave it vague until Phase 6.
- **Backup your work regularly** — *ongoing, starting the moment your GitHub repo exists in Phase 0.* Push daily, every session, not just at the end of a phase.
- **Keep a simple running log** — *ongoing, a few lines after every phase*, not just at the end. This is what saves you from reconstructing memory under pressure when you sit down to write the Phase 8 report.
- **Test your Telugu speaker's availability** — *before Phase 5 specifically*, since that's when voice integration starts and you need them to validate ASR/TTS output sounds natural, not just that it runs.
