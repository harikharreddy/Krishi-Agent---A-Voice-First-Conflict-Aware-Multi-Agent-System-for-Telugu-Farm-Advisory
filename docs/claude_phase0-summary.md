# Phase 0 Summary — Krishi-Agent (completed Aug 17, 2026)

## Repo
- `harikharreddy/Krishi-Agent---A-Voice-First-Conflict-Aware-Multi-Agent-System-for-Telugu-Farm-Advisory` (private, on a personal GitHub account, not an org)
- Folder structure: `agents/{disease,weather,price}`, `orchestrator/`, `ui/`, `shared/`, `notebooks/`, `data/` (gitignored on purpose — large datasets like PlantVillage/PlantDoc should never be committed), `docs/`, `tests/`
- All 3 teammates added as collaborators

## Colab
- Shared notebook `krishi-agent.ipynb`, T4 GPU, hosted on one teammate's personal Google account
- Not individually shared to the other 3 — team works together on one laptop (matches the project guide's actual workflow)

## ASR (Telugu speech-to-text)
- Model: `ai4bharat/indic-conformer-600m-multilingual`
- Works well — confirmed accurate on a real phone-recorded Telugu sentence (exact match on RNNT decoding; CTC decoding had one minor repeated-character artifact)
- Gated model on Hugging Face — needs a HF access token (read-only) and a one-time access request accepted on the model page
- Needs `onnxruntime` installed (plus `transformers`, `torchaudio`)
- Test sentence used: "నా పంట ఆకులపై మచ్చలు కనిపిస్తున్నాయి" (There are spots on my crop leaves)

## TTS (Telugu text-to-speech)
- Tried `ai4bharat/IndicF5` first — hit an unresolved `accelerate`/meta-device RuntimeError ("Tensor on device cpu is not on the expected device meta!"), abandoned after multiple fix attempts
- Switched to `ai4bharat/indic-parler-tts` — works, but fragile:
  - Requires `transformers==4.46.1` pinned exactly (newer versions break it — `ImportError: cannot import name 'isin_mps_friendly'`)
  - Requires `os.environ["USE_TF"]="0"` and `os.environ["USE_FLAX"]="0"` set *before* importing transformers/parler_tts (avoids a protobuf version conflict triggered by transformers auto-detecting TensorFlow)
  - **Gotcha:** if `transformers` gets imported earlier in the same Colab session (e.g. by running the ASR cell first) before the pin is applied, Python caches the old version in memory for that session — pinning the package on disk afterward does NOT fix it until you restart the Colab session (Runtime → Restart session; this keeps installed packages, just clears memory)
  - Gated model — needs a separate HF access request
  - Recommended Telugu voices: "Lalitha" or "Prakash", specified via a natural-language description prompt (e.g. "Lalitha's voice is calm and clear...")

## Mic capture
- Streamlit's `st.audio_input` confirmed working on Android/Chrome
- Tested via `ngrok http 8501` tunnel
- localtunnel was tried first and failed — doesn't reliably proxy Streamlit's websocket connection, causing a blank/unresponsive page on the phone
- ngrok requires a free account + authtoken (`ngrok config add-authtoken YOUR_TOKEN`)

## Working code locations (in repo)
- `notebooks/krishi_agent_phase0_final.ipynb`
- `ui/mic_test.py`

## Known fragility / lessons for Phase 1
- Colab sessions disconnected/reset more often than expected during a long working session, silently wiping installed packages and in-memory variables — budget time for re-running install cells when this happens
- "Runtime → Restart session" clears memory but keeps installed packages; "Runtime → Disconnect and delete runtime" wipes everything (full clean VM) — use the latter when debugging deep dependency conflicts, since accumulated installs across many restarts can create messy, hard-to-diagnose conflicts
- When installing a new pinned version of a package already imported earlier in the session, a session restart is required for it to take effect
- Consider whether Colab Pro (more stable sessions, less disconnection) is worth it for Phase 1's longer disease-classifier training runs
