import { DISTRICTS, CROPS, TELUGU_CROP_NAMES } from "./districts.js";

function teluguCropName(cropEnglish) {
  return TELUGU_CROP_NAMES[cropEnglish] || cropEnglish;
}

const PROFILE_KEY = "krishi_farm_profile";

// ---------- Farm profile (client-side only; resent with every request) ----------

function loadProfile() {
  try {
    const raw = localStorage.getItem(PROFILE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function saveProfile(profile) {
  localStorage.setItem(PROFILE_KEY, JSON.stringify(profile));
}

// ---------- Profile screen ----------

const profileScreen = document.getElementById("profile-screen");
const askScreen = document.getElementById("ask-screen");
const stateSelect = document.getElementById("state-select");
const districtSelect = document.getElementById("district-select");
const mandiInput = document.getElementById("mandi-input");
const cropSelect = document.getElementById("crop-select");
const profileForm = document.getElementById("profile-form");
const profileChip = document.getElementById("profile-chip");
const editProfileBtn = document.getElementById("edit-profile-btn");
const profileCancelBtn = document.getElementById("profile-cancel-btn");

function populateSelect(select, options, placeholder) {
  select.innerHTML = "";
  if (placeholder) {
    const ph = document.createElement("option");
    ph.value = "";
    ph.textContent = placeholder;
    ph.disabled = true;
    ph.selected = true;
    select.appendChild(ph);
  }
  for (const opt of options) {
    const el = document.createElement("option");
    el.value = opt;
    el.textContent = opt;
    select.appendChild(el);
  }
}

// Placeholders force an active choice (required + no auto-selected real
// value) -- without this, a farmer skipping the field silently saves
// whatever's alphabetically first (e.g. "Alluri Sitharama Raju" as
// District), not a deliberate choice.
function populateDistricts(state) {
  populateSelect(districtSelect, DISTRICTS[state] || [], "-- జిల్లా / District ఎంచుకోండి --");
}

populateSelect(stateSelect, Object.keys(DISTRICTS), "-- రాష్ట్రం / State ఎంచుకోండి --");
populateSelect(cropSelect, CROPS, "-- పంట / Crop ఎంచుకోండి --");
populateDistricts(stateSelect.value);

stateSelect.addEventListener("change", () => populateDistricts(stateSelect.value));

function showProfileScreen(existing) {
  if (existing) {
    stateSelect.value = existing.state;
    populateDistricts(existing.state);
    districtSelect.value = existing.district;
    mandiInput.value = existing.mandi;
    cropSelect.value = existing.crop;
  }
  // Only show a way back out if there's an existing profile to go back TO --
  // first-time setup has no ask screen behind it yet.
  profileCancelBtn.hidden = !existing;
  profileScreen.hidden = false;
  askScreen.hidden = true;
}

profileCancelBtn.addEventListener("click", () => {
  const existing = loadProfile();
  if (existing) showAskScreen(existing);
});

function showAskScreen(profile) {
  profileChip.textContent = `${profile.district}, ${profile.state} · ${profile.crop}`;
  profileScreen.hidden = true;
  askScreen.hidden = false;
}

profileForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const profile = {
    state: stateSelect.value,
    district: districtSelect.value,
    mandi: mandiInput.value.trim(),
    crop: cropSelect.value,
  };
  saveProfile(profile);
  showAskScreen(profile);
});

editProfileBtn.addEventListener("click", () => showProfileScreen(loadProfile()));

// ---------- WAV encoding ----------

function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  function writeString(offset, str) {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  }

  writeString(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  writeString(36, "data");
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([view], { type: "audio/wav" });
}

// ---------- Mic recorder (16kHz raw PCM via AudioWorklet -- no server-side resampling needed) ----------

class WavRecorder {
  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1 } });
    this.audioContext = new AudioContext({ sampleRate: 16000 });
    await this.audioContext.audioWorklet.addModule("/js/recorder-worklet.js");
    this.source = this.audioContext.createMediaStreamSource(this.stream);
    this.workletNode = new AudioWorkletNode(this.audioContext, "pcm-recorder");
    this.chunks = [];
    this.workletNode.port.onmessage = (e) => this.chunks.push(e.data);
    this.source.connect(this.workletNode);
  }

  stop() {
    const sampleRate = this.audioContext.sampleRate;
    this.workletNode.disconnect();
    this.source.disconnect();
    this.stream.getTracks().forEach((t) => t.stop());
    this.audioContext.close();

    const totalLength = this.chunks.reduce((a, c) => a + c.length, 0);
    const merged = new Float32Array(totalLength);
    let offset = 0;
    for (const c of this.chunks) {
      merged.set(c, offset);
      offset += c.length;
    }
    return encodeWav(merged, sampleRate);
  }
}

// ---------- State machine: idle -> listening -> thinking -> ready ----------

const stage = document.getElementById("stage");
const micBtn = document.getElementById("mic-btn");
const stageCaption = document.getElementById("stage-caption");
const photoBtn = document.getElementById("photo-btn");
const photoInput = document.getElementById("photo-input");
const photoPreview = document.getElementById("photo-preview");
const photoRemoveBtn = document.getElementById("photo-remove-btn");
const questionInput = document.getElementById("question-input");
const askTypedBtn = document.getElementById("ask-typed-btn");
const mismatchBanner = document.getElementById("mismatch-banner");
const mismatchText = document.getElementById("mismatch-text");
const mismatchUpdateBtn = document.getElementById("mismatch-update-btn");
const answerCard = document.getElementById("answer-card");
const heardTextEl = document.getElementById("heard-text");
const detectedCropChip = document.getElementById("detected-crop-chip");
const detectedCropText = document.getElementById("detected-crop-text");
const answerTextEl = document.getElementById("answer-text");
const answerAudio = document.getElementById("answer-audio");
const askAgainBtn = document.getElementById("ask-again-btn");
const errorBanner = document.getElementById("error-banner");
const errorText = document.getElementById("error-text");
const errorDismissBtn = document.getElementById("error-dismiss-btn");
const recordTimerEl = document.getElementById("record-timer");

// In-page, bilingual, dismissable -- replaces alert(), which is jarring,
// English-only, and breaks out of the app's whole visual design.
function showError(teluguMsg, englishMsg) {
  errorText.textContent = englishMsg ? `${teluguMsg} (${englishMsg})` : teluguMsg;
  errorBanner.hidden = false;
}

function hideError() {
  errorBanner.hidden = true;
}

errorDismissBtn.addEventListener("click", hideError);

const CAPTIONS = {
  idle: "నొక్కి మీ ప్రశ్న అడగండి / Tap to ask",
  listening: "వింటున్నాను... / Listening...",
  thinking: "ఆలోచిస్తున్నాను... / Thinking...",
  ready: "సమాధానం సిద్ధం / Answer ready",
};

let currentPhoto = null; // { file, dataUrl }
let recorder = null;
let isSubmitting = false;
let recordingTimerInterval = null;
let recordingStartedAt = null;

const MAX_RECORDING_MS = 45000;
const MIN_RECORDING_MS = 600;

function setState(next) {
  stage.dataset.state = next;
  stage.classList.remove("audio-playing");
  stageCaption.textContent = CAPTIONS[next];
  recordTimerEl.textContent = "";
  const busy = next === "thinking";
  // Blocks every submission trigger while a request is in flight -- each one
  // is a real, expensive backend call (ASR/pipeline/TTS), and firing two at
  // once would reintroduce the exact memory contention the Ollama-unload fix
  // was built to solve.
  micBtn.disabled = busy;
  askTypedBtn.disabled = busy;
  photoBtn.disabled = busy;
}

setState("idle");

function updateRecordingTimer() {
  const elapsedMs = Date.now() - recordingStartedAt;
  const seconds = Math.floor(elapsedMs / 1000);
  recordTimerEl.textContent = `0:${String(seconds).padStart(2, "0")}`;
  if (elapsedMs >= MAX_RECORDING_MS) {
    finishRecording();
  }
}

async function finishRecording() {
  clearInterval(recordingTimerInterval);
  const elapsedMs = Date.now() - recordingStartedAt;
  const wavBlob = recorder.stop();

  if (elapsedMs < MIN_RECORDING_MS) {
    showError(
      "రికార్డింగ్ చాలా చిన్నది, మళ్ళీ ప్రయత్నించండి.",
      "Recording was too short — please try again."
    );
    setState("idle");
    return;
  }
  setState("thinking");
  await submitQuestion({ audioBlob: wavBlob });
}

micBtn.addEventListener("click", async () => {
  const state = stage.dataset.state;
  if (state === "idle") {
    hideError();
    try {
      recorder = new WavRecorder();
      await recorder.start();
      setState("listening");
      recordingStartedAt = Date.now();
      updateRecordingTimer();
      recordingTimerInterval = setInterval(updateRecordingTimer, 500);
    } catch (err) {
      showError(
        "మైక్ యాక్సెస్ ఇవ్వలేదు, దయచేసి అనుమతి ఇవ్వండి.",
        "Couldn't access the microphone. Please allow microphone access and try again."
      );
      console.error(err);
    }
  } else if (state === "listening") {
    await finishRecording();
  } else if (state === "ready") {
    toggleAnswerPlayback();
  }
});

function toggleAnswerPlayback() {
  if (answerAudio.paused) {
    answerAudio.play().catch(() => {});
  } else {
    answerAudio.pause();
    answerAudio.currentTime = 0;
  }
}

answerAudio.addEventListener("play", () => stage.classList.add("audio-playing"));
answerAudio.addEventListener("pause", () => stage.classList.remove("audio-playing"));
answerAudio.addEventListener("ended", () => stage.classList.remove("audio-playing"));

askAgainBtn.addEventListener("click", () => {
  answerCard.hidden = true;
  mismatchBanner.hidden = true;
  hideError();
  heardTextEl.textContent = "";
  answerTextEl.textContent = "";
  detectedCropChip.hidden = true;
  questionInput.value = "";
  clearPhoto();
  setState("idle");
});

// ---------- Photo attach ----------

photoBtn.addEventListener("click", () => photoInput.click());

photoInput.addEventListener("change", () => {
  const file = photoInput.files[0];
  if (!file) return;
  currentPhoto = file;
  const reader = new FileReader();
  reader.onload = () => {
    photoPreview.src = reader.result;
    photoPreview.hidden = false;
    photoRemoveBtn.hidden = false;
  };
  reader.readAsDataURL(file);
});

photoRemoveBtn.addEventListener("click", clearPhoto);

function clearPhoto() {
  currentPhoto = null;
  photoInput.value = "";
  photoPreview.src = "";
  photoPreview.hidden = true;
  photoRemoveBtn.hidden = true;
}

// ---------- Typed-question fallback ----------

askTypedBtn.addEventListener("click", async () => {
  const text = questionInput.value.trim();
  if (!text) {
    questionInput.focus();
    return;
  }
  hideError();
  setState("thinking");
  await submitQuestion({ questionText: text });
});

questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") askTypedBtn.click();
});

// ---------- Submit to backend ----------

async function submitQuestion({ audioBlob, questionText }) {
  if (isSubmitting) return;
  isSubmitting = true;

  const profile = loadProfile();
  const form = new FormData();
  form.append("state", profile.state);
  form.append("district", profile.district);
  form.append("mandi", profile.mandi);
  form.append("crop", profile.crop);
  if (audioBlob) form.append("audio", audioBlob, "question.wav");
  if (questionText) form.append("question", questionText);
  if (currentPhoto) form.append("photo", currentPhoto, currentPhoto.name);

  try {
    const resp = await fetch("/api/ask", { method: "POST", body: form });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || err.error || `HTTP ${resp.status}`);
    }
    const data = await resp.json();
    renderAnswer(data, profile);
  } catch (err) {
    console.error(err);
    showError(
      "సమాధానం రావడంలో సమస్య వచ్చింది, మళ్ళీ ప్రయత్నించండి.",
      "Something went wrong getting your answer. Please try again."
    );
    setState("idle");
  } finally {
    isSubmitting = false;
  }
}

function renderAnswer(data, profile) {
  if (data.heard_text) {
    heardTextEl.hidden = false;
    heardTextEl.textContent = `"${data.heard_text}"`;
  } else {
    heardTextEl.hidden = true;
  }
  answerTextEl.textContent = data.answer_text;
  answerAudio.src = `data:audio/wav;base64,${data.audio_wav_b64}`;

  if (data.detected_crop) {
    detectedCropText.textContent = `${teluguCropName(data.detected_crop)} ఆకు`;
    detectedCropChip.hidden = false;
  } else {
    detectedCropChip.hidden = true;
  }

  if (data.disease_mismatch) {
    const { detected_crop, profile_crop } = data.disease_mismatch;
    const detectedTe = teluguCropName(detected_crop);
    const profileTe = teluguCropName(profile_crop);
    mismatchText.textContent =
      `ఈ ఫోటో ${detectedTe} ఆకు లా ఉంది, మీ ప్రొఫైల్‌లో ${profileTe} ఉంది. ` +
      `(This looks like a ${detected_crop} leaf, not ${profile_crop} — your saved profile.)`;
    mismatchUpdateBtn.textContent = `${detectedTe} గా మార్చు / Switch to ${detected_crop}`;
    mismatchUpdateBtn.onclick = () => {
      const updated = { ...profile, crop: detected_crop };
      saveProfile(updated);
      profileChip.textContent = `${updated.district}, ${updated.state} · ${updated.crop}`;
      mismatchBanner.hidden = true;
    };
    mismatchBanner.hidden = false;
  } else {
    mismatchBanner.hidden = true;
  }

  answerCard.hidden = false;
  setState("ready");
  answerAudio.play().catch(() => {
    // Autoplay may be blocked -- the big button still doubles as a play control.
  });
}

// ---------- Boot ----------

const existingProfile = loadProfile();
if (existingProfile) {
  showAskScreen(existingProfile);
} else {
  showProfileScreen(null);
}
