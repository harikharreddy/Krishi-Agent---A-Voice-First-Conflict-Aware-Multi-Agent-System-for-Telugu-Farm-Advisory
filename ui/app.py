import os
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

import tempfile
import torch
import soundfile as sf
import numpy as np
from transformers import AutoModel
import streamlit as st

from orchestrator.pipeline import run_pipeline


@st.cache_resource
def load_asr_model():
    model = AutoModel.from_pretrained(
        "ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True
    )
    model.eval()
    return model


def transcribe_audio(audio_bytes):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    wav_np, sr = sf.read(tmp_path, dtype="float32")
    if sr != 16000:
        raise ValueError(f"Expected 16000 Hz audio, got {sr} Hz")

    if wav_np.ndim == 1:
        wav_np = wav_np[np.newaxis, :]
    wav = torch.from_numpy(wav_np)

    asr_model = load_asr_model()
    with torch.no_grad():
        transcription = asr_model(wav, "te", "rnnt")
    return transcription


st.set_page_config(page_title="Krishi-Agent", page_icon="🌾")

st.title("Krishi-Agent")

if "farm_profile" not in st.session_state:
    st.session_state.farm_profile = None

with st.form("farm_profile_form"):
    district = st.text_input("District")
    mandi = st.text_input("Nearest Mandi")
    state = st.selectbox("State", ["Telangana", "Andhra Pradesh"])
    crop = st.selectbox("Crop", ["Tomato", "Potato"])
    submitted = st.form_submit_button("Save Profile")

    if submitted:
        if not district or not mandi:
            st.error("Please fill in both District and Mandi.")
        else:
            st.session_state.farm_profile = {
                "district": district,
                "mandi": mandi,
                "state": state,
                "crop": crop,
            }
            st.success("Farm profile saved for this session.")

if st.session_state.farm_profile:
    st.subheader("Current Profile")
    st.json(st.session_state.farm_profile)

st.divider()
st.subheader("Ask a Question")

if not st.session_state.farm_profile:
    st.info("Save your Farm Profile above before asking a question.")
else:
    audio_value = st.audio_input("Or record your question in Telugu")

    if audio_value is not None:
        st.success("Recording captured.")
        st.audio(audio_value)

    question = st.text_input("Type your question in Telugu")
    ask_submitted = st.button("Ask")

    if ask_submitted:
        final_question = None

        if audio_value is not None:
            with st.spinner("Transcribing your recording..."):
                final_question = transcribe_audio(audio_value.getvalue())
            st.write(f"Heard: {final_question}")
        elif question:
            final_question = question

        if not final_question:
            st.error("Please record or type a question.")
        else:
            with st.spinner("Thinking..."):
                trace = run_pipeline(final_question, st.session_state.farm_profile)
            st.subheader("Answer")
            st.write(trace["final_answer"])
