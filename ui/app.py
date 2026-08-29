import streamlit as st

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
from orchestrator.pipeline import run_pipeline

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
        if not question:
            st.error("Please type a question.")
        else:
            with st.spinner("Thinking..."):
                trace = run_pipeline(question, st.session_state.farm_profile)
            st.subheader("Answer")
            st.write(trace["final_answer"])
