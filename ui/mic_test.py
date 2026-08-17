import streamlit as st

st.title("Krishi-Agent Mic Test")

audio_value = st.audio_input("Record a short Telugu sentence")

if audio_value:
    st.audio(audio_value)
    with open("test_recording.wav", "wb") as f:
        f.write(audio_value.getvalue())
    st.success("Saved recording to test_recording.wav")
