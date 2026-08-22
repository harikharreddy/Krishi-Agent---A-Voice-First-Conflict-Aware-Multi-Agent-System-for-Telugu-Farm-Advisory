import streamlit as st

st.set_page_config(page_title="Krishi-Agent", page_icon="🌾")

st.title("Krishi-Agent")

if "farm_profile" not in st.session_state:
    st.session_state.farm_profile = None

with st.form("farm_profile_form"):
    district = st.text_input("District")
    mandi = st.text_input("Nearest Mandi")
    crop = st.selectbox("Crop", ["Tomato", "Potato"])
    submitted = st.form_submit_button("Save Profile")

    if submitted:
        if not district or not mandi:
            st.error("Please fill in both District and Mandi.")
        else:
            st.session_state.farm_profile = {
                "district": district,
                "mandi": mandi,
                "crop": crop,
            }
            st.success("Farm profile saved for this session.")

if st.session_state.farm_profile:
    st.subheader("Current Profile")
    st.json(st.session_state.farm_profile)
