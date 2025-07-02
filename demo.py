import os
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")

st.set_page_config(page_title="JokeGPT", page_icon=":smiley:", layout="centered")
st.title("JokeGPT :smiley:")

prompt = st.text_input("Prompt (optional)")
col1, col2 = st.columns(2)
with col1:
    temperature = st.slider("Temperature", min_value=0.2, max_value=2.0, value=1.0, step=0.1)
with col2:
    tokens = st.slider("Tokens", min_value=5, max_value=128, value=50, step=1)

if st.button("Generate Joke"):
    with st.spinner("Laughing at my own jokes..."):
        try:
            r = requests.get(f"{BACKEND_URL}/joke", params={
                "prompt": prompt,
                "temperature": temperature,
                "tokens": tokens
            }, timeout=30)
            r.raise_for_status()
            st.text_area("Joke", r.json().get("joke", "No joke generated."), height=200)
        except Exception as e:
            st.error(f"Request failed: {e}")