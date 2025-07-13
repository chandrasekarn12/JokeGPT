import os
import requests
import frontend as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")

st.set_page_config(page_title="JokeGPT", page_icon=":smiley:", layout="centered")
st.title("JokeGPT :smiley:")

col1, col2 = st.columns(2)
with col1:
    temperature = st.slider("Temperature", min_value=0.2, max_value=2.0, value=1.0, step=0.1)
with col2:
    tokens = st.slider("Tokens", min_value=5, max_value=30, value=15, step=1)

if st.button("Generate Joke"):
    with st.spinner("Laughing at my own jokes..."):
        try:
            response = requests.post(
                f"{BACKEND_URL}/generate", 
                json={"temperature": temperature, "tokens": tokens}
            )
            if response.status_code == 200:
                joke = response.json().get("joke", "No joke generated.")
                st.success(joke)
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")