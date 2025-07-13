import torch
from model import GPTLanguageModel, GPTConfig
from config import DATA_DIR, block_size
import streamlit as st
import os

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint = torch.load(os.path.join(DATA_DIR, 'checkpoint.pt'), map_location=device)
meta = checkpoint['meta']
stoi = meta['stoi']
itos = meta['itos']

def decode(indices):
    return ''.join([itos[i] for i in indices])

config = GPTConfig(
    vocab_size=meta["vocab_size"],
    block_size=block_size,
    n_layers=checkpoint["config"]["n_layers"],
    n_heads=checkpoint["config"]["n_heads"],
    n_embd=checkpoint["config"]["n_embd"],
    dropout=checkpoint["config"]["dropout"],
    start_token=stoi.get('\n', 0)
)
model = GPTLanguageModel(config).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Inference function
def generate_joke(tokens=15, temperature=1.0):
    with torch.no_grad():
        start_token = torch.tensor([[stoi.get('\n', 0)]], dtype=torch.long).to(device)
        out = model.generate(start_token, max_new_tokens=tokens, temperature=temperature)[0].tolist()
    return decode(out).strip()

# Streamlit UI
st.set_page_config(page_title="JokeGPT", layout="centered")
st.title("JokeGPT 🤖")
st.markdown("A GPT-based dad joke generator.")

col1, col2 = st.columns(2)
with col1:
    temperature = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1)
with col2:
    tokens = st.slider("Tokens", 50, 500, 250, 10)

if st.button("Generate Joke"):
    with st.spinner("Laughing at my own jokes..."):
        st.success(generate_joke(tokens, temperature))
