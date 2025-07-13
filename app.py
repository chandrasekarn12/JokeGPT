import os
import torch
import streamlit as st
from model import GPTLanguageModel, GPTConfig
from config import block_size

if torch.cuda.is_available():
    device = 'cuda'
else: 
    device = 'cpu'

checkpoint = torch.load(os.path.join("data", "checkpoint.pt"), map_location=device)
meta = checkpoint['meta']; stoi, itos = meta['stoi'], meta['itos']

config = GPTConfig(
    vocab_size=meta["vocab_size"], block_size=block_size,
    n_layers=checkpoint["config"]["n_layers"],
    n_heads=checkpoint["config"]["n_heads"],
    n_embd=checkpoint["config"]["n_embd"],
    dropout=checkpoint["config"]["dropout"],
    start_token=stoi.get('\n', 0)
)
model = GPTLanguageModel(config).to(device)
model.load_state_dict(checkpoint["model_state_dict"]); model.eval()

def decode(ids): return ''.join([itos[i] for i in ids]).strip()
def generate_joke(tokens=15, temperature=1.0):
    start = torch.tensor([[config.start_token]], dtype=torch.long).to(device)
    out = model.generate(start, max_new_tokens=tokens, temperature=temperature)[0].tolist()
    return decode(out)

st.set_page_config(page_title="JokeGPT", layout="centered")
st.title("JokeGPT 🤖")

tcol, scol = st.columns(2)
with tcol:
    temperature = st.slider("Temperature", 0.1, 1.5, 1.0, 0.1)
with scol:
    tokens = st.slider("Tokens", 50, 500, 250, 10)

if st.button("Tell me a joke"):
    st.write("Laughing at my own jokes...")
    joke = generate_joke(tokens, temperature)
    st.success(joke)
