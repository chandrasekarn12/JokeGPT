import os
import torch
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from model import GPTLanguageModel, GPTConfig
from config import DATA_DIR, block_size

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Load model and vocab
checkpoint_path = os.path.join(DATA_DIR, 'checkpoint.pt')
checkpoint = torch.load(checkpoint_path, map_location=device)
meta = checkpoint['meta']

stoi = meta['stoi']
itos = meta['itos']
def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

config = GPTConfig(
    vocab_size=meta["vocab_size"],
    block_size=block_size,
    n_layers=checkpoint["config"]["n_layers"],
    n_heads=checkpoint["config"]["n_heads"],
    n_embd=checkpoint["config"]["n_embd"],
    dropout=checkpoint["config"]["dropout"]
)
model = GPTLanguageModel(config).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

app = FastAPI(title="JokeGPT API", version="0.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ready for container orchestration
@app.get("/")
def root():
    return {"message": "Welcome to JokeGPT API"}
 
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/joke")
def joke(temperature: float = Query(1.0, ge=0.2, le=2.0, description="Temperature"),
    tokens: int = Query(200, ge=50, le=500, description="Tokens to generate")):
    
    model.eval()
    with torch.no_grad():
        x = torch.zeros((1, 1), dtype=torch.long).to(device)
        y = model.generate(max_new_tokens=tokens, temperature=temperature)
        joke = ''.join([itos[i] for i in y[0].tolist()])
    return {"joke": joke.strip()}