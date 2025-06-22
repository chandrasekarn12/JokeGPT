import torch
import pickle
import os
from config import DATA_DIR, block_size
from model import GPTLanguageModel, GPTConfig

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

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
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Generate inferences
def generate_text(prompt = "", max_new_tokens = 10):
    if prompt:
        context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
    
    output = model.generate(context, max_new_tokens=200, temperature=0.8)[0].tolist()
    return decode(output)

print(generate_text())