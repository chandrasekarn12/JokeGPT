import torch
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
    dropout=checkpoint["config"]["dropout"],
    start_token=stoi.get('\n', 0)
)

model = GPTLanguageModel(config).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Generate inferences
def generate_text(max_new_tokens = 2000, temperature=1.0):
    output = model.generate(idx=None, max_new_tokens=max_new_tokens, temperature=temperature)[0].tolist()
    return decode(output)

if __name__ == "__main__":
    temperature = float(input("Enter temperature (default 1.0): ") or 1.0)
    tokens = int(input("Enter number of tokens to generate (default 200): ") or 200)
    print("Generating text...")
    print(generate_text(max_new_tokens=tokens, temperature=temperature))