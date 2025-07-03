import torch
import os
import pickle
from transformers import GPT2TokenizerFast
from config import DATA_DIR, block_size, n_layers, n_heads, n_embd, dropout
from modelGPT1 import GPTLanguageModel, GPTConfig

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Load model and vocab
checkpoint_path = os.path.join(DATA_DIR, 'checkpoint.pt')
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

with open(os.path.join(DATA_DIR, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)

stoi = meta['stoi']
itos = meta['itos']

def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[int(i)] for i in l])

config = GPTConfig(
    vocab_size=meta['vocab_size'],
    block_size=block_size,
    n_layers=n_layers,
    n_heads=n_heads,
    n_embd=n_embd,
    dropout=dropout
)
model = GPTLanguageModel(config).to(device)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval()

# Generate inferences
@torch.no_grad()
def generate_text(prompt = "", max_new_tokens = 10, temperature=1.0):
    input_ids = torch.tensor(encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    if input_ids.numel() == 0:
        input_ids = torch.tensor([[tokenizer.bos_token_id]], device=device)
    
    output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temperature)
    return decode(output_ids[0])

if __name__ == "__main__":
    prompt = input("Enter a prompt: ")
    if not prompt.strip():
        prompt = " "
    temperature = float(input("Enter temperature (default 1.0): ") or 1.0)
    tokens = int(input("Enter number of tokens to generate (default 50): ") or 50)
    print("Generating text...")
    print(generate_text(prompt, max_new_tokens=tokens, temperature=temperature))