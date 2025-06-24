import torch
import pickle
import os
from transformers import GPT2TokenizerFast
from config import DATA_DIR, block_size
from modelGPT2 import GPT2, GPT2Config

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

tokenizer = GPT2TokenizerFast.from_pretrained("tokenizer_files", local_files_only=True)

# Load model and vocab
checkpoint_path = os.path.join(DATA_DIR, 'checkpoint.pt')
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
meta = checkpoint['meta']

config = GPT2Config(vocab_size=meta["vocab_size"])
model = GPT2(config).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def encode(text):
    return tokenizer.encode(text, return_tensors='pt').to(device)

def decode(tok_ids):
    if tok_ids is None or tok_ids.size(1) == 0:
        return ""
    return tokenizer.decode(tok_ids[0], skip_special_tokens=True)


# Generate inferences
@torch.no_grad()
def generate_text(prompt = "", max_new_tokens = 10, temperature=1.0):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    if input_ids.size(1) == 0:
        input_ids = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long, device=device)
    
    output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temperature)
    return decode(output_ids)

if __name__ == "__main__":
    prompt = input("Enter a prompt: ")
    if not prompt.strip():
        prompt = " "
    temperature = float(input("Enter temperature (default 1.0): ") or 1.0)
    tokens = int(input("Enter number of tokens to generate (default 50): ") or 50)
    print("Generating text...")
    print(generate_text(prompt, max_new_tokens=tokens, temperature=temperature))