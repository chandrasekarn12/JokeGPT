import torch
import os
from transformers import GPT2TokenizerFast
from config import DATA_DIR
from modelGPT2 import GPT2, GPT2Config

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Load model and vocab
checkpoint_path = os.path.join(DATA_DIR, 'checkpoint.pt')
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

config = GPT2Config(vocab_size=tokenizer.vocab_size)
model = GPT2(config).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# def encode(text):
#     return tokenizer.encode(text, return_tensors='pt').to(device)

def decode(tok_ids):
    return tokenizer.decode(tok_ids, skip_special_tokens=True)


# Generate inferences
@torch.no_grad()
def generate_text(prompt = "", max_new_tokens = 10, temperature=1.0):
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
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