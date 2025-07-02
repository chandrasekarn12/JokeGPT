import os
import torch
from fastapi import FastAPI, Query
from transformers import GPT2TokenizerFast
from modelGPT2 import GPT2, GPT2Config
from config import DATA_DIR, TOKENIZER_DIR

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

checkpoint_path = os.path.join(DATA_DIR, "checkpoint.pt")
checkpoint = torch.load(checkpoint_path, map_location=device)
tokenizer = GPT2TokenizerFast.from_pretrained(TOKENIZER_DIR, local_files_only=True)

config = GPT2Config(tokenizer.vocab_size)
model = GPT2(config).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

app = FastAPI(title="JokeGPT API", version="0.1")

# Ready for container orchestration
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/joke")
def joke(prompt: str = Query("", description="Optional text prompt"),
         temperature: float = Query(1.0, ge=0.2, le=2.0, description="Temperature for text generation"),
         tokens: int = Query(50, ge=5, le=128, description="Number of tokens to generate")):
    with torch.no_grad():
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        if ids.numel() == 0:
            ids = torch.tensor([[tokenizer.bos_token_id]], device=device)
        output_ids = model.generate(ids, max_new_tokens=tokens, temperature=temperature)
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return {"joke": text.strip()}