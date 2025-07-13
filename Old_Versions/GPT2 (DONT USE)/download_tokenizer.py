import os
import requests

os.makedirs("tokenizer_files", exist_ok=True)

# File URLs from Hugging Face GPT-2
files = {
    "vocab.json": "https://huggingface.co/gpt2/resolve/main/vocab.json",
    "merges.txt": "https://huggingface.co/gpt2/resolve/main/merges.txt",
    "tokenizer_config.json": "https://huggingface.co/gpt2/resolve/main/tokenizer_config.json"
}

for name, url in files.items():
    dest = os.path.join("tokenizer_files", name)
    try:
        print(f"Downloading {name}...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(dest, "wb") as f:
            f.write(response.content)
        print(f"Saved {name} to {dest}")
    except Exception as e:
        print(f"Failed to download {name}: {e}")
