from transformers import GPT2TokenizerFast
import pandas as pd
import numpy as np
import os
import torch
from torch.nn.utils.rnn import pad_sequence
import re
import matplotlib.pyplot as plt
from config import DATA_DIR, TRAIN_FILE, VAL_FILE, block_size

os.makedirs(DATA_DIR, exist_ok=True)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

df = pd.read_csv(os.path.join(DATA_DIR, 'dad_jokes.csv'))

jokes = df['joke'].dropna().astype(str).tolist()

encoded_chunks = []
count = 0
for j in jokes:
    j = re.sub(r'\s+', ' ', j).strip()
    if re.search(r'(https?://|www\.|img)', j, re.IGNORECASE):
        continue
    non_ascii = False
    for c in j:
        if ord(c) >= 128:
            non_ascii = True
            break
    if non_ascii:
        continue
    tokens = tokenizer.encode(j, truncation=True, max_length=block_size)
    if len(tokens) < 10:
        continue
    if len(tokens) <= block_size:
        encoded_chunks.append(torch.tensor(tokens, dtype=torch.long))
print(f'Loaded {len(encoded_chunks)} chunks/{len(jokes)} jokes from CSV.')

padded = pad_sequence(encoded_chunks, batch_first=True, padding_value=tokenizer.pad_token_id)
indices = torch.randperm(padded.size(0))
padded = padded[indices]

split = int(0.9 * len(padded))
train_data = padded[:split]
val_data = padded[split:]

np.array(train_data.cpu(), dtype=np.uint16).tofile(TRAIN_FILE)
np.array(val_data.cpu(), dtype=np.uint16).tofile(VAL_FILE)

print("Tokenization complete")