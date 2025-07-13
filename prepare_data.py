import os
import pickle
import numpy as np
import pandas as pd

from config import DATA_DIR, META_FILE, TRAIN_FILE, VAL_FILE, JOKES_CSV

os.makedirs(DATA_DIR, exist_ok=True)

# Load jokes csv
df = pd.read_csv(JOKES_CSV)
jokes = df['joke'].dropna().tolist()
data = "\n".join(jokes)

# Vocubulary
chars = sorted(set(data))
vocab_size = len(chars)
stoi = {}
for i, ch in enumerate(chars):
    stoi[ch] = i
itos = {}
for ch, i in stoi.items():
    itos[i] = ch

# Encoder and decoder functions
def encode(s):
    result = []
    for c in s:
        result.append(stoi[c])
    return result
def decode(l):
    result = []
    for i in l:
        result.append(itos[i])
    return ''.join(result)

# Data splitting
n = len(data)
train_data = data[:int(n * 0.9)]
val_data = data[int(n * 0.9):]
train_ids = np.array(encode(train_data), dtype=np.uint16)
val_ids = np.array(encode(val_data), dtype=np.uint16)

# Save encoded data
train_ids.tofile(TRAIN_FILE)
val_ids.tofile(VAL_FILE)

# Save metadata
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(META_FILE, 'wb') as f:
    pickle.dump(meta, f)

print("Data preparation complete.")