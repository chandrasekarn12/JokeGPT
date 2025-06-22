import os
import pickle
import numpy as np
import pandas as pd

from config import DATA_DIR, META_FILE, TRAIN_FILE, VAL_FILE

os.makedirs(DATA_DIR, exist_ok=True)

# Load jokes csv from kaggle dataset
csv_path = os.path.join(DATA_DIR, 'reddit_dadjokes.csv')
df = pd.read_csv(csv_path)

df = df[df['joke'].notna() & df['joke'].str.strip().ne('')]
jokes = df['joke'].str.strip().tolist()
data = '\n'.join(jokes)

# Vocabulary
chars = sorted(set(data))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])

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
