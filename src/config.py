# Relevant file paths
DATA_DIR = "data"
TRAIN_FILE = f"{DATA_DIR}/train.bin"
VAL_FILE = f"{DATA_DIR}/val.bin"
META_FILE = f"{DATA_DIR}/meta.pkl"
JOKES_CSV = f"{DATA_DIR}/dad_jokes.csv"

# GPT model parameters
block_size = 128
batch_size = 64
n_layers = 6
n_heads = 6
n_embd = 192
dropout = 0.1

# Training parameters
learning_rate = 0.0005
max_iters = 20000
eval_interval = 500
eval_iters = 10
patience = 6
delta = 0.0005