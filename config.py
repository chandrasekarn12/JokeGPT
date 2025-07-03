# Relevant file paths
DATA_DIR = "data"
TRAIN_FILE = f"{DATA_DIR}/train.bin"
VAL_FILE = f"{DATA_DIR}/val.bin"
META_FILE = f"{DATA_DIR}/meta.pkl"

# GPT model parameters
block_size = 512
batch_size = 64
n_layers = 12
n_heads = 12
n_embd = 768
dropout = 0.1

# Training parameters
learning_rate = 0.001
max_iters = 1000
eval_interval = 100
eval_iters = 25