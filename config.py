# Relevant file paths
DATA_DIR = "data"
TRAIN_FILE = f"{DATA_DIR}/train.bin"
VAL_FILE = f"{DATA_DIR}/val.bin"

# GPT model parameters
block_size = 128
batch_size = 32
n_layers = 2
n_heads = 2
n_embd = 128
dropout = 0.1

# Training parameters
learning_rate = 0.001
max_iters = 500
eval_interval = 100
eval_iters = 25