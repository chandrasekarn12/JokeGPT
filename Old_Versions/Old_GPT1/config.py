# Relevant file paths
DATA_DIR = "data"
TRAIN_FILE = f"{DATA_DIR}/train.bin"
VAL_FILE = f"{DATA_DIR}/val.bin"

# GPT model parameters
block_size = 192
batch_size = 32
n_layers = 8
n_heads = 8
n_embd = 320
dropout = 0.15
# block_size = 512
# batch_size = 64
# n_layers = 12
# n_heads = 12
# n_embd = 768
# dropout = 0.1

# Training parameters
learning_rate = 0.0003
max_iters = 12000
eval_interval = 200
eval_iters = 40 

# Early stopping
patience = 10
min_delta = 0.0005