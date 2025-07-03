# Relevant file paths
DATA_DIR = "data"
TRAIN_FILE = f"{DATA_DIR}/train.bin"
VAL_FILE = f"{DATA_DIR}/val.bin"

# GPT model parameters
block_size = 256
batch_size = 32
n_layers = 6
n_heads = 6
n_embd = 384
dropout = 0.1
# block_size = 512
# batch_size = 64
# n_layers = 12
# n_heads = 12
# n_embd = 768
# dropout = 0.1

# Training parameters
learning_rate = 0.001
max_iters = 5000
eval_interval = 100
eval_iters = 25