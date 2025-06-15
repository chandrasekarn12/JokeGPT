# Relevant file paths
DATA_DIR = "data"
TRAIN_FILE = f"{DATA_DIR}/train.bin"
VAL_FILE = f"{DATA_DIR}/val.bin"
META_FILE = f"{DATA_DIR}/meta.pkl"
JOKES_CSV = "dad_jokes.csv"

# GPT model parameters
block_size = 128
batch_size = 64
n_layers = 4
n_heads = 4
n_embd = 128
dropout = 0.1

# Training parameters
learning_rate = 0.001
max_iters = 2000
eval_interval = 100