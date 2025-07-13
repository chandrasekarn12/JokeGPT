import os, pickle, time, math, torch
import numpy as np
from torch.nn import functional as F
import matplotlib.pyplot as plt
import wandb
from config import (
    DATA_DIR, TRAIN_FILE, VAL_FILE, META_FILE,
    block_size, batch_size, learning_rate, max_iters, eval_interval, 
    eval_iters, n_layers, n_heads, n_embd, dropout, patience, delta
)
from model import GPTLanguageModel, GPTConfig

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# Load tokenized data
train_data = np.memmap(TRAIN_FILE, dtype=np.uint16, mode='r')
val_data = np.memmap(VAL_FILE, dtype=np.uint16, mode='r')

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x_list = []
    y_list = []
    for i in ix:
        x_item = torch.from_numpy(data[i : i + block_size].copy()).long()
        y_item = torch.from_numpy(data[i + 1 : i + 1 + block_size].copy()).long()
        x_list.append(x_item)
        y_list.append(y_item)
    x = torch.stack(x_list)
    y = torch.stack(y_list)
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Build the actual model
with open(META_FILE, 'rb') as f:
    meta = pickle.load(f)
vocab_size = meta['vocab_size']

config = GPTConfig(
    vocab_size=vocab_size,
    block_size=block_size,
    n_layers=n_layers,
    n_heads=n_heads,
    n_embd=n_embd,
    dropout=dropout,
)
model = GPTLanguageModel(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Initialize run
run = wandb.init(
        project="jokegpt",
        config = dict(
            block_size   = block_size,
            batch_size   = batch_size,
            n_layers     = n_layers,
            n_heads      = n_heads,
            n_embd       = n_embd,
            dropout      = dropout,
            learning_rate= learning_rate,
            max_iters    = max_iters)
)
wandb.watch_called = False
wandb.watch(model, log='gradients', log_freq=100)

# For plotting
train_losses = []
val_losses = []
iters = []
best_val_loss = float('inf')
epochs_no_improve = 0

# W and B logging
run = wandb.init(
    project = "jokegpt",
    name    = "gpt1_8x320_bs32",
    config  = dict(
        block_size=block_size, batch_size=batch_size,
        n_layers=n_layers,   n_heads=n_heads, n_embd=n_embd,
        dropout=dropout,     lr=learning_rate,
        max_iters=max_iters)
)
wandb.watch(model, log="gradients", log_freq=100)

# Training loop
t0 = time.time()
for iter in range(1, max_iters + 1):
    # Get batch
    xb, yb = get_batch('train')

    # Forward pass
    logits, loss = model(xb, yb)

    # Backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
        
    if iter % 10 == 0:
        wandb.log({
            "train/loss": loss.item(),
            "train/lr":   optimizer.param_groups[0]['lr'],
            "iter":       iter
        }, step=iter)
    
    # Logging and eval
    if iter % eval_interval == 0:
        losses = estimate_loss(model)

        train_loss = losses['train'].item()
        val_loss = losses['val'].item()

        print(f"iter {iter:6d} | train {train_loss:.4f} | "
              f"val {val_loss:.4f} | time {time.time()-t0:,.0f}s")

        wandb.log({
            "eval/train_loss": train_loss,
            "eval/val_loss":   val_loss,
        }, step=iter)

        if val_loss < best_val_loss - delta:
            best_val_loss = val_loss
            epochs_no_improve = 0
            ckpt_p = os.path.join(DATA_DIR, 'best_checkpoint.pt')
            torch.save({'model_state_dict': model.state_dict(),
                        'config': config.__dict__,
                        'iter': iter,
                        'val_loss': val_loss}, ckpt_p)
            print(f"  ✓  New best checkpoint saved to {ckpt_p}")

            # push artifact every new best
            art = wandb.Artifact('model', type='checkpoint')
            art.add_file(ckpt_p)
            run.log_artifact(art, aliases=['best', f'iter_{iter}'])
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping at iter {iter} "
                      f"(best val loss {best_val_loss:.4f})")
                break

        # Store losses for plotting
        train_losses.append(losses['train'].item())
        val_losses.append(losses['val'].item())
        iters.append(iter)

# Final checkpoint
final_checkpoint_path = os.path.join(DATA_DIR, 'checkpoint.pt')
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config.__dict__,
    'meta': meta
}, final_checkpoint_path)
print(f"Final model checkpoint saved to {final_checkpoint_path}")

run.finish()

# Plot losses
plt.figure()
plt.plot(iters, train_losses, label='Train Loss')
plt.plot(iters, val_losses, label='Val Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid()
plt.savefig(os.path.join(DATA_DIR, 'loss_curve.png'))
plt.show()

# optional shutdown
#os.system("shutdown /s /t 1")