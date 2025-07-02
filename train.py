import os, time, torch
import numpy as np
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
from transformers import GPT2TokenizerFast
from config import (
    DATA_DIR, TRAIN_FILE, VAL_FILE, TOKENIZER_DIR,
    block_size, batch_size,
    learning_rate, max_iters, eval_interval, eval_iters
)
from modelGPT2 import GPT2, GPT2Config

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# Load tokenized data and vocab
train_data = np.memmap(TRAIN_FILE, dtype=np.uint16, mode='r')
val_data = np.memmap(VAL_FILE, dtype=np.uint16, mode='r')
tokenizer = GPT2TokenizerFast.from_pretrained(TOKENIZER_DIR, local_files_only=True)

# Make model and optimizer
config = GPT2Config(vocab_size = tokenizer.vocab_size)
model = GPT2(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=0.00001)
scaler = torch.amp.GradScaler(device)

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x_list = []
    y_list = []
    for i in ix:
        chunk = torch.from_numpy(data[i : i + block_size + 1].copy()).long()
        x_list.append(chunk[:-1])
        y_list.append(chunk[1:])
    x = torch.stack(x_list).to(device)
    y = torch.stack(y_list).to(device)
    return x, y

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

# For plotting
train_losses = []
val_losses = []
iters = []

# Training loop
t0 = time.time()
for iter in range(1, max_iters + 1):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    # Logging and eval
    if iter % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"iter {iter:6d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | time {time.time()-t0:,.0f}s")
        checkpoint_path = os.path.join(DATA_DIR, f'checkpoint_iter{iter}.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config.__dict__,
        }, checkpoint_path)
        print(f"Model checkpoint saved to {checkpoint_path}")

        train_losses.append(losses['train'].item())
        val_losses.append(losses['val'].item())
        iters.append(iter)

final_checkpoint_path = os.path.join(DATA_DIR, 'checkpoint.pt')
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config.__dict__,
}, final_checkpoint_path)
print(f"Final model checkpoint saved to {final_checkpoint_path}")

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