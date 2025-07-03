import os, time, torch
import numpy as np
import wandb
from torch.nn import functional as F
from transformers import get_linear_schedule_with_warmup
from transformers import GPT2TokenizerFast
import matplotlib.pyplot as plt
from config import (
    DATA_DIR, TRAIN_FILE, VAL_FILE,
    block_size, batch_size, n_layers, n_heads, n_embd, dropout,
    learning_rate, max_iters, eval_interval, eval_iters
)
from modelGPT1 import GPTLanguageModel, GPTConfig

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# Load tokenized data and vocab
train_data = np.memmap(TRAIN_FILE, dtype=np.uint16, mode='r')
val_data = np.memmap(VAL_FILE, dtype=np.uint16, mode='r')
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
vocab = tokenizer.get_vocab()

# Make model and optimizer
config = GPTConfig(vocab_size=len(vocab), pad_token_id=tokenizer.pad_token_id)
model = GPTLanguageModel(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=200, num_training_steps=max_iters)

# Setup W and B
wandb_config = {
        'project': 'jokegpt',
        'job_type': 'train'
    }
config_wandb = {
            "block_size":   block_size,
            "batch_size":   batch_size,
            "n_layers":     n_layers,
            "n_heads":      n_heads,
            "n_embd":       n_embd,
            "dropout":      dropout,
            "learning_rate":learning_rate,
            "max_iters":    max_iters,
        }
run = wandb.init(**wandb_config, config=config_wandb)
wandb.watch_called = False
wandb.watch(models=model, log='gradients', log_freq=100)

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
    loss.backward()
    optimizer.step()
    scheduler.step()

    # Logging and eval
    if iter % eval_interval == 0:
        # W and B
        wandb.log({
            "iter":       iter,
            "train/loss": loss.item(),
            "train/lr":   scheduler.get_last_lr()[0],
        }, step=iter)

        losses = estimate_loss(model)
        print(f"iter {iter:6d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | time {time.time()-t0:,.0f}s")
        
        wandb.log({
            "eval/train_loss": losses['train'].item(),
            "eval/val_loss":   losses['val'].item(),
        }, step=iter)
        
        checkpoint_path = os.path.join(DATA_DIR, f'checkpoint_iter{iter}.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config.__dict__,
        }, checkpoint_path)
        print(f"Model checkpoint saved to {checkpoint_path}")

        if iter % (5 * eval_interval) == 0:
            # Push to W and B
            artifact = wandb.Artifact('model', type='checkpoint')
            artifact.add_file(checkpoint_path)
            run.log_artifact(artifact, aliases=[f"iter_{iter}"])

        train_losses.append(losses['train'].item())
        val_losses.append(losses['val'].item())
        iters.append(iter)

final_checkpoint_path = os.path.join(DATA_DIR, 'checkpoint.pt')
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config.__dict__,
}, final_checkpoint_path)
print(f"Final model checkpoint saved to {final_checkpoint_path}")
run.finish()

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