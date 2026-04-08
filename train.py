"""
EECE 4520 - Milestone 3 Part 1
Training loop: AdamW optimizer, cosine LR schedule, gradient clipping,
periodic loss logging, model checkpointing, and learning-curve plotting.

Usage:
    python train.py
"""

import os
import math
import time
import json
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from dataset import build_dataloaders
from model_factory import ModelFactory


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters — edit these to match your hardware
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = dict(
    # Model
    vocab_size  = 8000,
    block_size  = 128,
    d_model     = 256,
    n_layers    = 4,
    n_heads     = 4,
    dropout     = 0.1,

    # Training
    batch_size  = 32,       # lower to 8 or 16 if you run out of GPU memory
    epochs      = 3,
    lr          = 3e-4,
    weight_decay= 0.1,
    grad_clip   = 1.0,
    warmup_steps= 100,      # linear warm-up before cosine decay

    # Logging / saving
    log_interval  = 50,     # print loss every N gradient steps
    ckpt_interval = 200,    # save checkpoint every N gradient steps
    ckpt_dir      = "checkpoints",
    tokenizer_path= "bpe_tokenizer.json",
)


# ─────────────────────────────────────────────────────────────────────────────
# Learning-rate schedule: linear warm-up → cosine decay
# ─────────────────────────────────────────────────────────────────────────────

def get_lr(step: int, total_steps: int, warmup_steps: int, max_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * step / max_lr(warmup_steps, 1)
    if step >= total_steps:
        return max_lr * 0.1                              # floor at 10 % of peak
    # cosine decay
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return max_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device) -> float:
    """Return mean cross-entropy loss over a DataLoader."""
    model.eval()
    total, count = 0.0, 0
    for x, y in loader:
        x, y      = x.to(device), y.to(device)
        _, loss   = model(x, y)
        total    += loss.item()
        count    += 1
    model.train()
    return total / max(count, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Learning-curve plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_learning_curve(
    train_records: list[tuple[int, float]],   # (global_step, loss)
    val_records:   list[tuple[int, float]],   # (global_step, loss)
    save_path: str = "learning_curve.png",
):
    """
    Plot training and validation loss against gradient-update steps.
    Both step-level train loss and epoch-level validation loss are shown.
    """
    t_steps, t_losses = zip(*train_records)
    v_steps, v_losses = zip(*val_records)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(t_steps, t_losses, color="#2563EB", linewidth=1.5,
            alpha=0.8, label="Train loss (per log interval)")
    ax.plot(v_steps, v_losses, color="#DC2626", linewidth=2.0,
            marker="o", markersize=7, linestyle="--",
            label="Validation loss (per epoch)")

    ax.set_xlabel("Gradient update steps", fontsize=13)
    ax.set_ylabel("Cross-Entropy Loss",    fontsize=13)
    ax.set_title("GPT-2-like Transformer – Learning Curve",
                 fontsize=15, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nLearning curve saved → '{save_path}'")


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train():
    # ── Device ───────────────────────────────────────────────────────────────
    device = (
        "cuda"  if torch.cuda.is_available() else
        "mps"   if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Using device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = build_dataloaders(
        tokenizer_path = CONFIG["tokenizer_path"],
        block_size     = CONFIG["block_size"],
        batch_size     = CONFIG["batch_size"],
    )

    # ── Model ────────────────────────────────────────────────────────────────
    factory = ModelFactory()
    model   = factory.create_model('standard').to(device)
    cfg     = model.cfg

    # ── Optimizer ────────────────────────────────────────────────────────────
    # Separate weight-decayed params (weights) from non-decayed (biases, LN)
    decay_params   = [p for n, p in model.named_parameters()
                      if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and p.dim() < 2]
    optimiser = torch.optim.AdamW(
        [
            {"params": decay_params,   "weight_decay": CONFIG["weight_decay"]},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=CONFIG["lr"],
        betas=(0.9, 0.95),
    )

    # ── LR schedule ──────────────────────────────────────────────────────────
    total_steps  = CONFIG["epochs"] * len(train_loader)
    warmup_steps = CONFIG["warmup_steps"]

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)

    # ── Checkpoint directory ─────────────────────────────────────────────────
    os.makedirs(CONFIG["ckpt_dir"], exist_ok=True)

    # ── Training ─────────────────────────────────────────────────────────────
    train_records: list[tuple[int, float]] = []
    val_records:   list[tuple[int, float]] = []
    global_step = 0

    print("\n" + "="*60)
    print("Starting training")
    print("="*60)

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        epoch_loss, n = 0.0, 0
        t0 = time.time()

        for batch_idx, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device), y.to(device)

            # Forward + backward
            _, loss = model(x, y)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            optimiser.step()
            scheduler.step()

            epoch_loss  += loss.item()
            n           += 1
            global_step += 1

            # ── Log training loss ─────────────────────────────────────────
            if global_step % CONFIG["log_interval"] == 0:
                avg = epoch_loss / n
                lr  = scheduler.get_last_lr()[0]
                print(f"  epoch {epoch} | step {global_step:>6} "
                      f"| train_loss {avg:.4f} | lr {lr:.2e}")
                train_records.append((global_step, avg))

            # ── Checkpoint ───────────────────────────────────────────────
            if global_step % CONFIG["ckpt_interval"] == 0:
                ckpt_path = os.path.join(
                    CONFIG["ckpt_dir"], f"ckpt_step{global_step}.pt"
                )
                torch.save({
                    "step":        global_step,
                    "model_state": model.state_dict(),
                    "optim_state": optimiser.state_dict(),
                    "config":      cfg,
                    "train_loss":  epoch_loss / n,
                }, ckpt_path)
                print(f"  ✓ Checkpoint saved → {ckpt_path}")

        # ── End-of-epoch validation ───────────────────────────────────────
        val_loss = evaluate(model, val_loader, device)
        elapsed  = time.time() - t0
        print(f"\nEpoch {epoch}/{CONFIG['epochs']} complete | "
              f"val_loss {val_loss:.4f} | {elapsed:.0f}s\n")
        val_records.append((global_step, val_loss))

    # ── Save final model ──────────────────────────────────────────────────────
    torch.save({
        "model_state": model.state_dict(),
        "config":      cfg,
    }, "gpt2_final.pt")
    print("Final model saved → 'gpt2_final.pt'")

    # Save config for reproducibility
    with open("train_config.json", "w") as f:
        json.dump(CONFIG, f, indent=2)

    # ── Plot learning curve ───────────────────────────────────────────────────
    plot_learning_curve(train_records, val_records, "learning_curve.png")


if __name__ == "__main__":
    train()