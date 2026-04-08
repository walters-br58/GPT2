"""
EECE 4520 - Milestone 3 Part 1
Evaluation: compute perplexity on the test set and compare with
a HuggingFace GPT-2 baseline (as required by the project spec).

Usage:
    python evaluate.py
"""

import math
import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from model         import GPT2Like, GPTConfig
from dataset       import build_dataloaders
from model_factory import ModelFactory


# ─────────────────────────────────────────────────────────────────────────────
# Perplexity of our custom model
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_perplexity(model, loader: DataLoader, device: str) -> float:
    """
    Perplexity = exp( mean NLL over all tokens in the test set ).
    Lower is better. A random model over V tokens scores roughly V.
    """
    model.eval()
    total_loss = 0.0
    total_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total_loss    += loss.item()
        total_batches += 1

    mean_nll   = total_loss / max(total_batches, 1)
    perplexity = math.exp(mean_nll)
    return perplexity


# ─────────────────────────────────────────────────────────────────────────────
# Perplexity of HuggingFace GPT-2 baseline (same test data, word-level)
# ─────────────────────────────────────────────────────────────────────────────

def compute_hf_baseline_perplexity(device: str) -> float:
    """
    Evaluate the smallest pretrained HuggingFace GPT-2 (117 M params) on
    Wikitext-2 test set using stride = block_size // 2 to avoid edge effects.

    This serves as the baseline comparison required by the project spec.
    """
    try:
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        from datasets import load_dataset
        import torch

        print("\nLoading HuggingFace GPT-2 baseline...")
        hf_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        hf_model     = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        hf_model.eval()

        dataset  = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text     = "\n\n".join(s["text"] for s in dataset if s["text"].strip())
        encodings= hf_tokenizer(text, return_tensors="pt")

        block_size = 128
        stride     = block_size // 2
        seq_len    = encodings.input_ids.size(1)

        nlls = []
        for begin in range(0, seq_len - 1, stride):
            end          = min(begin + block_size, seq_len)
            input_ids    = encodings.input_ids[:, begin:end].to(device)
            target_start = max(begin, stride) - begin
            with torch.no_grad():
                out  = hf_model(input_ids, labels=input_ids)
                # Only count loss on non-overlapping target tokens
                logits  = out.logits[:, :-1, :]
                targets = input_ids[:, 1:]
                loss    = torch.nn.functional.cross_entropy(
                    logits[:, target_start:, :].reshape(-1, logits.size(-1)),
                    targets[:, target_start:].reshape(-1),
                    reduction="mean",
                )
            nlls.append(loss.item())

        baseline_ppl = math.exp(sum(nlls) / len(nlls))
        return baseline_ppl

    except Exception as e:
        print(f"  Could not compute HF baseline: {e}")
        return float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    device = (
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device: {device}")

    # Load saved model
    print("\nLoading trained model from 'gpt2_final.pt'...")
    checkpoint = torch.load("gpt2_final.pt", map_location=device)
    cfg        = checkpoint["config"]
    factory    = ModelFactory()
    model      = factory.create_model('standard').to(device)
    model.load_state_dict(checkpoint["model_state"])

    # Build test DataLoader
    _, _, test_loader = build_dataloaders(
        tokenizer_path = "bpe_tokenizer.json",
        block_size     = cfg.block_size,
        batch_size     = 32,
    )

    # Our model's perplexity
    our_ppl = compute_perplexity(model, test_loader, device)

    # HF GPT-2 baseline
    baseline_ppl = compute_hf_baseline_perplexity(device)

    # ── Report ───────────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("          PERPLEXITY RESULTS (Test Set)")
    print("="*50)
    print(f"  Our GPT-2-like model  : {our_ppl:>10.2f}")
    print(f"  HF GPT-2 baseline     : {baseline_ppl:>10.2f}")
    if not math.isnan(baseline_ppl):
        ratio = our_ppl / baseline_ppl
        print(f"  Ratio (ours/baseline) : {ratio:>10.2f}x")
        note = ("(Our small model trained from scratch on Wikitext-2 only — "
                "expected to be higher than the large pretrained baseline.)")
        print(f"\n  Note: {note}")
    print("="*50)

    # Save results
    with open("perplexity_results.txt", "w") as f:
        f.write("PERPLEXITY RESULTS\n")
        f.write(f"Our GPT-2-like model : {our_ppl:.2f}\n")
        f.write(f"HF GPT-2 baseline    : {baseline_ppl:.2f}\n")
    print("\nResults saved → 'perplexity_results.txt'")


if __name__ == "__main__":
    main()