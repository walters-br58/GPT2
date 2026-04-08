"""
EECE 4520 - Milestone 3 Part 1
Text generation using three decoding strategies:
  1. Greedy   – always picks the highest-probability token
  2. Top-k    – samples from the top-k tokens  (k = 50)
  3. Nucleus  – samples from the smallest set whose cumulative prob >= p  (p = 0.9)

Usage:
    python generate.py
"""

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model         import GPT2Like, GPTConfig
from model_factory import ModelFactory

# ─────────────────────────────────────────────────────────────────────────────
# Core generation function
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(
    model:           GPT2Like,
    tokenizer:       Tokenizer,
    prompt:          str,
    max_new_tokens:  int   = 150,
    strategy:        str   = "greedy",   # "greedy" | "top_k" | "top_p"
    temperature:     float = 1.0,
    top_k:           int   = 50,
    top_p:           float = 0.9,
    device:          str   = "cpu",
) -> str:
    """
    Autoregressively generate tokens from a text prompt.

    Args:
        model           : Trained GPT2Like model.
        tokenizer       : Trained BPE tokenizer.
        prompt          : Seed text string.
        max_new_tokens  : Number of new tokens to generate.
        strategy        : Decoding strategy — "greedy", "top_k", or "top_p".
        temperature     : Softens (>1) or sharpens (<1) the distribution.
        top_k           : For "top_k" strategy: number of candidates kept.
        top_p           : For "top_p" strategy: nucleus probability threshold.
        device          : Torch device string.

    Returns:
        Full decoded string (prompt + generated continuation).
    """
    model.eval()

    # Encode prompt → token ids
    ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    eos_id = tokenizer.token_to_id("[EOS]")

    for _ in range(max_new_tokens):
        # Crop context to block_size
        idx_cond  = idx[:, -model.cfg.block_size:]
        logits, _ = model(idx_cond)
        logits    = logits[:, -1, :] / temperature          # (1, V)

        # ── Greedy ───────────────────────────────────────────────────────────
        if strategy == "greedy":
            next_id = logits.argmax(dim=-1, keepdim=True)   # (1, 1)

        # ── Top-k sampling ───────────────────────────────────────────────────
        elif strategy == "top_k":
            # Zero out all logits below the k-th highest value
            topk_vals, _ = torch.topk(logits, top_k, dim=-1)
            threshold     = topk_vals[:, -1].unsqueeze(-1)
            logits        = logits.masked_fill(logits < threshold, float("-inf"))
            probs         = F.softmax(logits, dim=-1)
            next_id       = torch.multinomial(probs, num_samples=1)

        # ── Nucleus (top-p) sampling ─────────────────────────────────────────
        elif strategy == "top_p":
            # Sort descending, compute cumulative probability
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            sorted_probs  = F.softmax(sorted_logits, dim=-1)
            cumprobs      = torch.cumsum(sorted_probs, dim=-1)

            # Remove tokens once cumulative probability exceeds top_p
            # (keep at least 1 token even if it already exceeds threshold)
            remove_mask             = cumprobs - sorted_probs > top_p
            sorted_logits[remove_mask] = float("-inf")

            # Scatter filtered logits back to original ordering
            filtered_logits = torch.zeros_like(logits).scatter_(
                1, sorted_idx, sorted_logits
            )
            probs   = F.softmax(filtered_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        else:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Choose: greedy / top_k / top_p"
            )

        # Append predicted token
        idx = torch.cat([idx, next_id], dim=1)

        # Stop early if EOS is generated
        if next_id.item() == eos_id:
            break

    generated_ids = idx[0].tolist()
    return tokenizer.decode(generated_ids)


# ─────────────────────────────────────────────────────────────────────────────
# Main — run all three strategies on several prompts
# ─────────────────────────────────────────────────────────────────────────────

PROMPTS = [
    "The history of artificial intelligence began",
    "In the field of natural language processing",
    "Scientists recently discovered that",
    "The economic impact of climate change",
]

def main():
    device = (
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device: {device}")

    # Load model
    print("Loading trained model from 'gpt2_final.pt'...")
    checkpoint = torch.load("gpt2_final.pt", map_location=device)
    factory    = ModelFactory()
    model      = factory.create_model('standard').to(device)
    model.load_state_dict(checkpoint["model_state"])

    # Load tokenizer
    tokenizer = Tokenizer.from_file("bpe_tokenizer.json")

    # Output file
    results = []

    strategies = [
        dict(strategy="greedy", label="Greedy Decoding"),
        dict(strategy="top_k",  label="Top-k Sampling  (k=50)",  top_k=50,  temperature=0.8),
        dict(strategy="top_p",  label="Nucleus Sampling (p=0.9)", top_p=0.9, temperature=0.8),
    ]

    sep = "─" * 70

    for prompt in PROMPTS:
        header = f"\nPROMPT: \"{prompt}\""
        print("\n" + "="*70)
        print(header)
        print("="*70)
        results.append("\n" + "="*70)
        results.append(header)
        results.append("="*70)

        for s in strategies:
            label    = s.pop("label")
            text     = generate(
                model, tokenizer, prompt,
                max_new_tokens=120,
                device=device,
                **s,
            )
            s["label"] = label   # restore for next prompt

            block = f"\n[{label}]\n{text}\n{sep}"
            print(block)
            results.append(block)

    # Save to file
    with open("generated_samples.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(results))
    print("\nGenerated samples saved → 'generated_samples.txt'")


if __name__ == "__main__":
    main()