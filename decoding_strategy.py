from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F

class DecodingStrategy(ABC):
    @abstractmethod
    def select_token(self, logits: torch.Tensor) -> torch.Tensor: ...

class GreedyStrategy(DecodingStrategy):
    def select_token(self, logits):
        return logits.argmax(dim=-1, keepdim=True)

class TopKStrategy(DecodingStrategy):
    def __init__(self, k=50):
        self.k = k
    def select_token(self, logits):
        topk_vals, _ = torch.topk(logits, self.k, dim=-1)
        threshold = topk_vals[:, -1].unsqueeze(-1)
        filtered = logits.masked_fill(logits < threshold, float("-inf"))
        return torch.multinomial(F.softmax(filtered, dim=-1), num_samples=1)

class NucleusStrategy(DecodingStrategy):
    def __init__(self, p=0.9):
        self.p = p
    def select_token(self, logits):
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(sorted_probs, dim=-1)
        remove_mask = cumprobs - sorted_probs > self.p
        sorted_logits[remove_mask] = float("-inf")
        filtered = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)
        return torch.multinomial(F.softmax(filtered, dim=-1), num_samples=1)
