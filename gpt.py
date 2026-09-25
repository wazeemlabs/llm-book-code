"""A small, readable GPT (decoder-only Transformer), in PyTorch.

This is the model the course builds toward. Notebook 08 walks through every class here;
notebooks 09 and 10 import it to train and sample. Architecture follows GPT-2 / nanoGPT:
token + positional embeddings -> N pre-norm Transformer blocks -> final LayerNorm -> LM head.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 65      # number of distinct tokens
    block_size: int = 128     # maximum context length (tokens the model can look back over)
    n_layer: int = 4          # number of Transformer blocks
    n_head: int = 4           # attention heads per block
    n_embd: int = 128         # embedding / residual-stream width
    dropout: float = 0.0      # dropout probability (set >0 to regularize larger runs)


class Head(nn.Module):
    """One head of causal self-attention (notebook 05, now with learned, trainable weights)."""

    def __init__(self, cfg: GPTConfig, head_size: int):
        super().__init__()
        self.key = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.query = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.value = nn.Linear(cfg.n_embd, head_size, bias=False)
        # a constant lower-triangular mask; registered as a buffer so it moves with .to(device)
        self.register_buffer("tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size)))
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)            # (B, T, head_size)
        q = self.query(x)          # (B, T, head_size)
        # scaled dot-product affinities, then mask out the future
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5   # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)          # (B, T, head_size)
        return wei @ v             # (B, T, head_size)


class MultiHeadAttention(nn.Module):
    """Several attention heads in parallel, concatenated and projected (notebook 06)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        head_size = cfg.n_embd // cfg.n_head
        self.heads = nn.ModuleList([Head(cfg, head_size) for _ in range(cfg.n_head)])
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)   # (B, T, n_embd)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Per-token MLP: expand 4x, nonlinearity, project back. Where most parameters live."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communicate (attention), then compute (MLP). Pre-norm + residuals."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.sa = MultiHeadAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.ffwd = FeedForward(cfg)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))     # residual connection around attention
        x = x + self.ffwd(self.ln2(x))   # residual connection around the feed-forward
        return x


class GPT(nn.Module):
    """The full model: embeddings -> stacked blocks -> final norm -> next-token logits."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.position_embedding = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_embedding(idx)                                  # (B, T, n_embd)
        pos = self.position_embedding(torch.arange(T, device=idx.device))  # (T, n_embd)
        x = tok + pos                                                    # add position info
        x = self.blocks(x)                                              # (B, T, n_embd)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                        # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressively sample max_new_tokens, one token at a time."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]      # never look back further than block_size
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature       # focus on the last position
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
