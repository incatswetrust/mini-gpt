"""A small GPT: decoder-only transformer, written out piece by piece.

tokens -> token embedding + position embedding
       -> n_layer x Block [ x + MultiHeadAttention(LayerNorm(x)); x + FeedForward(LayerNorm(x)) ]
       -> LayerNorm -> Linear -> logits over the vocabulary
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 256  # max context length
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2


class Head(nn.Module):
    """One head of causal self-attention.

    Every token emits a query ("what am I looking for"), a key ("what do I
    contain") and a value ("what I pass on if you attend to me"). The affinity
    of token t to token s is q_t . k_s; tokens may only look at the past.
    """

    def __init__(self, config, head_size):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        # Lower-triangular mask: position t sees positions 0..t. A buffer, not a parameter.
        self.register_buffer("tril", torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, hs)
        q = self.query(x)  # (B, T, hs)
        # Scale by 1/sqrt(hs): otherwise the variance of q.k grows with hs and
        # softmax turns into a one-hot argmax at init.
        wei = q @ k.transpose(-2, -1) / math.sqrt(k.shape[-1])  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # no peeking at the future
        wei = F.softmax(wei, dim=-1)  # each row: weights over the past, summing to 1
        wei = self.dropout(wei)
        v = self.value(x)  # (B, T, hs)
        return wei @ v  # weighted average of the values, (B, T, hs)


class MultiHeadAttention(nn.Module):
    """Several heads in parallel, each with its own smaller q/k/v; outputs concatenated
    and mixed back together by a linear projection."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        head_size = config.n_embd // config.n_head
        self.heads = nn.ModuleList([Head(config, head_size) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Per-token MLP: attention lets tokens exchange information, this lets each
    token think about what it gathered. Hidden size 4x as in the paper."""

    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Communication (attention) followed by computation (feed-forward).

    Residual connections: each sublayer adds to x instead of replacing it, so
    gradients flow straight through the `+` to early layers. Pre-norm: LayerNorm
    is applied to the sublayer input, keeping the residual stream untouched.
    """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffwd = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        # Attention itself is order-blind (a weighted sum over a set), so position
        # has to be added to every token explicitly.
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        # Small init (std 0.02, as in GPT-2): logits start near 0, loss near ln(vocab_size).
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size, f"sequence of {T} tokens, block_size is {self.config.block_size}"
        tok = self.token_embedding(idx)  # (B, T, C)
        pos = self.position_embedding(torch.arange(T, device=idx.device))  # (T, C)
        x = self.drop(tok + pos)
        x = self.blocks(x)
        logits = self.lm_head(self.ln_f(x))  # (B, T, vocab)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, generator=None):
        """Predict the next token, append it, repeat. idx: (B, T) context.

        temperature < 1 sharpens the distribution (safer, more repetitive),
        > 1 flattens it (more varied, more mistakes). top_k keeps only the k
        most likely tokens before sampling.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]  # position embeddings only go up to block_size
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # only the last position predicts the next token
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs.cpu(), num_samples=1, generator=generator).to(idx.device)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
