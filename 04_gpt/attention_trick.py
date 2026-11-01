"""4.3: the "mathematical trick" behind self-attention, step by step.

Goal: each token t gets the average of tokens 0..t (never the future).
Version 1 loops, version 2 multiplies by a lower-triangular matrix, version 3
builds that matrix with softmax over a mask — which is exactly what attention
does, except the weights come from data (q . k) instead of being uniform.

Run: python 04_gpt/attention_trick.py
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)
B, T, C = 4, 8, 2
x = torch.randn(B, T, C)

# Version 1: explicit loops. xbow = "bag of words", the mean of all previous tokens.
xbow = torch.zeros(B, T, C)
for b in range(B):
    for t in range(T):
        xbow[b, t] = x[b, : t + 1].mean(0)

# Version 2: one matrix multiply. Row t of wei has 1/(t+1) in columns 0..t.
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x  # (T, T) @ (B, T, C) -> (B, T, C)
assert torch.allclose(xbow, xbow2, atol=1e-6)

# Version 3: softmax. Start from affinities (here all 0), set the future to -inf,
# softmax per row -> the same averaging matrix. exp(-inf) = 0: no weight on the future.
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros(T, T)
wei = wei.masked_fill(tril == 0, float("-inf"))
wei = F.softmax(wei, dim=-1)
xbow3 = wei @ x
assert torch.allclose(xbow, xbow3, atol=1e-6)
print("loops == lower-triangular matmul == masked softmax: OK")
print("uniform weights:\n", wei.round(decimals=2))

# Version 4: self-attention. The affinities are no longer 0: each token emits a
# query and a key, and wei[t, s] = q_t . k_s. Tokens decide whom to listen to.
B, T, C, head_size = 4, 8, 32, 16
x = torch.randn(B, T, C)
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k, q = key(x), query(x)
wei = q @ k.transpose(-2, -1) / math.sqrt(head_size)  # (B, T, T)
wei = wei.masked_fill(torch.tril(torch.ones(T, T)) == 0, float("-inf"))
wei = F.softmax(wei, dim=-1)
out = wei @ value(x)
print("\ndata-dependent weights, first sequence:\n", wei[0].detach().round(decimals=2))
print("out:", tuple(out.shape))

# Why divide by sqrt(head_size): q and k have unit variance, so q . k (a sum of
# head_size products) has variance head_size. Large values make softmax peaky —
# at init every token would attend to just one other token.
q, k = torch.randn(B, T, head_size), torch.randn(B, T, head_size)
raw = q @ k.transpose(-2, -1)
scaled = raw / math.sqrt(head_size)
print(f"\nvar(q.k) = {raw.var():.1f} (head_size {head_size}), scaled: {scaled.var():.2f}")
logits = torch.tensor([0.1, -0.2, 0.3, -0.2, 0.5])
print("softmax(x):     ", [round(p, 2) for p in F.softmax(logits, -1).tolist()])
print("softmax(x * 8): ", [round(p, 2) for p in F.softmax(logits * 8, -1).tolist()], "<- nearly one-hot")
