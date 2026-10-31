"""Learning rate search: one training run where the lr grows exponentially from
1e-3 to 1 over 1000 steps. Where the loss falls fastest and is still stable is
a good lr; where it blows up is too much.

Run: python 03_mlp/lr_finder.py   (saves lr_finder.png)
"""
import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from data import device, splits
from mlp import HERE, MLP

STEPS = 1000

stoi, itos, data = splits()
Xtr, Ytr = data["train"]
model = MLP()
params = model.parameters()
g = torch.Generator().manual_seed(42)

lre = torch.linspace(-3, 0, STEPS)  # lr exponents
lrs = 10**lre
losses = []
for k in range(STEPS):
    ix = torch.randint(0, Xtr.shape[0], (32,), generator=g).to(device)
    loss = F.cross_entropy(model(Xtr[ix]), Ytr[ix])
    for p in params:
        p.grad = None
    loss.backward()
    for p in params:
        p.data -= lrs[k].item() * p.grad
    losses.append(loss.item())

# Smooth the noisy minibatch losses with a moving average to read the curve.
window = 50
smooth = torch.tensor(losses).unfold(0, window, 1).mean(1)
x = lre[window - 1:]
print(f"lowest smoothed loss at lr = 10^{x[smooth.argmin()].item():.2f} = {10**x[smooth.argmin()].item():.3f}")

plt.figure(figsize=(8, 4.5))
plt.plot(lre, losses, alpha=0.3, label="minibatch loss")
plt.plot(x, smooth, label=f"moving average ({window})")
plt.xlabel("log10(learning rate)")
plt.ylabel("loss")
plt.legend()
path = os.path.join(HERE, "lr_finder.png")
plt.savefig(path, dpi=120, bbox_inches="tight")
print(f"saved {path}")
