"""3.4: train with 2-dimensional embeddings and plot every character on the plane.

Run: python 03_mlp/embed2d.py   (saves embeddings_2d.png)
"""
import os

import matplotlib.pyplot as plt

from data import splits
from mlp import HERE, MLP, train

stoi, itos, data = splits()
model = MLP(n_embd=2)
history = train(model, data, steps=100_000, eval_every=50_000)
print(f"val loss with 2-dim embeddings: {history[-1][2]:.4f}")

C = model.C.detach().cpu()
vowels = set("aeiou")
plt.figure(figsize=(7, 7))
for i in range(C.shape[0]):
    ch = itos[i]
    color = "tab:red" if ch in vowels else "tab:gray" if ch == "." else "tab:blue"
    plt.scatter(C[i, 0], C[i, 1], s=300, color=color, alpha=0.3)
    plt.text(C[i, 0], C[i, 1], ch, ha="center", va="center", fontsize=12)
plt.title("2-dim character embeddings (red: vowels)")
plt.grid(alpha=0.3)
path = os.path.join(HERE, "embeddings_2d.png")
plt.savefig(path, dpi=120, bbox_inches="tight")
print(f"saved {path}")
