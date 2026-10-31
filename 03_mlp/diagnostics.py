"""3.3: why initialization and normalization matter.

Part 1, one-hidden-layer MLP: naive randn init vs scaled init.
  - initial loss vs the expected -ln(1/27) = 3.30
  - tanh saturation: share of hidden activations with |h| > 0.99, dead neurons
Part 2, deep net (5 hidden layers of 100) from own layers:
  no gain / tanh gain 5/3 / BatchNorm; activation stats per layer at init and
  loss after a short training run.

Run: python 03_mlp/diagnostics.py   (saves init_activations.png, deep_activations.png)
"""
import math
import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from data import device, splits
from layers import Tanh
from mlp import HERE, MLP, evaluate, train
from mlp_bn import build_model

stoi, itos, data = splits()
Xtr, Ytr = data["train"]
g = torch.Generator().manual_seed(0)
batch = torch.randint(0, Xtr.shape[0], (1000,), generator=g).to(device)
print(f"expected initial loss: -ln(1/27) = {-math.log(1 / 27):.4f}\n")

# --- Part 1 -----------------------------------------------------------------

print("Part 1: MLP(emb 10, hidden 200), naive vs scaled init")
fig, axes = plt.subplots(2, 2, figsize=(12, 7))
for col, scaled in enumerate([False, True]):
    model = MLP(scaled_init=scaled)
    with torch.no_grad():
        loss = F.cross_entropy(model(Xtr[batch]), Ytr[batch]).item()
        logits = model(Xtr[batch])
    h = model.h.cpu()
    saturated = (h.abs() > 0.99)
    dead = saturated.all(dim=0).sum().item()  # neuron saturated for every example: gradient ~0 always
    name = "scaled" if scaled else "naive"
    print(f"  {name:>6}: initial loss {loss:.4f}, logits std {logits.std().item():.2f}, "
          f"saturated {saturated.float().mean().item():.0%}, dead neurons {dead}/200")

    axes[0, col].hist(h.view(-1).tolist(), bins=50)
    axes[0, col].set_title(f"{name} init: tanh outputs (saturated {saturated.float().mean().item():.0%})")
    axes[1, col].imshow(saturated[:50].T, cmap="gray", interpolation="nearest", aspect="auto")
    axes[1, col].set_title(f"|h| > 0.99 (white), 200 neurons x 50 examples; dead: {dead}")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "init_activations.png"), dpi=100)

print("\n  training both for 30k steps:")
for scaled in [False, True]:
    model = MLP(scaled_init=scaled)
    history = train(model, data, steps=30_000, eval_every=30_000, verbose=False)
    print(f"  {'scaled' if scaled else 'naive':>6}: step 0 train {history[0][1]:.4f} -> val {history[-1][2]:.4f}")

# --- Part 2 -----------------------------------------------------------------

print("\nPart 2: deep net, 5 hidden layers of 100")
configs = {
    "no gain": dict(batchnorm=False, tanh_gain=1.0),
    "gain 5/3": dict(batchnorm=False, tanh_gain=5 / 3),
    "BatchNorm": dict(batchnorm=True),
}
fig, axes = plt.subplots(1, len(configs), figsize=(15, 4))
for ax, (name, cfg) in zip(axes, configs.items()):
    model = build_model(n_hidden=100, n_layers=5, **cfg)
    with torch.no_grad():
        model(Xtr[batch])
    print(f"  {name}:")
    for i, layer in enumerate(model.layers):
        if isinstance(layer, Tanh):
            t = layer.out.cpu()
            sat = (t.abs() > 0.97).float().mean().item()
            print(f"    layer {i:2d} tanh: mean {t.mean():+.2f}, std {t.std():.2f}, saturated {sat:.1%}")
            ax.hist(t.view(-1).tolist(), bins=50, histtype="step", label=f"layer {i}")
    history = train(model, data, steps=10_000, eval_every=10_000, verbose=False)
    print(f"    loss at init {history[0][1]:.4f}, val after 10k steps {history[-1][2]:.4f}")
    ax.set_title(f"{name}: tanh outputs per layer at init")
    ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "deep_activations.png"), dpi=100)
print("\nsaved init_activations.png, deep_activations.png")
