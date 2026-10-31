"""Stage 1.4 experiments: learning rate sweep and forgetting to zero gradients.

Run: python 01_micrograd/experiments.py
Saves loss curves to experiments.png. Findings are written up in NOTES.md.
"""
import os

import matplotlib.pyplot as plt

from train_moons import train

STEPS = 100

runs = {}
for lr in [0.01, 0.1, 1.0, 10.0]:
    print(f"--- lr={lr}")
    _, _, _, runs[f"lr={lr}"] = train(lr=lr, steps=STEPS, verbose=False)
print("--- lr=1.0, no zero_grad")
_, _, _, runs["lr=1.0, no zero_grad"] = train(lr=1.0, steps=STEPS, zero_grad=False, verbose=False)

print(f"\n{'run':<22} {'final loss':>12} {'final acc':>10}")
for name, history in runs.items():
    loss, acc = history[-1]
    print(f"{name:<22} {loss:>12.4f} {acc:>10.0%}")

fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))
for name, history in runs.items():
    ax_loss.plot([h[0] for h in history], label=name)
    ax_acc.plot([h[1] for h in history], label=name)
ax_loss.set(title="loss", xlabel="step", yscale="log")
ax_acc.set(title="accuracy", xlabel="step", ylim=(0, 1.05))
ax_loss.legend()
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments.png")
fig.savefig(path, dpi=120, bbox_inches="tight")
print(f"\nsaved {path}")
