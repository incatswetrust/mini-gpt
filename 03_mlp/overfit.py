"""3.2: what overfitting looks like. A big MLP trained on only 500 names:
train loss keeps falling, val loss turns around and grows.

Run: python 03_mlp/overfit.py   (saves overfit.png)
"""
import os

from data import splits
from mlp import HERE, MLP, plot_history, train

stoi, itos, data = splits()
X, Y = data["train"]
# Examples of the first 500 names: every name ends with a '.' target.
end = (Y == 0).nonzero()[499].item() + 1
small = dict(data, train=(X[:end], Y[:end]))
print(f"training on {end} examples (500 names) instead of {X.shape[0]}")

model = MLP(n_embd=10, n_hidden=300)
history = train(model, small, steps=30_000, lr=0.1, eval_every=1000, verbose=False)
best = min(history, key=lambda h: h[2])
print(f"best val {best[2]:.4f} at step {best[0]} (train {best[1]:.4f})")
print(f"final   train {history[-1][1]:.4f}  val {history[-1][2]:.4f}")
plot_history(history, os.path.join(HERE, "overfit.png"), "Overfitting: hidden 300 on 500 names")
