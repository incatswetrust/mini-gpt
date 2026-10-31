"""Train an MLP on two moons with our own autograd and plot the decision boundary.

Run: python 01_micrograd/train_moons.py
"""
import random

import matplotlib.pyplot as plt
import numpy as np

from data import make_moons
from engine import Value
from nn import MLP


def loss_fn(model, X, y, alpha=1e-4):
    """SVM max-margin (hinge) loss + L2 regularization, labels are -1 / +1."""
    scores = [model([Value(x0), Value(x1)]) for x0, x1 in X]
    losses = [(1 + -yi * si).relu() for yi, si in zip(y, scores)]
    data_loss = sum(losses) * (1.0 / len(losses))
    reg_loss = alpha * sum(p * p for p in model.parameters())
    accuracy = sum((yi > 0) == (si.data > 0) for yi, si in zip(y, scores)) / len(y)
    return data_loss + reg_loss, accuracy


def train(lr=1.0, steps=100, zero_grad=True, seed=1337, verbose=True):
    """Returns the model and a history of (loss, accuracy) per step.

    lr decays linearly from lr to 0.1*lr over training.
    """
    random.seed(seed)
    X, y = make_moons(n_samples=100, noise=0.1)
    y = y * 2 - 1  # {0, 1} -> {-1, +1}
    model = MLP(2, [16, 16, 1])

    history = []
    for k in range(steps):
        loss, acc = loss_fn(model, X, y)  # forward
        if zero_grad:
            model.zero_grad()
        loss.backward()  # backward
        step_lr = lr * (1.0 - 0.9 * k / steps)
        for p in model.parameters():  # gradient descent step
            p.data -= step_lr * p.grad
        history.append((loss.data, acc))
        if verbose and (k % 10 == 0 or k == steps - 1):
            print(f"step {k:3d}  loss {loss.data:.4f}  accuracy {acc:.0%}")
    return model, X, y, history


def plot_decision_boundary(model, X, y, path):
    h = 0.05
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    scores = [model([Value(a), Value(b)]).data for a, b in zip(xx.ravel(), yy.ravel())]
    Z = (np.array(scores) > 0).reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral, edgecolors="k")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("MLP(2, [16, 16, 1]) on two moons")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    print(f"saved {path}")


if __name__ == "__main__":
    import os

    model, X, y, history = train()
    print(f"\nparameters: {len(model.parameters())}")
    print(f"final accuracy: {history[-1][1]:.0%}")
    out_dir = os.path.dirname(os.path.abspath(__file__))
    plot_decision_boundary(model, X, y, os.path.join(out_dir, "decision_boundary.png"))
