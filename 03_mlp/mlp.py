"""MLP language model (Bengio et al. 2003) on names.txt, written with raw tensors.

3 previous chars -> embedding table C (10 dims each) -> concat (30) ->
hidden layer (200, tanh) -> output layer (27) -> softmax.

Also holds the training loop shared by the other scripts of this stage: it works
with any model that has __call__(X) -> logits, parameters(), train() and eval().

Run: python 03_mlp/mlp.py   (saves loss_curves.png)
"""
import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from data import device, sample_names, splits

HERE = os.path.dirname(os.path.abspath(__file__))


class MLP:
    def __init__(self, vocab_size=27, block_size=3, n_embd=10, n_hidden=200, seed=2147483647, scaled_init=True):
        """scaled_init=False is the naive "randn everywhere" init, kept for comparison
        (see diagnostics.py and NOTES.md, 3.3)."""
        g = torch.Generator().manual_seed(seed)
        fan_in = n_embd * block_size
        self.C = torch.randn((vocab_size, n_embd), generator=g)
        self.W1 = torch.randn((fan_in, n_hidden), generator=g)
        self.b1 = torch.randn(n_hidden, generator=g)
        self.W2 = torch.randn((n_hidden, vocab_size), generator=g)
        self.b2 = torch.randn(vocab_size, generator=g)
        if scaled_init:
            self.W1 *= (5 / 3) / fan_in**0.5  # Kaiming init with tanh gain: pre-activations ~N(0, 1)
            self.b1 *= 0.01
            self.W2 *= 0.01  # near-zero logits -> uniform softmax -> initial loss ~ -ln(1/27)
            self.b2 *= 0
        for name in ("C", "W1", "b1", "W2", "b2"):
            setattr(self, name, getattr(self, name).to(device).requires_grad_())

    def __call__(self, X):
        emb = self.C[X]  # (batch, block_size, n_embd)
        self.hpreact = emb.view(emb.shape[0], -1) @ self.W1 + self.b1  # kept for diagnostics
        self.h = torch.tanh(self.hpreact)  # (batch, n_hidden)
        return self.h @ self.W2 + self.b2  # logits (batch, vocab)

    def parameters(self):
        return [self.C, self.W1, self.b1, self.W2, self.b2]

    def train(self, mode=True):  # no dropout / batchnorm: nothing changes between modes
        return self

    def eval(self):
        return self


@torch.no_grad()
def evaluate(model, X, Y):
    model.eval()
    loss = F.cross_entropy(model(X), Y).item()
    model.train()
    return loss


def lr_schedule(step, steps, lr):
    """Step decay: full lr for the first half, lr/10 for the second."""
    return lr if step < steps // 2 else lr / 10


def train(model, data, steps=200_000, batch_size=32, lr=0.1, eval_every=5000, seed=42, verbose=True):
    Xtr, Ytr = data["train"]
    params = model.parameters()
    g = torch.Generator().manual_seed(seed)
    history = []  # (step, train loss, val loss)
    t0 = time.time()
    model.train()
    for k in range(steps):
        ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g).to(device)
        loss = F.cross_entropy(model(Xtr[ix]), Ytr[ix])
        for p in params:
            p.grad = None
        loss.backward()
        step_lr = lr_schedule(k, steps, lr)
        for p in params:
            p.data -= step_lr * p.grad

        if k % eval_every == 0 or k == steps - 1:
            # Loss on the full splits: the minibatch loss is too noisy to compare.
            tr, va = evaluate(model, *data["train"]), evaluate(model, *data["val"])
            history.append((k, tr, va))
            if verbose:
                print(f"step {k:6d}  lr {step_lr:.3f}  train {tr:.4f}  val {va:.4f}  ({time.time() - t0:.0f}s)")
    return history


def plot_history(history, path, title):
    steps, tr, va = zip(*history)
    plt.figure(figsize=(8, 4.5))
    plt.plot(steps, tr, label="train")
    plt.plot(steps, va, label="val")
    plt.axhline(2.4546, color="gray", linestyle="--", linewidth=1, label="bigram (2.45)")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title(title)
    plt.legend()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    print(f"saved {path}")


def generate(model, itos, block_size, n=20, seed=2147483657):
    model.eval()
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        return sample_names(lambda ctx: F.softmax(model(ctx), dim=1), itos, block_size, n, g)


if __name__ == "__main__":
    stoi, itos, data = splits(block_size=3)
    print(f"device {device}; examples: " + ", ".join(f"{k} {v[0].shape[0]}" for k, v in data.items()))
    model = MLP()
    print(f"parameters: {sum(p.nelement() for p in model.parameters())}\n")

    history = train(model, data)
    print(f"\nfinal  train {history[-1][1]:.4f}  val {history[-1][2]:.4f}  test {evaluate(model, *data['test']):.4f}")
    plot_history(history, os.path.join(HERE, "loss_curves.png"), "MLP: emb 10, hidden 200, block_size 3")
    print("\nsamples:", ", ".join(generate(model, itos, 3)))
