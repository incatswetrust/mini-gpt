"""Deeper MLP built from own layers (layers.py) with BatchNorm.

Embedding -> Flatten -> n_layers x [Linear -> BatchNorm -> Tanh] -> Linear -> BatchNorm

Run: python 03_mlp/mlp_bn.py   (saves loss_curves_bn.png)
"""
import os

import torch

from data import device, splits
from layers import BatchNorm1d, Embedding, Flatten, Linear, Sequential, Tanh
from mlp import HERE, evaluate, generate, plot_history, train


def build_model(vocab_size=27, block_size=3, n_embd=10, n_hidden=200, n_layers=2,
                batchnorm=True, tanh_gain=5 / 3, seed=2147483647):
    """batchnorm=False and tanh_gain are there for the comparison in diagnostics.py."""
    g = torch.Generator().manual_seed(seed)
    layers = [Embedding(vocab_size, n_embd, generator=g), Flatten()]
    fan_in = n_embd * block_size
    for _ in range(n_layers):
        # The bias is useless before BatchNorm: BN subtracts the mean and has its own beta.
        lin = Linear(fan_in, n_hidden, bias=not batchnorm, generator=g)
        lin.weight *= tanh_gain  # tanh squashes, the gain compensates to keep activations ~unit scale
        layers += [lin, BatchNorm1d(n_hidden), Tanh()] if batchnorm else [lin, Tanh()]
        fan_in = n_hidden
    out = Linear(fan_in, vocab_size, bias=not batchnorm, generator=g)
    if batchnorm:
        layers += [out, BatchNorm1d(vocab_size)]
        layers[-1].gamma *= 0.1  # small logits at init -> loss ~ -ln(1/27)
    else:
        layers.append(out)
        out.weight *= 0.1
    model = Sequential(layers).to(device)
    for p in model.parameters():
        p.requires_grad_()
    return model


if __name__ == "__main__":
    stoi, itos, data = splits(block_size=3)
    model = build_model()
    print(f"device {device}; parameters: {sum(p.nelement() for p in model.parameters())}\n")

    history = train(model, data)
    print(f"\nfinal  train {history[-1][1]:.4f}  val {history[-1][2]:.4f}  test {evaluate(model, *data['test']):.4f}")
    plot_history(history, os.path.join(HERE, "loss_curves_bn.png"), "MLP + BatchNorm: emb 10, 2 x hidden 200")
    print("\nsamples:", ", ".join(generate(model, itos, 3)))
