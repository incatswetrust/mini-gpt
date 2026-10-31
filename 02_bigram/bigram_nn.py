"""The same bigram model as a neural network: one-hot -> one linear layer -> softmax.

The 27x27 weight matrix W plays the role of the log-counts table. Trained with
cross-entropy, it converges to (almost) the same probabilities as counting.

Run: python 02_bigram/bigram_nn.py
"""
import torch
import torch.nn.functional as F

from bigram_counts import count_table, nll, probabilities
from data import bigram_tensors, build_vocab, load_words, sample_names

device = "mps" if torch.backends.mps.is_available() else "cpu"


def forward(W, xs):
    xenc = F.one_hot(xs, num_classes=W.shape[0]).float()  # (n, 27)
    logits = xenc @ W  # log-counts, (n, 27)
    # softmax: exp makes everything positive ("counts"), dividing by the row sum
    # turns counts into probabilities
    counts = logits.exp()
    return counts / counts.sum(dim=1, keepdim=True)


def loss_fn(W, xs, ys, reg=0.001):
    probs = forward(W, xs)
    # cross-entropy: -log of the probability the model gave to the right answer.
    # reg pulls W toward 0 (uniform), the counterpart of add-one smoothing.
    return -probs[torch.arange(len(ys), device=ys.device), ys].log().mean() + reg * (W**2).mean()


def train(xs, ys, vocab_size, steps=1000, lr=50.0, reg=0.001, seed=2147483647, verbose=True):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn((vocab_size, vocab_size), generator=g).to(device).requires_grad_()
    xs, ys = xs.to(device), ys.to(device)
    for k in range(steps):
        loss = loss_fn(W, xs, ys, reg)
        W.grad = None
        loss.backward()
        W.data -= lr * W.grad
        if verbose and (k % 100 == 0 or k == steps - 1):
            print(f"step {k:3d}  loss {loss.item():.4f}")
    return W.detach()


if __name__ == "__main__":
    words = load_words()
    stoi, itos = build_vocab(words)
    xs, ys = bigram_tensors(words, stoi)
    print(f"device: {device}, {len(xs)} bigrams\n")

    W = train(xs, ys, len(stoi))
    # Feed every char once: row i is the predicted distribution after char i.
    P_nn = forward(W, torch.arange(len(stoi), device=device)).cpu()
    P_counts = probabilities(count_table(xs, ys, len(stoi)))

    print(f"\nNLL neural net: {nll(P_nn, xs, ys):.4f}")
    print(f"NLL counts:     {nll(P_counts, xs, ys):.4f}")
    print(f"max |P_nn - P_counts|: {(P_nn - P_counts).abs().max().item():.4f}")

    g = torch.Generator().manual_seed(2147483647)
    print("\nsamples (nn):    ", ", ".join(sample_names(P_nn, itos, 10, g)))
    g = torch.Generator().manual_seed(2147483647)
    print("samples (counts):", ", ".join(sample_names(P_counts, itos, 10, g)))
