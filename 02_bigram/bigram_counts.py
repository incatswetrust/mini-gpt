"""Bigram model from counts: how often does char B follow char A.

Run: python 02_bigram/bigram_counts.py
"""
import os

import matplotlib.pyplot as plt
import torch

from data import bigram_tensors, build_vocab, load_words, sample_names


def count_table(xs, ys, vocab_size):
    N = torch.zeros((vocab_size, vocab_size), dtype=torch.int32)
    N.index_put_((xs, ys), torch.ones_like(xs, dtype=torch.int32), accumulate=True)
    return N


def probabilities(N, smoothing=1):
    """Normalize each row to sum to 1. Add-one smoothing avoids zero probabilities
    (log 0 = -inf), e.g. for 'jq', which never occurs in the data."""
    P = (N + smoothing).float()
    return P / P.sum(dim=1, keepdim=True)


def nll(P, xs, ys):
    """Average negative log likelihood of the data under the model."""
    return -P[xs, ys].log().mean().item()


def plot_counts(N, itos, path):
    V = N.shape[0]
    plt.figure(figsize=(16, 16))
    plt.imshow(N, cmap="Blues")
    for i in range(V):
        for j in range(V):
            color = "white" if N[i, j] > N.max() / 2 else "gray"
            plt.text(j, i, itos[i] + itos[j], ha="center", va="bottom", color=color)
            plt.text(j, i, N[i, j].item(), ha="center", va="top", color=color)
    plt.axis("off")
    plt.savefig(path, dpi=80, bbox_inches="tight")
    print(f"saved {path}")


if __name__ == "__main__":
    words = load_words()
    stoi, itos = build_vocab(words)
    xs, ys = bigram_tensors(words, stoi)
    print(f"{len(words)} names, {len(xs)} bigrams, vocab {len(stoi)}")

    N = count_table(xs, ys, len(stoi))
    V = len(stoi)
    top = torch.topk(N.flatten(), 5)
    print("most common bigrams:", [(itos[i // V] + itos[i % V], c) for c, i in zip(top.values.tolist(), top.indices.tolist())])

    P = probabilities(N)
    print(f"\nNLL, add-one smoothing: {nll(P, xs, ys):.4f}")
    print(f"NLL, no smoothing:      {nll(probabilities(N, smoothing=0), xs, ys):.4f}")
    print(f"NLL, uniform (1/27):    {-torch.log(torch.tensor(1 / 27)).item():.4f}")
    print(f"log-likelihood of 'andrejq': {sum(P[stoi[a], stoi[b]].log().item() for a, b in zip('.andrejq', 'andrejq.')):.2f}")

    g = torch.Generator().manual_seed(2147483647)
    print("\nsamples:", ", ".join(sample_names(P, itos, 15, g)))

    plot_counts(N, itos, os.path.join(os.path.dirname(os.path.abspath(__file__)), "bigram_counts.png"))
