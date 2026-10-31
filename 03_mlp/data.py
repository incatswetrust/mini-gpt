"""names.txt -> (context of block_size chars, next char) examples, split 80/10/10."""
import os
import random

import torch

NAMES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "02_bigram", "names.txt")
device = "mps" if torch.backends.mps.is_available() else "cpu"


def load_words(path=NAMES_PATH):
    with open(path) as f:
        return f.read().splitlines()


def build_vocab(words):
    chars = sorted(set("".join(words)))
    stoi = {ch: i + 1 for i, ch in enumerate(chars)}
    stoi["."] = 0
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def build_dataset(words, stoi, block_size):
    """'emma', block_size=3 -> '...'->e, '..e'->m, '.em'->m, 'emm'->a, 'mma'->."""
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)


def splits(block_size=3, seed=42):
    """Shuffle names (not examples, so one name never lands in two splits),
    then split 80 / 10 / 10. Returns stoi, itos and {name: (X, Y)} on device."""
    words = load_words()
    stoi, itos = build_vocab(words)
    random.Random(seed).shuffle(words)
    n1, n2 = int(0.8 * len(words)), int(0.9 * len(words))
    parts = {"train": words[:n1], "val": words[n1:n2], "test": words[n2:]}
    data = {}
    for name, ws in parts.items():
        X, Y = build_dataset(ws, stoi, block_size)
        data[name] = (X.to(device), Y.to(device))
    return stoi, itos, data


def sample_names(next_probs, itos, block_size, n, generator):
    """next_probs(context (1, block_size) tensor) -> (1, vocab) probabilities."""
    names = []
    for _ in range(n):
        out, context = [], [0] * block_size
        while True:
            probs = next_probs(torch.tensor([context], device=device)).cpu()
            ix = torch.multinomial(probs, num_samples=1, generator=generator).item()
            if ix == 0:
                break
            out.append(itos[ix])
            context = context[1:] + [ix]
        names.append("".join(out))
    return names
