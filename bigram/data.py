"""names.txt and the character vocabulary.

names.txt: ~32k names, one per line, from karpathy/makemore (MIT license).
"""
import os

import torch

NAMES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "names.txt")


def load_words(path=NAMES_PATH):
    with open(path) as f:
        return f.read().splitlines()


def build_vocab(words):
    """stoi: char -> int, itos: int -> char. '.' (index 0) marks start and end of a name."""
    chars = sorted(set("".join(words)))
    stoi = {ch: i + 1 for i, ch in enumerate(chars)}
    stoi["."] = 0
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def bigrams(word):
    """'emma' -> ('.', 'e'), ('e', 'm'), ('m', 'm'), ('m', 'a'), ('a', '.')"""
    chs = ["."] + list(word) + ["."]
    return zip(chs, chs[1:])


def bigram_tensors(words, stoi):
    """Inputs xs (current char) and targets ys (next char) for every bigram."""
    xs, ys = [], []
    for w in words:
        for ch1, ch2 in bigrams(w):
            xs.append(stoi[ch1])
            ys.append(stoi[ch2])
    return torch.tensor(xs), torch.tensor(ys)


def sample_names(P, itos, n, generator):
    """P[i, j] = probability that char j follows char i. Sample until '.'."""
    names = []
    for _ in range(n):
        out, ix = [], 0
        while True:
            ix = torch.multinomial(P[ix], num_samples=1, generator=generator).item()
            if ix == 0:
                break
            out.append(itos[ix])
        names.append("".join(out))
    return names
