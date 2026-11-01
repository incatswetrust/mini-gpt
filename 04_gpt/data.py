"""Tiny Shakespeare (~1.1M chars): char tokenizer, train/val split, random batches."""
import os
import urllib.request

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "input.txt")
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_text(path=DATA_PATH):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, path)
    with open(path, encoding="utf-8") as f:
        return f.read()


class CharTokenizer:
    def __init__(self, chars):
        self.chars = list(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = dict(enumerate(self.chars))

    @classmethod
    def from_text(cls, text):
        return cls(sorted(set(text)))

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


def load_splits(val_fraction=0.1):
    """Returns tokenizer, train and val tensors of token ids (first 90% / last 10%)."""
    text = load_text()
    tok = CharTokenizer.from_text(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int((1 - val_fraction) * len(data))
    return tok, data[:n], data[n:]


def get_batch(data, block_size, batch_size, device, generator=None):
    """Random chunks of length block_size (x) and the same chunks shifted by one (y):
    y[b, t] is the token that follows x[b, :t+1]."""
    ix = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)
