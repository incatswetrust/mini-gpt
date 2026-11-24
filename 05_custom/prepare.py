"""Tokenize the corpus once into binary files for training.

  python 05_custom/prepare.py --tokenizer bpe4000   # BPE from data/bpe4000.json
  python 05_custom/prepare.py --tokenizer char      # one token per character

Writes data/<tokenizer>/{train,val}.bin (numpy uint16 token ids) and
meta.json (tokenizer, token and character counts). Both tokenizations use
the same train/val text, so their results can be compared per character.
"""
import argparse
import json
import os

import numpy as np

from bpe import BPETokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CORPUS = os.path.join(DATA, "franko.txt")


def split_text(text, n_segments=100, val_every=10):
    """Cut the corpus into n_segments pieces at paragraph boundaries and send
    every val_every-th piece to val. Taking the last 10% instead would make val
    a handful of works of one kind (the corpus is ordered by work)."""
    cuts = [0]
    for k in range(1, n_segments):
        pos = text.find("\n\n", k * len(text) // n_segments)
        cuts.append(pos + 2 if pos != -1 else len(text))
    cuts.append(len(text))
    pieces = [text[a:b] for a, b in zip(cuts, cuts[1:])]
    train = "".join(p for i, p in enumerate(pieces) if i % val_every != val_every - 1)
    val = "".join(p for i, p in enumerate(pieces) if i % val_every == val_every - 1)
    return train, val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default="bpe4000", help="'char' or the name of data/<name>.json")
    args = parser.parse_args()

    with open(CORPUS, encoding="utf-8") as f:
        text = f.read()
    train_text, val_text = split_text(text)

    if args.tokenizer == "char":
        chars = sorted(set(text))
        stoi = {c: i for i, c in enumerate(chars)}
        encode = lambda s: [stoi[c] for c in s]  # noqa: E731
        meta = {"type": "char", "chars": chars, "vocab_size": len(chars)}
    else:
        tok = BPETokenizer.load(os.path.join(DATA, f"{args.tokenizer}.json"))
        encode = tok.encode
        meta = {"type": "bpe", "merges": [[a, b, i] for (a, b), i in tok.merges.items()], "vocab_size": tok.vocab_size}

    out_dir = os.path.join(DATA, args.tokenizer)
    os.makedirs(out_dir, exist_ok=True)
    for name, part in (("train", train_text), ("val", val_text)):
        ids = np.array(encode(part), dtype=np.uint16)
        assert ids.max() < 2**16
        ids.tofile(os.path.join(out_dir, f"{name}.bin"))
        meta[f"{name}_tokens"], meta[f"{name}_chars"] = len(ids), len(part)
        print(f"{name}: {len(part):,} chars -> {len(ids):,} tokens ({len(part) / len(ids):.2f} chars/token)")
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False)
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
