"""Byte-level BPE tokenizer, written from scratch (after Karpathy's minbpe).

  python franko/bpe.py --vocab_size 4000   # train on data/franko.txt -> data/bpe4000.json

Tokens start as the 256 byte values of the UTF-8 text. Training repeatedly
finds the most frequent adjacent pair of tokens and merges it into a new
token, until the vocabulary has vocab_size entries. Encoding applies the
learned merges in the order they were learned; decoding concatenates the
bytes of each token.

Before counting pairs, text is split into chunks (words with their leading
space, numbers, punctuation, whitespace), as in GPT-2/4: merges never cross
a chunk boundary, so there is no token for "word. The".
"""
import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))

# Letters (with an apostrophe inside a word: м'ясо, п'ять), numbers, other
# symbols, whitespace. Each may carry one leading space.
SPLIT = re.compile(r" ?(?:[^\W\d_]|'(?=[^\W\d_]))+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+")


def merge(ids, pair, new_id):
    """Replace every occurrence of pair in ids with new_id."""
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, merges=None):
        self.merges = merges or {}  # (id, id) -> new id, in learning order
        self.vocab = self._build_vocab()
        self.cache = {}

    def _build_vocab(self):
        vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), idx in self.merges.items():
            vocab[idx] = vocab[a] + vocab[b]
        return vocab

    @property
    def vocab_size(self):
        return len(self.vocab)

    @classmethod
    def train(cls, text, vocab_size, verbose=False):
        """Learn vocab_size - 256 merges from text.

        Identical chunks are counted once with their frequency, and pair counts
        are kept up to date incrementally: a merge only touches the chunks that
        contain the merged pair, instead of recounting the whole corpus.
        """
        assert vocab_size >= 256
        chunk_freq = Counter(SPLIT.findall(text))
        words = [list(w.encode("utf-8")) for w in chunk_freq]
        freqs = list(chunk_freq.values())

        pair_counts = Counter()
        where = defaultdict(set)  # pair -> indices of words that contain it
        for wi, (w, f) in enumerate(zip(words, freqs)):
            for p in zip(w, w[1:]):
                pair_counts[p] += f
                where[p].add(wi)

        merges, t0 = {}, time.time()
        for new_id in range(256, vocab_size):
            if not pair_counts:
                break  # nothing left to merge: every chunk is a single token
            pair = max(pair_counts, key=lambda p: (pair_counts[p], -p[0], -p[1]))  # ties: smallest ids
            merges[pair] = new_id
            for wi in where.pop(pair):
                w, f = words[wi], freqs[wi]
                for p in zip(w, w[1:]):  # take the old pairs out...
                    pair_counts[p] -= f
                    if pair_counts[p] <= 0:
                        del pair_counts[p]
                w = words[wi] = merge(w, pair, new_id)
                for p in zip(w, w[1:]):  # ...and put the new ones in
                    pair_counts[p] += f
                    where[p].add(wi)
            if verbose and (new_id - 255) % 500 == 0:
                tok = cls(merges)
                print(f"merge {new_id - 255:5d}: {tok.vocab[new_id].decode('utf-8', errors='replace')!r:>16}"
                      f"  ({time.time() - t0:.0f} s)")
        return cls(merges)

    def _encode_chunk(self, chunk):
        if chunk in self.cache:
            return self.cache[chunk]
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # The pair learned earliest must be merged first, as in training.
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        self.cache[chunk] = ids
        return ids

    def encode(self, text):
        return [i for chunk in SPLIT.findall(text) for i in self._encode_chunk(chunk)]

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"merges": [[a, b, idx] for (a, b), idx in self.merges.items()]}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        return cls({(a, b): idx for a, b, idx in data["merges"]})


def make_tokenizer(meta):
    """(encode, decode) for a tokenizer described as in prepare.py's meta.json:
    {"type": "bpe", "merges": [[a, b, id], ...]} or {"type": "char", "chars": [...]}."""
    if meta["type"] == "char":
        stoi = {c: i for i, c in enumerate(meta["chars"])}
        return (lambda s: [stoi[c] for c in s]), (lambda ids: "".join(meta["chars"][i] for i in ids))
    tok = BPETokenizer({(a, b): i for a, b, i in meta["merges"]})
    return tok.encode, tok.decode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=4000)
    parser.add_argument("--corpus", default=os.path.join(HERE, "data", "franko.txt"))
    args = parser.parse_args()

    with open(args.corpus, encoding="utf-8") as f:
        text = f.read()
    t0 = time.time()
    tok = BPETokenizer.train(text, args.vocab_size, verbose=True)
    out = os.path.join(os.path.dirname(args.corpus), f"bpe{args.vocab_size}.json")
    tok.save(out)
    print(f"trained {tok.vocab_size} tokens in {time.time() - t0:.0f} s -> {out}")

    n = len(tok.encode(text))
    print(f"corpus: {len(text):,} chars, {len(text.encode()):,} bytes -> {n:,} tokens "
          f"({len(text) / n:.2f} chars/token)")
    longest = sorted(tok.vocab.values(), key=len)[-15:]
    print("longest tokens:", ", ".join(repr(t.decode("utf-8", errors="replace")) for t in longest))


if __name__ == "__main__":
    main()
