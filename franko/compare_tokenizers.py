"""Compare our BPE with GPT-4's tokenizers on the same text.

  python franko/compare_tokenizers.py [--tokenizer data/bpe4000.json]

Needs `pip install tiktoken` (downloads the GPT vocabularies on first use).
"""
import argparse
import os

import tiktoken

from bpe import BPETokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = "Сумно і непривітно тепер в нашій Тухольщині! Правда, і Стрий, і Опір однаково миють її рінисті, зелені узберіжжя."
SHAKESPEARE = os.path.join(HERE, "..", "gpt", "data", "input.txt")


def show(name, pieces):
    print(f"  {name:<16} {len(pieces):3d} tokens: " + "|".join(pieces))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default=os.path.join(HERE, "data", "bpe4000.json"))
    args = parser.parse_args()

    ours = BPETokenizer.load(args.tokenizer)
    gpt = {name: tiktoken.get_encoding(name) for name in ("cl100k_base", "o200k_base")}
    labels = {"cl100k_base": "GPT-4", "o200k_base": "GPT-4o"}

    print(f"Sample ({len(SAMPLE)} chars, {len(SAMPLE.encode())} bytes):")
    show(f"ours ({ours.vocab_size})", [ours.decode([i]) for i in ours.encode(SAMPLE)])
    for name, enc in gpt.items():
        show(f"{labels[name]} ({enc.n_vocab // 1000}k)",
             [enc.decode_single_token_bytes(i).decode("utf-8", errors="replace") for i in enc.encode(SAMPLE)])

    texts = {"Franko (uk)": open(os.path.join(HERE, "data", "franko.txt"), encoding="utf-8").read()[-1_000_000:]}
    if os.path.exists(SHAKESPEARE):
        texts["Shakespeare (en)"] = open(SHAKESPEARE, encoding="utf-8").read()[-1_000_000:]
    print("\nCharacters per token on 1M characters (more = cheaper):")
    print(f"  {'':18}" + "".join(f"{t:>18}" for t in texts))
    rows = [(f"ours ({ours.vocab_size})", ours.encode)] + [(labels[n], e.encode_ordinary) for n, e in gpt.items()]
    for label, encode in rows:
        print(f"  {label:<18}" + "".join(f"{len(t) / len(encode(t)):>18.2f}" for t in texts.values()))


if __name__ == "__main__":
    main()
