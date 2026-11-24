"""Generate text with a model trained by 05_custom/train.py.

  python 05_custom/generate.py --data bpe4000 --prompt "— Ні," --chars 600
  python 05_custom/generate.py --data char --seed 7

Generation stops after --chars characters, not tokens, so a BPE model and a
character model can be compared on the same amount of text.
"""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "04_gpt"))
from model import GPT, GPTConfig  # noqa: E402

from bpe import make_tokenizer  # noqa: E402


def load(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, *make_tokenizer(ckpt["tokenizer"])


def generate(model, encode, decode, prompt, n_chars, temperature=0.8, seed=None):
    generator = torch.Generator()
    generator.manual_seed(seed) if seed is not None else generator.seed()
    device = next(model.parameters()).device
    ids = encode(prompt)
    new = []
    # Every token is at least one character, so n_chars tokens always suffice.
    for token in model.generate_stream(torch.tensor([ids], device=device), n_chars,
                                       temperature=temperature, generator=generator):
        new.append(token)
        if len(decode(new)) >= n_chars:
            break
    return prompt + decode(new)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="bpe4000", help="run name: out/<data>/best.pt")
    parser.add_argument("--prompt", default="\n")
    parser.add_argument("--chars", type=int, default=600)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    model, encode, decode = load(os.path.join(HERE, "out", args.data, "best.pt"), args.device)
    print(generate(model, encode, decode, args.prompt, args.chars, args.temperature, args.seed))


if __name__ == "__main__":
    main()
