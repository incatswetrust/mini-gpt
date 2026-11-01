"""Generate text with a trained GPT.

  python 04_gpt/generate.py --prompt "ROMEO:" --temperature 0.8 --max_tokens 500
"""
import argparse
import os

import torch

from data import HERE
from train import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="\n")
    parser.add_argument("--max_tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ckpt", default=os.path.join(HERE, "out", "main", "ckpt.pt"))
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    model, tok, ckpt = load_model(args.ckpt, args.device)
    unknown = sorted(set(args.prompt) - set(tok.chars))
    if unknown:
        parser.error(f"prompt has characters the model doesn't know: {unknown}")

    generator = torch.Generator()
    if args.seed is not None:
        generator.manual_seed(args.seed)
    else:
        generator.seed()
    context = torch.tensor([tok.encode(args.prompt)], device=args.device)
    out = model.generate(context, args.max_tokens, temperature=args.temperature, top_k=args.top_k, generator=generator)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
