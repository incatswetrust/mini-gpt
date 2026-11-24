"""Trained weights: export them for a GitHub Release, or download them from one.

  python web/weights.py download   # get the released models, then: python web/server.py
  python web/weights.py export     # (maintainer) best.pt -> dist/<model>.pt for a release

A training checkpoint also holds the optimizer state and RNG, three times the
size of the weights. The release files keep only what generation needs: the
weights, the model config, the tokenizer and the validation loss. They load
with the same code as a checkpoint: the web server, gpt/generate.py and
franko/generate.py read them from where training would have written them.
"""
import argparse
import os
import urllib.request

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RELEASE_URL = "https://github.com/incatswetrust/mini-gpt/releases/latest/download"
# release file -> where training writes the checkpoint
MODELS = {
    "shakespeare-main.pt": "gpt/out/main/best.pt",
    "franko-bpe4000.pt": "franko/out/bpe4000/best.pt",
    "franko-char.pt": "franko/out/char/best.pt",
}
KEEP = ("model", "model_config", "tokenizer", "chars", "step", "best_val")


def export(out_dir):
    import torch

    os.makedirs(out_dir, exist_ok=True)
    for name, rel in MODELS.items():
        ckpt = torch.load(os.path.join(ROOT, rel), map_location="cpu", weights_only=False)
        slim = {k: ckpt[k] for k in KEEP if k in ckpt}
        path = os.path.join(out_dir, name)
        torch.save(slim, path)
        print(f"{rel} ({os.path.getsize(os.path.join(ROOT, rel)) / 1e6:.0f} MB) -> {path} "
              f"({os.path.getsize(path) / 1e6:.0f} MB)")


def download(force):
    for name, rel in MODELS.items():
        path = os.path.join(ROOT, rel)
        if os.path.exists(path) and not force:
            print(f"{rel} exists, skipping (--force to replace)")
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"{RELEASE_URL}/{name} -> {rel}")
        urllib.request.urlretrieve(f"{RELEASE_URL}/{name}", path + ".part")
        os.replace(path + ".part", path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("download", "export"))
    parser.add_argument("--out_dir", default=os.path.join(ROOT, "dist"))
    parser.add_argument("--force", action="store_true", help="download: replace existing checkpoints")
    args = parser.parse_args()
    export(args.out_dir) if args.command == "export" else download(args.force)


if __name__ == "__main__":
    main()
