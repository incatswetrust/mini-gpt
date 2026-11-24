"""Train the stage 4 GPT on the Franko corpus.

  python franko/train.py --data bpe4000            # after prepare.py --tokenizer bpe4000
  python franko/train.py --data char
  python franko/train.py --data bpe4000 --resume   # continue from out/<data>/ckpt.pt

Reuses the model, optimizer, lr schedule and plotting from gpt; the data
comes from the pre-tokenized .bin files. Writes franko/out/<data>/:
ckpt.pt (latest), best.pt (lowest val loss), log.csv, loss.png.

Loss per token can't be compared across tokenizers (a BPE token is ~3
characters), so the log also reports val loss in bits per character.
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gpt"))
from data import get_batch  # noqa: E402  (gpt/data.py)
from model import GPT, GPTConfig  # noqa: E402
from train import estimate_loss, get_lr, make_optimizer, plot_log  # noqa: E402

from bpe import make_tokenizer  # noqa: E402

# Same architecture as gpt "main"; the token embedding and output layer grow with the vocabulary.
MODEL = dict(block_size=256, n_layer=6, n_head=6, n_embd=384, dropout=0.2)
# batch_size 64 = 2 micro-batches of 32 (grad_accum): the same update, half the peak
# memory. A full batch of 64 with a 4000-token output pushes the 8 GB Air into swap.
TRAIN = dict(batch_size=64, grad_accum=2, max_iters=3000, lr=1e-3, min_lr=1e-4, warmup_iters=100,
             eval_interval=250, eval_iters=100, ckpt_interval=250)
GRAD_CLIP = 1.0


def load_data(name):
    d = os.path.join(HERE, "data", name)
    with open(os.path.join(d, "meta.json")) as f:
        meta = json.load(f)
    data = {s: torch.from_numpy(np.fromfile(os.path.join(d, f"{s}.bin"), dtype=np.uint16).astype(np.int64))
            for s in ("train", "val")}
    return meta, data


def bits_per_char(loss, meta, split="val"):
    """nats per token -> bits per character of the same text."""
    return loss * meta[f"{split}_tokens"] / meta[f"{split}_chars"] / math.log(2)


def save_checkpoint(path, model, optimizer, step, best_val, cfg, meta, generator, elapsed):
    tokenizer = {k: v for k, v in meta.items() if k in ("type", "chars", "merges", "vocab_size")}
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "model_config": asdict(model.config), "train_config": cfg,
        "step": step, "best_val": best_val, "rng": generator.get_state(), "elapsed": elapsed,
        "tokenizer": tokenizer,
        **({"chars": meta["chars"]} if meta["type"] == "char" else {}),  # what gpt and the web server read
    }, path + ".tmp")
    os.replace(path + ".tmp", path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="bpe4000")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max_iters", type=int)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    cfg = dict(TRAIN) | ({"max_iters": args.max_iters} if args.max_iters else {})
    out_dir = os.path.join(HERE, "out", args.data)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path, best_path, log_path = (os.path.join(out_dir, f) for f in ("ckpt.pt", "best.pt", "log.csv"))
    device = args.device

    meta, splits = load_data(args.data)
    encode, decode = make_tokenizer(meta)
    generator = torch.Generator()

    if args.resume:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = GPT(GPTConfig(**ckpt["model_config"])).to(device)
        model.load_state_dict(ckpt["model"])
        optimizer = make_optimizer(model, cfg["lr"])
        optimizer.load_state_dict(ckpt["optimizer"])
        cfg = ckpt["train_config"] | ({"max_iters": args.max_iters} if args.max_iters else {})
        start, best_val, prev_elapsed = ckpt["step"], ckpt["best_val"], ckpt["elapsed"]
        generator.set_state(ckpt["rng"])
        print(f"resumed from {ckpt_path} at step {start}")
    else:
        model = GPT(GPTConfig(vocab_size=meta["vocab_size"], **MODEL)).to(device)
        optimizer = make_optimizer(model, cfg["lr"])
        start, best_val, prev_elapsed = 0, float("inf"), 0.0
        generator.manual_seed(1337)
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "train", "val", "lr", "elapsed_s", "val_bpc"])

    config = model.config
    print(f"device {device}, {model.num_params() / 1e6:.2f}M parameters, config {asdict(config)}")
    print(f"{meta['type']} tokenizer, vocab {meta['vocab_size']}; train {meta['train_tokens']:,} tokens, "
          f"val {meta['val_tokens']:,} tokens; settings {cfg}\n")

    block_size, batch_size = config.block_size, cfg["batch_size"]
    model.train()
    t0 = time.time() - prev_elapsed
    t_last, last_eval_step, step = time.time(), start, start
    try:
        for step in range(start, cfg["max_iters"] + 1):
            if (step % cfg["eval_interval"] == 0 or step == cfg["max_iters"]) and not (args.resume and step == start):
                ms = (time.time() - t_last) / max(1, step - last_eval_step) * 1000
                losses = estimate_loss(model, splits, block_size, batch_size // cfg["grad_accum"], cfg["eval_iters"], device)
                lr, elapsed = get_lr(step, cfg), time.time() - t0
                bpc = bits_per_char(losses["val"], meta)
                with open(log_path, "a", newline="") as f:
                    csv.writer(f).writerow([step, f"{losses['train']:.4f}", f"{losses['val']:.4f}",
                                            f"{lr:.2e}", f"{elapsed:.0f}", f"{bpc:.4f}"])
                print(f"step {step:5d}  train {losses['train']:.4f}  val {losses['val']:.4f}  "
                      f"val {bpc:.3f} bits/char  lr {lr:.2e}  {ms:.0f} ms/iter  {elapsed / 60:.1f} min", flush=True)
                if losses["val"] < best_val:
                    best_val = losses["val"]
                    save_checkpoint(best_path, model, optimizer, step, best_val, cfg, meta, generator, elapsed)
                t_last, last_eval_step = time.time(), step
            if step == cfg["max_iters"]:
                break
            if step > start and step % cfg["ckpt_interval"] == 0:
                save_checkpoint(ckpt_path, model, optimizer, step, best_val, cfg, meta, generator, time.time() - t0)

            for group in optimizer.param_groups:
                group["lr"] = get_lr(step, cfg)
            optimizer.zero_grad(set_to_none=True)
            micro = batch_size // cfg["grad_accum"]
            for _ in range(cfg["grad_accum"]):
                x, y = get_batch(splits["train"], block_size, micro, device, generator)
                _, loss = model(x, y)
                (loss / cfg["grad_accum"]).backward()  # gradients add up to the mean over the full batch
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
    except KeyboardInterrupt:
        print(f"\ninterrupted at step {step}, saving checkpoint")

    save_checkpoint(ckpt_path, model, optimizer, step, best_val, cfg, meta, generator, time.time() - t0)
    plot_log(log_path, os.path.join(out_dir, "loss.png"),
             f"GPT on Franko, {meta['type']} tokens: {model.num_params() / 1e6:.2f}M params")
    print(f"\nsaved {ckpt_path}; best val {best_val:.4f} ({bits_per_char(best_val, meta):.3f} bits/char) in {best_path}")

    model.eval()
    context = torch.tensor([encode("\n")], device=device)
    sample = model.generate(context, 300, temperature=0.8, generator=torch.Generator().manual_seed(42))
    print("\n" + decode(sample[0].tolist()))


if __name__ == "__main__":
    main()
