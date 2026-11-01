"""Train the GPT on Tiny Shakespeare.

  python 04_gpt/train.py --config small          # sanity check, minutes
  python 04_gpt/train.py --config main           # ~10M params, the real run
  python 04_gpt/train.py --config main --resume  # continue from out/main/ckpt.pt

Writes to 04_gpt/out/<config>/: ckpt.pt (every ckpt_interval steps, on Ctrl+C
and at the end), log.csv (train/val loss), loss.png.
"""
import argparse
import csv
import math
import os
import time
from dataclasses import asdict

import matplotlib
import torch

from data import HERE, CharTokenizer, get_batch, load_splits
from model import GPT, GPTConfig

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Training settings follow nanoGPT's shakespeare_char config.
CONFIGS = {
    "small": dict(
        model=dict(block_size=64, n_layer=4, n_head=4, n_embd=128, dropout=0.0),
        batch_size=32, max_iters=5000, lr=1e-3, min_lr=1e-4, warmup_iters=100,
        eval_interval=250, eval_iters=100, ckpt_interval=1000,
    ),
    "main": dict(
        model=dict(block_size=256, n_layer=6, n_head=6, n_embd=384, dropout=0.2),
        batch_size=64, max_iters=5000, lr=1e-3, min_lr=1e-4, warmup_iters=100,
        eval_interval=250, eval_iters=200, ckpt_interval=500,
    ),
}
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.99)  # beta2 0.99 instead of 0.999: few tokens per step, a bit more reactive
GRAD_CLIP = 1.0


def get_lr(it, cfg):
    """Linear warmup, then cosine decay from lr to min_lr."""
    if it < cfg["warmup_iters"]:
        return cfg["lr"] * (it + 1) / cfg["warmup_iters"]
    progress = (it - cfg["warmup_iters"]) / max(1, cfg["max_iters"] - cfg["warmup_iters"])
    coeff = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))  # 1 -> 0
    return cfg["min_lr"] + coeff * (cfg["lr"] - cfg["min_lr"])


def make_optimizer(model, lr):
    # Weight decay on 2-D params (weight matrices, embeddings) only, as in nanoGPT;
    # biases and LayerNorm gains/biases are left alone.
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": WEIGHT_DECAY}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=BETAS)


@torch.no_grad()
def estimate_loss(model, splits, block_size, batch_size, eval_iters, device, seed=1234):
    """Average loss over eval_iters random batches per split, without gradients
    and with dropout off. Same seed every time, so evals are comparable."""
    model.eval()
    out = {}
    for name, data in splits.items():
        g = torch.Generator().manual_seed(seed)
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device, g)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def save_checkpoint(path, model, optimizer, step, best_val, cfg, tok, generator, elapsed):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": asdict(model.config),
        "train_config": cfg,
        "step": step,
        "best_val": best_val,
        "chars": tok.chars,
        "rng": generator.get_state(),
        "elapsed": elapsed,
    }, path + ".tmp")
    os.replace(path + ".tmp", path)  # never leave a half-written checkpoint behind


def load_model(path, device):
    """Model + tokenizer from a checkpoint, for generation."""
    ckpt = torch.load(path, map_location=device)
    model = GPT(GPTConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, CharTokenizer(ckpt["chars"]), ckpt


def plot_log(log_path, png_path, title):
    with open(log_path) as f:
        rows = [r for r in csv.DictReader(f)]
    steps = [int(r["step"]) for r in rows]
    plt.figure(figsize=(8, 4.5))
    plt.plot(steps, [float(r["train"]) for r in rows], label="train")
    plt.plot(steps, [float(r["val"]) for r in rows], label="val")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=CONFIGS, default="small")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max_iters", type=int, help="override the config")
    parser.add_argument("--batch_size", type=int, help="override the config")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--out_dir")
    args = parser.parse_args()

    cfg = {k: v for k, v in CONFIGS[args.config].items() if k != "model"}
    for key in ("max_iters", "batch_size"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    out_dir = args.out_dir or os.path.join(HERE, "out", args.config)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path, log_path = os.path.join(out_dir, "ckpt.pt"), os.path.join(out_dir, "log.csv")
    device = args.device

    tok, train_data, val_data = load_splits()
    splits = {"train": train_data, "val": val_data}
    generator = torch.Generator()

    if args.resume:
        ckpt = torch.load(ckpt_path, map_location=device)
        config = GPTConfig(**ckpt["model_config"])
        model = GPT(config).to(device)
        model.load_state_dict(ckpt["model"])
        optimizer = make_optimizer(model, cfg["lr"])
        optimizer.load_state_dict(ckpt["optimizer"])
        cfg = ckpt["train_config"] | {k: cfg[k] for k in ("max_iters", "batch_size") if getattr(args, k) is not None}
        start, best_val, prev_elapsed = ckpt["step"], ckpt["best_val"], ckpt["elapsed"]
        generator.set_state(ckpt["rng"])
        print(f"resumed from {ckpt_path} at step {start}")
    else:
        config = GPTConfig(vocab_size=tok.vocab_size, **CONFIGS[args.config]["model"])
        model = GPT(config).to(device)
        optimizer = make_optimizer(model, cfg["lr"])
        start, best_val, prev_elapsed = 0, float("inf"), 0.0
        generator.manual_seed(1337)
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "train", "val", "lr", "elapsed_s"])

    print(f"device {device}, {model.num_params() / 1e6:.2f}M parameters, config {asdict(config)}")
    print(f"train {len(train_data)} tokens, val {len(val_data)} tokens, settings {cfg}\n")

    block_size, batch_size = config.block_size, cfg["batch_size"]
    model.train()
    t0 = time.time() - prev_elapsed  # elapsed time carries over across resumes
    t_last, last_eval_step = time.time(), start
    step = start
    try:
        for step in range(start, cfg["max_iters"] + 1):
            resumed_here = args.resume and step == start  # this eval is already in the log
            if (step % cfg["eval_interval"] == 0 or step == cfg["max_iters"]) and not resumed_here:
                ms = (time.time() - t_last) / max(1, step - last_eval_step) * 1000  # training only, not eval
                losses = estimate_loss(model, splits, block_size, batch_size, cfg["eval_iters"], device)
                lr = get_lr(step, cfg)
                elapsed = time.time() - t0
                with open(log_path, "a", newline="") as f:
                    csv.writer(f).writerow([step, f"{losses['train']:.4f}", f"{losses['val']:.4f}", f"{lr:.2e}", f"{elapsed:.0f}"])
                print(f"step {step:5d}  train {losses['train']:.4f}  val {losses['val']:.4f}  lr {lr:.2e}  "
                      f"{ms:.0f} ms/iter  {elapsed / 60:.1f} min")
                best_val = min(best_val, losses["val"])
                t_last, last_eval_step = time.time(), step
            if step == cfg["max_iters"]:
                break
            if step > start and step % cfg["ckpt_interval"] == 0:
                save_checkpoint(ckpt_path, model, optimizer, step, best_val, cfg, tok, generator, time.time() - t0)

            for group in optimizer.param_groups:
                group["lr"] = get_lr(step, cfg)
            x, y = get_batch(train_data, block_size, batch_size, device, generator)
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)  # guard against rare huge gradients
            optimizer.step()
    except KeyboardInterrupt:
        print(f"\ninterrupted at step {step}, saving checkpoint")

    save_checkpoint(ckpt_path, model, optimizer, step, best_val, cfg, tok, generator, time.time() - t0)
    plot_log(log_path, os.path.join(out_dir, "loss.png"), f"GPT {args.config}: {model.num_params() / 1e6:.2f}M params")
    print(f"\nsaved {ckpt_path}, best val {best_val:.4f}")

    context = torch.tensor([tok.encode("\n")], device=device)
    sample = model.generate(context, 300, temperature=0.8, generator=torch.Generator().manual_seed(42))
    print("\n" + tok.decode(sample[0].tolist()))


if __name__ == "__main__":
    main()
