# Stage 4: mini-GPT

A decoder-only transformer written out piece by piece, trained on Tiny
Shakespeare (1.1M characters, 65-char vocabulary) to generate text in its style.

| File | What |
|---|---|
| `data.py` | Download Tiny Shakespeare, char tokenizer, 90 / 10 split, `get_batch()` |
| `bigram.py` | 4.2 baseline: bigram with `nn.Embedding`, `generate()`, `estimate_loss()` |
| `attention_trick.py` | 4.3: averaging the past with loops → triangular matmul → masked softmax → attention |
| `model.py` | `Head`, `MultiHeadAttention`, `FeedForward`, `Block` (pre-norm + residual), `GPT` with temperature / top-k generation |
| `train.py` | Configs, AdamW, warmup + cosine lr, checkpoints and resume, loss log and plot |
| `generate.py` | CLI for generating text from a checkpoint |
| `test_gpt.py` | Tokenizer, batches, causal mask, parameter count, generation, lr schedule |
| `NOTES.md` | Results and explanations |

## Configs

| | `small` | `main` |
|---|---|---|
| layers / heads / `n_embd` | 4 / 4 / 128 | 6 / 6 / 384 |
| `block_size` | 64 | 256 |
| dropout | 0.0 | 0.2 |
| batch | 32 | 64 |
| parameters | 0.82M | 10.8M |
| purpose | sanity check, minutes | the real run, target val ≈ 1.5 |

Both: 5000 steps, AdamW (β = 0.9, 0.99, weight decay 0.1), lr 1e-3 with 100
warmup steps and cosine decay to 1e-4, grad clipping at 1.0 — the settings of
nanoGPT's `shakespeare_char`.

## Run

```bash
pytest 04_gpt
python 04_gpt/bigram.py
python 04_gpt/attention_trick.py

python 04_gpt/train.py --config small
python 04_gpt/train.py --config main
python 04_gpt/train.py --config main --resume     # after Ctrl+C or a crash

python 04_gpt/generate.py --prompt "ROMEO:" --temperature 0.8 --max_tokens 500
python 04_gpt/generate.py --ckpt 04_gpt/out/small/ckpt.pt --top_k 10
```

Outputs go to `04_gpt/out/<config>/` (git-ignored): `ckpt.pt`, `log.csv`,
`loss.png`. A checkpoint is written every 500 steps for `main` (1000 for
`small`), on Ctrl+C, and at the end; `--resume` continues with the same
optimizer state, lr schedule position and data order.

**MacBook Air:** it has no fan and throttles after 15–20 minutes of load, so
`main` will slow down as it runs. Watch `ms/iter` in the log; if a run is too
long, stop it with Ctrl+C and resume later, or lower `--max_iters`.
