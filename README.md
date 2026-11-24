# mini-gpt

Building and training a small GPT-style language model from scratch, step by step:
micrograd → bigrams → MLP → GPT → own tokenizer on Ivan Franko → web chat. Everything trains locally on a
MacBook Air M2 via the Apple GPU (MPS).

## Structure

| Step | Folder | What's inside |
|---|---|---|
| 0 | `setup` | Environment check and CPU vs MPS benchmark |
| 1 | `micrograd` | Own autograd engine and MLP |
| 2 | `bigram` | Character-level bigram model |
| 3 | `mlp` | MLP with embeddings (Bengio 2003) |
| 4 | `gpt` | Transformer (mini-GPT) on Tiny Shakespeare |
| 5 | `franko` | Ivan Franko corpus, own BPE tokenizer, GPT trained on BPE vs characters |
| 6 | `web` | Web chat with the models: FastAPI + streaming UI, Docker (`docker compose up`) |

## Setup (macOS, Apple Silicon)

Requires Python 3.11+.

```bash
# Python via Homebrew (or: pyenv install 3.12 && pyenv local 3.12)
brew install python@3.12

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Check the environment and run the benchmark:

```bash
python setup/check_env.py   # MPS available: True
python setup/benchmark.py   # CPU vs MPS matmul timing
```

## Device conventions

Always pick the device like this:

```python
device = "mps" if torch.backends.mps.is_available() else "cpu"
```

If an operation is not supported on MPS, fall back to CPU for it:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

The Air has no fan and throttles after 15–20 minutes of sustained load, so long
training runs must save checkpoints.
