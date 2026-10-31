# mini-gpt

Building and training a small GPT-style language model from scratch, step by step:
micrograd → bigrams → MLP → GPT → custom dataset. Everything trains locally on a
MacBook Air M2 via the Apple GPU (MPS).

## Structure

| Folder | Stage |
|---|---|
| `00_setup` | Environment check and CPU vs MPS benchmark |
| `01_micrograd` | Own autograd engine and MLP |
| `02_bigram` | Character-level bigram model |
| `03_mlp` | MLP with embeddings (Bengio 2003) |
| `04_gpt` | Transformer (mini-GPT) on Tiny Shakespeare |
| `05_custom` | Custom corpus and BPE tokenizer |

Jupyter notebooks are for experiments, `.py` scripts hold the final code.

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
python 00_setup/check_env.py   # MPS available: True
python 00_setup/benchmark.py   # CPU vs MPS matmul timing
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
