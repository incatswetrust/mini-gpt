# Stage 3: MLP with embeddings

Character-level MLP language model after Bengio et al. 2003: the previous 3
characters are embedded, concatenated and passed through a tanh hidden layer.
Val loss 2.10 on names.txt vs 2.45 for the bigram.

| File | What |
|---|---|
| `data.py` | Dataset of (3-char context → next char), split 80 / 10 / 10 by name; reuses `02_bigram/names.txt` |
| `mlp.py` | MLP with raw tensors + the shared training loop (minibatches, lr decay, train/val tracking) |
| `lr_finder.py` | Learning rate search |
| `overfit.py` | Overfitting demo on 500 names |
| `layers.py` | Own `Embedding`, `Flatten`, `Linear`, `BatchNorm1d`, `Tanh`, `Sequential` |
| `mlp_bn.py` | Deeper MLP with BatchNorm built from `layers.py` |
| `diagnostics.py` | Init and activation statistics: naive vs scaled init, deep net with/without BatchNorm |
| `embed2d.py` | 2-dim embeddings plotted on a plane |
| `test_mlp.py` | Dataset, splits, initial loss, own layers vs `torch.nn` |
| `NOTES.md` | Results, plots and explanations |

## Run

```bash
pytest 03_mlp
python 03_mlp/lr_finder.py     # ~10 s
python 03_mlp/mlp.py           # 200k steps, ~12 min on CPU
python 03_mlp/overfit.py       # ~30 s
python 03_mlp/embed2d.py       # ~1 min
python 03_mlp/diagnostics.py   # ~10 min
python 03_mlp/mlp_bn.py        # ~10 min
```

Timings are from a 2-core x86 CPU; everything uses MPS when available.
