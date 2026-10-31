# Stage 2: character bigram model

The simplest language model: predicts the next character from only the
previous one. Trained on `names.txt` (32 033 names, from
[karpathy/makemore](https://github.com/karpathy/makemore), MIT license), it
generates new name-like strings.

| File | What |
|---|---|
| `data.py` | Load names, `stoi` / `itos` vocab with `.` as start/end token, bigram tensors, sampling |
| `bigram_counts.py` | 27×27 count table → probabilities → samples, NLL; saves `bigram_counts.png` |
| `bigram_nn.py` | Same model as a network: one-hot → linear 27×27 → softmax, cross-entropy, gradient descent |
| `test_bigram.py` | Vocab, counting, NLL, softmax ≡ counts, loss vs `F.cross_entropy`, sampling |
| `NOTES.md` | Results and explanations of softmax, cross-entropy and sampling |

## Run

```bash
pytest 02_bigram
python 02_bigram/bigram_counts.py
python 02_bigram/bigram_nn.py      # ~1000 full-batch steps; uses MPS if available
```
