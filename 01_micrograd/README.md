# Stage 1: micrograd

A scalar autograd engine in pure Python and a small neural network trained on it.

| File | What |
|---|---|
| `engine.py` | `Value`: one number + its gradient + the graph it came from; `backward()` |
| `nn.py` | `Neuron`, `Layer`, `MLP` built from `Value`s |
| `viz.py` | Draw the computation graph with graphviz |
| `data.py` | `make_moons` toy dataset (numpy, no sklearn) |
| `train_moons.py` | Train `MLP(2, [16, 16, 1])` on two moons, plot the decision boundary |
| `experiments.py` | Learning rate sweep and "forgot to zero grads" run |
| `test_engine.py` | Gradients vs numerical derivatives and vs PyTorch |
| `NOTES.md` | Results and observations |

## Run

```bash
pytest 01_micrograd                     # gradient checks
python 01_micrograd/train_moons.py      # -> decision_boundary.png
python 01_micrograd/experiments.py      # -> experiments.png
python 01_micrograd/viz.py              # -> neuron_graph.svg (needs `brew install graphviz`)
```

## How it works

Every operation on `Value`s (`+`, `*`, `**`, `exp`, `tanh`, `relu`; `-` and `/`
are built from those) returns a new `Value` that remembers its parents and a
`_backward` function with the local derivative. `backward()` sorts the graph
topologically and calls `_backward` from the output back to the inputs, so
each node's gradient is the chain rule product of the local derivatives on
the way. Gradients are accumulated with `+=`, which is why a value used twice
gets both contributions, and why they must be zeroed before every step.

Training loop: forward pass → hinge loss + L2 → `zero_grad()` → `backward()`
→ `p.data -= lr * p.grad`.
