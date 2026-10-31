"""Check our gradients against numerical derivatives and against PyTorch.

Run: pytest 01_micrograd
"""
import math

import pytest
import torch

from engine import Value
from nn import MLP

H = 1e-6
TOL = 1e-4


def numerical_grads(f, xs):
    """Central difference (f(x+h) - f(x-h)) / 2h for each input."""
    grads = []
    for i in range(len(xs)):
        plus = list(xs); plus[i] += H
        minus = list(xs); minus[i] -= H
        grads.append((f(*plus) - f(*minus)) / (2 * H))
    return grads


def our_grads(f, xs):
    vs = [Value(x) for x in xs]
    out = f(*vs)
    out.backward()
    return out.data, [v.grad for v in vs]


CASES = {
    "add": lambda a, b: a + b,
    "mul": lambda a, b: a * b,
    "sub": lambda a, b: a - b,
    "div": lambda a, b: a / b,
    "pow": lambda a, b: a**3 + b**-2,
    "exp": lambda a, b: (a * b).exp(),
    "tanh": lambda a, b: (a + b).tanh(),
    "relu": lambda a, b: (a * b).relu() + (a - b).relu(),
    "reuse": lambda a, b: a * a + a * b,  # same node used several times
    "consts": lambda a, b: 2 * a - 3 / b + 1 - b,
    "neuron": lambda a, b: (a * -3.0 + b * 1.0 + 6.88).tanh(),
}


@pytest.mark.parametrize("name", CASES)
def test_against_numerical(name):
    f = CASES[name]
    xs = [1.3, -0.7]
    _, grads = our_grads(f, xs)
    expected = numerical_grads(lambda *a: f(*(Value(x) for x in a)).data, xs)
    assert grads == pytest.approx(expected, abs=TOL)


def test_against_pytorch():
    # Expression from Karpathy's micrograd README: exercises every op.
    def expr(x):
        z = 2 * x + 2 + x
        q = z.relu() + z * x
        h = (z * z).relu()
        y = h + q + q * x
        return y

    x = Value(-4.0)
    y = expr(x)
    y.backward()

    xt = torch.tensor(-4.0, dtype=torch.double, requires_grad=True)
    z = 2 * xt + 2 + xt
    q = z.relu() + z * xt
    h = (z * z).relu()
    yt = h + q + q * xt
    yt.backward()

    assert y.data == pytest.approx(yt.item())
    assert x.grad == pytest.approx(xt.grad.item())


def test_against_pytorch_all_ops():
    a, b = Value(-4.0), Value(2.0)
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f + (a / 10).tanh() + (b / 10).exp()
    g.backward()

    at = torch.tensor(-4.0, dtype=torch.double, requires_grad=True)
    bt = torch.tensor(2.0, dtype=torch.double, requires_grad=True)
    c = at + bt
    d = at * bt + bt**3
    c = c + c + 1
    c = c + 1 + c + (-at)
    d = d + d * 2 + (bt + at).relu()
    d = d + 3 * d + (bt - at).relu()
    e = c - d
    f = e**2
    gt = f / 2.0
    gt = gt + 10.0 / f + (at / 10).tanh() + (bt / 10).exp()
    gt.backward()

    assert g.data == pytest.approx(gt.item())
    assert a.grad == pytest.approx(at.grad.item())
    assert b.grad == pytest.approx(bt.grad.item())


def test_mlp_matches_pytorch():
    model = MLP(3, [4, 4, 1])
    x = [2.0, 3.0, -1.0]
    out = model(x)
    out.backward()

    # Rebuild the same network in PyTorch from our weights.
    xt = torch.tensor(x, dtype=torch.double)
    layer_params = []
    for layer in model.layers:
        W = torch.tensor([[w.data for w in n.w] for n in layer.neurons], dtype=torch.double, requires_grad=True)
        B = torch.tensor([n.b.data for n in layer.neurons], dtype=torch.double, requires_grad=True)
        layer_params.append((W, B))
    h = xt
    for i, (W, B) in enumerate(layer_params):
        h = W @ h + B
        if i < len(layer_params) - 1:
            h = h.relu()
    h.sum().backward()

    assert out.data == pytest.approx(h.item())
    for layer, (W, B) in zip(model.layers, layer_params):
        for j, n in enumerate(layer.neurons):
            assert [w.grad for w in n.w] == pytest.approx(W.grad[j].tolist())
            assert n.b.grad == pytest.approx(B.grad[j].item())


def test_deep_graph_no_recursion_error():
    x = Value(0.5)
    y = x
    for _ in range(5000):
        y = y * 1.0 + 0.0
    y.backward()
    assert x.grad == pytest.approx(1.0)
