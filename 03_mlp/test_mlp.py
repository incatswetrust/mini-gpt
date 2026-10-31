"""Run: pytest 03_mlp"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import build_dataset, build_vocab, splits
from layers import BatchNorm1d, Embedding, Flatten, Linear, Sequential, Tanh
from mlp import MLP
from mlp_bn import build_model


def test_build_dataset():
    stoi, itos = build_vocab(["emma"])
    X, Y = build_dataset(["emma"], stoi, block_size=3)
    contexts = ["".join(itos[i] for i in row) for row in X.tolist()]
    targets = "".join(itos[i] for i in Y.tolist())
    assert contexts == ["...", "..e", ".em", "emm", "mma"]
    assert targets == "emma."


def test_splits_are_80_10_10():
    _, _, data = splits()
    sizes = {k: v[0].shape[0] for k, v in data.items()}
    total = sum(sizes.values())
    assert total == 228146  # every bigram example from stage 2
    assert abs(sizes["train"] / total - 0.8) < 0.01
    names = {k: int((v[1] == 0).sum()) for k, v in data.items()}
    assert names == {"train": 25626, "val": 3203, "test": 3204}


def test_initial_loss_is_close_to_uniform():
    _, _, data = splits()
    X, Y = data["train"]
    with torch.no_grad():
        loss = F.cross_entropy(MLP()(X[:5000]), Y[:5000]).item()
    assert abs(loss - math.log(27)) < 0.05


def test_linear_and_tanh_match_torch():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(8, 5, generator=g)
    lin = Linear(5, 3, generator=g)
    ref = nn.Linear(5, 3)
    with torch.no_grad():
        ref.weight.copy_(lin.weight.T)
        ref.bias.copy_(lin.bias)
    assert torch.allclose(lin(x), ref(x), atol=1e-6)
    assert torch.allclose(Tanh()(x), torch.tanh(x))


def test_batchnorm_matches_torch_in_train_and_eval():
    g = torch.Generator().manual_seed(0)
    bn, ref = BatchNorm1d(4), nn.BatchNorm1d(4)
    for _ in range(3):  # several steps so running stats accumulate
        x = torch.randn(16, 4, generator=g) * 3 + 1
        assert torch.allclose(bn(x), ref(x), atol=1e-5)
    assert torch.allclose(bn.running_mean, ref.running_mean, atol=1e-6)
    assert torch.allclose(bn.running_var, ref.running_var, atol=1e-5)
    bn.eval(); ref.eval()
    x = torch.randn(1, 4, generator=g)  # a single example works in eval mode
    assert torch.allclose(bn(x), ref(x), atol=1e-5)


def test_batchnorm_output_is_normalized():
    x = torch.randn(64, 10) * 5 + 3
    out = BatchNorm1d(10)(x)
    assert torch.allclose(out.mean(0), torch.zeros(10), atol=1e-5)
    assert torch.allclose(out.std(0, unbiased=False), torch.ones(10), atol=1e-3)


def test_sequential_shapes_and_modes():
    model = Sequential([Embedding(27, 10), Flatten(), Linear(30, 20), BatchNorm1d(20), Tanh(), Linear(20, 27)])
    assert model(torch.randint(0, 27, (32, 3))).shape == (32, 27)
    model.eval()
    assert model.layers[3].training is False
    model.train()
    assert model.layers[3].training is True


def test_bn_model_initial_loss():
    _, _, data = splits()
    X, Y = data["train"]
    model = build_model()
    with torch.no_grad():
        loss = F.cross_entropy(model(X[:5000]), Y[:5000]).item()
    assert abs(loss - math.log(27)) < 0.05
