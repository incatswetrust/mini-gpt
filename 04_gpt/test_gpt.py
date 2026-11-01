"""Run: pytest 04_gpt"""
import math

import pytest
import torch

from data import CharTokenizer, get_batch, load_splits
from model import GPT, GPTConfig
from train import CONFIGS, get_lr

SMALL = GPTConfig(vocab_size=65, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)


def test_tokenizer_roundtrip():
    tok, train, val = load_splits()
    assert tok.vocab_size == 65
    text = "First Citizen:\nBefore we proceed any further, hear me speak."
    assert tok.decode(tok.encode(text)) == text
    assert len(train) + len(val) == 1115394
    assert abs(len(train) / (len(train) + len(val)) - 0.9) < 1e-4


def test_get_batch_targets_are_shifted_inputs():
    data = torch.arange(100)
    x, y = get_batch(data, block_size=8, batch_size=4, device="cpu", generator=torch.Generator().manual_seed(0))
    assert x.shape == y.shape == (4, 8)
    assert torch.equal(y[:, :-1], x[:, 1:])
    assert torch.equal(y[:, -1], x[:, -1] + 1)


def test_forward_shapes_and_initial_loss():
    torch.manual_seed(0)
    model = GPT(SMALL)
    x = torch.randint(0, 65, (4, 16))
    logits, loss = model(x, x)
    assert logits.shape == (4, 16, 65)
    assert abs(loss.item() - math.log(65)) < 0.1  # near-uniform predictions at init


def test_causal_no_information_from_the_future():
    """Changing token 10 must not change the outputs at positions 0..9."""
    torch.manual_seed(0)
    model = GPT(SMALL).eval()
    x = torch.randint(0, 65, (1, 16))
    x2 = x.clone()
    x2[0, 10] = (x[0, 10] + 1) % 65
    out1, _ = model(x)
    out2, _ = model(x2)
    assert torch.allclose(out1[0, :10], out2[0, :10], atol=1e-5)
    assert not torch.allclose(out1[0, 10:], out2[0, 10:], atol=1e-5)


def test_main_config_is_about_10M_params():
    model = GPT(GPTConfig(vocab_size=65, **CONFIGS["main"]["model"]))
    assert 10e6 < model.num_params() < 11e6


def test_generate_length_and_context_crop():
    torch.manual_seed(0)
    model = GPT(SMALL).eval()
    out = model.generate(torch.zeros((1, 1), dtype=torch.long), 40)  # longer than block_size
    assert out.shape == (1, 41)


def test_top_k_only_samples_from_top_k():
    torch.manual_seed(0)
    model = GPT(SMALL).eval()
    with torch.no_grad():  # tokens 3 and 7 likely, the other 63 together still ~20% per step
        model.lm_head.bias.fill_(0)
        model.lm_head.bias[[3, 7]] = 5.0
    ctx = torch.zeros((1, 1), dtype=torch.long)
    free = model.generate(ctx, 100, generator=torch.Generator().manual_seed(0))
    assert set(free[0, 1:].tolist()) - {3, 7}  # without top_k other tokens show up
    out = model.generate(ctx, 100, top_k=2, generator=torch.Generator().manual_seed(0))
    assert set(out[0, 1:].tolist()) <= {3, 7}


def test_low_temperature_is_greedy():
    torch.manual_seed(0)
    model = GPT(SMALL).eval()
    g = torch.Generator().manual_seed(0)
    ctx = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(ctx, 1, temperature=1e-4, generator=g)
    logits, _ = model(ctx)
    assert out[0, 1].item() == logits[0, -1].argmax().item()


def test_lr_schedule():
    cfg = CONFIGS["main"]
    assert get_lr(0, cfg) == pytest.approx(cfg["lr"] / cfg["warmup_iters"])
    assert get_lr(cfg["warmup_iters"], cfg) == pytest.approx(cfg["lr"])
    assert get_lr(cfg["max_iters"], cfg) == pytest.approx(cfg["min_lr"])
    lrs = [get_lr(i, cfg) for i in range(cfg["warmup_iters"], cfg["max_iters"])]
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))  # monotone decay after warmup


def test_char_tokenizer_unknown_char_fails():
    tok = CharTokenizer("ab")
    with pytest.raises(KeyError):
        tok.encode("abc")
