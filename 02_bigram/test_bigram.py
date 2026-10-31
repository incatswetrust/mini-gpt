"""Run: pytest 02_bigram"""
import torch
import torch.nn.functional as F

from bigram_counts import count_table, nll, probabilities
from bigram_nn import forward, loss_fn
from data import bigram_tensors, bigrams, build_vocab, load_words, sample_names


def test_vocab_roundtrip():
    words = load_words()
    stoi, itos = build_vocab(words)
    assert len(stoi) == 27 and stoi["."] == 0
    assert all(itos[stoi[ch]] == ch for ch in stoi)


def test_bigrams_of_word():
    assert list(bigrams("emma")) == [(".", "e"), ("e", "m"), ("m", "m"), ("m", "a"), ("a", ".")]


def test_counts_and_probabilities():
    words = ["ab", "a"]
    stoi, _ = build_vocab(words)
    xs, ys = bigram_tensors(words, stoi)
    N = count_table(xs, ys, len(stoi))
    assert N[stoi["."], stoi["a"]] == 2  # both names start with 'a'
    assert N[stoi["a"], stoi["b"]] == 1
    assert N.sum() == 5
    P = probabilities(N)
    assert torch.allclose(P.sum(dim=1), torch.ones(len(stoi)))
    assert (P > 0).all()  # smoothing: no zero probabilities


def test_nll_of_uniform_model():
    P = torch.full((27, 27), 1 / 27)
    xs, ys = torch.tensor([0, 5, 9]), torch.tensor([1, 2, 0])
    assert abs(nll(P, xs, ys) - torch.log(torch.tensor(27.0)).item()) < 1e-5


def test_nn_softmax_matches_counts_when_W_is_log_counts():
    # If W = log(counts), softmax(one_hot @ W) is exactly the normalized count table.
    words = load_words()[:500]
    stoi, _ = build_vocab(words)
    xs, ys = bigram_tensors(words, stoi)
    P = probabilities(count_table(xs, ys, len(stoi)))
    W = P.log()
    P_nn = forward(W, torch.arange(len(stoi)))
    assert torch.allclose(P_nn, P, atol=1e-6)


def test_loss_matches_pytorch_cross_entropy():
    g = torch.Generator().manual_seed(0)
    W = torch.randn(27, 27, generator=g)
    xs = torch.randint(0, 27, (100,), generator=g)
    ys = torch.randint(0, 27, (100,), generator=g)
    expected = F.cross_entropy(F.one_hot(xs, 27).float() @ W, ys)
    assert torch.allclose(loss_fn(W, xs, ys, reg=0.0), expected, atol=1e-6)


def test_sampling_stops_at_end_token():
    # From '.', always go to 'a'; from 'a', always go to '.' -> every name is "a".
    P = torch.zeros(27, 27)
    P[0, 1] = 1.0
    P[1, 0] = 1.0
    g = torch.Generator().manual_seed(0)
    assert sample_names(P, {1: "a"}, 3, g) == ["a", "a", "a"]
