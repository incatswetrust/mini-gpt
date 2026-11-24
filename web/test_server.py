import json
import os
import sys
from dataclasses import asdict

import pytest
import torch
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server import GenerateRequest, build_prompt, create_app  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gpt"))
from model import GPT, GPTConfig  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "franko"))
from bpe import BPETokenizer  # noqa: E402

CHARS = sorted(set("\n !',.:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"))


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """App with one tiny random model saved in the real checkpoint format."""
    models_dir = tmp_path_factory.mktemp("out")
    os.makedirs(models_dir / "tiny")
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=len(CHARS), block_size=32, n_layer=1, n_head=2, n_embd=16, dropout=0.0))
    torch.save({"model": model.state_dict(), "model_config": asdict(model.config), "chars": CHARS,
                "step": 100, "best_val": 2.5}, models_dir / "tiny" / "best.pt")
    return TestClient(create_app(str(models_dir), device="cpu"))


def stream_events(response):
    return [json.loads(line[len("data: "):]) for line in response.text.splitlines() if line.startswith("data: ")]


def generate(client, **kw):
    body = {"model": "shakespeare-tiny", "messages": [{"role": "user", "content": "Good morrow."}],
            "max_tokens": 20, "seed": 1} | kw
    return client.post("/api/generate", json=body)


def test_models(client):
    data = client.get("/api/models").json()
    assert data["device"] == "cpu"
    (m,) = data["models"]
    assert m["name"] == "shakespeare-tiny" and m["chat_format"] == "play" and m["tokenizer"] == "char" and m["block_size"] == 32
    assert m["vocab_size"] == len(CHARS) and m["params"] > 0 and m["val_loss"] == 2.5


def test_streamed_generation(client):
    r = generate(client)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = stream_events(r)
    assert events[0]["type"] == "start" and events[-1]["type"] == "done"
    tokens = [e for e in events if e["type"] == "token"]
    assert tokens and all(set(e["text"]) <= set(CHARS) for e in tokens)
    reply = "".join(e["text"] for e in tokens)
    assert "\n\n" not in reply and not reply.endswith("\n")  # end of turn is never sent
    done = events[-1]
    assert done["stop_reason"] in ("max_tokens", "end_of_turn")
    assert done["tokens"] <= 20 and done["context_used"] <= done["block_size"] == 32


def test_seed_makes_generation_repeatable(client):
    texts = ["".join(e.get("text", "") for e in stream_events(generate(client, seed=7))) for _ in range(2)]
    assert texts[0] == texts[1]


def test_unknown_characters_are_rejected(client):
    r = generate(client, messages=[{"role": "user", "content": "Привіт 🙂"}])
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert set(detail["unknown_chars"]) == set("Привіт🙂") and "doesn't know" in detail["error"]


def test_unknown_model(client):
    assert generate(client, model="nope").status_code == 404


def test_prompt_drops_oldest_turns_first():
    msgs = [{"role": "user", "content": "first turn here"}, {"role": "model", "content": "second"},
            {"role": "user", "content": "third"}]
    req = GenerateRequest(model="tiny", messages=msgs, user_name="A", model_name="B")
    prompt, kept = build_prompt(req, block_size=40)  # default encode=list: one token per character
    assert "".join(prompt) == "B:\nsecond\n\nA:\nthird\n\nB:\n" and kept == 2
    prompt, kept = build_prompt(req, block_size=8)  # even the newest turn doesn't fit: cut from the left
    assert len(prompt) == 8 and "".join(prompt).endswith("\n\nB:\n") and kept == 1


def test_dialogue_prompt():
    msgs = [{"role": "user", "content": "— Добрий день."}, {"role": "model", "content": "Здорові були!"}]
    req = GenerateRequest(model="x", messages=msgs)
    prompt, kept = build_prompt(req, block_size=100, chat_format="dialogue")
    assert "".join(prompt) == "— Добрий день.\n\n— Здорові були!\n\n—" and kept == 2


@pytest.fixture(scope="module")
def franko_client(tmp_path_factory):
    """A tiny random BPE model saved the way franko/train.py saves it."""
    tok = BPETokenizer.train("Сумно і непривітно тепер в нашій Тухольщині! " * 20, vocab_size=280)
    models_dir = tmp_path_factory.mktemp("out")
    os.makedirs(models_dir / "bpe")
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=tok.vocab_size, block_size=32, n_layer=1, n_head=2, n_embd=16, dropout=0.0))
    meta = {"type": "bpe", "merges": [[a, b, i] for (a, b), i in tok.merges.items()], "vocab_size": tok.vocab_size}
    torch.save({"model": model.state_dict(), "model_config": asdict(model.config), "tokenizer": meta,
                "step": 10, "best_val": 4.0}, models_dir / "bpe" / "best.pt")
    return TestClient(create_app(str(models_dir), device="cpu"))


def test_bpe_model(franko_client):
    (m,) = franko_client.get("/api/models").json()["models"]
    assert m["name"] == "franko-bpe" and m["tokenizer"] == "bpe" and m["chat_format"] == "dialogue"
    # Any text encodes into bytes, so nothing is rejected.
    r = franko_client.post("/api/generate", json={"model": "franko-bpe", "max_tokens": 40, "seed": 3,
                                                   "messages": [{"role": "user", "content": "Привіт 🙂 ’"}]})
    assert r.status_code == 200
    events = stream_events(r)
    assert events[0]["context_tokens"] <= 32 and events[-1]["type"] == "done"
    # A random model emits arbitrary bytes, so U+FFFD can appear mid-text; but the
    # server never sends a character whose second byte hasn't been generated yet.
    reply = "".join(e["text"] for e in events if e["type"] == "token")
    assert reply and not reply.endswith("\ufffd")
