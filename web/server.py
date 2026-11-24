"""Web chat with the trained models.

  python web/server.py               # http://localhost:8000, MPS on the Mac
  python web/server.py --port 8080

Loads every checkpoint found as <dir>/<run>/best.pt (or ckpt.pt when there
is no best.pt) for each dir in MODELS_DIR (separated by ":"); the default is
gpt/out (Shakespeare) and franko/out (Franko). Serves the chat UI from
web/static and the API under /api.

The models are base language models, not assistants: a "reply" is the model
continuing a transcript written the way its training text writes dialogue.
Shakespeare's is a play, each turn a speech under the speaker's name:

    YOU:
    Good morrow, sir.

    BARD:
    <the model writes here, until it ends the speech with a blank line>

Franko's prose puts each line of dialogue in its own paragraph after a dash:

    — Добрий день.

    — <the model writes here, until the paragraph ends>
"""
import argparse
import asyncio
import glob
import json
import os
import sys
import time

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gpt"))
sys.path.insert(0, os.path.join(HERE, "..", "franko"))
from model import GPT, GPTConfig  # noqa: E402

from bpe import make_tokenizer  # noqa: E402  (franko/bpe.py)

DEFAULT_MODELS_DIRS = [os.path.join(HERE, "..", "gpt", "out"), os.path.join(HERE, "..", "franko", "out")]
STATIC_DIR = os.path.join(HERE, "static")
END_OF_TURN = "\n\n"  # a blank line ends a speech in the training text


def pick_device():
    if os.environ.get("DEVICE"):
        return os.environ["DEVICE"]
    return "mps" if torch.backends.mps.is_available() else "cpu"


# Typing variants the Franko corpus normalized away (franko/extract.py).
FRANKO_INPUT = str.maketrans({"’": "'", "ʼ": "'", "`": "'", "–": "—", "\u00a0": " "})


class LoadedModel:
    def __init__(self, run, path, device):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        self.path = path
        self.model = GPT(GPTConfig(**ckpt["model_config"])).to(device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        # Stage 5 checkpoints describe their tokenizer (char or BPE) and come
        # from the Franko corpus; stage 4 ones only carry a character list.
        tok = ckpt.get("tokenizer") or {"type": "char", "chars": ckpt["chars"]}
        self.tokenizer_type, self.chars = tok["type"], set(tok.get("chars", ()))
        self.encode, self.decode = make_tokenizer(tok)
        self.corpus = "franko" if "tokenizer" in ckpt else "shakespeare"
        self.chat_format = "dialogue" if self.corpus == "franko" else "play"
        self.name = f"{self.corpus}-{run}"
        self.step, self.val_loss = ckpt.get("step"), ckpt.get("best_val")
        self.device = device

    @property
    def block_size(self):
        return self.model.config.block_size

    def info(self):
        cfg = self.model.config
        return {
            "name": self.name,
            "corpus": self.corpus,
            "chat_format": self.chat_format,
            "params": self.model.num_params(),
            "tokenizer": self.tokenizer_type,
            "vocab_size": cfg.vocab_size,
            "block_size": cfg.block_size,
            "n_layer": cfg.n_layer,
            "n_head": cfg.n_head,
            "n_embd": cfg.n_embd,
            "step": self.step,
            "val_loss": round(self.val_loss, 4) if self.val_loss is not None else None,
        }

    def prepare(self, text):
        return text.translate(FRANKO_INPUT) if self.corpus == "franko" else text

    def unknown_chars(self, text):
        # A byte-level BPE can encode anything; a character model only its vocabulary.
        return sorted(set(text) - self.chars) if self.tokenizer_type == "char" else []


def load_models(models_dirs, device):
    models = {}
    for models_dir in models_dirs:
        for d in sorted(glob.glob(os.path.join(models_dir, "*"))):
            path = next((p for p in (os.path.join(d, "best.pt"), os.path.join(d, "ckpt.pt")) if os.path.exists(p)), None)
            if path:
                m = LoadedModel(os.path.basename(d), path, device)
                models[m.name] = m
                print(f"loaded {m.name}: {path} ({m.model.num_params() / 1e6:.2f}M params, {m.tokenizer_type}, {device})")
    return models


class Message(BaseModel):
    role: str = Field(pattern="^(user|model)$")
    content: str


class GenerateRequest(BaseModel):
    model: str
    messages: list[Message] = Field(min_length=1)
    temperature: float = Field(0.8, gt=0, le=3)
    top_k: int | None = Field(None, ge=1)
    max_tokens: int = Field(300, ge=1, le=2000)
    seed: int | None = None
    user_name: str = "YOU"
    model_name: str = "BARD"


def speech(name, text):
    return f"{name}:\n{text.strip()}{END_OF_TURN}"


def dialogue_line(text):
    return f"— {text.strip().lstrip('—-').strip()}{END_OF_TURN}"


def build_prompt(req, block_size, chat_format="play", encode=list):
    """Token ids of the conversation transcript, ending where the model's
    next turn starts.

    The model sees at most block_size tokens, so the oldest turns are dropped
    first; the newest turn is cut from the left if it alone is too long.
    Returns (token ids, number of turns kept).
    """
    if chat_format == "dialogue":
        header, turns = "— ", [dialogue_line(m.content) for m in req.messages]
    else:
        header = f"{req.model_name}:\n"
        turns = [speech(req.user_name if m.role == "user" else req.model_name, m.content) for m in req.messages]
    kept, length = [], len(encode(header))
    for t in reversed(turns):
        n = len(encode(t))
        if length + n > block_size and kept:
            break
        kept.insert(0, t)
        length += n
    return encode("".join(kept) + header)[-block_size:], len(kept)


def create_app(models_dir=None, device=None):
    device = device or pick_device()
    dirs = (models_dir or os.environ.get("MODELS_DIR", "")).split(":")
    models = load_models([d for d in dirs if d] or DEFAULT_MODELS_DIRS, device)
    lock = asyncio.Lock()  # one generation at a time: they share one GPU
    app = FastAPI(title="mini-gpt chat")

    @app.get("/api/models")
    def list_models():
        return {"device": device, "models": [m.info() for m in models.values()]}

    @app.post("/api/generate")
    async def generate(req: GenerateRequest, request: Request):
        m = models.get(req.model)
        if m is None:
            raise HTTPException(404, {"error": f"unknown model {req.model!r}", "models": list(models)})
        for msg in req.messages:
            msg.content = m.prepare(msg.content)
        names = req.user_name + req.model_name if m.chat_format == "play" else ""
        text = names + "".join(msg.content for msg in req.messages)
        unknown = m.unknown_chars(text)
        if unknown:
            raise HTTPException(400, {
                "error": f"the {req.model} model doesn't know these characters: {' '.join(map(repr, unknown))}",
                "unknown_chars": unknown,
            })
        ids, turns_kept = build_prompt(req, m.block_size, m.chat_format, m.encode)
        context = torch.tensor([ids], dtype=torch.long, device=device)
        generator = torch.Generator().manual_seed(req.seed) if req.seed is not None else None

        async def events():
            def send(obj):
                return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

            yield send({"type": "start", "context_tokens": context.size(1), "block_size": m.block_size,
                        "turns_kept": turns_kept, "turns_total": len(req.messages)})
            async with lock:
                stream = m.model.generate_stream(context, req.max_tokens, req.temperature, req.top_k, generator)
                tokens, reply, sent, t0, stop = [], "", 0, time.time(), "max_tokens"
                while True:
                    if await request.is_disconnected():
                        stop = "cancelled"
                        break
                    token = await asyncio.to_thread(next, stream, None)  # keep the event loop free
                    if token is None:
                        break
                    tokens.append(token)
                    # Decode the whole reply: a BPE token can be half of a
                    # 2-byte Cyrillic letter, which decodes only with its other half.
                    reply = m.decode(tokens)
                    end = reply.find(END_OF_TURN, sent)
                    if end != -1:  # the model finished its speech
                        if end > sent:
                            yield send({"type": "token", "text": reply[sent:end]})
                        stop = "end_of_turn"
                        break
                    # Hold back trailing newlines (they may turn out to be the end of
                    # the turn) and an unfinished multi-byte character (decoded as U+FFFD).
                    safe = len(reply.rstrip("\n\ufffd"))
                    if safe > sent:
                        yield send({"type": "token", "text": reply[sent:safe]})
                        sent = safe
                stream.close()
            elapsed, n = time.time() - t0, len(tokens)
            yield send({"type": "done", "stop_reason": stop, "tokens": n,
                        "tokens_per_s": round(n / elapsed, 1) if elapsed > 0 else None,
                        "context_used": min(m.block_size, context.size(1) + n), "block_size": m.block_size})

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    if os.path.isdir(STATIC_DIR):
        @app.get("/")
        def index():
            return FileResponse(os.path.join(STATIC_DIR, "index.html"))

        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    return app


def main():
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--models_dir", default=None)
    args = parser.parse_args()
    uvicorn.run(create_app(args.models_dir), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
