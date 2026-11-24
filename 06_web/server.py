"""Web chat with the trained models.

  python 06_web/server.py               # http://localhost:8000, MPS on the Mac
  python 06_web/server.py --port 8080

Loads every checkpoint found as <MODELS_DIR>/<name>/best.pt (or ckpt.pt when
there is no best.pt); MODELS_DIR defaults to 04_gpt/out. Serves the chat UI
from 06_web/static and the API under /api.

The models are base language models, not assistants: a "reply" is the model
continuing a play-like transcript. Each turn becomes a speech:

    YOU:
    Good morrow, sir.

    BARD:
    <the model writes here, until it ends the speech with a blank line>
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
sys.path.insert(0, os.path.join(HERE, "..", "04_gpt"))
from data import CharTokenizer  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402

DEFAULT_MODELS_DIR = os.path.join(HERE, "..", "04_gpt", "out")
STATIC_DIR = os.path.join(HERE, "static")
END_OF_TURN = "\n\n"  # a blank line ends a speech in the training text


def pick_device():
    if os.environ.get("DEVICE"):
        return os.environ["DEVICE"]
    return "mps" if torch.backends.mps.is_available() else "cpu"


class LoadedModel:
    def __init__(self, name, path, device):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        self.name, self.path = name, path
        self.model = GPT(GPTConfig(**ckpt["model_config"])).to(device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        # Stage 4 checkpoints carry a character vocabulary; the stage 5 BPE
        # model will plug in here with its own tokenizer type.
        self.tokenizer, self.tokenizer_type = CharTokenizer(ckpt["chars"]), "char"
        self.step, self.val_loss = ckpt.get("step"), ckpt.get("best_val")
        self.device = device

    @property
    def block_size(self):
        return self.model.config.block_size

    def info(self):
        cfg = self.model.config
        return {
            "name": self.name,
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

    def unknown_chars(self, text):
        return sorted(set(text) - set(self.tokenizer.stoi))


def load_models(models_dir, device):
    models = {}
    for d in sorted(glob.glob(os.path.join(models_dir, "*"))):
        path = next((p for p in (os.path.join(d, "best.pt"), os.path.join(d, "ckpt.pt")) if os.path.exists(p)), None)
        if path:
            name = os.path.basename(d)
            models[name] = LoadedModel(name, path, device)
            print(f"loaded {name}: {path} ({models[name].model.num_params() / 1e6:.2f}M params, {device})")
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


def build_prompt(req, block_size):
    """Transcript of the conversation ending with the model's speaker line.

    The model sees at most block_size tokens, so the oldest turns are dropped
    first; the newest turn is cut from the left if it alone is too long.
    Returns (prompt, number of turns kept).
    """
    header = f"{req.model_name}:\n"
    turns = [speech(req.user_name if m.role == "user" else req.model_name, m.content) for m in req.messages]
    kept, length = [], len(header)
    for t in reversed(turns):
        if length + len(t) > block_size and kept:
            break
        kept.insert(0, t)
        length += len(t)
    prompt = "".join(kept) + header
    return prompt[-block_size:], len(kept)


def create_app(models_dir=None, device=None):
    device = device or pick_device()
    models = load_models(models_dir or os.environ.get("MODELS_DIR", DEFAULT_MODELS_DIR), device)
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
        text = req.user_name + req.model_name + "".join(msg.content for msg in req.messages)
        unknown = m.unknown_chars(text)
        if unknown:
            raise HTTPException(400, {
                "error": f"the {req.model} model doesn't know these characters: {' '.join(map(repr, unknown))}",
                "unknown_chars": unknown,
            })
        prompt, turns_kept = build_prompt(req, m.block_size)
        context = torch.tensor([m.tokenizer.encode(prompt)], dtype=torch.long, device=device)
        generator = torch.Generator().manual_seed(req.seed) if req.seed is not None else None

        async def events():
            def send(obj):
                return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

            yield send({"type": "start", "context_tokens": context.size(1), "block_size": m.block_size,
                        "turns_kept": turns_kept, "turns_total": len(req.messages)})
            async with lock:
                stream = m.model.generate_stream(context, req.max_tokens, req.temperature, req.top_k, generator)
                reply, sent, n, t0, stop = "", 0, 0, time.time(), "max_tokens"
                while True:
                    if await request.is_disconnected():
                        stop = "cancelled"
                        break
                    token = await asyncio.to_thread(next, stream, None)  # keep the event loop free
                    if token is None:
                        break
                    reply += m.tokenizer.decode([token])
                    n += 1
                    end = reply.find(END_OF_TURN, sent)
                    if end != -1:  # the model finished its speech
                        if end > sent:
                            yield send({"type": "token", "text": reply[sent:end]})
                        stop = "end_of_turn"
                        break
                    # Hold back trailing newlines: they may turn out to be the end of the turn.
                    safe = len(reply.rstrip("\n"))
                    if safe > sent:
                        yield send({"type": "token", "text": reply[sent:safe]})
                        sent = safe
                stream.close()
            elapsed = time.time() - t0
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
