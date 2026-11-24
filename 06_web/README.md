# Stage 6: web chat

A browser chat with the trained models: conversation in the middle, settings
in the sidebar, replies streamed token by token.

The models are **base language models**, not assistants. A reply is the model
continuing a play: each message becomes a speech under a speaker name, and
the reply ends when the model closes its speech with a blank line.

```
YOU:
Good morrow, sir. What news from Verona?

BARD:
<the model writes here>
```

## Run locally (MPS on the Mac)

```bash
pip install -r requirements.txt
python 06_web/server.py            # http://localhost:8000
```

The server loads every `04_gpt/out/<name>/best.pt` (or `ckpt.pt` when there
is no `best.pt`) at startup; set `MODELS_DIR` to look elsewhere and `DEVICE`
to force `cpu`/`mps`.

## Run in Docker (two containers)

```bash
docker compose up --build          # http://localhost:8080
```

```
browser ──► web (nginx)  static UI; /api/* proxied without buffering (SSE)
               │
               ▼
            api (FastAPI + torch CPU)  checkpoints mounted read-only from 04_gpt/out
```

- Docker on macOS runs Linux in a VM with no access to the Apple GPU, so the
  `api` container runs on CPU. For one-token-at-a-time generation with a 10M
  model that's about as fast as MPS; training stays outside Docker.
- Checkpoints are mounted, not copied into the image: retrain and restart
  `api`, no rebuild needed.
- The image installs the CPU-only torch wheel (~200 MB instead of ~2 GB).

## API

`GET /api/models`: loaded models with parameter count, tokenizer, vocabulary
size, `block_size` and best val loss.

`POST /api/generate` streams Server-Sent Events:

```json
{"model": "main", "messages": [{"role": "user", "content": "Good morrow."}],
 "temperature": 0.8, "top_k": null, "max_tokens": 300, "seed": null,
 "user_name": "YOU", "model_name": "BARD"}
```

```
data: {"type": "start", "context_tokens": 34, "block_size": 256, "turns_kept": 1, "turns_total": 1}
data: {"type": "token", "text": "I"}
...
data: {"type": "done", "stop_reason": "end_of_turn", "tokens": 137, "tokens_per_s": 66.5, "context_used": 171, "block_size": 256}
```

- **Context**: the model sees `block_size` tokens. The oldest turns are dropped
  first; a single turn longer than that is cut from the left.
- **Vocabulary**: the character model knows 65 symbols. Anything else
  (Cyrillic, digits other than 3, emoji) gets a `400` with the list of
  unknown characters; the UI puts the message back into the input box.
- **Stop**: closing the request (the Stop button aborts the fetch) ends
  generation on the server; `stop_reason` is `cancelled`.
- `stop_reason` is `end_of_turn` when the model ends its speech, otherwise
  `max_tokens`.

## UI

Plain HTML/CSS/JS in `static/`, no build step. Nord palette, light and dark
themes (follows the system, toggle in the sidebar). Enter sends, Shift+Enter
adds a line. Chats and settings are kept in `localStorage`. Each reply shows
its token count, speed and why it stopped; the bar under the input shows how
much of the context window the last request used.

## Tests

```bash
cd 06_web && python -m pytest -q
```

They build a tiny random model in the real checkpoint format and check the
model list, a streamed generation (end of turn never sent, seed repeatable),
the vocabulary error, an unknown model, and how the prompt drops old turns.
