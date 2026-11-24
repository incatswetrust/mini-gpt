# Stage 5: own corpus and own tokenizer

The stage 4 GPT trained on a corpus built for it: the works of Ivan Franko
(7.7M characters, modern Ukrainian spelling), tokenized by a byte-level BPE
written from scratch. The same model is trained twice, on BPE tokens and on
characters, and the two are compared per character of the same text.

| File | What |
|---|---|
| `download.py` | Scrape Franko's works from ukrlib.com.ua (1 request/s, cached in `data/raw/`) |
| `extract.py` | HTML → one clean text: verse and prose, footnotes, homoglyphs, duplicates, old spelling |
| `bpe.py` | Byte-level BPE: training with incremental pair counts, encode / decode, save / load |
| `compare_tokenizers.py` | Characters per token: ours vs GPT-4 and GPT-4o, on Franko and on Shakespeare |
| `prepare.py` | Tokenize once into uint16 `.bin` files, interleaved train / val split |
| `train.py` | Train the stage 4 model on the `.bin` files, gradient accumulation, val in bits/char |
| `generate.py` | Generate a fixed number of characters from a trained model |
| `compare_runs.py` | Plot val bits/char of the two runs → `bpc_compare.png` |
| `test_custom.py` | BPE round trip, merge order, incremental = naive training, save / load, text cleaning |
| `NOTES.md` | Corpus cleaning, what BPE learned, why Cyrillic is expensive, BPE vs characters |

## Run

```bash
python franko/download.py                          # ~1 h
python franko/extract.py                           # -> data/franko.txt
python franko/bpe.py --vocab_size 4000             # 41 s -> data/bpe4000.json
python franko/compare_tokenizers.py                # needs tiktoken
python franko/prepare.py --tokenizer bpe4000
python franko/prepare.py --tokenizer char
python franko/train.py --data bpe4000              # ~75 min on the M2 Air
python franko/train.py --data char                 # ~67 min
python franko/compare_runs.py
python franko/generate.py --data bpe4000 --prompt "— Добрий вечір, куме!" --chars 500
```

`data/` and `out/` are git-ignored: the corpus is rebuilt by the scripts,
the trained models are in the GitHub release (`python web/weights.py download`).
