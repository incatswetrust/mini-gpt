# Stage 5 notes

## 5.1 Corpus: Ivan Franko

```bash
python 05_custom/download.py   # ~1 h: 379 works, 1073 pages, 1 request/s, cached in data/raw/
python 05_custom/extract.py    # -> data/franko.txt
```

Source: ukrlib.com.ua, which has Franko's works in modern spelling. Franko died
in 1916, so the texts are in the public domain. The corpus itself is not
committed (`data/` is git-ignored); the scripts rebuild it.

**Result: 289 works, 7 701 328 characters (13.8 MB UTF-8), 143 distinct
characters** — 7× Tiny Shakespeare.

What cleaning had to deal with:

| Problem | Example | Fix |
|---|---|---|
| Verse has no `<p>`: bare lines with `<br>` | 234 works (all poetry, *Мойсей*) came out empty at first | take all text of the article, skip every `<div>` (ads, audiobook links, "read also") |
| Latin look-alikes in Cyrillic words | `вiн`, `тiльки`, lone `i` for *і* | swap Latin → Cyrillic in mostly-Cyrillic words; lone `i` in Cyrillic paragraphs |
| Editor's footnotes inline | `* (* Звідини — розвідка.)` | regex, one level of nested `()` |
| Editor's notes on the source edition | *Повість друкується за виданням творів…* | drop such paragraphs |
| Collections repeat poems listed on their own | *Semper tiro* and *Semper tiro (збірка)* | drop repeated paragraphs > 40 chars (208K chars, 2.6%) |
| Old Galician spelling in 2 works | *дїло, приняв ся, по роцї* | drop works with "ї" after consonants (0.7%) |
| Typography | `"…"`, `–`, `...`, `’ ʼ`, stress marks | `«…»`, `—`, `…`, `'`, removed |

Paragraphs and stanzas are separated by a blank line, works by two — the same
layout as Tiny Shakespeare, so the model can learn where a speech or stanza
ends.

## 5.2 BPE tokenizer

```bash
python 05_custom/bpe.py --vocab_size 4000   # 41 s -> data/bpe4000.json
python 05_custom/compare_tokenizers.py
```

Byte-level BPE from scratch (`bpe.py`): start from the 256 byte values,
repeatedly merge the most frequent adjacent pair. Text is first split into
chunks (a word with its leading space, a number, punctuation, whitespace), so
merges never cross word boundaries.

Naive training recounts all pairs after every merge. Here identical chunks
are counted once with their frequency, and after a merge only the chunks
containing that pair are updated — 3744 merges over 7.7M characters take
41 s in pure Python. A test checks that this gives exactly the same merges
as the naive version (ties are broken by the smallest ids, so the result
is deterministic).

What it learned: first common letter pairs, then endings and short words
(`ані`, `увався`), then whole frequent words. The longest tokens are whole
words with their leading space: `' розуміється'`, `' канцелярії'`,
`' робітників'`, and names from the novels — `' Борислава'`, `' Стальський'`,
`' Панталаха'`.

**2 637 017 tokens for the corpus: 2.92 characters per token.**

### Why Cyrillic is "expensive" for GPT-4

The same sentence (113 characters, 204 bytes in UTF-8):

```
ours (4000)   46 tokens: С|ум|но| і| неп|рив|іт|но| тепер| в| нашій| Т|ухоль|щ|ині|!| Правда|,| …
GPT-4 (100k)  74 tokens: С|ум|но| і| н|еп|р|ив|і|т|но| т|еп|ер| в| на|ш|і|й| Т|у|х|оль|щ|ин|і|!| П|рав|да|,| …
GPT-4o (200k) 46 tokens: С|ум|но| і| непр|ив|іт|но| теп|ер| в| наш|ій| Т|ух|оль|щ|ині|!| Правда|,| …
```

Characters per token on 1M characters (more is cheaper):

| | Franko (uk) | Shakespeare (en) |
|---|---|---|
| ours, 4k vocab | **2.85** | 1.19 |
| GPT-4, 100k | 1.69 | **3.70** |
| GPT-4o, 200k | 2.64 | 3.75 |

- Every Cyrillic letter is 2 bytes in UTF-8, so a byte-level tokenizer
  starts at 2× the cost of English before any merges.
- BPE spends its vocabulary on what is frequent in *its* training data.
  GPT-4's was mostly English: 100k tokens, yet few Ukrainian merges, so
  Ukrainian costs 1.69 chars/token against 3.70 for English — **2.2× more
  tokens for the same amount of text**, i.e. 2.2× the price and half the
  effective context. Even `її` is 4 tokens: each `ї` is split into its two
  bytes.
- GPT-4o doubled the vocabulary with more multilingual data and gets close
  to our tokenizer on Ukrainian.
- Our 4k tokenizer beats both on Franko, because all 3744 merges went to
  Ukrainian, but it's useless for English (1.19: barely better than bytes).
  The Franko number is slightly flattering — the tokenizer was trained on
  this text — but the gap to GPT-4 is far bigger than that effect.
