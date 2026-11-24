from bpe import SPLIT, BPETokenizer
from extract import clean_paragraph

TEXT = ("Малий Мирон один сидить спокійний, майже веселий у лавці. "
        "Він дивується, чому се нараз так тихо стало в класі! М'ясо, п'ять — «так». "
        "Schon Schreiben 1916 р.\n\nНовий рядок.")


def test_roundtrip():
    tok = BPETokenizer.train(TEXT * 3, 300)
    for s in (TEXT, "", "ґанок їжак є", "emoji 🙂 and 中文", "  spaces\n\n\ttabs"):
        assert tok.decode(tok.encode(s)) == s


def test_merges_shorten_text_and_follow_training_order():
    tok = BPETokenizer.train(TEXT * 3, 320)
    assert tok.vocab_size == 320
    assert len(tok.encode(TEXT)) < len(TEXT.encode("utf-8"))
    # Every merged token is built from tokens that existed before it.
    for (a, b), idx in tok.merges.items():
        assert a < idx and b < idx


def test_naive_and_incremental_training_agree():
    """The incremental pair-count update must pick the same merges as recounting from scratch."""
    from collections import Counter
    from bpe import merge
    vocab_size = 290
    ids_per_chunk = [list(c.encode()) for c in SPLIT.findall(TEXT)]
    merges = {}
    for new_id in range(256, vocab_size):
        counts = Counter(p for ids in ids_per_chunk for p in zip(ids, ids[1:]))
        pair = max(counts, key=lambda p: (counts[p], -p[0], -p[1]))
        merges[pair] = new_id
        ids_per_chunk = [merge(ids, pair, new_id) for ids in ids_per_chunk]
    assert BPETokenizer.train(TEXT, vocab_size).merges == merges


def test_save_load(tmp_path):
    tok = BPETokenizer.train(TEXT, 300)
    tok.save(tmp_path / "t.json")
    tok2 = BPETokenizer.load(tmp_path / "t.json")
    assert tok2.encode(TEXT) == tok.encode(TEXT)


def test_split_keeps_apostrophe_words():
    assert " п'ять" in SPLIT.findall("та п'ять")


def test_clean_paragraph():
    raw = ('Вiн не помiтується * (* П о м i т у в а т и с я — цуратися.) хоч тростiвки i нiколи - '
           '"красне писання" (* Краснопис, калiграфiя (нiм.).) Schon Schreiben...')
    assert clean_paragraph(raw) == "Він не помітується хоч тростівки і ніколи — «красне писання» Schon Schreiben…"
