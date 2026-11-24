"""Turn the downloaded HTML (download.py) into one clean text corpus.

  python 05_custom/extract.py

Reads 05_custom/data/raw/, writes 05_custom/data/franko.txt (works separated
by blank lines, each starting with its title) and prints corpus statistics.
"""
import glob
import os
import re
import unicodedata
from collections import Counter
from html.parser import HTMLParser

from download import RAW_DIR

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "data", "franko.txt")

# Editor's notes about the source edition: "Повість друкується за виданням
# творів І. Франка в двадцяти томах, т. VI, К., 1951, с. 7—140."
# The old Galician spelling (Zhelekhivka: дїло, нїч, тїло) puts "ї" after
# consonants, which modern spelling never does. A couple of works on the site
# use it; they are left out so that each word has one spelling.
OLD_SPELLING = re.compile(r"[бвгґджзклмнпрстфхцчшщ]ї", re.I)
EDITORIAL = re.compile(r"(друкується|подається|публікується) за (виданням|текстом|першодруком|автографом|рукописом)", re.I)

# Latin letters that look like Cyrillic ones. Some texts on the site are typed
# with Latin "i" instead of Ukrainian "і" ("вiн"), which would split every such
# word into two spellings for the tokenizer.
HOMOGLYPHS = str.maketrans("iIaAeEoOpPcCxXyKMHTB", "іІаАеЕоОрРсСхХуКМНТВ")
CYRILLIC = re.compile(r"[а-яА-ЯіІїЇєЄґҐ']")
WORD = re.compile(r"[\w']+")

# Characters kept in the corpus; everything else is dropped after normalization.
ALLOWED = set(
    "абвгґдеєжзиіїйклмнопрстуфхцчшщьюяАБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯ"
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789 \n.,:;!?'«»—-()…"
)


class ParagraphParser(HTMLParser):
    """Collects the text of <article id="content">, split into paragraphs.

    Prose comes in <p> elements; verse comes as bare lines separated by <br>,
    with a blank line between stanzas. Everything inside <div>, <script>,
    <style> and <ins> is skipped: on this site those hold only ads, audiobook
    links and "read also" lists.
    """

    SKIP = ("div", "script", "style", "ins")
    BLOCK = ("p", "blockquote", "h1", "h2", "h3", "h4", "center")

    def __init__(self):
        super().__init__()
        self.in_article = False
        self.skip = 0  # nesting depth inside skipped elements
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag == "article" and dict(attrs).get("id") == "content":
            self.in_article = True
        elif not self.in_article:
            return
        elif tag in self.SKIP:
            self.skip += 1
        elif self.skip:
            return
        elif tag == "br":
            self.parts.append("\n")
        elif tag in self.BLOCK:
            self.parts.append("\n\n")

    def handle_startendtag(self, tag, attrs):
        if tag == "br" and self.in_article and not self.skip:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if not self.in_article:
            return
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag == "article":
            self.in_article = False
        elif tag in self.BLOCK and not self.skip:
            self.parts.append("\n\n")

    def handle_data(self, data):
        if self.in_article and not self.skip:
            # Source line breaks inside HTML are just formatting; <br> marks real ones.
            self.parts.append(data.replace("\r", "").replace("\n", " "))

    @property
    def paragraphs(self):
        text = "".join(self.parts)
        lines = "\n".join(line.strip() for line in text.split("\n"))
        return [p.strip() for p in re.split(r"\n{2,}", lines) if p.strip()]


def fix_homoglyphs(text):
    """Swap Latin look-alikes for Cyrillic in words that are mostly Cyrillic;
    real Latin words (German, Latin quotes) are left alone."""
    def fix(m):
        w = m.group(0)
        cyr = len(CYRILLIC.findall(w))
        return w.translate(HOMOGLYPHS) if cyr and cyr >= len(w) - cyr else w
    text = WORD.sub(fix, text)
    # The conjunction "і" typed as a lone Latin "i" has no Cyrillic in it:
    # fix it when the paragraph as a whole is Cyrillic.
    if len(CYRILLIC.findall(text)) > len(re.findall(r"[a-zA-Z]", text)):
        text = re.sub(r"\b[iI]\b", lambda m: m.group(0).translate(HOMOGLYPHS), text)
    return text


def normalize_quotes(text):
    # Typographic and straight double quotes -> «...»: a quote after a space,
    # line start or opening bracket opens, any other closes.
    text = re.sub(r'[“”„‟″]', '"', text)
    text = re.sub(r'(^|[\s(\[—-])"', r"\1«", text, flags=re.M)
    return text.replace('"', "»")


def clean_paragraph(p):
    p = unicodedata.normalize("NFC", p)
    # Editor's footnotes: "* (* Звідини — розвідка.)", "(* Краснопис (нім.).)"
    p = re.sub(r"\*?\s*\(\*(?:[^()]|\([^()]*\))*\)", "", p)  # one level of nested ()
    p = p.replace("*", "")
    p = p.replace(" ", " ").replace("́", "")  # nbsp, stress marks
    p = re.sub(r"[’ʼ`´‘]", "'", p)  # all apostrophe forms -> '
    p = re.sub(r"(?<=\s)[-–―](?=\s)|^[-–―](?=\s)|--", "—", p, flags=re.M)  # dashes
    p = p.replace("–", "—").replace("...", "…")
    p = normalize_quotes(fix_homoglyphs(p))
    p = "".join(ch for ch in p if ch in ALLOWED)
    p = re.sub(r"[ \t]+", " ", p)
    p = "\n".join(line.strip() for line in p.split("\n"))
    return re.sub(r"\n{2,}", "\n", p).strip()


def extract_work(tid):
    pages = sorted(glob.glob(os.path.join(RAW_DIR, f"{tid}_*.html")),
                   key=lambda f: int(f.rsplit("_", 1)[1].split(".")[0]))
    paragraphs = []
    for path in pages:
        parser = ParagraphParser()
        with open(path, encoding="utf-8") as f:
            parser.feed(f.read())
        paragraphs += [c for c in map(clean_paragraph, parser.paragraphs) if c and not EDITORIAL.search(c)]
    return paragraphs


def main():
    tids = sorted({int(os.path.basename(f).split("_")[0])
                   for f in glob.glob(os.path.join(RAW_DIR, "[0-9]*_*.html"))})
    # Collections repeat poems and fables that are also listed on their own:
    # keep only the first copy of each paragraph (short lines like "І" may repeat).
    seen, works, dropped = set(), [], 0
    for t in tids:
        paragraphs = []
        for p in extract_work(t):
            if len(p) > 40 and p in seen:
                dropped += len(p)
                continue
            seen.add(p)
            paragraphs.append(p)
        work = "\n\n".join(paragraphs)  # blank line between paragraphs / stanzas, as in Tiny Shakespeare
        if len(work) > 100 and len(OLD_SPELLING.findall(work)) <= len(work) / 1000:
            works.append(work)
    text = "\n\n\n".join(works) + "\n"  # two blank lines between works
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(text)

    counts = Counter(text)
    print(f"{len(works)} works, {len(text):,} characters, "
          f"{len(text.encode()) / 1e6:.1f} MB UTF-8, {len(counts)} distinct characters")
    print(f"dropped {dropped:,} characters of repeated paragraphs")
    print("rarest:", " ".join(f"{c!r}:{n}" for c, n in counts.most_common()[-15:]))
    print(f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
