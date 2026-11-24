"""Download Ivan Franko's works from ukrlib.com.ua.

  python franko/download.py            # all works, ~15-20 min (1 request/s)
  python franko/download.py --limit 5  # first 5 works, for a quick check

Franko died in 1916, so the texts are in the public domain; ukrlib.com.ua
has them in modern spelling. Raw HTML pages are cached in
franko/data/raw/, so a rerun downloads only what's missing, and
extract.py can be changed and rerun without downloading again.
"""
import argparse
import os
import re
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "data", "raw")
BASE = "https://www.ukrlib.com.ua/books/"
AUTHOR_ID = 2  # Ivan Franko on ukrlib
USER_AGENT = "mini-gpt-corpus-builder/0.1"
DELAY = 1.0  # seconds between requests, to be polite to the site


def fetch(url, retries=4):
    """GET a page and decode it from cp1251 (the site's encoding)."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
            time.sleep(DELAY)
            return data.decode("cp1251", errors="replace")
        except (urllib.error.URLError, TimeoutError) as e:
            wait = 5 * (attempt + 1)
            print(f"  {url}: {e}, retry in {wait} s")
            time.sleep(wait)
    raise RuntimeError(f"failed to fetch {url}")


def cached(name, url):
    """Return the page from the cache, downloading it if needed."""
    path = os.path.join(RAW_DIR, name)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    html = fetch(url)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return html


def list_works():
    """All (tid, title) pairs from the author's catalogue pages."""
    works, page = {}, 1
    while True:
        html = cached(f"author_{page}.html", f"{BASE}author.php?id={AUTHOR_ID}&page={page}")
        found = dict(re.findall(r'href="printit\.php\?tid=(\d+)">([^<]+)</a>', html))
        # Past the last page the site repeats works already listed, so stop at
        # the first page with nothing new.
        if not found.keys() - works.keys():
            break
        works.update(found)
        page += 1
    return sorted(works.items(), key=lambda w: int(w[0]))


def download_work(tid):
    """Download every page of one work; returns the number of pages."""
    page = 1
    while True:
        url = f"{BASE}printit.php?tid={tid}" + (f"&page={page}" if page > 1 else "")
        html = cached(f"{tid}_{page}.html", url)
        # The pager shows a sliding window of links, so just check for the next one.
        if f"tid={tid}&amp;page={page + 1}" not in html:
            return page
        page += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(RAW_DIR, exist_ok=True)
    works = list_works()[: args.limit]
    print(f"{len(works)} works")
    for i, (tid, title) in enumerate(works, 1):
        pages = download_work(tid)
        print(f"[{i}/{len(works)}] {tid} {title} ({pages} p.)", flush=True)


if __name__ == "__main__":
    main()
