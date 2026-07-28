"""Scrape external Gunn-related websites into RAG documents.

Reads a curated URL list (links.txt) and scrapes the pages worth indexing,
plus the Wikipedia article. Uses trafilatura to pull main content and drop
nav/footer boilerplate.

Skipped automatically:
  - gunnwiki.org        (already ingested cleanly via the MediaWiki API)
  - gunnwatt.web.app    (JS app; its data is already ingested from WATT's JSON)
  - nutrislice/classlink (menu via API; classlink is a login wall)
  - junk/test/stale pages, and district admin-noise on www.pausd.org
"""
import os
import re
import sys
import time
from urllib.parse import urlparse

import requests
import trafilatura

import config

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36")

LINKS_FILE = os.path.join(config.HERE, "links.txt")

SKIP_HOSTS = {"gunnwiki.org", "gunnwatt.web.app", "pausd.nutrislice.com",
              "launchpad.classlink.com"}
# Low-value / junk / stale (date-specific content that goes out of date).
SKIP_PATH = re.compile(
    r"(__trashed|/test-|/test-page|/test-tables|/test-components|/components/|"
    r"verify-login|/shop|/author/|hudl|/fan-zone|/gallery|"
    r"interactive-subscription-map|picture-of-us|"
    r"/schedule|/calendar|/scores|standings|dismissal|sync-schedule)", re.I)
# District (www.pausd.org) admin noise not useful to a Gunn student.
SKIP_PAUSD = re.compile(
    r"(committees-task-forces|board-of-education|superintendent|/funding|/hr/|"
    r"/news/|parcel-tax|/lcap|/promise|/enrollment|school-construction|"
    r"/schools/|public-data|/the-team)", re.I)

SOURCE_BY_HOST = {
    "gunn.pausd.org": "gunn-official",
    "www.gunnathletics.com": "athletics",
    "www.gunnsec.org": "student-activities",
    "www.pausd.org": "pausd-district",
}


def _keep(url):
    host = urlparse(url).netloc
    if host in SKIP_HOSTS:
        return False
    if SKIP_PATH.search(url):
        return False
    if host == "www.pausd.org" and SKIP_PAUSD.search(url):
        return False
    return host in SOURCE_BY_HOST


def _fetch(url):
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=20, allow_redirects=True)
        if r.status_code == 200 and "text/html" in r.headers.get("content-type", ""):
            return r.text
    except Exception:
        pass
    return None


def _extract(url, html, source):
    text = trafilatura.extract(html, include_comments=False, include_tables=True,
                               favor_precision=True, url=url)
    if not text or len(text) < 200:
        return None
    md = trafilatura.extract_metadata(html)
    title = (md.title if md and md.title else "")
    if not title:
        m = re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I)
        title = re.sub(r"\s+", " ", m.group(1)).strip() if m else url
    return {
        "id": f"{source}-{abs(hash(url)) % (10 ** 9)}",
        "title": title[:120],
        "url": url,
        "source": source,
        "text": f"{title}\n\n{' '.join(text.split())}",
    }


def scrape_list(path=LINKS_FILE, delay=0.2):
    if not os.path.exists(path):
        print(f"  (no {path}; skipping curated scrape)", file=sys.stderr)
        return []
    urls, seen = [], set()
    for line in open(path):
        u = line.strip()
        if u and u not in seen and _keep(u):
            seen.add(u)
            urls.append(u)
    print(f"  scraping {len(urls)} curated pages...", file=sys.stderr)
    docs = []
    for i, url in enumerate(urls):
        html = _fetch(url)
        if html:
            doc = _extract(url, html, SOURCE_BY_HOST[urlparse(url).netloc])
            if doc:
                docs.append(doc)
        if (i + 1) % 25 == 0:
            print(f"    {i + 1}/{len(urls)} fetched, {len(docs)} kept...", file=sys.stderr)
        time.sleep(delay)
    return docs


def wikipedia_docs():
    r = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={"action": "query", "prop": "extracts", "explaintext": "1",
                "redirects": "1", "titles": "Gunn High School", "format": "json"},
        headers={"User-Agent": UA}, timeout=30,
    )
    r.raise_for_status()
    docs = []
    for page in r.json().get("query", {}).get("pages", {}).values():
        extract = (page.get("extract") or "").strip()
        if len(extract) < 100:
            continue
        docs.append({
            "id": f"wikipedia-{page.get('pageid')}",
            "title": "Wikipedia: Gunn High School",
            "url": "https://en.wikipedia.org/wiki/Gunn_High_School",
            "source": "wikipedia",
            "text": f"Wikipedia article on Gunn High School:\n\n{extract}",
        })
    return docs


def build_scrape_docs():
    docs = []
    for label, fn in [("curated websites", scrape_list), ("wikipedia", wikipedia_docs)]:
        try:
            got = fn()
            print(f"  {label}: {len(got)} docs", file=sys.stderr)
            docs += got
        except Exception as e:
            print(f"  !! {label} failed: {e}", file=sys.stderr)
    return docs


if __name__ == "__main__":
    d = build_scrape_docs()
    print(f"\nTOTAL scraped: {len(d)} docs", file=sys.stderr)
    from collections import Counter
    for src, n in Counter(x["source"] for x in d).most_common():
        print(f"  {src}: {n}", file=sys.stderr)
