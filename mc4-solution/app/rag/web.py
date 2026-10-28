"""Web side of the Match 4 agent: HTML -> text, and a query-guided same-site crawler.

Dependency-free HTML parsing (html.parser) so nothing new has to be installed in
the image; fetching uses `requests` (already in the image).
"""

from __future__ import annotations

import heapq
import re
import time
from html.parser import HTMLParser
from urllib.parse import urljoin, urldefrag, urlparse

from .search import tokenize

SKIP_TAGS = {"script", "style", "noscript", "svg", "template", "iframe", "head"}
BLOCK_TAGS = {"p", "div", "section", "article", "li", "ul", "ol", "br", "tr", "table",
              "h1", "h2", "h3", "h4", "h5", "h6", "header", "footer", "main", "nav",
              "dl", "dt", "dd", "pre", "blockquote", "form", "label", "figure", "figcaption"}
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/126.0 Safari/537.36 AMD-League-Match4-Agent")


class _Extractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.links: list[tuple[str, str]] = []
        self.title = ""
        self._skip = 0
        self._in_title = False
        self._cell: list[str] | None = None
        self._row: list[str] | None = None
        self._a_href: str | None = None
        self._a_text: list[str] = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in SKIP_TAGS and tag != "head":
            self._skip += 1
            return
        if tag == "title":
            self._in_title = True
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell = []
        elif tag == "a" and a.get("href"):
            self._a_href, self._a_text = a.get("href"), []
        elif tag == "img" and a.get("alt"):
            self._emit(f" [{a.get('alt')}] ")
        elif tag == "meta" and a.get("name", "").lower() == "description" and a.get("content"):
            self.out.append("\n" + a["content"] + "\n")
        if tag in BLOCK_TAGS and self._cell is None:
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in SKIP_TAGS and tag != "head":
            self._skip = max(0, self._skip - 1)
            return
        if tag == "title":
            self._in_title = False
        if tag in ("td", "th") and self._cell is not None:
            txt = re.sub(r"\s+", " ", "".join(self._cell)).strip()
            if self._row is not None:
                self._row.append(txt)
            else:
                self.out.append(txt + " ")
            self._cell = None
        elif tag == "tr" and self._row is not None:
            cells = [c for c in self._row if c]
            if cells:
                self.out.append("\n" + " | ".join(cells) + "\n")
            self._row = None
        elif tag == "a" and self._a_href is not None:
            self.links.append((self._a_href, re.sub(r"\s+", " ", "".join(self._a_text)).strip()))
            self._a_href = None
        if tag in BLOCK_TAGS and self._cell is None:
            self.out.append("\n")

    def _emit(self, data):
        if self._cell is not None:
            self._cell.append(data)
        else:
            self.out.append(data)
        if self._a_href is not None:
            self._a_text.append(data)

    def handle_data(self, data):
        if self._in_title:
            self.title += data
            return
        if self._skip:
            return
        self._emit(data)


def html_to_text(html: str):
    """-> (title, text, links[(href, anchor_text)])"""
    p = _Extractor()
    try:
        p.feed(html)
        p.close()
    except Exception:
        pass
    text = "".join(p.out)
    lines = [re.sub(r"[ \t\r\f\v]+", " ", l).strip() for l in text.splitlines()]
    out, prev = [], None
    for l in lines:
        if l and l != prev:
            out.append(l)
        prev = l
    title = re.sub(r"\s+", " ", p.title).strip()
    body = "\n".join(out)
    if title:
        body = f"# {title}\n{body}"
    return title, body, p.links


def _norm_url(base: str, href: str) -> str | None:
    if not href or href.startswith(("mailto:", "tel:", "javascript:", "#", "data:")):
        return None
    u = urldefrag(urljoin(base, href))[0]
    pu = urlparse(u)
    if pu.scheme not in ("http", "https"):
        return None
    if re.search(r"\.(png|jpe?g|gif|svg|webp|ico|css|js|zip|gz|tar|mp4|mp3|woff2?|ttf|exe|dmg)$",
                 pu.path, re.I):
        return None
    return u


def _site(u: str) -> str:
    host = urlparse(u).netloc.lower().split(":")[0]
    parts = host.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def _score(query_toks: set, text: str) -> float:
    toks = tokenize(text)
    if not toks:
        return 0.0
    return sum(1.0 for t in toks if t in query_toks) / (1 + 0.15 * len(toks)) + \
        sum(1.0 for t in set(toks) if t in query_toks)


def crawl(start_url: str, query: str, budget_s: float = 15.0, max_pages: int = 25,
          per_request_timeout: float = 5.0, log=print):
    t0 = time.time()
    pages = _crawl(start_url, query, budget_s, max_pages, per_request_timeout, log)
    if not pages and start_url.startswith("https://") and time.time() - t0 < budget_s - 2:
        pages = _crawl("http://" + start_url[8:], query, budget_s - (time.time() - t0),
                       max_pages, per_request_timeout, log)
    return pages


def _crawl(start_url, query, budget_s, max_pages, per_request_timeout, log):
    """Best-first crawl of the start page's site, following links whose anchor text and
    URL best match the query. Returns [(url, text)] in fetch order."""
    import requests

    t0 = time.time()
    q = set(tokenize(query))
    site = _site(start_url)
    seen, pages = set(), []
    heap = [(-1e9, 0, start_url)]
    n = 0
    sess = requests.Session()
    sess.headers.update({"User-Agent": UA, "Accept": "text/html,application/xhtml+xml,*/*"})
    while heap and len(pages) < max_pages and time.time() - t0 < budget_s:
        _, _, url = heapq.heappop(heap)
        if url in seen:
            continue
        seen.add(url)
        left = budget_s - (time.time() - t0)
        if left < 1.0:
            break
        try:
            r = sess.get(url, timeout=min(per_request_timeout, left), allow_redirects=True)
            ctype = r.headers.get("content-type", "")
            if r.status_code >= 400 or ("html" not in ctype and "text" not in ctype):
                continue
            r.encoding = r.encoding or r.apparent_encoding
            title, text, links = html_to_text(r.text)
        except Exception as e:
            log(f"[crawl] fail {url}: {e!r}")
            continue
        final = r.url or url
        pages.append((final, text))
        log(f"[crawl] {len(pages)} {final} ({len(text)} chars, {time.time()-t0:.1f}s)")
        for href, anchor in links:
            u = _norm_url(final, href)
            if not u or u in seen or _site(u) != site:
                continue
            path_words = re.sub(r"[/_\-.]+", " ", urlparse(u).path)
            s = 3 * _score(q, anchor) + _score(q, path_words)
            n += 1
            heapq.heappush(heap, (-s, n, u))
    return pages
