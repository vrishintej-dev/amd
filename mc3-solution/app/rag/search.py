"""Small dependency-free BM25 over chunks, plus identifier helpers."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

STOP = set("""
a an the of in on at to for from by with and or is are was were be been being
what which who whom whose when where why how does do did has have had this that
these those it its as into than then there their they them i we you our your
can could should would will shall may might must not no yes per s about over
under up down out if so such any all each some more most other same very
""".split())

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-_.#/][A-Za-z0-9]+)*#?")
# identifiers: tokens mixing letters and digits, or codes like E7731, ORR-1847, REV-C2, 4.3.2
IDENT_RE = re.compile(r"\b(?=[A-Za-z0-9._#-]*\d)(?=[A-Za-z0-9._#-]*[A-Za-z])[A-Za-z0-9]+(?:[-_.][A-Za-z0-9]+)*\b"
                      r"|\b\d+(?:\.\d+){2,}\b")


def _stem(w: str) -> str:
    for suf in ("ies", "ing", "ed", "es", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


def tokenize(text: str) -> list[str]:
    out = []
    for m in WORD_RE.finditer(text.lower()):
        tok = m.group(0)
        parts = re.split(r"[-_.#/]", tok)
        if len(parts) > 1:
            out.append(tok.rstrip("#"))           # whole identifier, e.g. orr-1847
            out.append(re.sub(r"[-_.#/]", "", tok))  # squashed, e.g. orr1847
        for p in parts:
            if p and p not in STOP:
                out.append(_stem(p))
    return out


def identifiers(text: str) -> set[str]:
    return {m.group(0).upper() for m in IDENT_RE.finditer(text) if len(m.group(0)) >= 3}


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.4, b: float = 0.6):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = (sum(self.lens) / len(self.lens)) if self.lens else 1.0
        df = defaultdict(int)
        for d in self.docs:
            for t in d:
                df[t] += 1
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str, boost: dict[str, float] | None = None) -> list[float]:
        q = Counter(tokenize(query))
        if boost:
            for t, w in boost.items():
                for tok in tokenize(t):
                    q[tok] += w
        res = []
        for d, L in zip(self.docs, self.lens):
            s = 0.0
            for t, qf in q.items():
                f = d.get(t)
                if not f:
                    continue
                idf = self.idf.get(t, 0.0)
                s += qf * idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * L / self.avg))
            res.append(s)
        return res
