"""Minimal Okapi BM25 with an inverted index — enough for a few hundred thousand chunks."""
import math
import re
from collections import Counter, defaultdict

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")
_STOP = set("""
a an and are as at be but by for from has have in into is it its of on or that the their them then there these
this to was were which with we our can not no such than also been being do does how what when where who why will
""".split())


def tokenize(text: str) -> list[str]:
    tokens = []
    for t in _TOKEN.findall(text.lower()):
        if t in _STOP or len(t) < 2:
            continue
        # cheap plural folding: "transformers" -> "transformer"
        if len(t) > 4 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        tokens.append(t)
    return tokens


class BM25:
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n_docs = len(corpus)
        self.doc_len = np.array([len(d) for d in corpus], dtype=np.float32)
        self.avg_len = float(self.doc_len.mean()) if self.n_docs else 0.0

        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, doc in enumerate(corpus):
            for term, tf in Counter(doc).items():
                self.postings[term].append((i, tf))

        self.idf = {
            term: math.log(1 + (self.n_docs - len(p) + 0.5) / (len(p) + 0.5))
            for term, p in self.postings.items()
        }

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(self.n_docs, dtype=np.float32)
        if not self.n_docs:
            return out
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / max(self.avg_len, 1e-9))
        for term in set(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in self.postings[term]:
                out[i] += idf * tf * (self.k1 + 1) / (tf + norm[i])
        return out
