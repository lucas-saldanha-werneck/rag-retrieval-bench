import math
from collections import Counter
from typing import Protocol

from .text import char_ngrams, tokens


class Retriever(Protocol):
    def rank(self, query: str) -> list[str]: ...


class BM25:
    """Keyword retrieval (BM25, Okapi variant) over accent-folded tokens."""

    def __init__(self, docs: dict[str, str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = {i: Counter(tokens(t)) for i, t in docs.items()}
        self.len = {i: sum(c.values()) for i, c in self.tf.items()}
        self.avg = sum(self.len.values()) / max(len(self.len), 1)
        df: Counter[str] = Counter()
        for c in self.tf.values():
            df.update(c.keys())
        n = len(docs)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}

    def score(self, query: str, doc_id: str) -> float:
        tf, dl, s = self.tf[doc_id], self.len[doc_id], 0.0
        for w in set(tokens(query)):
            f = tf.get(w, 0)
            if f:
                s += self.idf[w] * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * dl / self.avg)
                )
        return s

    def rank(self, query: str) -> list[str]:
        return _sorted(self.tf, lambda i: self.score(query, i))


class NgramVectorIndex:
    """Dense-style retrieval without a model: TF-IDF vectors of character
    3-grams, cosine similarity. It is a vector index, but not a neural one:
    it matches word shapes ('cancelar' ~ 'cancelamento'), not meaning."""

    def __init__(self, docs: dict[str, str], n: int = 3):
        self.n = n
        counts = {i: Counter(char_ngrams(t, n)) for i, t in docs.items()}
        df: Counter[str] = Counter()
        for c in counts.values():
            df.update(c.keys())
        size = len(docs)
        self.idf = {g: math.log((1 + size) / (1 + d)) + 1 for g, d in df.items()}
        self.vec = {i: self._normalize(c) for i, c in counts.items()}

    def _normalize(self, counts: Counter) -> dict[str, float]:
        v = {g: (1 + math.log(c)) * self.idf.get(g, 0.0) for g, c in counts.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {g: x / norm for g, x in v.items()}

    def rank(self, query: str) -> list[str]:
        q = self._normalize(Counter(char_ngrams(query, self.n)))
        return _sorted(
            self.vec, lambda i: sum(w * self.vec[i].get(g, 0.0) for g, w in q.items())
        )


class Hybrid:
    """Reciprocal rank fusion of several retrievers: score = sum 1/(k + rank)."""

    def __init__(self, parts: list[Retriever], k: int = 60):
        self.parts, self.k = parts, k

    def rank(self, query: str) -> list[str]:
        fused: Counter[str] = Counter()
        for p in self.parts:
            for pos, doc_id in enumerate(p.rank(query), start=1):
                fused[doc_id] += 1.0 / (self.k + pos)
        return _sorted(fused, lambda i: fused[i])


def _sorted(ids, key) -> list[str]:
    # Ties broken by id so results are deterministic.
    return sorted(ids, key=lambda i: (-key(i), i))
