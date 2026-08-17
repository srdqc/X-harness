"""Dependency-free BM25 keyword retrieval over small in-memory corpora.

A self-contained Okapi BM25 (no ``rank_bm25`` / ``jieba`` / ``nltk``) plus a
CJK-aware tokenizer, shared by anything that needs cheap keyword ranking over
a few hundred short documents — file-based skills, tool catalogs, etc.

Tokenization splits on word boundaries and treats each Chinese ideograph as a
single token, so Chinese queries match instead of collapsing to empty.
"""

from __future__ import annotations

import math
import re

# 匹配长度至少为 2 的字母数字串，或单个 CJK 表意字符。
# 在模块级预编译 ``re``，避免每次调用都初始化正则表达式。
_TOKEN_RE = re.compile(r"[a-z0-9]{2,}|[一-鿿]")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Okapi:
    """Minimal Okapi BM25 — same formula as rank_bm25 / Lucene defaults.

    ``score(D, Q) = Σ idf(q_i) * f(q_i, D) * (k1 + 1)
                          / (f(q_i, D) + k1 * (1 - b + b * |D| / avgdl))``
    """

    def __init__(
        self,
        tokenized_corpus: list[list[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.corpus_size = len(tokenized_corpus)
        self.doc_lens = [len(d) for d in tokenized_corpus]
        self.avgdl = sum(self.doc_lens) / self.corpus_size if self.corpus_size else 0.0

        self.doc_freqs: list[dict[str, int]] = []
        df: dict[str, int] = {}
        for doc in tokenized_corpus:
            freqs: dict[str, int] = {}
            for tok in doc:
                freqs[tok] = freqs.get(tok, 0) + 1
            self.doc_freqs.append(freqs)
            for tok in freqs:
                df[tok] = df.get(tok, 0) + 1

        n = self.corpus_size
        # ``log(1 + (N - n + 0.5) / (n + 0.5))``——Robertson-Spärck-Jones 公式
        # ``1 +`` 保护项可在 n ≈ N 时确保权重不为负数。
        self.idf = {term: math.log(1 + (n - count + 0.5) / (count + 0.5)) for term, count in df.items()}

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * self.corpus_size
        if not query_tokens or self.corpus_size == 0:
            return scores
        for term in query_tokens:
            idf = self.idf.get(term, 0.0)
            if idf <= 0.0:
                continue
            for i, freqs in enumerate(self.doc_freqs):
                f = freqs.get(term, 0)
                if f == 0:
                    continue
                dl = self.doc_lens[i]
                norm = self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
                scores[i] += idf * f * (self.k1 + 1) / (f + norm)
        return scores
