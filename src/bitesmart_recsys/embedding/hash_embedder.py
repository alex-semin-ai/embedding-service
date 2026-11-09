"""Deterministic hashing embedder. Offline test double with real lexical similarity."""

from __future__ import annotations

import hashlib
import re
from itertools import pairwise

import numpy as np

from .base import Matrix, l2_normalize

_TOKEN = re.compile(r"\w+", re.UNICODE)


class HashEmbedder:
    """Signed feature hashing of word unigrams and bigrams into `dim` buckets.

    Texts sharing words get a high cosine similarity, which is enough to exercise the
    API, the indexes and the recommender without downloading a model.
    """

    def __init__(self, dim: int = 384, model_name: str = "hash-embedder") -> None:
        self.dim = dim
        self.model_name = model_name

    @staticmethod
    def _tokens(text: str) -> list[str]:
        words = _TOKEN.findall(text.lower())
        return words + [f"{a}_{b}" for a, b in pairwise(words)]

    def _bucket(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        return value % self.dim, 1.0 if (value >> 63) & 1 else -1.0

    def embed(self, texts: list[str]) -> Matrix:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in self._tokens(text):
                idx, sign = self._bucket(token)
                out[row, idx] += sign
        return l2_normalize(out)

    def count_tokens(self, texts: list[str]) -> int:
        return sum(len(_TOKEN.findall(t)) for t in texts)
