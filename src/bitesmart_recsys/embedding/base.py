"""Embedder protocol shared by the ONNX model and the deterministic test double."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Matrix = NDArray[np.float32]


class Embedder(Protocol):
    model_name: str
    dim: int

    def embed(self, texts: list[str]) -> Matrix:
        """Return an (n, dim) float32 matrix of L2-normalised vectors."""
        ...

    def count_tokens(self, texts: list[str]) -> int:
        """Number of model tokens in `texts`, used for the usage field of the API."""
        ...


def l2_normalize(x: Matrix) -> Matrix:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norms, 1e-12)).astype(np.float32)


def with_prefix(text: str, default_prefix: str) -> str:
    """E5 models need a 'query: ' or 'passage: ' prefix. Keep an explicit one, else add default."""
    if text.startswith(("query: ", "passage: ")):
        return text
    return default_prefix + text
