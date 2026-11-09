from __future__ import annotations

from bitesmart_recsys.settings import Settings

from .base import Embedder
from .hash_embedder import HashEmbedder


def create_embedder(settings: Settings) -> Embedder:
    if settings.embedder_backend == "hash":
        return HashEmbedder(dim=settings.embedding_dim)
    from .onnx_embedder import OnnxEmbedder

    return OnnxEmbedder(
        model_name=settings.model_name,
        onnx_file=settings.onnx_file,
        max_tokens=settings.max_input_tokens,
    )
