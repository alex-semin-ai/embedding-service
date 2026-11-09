"""Application settings, read from BITESMART_* environment variables or a .env file."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BITESMART_", env_file=".env", extra="ignore")

    # Embedding model
    model_name: str = "intfloat/multilingual-e5-small"
    embedder_backend: Literal["onnx", "hash"] = "onnx"
    onnx_file: str = "onnx/model.onnx"
    embedding_dim: int = 384
    max_input_tokens: int = 256
    default_prefix: str = "query: "

    # Dynamic batching
    max_batch_size: int = 32
    max_wait_ms: float = 10.0
    max_queue_size: int = 1024

    # Server
    host: str = "0.0.0.0"  # noqa: S104 - the server is meant to run inside a container
    port: int = 8000
    api_key: str | None = None
    log_level: str = "INFO"

    # Catalog and index used by /v1/search and /v1/recommend
    catalog_path: str | None = None
    index_backend: str = "faiss_hnsw"
    qdrant_url: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
