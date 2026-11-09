from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from bitesmart_recsys.api import create_app
from bitesmart_recsys.embedding import HashEmbedder
from bitesmart_recsys.indexes.base import Matrix
from bitesmart_recsys.recsys import Catalog, Product, generate_products, write_products
from bitesmart_recsys.settings import Settings


@pytest.fixture(scope="session")
def embedder() -> HashEmbedder:
    return HashEmbedder(dim=64)


@pytest.fixture(scope="session")
def products() -> list[Product]:
    return generate_products(400, seed=7)


@pytest.fixture(scope="session")
def catalog(products: list[Product], embedder: HashEmbedder) -> Catalog:
    return Catalog.from_products(products, embedder)


@pytest.fixture(scope="session")
def random_data() -> tuple[Matrix, list[dict[str, object]], Matrix]:
    rng = np.random.default_rng(0)
    n, d = 1500, 32
    centers = rng.normal(size=(15, d))
    labels = rng.integers(0, 15, n)
    v = centers[labels] + 0.3 * rng.normal(size=(n, d))
    v = (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)
    payloads: list[dict[str, object]] = [
        {
            "id": str(i),
            "group": int(labels[i]),
            "tags": ["a", "b"] if i % 3 == 0 else ["c"],
            "nutriments": {"sugar": float(i % 50)},
        }
        for i in range(n)
    ]
    q = v[:25] + 0.02 * rng.normal(size=(25, d)).astype(np.float32)
    q = (q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float32)
    return v, payloads, q


@pytest.fixture
def catalog_file(tmp_path: Path, products: list[Product]) -> Path:
    path = tmp_path / "catalog.jsonl"
    write_products(products, path)
    return path


@pytest.fixture
def settings(catalog_file: Path) -> Settings:
    return Settings(
        embedder_backend="hash",
        embedding_dim=64,
        model_name="hash-embedder",
        catalog_path=str(catalog_file),
        index_backend="faiss_flat",
        max_wait_ms=2.0,
        api_key=None,
    )


@pytest.fixture
def client(settings: Settings, embedder: HashEmbedder) -> Iterator[TestClient]:
    with TestClient(create_app(settings, embedder)) as c:
        yield c
