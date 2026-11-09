"""Product catalog with embeddings, loadable from JSONL."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from bitesmart_recsys.embedding import Embedder
from bitesmart_recsys.indexes.base import Matrix

from .models import Product

logger = logging.getLogger(__name__)


class Catalog:
    def __init__(self, products: list[Product], vectors: Matrix) -> None:
        if len(products) != len(vectors):
            raise ValueError("products and vectors must have the same length")
        self.products = products
        self.vectors = vectors
        self.payloads: list[dict[str, Any]] = [p.to_payload() for p in products]
        self._row_by_id = {p.id: i for i, p in enumerate(products)}

    def __len__(self) -> int:
        return len(self.products)

    def row_of(self, product_id: str) -> int:
        try:
            return self._row_by_id[product_id]
        except KeyError:
            raise KeyError(f"unknown product id {product_id}") from None

    @classmethod
    def from_products(
        cls, products: list[Product], embedder: Embedder, batch_size: int = 64
    ) -> Catalog:
        texts = [p.to_passage_text() for p in products]
        chunks = [
            embedder.embed(texts[i : i + batch_size]) for i in range(0, len(texts), batch_size)
        ]
        vectors = np.concatenate(chunks) if chunks else np.zeros((0, embedder.dim), np.float32)
        logger.info("catalog embedded", extra={"products": len(products), "dim": embedder.dim})
        return cls(products, vectors.astype(np.float32))


def read_products(path: str | Path) -> list[Product]:
    with Path(path).open(encoding="utf-8") as fh:
        return [Product.model_validate_json(line) for line in fh if line.strip()]


def write_products(products: Iterable[Product], path: str | Path) -> int:
    count = 0
    with Path(path).open("w", encoding="utf-8") as fh:
        for product in products:
            fh.write(json.dumps(product.model_dump(), ensure_ascii=False) + "\n")
            count += 1
    return count
