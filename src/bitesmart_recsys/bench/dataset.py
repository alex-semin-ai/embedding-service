"""Benchmark dataset. Clustered unit vectors with catalog metadata attached.

Real product embeddings are clustered by category, so the vectors are drawn around one
centroid per leaf category. Filters on metadata that correlates with the category
(vegan, allergens) are the hard case for filtered ANN, and the benchmark includes them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from bitesmart_recsys.indexes.base import Matrix
from bitesmart_recsys.indexes.filters import AnyOf, Condition, Filter
from bitesmart_recsys.recsys.synthetic import generate_products, iter_leaves


@dataclass
class BenchData:
    vectors: Matrix
    payloads: list[dict[str, Any]]
    queries: Matrix

    @property
    def n(self) -> int:
        return len(self.vectors)

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])


def make_clustered(
    n: int, dim: int = 384, n_queries: int = 200, seed: int = 42, noise: float = 0.9
) -> BenchData:
    rng = np.random.default_rng(seed)
    products = generate_products(n + n_queries, seed)
    leaves = {leaf.leaf: i for i, leaf in enumerate(iter_leaves())}
    centroids = rng.normal(size=(len(leaves), dim)).astype(np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

    leaf_idx = np.array([leaves[p.categories_hierarchy[-1]] for p in products])
    vectors = centroids[leaf_idx] + noise / np.sqrt(dim) * rng.normal(size=(len(products), dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    payloads = [p.to_payload() for p in products]
    return BenchData(
        vectors=np.ascontiguousarray(vectors[:n]),
        payloads=payloads[:n],
        queries=np.ascontiguousarray(vectors[n:]),
    )


def benchmark_filters() -> dict[str, Filter | None]:
    """Named filters from unfiltered to very selective."""
    return {
        "none": None,
        "country_finland": Filter(must=(Condition("countries_tags", "eq", "en:finland"),)),
        "no_milk_low_sugar": Filter(
            must=(Condition("nutriments.sugars_100g", "lte", 8.0),),
            must_not=(Condition("allergens_hierarchy", "any", ("en:milk",)),),
        ),
        "vegan_finland": Filter(
            must=(
                AnyOf(
                    (
                        Condition("ingredients_analysis_tags", "eq", "en:vegan"),
                        Condition("labels_tags", "eq", "en:vegan"),
                    )
                ),
                Condition("countries_tags", "eq", "en:finland"),
            )
        ),
        "rare_halal_vegan_low_sugar": Filter(
            must=(
                Condition("labels_tags", "eq", "en:halal"),
                Condition("labels_tags", "eq", "en:vegan"),
                Condition("nutriments.sugars_100g", "lte", 5.0),
                Condition("countries_tags", "eq", "en:germany"),
            )
        ),
    }
