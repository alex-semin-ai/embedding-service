from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bitesmart_recsys.indexes import Filter, create_index
from bitesmart_recsys.indexes.base import IdMatrix, Matrix, VectorIndex
from bitesmart_recsys.recsys import (
    Catalog,
    RecommendationConfig,
    Recommender,
    load_recommendation_config,
)
from bitesmart_recsys.recsys.constraints import build_filter, nutriment_key, verify_hard_constraints
from bitesmart_recsys.recsys.rerank import Reranker, hierarchy_transition_score
from bitesmart_recsys.recsys.scoring import (
    Candidate,
    CandidateScorer,
    ScoredCandidate,
    ScoringFeatures,
    normalize_scores,
    normalized_entropy,
)


class LeakyIndex(VectorIndex):
    """Ignores every filter. Used to prove verification catches a misbehaving backend."""

    name = "leaky"

    def _build(self, vectors: Matrix, payloads: Sequence[Mapping[str, Any]]) -> None:
        self._v = vectors

    def search(self, queries: Matrix, k: int, flt: Filter | None = None) -> tuple[IdMatrix, Matrix]:
        scores = queries @ self._v.T
        ids = np.argsort(-scores, axis=1)[:, :k]
        return ids.astype(np.int64), np.take_along_axis(scores, ids, axis=1).astype(np.float32)


def config_with(**hard: Any) -> RecommendationConfig:
    cfg = RecommendationConfig()
    hc = cfg.candidate_search.hard_constraints
    for key, value in hard.items():
        if key == "vegan":
            hc.dietary.vegan = value
        else:
            setattr(hc, key, value)
    return cfg


@pytest.fixture(scope="module")
def recommender(catalog: Catalog) -> Recommender:
    return Recommender.build(catalog, create_index("faiss_flat"))


def pick(catalog: Catalog, leaf: str) -> str:
    return next(p.id for p in catalog.products if p.categories_hierarchy[-1] == leaf)


@pytest.mark.parametrize("allergen", ["en:milk", "en:gluten", "en:nuts"])
def test_forbidden_allergens_never_appear(
    recommender: Recommender, catalog: Catalog, allergen: str
) -> None:
    cfg = config_with(forbidden_allergens=[allergen])
    for leaf in ("en:fruit-yogurts", "en:muesli", "en:chocolate-bars"):
        result = recommender.recommend(pick(catalog, leaf), cfg, top_n=20)
        for item in result.items:
            payload = catalog.payloads[catalog.row_of(item.id)]
            assert allergen not in payload["allergens_hierarchy"]


def test_vegan_and_same_category_constraints(recommender: Recommender, catalog: Catalog) -> None:
    source = pick(catalog, "en:oat-drinks")
    cfg = config_with(vegan=True, same_category_as_source=True)
    result = recommender.recommend(source, cfg, top_n=10)
    assert result.items
    for item in result.items:
        payload = catalog.payloads[catalog.row_of(item.id)]
        assert "en:vegan" in payload["labels_tags"] + payload["ingredients_analysis_tags"]
        assert payload["categories_hierarchy"][0] == "en:plant-based-foods"
    assert source not in {i.id for i in result.items}


def test_source_relative_nutrient_constraints(recommender: Recommender, catalog: Catalog) -> None:
    source_id = pick(catalog, "en:chocolate-bars")
    source = catalog.payloads[catalog.row_of(source_id)]["nutriments"]
    cfg = RecommendationConfig()
    cfg.candidate_search.hard_constraints.source_nutrients.enforce = True
    for item in recommender.recommend(source_id, cfg, top_n=20).items:
        assert item.nutriments["sugars_100g"] <= source["sugars_100g"]
        assert item.nutriments["proteins_100g"] >= source["proteins_100g"]


def test_verification_blocks_a_backend_that_ignores_filters(catalog: Catalog) -> None:
    leaky = Recommender.build(catalog, LeakyIndex())
    cfg = config_with(forbidden_allergens=["en:milk"])
    result = leaky.recommend(pick(catalog, "en:fruit-yogurts"), cfg, top_n=10)
    assert result.stats["retrieved"] > result.stats["after_verification"]
    for item in result.items:
        assert "en:milk" not in catalog.payloads[catalog.row_of(item.id)]["allergens_hierarchy"]


def test_unknown_source_raises(recommender: Recommender) -> None:
    with pytest.raises(KeyError, match="unknown product"):
        recommender.recommend("missing")


def test_results_are_sorted_and_explained(recommender: Recommender, catalog: Catalog) -> None:
    result = recommender.recommend(pick(catalog, "en:colas"), top_n=7)
    scores = [i.score for i in result.items]
    assert scores == sorted(scores, reverse=True)
    assert all(i.reasons for i in result.items)
    assert result.stats["returned"] == len(result.items) <= 7


def test_search_vector_applies_min_score(recommender: Recommender, catalog: Catalog) -> None:
    cfg = RecommendationConfig()
    cfg.similarity.min_score = 0.999
    row = 5
    hits = recommender.search_vector(catalog.vectors[row], 5, cfg)
    assert [h.id for h in hits] == [catalog.products[row].id]


def test_build_filter_contents(catalog: Catalog) -> None:
    cfg = config_with(
        forbidden_allergens=["en:milk"],
        forbidden_traces=["en:nuts"],
        nutrient_caps={"sugar": 5},
    )
    cfg.candidate_search.hard_constraints.country_filter.enabled = True
    cfg.candidate_search.hard_constraints.country_filter.country_tag = "en:finland"
    cfg.candidate_search.exclude_ids = ["42"]
    flt = build_filter(cfg, None, source_id="7")
    assert len(flt.must_not) == 3  # ids, allergens, traces
    assert len(flt.must) == 2  # country, sugar cap
    assert build_filter(RecommendationConfig(), None).is_empty


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("sugar", "nutriments.sugars_100g"),
        ("nutriments.salt", "nutriments.salt_100g"),
        ("kcal", "nutriments.energy-kcal_100g"),
        ("fiber_100g", "nutriments.fiber_100g"),
    ],
)
def test_nutriment_key_aliases(name: str, expected: str) -> None:
    assert nutriment_key(name) == expected


def test_verify_removes_violations() -> None:
    cfg = config_with(forbidden_allergens=["en:milk"])
    good = Candidate(id="1", similarity=0.9, payload={"id": "1", "allergens_hierarchy": []})
    bad = Candidate(id="2", similarity=0.9, payload={"id": "2", "allergens_hierarchy": ["en:milk"]})
    assert verify_hard_constraints([good, bad], cfg, None) == [good]


@pytest.mark.parametrize(
    ("src", "cand", "expected_score", "reason"),
    [
        (["a", "b"], ["a", "b"], 1.0, "exact_match"),
        (["a", "b"], ["x", "y"], 0.0, "no_shared_category"),
        (["a", "b"], ["a", "y"], 0.675, "partial_overlap"),
        ([], ["a"], 0.0, "missing_hierarchy"),
    ],
)
def test_hierarchy_transition_score(
    src: list[str], cand: list[str], expected_score: float, reason: str
) -> None:
    score, why = hierarchy_transition_score(src, cand)
    assert score == pytest.approx(expected_score)
    assert why == reason


def scored(id_: str, score: float, **payload: Any) -> ScoredCandidate:
    return ScoredCandidate(
        id=id_,
        score=score,
        features=ScoringFeatures(similarity=score, brand_diversity=0, category_diversity=0),
        payload=payload,
    )


def test_category_transition_penalises_other_branches() -> None:
    cfg = RecommendationConfig()
    cfg.category_transition.enabled = True
    items = [
        scored("far", 0.9, categories_hierarchy=["x", "y"]),
        scored("near", 0.8, categories_hierarchy=["a", "b"]),
    ]
    out = Reranker(cfg).apply(items, {"categories_hierarchy": ["a", "b"]})
    assert [i.id for i in out] == ["near", "far"]
    far = next(i for i in out if i.id == "far")
    assert far.penalties["category_transition"] == pytest.approx(0.15)
    assert far.score == pytest.approx(0.75)


def test_diversity_penalty_after_limit() -> None:
    cfg = RecommendationConfig()
    cfg.diversity.max_per_brand = 1
    cfg.diversity.brand_penalty = 0.2
    items = [
        scored(str(i), 1.0 - i * 0.01, brand="Same", categories_hierarchy=[f"c{i}"])
        for i in range(3)
    ]
    out = Reranker(cfg).apply(items)
    penalties = {i.id: i.penalties.get("diversity", 0) for i in out}
    assert penalties == {"0": 0, "1": 0.2, "2": 0.4}
    assert Reranker(cfg).apply([]) == []


def test_scoring_rewards_diverse_brands() -> None:
    cfg = RecommendationConfig()
    cfg.diversity.incremental_enabled = False
    same = [
        Candidate(id=str(i), similarity=0.8, payload={"brand": "A", "categories_hierarchy": ["c"]})
        for i in range(3)
    ]
    rare = Candidate(id="r", similarity=0.8, payload={"brand": "B", "categories_hierarchy": ["c"]})
    out = CandidateScorer(cfg).score([*same, rare])
    assert out[0].id == "r"
    assert CandidateScorer(cfg).score([]) == []


def test_scoring_top_k_and_incremental_bonus() -> None:
    cfg = RecommendationConfig()
    cfg.candidate_search.top_k = 2
    cands = [Candidate(id=str(i), similarity=0.5, payload={"brand": f"B{i}"}) for i in range(5)]
    out = CandidateScorer(cfg).score(cands)
    assert len(out) == 2
    assert "diversity_bonus" in out[0].debug


def test_entropy_and_normalisation() -> None:
    assert normalized_entropy(["a", "a", "a"]) == 0.0
    assert normalized_entropy(["a", "b"]) == pytest.approx(1.0)
    assert normalized_entropy([None, ""]) == 0.0
    items = [scored("a", 2.0), scored("b", 4.0), scored("c", 3.0)]
    assert [round(i.score, 2) for i in normalize_scores(items)] == [0.0, 1.0, 0.5]
    flat = normalize_scores([scored("a", 1.0), scored("b", 1.0)])
    assert [i.score for i in flat] == [1.0, 1.0]
    assert normalize_scores([]) == []


def test_config_validation_and_yaml(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="top_k"):
        RecommendationConfig(candidate_search={"limit": 10, "top_k": 20})  # type: ignore[arg-type]
    path = tmp_path / "rec.yaml"
    path.write_text("similarity:\n  min_score: 0.4\nmax_reasons: 2\n", encoding="utf-8")
    cfg = load_recommendation_config(path)
    assert cfg.similarity.min_score == 0.4
    assert cfg.max_reasons == 2


def test_explanations_can_be_disabled(recommender: Recommender, catalog: Catalog) -> None:
    cfg = RecommendationConfig(enable_explanations=False)
    result = recommender.recommend(pick(catalog, "en:colas"), cfg, top_n=3)
    assert all(i.reasons == [] for i in result.items)


def test_catalog_validates_and_roundtrips(
    tmp_path: Path, products: list[Any], embedder: Any
) -> None:
    from bitesmart_recsys.recsys import read_products, write_products

    with pytest.raises(ValueError, match="same length"):
        Catalog(products[:2], np.zeros((3, 4), np.float32))
    path = tmp_path / "c.jsonl"
    assert write_products(products[:5], path) == 5
    assert [p.id for p in read_products(path)] == [p.id for p in products[:5]]
    assert "passage: " in products[0].to_passage_text()
