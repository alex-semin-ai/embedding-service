"""Recommendation configuration (ported from the BiteSmart substitution recommender)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class SimilarityConfig(BaseModel):
    min_score: float = Field(0.1, ge=0.0, le=1.0, description="Minimum cosine similarity")


class DiversityConfig(BaseModel):
    brand_penalty: float = Field(0.05, ge=0.0)
    category_penalty: float = Field(0.03, ge=0.0)
    max_per_brand: int = Field(3, ge=1)
    max_per_category: int = Field(5, ge=1)
    incremental_enabled: bool = True
    brand_alpha: float = Field(0.25, ge=0.0)
    category_alpha: float = Field(0.15, ge=0.0)


class ScoringWeights(BaseModel):
    similarity: float = Field(1.0, ge=0.0)
    brand_diversity: float = Field(0.10, ge=0.0)
    category_diversity: float = Field(0.10, ge=0.0)


class DietaryConstraints(BaseModel):
    vegan: bool = False
    vegetarian: bool = False
    lactose_free: bool = False
    halal: bool = False
    kosher: bool = False
    pescatarian: bool = False
    paleo: bool = False
    low_glycemic: bool = False


class SourceNutrientConstraints(BaseModel):
    """Candidate must be at least as good as the source product on the enabled nutrients."""

    enforce: bool = False
    proteins_gte_source: bool = True
    sugars_lte_source: bool = True
    salt_lte_source: bool = True
    kcal_lte_source: bool = True
    saturated_fat_lte_source: bool = True


class CountryFilter(BaseModel):
    enabled: bool = False
    country_tag: str | None = None


class HardConstraints(BaseModel):
    forbidden_allergens: list[str] = Field(default_factory=list)
    forbidden_traces: list[str] = Field(default_factory=list)
    same_category_as_source: bool = False
    dietary: DietaryConstraints = Field(default_factory=DietaryConstraints)
    source_nutrients: SourceNutrientConstraints = Field(default_factory=SourceNutrientConstraints)
    country_filter: CountryFilter = Field(default_factory=CountryFilter)
    nutrient_caps: dict[str, float] = Field(
        default_factory=dict, description="Absolute caps per 100g, e.g. {'sugars_100g': 5}"
    )


class CandidateSearchConfig(BaseModel):
    limit: int = Field(200, ge=1, description="ANN candidates fetched from the index")
    top_k: int = Field(40, ge=1, description="Candidates kept after scoring")
    hard_constraints: HardConstraints = Field(default_factory=HardConstraints)
    exclude_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _top_k_not_above_limit(self) -> CandidateSearchConfig:
        if self.top_k > self.limit:
            raise ValueError("top_k cannot exceed limit")
        return self


class CategoryTransitionConfig(BaseModel):
    """Soft penalty for candidates from a different category branch than the source."""

    enabled: bool = False
    alpha: float = Field(0.15, ge=0.0, le=1.0)
    threshold: float = Field(0.50, ge=0.0, le=1.0)
    gamma: float = Field(2.0, ge=0.1, le=5.0)
    fallback_missing: Literal["skip", "penalize_unknown"] = "skip"


class RecommendationConfig(BaseModel):
    model_version: str = "rec-v1"
    similarity: SimilarityConfig = Field(default_factory=SimilarityConfig)
    diversity: DiversityConfig = Field(default_factory=DiversityConfig)
    scoring: ScoringWeights = Field(default_factory=ScoringWeights)
    candidate_search: CandidateSearchConfig = Field(default_factory=CandidateSearchConfig)
    category_transition: CategoryTransitionConfig = Field(default_factory=CategoryTransitionConfig)
    enable_explanations: bool = True
    max_reasons: int = Field(3, ge=1, le=4)


def load_recommendation_config(path: str | Path) -> RecommendationConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return RecommendationConfig(**data)
