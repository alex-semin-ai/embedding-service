"""Hard constraints, ported from the substitution recommender.

`build_filter` turns user constraints into a backend-neutral index filter (pushed down to
the vector index). `verify_hard_constraints` re-checks every returned candidate with the
reference evaluator, as defence in depth. An ANN backend that mishandles a filter can then
never leak an unsafe product (for example an allergen) into the result.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from bitesmart_recsys.indexes.filters import AnyOf, Condition, Filter, Node, matches

from .config import RecommendationConfig

logger = logging.getLogger(__name__)

# dietary flag -> (payload field, tag) pairs, any of which proves the flag
DIETARY_TAG_SOURCES: dict[str, list[tuple[str, str]]] = {
    "vegan": [
        ("ingredients_analysis_tags", "en:vegan"),
        ("labels_tags", "en:vegan"),
    ],
    "vegetarian": [
        ("ingredients_analysis_tags", "en:vegetarian"),
        ("labels_tags", "en:vegetarian"),
    ],
    "lactose_free": [("labels_tags", "en:no-lactose"), ("labels_tags", "en:lactose-free")],
    "halal": [("labels_tags", "en:halal")],
    "kosher": [("labels_tags", "en:kosher")],
    "pescatarian": [("labels_tags", "en:pescatarian")],
    "paleo": [("labels_tags", "en:paleo")],
    "low_glycemic": [("labels_tags", "en:low-glycemic-index")],
}

NUTRIENT_ALIASES: dict[str, str] = {
    "sugar": "sugars_100g",
    "sugars": "sugars_100g",
    "salt": "salt_100g",
    "kcal": "energy-kcal_100g",
    "energy-kcal": "energy-kcal_100g",
    "energy_kcal": "energy-kcal_100g",
    "protein": "proteins_100g",
    "proteins": "proteins_100g",
}


def nutriment_key(name: str) -> str:
    name = name.removeprefix("nutriments.")
    return "nutriments." + NUTRIENT_ALIASES.get(name, name)


def _source_value(source: Mapping[str, Any] | None, key: str) -> float | None:
    nutr = (source or {}).get("nutriments")
    value = nutr.get(key) if isinstance(nutr, Mapping) else None
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def build_filter(
    config: RecommendationConfig,
    source_payload: Mapping[str, Any] | None,
    source_id: str | None = None,
) -> Filter:
    hard = config.candidate_search.hard_constraints
    must: list[Node] = []
    must_not: list[Node] = []

    excluded = list(config.candidate_search.exclude_ids)
    if source_id:
        excluded.append(source_id)
    if excluded:
        must_not.append(Condition("id", "any", tuple(excluded)))
    if hard.forbidden_allergens:
        must_not.append(Condition("allergens_hierarchy", "any", tuple(hard.forbidden_allergens)))
    if hard.forbidden_traces:
        must_not.append(Condition("traces_hierarchy", "any", tuple(hard.forbidden_traces)))

    for flag, pairs in DIETARY_TAG_SOURCES.items():
        if getattr(hard.dietary, flag):
            conditions = tuple(Condition(f, "eq", tag) for f, tag in pairs)
            must.append(conditions[0] if len(conditions) == 1 else AnyOf(conditions))

    categories = (source_payload or {}).get("categories_hierarchy")
    if hard.same_category_as_source and isinstance(categories, list) and categories:
        must.append(Condition("categories_hierarchy", "eq", categories[0]))

    if hard.country_filter.enabled and hard.country_filter.country_tag:
        must.append(Condition("countries_tags", "eq", hard.country_filter.country_tag))

    rel = hard.source_nutrients
    if rel.enforce:
        relative = (
            (rel.proteins_gte_source, "proteins_100g", "gte"),
            (rel.sugars_lte_source, "sugars_100g", "lte"),
            (rel.salt_lte_source, "salt_100g", "lte"),
            (rel.kcal_lte_source, "energy-kcal_100g", "lte"),
            (rel.saturated_fat_lte_source, "saturated-fat_100g", "lte"),
        )
        for enabled, key, op in relative:
            value = _source_value(source_payload, key)
            if enabled and value is not None:
                must.append(Condition(f"nutriments.{key}", op, value))  # type: ignore[arg-type]

    for name, cap in hard.nutrient_caps.items():
        must.append(Condition(nutriment_key(name), "lte", float(cap)))

    return Filter(must=tuple(must), must_not=tuple(must_not))


def verify_hard_constraints(
    candidates: list[Any],
    config: RecommendationConfig,
    source_payload: Mapping[str, Any] | None,
    source_id: str | None = None,
) -> list[Any]:
    """Drop candidates that violate a hard constraint. Each candidate needs a `.payload`."""
    flt = build_filter(config, source_payload, source_id)
    kept = [c for c in candidates if matches(c.payload, flt)]
    removed = len(candidates) - len(kept)
    if removed:
        logger.warning(
            "hard constraint verification removed candidates",
            extra={"removed": removed, "kept": len(kept), "total": len(candidates)},
        )
    return kept
