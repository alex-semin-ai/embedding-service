"""Product schema (Open Food Facts style) and its text serialisation for embedding."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# Fields that user constraints filter on, with the Qdrant payload index type for each.
PAYLOAD_SCHEMA: dict[str, str] = {
    "id": "keyword",
    "allergens_hierarchy": "keyword",
    "traces_hierarchy": "keyword",
    "labels_tags": "keyword",
    "ingredients_analysis_tags": "keyword",
    "categories_hierarchy": "keyword",
    "countries_tags": "keyword",
    "nutriments.proteins_100g": "float",
    "nutriments.sugars_100g": "float",
    "nutriments.salt_100g": "float",
    "nutriments.energy-kcal_100g": "float",
    "nutriments.saturated-fat_100g": "float",
}


class Product(BaseModel):
    id: str
    product_name: str
    brand: str | None = None
    categories_hierarchy: list[str] = Field(default_factory=list)
    allergens_hierarchy: list[str] = Field(default_factory=list)
    traces_hierarchy: list[str] = Field(default_factory=list)
    labels_tags: list[str] = Field(default_factory=list)
    ingredients_analysis_tags: list[str] = Field(default_factory=list)
    ingredients_text: str = ""
    countries_tags: list[str] = Field(default_factory=list)
    nutriscore_grade: str | None = None
    nova_group: int | None = None
    nutriments: dict[str, float] = Field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return self.model_dump()

    def to_passage_text(self) -> str:
        """Text sent to the embedding model. E5 expects the 'passage: ' prefix for documents."""
        parts = [self.product_name]
        if self.brand:
            parts.append(f"brand {self.brand}")
        if self.categories_hierarchy:
            parts.append("category " + ", ".join(c.removeprefix("en:") for c in self.categories_hierarchy))
        if self.labels_tags:
            parts.append("labels " + ", ".join(t.removeprefix("en:") for t in self.labels_tags))
        if self.ingredients_text:
            parts.append(f"ingredients {self.ingredients_text}")
        return "passage: " + ". ".join(parts)
