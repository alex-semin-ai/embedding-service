"""Synthetic Open Food Facts style catalog.

Real OFF dumps are huge and not needed to demonstrate the system, so the demo, the tests
and the benchmark use a seeded generator with realistic taxonomy, allergens, labels,
countries and nutrient profiles per category.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from .models import Product


@dataclass(frozen=True)
class Leaf:
    root: str
    mid: str
    leaf: str
    flavors: tuple[str, ...]
    # mean nutrients per 100g: kcal, protein, sugars, salt, saturated fat
    profile: tuple[float, float, float, float, float]
    allergens: tuple[str, ...]
    animal_free_ratio: float


_LEAVES: tuple[Leaf, ...] = (
    Leaf("en:dairies", "en:yogurts", "en:fruit-yogurts", ("strawberry", "peach", "blueberry"), (95, 3.8, 13.0, 0.10, 1.6), ("en:milk",), 0.0),
    Leaf("en:dairies", "en:yogurts", "en:greek-yogurts", ("natural", "honey", "vanilla"), (110, 9.0, 5.0, 0.12, 3.5), ("en:milk",), 0.0),
    Leaf("en:dairies", "en:cheeses", "en:cream-cheeses", ("garlic", "natural", "herbs"), (250, 6.0, 3.0, 0.9, 16.0), ("en:milk",), 0.0),
    Leaf("en:dairies", "en:milks", "en:semi-skimmed-milks", ("classic", "organic"), (46, 3.4, 4.8, 0.10, 1.0), ("en:milk",), 0.0),
    Leaf("en:plant-based-foods", "en:plant-milks", "en:oat-drinks", ("natural", "barista", "chocolate"), (45, 1.0, 4.0, 0.10, 0.2), ("en:gluten",), 1.0),
    Leaf("en:plant-based-foods", "en:plant-milks", "en:soy-drinks", ("natural", "vanilla", "unsweetened"), (40, 3.3, 2.5, 0.10, 0.4), ("en:soybeans",), 1.0),
    Leaf("en:plant-based-foods", "en:plant-yogurts", "en:soy-yogurts", ("natural", "peach", "coconut"), (65, 4.0, 6.0, 0.08, 0.6), ("en:soybeans",), 1.0),
    Leaf("en:snacks", "en:sweet-snacks", "en:chocolate-bars", ("hazelnut", "dark", "caramel"), (520, 6.0, 48.0, 0.15, 18.0), ("en:milk", "en:nuts"), 0.15),
    Leaf("en:snacks", "en:sweet-snacks", "en:cereal-bars", ("oat", "berry", "apple"), (390, 6.5, 24.0, 0.30, 3.0), ("en:gluten",), 0.6),
    Leaf("en:snacks", "en:salty-snacks", "en:potato-chips", ("paprika", "salted", "sour-cream"), (530, 6.0, 1.5, 1.4, 4.5), (), 0.8),
    Leaf("en:snacks", "en:salty-snacks", "en:nut-mixes", ("roasted", "salted", "unsalted"), (610, 21.0, 4.5, 0.5, 6.0), ("en:nuts",), 1.0),
    Leaf("en:beverages", "en:juices", "en:orange-juices", ("not-from-concentrate", "with-pulp"), (44, 0.7, 8.5, 0.01, 0.0), (), 1.0),
    Leaf("en:beverages", "en:sodas", "en:colas", ("classic", "zero", "cherry"), (42, 0.0, 10.5, 0.02, 0.0), (), 1.0),
    Leaf("en:beverages", "en:sodas", "en:lemonades", ("lemon", "lime", "sparkling"), (38, 0.0, 9.0, 0.02, 0.0), (), 1.0),
    Leaf("en:breakfasts", "en:cereals", "en:muesli", ("fruit", "nut", "crunchy"), (370, 9.0, 17.0, 0.05, 2.0), ("en:gluten", "en:nuts"), 0.9),
    Leaf("en:breakfasts", "en:cereals", "en:corn-flakes", ("classic", "honey", "chocolate"), (375, 7.0, 8.0, 1.0, 0.7), ("en:gluten",), 0.9),
    Leaf("en:meals", "en:ready-meals", "en:pasta-dishes", ("bolognese", "carbonara", "pesto"), (140, 6.0, 3.0, 0.9, 2.5), ("en:gluten", "en:milk", "en:eggs"), 0.2),
    Leaf("en:meals", "en:ready-meals", "en:vegetable-soups", ("tomato", "pumpkin", "minestrone"), (45, 1.4, 4.0, 0.8, 0.4), (), 0.85),
)

_BRANDS = (
    "Valio", "Arla", "Danone", "Alpro", "Oatly", "Nestle", "Kellogg's", "Mondelez", "Lays",
    "Pepsi", "Coca-Cola", "Juicy", "Fazer", "Pirkka", "Rainbow", "Gutfood", "Alnatura",
    "Dr. Oetker", "Barilla", "Knorr",
)
_COUNTRIES = ("en:finland", "en:germany", "en:france", "en:united-states", "en:sweden")


def _grade(sugars: float, sat_fat: float, salt: float) -> str:
    points = sugars / 4.5 + sat_fat / 1.0 + salt / 0.45
    for grade, limit in zip("abcd", (2.0, 5.0, 9.0, 14.0), strict=True):
        if points <= limit:
            return grade
    return "e"


def _make_product(rng: np.random.Generator, index: int, leaf: Leaf) -> Product:
    brand = str(rng.choice(_BRANDS))
    flavor = str(rng.choice(leaf.flavors))
    scale = rng.lognormal(0.0, 0.25, size=5)
    kcal, protein, sugars, salt, sat_fat = (np.array(leaf.profile) * scale).round(2).tolist()

    animal_free = rng.random() < leaf.animal_free_ratio
    labels: list[str] = []
    analysis: list[str] = []
    allergens = list(leaf.allergens)
    if animal_free:
        labels.append("en:vegan")
        analysis += ["en:vegan", "en:vegetarian"]
        allergens = [a for a in allergens if a != "en:milk"]
    elif rng.random() < 0.6:
        analysis.append("en:vegetarian")
        labels.append("en:vegetarian")
    if "en:milk" not in allergens:
        labels.append("en:no-lactose")
    if rng.random() < 0.15:
        labels.append("en:halal")
    if rng.random() < 0.20:
        labels.append("en:organic")

    traces = [str(rng.choice(["en:nuts", "en:gluten", "en:soybeans", "en:milk"]))] if rng.random() < 0.2 else []
    countries = rng.choice(_COUNTRIES, size=int(rng.integers(1, 4)), replace=False).tolist()
    name = f"{brand} {flavor.replace('-', ' ')} {leaf.leaf.removeprefix('en:').replace('-', ' ')}"
    ingredients = ", ".join(
        [flavor.replace("-", " "), leaf.leaf.removeprefix("en:").replace("-", " "), "water", "salt"]
    )
    return Product(
        id=f"{3000000000000 + index}",
        product_name=name,
        brand=brand,
        categories_hierarchy=[leaf.root, leaf.mid, leaf.leaf],
        allergens_hierarchy=allergens,
        traces_hierarchy=traces,
        labels_tags=labels,
        ingredients_analysis_tags=analysis,
        ingredients_text=ingredients,
        countries_tags=[str(c) for c in countries],
        nutriscore_grade=_grade(sugars, sat_fat, salt),
        nova_group=int(rng.integers(1, 5)),
        nutriments={
            "energy-kcal_100g": kcal,
            "proteins_100g": protein,
            "sugars_100g": sugars,
            "salt_100g": salt,
            "saturated-fat_100g": sat_fat,
        },
    )


def generate_products(n: int, seed: int = 42) -> list[Product]:
    rng = np.random.default_rng(seed)
    leaf_ids = rng.integers(0, len(_LEAVES), size=n)
    return [_make_product(rng, i, _LEAVES[int(leaf)]) for i, leaf in enumerate(leaf_ids)]


def leaf_names() -> list[str]:
    return [leaf.leaf for leaf in _LEAVES]


def iter_leaves() -> Iterator[Leaf]:
    return iter(_LEAVES)
