"""Open Food Facts subset builder.

Reads the large `.jsonl.zst` files as a stream (nothing is extracted to disk), maps each record
to the project's `Product` schema and builds one reproducible subset. The subset is shuffled
with a fixed seed, so its first 10 000 and first 50 000 products are nested subsets of the
full set, which is what the dataset-size benchmark needs.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import random
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import zstandard

from bitesmart_recsys.recsys.models import Product

logger = logging.getLogger(__name__)

KJ_PER_KCAL = 4.184
SALT_PER_SODIUM = 2.5


def iter_jsonl_zst(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield the JSON objects of a zstd compressed JSON lines file. Bad lines are skipped."""
    skipped = 0
    with Path(path).open("rb") as fh:
        reader = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8")
        for line in reader:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if isinstance(doc, dict):
                yield doc
    if skipped:
        logger.warning("skipped malformed lines", extra={"path": str(path), "skipped": skipped})


def normalize_code(code: object) -> str:
    """Barcode as a comparable key. CSV exports and JSON dumps disagree on leading zeros."""
    return str(code).strip().lstrip("0")


def _float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _str_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if isinstance(v, str) and v.strip()]
    return []


def extract_nutriments(raw: object) -> dict[str, float]:
    """Per 100 g values for the nutrients the recommender uses. Missing ones are left out."""
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, float] = {}
    kcal = _float(raw.get("energy-kcal_100g"))
    if kcal is None:
        kj = _float(raw.get("energy_100g"))
        kcal = None if kj is None else round(kj / KJ_PER_KCAL, 2)
    salt = _float(raw.get("salt_100g"))
    if salt is None:
        sodium = _float(raw.get("sodium_100g"))
        salt = None if sodium is None else round(sodium * SALT_PER_SODIUM, 4)
    values = {
        "energy-kcal_100g": kcal,
        "proteins_100g": _float(raw.get("proteins_100g")),
        "sugars_100g": _float(raw.get("sugars_100g")),
        "salt_100g": salt,
        "saturated-fat_100g": _float(raw.get("saturated-fat_100g")),
        "fiber_100g": _float(raw.get("fiber_100g")),
    }
    for key, value in values.items():
        if value is not None and value >= 0:
            out[key] = value
    return out


def doc_to_product(doc: Mapping[str, Any]) -> Product | None:
    """Map a raw record to a Product, or None when it lacks a code, a name or categories."""
    code = doc.get("code")
    name = str(doc.get("product_name") or "").strip()
    categories = _str_list(doc.get("categories_hierarchy"))
    if code is None or len(name) < 2 or not categories:
        return None
    brands = _str_list(doc.get("brands_tags"))
    grade = str(doc.get("nutriscore_grade") or "").lower()
    nova = _float(doc.get("nova_group"))
    return Product(
        id=str(code).strip(),
        product_name=name,
        brand=brands[0] if brands else None,
        categories_hierarchy=categories,
        allergens_hierarchy=_str_list(doc.get("allergens_hierarchy")),
        traces_hierarchy=_str_list(doc.get("traces_hierarchy")),
        labels_tags=_str_list(doc.get("labels_tags")),
        ingredients_analysis_tags=_str_list(doc.get("ingredients_analysis_tags")),
        ingredients_text=str(doc.get("ingredients_text") or "").strip(),
        countries_tags=_str_list(doc.get("countries_tags")),
        nutriscore_grade=grade if grade in {"a", "b", "c", "d", "e"} else None,
        nova_group=int(nova) if nova is not None and 1 <= nova <= 4 else None,
        nutriments=extract_nutriments(doc.get("nutriments")),
    )


def read_labelled_ids(path: str | Path) -> set[str]:
    """Product ids used in a substitution pair CSV (src_id and alt_id columns).

    The thesis CSV files contain a few malformed rows with extra fields, those are skipped.
    """
    csv.field_size_limit(2**31 - 1)  # sys.maxsize overflows a 32-bit C long on Windows
    ids: set[str] = set()
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, restkey="__extra__"):
            if "__extra__" in row:
                continue
            for column in ("src_id", "alt_id"):
                value = row.get(column)
                if value:
                    ids.add(normalize_code(value))
    return ids


def describe(products: Iterable[Product]) -> dict[str, Any]:
    """Counts and field coverage, written to the manifest next to the subset."""
    items = list(products)
    n = max(len(items), 1)
    countries: Counter[str] = Counter()
    roots: Counter[str] = Counter()
    for p in items:
        countries.update(p.countries_tags)
        roots[p.categories_hierarchy[0]] += 1
    complete = (
        "energy-kcal_100g",
        "proteins_100g",
        "sugars_100g",
        "salt_100g",
        "saturated-fat_100g",
    )

    def share(flag: Callable[[Product], bool]) -> float:
        return round(sum(1 for p in items if flag(p)) / n, 4)

    return {
        "products": len(items),
        "countries_top": dict(countries.most_common(10)),
        "root_categories_top": dict(roots.most_common(10)),
        "distinct_leaf_categories": len({p.categories_hierarchy[-1] for p in items}),
        "share_with": {
            "allergens": share(lambda p: bool(p.allergens_hierarchy)),
            "labels": share(lambda p: bool(p.labels_tags)),
            "vegan_label": share(lambda p: "en:vegan" in p.labels_tags),
            "vegetarian_label": share(lambda p: "en:vegetarian" in p.labels_tags),
            "brand": share(lambda p: p.brand is not None),
            "ingredients_text": share(lambda p: bool(p.ingredients_text)),
            "nutriscore": share(lambda p: p.nutriscore_grade is not None),
            "nova": share(lambda p: p.nova_group is not None),
            "all_five_nutrients": share(lambda p: all(k in p.nutriments for k in complete)),
        },
    }


@dataclass
class SubsetResult:
    products: list[Product]
    stats: dict[str, Any] = field(default_factory=dict)


def build_subset(
    core_path: str | Path,
    index_path: str | Path,
    size: int = 100_000,
    seed: int = 42,
    labelled_ids: Iterable[str] = (),
    progress: Callable[[str], None] | None = None,
) -> SubsetResult:
    """Build a shuffled subset of `size` products.

    The core file (complete records) is read first. The large index file is then streamed once
    and sampled uniformly (reservoir sampling) to fill the rest. Products named in
    `labelled_ids` are always kept when they exist in either file, so the quality evaluation
    can use them.
    """
    say = progress or (lambda _msg: None)
    rng = random.Random(seed)
    labelled = {normalize_code(x) for x in labelled_ids}

    pool: dict[str, Product] = {}
    rejected_core = 0
    for doc in iter_jsonl_zst(core_path):
        product = doc_to_product(doc)
        if product is None:
            rejected_core += 1
        else:
            pool.setdefault(normalize_code(product.id), product)
    core_keys = set(pool)
    say(f"core file read, {len(pool)} products")

    fill_needed = max(size - len(core_keys), 0)
    reservoir: list[Product] = []
    labelled_extra: dict[str, Product] = {}
    seen = 0
    rejected_index = 0
    for doc in iter_jsonl_zst(index_path):
        product = doc_to_product(doc)
        if product is None:
            rejected_index += 1
            continue
        key = normalize_code(product.id)
        if key in core_keys or key in labelled_extra:
            continue
        if key in labelled:
            labelled_extra[key] = product
            continue
        seen += 1
        if len(reservoir) < fill_needed:
            reservoir.append(product)
        else:
            slot = rng.randrange(seen)
            if slot < fill_needed:
                reservoir[slot] = product
    say(f"index file streamed, {seen} candidates seen")

    required = {k: p for k, p in pool.items() if k in labelled} | labelled_extra
    optional = [p for k, p in pool.items() if k not in labelled] + reservoir
    rng.shuffle(optional)
    room = max(size - len(required), 0)
    chosen = list(required.values()) + optional[:room]
    rng.shuffle(chosen)

    stats = describe(chosen)
    stats.update(
        {
            "requested_size": size,
            "seed": seed,
            "core_products": len(core_keys),
            "index_candidates_seen": seen,
            "rejected_core": rejected_core,
            "rejected_index": rejected_index,
            "labelled_ids_given": len(labelled),
            "labelled_ids_found": len(required),
        }
    )
    return SubsetResult(products=chosen, stats=stats)
