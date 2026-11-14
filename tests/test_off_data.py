from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import zstandard

from bitesmart_recsys.data import (
    build_subset,
    describe,
    doc_to_product,
    extract_nutriments,
    iter_jsonl_zst,
    normalize_code,
    read_labelled_ids,
)


def write_zst(path: Path, lines: list[str]) -> Path:
    path.write_bytes(zstandard.ZstdCompressor().compress(("\n".join(lines) + "\n").encode()))
    return path


def doc(code: object, name: str = "Greek yogurt", **extra: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "code": code,
        "product_name": name,
        "categories_hierarchy": ["en:dairies", "en:yogurts"],
        "countries_tags": ["en:france"],
    }
    return base | extra


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            {"energy-kcal_100g": 120, "proteins_100g": "3.5"},
            {"energy-kcal_100g": 120.0, "proteins_100g": 3.5},
        ),
        ({"energy_100g": 418.4}, {"energy-kcal_100g": 100.0}),  # kJ converted to kcal
        ({"sodium_100g": 0.4}, {"salt_100g": 1.0}),  # salt estimated from sodium
        ({"salt_100g": 0.2, "sodium_100g": 9}, {"salt_100g": 0.2}),  # an explicit salt value wins
        (
            {"sugars_100g": -1, "fiber_100g": "n/a", "proteins_100g": True},
            {},
        ),  # invalid values dropped
        (None, {}),
        ("text", {}),
    ],
)
def test_extract_nutriments(raw: object, expected: dict[str, float]) -> None:
    assert extract_nutriments(raw) == pytest.approx(expected)


def test_doc_to_product_maps_and_cleans_fields() -> None:
    product = doc_to_product(
        doc(
            "0123",
            "  Skyr  ",
            brands_tags=["arla"],
            nutriscore_grade="B",
            nova_group="3",
            allergens_hierarchy=["en:milk", 5],
            labels_tags=["en:vegan"],
            nutriments={"proteins_100g": 11},
        )
    )
    assert product is not None
    assert (product.id, product.product_name, product.brand) == ("0123", "Skyr", "arla")
    assert product.nutriscore_grade == "b"
    assert product.nova_group == 3
    assert product.allergens_hierarchy == ["en:milk"]
    assert product.nutriments == {"proteins_100g": 11.0}


@pytest.mark.parametrize(
    "bad",
    [
        doc(None),
        doc("1", name=" "),
        doc("1", name="x"),
        doc("1", categories_hierarchy=[]),
        {"code": "1", "product_name": "No categories"},
    ],
)
def test_incomplete_records_are_rejected(bad: dict[str, Any]) -> None:
    assert doc_to_product(bad) is None


def test_invalid_grade_and_nova_become_none() -> None:
    product = doc_to_product(doc("9", nutriscore_grade="unknown", nova_group=7))
    assert product is not None
    assert product.nutriscore_grade is None
    assert product.nova_group is None


def test_iter_skips_bad_lines_and_non_objects(tmp_path: Path) -> None:
    path = write_zst(
        tmp_path / "a.jsonl.zst",
        [json.dumps(doc("1")), "{broken", "", "[1, 2]", json.dumps(doc("2"))],
    )
    assert [d["code"] for d in iter_jsonl_zst(path)] == ["1", "2"]


def test_normalize_code_ignores_leading_zeros() -> None:
    assert normalize_code("0000123") == normalize_code(123) == "123"


def test_read_labelled_ids_skips_malformed_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "pairs.csv"
    csv_path.write_text(
        "src_id,alt_id,quality_label\n"
        "001,200,best\n"
        "3,4,good,unexpected,extra\n"  # more fields than the header, skipped
        "5,,bad\n",
        encoding="utf-8",
    )
    assert read_labelled_ids(csv_path) == {"1", "200", "5"}


@pytest.fixture
def files(tmp_path: Path) -> tuple[Path, Path]:
    core = write_zst(
        tmp_path / "core.jsonl.zst", [json.dumps(doc(i, f"core {i}")) for i in range(1, 11)]
    )
    index_docs = [
        doc(i, f"index {i}") for i in range(5, 105)
    ]  # 5 to 10 also exist in the core file
    index_docs.append(doc(None, "no code"))
    return core, write_zst(tmp_path / "index.jsonl.zst", [json.dumps(d) for d in index_docs])


def test_build_subset_size_and_no_duplicates(files: tuple[Path, Path]) -> None:
    result = build_subset(*files, size=50, seed=1)
    ids = [p.id for p in result.products]
    assert len(ids) == len(set(ids)) == 50
    assert {str(i) for i in range(1, 11)} <= set(ids)  # the core file is kept first
    assert result.stats["core_products"] == 10
    assert result.stats["rejected_index"] == 1


def test_build_subset_is_reproducible_and_seed_dependent(files: tuple[Path, Path]) -> None:
    a = [p.id for p in build_subset(*files, size=40, seed=7).products]
    b = [p.id for p in build_subset(*files, size=40, seed=7).products]
    c = [p.id for p in build_subset(*files, size=40, seed=8).products]
    assert a == b
    assert a != c


def test_prefixes_of_the_order_are_nested_subsets(files: tuple[Path, Path]) -> None:
    products = build_subset(*files, size=60, seed=3).products
    small, medium = products[:20], products[:40]
    assert {p.id for p in small} <= {p.id for p in medium}


def test_labelled_products_are_always_kept(files: tuple[Path, Path]) -> None:
    labelled = {"3", "00099", "100", "424242"}  # core, index only, index only, found nowhere
    result = build_subset(*files, size=15, seed=2, labelled_ids=labelled)
    ids = {normalize_code(p.id) for p in result.products}
    assert {"3", "99", "100"} <= ids
    assert len(result.products) == 15
    assert result.stats["labelled_ids_given"] == 4
    assert result.stats["labelled_ids_found"] == 3


def test_core_larger_than_size_is_trimmed(files: tuple[Path, Path]) -> None:
    assert len(build_subset(*files, size=6, seed=1).products) == 6


def test_describe_reports_shares() -> None:
    items = [
        doc_to_product(doc("1", labels_tags=["en:vegan"], nutriscore_grade="a")),
        doc_to_product(doc("2", allergens_hierarchy=["en:milk"])),
    ]
    stats = describe(p for p in items if p is not None)
    assert stats["products"] == 2
    assert stats["share_with"]["vegan_label"] == 0.5
    assert stats["share_with"]["allergens"] == 0.5
    assert stats["countries_top"] == {"en:france": 2}
