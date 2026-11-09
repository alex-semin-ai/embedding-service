from __future__ import annotations

from typing import Any

import pytest

from bitesmart_recsys.indexes.filters import (
    AnyOf,
    Condition,
    Filter,
    compute_mask,
    get_field,
    matches,
)

PRODUCT: dict[str, Any] = {
    "id": "1",
    "labels": ["vegan", "organic"],
    "country": "fi",
    "nutriments": {"sugars": 4.0, "salt": "0.5", "bad": "n/a", "flag": True},
}


@pytest.mark.parametrize(
    ("cond", "expected"),
    [
        (Condition("country", "eq", "fi"), True),
        (Condition("country", "eq", "se"), False),
        (Condition("labels", "eq", "vegan"), True),
        (Condition("labels", "any", ("halal", "organic")), True),
        (Condition("labels", "any", ("halal",)), False),
        (Condition("country", "any", ["fi", "se"]), True),
        (Condition("nutriments.sugars", "lte", 4.0), True),
        (Condition("nutriments.sugars", "lte", 3.9), False),
        (Condition("nutriments.sugars", "gte", 4.0), True),
        (Condition("nutriments.salt", "lte", 1.0), True),  # numeric strings are accepted
        (Condition("nutriments.bad", "lte", 100.0), False),  # non numeric never matches
        (Condition("nutriments.flag", "gte", 0.0), False),  # booleans are not numbers
        (Condition("nutriments.missing", "lte", 100.0), False),
        (Condition("missing.deep.path", "eq", "x"), False),
    ],
)
def test_condition_semantics(cond: Condition, expected: bool) -> None:
    assert matches(PRODUCT, Filter(must=(cond,))) is expected


def test_must_not_and_anyof() -> None:
    vegan_or_halal = AnyOf((Condition("labels", "eq", "halal"), Condition("labels", "eq", "vegan")))
    assert matches(PRODUCT, Filter(must=(vegan_or_halal,)))
    assert not matches(
        PRODUCT, Filter(must=(vegan_or_halal,), must_not=(Condition("id", "eq", "1"),))
    )
    # a must_not on a missing field excludes nothing
    assert matches(PRODUCT, Filter(must_not=(Condition("allergens", "any", ("milk",)),)))


def test_none_and_empty_filter_match_everything() -> None:
    assert matches(PRODUCT, None)
    assert Filter().is_empty
    assert matches(PRODUCT, Filter())


def test_get_field_and_mask() -> None:
    assert get_field(PRODUCT, "nutriments.sugars") == 4.0
    assert get_field(PRODUCT, "country.x") is None
    payloads = [{"a": 1}, {"a": 5}, {"a": 9}]
    mask = compute_mask(payloads, Filter(must=(Condition("a", "gte", 5.0),)))
    assert mask.tolist() == [False, True, True]
