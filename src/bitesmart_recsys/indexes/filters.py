"""Backend-neutral metadata filters.

One small filter language is translated to FAISS/hnswlib id masks, to NumPy masks and to
Qdrant conditions. The reference evaluator `matches` defines the semantics, so every
backend can be checked against it.

Semantics
- `eq`   the field equals the value, or the field is a list containing the value
- `any`  the field (or one of its list items) is in the given values
- `gte` / `lte`  numeric comparison, a missing or non numeric field never matches
- `must` nodes are AND-ed, `must_not` nodes exclude a point when they match
- `AnyOf` is an OR group (used for "vegan in any of three tag fields")
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

Op = Literal["eq", "any", "gte", "lte"]


@dataclass(frozen=True)
class Condition:
    field: str
    op: Op
    value: Any

    def __post_init__(self) -> None:
        if self.op == "any" and not isinstance(self.value, tuple):
            object.__setattr__(self, "value", tuple(self.value))


@dataclass(frozen=True)
class AnyOf:
    conditions: tuple[Condition, ...]


Node = Condition | AnyOf


@dataclass(frozen=True)
class Filter:
    must: tuple[Node, ...] = ()
    must_not: tuple[Node, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.must and not self.must_not


def get_field(payload: Mapping[str, Any], path: str) -> Any:
    """Resolve a dotted path such as 'nutriments.sugars_100g'. Returns None when absent."""
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _to_float(value: Any) -> float | None:
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


def _as_items(value: Any) -> Sequence[Any]:
    if value is None:
        return ()
    if isinstance(value, list | tuple | set):
        return list(value)
    return (value,)


def _condition_matches(payload: Mapping[str, Any], cond: Condition) -> bool:
    raw = get_field(payload, cond.field)
    if cond.op == "eq":
        return cond.value in _as_items(raw)
    if cond.op == "any":
        return any(item in cond.value for item in _as_items(raw))
    number = _to_float(raw)
    if number is None:
        return False
    bound = float(cond.value)
    return number >= bound if cond.op == "gte" else number <= bound


def node_matches(payload: Mapping[str, Any], node: Node) -> bool:
    if isinstance(node, AnyOf):
        return any(_condition_matches(payload, c) for c in node.conditions)
    return _condition_matches(payload, node)


def matches(payload: Mapping[str, Any], flt: Filter | None) -> bool:
    if flt is None:
        return True
    return all(node_matches(payload, n) for n in flt.must) and not any(
        node_matches(payload, n) for n in flt.must_not
    )


def compute_mask(payloads: Iterable[Mapping[str, Any]], flt: Filter) -> NDArray[np.bool_]:
    return np.fromiter((matches(p, flt) for p in payloads), dtype=np.bool_)
