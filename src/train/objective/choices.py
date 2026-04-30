from __future__ import annotations

from collections.abc import Iterable
from typing import Any
from typing import TypeVar

T = TypeVar("T", bound=str)


def checked_choice(
    value: Any,
    *,
    allowed: Iterable[T],
    label: str,
    lower: bool = True,
) -> T:
    choice = str(value)
    if lower:
        choice = choice.lower()
    allowed_values = tuple(allowed)
    for option in allowed_values:
        if choice == option:
            return option
    expected = ", ".join(sorted(allowed_values))
    raise ValueError(f"Unknown {label}={choice!r}; expected one of: {expected}.")
