from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch


CameraSet = Literal["train", "heldout"]


@dataclass(frozen=True)
class CameraSwapPair:
    """One source-world / query-camera / target-GT training or eval item."""

    source_set: CameraSet
    source_view: int
    query_set: CameraSet
    query_view: int
    target_set: CameraSet
    target_view: int
    source_name: str | None = None
    query_name: str | None = None
    target_name: str | None = None

    @property
    def is_self_reconstruction(self) -> bool:
        return (
            self.source_set == "train"
            and self.query_set == "train"
            and self.target_set == "train"
            and self.source_view == self.query_view == self.target_view
        )

    @property
    def is_train_cross_view(self) -> bool:
        return (
            self.source_set == "train"
            and self.query_set == "train"
            and self.target_set == "train"
            and self.query_view == self.target_view
            and self.source_view != self.target_view
        )

    @property
    def is_heldout_query(self) -> bool:
        return self.query_set == "heldout" or self.target_set == "heldout"


def _validate_view_count(view_count: int, *, name: str) -> int:
    count = int(view_count)
    if count < 1:
        raise ValueError(f"{name} must be >= 1, got {view_count}.")
    return count


def _validate_names(names: Sequence[str] | None, view_count: int, *, name: str) -> tuple[str, ...] | None:
    if names is None:
        return None
    values = tuple(str(item) for item in names)
    if len(values) != view_count:
        raise ValueError(f"{name} must have {view_count} entries, got {len(values)}.")
    return values


def _name_for(names: tuple[str, ...] | None, view: int) -> str | None:
    if names is None:
        return None
    return names[int(view)]


def build_train_camera_swap_pairs(
    train_view_count: int,
    *,
    include_self: bool = True,
    include_cross: bool = True,
    train_camera_names: Sequence[str] | None = None,
) -> tuple[CameraSwapPair, ...]:
    """Build source/query pairs for train cameras.

    For two train cameras this returns the four canonical items:
    `W_0+C_0`, `W_0+C_1`, `W_1+C_1`, and `W_1+C_0`, depending on the
    include flags.
    """

    view_count = _validate_view_count(train_view_count, name="train_view_count")
    if not include_self and not include_cross:
        raise ValueError("At least one of include_self/include_cross must be true.")
    names = _validate_names(train_camera_names, view_count, name="train_camera_names")

    pairs = []
    for source_view in range(view_count):
        if include_self:
            pairs.append(
                CameraSwapPair(
                    source_set="train",
                    source_view=source_view,
                    query_set="train",
                    query_view=source_view,
                    target_set="train",
                    target_view=source_view,
                    source_name=_name_for(names, source_view),
                    query_name=_name_for(names, source_view),
                    target_name=_name_for(names, source_view),
                )
            )
        if include_cross:
            for target_view in range(view_count):
                if target_view == source_view:
                    continue
                pairs.append(
                    CameraSwapPair(
                        source_set="train",
                        source_view=source_view,
                        query_set="train",
                        query_view=target_view,
                        target_set="train",
                        target_view=target_view,
                        source_name=_name_for(names, source_view),
                        query_name=_name_for(names, target_view),
                        target_name=_name_for(names, target_view),
                    )
                )
    return tuple(pairs)


def build_heldout_camera_swap_pairs(
    train_view_count: int,
    heldout_view_count: int,
    *,
    train_camera_names: Sequence[str] | None = None,
    heldout_camera_names: Sequence[str] | None = None,
) -> tuple[CameraSwapPair, ...]:
    """Build eval queries `W_train + C_heldout -> GT_heldout`."""

    train_count = _validate_view_count(train_view_count, name="train_view_count")
    heldout_count = _validate_view_count(heldout_view_count, name="heldout_view_count")
    train_names = _validate_names(train_camera_names, train_count, name="train_camera_names")
    heldout_names = _validate_names(heldout_camera_names, heldout_count, name="heldout_camera_names")

    pairs = []
    for source_view in range(train_count):
        for heldout_view in range(heldout_count):
            pairs.append(
                CameraSwapPair(
                    source_set="train",
                    source_view=source_view,
                    query_set="heldout",
                    query_view=heldout_view,
                    target_set="heldout",
                    target_view=heldout_view,
                    source_name=_name_for(train_names, source_view),
                    query_name=_name_for(heldout_names, heldout_view),
                    target_name=_name_for(heldout_names, heldout_view),
                )
            )
    return tuple(pairs)


def shuffle_camera_swap_pairs(
    pairs: Sequence[CameraSwapPair],
    *,
    generator: torch.Generator | None = None,
) -> tuple[CameraSwapPair, ...]:
    if not pairs:
        return ()
    order = torch.randperm(len(pairs), generator=generator).tolist()
    return tuple(pairs[index] for index in order)


def _sample_from(pairs: Sequence[CameraSwapPair], *, generator: torch.Generator | None = None) -> CameraSwapPair:
    index = int(torch.randint(len(pairs), (1,), generator=generator).item())
    return pairs[index]


def sample_train_camera_swap_pairs(
    train_view_count: int,
    *,
    pairs_per_step: int,
    include_self: bool = True,
    include_cross: bool = True,
    self_pair_probability: float | None = None,
    train_camera_names: Sequence[str] | None = None,
    generator: torch.Generator | None = None,
) -> tuple[CameraSwapPair, ...]:
    """Sample train camera-token swap items for one optimization step.

    `pairs_per_step <= 0` means "use all eligible pairs this step", shuffled.
    When `self_pair_probability` is set, each sampled item first chooses self
    vs cross class, then samples inside that class. This keeps same-camera
    reconstruction present even when there are many cross-camera pairs.
    """

    all_pairs = build_train_camera_swap_pairs(
        train_view_count,
        include_self=include_self,
        include_cross=include_cross,
        train_camera_names=train_camera_names,
    )
    count = int(pairs_per_step)
    if count <= 0:
        return shuffle_camera_swap_pairs(all_pairs, generator=generator)

    if self_pair_probability is None:
        if count <= len(all_pairs):
            return shuffle_camera_swap_pairs(all_pairs, generator=generator)[:count]
        indices = torch.randint(len(all_pairs), (count,), generator=generator).tolist()
        return tuple(all_pairs[int(index)] for index in indices)

    probability = float(self_pair_probability)
    if probability < 0.0 or probability > 1.0:
        raise ValueError(f"self_pair_probability must be in [0, 1], got {self_pair_probability}.")
    self_pairs = tuple(pair for pair in all_pairs if pair.is_self_reconstruction)
    cross_pairs = tuple(pair for pair in all_pairs if pair.is_train_cross_view)
    if not self_pairs and not cross_pairs:
        raise ValueError("No eligible camera swap pairs were built.")

    sampled = []
    for _ in range(count):
        want_self = bool(torch.rand((), generator=generator).item() < probability)
        if want_self and self_pairs:
            sampled.append(_sample_from(self_pairs, generator=generator))
        elif (not want_self) and cross_pairs:
            sampled.append(_sample_from(cross_pairs, generator=generator))
        elif self_pairs:
            sampled.append(_sample_from(self_pairs, generator=generator))
        else:
            sampled.append(_sample_from(cross_pairs, generator=generator))
    return tuple(sampled)


def camera_swap_pair_counts(pairs: Sequence[CameraSwapPair]) -> dict[str, int]:
    return {
        "total": len(pairs),
        "self": sum(1 for pair in pairs if pair.is_self_reconstruction),
        "train_cross": sum(1 for pair in pairs if pair.is_train_cross_view),
        "heldout": sum(1 for pair in pairs if pair.is_heldout_query),
    }


__all__ = [
    "CameraSet",
    "CameraSwapPair",
    "build_heldout_camera_swap_pairs",
    "build_train_camera_swap_pairs",
    "camera_swap_pair_counts",
    "sample_train_camera_swap_pairs",
    "shuffle_camera_swap_pairs",
]
