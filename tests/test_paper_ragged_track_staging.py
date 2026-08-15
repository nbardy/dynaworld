from __future__ import annotations

from typing import Any

import pytest
import torch
from camera import CameraSpec
from paper_ragged_track_staging import adapt_paper_spacetime_batch_to_track_groups
from paper_training_protocol import SpacetimeEpochSampler
from paper_training_types import SpacetimeBatch, SpacetimeSample
from powerfoam_training_data import PowerFoamRayProvider, PowerFoamTargetProvider


class RecordingTargetSource:
    view_count = 3
    frame_count = 5
    height = 4
    width = 5

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def select_view_frames(
        self,
        view_indices: tuple[int, ...],
        frame_indices: tuple[int, ...],
    ) -> torch.Tensor:
        self.calls.append((view_indices, frame_indices))
        return torch.stack(
            [
                torch.full(
                    (3, self.height, self.width),
                    float(view * 10 + frame),
                    dtype=torch.float32,
                )
                for view, frame in zip(view_indices, frame_indices, strict=True)
            ]
        )

    def residency(self) -> dict[str, Any]:
        return {
            "source_kind": "recording_lazy_fixture",
            "source_device": "disk",
            "logical_bytes": self.view_count * self.frame_count * 3 * self.height * self.width * 4,
            "resident_bytes": 0,
            "full_source_resident": False,
            "disk_lazy_decode": True,
        }


def _cameras(*, views: int = 3, frames: int = 5) -> tuple[tuple[CameraSpec, ...], ...]:
    return tuple(
        tuple(
            CameraSpec(
                fx=5.0,
                fy=4.0,
                cx=2.5,
                cy=2.0,
                camera_to_world=torch.eye(4),
            )
            for _frame in range(frames)
        )
        for _view in range(views)
    )


def _providers() -> tuple[RecordingTargetSource, PowerFoamTargetProvider, PowerFoamRayProvider]:
    source = RecordingTargetSource()
    device = torch.device("cpu")
    return (
        source,
        PowerFoamTargetProvider(source=source, device=device),
        PowerFoamRayProvider(
            cameras=_cameras(),
            height=source.height,
            width=source.width,
            device=device,
        ),
    )


def _ragged_batch() -> SpacetimeBatch:
    return SpacetimeBatch(
        samples=(
            SpacetimeSample(view_index=2, frame_index=4),
            SpacetimeSample(view_index=0, frame_index=1),
            SpacetimeSample(view_index=2, frame_index=0),
            SpacetimeSample(view_index=1, frame_index=3),
            SpacetimeSample(view_index=0, frame_index=4),
        ),
        epoch=3,
        batch_index=7,
        completes_epoch=False,
    )


def test_paper_batch_groups_by_view_without_rectangularizing_observations() -> None:
    _source, target_provider, ray_provider = _providers()
    frame_times = torch.tensor([0.0, 0.125, 0.4, 0.8, 1.0])

    adapted = adapt_paper_spacetime_batch_to_track_groups(
        _ragged_batch(),
        target_provider=target_provider,
        ray_provider=ray_provider,
        frame_times=frame_times,
        height=2,
        width=3,
        device=torch.device("cpu"),
    )

    assert adapted.active_view_count == 3
    assert adapted.pixel_count == 6
    assert adapted.observation_count == 5
    assert adapted.global_rgb_element_count == 6 * 5 * 3
    assert [group.view_index for group in adapted.groups] == [0, 1, 2]
    assert [group.batch_positions.tolist() for group in adapted.groups] == [[1, 4], [3], [0, 2]]
    assert [(group.logical_sample_start, group.logical_sample_end) for group in adapted.groups] == [
        (0, 2),
        (2, 3),
        (3, 5),
    ]
    assert [group.staging_plan.sample_indices.tolist() for group in adapted.groups] == [
        [1, 4],
        [8],
        [14, 10],
    ]
    for actual, expected in zip(
        (group.staging_plan.sample_times for group in adapted.groups),
        (torch.tensor([0.125, 1.0]), torch.tensor([0.8]), torch.tensor([1.0, 0.0])),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)
    assert [group.observation_count for group in adapted.groups] == [2, 1, 2]
    assert [group.logical_step_weight for group in adapted.groups] == [0.4, 0.2, 0.4]
    assert sum(group.logical_step_weight for group in adapted.groups) == 1.0
    assert sum(group.observation_count for group in adapted.groups) == adapted.observation_count
    assert sum(group.local_rgb_element_count for group in adapted.groups) == adapted.global_rgb_element_count

    accounting = adapted.accounting()
    assert accounting["adapter_owned_dense_pixel_observation_tensors"] == 0
    assert accounting["adapter_owned_explicit_ray_tensors"] == 0
    assert accounting["adapter_owned_tensor_storage_bytes"] == 6 * 8 + 5 * (8 + 8 + 4)
    assert accounting["view_observation_counts"] == {"0": 2, "1": 1, "2": 2}
    assert accounting["view_logical_sample_ranges"] == {"0": [0, 2], "1": [2, 3], "2": [3, 5]}
    assert accounting["view_logical_step_weights"] == {"0": 0.4, "1": 0.2, "2": 0.4}
    assert accounting["target_provider_residency"]["full_source_resident"] is False
    assert accounting["ray_provider_camera_record_count"] == 3 * 5


def test_real_paper_sampler_batch_preserves_every_observation_through_ragged_grouping() -> None:
    _source, target_provider, ray_provider = _providers()
    batch = SpacetimeEpochSampler(
        view_count=3,
        frame_indices=range(5),
        batch_size=5,
        same_time_count=2,
        local_time_count=1,
        local_time_radius=1,
        seed=0,
    ).next_batch()

    adapted = adapt_paper_spacetime_batch_to_track_groups(
        batch,
        target_provider=target_provider,
        ray_provider=ray_provider,
        frame_times=torch.linspace(0.0, 1.0, 5),
        height=2,
        width=3,
        device=torch.device("cpu"),
    )

    assert sorted(group.observation_count for group in adapted.groups) == [1, 2, 2]
    recovered: list[tuple[int, int] | None] = [None] * len(batch.samples)
    for group in adapted.groups:
        for position, flat_index in zip(
            group.batch_positions.tolist(),
            group.staging_plan.sample_indices.tolist(),
            strict=True,
        ):
            recovered[position] = divmod(flat_index, target_provider.frame_count)
    assert recovered == [(sample.view_index, sample.frame_index) for sample in batch.samples]
    assert adapted.global_rgb_element_count == 6 * len(batch.samples) * 3


def test_ragged_groups_fit_existing_native_token_scalar_contract_without_padding() -> None:
    _source, target_provider, ray_provider = _providers()
    adapted = adapt_paper_spacetime_batch_to_track_groups(
        _ragged_batch(),
        target_provider=target_provider,
        ray_provider=ray_provider,
        frame_times=torch.linspace(0.0, 1.0, 5),
        height=2,
        width=3,
        device=torch.device("cpu"),
    )

    coordinates = [group.native_sample_state_coordinates() for group in adapted.groups]
    assert coordinates == [
        {
            "global_track_count": 6,
            "global_sample_count": 5,
            "global_sample_start": 0,
            "global_sample_end": 2,
            "global_loss_element_count": 6 * 5 * 3,
        },
        {
            "global_track_count": 6,
            "global_sample_count": 5,
            "global_sample_start": 2,
            "global_sample_end": 3,
            "global_loss_element_count": 6 * 5 * 3,
        },
        {
            "global_track_count": 6,
            "global_sample_count": 5,
            "global_sample_start": 3,
            "global_sample_end": 5,
            "global_loss_element_count": 6 * 5 * 3,
        },
    ]
    assert all(
        row["global_loss_element_count"] == row["global_track_count"] * row["global_sample_count"] * 3
        for row in coordinates
    )
    assert sum(row["global_sample_end"] - row["global_sample_start"] for row in coordinates) == 5

    second_chart = adapted.groups[2].native_sample_state_coordinates(
        local_sample_start=1,
        local_sample_end=2,
    )
    assert (second_chart["global_sample_start"], second_chart["global_sample_end"]) == (4, 5)


def test_ragged_target_staging_keeps_one_global_denominator_and_decodes_one_frame_at_a_time() -> None:
    source, target_provider, ray_provider = _providers()
    adapted = adapt_paper_spacetime_batch_to_track_groups(
        _ragged_batch(),
        target_provider=target_provider,
        ray_provider=ray_provider,
        frame_times=torch.tensor([0.0, 0.125, 0.4, 0.8, 1.0]),
        height=2,
        width=3,
        device=torch.device("cpu"),
        loss_normalization_id="paper-step-17",
    )

    staged = [group.stage_targets(track_start=1, track_end=5) for group in adapted.groups]

    assert adapted.loss_normalization_id == "paper-step-17"
    assert [block.batch_positions.tolist() for block in staged] == [[1, 4], [3], [0, 2]]
    assert [(block.logical_sample_start, block.logical_sample_end) for block in staged] == [
        (0, 2),
        (2, 3),
        (3, 5),
    ]
    assert {block.normalization.global_rgb_element_count for block in staged} == {6 * 5 * 3}
    assert [block.staged.normalization.global_rgb_element_count for block in staged] == [
        6 * 2 * 3,
        6 * 1 * 3,
        6 * 2 * 3,
    ]
    assert {block.normalization.global_track_count for block in staged} == {6}
    assert {block.normalization.global_sample_count for block in staged} == {5}
    assert [block.normalization.block_rgb_element_count for block in staged] == [4 * 2 * 3, 4 * 1 * 3, 4 * 2 * 3]
    assert all(block.targets.shape == (4, block.batch_positions.numel(), 3) for block in staged)
    assert not any(hasattr(block, "rays") for block in staged)
    assert all(block.accounting["global_denominator_preserved"] is True for block in staged)
    assert all(block.accounting["explicit_rays_staged"] is False for block in staged)
    assert all(len(views) == len(frames) == 1 for views, frames in source.calls)
    assert source.calls == [
        ((0,), (1,)),
        ((0,), (4,)),
        ((1,), (3,)),
        ((2,), (4,)),
        ((2,), (0,)),
    ]
    for actual, value in (
        (staged[0].targets[:, 0], 1.0),
        (staged[0].targets[:, 1], 4.0),
        (staged[1].targets[:, 0], 13.0),
        (staged[2].targets[:, 0], 24.0),
        (staged[2].targets[:, 1], 20.0),
    ):
        torch.testing.assert_close(actual, torch.full((4, 3), value), rtol=0.0, atol=2.0e-6)


def test_ragged_adapter_fails_closed_on_dataset_or_time_identity_drift() -> None:
    _source, target_provider, ray_provider = _providers()
    batch = _ragged_batch()

    with pytest.raises(ValueError, match="exactly one value per provider frame"):
        adapt_paper_spacetime_batch_to_track_groups(
            batch,
            target_provider=target_provider,
            ray_provider=ray_provider,
            frame_times=torch.tensor([0.0, 1.0]),
        )

    mismatched_rays = PowerFoamRayProvider(
        cameras=_cameras(frames=4),
        height=target_provider.height,
        width=target_provider.width,
        device=torch.device("cpu"),
    )
    with pytest.raises(ValueError, match="share one view/frame grid"):
        adapt_paper_spacetime_batch_to_track_groups(
            batch,
            target_provider=target_provider,
            ray_provider=mismatched_rays,
            frame_times=torch.linspace(0.0, 1.0, 5),
        )

    invalid_batch = SpacetimeBatch(
        samples=(SpacetimeSample(view_index=3, frame_index=0),),
        epoch=0,
        batch_index=0,
        completes_epoch=False,
    )
    with pytest.raises(IndexError, match="sample view 3"):
        adapt_paper_spacetime_batch_to_track_groups(
            invalid_batch,
            target_provider=target_provider,
            ray_provider=ray_provider,
            frame_times=torch.linspace(0.0, 1.0, 5),
        )
