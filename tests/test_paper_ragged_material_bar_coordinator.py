from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import paper_ragged_material_bar_coordinator as coordinator
import pytest
import torch
from camera import CameraSpec
from paper_ragged_material_bar_coordinator import (
    begin_paper_ragged_material_bar_step,
    consume_paper_ragged_compact_material_bar_result,
    finalize_paper_ragged_material_bar_step,
    prepare_paper_ragged_material_spatial_block,
    prepare_paper_ragged_material_view_program,
    run_paper_ragged_material_bar_step,
    seal_paper_ragged_compact_material_bar_result,
    stage_next_paper_ragged_material_bar_request,
)
from paper_ragged_track_staging import adapt_paper_spacetime_batch_to_track_groups
from paper_training_types import SpacetimeBatch, SpacetimeSample
from powerfoam_training_data import PowerFoamRayProvider, PowerFoamTargetProvider


class _LazyTargets:
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
            "source_kind": "lazy_test_source",
            "source_device": "disk",
            "logical_bytes": self.view_count * self.frame_count * 3 * self.height * self.width * 4,
            "resident_bytes": 0,
            "full_source_resident": False,
            "disk_lazy_decode": True,
        }


@dataclass
class _WorldToken:
    generation_digest: str
    source_site_ids_i64: torch.Tensor | None = None
    stale: bool = False

    def assert_current(self) -> None:
        if self.stale:
            raise ValueError("fake view world is stale")


class _CountingOptimizer:
    def __init__(self) -> None:
        self.step_count = 0
        self.seen_gradient: torch.Tensor | None = None

    def step(self, result) -> None:
        self.step_count += 1
        self.seen_gradient = result.grad_global_site_rgba_f32.clone()


def _cameras() -> tuple[tuple[CameraSpec, ...], ...]:
    return tuple(
        tuple(
            CameraSpec(
                fx=5.0,
                fy=4.0,
                cx=2.5,
                cy=2.0,
                camera_to_world=torch.eye(4),
            )
            for _frame in range(5)
        )
        for _view in range(3)
    )


def _ragged_case(*, normalization_id: str = "paper-ragged-step-17"):
    source = _LazyTargets()
    target_provider = PowerFoamTargetProvider(source=source, device=torch.device("cpu"))
    ray_provider = PowerFoamRayProvider(
        cameras=_cameras(),
        height=source.height,
        width=source.width,
        device=torch.device("cpu"),
    )
    # Unequal K_v = (3, 1, 2), deliberately interleaved in sampler order.
    batch = SpacetimeBatch(
        samples=(
            SpacetimeSample(view_index=2, frame_index=4),
            SpacetimeSample(view_index=0, frame_index=1),
            SpacetimeSample(view_index=2, frame_index=0),
            SpacetimeSample(view_index=1, frame_index=3),
            SpacetimeSample(view_index=0, frame_index=4),
            SpacetimeSample(view_index=0, frame_index=2),
        ),
        epoch=3,
        batch_index=7,
        completes_epoch=False,
    )
    adapted = adapt_paper_spacetime_batch_to_track_groups(
        batch,
        target_provider=target_provider,
        ray_provider=ray_provider,
        frame_times=torch.tensor([0.0, 0.125, 0.4, 0.8, 1.0]),
        height=2,
        width=3,
        device=torch.device("cpu"),
        loss_normalization_id=normalization_id,
    )
    return source, adapted


def _programs(adapted, track_ranges: tuple[tuple[int, int], ...]):
    programs = []
    for group in adapted.groups:
        blocks = []
        for block_index, (track_start, track_end) in enumerate(track_ranges):
            generation = f"view-{group.view_index}-block-{block_index}-generation"
            world = _WorldToken(
                generation,
                source_site_ids_i64=torch.tensor([0, 1, 1], dtype=torch.int64),
            )
            blocks.append(
                prepare_paper_ragged_material_spatial_block(
                    block_id=f"view-{group.view_index}-pixels-{track_start}-{track_end}",
                    view_index=group.view_index,
                    track_start=track_start,
                    track_end=track_end,
                    world_token=world,
                    world_generation_id=generation,
                    # Site 1 deliberately appears twice per compact block and
                    # every global site id is reused across views/partitions.
                    source_site_ids=torch.tensor([0, 1, 1], dtype=torch.int64),
                    global_site_count=4,
                    device="cpu",
                )
            )
        programs.append(
            prepare_paper_ragged_material_view_program(
                view_index=group.view_index,
                global_track_count=adapted.pixel_count,
                global_site_count=4,
                blocks=blocks,
            )
        )
    # Intentionally return a noncanonical program order. Program registration
    # must not impose a numerical view order on the reduction.
    return tuple(reversed(programs))


def _executor(optimizer: _CountingOptimizer, requests: list[tuple[int, int, int, int]]):
    def execute(request):
        assert optimizer.step_count == 0
        assert request.global_loss_element_count == request.global_track_count * request.global_observation_count * 3
        assert request.loss_normalization_id == "paper-ragged-step-17"
        requests.append(
            (
                request.view_index,
                request.block.track_start,
                request.logical_sample_start,
                request.logical_sample_end,
            )
        )
        target = request.target_rgb
        scale = request.global_loss_scale
        rgb_sum = target.sum(dim=(0, 1)) * scale
        sample_pixels = target.shape[0] * target.shape[1]
        gradient = torch.zeros((3, 4), dtype=torch.float32)
        factors = (1.0, 0.25, 0.75)
        for compact_row, factor in enumerate(factors):
            gradient[compact_row, :3] = rgb_sum * factor
            gradient[compact_row, 3] = float(sample_pixels) * scale * factor
        return seal_paper_ragged_compact_material_bar_result(
            request,
            loss_f32=(target.square().sum() * scale).reshape(1).contiguous(),
            grad_compact_site_rgba_f32=gradient,
        )

    return execute


def _run(adapted, *, track_ranges, sample_block_size: int, view_order):
    optimizer = _CountingOptimizer()
    requests: list[tuple[int, int, int, int]] = []
    global_gradient = torch.full((4, 4), 123.0, dtype=torch.float32)
    result = run_paper_ragged_material_bar_step(
        adapted,
        programs=_programs(adapted, track_ranges),
        global_grad_site_rgba_f32=global_gradient,
        executor=_executor(optimizer, requests),
        optimizer_update=optimizer.step,
        sample_block_size=sample_block_size,
        view_order=view_order,
    )
    return optimizer, requests, global_gradient, result


def test_ragged_coordinator_is_invariant_to_view_order_and_block_partition() -> None:
    source_a, adapted_a = _ragged_case()
    optimizer_a, requests_a, gradient_a, result_a = _run(
        adapted_a,
        track_ranges=((0, 6),),
        sample_block_size=2,
        view_order=(0, 1, 2),
    )
    source_b, adapted_b = _ragged_case()
    optimizer_b, requests_b, gradient_b, result_b = _run(
        adapted_b,
        track_ranges=((0, 2), (2, 6)),
        sample_block_size=1,
        view_order=(2, 0, 1),
    )

    assert [group.observation_count for group in adapted_a.groups] == [3, 1, 2]
    torch.testing.assert_close(result_a.step.loss_f32, result_b.step.loss_f32)
    torch.testing.assert_close(gradient_a, gradient_b)
    torch.testing.assert_close(optimizer_a.seen_gradient, gradient_a)
    torch.testing.assert_close(optimizer_b.seen_gradient, gradient_b)
    assert optimizer_a.step_count == optimizer_b.step_count == 1
    assert result_a.optimizer_update_callback_count == result_b.optimizer_update_callback_count == 1
    assert requests_a[0][0] == 0
    assert requests_b[0][0] == 2

    values = [1.0, 4.0, 2.0, 13.0, 24.0, 20.0]
    total_rgb_sum = sum(value * adapted_a.pixel_count for value in values)
    total_pixel_samples = adapted_a.pixel_count * adapted_a.observation_count
    scale = 1.0 / adapted_a.global_rgb_element_count
    expected = torch.zeros_like(gradient_a)
    expected[0, :3] = total_rgb_sum * scale
    expected[0, 3] = total_pixel_samples * scale
    # Compact rows 1 and 2 both map to global site 1, with factors summing to 1.
    expected[1] = expected[0]
    torch.testing.assert_close(gradient_a, expected)

    accounting = result_b.step.accounting
    assert accounting["global_loss_element_count"] == 6 * 6 * 3
    assert accounting["consumed_rgb_element_count"] == accounting["global_loss_element_count"]
    assert accounting["loss_normalization_id"] == "paper-ragged-step-17"
    assert accounting["global_site_gradient_buffer_count"] == 1
    assert accounting["per_view_global_gradient_buffers"] == 0
    assert accounting["persistent_target_tensor_bytes"] == 0
    assert accounting["persistent_explicit_ray_tensor_bytes"] == 0
    assert accounting["peak_in_flight_target_blocks"] == 1
    assert accounting["sample_partition_records_retained"] == 0
    assert accounting["view_time_cartesian_tensor_allocated"] is False
    assert accounting["peak_staged_target_bytes"] < adapted_b.global_rgb_element_count * 4
    assert all(len(views) == len(frames) == 1 for views, frames in source_a.calls + source_b.calls)


def test_coverage_ledger_rejects_duplicate_gap_and_incomplete_finalize() -> None:
    _source, adapted = _ragged_case()
    programs = _programs(adapted, ((0, 6),))
    gradient = torch.zeros((4, 4), dtype=torch.float32)
    ledger = begin_paper_ragged_material_bar_step(
        adapted,
        programs=programs,
        global_grad_site_rgba_f32=gradient,
    )
    optimizer = _CountingOptimizer()
    request = stage_next_paper_ragged_material_bar_request(
        ledger,
        view_index=0,
        block_id="view-0-pixels-0-6",
        local_sample_start=0,
        local_sample_end=1,
    )
    result = _executor(optimizer, [])(request)
    consume_paper_ragged_compact_material_bar_result(ledger, request, result)

    with pytest.raises(ValueError, match="duplicate or overlapping"):
        stage_next_paper_ragged_material_bar_request(
            ledger,
            view_index=0,
            block_id="view-0-pixels-0-6",
            local_sample_start=0,
            local_sample_end=1,
        )
    with pytest.raises(ValueError, match="gap"):
        stage_next_paper_ragged_material_bar_request(
            ledger,
            view_index=0,
            block_id="view-0-pixels-0-6",
            local_sample_start=2,
            local_sample_end=3,
        )
    with pytest.raises(ValueError, match="missing logical sample coverage"):
        finalize_paper_ragged_material_bar_step(ledger)
    assert optimizer.step_count == 0


def test_result_provenance_rejects_other_world_site_map_denominator_and_stale_tokens() -> None:
    _source_a, adapted_a = _ragged_case()
    programs_a = _programs(adapted_a, ((0, 6),))
    ledger_a = begin_paper_ragged_material_bar_step(
        adapted_a,
        programs=programs_a,
        global_grad_site_rgba_f32=torch.zeros((4, 4), dtype=torch.float32),
    )
    request_a = stage_next_paper_ragged_material_bar_request(
        ledger_a,
        view_index=0,
        block_id="view-0-pixels-0-6",
        local_sample_start=0,
        local_sample_end=1,
    )

    _source_b, adapted_b = _ragged_case(normalization_id="different-denominator-id")
    programs_b = _programs(adapted_b, ((0, 6),))
    ledger_b = begin_paper_ragged_material_bar_step(
        adapted_b,
        programs=programs_b,
        global_grad_site_rgba_f32=torch.zeros((4, 4), dtype=torch.float32),
    )
    request_b = stage_next_paper_ragged_material_bar_request(
        ledger_b,
        view_index=0,
        block_id="view-0-pixels-0-6",
        local_sample_start=0,
        local_sample_end=1,
    )
    foreign_result = seal_paper_ragged_compact_material_bar_result(
        request_b,
        loss_f32=torch.zeros((1,), dtype=torch.float32),
        grad_compact_site_rgba_f32=torch.zeros((3, 4), dtype=torch.float32),
    )
    with pytest.raises(ValueError, match="world, site-map, denominator, or request provenance"):
        consume_paper_ragged_compact_material_bar_result(ledger_a, request_a, foreign_result)

    request_a.block.world_token.stale = True
    with pytest.raises(ValueError, match="fake view world is stale"):
        seal_paper_ragged_compact_material_bar_result(
            request_a,
            loss_f32=torch.zeros((1,), dtype=torch.float32),
            grad_compact_site_rgba_f32=torch.zeros((3, 4), dtype=torch.float32),
        )
    request_a.block.world_token.stale = False
    request_a.block.source_site_ids_i64[0] = 3
    with pytest.raises(ValueError, match="site mapping is stale"):
        seal_paper_ragged_compact_material_bar_result(
            request_a,
            loss_f32=torch.zeros((1,), dtype=torch.float32),
            grad_compact_site_rgba_f32=torch.zeros((3, 4), dtype=torch.float32),
        )

    generation = "mismatched-world-map-generation"
    with pytest.raises(ValueError, match="does not match its world token"):
        prepare_paper_ragged_material_spatial_block(
            block_id="mismatched-world-map",
            view_index=0,
            track_start=0,
            track_end=6,
            world_token=_WorldToken(
                generation,
                source_site_ids_i64=torch.tensor([0, 2], dtype=torch.int64),
            ),
            world_generation_id=generation,
            source_site_ids=torch.tensor([0, 1], dtype=torch.int64),
            global_site_count=4,
            device="cpu",
        )


def test_optimizer_authorization_is_single_use_and_issued_only_after_coverage() -> None:
    _source, adapted = _ragged_case()
    programs = _programs(adapted, ((0, 6),))
    ledger = begin_paper_ragged_material_bar_step(
        adapted,
        programs=programs,
        global_grad_site_rgba_f32=torch.zeros((4, 4), dtype=torch.float32),
    )
    optimizer = _CountingOptimizer()
    executor = _executor(optimizer, [])
    for view_index in (2, 1, 0):
        group = next(group for group in adapted.groups if group.view_index == view_index)
        request = stage_next_paper_ragged_material_bar_request(
            ledger,
            view_index=view_index,
            block_id=f"view-{view_index}-pixels-0-6",
            local_sample_start=0,
            local_sample_end=group.observation_count,
        )
        consume_paper_ragged_compact_material_bar_result(ledger, request, executor(request))
        assert optimizer.step_count == 0

    authorization = finalize_paper_ragged_material_bar_step(ledger)
    assert optimizer.step_count == 0
    authorization.consume(optimizer.step)
    assert optimizer.step_count == 1
    with pytest.raises(ValueError, match="already consumed"):
        authorization.consume(optimizer.step)
    assert optimizer.step_count == 1


def test_warm_accumulation_and_finalize_do_not_sync_or_hash_device_payloads() -> None:
    for function in (
        consume_paper_ragged_compact_material_bar_result,
        finalize_paper_ragged_material_bar_step,
        coordinator._assert_ledger_current,
        coordinator._assert_world_token_current,
        coordinator.PaperRaggedMaterialSpatialBlock.assert_current,
        coordinator.PaperRaggedMaterialBarRequest.assert_current,
        coordinator.PaperRaggedCompactMaterialBarResult.assert_current,
    ):
        source = inspect.getsource(function)
        for forbidden in (".cpu(", ".item(", ".tolist(", "_digest_parts(", "_step_generation_id("):
            assert forbidden not in source
