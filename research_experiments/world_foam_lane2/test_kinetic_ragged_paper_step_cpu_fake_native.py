from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Any

import kinetic_ragged_paper_step_cpu_fake_native as paper_step
import pytest
import torch
from camera import CameraSpec
from kinetic_multichart_transfer_program import compile_kinetic_multichart_p0_program
from kinetic_native_equal_rank_lowering import (
    iter_materialize_kinetic_native_equal_rank_blocks,
    kinetic_native_equal_rank_chart_sources_for_track,
    lower_kinetic_native_equal_rank_buckets,
)
from kinetic_native_equal_rank_runtime_adapter import (
    prepare_kinetic_native_equal_rank_runtime_block,
)
from kinetic_native_topology_lowering import lower_kinetic_multichart_to_native_topology
from kinetic_owner_chart_compiler import compile_exact_kinetic_owner_charts
from kinetic_power_word_compiler import AffineKineticPowerSites
from paper_kinetic_ragged_sample_plan import (
    iter_paper_kinetic_row_ragged_sample_blocks,
    prepare_paper_kinetic_row_ragged_sampler,
)
from paper_kinetic_union_local_bar_assembly import (
    prepare_paper_kinetic_union_local_spatial_bundle,
)
from paper_ragged_material_bar_coordinator import (
    prepare_paper_ragged_material_spatial_block,
    prepare_paper_ragged_material_view_program,
)
from paper_ragged_track_staging import adapt_paper_spacetime_batch_to_track_groups
from paper_training_types import SpacetimeBatch, SpacetimeSample
from powerfoam_training_data import PowerFoamRayProvider, PowerFoamTargetProvider


def _node_charts(
    word_offsets_i32: torch.Tensor,
    word_owner_i32: torch.Tensor,
    node_physical_length_f32: torch.Tensor,
    site_rgba_f32: torch.Tensor,
) -> torch.Tensor:
    rows = []
    for row_index in range(word_offsets_i32.numel() - 1):
        start = int(word_offsets_i32[row_index])
        end = int(word_offsets_i32[row_index + 1])
        nodes = []
        for node_index in range(node_physical_length_f32.shape[0]):
            total_kappa = torch.zeros((), dtype=torch.float32)
            total_beta = torch.ones((), dtype=torch.float32)
            total_moment = torch.zeros((3,), dtype=torch.float32)
            for word_index in range(start, end):
                owner = word_owner_i32[word_index].to(dtype=torch.int64)
                rgba = site_rgba_f32[owner]
                optical_depth = rgba[3] * node_physical_length_f32[node_index, word_index]
                beta = torch.exp(-optical_depth)
                alpha = -torch.expm1(-optical_depth)
                total_moment = total_moment + total_beta * alpha * rgba[:3]
                total_beta = total_beta * beta
                total_kappa = total_kappa + optical_depth
            inverse_phi = _inverse_phi(total_kappa)
            nodes.append(torch.cat((total_kappa.reshape(1), inverse_phi * total_moment)))
        rows.append(torch.stack(nodes))
    return torch.stack(rows).contiguous()


def _inverse_phi(kappa: torch.Tensor) -> torch.Tensor:
    small = torch.abs(kappa) < 1.0e-4
    k2 = kappa.square()
    series = 1.0 + 0.5 * kappa + k2 / 12.0 - k2.square() / 720.0 + k2.pow(3) / 30240.0
    denominator = -torch.expm1(-kappa)
    return torch.where(
        small,
        series,
        kappa / torch.where(small, torch.ones_like(denominator), denominator),
    )


def _phi(kappa: torch.Tensor) -> torch.Tensor:
    small = torch.abs(kappa) < 1.0e-4
    k2 = kappa.square()
    series = 1.0 - 0.5 * kappa + k2 / 6.0 - k2 * kappa / 24.0 + k2.square() / 120.0
    return torch.where(
        small,
        series,
        -torch.expm1(-kappa) / torch.where(small, torch.ones_like(kappa), kappa),
    )


@dataclass(frozen=True)
class _PreparedFakeRaggedSample:
    node_chart_f32: torch.Tensor
    sample_row_i32: torch.Tensor
    sample_to_node_f32: torch.Tensor
    target_rgb_f32: torch.Tensor
    background_rgb_f32: torch.Tensor
    loss_scale: float
    cone_tolerance: float


class _FakeNativeOps:
    """CPU object exposing the exact production word and sample op surface."""

    def __init__(self) -> None:
        self.forward_calls = 0
        self.vjp_calls = 0
        self.material_vjp_calls = 0
        self.sample_prepare_calls = 0
        self.sample_launch_calls = 0

    def prepare_kinetic_ragged_p0_lie_sample_block(
        self,
        node_chart_f32,
        sample_row_i32,
        sample_to_node_f32,
        target_rgb_f32,
        background_rgb_f32,
        *,
        loss_scale,
        cone_tolerance=1.0e-5,
    ):
        assert sample_row_i32.device.type == "cpu"
        assert sample_to_node_f32.device == node_chart_f32.device
        assert target_rgb_f32.device == node_chart_f32.device
        assert background_rgb_f32.device == node_chart_f32.device
        assert tuple(sample_to_node_f32.shape) == (
            sample_row_i32.numel(),
            node_chart_f32.shape[1],
        )
        self.sample_prepare_calls += 1
        return _PreparedFakeRaggedSample(
            node_chart_f32=node_chart_f32,
            sample_row_i32=sample_row_i32,
            sample_to_node_f32=sample_to_node_f32,
            target_rgb_f32=target_rgb_f32,
            background_rgb_f32=background_rgb_f32,
            loss_scale=float(loss_scale),
            cone_tolerance=float(cone_tolerance),
        )

    def kinetic_ragged_p0_lie_sample_accumulate_loss_only_launch_only(
        self,
        prepared,
        loss_f32,
        grad_node_chart_f32,
        cone_diagnostic_i32,
    ):
        assert isinstance(prepared, _PreparedFakeRaggedSample)
        rows = prepared.sample_row_i32.to(dtype=torch.int64)
        selected = prepared.node_chart_f32.index_select(0, rows)
        chart = torch.sum(selected * prepared.sample_to_node_f32[:, :, None], dim=1)
        kappa = chart[:, 0]
        velocity = chart[:, 1:]
        cone_violation = torch.maximum(
            torch.maximum(-kappa, -torch.amin(velocity, dim=1)),
            torch.amax(velocity, dim=1) - kappa,
        )
        assert bool(torch.all(cone_violation <= prepared.cone_tolerance).item())
        phi, phi_prime = paper_step._lie_phi_and_derivative_f32(kappa)
        beta = torch.exp(-kappa)
        prediction = (
            phi[:, None] * velocity
            + beta[:, None] * prepared.background_rgb_f32
        )
        residual = prediction - prepared.target_rgb_f32
        loss_f32.add_(residual.square().sum() * prepared.loss_scale)
        grad_prediction = (2.0 * prepared.loss_scale) * residual
        grad_beta = torch.sum(
            grad_prediction * prepared.background_rgb_f32,
            dim=1,
        )
        grad_chart = torch.cat(
            (
                (
                    -beta * grad_beta
                    + phi_prime * torch.sum(velocity * grad_prediction, dim=1)
                )[:, None],
                phi[:, None] * grad_prediction,
            ),
            dim=1,
        )
        contribution = prepared.sample_to_node_f32[:, :, None] * grad_chart[:, None, :]
        grad_node_chart_f32.index_add_(0, rows, contribution)
        cone_diagnostic_i32[0] += int(prepared.sample_row_i32.numel())
        self.sample_launch_calls += 1

    def kinetic_precompiled_length_p0_lie_node_forward_launch_only(
        self,
        word_offsets_i32,
        word_owner_i32,
        node_physical_length_f32,
        site_rgba_f32,
        config_i32,
        config_f32,
        *,
        track_count,
        node_count,
    ):
        self._validate(
            word_offsets_i32,
            word_owner_i32,
            node_physical_length_f32,
            site_rgba_f32,
            config_i32,
            config_f32,
            track_count,
            node_count,
        )
        self.forward_calls += 1
        return _node_charts(
            word_offsets_i32,
            word_owner_i32,
            node_physical_length_f32,
            site_rgba_f32,
        )

    def kinetic_precompiled_length_p0_lie_node_forward_into_launch_only_v1(
        self,
        word_offsets_i32,
        word_owner_i32,
        node_physical_length_f32,
        site_rgba_f32,
        config_i32,
        config_f32,
        node_chart_out_f32,
        *,
        track_count,
        node_count,
    ):
        self._validate(
            word_offsets_i32,
            word_owner_i32,
            node_physical_length_f32,
            site_rgba_f32,
            config_i32,
            config_f32,
            track_count,
            node_count,
        )
        expected = _node_charts(
            word_offsets_i32,
            word_owner_i32,
            node_physical_length_f32,
            site_rgba_f32,
        )
        assert tuple(node_chart_out_f32.shape) == tuple(expected.shape)
        node_chart_out_f32.copy_(expected)
        self.forward_calls += 1
        return None

    def kinetic_precompiled_length_p0_lie_node_vjp_accumulate_launch_only(
        self,
        word_offsets_i32,
        word_owner_i32,
        node_physical_length_f32,
        site_rgba_f32,
        node_chart_f32,
        grad_node_chart_f32,
        grad_site_rgba_f32,
        config_i32,
        config_f32,
        *,
        track_count,
        node_count,
    ):
        self._validate(
            word_offsets_i32,
            word_owner_i32,
            node_physical_length_f32,
            site_rgba_f32,
            config_i32,
            config_f32,
            track_count,
            node_count,
        )
        assert tuple(node_chart_f32.shape) == (track_count, node_count, 4)
        with torch.enable_grad():
            rgba = site_rgba_f32.detach().clone().requires_grad_(True)
            lengths = node_physical_length_f32.detach().clone().requires_grad_(True)
            node_chart = _node_charts(word_offsets_i32, word_owner_i32, lengths, rgba)
            grad_rgba, grad_lengths = torch.autograd.grad(
                torch.sum(node_chart * grad_node_chart_f32.detach()),
                (rgba, lengths),
            )
        grad_site_rgba_f32.add_(grad_rgba)
        self.vjp_calls += 1
        return grad_site_rgba_f32, grad_lengths.contiguous()

    def kinetic_precompiled_length_p0_lie_material_node_vjp_accumulate_launch_only(
        self,
        word_offsets_i32,
        word_owner_i32,
        node_physical_length_f32,
        site_rgba_f32,
        node_chart_f32,
        grad_node_chart_f32,
        grad_site_rgba_f32,
        config_i32,
        config_f32,
        *,
        track_count,
        node_count,
    ):
        self._validate(
            word_offsets_i32,
            word_owner_i32,
            node_physical_length_f32,
            site_rgba_f32,
            config_i32,
            config_f32,
            track_count,
            node_count,
        )
        assert tuple(node_chart_f32.shape) == (track_count, node_count, 4)
        with torch.enable_grad():
            rgba = site_rgba_f32.detach().clone().requires_grad_(True)
            node_chart = _node_charts(
                word_offsets_i32,
                word_owner_i32,
                node_physical_length_f32,
                rgba,
            )
            (grad_rgba,) = torch.autograd.grad(
                torch.sum(node_chart * grad_node_chart_f32.detach()),
                (rgba,),
            )
        grad_site_rgba_f32.add_(grad_rgba)
        self.material_vjp_calls += 1
        return grad_site_rgba_f32

    @staticmethod
    def _validate(
        word_offsets_i32,
        word_owner_i32,
        node_physical_length_f32,
        site_rgba_f32,
        config_i32,
        config_f32,
        track_count,
        node_count,
    ) -> None:
        assert tuple(config_i32.tolist()) == (
            track_count,
            node_count,
            site_rgba_f32.shape[0],
            word_owner_i32.numel(),
        )
        assert tuple(config_f32.shape) == (1,)
        assert tuple(word_offsets_i32.shape) == (track_count + 1,)
        assert tuple(node_physical_length_f32.shape) == (
            node_count,
            word_owner_i32.numel(),
        )


class _Targets:
    view_count = 1
    height = 1
    width = 3

    def __init__(self, frame_count: int) -> None:
        self.frame_count = frame_count

    def select_view_frames(
        self,
        view_indices: tuple[int, ...],
        frame_indices: tuple[int, ...],
    ) -> torch.Tensor:
        frames = []
        for view, frame in zip(view_indices, frame_indices, strict=True):
            assert view == 0
            normalized = frame / max(1, self.frame_count - 1)
            pixels = torch.arange(self.width, dtype=torch.float32)
            frames.append(
                torch.stack(
                    (
                        0.12 + 0.19 * normalized + 0.03 * pixels,
                        0.31 - 0.08 * normalized + 0.02 * pixels,
                        0.18 + 0.11 * normalized + 0.01 * pixels,
                    )
                ).reshape(3, self.height, self.width)
            )
        return torch.stack(frames)

    def residency(self) -> dict[str, Any]:
        return {
            "source_kind": "kinetic_cpu_fake_native_e2e_fixture",
            "source_device": "cpu",
            "logical_bytes": self.frame_count * 3 * self.height * self.width * 4,
            "resident_bytes": 0,
            "full_source_resident": False,
            "disk_lazy_decode": False,
        }


def _sites() -> AffineKineticPowerSites:
    slopes = [(0, 0), (-2, 0)]
    intercepts = [(0, 0, 0), (1, -1, 0)]
    positions = []
    velocities = []
    weights = []
    for (slope0, slope1), (bias0, bias1, bias2) in zip(
        slopes,
        intercepts,
        strict=True,
    ):
        position = -Fraction(slope0) / 2
        velocity = -Fraction(slope1) / 2
        positions.append((position, Fraction(0), Fraction(0)))
        velocities.append((velocity, Fraction(0), Fraction(0)))
        weights.append(
            (
                position * position - Fraction(bias0),
                2 * position * velocity - Fraction(bias1),
                velocity * velocity - Fraction(bias2),
            )
        )
    positions.extend(
        (
            (Fraction(100), Fraction(0), Fraction(0)),
            (Fraction(200), Fraction(0), Fraction(0)),
        )
    )
    velocities.extend(
        (
            (Fraction(0), Fraction(0), Fraction(0)),
            (Fraction(0), Fraction(0), Fraction(0)),
        )
    )
    weights.extend(
        (
            (Fraction(0), Fraction(0), Fraction(0)),
            (Fraction(0), Fraction(0), Fraction(0)),
        )
    )
    return AffineKineticPowerSites(
        positions0=torch.tensor(positions, dtype=torch.float64),
        velocities=torch.tensor(velocities, dtype=torch.float64),
        weight_coefficients=torch.tensor(weights, dtype=torch.float64),
    )


def _program(
    sites: AffineKineticPowerSites,
    *,
    ray_origin_x: int,
    node_count: int,
):
    ray = torch.tensor(
        [ray_origin_x, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        dtype=torch.float64,
    )
    owner_program = compile_exact_kinetic_owner_charts(
        sites,
        ray,
        t_min=-2,
        t_max=2,
        near=0,
        far=1,
    )
    return compile_kinetic_multichart_p0_program(
        owner_program,
        sites,
        ray,
        node_count=node_count,
    )


@dataclass(frozen=True)
class _CompiledCase:
    sampler: Any
    payloads: tuple[Any, ...]
    lanes: tuple[Any, ...]
    view_program: Any
    native_ops: _FakeNativeOps
    global_site_rgba_f32: torch.Tensor
    background_rgb_f32: torch.Tensor

    @property
    def lane(self):
        if len(self.lanes) != 1:
            raise ValueError("singular lane is available only for the one-block fixture")
        return self.lanes[0]


def _compiled_case(
    track_ranges: tuple[tuple[int, int], ...] = ((0, 3),),
) -> _CompiledCase:
    sites = _sites()
    programs = (
        _program(sites, ray_origin_x=-2, node_count=3),
        _program(sites, ray_origin_x=-1, node_count=4),
        _program(sites, ray_origin_x=0, node_count=5),
    )
    sources = tuple(
        source
        for track_id, program in enumerate(programs)
        for source in kinetic_native_equal_rank_chart_sources_for_track(
            track_id,
            program,
            lowering=lower_kinetic_multichart_to_native_topology(program),
        )
    )
    lowering = lower_kinetic_native_equal_rank_buckets(
        tuple(reversed(sources)),
        maximum_rows_per_block=2,
    )
    sampler = prepare_paper_kinetic_row_ragged_sampler(
        view_index=0,
        lowering=lowering,
        sources=sources,
    )
    payloads = tuple(iter_materialize_kinetic_native_equal_rank_blocks(lowering, sources))
    native_ops = _FakeNativeOps()
    runtimes = tuple(
        prepare_kinetic_native_equal_rank_runtime_block(
            payload,
            lowering=lowering,
            sources=sources,
            native_ops=native_ops,
            device="cpu",
        )
        for payload in payloads
    )
    runtime_by_digest = {runtime.payload.block.generation_digest: runtime for runtime in runtimes}
    spatial_blocks = []
    lanes = []
    for track_start, track_end in track_ranges:
        bundle = prepare_paper_kinetic_union_local_spatial_bundle(
            sampler,
            track_ids=tuple(range(track_start, track_end)),
            device="cpu",
        )
        spatial_block = prepare_paper_ragged_material_spatial_block(
            block_id=f"view-0-tracks-{track_start}-{track_end}",
            view_index=0,
            track_start=track_start,
            track_end=track_end,
            world_token=bundle,
            world_generation_id=bundle.generation_digest,
            source_site_ids=bundle.source_site_ids_i64,
            global_site_count=bundle.global_site_count,
            device="cpu",
        )
        spatial_blocks.append(spatial_block)
        lanes.append(
            paper_step.prepare_paper_kinetic_cpu_fake_native_spatial_lane(
                spatial_block,
                bundle,
                runtimes=tuple(
                    runtime_by_digest[binding.native_block_generation_digest] for binding in bundle.native_blocks
                ),
            )
        )
    view_program = prepare_paper_ragged_material_view_program(
        view_index=0,
        global_track_count=3,
        global_site_count=lowering.global_site_count,
        blocks=tuple(spatial_blocks),
    )
    global_rgba = torch.tensor(
        [
            [0.76, 0.18, 0.09, 0.43],
            [0.12, 0.68, 0.83, 0.71],
            [0.44, 0.33, 0.22, 0.25],
            [0.31, 0.42, 0.53, 0.19],
        ],
        dtype=torch.float32,
    )
    return _CompiledCase(
        sampler=sampler,
        payloads=payloads,
        lanes=tuple(lanes),
        view_program=view_program,
        native_ops=native_ops,
        global_site_rgba_f32=global_rgba,
        background_rgb_f32=torch.tensor([0.04, 0.07, 0.11], dtype=torch.float32),
    )


def _adapted(frame_count: int):
    source = _Targets(frame_count)
    target_provider = PowerFoamTargetProvider(source=source, device=torch.device("cpu"))
    cameras = (
        tuple(
            CameraSpec(
                fx=3.0,
                fy=3.0,
                cx=1.5,
                cy=0.5,
                camera_to_world=torch.eye(4),
            )
            for _frame in range(frame_count)
        ),
    )
    ray_provider = PowerFoamRayProvider(
        cameras=cameras,
        height=source.height,
        width=source.width,
        device=torch.device("cpu"),
    )
    batch = SpacetimeBatch(
        samples=tuple(SpacetimeSample(view_index=0, frame_index=frame) for frame in range(frame_count)),
        epoch=0,
        batch_index=frame_count,
        completes_epoch=False,
    )
    return adapt_paper_spacetime_batch_to_track_groups(
        batch,
        target_provider=target_provider,
        ray_provider=ray_provider,
        frame_times=torch.linspace(-2.0, 2.0, frame_count),
        height=1,
        width=3,
        device="cpu",
        loss_normalization_id=f"kinetic-e2e-f{frame_count}",
    )


@dataclass
class _RunCapture:
    optimizer_calls: int = 0
    optimizer_loss: torch.Tensor | None = None
    optimizer_gradient: torch.Tensor | None = None

    def optimizer_step(self, result) -> None:
        self.optimizer_calls += 1
        self.optimizer_loss = result.loss_f32.clone()
        self.optimizer_gradient = result.grad_global_site_rgba_f32.clone()


def _run(
    compiled: _CompiledCase,
    adapted,
    *,
    request_size: int,
    sample_launch_size: int,
):
    capture = _RunCapture()
    global_bar = torch.full_like(compiled.global_site_rgba_f32, 91.0)
    result = paper_step.run_paper_kinetic_cpu_fake_native_material_step(
        adapted,
        programs=(compiled.view_program,),
        lanes=compiled.lanes,
        global_site_rgba_f32=compiled.global_site_rgba_f32,
        global_grad_site_rgba_f32=global_bar,
        background_rgb_f32=compiled.background_rgb_f32,
        maximum_observations_per_request=request_size,
        maximum_samples_per_launch=sample_launch_size,
        optimizer_update=capture.optimizer_step,
    )
    return result, capture, global_bar


def _direct_oracle(compiled: _CompiledCase, adapted) -> tuple[torch.Tensor, torch.Tensor]:
    global_rgba = compiled.global_site_rgba_f32.detach().clone().requires_grad_(True)
    payload_by_digest = {payload.block.generation_digest: payload for payload in compiled.payloads}
    node_by_digest = {
        digest: _node_charts(
            payload.word_offsets_i32,
            payload.word_owner_i32,
            payload.node_physical_length_f32,
            global_rgba.index_select(0, payload.source_site_ids_i64),
        )
        for digest, payload in payload_by_digest.items()
    }
    group = adapted.groups[0]
    staged = group.stage_targets(
        track_start=0,
        track_end=adapted.pixel_count,
        sample_start=0,
        sample_end=group.observation_count,
    )
    loss = torch.zeros((), dtype=torch.float32)
    for block in iter_paper_kinetic_row_ragged_sample_blocks(
        compiled.sampler,
        staged,
        loss_normalization_id=adapted.loss_normalization_id,
        maximum_samples_per_launch=7,
    ):
        rows = block.sample_row_i32.to(dtype=torch.int64)
        selected = node_by_digest[block.native_block_generation_digest].index_select(0, rows)
        chart = torch.sum(selected * block.sample_to_node_f32[:, :, None], dim=1)
        kappa = chart[:, 0]
        prediction = _phi(kappa)[:, None] * chart[:, 1:] + torch.exp(-kappa)[:, None] * compiled.background_rgb_f32
        loss = loss + (prediction - block.target_rgb_f32).square().sum() * block.loss_scale
    loss.backward()
    assert global_rgba.grad is not None
    return loss.detach().reshape(1), global_rgba.grad.detach()


def test_cross_k_deferred_step_matches_oracle_and_runs_each_word_reverse_once() -> None:
    compiled = _compiled_case()
    adapted = _adapted(9)
    oracle_loss, oracle_gradient = _direct_oracle(compiled, adapted)

    before_forward = compiled.native_ops.forward_calls
    before_vjp = compiled.native_ops.material_vjp_calls
    k1, capture_k1, bar_k1 = _run(
        compiled,
        adapted,
        request_size=1,
        sample_launch_size=2,
    )
    k1_forward = compiled.native_ops.forward_calls - before_forward
    k1_vjp = compiled.native_ops.material_vjp_calls - before_vjp

    before_forward = compiled.native_ops.forward_calls
    before_vjp = compiled.native_ops.material_vjp_calls
    k4, capture_k4, bar_k4 = _run(
        compiled,
        adapted,
        request_size=4,
        sample_launch_size=2,
    )
    k4_forward = compiled.native_ops.forward_calls - before_forward
    k4_vjp = compiled.native_ops.material_vjp_calls - before_vjp

    torch.testing.assert_close(k1.step.loss_f32, oracle_loss, rtol=2.0e-5, atol=2.0e-6)
    torch.testing.assert_close(k1.step.grad_global_site_rgba_f32, oracle_gradient, rtol=3.0e-5, atol=3.0e-6)
    torch.testing.assert_close(k4.step.loss_f32, oracle_loss, rtol=2.0e-5, atol=2.0e-6)
    torch.testing.assert_close(k4.step.grad_global_site_rgba_f32, oracle_gradient, rtol=3.0e-5, atol=3.0e-6)
    torch.testing.assert_close(bar_k1, bar_k4, rtol=3.0e-5, atol=3.0e-6)
    torch.testing.assert_close(k1.step.loss_f32, k4.step.loss_f32, rtol=2.0e-5, atol=2.0e-6)

    active = k1.accounting.native_active_block_count
    assert active > 1
    assert k1_forward == k1_vjp == k4_forward == k4_vjp == active
    assert k1.accounting.native_node_forward_invocation_count == active
    assert k1.accounting.native_word_vjp_invocation_count == active
    assert k4.accounting.native_node_forward_invocation_count == active
    assert k4.accounting.native_word_vjp_invocation_count == active
    assert k1.accounting.temporal_request_count == 9
    assert k4.accounting.temporal_request_count == 3
    assert k1.accounting.sample_kernel_call_count > k4.accounting.sample_kernel_call_count
    assert capture_k1.optimizer_calls == capture_k4.optimizer_calls == 1
    assert compiled.native_ops.vjp_calls == 0
    assert k1.accounting.native_length_bar_tensor_bytes == 0
    assert k1.accounting.native_length_bar_callback_count == 0
    assert not k1.accounting.geometry_length_bar_delivered
    assert not k1.accounting.geometry_parameter_vjp_implemented
    assert k1.step.accounting["optimizer_update_authorization_count"] == 1
    assert {runtime.node_count for runtime in compiled.lane.runtimes} == {3, 4, 5}
    assert not k1.accounting.global_common_temporal_refinement_used
    assert k1.accounting.ordered_run_node_interactions == sum(
        runtime.node_count * runtime.word_count for runtime in compiled.lane.runtimes
    )
    repeated_source_ids = [
        source_id for runtime in compiled.lane.runtimes for source_id in runtime.payload.block.source_site_ids
    ]
    assert len(repeated_source_ids) > len(set(repeated_source_ids))
    assert torch.count_nonzero(bar_k1[2:]) == 0


def test_denser_f_keeps_runtime_and_word_work_fixed_while_sample_work_grows() -> None:
    compiled = _compiled_case()
    sparse, sparse_capture, _sparse_bar = _run(
        compiled,
        _adapted(5),
        request_size=3,
        sample_launch_size=2,
    )
    dense, dense_capture, _dense_bar = _run(
        compiled,
        _adapted(41),
        request_size=3,
        sample_launch_size=2,
    )

    assert sparse.accounting.requested_observation_count == 5
    assert dense.accounting.requested_observation_count == 41
    for field_name in (
        "native_active_block_count",
        "native_node_forward_invocation_count",
        "native_word_vjp_invocation_count",
        "ordered_run_node_interactions",
        "retained_topology_runtime_tensor_bytes",
        "peak_spatial_node_state_tensor_bytes",
    ):
        assert getattr(sparse.accounting, field_name) == getattr(dense.accounting, field_name)
    assert dense.accounting.temporal_request_count > sparse.accounting.temporal_request_count
    assert dense.accounting.sample_kernel_call_count > sparse.accounting.sample_kernel_call_count
    assert dense.accounting.sample_to_node_interactions > sparse.accounting.sample_to_node_interactions
    assert dense.accounting.streamed_sample_count == 3 * 41
    assert sparse.accounting.streamed_sample_count == 3 * 5
    assert dense.accounting.peak_staged_target_tensor_bytes <= 3 * 3 * 3 * 4
    assert sparse.accounting.peak_staged_target_tensor_bytes <= 3 * 3 * 3 * 4
    assert dense.accounting.peak_sample_launch_tensor_bytes == sparse.accounting.peak_sample_launch_tensor_bytes
    assert dense.accounting.persistent_frame_tensor_bytes == 0
    assert dense.accounting.persistent_target_tensor_bytes == 0
    assert dense.accounting.persistent_prediction_tensor_bytes == 0
    assert dense.accounting.caller_global_material_bar_count == 1
    assert dense.accounting.global_denominator_preserved
    assert dense.accounting.block_major_temporal_streaming
    assert not dense.accounting.node_forward_depends_on_requested_frames
    assert not dense.accounting.native_word_vjp_depends_on_requested_frames
    assert sparse_capture.optimizer_calls == dense_capture.optimizer_calls == 1


def test_block_major_schedule_peaks_at_largest_spatial_bundle_not_sum_of_bundles() -> None:
    adapted = _adapted(13)
    single = _compiled_case()
    split = _compiled_case(((0, 1), (1, 3)))
    single_result, _single_capture, single_bar = _run(
        single,
        adapted,
        request_size=2,
        sample_launch_size=2,
    )
    split_result, split_capture, split_bar = _run(
        split,
        adapted,
        request_size=2,
        sample_launch_size=2,
    )

    torch.testing.assert_close(split_result.step.loss_f32, single_result.step.loss_f32)
    torch.testing.assert_close(split_bar, single_bar, rtol=4.0e-5, atol=4.0e-6)
    assert split_capture.optimizer_calls == 1
    assert split_result.accounting.spatial_block_count == 2
    assert split_result.accounting.native_node_forward_invocation_count == len(split.payloads)
    assert split_result.accounting.native_word_vjp_invocation_count == len(split.payloads)
    assert split.native_ops.vjp_calls == 0

    lane_bounds = []
    for lane in split.lanes:
        base = lane.bundle.union_site_count * 4 * 4
        compact_bar_peak = 0
        for runtime in lane.runtimes:
            compact_bytes = runtime.compact_site_count * 4 * 4
            node_bytes = runtime.row_count * runtime.node_count * 4 * 4
            base += compact_bytes + 2 * node_bytes
            compact_bar_peak = max(compact_bar_peak, compact_bytes)
        lane_bounds.append(base + compact_bar_peak)
    assert split_result.accounting.peak_spatial_node_state_tensor_bytes <= max(lane_bounds)
    assert split_result.accounting.peak_spatial_node_state_tensor_bytes < sum(lane_bounds)
    assert (
        split_result.accounting.peak_spatial_node_state_tensor_bytes
        < single_result.accounting.peak_spatial_node_state_tensor_bytes
    )


def test_stream_coverage_and_lane_provenance_fail_closed_before_optimizer(monkeypatch) -> None:
    compiled = _compiled_case()
    adapted = _adapted(7)
    capture = _RunCapture()
    original = paper_step.iter_paper_kinetic_row_ragged_request_blocks

    def drop_last(*args, **kwargs):
        blocks = tuple(original(*args, **kwargs))
        yield from blocks[:-1]

    monkeypatch.setattr(
        paper_step,
        "iter_paper_kinetic_row_ragged_request_blocks",
        drop_last,
    )
    with pytest.raises(ValueError, match="did not exactly match request work"):
        paper_step.run_paper_kinetic_cpu_fake_native_material_step(
            adapted,
            programs=(compiled.view_program,),
            lanes=(compiled.lane,),
            global_site_rgba_f32=compiled.global_site_rgba_f32,
            global_grad_site_rgba_f32=torch.zeros_like(compiled.global_site_rgba_f32),
            background_rgb_f32=compiled.background_rgb_f32,
            maximum_observations_per_request=2,
            maximum_samples_per_launch=2,
            optimizer_update=capture.optimizer_step,
        )
    assert capture.optimizer_calls == 0
    assert compiled.native_ops.vjp_calls == 0

    stale = replace(
        compiled.lane,
        runtime_block_generation_digests=tuple(reversed(compiled.lane.runtime_block_generation_digests)),
    )
    with pytest.raises(ValueError, match="identity/provenance changed"):
        stale.assert_current()
    with pytest.raises(ValueError, match="every and only"):
        paper_step.prepare_paper_kinetic_cpu_fake_native_spatial_lane(
            compiled.lane.spatial_block,
            compiled.lane.bundle,
            runtimes=compiled.lane.runtimes[:-1],
        )
