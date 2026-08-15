from __future__ import annotations

import inspect
from dataclasses import replace
from fractions import Fraction

import pytest
import torch
from kinetic_multichart_transfer_program import compile_kinetic_multichart_p0_program
from kinetic_native_equal_rank_lowering import (
    iter_materialize_kinetic_native_equal_rank_blocks,
    kinetic_native_equal_rank_chart_sources_for_track,
    lower_kinetic_native_equal_rank_buckets,
)
from kinetic_native_equal_rank_runtime_adapter import (
    FORWARD_OP_NAME,
    MATERIAL_VJP_OP_NAME,
    RUNTIME_STATUS,
    VJP_OP_NAME,
    KineticNativeEqualRankMaterialVJPResult,
    KineticNativeEqualRankRuntimeBlock,
    KineticNativeEqualRankVJPResult,
    KineticNativeEqualRankWorld,
    execute_kinetic_native_equal_rank_material_node_vjp,
    execute_kinetic_native_equal_rank_node_vjp,
    materialize_kinetic_native_equal_rank_runtime_block,
    prepare_kinetic_native_equal_rank_runtime_block,
    prepare_kinetic_native_equal_rank_runtime_construction_lifetime,
    refresh_kinetic_native_equal_rank_world,
)
from kinetic_owner_chart_compiler import compile_exact_kinetic_owner_charts
from kinetic_power_word_compiler import AffineKineticPowerSites

DTYPE = torch.float64


def _node_charts(
    word_offsets_i32: torch.Tensor,
    word_owner_i32: torch.Tensor,
    node_physical_length_f32: torch.Tensor,
    site_rgba_f32: torch.Tensor,
) -> torch.Tensor:
    tracks = []
    for track_index in range(word_offsets_i32.numel() - 1):
        start = int(word_offsets_i32[track_index])
        end = int(word_offsets_i32[track_index + 1])
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
            denominator = -torch.expm1(-total_kappa)
            kappa2 = total_kappa.square()
            series = (
                1.0
                + 0.5 * total_kappa
                + kappa2 / 12.0
                - kappa2.square() / 720.0
                + kappa2.pow(3) / 30240.0
            )
            small = total_kappa.abs() < 1.0e-4
            inverse_phi = torch.where(
                small,
                series,
                total_kappa / torch.where(small, torch.ones_like(denominator), denominator),
            )
            nodes.append(torch.cat((total_kappa.reshape(1), inverse_phi * total_moment)))
        tracks.append(torch.stack(nodes))
    return torch.stack(tracks).contiguous()


def _reference_vjp(
    payload,
    compact_rgba: torch.Tensor,
    grad_node_chart: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.enable_grad():
        rgba = compact_rgba.detach().clone().requires_grad_(True)
        lengths = payload.node_physical_length_f32.detach().clone().requires_grad_(True)
        charts = _node_charts(
            payload.word_offsets_i32,
            payload.word_owner_i32,
            lengths,
            rgba,
        )
        objective = torch.sum(charts * grad_node_chart)
        return torch.autograd.grad(objective, (rgba, lengths))


class _FakeNativeOps:
    """CPU double with the exact low-level precompiled-length ABI."""

    def __init__(self) -> None:
        self.forward_calls = 0
        self.vjp_calls = 0

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
        assert tuple(grad_node_chart_f32.shape) == (track_count, node_count, 4)
        grad_rgba, grad_lengths = _reference_vjp(
            _PayloadView(word_offsets_i32, word_owner_i32, node_physical_length_f32),
            site_rgba_f32,
            grad_node_chart_f32,
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
        assert tuple(grad_node_chart_f32.shape) == (track_count, node_count, 4)
        grad_rgba, _discarded_grad_lengths = _reference_vjp(
            _PayloadView(word_offsets_i32, word_owner_i32, node_physical_length_f32),
            site_rgba_f32,
            grad_node_chart_f32,
        )
        grad_site_rgba_f32.add_(grad_rgba)
        self.vjp_calls += 1
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
        assert tuple(node_physical_length_f32.shape) == (node_count, word_owner_i32.numel())
        assert all(
            tensor.device.type == "cpu" and tensor.is_contiguous()
            for tensor in (
                word_offsets_i32,
                word_owner_i32,
                node_physical_length_f32,
                site_rgba_f32,
                config_i32,
                config_f32,
            )
        )


class _BadForwardNativeOps(_FakeNativeOps):
    def kinetic_precompiled_length_p0_lie_node_forward_launch_only(self, *args, **kwargs):
        return super().kinetic_precompiled_length_p0_lie_node_forward_launch_only(
            *args,
            **kwargs,
        ).reshape(-1, 4)


class _PayloadView:
    def __init__(self, offsets, owners, lengths) -> None:
        self.word_offsets_i32 = offsets
        self.word_owner_i32 = owners
        self.node_physical_length_f32 = lengths


def _sites() -> AffineKineticPowerSites:
    slopes = ((0, 0), (-2, 0))
    intercepts = ((0, 0, 0), (1, -1, 0))
    positions = []
    velocities = []
    weights = []
    for (slope0, slope1), (bias0, bias1, bias2) in zip(slopes, intercepts, strict=True):
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
    return AffineKineticPowerSites(
        positions0=torch.tensor(positions, dtype=DTYPE),
        velocities=torch.tensor(velocities, dtype=DTYPE),
        weight_coefficients=torch.tensor(weights, dtype=DTYPE),
    )


def _case(*, maximum_rows_per_block: int):
    sites = _sites()
    ray = torch.tensor([-1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=DTYPE)
    owners = compile_exact_kinetic_owner_charts(
        sites,
        ray,
        t_min=-2,
        t_max=2,
        near=0,
        far=1,
    )
    program = compile_kinetic_multichart_p0_program(owners, sites, ray, node_count=4)
    assert program.chart_count == 2
    sources = kinetic_native_equal_rank_chart_sources_for_track(7, program)
    lowering = lower_kinetic_native_equal_rank_buckets(
        sources,
        maximum_rows_per_block=maximum_rows_per_block,
    )
    payloads = tuple(iter_materialize_kinetic_native_equal_rank_blocks(lowering, sources))
    global_rgba = torch.tensor(
        [[0.81, 0.17, 0.11, 0.37], [0.14, 0.72, 0.91, 0.68]],
        dtype=torch.float32,
    )
    return sources, lowering, payloads, global_rgba


def _runtime_world(payload, lowering, sources, global_rgba, fake):
    runtime = prepare_kinetic_native_equal_rank_runtime_block(
        payload,
        lowering=lowering,
        sources=sources,
        native_ops=fake,
        device="cpu",
    )
    compact = global_rgba.index_select(0, runtime.source_site_ids_i64).contiguous()
    world = refresh_kinetic_native_equal_rank_world(runtime, compact)
    return runtime, world


def test_runtime_two_phase_construction_installs_sources_before_materialization() -> None:
    sources, lowering, payloads, _global_rgba = _case(maximum_rows_per_block=2)
    payload = payloads[0]
    lifetime = prepare_kinetic_native_equal_rank_runtime_construction_lifetime(
        payload,
        lowering=lowering,
        sources=sources,
        native_ops=_FakeNativeOps(),
        device="cpu",
    )

    assert lifetime.phase == "installed"
    assert lifetime.runtime is None
    assert lifetime.transferred_tensors == []
    assert lifetime.transfer_intermediates == []
    assert tuple(id(tensor) for tensor in lifetime.source_tensors) == tuple(
        id(tensor)
        for tensor in (
            payload.source_site_ids_i64,
            payload.word_offsets_i32,
            payload.word_owner_i32,
            payload.node_physical_length_f32,
            payload.config_i32,
        )
    )

    runtime = materialize_kinetic_native_equal_rank_runtime_block(lifetime)

    assert lifetime.phase == "materialized"
    assert lifetime.runtime is runtime
    assert len(lifetime.transferred_tensors) == len(lifetime.source_tensors) + 1
    assert lifetime.current_transfer_source is None
    assert lifetime.transferred_tensors[-1][0] is runtime.config_f32
    with pytest.raises(ValueError, match="already used"):
        materialize_kinetic_native_equal_rank_runtime_block(lifetime)


def test_equal_rank_runtime_executes_multirow_forward_and_caller_owned_vjp() -> None:
    sources, lowering, payloads, global_rgba = _case(maximum_rows_per_block=2)
    assert len(payloads) == 1 and payloads[0].row_count == 2
    fake = _FakeNativeOps()
    runtime, world = _runtime_world(payloads[0], lowering, sources, global_rgba, fake)
    expected_nodes = _node_charts(
        runtime.word_offsets_i32,
        runtime.word_owner_i32,
        runtime.node_physical_length_f32,
        world.compact_site_rgba_f32,
    )
    torch.testing.assert_close(world.node_chart_f32, expected_nodes)
    assert fake.forward_calls == 1
    assert runtime.runtime_status == world.runtime_status == RUNTIME_STATUS
    assert world.source_site_ids_i64 is runtime.source_site_ids_i64

    grad_node = torch.linspace(
        -0.53,
        0.71,
        runtime.row_count * runtime.node_count * 4,
        dtype=torch.float32,
    ).reshape(runtime.row_count, runtime.node_count, 4)
    expected_compact, expected_lengths = _reference_vjp(
        runtime,
        world.compact_site_rgba_f32,
        grad_node,
    )
    compact_bar = torch.full_like(world.compact_site_rgba_f32, 99.0)
    global_bar = torch.full((lowering.global_site_count, 4), 0.125, dtype=torch.float32)
    initial_global = global_bar.clone()
    result = execute_kinetic_native_equal_rank_node_vjp(
        world,
        grad_node,
        compact_grad_site_rgba_f32=compact_bar,
        global_grad_site_rgba_f32=global_bar,
    )

    assert isinstance(result, KineticNativeEqualRankVJPResult)
    assert result.grad_compact_site_rgba_f32 is compact_bar
    assert result.grad_global_site_rgba_f32 is global_bar
    torch.testing.assert_close(compact_bar, expected_compact)
    torch.testing.assert_close(result.grad_node_physical_length_f32, expected_lengths)
    expected_global = initial_global.clone()
    expected_global.index_add_(0, runtime.source_site_ids_i64, expected_compact)
    torch.testing.assert_close(global_bar, expected_global)
    assert fake.vjp_calls == 1
    assert result.accounting["adapter_allocated_compact_material_bar_bytes"] == 0
    assert result.accounting["adapter_allocated_global_material_bar_bytes"] == 0
    assert result.accounting["global_scatter_performed"]
    assert result.accounting["requested_frame_count_used"] == 0
    assert not result.geometry_parameter_vjp_implemented


def test_compact_only_vjp_allocates_no_global_bar_and_cross_block_ids_sum() -> None:
    sources, lowering, payloads, global_rgba = _case(maximum_rows_per_block=1)
    assert len(payloads) == 2

    fake = _FakeNativeOps()
    first_runtime, first_world = _runtime_world(
        payloads[0],
        lowering,
        sources,
        global_rgba,
        fake,
    )
    compact_only = torch.empty_like(first_world.compact_site_rgba_f32)
    compact_only_result = execute_kinetic_native_equal_rank_node_vjp(
        first_world,
        torch.ones_like(first_world.node_chart_f32),
        compact_grad_site_rgba_f32=compact_only,
    )
    assert compact_only_result.grad_global_site_rgba_f32 is None
    assert compact_only_result.accounting["global_material_bar_tensor_bytes"] == 0
    assert compact_only_result.accounting["adapter_allocated_global_material_bar_bytes"] == 0
    assert not compact_only_result.accounting["global_scatter_performed"]

    shared_global = torch.zeros((lowering.global_site_count, 4), dtype=torch.float32)
    expected_global = torch.zeros_like(shared_global)
    repeated_ids = []
    for scale, payload in enumerate(payloads, start=1):
        runtime, world = _runtime_world(payload, lowering, sources, global_rgba, fake)
        grad_node = torch.full_like(world.node_chart_f32, float(scale) / 3.0)
        expected_compact, _expected_lengths = _reference_vjp(
            runtime,
            world.compact_site_rgba_f32,
            grad_node,
        )
        compact_bar = torch.empty_like(world.compact_site_rgba_f32)
        execute_kinetic_native_equal_rank_node_vjp(
            world,
            grad_node,
            compact_grad_site_rgba_f32=compact_bar,
            global_grad_site_rgba_f32=shared_global,
        )
        expected_global.index_add_(0, runtime.source_site_ids_i64, expected_compact)
        repeated_ids.extend(runtime.source_site_ids_i64.tolist())
    assert len(repeated_ids) > len(set(repeated_ids))
    torch.testing.assert_close(shared_global, expected_global)


def test_material_only_vjp_returns_no_node_length_bar() -> None:
    sources, lowering, payloads, global_rgba = _case(maximum_rows_per_block=2)
    fake = _FakeNativeOps()
    runtime, world = _runtime_world(payloads[0], lowering, sources, global_rgba, fake)
    grad_node = torch.linspace(
        -0.41,
        0.63,
        runtime.row_count * runtime.node_count * 4,
        dtype=torch.float32,
    ).reshape(runtime.row_count, runtime.node_count, 4)
    expected_compact, _unused_lengths = _reference_vjp(
        runtime,
        world.compact_site_rgba_f32,
        grad_node,
    )
    compact_bar = torch.full_like(world.compact_site_rgba_f32, 99.0)
    result = execute_kinetic_native_equal_rank_material_node_vjp(
        world,
        grad_node,
        compact_grad_site_rgba_f32=compact_bar,
    )

    assert isinstance(result, KineticNativeEqualRankMaterialVJPResult)
    assert result.grad_compact_site_rgba_f32 is compact_bar
    assert result.grad_global_site_rgba_f32 is None
    torch.testing.assert_close(compact_bar, expected_compact)
    assert result.accounting["node_physical_length_bar_tensor_bytes"] == 0
    assert result.accounting["native_vjp_output_length_bar_bytes"] == 0
    assert not result.geometry_length_bar_returned
    assert fake.vjp_calls == 1


def test_logical_memory_accounting_is_exact_and_frame_density_invariant() -> None:
    sources, lowering, payloads, global_rgba = _case(maximum_rows_per_block=2)
    runtime, world = _runtime_world(
        payloads[0],
        lowering,
        sources,
        global_rgba,
        _FakeNativeOps(),
    )
    small = runtime.memory_accounting(3)
    huge = runtime.memory_accounting(30_000_000)
    for name in small.__dataclass_fields__:
        if name != "requested_frame_count":
            assert getattr(small, name) == getattr(huge, name)
    launch_bytes = sum(tensor.numel() * tensor.element_size() for tensor in runtime._launch_tensors())
    assert small.runtime_launch_tensor_bytes == launch_bytes
    assert small.runtime_config_f32_tensor_bytes == 4
    assert small.runtime_owned_persistent_tensor_bytes == 4
    assert small.runtime_launch_aliased_payload_tensor_bytes == launch_bytes - 4
    assert small.unique_retained_tensor_bytes == runtime.payload.retained_tensor_bytes + 4
    assert small.persistent_frame_tensor_bytes == 0
    assert small.persistent_sample_tensor_bytes == 0
    assert small.persistent_target_tensor_bytes == 0
    assert small.persistent_prediction_tensor_bytes == 0
    assert small.dense_row_by_global_time_tensor_bytes == 0
    assert not small.allocator_storage_bytes_measured
    assert not small.allocator_peak_measured
    assert not small.python_object_bytes_measured
    assert world.memory_accounting["adapter_allocated_compact_material_tensor_bytes"] == 0
    assert world.memory_accounting["node_chart_tensor_bytes"] == world.node_chart_f32.numel() * 4


def test_warm_gates_detect_mutation_and_bad_native_outputs_without_content_reads() -> None:
    sources, lowering, payloads, global_rgba = _case(maximum_rows_per_block=2)
    runtime, world = _runtime_world(
        payloads[0],
        lowering,
        sources,
        global_rgba,
        _FakeNativeOps(),
    )
    assert isinstance(runtime, KineticNativeEqualRankRuntimeBlock)
    assert isinstance(world, KineticNativeEqualRankWorld)

    runtime.config_f32.add_(0.25)
    with pytest.raises(ValueError, match="launch tensor identity/layout/version changed"):
        runtime.assert_warm_layout()

    fresh_runtime, fresh_world = _runtime_world(
        payloads[0],
        lowering,
        sources,
        global_rgba,
        _FakeNativeOps(),
    )
    fresh_world.compact_site_rgba_f32[0, 0].add_(0.125)
    with pytest.raises(ValueError, match="world tensor identity/layout/version changed"):
        fresh_world.assert_warm_layout()

    other_native = replace(fresh_runtime, native_ops=_FakeNativeOps())
    with pytest.raises(ValueError, match="different native ops"):
        other_native.assert_warm_layout()

    bad_runtime = prepare_kinetic_native_equal_rank_runtime_block(
        payloads[0],
        lowering=lowering,
        sources=sources,
        native_ops=_BadForwardNativeOps(),
        device="cpu",
    )
    compact = global_rgba.index_select(0, bad_runtime.source_site_ids_i64).contiguous()
    with pytest.raises(ValueError, match="native node_chart_f32"):
        refresh_kinetic_native_equal_rank_world(bad_runtime, compact)


def test_warm_validation_and_execution_source_excludes_host_content_checks() -> None:
    warm_functions = (
        KineticNativeEqualRankRuntimeBlock.assert_warm_layout,
        KineticNativeEqualRankWorld.assert_warm_layout,
        KineticNativeEqualRankVJPResult.assert_warm_layout,
        KineticNativeEqualRankMaterialVJPResult.assert_warm_layout,
        refresh_kinetic_native_equal_rank_world,
        execute_kinetic_native_equal_rank_material_node_vjp,
        execute_kinetic_native_equal_rank_node_vjp,
    )
    forbidden = (
        ".cpu(",
        ".item(",
        ".tolist(",
        "_tensor_digest(",
        "torch.as_tensor(",
        "torch.tensor(",
        "torch.zeros(",
        "torch.empty(",
        ".clone(",
        ".to(",
    )
    for function in warm_functions:
        source = inspect.getsource(function)
        for fragment in forbidden:
            assert fragment not in source


def test_adapter_requires_exact_native_abi_and_cold_epsilon_gate() -> None:
    sources, lowering, payloads, _global_rgba = _case(maximum_rows_per_block=2)

    class _Incomplete:
        pass

    with pytest.raises(TypeError, match=FORWARD_OP_NAME):
        prepare_kinetic_native_equal_rank_runtime_block(
            payloads[0],
            lowering=lowering,
            sources=sources,
            native_ops=_Incomplete(),
            device="cpu",
        )
    incomplete = _Incomplete()
    setattr(incomplete, FORWARD_OP_NAME, lambda *args, **kwargs: None)
    with pytest.raises(TypeError, match=VJP_OP_NAME):
        prepare_kinetic_native_equal_rank_runtime_block(
            payloads[0],
            lowering=lowering,
            sources=sources,
            native_ops=incomplete,
            device="cpu",
        )
    setattr(incomplete, VJP_OP_NAME, lambda *args, **kwargs: None)
    with pytest.raises(TypeError, match=MATERIAL_VJP_OP_NAME):
        prepare_kinetic_native_equal_rank_runtime_block(
            payloads[0],
            lowering=lowering,
            sources=sources,
            native_ops=incomplete,
            device="cpu",
        )
    minimum_length = float(payloads[0].node_physical_length_f32.min())
    with pytest.raises(ValueError, match="strictly above epsilon"):
        prepare_kinetic_native_equal_rank_runtime_block(
            payloads[0],
            lowering=lowering,
            sources=sources,
            native_ops=_FakeNativeOps(),
            device="cpu",
            physical_length_epsilon=minimum_length,
        )
    with pytest.raises(TypeError, match="compiled kinetic ABI attestation"):
        prepare_kinetic_native_equal_rank_runtime_block(
            payloads[0],
            lowering=lowering,
            sources=sources,
            native_ops=_FakeNativeOps(),
            device="mps",
        )
