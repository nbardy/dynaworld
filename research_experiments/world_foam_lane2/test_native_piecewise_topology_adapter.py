from __future__ import annotations

import dataclasses
import hashlib
from contextlib import contextmanager
from fractions import Fraction
from types import SimpleNamespace

import pytest
import torch
from camera import CameraSpec
from compact_lie_schedule import compact_lie_world_schedule_from_atlas
from compiled_lie_world_adjoint import (
    AdaptiveCompiledLieWorldAtlas,
    AdaptiveLieWorldCompilePolicy,
    compile_lie_world_atlas,
)
from compiled_transfer_adjoint import make_stable_cell_word
from native_piecewise_topology_adapter import (
    NativePiecewiseTopologyChartPayload,
    describe_native_piecewise_topology_chart,
    execute_native_piecewise_topology_track_block,
    make_native_algebraic_topology_event_guard,
    make_native_piecewise_topology_program,
)
from power_topology_event_predicates import (
    CertifiedEventRoot,
    RationalPolynomial,
    TopologyEventIsolation,
    TopologyEventPredicate,
    isolate_topology_event_roots,
)
from powerfoam_track_staging import PowerFoamTrackStagingPlan
from powerfoam_training_data import PowerFoamRayProvider, PowerFoamTargetProvider
from prepared_track_block import prepare_worldfoam_track_block
from staged_compiled_lie_adjoint import (
    allocate_compact_spatial_gradient_buffers,
    begin_compact_spatial_step_v2,
    finalize_compact_spatial_step,
    prepare_compact_staged_lie_world_snapshot_v2,
)


class _FakeBinding:
    binding_mode = "strict_frozen_evaluation"
    paper_evidence_eligible = True

    def __init__(self, prepared, *, chart_id: str, marker: float) -> None:
        self._prepared = prepared
        self.marker = marker
        self.canonical_digest = hashlib.sha256(f"binding:{chart_id}".encode()).hexdigest()
        self.charts = tuple(
            SimpleNamespace(chart_digest=f"{chart_id}:native:{index}")
            for index in range(len(prepared.world_snapshot.atlas.charts))
        )

    def assert_current(self) -> None:
        self._prepared.assert_current()


class _FakeNativeLifecycle:
    """CPU double for the token boundaries used by the composition layer."""

    def __init__(self) -> None:
        self.world_grad_calls = []
        self.sample_targets = []
        self.active_chart = None

    def prepare_fixed_word_p0_topology_token(self, *tensors, **kwargs):
        return SimpleNamespace(
            tensors=tensors,
            track_count=kwargs["track_count"],
            site_count=kwargs["site_count"],
            certificate_binding=kwargs["certificate_binding"],
        )

    def refresh_fixed_word_p0_world_token(
        self,
        topology,
        sites_f32,
        site_rgba_f32,
        track_ray_coeff_f32,
        replay_config,
        **_kwargs,
    ):
        return SimpleNamespace(
            topology=topology,
            sites_f32=sites_f32,
            site_rgba_f32=site_rgba_f32,
            track_ray_coeff_f32=track_ray_coeff_f32,
            replay_config=replay_config,
        )

    def fixed_word_p0_lie_world_grad_init_launch_only(self, world, **kwargs):
        self.world_grad_calls.append(kwargs)
        return SimpleNamespace(
            world=world,
            grad_site_rgba_f32=torch.zeros_like(world.site_rgba_f32),
            boundary_finalized=False,
        )

    def prepare_fixed_word_p0_chart_token(self, world, compiler_node_t_f32, *, chart_index):
        assert self.active_chart is None
        self.active_chart = (world.topology.certificate_binding.canonical_digest, chart_index)
        expected = world.topology.certificate_binding.charts[chart_index]
        return SimpleNamespace(
            world=world,
            chart_index=chart_index,
            chart_generation_id=expected.chart_digest,
            node_count=int(compiler_node_t_f32.numel()),
        )

    def prepare_fixed_word_p0_sample_state_token(self, chart, **kwargs):
        return SimpleNamespace(
            chart=chart,
            loss_f32=torch.zeros((), dtype=torch.float32),
            global_loss_scale=1.0 / float(kwargs["global_loss_element_count"]),
            kwargs=kwargs,
        )

    def prepare_fixed_word_p0_sample_block_token(
        self,
        sample_state,
        target_rgb_f32,
        background_rgb_f32,
        *,
        sample_t_f64,
        sample_block_id,
        global_sample_start,
        global_sample_end,
    ):
        marker = sample_state.chart.world.topology.certificate_binding.marker
        assert tuple(sample_t_f64.shape) == (global_sample_end - global_sample_start,)
        self.sample_targets.append((marker, target_rgb_f32.clone()))
        return SimpleNamespace(
            sample_state=sample_state,
            target_rgb_f32=target_rgb_f32,
            background_rgb_f32=background_rgb_f32,
            sample_block_id=sample_block_id,
            global_sample_start=global_sample_start,
            global_sample_end=global_sample_end,
        )

    def fixed_word_p0_lie_sample_accumulate_loss_only_launch_only(
        self,
        sample_block,
        sample_state,
    ):
        sample_state.loss_f32.add_(sample_block.target_rgb_f32.square().sum() * sample_state.global_loss_scale)

    def fixed_word_p0_lie_node_vjp_accumulate_launch_only(self, chart, _sample_state, world_grad):
        identity = (world_grad.world.topology.certificate_binding.canonical_digest, chart.chart_index)
        assert self.active_chart == identity
        marker = world_grad.world.topology.certificate_binding.marker
        world_grad.grad_site_rgba_f32.add_(marker)
        self.active_chart = None

    def fixed_word_p0_sparse_mobius_boundary_finalize_launch_only(self, world_grad):
        world_grad.boundary_finalized = True
        return torch.empty((0, 5), dtype=torch.float32)

    def fixed_word_p0_site_geometry_finalize_launch_only(self, world_grad):
        assert world_grad.boundary_finalized
        marker = world_grad.world.topology.certificate_binding.marker
        return torch.full(
            (world_grad.world.topology.site_count, 5),
            10.0 * marker,
            dtype=torch.float32,
        )


class _PayloadProvider:
    def __init__(self, payloads) -> None:
        self.payloads = payloads
        self.active = 0
        self.maximum_active = 0
        self.loaded = []
        self.released = []

    @contextmanager
    def __call__(self, spec):
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        self.loaded.append(spec.chart_id)
        try:
            yield self.payloads[spec.chart_id]
        finally:
            self.released.append(spec.chart_id)
            self.active -= 1


def _camera() -> CameraSpec:
    return CameraSpec(
        fx=3.0,
        fy=3.0,
        cx=1.0,
        cy=1.0,
        camera_to_world=torch.eye(4, dtype=torch.float32),
    )


def _exact_zero_isolation() -> TopologyEventIsolation:
    predicate = TopologyEventPredicate(
        kind="test_zero_run_birth_death",
        polynomial=RationalPolynomial((Fraction(0), Fraction(1))),
        site_ids=(0, 1),
        pair_differences=(),
        fixed_depth=None,
        derivation="unit-test event p(t)=t",
    )
    return TopologyEventIsolation(
        predicate=predicate,
        t_min=Fraction(-1),
        t_max=Fraction(1),
        roots=(
            CertifiedEventRoot(
                lower_bound=Fraction(0),
                upper_bound=Fraction(0),
                exact=True,
                multiplicity=1,
                sturm_root_count=1,
                polynomial_sign_at_lower=0,
                polynomial_sign_at_upper=0,
            ),
        ),
    )


def _prepare_chart(
    *,
    chart_id: str,
    t_min: float,
    t_max: float,
    owner: int,
    geometry: torch.Tensor,
    rays: torch.Tensor,
    density: torch.Tensor,
    color: torch.Tensor,
):
    words = tuple(make_stable_cell_word([owner], [-1], [-2]) for _ in range(rays.shape[0]))
    boundary = torch.empty((0, 5), dtype=torch.float64)
    compiled = compile_lie_world_atlas(
        boundary=boundary,
        ray_coefficients=rays.to(dtype=torch.float64),
        words=words,
        site_density=density.to(dtype=torch.float64),
        site_color=color.to(dtype=torch.float64),
        t_min=t_min,
        t_max=t_max,
        near=0.1,
        far=1.0,
        node_count=2,
    )
    atlas = AdaptiveCompiledLieWorldAtlas(
        charts=(compiled,),
        selections=(),
        policy=AdaptiveLieWorldCompilePolicy(node_count_schedule=(2,)),
        supplied_word_ordering_check=compiled.supplied_word_ordering_check,
    )
    schedule = compact_lie_world_schedule_from_atlas(atlas)
    topology = prepare_worldfoam_track_block(
        words,
        torch.empty((0, 2), dtype=torch.int64),
        site_count=geometry.shape[0],
        track_start=0,
        track_end=rays.shape[0],
    )
    prepared = prepare_compact_staged_lie_world_snapshot_v2(
        schedule,
        topology,
        site_geometry=geometry,
        ray_coefficients=rays,
        site_density=density,
        site_color=color,
    )
    binding = _FakeBinding(
        prepared,
        chart_id=chart_id,
        marker=1.0 if owner == 0 else 2.0,
    )
    spec = describe_native_piecewise_topology_chart(
        chart_id=chart_id,
        prepared=prepared,
        certificate_binding=binding,
        chart_provenance=f"unit-test:{chart_id}",
    )
    return schedule, spec, NativePiecewiseTopologyChartPayload(prepared, binding)


def _case(*, sample_block_size: int, execute: bool = True):
    frames = torch.arange(4 * 3 * 2 * 2, dtype=torch.float32).reshape(1, 4, 3, 2, 2)
    frames = frames / float(frames.numel())
    target_provider = PowerFoamTargetProvider.from_resident_frames(frames, device=torch.device("cpu"))
    ray_provider = PowerFoamRayProvider(
        (tuple(_camera() for _ in range(4)),),
        height=2,
        width=2,
        device=torch.device("cpu"),
    )
    plan = PowerFoamTrackStagingPlan(
        target_provider,
        ray_provider,
        torch.tensor([0, 1, 2, 3]),
        torch.tensor([3, 0, 2, 1]),
        sample_times=torch.tensor([0.75, -0.75, 0.0, -0.25]),
    )
    staged = plan.stage(require_affine_ray_program=True)
    assert staged.affine_ray_program is not None
    rays = staged.affine_ray_program.coefficients[0].to(dtype=torch.float32)
    geometry = torch.tensor(
        [[-0.2, 0.0, 0.4, -0.1, 0.0], [0.2, 0.0, 0.6, 0.1, 0.0]],
        dtype=torch.float32,
    )
    density = torch.tensor([0.4, 0.7], dtype=torch.float32)
    color = torch.tensor([[0.8, 0.1, 0.2], [0.1, 0.7, 0.3]], dtype=torch.float32)
    left_schedule, left_spec, left_payload = _prepare_chart(
        chart_id="owner-0",
        t_min=-1.0,
        t_max=0.0,
        owner=0,
        geometry=geometry,
        rays=rays,
        density=density,
        color=color,
    )
    _right_schedule, right_spec, right_payload = _prepare_chart(
        chart_id="owner-1",
        t_min=0.0,
        t_max=1.0,
        owner=1,
        geometry=geometry,
        rays=rays,
        density=density,
        color=color,
    )
    guard = make_native_algebraic_topology_event_guard(
        _exact_zero_isolation(),
        root_index=0,
        event_id="owner-swap-at-zero",
        left_chart_id="owner-0",
        right_chart_id="owner-1",
        source_track_id=0,
        geometry_ray_content_digest=left_spec.geometry_ray_content_digest,
        compiler_provenance="unit-test-exact-event-compiler-v1",
    )
    program = make_native_piecewise_topology_program(
        (left_spec, right_spec),
        (guard,),
        domain_t_min=-1,
        domain_t_max=1,
        compiler_provenance="unit-test-piecewise-native-program-v1",
    )
    gradients = allocate_compact_spatial_gradient_buffers(
        site_geometry=geometry,
        site_density=density,
        site_color=color,
    )
    ledger = begin_compact_spatial_step_v2(
        schedule=left_schedule,
        site_geometry=geometry,
        ray_coefficients=rays,
        site_density=density,
        site_color=color,
        gradients=gradients,
        global_track_count=4,
        global_frame_count=4,
        loss_normalization_id="piecewise-paper-logical-batch",
        expected_blocks=(("all-pixels", 0, 4),),
        expected_block_schedule_generations=(("all-pixels", program.generation_digest),),
    )
    payloads = {"owner-0": left_payload, "owner-1": right_payload}
    provider = _PayloadProvider(payloads)
    native = _FakeNativeLifecycle()
    if not execute:
        return plan, program, provider, native, ledger, payloads
    result = execute_native_piecewise_topology_track_block(
        ledger,
        block_id="all-pixels",
        program=program,
        payload_provider=provider,
        staging_plan=plan,
        background_rgb=(0.0, 0.0, 0.0),
        replay_config=SimpleNamespace(near=0.1, far=1.0),
        sample_block_size=sample_block_size,
        native_ops=native,
    )
    return plan, program, provider, native, result, finalize_compact_spatial_step(ledger)


def test_streams_distinct_topologies_with_one_denominator_and_one_site_ledger() -> None:
    plan, program, provider, native, result, final = _case(sample_block_size=1)
    torch.testing.assert_close(
        final.loss,
        plan.stage().targets.square().sum() / float(4 * 4 * 3),
    )
    torch.testing.assert_close(
        final.gradients.grad_site_geometry,
        torch.tensor([[10.0, 10.0, 10.0, 10.0], [20.0, 20.0, 20.0, 20.0]]),
    )
    torch.testing.assert_close(final.gradients.grad_site_weight, torch.tensor([10.0, 20.0]))
    torch.testing.assert_close(
        final.gradients.grad_site_color,
        torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
    )
    torch.testing.assert_close(final.gradients.grad_site_density, torch.tensor([1.0, 2.0]))
    assert provider.loaded == ["owner-0", "owner-1"]
    assert provider.released == provider.loaded
    assert provider.active == 0
    assert provider.maximum_active == 1
    assert result.maximum_resident_payload_count == 1
    assert result.maximum_resident_target_elements == 4 * 1 * 3
    assert result.maximum_resident_sample_time_bytes == 1 * 8
    assert result.global_loss_element_count == 4 * 4 * 3
    assert [(call["resident_sample_start"], call["resident_sample_end"]) for call in native.world_grad_calls] == [
        (0, 2),
        (2, 4),
    ]
    assert {call["global_loss_element_count"] for call in native.world_grad_calls} == {4 * 4 * 3}
    assert result.exact_binary_sample_dispatch
    assert result.continuous_real_native_boundary_equivalence_certified
    assert result.paper_evidence_eligible
    assert program.chart_index_for_binary_sample(0.0) == 1
    assert len(result.event_gradients) == 1
    event = result.event_gradients[0]
    assert event.seam_sample_assignment == "right_chart"
    assert event.frozen_topology_parameter_vjp == "right_one_sided"
    assert event.event_time_vjp == "not_implemented"
    assert event.algebraic_event_dispatch_vjp == "unresolved"
    right_targets = torch.cat([target for marker, target in native.sample_targets if marker == 2.0], dim=1)
    assert right_targets.shape[1] == 2


def test_k_partition_preserves_loss_gradients_and_bounds_target_residency() -> None:
    *_, unit, unit_final = _case(sample_block_size=1)
    *_, full, full_final = _case(sample_block_size=4)
    torch.testing.assert_close(full_final.loss, unit_final.loss)
    for actual, expected in zip(full_final.gradients.tensors, unit_final.gradients.tensors, strict=True):
        torch.testing.assert_close(actual, expected)
    assert unit.maximum_resident_target_elements == 4 * 1 * 3
    assert full.maximum_resident_target_elements == 4 * 2 * 3
    assert unit.maximum_resident_sample_time_bytes == 1 * 8
    assert full.maximum_resident_sample_time_bytes == 2 * 8
    assert unit.sample_block_count == 4
    assert full.sample_block_count == 2


def test_irrational_polynomial_guard_dispatches_exact_samples_but_is_not_continuous_native_evidence() -> None:
    predicate = TopologyEventPredicate(
        kind="triple_concurrence",
        polynomial=RationalPolynomial((Fraction(-2), Fraction(0), Fraction(1))),
        site_ids=(0, 1, 2),
        pair_differences=(),
        fixed_depth=None,
        derivation="unit-test p(t)=t^2-2",
    )
    isolation = isolate_topology_event_roots(
        predicate,
        t_min=1,
        t_max=2,
        max_interval_width=Fraction(1, 1 << 30),
    )
    positive_root = isolation.roots[0]
    guard = make_native_algebraic_topology_event_guard(
        isolation,
        root_index=0,
        event_id="sqrt-two-event",
        left_chart_id="owner-0",
        right_chart_id="owner-1",
        source_track_id=0,
        geometry_ray_content_digest=hashlib.sha256(b"temporary").hexdigest(),
        compiler_provenance="unit-test-sturm-v1",
    )
    assert not positive_root.exact
    assert guard.compare_binary_sample(Fraction(7, 5)) == -1
    assert guard.compare_binary_sample(Fraction(3, 2)) == 1

    _plan, exact_program, _provider, _native, _result, _final = _case(sample_block_size=2)
    guard = make_native_algebraic_topology_event_guard(
        isolation,
        root_index=0,
        event_id="sqrt-two-event",
        left_chart_id="owner-0",
        right_chart_id="owner-1",
        source_track_id=0,
        geometry_ray_content_digest=exact_program.charts[0].geometry_ray_content_digest,
        compiler_provenance="unit-test-sturm-v1",
    )
    irrational_program = make_native_piecewise_topology_program(
        exact_program.charts,
        (guard,),
        domain_t_min=1,
        domain_t_max=2,
        compiler_provenance="unit-test-irrational-program-v1",
    )
    assert not irrational_program.continuous_real_native_boundary_equivalence_certified


def test_tampered_guard_and_streamed_payload_fail_closed() -> None:
    _plan, program, _provider, _native, _result, _final = _case(sample_block_size=2)
    with pytest.raises(ValueError, match="stale or fabricated"):
        dataclasses.replace(program.event_guards[0], compiler_provenance="tampered").assert_current()

    plan, program, _provider, native, ledger, payloads = _case(
        sample_block_size=2,
        execute=False,
    )
    wrong = _PayloadProvider(
        {
            "owner-0": payloads["owner-1"],
            "owner-1": payloads["owner-1"],
        }
    )
    with pytest.raises(ValueError, match="chart schedule|compact CSR"):
        execute_native_piecewise_topology_track_block(
            ledger,
            block_id="all-pixels",
            program=program,
            payload_provider=wrong,
            staging_plan=plan,
            background_rgb=(0.0, 0.0, 0.0),
            replay_config=SimpleNamespace(near=0.1, far=1.0),
            sample_block_size=2,
            native_ops=native,
        )
    assert wrong.active == 0
    with pytest.raises(ValueError, match="previous native spatial operation"):
        execute_native_piecewise_topology_track_block(
            ledger,
            block_id="all-pixels",
            program=program,
            payload_provider=_provider,
            staging_plan=plan,
            background_rgb=(0.0, 0.0, 0.0),
            replay_config=SimpleNamespace(near=0.1, far=1.0),
            sample_block_size=2,
            native_ops=native,
        )
