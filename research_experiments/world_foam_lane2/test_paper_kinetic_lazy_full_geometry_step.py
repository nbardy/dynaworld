from __future__ import annotations

from types import SimpleNamespace

import torch
from paper_kinetic_active_track_program_factory import (
    PaperKineticActiveP0TrackProgramFactoryConfig,
    prepare_paper_kinetic_active_p0_track_program_factory,
)
from kinetic_lazy_native_material_step import (
    paper_kinetic_observation_manifest_digest,
    prepare_paper_kinetic_lazy_native_trainer_state,
)
from paper_kinetic_fixed_camera_full_geometry_step import (
    paper_kinetic_fixed_camera_provider_geometry_generation_id,
)
from paper_kinetic_lazy_full_geometry_step import (
    FUSED_UNION_V2,
    STAGED_SPARSE,
    PaperKineticLazyFullGeometryMemoryPolicy,
    PaperKineticLazyNativeFullGeometryStepResult,
    run_paper_kinetic_lazy_native_full_geometry_step,
)
from test_kinetic_lazy_native_material_step import (
    _FakeNativeOps,
    _node_charts,
    _background,
    _material,
    _memory_policy,
    _observations,
    _provider,
)


_DIRECT_TENSOR_NAMES = (
    "word_offsets_i32",
    "word_owner_i32",
    "source_site_ids_i64",
    "node_physical_length_f32",
    "site_rgba_f32",
    "node_chart_f32",
    "row_node_time_f32",
    "row_near_far_f32",
    "row_ray_coeff_f32",
    "compact_positions0_f32",
    "compact_velocities_f32",
    "compact_weight_coefficients_f32",
    "config_i32",
    "config_f32",
)


def _tensor_bytes(*tensors: torch.Tensor) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


class _FakeUnionV2NativeOps(_FakeNativeOps):
    """CPU oracle for coordinator parity; native union math remains unclaimed."""

    def kinetic_fused_direct_full_vjp_validation_status_init_v1(self, _rgba):
        return torch.zeros((1,), dtype=torch.int32)

    def kinetic_fused_direct_full_vjp_accumulate_launch_only_v1(self, *_args, **_kwargs):
        raise AssertionError("union-v2 coordinator must not execute fused-direct v1")

    def prepare_kinetic_fused_direct_full_vjp_v1(
        self,
        *payload,
        global_site_count,
        physical_length_epsilon=1.0e-8,
        minimum_absolute_cut_denominator=1.0e-7,
        minimum_cut_cosine=1.0e-8,
        minimum_coordinate_length=1.0e-8,
        minimum_ray_speed=1.0e-7,
        depth_closure_relative_tolerance=2.0e-5,
        active_tie_relative_tolerance=2.0e-5,
    ):
        config_i32 = torch.tensor([global_site_count], dtype=torch.int32)
        config_f32 = torch.tensor(
            (
                physical_length_epsilon,
                minimum_absolute_cut_denominator,
                minimum_ray_speed,
                depth_closure_relative_tolerance,
                active_tie_relative_tolerance,
                minimum_cut_cosine,
                minimum_coordinate_length,
            ),
            dtype=torch.float32,
        )
        tensors = (*payload, config_i32, config_f32)
        return SimpleNamespace(
            **dict(zip(_DIRECT_TENSOR_NAMES, tensors, strict=True)),
            tensor_owned_by_preparer=(False,) * 12 + (True, True),
            retained_logical_tensor_bytes=_tensor_bytes(*tensors),
            preparer_owned_logical_tensor_bytes=_tensor_bytes(
                config_i32,
                config_f32,
            ),
            persistent_frame_tensor_bytes=0,
            persistent_sample_tensor_bytes=0,
            persistent_target_tensor_bytes=0,
            persistent_prediction_tensor_bytes=0,
            weight_coefficient_count=int(payload[-1].shape[1]),
            row_count=int(payload[0].numel() - 1),
            node_count=int(payload[3].shape[0]),
            word_count=int(payload[1].numel()),
            compact_site_count=int(payload[2].numel()),
            global_site_count=int(global_site_count),
            runtime_status=(
                "raw_fixed_camera_source_only_until_native_rebuild_and_sparse_oracle_parity"
            ),
        )

    def prepare_kinetic_fused_union_full_vjp_v2(
        self,
        word_offsets_i32,
        word_owner_i32,
        source_site_ids_i64,
        compact_to_geometry_output_i64,
        geometry_output_source_site_ids_i64,
        node_physical_length_f32,
        site_rgba_f32,
        node_chart_f32,
        row_node_time_f32,
        row_near_far_f32,
        row_ray_coeff_f32,
        compact_positions0_f32,
        compact_velocities_f32,
        compact_weight_coefficients_f32,
        *,
        global_site_count,
        **thresholds,
    ):
        direct = self.prepare_kinetic_fused_direct_full_vjp_v1(
            word_offsets_i32,
            word_owner_i32,
            source_site_ids_i64,
            node_physical_length_f32,
            site_rgba_f32,
            node_chart_f32,
            row_node_time_f32,
            row_near_far_f32,
            row_ray_coeff_f32,
            compact_positions0_f32,
            compact_velocities_f32,
            compact_weight_coefficients_f32,
            global_site_count=global_site_count,
            **thresholds,
        )
        return SimpleNamespace(
            direct_v1_oracle=direct,
            compact_to_geometry_output_i64=compact_to_geometry_output_i64,
            geometry_output_source_site_ids_i64=(
                geometry_output_source_site_ids_i64
            ),
            config_i32=torch.tensor(
                (
                    int(geometry_output_source_site_ids_i64.numel()),
                    int(global_site_count),
                ),
                dtype=torch.int32,
            ),
            mapping_tensor_owned_by_preparer=(False, False),
            union_site_count=int(geometry_output_source_site_ids_i64.numel()),
            global_site_count=int(global_site_count),
        )

    def kinetic_fused_union_full_vjp_validation_status_init_v2(self, _rgba):
        return torch.zeros((1,), dtype=torch.int32)

    def kinetic_fused_union_full_vjp_accumulate_launch_only_v2(
        self,
        raw,
        node_bar,
        compact_bar,
        grad_positions,
        grad_velocities,
        grad_weights,
        *,
        validation_status_i32,
        launch_phase,
        **_kwargs,
    ):
        if launch_phase == "accumulate":
            direct = raw.direct_v1_oracle
            with torch.enable_grad():
                rgba = direct.site_rgba_f32.detach().clone().requires_grad_(True)
                nodes = _node_charts(
                    direct.word_offsets_i32,
                    direct.word_owner_i32,
                    direct.node_physical_length_f32,
                    rgba,
                )
                (grad_rgba,) = torch.autograd.grad(
                    torch.sum(nodes * node_bar.detach()),
                    (rgba,),
                )
            compact_bar.add_(grad_rgba)
            # The one-site fixed-ray fixture has an exactly zero geometry bar.
            assert not torch.count_nonzero(grad_positions)
            assert not torch.count_nonzero(grad_velocities)
            assert not torch.count_nonzero(grad_weights)
        return SimpleNamespace(
            grad_site_rgba_f32=compact_bar,
            grad_union_positions0_f32=grad_positions,
            grad_union_velocities_f32=grad_velocities,
            grad_union_weight_coefficients_f32=grad_weights,
            validation_status_i32=validation_status_i32,
            accumulation_enqueued=launch_phase == "accumulate",
            finalization_enqueued=launch_phase == "finalize",
            shared_status_reused=True,
            geometry_output_index_space="request_union",
            runtime_status=(
                "raw_union_v2_source_only_until_native_rebuild_v1_sparse_parity_and_allocator_evidence"
            ),
        )


def _run(mode: str, *, step_index: int, native_ops):
    factory = prepare_paper_kinetic_active_p0_track_program_factory(
        PaperKineticActiveP0TrackProgramFactoryConfig(
            near=0.0,
            far=2.0,
            node_count=2,
            maximum_sites_per_track_compile=8,
            maximum_charts_per_track=16,
            maximum_owner_runs_per_chart=8,
            rank_selection_provenance="full-geometry-compile-receipt-test-v1",
        )
    )
    _, _, provider = _provider(
        maximum_tracks_per_bundle=1,
        maximum_observations_per_bundle=3,
        factory=factory,
    )
    observations = _observations(((0, 0, 0), (0, 1, 0), (0, 2, 0)))
    state = prepare_paper_kinetic_lazy_native_trainer_state(provider, device="cpu")
    material = _material()
    material_bar = torch.empty_like(material)
    position_bar = torch.empty((provider.world.site_count, 3), dtype=torch.float64)
    velocity_bar = torch.empty_like(position_bar)
    weight_bar = torch.empty_like(provider.world.sites.weight_coefficients)
    captures = []
    result = run_paper_kinetic_lazy_native_full_geometry_step(
        state,
        provider,
        observations,
        step_index=step_index,
        expected_observation_count=len(observations),
        expected_observation_manifest_digest=(
            paper_kinetic_observation_manifest_digest(observations)
        ),
        loss_normalization_id="lazy-full-geometry-parity",
        material_generation_id="a" * 64,
        geometry_generation_id=(
            paper_kinetic_fixed_camera_provider_geometry_generation_id(provider)
        ),
        background_generation_id="b" * 64,
        global_site_rgba_f32=material,
        global_grad_site_rgba_f32=material_bar,
        grad_positions0_f64_cpu=position_bar,
        grad_velocities_f64_cpu=velocity_bar,
        grad_weight_coefficients_f64_cpu=weight_bar,
        background_rgb_f32=_background(),
        native_ops=native_ops,
        maximum_samples_per_launch=2,
        memory_policy=_memory_policy(provider),
        full_geometry_memory_policy=PaperKineticLazyFullGeometryMemoryPolicy(
            maximum_global_geometry_bar_logical_tensor_bytes=1_000_000,
            maximum_geometry_bridge_visible_peak_logical_tensor_bytes=10_000_000,
            maximum_fused_union_transaction_scratch_tensor_bytes=(
                0 if mode == STAGED_SPARSE else 1_000_000
            ),
        ),
        reverse_mode=mode,
        optimizer_update=captures.append,
    )
    assert type(result) is PaperKineticLazyNativeFullGeometryStepResult
    assert captures == [result]
    return result


def test_staged_sparse_and_fused_union_v2_cpu_coordinator_parity() -> None:
    staged = _run(STAGED_SPARSE, step_index=0, native_ops=_FakeNativeOps())
    fused = _run(FUSED_UNION_V2, step_index=0, native_ops=_FakeUnionV2NativeOps())

    torch.testing.assert_close(fused.loss_f32, staged.loss_f32)
    torch.testing.assert_close(
        fused.grad_global_site_rgba_f32,
        staged.grad_global_site_rgba_f32,
    )
    torch.testing.assert_close(
        fused.grad_positions0_f64_cpu,
        staged.grad_positions0_f64_cpu,
    )
    torch.testing.assert_close(
        fused.grad_velocities_f64_cpu,
        staged.grad_velocities_f64_cpu,
    )
    torch.testing.assert_close(
        fused.grad_weight_coefficients_f64_cpu,
        staged.grad_weight_coefficients_f64_cpu,
    )
    assert staged.accounting["native_full_geometry_vjp_launch_count"] == 1
    assert fused.accounting["native_fused_union_v2_transaction_count"] == 1
    compiler_keys = (
        "compile_track_count",
        "compiler_work_receipt_count",
        "compiler_work_receipt_bundle_count",
        "compiler_work_receipt_chain_link_count",
        "root_complement_witness_count",
        "candidate_source_attempt_count",
        "all_site_witness_check_count",
        "unique_pair_difference_count",
        "per_witness_candidate_bound_verified",
        "exhaustive_triple_enumeration_used",
        "requested_frame_sampling_used",
        "active_compiler_accounting_complete",
        "all_track_receipt_digests_verified",
        "compiler_work_receipt_provenance",
        "compiler_work_receipt_chain_digest",
    )
    assert {key: staged.accounting[key] for key in compiler_keys} == {
        key: fused.accounting[key] for key in compiler_keys
    }
    assert staged.accounting["compile_track_count"] == 1
    assert staged.accounting["compiler_work_receipt_count"] == 1
    assert staged.accounting["compiler_work_receipt_chain_link_count"] == 1
    assert staged.accounting["active_compiler_accounting_complete"] is True
    assert staged.accounting["all_track_receipt_digests_verified"] is True
    assert staged.accounting["per_witness_candidate_bound_verified"] is True
    assert staged.accounting["exhaustive_triple_enumeration_used"] is False
    assert staged.accounting["requested_frame_sampling_used"] is False
    assert staged.accounting["retained_compiled_program_count"] == 0
    assert staged.accounting["retained_compiler_receipt_entry_count"] == 0
    assert staged.accounting["retained_compiler_tensor_bytes"] == 0
    assert len(staged.accounting["compiler_work_receipt_chain_digest"]) == 64
    assert staged.accounting["camera_ray_slice_work_count"] == 3
    assert staged.accounting["camera_ray_slice_scalar_count"] == 18
    assert fused.accounting["camera_ray_slice_work_count"] == 3
    assert fused.accounting["camera_ray_slice_scalar_count"] == 18
    assert fused.geometry_d2h_receipts[0].device_to_host_tensor_count == 3
    assert fused.geometry_d2h_receipts[0].exact_request_union_identity_certified
