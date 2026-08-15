from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
from pathlib import Path

import worldfoam_training_memory_spatial_native_driver as driver
from test_kinetic_lazy_native_material_step import _FakeNativeOps
from test_paper_kinetic_lazy_full_geometry_step import _FakeUnionV2NativeOps
from worldfoam_training_memory_ablation_adapter import (
    run_worldfoam_training_memory_ablation_adapter,
)


HERE = Path(__file__).resolve().parent


class _TrainingUnionV2FakeOps(_FakeUnionV2NativeOps):
    """Deterministic CPU lifecycle double with nonzero geometry updates."""

    def kinetic_fused_union_full_vjp_accumulate_launch_only_v2(
        self,
        *args,
        launch_phase,
        **kwargs,
    ):
        result = super().kinetic_fused_union_full_vjp_accumulate_launch_only_v2(
            *args,
            launch_phase=launch_phase,
            **kwargs,
        )
        if launch_phase == "accumulate":
            result.grad_union_positions0_f32.add_(1.0e-3)
            result.grad_union_velocities_f32.add_(2.0e-3)
            result.grad_union_weight_coefficients_f32.add_(3.0e-3)
        return result


def _config():
    return json.loads(
        (HERE / "worldfoam_training_memory_ablation_v1.json").read_text(
            encoding="utf-8"
        )
    )


def _tiny_config():
    config = copy.deepcopy(_config())
    config["procedural_world"].update(rows=2, columns=2, site_count=4)
    config["temporal_grid"]["dataset_frame_count"] = 2
    config["optimizer"]["lifecycle_frame_count"] = 2
    config["track_manifest"].update(
        grid_rows=1,
        grid_columns=2,
        track_count=2,
    )
    tracks = tuple(
        {
            "track_id": index,
            "row": 6,
            "column": 8 + 16 * index,
            "pixel_id": 6 * 512 + (8 + 16 * index),
        }
        for index in range(2)
    )
    encoded_tracks = json.dumps(
        tracks,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    config["track_manifest"]["ordered_manifest_sha256"] = hashlib.sha256(
        encoded_tracks
    ).hexdigest()
    config["compiler"]["maximum_sites_per_track_compile"] = 4
    config["spatial_streaming"].update(
        maximum_tracks_per_request=1,
        maximum_observations_per_chunk=2,
        maximum_samples_per_launch=2,
    )
    return config


def test_fixed_world_tracks_and_time_subsets_are_exact_and_repeat_invariant() -> None:
    config = _config()
    tracks = driver.build_fixed_track_manifest(config)
    positions, velocities, weights, rgba = driver.build_procedural_world_rows(config)
    assert len(tracks) == 512
    assert len(positions) == len(velocities) == len(weights) == len(rgba) == 1024
    assert driver.endpoint_including_frame_indices(
        dataset_frame_count=300, requested_frame_count=8
    ) == (0, 43, 85, 128, 171, 214, 256, 299)
    for frame_count in (8, 64, 300):
        selected = driver.endpoint_including_frame_indices(
            dataset_frame_count=300, requested_frame_count=frame_count
        )
        assert len(selected) == frame_count
        assert selected[0] == 0 and selected[-1] == 299
        receipt = driver.build_training_structure_receipt(
            config, requested_frame_count=frame_count
        )
        assert receipt["expected_observation_count"] == 512 * frame_count
        assert receipt["loss_element_count"] == 512 * frame_count * 3
        assert receipt["dataset_is_procedural_synthetic"] is True
        assert receipt["full_frame_target_materialization_used"] is False


def test_target_stream_is_bounded_and_never_materializes_a_video() -> None:
    config = _config()
    for frame_count, expected_chunks in ((8, 1), (64, 8), (300, 38)):
        chunks = tuple(
            driver.iter_direct_selected_pixel_target_chunks(
                config, requested_frame_count=frame_count
            )
        )
        assert len(chunks) == expected_chunks
        assert sum(chunk["observation_count"] for chunk in chunks) == 512 * frame_count
        assert max(chunk["observation_count"] for chunk in chunks) <= 4096
        assert max(chunk["logical_target_tensor_bytes"] for chunk in chunks) <= 49152
        assert all(chunk["full_frame_materialized"] is False for chunk in chunks)


def test_sequential_same_representation_control_is_never_preflight_censored() -> None:
    config = _config()
    for frame_count in (8, 64, 300):
        preflight = driver.sequential_control_launch_policy(
            config, requested_frame_count=frame_count
        )
        assert preflight["decision"] == "launch"
        assert preflight["censor_reason"] is None
        assert preflight["model"]["censorship_permitted"] is False
        assert preflight["model"]["same_representation_and_native_kernels_required"] is True
        assert preflight["model"]["sequential_frame_release_required"] is True
        assert "retained_float32_scalars_per_node_site_observation" not in preflight["model"]


def test_capability_manifest_is_a_request_and_promotion_remains_fail_closed() -> None:
    capabilities = driver.WORLDFOAM_TRAINING_MEMORY_DRIVER_CAPABILITIES
    assert capabilities["production_adapter_status"] == "source_complete"
    assert capabilities["sequential_control_adapter_status"] == "source_complete"
    assert capabilities["compiled_framewise_control_provenance"] == (
        "paper-kinetic-compiled-framewise-full-geometry-control-v1"
    )
    assert "capabilities" not in capabilities
    assert capabilities["required_runtime_capabilities"] == sorted(
        capabilities["required_runtime_capabilities"]
    )
    assert capabilities["production_core_callable"] == (
        "run_paper_kinetic_lazy_native_full_geometry_step"
    )


def test_compiled_framewise_control_routes_raw_receipts_through_adapter() -> None:
    config = _tiny_config()
    inputs = driver.build_training_inputs(config, requested_frame_count=2)
    result = run_worldfoam_training_memory_ablation_adapter(
        {
            "backend": "cpu",
            "allow_cpu_fake_native": True,
            "worker_kind": "control",
            "mode": config["ablation"]["control_mode"],
            "frame_count": 2,
            "repeat_index": 0,
            "config": config,
            "inputs": inputs,
            "native_ops": _FakeNativeOps(),
            "preflight": driver.sequential_control_launch_policy(
                config,
                requested_frame_count=2,
            ),
        }
    )
    assert set(result) == {"native_ops_used", "row"}
    row = result["row"]
    execution = row["execution"]
    precompile = row["work"]["precompile_receipt"]
    accounting = row["work"]["accounting"]
    update = execution["update_receipt"]
    assert execution["control_result_receipt"]["provenance"] == (
        "paper-kinetic-compiled-framewise-full-geometry-control-v1"
    )
    assert precompile["compile_pass_count"] == 1
    assert precompile["request_count"] == 2
    assert precompile["track_count"] == 2
    assert accounting["per_frame_replay_count"] == 2
    assert accounting["compiled_artifact_warm_hit_count"] == 4
    assert accounting["per_frame_continuous_recompile_count"] == 0
    assert accounting["maximum_simultaneously_live_frame_count"] == 1
    assert accounting["direct_selected_pixel_target_stream"] is True
    assert accounting["persistent_frame_tensor_bytes"] == 0
    assert accounting["persistent_sample_tensor_bytes"] == 0
    assert accounting["persistent_target_tensor_bytes"] == 0
    assert accounting["persistent_prediction_tensor_bytes"] == 0
    assert update["cpu_optimizer_mutation_count"] == 1
    assert update["geometry_mutation_count"] == 1
    assert update["terminal_control_generation"] is True
    assert any(
        float(update[key]) > 0.0
        for key in (
            "raw_color_parameter_delta_l2_norm",
            "raw_density_parameter_delta_l2_norm",
            "positions0_parameter_delta_l2_norm",
            "velocities_parameter_delta_l2_norm",
            "weight_coefficients_parameter_delta_l2_norm",
        )
    )
    assert row["memory"]["logical_accounting"] == {
        key: accounting[key] for key in row["memory"]["logical_accounting"]
    }
    assert row["quality"]["loss"] == update["loss"]
    assert row["preflight"]["decision"] == "launch"
    assert execution["adapter_measurements"][
        "continuous_precompile_wall_time_seconds"
    ] >= 0.0
    assert execution["adapter_measurements"][
        "control_transaction_wall_time_seconds"
    ] >= accounting["step_wall_time_seconds"]


def test_fused_f8_scaling_row_is_one_step_and_auxiliary_worker_owns_lifecycle(
    tmp_path: Path,
) -> None:
    config = _tiny_config()
    common = {
        "backend": "cpu",
        "allow_cpu_fake_native": True,
        "mode": config["ablation"]["fused_mode"],
        "frame_count": 2,
        "repeat_index": 0,
        "config": config,
        "native_ops": _TrainingUnionV2FakeOps(),
        "worker_output_dir": str(tmp_path),
    }
    primary = run_worldfoam_training_memory_ablation_adapter(
        {
            **common,
            "worker_kind": "primary",
            "inputs": driver.build_training_inputs(
                config, requested_frame_count=2
            ),
        }
    )
    assert set(primary) == {"native_ops_used", "row", "parity_payload"}
    execution = primary["row"]["execution"]
    assert execution["cpu_optimizer_mutation_count"] == 1
    assert execution["cold_cpu_compile_measurement_count"] == 1
    assert execution["worker_measurement_scope"] == (
        "single_optimizer_step_scaling_row_v2"
    )
    assert (
        execution["worker_measurement_covers_checkpoint_and_uninterrupted_step_2"]
        is False
    )

    auxiliary = run_worldfoam_training_memory_ablation_adapter(
        {
            **common,
            "worker_kind": "restart",
            "inputs": driver.build_training_inputs(
                config, requested_frame_count=2
            ),
            "fresh_inputs_factory": lambda: driver.build_training_inputs(
                config, requested_frame_count=2
            ),
        }
    )["restart_result"]
    primary_parity = primary["parity_payload"]
    primary_gradient_digest = hashlib.sha256(
        json.dumps(
            {
                "loss": primary_parity["loss"],
                "material_gradient": primary_parity["material_gradient"],
                "geometry_gradient": primary_parity["geometry_gradient"],
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    step_1 = auxiliary["auxiliary_step_1"]
    assert step_1["gradient_sha256"] == primary_gradient_digest
    primary_parameters_digest = hashlib.sha256(
        json.dumps(
            primary_parity["parameters_after_step"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    primary_deltas = {
        key: float(execution["gradient_update"][key])
        for key in (
            "raw_color_parameter_delta_l2_norm",
            "raw_density_parameter_delta_l2_norm",
            "positions0_parameter_delta_l2_norm",
            "velocities_parameter_delta_l2_norm",
            "weight_coefficients_parameter_delta_l2_norm",
        )
    }
    primary_update_content_digest = hashlib.sha256(
        json.dumps(
            {
                "loss_pre_update": float(primary_parity["loss"]),
                "gradient_sha256": primary_gradient_digest,
                "parameters_after_step_sha256": primary_parameters_digest,
                "parameter_delta_l2": primary_deltas,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert step_1["update_content_sha256"] == primary_update_content_digest
    for key in (
        "loss_pre_update",
        "gradient_sha256",
        "parameters_after_step_sha256",
        "state_sha256",
        "parameter_delta_l2",
        "update_content_sha256",
    ):
        assert auxiliary["uninterrupted_step_2"][key] == (
            auxiliary["restored_step_2"][key]
        )
    assert auxiliary["auxiliary_optimizer_mutation_count"] == 3
    assert auxiliary["maximum_simultaneously_retained_world_count"] == 1
    assert auxiliary["uninterrupted_world_released_before_restore"] is True
    assert auxiliary["lifecycle_executed_outside_primary_scaling_worker"] is True
    assert Path(auxiliary["checkpoint_path"]).is_file()


def test_native_full_geometry_attestation_names_exact_compiled_schemas() -> None:
    variant = (
        driver.ROOT
        / "third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0"
    )
    ops_source = (
        variant / "torch_world_foam_lane2_fused_slab/ops.py"
    ).read_text(encoding="utf-8")
    bindings_source = (variant / "csrc/bindings.cpp").read_text(encoding="utf-8")
    assert "def assert_kinetic_lazy_full_geometry_compiled_abi_registered" in ops_source
    required = {
        "kinetic_precompiled_length_p0_lie_node_vjp_accumulate_launch_only",
        "kinetic_fused_direct_full_vjp_accumulate_launch_only_v1",
        "kinetic_fused_direct_full_vjp_validate_shared_status_launch_only_v1",
        "kinetic_fused_direct_full_vjp_accumulate_shared_status_launch_only_v1",
        "kinetic_fused_direct_full_vjp_finalize_shared_status_launch_only_v1",
        "kinetic_fused_union_full_vjp_validate_shared_status_launch_only_v2",
        "kinetic_fused_union_full_vjp_accumulate_shared_status_launch_only_v2",
        "kinetic_fused_union_full_vjp_finalize_shared_status_launch_only_v2",
    }
    assignment = next(
        node
        for node in ast.parse(ops_source).body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "_KINETIC_LAZY_FULL_GEOMETRY_COMPILED_SCHEMAS"
            for target in node.targets
        )
    )
    attested = dict(ast.literal_eval(assignment.value))
    compiled = {
        schema.split("(", 1)[0]: schema
        for schema in re.findall(
            r'm\.def\(\s*"([^"]+)"', bindings_source, flags=re.DOTALL
        )
    }
    assert set(attested) == required
    for name, schema in attested.items():
        assert schema == compiled[name]
