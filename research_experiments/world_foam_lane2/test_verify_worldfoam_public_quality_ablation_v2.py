from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import verify_worldfoam_public_quality_ablation_v2 as verifier
from worldfoam_g4_selected_ray_contract import (
    DEFAULT_CONFIG,
    REQUIRED_ROUTES,
    REQUIRED_SCENES,
    REQUIRED_SEEDS,
    build_matrix_workload_receipts,
    canonical_sha256,
    file_sha256,
)
from worldfoam_g4_v2_capability import required_source_capability


ROOT = Path(__file__).resolve().parents[2]


def _identity(path: str, **extra: object) -> dict[str, object]:
    return {"path": path, "bytes": 17, "sha256": "a" * 64, **extra}


def _worldfoam_heldout_receipt(workload: object) -> dict[str, object]:
    target_pixels = int(workload.heldout_target_pixels)
    frames = int(workload.heldout_frame_count)
    pixels = int(workload.image_height * workload.image_width)
    session_payload = {
        "schema_version": 1,
        "kind": "worldfoam-spatial-major-full-temporal-heldout-v1",
        "camera_count": 1,
        "frame_count": frames,
        "image_height": int(workload.image_height),
        "image_width": int(workload.image_width),
        "cross_time_track_block_size": int(
            workload.heldout_cross_time_track_block_size
        ),
        "render_call_count": int(
            workload.heldout_spatial_major_render_call_count
        ),
        "cold_track_compile_count": int(workload.heldout_cold_track_compile_count),
        "complete_camera_record_validation_count": int(
            workload.heldout_complete_camera_record_validation_count
        ),
        "admitted_site_reference_upper_bound": int(
            workload.heldout_admitted_site_reference_upper_bound
        ),
        "native_bundle_count": int(workload.heldout_native_bundle_count),
        "native_tracks_per_bundle_limit": 13,
        "expected_native_bundle_count": int(workload.heldout_native_bundle_count),
        "native_sample_count": target_pixels,
        "native_prediction_target_observation_read_count": target_pixels,
        "spatial_target_staging_call_count": int(
            workload.heldout_spatial_major_render_call_count
        ),
        "spatial_target_staging_observation_count": target_pixels,
        "spatial_target_staging_peak_logical_bytes": 1_000_000,
        "prediction_receipt_chain_sha256": "b" * 64,
        "target_receipt_chain_sha256": "c" * 64,
        "target_ray_tensor_bytes": 0,
        "full_pixel_full_temporal": True,
        "frame_major_recompile_per_time_used": False,
        "prediction_spool_dtype": "float32",
    }
    session = {
        **session_payload,
        "generation_digest": canonical_sha256(session_payload),
    }
    frame_hashes = ["d" * 64 for _ in range(frames)]
    prediction_hash = "e" * 64
    target_hash = "f" * 64
    coverage_hash = "1" * 64
    receipt_payload = {
        "schema_version": 1,
        "kind": "worldfoam-spatial-major-heldout-evaluation-v1",
        "camera_count": 1,
        "frame_count": frames,
        "image_height": int(workload.image_height),
        "image_width": int(workload.image_width),
        "target_pixel_count": target_pixels,
        "rgb_scalar_count": target_pixels * 3,
        "spatial_track_count": pixels,
        "spatial_track_block_limit": int(
            workload.heldout_cross_time_track_block_size
        ),
        "spatial_track_block_count": int(
            workload.heldout_spatial_major_render_call_count
        ),
        "maximum_observations_per_spatial_call": (
            frames * int(workload.heldout_cross_time_track_block_size)
        ),
        "write_superblock_track_limit": 1024,
        "write_superblock_count": math.ceil(pixels / 1024),
        "peak_buffered_prediction_and_target_bytes": frames * 1024 * 3 * 8,
        "prediction_spool_bytes": target_pixels * 3 * 4,
        "target_spool_bytes": target_pixels * 3,
        "total_spool_bytes": target_pixels * 15,
        "prediction_spool_dtype": "float32",
        "target_spool_dtype": "uint8",
        "spool_shape": [1, frames, pixels, 3],
        "prediction_spool_darwin_f_nocache": True,
        "target_spool_darwin_f_nocache": True,
        "spools_cleaned_before_return": True,
        "dense_device_video_used": False,
        "persistent_device_video_bytes": 0,
        "target_ray_tensor_bytes": 0,
        "metric_pixel_chunk_limit": 32_768,
        "metric_pixel_chunk_count": frames * math.ceil(pixels / 32_768),
        "lpips_evaluation_count": frames,
        "media_frame_count": frames,
        "metric_and_media_order": (
            "camera_major_then_frame_then_ascending_pixel_chunks"
        ),
        "metric_target_spool_observation_read_count": target_pixels,
        "native_prediction_target_source_observation_read_count": target_pixels,
        "target_spool_source_observation_read_count": target_pixels,
        "total_target_source_observation_read_count": 2 * target_pixels,
        "total_target_observation_traversal_count": 3 * target_pixels,
        "forward_only_prediction_native_op_used": False,
        "heldout_wall_time_target_io_matched_across_routes": False,
        "track_request_manifest_sha256": "2" * 64,
        "prediction_block_content_sha256": "3" * 64,
        "target_block_rgb8_content_sha256": "4" * 64,
        "prediction_spool_file_sha256": prediction_hash,
        "target_spool_file_sha256": target_hash,
        "target_read_receipt_manifest_sha256": "5" * 64,
        "spool_read_request_manifest_sha256": coverage_hash,
        "prediction_spool_read_content_sha256": prediction_hash,
        "target_spool_read_content_sha256": target_hash,
        "target_source_frame_sha256s": frame_hashes,
        "target_spool_frame_read_sha256s": frame_hashes,
        "target_source_to_spool_frame_hashes_equal": True,
        "metrics_sha256": "6" * 64,
        "heldout_coverage_sha256": coverage_hash,
        "session_receipt": session,
        "session_receipt_generation_digest": session["generation_digest"],
        "exact_rgb8_roundtrip_verified": True,
        "exact_full_pixel_full_temporal_coverage": True,
        "one_cold_compile_per_view_pixel_track": True,
    }
    return {
        **receipt_payload,
        "generation_digest": canonical_sha256(receipt_payload),
    }


def _source_read_receipt(route: str, workload: object) -> dict[str, object]:
    worldfoam = route.startswith("worldfoam_")
    payload = {
        "schema_version": 1,
        "kind": (
            "worldfoam-native-internal-selected-target-reads-v1"
            if worldfoam
            else "row-worker-external-selected-target-reads-v1"
        ),
        "ownership": (
            "executor_internal_single_read"
            if worldfoam
            else "row_worker_external_single_read"
        ),
        "selected_pixel_read_call_count": (
            2 * workload.selected_pixel_chunk_count
            if worldfoam
            else workload.selected_pixel_chunk_count
        ),
        "selected_pixel_read_observation_count": workload.selected_target_pixels,
        "full_frame_target_materialization_count": 0,
        "external_row_worker_target_read_call_count": (
            0 if worldfoam else workload.selected_pixel_chunk_count
        ),
        "request_schedule_sha256": workload.sample_schedule_sha256,
    }
    return {**payload, "generation_digest": canonical_sha256(payload)}


def _gaussian_heldout_receipt(workload: object) -> dict[str, object]:
    payload = {
        "schema_version": 1,
        "kind": "gaussian-frame-major-full-image-heldout-v1",
        "camera_count": 1,
        "frame_count": workload.heldout_frame_count,
        "target_pixel_count": workload.heldout_target_pixels,
        "pixel_chunk_count": workload.heldout_frame_count * 6,
        "coverage_sha256": "7" * 64,
        "full_pixel_full_temporal": True,
    }
    return {**payload, "generation_digest": canonical_sha256(payload)}


def _mps_limit_receipt() -> dict[str, object]:
    payload = {
        "schema_version": 1,
        "kind": "worldfoam-g4-v2-row-mps-working-set-limit-v1",
        "requested_working_set_limit_bytes": 2 * 1024**3,
        "recommended_max_memory_bytes": 8 * 1024**3,
        "effective_fraction": 0.25,
        "effective_working_set_limit_bytes": 2 * 1024**3,
        "installed_before_dataset_executor_native_or_tensor_allocation": True,
    }
    return {**payload, "generation_digest": canonical_sha256(payload)}


def _watchdog(
    *,
    row: dict[str, object],
    row_path: str,
    config_path: Path,
    source_capability: dict[str, object],
) -> dict[str, object]:
    argv = [
        str(ROOT / ".venv" / "bin" / "python"),
        str((ROOT / "src/train/train_worldfoam_native4d_public_quality_row_v2.py").resolve()),
        "--execute",
        "--g4-v2-config",
        str(config_path.resolve()),
        "--protocol",
        str((ROOT / str(row["protocol_path"])).resolve()),
        "--scene",
        str(row["scene"]),
        "--seed",
        str(row["seed"]),
        "--route",
        str(row["route"]),
        "--output",
        str((ROOT / row_path).resolve()),
        "--maximum-mps-working-set-bytes",
        str(2 * 1024**3),
        "--allow-local-mps-execution",
    ]
    measurement = {
        "returncode": 0,
        "elapsed_seconds": 10.0,
        "rss_measurement_kind": verifier.ROW_WATCHDOG_MEASUREMENT_KIND,
        "rss_sampling_interval_seconds": 0.25,
        "sampled_process_group_rss_high_water_bytes": 512 * 1024**2,
        "sample_count": 40,
        "worker_timeout_seconds": 43_200.0,
        "worker_process_group_rss_limit_bytes": 4 * 1024**3,
        "watchdog_completed": True,
        "process_group_empty_after_exit": True,
        "worker_terminated_by_watchdog": False,
    }
    payload = {
        "schema_version": 1,
        "kind": verifier.ROW_WATCHDOG_KIND,
        "row_id": row["row_id"],
        "worker_argv": argv,
        "worker_command_sha256": canonical_sha256(argv),
        "v2_config_sha256": file_sha256(config_path),
        "source_capability_sha256": source_capability["capability_sha256"],
        "row_file": _identity(row_path),
        "stdout_log": _identity(str(Path(row_path).with_name("row_worker.stdout.log"))),
        "stderr_log": _identity(str(Path(row_path).with_name("row_worker.stderr.log"))),
        "measurement": measurement,
        "pre_worker_host_resource_guard": _host_guard(),
        "parent_only_rusage_is_not_total_host_memory": True,
        "cross_route_host_memory_field": (
            "measurement.sampled_process_group_rss_high_water_bytes"
        ),
    }
    receipt = {**payload, "generation_digest": canonical_sha256(payload)}
    return {
        **receipt,
        "receipt_file": _identity(
            str(Path(row_path).with_name(verifier.ROW_WATCHDOG_FILENAME))
        ),
    }


def _host_guard() -> dict[str, object]:
    policy = {
        "required": True,
        "minimum_free_disk_bytes": 8 * 1024**3,
        "minimum_available_memory_bytes": 8 * 1024**3,
        "maximum_swap_used_bytes": 2 * 1024**3,
        "maximum_load_average": 8.0,
        "default_dry_plan_samples_host_resources": False,
        "rechecked_immediately_before_every_row": True,
    }
    payload = {
        "schema_version": 1,
        "kind": "worldfoam-g4-v2-pre-worker-host-resource-guard-v1",
        "policy": policy,
        "snapshot": {
            "platform": "darwin",
            "available_memory_bytes": 10 * 1024**3,
            "swap_used_bytes": 1024**3,
            "free_disk_bytes": 12 * 1024**3,
            "load_average_1m": 1.0,
            "load_average_5m": 1.0,
            "load_average_15m": 1.0,
        },
        "failures": [],
        "passed_immediately_before_worker": True,
    }
    return {**payload, "generation_digest": canonical_sha256(payload)}


def _complete_artifact() -> dict[str, object]:
    config_path = DEFAULT_CONFIG.resolve()
    config, base, base_path, workloads = build_matrix_workload_receipts(config_path)
    scene_receipts = verifier._scene_receipts(base, base_path)
    route_specs = verifier._route_specs(base)
    source_capability = required_source_capability(config_path)
    source_commit = "8" * 40
    rows: list[dict[str, object]] = []
    for scene in REQUIRED_SCENES:
        for seed in REQUIRED_SEEDS:
            workload = workloads[(scene, seed)]
            representation = canonical_sha256(
                {"scene": scene, "seed": seed, "representation": "worldfoam"}
            )
            evaluator = canonical_sha256({"scene": scene, "seed": seed, "evaluator": 1})
            for route in REQUIRED_ROUTES:
                route_spec = route_specs[route]
                scene_receipt = scene_receipts[scene]
                prefix = Path(config["output_root"]) / scene / f"seed_{seed}" / route
                row_path = str(prefix / "g4_v2_row.json")
                checkpoint = _identity(
                    str(prefix / "checkpoint_final.pt"),
                    step=300,
                    training_loss_contract_sha256=(
                        workload.training_loss_contract_sha256
                    ),
                    sample_schedule_sha256=workload.sample_schedule_sha256,
                    v2_config_sha256=workload.v2_config_sha256,
                    workload_receipt_generation_digest=workload.generation_digest,
                    route_schedule_sha256=workload.route_schedule_sha256,
                )
                metrics = {
                    "heldout_eval_psnr": {
                        "worldfoam_native4d": 20.0,
                        "worldfoam_framewise_replay": 20.0,
                        "world_tubes": 21.0,
                        "dynamic_3dgs": 22.0,
                    }[route],
                    "heldout_eval_ssim": 0.6,
                    "heldout_eval_lpips": 0.2,
                    "heldout_eval_l1": 0.1,
                }
                raw: dict[str, object] = {
                    "schema_version": 2,
                    "row_kind": verifier.ROW_KIND,
                    "row_id": f"{scene}/seed_{seed}/{route}",
                    "scene": scene,
                    "seed": seed,
                    "route": route,
                    "lane": route_spec["lane"],
                    "execution_mode": route_spec["execution_mode"],
                    "backend": route_spec["backend"],
                    "protocol_path": scene_receipt["protocol_path"],
                    "protocol_sha256": scene_receipt["protocol_sha256"],
                    "dataset_manifest_path": scene_receipt["manifest_path"],
                    "dataset_manifest_sha256": scene_receipt["manifest_sha256"],
                    "sample_id": scene_receipt["sample_id"],
                    "train_cameras": scene_receipt["train_cameras"],
                    "heldout_cameras": scene_receipt["heldout_cameras"],
                    "frame_count": 300,
                    "image_size": [384, 512],
                    "optimizer_steps": 300,
                    "frames_per_step": 4,
                    "primitive_state_temporal_scope": (
                        "per_frame" if route == "dynamic_3dgs" else "shared_across_time"
                    ),
                    "target_pixel_budget": workload.selected_target_pixels,
                    "sample_schedule_sha256": workload.sample_schedule_sha256,
                    "evaluator_sha256": evaluator,
                    "representation_sha256": (
                        representation
                        if route.startswith("worldfoam_")
                        else canonical_sha256({"scene": scene, "seed": seed, "route": route})
                    ),
                    "source_commit": source_commit,
                    "source_dirty": False,
                    "public_quality_evidence": True,
                    "paper_evidence_eligible": True,
                    "proxy_or_test_artifact": False,
                    "measurement_is_simulated": False,
                    "smoke": False,
                    "dataset_is_public": True,
                    "calibrated_multiview": True,
                    "final_checkpoint_evaluation": True,
                    "full_temporal_heldout_evaluation": True,
                    "route_attestation": {
                        "real_native": True,
                        "native_extension_attested": False,
                        "fake_native": False,
                        "source_only": False,
                        "procedural_target": False,
                        "public_target_provider": True,
                        "heldout_evaluator": True,
                        "full_geometry_trainable": True,
                        "compiled_shared_adjoint": route == "worldfoam_native4d",
                        "same_representation_framewise_replay": (
                            route == "worldfoam_framewise_replay"
                        ),
                    },
                    "checkpoint": checkpoint,
                    "heldout_media": _identity(
                        str(prefix / "heldout_full_temporal.mp4"),
                        camera_ids=scene_receipt["heldout_cameras"],
                        frame_count=300,
                    ),
                    "wandb_run_file": _identity(
                        str(prefix / "run.wandb"), run_id=f"{scene}-{seed}-{route}"
                    ),
                    "metrics": metrics,
                    "cost": {
                        "optimizer_steps": 300,
                        "target_pixels": workload.selected_target_pixels,
                        "rasterized_pixels": (
                            workload.selected_target_pixels
                            if route.startswith("worldfoam_")
                            else 300 * 4 * 384 * 512
                        ),
                        "parameter_count": 1024,
                        "parameter_bytes": 65_536,
                        "serialized_checkpoint_bytes": checkpoint["bytes"],
                        "final_active_primitive_count_per_render": 1024,
                        "stored_primitive_state_count": (
                            307_200 if route == "dynamic_3dgs" else 1024
                        ),
                        "process_lifetime_peak_rss_through_checkpoint_bytes": 1,
                        "sampled_peak_mps_driver_during_training_and_checkpoint_bytes": 1,
                        "training_and_checkpoint_elapsed_s": 1.0,
                        "process_lifetime_peak_rss_through_heldout_evaluation_bytes": 2,
                        "sampled_peak_mps_driver_through_heldout_evaluation_bytes": 2,
                        "executor_dataset_and_model_setup_elapsed_s": 1.0,
                        "heldout_evaluation_elapsed_s": 1.0,
                        "full_row_through_heldout_evaluation_elapsed_s": 3.0,
                    },
                    "v2_config_path": str(config_path.relative_to(ROOT)),
                    "v2_config_sha256": workload.v2_config_sha256,
                    "base_g4_v1_sha256": workload.base_g4_v1_sha256,
                    "training_sampling_kind": config["training_sampling"]["kind"],
                    "training_loss_contract": config["training_loss"],
                    "training_loss_contract_sha256": (
                        workload.training_loss_contract_sha256
                    ),
                    "selected_pixels_per_spacetime_sample": (
                        workload.selected_pixels_per_spacetime_sample
                    ),
                    "selected_loss_scalar_count": workload.selected_loss_scalar_count,
                    "route_schedule_sha256": workload.route_schedule_sha256,
                    "workload_receipt": workload.as_dict(),
                    "workload_receipt_generation_digest": workload.generation_digest,
                    "full_heldout_target_pixels": workload.heldout_target_pixels,
                    "training_rasterized_work_claimed_equal": False,
                    "target_source_read_receipt": _source_read_receipt(route, workload),
                    "heldout_execution_receipt": (
                        _worldfoam_heldout_receipt(workload)
                        if route.startswith("worldfoam_")
                        else _gaussian_heldout_receipt(workload)
                    ),
                    "mps_working_set_limit_receipt": _mps_limit_receipt(),
                    "parent_rusage_memory_scope": (
                        "worker_parent_only_excludes_children_use_process_group_watchdog"
                    ),
                    "heldout_wall_time_cross_route_comparable": False,
                }
                assert set(raw) == set(verifier.ROW_KEYS)
                raw_identity = _identity(row_path)
                rows.append(
                    {
                        **raw,
                        "receipt": raw_identity,
                        "execution_watchdog": _watchdog(
                            row=raw,
                            row_path=row_path,
                            config_path=config_path,
                            source_capability=source_capability,
                        ),
                    }
                )
    workload_payload = {
        f"{scene}/seed_{seed}": receipt.as_dict()
        for (scene, seed), receipt in workloads.items()
    }
    acceptance = verifier.compute_acceptance(rows, base)
    artifact: dict[str, object] = {
        "schema_version": 2,
        "artifact_kind": verifier.ARTIFACT_KIND,
        "status": "measured",
        "public_quality_evidence": True,
        "proxy_or_test_artifact": False,
        "measurement_is_simulated": False,
        "matrix_config": str(config_path.relative_to(ROOT)),
        "matrix_config_sha256": file_sha256(config_path),
        "base_g4_v1_config": str(base_path.relative_to(ROOT)),
        "base_g4_v1_sha256": file_sha256(base_path),
        "source_commit": source_commit,
        "cross_route_host_memory_source": (
            "rows[].execution_watchdog.measurement."
            "sampled_process_group_rss_high_water_bytes"
        ),
        "raw_row_rusage_scope": "worker_parent_only_excludes_child_processes",
        "workload_receipts": workload_payload,
        "workload_receipts_sha256": canonical_sha256(workload_payload),
        "rows": rows,
        "acceptance": acceptance,
        "failures": [],
        "artifact_sha256": "",
    }
    artifact["artifact_sha256"] = verifier.artifact_sha256(artifact)
    return artifact


def _write_artifact(path: Path, artifact: dict[str, object]) -> None:
    path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_pure_collected_artifact_verifier_accepts_complete_matrix(
    tmp_path: Path,
) -> None:
    artifact = _complete_artifact()
    path = tmp_path / "g4_v2_complete.json"
    _write_artifact(path, artifact)
    report = verifier.verify_artifact_file(path)
    assert report["accepted"] is True, report["failures"]
    assert report["public_quality_evidence"] is True
    assert report["row_count"] == 36


def test_pure_collected_artifact_verifier_rejects_rebound_memory_tamper(
    tmp_path: Path,
) -> None:
    artifact = copy.deepcopy(_complete_artifact())
    watchdog = artifact["rows"][0]["execution_watchdog"]
    watchdog["measurement"]["sampled_process_group_rss_high_water_bytes"] = (
        4 * 1024**3 + 1
    )
    payload = {
        key: value
        for key, value in watchdog.items()
        if key not in {"generation_digest", "receipt_file"}
    }
    watchdog["generation_digest"] = canonical_sha256(payload)
    artifact["artifact_sha256"] = verifier.artifact_sha256(artifact)
    path = tmp_path / "g4_v2_tampered.json"
    _write_artifact(path, artifact)
    report = verifier.verify_artifact_file(path)
    assert report["accepted"] is False
    assert any("process-group RSS evidence failed" in item for item in report["failures"])
