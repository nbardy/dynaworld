from __future__ import annotations

import copy
import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path

import pytest

from config_utils import load_config_file
from paper_training_protocol import (
    paper_evaluator_contract,
    paper_runtime_source_tree_identity,
    resolve_paper_training_protocol,
)
from research_experiments.paper_runner_suite import (
    run_unified_paper_matrix as matrix_runner,
)
from research_experiments.paper_runner_suite.run_unified_paper_matrix import (
    DEFAULT_MATRIX,
    MatrixRun,
    aggregate_rows,
    collect_existing_records,
    expand_matrix,
    flatten_summary,
    matrix_failure_payload,
    matrix_output_root,
    matrix_preflight,
    matrix_progress_payload,
    resolve_matrix_output_dir,
    select_matrix_runs,
    write_artifacts,
)


def _evidence(offset: float) -> dict:
    return {
        "schema_version": 2,
        "quality": {
            "eval_psnr": 20.0 + offset,
            "eval_ssim": 0.8,
            "eval_l1": 0.1,
            "heldout_eval_psnr": 18.0 + offset,
            "heldout_eval_ssim": 0.7,
            "heldout_eval_l1": 0.15,
            "heldout_eval_lpips": 0.25,
        },
        "cost": {
            "optimizer_steps": 2,
            "target_frames": 4,
            "rasterized_frames": 4,
            "target_pixels": 30_720,
            "rasterized_pixels": 30_720,
            "parameter_count": 100,
            "trainable_parameter_count": 100,
            "parameter_bytes": 400,
            "optimizer_state_bytes": 800,
            "serialized_checkpoint_bytes": 1_024,
            "sampled_peak_current_allocated_bytes": 2_048,
            "sampled_peak_driver_allocated_bytes": 4_096,
            "elapsed_s": 1.0,
        },
        "timing": {
            "cold_compile_forward_s": 0.2,
            "steady_forward_s": 0.3,
            "steady_forward_calls": 1,
            "backward_s": 0.4,
            "backward_calls": 2,
            "optimizer_s": 0.1,
            "optimizer_calls": 2,
            "train_wall_s": 1.0,
        },
        "diagnostics": {"active_count": 100},
    }


def _hashed_contract(schema_version: int, **values) -> dict:
    payload = {"schema_version": schema_version, **values}
    return {
        **payload,
        "sha256": hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest(),
    }


@lru_cache(maxsize=None)
def _manifest_validation(protocol) -> dict:
    return matrix_runner.single.validate_manifest(protocol)


def _paper_lane_payload(
    lane_name: str,
    *,
    protocol,
    common_contract: dict,
    route_native: dict,
    offset: float,
) -> dict:
    base = _evidence(offset)
    base["cost"].update(
        {
            "optimizer_steps": protocol.steps,
            "target_frames": protocol.target_frame_budget,
            "rasterized_frames": protocol.target_frame_budget,
            "target_pixels": protocol.target_pixel_budget,
            "rasterized_pixels": protocol.target_pixel_budget,
        }
    )
    paper_protocol = {
        "enabled": True,
        "sampling": {
            "mode": "spacetime_epoch",
            "same_time_count": protocol.same_time_count,
            "local_time_count": protocol.local_time_count,
            "local_time_radius": protocol.local_time_radius,
        },
        "stages": [stage.as_dict() for stage in protocol.stages],
        "sample_schedule": common_contract["sample_schedule"],
        "cost": base["cost"],
        "timing": base["timing"],
        "paper_dataset_bundle": common_contract["decoded_dataset_bundle"],
        "paper_evaluator": common_contract["evaluator"],
        "paper_runtime": common_contract["runtime"],
        "route_native_extension": route_native,
    }
    metrics = dict(base["quality"])
    payload: dict = {
        "metrics": metrics,
        "paper_protocol": paper_protocol,
        "steps": protocol.steps,
    }
    if lane_name == "world_tubes":
        payload.update(
            {
                "tube_count": 100,
                "camera_sequence_mode": "static_view",
                "segment_frames": protocol.dataset.frame_count,
                "metal_stats": {
                    "rows": [
                        {
                            "stats": {
                                "unstable_tile_fraction": 0.0,
                                "projected_trace_count": 100.0,
                                "uvt_tile_tube_pairs": 200.0,
                                "summed_per_frame_tile_splat_pairs": 400.0,
                                "effective_pair_ratio_after_unstable_fallback": 0.5,
                                "overflow_tile_count": 0.0,
                                "metal_buffer_memory": 4096.0,
                            }
                        }
                    ]
                },
            }
        )
    elif lane_name == "dynamic_3dgs":
        payload["splat_count"] = 100
    elif lane_name == "worldfoam":
        metrics.update(
            {
                "state_cell_count": 100,
                "aux_visible_fraction": 0.5,
                "aux_visible_cell_frame_events": 200,
                "aux_possible_cell_frame_events": 400,
                "aux_mean_visible_cells_per_frame": 50.0,
                "aux_median_depth_valid_fraction": 1.0,
                "aux_mean_contrib": 0.1,
                "aux_max_contrib": 0.5,
            }
        )
    else:
        raise ValueError(f"unsupported fixture lane: {lane_name}")
    return payload


def _summary(protocol=None, *, seed: int = 17) -> dict:
    source = {
        "repository_commit": "a" * 40,
        "repository_dirty": False,
        "star_uvt_commit": "b" * 40,
        "star_uvt_dirty": False,
    }
    manifest_validation = (
        copy.deepcopy(_manifest_validation(protocol))
        if protocol is not None
        else {
            "manifest": "manifest.jsonl",
            "sample_id": "scene_triplet",
            "dataset": "neural_3d_video",
            "expected_pose_source": (
                "neural_3d_llff_opencv_relative_pinhole_v2"
            ),
            "checks": {"synthetic": True},
            "input_identity": _hashed_contract(
                1,
                dataset="neural_3d_video",
                files=[],
            ),
        }
    )
    common_contract = {
        "schema_version": 1,
        "dataset_input_identity": manifest_validation["input_identity"],
        "decoded_dataset_bundle": _hashed_contract(
            1,
            train_frames="1" * 64,
            heldout_frames="2" * 64,
            cameras="3" * 64,
            pose_source=manifest_validation["expected_pose_source"],
        ),
        "evaluator": paper_evaluator_contract(),
        "runtime": _hashed_contract(1, host="test-host"),
        "sample_schedule": {
            "schema_version": 1,
            "algorithm": "spacetime_epoch_v1",
            "sampler_seed": seed + (
                protocol.sampler_seed_offset if protocol is not None else 7001
            ),
            "record_count": protocol.steps if protocol is not None else 2,
            "sha256": "c" * 64,
        },
    }
    native_path = Path(__file__).resolve()
    route_native = {
        "module": "test._C",
        "path": str(native_path),
        "bytes": native_path.stat().st_size,
        "sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
        "runtime_source_tree": paper_runtime_source_tree_identity(
            native_path.parent
        ),
    }
    summary = {
        "status": "complete",
        "seed": seed,
        "world_tubes_requested_backward_policy": "fast_exploration",
        "world_tubes_backward_policy": "fast_exploration",
        "uvt_world_representation": "legacy_tube",
        "uvt_alpha_mode": "peak_splat",
        "uvt_render_backend": "metal_tile",
        "uvt_amplitude_convention": "fiber_integrated",
        "uvt_opacity_semantics": "peak_alpha_amplitude",
        "uvt_retained_depth_samples": 48,
        "uvt_retained_sigma_extent": 6.0,
        "uvt_order_certificate_sigma": 6.0,
        "uvt_order_certificate_min_gap": 0.0,
        "uvt_spd4_init_precision_z": None,
        "frozen_world_replay_compiled": False,
        "frozen_world_max_frames": 0,
        "worldfoam_initializer": "base_config",
        "worldfoam_initializer_identity": _hashed_contract(
            1,
            requested_initializer="base_config",
            file={
                "role": "worldfoam_initializer",
                "path": str(native_path),
                "bytes": native_path.stat().st_size,
                "sha256": hashlib.sha256(
                    native_path.read_bytes()
                ).hexdigest(),
            },
        ),
        "source": source,
        "source_finish": dict(source),
        "common_evidence_contract": common_contract,
        "manifest_validation": manifest_validation,
        "execution_safety": {
            "high_risk": False,
            "live_resources": {},
        },
        "protocol": {
            "name": "smoke",
            "dataset": {
                "sample_id": "scene_triplet",
                "train_cameras": ["a", "b"],
                "heldout_cameras": ["c"],
            },
        },
        "lanes": {
            "world_tubes": {
                "evidence": _evidence(1.0),
                "wandb": {"mode": "offline", "run_id": "world-tubes-test"},
                "route_native_extension": route_native,
            },
            "worldfoam": {
                "evidence": _evidence(0.0),
                "wandb": {"mode": "offline", "run_id": "worldfoam-test"},
                "route_native_extension": route_native,
            },
            "dynamic_3dgs": {
                "evidence": _evidence(-1.0),
                "wandb": {"mode": "offline", "run_id": "dynamic-test"},
                "route_native_extension": route_native,
            },
        },
    }
    if protocol is not None:
        summary["protocol"] = protocol.as_dict()
        offsets = {
            "world_tubes": 1.0,
            "worldfoam": 0.0,
            "dynamic_3dgs": -1.0,
        }
        for lane_name, lane in summary["lanes"].items():
            payload = _paper_lane_payload(
                lane_name,
                protocol=protocol,
                common_contract=common_contract,
                route_native=route_native,
                offset=offsets[lane_name],
            )
            lane["metrics"] = payload["metrics"]
            lane["paper_protocol"] = payload["paper_protocol"]
            lane["evidence"] = matrix_runner.single.build_lane_evidence(
                lane_name,
                payload,
                frame_count=protocol.dataset.frame_count,
            )
    return summary


def _write_media(run_dir: Path, *, steps: int) -> None:
    comparison_dir = run_dir / "world_tubes_dynamic_3dgs"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    for prefix in ("star_uvt", "free_dynamic_splats"):
        for split in ("train", "heldout"):
            (comparison_dir / f"{prefix}_{split}_view0_side_by_side.mp4").write_bytes(
                b"media"
            )
    worldfoam_dir = run_dir / "worldfoam"
    worldfoam_dir.mkdir(parents=True, exist_ok=True)
    (worldfoam_dir / f"side_by_side_step_{steps:04d}.mp4").write_bytes(b"media")
    (
        worldfoam_dir / f"heldout_side_by_side_step_{steps:04d}.mp4"
    ).write_bytes(b"media")


def _write_existing_run(out_dir: Path, run: MatrixRun) -> None:
    protocol = resolve_paper_training_protocol(load_config_file(run.protocol_path))
    run_dir = out_dir / protocol.name / f"seed_{run.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = _summary(protocol, seed=run.seed)
    _write_media(run_dir, steps=protocol.steps)
    _write_execution_identities(run_dir, run, summary)
    (run_dir / "run_summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )


def _write_execution_identities(
    run_dir: Path,
    run: MatrixRun,
    summary: dict,
) -> None:
    protocol = resolve_paper_training_protocol(
        load_config_file(run.protocol_path)
    )
    protocol_sha256 = hashlib.sha256(run.protocol_path.read_bytes()).hexdigest()
    source_digest = hashlib.sha256(
        json.dumps(
            summary["source"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    def identity(path: Path, role: str) -> dict:
        return {
            "role": role,
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    def wandb_files_dir(lane_name: str) -> Path:
        path = run_dir / "wandb_test_files" / lane_name / "files"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def run_file(lane_name: str, run_id: str) -> dict:
        path = wandb_files_dir(lane_name).parent / f"run-{run_id}.wandb"
        path.write_bytes(f"wandb:{lane_name}:{run_id}".encode())
        return identity(path, "wandb_run_file")

    comparison_root = run_dir / "world_tubes_dynamic_3dgs"
    comparison_root.mkdir(parents=True, exist_ok=True)
    route_native = summary["lanes"]["world_tubes"]["route_native_extension"]
    runtime_source_tree = route_native["runtime_source_tree"]
    star_native = {
        **route_native,
        "source_tree_sha256": runtime_source_tree["sha256"],
        "source_file_count": runtime_source_tree["file_count"],
    }
    camera_projection, camera_sequence_mode, segment_frames = (
        matrix_runner.single.paper_world_tubes_camera_policy(protocol)
    )

    def retained_lane(lane_name: str) -> dict:
        summary_lane = summary["lanes"][lane_name]
        lane = {
            "lane": lane_name,
            "steps": protocol.steps,
            "metrics": summary_lane["metrics"],
            "paper_protocol": summary_lane["paper_protocol"],
        }
        diagnostics = summary_lane["evidence"]["diagnostics"]
        if lane_name == "world_tubes":
            lane.update(
                {
                    "tube_count": diagnostics["active_trace_count"],
                    "camera_sequence_mode": diagnostics["camera_chart_mode"],
                    "segment_frames": protocol.dataset.frame_count,
                    "metal_stats": {
                        "rows": [
                            {
                                "stats": {
                                    "unstable_tile_fraction": diagnostics[
                                        "unstable_tile_fraction_mean"
                                    ],
                                    "projected_trace_count": diagnostics[
                                        "compiled_trace_count_mean"
                                    ],
                                    "uvt_tile_tube_pairs": diagnostics[
                                        "tile_trace_pairs_mean"
                                    ],
                                    "summed_per_frame_tile_splat_pairs": diagnostics[
                                        "per_frame_tile_trace_pairs_mean"
                                    ],
                                    "effective_pair_ratio_after_unstable_fallback": diagnostics[
                                        "effective_pair_ratio_after_fallback_mean"
                                    ],
                                    "overflow_tile_count": diagnostics[
                                        "overflow_tile_count_mean"
                                    ],
                                    "metal_buffer_memory": diagnostics[
                                        "metal_buffer_bytes_mean"
                                    ],
                                }
                            }
                        ]
                    },
                    "frozen_world_replay_compiled": None,
                }
            )
        elif lane_name == "dynamic_3dgs":
            lane["splat_count"] = diagnostics["active_splats_per_frame"]
        return lane

    merged_report = {
        "meta": {
            "only_lane": "isolated_merged",
            "skip_splats": False,
            "execution_model": "one_child_process_per_representation",
            "device": "mps",
            "seed": run.seed,
            "train_cameras": list(protocol.dataset.train_cameras),
            "heldout_cameras": list(protocol.dataset.heldout_cameras),
            "frame_count": protocol.dataset.frame_count,
            "pose_source": summary["manifest_validation"][
                "expected_pose_source"
            ],
            "uvt_world_representation": summary[
                "uvt_world_representation"
            ],
            "uvt_alpha_mode": summary["uvt_alpha_mode"],
            "uvt_render_backend": summary["uvt_render_backend"],
            "uvt_amplitude_convention": summary[
                "uvt_amplitude_convention"
            ],
            "uvt_opacity_semantics": summary["uvt_opacity_semantics"],
            "uvt_retained_depth_samples": summary[
                "uvt_retained_depth_samples"
            ],
            "uvt_retained_sigma_extent": summary[
                "uvt_retained_sigma_extent"
            ],
            "uvt_order_certificate_sigma": summary[
                "uvt_order_certificate_sigma"
            ],
            "uvt_order_certificate_min_gap": summary[
                "uvt_order_certificate_min_gap"
            ],
            "uvt_spd4_init_precision_z": summary[
                "uvt_spd4_init_precision_z"
            ],
            "uvt_camera_projection": camera_projection,
            "uvt_camera_sequence_mode": camera_sequence_mode,
            "uvt_segment_frames": segment_frames,
            "uvt_backward_policy": {
                "name": summary["world_tubes_backward_policy"]
            },
            "frozen_world_replay_compiled": False,
            "frozen_world_max_frames": 0,
            "star_uvt_native_extension": star_native,
            "route_native_extension": route_native,
            "route_native_extensions": {
                lane_name: summary["lanes"][lane_name][
                    "route_native_extension"
                ]
                for lane_name in ("world_tubes", "dynamic_3dgs")
            },
            "paper_dataset_bundle": summary[
                "common_evidence_contract"
            ]["decoded_dataset_bundle"],
            "paper_evaluator": summary["common_evidence_contract"][
                "evaluator"
            ],
            "paper_runtime": summary["common_evidence_contract"]["runtime"],
        },
        "star_uvt": retained_lane("world_tubes"),
        "star_uvt_selected": {"checkpoint": "final"},
        "free_dynamic_splats": retained_lane("dynamic_3dgs"),
    }
    (comparison_root / "comparison_report.json").write_text(
        json.dumps(merged_report),
        encoding="utf-8",
    )
    merged_report_sha256 = matrix_runner.canonical_json_sha256(merged_report)
    comparison_commands = matrix_runner.single.comparison_lane_commands(
        run.protocol_path,
        protocol,
        run.seed,
        comparison_root,
        backward_policy=run.backward_policy,
        device="mps",
        allow_local_mps_execution=True,
        python=sys.executable,
    )
    for lane_name in ("world_tubes", "dynamic_3dgs"):
        lane_dir = comparison_root / lane_name
        lane_dir.mkdir(parents=True, exist_ok=True)
        report_path = lane_dir / "comparison_report.json"
        report_key = matrix_runner.single.LANE_REPORT_KEYS[lane_name]
        isolated_report = {
            "meta": {
                **merged_report["meta"],
                "only_lane": lane_name,
                "route_native_extension": summary["lanes"][lane_name][
                    "route_native_extension"
                ],
            },
            "star_uvt": (
                merged_report["star_uvt"]
                if lane_name == "world_tubes"
                else None
            ),
            "star_uvt_selected": (
                merged_report["star_uvt_selected"]
                if lane_name == "world_tubes"
                else None
            ),
            "free_dynamic_splats": (
                merged_report[report_key]
                if lane_name == "dynamic_3dgs"
                else None
            ),
        }
        report_path.write_text(json.dumps(isolated_report), encoding="utf-8")
        reported_wandb = summary["lanes"][lane_name].get("wandb")
        valid_wandb = (
            isinstance(reported_wandb, dict)
            and bool(str(reported_wandb.get("run_id", "")).strip())
        )
        run_id = (
            str(reported_wandb["run_id"])
            if valid_wandb
            else f"{lane_name}-sidecar"
        )
        wandb_identity = {
            "schema_version": 1,
            "project": "dynaworld",
            "name": f"test-{lane_name}",
            "mode": "offline",
            "run_id": run_id,
            "run_dir": str(wandb_files_dir(lane_name).resolve()),
            "source_digest": source_digest,
            "comparison_report_sha256": merged_report_sha256,
            "config_sha256": "d" * 64,
            "run_file": run_file(lane_name, run_id),
        }
        wandb_identity_path = lane_dir / "wandb_identity.json"
        wandb_identity_path.write_text(
            json.dumps(wandb_identity),
            encoding="utf-8",
        )
        if valid_wandb:
            summary["lanes"][lane_name]["wandb"] = wandb_identity
        (lane_dir / "execution_identity.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "lane": lane_name,
                    "protocol": protocol.as_dict(),
                    "source_start": summary["source"],
                    "source_finish": summary["source"],
                    "dataset_input_identity": summary[
                        "common_evidence_contract"
                    ]["dataset_input_identity"],
                    "protocol_sha256": protocol_sha256,
                    "command": comparison_commands[lane_name],
                    "comparison_report_sha256": hashlib.sha256(
                        report_path.read_bytes()
                    ).hexdigest(),
                    "wandb_identity": identity(
                        wandb_identity_path,
                        f"{lane_name}:wandb_identity",
                    ),
                }
            ),
            encoding="utf-8",
        )
    worldfoam_dir = run_dir / "worldfoam"
    worldfoam_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = {
        "paper_protocol_summary": worldfoam_dir / "paper_protocol_summary.json",
        "best_metrics": worldfoam_dir / "best_metrics.json",
        "eval_metrics_history": worldfoam_dir / "eval_metrics_history.jsonl",
        "resolved_config": worldfoam_dir / "resolved_config.json",
        "checkpoint_final": worldfoam_dir / "checkpoint_final.pt",
        "train_metrics_history": worldfoam_dir / "train_metrics_history.jsonl",
        "final_train_media": worldfoam_dir
        / f"side_by_side_step_{summary['protocol'].get('steps', 2):04d}.mp4",
        "final_heldout_media": worldfoam_dir
        / (
            "heldout_side_by_side_step_"
            f"{summary['protocol'].get('steps', 2):04d}.mp4"
        ),
    }
    powerfoam_summary = summary["lanes"]["worldfoam"]["paper_protocol"]
    powerfoam_metrics = summary["lanes"]["worldfoam"]["metrics"]
    artifact_paths["paper_protocol_summary"].write_text(
        json.dumps(powerfoam_summary),
        encoding="utf-8",
    )
    artifact_paths["best_metrics"].write_text(
        json.dumps(
            {
                "best_metric_name": "heldout_eval_psnr",
                "best_metric_value": powerfoam_metrics[
                    "heldout_eval_psnr"
                ],
            }
        ),
        encoding="utf-8",
    )
    artifact_paths["eval_metrics_history"].write_text(
        json.dumps({"step": protocol.steps, "metrics": powerfoam_metrics})
        + "\n",
        encoding="utf-8",
    )
    expected_powerfoam_config = matrix_runner.single.powerfoam_config(
        load_config_file(run.protocol_path),
        protocol,
        run.seed,
        worldfoam_dir,
        wandb_mode="offline",
        device="mps",
        worldfoam_initializer=run.worldfoam_initializer,
    )
    summary["worldfoam_initializer_identity"] = (
        matrix_runner.single.powerfoam_initializer_identity(
            expected_powerfoam_config,
            requested_initializer=run.worldfoam_initializer,
        )
    )
    resolved_powerfoam_config = matrix_runner.serialize_config_value(
        expected_powerfoam_config
    )
    resolved_powerfoam_config["render"].update(
        {
            "background_mode": "fixed",
            "background": [0.0, 0.0, 0.0],
            "eval_color_calibration": "none",
        }
    )
    artifact_paths["resolved_config"].write_text(
        json.dumps(resolved_powerfoam_config),
        encoding="utf-8",
    )
    resolved_config_binding = (
        matrix_runner.single.worldfoam_resolved_config_binding(
            expected_powerfoam_config,
            artifact_paths["resolved_config"],
        )
    )
    summary["lanes"]["worldfoam"][
        "resolved_config_binding"
    ] = resolved_config_binding
    artifact_paths["checkpoint_final"].write_bytes(b"x" * 1_024)
    artifact_paths["train_metrics_history"].write_text(
        json.dumps(
            {
                "step": protocol.steps,
                "loss": 0.1,
                "train_wall_s": powerfoam_summary["timing"][
                    "train_wall_s"
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    for name in ("final_train_media", "final_heldout_media"):
        if not artifact_paths[name].exists():
            artifact_paths[name].write_bytes(b"media")
    reported_worldfoam_wandb = summary["lanes"]["worldfoam"].get("wandb")
    valid_worldfoam_wandb = (
        isinstance(reported_worldfoam_wandb, dict)
        and bool(str(reported_worldfoam_wandb.get("run_id", "")).strip())
    )
    worldfoam_run_id = (
        str(reported_worldfoam_wandb["run_id"])
        if valid_worldfoam_wandb
        else "worldfoam-sidecar"
    )
    worldfoam_wandb_identity = {
        "schema_version": 1,
        "project": "dynaworld",
        "name": "test-worldfoam",
        "mode": "offline",
        "run_id": worldfoam_run_id,
        "run_dir": str(wandb_files_dir("worldfoam").resolve()),
        "source_digest": source_digest,
        "paper_protocol_summary_sha256": hashlib.sha256(
            artifact_paths["paper_protocol_summary"].read_bytes()
        ).hexdigest(),
        "resolved_config_sha256": hashlib.sha256(
            artifact_paths["resolved_config"].read_bytes()
        ).hexdigest(),
        "finalized": True,
        "run_file": run_file("worldfoam", worldfoam_run_id),
    }
    worldfoam_wandb_path = worldfoam_dir / "wandb_identity.json"
    worldfoam_wandb_path.write_text(
        json.dumps(worldfoam_wandb_identity),
        encoding="utf-8",
    )
    artifact_paths["wandb_identity"] = worldfoam_wandb_path
    if valid_worldfoam_wandb:
        summary["lanes"]["worldfoam"]["wandb"] = worldfoam_wandb_identity
    (worldfoam_dir / "execution_identity.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "lane": "worldfoam",
                "protocol": protocol.as_dict(),
                "source_start": summary["source"],
                "source_finish": summary["source"],
                "dataset_input_identity": summary[
                    "common_evidence_contract"
                ]["dataset_input_identity"],
                "protocol_sha256": protocol_sha256,
                "command": matrix_runner.single.worldfoam_lane_command(
                    run.protocol_path,
                    run.seed,
                    worldfoam_dir,
                    device="mps",
                    wandb_mode="offline",
                    worldfoam_initializer=run.worldfoam_initializer,
                    allow_local_mps_execution=True,
                    python=sys.executable,
                ),
                "initializer_identity": summary[
                    "worldfoam_initializer_identity"
                ],
                "resolved_config_binding": resolved_config_binding,
                "artifacts": {
                    name: identity(path, f"worldfoam:{name}")
                    for name, path in artifact_paths.items()
                },
            }
        ),
        encoding="utf-8",
    )


def test_submission_matrix_expands_to_the_frozen_seven_runs() -> None:
    runs = expand_matrix(load_config_file(DEFAULT_MATRIX))

    assert len(runs) == 7
    assert [run.seed for run in runs[:3]] == [17, 29, 43]
    assert sum(run.role == "pixel_matched_control" for run in runs) == 3
    assert sum(run.role == "sampler_control" for run in runs) == 1
    assert all(run.worldfoam_initializer == "base_config" for run in runs)


def test_matrix_selection_preserves_declared_order_and_rejects_drift() -> None:
    runs = expand_matrix(load_config_file(DEFAULT_MATRIX))

    selected = select_matrix_runs(
        runs,
        [runs[4].key, runs[1].key],
    )

    assert [run.key for run in selected] == [runs[1].key, runs[4].key]
    with pytest.raises(ValueError, match="must be unique"):
        select_matrix_runs(runs, [runs[0].key, runs[0].key])
    with pytest.raises(ValueError, match="not declared"):
        select_matrix_runs(runs, ["unknown/seed_17/fast_exploration"])


def test_matrix_progress_is_compact_and_never_marks_a_subset_complete() -> None:
    runs = expand_matrix(load_config_file(DEFAULT_MATRIX))
    accepted = [{"run": runs[0].as_dict(), "summary": {"large": "omitted"}}]

    progress = matrix_progress_payload(
        matrix_name="world_tubes_submission_matrix_v1",
        runs=runs,
        accepted_records=accepted,
        status="partial",
        selected_runs=[runs[0]],
        new_run_count=1,
    )

    assert progress["status"] == "partial"
    assert progress["accepted_run_count"] == 1
    assert progress["accepted_lane_row_count"] == 3
    assert progress["accepted_runs"] == [runs[0].as_dict()]
    assert "summary" not in progress["accepted_runs"][0]
    assert progress["missing_runs"] == [run.key for run in runs[1:]]


def test_matrix_failure_progress_preserves_resume_boundary() -> None:
    runs = expand_matrix(load_config_file(DEFAULT_MATRIX))
    accepted = [{"run": runs[0].as_dict(), "summary": {"large": "omitted"}}]

    progress = matrix_failure_payload(
        matrix_name="world_tubes_submission_matrix_v1",
        runs=runs,
        accepted_records=accepted,
        selected_runs=[runs[1]],
        new_run_count=0,
        failed_run=runs[1],
        error=RuntimeError("child exited 1"),
    )

    assert progress["status"] == "failed"
    assert progress["accepted_run_count"] == 1
    assert progress["failed_run"]["key"] == runs[1].key
    assert progress["failure"] == {
        "exception_type": "RuntimeError",
        "message": "child exited 1",
    }
    assert progress["missing_runs"] == [run.key for run in runs[1:]]
    assert "summary" not in progress["accepted_runs"][0]


def test_official_matrix_output_roots_are_distinct_and_canonical() -> None:
    submission = load_config_file(DEFAULT_MATRIX)
    full_path = DEFAULT_MATRIX.parent / "world_tubes_full_public_matrix_v1.jsonc"
    full = load_config_file(full_path)

    submission_root = matrix_output_root(submission)
    full_root = matrix_output_root(full)

    assert submission_root == matrix_runner.DEFAULT_OUT_DIR
    assert submission_root.name == (
        "2026-07-28_world_tubes_submission_matrix_schema2"
    )
    assert full_root.name == "2026-07-28_world_tubes_full_public_matrix_schema2"
    assert submission_root != full_root


def test_declared_matrix_output_root_rejects_explicit_mismatch(
    tmp_path: Path,
) -> None:
    matrix = load_config_file(DEFAULT_MATRIX)

    assert resolve_matrix_output_dir(matrix, None) == matrix_runner.DEFAULT_OUT_DIR
    assert (
        resolve_matrix_output_dir(matrix, matrix_runner.DEFAULT_OUT_DIR)
        == matrix_runner.DEFAULT_OUT_DIR
    )
    with pytest.raises(ValueError, match="disagrees"):
        resolve_matrix_output_dir(matrix, tmp_path / "wrong-root")


def test_matrix_preflight_reports_source_estimate_and_live_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = expand_matrix(load_config_file(DEFAULT_MATRIX))[0]
    monkeypatch.setattr(
        matrix_runner.single,
        "source_provenance",
        lambda: {
            "repository_commit": "a" * 40,
            "repository_dirty": True,
            "star_uvt_commit": "b" * 40,
            "star_uvt_dirty": False,
        },
    )
    monkeypatch.setattr(
        matrix_runner.single,
        "local_mps_safety_estimate",
        lambda _protocol: {"high_risk": True, "estimated_peak_gib": 18.0},
    )
    monkeypatch.setattr(
        matrix_runner.single,
        "live_resource_snapshot",
        lambda: {"platform": "darwin", "swap_used_bytes": 3},
    )
    monkeypatch.setattr(
        matrix_runner.single,
        "require_live_resources",
        lambda _snapshot: (_ for _ in ()).throw(
            RuntimeError("live resource rejection")
        ),
    )
    monkeypatch.setattr(
        matrix_runner,
        "lpips_alex_asset_status",
        lambda: {"status": "pass"},
    )
    monkeypatch.setattr(
        matrix_runner,
        "wandb_local_readiness",
        lambda *_args, **_kwargs: {"status": "pass"},
    )
    monkeypatch.setattr(
        matrix_runner,
        "matrix_retained_output_budget",
        lambda _runs: {"required_free_bytes": 1},
    )
    monkeypatch.setattr(matrix_runner, "_disk_free_bytes", lambda _path: 2)

    preflight = matrix_preflight([run], device="mps")

    assert preflight["status"] == "rejected"
    assert preflight["checks"] == {
        "clean_superproject_and_star_source": False,
        "all_protocol_estimates_below_incident_limit": False,
        "live_resource_gate": False,
        "supported_device": True,
        "lpips_alex_assets_exact": True,
        "wandb_local_readiness": True,
        "retained_output_disk_budget": True,
    }
    assert preflight["live_resource_error"] == "live resource rejection"
    assert preflight["high_risk_protocols"] == [
        str(run.protocol_path.relative_to(matrix_runner.ROOT))
    ]


def test_full_public_matrix_unifies_controls_triplets_scenes_and_dnerf() -> None:
    matrix = DEFAULT_MATRIX.parent / "world_tubes_full_public_matrix_v1.jsonc"
    runs = expand_matrix(load_config_file(matrix))

    assert len(runs) == 21
    assert sum(run.role == "primary_progressive" for run in runs) == 3
    assert sum(run.role == "pixel_matched_control" for run in runs) == 3
    assert sum(run.role == "sampler_control" for run in runs) == 1
    assert sum(run.role == "camera_split_breadth" for run in runs) == 6
    assert sum(run.role == "scene_breadth" for run in runs) == 6
    assert sum(run.role == "controlled_dnerf_breadth" for run in runs) == 1
    assert sum(run.role == "deterministic_correctness_timing" for run in runs) == 1
    deterministic = next(run for run in runs if run.role == "deterministic_correctness_timing")
    assert deterministic.backward_policy == "deterministic_quality"


def test_full_public_matrix_existing_accounting_is_three_of_twenty_one(
    tmp_path: Path,
) -> None:
    matrix = DEFAULT_MATRIX.parent / "world_tubes_full_public_matrix_v1.jsonc"
    runs = expand_matrix(load_config_file(matrix))
    for run in runs[:3]:
        _write_existing_run(tmp_path, run)

    records, missing = collect_existing_records(runs, tmp_path)

    assert len(runs) == 21
    assert [record["run"]["key"] for record in records] == [
        run.key for run in runs[:3]
    ]
    assert missing == [run.key for run in runs[3:]]
    assert len(missing) == 18
    assert len(records) * len(matrix_runner.LANE_ORDER) == 9


def test_matrix_can_select_scene_specific_worldfoam_initialization() -> None:
    protocol = DEFAULT_MATRIX.parent / "coffee_martini_protocol_smoke_2step.jsonc"
    runs = expand_matrix(
        {
            "runs": [
                {
                    "role": "breadth",
                    "protocol": str(protocol),
                    "seeds": [17],
                    "world_tubes_backward_policy": "fast_exploration",
                    "worldfoam_initializer": "video",
                }
            ]
        }
    )

    assert runs[0].worldfoam_initializer == "video"


def test_matrix_rejects_variants_that_share_an_actual_output_directory() -> None:
    protocol = (
        DEFAULT_MATRIX.parent / "coffee_martini_protocol_smoke_2step.jsonc"
    )

    with pytest.raises(ValueError, match="actual output directory"):
        expand_matrix(
            {
                "runs": [
                    {
                        "role": "fast",
                        "protocol": str(protocol),
                        "seeds": [17],
                        "world_tubes_backward_policy": "fast_exploration",
                    },
                    {
                        "role": "deterministic",
                        "protocol": str(protocol),
                        "seeds": [17],
                        "world_tubes_backward_policy": "deterministic_quality",
                    },
                ]
            }
        )


def test_matrix_artifacts_are_generated_from_validated_evidence(tmp_path: Path) -> None:
    run = MatrixRun(
        role="mechanical_smoke",
        protocol_path=DEFAULT_MATRIX,
        seed=17,
        backward_policy="fast_exploration",
    )
    summary = _summary()
    rows = flatten_summary(run, summary)

    assert [row["lane"] for row in rows] == ["world_tubes", "worldfoam", "dynamic_3dgs"]
    artifacts = write_artifacts(
        tmp_path,
        "test_matrix",
        [{"run": run.as_dict(), "summary": summary}],
    )

    assert all((tmp_path / Path(path).name).exists() for path in artifacts.values())
    payload = json.loads((tmp_path / "paper_rows.json").read_text(encoding="utf-8"))
    assert len(payload["rows"]) == 3
    assert len(payload["aggregated"]) == 3
    assert "LPIPS" in (tmp_path / "paper_table.tex").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "field",
    (
        "repository_commit",
        "star_uvt_commit",
        "dataset_input_sha256",
        "decoded_dataset_sha256",
        "evaluator_sha256",
        "runtime_sha256",
        "paper_backward_policy",
        "route_native_sha256",
    ),
)
def test_aggregation_rejects_cross_seed_contract_drift(field: str) -> None:
    run_17 = MatrixRun(
        role="repeat",
        protocol_path=DEFAULT_MATRIX,
        seed=17,
        backward_policy="fast_exploration",
    )
    run_29 = MatrixRun(
        role="repeat",
        protocol_path=DEFAULT_MATRIX,
        seed=29,
        backward_policy="fast_exploration",
    )
    row_17 = flatten_summary(run_17, _summary(seed=17))[0]
    row_29 = flatten_summary(run_29, _summary(seed=29))[0]
    row_29[field] = "drifted"

    with pytest.raises(ValueError, match=field):
        aggregate_rows([row_17, row_29])


def test_aggregation_rejects_duplicate_seeds() -> None:
    run = MatrixRun(
        role="repeat",
        protocol_path=DEFAULT_MATRIX,
        seed=17,
        backward_policy="fast_exploration",
    )
    row = flatten_summary(run, _summary(seed=17))[0]

    with pytest.raises(ValueError, match="duplicate seeds"):
        aggregate_rows([row, copy.deepcopy(row)])


def test_existing_evidence_collection_ignores_partial_lane_debris(tmp_path: Path) -> None:
    protocol_path = (
        DEFAULT_MATRIX.parent / "coffee_martini_protocol_smoke_2step.jsonc"
    )
    protocol = resolve_paper_training_protocol(load_config_file(protocol_path))
    protocol_name = protocol.name
    complete = MatrixRun(
        role="mechanical_smoke",
        protocol_path=protocol_path,
        seed=17,
        backward_policy="fast_exploration",
    )
    incomplete = MatrixRun(
        role="mechanical_smoke",
        protocol_path=protocol_path,
        seed=29,
        backward_policy="fast_exploration",
    )
    summary = _summary(protocol)
    complete_dir = tmp_path / protocol_name / "seed_17"
    complete_dir.mkdir(parents=True)
    _write_media(complete_dir, steps=protocol.steps)
    _write_execution_identities(complete_dir, complete, summary)
    (complete_dir / "run_summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    partial_dir = tmp_path / protocol_name / "seed_29" / "world_tubes_dynamic_3dgs"
    partial_dir.mkdir(parents=True)
    (partial_dir / "comparison_report.json").write_text("{}", encoding="utf-8")

    records, missing = collect_existing_records([complete, incomplete], tmp_path)

    assert [record["run"]["seed"] for record in records] == [17]
    assert missing == [incomplete.key]


def test_existing_evidence_rejects_merged_comparison_report_drift(
    tmp_path: Path,
) -> None:
    protocol_path = (
        DEFAULT_MATRIX.parent / "coffee_martini_protocol_smoke_2step.jsonc"
    )
    protocol = resolve_paper_training_protocol(load_config_file(protocol_path))
    run = MatrixRun(
        role="mechanical_smoke",
        protocol_path=protocol_path,
        seed=17,
        backward_policy="fast_exploration",
    )
    _write_existing_run(tmp_path, run)
    merged_report_path = (
        tmp_path
        / protocol.name
        / "seed_17"
        / "world_tubes_dynamic_3dgs"
        / "comparison_report.json"
    )
    merged_report_path.write_text(
        json.dumps({"fixture": "mutated_after_wandb_finish"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="merged comparison report is invalid"):
        matrix_runner.load_existing_summary(run, tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("schema_version", 0, "execution contract drifted"),
        ("lane", "wrong_lane", "execution contract drifted"),
        ("protocol", {"name": "wrong"}, "execution contract drifted"),
        ("command", None, "command drifted"),
    ),
)
def test_existing_evidence_rejects_child_execution_contract_drift(
    tmp_path: Path,
    field: str,
    value,
    message: str,
) -> None:
    protocol_path = (
        DEFAULT_MATRIX.parent / "coffee_martini_protocol_smoke_2step.jsonc"
    )
    protocol = resolve_paper_training_protocol(load_config_file(protocol_path))
    run = MatrixRun(
        role="mechanical_smoke",
        protocol_path=protocol_path,
        seed=17,
        backward_policy="fast_exploration",
    )
    _write_existing_run(tmp_path, run)
    identity_path = (
        tmp_path
        / protocol.name
        / "seed_17"
        / "world_tubes_dynamic_3dgs"
        / "world_tubes"
        / "execution_identity.json"
    )
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if field == "command":
        identity[field] = [*identity[field], "--unexpected-option"]
    else:
        identity[field] = value
    identity_path.write_text(json.dumps(identity), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        matrix_runner.load_existing_summary(run, tmp_path)


def test_existing_evidence_rejects_parent_evidence_not_derived_from_report(
    tmp_path: Path,
) -> None:
    protocol_path = (
        DEFAULT_MATRIX.parent / "coffee_martini_protocol_smoke_2step.jsonc"
    )
    protocol = resolve_paper_training_protocol(load_config_file(protocol_path))
    run = MatrixRun(
        role="mechanical_smoke",
        protocol_path=protocol_path,
        seed=17,
        backward_policy="fast_exploration",
    )
    _write_existing_run(tmp_path, run)
    summary_path = (
        tmp_path / protocol.name / "seed_17" / "run_summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["lanes"]["world_tubes"]["evidence"]["quality"][
        "heldout_eval_psnr"
    ] += 1.0
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="evidence does not match"):
        matrix_runner.load_existing_summary(run, tmp_path)


def test_existing_evidence_rejects_merged_and_isolated_report_drift(
    tmp_path: Path,
) -> None:
    protocol_path = (
        DEFAULT_MATRIX.parent / "coffee_martini_protocol_smoke_2step.jsonc"
    )
    protocol = resolve_paper_training_protocol(load_config_file(protocol_path))
    run = MatrixRun(
        role="mechanical_smoke",
        protocol_path=protocol_path,
        seed=17,
        backward_policy="fast_exploration",
    )
    _write_existing_run(tmp_path, run)
    lane_dir = (
        tmp_path
        / protocol.name
        / "seed_17"
        / "world_tubes_dynamic_3dgs"
        / "world_tubes"
    )
    report_path = lane_dir / "comparison_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["star_uvt"]["metrics"]["heldout_eval_psnr"] += 1.0
    report_path.write_text(json.dumps(report), encoding="utf-8")
    identity_path = lane_dir / "execution_identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    identity["comparison_report_sha256"] = hashlib.sha256(
        report_path.read_bytes()
    ).hexdigest()
    identity_path.write_text(json.dumps(identity), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match.*isolated"):
        matrix_runner.load_existing_summary(run, tmp_path)


def test_existing_evidence_rejects_worldfoam_eval_not_reflected_in_parent(
    tmp_path: Path,
) -> None:
    protocol_path = (
        DEFAULT_MATRIX.parent / "coffee_martini_protocol_smoke_2step.jsonc"
    )
    protocol = resolve_paper_training_protocol(load_config_file(protocol_path))
    run = MatrixRun(
        role="mechanical_smoke",
        protocol_path=protocol_path,
        seed=17,
        backward_policy="fast_exploration",
    )
    _write_existing_run(tmp_path, run)
    worldfoam_dir = tmp_path / protocol.name / "seed_17" / "worldfoam"
    history_path = worldfoam_dir / "eval_metrics_history.jsonl"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    history["metrics"]["heldout_eval_psnr"] += 1.0
    history_path.write_text(json.dumps(history) + "\n", encoding="utf-8")
    identity_path = worldfoam_dir / "execution_identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    identity["artifacts"]["eval_metrics_history"]["sha256"] = (
        hashlib.sha256(history_path.read_bytes()).hexdigest()
    )
    identity_path.write_text(json.dumps(identity), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match retained artifacts"):
        matrix_runner.load_existing_summary(run, tmp_path)


def test_reuse_existing_skips_single_execute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol_path = (
        DEFAULT_MATRIX.parent / "coffee_martini_protocol_smoke_2step.jsonc"
    )
    run = MatrixRun(
        role="mechanical_smoke",
        protocol_path=protocol_path,
        seed=17,
        backward_policy="fast_exploration",
    )
    out_dir = tmp_path / "outputs"
    _write_existing_run(out_dir, run)
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(
        json.dumps(
            {
                "name": "reuse_test",
                "runs": [
                    {
                        "role": run.role,
                        "protocol": str(protocol_path),
                        "seeds": [run.seed],
                        "world_tubes_backward_policy": run.backward_policy,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def unexpected_execute(*_args, **_kwargs):
        pytest.fail("single.execute must not run for accepted reusable evidence")

    monkeypatch.setattr(matrix_runner.single, "execute", unexpected_execute)
    monkeypatch.setattr(
        matrix_runner.sys,
        "argv",
        [
            "run_unified_paper_matrix.py",
            "--execute",
            "--reuse-existing",
            "--matrix",
            str(matrix_path),
            "--out-dir",
            str(out_dir),
        ],
    )

    matrix_runner.main()

    result = json.loads((out_dir / "matrix_summary.json").read_text(encoding="utf-8"))
    assert result["status"] == "complete"
    assert result["run_count"] == 1
    assert result["runs"][0]["run"]["key"] == run.key


def test_bounded_execution_runs_one_new_row_without_claiming_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol_path = (
        DEFAULT_MATRIX.parent / "coffee_martini_protocol_smoke_2step.jsonc"
    )
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(
        json.dumps(
            {
                "name": "bounded_test",
                "runs": [
                    {
                        "role": "mechanical_smoke",
                        "protocol": str(protocol_path),
                        "seeds": [17, 29],
                        "world_tubes_backward_policy": "fast_exploration",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "outputs"
    executed_seeds: list[int] = []

    def fake_execute(*_args, seed: int, **_kwargs) -> dict:
        executed_seeds.append(seed)
        return {"status": "complete", "seed": seed}

    monkeypatch.setattr(matrix_runner.single, "execute", fake_execute)
    monkeypatch.setattr(
        matrix_runner,
        "validate_existing_summary",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        matrix_runner.sys,
        "argv",
        [
            "run_unified_paper_matrix.py",
            "--execute",
            "--reuse-existing",
            "--max-new-runs",
            "1",
            "--allow-dirty-source",
            "--matrix",
            str(matrix_path),
            "--out-dir",
            str(out_dir),
        ],
    )

    matrix_runner.main()

    progress = json.loads(
        (out_dir / "matrix_progress.json").read_text(encoding="utf-8")
    )
    assert executed_seeds == [17]
    assert progress["status"] == "partial"
    assert progress["accepted_run_count"] == 1
    assert progress["new_run_count"] == 1
    assert progress["missing_runs"] == [
        (
            "coffee_martini_protocol_smoke_2step/"
            "seed_29/fast_exploration"
        )
    ]
    assert not (out_dir / "matrix_summary.json").exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda summary: summary["lanes"]["world_tubes"]["evidence"]["cost"].__setitem__(
                "rasterized_pixels",
                1,
            ),
            "evidence does not match",
        ),
        (
            lambda summary: summary["lanes"]["dynamic_3dgs"]["evidence"][
                "cost"
            ].__setitem__("rasterized_frames", 1),
            "evidence does not match",
        ),
        (
            lambda summary: summary["manifest_validation"].__setitem__(
                "sample_id",
                "wrong-sample",
            ),
            "dataset identity drifted",
        ),
        (
            lambda summary: summary["manifest_validation"].__setitem__(
                "expected_pose_source",
                "neural_3d_llff_relative_pinhole",
            ),
            "dataset pose-source contract drifted",
        ),
        (
            lambda summary: summary["lanes"]["worldfoam"].__setitem__("wandb", None),
            "W&B provenance",
        ),
        (
            lambda summary: summary["lanes"]["world_tubes"]["wandb"].__setitem__(
                "run_id",
                " ",
            ),
            "W&B provenance",
        ),
        (
            lambda summary: summary["source"].__setitem__(
                "repository_commit",
                "not-a-sha",
            ),
            "invalid repository_commit",
        ),
        (
            lambda summary: summary["source"].__setitem__(
                "repository_dirty",
                0,
            ),
            "dirty source provenance",
        ),
    ),
)
def test_existing_evidence_rejects_incomplete_acceptance_contract(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    protocol_path = (
        DEFAULT_MATRIX.parent / "coffee_martini_protocol_smoke_2step.jsonc"
    )
    protocol = resolve_paper_training_protocol(load_config_file(protocol_path))
    run = MatrixRun(
        role="mechanical_smoke",
        protocol_path=protocol_path,
        seed=17,
        backward_policy="fast_exploration",
    )
    summary = copy.deepcopy(_summary(protocol))
    mutation(summary)
    run_dir = tmp_path / protocol.name / "seed_17"
    run_dir.mkdir(parents=True)
    _write_media(run_dir, steps=protocol.steps)
    _write_execution_identities(run_dir, run, summary)
    (run_dir / "run_summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        collect_existing_records([run], tmp_path)


@pytest.mark.parametrize(
    ("relative_path", "mode", "message"),
    (
        (
            Path("world_tubes_dynamic_3dgs")
            / "star_uvt_heldout_view0_side_by_side.mp4",
            "missing",
            "missing world_tubes media",
        ),
        (
            Path("worldfoam") / "side_by_side_step_0002.mp4",
            "empty",
            "missing worldfoam media",
        ),
    ),
)
def test_existing_evidence_rejects_missing_or_empty_media(
    tmp_path: Path,
    relative_path: Path,
    mode: str,
    message: str,
) -> None:
    protocol_path = (
        DEFAULT_MATRIX.parent / "coffee_martini_protocol_smoke_2step.jsonc"
    )
    run = MatrixRun(
        role="mechanical_smoke",
        protocol_path=protocol_path,
        seed=17,
        backward_policy="fast_exploration",
    )
    _write_existing_run(tmp_path, run)
    protocol = resolve_paper_training_protocol(load_config_file(protocol_path))
    media_path = tmp_path / protocol.name / "seed_17" / relative_path
    if mode == "missing":
        media_path.unlink()
    else:
        media_path.write_bytes(b"")

    with pytest.raises(ValueError, match=message):
        collect_existing_records([run], tmp_path)
