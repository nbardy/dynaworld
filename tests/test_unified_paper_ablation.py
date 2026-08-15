from __future__ import annotations

import hashlib
import json
import subprocess
from functools import lru_cache
from pathlib import Path

import pytest
import torch

from config_utils import load_config_file
from paper_training_protocol import (
    PaperRGBMetricAccumulator,
    paper_evaluator_contract,
    paper_runtime_source_tree_identity,
    resolve_paper_training_protocol,
)
from research_experiments.paper_runner_suite.run_unified_paper_ablation import (
    DEFAULT_PROTOCOL,
    FROZEN_WORLD_ACCEPTANCE,
    build_lane_evidence,
    build_dry_run_manifest,
    comparison_command,
    comparison_lane_commands,
    kernel_specs,
    load_final_powerfoam_metrics,
    local_mps_safety_estimate,
    materialize_isolated_comparison_report,
    merge_comparison_lane_reports,
    paper_camera_rig_init,
    paper_world_tubes_camera_policy,
    paper_scene_tag,
    powerfoam_config,
    require_execution_safety_acknowledgement,
    require_clean_provenance,
    source_provenance,
    validate_comparison_report,
    validate_comparison_pose_source,
    validate_frozen_world_evidence,
    validate_lane_cost,
    validate_lane_evidence,
    validate_manifest,
    validate_route_native_extension_identity,
    validate_wandb_identity,
    wandb_file_identity,
    worldfoam_resolved_config_binding,
    worldfoam_lane_command,
)


ROOT = Path(__file__).resolve().parents[1]
SMOKE_PROTOCOL = (
    ROOT
    / "src"
    / "train_configs"
    / "paper_protocols"
    / "coffee_martini_protocol_smoke_2step.jsonc"
)


def _protocol(path: Path = SMOKE_PROTOCOL):
    raw = load_config_file(path)
    return raw, resolve_paper_training_protocol(raw)


def _value_after(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def _sample_schedule(seed: int = 17) -> dict:
    return {
        "schema_version": 1,
        "algorithm": "spacetime_epoch_v1",
        "sampler_seed": seed + 7001,
        "record_count": 2,
        "sha256": "a" * 64,
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


def _world_state_digest(state: dict[str, torch.Tensor], metadata: dict) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(json.dumps(list(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _refresh_checkpoint_file_identity(frozen: dict) -> None:
    path = Path(frozen["checkpoint"]["path"])
    frozen["checkpoint"]["bytes"] = path.stat().st_size
    frozen["checkpoint"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_frozen_world_evidence(tmp_path: Path) -> dict:
    frame_indices = (0, 1, 2, 3)
    centered_frame_times = (-1.5, -0.5, 0.5, 1.5)
    metadata = {
        "representation": "legacy_tube",
        "frame_count": 4,
        "active_tube_count": 2,
        "tube_count": 2,
        "alpha_mode": "peak_splat",
        "amplitude_convention": "fiber_integrated",
        "min_precision_xy": 1.0e-4,
        "min_lambda_t": 1.0e-4,
        "parameter_names": [
            "x0",
            "velocity",
            "raw_precision_xy",
            "raw_lambda_t",
            "raw_opacity",
            "raw_color",
            "t0",
        ],
    }
    state = {
        "x0": torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=torch.float32,
        ),
        "velocity": torch.zeros((2, 3), dtype=torch.float32),
        "raw_precision_xy": torch.ones((2, 2), dtype=torch.float32),
        "raw_lambda_t": torch.ones((2,), dtype=torch.float32),
        "raw_opacity": torch.zeros((2,), dtype=torch.float32),
        "raw_color": torch.zeros((2, 3), dtype=torch.float32),
        "t0": torch.zeros((2,), dtype=torch.float32),
    }
    world_state_sha256 = _world_state_digest(state, metadata)
    checkpoint_path = tmp_path / "tiny_frozen_world.pt"
    torch.save(
        {
            "schema_version": 1,
            **metadata,
            "world_state_sha256": world_state_sha256,
            "state_dict": state,
        },
        checkpoint_path,
    )
    checkpoint = {
        "path": str(checkpoint_path),
        "parameter_tensor_count": len(metadata["parameter_names"]),
        "world_state_sha256": world_state_sha256,
        **metadata,
    }
    frozen = {
        "schema_version": 2,
        "status": "complete",
        "accepted": True,
        "scope": "tiny validator fixture",
        "checkpoint": checkpoint,
        "world_state": {
            "checkpoint_sha256": world_state_sha256,
            "before_routes_sha256": world_state_sha256,
            "after_replay_sha256": world_state_sha256,
            "after_compiled_sha256": world_state_sha256,
            "matches_checkpoint": True,
        },
        "heldout_camera": "cam06",
        "frame_count": 4,
        "full_dataset_frame_count": 4,
        "frame_indices": list(frame_indices),
        "centered_frame_times": list(centered_frame_times),
        "temporal_sampling": "ordered_full_interval_integer_lattice_v1",
        "image_size": [96, 128],
        "loss": {
            "name": "robust_l1",
            "replay": 1.0,
            "compiled": 1.0,
            "absolute_delta": 0.0,
        },
        "image": {"max_abs_error": 0.0, "mean_abs_error": 0.0},
        "gradient": {
            "global_normalized_l2_error": 0.0,
            "cosine_similarity": 1.0,
            "replay_l2_norm": 1.0,
            "compiled_l2_norm": 1.0,
            "parameter_tensor_count": len(metadata["parameter_names"]),
            "replay_gradient_tensor_count": len(metadata["parameter_names"]),
            "compiled_gradient_tensor_count": len(metadata["parameter_names"]),
            "gradient_coverage_matches": True,
            "replay_gradient_parameters": metadata["parameter_names"],
            "compiled_gradient_parameters": metadata["parameter_names"],
            "max_parameter_normalized_l2_error": 0.0,
            "per_parameter_normalized_l2_error": {
                name: 0.0 for name in metadata["parameter_names"]
            },
        },
        "timing_s": {
            "replay_total_forward": 0.4,
            "replay_total_backward": 0.8,
            "replay_per_frame_forward": 0.1,
            "replay_per_frame_backward": 0.2,
            "compiled_atlas_compile": 0.05,
            "compiled_total_forward": 0.4,
            "compiled_total_backward": 0.8,
            "compiled_per_frame_forward": 0.1,
            "compiled_per_frame_backward": 0.2,
            "parity_replay_total_forward": 0.4,
        },
        "payload_bytes": {
            "definition": "logical tensor bytes only",
            "topology_bytes_included": False,
            "storage_claim_eligible": False,
            "replay_cumulative_logical_tensor_bytes": 400,
            "compiled_trace_table_logical_tensor_bytes": 100,
            "compiled_to_replay_logical_volume_ratio": 0.25,
        },
        "atlas": {
            "trace_count": 2,
            "cell_count": 2,
            "interval_trace_entries": 4,
            "dense_trace_samples": 8,
            "interval_to_dense_trace_sample_ratio": 0.5,
            "fallback_cells": 0,
            "total_tile_samples": 8,
            "fallback_tile_samples": 0,
            "fallback_fraction": 0.0,
            "fallback_reasons": [],
        },
        "contract": {
            "same_checkpoint": True,
            "same_heldout_camera": True,
            "same_target_frames": True,
            "same_loss": True,
            "same_precision": True,
            "same_alpha_mode": True,
            "bounded_device_frame_residency": True,
            "host_target_storage": "eager_cpu_selected_frames",
            "resident_chunk_frames": 2,
            "timing_excludes_parity_replay": True,
        },
        "contract_hashes": {
            "target_frames_sha256": "a" * 64,
            "camera_program_sha256": "b" * 64,
            "frame_indices_sha256": hashlib.sha256(
                json.dumps(
                    list(frame_indices),
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "centered_frame_times_sha256": hashlib.sha256(
                json.dumps(
                    list(centered_frame_times),
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "evaluation_contract_sha256": "c" * 64,
        },
        "acceptance": dict(FROZEN_WORLD_ACCEPTANCE),
        "checks": {
            "checkpoint_matches": True,
            "image_matches": True,
            "loss_matches": True,
            "world_vjp_matches": True,
            "world_vjp_per_parameter_matches": True,
            "world_vjp_nonzero": True,
            "world_vjp_coverage_matches": True,
            "fallback_within_budget": True,
        },
    }
    _refresh_checkpoint_file_identity(frozen)
    return frozen


@lru_cache(maxsize=None)
def _runtime_source_tree(_path: Path) -> dict:
    return paper_runtime_source_tree_identity(
        ROOT
        / "third_party"
        / "fast-mac-gsplat"
        / "variants"
        / "star_uvt_v0"
        / "csrc"
        / "metal"
    )


@lru_cache(maxsize=1)
def _source_identity() -> dict:
    return source_provenance()


@lru_cache(maxsize=None)
def _manifest_validation(protocol) -> dict:
    return validate_manifest(protocol)


def _manifest_input_identity(protocol) -> dict:
    return _manifest_validation(protocol)["input_identity"]


def _isolated_reports(
    *,
    uvt_world_representation: str = "legacy_tube",
    uvt_alpha_mode: str = "peak_splat",
    uvt_render_backend: str = "metal_tile",
    uvt_amplitude_convention: str = "fiber_integrated",
    uvt_retained_depth_samples: int = 48,
    uvt_retained_sigma_extent: float = 6.0,
    uvt_order_certificate_sigma: float = 6.0,
    uvt_order_certificate_min_gap: float = 0.0,
    frozen_world_replay_compiled: bool = False,
    frozen_world_max_frames: int = 0,
    device: str = "mps",
) -> dict:
    if uvt_alpha_mode == "peak_splat":
        opacity_semantics = "peak_alpha_amplitude"
    elif uvt_amplitude_convention == "fiber_integrated":
        opacity_semantics = (
            "nonnegative_fiber_integrated_peak_optical_thickness"
        )
    else:
        opacity_semantics = "nonnegative_world_peak_extinction_density"
    native_path = Path(__file__).resolve()
    meta = {
        "baseline_config": "baseline.jsonc",
        "target_size": [96, 128],
        "image_size": [96, 128],
        "max_frames": 4,
        "frame_count": 4,
        "train_seconds": 1.0,
        "device": device,
        "seed": 17,
        "train_cameras": ["cam04", "cam09"],
        "heldout_cameras": ["cam06"],
        "pose_source": "neural_3d_llff_opencv_relative_pinhole_v2",
        "uvt_world_representation": uvt_world_representation,
        "uvt_alpha_mode": uvt_alpha_mode,
        "uvt_render_backend": uvt_render_backend,
        "uvt_amplitude_convention": uvt_amplitude_convention,
        "uvt_opacity_semantics": opacity_semantics,
        "uvt_retained_depth_samples": uvt_retained_depth_samples,
        "uvt_retained_sigma_extent": uvt_retained_sigma_extent,
        "uvt_order_certificate_sigma": uvt_order_certificate_sigma,
        "uvt_order_certificate_min_gap": uvt_order_certificate_min_gap,
        "uvt_camera_projection": "dataset_lens",
        "uvt_camera_sequence_mode": "static_view",
        "uvt_segment_frames": 4,
        "uvt_backward_policy": (
            None
            if uvt_render_backend == "retained_fiber_metal"
            else {"name": "fast_exploration"}
        ),
        "splat_camera_projection": "dataset_lens",
        "eval_chunk_frames": 2,
        "eval_media_max_frames": 32,
        "star_uvt_native_extension": {
            "module": "test.star_uvt._C",
            "path": str(native_path),
            "sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
            "bytes": native_path.stat().st_size,
            "source_tree_sha256": "d" * 64,
            "source_file_count": 1,
            "runtime_source_tree": _runtime_source_tree(native_path),
        },
        "route_native_extension": {
            "module": "test._C",
            "path": str(native_path),
            "sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
            "bytes": native_path.stat().st_size,
            "runtime_source_tree": _runtime_source_tree(native_path),
        },
        "paper_dataset_bundle": _hashed_contract(
            1,
            sample_id="smoke",
            train_frames_sha256="1" * 64,
            heldout_frames_sha256="2" * 64,
            camera_program_sha256="3" * 64,
            pose_source="neural_3d_llff_opencv_relative_pinhole_v2",
        ),
        "paper_evaluator": paper_evaluator_contract(),
        "paper_runtime": _hashed_contract(
            1,
            os="test",
            hardware="test",
            python="test",
            torch="test",
        ),
        "frozen_world_replay_compiled": frozen_world_replay_compiled,
        "frozen_world_max_frames": frozen_world_max_frames,
    }
    frozen_report = (
        {
            "schema_version": 2,
            "status": "complete",
            "accepted": True,
            "frame_count": (
                4 if frozen_world_max_frames <= 0 else frozen_world_max_frames
            ),
            "checkpoint": {},
            "loss": {},
            "image": {},
            "gradient": {},
            "timing_s": {},
            "payload_bytes": {},
            "atlas": {},
            "contract": {"same_checkpoint": True},
            "acceptance": {},
            "checks": {},
        }
        if frozen_world_replay_compiled
        else None
    )
    return {
        "world_tubes": {
            "meta": {**meta, "only_lane": "world_tubes"},
            "star_uvt": {
                "lane": "world_tubes",
                "frozen_world_replay_compiled": frozen_report,
                "paper_protocol": {
                    "sample_schedule": _sample_schedule(meta["seed"]),
                },
            },
            "star_uvt_selected": {"checkpoint": "final"},
            "free_dynamic_splats": None,
        },
        "dynamic_3dgs": {
            "meta": {
                **meta,
                "only_lane": "dynamic_3dgs",
                "frozen_world_replay_compiled": False,
                "frozen_world_max_frames": 0,
            },
            "star_uvt": None,
            "star_uvt_selected": None,
            "free_dynamic_splats": {
                "lane": "dynamic_3dgs",
                "paper_protocol": {
                    "sample_schedule": _sample_schedule(meta["seed"]),
                },
            },
        },
    }


def _write_isolated_reports(
    comparison_dir: Path,
    protocol,
    **kwargs,
) -> dict:
    reports = _isolated_reports(**kwargs)
    command_keys = {
        "uvt_world_representation",
        "uvt_alpha_mode",
        "uvt_render_backend",
        "uvt_amplitude_convention",
        "uvt_retained_depth_samples",
        "uvt_retained_sigma_extent",
        "uvt_order_certificate_sigma",
        "uvt_order_certificate_min_gap",
        "frozen_world_replay_compiled",
        "frozen_world_max_frames",
    }
    commands = comparison_lane_commands(
        SMOKE_PROTOCOL,
        protocol,
        17,
        comparison_dir,
        backward_policy="fast_exploration",
        device=str(kwargs.get("device", "mps")),
        python="python",
        **{key: value for key, value in kwargs.items() if key in command_keys},
    )
    dataset_identity = _manifest_input_identity(protocol)
    source = _source_identity()
    protocol_sha256 = hashlib.sha256(SMOKE_PROTOCOL.read_bytes()).hexdigest()
    for lane_name, report in reports.items():
        lane_dir = comparison_dir / lane_name
        lane_dir.mkdir(parents=True)
        report_path = lane_dir / "comparison_report.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        (lane_dir / "execution_identity.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "lane": lane_name,
                    "source_start": source,
                    "source_finish": source,
                    "protocol_sha256": protocol_sha256,
                    "command": commands[lane_name],
                    "dataset_input_identity": dataset_identity,
                    "comparison_report_sha256": hashlib.sha256(
                        report_path.read_bytes()
                    ).hexdigest(),
                }
            ),
            encoding="utf-8",
        )
    return reports


def test_unified_command_selects_the_practical_metal_lanes(tmp_path: Path) -> None:
    _, protocol = _protocol()
    command = comparison_command(
        SMOKE_PROTOCOL,
        protocol,
        29,
        tmp_path / "compare",
        backward_policy="fast_exploration",
        device="mps",
        python="python",
    )

    assert _value_after(command, "--paper-protocol") == str(SMOKE_PROTOCOL)
    assert _value_after(command, "--uvt-loss-scope") == "paper_batch"
    assert _value_after(command, "--uvt-backward-policy") == "fast_exploration"
    assert _value_after(command, "--uvt-world-representation") == "legacy_tube"
    assert _value_after(command, "--uvt-alpha-mode") == "peak_splat"
    assert _value_after(command, "--uvt-render-backend") == "metal_tile"
    assert _value_after(command, "--uvt-amplitude-convention") == "fiber_integrated"
    assert _value_after(command, "--uvt-retained-depth-samples") == "48"
    assert _value_after(command, "--uvt-retained-sigma-extent") == "6.0"
    assert _value_after(command, "--uvt-order-certificate-sigma") == "6.0"
    assert _value_after(command, "--uvt-order-certificate-min-gap") == "0.0"
    assert _value_after(command, "--splat-renderer") == "fast_mac"
    assert _value_after(command, "--max-frames") == "4"
    assert _value_after(command, "--max-steps") == "2"
    assert _value_after(command, "--uvt-tubes") == "256"
    assert _value_after(command, "--splat-count") == "256"
    assert _value_after(command, "--eval-chunk-frames") == "2"
    assert _value_after(command, "--eval-media-max-frames") == "32"
    assert _value_after(command, "--only-lane") == "combined"
    assert "--allow-paper-local-mps-execution" not in command


def test_frozen_world_command_is_world_tubes_only_and_fail_closed(
    tmp_path: Path,
) -> None:
    _, protocol = _protocol()
    commands = comparison_lane_commands(
        SMOKE_PROTOCOL,
        protocol,
        17,
        tmp_path / "compare",
        backward_policy="fast_exploration",
        device="mps",
        frozen_world_replay_compiled=True,
        frozen_world_max_frames=4,
        python="python",
    )

    assert "--frozen-world-replay-compiled" in commands["world_tubes"]
    assert _value_after(
        commands["world_tubes"],
        "--frozen-world-max-frames",
    ) == "4"
    assert "--frozen-world-replay-compiled" not in commands["dynamic_3dgs"]

    with pytest.raises(ValueError, match="legacy_tube"):
        comparison_command(
            SMOKE_PROTOCOL,
            protocol,
            17,
            tmp_path / "spd4",
            backward_policy="fast_exploration",
            device="mps",
            uvt_world_representation="full_spd4",
            frozen_world_replay_compiled=True,
            python="python",
        )
    with pytest.raises(ValueError, match="requires frozen_world_replay_compiled"):
        comparison_command(
            SMOKE_PROTOCOL,
            protocol,
            17,
            tmp_path / "frames-only",
            backward_policy="fast_exploration",
            device="mps",
            frozen_world_max_frames=4,
            python="python",
        )


def _validate_tiny_frozen_world(frozen: dict) -> None:
    validate_frozen_world_evidence(
        frozen,
        expected_frames=4,
        expected_full_frames=4,
        expected_image_size=(96, 128),
        expected_heldout_camera="cam06",
        expected_active_tubes=2,
    )


def test_frozen_world_evidence_binds_checkpoint_semantics(tmp_path: Path) -> None:
    frozen = _valid_frozen_world_evidence(tmp_path)

    _validate_tiny_frozen_world(frozen)

    frozen["checkpoint"]["min_precision_xy"] = 2.0e-4
    with pytest.raises(ValueError, match="metadata does not match report"):
        _validate_tiny_frozen_world(frozen)


def test_frozen_world_evidence_recomputes_checkpoint_world_state(
    tmp_path: Path,
) -> None:
    frozen = _valid_frozen_world_evidence(tmp_path)
    checkpoint_path = Path(frozen["checkpoint"]["path"])
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    payload["state_dict"]["x0"][0, 0] += 1.0
    torch.save(payload, checkpoint_path)
    _refresh_checkpoint_file_identity(frozen)

    with pytest.raises(ValueError, match="SHA-256 does not match contents"):
        _validate_tiny_frozen_world(frozen)


def test_frozen_world_evidence_rejects_checkpoint_payload_drift(
    tmp_path: Path,
) -> None:
    frozen = _valid_frozen_world_evidence(tmp_path)
    checkpoint_path = Path(frozen["checkpoint"]["path"])
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    payload["schema_version"] = 2
    torch.save(payload, checkpoint_path)
    _refresh_checkpoint_file_identity(frozen)

    with pytest.raises(ValueError, match="payload schema is stale"):
        _validate_tiny_frozen_world(frozen)


def test_frozen_world_evidence_rejects_checkpoint_tensor_schema_drift(
    tmp_path: Path,
) -> None:
    frozen = _valid_frozen_world_evidence(tmp_path)
    checkpoint_path = Path(frozen["checkpoint"]["path"])
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    payload["state_dict"]["x0"] = payload["state_dict"]["x0"][:, :2]
    torch.save(payload, checkpoint_path)
    _refresh_checkpoint_file_identity(frozen)

    with pytest.raises(ValueError, match="tensor schema is invalid"):
        _validate_tiny_frozen_world(frozen)


def test_frozen_world_evidence_validates_atlas_ratios_and_counts(
    tmp_path: Path,
) -> None:
    frozen = _valid_frozen_world_evidence(tmp_path)
    frozen["atlas"]["interval_to_dense_trace_sample_ratio"] = 0.25
    with pytest.raises(ValueError, match="trace ratio is inconsistent"):
        _validate_tiny_frozen_world(frozen)

    frozen = _valid_frozen_world_evidence(tmp_path)
    frozen["atlas"]["interval_trace_entries"] = 9
    with pytest.raises(ValueError, match="structural counts"):
        _validate_tiny_frozen_world(frozen)


def test_frozen_world_evidence_binds_full_interval_time_grid(
    tmp_path: Path,
) -> None:
    frozen = _valid_frozen_world_evidence(tmp_path)
    frozen["frame_indices"][1] = 2

    with pytest.raises(ValueError, match="fixed-program time grid"):
        _validate_tiny_frozen_world(frozen)


def test_beer_lambert_command_is_scoped_to_full_spd4_direct_atomic(
    tmp_path: Path,
) -> None:
    _, protocol = _protocol()
    command = comparison_command(
        SMOKE_PROTOCOL,
        protocol,
        29,
        tmp_path / "compare",
        backward_policy="fast_exploration",
        device="mps",
        uvt_world_representation="full_spd4",
        uvt_alpha_mode="beer_lambert",
        python="python",
    )

    assert _value_after(command, "--uvt-world-representation") == "full_spd4"
    assert _value_after(command, "--uvt-alpha-mode") == "beer_lambert"
    assert _value_after(command, "--uvt-backward-policy") == "fast_exploration"
    assert _value_after(command, "--uvt-camera-sequence-mode") == "static_view"

    with pytest.raises(ValueError, match="scoped to full_spd4"):
        comparison_command(
            SMOKE_PROTOCOL,
            protocol,
            29,
            tmp_path / "legacy",
            backward_policy="fast_exploration",
            device="mps",
            uvt_alpha_mode="beer_lambert",
            python="python",
        )
    with pytest.raises(ValueError, match="direct_atomic\\+index_add"):
        comparison_command(
            SMOKE_PROTOCOL,
            protocol,
            29,
            tmp_path / "deterministic",
            backward_policy="deterministic_quality",
            device="mps",
            uvt_world_representation="full_spd4",
            uvt_alpha_mode="beer_lambert",
            python="python",
        )


@pytest.mark.parametrize(
    ("backend", "expected_policy"),
    (
        ("retained_fiber_metal", "manual"),
        ("hybrid_retained_fiber", "fast_exploration"),
    ),
)
def test_physical_backend_command_threads_quadrature_and_certificate_axes(
    tmp_path: Path,
    backend: str,
    expected_policy: str,
) -> None:
    _, protocol = _protocol()
    command = comparison_command(
        SMOKE_PROTOCOL,
        protocol,
        29,
        tmp_path / backend,
        backward_policy="fast_exploration",
        device="mps",
        uvt_world_representation="full_spd4",
        uvt_alpha_mode="beer_lambert",
        uvt_render_backend=backend,
        uvt_amplitude_convention="peak_density",
        uvt_retained_depth_samples=32,
        uvt_retained_sigma_extent=4.5,
        uvt_order_certificate_sigma=5.5,
        uvt_order_certificate_min_gap=0.125,
        python="python",
    )

    assert _value_after(command, "--uvt-render-backend") == backend
    assert _value_after(command, "--uvt-amplitude-convention") == "peak_density"
    assert _value_after(command, "--uvt-retained-depth-samples") == "32"
    assert _value_after(command, "--uvt-retained-sigma-extent") == "4.5"
    assert _value_after(command, "--uvt-order-certificate-sigma") == "5.5"
    assert _value_after(command, "--uvt-order-certificate-min-gap") == "0.125"
    assert _value_after(command, "--uvt-backward-policy") == expected_policy


def test_physical_backend_contract_rejects_unsupported_combinations(
    tmp_path: Path,
) -> None:
    _, protocol = _protocol()
    common = {
        "backward_policy": "fast_exploration",
        "uvt_world_representation": "full_spd4",
        "uvt_alpha_mode": "beer_lambert",
        "python": "python",
    }

    with pytest.raises(ValueError, match="requires local MPS"):
        comparison_command(
            SMOKE_PROTOCOL,
            protocol,
            29,
            tmp_path / "cpu",
            device="cpu",
            uvt_render_backend="retained_fiber_metal",
            **common,
        )
    with pytest.raises(ValueError, match="requires beer_lambert"):
        comparison_command(
            SMOKE_PROTOCOL,
            protocol,
            29,
            tmp_path / "peak",
            backward_policy="fast_exploration",
            device="mps",
            uvt_world_representation="full_spd4",
            uvt_alpha_mode="peak_splat",
            uvt_render_backend="hybrid_retained_fiber",
            python="python",
        )
    with pytest.raises(ValueError, match="direct_atomic\\+index_add"):
        comparison_command(
            SMOKE_PROTOCOL,
            protocol,
            29,
            tmp_path / "deterministic-hybrid",
            backward_policy="deterministic_quality",
            device="mps",
            uvt_world_representation="full_spd4",
            uvt_alpha_mode="beer_lambert",
            uvt_render_backend="hybrid_retained_fiber",
            python="python",
        )
    with pytest.raises(ValueError, match="peak_density requires beer_lambert"):
        comparison_command(
            SMOKE_PROTOCOL,
            protocol,
            29,
            tmp_path / "density-alpha",
            backward_policy="fast_exploration",
            device="mps",
            uvt_world_representation="full_spd4",
            uvt_alpha_mode="peak_splat",
            uvt_amplitude_convention="peak_density",
            python="python",
        )
    with pytest.raises(ValueError, match="at least uvt_retained_sigma_extent"):
        comparison_command(
            SMOKE_PROTOCOL,
            protocol,
            29,
            tmp_path / "certificate",
            device="mps",
            uvt_order_certificate_sigma=3.0,
            uvt_retained_sigma_extent=4.0,
            **common,
        )


def test_unified_commands_isolate_each_allocator_and_worldfoam_inherits_device(tmp_path: Path) -> None:
    _, protocol = _protocol()
    compare_dir = tmp_path / "compare"
    commands = comparison_lane_commands(
        SMOKE_PROTOCOL,
        protocol,
        29,
        compare_dir,
        backward_policy="fast_exploration",
        device="cpu",
        uvt_world_representation="full_spd4",
        uvt_alpha_mode="beer_lambert",
        uvt_spd4_init_precision_z=1.0e6,
        python="python",
    )

    assert set(commands) == {"world_tubes", "dynamic_3dgs"}
    for lane_name, command in commands.items():
        assert _value_after(command, "--only-lane") == lane_name
        assert _value_after(command, "--out-dir") == str(compare_dir / lane_name)
        assert _value_after(command, "--device") == "cpu"
        assert _value_after(command, "--uvt-world-representation") == "full_spd4"
        assert _value_after(command, "--uvt-alpha-mode") == "beer_lambert"
        assert _value_after(command, "--uvt-spd4-init-precision-z") == "1000000.0"

    worldfoam = worldfoam_lane_command(
        SMOKE_PROTOCOL,
        29,
        tmp_path / "worldfoam",
        device="cpu",
        wandb_mode="offline",
        python="python",
    )
    assert "--execute" in worldfoam
    assert _value_after(worldfoam, "--device") == "cpu"
    assert "--allow-local-mps-execution" not in worldfoam

    approved = comparison_command(
        SMOKE_PROTOCOL,
        protocol,
        29,
        compare_dir,
        backward_policy="fast_exploration",
        device="mps",
        only_lane="world_tubes",
        allow_local_mps_execution=True,
        python="python",
    )
    assert "--allow-paper-local-mps-execution" in approved


def test_isolated_comparison_reports_merge_only_when_metadata_matches() -> None:
    reports = _isolated_reports()

    merged = merge_comparison_lane_reports(reports)
    assert merged["star_uvt"]["lane"] == "world_tubes"
    assert merged["free_dynamic_splats"]["lane"] == "dynamic_3dgs"
    assert merged["meta"]["execution_model"] == "one_child_process_per_representation"
    assert merged["meta"]["uvt_world_representation"] == "legacy_tube"
    assert merged["meta"]["uvt_alpha_mode"] == "peak_splat"
    assert merged["meta"]["uvt_render_backend"] == "metal_tile"
    assert merged["meta"]["uvt_amplitude_convention"] == "fiber_integrated"
    assert merged["meta"]["uvt_opacity_semantics"] == "peak_alpha_amplitude"
    assert merged["meta"]["uvt_retained_depth_samples"] == 48
    assert merged["meta"]["uvt_retained_sigma_extent"] == 6.0
    assert merged["meta"]["uvt_order_certificate_sigma"] == 6.0
    assert merged["meta"]["uvt_order_certificate_min_gap"] == 0.0

    reports["dynamic_3dgs"]["meta"]["seed"] = 29
    with pytest.raises(ValueError, match="metadata drifted: seed"):
        merge_comparison_lane_reports(reports)

    reports = _isolated_reports()
    del reports["world_tubes"]["meta"]["star_uvt_native_extension"]
    with pytest.raises(ValueError, match="native extension identity is missing"):
        merge_comparison_lane_reports(reports)

    reports = _isolated_reports()
    reports["dynamic_3dgs"]["free_dynamic_splats"]["paper_protocol"][
        "sample_schedule"
    ]["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="same sample schedule"):
        merge_comparison_lane_reports(reports)

    reports = _isolated_reports()
    reports["dynamic_3dgs"]["meta"]["paper_dataset_bundle"] = _hashed_contract(
        1,
        sample_id="mutated",
    )
    with pytest.raises(ValueError, match="paper_dataset_bundle"):
        merge_comparison_lane_reports(reports)

    reports = _isolated_reports()
    reports["dynamic_3dgs"]["meta"]["paper_runtime"] = _hashed_contract(
        1,
        os="different-host",
    )
    with pytest.raises(ValueError, match="paper_runtime"):
        merge_comparison_lane_reports(reports)


def test_isolated_comparison_merge_rejects_representation_drift_and_normalizes_old_legacy_reports() -> None:
    reports = _isolated_reports()
    reports["dynamic_3dgs"]["meta"]["uvt_world_representation"] = "full_spd4"
    with pytest.raises(ValueError, match="metadata drifted: uvt_world_representation"):
        merge_comparison_lane_reports(reports)

    old_reports = _isolated_reports()
    for report in old_reports.values():
        for key in (
            "uvt_world_representation",
            "uvt_alpha_mode",
            "uvt_render_backend",
            "uvt_amplitude_convention",
            "uvt_opacity_semantics",
            "uvt_retained_depth_samples",
            "uvt_retained_sigma_extent",
            "uvt_order_certificate_sigma",
            "uvt_order_certificate_min_gap",
        ):
            del report["meta"][key]
    merged = merge_comparison_lane_reports(old_reports)
    assert merged["meta"]["uvt_world_representation"] == "legacy_tube"
    assert merged["meta"]["uvt_alpha_mode"] == "peak_splat"
    assert merged["meta"]["uvt_render_backend"] == "metal_tile"
    assert merged["meta"]["uvt_amplitude_convention"] == "fiber_integrated"
    assert merged["meta"]["uvt_opacity_semantics"] == "peak_alpha_amplitude"


def test_isolated_comparison_merge_rejects_alpha_semantics_drift() -> None:
    reports = _isolated_reports(uvt_alpha_mode="beer_lambert")
    reports["dynamic_3dgs"]["meta"]["uvt_opacity_semantics"] = (
        "peak_alpha_amplitude"
    )

    with pytest.raises(ValueError, match="metadata drifted: uvt_opacity_semantics"):
        merge_comparison_lane_reports(reports)


def test_isolated_comparison_merge_rejects_physical_backend_drift() -> None:
    reports = _isolated_reports()
    reports["dynamic_3dgs"]["meta"]["uvt_render_backend"] = (
        "hybrid_retained_fiber"
    )

    with pytest.raises(ValueError, match="metadata drifted: uvt_render_backend"):
        merge_comparison_lane_reports(reports)


def test_comparison_report_validation_rejects_a_stale_world_representation() -> None:
    _, protocol = _protocol()
    report = {
        "meta": {
            "train_cameras": list(protocol.dataset.train_cameras),
            "heldout_cameras": list(protocol.dataset.heldout_cameras),
            "frame_count": protocol.dataset.frame_count,
            "uvt_backward_policy": {"name": "fast_exploration"},
            "uvt_world_representation": "legacy_tube",
            "uvt_alpha_mode": "peak_splat",
            "uvt_opacity_semantics": "peak_alpha_amplitude",
        }
    }

    with pytest.raises(ValueError, match="World Tubes representation drifted"):
        validate_comparison_report(
            report,
            protocol,
            backward_policy="fast_exploration",
            manifest_validation=_manifest_validation(protocol),
            uvt_world_representation="full_spd4",
        )


def test_comparison_report_rejects_hash_consistent_legacy_neural3d_pose_source() -> None:
    _, protocol = _protocol()
    reports = _isolated_reports()
    legacy_pose_source = "neural_3d_llff_relative_pinhole"
    for report in reports.values():
        meta = report["meta"]
        meta["pose_source"] = legacy_pose_source
        bundle = {
            key: value
            for key, value in meta["paper_dataset_bundle"].items()
            if key not in {"schema_version", "sha256", "pose_source"}
        }
        meta["paper_dataset_bundle"] = _hashed_contract(
            1,
            **bundle,
            pose_source=legacy_pose_source,
        )
    merged = merge_comparison_lane_reports(reports)

    with pytest.raises(ValueError, match="pose source does not match"):
        validate_comparison_report(
            merged,
            protocol,
            backward_policy="fast_exploration",
            manifest_validation=_manifest_validation(protocol),
        )


def test_comparison_pose_source_contract_preserves_dnerf() -> None:
    pose_source = "dnerf_matched_time_blender_to_opencv_relative_pinhole"
    meta = {
        "pose_source": pose_source,
        "paper_dataset_bundle": _hashed_contract(
            1,
            pose_source=pose_source,
        ),
    }
    manifest_validation = {
        "dataset": "dnerf",
        "expected_pose_source": pose_source,
        "input_identity": _hashed_contract(
            1,
            dataset="dnerf",
            files=[],
        ),
    }

    assert (
        validate_comparison_pose_source(meta, manifest_validation)
        == pose_source
    )


def test_comparison_report_validation_rejects_stale_alpha_semantics() -> None:
    _, protocol = _protocol()
    report = {
        "meta": {
            "train_cameras": list(protocol.dataset.train_cameras),
            "heldout_cameras": list(protocol.dataset.heldout_cameras),
            "frame_count": protocol.dataset.frame_count,
            "uvt_backward_policy": {"name": "fast_exploration"},
            "uvt_world_representation": "full_spd4",
            "uvt_alpha_mode": "beer_lambert",
            "uvt_opacity_semantics": "peak_alpha_amplitude",
        }
    }

    with pytest.raises(ValueError, match="opacity semantics drifted"):
        validate_comparison_report(
            report,
            protocol,
            backward_policy="fast_exploration",
            manifest_validation=_manifest_validation(protocol),
            uvt_world_representation="full_spd4",
            uvt_alpha_mode="beer_lambert",
        )


def test_comparison_report_validation_rejects_spd4_initialization_drift() -> None:
    _, protocol = _protocol()
    report = {
        "meta": {
            "train_cameras": list(protocol.dataset.train_cameras),
            "heldout_cameras": list(protocol.dataset.heldout_cameras),
            "frame_count": protocol.dataset.frame_count,
            "uvt_backward_policy": {"name": "fast_exploration"},
            "uvt_world_representation": "full_spd4",
            "uvt_alpha_mode": "peak_splat",
            "uvt_opacity_semantics": "peak_alpha_amplitude",
            "uvt_spd4_init_precision_z": 30.0,
        }
    }

    with pytest.raises(ValueError, match="depth initialization drifted"):
        validate_comparison_report(
            report,
            protocol,
            backward_policy="fast_exploration",
            manifest_validation=_manifest_validation(protocol),
            uvt_world_representation="full_spd4",
            uvt_spd4_init_precision_z=1.0e6,
        )


def test_comparison_report_validation_rejects_stale_physical_renderer_axes() -> None:
    _, protocol = _protocol()
    report = {
        "meta": {
            "device": "mps",
            "train_cameras": list(protocol.dataset.train_cameras),
            "heldout_cameras": list(protocol.dataset.heldout_cameras),
            "frame_count": protocol.dataset.frame_count,
            "uvt_backward_policy": {"name": "fast_exploration"},
            "uvt_world_representation": "full_spd4",
            "uvt_alpha_mode": "beer_lambert",
            "uvt_render_backend": "metal_tile",
            "uvt_amplitude_convention": "fiber_integrated",
            "uvt_opacity_semantics": (
                "nonnegative_fiber_integrated_peak_optical_thickness"
            ),
            "uvt_retained_depth_samples": 48,
            "uvt_retained_sigma_extent": 6.0,
            "uvt_order_certificate_sigma": 6.0,
            "uvt_order_certificate_min_gap": 0.0,
        }
    }

    with pytest.raises(ValueError, match="render backend drifted"):
        validate_comparison_report(
            report,
            protocol,
            backward_policy="fast_exploration",
            manifest_validation=_manifest_validation(protocol),
            uvt_world_representation="full_spd4",
            uvt_alpha_mode="beer_lambert",
            uvt_render_backend="hybrid_retained_fiber",
        )

    report["meta"]["uvt_render_backend"] = "hybrid_retained_fiber"
    with pytest.raises(ValueError, match="uvt_retained_depth_samples drifted"):
        validate_comparison_report(
            report,
            protocol,
            backward_policy="fast_exploration",
            manifest_validation=_manifest_validation(protocol),
            uvt_world_representation="full_spd4",
            uvt_alpha_mode="beer_lambert",
            uvt_render_backend="hybrid_retained_fiber",
            uvt_retained_depth_samples=32,
        )


def test_isolated_comparison_resume_reuses_completed_lane_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, protocol = _protocol()
    comparison_dir = tmp_path / "compare"
    _write_isolated_reports(comparison_dir, protocol)

    def unexpected_run(*_args, **_kwargs):
        raise AssertionError("completed isolated lane must not be relaunched")

    monkeypatch.setattr("subprocess.run", unexpected_run)
    report_path = materialize_isolated_comparison_report(
        SMOKE_PROTOCOL,
        protocol,
        17,
        comparison_dir,
        backward_policy="fast_exploration",
        device="mps",
        reuse_existing=True,
        expected_source=_source_identity(),
        python="python",
    )

    merged = json.loads(report_path.read_text(encoding="utf-8"))
    assert merged["meta"]["only_lane"] == "isolated_merged"
    assert merged["star_uvt"]["lane"] == "world_tubes"
    assert merged["free_dynamic_splats"]["lane"] == "dynamic_3dgs"


def test_isolated_comparison_materializer_requires_source_identity(
    tmp_path: Path,
) -> None:
    _, protocol = _protocol()

    with pytest.raises(ValueError, match="requires an expected source identity"):
        materialize_isolated_comparison_report(
            SMOKE_PROTOCOL,
            protocol,
            17,
            tmp_path / "compare",
            backward_policy="fast_exploration",
            device="mps",
            reuse_existing=False,
            expected_source={},
            python="python",
        )


def test_isolated_comparison_resume_preserves_frozen_world_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, protocol = _protocol()
    comparison_dir = tmp_path / "compare"
    _write_isolated_reports(
        comparison_dir,
        protocol,
        frozen_world_replay_compiled=True,
        frozen_world_max_frames=4,
    )

    def unexpected_run(*_args, **_kwargs):
        raise AssertionError("matching frozen-world lanes must not be relaunched")

    monkeypatch.setattr("subprocess.run", unexpected_run)
    report_path = materialize_isolated_comparison_report(
        SMOKE_PROTOCOL,
        protocol,
        17,
        comparison_dir,
        backward_policy="fast_exploration",
        device="mps",
        reuse_existing=True,
        expected_source=_source_identity(),
        frozen_world_replay_compiled=True,
        frozen_world_max_frames=4,
        python="python",
    )

    merged = json.loads(report_path.read_text(encoding="utf-8"))
    assert merged["meta"]["frozen_world_replay_compiled"] is True
    assert (
        merged["star_uvt"]["frozen_world_replay_compiled"]["contract"][
            "same_checkpoint"
        ]
        is True
    )


def test_isolated_comparison_materializer_threads_full_spd4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, protocol = _protocol()
    comparison_dir = tmp_path / "compare"
    _write_isolated_reports(
        comparison_dir,
        protocol,
        uvt_world_representation="full_spd4",
        uvt_alpha_mode="beer_lambert",
    )

    def unexpected_run(*_args, **_kwargs):
        raise AssertionError("completed isolated lane must not be relaunched")

    monkeypatch.setattr("subprocess.run", unexpected_run)
    report_path = materialize_isolated_comparison_report(
        SMOKE_PROTOCOL,
        protocol,
        17,
        comparison_dir,
        backward_policy="fast_exploration",
        device="mps",
        reuse_existing=True,
        expected_source=_source_identity(),
        uvt_world_representation="full_spd4",
        uvt_alpha_mode="beer_lambert",
        python="python",
    )

    merged = json.loads(report_path.read_text(encoding="utf-8"))
    assert merged["meta"]["uvt_world_representation"] == "full_spd4"
    assert merged["meta"]["uvt_alpha_mode"] == "beer_lambert"


def test_isolated_materializer_reuses_matching_retained_fiber_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, protocol = _protocol()
    comparison_dir = tmp_path / "compare"
    _write_isolated_reports(
        comparison_dir,
        protocol,
        uvt_world_representation="full_spd4",
        uvt_alpha_mode="beer_lambert",
        uvt_render_backend="retained_fiber_metal",
        uvt_amplitude_convention="peak_density",
        uvt_retained_depth_samples=24,
        uvt_retained_sigma_extent=4.0,
        uvt_order_certificate_sigma=5.0,
        uvt_order_certificate_min_gap=0.25,
    )

    def unexpected_run(*_args, **_kwargs):
        raise AssertionError("matching retained-fiber lanes must be reused")

    monkeypatch.setattr("subprocess.run", unexpected_run)
    report_path = materialize_isolated_comparison_report(
        SMOKE_PROTOCOL,
        protocol,
        17,
        comparison_dir,
        backward_policy="fast_exploration",
        device="mps",
        reuse_existing=True,
        expected_source=_source_identity(),
        uvt_world_representation="full_spd4",
        uvt_alpha_mode="beer_lambert",
        uvt_render_backend="retained_fiber_metal",
        uvt_amplitude_convention="peak_density",
        uvt_retained_depth_samples=24,
        uvt_retained_sigma_extent=4.0,
        uvt_order_certificate_sigma=5.0,
        uvt_order_certificate_min_gap=0.25,
        python="python",
    )

    merged = json.loads(report_path.read_text(encoding="utf-8"))
    assert merged["meta"]["uvt_render_backend"] == "retained_fiber_metal"
    assert merged["meta"]["uvt_amplitude_convention"] == "peak_density"
    assert merged["meta"]["uvt_opacity_semantics"] == (
        "nonnegative_world_peak_extinction_density"
    )


def test_isolated_materializer_does_not_reuse_stale_backend_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, protocol = _protocol()
    comparison_dir = tmp_path / "compare"
    _write_isolated_reports(
        comparison_dir,
        protocol,
        uvt_world_representation="full_spd4",
        uvt_alpha_mode="beer_lambert",
    )

    launched: list[list[str]] = []
    original_run = subprocess.run

    def record_relaunch(command, **kwargs):
        # subprocess.check_output implements itself through subprocess.run.
        # Preserve the real read-only git provenance probes and intercept only
        # the trainer child process this behavior test is about.
        if command and command[0] in {"git", "vm_stat", "sysctl"}:
            return original_run(command, **kwargs)
        launched.append(command)
        raise RuntimeError("stale backend relaunch")

    # This test protects stale-report invalidation, not the independent live
    # host resource gate.  Keep it deterministic in restricted test runners
    # where macOS denies vm.swapusage to child processes.
    monkeypatch.setattr(
        "research_experiments.paper_runner_suite.run_unified_paper_ablation.live_resource_snapshot",
        lambda: {"platform": "darwin"},
    )
    monkeypatch.setattr(
        "research_experiments.paper_runner_suite.run_unified_paper_ablation.require_live_resources",
        lambda _snapshot: None,
    )
    monkeypatch.setattr("subprocess.run", record_relaunch)
    with pytest.raises(RuntimeError, match="stale backend relaunch"):
        materialize_isolated_comparison_report(
            SMOKE_PROTOCOL,
            protocol,
            17,
            comparison_dir,
            backward_policy="fast_exploration",
            device="mps",
            reuse_existing=True,
            expected_source=_source_identity(),
            uvt_world_representation="full_spd4",
            uvt_alpha_mode="beer_lambert",
            uvt_render_backend="hybrid_retained_fiber",
            python="python",
        )

    assert len(launched) == 1
    assert _value_after(launched[0], "--uvt-render-backend") == (
        "hybrid_retained_fiber"
    )


def test_unified_powerfoam_config_uses_the_same_protocol(tmp_path: Path) -> None:
    raw, protocol = _protocol()
    cfg = powerfoam_config(raw, protocol, 43, tmp_path / "worldfoam", wandb_mode="offline")

    assert cfg["data"]["max_frames"] == 4
    assert cfg["data"]["multicam_train_cameras"] == ["cam04", "cam09"]
    assert cfg["data"]["multicam_heldout_camera"] == "cam06"
    assert cfg["render"]["image_size"] == [96, 128]
    assert cfg["model"]["cells"] == 256
    assert cfg["train"]["steps"] == 2
    assert cfg["train"]["device"] == "mps"
    assert cfg["logging"]["image_log_every"] == 2
    assert cfg["logging"]["video_log_every"] == 2
    assert cfg["paper_protocol"] == raw
    assert cfg["logging"]["wandb_enabled"] is True
    assert cfg["logging"]["wandb_mode"] == "offline"
    assert cfg["logging"]["wandb_resume"] == "never"
    assert "scene-neural3d_coffee_martini" in cfg["logging"]["wandb_tags"]


def test_worldfoam_resolved_config_is_bound_to_the_runner_contract(
    tmp_path: Path,
) -> None:
    raw, protocol = _protocol()
    expected = powerfoam_config(
        raw,
        protocol,
        43,
        tmp_path / "worldfoam",
        wandb_mode="offline",
    )
    resolved = json.loads(json.dumps(expected))
    resolved["render"].update(
        {
            "background_mode": "fixed",
            "background": [0.0, 0.0, 0.0],
            "eval_color_calibration": "none",
        }
    )
    resolved_path = tmp_path / "worldfoam" / "resolved_config.json"
    resolved_path.parent.mkdir(parents=True)
    resolved_path.write_text(json.dumps(resolved), encoding="utf-8")

    binding = worldfoam_resolved_config_binding(expected, resolved_path)

    assert binding["schema_version"] == 1
    assert binding["matched_expected_leaf_count"] > 0
    assert binding["resolved_config_sha256"] == hashlib.sha256(
        resolved_path.read_bytes()
    ).hexdigest()

    resolved["train"]["steps"] += 1
    resolved_path.write_text(json.dumps(resolved), encoding="utf-8")
    with pytest.raises(ValueError, match="train.steps"):
        worldfoam_resolved_config_binding(expected, resolved_path)


def test_wandb_identity_requires_the_exact_run_id_file_and_directory(
    tmp_path: Path,
) -> None:
    run_id = "paexact"
    run_dir = tmp_path / "wandb" / "offline-run-test" / "files"
    run_dir.mkdir(parents=True)
    (run_dir.parent / "run-other.wandb").write_bytes(b"wrong")

    with pytest.raises(FileNotFoundError, match="run-paexact.wandb"):
        wandb_file_identity(run_dir, run_id)

    expected_path = run_dir.parent / f"run-{run_id}.wandb"
    expected_path.write_bytes(b"right")
    artifact = wandb_file_identity(run_dir, run_id)
    identity = {
        "run_id": run_id,
        "mode": "offline",
        "run_dir": str(run_dir.resolve()),
        "source_digest": "a" * 64,
        "comparison_report_sha256": "b" * 64,
        "config_sha256": "c" * 64,
        "run_file": artifact,
    }
    validate_wandb_identity(
        identity,
        run_id=run_id,
        mode="offline",
        source_digest="a" * 64,
        report_digest="b" * 64,
        config_digest="c" * 64,
    )

    foreign_run_dir = tmp_path / "foreign" / "offline-run-test" / "files"
    foreign_run_dir.mkdir(parents=True)
    (foreign_run_dir.parent / f"run-{run_id}.wandb").write_bytes(b"right")
    identity["run_file"] = wandb_file_identity(foreign_run_dir, run_id)
    with pytest.raises(ValueError, match="run-file identity is invalid"):
        validate_wandb_identity(
            identity,
            run_id=run_id,
            mode="offline",
            source_digest="a" * 64,
            report_digest="b" * 64,
            config_digest="c" * 64,
        )


def test_route_native_identity_rejects_added_runtime_source(
    tmp_path: Path,
) -> None:
    native_path = tmp_path / "native.so"
    native_path.write_bytes(b"native")
    source_root = tmp_path / "metal"
    source_root.mkdir()
    (source_root / "kernel.metal").write_text("kernel", encoding="utf-8")
    native = {
        "module": "fixture.native",
        "path": str(native_path.resolve()),
        "bytes": native_path.stat().st_size,
        "sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
        "runtime_source_tree": paper_runtime_source_tree_identity(source_root),
    }
    validate_route_native_extension_identity("fixture", native)

    (source_root / "new_kernel.metal").write_text("new", encoding="utf-8")
    with pytest.raises(ValueError, match="runtime source provenance drifted"):
        validate_route_native_extension_identity("fixture", native)


def test_paper_scene_tag_is_derived_from_each_protocol() -> None:
    _, coffee = _protocol()
    raw = load_config_file(
        ROOT / "src" / "train_configs" / "paper_protocols" / "cook_spinach_full_300f_progressive_512_v1.jsonc"
    )
    spinach = resolve_paper_training_protocol(raw)

    assert paper_scene_tag(coffee) == "scene-neural3d_coffee_martini"
    assert paper_scene_tag(spinach) == "scene-neural3d_cook_spinach"


def test_dnerf_protocol_routes_both_trainers_through_the_posed_trajectory_adapter(tmp_path: Path) -> None:
    path = (
        ROOT
        / "src"
        / "train_configs"
        / "paper_protocols"
        / "dnerf_bouncingballs_matched_20f_progressive_512_v1.jsonc"
    )
    raw = load_config_file(path)
    protocol = resolve_paper_training_protocol(raw)
    command = comparison_command(
        path,
        protocol,
        17,
        tmp_path / "compare",
        backward_policy="fast_exploration",
        device="mps",
        python="python",
    )
    cfg = powerfoam_config(
        raw,
        protocol,
        17,
        tmp_path / "worldfoam",
        wandb_mode="offline",
        worldfoam_initializer="video",
    )

    assert paper_camera_rig_init(protocol) == "dnerf"
    assert _value_after(command, "--camera-rig-init") == "dnerf"
    assert _value_after(command, "--uvt-camera-projection") == "legacy_pinhole"
    assert paper_world_tubes_camera_policy(protocol) == ("legacy_pinhole", "segmented", 1)
    assert _value_after(command, "--uvt-camera-sequence-mode") == "segmented"
    assert _value_after(command, "--uvt-segment-frames") == "1"
    assert cfg["camera"]["rig_init"] == "dnerf"
    assert cfg["data"]["multicam_train_cameras"] == ["train_trajectory"]
    assert cfg["data"]["multicam_heldout_camera"] == "test_trajectory"

    with pytest.raises(ValueError, match="full_spd4 has no segmented camera compiler"):
        comparison_command(
            path,
            protocol,
            17,
            tmp_path / "full_spd4",
            backward_policy="fast_exploration",
            device="mps",
            uvt_world_representation="full_spd4",
            python="python",
        )
    with pytest.raises(ValueError, match="scoped to static_view"):
        comparison_command(
            path,
            protocol,
            17,
            tmp_path / "beer_lambert",
            backward_policy="fast_exploration",
            device="mps",
            uvt_world_representation="full_spd4",
            uvt_alpha_mode="beer_lambert",
            python="python",
        )


def test_worldfoam_paper_metrics_are_from_the_final_checkpoint(tmp_path: Path) -> None:
    history = tmp_path / "eval_metrics_history.jsonl"
    history.write_text(
        '{"step":0,"metrics":{"heldout_eval_psnr":9.0}}\n'
        '{"step":600,"metrics":{"heldout_eval_psnr":8.0,"heldout_eval_lpips":0.3}}\n',
        encoding="utf-8",
    )

    metrics = load_final_powerfoam_metrics(history, expected_step=600)

    assert metrics == {"heldout_eval_psnr": 8.0, "heldout_eval_lpips": 0.3}
    with pytest.raises(ValueError, match="no evaluation at final step"):
        load_final_powerfoam_metrics(history, expected_step=601)


def test_worldfoam_initializer_cannot_leak_coffee_geometry_into_breadth_rows(tmp_path: Path) -> None:
    raw, protocol = _protocol()
    cfg = powerfoam_config(
        raw,
        protocol,
        17,
        tmp_path / "worldfoam",
        wandb_mode="offline",
        worldfoam_initializer="video",
    )

    assert cfg["model"]["init_from_video"] is True
    assert cfg["model"]["init_point_cloud_path"] is None
    with pytest.raises(FileNotFoundError, match="initializer does not exist"):
        powerfoam_config(
            raw,
            protocol,
            17,
            tmp_path / "missing",
            wandb_mode="offline",
            worldfoam_initializer=str(tmp_path / "missing.ply"),
        )


def test_submission_source_gate_records_both_repository_revisions() -> None:
    provenance = source_provenance()

    assert len(provenance["repository_commit"]) == 40
    assert len(provenance["star_uvt_commit"]) == 40
    with pytest.raises(RuntimeError, match="repository_dirty"):
        require_clean_provenance(
            {
                "repository_commit": provenance["repository_commit"],
                "repository_dirty": True,
                "star_uvt_commit": provenance["star_uvt_commit"],
                "star_uvt_dirty": False,
            }
        )
    with pytest.raises(RuntimeError, match="literal clean source flags"):
        require_clean_provenance(
            {
                "repository_commit": provenance["repository_commit"],
                "repository_dirty": 0,
                "star_uvt_commit": provenance["star_uvt_commit"],
                "star_uvt_dirty": False,
            }
        )
    with pytest.raises(RuntimeError, match="40-character commit hashes"):
        require_clean_provenance(
            {
                "repository_commit": "not-a-commit",
                "repository_dirty": False,
                "star_uvt_commit": provenance["star_uvt_commit"],
                "star_uvt_dirty": False,
            }
        )


def test_local_mps_execution_is_fail_closed_and_full_protocol_needs_second_acknowledgement() -> None:
    _, smoke = _protocol()
    full_raw = load_config_file(DEFAULT_PROTOCOL)
    full = resolve_paper_training_protocol(full_raw)

    assert local_mps_safety_estimate(smoke)["high_risk"] is False
    assert local_mps_safety_estimate(full)["high_risk"] is True
    with pytest.raises(RuntimeError, match="allow-local-mps-execution"):
        require_execution_safety_acknowledgement(
            smoke,
            device="mps",
            allow_local_mps_execution=False,
            allow_high_risk_local_mps=False,
        )
    with pytest.raises(RuntimeError, match="allow-high-risk-local-mps"):
        require_execution_safety_acknowledgement(
            full,
            device="mps",
            allow_local_mps_execution=True,
            allow_high_risk_local_mps=False,
        )
    estimate = require_execution_safety_acknowledgement(
        full,
        device="cpu",
        allow_local_mps_execution=False,
        allow_high_risk_local_mps=False,
    )
    assert estimate["estimated_peak_bytes"] > estimate["safety_limit_bytes"]


def test_checked_in_full_protocol_manifest_is_all_300_frames() -> None:
    raw = load_config_file(DEFAULT_PROTOCOL)
    protocol = resolve_paper_training_protocol(raw)
    validation = validate_manifest(protocol)

    assert validation["dataset"] == "neural_3d_video"
    assert validation["expected_pose_source"] == (
        "neural_3d_llff_opencv_relative_pinhole_v2"
    )
    assert protocol.dataset.frame_count == 300
    assert protocol.dataset.samples_per_epoch == 600
    assert protocol.nominal_epoch_coverage == 4.0
    assert all(validation["checks"].values())
    assert validation["duration_seconds"] == 10.0
    assert validation["source_image_size"] == [2028, 2704]


def test_dry_run_declares_costs_kernels_and_artifacts(tmp_path: Path) -> None:
    raw, protocol = _protocol()
    manifest = build_dry_run_manifest(
        SMOKE_PROTOCOL,
        raw,
        protocol,
        seed=17,
        out_dir=tmp_path,
        backward_policy="fast_exploration",
        device="mps",
        wandb_mode="offline",
        uvt_world_representation="full_spd4",
        uvt_alpha_mode="beer_lambert",
        uvt_spd4_init_precision_z=1.0e6,
    )

    assert manifest["status"] == "dry_run"
    assert manifest["uvt_world_representation"] == "full_spd4"
    assert manifest["uvt_alpha_mode"] == "beer_lambert"
    assert manifest["uvt_opacity_semantics"] == (
        "nonnegative_fiber_integrated_peak_optical_thickness"
    )
    assert manifest["uvt_render_backend"] == "metal_tile"
    assert manifest["uvt_amplitude_convention"] == "fiber_integrated"
    assert manifest["uvt_retained_depth_samples"] == 48
    assert manifest["uvt_retained_sigma_extent"] == 6.0
    assert manifest["uvt_order_certificate_sigma"] == 6.0
    assert manifest["uvt_order_certificate_min_gap"] == 0.0
    assert manifest["uvt_effective_backward_policy"] == "fast_exploration"
    assert manifest["uvt_spd4_init_precision_z"] == 1.0e6
    assert manifest["protocol"]["target_frame_budget"] == 4
    assert manifest["protocol"]["target_pixel_budget"] == 30_720
    assert manifest["kernels"]["world_tubes"]["forward"] == "metal_tile_selected_time"
    assert manifest["kernels"]["world_tubes"]["backward"] == "direct_atomic+index_add"
    assert manifest["kernels"]["worldfoam"]["forward"] == "raytrace"
    assert manifest["kernels"]["dynamic_3dgs"]["forward"] == "fast_mac"
    assert set(manifest["comparison_lane_commands"]) == {"world_tubes", "dynamic_3dgs"}
    assert all(
        _value_after(command, "--uvt-world-representation") == "full_spd4"
        for command in manifest["comparison_lane_commands"].values()
    )
    assert all(
        _value_after(command, "--uvt-alpha-mode") == "beer_lambert"
        for command in manifest["comparison_lane_commands"].values()
    )
    assert all(
        _value_after(command, "--uvt-render-backend") == "metal_tile"
        for command in manifest["comparison_lane_commands"].values()
    )
    assert all(
        _value_after(command, "--uvt-amplitude-convention")
        == "fiber_integrated"
        for command in manifest["comparison_lane_commands"].values()
    )
    assert all(
        _value_after(command, "--uvt-spd4-init-precision-z") == "1000000.0"
        for command in manifest["comparison_lane_commands"].values()
    )
    assert "--execute" in manifest["worldfoam_lane_command"]
    assert manifest["expected_artifacts"]["run_summary"].endswith("run_summary.json")


def test_hybrid_dry_run_reports_physical_renderer_contract(tmp_path: Path) -> None:
    raw, protocol = _protocol()
    manifest = build_dry_run_manifest(
        SMOKE_PROTOCOL,
        raw,
        protocol,
        seed=17,
        out_dir=tmp_path,
        backward_policy="fast_exploration",
        device="mps",
        wandb_mode="offline",
        uvt_world_representation="full_spd4",
        uvt_alpha_mode="beer_lambert",
        uvt_render_backend="hybrid_retained_fiber",
        uvt_amplitude_convention="peak_density",
        uvt_retained_depth_samples=20,
        uvt_retained_sigma_extent=3.5,
        uvt_order_certificate_sigma=4.0,
        uvt_order_certificate_min_gap=0.2,
    )

    assert manifest["uvt_render_backend"] == "hybrid_retained_fiber"
    assert manifest["uvt_amplitude_convention"] == "peak_density"
    assert manifest["uvt_opacity_semantics"] == (
        "nonnegative_world_peak_extinction_density"
    )
    assert manifest["uvt_retained_depth_samples"] == 20
    assert manifest["uvt_retained_sigma_extent"] == 3.5
    assert manifest["uvt_order_certificate_sigma"] == 4.0
    assert manifest["uvt_order_certificate_min_gap"] == 0.2
    assert manifest["uvt_effective_backward_policy"] == "fast_exploration"
    assert manifest["kernels"]["world_tubes"]["forward"] == (
        "hybrid_retained_fiber_selected_time"
    )
    assert manifest["kernels"]["world_tubes"]["backward"] == (
        "direct_atomic+index_add+retained_fiber_metal_autograd"
    )


def test_cost_validator_keeps_target_budget_separate_from_extra_rasterization() -> None:
    _, protocol = _protocol()
    lane = {
        "steps": 2,
        "paper_protocol": {
            "enabled": True,
            "sampling": {
                "mode": "spacetime_epoch",
                "same_time_count": protocol.same_time_count,
                "local_time_count": protocol.local_time_count,
                "local_time_radius": protocol.local_time_radius,
            },
            "sample_schedule": _sample_schedule(),
            "stages": [stage.as_dict() for stage in protocol.stages],
            "cost": {
                "optimizer_steps": 2,
                "target_frames": 4,
                "target_pixels": 30_720,
                "rasterized_frames": 16,
            },
        },
    }

    validate_lane_cost("world_tubes", lane, protocol)
    assert lane["paper_protocol"]["cost"]["rasterized_frames"] > protocol.target_frame_budget
    lane["paper_protocol"]["sample_schedule"]["record_count"] = 1
    with pytest.raises(ValueError, match="step count"):
        validate_lane_cost("world_tubes", lane, protocol)
    lane["paper_protocol"]["sample_schedule"]["record_count"] = protocol.steps
    lane["paper_protocol"]["cost"]["target_pixels"] = 1
    with pytest.raises(ValueError, match="target-pixel"):
        validate_lane_cost("world_tubes", lane, protocol)


def test_kernel_registry_separates_fast_and_deterministic_world_tubes() -> None:
    fast = kernel_specs("fast_exploration")["world_tubes"]
    reference = kernel_specs("deterministic_quality")["world_tubes"]

    assert fast.deterministic is False
    assert reference.deterministic is True
    assert fast.backward != reference.backward


def test_paper_evaluator_is_layout_chunk_and_view_aggregation_invariant() -> None:
    target = torch.zeros(2, 2, 3, 5, 7)
    prediction = target.clone()
    prediction[0] = 0.1
    prediction[1] = 0.8

    whole = PaperRGBMetricAccumulator()
    whole.update(prediction.reshape(-1, 3, 5, 7), target.reshape(-1, 3, 5, 7))

    chunked = PaperRGBMetricAccumulator()
    for view in range(2):
        chunked.update(
            prediction[view].permute(0, 2, 3, 1),
            target[view].permute(0, 2, 3, 1),
        )

    assert chunked.metrics() == pytest.approx(whole.metrics(), abs=1.0e-8)
    per_view_psnr = []
    for view in range(2):
        accumulator = PaperRGBMetricAccumulator()
        accumulator.update(prediction[view], target[view])
        per_view_psnr.append(accumulator.metrics()["eval_psnr"])
    assert sum(per_view_psnr) / len(per_view_psnr) != pytest.approx(
        whole.metrics()["eval_psnr"],
        abs=1.0e-3,
    )


def test_paper_evidence_is_fail_closed_and_keeps_trace_diagnostics() -> None:
    lane = {
        "tube_count": 256,
        "metrics": {
            "eval_psnr": 20.0,
            "eval_ssim": 0.8,
            "eval_l1": 0.1,
            "heldout_eval_psnr": 18.0,
            "heldout_eval_ssim": 0.7,
            "heldout_eval_l1": 0.15,
            "heldout_eval_lpips": 0.25,
        },
        "paper_protocol": {
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
                "definition": "test",
                "cold_compile_forward_s": 0.2,
                "steady_forward_s": 0.3,
                "steady_forward_calls": 1,
                "backward_s": 0.4,
                "backward_calls": 2,
                "optimizer_s": 0.1,
                "optimizer_calls": 2,
                "train_wall_s": 1.0,
            },
        },
        "metal_stats": {
            "rows": [
                {
                    "stats": {
                        "projected_trace_count": 256,
                        "uvt_tile_tube_pairs": 20,
                        "summed_per_frame_tile_splat_pairs": 40,
                        "effective_pair_ratio_after_unstable_fallback": 0.5,
                        "unstable_tile_fraction": 0.1,
                        "overflow_tile_count": 0,
                        "metal_buffer_memory": 8192,
                    }
                }
            ]
        },
    }

    evidence = build_lane_evidence("world_tubes", lane, frame_count=4)
    assert evidence["quality"]["heldout_eval_lpips"] == 0.25
    assert evidence["cost"]["serialized_checkpoint_bytes"] == 1_024
    assert evidence["diagnostics"]["active_trace_count"] == 256
    assert evidence["diagnostics"]["compiled_trace_count_mean"] == 256

    for section, key, value, message in (
        ("quality", "heldout_eval_lpips", "nan", "not real numeric"),
        ("timing", "steady_forward_s", "inf", "not real numeric"),
        ("cost", "optimizer_steps", True, "not real numeric"),
        ("quality", "eval_psnr", float("nan"), "non-finite"),
        ("cost", "elapsed_s", float("inf"), "non-finite"),
    ):
        invalid = json.loads(json.dumps(evidence))
        invalid[section][key] = value
        with pytest.raises(ValueError, match=message):
            validate_lane_evidence("world_tubes", invalid)

    del lane["metrics"]["heldout_eval_lpips"]
    with pytest.raises(ValueError, match="heldout_eval_lpips"):
        build_lane_evidence("world_tubes", lane, frame_count=4)
