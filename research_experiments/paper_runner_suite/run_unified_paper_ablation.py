from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import wandb

from config_utils import load_config_file, serialize_config_value
from paper_training_protocol import apply_paper_dataset_contract, resolve_paper_training_protocol
from paper_training_types import MetalKernelSpec, PaperTrainingProtocol


ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = (
    ROOT
    / "src"
    / "train_configs"
    / "local_mac_powerfoam_metal_multicam_neural3d_coffee_train2_holdout1_feature_triangulation_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux.jsonc"
)
DEFAULT_WORLDFOAM_INITIALIZER = "base_config"
COMPARE_SCRIPT = (
    ROOT
    / "third_party"
    / "fast-mac-gsplat"
    / "variants"
    / "star_uvt_v0"
    / "research_project"
    / "benchmarks"
    / "multicam_heldout_compare.py"
)
WORLDFOAM_LANE_SCRIPT = (
    ROOT / "research_experiments" / "paper_runner_suite" / "run_worldfoam_paper_lane.py"
)
DEFAULT_PROTOCOL = (
    ROOT
    / "src"
    / "train_configs"
    / "paper_protocols"
    / "coffee_martini_full_300f_progressive_512_v1.jsonc"
)
DEFAULT_OUT_DIR = ROOT / "outputs" / "benchmarks" / "2026-07-19_unified_paper_ablation"
LANE_REPORT_KEYS = {
    "world_tubes": "star_uvt",
    "dynamic_3dgs": "free_dynamic_splats",
}
PAPER_EVIDENCE_SCHEMA_VERSION = 1
REQUIRED_QUALITY_KEYS = (
    "eval_psnr",
    "eval_ssim",
    "eval_l1",
    "heldout_eval_psnr",
    "heldout_eval_ssim",
    "heldout_eval_l1",
    "heldout_eval_lpips",
)
REQUIRED_COST_KEYS = (
    "optimizer_steps",
    "target_frames",
    "rasterized_frames",
    "target_pixels",
    "rasterized_pixels",
    "parameter_count",
    "trainable_parameter_count",
    "parameter_bytes",
    "optimizer_state_bytes",
    "serialized_checkpoint_bytes",
    "sampled_peak_current_allocated_bytes",
    "sampled_peak_driver_allocated_bytes",
    "elapsed_s",
)
REQUIRED_TIMING_KEYS = (
    "cold_compile_forward_s",
    "steady_forward_s",
    "steady_forward_calls",
    "backward_s",
    "backward_calls",
    "optimizer_s",
    "optimizer_calls",
    "train_wall_s",
)


def resolve_root_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()


def display_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def paper_scene_tag(protocol: PaperTrainingProtocol) -> str:
    """Return a stable scene tag without hard-coding the first paper scene."""
    scene_id = protocol.dataset.sample_id.split("_train_", 1)[0]
    return f"scene-{scene_id}"


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def load_final_powerfoam_metrics(path: Path, *, expected_step: int) -> dict[str, Any]:
    """Load the final-checkpoint evaluation, never an earlier best checkpoint."""
    if not path.exists():
        raise FileNotFoundError(f"WorldFoam evaluation history is missing: {path}")
    matches = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if int(row.get("step", -1)) == int(expected_step):
            matches.append(row)
    if not matches:
        raise ValueError(f"WorldFoam has no evaluation at final step {expected_step}: {path}")
    metrics = matches[-1].get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"WorldFoam final evaluation has no metrics object: {path}")
    return metrics


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(serialize_config_value(dict(payload)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def source_provenance() -> dict[str, Any]:
    star_root = ROOT / "third_party" / "fast-mac-gsplat" / "variants" / "star_uvt_v0"

    def git(*args: str, cwd: Path) -> str:
        return subprocess.check_output(("git", *args), cwd=cwd, text=True).strip()

    return {
        "repository_commit": git("rev-parse", "HEAD", cwd=ROOT),
        "repository_dirty": bool(git("status", "--porcelain", cwd=ROOT)),
        "star_uvt_commit": git("rev-parse", "HEAD", cwd=star_root),
        "star_uvt_dirty": bool(git("status", "--porcelain", cwd=star_root)),
    }


def require_clean_provenance(provenance: Mapping[str, Any]) -> None:
    dirty = [key for key in ("repository_dirty", "star_uvt_dirty") if bool(provenance[key])]
    if dirty:
        raise RuntimeError(f"paper submission runs require clean source state; dirty flags: {dirty}")


def host_physical_memory_bytes() -> int:
    if sys.platform == "darwin":
        return int(subprocess.check_output(("sysctl", "-n", "hw.memsize"), text=True).strip())
    if hasattr(os, "sysconf"):
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        page_count = int(os.sysconf("SC_PHYS_PAGES"))
        return page_size * page_count
    raise RuntimeError("cannot determine host physical memory for paper-run safety preflight")


def local_mps_safety_estimate(protocol: PaperTrainingProtocol) -> dict[str, Any]:
    """Retain the incident-calibrated eager upper bound as a fail-closed guard.

    Lane isolation now releases allocator state between representations, but it
    has not been profiled safely at full scale. Until streaming or off-machine
    evidence replaces this bound, keep the older combined estimate so the code
    change cannot silently authorize the workload that crashed the host.
    """

    float_bytes = 4
    rgb_channels = 3
    rendered_channels = 4
    frames = protocol.dataset.frame_count
    train_views = len(protocol.dataset.train_cameras)
    total_views = train_views + len(protocol.dataset.heldout_cameras)
    final_pixels = protocol.final_stage.image_size.pixels
    stage_pixels = sum(stage.image_size.pixels for stage in protocol.stages)
    bundle_bytes = total_views * frames * final_pixels * rgb_channels * float_bytes
    stage_cache_bytes_per_lane = train_views * frames * stage_pixels * rgb_channels * float_bytes
    eval_bytes_per_lane = total_views * frames * final_pixels * rendered_channels * float_bytes
    raw_combined_bytes = bundle_bytes + 2 * stage_cache_bytes_per_lane + 2 * eval_bytes_per_lane
    estimated_peak_bytes = math.ceil(1.75 * raw_combined_bytes)
    host_bytes = host_physical_memory_bytes()
    safety_limit_bytes = math.floor(0.60 * host_bytes)
    return {
        "definition": "incident-calibrated legacy combined eager upper bound; intentionally not relaxed by unprofiled lane isolation",
        "execution_model": "one_child_process_per_representation",
        "bundle_bytes": bundle_bytes,
        "stage_cache_bytes_per_lane": stage_cache_bytes_per_lane,
        "eval_bytes_per_lane": eval_bytes_per_lane,
        "raw_combined_bytes": raw_combined_bytes,
        "estimated_peak_bytes": estimated_peak_bytes,
        "host_physical_memory_bytes": host_bytes,
        "safety_limit_bytes": safety_limit_bytes,
        "estimated_peak_gib": estimated_peak_bytes / float(1 << 30),
        "host_physical_memory_gib": host_bytes / float(1 << 30),
        "high_risk": estimated_peak_bytes > safety_limit_bytes,
        "incident_reference": "agent_notes/loose_notes/2026-07-22_19-26-04_mps_memory_pressure_kernel_task_incident.md",
    }


def require_execution_safety_acknowledgement(
    protocol: PaperTrainingProtocol,
    *,
    device: str,
    allow_local_mps_execution: bool,
    allow_high_risk_local_mps: bool,
) -> dict[str, Any]:
    estimate = local_mps_safety_estimate(protocol)
    if str(device).lower() != "mps":
        return estimate
    if not allow_local_mps_execution:
        raise RuntimeError(
            "Local MPS execution is fail-closed after the 2026-07-22 memory-pressure incident. "
            "Do not enable it without explicit user approval; then pass --allow-local-mps-execution."
        )
    if bool(estimate["high_risk"]) and not allow_high_risk_local_mps:
        raise RuntimeError(
            f"Estimated local MPS peak is {estimate['estimated_peak_gib']:.2f} GiB on a "
            f"{estimate['host_physical_memory_gib']:.2f} GiB host. Paper-scale execution remains blocked; "
            "use streamed/lane-isolated execution or, only after explicit approval, pass "
            "--allow-high-risk-local-mps."
        )
    return estimate


def kernel_specs(backward_policy: str) -> dict[str, MetalKernelSpec]:
    uvt_backward = {
        "fast_exploration": ("direct_atomic+index_add", False),
        "deterministic_quality": ("tile_pair+key_sort_scan_metal", True),
        "deterministic_compact": ("compact_tile_pair+key_sort_scan_metal", True),
    }
    if backward_policy not in uvt_backward:
        raise ValueError(f"unsupported World Tubes backward policy: {backward_policy}")
    backward, deterministic = uvt_backward[backward_policy]
    return {
        "world_tubes": MetalKernelSpec(
            representation="world_tubes",
            family="star_uvt",
            forward="metal_tile_selected_time",
            backward=backward,
            deterministic=deterministic,
            implementation="third_party/fast-mac-gsplat/variants/star_uvt_v0",
        ),
        "worldfoam": MetalKernelSpec(
            representation="worldfoam",
            family="powerfoam_metal",
            forward="raytrace",
            backward="powerfoam_metal_autograd",
            deterministic=False,
            implementation="third_party/powerfoam-metal",
        ),
        "dynamic_3dgs": MetalKernelSpec(
            representation="dynamic_3dgs",
            family="fast_mac",
            forward="fast_mac",
            backward="fast_mac_autograd",
            deterministic=False,
            implementation="third_party/fast-mac-gsplat",
        ),
    }


def paper_dataset_record(protocol: PaperTrainingProtocol) -> tuple[Path, dict[str, Any]]:
    manifest_path = resolve_root_path(protocol.dataset.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"paper manifest does not exist: {manifest_path}")
    records = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matches = [record for record in records if record.get("sample_id") == protocol.dataset.sample_id]
    if len(matches) != 1:
        raise ValueError(f"expected one manifest row for {protocol.dataset.sample_id}, found {len(matches)}")
    return manifest_path, matches[0]


def paper_camera_rig_init(protocol: PaperTrainingProtocol) -> str:
    _manifest_path, record = paper_dataset_record(protocol)
    return "dnerf" if str(record.get("dataset", "")).lower() == "dnerf" else "neural_3d_video"


def paper_world_tubes_camera_policy(protocol: PaperTrainingProtocol) -> tuple[str, str, int]:
    _manifest_path, record = paper_dataset_record(protocol)
    if str(record.get("dataset", "")).lower() != "dnerf":
        return "dataset_lens", "static_view", 4
    mode = str(record.get("world_tubes_camera_sequence_mode", ""))
    segment_frames = int(record.get("world_tubes_segment_frames", 0))
    if mode != "segmented" or segment_frames != 1:
        raise ValueError(
            "D-NeRF paper rows require the declared one-frame gauged fallback because official poses are discontinuous"
        )
    return "legacy_pinhole", mode, segment_frames


def validate_manifest(protocol: PaperTrainingProtocol) -> dict[str, Any]:
    manifest_path, record = paper_dataset_record(protocol)
    is_dnerf = str(record.get("dataset", "")).lower() == "dnerf"
    checks = {
        "train_cameras": tuple(record.get("train_cameras", ())) == protocol.dataset.train_cameras,
        "heldout_cameras": tuple(record.get("heldout_cameras", ())) == protocol.dataset.heldout_cameras,
        "frame_count_available": int(record.get("frame_count", -1)) >= protocol.dataset.frame_count,
        "fps": float(record.get("fps", -1.0)) == protocol.dataset.fps,
        "start_at_zero": (
            bool(record.get("dnerf_times")) and float(record["dnerf_times"][0]) == 0.0
            if is_dnerf
            else float(record.get("source_start_seconds", -1.0)) == 0.0
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"paper manifest contract failed: {', '.join(failed)}")
    scene_dir = resolve_root_path(record["dataset_scene_dir"])
    if is_dnerf:
        split_map = record.get("dnerf_camera_splits", {})
        index_map = record.get("dnerf_frame_indices", {})
        camera_paths: dict[str, Path] = {}
        image_paths: dict[str, list[Path]] = {}
        for camera in (*protocol.dataset.train_cameras, *protocol.dataset.heldout_cameras):
            split = split_map.get(camera)
            indices = index_map.get(camera)
            if not isinstance(split, str) or not isinstance(indices, list):
                raise ValueError(f"D-NeRF manifest is missing split/indices for {camera}")
            transforms_path = scene_dir / f"transforms_{split}.json"
            camera_paths[camera] = transforms_path
            payload = load_json(transforms_path)
            frames = payload.get("frames", [])
            image_paths[camera] = [
                (scene_dir / str(frames[int(index)]["file_path"])).with_suffix(".png")
                for index in indices[: protocol.dataset.frame_count]
            ]
        missing = [
            str(path)
            for path in (*camera_paths.values(), *(path for paths in image_paths.values() for path in paths))
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(f"paper D-NeRF inputs are missing: {missing}")
    else:
        camera_paths = {
            camera: scene_dir / f"{camera}.mp4"
            for camera in (*protocol.dataset.train_cameras, *protocol.dataset.heldout_cameras)
        }
        missing = [str(path) for path in camera_paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"paper camera videos are missing: {missing}")
    return {
        "manifest": display_path(manifest_path),
        "sample_id": protocol.dataset.sample_id,
        "dataset": record.get("dataset"),
        "checks": checks,
        "camera_inputs": {camera: display_path(path) for camera, path in camera_paths.items()},
        "camera_videos": (
            None if is_dnerf else {camera: display_path(path) for camera, path in camera_paths.items()}
        ),
        "source_image_size": record.get("source_image_size"),
        "duration_seconds": record.get("duration_seconds"),
        "sample_layout": record.get("sample_layout", "synchronized_multicamera"),
    }


def comparison_command(
    protocol_path: Path,
    protocol: PaperTrainingProtocol,
    seed: int,
    out_dir: Path,
    *,
    backward_policy: str,
    device: str,
    only_lane: str = "combined",
    allow_local_mps_execution: bool = False,
    python: str = sys.executable,
) -> list[str]:
    if only_lane not in {"combined", *LANE_REPORT_KEYS}:
        raise ValueError(f"unsupported comparison lane: {only_lane}")
    camera_rig_init = paper_camera_rig_init(protocol)
    camera_projection, camera_sequence_mode, segment_frames = paper_world_tubes_camera_policy(protocol)
    command = [
        python,
        str(COMPARE_SCRIPT),
        "--baseline-config",
        str(BASE_CONFIG),
        "--target-size",
        str(protocol.final_stage.image_size.width),
        "--max-frames",
        str(protocol.dataset.frame_count),
        "--train-seconds",
        str(protocol.max_train_seconds),
        "--max-steps",
        str(protocol.steps),
        "--device",
        device,
        "--seed",
        str(seed),
        "--uvt-tubes",
        str(protocol.final_stage.primitive_count),
        "--uvt-render-backend",
        "metal_tile",
        "--uvt-backward-policy",
        backward_policy,
        "--uvt-camera-projection",
        camera_projection,
        "--uvt-camera-sequence-mode",
        camera_sequence_mode,
        "--uvt-segment-frames",
        str(segment_frames),
        "--uvt-init-views",
        "all_train",
        "--uvt-init-sampling",
        "grid",
        "--uvt-init-frames",
        "all",
        "--uvt-loss-scope",
        "paper_batch",
        "--uvt-train-schedule",
        "view_shuffled_cycle",
        "--splat-count",
        str(protocol.final_stage.primitive_count),
        "--splat-renderer",
        "fast_mac",
        "--splat-camera-projection",
        "dataset_lens",
        "--paper-protocol",
        str(protocol_path),
        "--eval-chunk-frames",
        str(max(stage.frames_per_step for stage in protocol.stages)),
        "--eval-media-max-frames",
        "32",
        "--camera-rig-init",
        camera_rig_init,
        "--out-dir",
        str(out_dir),
        "--only-lane",
        only_lane,
    ]
    if allow_local_mps_execution:
        command.append("--allow-paper-local-mps-execution")
    return command


def comparison_lane_commands(
    protocol_path: Path,
    protocol: PaperTrainingProtocol,
    seed: int,
    comparison_dir: Path,
    *,
    backward_policy: str,
    device: str,
    allow_local_mps_execution: bool = False,
    python: str = sys.executable,
) -> dict[str, list[str]]:
    """Build one process command per representation to bound allocator lifetime."""
    return {
        lane_name: comparison_command(
            protocol_path,
            protocol,
            seed,
            comparison_dir / lane_name,
            backward_policy=backward_policy,
            device=device,
            only_lane=lane_name,
            allow_local_mps_execution=allow_local_mps_execution,
            python=python,
        )
        for lane_name in LANE_REPORT_KEYS
    }


def worldfoam_lane_command(
    protocol_path: Path,
    seed: int,
    out_dir: Path,
    *,
    device: str,
    wandb_mode: str,
    worldfoam_initializer: str = DEFAULT_WORLDFOAM_INITIALIZER,
    allow_local_mps_execution: bool = False,
    allow_high_risk_local_mps: bool = False,
    python: str = sys.executable,
) -> list[str]:
    command = [
        python,
        str(WORLDFOAM_LANE_SCRIPT),
        "--execute",
        "--protocol",
        str(protocol_path),
        "--seed",
        str(seed),
        "--out-dir",
        str(out_dir),
        "--device",
        device,
        "--wandb-mode",
        wandb_mode,
        "--worldfoam-initializer",
        worldfoam_initializer,
    ]
    if allow_local_mps_execution:
        command.append("--allow-local-mps-execution")
    if allow_high_risk_local_mps:
        command.append("--allow-high-risk-local-mps")
    return command


def powerfoam_config(
    raw_protocol: Mapping[str, Any],
    protocol: PaperTrainingProtocol,
    seed: int,
    out_dir: Path,
    *,
    wandb_mode: str,
    device: str = "mps",
    worldfoam_initializer: str = DEFAULT_WORLDFOAM_INITIALIZER,
) -> dict[str, Any]:
    cfg = copy.deepcopy(load_config_file(BASE_CONFIG))
    cfg["camera"]["rig_init"] = paper_camera_rig_init(protocol)
    if worldfoam_initializer == "video":
        cfg["model"]["init_point_cloud_path"] = None
    elif worldfoam_initializer != DEFAULT_WORLDFOAM_INITIALIZER:
        init_path = resolve_root_path(worldfoam_initializer)
        if not init_path.exists():
            raise FileNotFoundError(f"WorldFoam initializer does not exist: {init_path}")
        cfg["model"]["init_point_cloud_path"] = str(init_path)
    cfg["data"] = apply_paper_dataset_contract(cfg["data"], protocol)
    cfg["render"]["render_size"] = protocol.final_stage.image_size.width
    cfg["render"]["image_size"] = protocol.final_stage.image_size.as_list()
    cfg["model"]["cells"] = protocol.final_stage.primitive_count
    cfg["model"]["resample_every"] = 0
    cfg["train"]["steps"] = protocol.steps
    cfg["train"]["frames_per_step"] = max(stage.frames_per_step for stage in protocol.stages)
    cfg["train"]["seed"] = int(seed)
    cfg["train"]["device"] = str(device)
    cfg["paper_protocol"] = copy.deepcopy(dict(raw_protocol))
    run_hash = hashlib.sha1(
        f"{protocol.name}:{seed}:worldfoam:evidence-v1-final-only".encode("utf-8")
    ).hexdigest()[:8]
    cfg["logging"].update(
        {
            "log_every": max(1, protocol.steps // 20),
            # Each artifact pass evaluates and encodes the full temporal set.
            # The paper contract needs clean-init and final quality, not six
            # redundant 300-frame videos inside one training row.
            "image_log_every": protocol.steps,
            "video_log_every": protocol.steps,
            "always_log_last_step": True,
            "eval_media_max_frames": 32,
            "output_dir": str(out_dir),
            "wandb_enabled": True,
            "wandb_mode": wandb_mode,
            "wandb_run_id": f"pf{run_hash}",
            "wandb_resume": "allow",
            "wandb_project": "dynaworld",
            "wandb_disable_git": True,
            "wandb_disable_code": True,
            "wandb_run_name": f"paper-{protocol.name}-worldfoam-seed{seed}",
            "wandb_tags": [
                "paper-ablation-v1",
                paper_scene_tag(protocol),
                "full-temporal" if protocol.dataset.frame_count == 300 else "mechanical-smoke",
                "worldfoam",
                "powerfoam-metal",
                protocol.name,
                f"seed-{seed}",
            ],
        }
    )
    return cfg


def validate_lane_cost(
    lane_name: str,
    lane: Mapping[str, Any],
    protocol: PaperTrainingProtocol,
) -> None:
    if int(lane["steps"]) != protocol.steps:
        raise ValueError(f"{lane_name} completed {lane['steps']} of {protocol.steps} required steps")
    paper = lane.get("paper_protocol")
    if not isinstance(paper, Mapping) or not bool(paper.get("enabled", False)):
        raise ValueError(f"{lane_name} did not report an enabled paper protocol")
    cost = paper.get("cost")
    if not isinstance(cost, Mapping):
        raise ValueError(f"{lane_name} did not report paper cost accounting")
    if int(cost["optimizer_steps"]) != protocol.steps:
        raise ValueError(f"{lane_name} optimizer-step cost does not match the protocol")
    if int(cost["target_frames"]) != protocol.target_frame_budget:
        raise ValueError(f"{lane_name} target-frame cost does not match the protocol")
    if int(cost["target_pixels"]) != protocol.target_pixel_budget:
        raise ValueError(f"{lane_name} target-pixel cost does not match the protocol")


def _mean_stat(rows: list[Mapping[str, Any]], key: str) -> float:
    values = [float(row["stats"][key]) for row in rows if key in row.get("stats", {})]
    if not values:
        raise ValueError(f"World Tubes metal diagnostics are missing {key}")
    return sum(values) / float(len(values))


def representation_diagnostics(
    lane_name: str,
    lane: Mapping[str, Any],
    *,
    frame_count: int,
) -> dict[str, Any]:
    metrics = lane["metrics"]
    if lane_name == "world_tubes":
        rows = lane.get("metal_stats", {}).get("rows", [])
        if not rows:
            raise ValueError("World Tubes did not report metal trace diagnostics")
        unstable_fraction = _mean_stat(rows, "unstable_tile_fraction")
        sequence_mode = str(lane.get("camera_sequence_mode", "static_view"))
        segment_frames = int(lane.get("segment_frames", frame_count))
        projected_counts = [
            float(row["stats"]["projected_trace_count"])
            for row in rows
            if "projected_trace_count" in row.get("stats", {})
        ]
        if projected_counts:
            compiled_trace_count = sum(projected_counts) / float(len(projected_counts))
        elif sequence_mode == "static_view":
            # Submission rows recorded before the explicit counter are exactly
            # one projected trace per active tube under the static chart.
            compiled_trace_count = float(lane["tube_count"])
        else:
            raise ValueError("moving-camera World Tubes diagnostics are missing projected_trace_count")
        return {
            "active_trace_count": int(lane["tube_count"]),
            "compiled_trace_count_mean": compiled_trace_count,
            "tile_trace_pairs_mean": _mean_stat(rows, "uvt_tile_tube_pairs"),
            "per_frame_tile_trace_pairs_mean": _mean_stat(rows, "summed_per_frame_tile_splat_pairs"),
            "effective_pair_ratio_after_fallback_mean": _mean_stat(
                rows, "effective_pair_ratio_after_unstable_fallback"
            ),
            "unstable_tile_fraction_mean": unstable_fraction,
            "fallback_fraction_mean": unstable_fraction,
            "overflow_tile_count_mean": _mean_stat(rows, "overflow_tile_count"),
            "metal_buffer_bytes_mean": _mean_stat(rows, "metal_buffer_memory"),
            "camera_chart_mode": sequence_mode,
            "camera_chart_count": (
                math.ceil(int(frame_count) / segment_frames) if sequence_mode == "segmented" else 1
            ),
            "camera_chart_fallback_fraction": 1.0 if sequence_mode == "segmented" else 0.0,
        }
    if lane_name == "dynamic_3dgs":
        active = int(lane["splat_count"])
        return {
            "active_splats_per_frame": active,
            "stored_frame_count": int(frame_count),
            "stored_splat_states": active * int(frame_count),
            "fallback_fraction": 0.0,
        }
    if lane_name == "worldfoam":
        return {
            "active_cell_count": int(metrics["state_cell_count"]),
            "visible_cell_fraction": float(metrics["aux_visible_fraction"]),
            "visible_cell_frame_events": int(metrics["aux_visible_cell_frame_events"]),
            "possible_cell_frame_events": int(metrics["aux_possible_cell_frame_events"]),
            "mean_visible_cells_per_frame": float(metrics["aux_mean_visible_cells_per_frame"]),
            "median_depth_valid_fraction": float(metrics["aux_median_depth_valid_fraction"]),
            "mean_cell_contribution": float(metrics["aux_mean_contrib"]),
            "max_cell_contribution": float(metrics["aux_max_contrib"]),
        }
    raise ValueError(f"unsupported paper lane: {lane_name}")


def build_lane_evidence(
    lane_name: str,
    lane: Mapping[str, Any],
    *,
    frame_count: int,
) -> dict[str, Any]:
    paper = lane["paper_protocol"]
    required_sources = (
        ("quality", lane["metrics"], REQUIRED_QUALITY_KEYS),
        ("cost", paper["cost"], REQUIRED_COST_KEYS),
        ("timing", paper.get("timing", {}), REQUIRED_TIMING_KEYS),
    )
    missing = {
        section: [key for key in keys if key not in source]
        for section, source, keys in required_sources
    }
    missing = {section: keys for section, keys in missing.items() if keys}
    if missing:
        detail = "; ".join(f"{section}: {', '.join(keys)}" for section, keys in missing.items())
        raise ValueError(f"{lane_name} cannot form paper evidence; missing {detail}")
    evidence = {
        "schema_version": PAPER_EVIDENCE_SCHEMA_VERSION,
        "quality": {key: lane["metrics"][key] for key in REQUIRED_QUALITY_KEYS},
        "cost": {key: paper["cost"][key] for key in REQUIRED_COST_KEYS},
        "timing": {key: paper["timing"][key] for key in REQUIRED_TIMING_KEYS},
        "timing_definition": paper["timing"].get("definition"),
        "diagnostics": representation_diagnostics(lane_name, lane, frame_count=frame_count),
    }
    validate_lane_evidence(lane_name, evidence)
    return evidence


def validate_lane_evidence(lane_name: str, evidence: Mapping[str, Any]) -> None:
    if int(evidence.get("schema_version", -1)) != PAPER_EVIDENCE_SCHEMA_VERSION:
        raise ValueError(f"{lane_name} paper evidence schema version is missing or stale")
    for section, keys in (
        ("quality", REQUIRED_QUALITY_KEYS),
        ("cost", REQUIRED_COST_KEYS),
        ("timing", REQUIRED_TIMING_KEYS),
    ):
        values = evidence.get(section)
        if not isinstance(values, Mapping):
            raise ValueError(f"{lane_name} paper evidence is missing {section}")
        missing = [key for key in keys if key not in values]
        if missing:
            raise ValueError(f"{lane_name} paper evidence {section} is missing: {', '.join(missing)}")
        nonfinite = [
            key
            for key in keys
            if isinstance(values[key], (int, float)) and not math.isfinite(float(values[key]))
        ]
        if nonfinite:
            raise ValueError(f"{lane_name} paper evidence {section} is non-finite: {', '.join(nonfinite)}")
    if float(evidence["quality"]["heldout_eval_lpips"]) < 0.0:
        raise ValueError(f"{lane_name} heldout LPIPS must be non-negative")
    for key in ("serialized_checkpoint_bytes", "parameter_bytes"):
        if int(evidence["cost"][key]) <= 0:
            raise ValueError(f"{lane_name} {key} must be positive")
    if not isinstance(evidence.get("diagnostics"), Mapping) or not evidence["diagnostics"]:
        raise ValueError(f"{lane_name} paper evidence is missing representation diagnostics")


def validate_comparison_report(
    report: Mapping[str, Any],
    protocol: PaperTrainingProtocol,
    *,
    backward_policy: str,
) -> None:
    meta = report["meta"]
    if tuple(meta["train_cameras"]) != protocol.dataset.train_cameras:
        raise ValueError("comparison report train cameras do not match the paper protocol")
    if tuple(meta["heldout_cameras"]) != protocol.dataset.heldout_cameras:
        raise ValueError("comparison report heldout cameras do not match the paper protocol")
    if int(meta["frame_count"]) != protocol.dataset.frame_count:
        raise ValueError("comparison report frame count does not match the paper protocol")
    if meta["uvt_backward_policy"]["name"] != backward_policy:
        raise ValueError("comparison report World Tubes backward policy drifted")
    for lane_name, report_key in LANE_REPORT_KEYS.items():
        lane = report.get(report_key)
        if not isinstance(lane, Mapping):
            raise ValueError(f"comparison report is missing {lane_name}")
        validate_lane_cost(lane_name, lane, protocol)
        build_lane_evidence(lane_name, lane, frame_count=protocol.dataset.frame_count)


def merge_comparison_lane_reports(
    lane_reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Merge isolated renderer reports, rejecting cross-lane protocol drift."""
    if set(lane_reports) != set(LANE_REPORT_KEYS):
        raise ValueError(
            f"isolated comparison reports must contain {sorted(LANE_REPORT_KEYS)}, "
            f"got {sorted(lane_reports)}"
        )
    meta_keys = (
        "baseline_config",
        "target_size",
        "image_size",
        "max_frames",
        "frame_count",
        "train_seconds",
        "device",
        "seed",
        "train_cameras",
        "heldout_cameras",
        "pose_source",
        "uvt_camera_projection",
        "uvt_camera_sequence_mode",
        "uvt_segment_frames",
        "uvt_backward_policy",
        "splat_camera_projection",
        "eval_chunk_frames",
        "eval_media_max_frames",
    )
    reference_name = "world_tubes"
    reference_meta = lane_reports[reference_name].get("meta")
    if not isinstance(reference_meta, Mapping):
        raise ValueError(f"isolated {reference_name} report has no metadata")
    for lane_name, report in lane_reports.items():
        meta = report.get("meta")
        if not isinstance(meta, Mapping):
            raise ValueError(f"isolated {lane_name} report has no metadata")
        if meta.get("only_lane") != lane_name:
            raise ValueError(f"isolated {lane_name} report was produced as {meta.get('only_lane')!r}")
        drift = [key for key in meta_keys if meta.get(key) != reference_meta.get(key)]
        if drift:
            raise ValueError(f"isolated {lane_name} report metadata drifted: {', '.join(drift)}")
        report_key = LANE_REPORT_KEYS[lane_name]
        if not isinstance(report.get(report_key), Mapping):
            raise ValueError(f"isolated {lane_name} report is missing {report_key}")
        foreign = [
            key
            for other_name, key in LANE_REPORT_KEYS.items()
            if other_name != lane_name and report.get(key) is not None
        ]
        if foreign:
            raise ValueError(f"isolated {lane_name} report unexpectedly contains {', '.join(foreign)}")

    merged_meta = dict(reference_meta)
    merged_meta.update(
        {
            "only_lane": "isolated_merged",
            "skip_splats": False,
            "execution_model": "one_child_process_per_representation",
        }
    )
    return {
        "meta": merged_meta,
        "star_uvt": lane_reports["world_tubes"]["star_uvt"],
        "star_uvt_selected": lane_reports["world_tubes"].get("star_uvt_selected"),
        "free_dynamic_splats": lane_reports["dynamic_3dgs"]["free_dynamic_splats"],
    }


def materialize_isolated_comparison_report(
    protocol_path: Path,
    protocol: PaperTrainingProtocol,
    seed: int,
    comparison_dir: Path,
    *,
    backward_policy: str,
    device: str,
    reuse_existing: bool,
    allow_local_mps_execution: bool = False,
    python: str = sys.executable,
) -> Path:
    """Run only missing lane children, then form their shared report contract."""
    comparison_report_path = comparison_dir / "comparison_report.json"
    if reuse_existing and comparison_report_path.exists():
        return comparison_report_path
    lane_reports: dict[str, Mapping[str, Any]] = {}
    for lane_name, command in comparison_lane_commands(
        protocol_path,
        protocol,
        seed,
        comparison_dir,
        backward_policy=backward_policy,
        device=device,
        allow_local_mps_execution=allow_local_mps_execution,
        python=python,
    ).items():
        lane_report_path = comparison_dir / lane_name / "comparison_report.json"
        if not (reuse_existing and lane_report_path.exists()):
            subprocess.run(command, cwd=ROOT, check=True)
        lane_reports[lane_name] = load_json(lane_report_path)
    write_json(comparison_report_path, merge_comparison_lane_reports(lane_reports))
    return comparison_report_path


def _comparison_wandb_log(
    report: Mapping[str, Any],
    protocol: PaperTrainingProtocol,
    *,
    lane_name: str,
    seed: int,
    report_dir: Path,
    wandb_mode: str,
) -> dict[str, str]:
    report_key = LANE_REPORT_KEYS[lane_name]
    lane = report[report_key]
    metrics = lane["metrics"]
    cost = lane["paper_protocol"]["cost"]
    run_id = hashlib.sha1(
        (
            f"{protocol.name}:{seed}:{lane_name}:"
            f"{lane['paper_protocol']['kernel']['backward']}:evidence-v1"
        ).encode("utf-8")
    ).hexdigest()[:8]
    run = wandb.init(
        project="dynaworld",
        name=f"paper-{protocol.name}-{lane_name}-seed{seed}",
        tags=["paper-ablation-v1", paper_scene_tag(protocol), protocol.name, lane_name, f"seed-{seed}"],
        mode=wandb_mode,
        id=f"pa{run_id}",
        resume="allow",
        config={
            "protocol": protocol.as_dict(),
            "seed": seed,
            "kernel": lane["paper_protocol"]["kernel"],
            "source": source_provenance(),
        },
        settings=wandb.Settings(disable_git=True, disable_code=True),
        reinit="finish_previous",
    )
    payload: dict[str, Any] = {
        "train/psnr": metrics["eval_psnr"],
        "train/ssim": metrics["eval_ssim"],
        "train/l1": metrics["eval_l1"],
        "heldout/psnr": metrics["heldout_eval_psnr"],
        "heldout/ssim": metrics["heldout_eval_ssim"],
        "heldout/l1": metrics["heldout_eval_l1"],
        "heldout/lpips": metrics["heldout_eval_lpips"],
        **{f"cost/{key}": value for key, value in cost.items()},
        **{f"timing/{key}": value for key, value in lane["paper_protocol"]["timing"].items() if isinstance(value, (int, float))},
    }
    media_prefix = "star_uvt" if lane_name == "world_tubes" else "free_dynamic_splats"
    for split in ("train", "heldout"):
        path = report_dir / f"{media_prefix}_{split}_view0_side_by_side.mp4"
        if not path.exists():
            path = report_dir / lane_name / path.name
        if path.exists():
            payload[f"media/{split}_view"] = wandb.Video(str(path), format="mp4")
    run.log(payload, step=protocol.steps)
    provenance = {"mode": wandb_mode, "run_id": str(run.id), "run_dir": str(run.dir)}
    run.finish()
    return provenance


def build_dry_run_manifest(
    protocol_path: Path,
    raw_protocol: Mapping[str, Any],
    protocol: PaperTrainingProtocol,
    *,
    seed: int,
    out_dir: Path,
    backward_policy: str,
    device: str,
    wandb_mode: str,
    worldfoam_initializer: str = DEFAULT_WORLDFOAM_INITIALIZER,
) -> dict[str, Any]:
    seed_dir = out_dir / protocol.name / f"seed_{seed}"
    specs = kernel_specs(backward_policy)
    pf_cfg = powerfoam_config(
        raw_protocol,
        protocol,
        seed,
        seed_dir / "worldfoam",
        wandb_mode=wandb_mode,
        device=device,
        worldfoam_initializer=worldfoam_initializer,
    )
    comparison_dir = seed_dir / "world_tubes_dynamic_3dgs"
    return {
        "status": "dry_run",
        "execution_safety": local_mps_safety_estimate(protocol),
        "protocol_path": display_path(protocol_path),
        "protocol": protocol.as_dict(),
        "manifest_validation": validate_manifest(protocol),
        "kernels": {name: spec.as_dict() for name, spec in specs.items()},
        "comparison_lane_commands": comparison_lane_commands(
            protocol_path,
            protocol,
            seed,
            comparison_dir,
            backward_policy=backward_policy,
            device=device,
        ),
        "worldfoam_lane_command": worldfoam_lane_command(
            protocol_path,
            seed,
            seed_dir / "worldfoam",
            device=device,
            wandb_mode=wandb_mode,
            worldfoam_initializer=worldfoam_initializer,
        ),
        "powerfoam": {
            "initializer": worldfoam_initializer,
            "output_dir": pf_cfg["logging"]["output_dir"],
            "image_size": pf_cfg["render"]["image_size"],
            "steps": pf_cfg["train"]["steps"],
            "final_cells": pf_cfg["model"]["cells"],
            "wandb_mode": pf_cfg["logging"]["wandb_mode"],
        },
        "expected_artifacts": {
            "comparison_report": display_path(
                comparison_dir / "comparison_report.json"
            ),
            "world_tubes_lane_report": display_path(
                comparison_dir / "world_tubes" / "comparison_report.json"
            ),
            "dynamic_3dgs_lane_report": display_path(
                comparison_dir / "dynamic_3dgs" / "comparison_report.json"
            ),
            "worldfoam_protocol_summary": display_path(seed_dir / "worldfoam" / "paper_protocol_summary.json"),
            "run_summary": display_path(seed_dir / "run_summary.json"),
        },
    }


def execute(
    protocol_path: Path,
    raw_protocol: Mapping[str, Any],
    protocol: PaperTrainingProtocol,
    *,
    seed: int,
    out_dir: Path,
    backward_policy: str,
    device: str,
    wandb_mode: str,
    reuse_existing: bool,
    worldfoam_initializer: str = DEFAULT_WORLDFOAM_INITIALIZER,
    require_clean_source: bool = False,
    allow_local_mps_execution: bool = False,
    allow_high_risk_local_mps: bool = False,
) -> dict[str, Any]:
    execution_safety = require_execution_safety_acknowledgement(
        protocol,
        device=device,
        allow_local_mps_execution=allow_local_mps_execution,
        allow_high_risk_local_mps=allow_high_risk_local_mps,
    )
    provenance = source_provenance()
    if require_clean_source:
        require_clean_provenance(provenance)
    seed_dir = out_dir / protocol.name / f"seed_{seed}"
    comparison_dir = seed_dir / "world_tubes_dynamic_3dgs"
    worldfoam_dir = seed_dir / "worldfoam"
    comparison_report_path = comparison_dir / "comparison_report.json"
    materialize_isolated_comparison_report(
        protocol_path,
        protocol,
        seed,
        comparison_dir,
        backward_policy=backward_policy,
        device=device,
        reuse_existing=reuse_existing,
        allow_local_mps_execution=allow_local_mps_execution,
    )
    comparison_report = load_json(comparison_report_path)
    validate_comparison_report(comparison_report, protocol, backward_policy=backward_policy)

    wandb_runs = {
        lane_name: _comparison_wandb_log(
            comparison_report,
            protocol,
            lane_name=lane_name,
            seed=seed,
            report_dir=comparison_dir,
            wandb_mode=wandb_mode,
        )
        for lane_name in LANE_REPORT_KEYS
    }

    powerfoam_summary_path = worldfoam_dir / "paper_protocol_summary.json"
    if not (reuse_existing and powerfoam_summary_path.exists()):
        subprocess.run(
            worldfoam_lane_command(
                protocol_path,
                seed,
                worldfoam_dir,
                device=device,
                wandb_mode=wandb_mode,
                worldfoam_initializer=worldfoam_initializer,
                allow_local_mps_execution=allow_local_mps_execution,
                allow_high_risk_local_mps=allow_high_risk_local_mps,
            ),
            cwd=ROOT,
            check=True,
        )
    powerfoam_summary = load_json(powerfoam_summary_path)
    powerfoam_best = load_json(worldfoam_dir / "best_metrics.json")
    powerfoam_final_metrics = load_final_powerfoam_metrics(
        worldfoam_dir / "eval_metrics_history.jsonl",
        expected_step=protocol.steps,
    )
    validate_lane_cost(
        "worldfoam",
        {
            "steps": powerfoam_summary["cost"]["optimizer_steps"],
            "paper_protocol": powerfoam_summary,
        },
        protocol,
    )

    lanes = {
        lane_name: {
            "metrics": comparison_report[report_key]["metrics"],
            "paper_protocol": comparison_report[report_key]["paper_protocol"],
            "wandb": wandb_runs[lane_name],
            "evidence": build_lane_evidence(
                lane_name,
                comparison_report[report_key],
                frame_count=protocol.dataset.frame_count,
            ),
        }
        for lane_name, report_key in LANE_REPORT_KEYS.items()
    }
    lanes["worldfoam"] = {
        "metrics": powerfoam_final_metrics,
        "reported_checkpoint": "final",
        "best_metric_name": powerfoam_best["best_metric_name"],
        "best_metric_value": powerfoam_best["best_metric_value"],
        "paper_protocol": powerfoam_summary,
        "evidence": build_lane_evidence(
            "worldfoam",
            {
                "metrics": powerfoam_final_metrics,
                "paper_protocol": powerfoam_summary,
            },
            frame_count=protocol.dataset.frame_count,
        ),
        "wandb": {
            "mode": wandb_mode,
            "run_id": powerfoam_config(
                raw_protocol,
                protocol,
                seed,
                worldfoam_dir,
                wandb_mode=wandb_mode,
                device=device,
                worldfoam_initializer=worldfoam_initializer,
            )["logging"]["wandb_run_id"],
        },
    }
    summary = {
        "status": "complete",
        "seed": seed,
        "protocol_path": display_path(protocol_path),
        "protocol": protocol.as_dict(),
        "manifest_validation": validate_manifest(protocol),
        "world_tubes_backward_policy": backward_policy,
        "comparison_report": display_path(comparison_report_path),
        "worldfoam_dir": display_path(worldfoam_dir),
        "worldfoam_initializer": worldfoam_initializer,
        "source": provenance,
        "execution_safety": execution_safety,
        "lanes": lanes,
    }
    write_json(seed_dir / "run_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--uvt-backward-policy",
        choices=("fast_exploration", "deterministic_quality", "deterministic_compact"),
        default="fast_exploration",
    )
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument(
        "--worldfoam-initializer",
        default=DEFAULT_WORLDFOAM_INITIALIZER,
        help="Use 'base_config', 'video', or a scene-specific point-cloud path.",
    )
    parser.add_argument("--require-clean-source", action="store_true")
    parser.add_argument(
        "--allow-local-mps-execution",
        action="store_true",
        help="Enable only after explicit user approval; local MPS execution is otherwise fail-closed.",
    )
    parser.add_argument(
        "--allow-high-risk-local-mps",
        action="store_true",
        help="Second acknowledgement for a preflight estimate above 60% of host physical memory.",
    )
    args = parser.parse_args()

    protocol_path = resolve_root_path(args.protocol)
    raw_protocol = load_config_file(protocol_path)
    protocol = resolve_paper_training_protocol(raw_protocol)
    out_dir = resolve_root_path(args.out_dir)
    dry_run = build_dry_run_manifest(
        protocol_path,
        raw_protocol,
        protocol,
        seed=args.seed,
        out_dir=out_dir,
        backward_policy=args.uvt_backward_policy,
        device=args.device,
        wandb_mode=args.wandb_mode,
        worldfoam_initializer=args.worldfoam_initializer,
    )
    if not args.execute:
        print(json.dumps(serialize_config_value(dry_run), indent=2, sort_keys=True))
        return
    summary = execute(
        protocol_path,
        raw_protocol,
        protocol,
        seed=args.seed,
        out_dir=out_dir,
        backward_policy=args.uvt_backward_policy,
        device=args.device,
        wandb_mode=args.wandb_mode,
        reuse_existing=args.reuse_existing,
        worldfoam_initializer=args.worldfoam_initializer,
        require_clean_source=args.require_clean_source,
        allow_local_mps_execution=args.allow_local_mps_execution,
        allow_high_risk_local_mps=args.allow_high_risk_local_mps,
    )
    print(json.dumps(serialize_config_value(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
