from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import wandb

from config_utils import load_config_file, serialize_config_value
from paper_training_protocol import apply_paper_dataset_contract, resolve_paper_training_protocol
from paper_training_types import MetalKernelSpec, PaperTrainingProtocol
from powerfoam_metal_trainer import run_training as run_powerfoam_training


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


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


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


def validate_manifest(protocol: PaperTrainingProtocol) -> dict[str, Any]:
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
    record = matches[0]
    checks = {
        "train_cameras": tuple(record.get("train_cameras", ())) == protocol.dataset.train_cameras,
        "heldout_cameras": tuple(record.get("heldout_cameras", ())) == protocol.dataset.heldout_cameras,
        "frame_count_available": int(record.get("frame_count", -1)) >= protocol.dataset.frame_count,
        "fps": float(record.get("fps", -1.0)) == protocol.dataset.fps,
        "start_at_zero": float(record.get("source_start_seconds", -1.0)) == 0.0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"paper manifest contract failed: {', '.join(failed)}")
    scene_dir = resolve_root_path(record["dataset_scene_dir"])
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
        "checks": checks,
        "camera_videos": {camera: display_path(path) for camera, path in camera_paths.items()},
        "source_image_size": record.get("source_image_size"),
        "duration_seconds": record.get("duration_seconds"),
    }


def comparison_command(
    protocol_path: Path,
    protocol: PaperTrainingProtocol,
    seed: int,
    out_dir: Path,
    *,
    backward_policy: str,
    device: str,
    python: str = sys.executable,
) -> list[str]:
    return [
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
        "dataset_lens",
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
        "--out-dir",
        str(out_dir),
    ]


def powerfoam_config(
    raw_protocol: Mapping[str, Any],
    protocol: PaperTrainingProtocol,
    seed: int,
    out_dir: Path,
    *,
    wandb_mode: str,
    worldfoam_initializer: str = DEFAULT_WORLDFOAM_INITIALIZER,
) -> dict[str, Any]:
    cfg = copy.deepcopy(load_config_file(BASE_CONFIG))
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
                "coffee_martini",
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
        return {
            "active_trace_count": int(lane["tube_count"]),
            "tile_trace_pairs_mean": _mean_stat(rows, "uvt_tile_tube_pairs"),
            "per_frame_tile_trace_pairs_mean": _mean_stat(rows, "summed_per_frame_tile_splat_pairs"),
            "effective_pair_ratio_after_fallback_mean": _mean_stat(
                rows, "effective_pair_ratio_after_unstable_fallback"
            ),
            "unstable_tile_fraction_mean": _mean_stat(rows, "unstable_tile_fraction"),
            "overflow_tile_count_mean": _mean_stat(rows, "overflow_tile_count"),
            "metal_buffer_bytes_mean": _mean_stat(rows, "metal_buffer_memory"),
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
        tags=["paper-ablation-v1", "coffee_martini", protocol.name, lane_name, f"seed-{seed}"],
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
        worldfoam_initializer=worldfoam_initializer,
    )
    return {
        "status": "dry_run",
        "protocol_path": display_path(protocol_path),
        "protocol": protocol.as_dict(),
        "manifest_validation": validate_manifest(protocol),
        "kernels": {name: spec.as_dict() for name, spec in specs.items()},
        "comparison_command": comparison_command(
            protocol_path,
            protocol,
            seed,
            seed_dir / "world_tubes_dynamic_3dgs",
            backward_policy=backward_policy,
            device=device,
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
                seed_dir / "world_tubes_dynamic_3dgs" / "comparison_report.json"
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
) -> dict[str, Any]:
    provenance = source_provenance()
    if require_clean_source:
        require_clean_provenance(provenance)
    seed_dir = out_dir / protocol.name / f"seed_{seed}"
    comparison_dir = seed_dir / "world_tubes_dynamic_3dgs"
    worldfoam_dir = seed_dir / "worldfoam"
    comparison_report_path = comparison_dir / "comparison_report.json"
    if not (reuse_existing and comparison_report_path.exists()):
        subprocess.run(
            comparison_command(
                protocol_path,
                protocol,
                seed,
                comparison_dir,
                backward_policy=backward_policy,
                device=device,
            ),
            cwd=ROOT,
            check=True,
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
        run_powerfoam_training(
            powerfoam_config(
                raw_protocol,
                protocol,
                seed,
                worldfoam_dir,
                wandb_mode=wandb_mode,
                worldfoam_initializer=worldfoam_initializer,
            )
        )
    powerfoam_summary = load_json(powerfoam_summary_path)
    powerfoam_best = load_json(worldfoam_dir / "best_metrics.json")
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
        "metrics": powerfoam_best["metrics"],
        "best_metric_name": powerfoam_best["best_metric_name"],
        "best_metric_value": powerfoam_best["best_metric_value"],
        "paper_protocol": powerfoam_summary,
        "evidence": build_lane_evidence(
            "worldfoam",
            {
                "metrics": powerfoam_best["metrics"],
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
    )
    print(json.dumps(serialize_config_value(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
