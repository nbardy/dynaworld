from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

from config_utils import load_config_file, serialize_config_value
from paper_training_protocol import (
    PAPER_DATASET_BUNDLE_SCHEMA_VERSION,
    PAPER_EVALUATOR_SCHEMA_VERSION,
    PAPER_RUNTIME_SCHEMA_VERSION,
    PAPER_RUNTIME_SOURCE_TREE_SCHEMA_VERSION,
    PAPER_SAMPLE_SCHEDULE_ALGORITHM,
    PAPER_SAMPLE_SCHEDULE_SCHEMA_VERSION,
    apply_paper_dataset_contract,
    paper_evaluator_contract,
    resolve_paper_training_protocol,
    validate_paper_runtime_source_tree_identity,
)
from paper_training_types import (
    MetalKernelSpec,
    PaperTrainingProtocol,
    expected_paper_pose_source,
)


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
UVT_WORLD_REPRESENTATIONS = ("legacy_tube", "full_spd4")
DEFAULT_UVT_WORLD_REPRESENTATION = "legacy_tube"
UVT_ALPHA_MODES = ("peak_splat", "beer_lambert")
DEFAULT_UVT_ALPHA_MODE = "peak_splat"
UVT_RENDER_BACKENDS = (
    "metal_tile",
    "retained_fiber_metal",
    "hybrid_retained_fiber",
)
DEFAULT_UVT_RENDER_BACKEND = "metal_tile"
UVT_AMPLITUDE_CONVENTIONS = ("fiber_integrated", "peak_density")
DEFAULT_UVT_AMPLITUDE_CONVENTION = "fiber_integrated"
DEFAULT_UVT_RETAINED_DEPTH_SAMPLES = 48
DEFAULT_UVT_RETAINED_SIGMA_EXTENT = 6.0
DEFAULT_UVT_ORDER_CERTIFICATE_SIGMA = 6.0
DEFAULT_UVT_ORDER_CERTIFICATE_MIN_GAP = 0.0
PAPER_EVIDENCE_SCHEMA_VERSION = 2
GIB = 1024**3
LIVE_RESOURCE_THRESHOLDS = {
    "available_memory_bytes": 10 * GIB,
    "maximum_swap_used_bytes": 2 * GIB,
    "disk_free_bytes": 32 * GIB,
    "maximum_load_1m_per_logical_cpu": 0.75,
}
FROZEN_WORLD_ACCEPTANCE = {
    "image_max_abs_error": 1.0e-5,
    "loss_absolute_delta": 1.0e-5,
    "gradient_global_normalized_l2_error": 1.0e-5,
    "gradient_max_parameter_normalized_l2_error": 1.0e-5,
    "min_world_vjp_l2_norm": 1.0e-12,
    "fallback_fraction": 0.20,
}
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


def uvt_opacity_semantics(
    alpha_mode: str,
    amplitude_convention: str = DEFAULT_UVT_AMPLITUDE_CONVENTION,
) -> str:
    if amplitude_convention not in UVT_AMPLITUDE_CONVENTIONS:
        raise ValueError(
            "unsupported STAR amplitude convention: "
            f"{amplitude_convention}; expected one of "
            f"{UVT_AMPLITUDE_CONVENTIONS}"
        )
    if alpha_mode == "peak_splat":
        if amplitude_convention != "fiber_integrated":
            raise ValueError("peak_splat requires fiber_integrated amplitude")
        return "peak_alpha_amplitude"
    if alpha_mode == "beer_lambert":
        if amplitude_convention == "fiber_integrated":
            return "nonnegative_fiber_integrated_peak_optical_thickness"
        return "nonnegative_world_peak_extinction_density"
    raise ValueError(
        f"unsupported STAR alpha mode: {alpha_mode}; expected one of {UVT_ALPHA_MODES}"
    )


def _validate_uvt_paper_lane_contract(
    protocol: PaperTrainingProtocol,
    *,
    backward_policy: str,
    world_representation: str,
    alpha_mode: str,
    render_backend: str = DEFAULT_UVT_RENDER_BACKEND,
    amplitude_convention: str = DEFAULT_UVT_AMPLITUDE_CONVENTION,
    retained_depth_samples: int = DEFAULT_UVT_RETAINED_DEPTH_SAMPLES,
    retained_sigma_extent: float = DEFAULT_UVT_RETAINED_SIGMA_EXTENT,
    order_certificate_sigma: float = DEFAULT_UVT_ORDER_CERTIFICATE_SIGMA,
    order_certificate_min_gap: float = DEFAULT_UVT_ORDER_CERTIFICATE_MIN_GAP,
    device: str | None = None,
) -> None:
    if world_representation not in UVT_WORLD_REPRESENTATIONS:
        raise ValueError(
            "unsupported World Tubes representation: "
            f"{world_representation}; expected one of {UVT_WORLD_REPRESENTATIONS}"
        )
    if alpha_mode not in UVT_ALPHA_MODES:
        raise ValueError(
            f"unsupported STAR alpha mode: {alpha_mode}; "
            f"expected one of {UVT_ALPHA_MODES}"
        )
    if render_backend not in UVT_RENDER_BACKENDS:
        raise ValueError(
            f"unsupported STAR render backend: {render_backend}; "
            f"expected one of {UVT_RENDER_BACKENDS}"
        )
    if amplitude_convention not in UVT_AMPLITUDE_CONVENTIONS:
        raise ValueError(
            "unsupported STAR amplitude convention: "
            f"{amplitude_convention}; expected one of "
            f"{UVT_AMPLITUDE_CONVENTIONS}"
        )
    if not 1 <= int(retained_depth_samples) <= 64:
        raise ValueError("uvt_retained_depth_samples must lie in [1, 64]")
    if (
        not math.isfinite(float(retained_sigma_extent))
        or float(retained_sigma_extent) <= 0.0
    ):
        raise ValueError("uvt_retained_sigma_extent must be finite and positive")
    if (
        not math.isfinite(float(order_certificate_sigma))
        or float(order_certificate_sigma) < float(retained_sigma_extent)
    ):
        raise ValueError(
            "uvt_order_certificate_sigma must be finite and at least "
            "uvt_retained_sigma_extent"
        )
    if (
        not math.isfinite(float(order_certificate_min_gap))
        or float(order_certificate_min_gap) < 0.0
    ):
        raise ValueError(
            "uvt_order_certificate_min_gap must be finite and nonnegative"
        )
    _camera_projection, camera_sequence_mode, _segment_frames = (
        paper_world_tubes_camera_policy(protocol)
    )
    if alpha_mode == "beer_lambert":
        if world_representation != "full_spd4":
            raise ValueError(
                "Beer-Lambert paper runs are currently scoped to full_spd4"
            )
        if (
            render_backend in {"metal_tile", "hybrid_retained_fiber"}
            and backward_policy != "fast_exploration"
        ):
            raise ValueError(
                "Beer-Lambert paper runs require the validated "
                "fast_exploration direct_atomic+index_add backward path"
            )
        if camera_sequence_mode != "static_view":
            raise ValueError(
                "Beer-Lambert paper runs are currently scoped to static_view "
                "camera sequences"
            )
    if amplitude_convention == "peak_density":
        if world_representation != "full_spd4":
            raise ValueError("peak_density requires full_spd4")
        if alpha_mode != "beer_lambert":
            raise ValueError("peak_density requires beer_lambert")
    if render_backend in {"retained_fiber_metal", "hybrid_retained_fiber"}:
        if world_representation != "full_spd4":
            raise ValueError(f"{render_backend} requires full_spd4")
        if alpha_mode != "beer_lambert":
            raise ValueError(f"{render_backend} requires beer_lambert")
        if device is not None and str(device).lower() != "mps":
            raise ValueError(f"{render_backend} requires local MPS execution")
    if render_backend == "retained_fiber_metal" and backward_policy != "fast_exploration":
        raise ValueError(
            "retained_fiber_metal uses native autograd and accepts only the "
            "runner's fast_exploration policy slot"
        )
    if render_backend == "hybrid_retained_fiber" and backward_policy != "fast_exploration":
        raise ValueError(
            "hybrid_retained_fiber requires fast_exploration "
            "direct_atomic+index_add"
        )
    if (
        world_representation == "full_spd4"
        and camera_sequence_mode
        not in {"static_view", "dynamic_first_order", "projective_first_order"}
    ):
        raise ValueError(
            "full_spd4 has no segmented camera compiler; "
            f"protocol {protocol.name!r} selects {camera_sequence_mode!r}"
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path, *, role: str) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"paper input is missing: {resolved}")
    return {
        "role": str(role),
        "path": display_path(resolved),
        "bytes": int(resolved.stat().st_size),
        "sha256": file_sha256(resolved),
    }


def dataset_input_identity(
    manifest_path: Path,
    record: Mapping[str, Any],
    protocol: PaperTrainingProtocol,
) -> dict[str, Any]:
    """Hash every raw file consumed by the selected paper dataset row."""

    scene_dir = resolve_root_path(str(record["dataset_scene_dir"]))
    dataset = str(record.get("dataset", "")).lower()
    entries = [file_identity(manifest_path, role="manifest")]
    cameras = (*protocol.dataset.train_cameras, *protocol.dataset.heldout_cameras)
    if dataset == "dnerf":
        split_map = record.get("dnerf_camera_splits", {})
        index_map = record.get("dnerf_frame_indices", {})
        for camera in cameras:
            split = split_map.get(camera)
            indices = index_map.get(camera)
            if not isinstance(split, str) or not isinstance(indices, list):
                raise ValueError(
                    f"D-NeRF manifest is missing split/indices for {camera}"
                )
            transforms_path = scene_dir / f"transforms_{split}.json"
            entries.append(
                file_identity(
                    transforms_path,
                    role=f"camera:{camera}:transforms",
                )
            )
            frames = load_json(transforms_path).get("frames", [])
            for position, index in enumerate(
                indices[: protocol.dataset.frame_count]
            ):
                image_path = scene_dir / str(frames[int(index)]["file_path"])
                if not image_path.suffix:
                    image_path = image_path.with_suffix(".png")
                entries.append(
                    file_identity(
                        image_path,
                        role=f"camera:{camera}:frame:{position:06d}",
                    )
                )
    else:
        for camera in cameras:
            entries.append(
                file_identity(
                    scene_dir / f"{camera}.mp4",
                    role=f"camera:{camera}:video",
                )
            )
        if dataset == "neural_3d_video":
            entries.append(
                file_identity(
                    scene_dir / "poses_bounds.npy",
                    role="camera_calibration:poses_bounds",
                )
            )
        elif record.get("models_path"):
            entries.append(
                file_identity(
                    resolve_root_path(str(record["models_path"])),
                    role="camera_calibration:models",
                )
            )
    entries = sorted(entries, key=lambda row: (row["role"], row["path"]))
    payload = {
        "schema_version": 1,
        "dataset": dataset,
        "sample_id": protocol.dataset.sample_id,
        "files": entries,
    }
    return {
        **payload,
        "sha256": hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest(),
    }


def validate_native_extension_identity(native: Mapping[str, Any]) -> None:
    path = Path(str(native.get("path", "")))
    for key in ("sha256", "source_tree_sha256"):
        value = native.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"STAR UVT native extension {key} is invalid")
    if (
        not path.is_absolute()
        or not path.is_file()
        or int(native.get("bytes", 0)) != path.stat().st_size
        or native["sha256"] != file_sha256(path)
        or int(native.get("source_file_count", 0)) <= 0
    ):
        raise ValueError("STAR UVT native extension provenance is invalid")
    validate_route_native_extension_identity("world_tubes", native)


def validate_route_native_extension_identity(
    lane_name: str,
    native: Any,
) -> None:
    if not isinstance(native, Mapping):
        raise ValueError(f"{lane_name} route-native extension identity is missing")
    path = Path(str(native.get("path", "")))
    sha256 = native.get("sha256")
    if (
        not isinstance(native.get("module"), str)
        or not native["module"]
        or not path.is_absolute()
        or not path.is_file()
        or int(native.get("bytes", 0)) != path.stat().st_size
        or not isinstance(sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
        or sha256 != file_sha256(path)
    ):
        raise ValueError(f"{lane_name} route-native extension provenance is invalid")
    source_tree = native.get("runtime_source_tree")
    validate_hashed_contract(
        f"{lane_name} route-native runtime source tree",
        source_tree,
        schema_version=PAPER_RUNTIME_SOURCE_TREE_SCHEMA_VERSION,
    )
    try:
        validate_paper_runtime_source_tree_identity(source_tree)
    except (FileNotFoundError, ValueError) as error:
        raise ValueError(
            f"{lane_name} route-native runtime source provenance drifted"
        ) from error


def validate_hashed_contract(
    name: str,
    contract: Any,
    *,
    schema_version: int,
) -> None:
    if not isinstance(contract, Mapping):
        raise ValueError(f"{name} contract is missing")
    if int(contract.get("schema_version", -1)) != int(schema_version):
        raise ValueError(f"{name} contract schema is missing or stale")
    reported_sha256 = contract.get("sha256")
    payload = {key: value for key, value in contract.items() if key != "sha256"}
    expected_sha256 = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    if reported_sha256 != expected_sha256:
        raise ValueError(f"{name} contract digest does not match its payload")


def _expected_wandb_file(run_dir: str | Path, run_id: str) -> Path:
    candidate = Path(str(run_dir))
    resolved_run_dir = candidate.resolve()
    if (
        not candidate.is_absolute()
        or resolved_run_dir.name != "files"
        or not resolved_run_dir.is_dir()
    ):
        raise FileNotFoundError(
            f"W&B run directory is missing for {run_id}: {resolved_run_dir}"
        )
    return resolved_run_dir.parent / f"run-{run_id}.wandb"


def _validate_wandb_run_file_identity(
    artifact: Any,
    *,
    run_dir: str | Path,
    run_id: str,
) -> None:
    if not isinstance(artifact, Mapping):
        raise ValueError("W&B run-file identity is missing")
    expected = _expected_wandb_file(run_dir, run_id)
    path = Path(str(artifact.get("path", "")))
    if (
        artifact.get("role") != "wandb_run_file"
        or not path.is_absolute()
        or path.resolve() != expected.resolve()
        or path.name != f"run-{run_id}.wandb"
        or path.parent.resolve() != Path(run_dir).resolve().parent
        or not path.is_file()
        or int(artifact.get("bytes", -1)) != path.stat().st_size
        or artifact.get("sha256") != file_sha256(path)
    ):
        raise ValueError("W&B run-file identity is invalid")


def wandb_file_identity(run_dir: str | Path, run_id: str) -> dict[str, Any]:
    expected = _expected_wandb_file(run_dir, run_id)
    if not expected.is_file():
        raise FileNotFoundError(
            f"expected exact W&B run artifact {expected.name} under "
            f"{expected.parent}"
        )
    identity = file_identity(expected, role="wandb_run_file")
    identity["path"] = str(expected.resolve())
    return identity


def validate_wandb_identity(
    identity: Any,
    *,
    run_id: str,
    mode: str,
    source_digest: str,
    report_digest: str,
    config_digest: str,
) -> None:
    if not isinstance(identity, Mapping):
        raise ValueError("W&B execution identity is missing")
    expected = {
        "run_id": run_id,
        "mode": mode,
        "source_digest": source_digest,
        "comparison_report_sha256": report_digest,
        "config_sha256": config_digest,
    }
    drift = [
        key for key, value in expected.items() if identity.get(key) != value
    ]
    artifact = identity.get("run_file")
    if drift or not isinstance(artifact, Mapping):
        raise ValueError(
            "W&B execution identity drifted"
            + (f": {', '.join(drift)}" if drift else "")
        )
    try:
        _validate_wandb_run_file_identity(
            artifact,
            run_dir=str(identity.get("run_dir", "")),
            run_id=run_id,
        )
    except (FileNotFoundError, ValueError) as error:
        raise ValueError("W&B run-file identity is invalid") from error


def finalize_worldfoam_wandb_identity(
    identity_path: Path,
    *,
    expected_run_id: str,
    expected_mode: str,
    source: Mapping[str, Any],
    paper_summary_path: Path,
    resolved_config_path: Path,
) -> dict[str, Any]:
    identity = load_json(identity_path)
    source_digest = hashlib.sha256(
        json.dumps(
            serialize_config_value(dict(source)),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    expected = {
        "run_id": expected_run_id,
        "mode": expected_mode,
        "source_digest": source_digest,
        "paper_protocol_summary_sha256": file_sha256(paper_summary_path),
        "resolved_config_sha256": file_sha256(resolved_config_path),
    }
    if identity.get("finalized") is True:
        drift = [
            key for key, value in expected.items() if identity.get(key) != value
        ]
        if drift:
            raise ValueError(
                "WorldFoam W&B identity drifted: " + ", ".join(drift)
            )
        artifact = identity.get("run_file")
        try:
            _validate_wandb_run_file_identity(
                artifact,
                run_dir=str(identity.get("run_dir", "")),
                run_id=expected_run_id,
            )
        except (FileNotFoundError, ValueError) as error:
            raise ValueError(
                "WorldFoam W&B run-file identity is invalid"
            ) from error
        return identity
    if (
        identity.get("run_id") != expected_run_id
        or identity.get("mode") != expected_mode
        or not str(identity.get("run_dir", "")).strip()
    ):
        raise ValueError("WorldFoam provisional W&B identity drifted")
    finalized = {
        **identity,
        **expected,
        "finalized": True,
        "run_file": wandb_file_identity(
            str(identity["run_dir"]),
            expected_run_id,
        ),
    }
    write_json(identity_path, finalized)
    return finalized


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
    invalid_dirty = [
        key
        for key in ("repository_dirty", "star_uvt_dirty")
        if provenance.get(key) is not False
    ]
    if invalid_dirty:
        raise RuntimeError(
            "paper submission runs require literal clean source flags; "
            f"invalid dirty flags: {invalid_dirty}"
        )
    invalid_commits = [
        key
        for key in ("repository_commit", "star_uvt_commit")
        if not isinstance(provenance.get(key), str)
        or len(provenance[key]) != 40
        or any(character not in "0123456789abcdef" for character in provenance[key])
    ]
    if invalid_commits:
        raise RuntimeError(
            "paper submission runs require exact 40-character commit hashes; "
            f"invalid commits: {invalid_commits}"
        )


def host_physical_memory_bytes() -> int:
    if sys.platform == "darwin":
        try:
            return int(
                subprocess.check_output(
                    ("sysctl", "-n", "hw.memsize"),
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            )
        except (OSError, subprocess.CalledProcessError, ValueError):
            # Sandboxed macOS processes can be denied sysctl even though the
            # POSIX page-count interface remains available.  Keep the guard
            # fail-closed, but use the independent exact page-count route
            # before declaring the host unknowable.
            pass
    if hasattr(os, "sysconf"):
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        page_count = int(os.sysconf("SC_PHYS_PAGES"))
        host_bytes = page_size * page_count
        if page_size > 0 and page_count > 0 and host_bytes > 0:
            return host_bytes
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


def _parse_swap_bytes(output: str) -> int:
    match = re.search(r"used\s*=\s*([0-9.]+)([KMG])", output)
    if match is None:
        raise ValueError("could not parse macOS swap usage")
    multiplier = {"K": 1024, "M": 1024**2, "G": GIB}[match.group(2)]
    return int(float(match.group(1)) * multiplier)


def live_resource_snapshot() -> dict[str, Any]:
    logical_cpu_count = int(os.cpu_count() or 1)
    load_1m, load_5m, load_15m = os.getloadavg()
    snapshot: dict[str, Any] = {
        "disk_free_bytes": int(shutil.disk_usage(ROOT).free),
        "platform": sys.platform,
        "logical_cpu_count": logical_cpu_count,
        "load_average_1m": float(load_1m),
        "load_average_5m": float(load_5m),
        "load_average_15m": float(load_15m),
        "load_1m_per_logical_cpu": float(load_1m) / logical_cpu_count,
    }
    if sys.platform != "darwin":
        return snapshot
    vm_output = subprocess.check_output(("vm_stat",), text=True)
    page_match = re.search(r"page size of\s+(\d+)\s+bytes", vm_output)
    if page_match is None:
        raise ValueError("could not parse macOS VM page size")
    page_size = int(page_match.group(1))
    page_counts: dict[str, int] = {}
    for line in vm_output.splitlines():
        match = re.match(r"([^:]+):\s+([0-9.]+)\.?$", line.strip())
        if match is not None:
            page_counts[match.group(1)] = int(float(match.group(2)))
    snapshot["available_memory_bytes"] = page_size * sum(
        page_counts.get(key, 0)
        for key in ("Pages free", "Pages inactive", "Pages speculative")
    )
    try:
        swap_output = subprocess.check_output(
            ("sysctl", "-n", "vm.swapusage"),
            text=True,
            stderr=subprocess.DEVNULL,
        )
        snapshot["swap_used_bytes"] = _parse_swap_bytes(swap_output)
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        # Unknown swap pressure is unsafe, never evidence of a healthy host.
        # Preserve a serializable reason and force the normal resource gate to
        # reject execution instead of leaking a platform subprocess traceback.
        snapshot["swap_used_bytes"] = sys.maxsize
        snapshot["swap_probe_error"] = type(error).__name__
    return snapshot


def require_live_resources(snapshot: Mapping[str, Any]) -> None:
    if snapshot.get("platform") != "darwin":
        raise RuntimeError("MPS paper execution requires a macOS resource audit")
    failures = []
    if int(snapshot.get("available_memory_bytes", 0)) < int(
        LIVE_RESOURCE_THRESHOLDS["available_memory_bytes"]
    ):
        failures.append("available_memory_bytes")
    if int(snapshot.get("swap_used_bytes", sys.maxsize)) > int(
        LIVE_RESOURCE_THRESHOLDS["maximum_swap_used_bytes"]
    ):
        failures.append("swap_used_bytes")
    if int(snapshot.get("disk_free_bytes", 0)) < int(
        LIVE_RESOURCE_THRESHOLDS["disk_free_bytes"]
    ):
        failures.append("disk_free_bytes")
    if float(snapshot.get("load_1m_per_logical_cpu", math.inf)) > float(
        LIVE_RESOURCE_THRESHOLDS["maximum_load_1m_per_logical_cpu"]
    ):
        failures.append("load_1m_per_logical_cpu")
    if failures:
        raise RuntimeError(
            "live resource gate rejected MPS paper execution: "
            + ", ".join(failures)
        )


def require_execution_safety_acknowledgement(
    protocol: PaperTrainingProtocol,
    *,
    device: str,
    allow_local_mps_execution: bool,
    allow_high_risk_local_mps: bool,
) -> dict[str, Any]:
    estimate = local_mps_safety_estimate(protocol)
    if str(device).lower() != "mps":
        return {
            **estimate,
            "live_resources": None,
            "live_resource_thresholds": LIVE_RESOURCE_THRESHOLDS,
        }
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
    live_resources = live_resource_snapshot()
    require_live_resources(live_resources)
    return {
        **estimate,
        "live_resources": live_resources,
        "live_resource_thresholds": LIVE_RESOURCE_THRESHOLDS,
    }


def effective_uvt_backward_policy(
    render_backend: str,
    backward_policy: str,
) -> str:
    """Map the runner policy slot to the benchmark's backend-specific selector."""

    if render_backend == "retained_fiber_metal":
        return "manual"
    return backward_policy


def kernel_specs(
    backward_policy: str,
    *,
    uvt_render_backend: str = DEFAULT_UVT_RENDER_BACKEND,
) -> dict[str, MetalKernelSpec]:
    uvt_backward = {
        "fast_exploration": ("direct_atomic+index_add", False),
        "deterministic_quality": ("tile_pair+key_sort_scan_metal", True),
        "deterministic_compact": ("compact_tile_pair+key_sort_scan_metal", True),
    }
    if backward_policy not in uvt_backward:
        raise ValueError(f"unsupported World Tubes backward policy: {backward_policy}")
    if uvt_render_backend not in UVT_RENDER_BACKENDS:
        raise ValueError(f"unsupported STAR render backend: {uvt_render_backend}")
    backward, deterministic = uvt_backward[backward_policy]
    forward = f"{uvt_render_backend}_selected_time"
    if uvt_render_backend == "retained_fiber_metal":
        backward = "retained_fiber_metal_autograd"
        deterministic = False
    elif uvt_render_backend == "hybrid_retained_fiber":
        backward = "direct_atomic+index_add+retained_fiber_metal_autograd"
        deterministic = False
    return {
        "world_tubes": MetalKernelSpec(
            representation="world_tubes",
            family="star_uvt",
            forward=forward,
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
    dataset_family = str(record.get("dataset", "")).strip().lower()
    expected_pose_source = expected_paper_pose_source(dataset_family)
    is_dnerf = dataset_family == "dnerf"
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
        "dataset": dataset_family,
        "expected_pose_source": expected_pose_source,
        "checks": checks,
        "camera_inputs": {camera: display_path(path) for camera, path in camera_paths.items()},
        "camera_videos": (
            None if is_dnerf else {camera: display_path(path) for camera, path in camera_paths.items()}
        ),
        "source_image_size": record.get("source_image_size"),
        "duration_seconds": record.get("duration_seconds"),
        "sample_layout": record.get("sample_layout", "synchronized_multicamera"),
        "input_identity": dataset_input_identity(
            manifest_path,
            record,
            protocol,
        ),
    }


def validate_comparison_pose_source(
    meta: Mapping[str, Any],
    manifest_validation: Mapping[str, Any],
) -> str:
    """Bind decoded camera semantics to the validated manifest family."""

    dataset_family = manifest_validation.get("dataset")
    expected = expected_paper_pose_source(dataset_family)
    if manifest_validation.get("expected_pose_source") != expected:
        raise ValueError(
            "validated paper manifest pose-source contract drifted: "
            f"dataset {dataset_family!r} requires {expected!r}"
        )
    input_identity = manifest_validation.get("input_identity")
    if (
        not isinstance(input_identity, Mapping)
        or input_identity.get("dataset") != dataset_family
    ):
        raise ValueError(
            "validated paper manifest dataset family does not match its raw "
            "input identity"
        )
    reported = meta.get("pose_source")
    if reported != expected:
        raise ValueError(
            "comparison report pose source does not match the validated "
            f"{dataset_family!r} manifest: expected {expected!r}, got {reported!r}"
        )
    dataset_bundle = meta.get("paper_dataset_bundle")
    if (
        not isinstance(dataset_bundle, Mapping)
        or dataset_bundle.get("pose_source") != expected
    ):
        raise ValueError(
            "decoded paper dataset bundle pose source does not match the "
            f"validated {dataset_family!r} manifest: expected {expected!r}"
        )
    return expected


def comparison_command(
    protocol_path: Path,
    protocol: PaperTrainingProtocol,
    seed: int,
    out_dir: Path,
    *,
    backward_policy: str,
    device: str,
    uvt_world_representation: str = DEFAULT_UVT_WORLD_REPRESENTATION,
    uvt_alpha_mode: str = DEFAULT_UVT_ALPHA_MODE,
    uvt_render_backend: str = DEFAULT_UVT_RENDER_BACKEND,
    uvt_amplitude_convention: str = DEFAULT_UVT_AMPLITUDE_CONVENTION,
    uvt_retained_depth_samples: int = DEFAULT_UVT_RETAINED_DEPTH_SAMPLES,
    uvt_retained_sigma_extent: float = DEFAULT_UVT_RETAINED_SIGMA_EXTENT,
    uvt_order_certificate_sigma: float = DEFAULT_UVT_ORDER_CERTIFICATE_SIGMA,
    uvt_order_certificate_min_gap: float = DEFAULT_UVT_ORDER_CERTIFICATE_MIN_GAP,
    uvt_spd4_init_precision_z: float | None = None,
    frozen_world_replay_compiled: bool = False,
    frozen_world_max_frames: int = 0,
    only_lane: str = "combined",
    allow_local_mps_execution: bool = False,
    python: str = sys.executable,
) -> list[str]:
    if only_lane not in {"combined", *LANE_REPORT_KEYS}:
        raise ValueError(f"unsupported comparison lane: {only_lane}")
    _validate_uvt_paper_lane_contract(
        protocol,
        backward_policy=backward_policy,
        world_representation=uvt_world_representation,
        alpha_mode=uvt_alpha_mode,
        render_backend=uvt_render_backend,
        amplitude_convention=uvt_amplitude_convention,
        retained_depth_samples=uvt_retained_depth_samples,
        retained_sigma_extent=uvt_retained_sigma_extent,
        order_certificate_sigma=uvt_order_certificate_sigma,
        order_certificate_min_gap=uvt_order_certificate_min_gap,
        device=device,
    )
    if (
        uvt_spd4_init_precision_z is not None
        and uvt_spd4_init_precision_z <= 0.0
    ):
        raise ValueError("uvt_spd4_init_precision_z must be positive when provided")
    if frozen_world_max_frames < 0:
        raise ValueError("frozen_world_max_frames must be nonnegative")
    if frozen_world_max_frames and not frozen_world_replay_compiled:
        raise ValueError(
            "frozen_world_max_frames requires frozen_world_replay_compiled"
        )
    if frozen_world_replay_compiled and (
        uvt_world_representation != "legacy_tube"
        or uvt_alpha_mode != "peak_splat"
        or uvt_render_backend != "metal_tile"
    ):
        raise ValueError(
            "frozen replay/compiled comparison requires "
            "legacy_tube + peak_splat + metal_tile"
        )
    camera_rig_init = paper_camera_rig_init(protocol)
    camera_projection, camera_sequence_mode, segment_frames = paper_world_tubes_camera_policy(protocol)
    if frozen_world_replay_compiled and camera_sequence_mode != "static_view":
        raise ValueError(
            "frozen replay/compiled comparison currently requires a static-view protocol"
        )
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
        "--uvt-world-representation",
        uvt_world_representation,
        "--uvt-alpha-mode",
        uvt_alpha_mode,
        "--uvt-amplitude-convention",
        uvt_amplitude_convention,
        "--uvt-render-backend",
        uvt_render_backend,
        "--uvt-retained-depth-samples",
        str(uvt_retained_depth_samples),
        "--uvt-retained-sigma-extent",
        str(uvt_retained_sigma_extent),
        "--uvt-order-certificate-sigma",
        str(uvt_order_certificate_sigma),
        "--uvt-order-certificate-min-gap",
        str(uvt_order_certificate_min_gap),
        "--uvt-backward-policy",
        effective_uvt_backward_policy(uvt_render_backend, backward_policy),
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
    if uvt_spd4_init_precision_z is not None:
        command.extend(
            (
                "--uvt-spd4-init-precision-z",
                str(uvt_spd4_init_precision_z),
            )
        )
    if frozen_world_replay_compiled and only_lane in {"combined", "world_tubes"}:
        command.append("--frozen-world-replay-compiled")
        if frozen_world_max_frames:
            command.extend(
                ("--frozen-world-max-frames", str(frozen_world_max_frames))
            )
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
    uvt_world_representation: str = DEFAULT_UVT_WORLD_REPRESENTATION,
    uvt_alpha_mode: str = DEFAULT_UVT_ALPHA_MODE,
    uvt_render_backend: str = DEFAULT_UVT_RENDER_BACKEND,
    uvt_amplitude_convention: str = DEFAULT_UVT_AMPLITUDE_CONVENTION,
    uvt_retained_depth_samples: int = DEFAULT_UVT_RETAINED_DEPTH_SAMPLES,
    uvt_retained_sigma_extent: float = DEFAULT_UVT_RETAINED_SIGMA_EXTENT,
    uvt_order_certificate_sigma: float = DEFAULT_UVT_ORDER_CERTIFICATE_SIGMA,
    uvt_order_certificate_min_gap: float = DEFAULT_UVT_ORDER_CERTIFICATE_MIN_GAP,
    uvt_spd4_init_precision_z: float | None = None,
    frozen_world_replay_compiled: bool = False,
    frozen_world_max_frames: int = 0,
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
            uvt_world_representation=uvt_world_representation,
            uvt_alpha_mode=uvt_alpha_mode,
            uvt_render_backend=uvt_render_backend,
            uvt_amplitude_convention=uvt_amplitude_convention,
            uvt_retained_depth_samples=uvt_retained_depth_samples,
            uvt_retained_sigma_extent=uvt_retained_sigma_extent,
            uvt_order_certificate_sigma=uvt_order_certificate_sigma,
            uvt_order_certificate_min_gap=uvt_order_certificate_min_gap,
            uvt_spd4_init_precision_z=uvt_spd4_init_precision_z,
            frozen_world_replay_compiled=frozen_world_replay_compiled,
            frozen_world_max_frames=frozen_world_max_frames,
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
    run_identity = {
        "protocol": protocol.as_dict(),
        "seed": int(seed),
        "lane": "worldfoam",
        "device": str(device),
        "wandb_mode": str(wandb_mode),
        "output_dir": str(out_dir.resolve()),
        "initializer": str(worldfoam_initializer),
        "evidence_schema_version": PAPER_EVIDENCE_SCHEMA_VERSION,
    }
    run_hash = hashlib.sha256(
        json.dumps(
            serialize_config_value(run_identity),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:12]
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
            "wandb_resume": "never",
            "wandb_project": "dynaworld",
            "wandb_disable_git": True,
            "wandb_disable_code": True,
            "wandb_run_name": f"paper-{protocol.name}-worldfoam-seed{seed}",
            "wandb_tags": [
                "paper-ablation-v2",
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


def worldfoam_resolved_config_binding(
    expected_config: Mapping[str, Any],
    resolved_config_path: Path,
) -> dict[str, Any]:
    """Bind the trainer-written config to every runner-owned config value."""

    expected = serialize_config_value(dict(expected_config))
    resolved = load_json(resolved_config_path)
    drift: list[str] = []
    matched_leaf_count = 0

    def compare_subset(expected_value: Any, actual_value: Any, path: str) -> None:
        nonlocal matched_leaf_count
        if isinstance(expected_value, Mapping):
            if not isinstance(actual_value, Mapping):
                drift.append(path or "<root>")
                return
            for key, value in expected_value.items():
                child_path = f"{path}.{key}" if path else str(key)
                if key not in actual_value:
                    drift.append(child_path)
                    continue
                compare_subset(value, actual_value[key], child_path)
            return
        matched_leaf_count += 1
        if actual_value != expected_value:
            drift.append(path)

    compare_subset(expected, resolved, "")
    required_physical_fields = {
        "render.background_mode": "fixed",
        "render.background": [0.0, 0.0, 0.0],
        "render.eval_color_calibration": "none",
    }
    for dotted_path, expected_value in required_physical_fields.items():
        actual_value: Any = resolved
        for key in dotted_path.split("."):
            if not isinstance(actual_value, Mapping) or key not in actual_value:
                actual_value = None
                break
            actual_value = actual_value[key]
        if actual_value != expected_value:
            drift.append(dotted_path)
    if drift:
        raise ValueError(
            "WorldFoam resolved config drifted from the runner contract: "
            + ", ".join(sorted(set(drift)))
        )
    expected_digest = hashlib.sha256(
        json.dumps(expected, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    payload = {
        "schema_version": 1,
        "expected_config_sha256": expected_digest,
        "resolved_config_sha256": file_sha256(resolved_config_path),
        "matched_expected_leaf_count": matched_leaf_count,
        "required_physical_fields": required_physical_fields,
    }
    return {
        **payload,
        "sha256": hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest(),
    }


def powerfoam_initializer_identity(
    cfg: Mapping[str, Any],
    *,
    requested_initializer: str,
) -> dict[str, Any]:
    raw_path = cfg["model"]["init_point_cloud_path"]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "requested_initializer": str(requested_initializer),
        "init_from_video": bool(cfg["model"]["init_from_video"]),
        "coordinate_frame": cfg["model"][
            "init_point_cloud_coordinate_frame"
        ],
        "normalize": cfg["model"]["init_point_cloud_normalize"],
        "visibility_filter": cfg["model"][
            "init_point_cloud_visibility_filter"
        ],
        "file": (
            None
            if raw_path is None
            else file_identity(
                resolve_root_path(str(raw_path)),
                role="worldfoam_initializer",
            )
        ),
    }
    return {
        **payload,
        "sha256": hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest(),
    }


def validate_lane_cost(
    lane_name: str,
    lane: Mapping[str, Any],
    protocol: PaperTrainingProtocol,
    *,
    seed: int | None = None,
) -> None:
    if int(lane["steps"]) != protocol.steps:
        raise ValueError(f"{lane_name} completed {lane['steps']} of {protocol.steps} required steps")
    paper = lane.get("paper_protocol")
    if not isinstance(paper, Mapping) or not bool(paper.get("enabled", False)):
        raise ValueError(f"{lane_name} did not report an enabled paper protocol")
    expected_sampling = {
        "mode": "spacetime_epoch",
        "same_time_count": protocol.same_time_count,
        "local_time_count": protocol.local_time_count,
        "local_time_radius": protocol.local_time_radius,
    }
    if paper.get("sampling") != expected_sampling:
        raise ValueError(f"{lane_name} sampling contract does not match the protocol")
    if paper.get("stages") != [stage.as_dict() for stage in protocol.stages]:
        raise ValueError(f"{lane_name} stage schedule does not match the protocol")
    sample_schedule = paper.get("sample_schedule")
    if not isinstance(sample_schedule, Mapping):
        raise ValueError(f"{lane_name} did not report the consumed sample schedule")
    if int(sample_schedule.get("schema_version", -1)) != (
        PAPER_SAMPLE_SCHEDULE_SCHEMA_VERSION
    ):
        raise ValueError(f"{lane_name} sample schedule schema is missing or stale")
    if sample_schedule.get("algorithm") != PAPER_SAMPLE_SCHEDULE_ALGORITHM:
        raise ValueError(f"{lane_name} sample schedule algorithm drifted")
    if int(sample_schedule.get("record_count", -1)) != protocol.steps:
        raise ValueError(f"{lane_name} sample schedule step count does not match the protocol")
    schedule_sha256 = sample_schedule.get("sha256")
    if (
        not isinstance(schedule_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", schedule_sha256) is None
    ):
        raise ValueError(f"{lane_name} sample schedule digest is invalid")
    if seed is not None:
        expected_sampler_seed = int(seed) + protocol.sampler_seed_offset
        if int(sample_schedule.get("sampler_seed", -1)) != expected_sampler_seed:
            raise ValueError(f"{lane_name} sample schedule seed does not match the protocol")
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
        nonnumeric = [
            key
            for key in keys
            if isinstance(values[key], bool)
            or not isinstance(values[key], Real)
        ]
        if nonnumeric:
            raise ValueError(
                f"{lane_name} paper evidence {section} is not real numeric: "
                + ", ".join(nonnumeric)
            )
        nonfinite = [
            key
            for key in keys
            if not math.isfinite(float(values[key]))
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


_FROZEN_WORLD_CHECKPOINT_SCHEMA_VERSION = 1
_FROZEN_WORLD_TEMPORAL_SAMPLING = (
    "ordered_full_interval_integer_lattice_v1"
)
_FROZEN_WORLD_CHECKPOINT_METADATA_KEYS = (
    "representation",
    "frame_count",
    "active_tube_count",
    "tube_count",
    "alpha_mode",
    "amplitude_convention",
    "min_precision_xy",
    "min_lambda_t",
    "parameter_names",
)
_FROZEN_WORLD_LEGACY_PARAMETER_NAMES = (
    "x0",
    "velocity",
    "raw_precision_xy",
    "raw_lambda_t",
    "raw_opacity",
    "raw_color",
    "t0",
)


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _frozen_world_full_interval_frame_indices(
    full_frames: int,
    frame_count: int,
) -> tuple[int, ...]:
    if full_frames < 1 or frame_count < 1 or frame_count > full_frames:
        raise ValueError(
            "frozen-world sampled frame count must be in [1, full_frames]"
        )
    if frame_count == 1:
        if full_frames > 1:
            raise ValueError(
                "frozen-world full-interval sampling requires at least two frames"
            )
        return (0,)
    denominator = frame_count - 1
    indices = tuple(
        (
            sample * (full_frames - 1) + denominator // 2
        )
        // denominator
        for sample in range(frame_count)
    )
    if len(set(indices)) != frame_count:
        raise ValueError("frozen-world full-interval time grid is not unique")
    return indices


def _frozen_world_sequence_sha256(
    values: tuple[int | float, ...],
) -> str:
    return hashlib.sha256(
        json.dumps(list(values), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_frozen_world_checkpoint_payload(
    checkpoint_path: Path,
    checkpoint: Mapping[str, Any],
) -> str:
    """Bind semantic checkpoint contents to the JSON evidence contract.

    The producer hashes canonical metadata followed by each named tensor in
    sorted-name order. Loading is CPU-only and weights-only; tensor bytes are
    fed to the digest in bounded chunks after reproducing the producer's
    contiguous C-order conversion.
    """

    try:
        import torch
    except Exception as exc:  # pragma: no cover - exercised only in broken envs
        raise ValueError(
            "frozen-world checkpoint validation requires PyTorch"
        ) from exc
    try:
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except Exception as exc:
        raise ValueError(
            "frozen-world checkpoint could not be loaded safely on CPU"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError("frozen-world checkpoint payload must be a mapping")
    expected_payload_keys = {
        "schema_version",
        *_FROZEN_WORLD_CHECKPOINT_METADATA_KEYS,
        "world_state_sha256",
        "state_dict",
    }
    if set(payload) != expected_payload_keys:
        raise ValueError("frozen-world checkpoint payload fields drifted")
    if (
        isinstance(payload.get("schema_version"), bool)
        or payload.get("schema_version")
        != _FROZEN_WORLD_CHECKPOINT_SCHEMA_VERSION
    ):
        raise ValueError("frozen-world checkpoint payload schema is stale")

    for key in _FROZEN_WORLD_CHECKPOINT_METADATA_KEYS:
        loaded_value = payload[key]
        reported_value = checkpoint.get(key)
        if (
            type(loaded_value) is not type(reported_value)
            or loaded_value != reported_value
        ):
            raise ValueError(
                f"frozen-world checkpoint metadata does not match report: {key}"
            )
    metadata = {
        key: payload[key] for key in _FROZEN_WORLD_CHECKPOINT_METADATA_KEYS
    }
    if any(
        type(metadata[key]) is not int
        for key in (
            "frame_count",
            "active_tube_count",
            "tube_count",
        )
    ) or any(
        type(metadata[key]) is not float
        for key in ("min_precision_xy", "min_lambda_t")
    ):
        raise ValueError("frozen-world checkpoint metadata types are invalid")
    parameter_names = metadata["parameter_names"]
    if (
        not isinstance(parameter_names, list)
        or not parameter_names
        or any(not isinstance(name, str) or not name for name in parameter_names)
        or len(set(parameter_names)) != len(parameter_names)
    ):
        raise ValueError("frozen-world checkpoint parameter names are invalid")
    state = payload["state_dict"]
    if not isinstance(state, Mapping) or set(state) != set(parameter_names):
        raise ValueError(
            "frozen-world checkpoint state_dict does not match parameter names"
        )
    if parameter_names != list(_FROZEN_WORLD_LEGACY_PARAMETER_NAMES):
        raise ValueError(
            "frozen-world checkpoint parameter schema is invalid"
        )
    if (
        isinstance(checkpoint.get("parameter_tensor_count"), bool)
        or checkpoint.get("parameter_tensor_count") != len(state)
    ):
        raise ValueError(
            "frozen-world checkpoint tensor count does not match state_dict"
        )
    tube_count = int(metadata["tube_count"])
    expected_shapes = {
        "x0": (tube_count, 3),
        "velocity": (tube_count, 3),
        "raw_precision_xy": (tube_count, 2),
        "raw_lambda_t": (tube_count,),
        "raw_opacity": (tube_count,),
        "raw_color": (tube_count, 3),
        "t0": (tube_count,),
    }
    invalid_tensor_schema = [
        name
        for name, expected_shape in expected_shapes.items()
        if not isinstance(state[name], torch.Tensor)
        or state[name].dtype != torch.float32
        or tuple(state[name].shape) != expected_shape
    ]
    if invalid_tensor_schema:
        raise ValueError(
            "frozen-world checkpoint tensor schema is invalid: "
            + ", ".join(invalid_tensor_schema)
        )

    digest = hashlib.sha256()
    digest.update(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    chunk_bytes = 1024 * 1024
    for name in sorted(state):
        value = state[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.layout != torch.strided
            or value.is_quantized
            or value.device.type != "cpu"
            or not value.is_contiguous()
        ):
            raise ValueError(
                f"frozen-world checkpoint state tensor is unsupported: {name}"
            )
        tensor = value.detach()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(json.dumps(list(tensor.shape)).encode("utf-8"))
        flat = tensor.reshape(-1)
        elements_per_chunk = max(1, chunk_bytes // max(1, tensor.element_size()))
        try:
            for start in range(0, flat.numel(), elements_per_chunk):
                digest.update(
                    flat[start : start + elements_per_chunk]
                    .numpy()
                    .tobytes(order="C")
                )
        except Exception as exc:
            raise ValueError(
                f"frozen-world checkpoint state tensor cannot be hashed: {name}"
            ) from exc
    recomputed_world_state_sha = digest.hexdigest()
    payload_world_state_sha = payload.get("world_state_sha256")
    reported_world_state_sha = checkpoint.get("world_state_sha256")
    if (
        not _valid_sha256(payload_world_state_sha)
        or not _valid_sha256(reported_world_state_sha)
        or payload_world_state_sha != reported_world_state_sha
    ):
        raise ValueError(
            "frozen-world checkpoint world-state hash does not match report"
        )
    if recomputed_world_state_sha != payload_world_state_sha:
        raise ValueError(
            "frozen-world checkpoint world-state SHA-256 does not match contents"
        )
    return recomputed_world_state_sha


def validate_frozen_world_evidence(
    frozen: Mapping[str, Any],
    *,
    expected_frames: int,
    expected_full_frames: int | None = None,
    expected_image_size: tuple[int, int] | None = None,
    expected_heldout_camera: str | None = None,
    expected_active_tubes: int | None = None,
) -> None:
    if int(frozen.get("schema_version", -1)) != 2:
        raise ValueError("frozen-world evidence schema is missing or stale")
    required_sections = (
        "checkpoint",
        "world_state",
        "loss",
        "image",
        "gradient",
        "timing_s",
        "payload_bytes",
        "atlas",
        "contract",
        "contract_hashes",
        "acceptance",
        "checks",
    )
    missing = [
        key for key in required_sections if not isinstance(frozen.get(key), Mapping)
    ]
    if missing:
        raise ValueError(
            "frozen-world evidence is missing sections: " + ", ".join(missing)
        )
    if frozen.get("status") != "complete":
        raise ValueError("frozen-world evidence is incomplete")
    if int(frozen.get("frame_count", 0)) != int(expected_frames):
        raise ValueError("frozen-world evidence frame count drifted")
    if (
        expected_full_frames is not None
        and int(frozen.get("full_dataset_frame_count", 0))
        != int(expected_full_frames)
    ):
        raise ValueError("frozen-world full dataset frame count drifted")
    if (
        expected_image_size is not None
        and tuple(frozen.get("image_size", ())) != tuple(expected_image_size)
    ):
        raise ValueError("frozen-world image size drifted")
    if (
        expected_heldout_camera is not None
        and frozen.get("heldout_camera") != expected_heldout_camera
    ):
        raise ValueError("frozen-world heldout camera drifted")
    full_frames = int(
        expected_full_frames
        if expected_full_frames is not None
        else frozen.get("full_dataset_frame_count", 0)
    )
    expected_frame_indices = _frozen_world_full_interval_frame_indices(
        full_frames,
        expected_frames,
    )
    expected_centered_times = tuple(
        float(frame) - 0.5 * float(full_frames - 1)
        for frame in expected_frame_indices
    )
    if (
        frozen.get("temporal_sampling")
        != _FROZEN_WORLD_TEMPORAL_SAMPLING
        or tuple(frozen.get("frame_indices", ()))
        != expected_frame_indices
        or tuple(frozen.get("centered_frame_times", ()))
        != expected_centered_times
    ):
        raise ValueError("frozen-world fixed-program time grid drifted")
    for key in (
        "same_checkpoint",
        "same_heldout_camera",
        "same_target_frames",
        "same_loss",
        "same_precision",
        "same_alpha_mode",
        "bounded_device_frame_residency",
        "timing_excludes_parity_replay",
    ):
        if frozen["contract"].get(key) is not True:
            raise ValueError(f"frozen-world contract failed: {key}")
    if frozen["contract"].get("host_target_storage") != (
        "eager_cpu_selected_frames"
    ):
        raise ValueError("frozen-world host target storage contract drifted")
    resident_chunk_frames = int(
        frozen["contract"].get("resident_chunk_frames", 0)
    )
    if resident_chunk_frames < 1 or resident_chunk_frames > expected_frames:
        raise ValueError("frozen-world resident chunk size is invalid")
    checkpoint = frozen["checkpoint"]
    checkpoint_path = Path(str(checkpoint.get("path", "")))
    checkpoint_sha = checkpoint.get("sha256")
    if not checkpoint_path.is_file() or not _valid_sha256(checkpoint_sha):
        raise ValueError("frozen-world checkpoint provenance is invalid")
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != checkpoint_sha:
        raise ValueError("frozen-world checkpoint SHA-256 does not match")
    if int(checkpoint.get("bytes", 0)) != checkpoint_path.stat().st_size:
        raise ValueError("frozen-world checkpoint byte count does not match")
    world_state_sha = checkpoint.get("world_state_sha256")
    checkpoint_parameter_names = checkpoint.get("parameter_names")
    if (
        not _valid_sha256(world_state_sha)
        or not isinstance(checkpoint_parameter_names, list)
        or not checkpoint_parameter_names
        or any(
            not isinstance(name, str) or not name
            for name in checkpoint_parameter_names
        )
        or len(set(checkpoint_parameter_names)) != len(checkpoint_parameter_names)
        or int(checkpoint.get("parameter_tensor_count", 0))
        != len(checkpoint_parameter_names)
        or int(checkpoint.get("active_tube_count", 0)) <= 0
        or int(checkpoint.get("tube_count", 0))
        < int(checkpoint.get("active_tube_count", 0))
        or int(checkpoint.get("frame_count", 0))
        != int(
            expected_full_frames
            if expected_full_frames is not None
            else checkpoint.get("frame_count", 0)
        )
        or checkpoint.get("representation") != "legacy_tube"
        or checkpoint.get("alpha_mode") != "peak_splat"
        or checkpoint.get("amplitude_convention") != "fiber_integrated"
        or isinstance(checkpoint.get("min_precision_xy"), bool)
        or not isinstance(checkpoint.get("min_precision_xy"), Real)
        or not math.isfinite(float(checkpoint["min_precision_xy"]))
        or float(checkpoint["min_precision_xy"]) <= 0.0
        or isinstance(checkpoint.get("min_lambda_t"), bool)
        or not isinstance(checkpoint.get("min_lambda_t"), Real)
        or not math.isfinite(float(checkpoint["min_lambda_t"]))
        or float(checkpoint["min_lambda_t"]) <= 0.0
    ):
        raise ValueError("frozen-world checkpoint semantics are invalid")
    if (
        expected_active_tubes is not None
        and (
            int(checkpoint["active_tube_count"]) != int(expected_active_tubes)
            or int(checkpoint["tube_count"]) != int(expected_active_tubes)
        )
    ):
        raise ValueError("frozen-world checkpoint primitive count drifted")
    recomputed_world_state_sha = _validate_frozen_world_checkpoint_payload(
        checkpoint_path,
        checkpoint,
    )
    if recomputed_world_state_sha != world_state_sha:
        raise ValueError("frozen-world checkpoint semantic hash drifted")
    world_state = frozen["world_state"]
    state_hashes = (
        world_state.get("checkpoint_sha256"),
        world_state.get("before_routes_sha256"),
        world_state.get("after_replay_sha256"),
        world_state.get("after_compiled_sha256"),
    )
    if (
        any(not isinstance(value, str) or len(value) != 64 for value in state_hashes)
        or len(set(state_hashes)) != 1
        or state_hashes[0] != world_state_sha
        or world_state.get("matches_checkpoint") is not True
    ):
        raise ValueError("frozen-world route state does not match the checkpoint")
    scalar_paths = (
        ("loss", "replay"),
        ("loss", "compiled"),
        ("loss", "absolute_delta"),
        ("image", "max_abs_error"),
        ("image", "mean_abs_error"),
        ("gradient", "global_normalized_l2_error"),
        ("gradient", "cosine_similarity"),
        ("gradient", "replay_l2_norm"),
        ("gradient", "compiled_l2_norm"),
        ("gradient", "max_parameter_normalized_l2_error"),
        ("timing_s", "replay_total_forward"),
        ("timing_s", "replay_total_backward"),
        ("timing_s", "replay_per_frame_forward"),
        ("timing_s", "replay_per_frame_backward"),
        ("timing_s", "compiled_atlas_compile"),
        ("timing_s", "compiled_total_forward"),
        ("timing_s", "compiled_total_backward"),
        ("timing_s", "compiled_per_frame_forward"),
        ("timing_s", "compiled_per_frame_backward"),
        ("timing_s", "parity_replay_total_forward"),
        ("payload_bytes", "compiled_to_replay_logical_volume_ratio"),
        ("atlas", "fallback_fraction"),
    )
    invalid_scalars = [
        f"{section}.{key}"
        for section, key in scalar_paths
        if isinstance(frozen[section].get(key), bool)
        or not isinstance(frozen[section].get(key), Real)
        or not math.isfinite(float(frozen[section][key]))
    ]
    if invalid_scalars:
        raise ValueError(
            "frozen-world evidence has invalid scalars: "
            + ", ".join(invalid_scalars)
        )
    expected_loss_delta = abs(
        float(frozen["loss"]["replay"]) - float(frozen["loss"]["compiled"])
    )
    if not math.isclose(
        float(frozen["loss"]["absolute_delta"]),
        expected_loss_delta,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("frozen-world loss delta is inconsistent")
    for section, key in (
        ("loss", "replay"),
        ("loss", "compiled"),
        ("loss", "absolute_delta"),
        ("image", "max_abs_error"),
        ("image", "mean_abs_error"),
        ("gradient", "global_normalized_l2_error"),
        ("gradient", "replay_l2_norm"),
        ("gradient", "compiled_l2_norm"),
        ("gradient", "max_parameter_normalized_l2_error"),
        ("timing_s", "replay_total_forward"),
        ("timing_s", "replay_total_backward"),
        ("timing_s", "replay_per_frame_forward"),
        ("timing_s", "replay_per_frame_backward"),
        ("timing_s", "compiled_atlas_compile"),
        ("timing_s", "compiled_total_forward"),
        ("timing_s", "compiled_total_backward"),
        ("timing_s", "compiled_per_frame_forward"),
        ("timing_s", "compiled_per_frame_backward"),
        ("timing_s", "parity_replay_total_forward"),
        ("payload_bytes", "compiled_to_replay_logical_volume_ratio"),
        ("atlas", "fallback_fraction"),
    ):
        if float(frozen[section][key]) < 0.0:
            raise ValueError(f"frozen-world evidence {section}.{key} is negative")
    cosine_similarity = float(frozen["gradient"]["cosine_similarity"])
    if cosine_similarity < -1.0 or cosine_similarity > 1.0:
        raise ValueError("frozen-world gradient cosine similarity is invalid")
    for total_key, per_frame_key in (
        ("replay_total_forward", "replay_per_frame_forward"),
        ("replay_total_backward", "replay_per_frame_backward"),
        ("compiled_total_forward", "compiled_per_frame_forward"),
        ("compiled_total_backward", "compiled_per_frame_backward"),
    ):
        if not math.isclose(
            float(frozen["timing_s"][per_frame_key]),
            float(frozen["timing_s"][total_key]) / float(expected_frames),
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                f"frozen-world timing {per_frame_key} is inconsistent"
            )
    for key in (
        "replay_cumulative_logical_tensor_bytes",
        "compiled_trace_table_logical_tensor_bytes",
    ):
        if int(frozen["payload_bytes"].get(key, 0)) <= 0:
            raise ValueError(f"frozen-world evidence payload_bytes.{key} is invalid")
    expected_payload_ratio = float(
        frozen["payload_bytes"]["compiled_trace_table_logical_tensor_bytes"]
    ) / float(
        frozen["payload_bytes"]["replay_cumulative_logical_tensor_bytes"]
    )
    if not math.isclose(
        float(
            frozen["payload_bytes"][
                "compiled_to_replay_logical_volume_ratio"
            ]
        ),
        expected_payload_ratio,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise ValueError("frozen-world payload ratio is inconsistent")
    if (
        frozen["payload_bytes"].get("topology_bytes_included") is not False
        or frozen["payload_bytes"].get("storage_claim_eligible") is not False
    ):
        raise ValueError(
            "frozen-world tensor payload must remain excluded from storage claims"
        )
    if (
        frozen["gradient"].get("gradient_coverage_matches") is not True
        or int(frozen["gradient"].get("replay_gradient_tensor_count", 0)) <= 0
        or int(frozen["gradient"].get("compiled_gradient_tensor_count", 0)) <= 0
        or int(frozen["gradient"].get("replay_gradient_tensor_count", 0))
        != int(frozen["gradient"].get("parameter_tensor_count", -1))
        or int(frozen["gradient"].get("compiled_gradient_tensor_count", 0))
        != int(frozen["gradient"].get("parameter_tensor_count", -1))
        or int(frozen["gradient"].get("parameter_tensor_count", 0))
        != int(checkpoint["parameter_tensor_count"])
        or frozen["gradient"].get("replay_gradient_parameters")
        != checkpoint_parameter_names
        or frozen["gradient"].get("compiled_gradient_parameters")
        != checkpoint_parameter_names
    ):
        raise ValueError("frozen-world gradient coverage is invalid")
    contract_hashes = frozen["contract_hashes"]
    for key in (
        "target_frames_sha256",
        "camera_program_sha256",
        "frame_indices_sha256",
        "centered_frame_times_sha256",
        "evaluation_contract_sha256",
    ):
        value = contract_hashes.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"frozen-world contract hash is invalid: {key}")
    if (
        contract_hashes["frame_indices_sha256"]
        != _frozen_world_sequence_sha256(expected_frame_indices)
        or contract_hashes["centered_frame_times_sha256"]
        != _frozen_world_sequence_sha256(expected_centered_times)
    ):
        raise ValueError("frozen-world fixed-program time-grid hash drifted")
    atlas_count_keys = (
        "trace_count",
        "cell_count",
        "interval_trace_entries",
        "dense_trace_samples",
        "fallback_cells",
        "total_tile_samples",
        "fallback_tile_samples",
    )
    for key in atlas_count_keys:
        value = frozen["atlas"].get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or int(value) != value
            or int(value) < 0
        ):
            raise ValueError(f"frozen-world evidence atlas.{key} is invalid")
    trace_count = int(frozen["atlas"]["trace_count"])
    cell_count = int(frozen["atlas"]["cell_count"])
    interval_trace_entries = int(frozen["atlas"]["interval_trace_entries"])
    dense_trace_samples = int(frozen["atlas"]["dense_trace_samples"])
    if (
        trace_count < 1
        or cell_count < 1
        or int(frozen["atlas"]["total_tile_samples"]) < 1
        or int(frozen["atlas"]["fallback_cells"])
        > cell_count
        or interval_trace_entries > dense_trace_samples
        or interval_trace_entries > trace_count * cell_count
        or (interval_trace_entries == 0) != (dense_trace_samples == 0)
    ):
        raise ValueError("frozen-world atlas structural counts are inconsistent")
    interval_ratio = frozen["atlas"].get(
        "interval_to_dense_trace_sample_ratio"
    )
    if interval_ratio is not None:
        if (
            isinstance(interval_ratio, bool)
            or not isinstance(interval_ratio, Real)
            or not math.isfinite(float(interval_ratio))
            or float(interval_ratio) < 0.0
        ):
            raise ValueError(
                "frozen-world interval-to-dense trace ratio is invalid"
            )
        expected_interval_ratio = (
            float(interval_trace_entries) / float(dense_trace_samples)
            if dense_trace_samples > 0
            else 0.0
        )
        if not math.isclose(
            float(interval_ratio),
            expected_interval_ratio,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "frozen-world interval-to-dense trace ratio is inconsistent"
            )
    total_tile_samples = int(frozen["atlas"]["total_tile_samples"])
    fallback_tile_samples = int(frozen["atlas"]["fallback_tile_samples"])
    if fallback_tile_samples > total_tile_samples:
        raise ValueError("frozen-world fallback tile count is invalid")
    expected_fallback_fraction = (
        float(fallback_tile_samples) / float(total_tile_samples)
        if total_tile_samples > 0
        else 0.0
    )
    if not math.isclose(
        float(frozen["atlas"]["fallback_fraction"]),
        expected_fallback_fraction,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise ValueError("frozen-world fallback fraction is inconsistent")

    if dict(frozen["acceptance"]) != FROZEN_WORLD_ACCEPTANCE:
        raise ValueError("frozen-world acceptance thresholds drifted")
    expected_checks = {
        "checkpoint_matches": world_state.get("matches_checkpoint") is True,
        "image_matches": float(frozen["image"]["max_abs_error"])
        <= float(frozen["acceptance"]["image_max_abs_error"]),
        "loss_matches": float(frozen["loss"]["absolute_delta"])
        <= float(frozen["acceptance"]["loss_absolute_delta"]),
        "world_vjp_matches": float(
            frozen["gradient"]["global_normalized_l2_error"]
        )
        <= FROZEN_WORLD_ACCEPTANCE["gradient_global_normalized_l2_error"],
        "world_vjp_per_parameter_matches": float(
            frozen["gradient"]["max_parameter_normalized_l2_error"]
        )
        <= FROZEN_WORLD_ACCEPTANCE[
            "gradient_max_parameter_normalized_l2_error"
        ],
        "world_vjp_nonzero": min(
            float(frozen["gradient"]["replay_l2_norm"]),
            float(frozen["gradient"]["compiled_l2_norm"]),
        )
        > FROZEN_WORLD_ACCEPTANCE["min_world_vjp_l2_norm"],
        "world_vjp_coverage_matches": (
            frozen["gradient"]["gradient_coverage_matches"] is True
            and int(frozen["gradient"]["replay_gradient_tensor_count"])
            == int(frozen["gradient"]["parameter_tensor_count"])
            and int(frozen["gradient"]["compiled_gradient_tensor_count"])
            == int(frozen["gradient"]["parameter_tensor_count"])
        ),
        "fallback_within_budget": float(frozen["atlas"]["fallback_fraction"])
        <= FROZEN_WORLD_ACCEPTANCE["fallback_fraction"],
    }
    if dict(frozen["checks"]) != expected_checks:
        raise ValueError("frozen-world acceptance checks do not match evidence")
    if frozen.get("accepted") != all(expected_checks.values()):
        raise ValueError("frozen-world accepted status does not match checks")


def validate_comparison_report(
    report: Mapping[str, Any],
    protocol: PaperTrainingProtocol,
    *,
    backward_policy: str,
    manifest_validation: Mapping[str, Any],
    uvt_world_representation: str = DEFAULT_UVT_WORLD_REPRESENTATION,
    uvt_alpha_mode: str = DEFAULT_UVT_ALPHA_MODE,
    uvt_render_backend: str = DEFAULT_UVT_RENDER_BACKEND,
    uvt_amplitude_convention: str = DEFAULT_UVT_AMPLITUDE_CONVENTION,
    uvt_retained_depth_samples: int = DEFAULT_UVT_RETAINED_DEPTH_SAMPLES,
    uvt_retained_sigma_extent: float = DEFAULT_UVT_RETAINED_SIGMA_EXTENT,
    uvt_order_certificate_sigma: float = DEFAULT_UVT_ORDER_CERTIFICATE_SIGMA,
    uvt_order_certificate_min_gap: float = DEFAULT_UVT_ORDER_CERTIFICATE_MIN_GAP,
    uvt_spd4_init_precision_z: float | None = None,
    frozen_world_replay_compiled: bool = False,
    frozen_world_max_frames: int = 0,
) -> None:
    meta = report["meta"]
    _validate_uvt_paper_lane_contract(
        protocol,
        backward_policy=backward_policy,
        world_representation=uvt_world_representation,
        alpha_mode=uvt_alpha_mode,
        render_backend=uvt_render_backend,
        amplitude_convention=uvt_amplitude_convention,
        retained_depth_samples=uvt_retained_depth_samples,
        retained_sigma_extent=uvt_retained_sigma_extent,
        order_certificate_sigma=uvt_order_certificate_sigma,
        order_certificate_min_gap=uvt_order_certificate_min_gap,
        device=str(meta.get("device", "")),
    )
    if tuple(meta["train_cameras"]) != protocol.dataset.train_cameras:
        raise ValueError("comparison report train cameras do not match the paper protocol")
    if tuple(meta["heldout_cameras"]) != protocol.dataset.heldout_cameras:
        raise ValueError("comparison report heldout cameras do not match the paper protocol")
    if int(meta["frame_count"]) != protocol.dataset.frame_count:
        raise ValueError("comparison report frame count does not match the paper protocol")
    if bool(meta.get("frozen_world_replay_compiled", False)) != bool(
        frozen_world_replay_compiled
    ):
        raise ValueError("comparison report frozen-world mode drifted")
    if int(meta.get("frozen_world_max_frames", 0)) != int(
        frozen_world_max_frames
    ):
        raise ValueError("comparison report frozen-world frame limit drifted")
    expected_backward_policy = effective_uvt_backward_policy(
        uvt_render_backend,
        backward_policy,
    )
    reported_backward_policy = meta.get("uvt_backward_policy")
    if expected_backward_policy == "manual":
        if reported_backward_policy is not None:
            raise ValueError("comparison report World Tubes backward policy drifted")
    elif (
        not isinstance(reported_backward_policy, Mapping)
        or reported_backward_policy.get("name") != expected_backward_policy
    ):
        raise ValueError("comparison report World Tubes backward policy drifted")
    reported_world_representation = meta.get(
        "uvt_world_representation",
        DEFAULT_UVT_WORLD_REPRESENTATION,
    )
    if reported_world_representation != uvt_world_representation:
        raise ValueError(
            "comparison report World Tubes representation drifted: "
            f"expected {uvt_world_representation}, got {reported_world_representation}"
        )
    reported_alpha_mode = meta.get("uvt_alpha_mode", DEFAULT_UVT_ALPHA_MODE)
    if reported_alpha_mode != uvt_alpha_mode:
        raise ValueError(
            "comparison report STAR alpha mode drifted: "
            f"expected {uvt_alpha_mode}, got {reported_alpha_mode}"
        )
    reported_render_backend = meta.get(
        "uvt_render_backend",
        DEFAULT_UVT_RENDER_BACKEND,
    )
    if reported_render_backend != uvt_render_backend:
        raise ValueError(
            "comparison report STAR render backend drifted: "
            f"expected {uvt_render_backend}, got {reported_render_backend}"
        )
    reported_amplitude_convention = meta.get(
        "uvt_amplitude_convention",
        DEFAULT_UVT_AMPLITUDE_CONVENTION,
    )
    if reported_amplitude_convention != uvt_amplitude_convention:
        raise ValueError(
            "comparison report STAR amplitude convention drifted: "
            f"expected {uvt_amplitude_convention}, "
            f"got {reported_amplitude_convention}"
        )
    expected_physical_settings = {
        "uvt_retained_depth_samples": uvt_retained_depth_samples,
        "uvt_retained_sigma_extent": uvt_retained_sigma_extent,
        "uvt_order_certificate_sigma": uvt_order_certificate_sigma,
        "uvt_order_certificate_min_gap": uvt_order_certificate_min_gap,
    }
    physical_defaults = {
        "uvt_retained_depth_samples": DEFAULT_UVT_RETAINED_DEPTH_SAMPLES,
        "uvt_retained_sigma_extent": DEFAULT_UVT_RETAINED_SIGMA_EXTENT,
        "uvt_order_certificate_sigma": DEFAULT_UVT_ORDER_CERTIFICATE_SIGMA,
        "uvt_order_certificate_min_gap": DEFAULT_UVT_ORDER_CERTIFICATE_MIN_GAP,
    }
    for key, expected in expected_physical_settings.items():
        reported = meta.get(key, physical_defaults[key])
        if reported != expected:
            raise ValueError(
                f"comparison report {key} drifted: "
                f"expected {expected}, got {reported}"
            )
    expected_opacity_semantics = uvt_opacity_semantics(
        uvt_alpha_mode,
        uvt_amplitude_convention,
    )
    reported_opacity_semantics = meta.get(
        "uvt_opacity_semantics",
        (
            uvt_opacity_semantics(
                DEFAULT_UVT_ALPHA_MODE,
                DEFAULT_UVT_AMPLITUDE_CONVENTION,
            )
            if reported_alpha_mode == DEFAULT_UVT_ALPHA_MODE
            else None
        ),
    )
    if reported_opacity_semantics != expected_opacity_semantics:
        raise ValueError(
            "comparison report STAR opacity semantics drifted: "
            f"expected {expected_opacity_semantics}, "
            f"got {reported_opacity_semantics}"
        )
    if uvt_spd4_init_precision_z is not None:
        reported_init_precision_z = meta.get("uvt_spd4_init_precision_z")
        if reported_init_precision_z != uvt_spd4_init_precision_z:
            raise ValueError(
                "comparison report SPD(4) depth initialization drifted: "
                f"expected {uvt_spd4_init_precision_z}, "
                f"got {reported_init_precision_z}"
            )
    native_extension = meta.get("star_uvt_native_extension")
    if not isinstance(native_extension, Mapping):
        raise ValueError("comparison report native extension identity is missing")
    validate_native_extension_identity(native_extension)
    validate_hashed_contract(
        "decoded paper dataset bundle",
        meta.get("paper_dataset_bundle"),
        schema_version=PAPER_DATASET_BUNDLE_SCHEMA_VERSION,
    )
    validate_comparison_pose_source(meta, manifest_validation)
    validate_hashed_contract(
        "paper evaluator",
        meta.get("paper_evaluator"),
        schema_version=PAPER_EVALUATOR_SCHEMA_VERSION,
    )
    if meta["paper_evaluator"] != paper_evaluator_contract():
        raise ValueError("comparison report paper evaluator is not canonical")
    validate_hashed_contract(
        "paper runtime",
        meta.get("paper_runtime"),
        schema_version=PAPER_RUNTIME_SCHEMA_VERSION,
    )
    route_native_extensions = meta.get("route_native_extensions")
    if (
        not isinstance(route_native_extensions, Mapping)
        or set(route_native_extensions) != set(LANE_REPORT_KEYS)
    ):
        raise ValueError("comparison report route-native identities are missing")
    for lane_name, route_native in route_native_extensions.items():
        validate_route_native_extension_identity(lane_name, route_native)
    seed = int(meta["seed"])
    lane_sample_schedules: dict[str, Mapping[str, Any]] = {}
    for lane_name, report_key in LANE_REPORT_KEYS.items():
        lane = report.get(report_key)
        if not isinstance(lane, Mapping):
            raise ValueError(f"comparison report is missing {lane_name}")
        validate_lane_cost(lane_name, lane, protocol, seed=seed)
        lane_sample_schedules[lane_name] = lane["paper_protocol"][
            "sample_schedule"
        ]
        build_lane_evidence(lane_name, lane, frame_count=protocol.dataset.frame_count)
    if (
        lane_sample_schedules["world_tubes"]
        != lane_sample_schedules["dynamic_3dgs"]
    ):
        raise ValueError(
            "World Tubes and dynamic 3DGS did not consume the same sample schedule"
        )
    frozen = report["star_uvt"].get("frozen_world_replay_compiled")
    if frozen_world_replay_compiled:
        if not isinstance(frozen, Mapping):
            raise ValueError("comparison report is missing frozen-world evidence")
        expected_frames = (
            protocol.dataset.frame_count
            if frozen_world_max_frames <= 0
            else min(frozen_world_max_frames, protocol.dataset.frame_count)
        )
        validate_frozen_world_evidence(
            frozen,
            expected_frames=expected_frames,
            expected_full_frames=protocol.dataset.frame_count,
            expected_image_size=(
                protocol.final_stage.image_size.height,
                protocol.final_stage.image_size.width,
            ),
            expected_heldout_camera=protocol.dataset.heldout_cameras[0],
            expected_active_tubes=protocol.final_stage.primitive_count,
        )
    elif frozen is not None:
        raise ValueError("comparison report unexpectedly contains frozen-world evidence")


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
        "uvt_world_representation",
        "uvt_alpha_mode",
        "uvt_render_backend",
        "uvt_amplitude_convention",
        "uvt_opacity_semantics",
        "uvt_retained_depth_samples",
        "uvt_retained_sigma_extent",
        "uvt_order_certificate_sigma",
        "uvt_order_certificate_min_gap",
        "uvt_spd4_init_precision_z",
        "uvt_camera_projection",
        "uvt_camera_sequence_mode",
        "uvt_segment_frames",
        "uvt_backward_policy",
        "splat_camera_projection",
        "eval_chunk_frames",
        "eval_media_max_frames",
        "star_uvt_native_extension",
        "paper_dataset_bundle",
        "paper_evaluator",
        "paper_runtime",
    )
    reference_name = "world_tubes"
    reference_meta = lane_reports[reference_name].get("meta")
    if not isinstance(reference_meta, Mapping):
        raise ValueError(f"isolated {reference_name} report has no metadata")
    reference_native_extension = reference_meta.get("star_uvt_native_extension")
    if not isinstance(reference_native_extension, Mapping):
        raise ValueError(
            "isolated world_tubes report native extension identity is missing"
        )
    validate_native_extension_identity(reference_native_extension)
    defaults = {
        "uvt_world_representation": DEFAULT_UVT_WORLD_REPRESENTATION,
        "uvt_alpha_mode": DEFAULT_UVT_ALPHA_MODE,
        "uvt_render_backend": DEFAULT_UVT_RENDER_BACKEND,
        "uvt_amplitude_convention": DEFAULT_UVT_AMPLITUDE_CONVENTION,
        "uvt_retained_depth_samples": DEFAULT_UVT_RETAINED_DEPTH_SAMPLES,
        "uvt_retained_sigma_extent": DEFAULT_UVT_RETAINED_SIGMA_EXTENT,
        "uvt_order_certificate_sigma": DEFAULT_UVT_ORDER_CERTIFICATE_SIGMA,
        "uvt_order_certificate_min_gap": DEFAULT_UVT_ORDER_CERTIFICATE_MIN_GAP,
    }

    def normalized_meta_value(meta: Mapping[str, Any], key: str) -> Any:
        if key == "uvt_opacity_semantics":
            return meta.get(
                key,
                uvt_opacity_semantics(
                    str(meta.get("uvt_alpha_mode", DEFAULT_UVT_ALPHA_MODE)),
                    str(
                        meta.get(
                            "uvt_amplitude_convention",
                            DEFAULT_UVT_AMPLITUDE_CONVENTION,
                        )
                    ),
                ),
            )
        return meta.get(key, defaults.get(key))

    for lane_name, report in lane_reports.items():
        meta = report.get("meta")
        if not isinstance(meta, Mapping):
            raise ValueError(f"isolated {lane_name} report has no metadata")
        if meta.get("only_lane") != lane_name:
            raise ValueError(f"isolated {lane_name} report was produced as {meta.get('only_lane')!r}")
        native_extension = meta.get("star_uvt_native_extension")
        if not isinstance(native_extension, Mapping):
            raise ValueError(
                f"isolated {lane_name} report native extension identity is missing"
            )
        validate_native_extension_identity(native_extension)
        validate_route_native_extension_identity(
            lane_name,
            meta.get("route_native_extension"),
        )
        validate_hashed_contract(
            f"isolated {lane_name} decoded paper dataset bundle",
            meta.get("paper_dataset_bundle"),
            schema_version=PAPER_DATASET_BUNDLE_SCHEMA_VERSION,
        )
        validate_hashed_contract(
            f"isolated {lane_name} paper evaluator",
            meta.get("paper_evaluator"),
            schema_version=PAPER_EVALUATOR_SCHEMA_VERSION,
        )
        validate_hashed_contract(
            f"isolated {lane_name} paper runtime",
            meta.get("paper_runtime"),
            schema_version=PAPER_RUNTIME_SCHEMA_VERSION,
        )
        drift = []
        for key in meta_keys:
            if normalized_meta_value(meta, key) != normalized_meta_value(
                reference_meta,
                key,
            ):
                drift.append(key)
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
    reference_schedule = lane_reports["world_tubes"]["star_uvt"].get(
        "paper_protocol",
        {},
    ).get("sample_schedule")
    comparison_schedule = lane_reports["dynamic_3dgs"][
        "free_dynamic_splats"
    ].get("paper_protocol", {}).get("sample_schedule")
    if not isinstance(reference_schedule, Mapping) or not isinstance(
        comparison_schedule,
        Mapping,
    ):
        raise ValueError("isolated paper lanes are missing sample schedules")
    if reference_schedule != comparison_schedule:
        raise ValueError(
            "isolated paper lanes did not consume the same sample schedule"
        )

    merged_meta = dict(reference_meta)
    merged_alpha_mode = str(
        reference_meta.get("uvt_alpha_mode", DEFAULT_UVT_ALPHA_MODE)
    )
    merged_amplitude_convention = str(
        reference_meta.get(
            "uvt_amplitude_convention",
            DEFAULT_UVT_AMPLITUDE_CONVENTION,
        )
    )
    merged_meta.update(
        {
            "only_lane": "isolated_merged",
            "skip_splats": False,
            "execution_model": "one_child_process_per_representation",
            "uvt_world_representation": reference_meta.get(
                "uvt_world_representation",
                DEFAULT_UVT_WORLD_REPRESENTATION,
            ),
            "uvt_alpha_mode": reference_meta.get(
                "uvt_alpha_mode",
                DEFAULT_UVT_ALPHA_MODE,
            ),
            "uvt_render_backend": reference_meta.get(
                "uvt_render_backend",
                DEFAULT_UVT_RENDER_BACKEND,
            ),
            "uvt_amplitude_convention": merged_amplitude_convention,
            "uvt_opacity_semantics": uvt_opacity_semantics(
                merged_alpha_mode,
                merged_amplitude_convention,
            ),
            "uvt_retained_depth_samples": reference_meta.get(
                "uvt_retained_depth_samples",
                DEFAULT_UVT_RETAINED_DEPTH_SAMPLES,
            ),
            "uvt_retained_sigma_extent": reference_meta.get(
                "uvt_retained_sigma_extent",
                DEFAULT_UVT_RETAINED_SIGMA_EXTENT,
            ),
            "uvt_order_certificate_sigma": reference_meta.get(
                "uvt_order_certificate_sigma",
                DEFAULT_UVT_ORDER_CERTIFICATE_SIGMA,
            ),
            "uvt_order_certificate_min_gap": reference_meta.get(
                "uvt_order_certificate_min_gap",
                DEFAULT_UVT_ORDER_CERTIFICATE_MIN_GAP,
            ),
            "route_native_extensions": {
                lane_name: lane_reports[lane_name]["meta"][
                    "route_native_extension"
                ]
                for lane_name in LANE_REPORT_KEYS
            },
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
    expected_source: Mapping[str, Any],
    uvt_world_representation: str = DEFAULT_UVT_WORLD_REPRESENTATION,
    uvt_alpha_mode: str = DEFAULT_UVT_ALPHA_MODE,
    uvt_render_backend: str = DEFAULT_UVT_RENDER_BACKEND,
    uvt_amplitude_convention: str = DEFAULT_UVT_AMPLITUDE_CONVENTION,
    uvt_retained_depth_samples: int = DEFAULT_UVT_RETAINED_DEPTH_SAMPLES,
    uvt_retained_sigma_extent: float = DEFAULT_UVT_RETAINED_SIGMA_EXTENT,
    uvt_order_certificate_sigma: float = DEFAULT_UVT_ORDER_CERTIFICATE_SIGMA,
    uvt_order_certificate_min_gap: float = DEFAULT_UVT_ORDER_CERTIFICATE_MIN_GAP,
    uvt_spd4_init_precision_z: float | None = None,
    frozen_world_replay_compiled: bool = False,
    frozen_world_max_frames: int = 0,
    allow_local_mps_execution: bool = False,
    expected_dataset_input_identity: Mapping[str, Any] | None = None,
    python: str = sys.executable,
) -> Path:
    """Run only missing lane children, then form their shared report contract."""
    if not isinstance(expected_source, Mapping) or not expected_source:
        raise ValueError(
            "isolated paper lane execution requires an expected source identity"
        )
    expected_source_identity = dict(expected_source)
    _validate_uvt_paper_lane_contract(
        protocol,
        backward_policy=backward_policy,
        world_representation=uvt_world_representation,
        alpha_mode=uvt_alpha_mode,
        render_backend=uvt_render_backend,
        amplitude_convention=uvt_amplitude_convention,
        retained_depth_samples=uvt_retained_depth_samples,
        retained_sigma_extent=uvt_retained_sigma_extent,
        order_certificate_sigma=uvt_order_certificate_sigma,
        order_certificate_min_gap=uvt_order_certificate_min_gap,
        device=device,
    )

    expected_identity: dict[str, Any] = {
        "device": device,
        "uvt_world_representation": uvt_world_representation,
        "uvt_alpha_mode": uvt_alpha_mode,
        "uvt_render_backend": uvt_render_backend,
        "uvt_amplitude_convention": uvt_amplitude_convention,
        "uvt_opacity_semantics": uvt_opacity_semantics(
            uvt_alpha_mode,
            uvt_amplitude_convention,
        ),
        "uvt_retained_depth_samples": uvt_retained_depth_samples,
        "uvt_retained_sigma_extent": uvt_retained_sigma_extent,
        "uvt_order_certificate_sigma": uvt_order_certificate_sigma,
        "uvt_order_certificate_min_gap": uvt_order_certificate_min_gap,
    }
    if uvt_spd4_init_precision_z is not None:
        expected_identity["uvt_spd4_init_precision_z"] = (
            uvt_spd4_init_precision_z
        )
    identity_defaults = {
        "uvt_world_representation": DEFAULT_UVT_WORLD_REPRESENTATION,
        "uvt_alpha_mode": DEFAULT_UVT_ALPHA_MODE,
        "uvt_render_backend": DEFAULT_UVT_RENDER_BACKEND,
        "uvt_amplitude_convention": DEFAULT_UVT_AMPLITUDE_CONVENTION,
        "uvt_opacity_semantics": uvt_opacity_semantics(
            DEFAULT_UVT_ALPHA_MODE,
            DEFAULT_UVT_AMPLITUDE_CONVENTION,
        ),
        "uvt_retained_depth_samples": DEFAULT_UVT_RETAINED_DEPTH_SAMPLES,
        "uvt_retained_sigma_extent": DEFAULT_UVT_RETAINED_SIGMA_EXTENT,
        "uvt_order_certificate_sigma": DEFAULT_UVT_ORDER_CERTIFICATE_SIGMA,
        "uvt_order_certificate_min_gap": DEFAULT_UVT_ORDER_CERTIFICATE_MIN_GAP,
    }
    expected_reported_policy = effective_uvt_backward_policy(
        uvt_render_backend,
        backward_policy,
    )

    def report_matches_identity(
        report: Mapping[str, Any],
        *,
        expect_frozen_world: bool,
    ) -> bool:
        meta = report.get("meta")
        if not isinstance(meta, Mapping):
            return False
        for key, expected in expected_identity.items():
            if meta.get(key, identity_defaults.get(key)) != expected:
                return False
        if bool(meta.get("frozen_world_replay_compiled", False)) != bool(
            expect_frozen_world
        ):
            return False
        expected_frozen_frames = (
            frozen_world_max_frames if expect_frozen_world else 0
        )
        if int(meta.get("frozen_world_max_frames", 0)) != int(
            expected_frozen_frames
        ):
            return False
        reported_policy = meta.get("uvt_backward_policy")
        if expected_reported_policy == "manual":
            return reported_policy is None
        return (
            isinstance(reported_policy, Mapping)
            and reported_policy.get("name") == expected_reported_policy
        )

    comparison_report_path = comparison_dir / "comparison_report.json"
    expected_dataset_identity = (
        validate_manifest(protocol)["input_identity"]
        if expected_dataset_input_identity is None
        else dict(expected_dataset_input_identity)
    )
    lane_reports: dict[str, Mapping[str, Any]] = {}
    for lane_name, command in comparison_lane_commands(
        protocol_path,
        protocol,
        seed,
        comparison_dir,
        backward_policy=backward_policy,
        device=device,
        uvt_world_representation=uvt_world_representation,
        uvt_alpha_mode=uvt_alpha_mode,
        uvt_render_backend=uvt_render_backend,
        uvt_amplitude_convention=uvt_amplitude_convention,
        uvt_retained_depth_samples=uvt_retained_depth_samples,
        uvt_retained_sigma_extent=uvt_retained_sigma_extent,
        uvt_order_certificate_sigma=uvt_order_certificate_sigma,
        uvt_order_certificate_min_gap=uvt_order_certificate_min_gap,
        uvt_spd4_init_precision_z=uvt_spd4_init_precision_z,
        frozen_world_replay_compiled=frozen_world_replay_compiled,
        frozen_world_max_frames=frozen_world_max_frames,
        allow_local_mps_execution=allow_local_mps_execution,
        python=python,
    ).items():
        lane_report_path = comparison_dir / lane_name / "comparison_report.json"
        lane_identity_path = comparison_dir / lane_name / "execution_identity.json"
        lane_report = (
            load_json(lane_report_path)
            if reuse_existing and lane_report_path.exists()
            else None
        )
        if lane_report is not None:
            lane_identity = (
                load_json(lane_identity_path)
                if lane_identity_path.exists()
                else None
            )
            if (
                not isinstance(lane_identity, Mapping)
                or lane_identity.get("source_start")
                != expected_source_identity
                or lane_identity.get("source_finish")
                != expected_source_identity
                or lane_identity.get("dataset_input_identity")
                != expected_dataset_identity
                or lane_identity.get("protocol_sha256")
                != file_sha256(protocol_path)
                or list(lane_identity.get("command", ())) != command
                or lane_identity.get("comparison_report_sha256")
                != file_sha256(lane_report_path)
            ):
                lane_report = None
        expect_frozen_world = (
            frozen_world_replay_compiled and lane_name == "world_tubes"
        )
        if lane_report is None or not report_matches_identity(
            lane_report,
            expect_frozen_world=expect_frozen_world,
        ):
            source_start = source_provenance()
            if source_start != expected_source_identity:
                raise RuntimeError(
                    f"source drifted before the {lane_name} paper lane"
                )
            lane_live_resources = None
            if str(device).lower() == "mps":
                lane_live_resources = live_resource_snapshot()
                require_live_resources(lane_live_resources)
            subprocess.run(command, cwd=ROOT, check=True)
            source_finish = source_provenance()
            if source_start != source_finish:
                raise RuntimeError(
                    f"source changed while the {lane_name} paper lane executed"
                )
            lane_report = load_json(lane_report_path)
            write_json(
                lane_identity_path,
                {
                    "schema_version": 1,
                    "lane": lane_name,
                    "protocol": protocol.as_dict(),
                    "protocol_sha256": file_sha256(protocol_path),
                    "command": command,
                    "source_start": source_start,
                    "source_finish": source_finish,
                    "dataset_input_identity": expected_dataset_identity,
                    "live_resources_at_launch": lane_live_resources,
                    "comparison_report": display_path(lane_report_path),
                    "comparison_report_sha256": file_sha256(lane_report_path),
                },
            )
        if not report_matches_identity(
            lane_report,
            expect_frozen_world=expect_frozen_world,
        ):
            raise ValueError(
                f"isolated {lane_name} report identity does not match the "
                "requested physical renderer configuration"
            )
        lane_reports[lane_name] = lane_report
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
    execution_source: Mapping[str, Any],
) -> dict[str, Any]:
    import wandb

    report_key = LANE_REPORT_KEYS[lane_name]
    lane = report[report_key]
    metrics = lane["metrics"]
    cost = lane["paper_protocol"]["cost"]
    world_representation = report["meta"].get(
        "uvt_world_representation",
        DEFAULT_UVT_WORLD_REPRESENTATION,
    )
    alpha_mode = report["meta"].get("uvt_alpha_mode", DEFAULT_UVT_ALPHA_MODE)
    render_backend = report["meta"].get(
        "uvt_render_backend",
        DEFAULT_UVT_RENDER_BACKEND,
    )
    amplitude_convention = report["meta"].get(
        "uvt_amplitude_convention",
        DEFAULT_UVT_AMPLITUDE_CONVENTION,
    )
    opacity_semantics = report["meta"].get(
        "uvt_opacity_semantics",
        uvt_opacity_semantics(alpha_mode, amplitude_convention),
    )
    retained_depth_samples = report["meta"].get(
        "uvt_retained_depth_samples",
        DEFAULT_UVT_RETAINED_DEPTH_SAMPLES,
    )
    retained_sigma_extent = report["meta"].get(
        "uvt_retained_sigma_extent",
        DEFAULT_UVT_RETAINED_SIGMA_EXTENT,
    )
    order_certificate_sigma = report["meta"].get(
        "uvt_order_certificate_sigma",
        DEFAULT_UVT_ORDER_CERTIFICATE_SIGMA,
    )
    order_certificate_min_gap = report["meta"].get(
        "uvt_order_certificate_min_gap",
        DEFAULT_UVT_ORDER_CERTIFICATE_MIN_GAP,
    )
    spd4_init_precision_z = report["meta"].get("uvt_spd4_init_precision_z")
    representation_hash_suffix = (
        "" if world_representation == DEFAULT_UVT_WORLD_REPRESENTATION else f":{world_representation}"
    )
    initialization_hash_suffix = (
        ""
        if spd4_init_precision_z is None
        else f":init-z-{float(spd4_init_precision_z):.9g}"
    )
    alpha_hash_suffix = (
        "" if alpha_mode == DEFAULT_UVT_ALPHA_MODE else f":{alpha_mode}"
    )
    physical_hash_suffix = (
        f":{render_backend}:{amplitude_convention}:"
        f"depth-{int(retained_depth_samples)}:"
        f"sigma-{float(retained_sigma_extent):.9g}:"
        f"cert-{float(order_certificate_sigma):.9g}:"
        f"gap-{float(order_certificate_min_gap):.9g}"
    )
    frozen_world_enabled = lane_name == "world_tubes" and bool(
        report["meta"].get("frozen_world_replay_compiled", False)
    )
    frozen_world_max_frames = int(
        report["meta"].get("frozen_world_max_frames", 0)
    )
    frozen_hash_suffix = (
        f":frozen-world-{frozen_world_max_frames or 'all'}"
        if frozen_world_enabled
        else ""
    )
    source_digest = hashlib.sha256(
        json.dumps(
            serialize_config_value(dict(execution_source)),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    report_digest = hashlib.sha256(
        json.dumps(
            serialize_config_value(dict(report)),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    run_id = hashlib.sha1(
        (
            f"{protocol.name}:{seed}:{lane_name}:"
            f"{lane['paper_protocol']['kernel']['backward']}:evidence-v2"
            f"{representation_hash_suffix}"
            f"{initialization_hash_suffix}"
            f"{alpha_hash_suffix}"
            f"{physical_hash_suffix}"
            f"{frozen_hash_suffix}"
            f":source-{source_digest}"
            f":report-{report_digest}"
        ).encode("utf-8")
    ).hexdigest()[:12]
    representation_name_suffix = (
        "" if world_representation == DEFAULT_UVT_WORLD_REPRESENTATION else f"-{world_representation}"
    )
    alpha_name_suffix = (
        "" if alpha_mode == DEFAULT_UVT_ALPHA_MODE else f"-{alpha_mode}"
    )
    backend_name_suffix = (
        ""
        if render_backend == DEFAULT_UVT_RENDER_BACKEND
        else f"-{render_backend}"
    )
    amplitude_name_suffix = (
        ""
        if amplitude_convention == DEFAULT_UVT_AMPLITUDE_CONVENTION
        else f"-{amplitude_convention}"
    )
    run_name = (
        f"paper-{protocol.name}-{lane_name}{representation_name_suffix}"
        f"{alpha_name_suffix}{backend_name_suffix}"
        f"{amplitude_name_suffix}"
        f"{'-frozen-world' if frozen_world_enabled else ''}-seed{seed}"
    )
    route_native_extensions = report["meta"].get("route_native_extensions")
    if isinstance(route_native_extensions, Mapping):
        route_native_extension = route_native_extensions.get(lane_name)
    elif report["meta"].get("only_lane") == lane_name:
        route_native_extension = report["meta"].get("route_native_extension")
    else:
        route_native_extension = None
    validate_route_native_extension_identity(
        lane_name,
        route_native_extension,
    )
    wandb_config = {
        "protocol": protocol.as_dict(),
        "seed": seed,
        "kernel": lane["paper_protocol"]["kernel"],
        "uvt_world_representation": world_representation,
        "uvt_alpha_mode": alpha_mode,
        "uvt_render_backend": render_backend,
        "uvt_amplitude_convention": amplitude_convention,
        "uvt_opacity_semantics": opacity_semantics,
        "uvt_retained_depth_samples": retained_depth_samples,
        "uvt_retained_sigma_extent": retained_sigma_extent,
        "uvt_order_certificate_sigma": order_certificate_sigma,
        "uvt_order_certificate_min_gap": order_certificate_min_gap,
        "uvt_spd4_init_precision_z": spd4_init_precision_z,
        "frozen_world_replay_compiled": frozen_world_enabled,
        "frozen_world_max_frames": frozen_world_max_frames,
        "source": dict(execution_source),
        "source_digest": source_digest,
        "comparison_report_sha256": report_digest,
        "star_uvt_native_extension": report["meta"].get(
            "star_uvt_native_extension"
        ),
        "route_native_extension": route_native_extension,
        "paper_dataset_bundle": report["meta"]["paper_dataset_bundle"],
        "paper_evaluator": report["meta"]["paper_evaluator"],
        "paper_runtime": report["meta"]["paper_runtime"],
    }
    config_digest = hashlib.sha256(
        json.dumps(
            serialize_config_value(wandb_config),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    resolved_run_id = f"pa{run_id}"
    identity_path = report_dir / lane_name / "wandb_identity.json"
    if identity_path.is_file():
        existing_identity = load_json(identity_path)
        validate_wandb_identity(
            existing_identity,
            run_id=resolved_run_id,
            mode=wandb_mode,
            source_digest=source_digest,
            report_digest=report_digest,
            config_digest=config_digest,
        )
        return existing_identity
    run = wandb.init(
        project="dynaworld",
        name=run_name,
        tags=[
            "paper-ablation-v2",
            paper_scene_tag(protocol),
            protocol.name,
            lane_name,
            f"world-representation-{world_representation}",
            f"alpha-mode-{alpha_mode}",
            f"render-backend-{render_backend}",
            f"amplitude-convention-{amplitude_convention}",
            *(
                [f"frozen-world-{frozen_world_max_frames or 'all'}"]
                if frozen_world_enabled
                else []
            ),
            f"seed-{seed}",
        ],
        mode=wandb_mode,
        id=resolved_run_id,
        resume="never",
        config=wandb_config,
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
    frozen = lane.get("frozen_world_replay_compiled")
    if isinstance(frozen, Mapping):
        payload.update(
            {
                "frozen_world/accepted": int(bool(frozen.get("accepted", False))),
                "frozen_world/image_max_abs_error": frozen["image"][
                    "max_abs_error"
                ],
                "frozen_world/loss_absolute_delta": frozen["loss"][
                    "absolute_delta"
                ],
                "frozen_world/gradient_global_normalized_l2_error": frozen[
                    "gradient"
                ]["global_normalized_l2_error"],
                "frozen_world/gradient_max_parameter_normalized_l2_error": frozen[
                    "gradient"
                ]["max_parameter_normalized_l2_error"],
                "frozen_world/logical_payload_volume_ratio_not_storage": frozen["payload_bytes"][
                    "compiled_to_replay_logical_volume_ratio"
                ],
                "frozen_world/fallback_fraction": frozen["atlas"][
                    "fallback_fraction"
                ],
                **{
                    f"frozen_world/timing_{key}": value
                    for key, value in frozen["timing_s"].items()
                },
            }
        )
    media_prefix = "star_uvt" if lane_name == "world_tubes" else "free_dynamic_splats"
    for split in ("train", "heldout"):
        path = report_dir / f"{media_prefix}_{split}_view0_side_by_side.mp4"
        if not path.exists():
            path = report_dir / lane_name / path.name
        if path.exists():
            payload[f"media/{split}_view"] = wandb.Video(str(path), format="mp4")
    run.log(payload, step=protocol.steps)
    run_dir = str(run.dir)
    actual_run_id = str(run.id)
    run.finish()
    provenance = {
        "schema_version": 1,
        "project": "dynaworld",
        "name": run_name,
        "mode": wandb_mode,
        "run_id": actual_run_id,
        "run_dir": run_dir,
        "source_digest": source_digest,
        "comparison_report_sha256": report_digest,
        "config_sha256": config_digest,
        "run_file": wandb_file_identity(run_dir, actual_run_id),
    }
    write_json(identity_path, provenance)
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
    uvt_world_representation: str = DEFAULT_UVT_WORLD_REPRESENTATION,
    uvt_alpha_mode: str = DEFAULT_UVT_ALPHA_MODE,
    uvt_render_backend: str = DEFAULT_UVT_RENDER_BACKEND,
    uvt_amplitude_convention: str = DEFAULT_UVT_AMPLITUDE_CONVENTION,
    uvt_retained_depth_samples: int = DEFAULT_UVT_RETAINED_DEPTH_SAMPLES,
    uvt_retained_sigma_extent: float = DEFAULT_UVT_RETAINED_SIGMA_EXTENT,
    uvt_order_certificate_sigma: float = DEFAULT_UVT_ORDER_CERTIFICATE_SIGMA,
    uvt_order_certificate_min_gap: float = DEFAULT_UVT_ORDER_CERTIFICATE_MIN_GAP,
    uvt_spd4_init_precision_z: float | None = None,
    frozen_world_replay_compiled: bool = False,
    frozen_world_max_frames: int = 0,
    worldfoam_initializer: str = DEFAULT_WORLDFOAM_INITIALIZER,
) -> dict[str, Any]:
    seed_dir = out_dir / protocol.name / f"seed_{seed}"
    specs = kernel_specs(
        backward_policy,
        uvt_render_backend=uvt_render_backend,
    )
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
        "uvt_world_representation": uvt_world_representation,
        "uvt_alpha_mode": uvt_alpha_mode,
        "uvt_render_backend": uvt_render_backend,
        "uvt_amplitude_convention": uvt_amplitude_convention,
        "uvt_opacity_semantics": uvt_opacity_semantics(
            uvt_alpha_mode,
            uvt_amplitude_convention,
        ),
        "uvt_retained_depth_samples": uvt_retained_depth_samples,
        "uvt_retained_sigma_extent": uvt_retained_sigma_extent,
        "uvt_order_certificate_sigma": uvt_order_certificate_sigma,
        "uvt_order_certificate_min_gap": uvt_order_certificate_min_gap,
        "uvt_effective_backward_policy": effective_uvt_backward_policy(
            uvt_render_backend,
            backward_policy,
        ),
        "uvt_spd4_init_precision_z": uvt_spd4_init_precision_z,
        "frozen_world_replay_compiled": frozen_world_replay_compiled,
        "frozen_world_max_frames": frozen_world_max_frames,
        "manifest_validation": validate_manifest(protocol),
        "kernels": {name: spec.as_dict() for name, spec in specs.items()},
        "comparison_lane_commands": comparison_lane_commands(
            protocol_path,
            protocol,
            seed,
            comparison_dir,
            backward_policy=backward_policy,
            device=device,
            uvt_world_representation=uvt_world_representation,
            uvt_alpha_mode=uvt_alpha_mode,
            uvt_render_backend=uvt_render_backend,
            uvt_amplitude_convention=uvt_amplitude_convention,
            uvt_retained_depth_samples=uvt_retained_depth_samples,
            uvt_retained_sigma_extent=uvt_retained_sigma_extent,
            uvt_order_certificate_sigma=uvt_order_certificate_sigma,
            uvt_order_certificate_min_gap=uvt_order_certificate_min_gap,
            uvt_spd4_init_precision_z=uvt_spd4_init_precision_z,
            frozen_world_replay_compiled=frozen_world_replay_compiled,
            frozen_world_max_frames=frozen_world_max_frames,
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
    uvt_world_representation: str = DEFAULT_UVT_WORLD_REPRESENTATION,
    uvt_alpha_mode: str = DEFAULT_UVT_ALPHA_MODE,
    uvt_render_backend: str = DEFAULT_UVT_RENDER_BACKEND,
    uvt_amplitude_convention: str = DEFAULT_UVT_AMPLITUDE_CONVENTION,
    uvt_retained_depth_samples: int = DEFAULT_UVT_RETAINED_DEPTH_SAMPLES,
    uvt_retained_sigma_extent: float = DEFAULT_UVT_RETAINED_SIGMA_EXTENT,
    uvt_order_certificate_sigma: float = DEFAULT_UVT_ORDER_CERTIFICATE_SIGMA,
    uvt_order_certificate_min_gap: float = DEFAULT_UVT_ORDER_CERTIFICATE_MIN_GAP,
    uvt_spd4_init_precision_z: float | None = None,
    frozen_world_replay_compiled: bool = False,
    frozen_world_max_frames: int = 0,
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
    manifest_validation = validate_manifest(protocol)
    materialize_isolated_comparison_report(
        protocol_path,
        protocol,
        seed,
        comparison_dir,
        backward_policy=backward_policy,
        device=device,
        reuse_existing=reuse_existing,
        uvt_world_representation=uvt_world_representation,
        uvt_alpha_mode=uvt_alpha_mode,
        uvt_render_backend=uvt_render_backend,
        uvt_amplitude_convention=uvt_amplitude_convention,
        uvt_retained_depth_samples=uvt_retained_depth_samples,
        uvt_retained_sigma_extent=uvt_retained_sigma_extent,
        uvt_order_certificate_sigma=uvt_order_certificate_sigma,
        uvt_order_certificate_min_gap=uvt_order_certificate_min_gap,
        uvt_spd4_init_precision_z=uvt_spd4_init_precision_z,
        frozen_world_replay_compiled=frozen_world_replay_compiled,
        frozen_world_max_frames=frozen_world_max_frames,
        allow_local_mps_execution=allow_local_mps_execution,
        expected_source=provenance,
        expected_dataset_input_identity=manifest_validation[
            "input_identity"
        ],
    )
    comparison_report = load_json(comparison_report_path)
    validate_comparison_report(
        comparison_report,
        protocol,
        backward_policy=backward_policy,
        manifest_validation=manifest_validation,
        uvt_world_representation=uvt_world_representation,
        uvt_alpha_mode=uvt_alpha_mode,
        uvt_render_backend=uvt_render_backend,
        uvt_amplitude_convention=uvt_amplitude_convention,
        uvt_retained_depth_samples=uvt_retained_depth_samples,
        uvt_retained_sigma_extent=uvt_retained_sigma_extent,
        uvt_order_certificate_sigma=uvt_order_certificate_sigma,
        uvt_order_certificate_min_gap=uvt_order_certificate_min_gap,
        uvt_spd4_init_precision_z=uvt_spd4_init_precision_z,
        frozen_world_replay_compiled=frozen_world_replay_compiled,
        frozen_world_max_frames=frozen_world_max_frames,
    )

    powerfoam_summary_path = worldfoam_dir / "paper_protocol_summary.json"
    powerfoam_best_path = worldfoam_dir / "best_metrics.json"
    powerfoam_eval_history_path = worldfoam_dir / "eval_metrics_history.jsonl"
    powerfoam_resolved_config_path = worldfoam_dir / "resolved_config.json"
    powerfoam_final_checkpoint_path = worldfoam_dir / "checkpoint_final.pt"
    powerfoam_train_history_path = worldfoam_dir / "train_metrics_history.jsonl"
    powerfoam_train_media_path = (
        worldfoam_dir / f"side_by_side_step_{protocol.steps:04d}.mp4"
    )
    powerfoam_heldout_media_path = (
        worldfoam_dir
        / f"heldout_side_by_side_step_{protocol.steps:04d}.mp4"
    )
    powerfoam_wandb_identity_path = worldfoam_dir / "wandb_identity.json"
    powerfoam_identity_path = worldfoam_dir / "execution_identity.json"
    powerfoam_expected_config = powerfoam_config(
        raw_protocol,
        protocol,
        seed,
        worldfoam_dir,
        wandb_mode=wandb_mode,
        device=device,
        worldfoam_initializer=worldfoam_initializer,
    )
    powerfoam_artifacts = {
        "paper_protocol_summary": powerfoam_summary_path,
        "best_metrics": powerfoam_best_path,
        "eval_metrics_history": powerfoam_eval_history_path,
        "resolved_config": powerfoam_resolved_config_path,
        "checkpoint_final": powerfoam_final_checkpoint_path,
        "train_metrics_history": powerfoam_train_history_path,
        "final_train_media": powerfoam_train_media_path,
        "final_heldout_media": powerfoam_heldout_media_path,
        "wandb_identity": powerfoam_wandb_identity_path,
    }
    powerfoam_init_identity = powerfoam_initializer_identity(
        powerfoam_expected_config,
        requested_initializer=worldfoam_initializer,
    )
    powerfoam_command = worldfoam_lane_command(
        protocol_path,
        seed,
        worldfoam_dir,
        device=device,
        wandb_mode=wandb_mode,
        worldfoam_initializer=worldfoam_initializer,
        allow_local_mps_execution=allow_local_mps_execution,
        allow_high_risk_local_mps=allow_high_risk_local_mps,
    )
    reuse_powerfoam = reuse_existing and powerfoam_summary_path.exists()
    powerfoam_config_binding: dict[str, Any] | None = None
    if reuse_powerfoam and powerfoam_resolved_config_path.is_file():
        try:
            powerfoam_config_binding = worldfoam_resolved_config_binding(
                powerfoam_expected_config,
                powerfoam_resolved_config_path,
            )
        except ValueError:
            reuse_powerfoam = False
    if reuse_powerfoam:
        powerfoam_identity = (
            load_json(powerfoam_identity_path)
            if powerfoam_identity_path.exists()
            else None
        )
        reuse_powerfoam = bool(
            isinstance(powerfoam_identity, Mapping)
            and powerfoam_identity.get("source_start") == provenance
            and powerfoam_identity.get("source_finish") == provenance
            and powerfoam_identity.get("dataset_input_identity")
            == manifest_validation["input_identity"]
            and powerfoam_identity.get("initializer_identity")
            == powerfoam_init_identity
            and powerfoam_identity.get("resolved_config_binding")
            == powerfoam_config_binding
            and powerfoam_identity.get("protocol_sha256")
            == file_sha256(protocol_path)
            and list(powerfoam_identity.get("command", ())) == powerfoam_command
            and isinstance(powerfoam_identity.get("artifacts"), Mapping)
            and all(
                path.is_file()
                and int(path.stat().st_size) > 0
                and powerfoam_identity.get("artifacts", {})
                .get(name, {})
                .get("sha256")
                == file_sha256(path)
                for name, path in powerfoam_artifacts.items()
            )
        )
    if not reuse_powerfoam:
        source_start = source_provenance()
        if source_start != provenance:
            raise RuntimeError("source drifted before the WorldFoam paper lane")
        powerfoam_live_resources = None
        if str(device).lower() == "mps":
            powerfoam_live_resources = live_resource_snapshot()
            require_live_resources(powerfoam_live_resources)
        subprocess.run(powerfoam_command, cwd=ROOT, check=True)
        source_finish = source_provenance()
        if source_start != source_finish:
            raise RuntimeError("source changed while the WorldFoam paper lane executed")
        for artifact_path in powerfoam_artifacts.values():
            if not artifact_path.is_file() or artifact_path.stat().st_size <= 0:
                raise FileNotFoundError(
                    f"WorldFoam paper lane did not produce {artifact_path}"
                )
        powerfoam_config_binding = worldfoam_resolved_config_binding(
            powerfoam_expected_config,
            powerfoam_resolved_config_path,
        )
        finalize_worldfoam_wandb_identity(
            powerfoam_wandb_identity_path,
            expected_run_id=str(
                powerfoam_expected_config["logging"]["wandb_run_id"]
            ),
            expected_mode=wandb_mode,
            source=provenance,
            paper_summary_path=powerfoam_summary_path,
            resolved_config_path=powerfoam_resolved_config_path,
        )
        write_json(
            powerfoam_identity_path,
            {
                "schema_version": 1,
                "lane": "worldfoam",
                "protocol": protocol.as_dict(),
                "protocol_sha256": file_sha256(protocol_path),
                "command": powerfoam_command,
                "source_start": source_start,
                "source_finish": source_finish,
                "dataset_input_identity": manifest_validation[
                    "input_identity"
                ],
                "initializer_identity": powerfoam_init_identity,
                "resolved_config_binding": powerfoam_config_binding,
                "live_resources_at_launch": powerfoam_live_resources,
                "artifacts": {
                    name: file_identity(path, role=f"worldfoam:{name}")
                    for name, path in powerfoam_artifacts.items()
                },
            },
        )
    if powerfoam_config_binding is None:
        powerfoam_config_binding = worldfoam_resolved_config_binding(
            powerfoam_expected_config,
            powerfoam_resolved_config_path,
        )
    worldfoam_wandb_identity = finalize_worldfoam_wandb_identity(
        powerfoam_wandb_identity_path,
        expected_run_id=str(
            powerfoam_expected_config["logging"]["wandb_run_id"]
        ),
        expected_mode=wandb_mode,
        source=provenance,
        paper_summary_path=powerfoam_summary_path,
        resolved_config_path=powerfoam_resolved_config_path,
    )
    powerfoam_summary = load_json(powerfoam_summary_path)
    if int(powerfoam_summary["cost"]["serialized_checkpoint_bytes"]) != int(
        powerfoam_final_checkpoint_path.stat().st_size
    ):
        raise ValueError(
            "WorldFoam serialized checkpoint size does not match cost evidence"
        )
    powerfoam_best = load_json(powerfoam_best_path)
    powerfoam_final_metrics = load_final_powerfoam_metrics(
        powerfoam_eval_history_path,
        expected_step=protocol.steps,
    )
    validate_lane_cost(
        "worldfoam",
        {
            "steps": powerfoam_summary["cost"]["optimizer_steps"],
            "paper_protocol": powerfoam_summary,
        },
        protocol,
        seed=seed,
    )
    for contract_name, schema_version in (
        ("paper_dataset_bundle", PAPER_DATASET_BUNDLE_SCHEMA_VERSION),
        ("paper_evaluator", PAPER_EVALUATOR_SCHEMA_VERSION),
        ("paper_runtime", PAPER_RUNTIME_SCHEMA_VERSION),
    ):
        validate_hashed_contract(
            f"WorldFoam {contract_name}",
            powerfoam_summary.get(contract_name),
            schema_version=schema_version,
        )
        if powerfoam_summary[contract_name] != comparison_report["meta"][
            contract_name
        ]:
            raise ValueError(
                f"WorldFoam {contract_name} does not match World Tubes and "
                "dynamic 3DGS"
            )
    validate_route_native_extension_identity(
        "worldfoam",
        powerfoam_summary.get("route_native_extension"),
    )
    comparison_sample_schedule = comparison_report["star_uvt"][
        "paper_protocol"
    ]["sample_schedule"]
    if powerfoam_summary["sample_schedule"] != comparison_sample_schedule:
        raise ValueError(
            "WorldFoam did not consume the same sample schedule as "
            "World Tubes and dynamic 3DGS"
        )

    wandb_runs = {
        lane_name: _comparison_wandb_log(
            comparison_report,
            protocol,
            lane_name=lane_name,
            seed=seed,
            report_dir=comparison_dir,
            wandb_mode=wandb_mode,
            execution_source=provenance,
        )
        for lane_name in LANE_REPORT_KEYS
    }
    for lane_name in LANE_REPORT_KEYS:
        lane_identity_path = (
            comparison_dir / lane_name / "execution_identity.json"
        )
        lane_identity = load_json(lane_identity_path)
        wandb_identity_path = (
            comparison_dir / lane_name / "wandb_identity.json"
        )
        write_json(
            lane_identity_path,
            {
                **lane_identity,
                "wandb_identity": file_identity(
                    wandb_identity_path,
                    role=f"{lane_name}:wandb_identity",
                ),
            },
        )

    lanes = {
        lane_name: {
            "metrics": comparison_report[report_key]["metrics"],
            "paper_protocol": comparison_report[report_key]["paper_protocol"],
            "wandb": wandb_runs[lane_name],
            "route_native_extension": comparison_report["meta"][
                "route_native_extensions"
            ][lane_name],
            "evidence": build_lane_evidence(
                lane_name,
                comparison_report[report_key],
                frame_count=protocol.dataset.frame_count,
            ),
        }
        for lane_name, report_key in LANE_REPORT_KEYS.items()
    }
    lanes["world_tubes"]["frozen_world_replay_compiled"] = comparison_report[
        "star_uvt"
    ].get("frozen_world_replay_compiled")
    lanes["worldfoam"] = {
        "metrics": powerfoam_final_metrics,
        "reported_checkpoint": "final",
        "best_metric_name": powerfoam_best["best_metric_name"],
        "best_metric_value": powerfoam_best["best_metric_value"],
        "paper_protocol": powerfoam_summary,
        "resolved_config_binding": powerfoam_config_binding,
        "route_native_extension": powerfoam_summary[
            "route_native_extension"
        ],
        "evidence": build_lane_evidence(
            "worldfoam",
            {
                "metrics": powerfoam_final_metrics,
                "paper_protocol": powerfoam_summary,
            },
            frame_count=protocol.dataset.frame_count,
        ),
        "wandb": {
            **worldfoam_wandb_identity,
        },
    }
    provenance_finish = source_provenance()
    if provenance_finish != provenance:
        raise RuntimeError("source changed while the unified paper run executed")
    if require_clean_source:
        require_clean_provenance(provenance_finish)
    summary = {
        "status": "complete",
        "seed": seed,
        "protocol_path": display_path(protocol_path),
        "protocol": protocol.as_dict(),
        "manifest_validation": manifest_validation,
        "world_tubes_requested_backward_policy": backward_policy,
        "world_tubes_backward_policy": effective_uvt_backward_policy(
            uvt_render_backend,
            backward_policy,
        ),
        "uvt_world_representation": uvt_world_representation,
        "uvt_alpha_mode": uvt_alpha_mode,
        "uvt_render_backend": uvt_render_backend,
        "uvt_amplitude_convention": uvt_amplitude_convention,
        "uvt_opacity_semantics": uvt_opacity_semantics(
            uvt_alpha_mode,
            uvt_amplitude_convention,
        ),
        "uvt_retained_depth_samples": uvt_retained_depth_samples,
        "uvt_retained_sigma_extent": uvt_retained_sigma_extent,
        "uvt_order_certificate_sigma": uvt_order_certificate_sigma,
        "uvt_order_certificate_min_gap": uvt_order_certificate_min_gap,
        "uvt_spd4_init_precision_z": uvt_spd4_init_precision_z,
        "frozen_world_replay_compiled": frozen_world_replay_compiled,
        "frozen_world_max_frames": frozen_world_max_frames,
        "comparison_report": display_path(comparison_report_path),
        "worldfoam_dir": display_path(worldfoam_dir),
        "worldfoam_initializer": worldfoam_initializer,
        "worldfoam_initializer_identity": powerfoam_init_identity,
        "common_evidence_contract": {
            "schema_version": 1,
            "dataset_input_identity": manifest_validation[
                "input_identity"
            ],
            "decoded_dataset_bundle": comparison_report["meta"][
                "paper_dataset_bundle"
            ],
            "evaluator": comparison_report["meta"]["paper_evaluator"],
            "runtime": comparison_report["meta"]["paper_runtime"],
            "sample_schedule": comparison_sample_schedule,
        },
        "source": provenance,
        "source_finish": provenance_finish,
        "source_provenance_scope": (
            "parent-runner bracket plus per-child execution_identity.json"
        ),
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
    parser.add_argument(
        "--uvt-world-representation",
        choices=UVT_WORLD_REPRESENTATIONS,
        default=DEFAULT_UVT_WORLD_REPRESENTATION,
        help="Select the historical restricted tube or the native mean+SPD(4) atom lane.",
    )
    parser.add_argument(
        "--uvt-alpha-mode",
        choices=UVT_ALPHA_MODES,
        default=DEFAULT_UVT_ALPHA_MODE,
        help="Select bounded peak-splat alpha or Beer-Lambert optical thickness.",
    )
    parser.add_argument(
        "--uvt-render-backend",
        choices=UVT_RENDER_BACKENDS,
        default=DEFAULT_UVT_RENDER_BACKEND,
        help="Select fast q-UVT, retained-fiber, or certified hybrid rendering.",
    )
    parser.add_argument(
        "--uvt-amplitude-convention",
        choices=UVT_AMPLITUDE_CONVENTIONS,
        default=DEFAULT_UVT_AMPLITUDE_CONVENTION,
        help="Select fiber-integrated amplitude or native world peak density.",
    )
    parser.add_argument(
        "--uvt-retained-depth-samples",
        type=int,
        default=DEFAULT_UVT_RETAINED_DEPTH_SAMPLES,
    )
    parser.add_argument(
        "--uvt-retained-sigma-extent",
        type=float,
        default=DEFAULT_UVT_RETAINED_SIGMA_EXTENT,
    )
    parser.add_argument(
        "--uvt-order-certificate-sigma",
        type=float,
        default=DEFAULT_UVT_ORDER_CERTIFICATE_SIGMA,
    )
    parser.add_argument(
        "--uvt-order-certificate-min-gap",
        type=float,
        default=DEFAULT_UVT_ORDER_CERTIFICATE_MIN_GAP,
    )
    parser.add_argument(
        "--uvt-spd4-init-precision-z",
        type=float,
        default=None,
        help=(
            "Optional full-SPD(4) conditional depth initialization. A large "
            "value gives a near-planar legacy-lift control."
        ),
    )
    parser.add_argument(
        "--frozen-world-replay-compiled",
        action="store_true",
        help=(
            "Train World Tubes once, then compare per-frame replay and one "
            "compiled interval atlas from the identical final checkpoint."
        ),
    )
    parser.add_argument(
        "--frozen-world-max-frames",
        type=int,
        default=0,
        help="Optional frozen-comparison prefix; zero evaluates all protocol frames.",
    )
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument(
        "--worldfoam-initializer",
        default=DEFAULT_WORLDFOAM_INITIALIZER,
        help="Use 'base_config', 'video', or a scene-specific point-cloud path.",
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--require-clean-source",
        action="store_true",
        help="Compatibility flag; paper execution requires clean source by default.",
    )
    source_group.add_argument(
        "--allow-dirty-source",
        action="store_true",
        help=(
            "Allow a labelled mechanical smoke from dirty source. Such a run "
            "is ineligible for paper aggregation."
        ),
    )
    parser.add_argument(
        "--allow-local-mps-execution",
        action="store_true",
        help="Enable only after explicit user approval; local MPS execution is otherwise fail-closed.",
    )
    parser.add_argument(
        "--allow-high-risk-local-mps",
        action="store_true",
        help="Second acknowledgement for a preflight estimate above 60%% of host physical memory.",
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
        uvt_world_representation=args.uvt_world_representation,
        uvt_alpha_mode=args.uvt_alpha_mode,
        uvt_render_backend=args.uvt_render_backend,
        uvt_amplitude_convention=args.uvt_amplitude_convention,
        uvt_retained_depth_samples=args.uvt_retained_depth_samples,
        uvt_retained_sigma_extent=args.uvt_retained_sigma_extent,
        uvt_order_certificate_sigma=args.uvt_order_certificate_sigma,
        uvt_order_certificate_min_gap=args.uvt_order_certificate_min_gap,
        uvt_spd4_init_precision_z=args.uvt_spd4_init_precision_z,
        frozen_world_replay_compiled=args.frozen_world_replay_compiled,
        frozen_world_max_frames=args.frozen_world_max_frames,
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
        uvt_world_representation=args.uvt_world_representation,
        uvt_alpha_mode=args.uvt_alpha_mode,
        uvt_render_backend=args.uvt_render_backend,
        uvt_amplitude_convention=args.uvt_amplitude_convention,
        uvt_retained_depth_samples=args.uvt_retained_depth_samples,
        uvt_retained_sigma_extent=args.uvt_retained_sigma_extent,
        uvt_order_certificate_sigma=args.uvt_order_certificate_sigma,
        uvt_order_certificate_min_gap=args.uvt_order_certificate_min_gap,
        uvt_spd4_init_precision_z=args.uvt_spd4_init_precision_z,
        frozen_world_replay_compiled=args.frozen_world_replay_compiled,
        frozen_world_max_frames=args.frozen_world_max_frames,
        worldfoam_initializer=args.worldfoam_initializer,
        require_clean_source=not args.allow_dirty_source,
        allow_local_mps_execution=args.allow_local_mps_execution,
        allow_high_risk_local_mps=args.allow_high_risk_local_mps,
    )
    print(json.dumps(serialize_config_value(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
