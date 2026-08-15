#!/usr/bin/env python3
"""Verify the measured WorldFoam native-4D G4 public-quality matrix.

This is deliberately stricter than the unified runner's generic lane schema.
G4 is accepted only from a complete 3-scene x 3-seed x 4-route matrix whose
raw row receipts, protocols, manifests, final checkpoints, held-out media, and
W&B run files remain hash-bound.  A smoke, procedural target, training-view
metric, source-only adapter, CPU/fake-native seam, or old per-frame PowerFoam
row cannot satisfy this contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "src" / "train"
for import_root in (ROOT, TRAIN):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from config_utils import load_config_file, serialize_config_value  # noqa: E402
from paper_training_protocol import resolve_paper_training_protocol  # noqa: E402


ARTIFACT_KIND = "worldfoam-native4d-public-quality-ablation-v1"
ROW_KIND = "worldfoam-native4d-public-quality-row-v1"
SCHEMA_VERSION = 1
DEFAULT_CONFIG = (
    ROOT
    / "src"
    / "train_configs"
    / "paper_protocols"
    / "worldfoam_native4d_g4_public_quality_v1.jsonc"
)
DEFAULT_ARTIFACT = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "worldfoam_native4d_g4_public_quality_v1"
    / "worldfoam_public_quality_ablation.json"
)

REQUIRED_ROUTES = (
    "worldfoam_native4d",
    "worldfoam_framewise_replay",
    "world_tubes",
    "dynamic_3dgs",
)
REQUIRED_METRICS = (
    "heldout_eval_psnr",
    "heldout_eval_ssim",
    "heldout_eval_lpips",
    "heldout_eval_l1",
)
REQUIRED_COST = (
    "optimizer_steps",
    "target_pixels",
    "rasterized_pixels",
    "parameter_count",
    "parameter_bytes",
    "serialized_checkpoint_bytes",
    "final_active_primitive_count_per_render",
    "stored_primitive_state_count",
    "process_lifetime_peak_rss_through_checkpoint_bytes",
    "sampled_peak_mps_driver_during_training_and_checkpoint_bytes",
    "training_and_checkpoint_elapsed_s",
    "process_lifetime_peak_rss_through_heldout_evaluation_bytes",
    "sampled_peak_mps_driver_through_heldout_evaluation_bytes",
    "executor_dataset_and_model_setup_elapsed_s",
    "heldout_evaluation_elapsed_s",
    "full_row_through_heldout_evaluation_elapsed_s",
)
ROW_KEYS = frozenset(
    {
        "schema_version",
        "row_kind",
        "row_id",
        "scene",
        "seed",
        "route",
        "lane",
        "execution_mode",
        "backend",
        "protocol_path",
        "protocol_sha256",
        "dataset_manifest_path",
        "dataset_manifest_sha256",
        "sample_id",
        "train_cameras",
        "heldout_cameras",
        "frame_count",
        "image_size",
        "optimizer_steps",
        "frames_per_step",
        "primitive_state_temporal_scope",
        "target_pixel_budget",
        "sample_schedule_sha256",
        "evaluator_sha256",
        "representation_sha256",
        "source_commit",
        "source_dirty",
        "public_quality_evidence",
        "paper_evidence_eligible",
        "proxy_or_test_artifact",
        "measurement_is_simulated",
        "smoke",
        "dataset_is_public",
        "calibrated_multiview",
        "final_checkpoint_evaluation",
        "full_temporal_heldout_evaluation",
        "route_attestation",
        "checkpoint",
        "heldout_media",
        "wandb_run_file",
        "metrics",
        "cost",
    }
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        serialize_config_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _is_git_commit(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _repo_path(value: Any, *, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty repository path")
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"{name} leaves the repository") from error
    return path


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return payload


def load_contract(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_config_file(path)
    validate_contract(config, config_path=path)
    return config


def _route_specs(config: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = config.get("routes")
    if not isinstance(rows, list):
        raise ValueError("G4 config routes must be a list")
    result = {
        str(_mapping(row, name="route").get("route")): _mapping(row, name="route")
        for row in rows
    }
    if tuple(result) != REQUIRED_ROUTES:
        raise ValueError("G4 routes must retain the frozen order and exact route set")
    return result


def _scene_specs(config: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = config.get("scenes")
    if not isinstance(rows, list):
        raise ValueError("G4 config scenes must be a list")
    result = {
        str(_mapping(row, name="scene").get("scene")): _mapping(row, name="scene")
        for row in rows
    }
    if tuple(result) != ("coffee_martini", "cook_spinach", "cut_roasted_beef"):
        raise ValueError("G4 requires the frozen three-scene Neural3D matrix")
    return result


def _manifest_row(path: Path, sample_id: str) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict) and row.get("sample_id") == sample_id:
            matches.append(row)
    if len(matches) != 1:
        raise ValueError(f"manifest must contain exactly one row for {sample_id}")
    return matches[0]


def validate_contract(
    config: Mapping[str, Any], *, config_path: Path = DEFAULT_CONFIG
) -> dict[str, Any]:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("G4 config schema is missing or stale")
    if config.get("artifact_kind") != ARTIFACT_KIND:
        raise ValueError("G4 config artifact kind changed")
    if config.get("dataset_family") != "neural_3d_video":
        raise ValueError("G4 config must use calibrated Neural3D public data")
    if config.get("device") != "mps":
        raise ValueError("G4 Metal contract must use MPS")
    if config.get("seeds") != [17, 29, 43]:
        raise ValueError("G4 seeds must remain [17, 29, 43]")
    routes = _route_specs(config)
    if routes["worldfoam_native4d"].get("same_representation_group") != routes[
        "worldfoam_framewise_replay"
    ].get("same_representation_group"):
        raise ValueError("compiled and replay WorldFoam routes changed representation")
    if routes["worldfoam_native4d"].get("execution_mode") != "compiled_shared_adjoint":
        raise ValueError("native4d route is not the compiled shared-adjoint route")
    if routes["worldfoam_framewise_replay"].get("execution_mode") != (
        "framewise_same_representation"
    ):
        raise ValueError("WorldFoam control is not same-representation framewise replay")
    if routes["world_tubes"].get("execution_mode") != "selected_time_uvt_replay":
        raise ValueError("World Tubes G4 row must be labelled selected-time UVT replay")

    public = _mapping(config.get("public_protocol"), name="public_protocol")
    expected_public = {
        "dataset_frame_count": 300,
        "image_size": [384, 512],
        "optimizer_steps": 300,
        "frames_per_step": 4,
        "primitive_count": 1024,
        "target_pixel_budget": 235929600,
        "require_calibrated_multiview": True,
        "require_final_checkpoint_heldout_evaluation": True,
        "require_full_temporal_heldout_evaluation": True,
        "require_identical_evaluator_within_scene_seed": True,
        "require_identical_sample_schedule_within_scene_seed": True,
        "require_wandb_run_file": True,
        "require_clean_source": True,
    }
    if dict(public) != expected_public:
        raise ValueError("G4 public protocol drifted from the frozen 512-wide contract")

    scene_receipts: dict[str, Any] = {}
    for scene, spec in _scene_specs(config).items():
        protocol_path = _repo_path(spec.get("protocol"), name=f"{scene}.protocol")
        protocol = resolve_paper_training_protocol(load_config_file(protocol_path))
        if (
            protocol.dataset.frame_count != 300
            or protocol.steps != 300
            or len(protocol.stages) != 1
            or protocol.final_stage.image_size.as_list() != [384, 512]
            or protocol.final_stage.primitive_count != 1024
            or protocol.final_stage.frames_per_step != 4
            or protocol.target_pixel_budget != 235929600
        ):
            raise ValueError(f"{scene} protocol is not the fixed matched-512 contract")
        manifest_path = _repo_path(
            protocol.dataset.manifest, name=f"{scene}.dataset.manifest"
        )
        manifest = _manifest_row(manifest_path, protocol.dataset.sample_id)
        if (
            manifest.get("dataset") != "neural_3d_video"
            or manifest.get("scene") != scene
            or manifest.get("frame_count") != 300
            or manifest.get("train_cameras") != list(protocol.dataset.train_cameras)
            or manifest.get("heldout_cameras") != list(protocol.dataset.heldout_cameras)
            or not manifest.get("heldout_cameras")
            or not manifest.get("dataset_scene_dir")
        ):
            raise ValueError(f"{scene} protocol/manifest calibration contract changed")
        scene_receipts[scene] = {
            "protocol_path": str(protocol_path.relative_to(ROOT)),
            "protocol_sha256": file_sha256(protocol_path),
            "manifest_path": str(manifest_path.relative_to(ROOT)),
            "manifest_sha256": file_sha256(manifest_path),
            "sample_id": protocol.dataset.sample_id,
            "train_cameras": list(protocol.dataset.train_cameras),
            "heldout_cameras": list(protocol.dataset.heldout_cameras),
        }
    return {
        "config_path": str(config_path.resolve()),
        "config_sha256": file_sha256(config_path),
        "scenes": scene_receipts,
        "expected_row_count": 36,
    }


def _verify_file_identity(value: Any, *, name: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, Mapping):
        return [f"{name} identity is missing"]
    try:
        path = _repo_path(value.get("path"), name=f"{name}.path")
    except (TypeError, ValueError) as error:
        return [str(error)]
    if not path.is_file():
        return [f"{name} file is missing: {path}"]
    if value.get("sha256") != file_sha256(path):
        errors.append(f"{name} sha256 changed")
    if value.get("bytes") != path.stat().st_size or path.stat().st_size <= 0:
        errors.append(f"{name} byte count changed or is empty")
    return errors


def _finite_metrics(value: Any, *, label: str) -> tuple[list[str], dict[str, float]]:
    errors: list[str] = []
    if not isinstance(value, Mapping) or set(value) != set(REQUIRED_METRICS):
        return [f"{label} metric keys changed"], {}
    metrics: dict[str, float] = {}
    for key in REQUIRED_METRICS:
        raw = value.get(key)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            errors.append(f"{label} {key} is not numeric")
            continue
        number = float(raw)
        if not math.isfinite(number):
            errors.append(f"{label} {key} is non-finite")
        metrics[key] = number
    if metrics.get("heldout_eval_psnr", -1.0) <= 0.0:
        errors.append(f"{label} heldout PSNR is not positive")
    if not 0.0 <= metrics.get("heldout_eval_ssim", -1.0) <= 1.0:
        errors.append(f"{label} heldout SSIM is outside [0,1]")
    if metrics.get("heldout_eval_lpips", -1.0) < 0.0:
        errors.append(f"{label} heldout LPIPS is negative")
    if metrics.get("heldout_eval_l1", -1.0) < 0.0:
        errors.append(f"{label} heldout L1 is negative")
    return errors, metrics


def _validate_row(
    row: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    config_receipt: Mapping[str, Any],
    artifact_source_commit: str,
) -> list[str]:
    label = str(row.get("row_id", "<unknown-row>"))
    errors: list[str] = []
    expected_keys = set(ROW_KEYS) | {"receipt"}
    if set(row) != expected_keys:
        errors.append(f"{label}: row keys changed")
    raw = {key: row.get(key) for key in ROW_KEYS}
    receipt = row.get("receipt")
    errors.extend(_verify_file_identity(receipt, name=f"{label} raw receipt"))
    if isinstance(receipt, Mapping):
        try:
            receipt_path = _repo_path(
                receipt.get("path"), name=f"{label}.receipt.path"
            )
            if receipt_path.is_file() and _load_json(receipt_path) != raw:
                errors.append(f"{label}: embedded row differs from its raw receipt")
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"{label}: raw receipt could not be checked: {error}")

    scene = str(row.get("scene", ""))
    route = str(row.get("route", ""))
    seed = row.get("seed")
    scenes = _scene_specs(config)
    routes = _route_specs(config)
    if scene not in scenes:
        errors.append(f"{label}: scene is outside the frozen matrix")
    if route not in routes:
        errors.append(f"{label}: route is outside the frozen matrix")
    if seed not in config.get("seeds", []):
        errors.append(f"{label}: seed is outside the frozen matrix")
    expected_id = f"{scene}/seed_{seed}/{route}"
    if row.get("row_id") != expected_id:
        errors.append(f"{label}: row id changed")
    route_spec = routes.get(route, {})
    for key in ("lane", "execution_mode", "backend"):
        if row.get(key) != route_spec.get(key):
            errors.append(f"{label}: {key} differs from route contract")

    scene_receipt = config_receipt.get("scenes", {}).get(scene, {})
    expected_protocol_path = scene_receipt.get("protocol_path")
    if row.get("protocol_path") != expected_protocol_path:
        errors.append(f"{label}: protocol path changed")
    if row.get("protocol_sha256") != scene_receipt.get("protocol_sha256"):
        errors.append(f"{label}: protocol digest changed")
    if row.get("dataset_manifest_path") != scene_receipt.get("manifest_path"):
        errors.append(f"{label}: dataset manifest path changed")
    if row.get("dataset_manifest_sha256") != scene_receipt.get("manifest_sha256"):
        errors.append(f"{label}: dataset manifest digest changed")
    for key in ("sample_id", "train_cameras", "heldout_cameras"):
        if row.get(key) != scene_receipt.get(key):
            errors.append(f"{label}: {key} changed")

    public = config["public_protocol"]
    exact_values = {
        "frame_count": public["dataset_frame_count"],
        "image_size": public["image_size"],
        "optimizer_steps": public["optimizer_steps"],
        "frames_per_step": public["frames_per_step"],
        "primitive_state_temporal_scope": (
            "per_frame" if route == "dynamic_3dgs" else "shared_across_time"
        ),
        "target_pixel_budget": public["target_pixel_budget"],
        "source_commit": artifact_source_commit,
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
    }
    for key, expected in exact_values.items():
        if row.get(key) != expected:
            errors.append(f"{label}: {key} is not publication eligible")
    for key in (
        "sample_schedule_sha256",
        "evaluator_sha256",
        "representation_sha256",
    ):
        if not _is_sha256(row.get(key)):
            errors.append(f"{label}: {key} is invalid")

    attestation = row.get("route_attestation")
    if not isinstance(attestation, Mapping):
        errors.append(f"{label}: route attestation is missing")
    else:
        common_attestation = {
            "real_native": True,
            "native_extension_attested": False,
            "fake_native": False,
            "source_only": False,
            "procedural_target": False,
            "public_target_provider": True,
            "heldout_evaluator": True,
            "full_geometry_trainable": True,
        }
        for key, expected in common_attestation.items():
            if attestation.get(key) is not expected:
                errors.append(f"{label}: route attestation {key} is not {expected}")
        if route == "worldfoam_native4d":
            if attestation.get("compiled_shared_adjoint") is not True:
                errors.append(f"{label}: compiled WorldFoam adjoint is not attested")
            if attestation.get("same_representation_framewise_replay") is not False:
                errors.append(f"{label}: compiled route was relabelled replay")
        elif route == "worldfoam_framewise_replay":
            if attestation.get("compiled_shared_adjoint") is not False:
                errors.append(f"{label}: replay route was relabelled compiled")
            if attestation.get("same_representation_framewise_replay") is not True:
                errors.append(f"{label}: same-representation replay is not attested")

    for key in ("checkpoint", "heldout_media", "wandb_run_file"):
        errors.extend(_verify_file_identity(row.get(key), name=f"{label} {key}"))
    checkpoint = row.get("checkpoint")
    if isinstance(checkpoint, Mapping) and checkpoint.get("step") != 300:
        errors.append(f"{label}: checkpoint is not the final optimizer step")
    media = row.get("heldout_media")
    if isinstance(media, Mapping):
        if media.get("camera_ids") != row.get("heldout_cameras"):
            errors.append(f"{label}: heldout media cameras changed")
        if media.get("frame_count") != 300:
            errors.append(f"{label}: heldout media is not full temporal")
    wandb = row.get("wandb_run_file")
    if isinstance(wandb, Mapping):
        if not isinstance(wandb.get("run_id"), str) or not wandb["run_id"].strip():
            errors.append(f"{label}: W&B run id is missing")
        if wandb.get("mode") not in {"online", "offline"}:
            errors.append(f"{label}: W&B mode is invalid")

    metric_errors, _metrics = _finite_metrics(row.get("metrics"), label=label)
    errors.extend(metric_errors)
    cost = row.get("cost")
    if not isinstance(cost, Mapping) or set(cost) != set(REQUIRED_COST):
        errors.append(f"{label}: cost keys changed")
    else:
        for key in REQUIRED_COST:
            value = cost.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                errors.append(f"{label}: cost {key} is not numeric")
            elif not math.isfinite(float(value)) or float(value) < 0.0:
                errors.append(f"{label}: cost {key} is invalid")
        if cost.get("optimizer_steps") != 300:
            errors.append(f"{label}: measured optimizer-step count changed")
        if cost.get("target_pixels") != 235929600:
            errors.append(f"{label}: measured target-pixel budget changed")
        if cost.get("final_active_primitive_count_per_render") != 1024:
            errors.append(f"{label}: final active primitive count changed")
        expected_stored_state_count = 307200 if route == "dynamic_3dgs" else 1024
        if cost.get("stored_primitive_state_count") != expected_stored_state_count:
            errors.append(f"{label}: stored primitive state count changed")
        full_rss = cost.get(
            "process_lifetime_peak_rss_through_heldout_evaluation_bytes"
        )
        checkpoint_rss = cost.get(
            "process_lifetime_peak_rss_through_checkpoint_bytes"
        )
        if (
            isinstance(full_rss, (int, float))
            and not isinstance(full_rss, bool)
            and isinstance(checkpoint_rss, (int, float))
            and not isinstance(checkpoint_rss, bool)
            and full_rss < checkpoint_rss
        ):
            errors.append(f"{label}: process RSS decreased after heldout evaluation")
        full_mps = cost.get("sampled_peak_mps_driver_through_heldout_evaluation_bytes")
        checkpoint_mps = cost.get(
            "sampled_peak_mps_driver_during_training_and_checkpoint_bytes"
        )
        if (
            isinstance(full_mps, (int, float))
            and not isinstance(full_mps, bool)
            and isinstance(checkpoint_mps, (int, float))
            and not isinstance(checkpoint_mps, bool)
            and full_mps < checkpoint_mps
        ):
            errors.append(f"{label}: sampled MPS peak decreased after heldout evaluation")
        full_elapsed = cost.get("full_row_through_heldout_evaluation_elapsed_s")
        if isinstance(full_elapsed, (int, float)) and not isinstance(
            full_elapsed, bool
        ):
            for key in (
                "executor_dataset_and_model_setup_elapsed_s",
                "training_and_checkpoint_elapsed_s",
                "heldout_evaluation_elapsed_s",
            ):
                component = cost.get(key)
                if (
                    isinstance(component, (int, float))
                    and not isinstance(component, bool)
                    and full_elapsed < component
                ):
                    errors.append(f"{label}: full-row timing is shorter than {key}")
        if isinstance(checkpoint, Mapping):
            if cost.get("serialized_checkpoint_bytes") != checkpoint.get("bytes"):
                errors.append(f"{label}: checkpoint byte accounting changed")
    return errors


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return sum(values) / float(len(values))


def compute_acceptance(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    by_key = {
        (str(row.get("scene")), int(row.get("seed", -1)), str(row.get("route"))): row
        for row in rows
    }
    native = [
        by_key[(scene, seed, "worldfoam_native4d")]
        for scene in _scene_specs(config)
        for seed in config["seeds"]
        if (scene, seed, "worldfoam_native4d") in by_key
    ]
    paired: list[dict[str, Any]] = []
    for scene in _scene_specs(config):
        for seed in config["seeds"]:
            keys = [(scene, seed, route) for route in REQUIRED_ROUTES]
            if not all(key in by_key for key in keys):
                continue
            group = {route: by_key[(scene, seed, route)] for route in REQUIRED_ROUTES}
            native_metrics = group["worldfoam_native4d"]["metrics"]
            replay_metrics = group["worldfoam_framewise_replay"]["metrics"]
            best_gaussian_psnr = max(
                float(group[route]["metrics"]["heldout_eval_psnr"])
                for route in ("world_tubes", "dynamic_3dgs")
            )
            paired.append(
                {
                    "scene": scene,
                    "seed": seed,
                    "native_minus_replay_psnr_db": float(
                        native_metrics["heldout_eval_psnr"]
                    )
                    - float(replay_metrics["heldout_eval_psnr"]),
                    "native_minus_replay_ssim": float(
                        native_metrics["heldout_eval_ssim"]
                    )
                    - float(replay_metrics["heldout_eval_ssim"]),
                    "native_minus_replay_lpips": float(
                        native_metrics["heldout_eval_lpips"]
                    )
                    - float(replay_metrics["heldout_eval_lpips"]),
                    "native_psnr_deficit_vs_best_gaussian_db": best_gaussian_psnr
                    - float(native_metrics["heldout_eval_psnr"]),
                }
            )
    thresholds = config["acceptance"]
    failures: list[str] = []
    if len(rows) != 36:
        failures.append(f"expected 36 measured rows, observed {len(rows)}")
    if len(native) != 9:
        failures.append(f"expected 9 native4d rows, observed {len(native)}")
    if len(paired) != 9:
        failures.append(f"expected 9 complete paired comparisons, observed {len(paired)}")

    native_psnr = [float(row["metrics"]["heldout_eval_psnr"]) for row in native]
    native_ssim = [float(row["metrics"]["heldout_eval_ssim"]) for row in native]
    replay_psnr = [row["native_minus_replay_psnr_db"] for row in paired]
    replay_ssim = [row["native_minus_replay_ssim"] for row in paired]
    replay_lpips = [row["native_minus_replay_lpips"] for row in paired]
    gaussian_deficit = [
        row["native_psnr_deficit_vs_best_gaussian_db"] for row in paired
    ]
    checks = {
        "native_psnr_floor": bool(native_psnr)
        and min(native_psnr) >= thresholds["minimum_native_heldout_psnr"],
        "native_ssim_floor": bool(native_ssim)
        and min(native_ssim) >= thresholds["minimum_native_heldout_ssim"],
        "mean_psnr_vs_replay": bool(replay_psnr)
        and _mean(replay_psnr)
        >= thresholds["minimum_mean_psnr_delta_vs_framewise_replay_db"],
        "worst_psnr_vs_replay": bool(replay_psnr)
        and min(replay_psnr)
        >= thresholds["minimum_worst_psnr_delta_vs_framewise_replay_db"],
        "mean_ssim_vs_replay": bool(replay_ssim)
        and _mean(replay_ssim)
        >= thresholds["minimum_mean_ssim_delta_vs_framewise_replay"],
        "mean_lpips_vs_replay": bool(replay_lpips)
        and _mean(replay_lpips)
        <= thresholds["maximum_mean_lpips_delta_vs_framewise_replay"],
        "mean_psnr_vs_best_gaussian": bool(gaussian_deficit)
        and _mean(gaussian_deficit)
        <= thresholds["maximum_mean_psnr_deficit_vs_best_gaussian_db"],
        "worst_psnr_vs_best_gaussian": bool(gaussian_deficit)
        and max(gaussian_deficit)
        <= thresholds["maximum_worst_psnr_deficit_vs_best_gaussian_db"],
    }
    failures.extend(f"acceptance check failed: {name}" for name, ok in checks.items() if not ok)
    aggregates = {
        "minimum_native_heldout_psnr": min(native_psnr) if native_psnr else None,
        "minimum_native_heldout_ssim": min(native_ssim) if native_ssim else None,
        "mean_native_minus_replay_psnr_db": _mean(replay_psnr) if replay_psnr else None,
        "worst_native_minus_replay_psnr_db": min(replay_psnr) if replay_psnr else None,
        "mean_native_minus_replay_ssim": _mean(replay_ssim) if replay_ssim else None,
        "mean_native_minus_replay_lpips": _mean(replay_lpips) if replay_lpips else None,
        "mean_native_psnr_deficit_vs_best_gaussian_db": (
            _mean(gaussian_deficit) if gaussian_deficit else None
        ),
        "worst_native_psnr_deficit_vs_best_gaussian_db": (
            max(gaussian_deficit) if gaussian_deficit else None
        ),
    }
    return {
        "accepted": not failures,
        "failures": failures,
        "observed_row_count": len(rows),
        "observed_native_row_count": len(native),
        "observed_paired_comparison_count": len(paired),
        "checks": checks,
        "aggregates": aggregates,
        "paired_comparisons": paired,
    }


def artifact_sha256(payload: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {key: value for key, value in payload.items() if key != "artifact_sha256"}
    )


def verify_artifact(
    payload: Mapping[str, Any],
    *,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    failures: list[str] = []
    try:
        config = load_contract(config_path)
        config_receipt = validate_contract(config, config_path=config_path)
    except Exception as error:
        return {
            "accepted": False,
            "failures": [f"G4 config could not be verified: {error}"],
            "observed_row_count": 0,
        }
    expected_top_keys = {
        "schema_version",
        "artifact_kind",
        "status",
        "public_quality_evidence",
        "proxy_or_test_artifact",
        "measurement_is_simulated",
        "matrix_config",
        "matrix_config_sha256",
        "source_commit",
        "rows",
        "acceptance",
        "artifact_sha256",
    }
    if set(payload) != expected_top_keys:
        failures.append("artifact key set changed")
    exact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "status": "measured",
        "public_quality_evidence": True,
        "proxy_or_test_artifact": False,
        "measurement_is_simulated": False,
        "matrix_config": str(config_path.resolve().relative_to(ROOT)),
        "matrix_config_sha256": file_sha256(config_path),
    }
    for key, expected in exact.items():
        if payload.get(key) != expected:
            failures.append(f"artifact {key} changed")
    source_commit = payload.get("source_commit")
    if not _is_git_commit(source_commit):
        failures.append("artifact source commit is invalid")
        source_commit = ""
    rows = payload.get("rows")
    if not isinstance(rows, list):
        failures.append("artifact rows are missing")
        rows = []
    seen: set[tuple[str, int, str]] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            failures.append(f"row {index} is not a mapping")
            continue
        key = (str(row.get("scene")), int(row.get("seed", -1)), str(row.get("route")))
        if key in seen:
            failures.append(f"duplicate row key: {key}")
        seen.add(key)
        failures.extend(
            _validate_row(
                row,
                config=config,
                config_receipt=config_receipt,
                artifact_source_commit=str(source_commit),
            )
        )
    expected_grid = {
        (scene, seed, route)
        for scene in _scene_specs(config)
        for seed in config["seeds"]
        for route in REQUIRED_ROUTES
    }
    missing = sorted(expected_grid - seen)
    extra = sorted(seen - expected_grid)
    if missing:
        failures.append(f"artifact matrix is missing {len(missing)} row(s)")
    if extra:
        failures.append(f"artifact matrix has {len(extra)} extra row(s)")
    evaluator_groups: dict[tuple[str, int], set[str]] = {}
    schedule_groups: dict[tuple[str, int], set[str]] = {}
    representation_groups: dict[tuple[str, int], set[str]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        group = (str(row.get("scene")), int(row.get("seed", -1)))
        evaluator_groups.setdefault(group, set()).add(str(row.get("evaluator_sha256")))
        schedule_groups.setdefault(group, set()).add(str(row.get("sample_schedule_sha256")))
        if row.get("route") in {"worldfoam_native4d", "worldfoam_framewise_replay"}:
            representation_groups.setdefault(group, set()).add(
                str(row.get("representation_sha256"))
            )
    if any(len(values) != 1 for values in evaluator_groups.values()):
        failures.append("paired routes do not share an identical evaluator")
    if any(len(values) != 1 for values in schedule_groups.values()):
        failures.append("paired routes do not share an identical sample schedule")
    if any(len(values) != 1 for values in representation_groups.values()):
        failures.append("compiled and replay WorldFoam changed representation")

    try:
        computed = compute_acceptance(rows, config)
    except Exception as error:
        failures.append(f"acceptance computation failed: {error}")
        computed = {"accepted": False, "failures": [str(error)]}
    if payload.get("acceptance") != computed:
        failures.append("stored G4 acceptance summary changed")
    if computed.get("accepted") is not True:
        failures.extend(str(value) for value in computed.get("failures", ()))
    if payload.get("artifact_sha256") != artifact_sha256(payload):
        failures.append("artifact canonical digest changed")
    return {
        "accepted": not failures,
        "failures": sorted(set(failures)),
        "observed_row_count": len(rows),
        "observed_scene_count": len({row.get("scene") for row in rows if isinstance(row, Mapping)}),
        "observed_seed_count": len({row.get("seed") for row in rows if isinstance(row, Mapping)}),
        "observed_route_count": len({row.get("route") for row in rows if isinstance(row, Mapping)}),
        "computed_acceptance": computed,
    }


def verify_artifact_file(
    path: Path, *, config_path: Path = DEFAULT_CONFIG
) -> dict[str, Any]:
    if not path.is_file():
        return {
            "accepted": False,
            "failures": [f"G4 artifact is missing: {path}"],
            "observed_row_count": 0,
        }
    try:
        payload = _load_json(path)
    except Exception as error:
        return {
            "accepted": False,
            "failures": [f"G4 artifact could not be loaded: {error}"],
            "observed_row_count": 0,
        }
    return verify_artifact(payload, config_path=config_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify WorldFoam native4d G4 public-quality evidence."
    )
    parser.add_argument("artifact", type=Path, nargs="?", default=DEFAULT_ARTIFACT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    report = verify_artifact_file(args.artifact, config_path=args.config)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


__all__ = [
    "ARTIFACT_KIND",
    "DEFAULT_ARTIFACT",
    "DEFAULT_CONFIG",
    "REQUIRED_ROUTES",
    "ROW_KEYS",
    "ROW_KIND",
    "artifact_sha256",
    "canonical_sha256",
    "compute_acceptance",
    "file_sha256",
    "load_contract",
    "validate_contract",
    "verify_artifact",
    "verify_artifact_file",
]
