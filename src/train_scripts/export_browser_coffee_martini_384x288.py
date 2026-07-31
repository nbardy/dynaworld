from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from config_utils import load_config_file
from export_dynaworld_browser_bundle import (
    BROWSER_MULTICAM_BUNDLE_VERSION,
    _resolve_seed_provenance,
    export_browser_multicam_dataset_bundle,
)
from multicam_video_data import (
    neural_3d_camera_from_poses_bounds,
    select_multicam_record,
    validate_multicam_camera_split,
    video_path_for_camera,
)
from PIL import Image
from powerfoam_point_cloud import load_point_cloud_xyz_rgb
from train_artifacts import write_json

RECIPE_VERSION = "dynaworld_browser_multicam_export_recipe/v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "src/train_configs/browser_coffee_martini_train17_holdout1_384x288_export.jsonc"
)
EXPECTED_FRAME_INDICES = [
    0,
    20,
    40,
    60,
    80,
    100,
    120,
    140,
    159,
    179,
    199,
    219,
    239,
    259,
    279,
    299,
]
GIB = 1024**3
MIB = 1024**2


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _required_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"Export recipe field {key!r} must be an object.")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_export_recipe(config_path: Path) -> dict[str, Any]:
    config = load_config_file(config_path)
    if config.get("version") != RECIPE_VERSION:
        raise ValueError(f"Export recipe must declare version {RECIPE_VERSION!r}.")

    dataset = _required_mapping(config, "dataset")
    seed = _required_mapping(config, "seed")
    output = _required_mapping(config, "output")
    preflight = _required_mapping(config, "preflight")
    target_size = tuple(int(value) for value in dataset.get("target_size", []))
    frame_indices = [int(value) for value in dataset.get("frame_indices", [])]
    if target_size != (288, 384):
        raise ValueError("The checked-in real-resolution lane must remain target_size=[288, 384].")
    if frame_indices != EXPECTED_FRAME_INDICES:
        raise ValueError("The checked-in lane must preserve the exact canonical 16-frame schedule.")
    if not bool(dataset.get("sparse_frame_decode")):
        raise ValueError("The 384x288 lane requires sparse_frame_decode=true.")

    manifest_path = _repo_path(dataset["manifest"])
    manifest_sha256 = _sha256(manifest_path)
    if manifest_sha256 != str(dataset["expected_manifest_sha256"]):
        raise ValueError(
            "Canonical manifest hash drifted from the checked-in export recipe: "
            f"{manifest_sha256}."
        )
    data_cfg = {
        "multicam_manifest": str(manifest_path),
        "multicam_split": str(dataset["split"]),
        "multicam_sample_id": str(dataset["sample_id"]),
    }
    record = select_multicam_record(data_cfg)
    train_cameras = [str(camera) for camera in record["train_cameras"]]
    heldout_cameras = [str(camera) for camera in record["heldout_cameras"]]
    expected_contract = {
        "frame_count": int(dataset["expected_source_frame_count"]),
        "train_camera_count": int(dataset["expected_train_camera_count"]),
        "heldout_cameras": [str(camera) for camera in dataset["expected_heldout_cameras"]],
        "anchor_camera": str(dataset["expected_anchor_camera"]),
    }
    actual_contract = {
        "frame_count": int(record["frame_count"]),
        "train_camera_count": len(train_cameras),
        "heldout_cameras": heldout_cameras,
        "anchor_camera": str(record["anchor_camera"]),
    }
    if actual_contract != expected_contract:
        raise ValueError(
            "Canonical Coffee Martini manifest drifted from the checked-in export recipe: "
            f"expected {expected_contract}, got {actual_contract}."
        )
    if set(train_cameras) & set(heldout_cameras):
        raise ValueError("Canonical train and heldout camera sets overlap.")
    validate_multicam_camera_split(
        train_cameras=train_cameras,
        heldout_cameras=heldout_cameras,
        anchor_camera=str(record["anchor_camera"]),
        condition_camera=str(record["condition_camera"]),
    )
    for camera_name in train_cameras + heldout_cameras:
        source_path = _repo_path(video_path_for_camera(record, camera_name))
        if not source_path.is_file():
            raise FileNotFoundError(f"Canonical source video is missing: {source_path}")

    seed_point_cloud_path = _repo_path(seed["point_cloud"])
    seed_report_path = _repo_path(seed["provenance_report"])
    if not seed_point_cloud_path.is_file():
        raise FileNotFoundError(
            f"Verified seed cloud is missing: {seed_point_cloud_path}. "
            "Rebuild it with the existing known-pose pycolmap recipe."
        )
    if not seed_report_path.is_file():
        raise FileNotFoundError(
            f"Verified seed provenance report is missing: {seed_report_path}."
        )
    point_cloud_sha256 = _sha256(seed_point_cloud_path)
    if point_cloud_sha256 != str(seed["expected_point_cloud_sha256"]):
        raise ValueError(
            "Verified seed point-cloud hash drifted from the checked-in export recipe: "
            f"{point_cloud_sha256}."
        )
    seed_count = int(seed["count"])
    provenance = _resolve_seed_provenance(
        report_path=seed_report_path,
        seed_point_cloud_path=seed_point_cloud_path,
        train_cameras=train_cameras,
        heldout_cameras=heldout_cameras,
        allow_unverified=False,
    )
    points, _ = load_point_cloud_xyz_rgb(seed_point_cloud_path)
    if int(points.shape[0]) < seed_count:
        raise ValueError(
            f"Verified seed cloud has {int(points.shape[0])} points but recipe requests {seed_count}."
        )

    output_directory = _repo_path(output["directory"])
    bundle_filename = str(output["bundle_filename"])
    if Path(bundle_filename).name != bundle_filename or not bundle_filename.endswith(".json"):
        raise ValueError("output.bundle_filename must be a plain JSON filename.")
    if "384x288" not in bundle_filename or "verified_sparse" not in bundle_filename:
        raise ValueError(
            "The real-resolution bundle name must distinguish 384x288 verified sparse output."
        )

    thresholds = {
        "minimum_available_memory_bytes": int(preflight["minimum_available_memory_bytes"]),
        "minimum_free_memory_fraction": float(preflight["minimum_free_memory_fraction"]),
        "maximum_swap_used_fraction": float(preflight["maximum_swap_used_fraction"]),
        "maximum_load_5m_per_logical_cpu": float(
            preflight["maximum_load_5m_per_logical_cpu"]
        ),
        "minimum_free_disk_bytes": int(preflight["minimum_free_disk_bytes"]),
        "working_set_headroom_multiplier": float(preflight["working_set_headroom_multiplier"]),
    }
    if thresholds["working_set_headroom_multiplier"] < 1.0:
        raise ValueError("working_set_headroom_multiplier must be at least 1.0.")

    return {
        "config_path": config_path.resolve(),
        "manifest_path": manifest_path,
        "record": record,
        "sample_id": str(dataset["sample_id"]),
        "split": str(dataset["split"]),
        "target_size": target_size,
        "frame_indices": frame_indices,
        "train_cameras": train_cameras,
        "heldout_cameras": heldout_cameras,
        "seed_point_cloud_path": seed_point_cloud_path,
        "seed_report_path": seed_report_path,
        "seed_count": seed_count,
        "manifest_sha256": manifest_sha256,
        "point_cloud_sha256": point_cloud_sha256,
        "seed_provenance": provenance,
        "output_directory": output_directory,
        "bundle_filename": bundle_filename,
        "preflight_report_path": _repo_path(output["preflight_report"]),
        "thresholds": thresholds,
    }


def estimate_export_resources(recipe: dict[str, Any]) -> dict[str, int]:
    height, width = recipe["target_size"]
    camera_count = len(recipe["train_cameras"]) + len(recipe["heldout_cameras"])
    frame_count = len(recipe["frame_indices"])
    decoded_rgb_f32_bytes = camera_count * frame_count * height * width * 3 * 4
    atlas_rgb8_bytes = camera_count * frame_count * height * width * 3
    legacy_eager_rgb_f32_bytes = (
        camera_count
        * (max(recipe["frame_indices"]) + 1)
        * height
        * width
        * 3
        * 4
    )
    estimated_peak_working_set_bytes = (
        2 * decoded_rgb_f32_bytes
        + 2 * atlas_rgb8_bytes
        + 512 * MIB
    )
    estimated_output_upper_bound_bytes = math.ceil(1.25 * atlas_rgb8_bytes) + 16 * MIB
    required_available_memory_bytes = max(
        recipe["thresholds"]["minimum_available_memory_bytes"],
        math.ceil(
            estimated_peak_working_set_bytes
            * recipe["thresholds"]["working_set_headroom_multiplier"]
        ),
    )
    required_free_disk_bytes = max(
        recipe["thresholds"]["minimum_free_disk_bytes"],
        4 * estimated_output_upper_bound_bytes,
    )
    return {
        "decoded_rgb_f32_bytes": decoded_rgb_f32_bytes,
        "atlas_rgb8_bytes": atlas_rgb8_bytes,
        "legacy_eager_rgb_f32_bytes": legacy_eager_rgb_f32_bytes,
        "legacy_eager_copy_peak_floor_bytes": 2 * legacy_eager_rgb_f32_bytes,
        "estimated_peak_working_set_bytes": estimated_peak_working_set_bytes,
        "estimated_output_upper_bound_bytes": estimated_output_upper_bound_bytes,
        "required_available_memory_bytes": required_available_memory_bytes,
        "required_free_disk_bytes": required_free_disk_bytes,
    }


def _command_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, timeout=10)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"Host preflight command failed: {' '.join(command)}: {error}") from error


def capture_host_resources(output_parent: Path) -> dict[str, Any]:
    physical_memory_bytes = int(_command_output(["sysctl", "-n", "hw.memsize"]).strip())
    pressure_output = _command_output(["memory_pressure"])
    free_match = re.search(r"System-wide memory free percentage:\s*([0-9.]+)%", pressure_output)
    if free_match is None:
        raise RuntimeError("Could not parse free-memory percentage from memory_pressure.")
    free_memory_fraction = float(free_match.group(1)) / 100.0

    swap_output = _command_output(["sysctl", "-n", "vm.swapusage"])
    swap_match = re.search(
        r"total\s*=\s*([0-9.]+)M\s+used\s*=\s*([0-9.]+)M\s+free\s*=\s*([0-9.]+)M",
        swap_output,
    )
    if swap_match is None:
        raise RuntimeError("Could not parse vm.swapusage.")
    swap_total_bytes = round(float(swap_match.group(1)) * MIB)
    swap_used_bytes = round(float(swap_match.group(2)) * MIB)

    existing_parent = output_parent
    while not existing_parent.exists() and existing_parent != existing_parent.parent:
        existing_parent = existing_parent.parent
    disk = shutil.disk_usage(existing_parent)
    load_1m, load_5m, load_15m = os.getloadavg()
    logical_cpu_count = int(os.cpu_count() or 1)
    return {
        "physical_memory_bytes": physical_memory_bytes,
        "free_memory_fraction": free_memory_fraction,
        "available_memory_bytes": math.floor(physical_memory_bytes * free_memory_fraction),
        "swap_total_bytes": swap_total_bytes,
        "swap_used_bytes": swap_used_bytes,
        "swap_used_fraction": 0.0 if swap_total_bytes == 0 else swap_used_bytes / swap_total_bytes,
        "free_disk_bytes": int(disk.free),
        "logical_cpu_count": logical_cpu_count,
        "load_1m_per_logical_cpu": float(load_1m) / logical_cpu_count,
        "load_5m_per_logical_cpu": float(load_5m) / logical_cpu_count,
        "load_15m_per_logical_cpu": float(load_15m) / logical_cpu_count,
    }


def evaluate_preflight(
    recipe: dict[str, Any],
    host: dict[str, Any],
    *,
    overwrite: bool,
) -> dict[str, Any]:
    estimates = estimate_export_resources(recipe)
    thresholds = recipe["thresholds"]
    failures = []
    if host["free_memory_fraction"] < thresholds["minimum_free_memory_fraction"]:
        failures.append("free_memory_fraction")
    if host["available_memory_bytes"] < estimates["required_available_memory_bytes"]:
        failures.append("available_memory_bytes")
    if host["swap_used_fraction"] > thresholds["maximum_swap_used_fraction"]:
        failures.append("swap_used_fraction")
    if (
        host["load_5m_per_logical_cpu"]
        > thresholds["maximum_load_5m_per_logical_cpu"]
    ):
        failures.append("load_5m_per_logical_cpu")
    if host["free_disk_bytes"] < estimates["required_free_disk_bytes"]:
        failures.append("free_disk_bytes")
    output_directory = recipe["output_directory"]
    output_is_occupied = (
        output_directory.exists() or output_directory.is_symlink()
    ) and (
        output_directory.is_symlink()
        or not output_directory.is_dir()
        or next(output_directory.iterdir(), None) is not None
    )
    if output_is_occupied and not overwrite:
        failures.append("output_directory_not_empty")
    return {
        "version": "dynaworld_browser_export_preflight/v1",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "safe" if not failures else "blocked",
        "failures": failures,
        "config_path": str(recipe["config_path"]),
        "output_directory": str(recipe["output_directory"]),
        "overwrite": bool(overwrite),
        "inputs": {
            "manifest_path": str(recipe["manifest_path"]),
            "manifest_sha256": recipe["manifest_sha256"],
            "seed_point_cloud_path": str(recipe["seed_point_cloud_path"]),
            "seed_point_cloud_sha256": recipe["point_cloud_sha256"],
            "seed_provenance_report_path": str(recipe["seed_report_path"]),
        },
        "host": host,
        "estimates": estimates,
        "thresholds": thresholds,
    }


def verify_generated_bundle(recipe: dict[str, Any], bundle_path: Path) -> dict[str, Any]:
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    if payload.get("version") != BROWSER_MULTICAM_BUNDLE_VERSION:
        raise ValueError("Generated bundle has an unsupported version.")
    width_height = [recipe["target_size"][1], recipe["target_size"][0]]
    if payload.get("decode_size") != width_height:
        raise ValueError(f"Generated decode_size is not {width_height}.")
    if payload.get("frame_indices") != recipe["frame_indices"]:
        raise ValueError("Generated frame schedule drifted from the checked-in recipe.")
    if payload.get("dataset_contract", {}).get("frame_decode") != "sparse_exact":
        raise ValueError("Generated bundle did not record sparse exact frame decoding.")
    if len(payload.get("seed_points_xyzrgb", [])) != recipe["seed_count"]:
        raise ValueError("Generated seed count drifted from the checked-in recipe.")
    provenance = payload.get("seed_provenance", {})
    if provenance.get("train_only_verified") is not True:
        raise ValueError("Generated bundle lost verified train-only seed provenance.")
    if set(provenance.get("input_cameras", [])) & set(recipe["heldout_cameras"]):
        raise ValueError("Generated seed provenance includes a heldout camera.")

    expected_roles = [
        *(("train", camera) for camera in recipe["train_cameras"]),
        *(("heldout", camera) for camera in recipe["heldout_cameras"]),
    ]
    actual_roles = [(camera["role"], camera["name"]) for camera in payload.get("cameras", [])]
    if actual_roles != expected_roles:
        raise ValueError(f"Generated camera roles drifted: {actual_roles}.")

    height, width = recipe["target_size"]
    atlas_size = (width * len(recipe["frame_indices"]), height)
    atlas_bytes = 0
    record = recipe["record"]
    rows_by_name = {camera["name"]: camera for camera in payload["cameras"]}
    for role, camera_name in expected_roles:
        del role
        row = rows_by_name[camera_name]
        atlas_path = bundle_path.parent / row["frame_atlas_url"].removeprefix("./")
        with Image.open(atlas_path) as atlas:
            if atlas.size != atlas_size:
                raise ValueError(f"Atlas {atlas_path} has size {atlas.size}, expected {atlas_size}.")
        atlas_bytes += atlas_path.stat().st_size

        K, _ = neural_3d_camera_from_poses_bounds(
            record,
            camera_name,
            H=height,
            W=width,
            device=torch.device("cpu"),
        )
        expected_intrinsics = [
            float(K[0, 0]) / width,
            float(K[1, 1]) / height,
            float(K[0, 2]) / width,
            float(K[1, 2]) / height,
        ]
        torch.testing.assert_close(
            torch.tensor(row["intrinsics"]),
            torch.tensor(expected_intrinsics),
            rtol=1.0e-6,
            atol=1.0e-7,
        )

    return {
        "version": "dynaworld_browser_export_result/v1",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "verified",
        "bundle_path": str(bundle_path),
        "bundle_bytes": bundle_path.stat().st_size,
        "atlas_bytes": atlas_bytes,
        "decode_size": width_height,
        "frame_indices": recipe["frame_indices"],
        "train_camera_count": len(recipe["train_cameras"]),
        "heldout_cameras": recipe["heldout_cameras"],
        "seed_count": recipe["seed_count"],
        "seed_train_only_verified": True,
    }


def run_export(recipe: dict[str, Any], *, overwrite: bool) -> Path:
    output_directory = recipe["output_directory"]
    output_parent = output_directory.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    staging_directory = Path(
        tempfile.mkdtemp(prefix=f".{output_directory.name}.", dir=output_parent)
    )
    try:
        staging_bundle = staging_directory / recipe["bundle_filename"]
        export_browser_multicam_dataset_bundle(
            manifest_path=recipe["manifest_path"],
            sample_id=recipe["sample_id"],
            split=recipe["split"],
            seed_point_cloud_path=recipe["seed_point_cloud_path"],
            output_path=staging_bundle,
            target_size=recipe["target_size"],
            frame_indices=recipe["frame_indices"],
            seed_count=recipe["seed_count"],
            seed_provenance_report_path=recipe["seed_report_path"],
            sparse_frame_decode=True,
        )
        result = verify_generated_bundle(recipe, staging_bundle)
        write_json(staging_directory / "export_report.json", result, sort_keys=False)
        if output_directory.exists() or output_directory.is_symlink():
            if not overwrite:
                raise FileExistsError(f"Output path already exists: {output_directory}")
            if output_directory.is_symlink() or not output_directory.is_dir():
                output_directory.unlink()
            else:
                shutil.rmtree(output_directory)
        os.replace(staging_directory, output_directory)
    finally:
        if staging_directory.exists():
            shutil.rmtree(staging_directory)
    return output_directory / recipe["bundle_filename"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed real-resolution Coffee Martini train17/holdout1 browser export."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate inputs and host resources without decoding or writing the bundle.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace a prior generated output directory after preflight passes.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    recipe = load_export_recipe(args.config.resolve())
    host = capture_host_resources(recipe["output_directory"].parent)
    preflight = evaluate_preflight(recipe, host, overwrite=args.overwrite)
    write_json(recipe["preflight_report_path"], preflight, sort_keys=False)
    print(json.dumps(preflight, indent=2))
    if preflight["status"] != "safe":
        print(f"Export blocked; see {recipe['preflight_report_path']}.")
        return 2
    if args.preflight_only:
        print(f"Preflight passed; report: {recipe['preflight_report_path']}")
        return 0
    bundle_path = run_export(recipe, overwrite=args.overwrite)
    print(f"Verified real-resolution browser bundle: {bundle_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
