from __future__ import annotations

import argparse
import json
import os
import shutil
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
LOCAL_PYTHON = ".venv/bin/python"
SRC_TRAIN = ROOT / "src/train"
if str(SRC_TRAIN) not in sys.path:
    sys.path.insert(0, str(SRC_TRAIN))

from config_utils import load_config_file  # noqa: E402

OFFICIAL_FIXTURE_SOURCE = (
    ROOT / "research_experiments/dynamic_foam/fixtures/powerfoam_tiny_height_sv_origin_parity_v1.json"
)
OFFICIAL_FIXTURE_OUTPUT = (
    ROOT / "research_experiments/dynamic_foam/fixtures/powerfoam_tiny_height_sv_official_camera_official_v1.json"
)
OPENCV_FISHEYE_CONFIG = (
    ROOT
    / "src/train_configs/"
    "local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux.jsonc"
)
ALIKED_OUTPUT_DIR = (
    ROOT
    / "research_experiments/dynamic_foam/artifacts"
)
POWERFOAM_OUTPUT_DIR = ROOT / "outputs/powerfoam_metal"
TRAIN_CONFIG_DIR = ROOT / "src/train_configs"
OFFICIAL_FIXTURE_EXPECTED_KEYS = {
    "rendered",
    "alpha",
    "normal_distance",
    "loss",
    "grad_points",
    "grad_radii",
    "grad_density",
    "grad_normals",
    "grad_texel_sites",
    "grad_texel_height",
    "grad_texel_sv_axis",
    "grad_texel_sv_rgb",
}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def command_to_string(command: list[str], env: dict[str, str]) -> str:
    prefixes = [f"{key}={shlex.quote(value)}" for key, value in sorted(env.items())]
    return " ".join([*prefixes, *(shlex.quote(part) for part in command)])


def load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return payload


def require_checks(name: str, checks: list[dict[str, object]]) -> None:
    report = {"name": name, "ok": all(bool(check["passed"]) for check in checks), "checks": checks}
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not report["ok"]:
        failed = [str(check["name"]) for check in checks if not bool(check["passed"])]
        raise RuntimeError(f"{name} failed checks: {', '.join(failed)}")


def default_aliked_output(matcher_type: str) -> Path:
    return (
        ALIKED_OUTPUT_DIR
        / "deepview_03_dog_8cam_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_"
        f"opencv_fisheye_aliked_n16rot_{matcher_type}_minucam2.ply"
    )


def default_train_slug(matcher_type: str) -> str:
    return (
        "local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_"
        "pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_"
        f"opencv_fisheye_aliked_n16rot_{matcher_type}_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux"
    )


def default_train_config_output(matcher_type: str) -> Path:
    return TRAIN_CONFIG_DIR / f"{default_train_slug(matcher_type)}.jsonc"


def default_train_output_dir(matcher_type: str) -> Path:
    return POWERFOAM_OUTPUT_DIR / default_train_slug(matcher_type)


def run_command(command: list[str], *, env: dict[str, str], dry_run: bool) -> None:
    print(command_to_string(command, env), flush=True)
    if dry_run:
        return
    process_env = os.environ.copy()
    process_env.update(env)
    subprocess.run(command, cwd=ROOT, env=process_env, check=True)


def require_canonical_official_fixture_for_tests(path: Path) -> None:
    if path.resolve() != OFFICIAL_FIXTURE_OUTPUT.resolve():
        raise ValueError(
            "official-tests loads the canonical fixture path used by tests/test_powerfoam_direct.py. "
            f"Write or copy the fixture to {rel(OFFICIAL_FIXTURE_OUTPUT)!r}; got {rel(path)!r}."
        )


def validate_official_fixture(path: Path) -> None:
    if not path.exists():
        require_checks("official_fixture", [{"name": "fixture_exists", "passed": False, "evidence": rel(path)}])
    fixture = load_json_object(path)
    metadata = fixture.get("metadata", {})
    expected = fixture.get("expected", {})
    checks = [
        {"name": "fixture_exists", "passed": path.exists(), "evidence": rel(path)},
        {
            "name": "backend_official",
            "passed": isinstance(metadata, dict) and metadata.get("backend") == "official",
            "evidence": metadata.get("backend") if isinstance(metadata, dict) else None,
        },
        {
            "name": "has_inputs",
            "passed": isinstance(fixture.get("inputs"), dict),
            "evidence": sorted(fixture.get("inputs", {}).keys()) if isinstance(fixture.get("inputs"), dict) else None,
        },
        {
            "name": "has_render_options",
            "passed": isinstance(fixture.get("render_options"), dict),
            "evidence": sorted(fixture.get("render_options", {}).keys())
            if isinstance(fixture.get("render_options"), dict)
            else None,
        },
        {
            "name": "has_forward_and_backward_expected_keys",
            "passed": isinstance(expected, dict) and OFFICIAL_FIXTURE_EXPECTED_KEYS.issubset(expected.keys()),
            "evidence": sorted(OFFICIAL_FIXTURE_EXPECTED_KEYS - set(expected.keys()))
            if isinstance(expected, dict)
            else sorted(OFFICIAL_FIXTURE_EXPECTED_KEYS),
        },
    ]
    require_checks("official_fixture", checks)


def check_prereqs() -> int:
    missing = 0
    if shutil.which("uv") is None:
        print("uv command not found", file=sys.stderr)
        missing += 1
    else:
        print("uv command ok")
    try:
        import torch
    except Exception as exc:
        print(f"torch import failed: {exc}", file=sys.stderr)
        missing += 1
    else:
        cuda = bool(torch.cuda.is_available())
        print(f"torch.cuda.is_available={cuda}")
        if not cuda:
            missing += 1
    try:
        import warp  # noqa: F401
    except Exception as exc:
        print(f"warp import failed: {exc}", file=sys.stderr)
        missing += 1
    else:
        print("warp import ok")
    try:
        import pycolmap
    except Exception as exc:
        print(f"pycolmap import failed: {exc}", file=sys.stderr)
        missing += 1
    else:
        print(f"pycolmap import ok: {getattr(pycolmap, '__version__', 'unknown')}")
        print(
            "ONNX-backed ALIKED/LightGlue support is not proven by check; "
            "validate it by running the aliked-cloud builder command."
        )
    return missing


def official_fixture_command(upstream_root: Path, output: Path) -> tuple[list[str], dict[str, str]]:
    return (
        [
            PYTHON,
            "research_experiments/dynamic_foam/make_powerfoam_official_parity_fixture.py",
            "--backend",
            "official",
            "--upstream-root",
            str(upstream_root),
            "--fixture",
            rel(OFFICIAL_FIXTURE_SOURCE),
            "--output",
            rel(output),
        ],
        {"PYTHONPATH": "src/train"},
    )


def official_tests_command() -> tuple[list[str], dict[str, str]]:
    return (
        [
            "uv",
            "run",
            "--with",
            "pytest",
            "python",
            "-m",
            "pytest",
            "tests/test_powerfoam_direct.py::test_powerfoam_direct_matches_official_cuda_fixture_if_present",
            "tests/test_powerfoam_direct.py::test_powerfoam_metal_matches_official_cuda_fixture_shared_backward_if_present",
            "-q",
            "-rs",
        ],
        {"PYTHONPATH": "src/train:third_party/powerfoam-metal"},
    )


def aliked_cloud_command(output: Path, matcher_type: str, max_features: int) -> tuple[list[str], dict[str, str]]:
    return (
        [
            "uv",
            "run",
            "--with",
            "pycolmap==4.0.4",
            "python",
            "research_experiments/dynamic_foam/build_pycolmap_known_pose_point_cloud.py",
            rel(OPENCV_FISHEYE_CONFIG),
            "--output",
            rel(output),
            "--target-size",
            "1024",
            "--frame-indices",
            "0",
            "4",
            "8",
            "12",
            "--camera-model",
            "opencv_fisheye",
            "--camera-mode",
            "per_image",
            "--feature-backend",
            "colmap_cli",
            "--feature-type",
            "aliked_n16rot",
            "--matcher-type",
            matcher_type,
            "--allow-onnx-models",
            "--pycolmap-use-gpu",
            "--pycolmap-device",
            "cuda",
            "--max-features",
            str(max_features),
            "--max-reproj-error",
            "8.0",
            "--xy-extent",
            "100",
            "--z-min",
            "-100",
            "--z-max",
            "100",
            "--min-unique-cameras",
            "2",
        ],
        {"PYTHONPATH": "src/train"},
    )


def aliked_training_command(config_path: Path) -> tuple[list[str], dict[str, str]]:
    return (
        [
            PYTHON,
            "src/train/train_powerfoam_metal.py",
            rel(config_path),
        ],
        {"PYTHONPATH": "src/train:third_party/powerfoam-metal"},
    )


def command_to_string_with_python(command: list[str], env: dict[str, str], *, python_executable: str) -> str:
    command = list(command)
    if command and command[0] == PYTHON:
        command[0] = python_executable
    return command_to_string(command, env)


def runner_command(
    task: str,
    matcher_type: str,
    *extra: str,
    python_executable: str = LOCAL_PYTHON,
) -> str:
    command = [
        python_executable,
        "research_experiments/dynamic_foam/run_powerfoam_external_blockers.py",
        task,
        "--matcher-type",
        matcher_type,
        *extra,
    ]
    return command_to_string(command, {"PYTHONDONTWRITEBYTECODE": "1"})


def build_handoff_manifest(
    *,
    upstream_root: Path,
    official_output: Path,
    aliked_output: Path,
    train_config_output: Path,
    train_output_dir: Path,
    matcher_type: str,
    max_features: int,
    min_point_count: int,
    min_track_mean: float,
    min_track_p90: float,
    max_reproj_median: float,
    max_reproj_p90: float,
    min_verified_pairs: int,
    min_heldout_psnr: float,
    min_heldout_ssim: float,
) -> dict[str, object]:
    official_fixture_cmd, official_fixture_env = official_fixture_command(upstream_root, official_output)
    official_tests_cmd, official_tests_env = official_tests_command()
    aliked_cmd, aliked_env = aliked_cloud_command(aliked_output, matcher_type, max_features)
    aliked_train_cmd, aliked_train_env = aliked_training_command(train_config_output)
    threshold_args = [
        "--min-point-count",
        str(min_point_count),
        "--min-track-mean",
        str(min_track_mean),
        "--min-track-p90",
        str(min_track_p90),
        "--max-reproj-median",
        str(max_reproj_median),
        "--max-reproj-p90",
        str(max_reproj_p90),
        "--min-verified-pairs",
        str(min_verified_pairs),
    ]
    heldout_args = [
        "--min-heldout-psnr",
        str(min_heldout_psnr),
        "--min-heldout-ssim",
        str(min_heldout_ssim),
    ]
    return {
        "objective": (
            "Close the remaining PowerFoam completion blockers by producing the official "
            "CUDA/Warp parity fixture and ONNX-backed ALIKED/LightGlue clean geometry, then "
            "running the matched W&B-backed Metal training row."
        ),
        "host_requirements": {
            "official_fixture_host": ["linux", "cuda", "torch cuda available", "warp-lang", "pinned upstream PowerFoam checkout"],
            "aliked_host": [
                "linux",
                "cuda",
                "COLMAP CLI with CUDA and ONNX feature extraction",
                "cuDNN runtime libraries for ONNX Runtime",
                "pycolmap for known-pose triangulation",
                "ALIKED/LightGlue model downloads allowed",
            ],
            "metal_training_host": ["macOS", "MPS", "built third_party/powerfoam-metal extension"],
        },
        "external_commands": [
            {
                "name": "host_prereq_check",
                "command": runner_command("check", matcher_type, python_executable="python"),
                "expected": "exit code 0 on a CUDA/Warp/pycolmap host; current Mac is expected to fail this check",
            },
            {
                "name": "official_cuda_warp_fixture",
                "command": command_to_string_with_python(
                    official_fixture_cmd,
                    official_fixture_env,
                    python_executable="python",
                ),
                "writes": [rel(official_output)],
                "validate": runner_command("verify-official-fixture", matcher_type, python_executable="python"),
            },
            {
                "name": "aliked_lightglue_point_cloud",
                "command": command_to_string(aliked_cmd, aliked_env),
                "writes": [rel(aliked_output), rel(aliked_output.with_suffix(".json"))],
                "validate": runner_command(
                    "verify-aliked-artifact",
                    matcher_type,
                    *threshold_args,
                    python_executable="python",
                ),
            },
        ],
        "copy_back_to_mac": [
            rel(official_output),
            rel(aliked_output),
            rel(aliked_output.with_suffix(".json")),
        ],
        "mac_after_copy_commands": [
            {
                "name": "validate_official_fixture_shape_and_keys",
                "command": runner_command("verify-official-fixture", matcher_type),
            },
            {
                "name": "run_official_fixture_parity_tests",
                "command": command_to_string(official_tests_cmd, official_tests_env),
            },
            {
                "name": "validate_aliked_artifact_quality",
                "command": runner_command("verify-aliked-artifact", matcher_type, *threshold_args),
            },
            {
                "name": "write_matched_training_config",
                "command": runner_command("write-train-config", matcher_type, "--overwrite-config"),
                "writes": [rel(train_config_output)],
            },
            {
                "name": "train_matched_metal_row",
                "command": command_to_string_with_python(
                    aliked_train_cmd,
                    aliked_train_env,
                    python_executable=LOCAL_PYTHON,
                ),
                "writes": [rel(train_output_dir)],
            },
            {
                "name": "validate_matched_training_row",
                "command": runner_command("verify-aliked-run", matcher_type, *threshold_args, *heldout_args),
            },
            {
                "name": "final_completion_audit",
                "command": command_to_string(
                    [
                        LOCAL_PYTHON,
                        "research_experiments/dynamic_foam/verify_powerfoam_completion_audit.py",
                        "--run-local-tests",
                    ],
                    {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": "src/train"},
                ),
            },
        ],
        "acceptance_thresholds": {
            "aliked_min_point_count": min_point_count,
            "aliked_min_track_mean": min_track_mean,
            "aliked_min_track_p90": min_track_p90,
            "aliked_max_reproj_median": max_reproj_median,
            "aliked_max_reproj_p90": max_reproj_p90,
            "aliked_min_verified_pairs": min_verified_pairs,
            "heldout_min_psnr": min_heldout_psnr,
            "heldout_min_ssim": min_heldout_ssim,
        },
    }


def _append_unique(values: list[str], extra: list[str]) -> list[str]:
    seen = set(values)
    for value in extra:
        if value not in seen:
            values.append(value)
            seen.add(value)
    return values


def load_aliked_artifact_summary(point_cloud: Path, matcher_type: str) -> dict[str, object]:
    summary_path = point_cloud.with_suffix(".json")
    if not point_cloud.exists():
        raise FileNotFoundError(point_cloud)
    if not summary_path.exists():
        raise FileNotFoundError(f"{summary_path} is required beside {point_cloud}.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise TypeError(f"{summary_path} must contain a JSON object.")
    point_count = int(summary.get("point_count", 0))
    if point_count <= 0:
        raise ValueError(f"{summary_path} has point_count={point_count}; refusing to prepare a training run.")
    actual_matcher = str(summary.get("matcher_type", "")).lower()
    if actual_matcher != matcher_type:
        raise ValueError(f"{summary_path} matcher_type={actual_matcher!r}, expected {matcher_type!r}.")
    output_path = Path(str(summary.get("output", "")))
    if output_path != Path(rel(point_cloud)) and output_path != point_cloud:
        raise ValueError(f"{summary_path} output={output_path!s}, expected {rel(point_cloud)!r}.")
    return summary


def validate_aliked_artifact(
    point_cloud: Path,
    matcher_type: str,
    *,
    min_point_count: int,
    min_track_mean: float,
    min_track_p90: float,
    max_reproj_median: float,
    max_reproj_p90: float,
    min_verified_pairs: int,
) -> None:
    summary_path = point_cloud.with_suffix(".json")
    if not point_cloud.exists() or not summary_path.exists():
        require_checks(
            "aliked_artifact",
            [
                {"name": "point_cloud_exists", "passed": point_cloud.exists(), "evidence": rel(point_cloud)},
                {"name": "summary_exists", "passed": summary_path.exists(), "evidence": rel(summary_path)},
            ],
        )
    summary = load_aliked_artifact_summary(point_cloud, matcher_type)
    track = summary.get("filtered_track_length", {})
    reproj = summary.get("filtered_reproj_error", {})
    checks = [
        {"name": "point_cloud_exists", "passed": point_cloud.exists(), "evidence": rel(point_cloud)},
        {
            "name": "summary_exists",
            "passed": point_cloud.with_suffix(".json").exists(),
            "evidence": rel(point_cloud.with_suffix(".json")),
        },
        {
            "name": "feature_type_aliked",
            "passed": str(summary.get("feature_type", "")).lower() == "aliked_n16rot",
            "evidence": summary.get("feature_type"),
        },
        {"name": "onnx_opt_in_recorded", "passed": bool(summary.get("allow_onnx_models")), "evidence": summary.get("allow_onnx_models")},
        {
            "name": "matcher_type_matches",
            "passed": str(summary.get("matcher_type", "")).lower() == matcher_type,
            "evidence": summary.get("matcher_type"),
        },
        {
            "name": "minimum_point_count",
            "passed": int(summary.get("point_count", 0)) >= int(min_point_count),
            "evidence": {"actual": int(summary.get("point_count", 0)), "required": int(min_point_count)},
        },
        {
            "name": "minimum_unique_camera_support",
            "passed": float(summary.get("filtered_unique_camera_track_length", {}).get("p90", 0.0)) >= 2.0,
            "evidence": summary.get("filtered_unique_camera_track_length"),
        },
        {
            "name": "true_multiframe_database",
            "passed": bool(summary.get("multi_frame_database")) and len(summary.get("frame_indices", [])) >= 4,
            "evidence": {"multi_frame_database": summary.get("multi_frame_database"), "frame_indices": summary.get("frame_indices")},
        },
        {
            "name": "track_length_threshold",
            "passed": float(track.get("mean", 0.0)) >= float(min_track_mean)
            and float(track.get("p90", 0.0)) >= float(min_track_p90),
            "evidence": {
                "mean": track.get("mean"),
                "p90": track.get("p90"),
                "required_mean": float(min_track_mean),
                "required_p90": float(min_track_p90),
            },
        },
        {
            "name": "reprojection_quality",
            "passed": float(reproj.get("median", float("inf"))) <= float(max_reproj_median)
            and float(reproj.get("p90", float("inf"))) <= float(max_reproj_p90),
            "evidence": {
                "median": reproj.get("median"),
                "p90": reproj.get("p90"),
                "required_median_max": float(max_reproj_median),
                "required_p90_max": float(max_reproj_p90),
            },
        },
        {
            "name": "verified_pair_threshold",
            "passed": int(summary.get("database_num_verified_image_pairs", 0)) >= int(min_verified_pairs),
            "evidence": {
                "actual": summary.get("database_num_verified_image_pairs"),
                "required": int(min_verified_pairs),
            },
        },
    ]
    require_checks("aliked_artifact", checks)


def validate_aliked_training_run(
    *,
    config_path: Path,
    point_cloud: Path,
    train_output_dir: Path,
    matcher_type: str,
    min_heldout_psnr: float,
    min_heldout_ssim: float,
) -> None:
    pre_checks = [
        {"name": "config_exists", "passed": config_path.exists(), "evidence": rel(config_path)},
        {
            "name": "artifact_summary_exists",
            "passed": point_cloud.exists() and point_cloud.with_suffix(".json").exists(),
            "evidence": {"ply": rel(point_cloud), "json": rel(point_cloud.with_suffix(".json"))},
        },
        {"name": "best_metrics_exists", "passed": (train_output_dir / "best_metrics.json").exists(), "evidence": rel(train_output_dir / "best_metrics.json")},
        {"name": "resolved_config_exists", "passed": (train_output_dir / "resolved_config.json").exists(), "evidence": rel(train_output_dir / "resolved_config.json")},
    ]
    if not all(bool(check["passed"]) for check in pre_checks):
        require_checks("aliked_training_run", pre_checks)
    _summary = load_aliked_artifact_summary(point_cloud, matcher_type)
    cfg = load_config_file(config_path)
    best_path = train_output_dir / "best_metrics.json"
    resolved_path = train_output_dir / "resolved_config.json"
    best = load_json_object(best_path) if best_path.exists() else {}
    metrics = best.get("metrics", {}) if isinstance(best, dict) else {}
    checks = [
        *pre_checks,
        {
            "name": "config_init_path_matches_artifact",
            "passed": cfg["model"]["init_point_cloud_path"] == rel(point_cloud),
            "evidence": {"config": cfg["model"]["init_point_cloud_path"], "artifact": rel(point_cloud)},
        },
        {
            "name": "config_output_dir_matches",
            "passed": cfg["logging"]["output_dir"] == rel(train_output_dir),
            "evidence": {"config": cfg["logging"]["output_dir"], "expected": rel(train_output_dir)},
        },
        {
            "name": "wandb_enabled",
            "passed": bool(cfg["logging"]["wandb_enabled"]),
            "evidence": cfg["logging"]["wandb_enabled"],
        },
        {
            "name": "wandb_mode_set",
            "passed": bool(cfg["logging"]["wandb_mode"]),
            "evidence": cfg["logging"]["wandb_mode"],
        },
        {
            "name": "offline_wandb_dir_exists_if_requested",
            "passed": str(cfg["logging"]["wandb_mode"]) != "offline" or any((ROOT / "wandb").glob("offline-run-*")),
            "evidence": str(ROOT / "wandb/offline-run-*"),
        },
        {
            "name": "selected_post_initial_step",
            "passed": int(best.get("step", 0)) > 0,
            "evidence": best.get("step"),
        },
        {
            "name": "heldout_psnr_threshold",
            "passed": float(metrics.get("heldout_eval_psnr", 0.0)) >= float(min_heldout_psnr),
            "evidence": {"actual": metrics.get("heldout_eval_psnr"), "required": float(min_heldout_psnr)},
        },
        {
            "name": "heldout_ssim_threshold",
            "passed": float(metrics.get("heldout_eval_ssim", 0.0)) >= float(min_heldout_ssim),
            "evidence": {"actual": metrics.get("heldout_eval_ssim"), "required": float(min_heldout_ssim)},
        },
    ]
    require_checks("aliked_training_run", checks)


def write_aliked_train_config(
    *,
    template: Path,
    config_output: Path,
    point_cloud: Path,
    train_output_dir: Path,
    matcher_type: str,
    wandb_enabled: bool,
    wandb_mode: str | None,
    overwrite: bool,
    dry_run: bool,
) -> None:
    if not template.exists():
        raise FileNotFoundError(template)
    artifact_summary = None if dry_run else load_aliked_artifact_summary(point_cloud, matcher_type)
    if not dry_run and config_output.exists() and not overwrite:
        raise FileExistsError(f"{config_output} already exists; pass --overwrite-config to replace it.")

    cfg = load_config_file(template)
    cfg["model"]["init_point_cloud_path"] = rel(point_cloud)
    cfg["logging"]["output_dir"] = rel(train_output_dir)
    cfg["logging"]["wandb_enabled"] = bool(wandb_enabled)
    cfg["logging"]["wandb_mode"] = wandb_mode
    cfg["logging"]["wandb_run_name"] = default_train_slug(matcher_type).replace("_", "-")
    cfg["logging"]["wandb_tags"] = _append_unique(
        [str(tag) for tag in cfg["logging"].get("wandb_tags", [])],
        ["aliked", "aliked-n16rot", matcher_type, "onnx", "wandb-paper-candidate"],
    )

    print(
        json.dumps(
            {
                "config_output": rel(config_output),
                "init_point_cloud_path": cfg["model"]["init_point_cloud_path"],
                "train_output_dir": cfg["logging"]["output_dir"],
                "wandb_enabled": cfg["logging"]["wandb_enabled"],
                "wandb_mode": cfg["logging"]["wandb_mode"],
                "wandb_run_name": cfg["logging"]["wandb_run_name"],
                "artifact_point_count": None if artifact_summary is None else int(artifact_summary["point_count"]),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    if dry_run:
        return
    config_output.parent.mkdir(parents=True, exist_ok=True)
    config_output.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run or print the external-host PowerFoam blockers: official CUDA/Warp fixture "
            "and ONNX-backed ALIKED/LightGlue clean geometry."
        )
    )
    parser.add_argument(
        "task",
        choices=[
            "check",
            "official-fixture",
            "official-tests",
            "aliked-cloud",
            "write-train-config",
            "train-aliked",
            "verify-official-fixture",
            "verify-aliked-artifact",
            "verify-aliked-run",
            "handoff",
            "all",
        ],
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--upstream-root", type=Path, default=Path("/tmp/powerfoam_official"))
    parser.add_argument("--official-output", type=Path, default=OFFICIAL_FIXTURE_OUTPUT)
    parser.add_argument("--aliked-output", type=Path, default=None)
    parser.add_argument("--train-template", type=Path, default=OPENCV_FISHEYE_CONFIG)
    parser.add_argument("--train-config-output", type=Path, default=None)
    parser.add_argument("--train-output-dir", type=Path, default=None)
    parser.add_argument("--overwrite-config", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-mode", default="offline")
    parser.add_argument("--min-point-count", type=int, default=2000)
    parser.add_argument("--min-track-mean", type=float, default=2.5)
    parser.add_argument("--min-track-p90", type=float, default=3.0)
    parser.add_argument("--max-reproj-median", type=float, default=4.0)
    parser.add_argument("--max-reproj-p90", type=float, default=8.0)
    parser.add_argument("--min-verified-pairs", type=int, default=28)
    parser.add_argument("--min-heldout-psnr", type=float, default=13.0)
    parser.add_argument("--min-heldout-ssim", type=float, default=0.15)
    parser.add_argument("--handoff-output", type=Path, default=None)
    parser.add_argument(
        "--matcher-type",
        choices=["aliked_bruteforce", "aliked_lightglue"],
        default="aliked_lightglue",
    )
    parser.add_argument("--max-features", type=int, default=12000)
    args = parser.parse_args()
    aliked_output = args.aliked_output or default_aliked_output(str(args.matcher_type))
    train_config_output = args.train_config_output or default_train_config_output(str(args.matcher_type))
    train_output_dir = args.train_output_dir or default_train_output_dir(str(args.matcher_type))

    if args.task == "handoff":
        manifest = build_handoff_manifest(
            upstream_root=args.upstream_root,
            official_output=args.official_output,
            aliked_output=aliked_output,
            train_config_output=train_config_output,
            train_output_dir=train_output_dir,
            matcher_type=str(args.matcher_type),
            max_features=int(args.max_features),
            min_point_count=int(args.min_point_count),
            min_track_mean=float(args.min_track_mean),
            min_track_p90=float(args.min_track_p90),
            max_reproj_median=float(args.max_reproj_median),
            max_reproj_p90=float(args.max_reproj_p90),
            min_verified_pairs=int(args.min_verified_pairs),
            min_heldout_psnr=float(args.min_heldout_psnr),
            min_heldout_ssim=float(args.min_heldout_ssim),
        )
        payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        print(payload, end="", flush=True)
        if args.handoff_output is not None:
            args.handoff_output.parent.mkdir(parents=True, exist_ok=True)
            args.handoff_output.write_text(payload, encoding="utf-8")
        return
    if args.task == "check":
        raise SystemExit(check_prereqs())
    if args.task in {"official-fixture", "all"}:
        command, env = official_fixture_command(args.upstream_root, args.official_output)
        run_command(command, env=env, dry_run=bool(args.dry_run))
        if not args.dry_run:
            validate_official_fixture(args.official_output)
    if args.task in {"official-tests", "all"}:
        require_canonical_official_fixture_for_tests(args.official_output)
        command, env = official_tests_command()
        run_command(command, env=env, dry_run=bool(args.dry_run))
    if args.task in {"aliked-cloud", "all"}:
        command, env = aliked_cloud_command(aliked_output, args.matcher_type, int(args.max_features))
        run_command(command, env=env, dry_run=bool(args.dry_run))
        if not args.dry_run:
            validate_aliked_artifact(
                aliked_output,
                args.matcher_type,
                min_point_count=int(args.min_point_count),
                min_track_mean=float(args.min_track_mean),
                min_track_p90=float(args.min_track_p90),
                max_reproj_median=float(args.max_reproj_median),
                max_reproj_p90=float(args.max_reproj_p90),
                min_verified_pairs=int(args.min_verified_pairs),
            )
    if args.task in {"write-train-config", "train-aliked", "all"}:
        write_aliked_train_config(
            template=args.train_template,
            config_output=train_config_output,
            point_cloud=aliked_output,
            train_output_dir=train_output_dir,
            matcher_type=str(args.matcher_type),
            wandb_enabled=not bool(args.no_wandb),
            wandb_mode=None if args.wandb_mode is None else str(args.wandb_mode),
            overwrite=bool(args.overwrite_config),
            dry_run=bool(args.dry_run),
        )
    if args.task in {"train-aliked", "all"}:
        command, env = aliked_training_command(train_config_output)
        run_command(command, env=env, dry_run=bool(args.dry_run))
        if not args.dry_run:
            validate_aliked_training_run(
                config_path=train_config_output,
                point_cloud=aliked_output,
                train_output_dir=train_output_dir,
                matcher_type=str(args.matcher_type),
                min_heldout_psnr=float(args.min_heldout_psnr),
                min_heldout_ssim=float(args.min_heldout_ssim),
            )
    if args.task == "verify-official-fixture":
        validate_official_fixture(args.official_output)
    if args.task == "verify-aliked-artifact":
        validate_aliked_artifact(
            aliked_output,
            args.matcher_type,
            min_point_count=int(args.min_point_count),
            min_track_mean=float(args.min_track_mean),
            min_track_p90=float(args.min_track_p90),
            max_reproj_median=float(args.max_reproj_median),
            max_reproj_p90=float(args.max_reproj_p90),
            min_verified_pairs=int(args.min_verified_pairs),
        )
    if args.task == "verify-aliked-run":
        validate_aliked_training_run(
            config_path=train_config_output,
            point_cloud=aliked_output,
            train_output_dir=train_output_dir,
            matcher_type=str(args.matcher_type),
            min_heldout_psnr=float(args.min_heldout_psnr),
            min_heldout_ssim=float(args.min_heldout_ssim),
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"{type(exc).__name__}: {exc}") from None
