from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC_TRAIN = ROOT / "src/train"
POWERFOAM_METAL = ROOT / "third_party/powerfoam-metal"
DYNAMIC_FOAM = Path(__file__).resolve().parent
if str(SRC_TRAIN) not in sys.path:
    sys.path.insert(0, str(SRC_TRAIN))
if str(POWERFOAM_METAL) not in sys.path:
    sys.path.insert(0, str(POWERFOAM_METAL))
if str(DYNAMIC_FOAM) not in sys.path:
    sys.path.insert(0, str(DYNAMIC_FOAM))

from verify_powerfoam_4k_benchmarks import (  # noqa: E402
    REGULAR_BENCHMARK,
    SELECTED_BENCHMARK,
    verify_benchmarks,
)
from config_utils import load_config_file  # noqa: E402
from verify_powerfoam_4k_trainability import (  # noqa: E402
    DEFAULT_ARTIFACT as FOUR_K_TRAINABILITY_ARTIFACT,
    verify_artifact as verify_four_k_trainability_artifact,
)
from verify_dynamic_powerfoam_geometry_run import (  # noqa: E402
    check_summary as check_dynamic_metal_geometry_summary,
    load_summary as load_dynamic_metal_geometry_summary,
)
from verify_powerfoam_cuda_smoke_results import (  # noqa: E402
    check_summary as check_cuda_smoke_summary,
    load_summary as load_cuda_smoke_summary,
)
from verify_powerfoam_paper_acceptance import OFFICIAL_FIXTURE, audit as paper_acceptance_audit  # noqa: E402


SAME_SPLIT_COMPARISON = ROOT / "outputs/comparisons/powerfoam_vs_splats_nearest0040_20260506.json"
METAL_DYNAMIC_GEOMETRY_SUMMARY = (
    ROOT
    / "outputs/dynamic_powerfoam_metal/"
    "local_mac_dynamic_powerfoam_metal_rbf_geometry_only_video_1024_16f_40step_smoke/"
    "dynamic_geometry_summary.json"
)
CUDA_DYNAMIC_GEOMETRY_SUMMARY = (
    ROOT / "outputs/powerfoam_cuda_smokes/cuda_dynamic_geometry_micro_20260506/summary.json"
)
NEAREST0040_TRAIN_CAMERAS = [
    "camera_0025",
    "camera_0039",
    "camera_0041",
    "camera_0012",
    "camera_0026",
    "camera_0023",
    "camera_0042",
    "camera_0038",
]
NEAREST0040_SAMPLE_ID = "deepview_03_Dog_camera_0001_to_camera_0040"


LOCAL_METAL_TEST_COMMAND = [
    "uv",
    "run",
    "--with",
    "pytest",
    "python",
    "-m",
    "pytest",
    "-p",
    "no:cacheprovider",
    "tests/test_multicam_video_data.py",
    "tests/test_powerfoam_direct.py",
    "-q",
]
PAPER_ACCEPTANCE_TEST_COMMAND = [
    "uv",
    "run",
    "--with",
    "pytest",
    "python",
    "-m",
    "pytest",
    "-p",
    "no:cacheprovider",
    "tests/test_powerfoam_paper_acceptance.py",
    "-q",
]
DYNAMIC_METAL_TEST_COMMAND = [
    "uv",
    "run",
    "--with",
    "pytest",
    "python",
    "-m",
    "pytest",
    "-p",
    "no:cacheprovider",
    "tests/test_dynamic_powerfoam_metal.py",
    "-q",
    "-rs",
]
CUDA_SMOKE_TEST_COMMAND = [
    "uv",
    "run",
    "--with",
    "pytest",
    "python",
    "-m",
    "pytest",
    "-p",
    "no:cacheprovider",
    "tests/test_powerfoam_cuda_smoke.py",
    "-q",
]
LOCAL_METAL_FIXTURE_TEST_NODE = (
    "tests/test_powerfoam_direct.py::test_powerfoam_metal_matches_official_camera_local_fixture_shared_backward"
)
OFFICIAL_DIRECT_TEST_NODE = (
    "tests/test_powerfoam_direct.py::test_powerfoam_direct_matches_official_cuda_fixture_if_present"
)
OFFICIAL_METAL_TEST_NODE = (
    "tests/test_powerfoam_direct.py::test_powerfoam_metal_matches_official_cuda_fixture_shared_backward_if_present"
)


def official_test_command(node_id: str) -> list[str]:
    return [
        "uv",
        "run",
        "--with",
        "pytest",
        "python",
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        node_id,
        "-q",
        "-rs",
    ]


OFFICIAL_PARITY_TEST_COMMAND = [
    "uv",
    "run",
    "--with",
    "pytest",
    "python",
    "-m",
    "pytest",
    "-p",
    "no:cacheprovider",
    OFFICIAL_DIRECT_TEST_NODE,
    OFFICIAL_METAL_TEST_NODE,
    "-q",
    "-rs",
]
RAYTRACE_PARITY_COMMAND = [
    "uv",
    "run",
    "--project",
    str(ROOT),
    "--with",
    "scipy",
    "python",
    "third_party/powerfoam-metal/tests/raytrace_check.py",
]


def check(condition: bool, name: str, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(condition), "evidence": evidence}


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def json_path_status(path: Path) -> dict[str, Any]:
    return {
        "path": display_path(path),
        "exists": path.exists(),
    }


def read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return payload


def repo_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_row_config(row: dict[str, Any]) -> dict[str, Any]:
    config_path = repo_path(row.get("config"))
    if config_path is None:
        return {}
    return load_config_file(config_path)


def numeric_row_metrics(row: dict[str, Any], names: tuple[str, ...]) -> bool:
    for name in names:
        try:
            float(row[name])
        except Exception:
            return False
    return True


def verify_same_split_comparison(path: Path = SAME_SPLIT_COMPARISON) -> dict[str, Any]:
    checks: list[dict[str, Any]] = [check(path.exists(), "comparison_artifact_exists", json_path_status(path))]
    if not path.exists():
        return {"ok": False, "checks": checks}
    try:
        payload = read_json_object(path)
    except Exception as exc:
        checks.append(check(False, "comparison_artifact_json_object", f"{type(exc).__name__}: {exc}"))
        return {"ok": False, "checks": checks}
    rows = payload.get("rows", [])
    by_label = {str(row.get("label")): row for row in rows if isinstance(row, dict)}
    raw_powerfoam = by_label.get("powerfoam_raw_nearest0040", {})
    calibrated_powerfoam = by_label.get("powerfoam_eval_rgb_calibrated_nearest0040", {})
    splat = by_label.get("matched_free_dynamic_3dgs", {})
    configs: dict[str, dict[str, Any]] = {}
    config_errors: dict[str, str] = {}
    for label, row in (
        ("powerfoam_raw_nearest0040", raw_powerfoam),
        ("powerfoam_eval_rgb_calibrated_nearest0040", calibrated_powerfoam),
        ("matched_free_dynamic_3dgs", splat),
    ):
        try:
            configs[label] = load_row_config(row) if row else {}
        except Exception as exc:
            configs[label] = {}
            config_errors[label] = f"{type(exc).__name__}: {exc}"
    raw_cfg = configs["powerfoam_raw_nearest0040"]
    calibrated_cfg = configs["powerfoam_eval_rgb_calibrated_nearest0040"]
    splat_cfg = configs["matched_free_dynamic_3dgs"]
    required_metric_names = ("heldout_eval_psnr", "heldout_eval_ssim", "heldout_eval_l1")
    eval_metric_names = ("eval_psnr", "eval_ssim", "eval_l1")
    caveat_text = "\n".join(str(item) for item in payload.get("caveats", []))
    row_artifact_status = {
        label: {
            "config": json_path_status(repo_path(row.get("config")) or Path("")),
            "output_dir": json_path_status(repo_path(row.get("output_dir")) or Path("")),
        }
        for label, row in (
            ("powerfoam_raw_nearest0040", raw_powerfoam),
            ("powerfoam_eval_rgb_calibrated_nearest0040", calibrated_powerfoam),
            ("matched_free_dynamic_3dgs", splat),
        )
        if row
    }
    split_fields = {
        label: {
            "sample_id": cfg.get("data", {}).get("multicam_sample_id"),
            "train_cameras": cfg.get("data", {}).get("multicam_train_cameras"),
            "heldout_camera": cfg.get("data", {}).get("multicam_heldout_camera"),
            "anchor_camera": cfg.get("data", {}).get("multicam_anchor_camera"),
            "max_frames": cfg.get("data", {}).get("max_frames"),
        }
        for label, cfg in configs.items()
    }
    checks.extend(
        [
            check(
                all(label in by_label for label in (
                    "powerfoam_raw_nearest0040",
                    "powerfoam_eval_rgb_calibrated_nearest0040",
                    "matched_free_dynamic_3dgs",
                )),
                "comparison_required_rows_present",
                sorted(by_label),
            ),
            check(
                raw_powerfoam.get("metric_semantics") == "raw"
                and splat.get("metric_semantics") == "raw"
                and calibrated_powerfoam.get("metric_semantics") == "calibrated_with_raw_disclosed"
                and numeric_row_metrics(calibrated_powerfoam, (
                    "raw_heldout_eval_psnr",
                    "raw_heldout_eval_ssim",
                    "raw_heldout_eval_l1",
                )),
                "raw_and_calibrated_metric_semantics_disclosed",
                {
                    "powerfoam_raw": raw_powerfoam.get("metric_semantics"),
                    "powerfoam_calibrated": calibrated_powerfoam.get("metric_semantics"),
                    "splat": splat.get("metric_semantics"),
                    "calibrated_raw_fields": {
                        key: calibrated_powerfoam.get(key)
                        for key in ("raw_heldout_eval_psnr", "raw_heldout_eval_ssim", "raw_heldout_eval_l1")
                    },
                },
            ),
            check(
                numeric_row_metrics(raw_powerfoam, required_metric_names)
                and numeric_row_metrics(splat, required_metric_names)
                and numeric_row_metrics(calibrated_powerfoam, required_metric_names),
                "comparison_measured_metrics_present",
                {
                    "powerfoam_raw": {key: raw_powerfoam.get(key) for key in required_metric_names},
                    "powerfoam_calibrated": {key: calibrated_powerfoam.get(key) for key in required_metric_names},
                    "splat": {key: splat.get(key) for key in required_metric_names},
                },
            ),
            check(
                numeric_row_metrics(raw_powerfoam, eval_metric_names)
                and numeric_row_metrics(splat, eval_metric_names)
                and numeric_row_metrics(calibrated_powerfoam, eval_metric_names),
                "comparison_eval_metrics_present",
                {
                    "powerfoam_raw": {key: raw_powerfoam.get(key) for key in eval_metric_names},
                    "powerfoam_calibrated": {key: calibrated_powerfoam.get(key) for key in eval_metric_names},
                    "splat": {key: splat.get(key) for key in eval_metric_names},
                },
            ),
            check(
                bool(row_artifact_status)
                and all(
                    item["config"]["exists"] and item["output_dir"]["exists"]
                    for item in row_artifact_status.values()
                )
                and not config_errors,
                "comparison_row_artifacts_exist",
                {"artifacts": row_artifact_status, "config_errors": config_errors},
            ),
            check(
                raw_powerfoam.get("primitive_count") == splat.get("primitive_count")
                and raw_powerfoam.get("train_frames") == splat.get("train_frames")
                and raw_powerfoam.get("render_size") == splat.get("render_size")
                and raw_powerfoam.get("steps") == splat.get("steps"),
                "raw_powerfoam_and_splat_same_core_settings",
                {
                    "powerfoam_raw": {
                        key: raw_powerfoam.get(key)
                        for key in ("primitive_count", "train_frames", "render_size", "steps")
                    },
                    "splat": {
                        key: splat.get(key)
                        for key in ("primitive_count", "train_frames", "render_size", "steps")
                    },
                },
            ),
            check(
                all(
                    fields.get("sample_id") == NEAREST0040_SAMPLE_ID
                    and fields.get("train_cameras") == NEAREST0040_TRAIN_CAMERAS
                    and fields.get("heldout_camera") == "camera_0040"
                    and fields.get("anchor_camera") == "camera_0025"
                    and fields.get("max_frames") == 16
                    for fields in (
                        split_fields["powerfoam_raw_nearest0040"],
                        split_fields["powerfoam_eval_rgb_calibrated_nearest0040"],
                        split_fields["matched_free_dynamic_3dgs"],
                    )
                ),
                "comparison_exact_nearest0040_split",
                split_fields,
            ),
            check(
                raw_cfg.get("train", {}).get("seed") == 23
                and splat_cfg.get("train", {}).get("seed") == 23
                and raw_cfg.get("train", {}).get("frames_per_step") == 4
                and splat_cfg.get("train", {}).get("frames_per_step") == 4
                and raw_cfg.get("train", {}).get("steps") == 40
                and splat_cfg.get("train", {}).get("steps") == 40
                and splat_cfg.get("train", {}).get("train_frame_count") == 16,
                "comparison_seed_and_sampling_match",
                {
                    "powerfoam_raw": {
                        key: raw_cfg.get("train", {}).get(key)
                        for key in ("seed", "frames_per_step", "steps")
                    },
                    "splat": {
                        key: splat_cfg.get("train", {}).get(key)
                        for key in ("seed", "frames_per_step", "steps", "train_frame_count")
                    },
                    "calibrated_powerfoam_steps": calibrated_cfg.get("train", {}).get("steps"),
                },
            ),
            check(
                raw_cfg.get("logging", {}).get("output_dir") == raw_powerfoam.get("output_dir")
                and calibrated_cfg.get("logging", {}).get("output_dir") == calibrated_powerfoam.get("output_dir")
                and splat_cfg.get("logging", {}).get("output_dir") == splat.get("output_dir"),
                "comparison_config_paths_match_rows",
                {
                    "powerfoam_raw": {
                        "cfg_output_dir": raw_cfg.get("logging", {}).get("output_dir"),
                        "row_output_dir": raw_powerfoam.get("output_dir"),
                    },
                    "powerfoam_calibrated": {
                        "cfg_output_dir": calibrated_cfg.get("logging", {}).get("output_dir"),
                        "row_output_dir": calibrated_powerfoam.get("output_dir"),
                    },
                    "splat": {
                        "cfg_output_dir": splat_cfg.get("logging", {}).get("output_dir"),
                        "row_output_dir": splat.get("output_dir"),
                    },
                },
            ),
            check(
                raw_powerfoam.get("primitive_count") is not None
                and calibrated_powerfoam.get("primitive_count") is not None
                and splat.get("primitive_count") is not None,
                "comparison_primitive_or_parameter_count_present",
                {
                    "powerfoam_raw": raw_powerfoam.get("primitive_count"),
                    "powerfoam_calibrated": calibrated_powerfoam.get("primitive_count"),
                    "splat": splat.get("primitive_count"),
                },
            ),
            check(
                float(splat.get("train_loop_elapsed_s", 0.0)) > 0.0,
                "comparison_wall_clock_recorded",
                {"splat_train_loop_elapsed_s": splat.get("train_loop_elapsed_s")},
            ),
            check(
                "Dynamic gsplat nearest0040 matched PowerFoam split" in (ROOT / "BASELINES.md").read_text(
                    encoding="utf-8"
                )
                and "powerfoam_vs_splats_nearest0040_20260506.json" in (ROOT / "BASELINES.md").read_text(
                    encoding="utf-8"
                ),
                "baselines_nearest0040_rows_recorded",
                {"path": "BASELINES.md"},
            ),
            check(
                "OPENCV_FISHEYE" in caveat_text and "pinhole" in caveat_text,
                "projection_caveat_disclosed",
                payload.get("caveats", []),
            ),
            check(
                numeric_row_metrics(raw_powerfoam, required_metric_names)
                and numeric_row_metrics(splat, required_metric_names),
                "raw_powerfoam_vs_splat_delta_recorded",
                {
                    "delta_raw_powerfoam_minus_splat": {
                        "heldout_eval_psnr": float(raw_powerfoam.get("heldout_eval_psnr", 0.0))
                        - float(splat.get("heldout_eval_psnr", 0.0)),
                        "heldout_eval_ssim": float(raw_powerfoam.get("heldout_eval_ssim", 0.0))
                        - float(splat.get("heldout_eval_ssim", 0.0)),
                        "heldout_eval_l1": float(raw_powerfoam.get("heldout_eval_l1", 0.0))
                        - float(splat.get("heldout_eval_l1", 0.0)),
                    }
                },
            ),
        ]
    )
    return {"ok": all(bool(item["passed"]) for item in checks), "checks": checks}


def verify_dynamic_metal_geometry_artifact(path: Path = METAL_DYNAMIC_GEOMETRY_SUMMARY) -> dict[str, Any]:
    checks: list[dict[str, Any]] = [check(path.exists(), "metal_dynamic_geometry_summary_exists", json_path_status(path))]
    if not path.exists():
        return {"ok": False, "checks": checks}
    try:
        summary = load_dynamic_metal_geometry_summary(path)
        checks.extend(
            check_dynamic_metal_geometry_summary(
                summary,
                require_geometry_motion=True,
                require_alpha_support_motion=True,
                require_appearance_freeze_control=True,
                min_screen_delta_px=1.0e-5,
                min_alpha_delta=1.0e-6,
                min_support_delta=0.0,
                max_feature_delta=1.0e-8,
            )
        )
        artifact_status = {
            name: json_path_status(repo_path(artifact_path) or Path(""))
            for name, artifact_path in summary.get("artifacts", {}).items()
        }
        config = summary.get("config", {})
        motion = summary.get("motion_vs_repaint", {})
        checks.extend(
            [
                check(
                    bool(artifact_status) and all(item["exists"] for item in artifact_status.values()),
                    "metal_dynamic_geometry_artifacts_exist",
                    artifact_status,
                ),
                check(
                    config.get("frames") == 16
                    and config.get("steps") == 40
                    and config.get("cells") == 1024
                    and config.get("render_size") == 64
                    and bool(config.get("dynamic_centers"))
                    and bool(config.get("dynamic_radii"))
                    and not bool(config.get("dynamic_features"))
                    and not bool(config.get("dynamic_densities")),
                    "metal_dynamic_geometry_smoke_scope",
                    config,
                ),
                check(
                    float(motion.get("state_mean_temporal_screen_delta_px", 0.0)) > 0.0
                    and float(motion.get("state_p95_temporal_screen_delta_px", 0.0)) > 0.0
                    and float(summary.get("final_eval", {}).get("state_mean_center_delta", 0.0)) > 0.0
                    and float(summary.get("final_eval", {}).get("state_mean_radius_delta", 0.0)) > 0.0,
                    "metal_dynamic_geometry_state_delta_present",
                    {"motion_vs_repaint": motion, "final_eval": summary.get("final_eval", {})},
                ),
            ]
        )
    except Exception as exc:
        checks.append(check(False, "metal_dynamic_geometry_verifier_exception", f"{type(exc).__name__}: {exc}"))
    return {"ok": all(bool(item["passed"]) for item in checks), "checks": checks}


def verify_cuda_dynamic_geometry_artifact(path: Path = CUDA_DYNAMIC_GEOMETRY_SUMMARY) -> dict[str, Any]:
    checks: list[dict[str, Any]] = [check(path.exists(), "cuda_dynamic_geometry_summary_exists", json_path_status(path))]
    if not path.exists():
        return {"ok": False, "checks": checks}
    try:
        summary = load_cuda_smoke_summary(path)
        checks.extend(
            check_cuda_smoke_summary(
                summary,
                allow_planned=False,
                require_official_fixture=False,
                require_dynamic_geometry=True,
            )
        )
        runs = summary.get("runs", [])
        by_name = {str(run.get("name")): run for run in runs if isinstance(run, dict)}
        settings = summary.get("settings", {})
        clip = summary.get("clip", {})
        host = summary.get("host", {})
        feature_dynamic = by_name.get("dynamic_feature_foam_cuda", {}).get("metrics", {}).get("dynamic", {})
        geometry_dynamic = by_name.get("dynamic_geometry_foam_cuda", {}).get("metrics", {}).get("dynamic", {})
        checks.extend(
            [
                check(
                    host.get("cuda_device_name") == "NVIDIA L40S"
                    and clip.get("frames") == 4
                    and clip.get("size") == 64
                    and settings.get("iterations") == 5
                    and settings.get("points") == 256
                    and settings.get("num_texel_sites") == 4
                    and settings.get("sv_dof") == 2
                    and settings.get("fixed_black_background") is True
                    and settings.get("dynamic_geometry") is True
                    and settings.get("skip_official_fixture") is True,
                    "cuda_dynamic_geometry_micro_contract",
                    {"host": host, "clip": clip, "settings": settings},
                ),
                check(
                    all(
                        by_name.get(name, {}).get("status") == "ok"
                        for name in (
                            "official_static_cuda",
                            "dynamic_feature_foam_cuda",
                            "dynamic_geometry_foam_cuda",
                        )
                    )
                    and bool(summary.get("comparisons", {}).get("available"))
                    and bool(summary.get("comparisons", {}).get("geometry_available")),
                    "cuda_dynamic_geometry_lanes_same_model_contract",
                    {
                        "runs": {name: by_name.get(name, {}).get("status") for name in by_name},
                        "comparisons": summary.get("comparisons"),
                        "settings": settings,
                    },
                ),
                check(
                    float(feature_dynamic.get("dynamic_time_rgb_delta_mean", 0.0)) > 0.0
                    and float(feature_dynamic.get("dynamic_time_alpha_delta_mean", 0.0)) == 0.0
                    and float(feature_dynamic.get("same_camera_support_delta_mean", 0.0)) == 0.0,
                    "cuda_feature_lane_rgb_only_negative_control",
                    feature_dynamic,
                ),
                check(
                    float(geometry_dynamic.get("dynamic_time_alpha_delta_mean", 0.0))
                    > float(feature_dynamic.get("dynamic_time_alpha_delta_mean", 0.0))
                    and float(geometry_dynamic.get("same_camera_support_delta_mean", 0.0))
                    > float(feature_dynamic.get("same_camera_support_delta_mean", 0.0)),
                    "cuda_geometry_alpha_support_exceeds_feature_control",
                    {"feature": feature_dynamic, "geometry": geometry_dynamic},
                ),
            ]
        )
    except Exception as exc:
        checks.append(check(False, "cuda_dynamic_geometry_verifier_exception", f"{type(exc).__name__}: {exc}"))
    return {"ok": all(bool(item["passed"]) for item in checks), "checks": checks}


def run_pytest_command(command: list[str], *, pythonpath: str = "src/train:third_party/powerfoam-metal") -> dict[str, Any]:
    env = {"PYTHONPATH": pythonpath, "PYTHONDONTWRITEBYTECODE": "1"}
    process_env = os.environ.copy()
    process_env.update(env)
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=process_env,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "command": " ".join(command),
        "returncode": int(result.returncode),
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
    }


def run_local_metal_tests() -> dict[str, Any]:
    return run_pytest_command(LOCAL_METAL_TEST_COMMAND)


def run_paper_acceptance_tests() -> dict[str, Any]:
    return run_pytest_command(PAPER_ACCEPTANCE_TEST_COMMAND)


def run_dynamic_metal_tests() -> dict[str, Any]:
    return run_pytest_command(DYNAMIC_METAL_TEST_COMMAND, pythonpath="src/train:third_party/dynamic-powerfoam-metal")


def run_cuda_smoke_tests() -> dict[str, Any]:
    return run_pytest_command(CUDA_SMOKE_TEST_COMMAND, pythonpath="src/train")


def run_official_node_test(node_id: str) -> dict[str, Any]:
    return run_pytest_command(official_test_command(node_id))


def output_has_no_skips(result: dict[str, Any]) -> bool:
    text = f"{result.get('stdout_tail', '')}\n{result.get('stderr_tail', '')}".lower()
    return "skipped" not in text and " skipped" not in text


def command_passed_without_skips(result: dict[str, Any] | None) -> bool:
    return result is not None and int(result["returncode"]) == 0 and output_has_no_skips(result)


def official_node_status(result: dict[str, Any] | None, *, node_id: str) -> Any:
    if result is not None:
        return result
    if not OFFICIAL_FIXTURE.exists():
        return {
            "status": "not_run_missing_fixture",
            "fixture": str(OFFICIAL_FIXTURE.relative_to(ROOT)),
            "node_id": node_id,
        }
    return {
        "status": "not_run",
        "command": "PYTHONPATH=src/train:third_party/powerfoam-metal "
        + " ".join(official_test_command(node_id)),
        "node_id": node_id,
    }


def official_node_passed(result: dict[str, Any] | None) -> bool:
    return command_passed_without_skips(result)


def local_node_status(result: dict[str, Any] | None, *, node_id: str) -> Any:
    if result is not None:
        return result
    return {
        "status": "not_run",
        "command": "PYTHONPATH=src/train:third_party/powerfoam-metal " + " ".join(official_test_command(node_id)),
        "node_id": node_id,
    }


def low_level_script_status(result: dict[str, Any] | None, *, command: list[str]) -> Any:
    if result is not None:
        return result
    return {
        "status": "not_run",
        "command": "PYTHONPATH=src/train:third_party/powerfoam-metal " + " ".join(command),
    }


def audit(
    *,
    run_local_tests: bool,
    allow_local_tests_unrun: bool,
    require_raw_quality: bool = False,
    same_split_comparison: Path = SAME_SPLIT_COMPARISON,
    metal_dynamic_geometry_summary: Path = METAL_DYNAMIC_GEOMETRY_SUMMARY,
    cuda_dynamic_geometry_summary: Path = CUDA_DYNAMIC_GEOMETRY_SUMMARY,
) -> dict[str, Any]:
    objective = (
        "PowerFoam proper on Metal: full/paper primitive on Metal, accurate forward/backward, "
        "official CUDA/Warp parity, fast 4K evidence, trainable paper-scale heldout quality, "
        "and P0 evidence for raw-quality gating, same-split splat comparison, Metal dynamic "
        "geometry, and CUDA dynamic geometry."
    )
    local_test_result = run_local_metal_tests() if run_local_tests else None
    paper_acceptance_test_result = run_paper_acceptance_tests() if run_local_tests else None
    dynamic_metal_test_result = run_dynamic_metal_tests() if run_local_tests else None
    cuda_smoke_test_result = run_cuda_smoke_tests() if run_local_tests else None
    local_metal_fixture_test_result = (
        run_official_node_test(LOCAL_METAL_FIXTURE_TEST_NODE) if run_local_tests else None
    )
    official_direct_test_result = (
        run_official_node_test(OFFICIAL_DIRECT_TEST_NODE) if run_local_tests and OFFICIAL_FIXTURE.exists() else None
    )
    official_metal_test_result = (
        run_official_node_test(OFFICIAL_METAL_TEST_NODE) if run_local_tests and OFFICIAL_FIXTURE.exists() else None
    )
    raytrace_parity_result = run_pytest_command(RAYTRACE_PARITY_COMMAND) if run_local_tests else None
    four_k_report: dict[str, Any]
    try:
        four_k_report = {
            "ok": True,
            "summary": verify_benchmarks(
                selected_path=SELECTED_BENCHMARK,
                regular_path=REGULAR_BENCHMARK,
                max_total_ms=1200.0,
                max_steps_cap=64,
            ),
        }
    except Exception as exc:  # pragma: no cover - verifier failures are reported as data.
        four_k_report = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        four_k_trainability_report = verify_four_k_trainability_artifact(FOUR_K_TRAINABILITY_ARTIFACT)
    except Exception as exc:  # pragma: no cover - verifier failures are reported as data.
        four_k_trainability_report = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    paper_report = paper_acceptance_audit(
        min_clean_heldout_psnr=13.0,
        min_clean_heldout_ssim=0.15,
        min_track_p90=3.0,
        min_track_mean=2.5,
        min_point_count=2000,
        require_raw_quality=require_raw_quality,
    )
    same_split_report = verify_same_split_comparison(same_split_comparison)
    dynamic_metal_report = verify_dynamic_metal_geometry_artifact(metal_dynamic_geometry_summary)
    cuda_dynamic_report = verify_cuda_dynamic_geometry_artifact(cuda_dynamic_geometry_summary)

    selected_clean = paper_report["selected_clean_deepview_candidate"]
    trained_clean = paper_report.get("selected_clean_post_initial_candidate")
    clean_point_cloud = paper_report["clean_point_cloud"]
    local_tests_passed = local_test_result is not None and int(local_test_result["returncode"]) == 0
    local_fixture_passed = command_passed_without_skips(local_metal_fixture_test_result) or (
        allow_local_tests_unrun and local_metal_fixture_test_result is None
    )
    raytrace_parity_passed = command_passed_without_skips(raytrace_parity_result) or (
        allow_local_tests_unrun and raytrace_parity_result is None
    )
    paper_acceptance_tests_passed = command_passed_without_skips(paper_acceptance_test_result) or (
        allow_local_tests_unrun and paper_acceptance_test_result is None
    )
    dynamic_metal_tests_passed = command_passed_without_skips(dynamic_metal_test_result) or (
        allow_local_tests_unrun and dynamic_metal_test_result is None
    )
    cuda_smoke_tests_passed = command_passed_without_skips(cuda_smoke_test_result) or (
        allow_local_tests_unrun and cuda_smoke_test_result is None
    )
    checks = [
        check(
            local_tests_passed or (allow_local_tests_unrun and local_test_result is None),
            "local_metal_forward_backward_regression_gate",
            {
                "status": "not_run" if local_test_result is None else local_test_result,
                "command": "PYTHONPATH=src/train:third_party/powerfoam-metal "
                + " ".join(LOCAL_METAL_TEST_COMMAND),
            },
        ),
        check(
            paper_acceptance_tests_passed,
            "p0_1_raw_quality_gate_pytest",
            {
                "status": "not_run" if paper_acceptance_test_result is None else paper_acceptance_test_result,
                "command": "PYTHONPATH=src/train:third_party/powerfoam-metal "
                + " ".join(PAPER_ACCEPTANCE_TEST_COMMAND),
            },
        ),
        check(
            dynamic_metal_tests_passed,
            "p0_3_dynamic_metal_geometry_pytest",
            {
                "status": "not_run" if dynamic_metal_test_result is None else dynamic_metal_test_result,
                "command": "PYTHONPATH=src/train:third_party/dynamic-powerfoam-metal "
                + " ".join(DYNAMIC_METAL_TEST_COMMAND),
            },
        ),
        check(
            cuda_smoke_tests_passed,
            "p0_4_cuda_dynamic_geometry_pytest",
            {
                "status": "not_run" if cuda_smoke_test_result is None else cuda_smoke_test_result,
                "command": "PYTHONPATH=src/train " + " ".join(CUDA_SMOKE_TEST_COMMAND),
            },
        ),
        check(
            local_fixture_passed,
            "local_metal_fixture_shared_backward_test_ran_passed",
            local_node_status(local_metal_fixture_test_result, node_id=LOCAL_METAL_FIXTURE_TEST_NODE),
        ),
        check(
            raytrace_parity_passed,
            "low_level_raytrace_forward_backward_parity_script",
            low_level_script_status(raytrace_parity_result, command=RAYTRACE_PARITY_COMMAND),
        ),
        check(
            OFFICIAL_FIXTURE.exists(),
            "official_cuda_warp_fixture_present",
            str(OFFICIAL_FIXTURE.relative_to(ROOT)),
        ),
        check(
            official_node_passed(official_direct_test_result),
            "official_direct_parity_test_ran_passed",
            official_node_status(official_direct_test_result, node_id=OFFICIAL_DIRECT_TEST_NODE),
        ),
        check(
            official_node_passed(official_metal_test_result),
            "official_metal_parity_test_ran_passed",
            official_node_status(official_metal_test_result, node_id=OFFICIAL_METAL_TEST_NODE),
        ),
        check(
            bool(four_k_report["ok"]),
            "saved_4k_height_sv_raytrace_benchmark",
            four_k_report,
        ),
        check(
            bool(four_k_trainability_report["ok"]),
            "saved_4k_optimizer_step_trainability_artifact",
            four_k_trainability_report,
        ),
        check(
            bool(paper_report["ok"]),
            "paper_acceptance_verifier",
            {
                "selected_clean_deepview_candidate": selected_clean,
                "selected_clean_post_initial_candidate": trained_clean,
                "clean_point_cloud": clean_point_cloud,
                "failed_checks": [
                    item for item in paper_report["checks"] if not bool(item["passed"])
                ],
                "missing_optional_clean_deepview_candidates": paper_report.get(
                    "missing_optional_clean_deepview_candidates",
                    [],
                ),
            },
        ),
        check(
            not require_raw_quality or bool(paper_report["raw_quality_ok"]),
            "paper_acceptance_raw_quality_status",
            {
                "require_raw_quality": bool(require_raw_quality),
                "raw_quality_ok": bool(paper_report["raw_quality_ok"]),
                "calibrated_quality_ok": bool(paper_report["calibrated_quality_ok"]),
                "raw_quality_checks": paper_report.get("raw_quality_checks", []),
                "selected_clean_deepview_candidate": selected_clean,
                "selected_clean_post_initial_raw_candidate": paper_report.get(
                    "selected_clean_post_initial_raw_candidate"
                ),
                "raw_quality_next_blockers": paper_report.get("raw_quality_next_blockers", []),
            },
        ),
        check(
            bool(selected_clean["wandb_enabled"]),
            "wandb_backed_selected_paper_row",
            {"wandb_enabled": bool(selected_clean["wandb_enabled"])},
        ),
        check(
            trained_clean is not None,
            "post_initial_paper_row_trained_and_quality_gated",
            {
                "best_step": int(selected_clean["step"]),
                "best_step_note": "best checkpoint may remain initialization if later trained rows do not improve PSNR",
                "post_initial_row": trained_clean,
                "required_step": "> 0",
                "required_psnr": 13.0,
                "required_ssim": 0.15,
                "required_state_delta": "> 1e-8",
            },
        ),
        check(
            float(selected_clean["heldout_eval_psnr"]) >= 13.0
            and float(selected_clean["heldout_eval_ssim"]) >= 0.15,
            "paper_scale_heldout_quality",
            {
                "heldout_eval_psnr": float(selected_clean["heldout_eval_psnr"]),
                "required_psnr": 13.0,
                "heldout_eval_ssim": float(selected_clean["heldout_eval_ssim"]),
                "required_ssim": 0.15,
            },
        ),
        check(
            bool(same_split_report["ok"]),
            "p0_2_same_split_powerfoam_vs_splat_comparison",
            {
                "path": display_path(same_split_comparison),
                "failed_checks": [item for item in same_split_report["checks"] if not bool(item["passed"])],
                "checks": same_split_report["checks"],
            },
        ),
        check(
            bool(dynamic_metal_report["ok"]),
            "p0_3_metal_dynamic_geometry_motion_not_repaint",
            {
                "path": display_path(metal_dynamic_geometry_summary),
                "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python "
                "research_experiments/dynamic_foam/verify_dynamic_powerfoam_geometry_run.py "
                f"{display_path(metal_dynamic_geometry_summary)} "
                "--require-geometry-motion --require-alpha-support-motion "
                "--require-appearance-freeze-control",
                "failed_checks": [item for item in dynamic_metal_report["checks"] if not bool(item["passed"])],
                "checks": dynamic_metal_report["checks"],
            },
        ),
        check(
            bool(cuda_dynamic_report["ok"]),
            "p0_4_cuda_dynamic_geometry_motion_not_rgb_only",
            {
                "path": display_path(cuda_dynamic_geometry_summary),
                "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python "
                "research_experiments/dynamic_foam/verify_powerfoam_cuda_smoke_results.py "
                f"{display_path(cuda_dynamic_geometry_summary)} --require-dynamic-geometry",
                "failed_checks": [item for item in cuda_dynamic_report["checks"] if not bool(item["passed"])],
                "checks": cuda_dynamic_report["checks"],
            },
        ),
    ]
    return {
        "ok": all(bool(item["passed"]) for item in checks),
        "objective": objective,
        "raw_quality_ok": bool(paper_report["raw_quality_ok"]),
        "calibrated_quality_ok": bool(paper_report["calibrated_quality_ok"]),
        "raw_quality_next_blockers": paper_report.get("raw_quality_next_blockers", []),
        "prompt_to_artifact_checklist": checks,
        "next_blockers": [
            item["name"] for item in checks if not bool(item["passed"])
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Top-level completion audit for the PowerFoam Metal objective.")
    parser.add_argument(
        "--run-local-tests",
        action="store_true",
        help="Run the focused PowerFoam Metal pytest gate instead of only reporting the command.",
    )
    parser.add_argument(
        "--allow-local-tests-unrun",
        action="store_true",
        help="Do not fail the audit merely because --run-local-tests was omitted.",
    )
    parser.add_argument(
        "--require-raw-quality",
        action="store_true",
        help="Fail if the selected clean paper row only passes after eval color calibration.",
    )
    parser.add_argument(
        "--same-split-comparison",
        type=Path,
        default=SAME_SPLIT_COMPARISON,
        help="Saved same-split PowerFoam-vs-splat comparison JSON.",
    )
    parser.add_argument(
        "--metal-dynamic-geometry-summary",
        type=Path,
        default=METAL_DYNAMIC_GEOMETRY_SUMMARY,
        help="Saved Metal dynamic-geometry run summary JSON or run directory.",
    )
    parser.add_argument(
        "--cuda-dynamic-geometry-summary",
        type=Path,
        default=CUDA_DYNAMIC_GEOMETRY_SUMMARY,
        help="Saved CUDA dynamic-geometry smoke summary JSON.",
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    report = audit(
        run_local_tests=bool(args.run_local_tests),
        allow_local_tests_unrun=bool(args.allow_local_tests_unrun),
        require_raw_quality=bool(args.require_raw_quality),
        same_split_comparison=args.same_split_comparison,
        metal_dynamic_geometry_summary=args.metal_dynamic_geometry_summary,
        cuda_dynamic_geometry_summary=args.cuda_dynamic_geometry_summary,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not bool(report["ok"]) and not bool(args.allow_incomplete):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
