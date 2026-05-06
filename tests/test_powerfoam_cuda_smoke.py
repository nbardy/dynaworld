from __future__ import annotations

import importlib.util
import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "research_experiments/dynamic_foam/powerfoam_cuda_smoke_runner.py"
MODAL_SMOKE = ROOT / "research_experiments/dynamic_foam/modal_powerfoam_cuda_smoke.py"
VERIFIER = ROOT / "research_experiments/dynamic_foam/verify_powerfoam_cuda_smoke_results.py"
COMPARE = ROOT / "research_experiments/dynamic_foam/compare_powerfoam_cuda_metal_smoke.py"
FEATURE_PATCH = ROOT / "research_experiments/dynamic_foam/cuda_forks/dynamic_feature_foam.patch"
GEOMETRY_PATCH = ROOT / "research_experiments/dynamic_foam/cuda_forks/dynamic_geometry_foam.patch"
OFFICIAL_COMMIT = "96392252ebd0059fe6ca98881b62e12295d9242f"


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _FakeModalImage:
    @classmethod
    def debian_slim(cls, **_kwargs):
        return cls()

    def apt_install(self, *_args, **_kwargs):
        return self

    def pip_install(self, *_args, **_kwargs):
        return self

    def workdir(self, *_args, **_kwargs):
        return self

    def add_local_dir(self, *_args, **_kwargs):
        return self

    def add_local_file(self, *_args, **_kwargs):
        return self


class _FakeModalApp:
    def __init__(self, *_args, **_kwargs):
        pass

    def function(self, *_args, **_kwargs):
        def decorator(fn):
            fn.remote = fn
            return fn

        return decorator

    def local_entrypoint(self, *_args, **_kwargs):
        def decorator(fn):
            return fn

        return decorator


def load_modal_smoke_module(monkeypatch):
    fake_modal = types.SimpleNamespace(Image=_FakeModalImage, App=_FakeModalApp)
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    spec = importlib.util.spec_from_file_location("modal_powerfoam_cuda_smoke_for_test", MODAL_SMOKE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_powerfoam_cuda_smoke_plan_validates_without_gpu(tmp_path: Path) -> None:
    # Guards the cheap deployment contract without spending Modal/GPU time.
    output_dir = tmp_path / "cuda_plan"
    run_command(
        [
            sys.executable,
            str(RUNNER),
            "--run-id",
            "pytest_cuda_plan",
            "--output-dir",
            str(output_dir),
            "--frames",
            "4",
            "--size",
            "64",
            "--iterations",
            "5",
            "--points",
            "256",
            "--num-texel-sites",
            "4",
            "--sv-dof",
            "2",
            "--max-gpu-minutes",
            "8",
            "--skip-official-fixture",
            "--fixed-black-background",
        ]
    )

    summary_path = output_dir / "summary.json"
    assert summary_path.exists()
    run_command([sys.executable, str(VERIFIER), str(summary_path), "--allow-planned"])

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "planned"
    assert summary["source"]["official_commit"] == OFFICIAL_COMMIT
    assert summary["settings"] == {
        "gpu": "L40S",
        "iterations": 5,
        "max_gpu_minutes": 8,
        "num_texel_sites": 4,
        "seed": 17,
        "dynamic_geometry": False,
        "dynamic_time_basis_count": 4,
        "dynamic_center_basis_count": 4,
        "dynamic_height_basis_count": 4,
        "skip_official_fixture": True,
        "fixed_black_background": True,
        "random_background": False,
        "output_dir": str(output_dir),
        "points": 256,
        "sv_dof": 2,
    }
    execute_command = summary["planned_commands"][-1]["command"]
    assert "--execute" in execute_command
    assert "--dynamic-geometry" not in execute_command
    assert "--skip-official-fixture" in execute_command
    assert "--fixed-black-background" in execute_command


def test_powerfoam_cuda_geometry_plan_selects_geometry_patch(tmp_path: Path) -> None:
    output_dir = tmp_path / "cuda_geometry_plan"
    run_command(
        [
            sys.executable,
            str(RUNNER),
            "--run-id",
            "pytest_cuda_geometry_plan",
            "--output-dir",
            str(output_dir),
            "--frames",
            "4",
            "--size",
            "64",
            "--iterations",
            "5",
            "--points",
            "256",
            "--num-texel-sites",
            "4",
            "--sv-dof",
            "2",
            "--dynamic-geometry",
            "--max-gpu-minutes",
            "8",
            "--skip-official-fixture",
            "--fixed-black-background",
        ]
    )

    summary_path = output_dir / "summary.json"
    run_command([sys.executable, str(VERIFIER), str(summary_path), "--allow-planned"])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "planned"
    assert summary["source"]["dynamic_feature_patch_sha256"] == sha256_file(FEATURE_PATCH)
    assert summary["source"]["dynamic_geometry_patch"] == "research_experiments/dynamic_foam/cuda_forks/dynamic_geometry_foam.patch"
    assert summary["source"]["dynamic_geometry_patch_sha256"] == sha256_file(GEOMETRY_PATCH)
    assert summary["settings"]["dynamic_geometry"] is True
    apply_command = summary["planned_commands"][-2]
    assert apply_command["command"][-1] == "research_experiments/dynamic_foam/cuda_forks/dynamic_geometry_foam.patch"
    assert "dynamic_geometry_foam_cuda" in apply_command["cwd"]
    execute_command = summary["planned_commands"][-1]["command"]
    assert "--dynamic-geometry" in execute_command
    failed = subprocess.run(
        [sys.executable, str(VERIFIER), str(summary_path), "--allow-planned", "--require-dynamic-geometry"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert failed.returncode != 0


def test_modal_cuda_wrapper_forwards_fast_deploy_flags(monkeypatch, tmp_path: Path) -> None:
    module = load_modal_smoke_module(monkeypatch)
    args = module.runner_args(
        execute=False,
        output_dir=tmp_path / "cuda_plan",
        run_id="pytest_modal_plan",
        preset="micro_clip_64_4f_5step",
        max_gpu_minutes=8,
        skip_official_fixture=True,
        fixed_black_background=True,
        dynamic_geometry=True,
    )

    def value_after(flag: str) -> str:
        return args[args.index(flag) + 1]

    assert args[0] == str(RUNNER)
    assert value_after("--frames") == "4"
    assert value_after("--size") == "64"
    assert value_after("--iterations") == "5"
    assert value_after("--points") == "256"
    assert value_after("--num-texel-sites") == "4"
    assert value_after("--sv-dof") == "2"
    assert value_after("--max-gpu-minutes") == "8"
    assert "--skip-official-fixture" in args
    assert "--fixed-black-background" in args
    assert "--dynamic-geometry" in args
    assert "--execute" not in args


def test_dynamic_cuda_patch_stays_small_and_appearance_side() -> None:
    patch = FEATURE_PATCH.read_text(encoding="utf-8")
    assert "diff --git a/configs/__init__.py" in patch
    assert "diff --git a/powerfoam/scene.py" in patch
    assert "dynamic_feature_foam" in patch
    assert "get_time_conditioned_texel_sv_rgb" in patch
    assert "diff --git a/powerfoam/rasterize" not in patch
    assert "diff --git a/powerfoam/raytrace" not in patch
    assert ".cu" not in patch


def test_dynamic_geometry_cuda_patch_stays_scene_side_geometry() -> None:
    patch = GEOMETRY_PATCH.read_text(encoding="utf-8")
    assert "diff --git a/configs/__init__.py" in patch
    assert "diff --git a/powerfoam/scene.py" in patch
    assert "dynamic_geometry_foam" in patch
    assert "dynamic_center_coeffs" in patch
    assert "dynamic_radius_coeffs" in patch
    assert "dynamic_height_coeffs" in patch
    assert "dynamic_state_delta_probe" in patch
    assert "texel_sites = points[:, None, :] + offsets" in patch
    assert "diff --git a/powerfoam/rasterize" not in patch
    assert "diff --git a/powerfoam/raytrace" not in patch
    assert ".cu" not in patch


def test_cuda_verifier_can_require_official_fixture(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    summary = {
        "schema_version": "powerfoam_cuda_smoke_v1",
        "status": "ok",
        "source": {
            "official_repo_url": "https://github.com/theialab/powerfoam",
            "official_commit": OFFICIAL_COMMIT,
            "dynamic_patch_kind": "feature",
            "dynamic_patch_sha256": sha256_file(FEATURE_PATCH),
        },
        "clip": {"frames": 4, "size": 64},
        "settings": {"iterations": 5},
        "host": {"torch_cuda_available": True, "cuda_device_name": "NVIDIA L40S"},
        "official_fixture": None,
        "comparisons": {
            "available": True,
            "static_warm_step_total_ms_mean": 1.0,
            "dynamic_warm_step_total_ms_mean": 1.0,
        },
        "runs": [
            {
                "name": "official_static_cuda",
                "status": "ok",
                "metrics": {"warm_timing_excluding_step0": {"step_total_ms_mean": 1.0}},
            },
            {
                "name": "dynamic_feature_foam_cuda",
                "status": "ok",
                "metrics": {
                    "warm_timing_excluding_step0": {"step_total_ms_mean": 1.0},
                    "dynamic": {
                        "camera_time_count": 4.0,
                        "camera_time_min": 0.0,
                        "camera_time_max": 1.0,
                        "dynamic_texel_sv_rgb_coeff_abs_mean": 1.0e-3,
                        "dynamic_texel_sv_rgb_coeff_abs_max": 1.0e-2,
                        "dynamic_time_rgb_delta_mean": 1.0e-5,
                        "dynamic_time_rgb_delta_max": 1.0e-4,
                    },
                },
            },
        ],
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    run_command([sys.executable, str(VERIFIER), str(summary_path)])
    unsupported_geometry = subprocess.run(
        [sys.executable, str(VERIFIER), str(summary_path), "--require-dynamic-geometry"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert unsupported_geometry.returncode != 0
    failed = subprocess.run(
        [sys.executable, str(VERIFIER), str(summary_path), "--require-official-fixture"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert failed.returncode != 0

    summary["official_fixture"] = {
        "status": "ok",
        "upstream_powerfoam_commit": OFFICIAL_COMMIT,
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    run_command([sys.executable, str(VERIFIER), str(summary_path), "--require-official-fixture"])


def test_cuda_verifier_requires_dynamic_geometry_alpha_support(tmp_path: Path) -> None:
    summary_path = tmp_path / "geometry_summary.json"
    summary = {
        "schema_version": "powerfoam_cuda_smoke_v1",
        "status": "ok",
        "source": {
            "official_repo_url": "https://github.com/theialab/powerfoam",
            "official_commit": OFFICIAL_COMMIT,
            "dynamic_patch_sha256": sha256_file(FEATURE_PATCH),
            "dynamic_feature_patch_sha256": sha256_file(FEATURE_PATCH),
            "dynamic_geometry_patch_sha256": sha256_file(GEOMETRY_PATCH),
        },
        "clip": {"frames": 4, "size": 64},
        "settings": {"iterations": 5, "dynamic_geometry": True},
        "host": {"torch_cuda_available": True, "cuda_device_name": "NVIDIA L40S"},
        "official_fixture": None,
        "comparisons": {
            "available": True,
            "dynamic_lane": "dynamic_feature_foam_cuda",
            "geometry_available": True,
            "static_warm_step_total_ms_mean": 1.0,
            "dynamic_warm_step_total_ms_mean": 1.0,
            "geometry_warm_step_total_ms_mean": 1.0,
        },
        "runs": [
            {
                "name": "official_static_cuda",
                "status": "ok",
                "metrics": {"warm_timing_excluding_step0": {"step_total_ms_mean": 1.0}},
            },
            {
                "name": "dynamic_feature_foam_cuda",
                "status": "ok",
                "metrics": {
                    "warm_timing_excluding_step0": {"step_total_ms_mean": 1.0},
                    "dynamic": {
                        "camera_time_count": 4.0,
                        "camera_time_min": 0.0,
                        "camera_time_max": 1.0,
                        "dynamic_texel_sv_rgb_coeff_abs_mean": 1.0e-3,
                        "dynamic_texel_sv_rgb_coeff_abs_max": 1.0e-2,
                        "dynamic_time_rgb_delta_mean": 1.0e-5,
                        "dynamic_time_rgb_delta_max": 1.0e-4,
                    },
                },
            },
            {
                "name": "dynamic_geometry_foam_cuda",
                "status": "ok",
                "metrics": {
                    "warm_timing_excluding_step0": {"step_total_ms_mean": 1.0},
                    "dynamic": {
                        "camera_time_count": 4.0,
                        "camera_time_min": 0.0,
                        "camera_time_max": 1.0,
                        "dynamic_center_coeffs_abs_mean": 1.0e-3,
                        "dynamic_center_coeffs_abs_max": 1.0e-2,
                        "dynamic_time_point_delta_mean": 1.0e-5,
                        "dynamic_time_point_delta_max": 1.0e-4,
                        "dynamic_time_alpha_delta_mean": 1.0e-5,
                        "dynamic_time_alpha_delta_max": 1.0e-4,
                        "dynamic_time_alpha_support_delta_fraction": 0.01,
                        "dynamic_time_alpha_support_pixels_0": 12.0,
                        "dynamic_time_alpha_support_pixels_1": 14.0,
                    },
                },
            },
        ],
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    run_command([sys.executable, str(VERIFIER), str(summary_path)])
    run_command([sys.executable, str(VERIFIER), str(summary_path), "--require-dynamic-geometry"])

    summary["runs"][2]["metrics"]["dynamic"]["dynamic_time_alpha_support_delta_fraction"] = 0.0
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    failed = subprocess.run(
        [sys.executable, str(VERIFIER), str(summary_path), "--require-dynamic-geometry"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert failed.returncode != 0


def test_cuda_metal_comparison_contract_writes_json(tmp_path: Path) -> None:
    cuda_summary = tmp_path / "cuda_summary.json"
    metal_output = tmp_path / "metal"
    report_path = tmp_path / "comparison.json"
    metal_output.mkdir()
    cuda_summary.write_text(
        json.dumps(
            {
                "schema_version": "powerfoam_cuda_smoke_v1",
                "status": "ok",
                "clip": {
                    "path": "test_data/test_video_small_128_4fps.mp4",
                    "frames": 4,
                    "size": 64,
                },
                "settings": {
                    "fixed_black_background": True,
                    "iterations": 5,
                    "points": 256,
                    "num_texel_sites": 4,
                    "sv_dof": 2,
                },
                "runs": [
                    {
                        "name": "official_static_cuda",
                        "status": "ok",
                        "metrics": {
                            "eval": {
                                "eval_l1": 0.49,
                                "eval_mse": 0.28,
                                "eval_psnr": 5.54,
                                "eval_ssim": 0.02,
                            },
                            "warm_timing_excluding_step0": {"step_total_ms_mean": 6.9},
                            "timing": {"step_total_ms_mean": 1200.0},
                            "dynamic": {},
                            "model": {"points": 256, "num_texel_sites": 4, "sv_dof": 2},
                        },
                    },
                    {
                        "name": "dynamic_feature_foam_cuda",
                        "status": "ok",
                        "metrics": {
                            "eval": {
                                "eval_l1": 0.48,
                                "eval_mse": 0.27,
                                "eval_psnr": 5.60,
                                "eval_ssim": 0.03,
                            },
                            "warm_timing_excluding_step0": {"step_total_ms_mean": 7.1},
                            "timing": {"step_total_ms_mean": 130.0},
                            "dynamic": {
                                "camera_time_min": 0.0,
                                "camera_time_max": 1.0,
                                "dynamic_time_rgb_delta_mean": 1.0e-4,
                            },
                            "model": {"points": 256, "num_texel_sites": 4, "sv_dof": 2},
                        },
                    },
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (metal_output / "best_metrics.json").write_text(
        json.dumps(
            {
                "step": 4,
                "best_metric_name": "eval_psnr",
                "metrics": {
                    "eval_l1": 0.50,
                    "eval_mse": 0.30,
                    "eval_psnr": 5.1,
                    "eval_ssim": 0.01,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (metal_output / "resolved_config.json").write_text(
        json.dumps(
            {
                "data": {"video_path": "test_data/test_video_small_128_4fps.mp4", "max_frames": 4},
                "render": {"render_size": 64, "use_raytrace": True, "background": [0.0, 0.0, 0.0]},
                "train": {"steps": 5, "device": "mps"},
                "model": {
                    "cells": 256,
                    "num_texel_sites": 4,
                    "sv_dof": 2,
                    "adjacency_mode": "cech_aabb",
                    "init_from_video": False,
                    "color_init_mode": "random",
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (metal_output / "train_metrics_history.jsonl").write_text(
        json.dumps({"step": 4, "elapsed_s": 1.25}) + "\n",
        encoding="utf-8",
    )

    run_command(
        [
            sys.executable,
            str(COMPARE),
            "--cuda-summary",
            str(cuda_summary),
            "--metal-output",
            str(metal_output),
            "--output",
            str(report_path),
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == "powerfoam_cuda_metal_smoke_comparison_v1"
    assert report["status"] == "ok"
    assert all(report["matched_contract"].values())
    assert set(report["lanes"]) == {
        "official_static_cuda",
        "dynamic_feature_foam_cuda",
        "powerfoam_metal_micro_match",
    }
