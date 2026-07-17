from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from export_powerfoam_smoke_dataset import export_dataset, sha256_file
except ModuleNotFoundError:
    from research_experiments.dynamic_foam.export_powerfoam_smoke_dataset import export_dataset, sha256_file

try:
    from .report_artifacts import PROJECT_ROOT, load_report_json, relative_to_project as rel, write_report_json
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import PROJECT_ROOT, load_report_json, relative_to_project as rel, write_report_json


ROOT = PROJECT_ROOT
UPSTREAM_REPO_URL = "https://github.com/theialab/powerfoam"
UPSTREAM_COMMIT = "96392252ebd0059fe6ca98881b62e12295d9242f"
DEFAULT_VIDEO = ROOT / "test_data/test_video_small_128_4fps.mp4"
DEFAULT_FIXTURE_SOURCE = ROOT / "research_experiments/dynamic_foam/fixtures/powerfoam_tiny_height_sv_origin_parity_v1.json"
DEFAULT_FIXTURE_NAME = "powerfoam_tiny_height_sv_official_camera_official_v1.json"
DYNAMIC_FEATURE_PATCH = ROOT / "research_experiments/dynamic_foam/cuda_forks/dynamic_feature_foam.patch"
DYNAMIC_GEOMETRY_PATCH = ROOT / "research_experiments/dynamic_foam/cuda_forks/dynamic_geometry_foam.patch"
DYNAMIC_PATCHES = {
    "feature": DYNAMIC_FEATURE_PATCH,
    "geometry": DYNAMIC_GEOMETRY_PATCH,
}
DYNAMIC_PATCH = DYNAMIC_FEATURE_PATCH
OFFICIAL_FIXTURE_SCRIPT = ROOT / "research_experiments/dynamic_foam/make_powerfoam_official_parity_fixture.py"


SMOKE_ENTRY = r'''
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import warp as wp

from data_loader import DataHandler
from powerfoam.metrics import psnr, ssim_eval
from powerfoam.scene import PowerfoamScene


def sync_cuda() -> None:
    torch.cuda.synchronize()


def timed_call(fn):
    sync_cuda()
    start = time.perf_counter()
    value = fn()
    sync_cuda()
    return value, (time.perf_counter() - start) * 1000.0


def camera_time_value(index: int, count: int) -> float:
    return 0.0 if count <= 1 else float(index) / float(count - 1)


def attach_time_indices(handler: DataHandler) -> None:
    count = len(handler.cameras)
    for index, camera in enumerate(handler.cameras):
        camera.time_index = camera_time_value(index, count)
    missing = [index for index, camera in enumerate(handler.cameras) if not hasattr(camera, "time_index")]
    if missing:
        raise RuntimeError(f"Missing time_index on cameras: {missing}")


def camera_time_stats(handler: DataHandler) -> dict[str, float]:
    values = [float(camera.time_index) for camera in handler.cameras]
    return {
        "camera_time_min": float(min(values)),
        "camera_time_max": float(max(values)),
        "camera_time_count": float(len(values)),
    }


def namespace_from_settings(settings: dict[str, Any]) -> SimpleNamespace:
    points = int(settings["points"])
    iterations = int(settings["iterations"])
    return SimpleNamespace(
        iterations=iterations,
        normal_weight=0.0,
        contribution_weight=0.0,
        interpenetration_weight=0.0,
        densify_from=10**9,
        densify_until=10**9,
        experiment_name=str(settings["run_name"]),
        dry_run=True,
        viewer=False,
        normal_supervision=False,
        dataset="blender",
        data_path=str(settings["dataset_root"]),
        scene=str(settings["scene_name"]),
        alpha_format_on_disk="straight",
        downsample=[1],
        downsample_iterations=[],
        use_metric3d=False,
        is_pinhole=True,
        eval=False,
        init_type="random_bounded",
        init_points=points,
        final_points=points,
        bkgd_color=[0.0, 0.0, 0.0],
        disable_coop_prim_load=False,
        disable_coop_adj_load=False,
        render_objective="volume",
        sv_dof=int(settings["sv_dof"]),
        num_texel_sites=int(settings["num_texel_sites"]),
        dynamic_feature_foam=bool(settings.get("dynamic_feature_foam", False)),
        dynamic_geometry_foam=bool(settings.get("dynamic_geometry_foam", False)),
        dynamic_time_basis_count=int(settings.get("dynamic_time_basis_count", 4)),
        dynamic_time_basis_sigma_scale=float(settings.get("dynamic_time_basis_sigma_scale", 0.75)),
        dynamic_texel_sv_rgb_lr_init=float(settings.get("dynamic_texel_sv_rgb_lr", 5.0e-3)),
        dynamic_texel_sv_rgb_lr_final=float(settings.get("dynamic_texel_sv_rgb_lr_final", 5.0e-4)),
        dynamic_center_lr_init=float(settings.get("dynamic_center_lr", 1.0e-3)),
        dynamic_center_lr_final=float(settings.get("dynamic_center_lr_final", 5.0e-5)),
        dynamic_radius_lr_init=float(settings.get("dynamic_radius_lr", 5.0e-5)),
        dynamic_radius_lr_final=float(settings.get("dynamic_radius_lr_final", 5.0e-6)),
        dynamic_quaternion_lr_init=float(settings.get("dynamic_quaternion_lr", 1.0e-2)),
        dynamic_quaternion_lr_final=float(settings.get("dynamic_quaternion_lr_final", 1.0e-3)),
        dynamic_height_lr_init=float(settings.get("dynamic_height_lr", 5.0e-3)),
        dynamic_height_lr_final=float(settings.get("dynamic_height_lr_final", 5.0e-4)),
        points_lr_init=1.0e-3,
        points_lr_final=5.0e-5,
        density_lr_init=1.0,
        density_lr_final=1.0,
        radii_lr_init=5.0e-5,
        radii_lr_final=5.0e-6,
        quaternions_lr_init=1.0e-1,
        quaternions_lr_final=1.0e-2,
        texel_sites_lr_init=1.0e-2,
        texel_sites_lr_final=1.0e-3,
        texel_sv_axis_lr_init=5.0e-2,
        texel_sv_axis_lr_final=5.0e-3,
        texel_sv_rgb_lr_init=5.0e-3,
        texel_sv_rgb_lr_final=5.0e-4,
        texel_height_lr_init=5.0e-3,
        texel_height_lr_final=5.0e-4,
    )


def prepare_model_for_camera(model: PowerfoamScene, camera) -> None:
    if hasattr(model, "set_active_time"):
        model.set_active_time(camera)
    elif hasattr(model, "_active_time_index"):
        model._active_time_index = float(getattr(camera, "time_index", 0.0))
    if bool(getattr(model.args, "dynamic_geometry_foam", False)):
        model.rebuild_adjacency()


def eval_model(model: PowerfoamScene, handler: DataHandler) -> dict[str, float]:
    l1_values: list[float] = []
    mse_values: list[float] = []
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    alpha_values: list[float] = []
    with torch.no_grad():
        for camera, rgb_gt, alpha_gt in zip(handler.cameras, handler.rgbs, handler.alphas):
            target = rgb_gt.cuda(non_blocking=True)
            alpha_target = alpha_gt.cuda(non_blocking=True)
            target = target + (1.0 - alpha_target[..., None]) * 0.0
            prepare_model_for_camera(model, camera)
            result = model.forward(camera, depth_quantiles=None, ray_gt=None, return_point_err=False)
            rgb = result[0].clamp(0.0, 1.0)
            alpha = result[1].clamp(0.0, 1.0)
            rgb = rgb + (1.0 - alpha[..., None]) * 0.0
            l1_values.append(float((rgb - target).abs().mean().detach().cpu()))
            mse_values.append(float(F.mse_loss(rgb, target).detach().cpu()))
            psnr_values.append(float(psnr(rgb, target).detach().cpu()))
            ssim_values.append(float(ssim_eval(rgb, target).detach().cpu()))
            alpha_values.append(float(alpha.mean().detach().cpu()))
    return {
        "eval_l1": float(np.mean(l1_values)),
        "eval_mse": float(np.mean(mse_values)),
        "eval_psnr": float(np.mean(psnr_values)),
        "eval_ssim": float(np.mean(ssim_values)),
        "eval_alpha_mean": float(np.mean(alpha_values)),
    }


def dynamic_time_causality_probe(model: PowerfoamScene, handler: DataHandler) -> dict[str, float]:
    has_feature_state = hasattr(model, "dynamic_texel_sv_rgb_coeff")
    has_geometry_state = any(
        hasattr(model, name)
        for name in (
            "dynamic_center_coeffs",
            "dynamic_radius_coeffs",
            "dynamic_quaternion_coeffs",
            "dynamic_height_coeffs",
        )
    )
    if not (has_feature_state or has_geometry_state):
        return {}
    camera = handler.cameras[0]
    original_time = getattr(camera, "time_index", None)
    original_model_time = getattr(model, "_active_time_index", None)
    with torch.no_grad():
        camera.time_index = 0.0
        prepare_model_for_camera(model, camera)
        result_0 = model.forward(camera, depth_quantiles=None, ray_gt=None, return_point_err=False)
        rgb_0 = result_0[0].detach()
        alpha_0 = result_0[1].detach()
        points_0 = None
        if hasattr(model, "get_points"):
            model._active_time_index = 0.0
            points_0 = model.get_points().detach()
        camera.time_index = 1.0
        prepare_model_for_camera(model, camera)
        result_1 = model.forward(camera, depth_quantiles=None, ray_gt=None, return_point_err=False)
        rgb_1 = result_1[0].detach()
        alpha_1 = result_1[1].detach()
        points_1 = None
        if hasattr(model, "get_points"):
            model._active_time_index = 1.0
            points_1 = model.get_points().detach()
    if original_time is not None:
        camera.time_index = original_time
    if original_model_time is not None:
        model._active_time_index = original_model_time
    elif hasattr(model, "_active_time_index"):
        delattr(model, "_active_time_index")
    support_0 = alpha_0 > 1.0e-4
    support_1 = alpha_1 > 1.0e-4
    support_union = support_0 | support_1
    metrics = {
        "dynamic_time_rgb_delta_mean": float((rgb_1 - rgb_0).abs().mean().cpu()),
        "dynamic_time_rgb_delta_max": float((rgb_1 - rgb_0).abs().max().cpu()),
        "dynamic_time_alpha_delta_mean": float((alpha_1 - alpha_0).abs().mean().cpu()),
        "dynamic_time_alpha_delta_max": float((alpha_1 - alpha_0).abs().max().cpu()),
        "dynamic_time_alpha_support_delta_fraction": float((support_1.float() - support_0.float()).abs().mean().cpu()),
        "same_camera_support_delta_mean": float((support_1.float() - support_0.float()).abs().mean().cpu()),
        "time_alpha_delta_mean": float((alpha_1 - alpha_0).abs().mean().cpu()),
        "time_rgb_delta_mean": float((rgb_1 - rgb_0).abs().mean().cpu()),
        "dynamic_time_alpha_support_iou": float(
            ((support_0 & support_1).float().sum() / support_union.float().sum().clamp_min(1.0)).cpu()
        ),
        "dynamic_time_alpha_support_pixels_0": float(support_0.float().sum().cpu()),
        "dynamic_time_alpha_support_pixels_1": float(support_1.float().sum().cpu()),
        "dynamic_time_probe_camera_index": 0.0,
    }
    if points_0 is not None and points_1 is not None:
        metrics.update(
            {
                "dynamic_time_point_delta_mean": float((points_1 - points_0).abs().mean().cpu()),
                "dynamic_time_point_delta_max": float((points_1 - points_0).abs().max().cpu()),
            }
        )
    if hasattr(model, "dynamic_state_delta_probe"):
        metrics.update(model.dynamic_state_delta_probe(0.0, 1.0))
    return metrics


def train_smoke(settings: dict[str, Any]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the PowerFoam CUDA smoke.")
    wp.init()
    seed = int(settings.get("seed", 17))
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    args = namespace_from_settings(settings)
    test_handler = DataHandler(args)
    test_handler.reload("all", downsample=1)
    attach_time_indices(test_handler)
    train_handler = DataHandler(args)
    train_handler.reload("all", downsample=1)
    attach_time_indices(train_handler)
    train_time_stats = camera_time_stats(train_handler)
    train_iter = train_handler.get_iter()

    model = PowerfoamScene(args)
    _, init_ms = timed_call(lambda: model.initialize_from_dataset(train_handler, device="cuda"))
    model.declare_optimizers(args, args.iterations)
    _, sort_ms = timed_call(model.sort_points)

    step_rows = []
    last_scalars: dict[str, float] = {}
    total_start = time.perf_counter()
    for step in range(int(settings["iterations"])):
        (camera, rgb_gt, alpha_gt, _normal_gt), load_ms = timed_call(lambda: next(train_iter))

        def zero_grad():
            model.optimizer.zero_grad(set_to_none=True)

        _, zero_grad_ms = timed_call(zero_grad)

        def rebuild():
            prepare_model_for_camera(model, camera)
            if not bool(getattr(model.args, "dynamic_geometry_foam", False)):
                model.rebuild_adjacency()

        _, adjacency_ms = timed_call(rebuild)

        def forward_loss():
            random_bkgd = torch.rand_like(rgb_gt) if bool(settings.get("random_background", True)) else torch.zeros_like(rgb_gt)
            target = rgb_gt + (1.0 - alpha_gt[..., None]) * random_bkgd
            result = model.forward(camera, depth_quantiles=None, ray_gt=target, return_point_err=True)
            rgb = result[0] + (1.0 - result[1][..., None]) * random_bkgd
            rgb_loss = F.mse_loss(rgb, target, reduction="none").sum(dim=-1).mean()
            loss = rgb_loss
            return loss, result, target, rgb_loss

        (loss, result, target, rgb_loss), forward_ms = timed_call(forward_loss)
        _, backward_ms = timed_call(loss.backward)

        def optimizer_step():
            model.optimizer.step()
            model.update_learning_rate(step)
            model.update_stats(result[6], result[7], result[8])

        _, optimizer_ms = timed_call(optimizer_step)
        rgb_psnr = psnr((result[0] + (1.0 - result[1][..., None]) * 0.0).clamp(0.0, 1.0), target.clamp(0.0, 1.0))
        last_scalars = {
            "loss": float(loss.detach().cpu()),
            "rgb_mse_loss": float(rgb_loss.detach().cpu()),
            "train_psnr": float(rgb_psnr.detach().cpu()),
            "alpha_mean": float(result[1].mean().detach().cpu()),
        }
        step_rows.append(
            {
                "step": int(step),
                "load_ms": load_ms,
                "zero_grad_ms": zero_grad_ms,
                "adjacency_ms": adjacency_ms,
                "forward_ms": forward_ms,
                "backward_ms": backward_ms,
                "optimizer_ms": optimizer_ms,
                **last_scalars,
            }
        )
    total_train_s = time.perf_counter() - total_start
    eval_metrics = eval_model(model, test_handler)
    dynamic_metrics: dict[str, float] = {}
    if hasattr(model, "dynamic_texel_sv_rgb_coeff") or hasattr(model, "dynamic_center_coeffs"):
        dynamic_metrics = {
            **train_time_stats,
            **dynamic_time_causality_probe(model, test_handler),
        }
        if hasattr(model, "dynamic_texel_sv_rgb_coeff"):
            coeff = model.dynamic_texel_sv_rgb_coeff.detach()
            dynamic_metrics.update(
                {
                    "dynamic_texel_sv_rgb_coeff_abs_mean": float(coeff.abs().mean().cpu()),
                    "dynamic_texel_sv_rgb_coeff_abs_max": float(coeff.abs().max().cpu()),
                }
            )
        for name in (
            "dynamic_center_coeffs",
            "dynamic_radius_coeffs",
            "dynamic_quaternion_coeffs",
            "dynamic_height_coeffs",
        ):
            if hasattr(model, name):
                coeff = getattr(model, name).detach()
                dynamic_metrics.update(
                    {
                        f"{name}_abs_mean": float(coeff.abs().mean().cpu()),
                        f"{name}_abs_max": float(coeff.abs().max().cpu()),
                    }
                )

    phase_keys = ["load_ms", "adjacency_ms", "forward_ms", "backward_ms", "optimizer_ms"]
    timing = {
        f"{key}_mean": float(np.mean([row[key] for row in step_rows]))
        for key in phase_keys
    }
    timing.update(
        {
            "init_ms": init_ms,
            "sort_ms": sort_ms,
            "train_total_s": total_train_s,
            "step_total_ms_mean": float(
                np.mean(
                    [
                        row["load_ms"] + row["adjacency_ms"] + row["forward_ms"] + row["backward_ms"] + row["optimizer_ms"]
                        for row in step_rows
                    ]
                )
            ),
        }
    )
    return {
        "schema_version": "powerfoam_cuda_lane_v1",
        "run_name": str(settings["run_name"]),
        "dynamic_feature_foam": bool(settings.get("dynamic_feature_foam", False)),
        "dynamic_geometry_foam": bool(settings.get("dynamic_geometry_foam", False)),
        "host": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_name": torch.cuda.get_device_name(0),
            "warp": getattr(wp, "__version__", "unknown"),
        },
        "settings": settings,
        "final_train": last_scalars,
        "eval": eval_metrics,
        "dynamic": dynamic_metrics,
        "timing": timing,
        "steps": step_rows,
        "model": {
            "points": int(model.points.shape[0]),
            "num_texel_sites": int(args.num_texel_sites),
            "sv_dof": int(args.sv_dof),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text(encoding="utf-8"))
    payload = train_smoke(settings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["eval"], sort_keys=True))


if __name__ == "__main__":
    main()
'''


def now_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dynamic_patch_path(kind: str) -> Path:
    try:
        return DYNAMIC_PATCHES[kind]
    except KeyError as exc:
        raise ValueError(f"Unknown dynamic patch kind {kind!r}; choices: {sorted(DYNAMIC_PATCHES)}") from exc


def run_command(
    command: list[str],
    *,
    cwd: Path,
    timeout_s: int | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    process_env = os.environ.copy()
    if env is not None:
        process_env.update(env)
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            env=process_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stdout = exc.stdout or ""
        raise RuntimeError(
            "Command failed "
            f"with exit code {exc.returncode}: {command}\n"
            f"cwd: {cwd}\n"
            f"stdout_tail:\n{stdout[-8000:]}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        raise RuntimeError(
            "Command timed out "
            f"after {timeout_s}s: {command}\n"
            f"cwd: {cwd}\n"
            f"stdout_tail:\n{stdout[-8000:]}"
        ) from exc


def command_record(command: list[str], cwd: Path) -> dict[str, Any]:
    return {"cwd": str(cwd), "command": command}


def preflight() -> dict[str, Any]:
    import torch
    import warp

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "warp": getattr(warp, "__version__", "unknown"),
    }


def clone_official_repo(*, repo_url: str, commit: str, dest: Path, timeout_s: int) -> dict[str, Any]:
    if dest.exists():
        shutil.rmtree(dest)
    clone = run_command(["git", "clone", "--no-checkout", repo_url, str(dest)], cwd=dest.parent, timeout_s=timeout_s)
    fetch = run_command(["git", "fetch", "--depth", "1", "origin", commit], cwd=dest, timeout_s=timeout_s)
    checkout = run_command(["git", "checkout", "FETCH_HEAD"], cwd=dest, timeout_s=timeout_s)
    actual = run_command(["git", "rev-parse", "HEAD"], cwd=dest, timeout_s=timeout_s).stdout.strip()
    if actual != commit:
        raise RuntimeError(f"Expected official commit {commit}, got {actual}")
    return {
        "repo_url": repo_url,
        "commit": commit,
        "clone_log": clone.stdout[-4000:],
        "fetch_log": fetch.stdout[-4000:],
        "checkout_log": checkout.stdout[-4000:],
    }


def copy_dynamic_fork(static_repo: Path, dynamic_repo: Path, *, patch_kind: str) -> dict[str, Any]:
    if dynamic_repo.exists():
        shutil.rmtree(dynamic_repo)
    shutil.copytree(static_repo, dynamic_repo, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    patch_path = dynamic_patch_path(patch_kind)
    patch_text = patch_path.read_text(encoding="utf-8")
    result = run_command(["git", "apply", str(patch_path)], cwd=dynamic_repo, timeout_s=120)
    return {
        "patch_kind": patch_kind,
        "patch": rel(patch_path),
        "patch_sha256": sha256_text(patch_text),
        "apply_log": result.stdout[-4000:],
    }


def write_entry(repo: Path) -> Path:
    path = repo / "dynaworld_cuda_smoke_entry.py"
    path.write_text(SMOKE_ENTRY.strip() + "\n", encoding="utf-8")
    return path


def run_official_fixture(*, upstream_root: Path, output_dir: Path, timeout_s: int) -> dict[str, Any]:
    output = output_dir / "fixtures" / DEFAULT_FIXTURE_NAME
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(OFFICIAL_FIXTURE_SCRIPT),
        "--backend",
        "official",
        "--upstream-root",
        str(upstream_root),
        "--fixture",
        str(DEFAULT_FIXTURE_SOURCE),
        "--output",
        str(output),
    ]
    completed = run_command(command, cwd=ROOT, timeout_s=timeout_s, env={"PYTHONPATH": str(ROOT / "src/train")})
    payload = load_report_json(output)
    return {
        "name": "official_cuda_warp_fixture",
        "status": "ok",
        "path": str(output),
        "relative_path": str(output.relative_to(output_dir)),
        "backend": payload.get("metadata", {}).get("backend"),
        "upstream_powerfoam_commit": payload.get("metadata", {}).get("upstream_powerfoam_commit"),
        "log": completed.stdout[-4000:],
    }


def base_smoke_settings(
    *,
    run_name: str,
    dataset_root: Path,
    scene_name: str,
    iterations: int,
    points: int,
    num_texel_sites: int,
    sv_dof: int,
    seed: int,
    random_background: bool,
    dynamic: bool,
    dynamic_patch_kind: str,
    dynamic_time_basis_count: int,
    dynamic_center_basis_count: int,
    dynamic_height_basis_count: int,
) -> dict[str, Any]:
    if dynamic_patch_kind not in DYNAMIC_PATCHES:
        raise ValueError(f"Unknown dynamic patch kind {dynamic_patch_kind!r}; choices: {sorted(DYNAMIC_PATCHES)}")
    return {
        "run_name": run_name,
        "dataset_root": str(dataset_root),
        "scene_name": scene_name,
        "iterations": int(iterations),
        "points": int(points),
        "num_texel_sites": int(num_texel_sites),
        "sv_dof": int(sv_dof),
        "seed": int(seed),
        "random_background": bool(random_background),
        "dynamic_patch_kind": dynamic_patch_kind,
        "dynamic_feature_foam": bool(dynamic and dynamic_patch_kind == "feature"),
        "dynamic_geometry_foam": bool(dynamic and dynamic_patch_kind == "geometry"),
        "dynamic_time_basis_count": int(dynamic_time_basis_count),
        "dynamic_center_basis_count": int(dynamic_center_basis_count),
        "dynamic_height_basis_count": int(dynamic_height_basis_count),
        "dynamic_time_basis_sigma_scale": 0.75,
    }


def run_lane(
    *,
    repo: Path,
    settings: dict[str, Any],
    output_dir: Path,
    timeout_s: int,
) -> dict[str, Any]:
    entry = write_entry(repo)
    lane_dir = output_dir / "lanes" / str(settings["run_name"])
    lane_dir.mkdir(parents=True, exist_ok=True)
    settings_path = lane_dir / "settings.json"
    metrics_path = lane_dir / "metrics.json"
    write_report_json(settings_path, settings)
    command = [sys.executable, str(entry), "--settings", str(settings_path), "--output", str(metrics_path)]
    completed = run_command(command, cwd=repo, timeout_s=timeout_s)
    metrics = load_report_json(metrics_path)
    warm_timing = warm_timing_from_steps(metrics.get("steps", []))
    return {
        "name": str(settings["run_name"]),
        "status": "ok",
        "repo": str(repo),
        "metrics_path": str(metrics_path),
        "settings_path": str(settings_path),
        "log": completed.stdout[-4000:],
        "metrics": {
            "final_train": metrics.get("final_train", {}),
            "eval": metrics.get("eval", {}),
            "timing": metrics.get("timing", {}),
            "warm_timing_excluding_step0": warm_timing,
            "dynamic": metrics.get("dynamic", {}),
            "model": metrics.get("model", {}),
        },
    }


def warm_timing_from_steps(steps: list[dict[str, Any]]) -> dict[str, float]:
    warm_steps = steps[1:]
    if not warm_steps:
        return {}
    keys = ["load_ms", "adjacency_ms", "forward_ms", "backward_ms", "optimizer_ms"]
    timing = {
        f"{key}_mean": float(sum(float(row[key]) for row in warm_steps) / len(warm_steps))
        for key in keys
        if all(key in row for row in warm_steps)
    }
    if all(key in timing for key in [f"{key}_mean" for key in keys]):
        timing["step_total_ms_mean"] = float(sum(timing[f"{key}_mean"] for key in keys))
    return timing


def compare_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {str(run["name"]): run for run in runs if run.get("status") == "ok"}
    static = by_name.get("official_static_cuda")
    dynamic = by_name.get("dynamic_feature_foam_cuda")
    geometry = by_name.get("dynamic_geometry_foam_cuda")
    if static is None or dynamic is None:
        return {"available": False}
    static_metrics = static["metrics"]
    dynamic_metrics = dynamic["metrics"]
    static_eval = static_metrics.get("eval", {})
    dynamic_eval = dynamic_metrics.get("eval", {})
    static_timing = static_metrics.get("timing", {})
    dynamic_timing = dynamic_metrics.get("timing", {})
    static_warm_timing = static_metrics.get("warm_timing_excluding_step0", {})
    dynamic_warm_timing = dynamic_metrics.get("warm_timing_excluding_step0", {})
    static_step_ms = float(static_timing.get("step_total_ms_mean", 0.0))
    dynamic_step_ms = float(dynamic_timing.get("step_total_ms_mean", 0.0))
    static_warm_step_ms = float(static_warm_timing.get("step_total_ms_mean", 0.0))
    dynamic_warm_step_ms = float(dynamic_warm_timing.get("step_total_ms_mean", 0.0))
    return {
        "available": True,
        "dynamic_lane": "dynamic_feature_foam_cuda",
        "delta_eval_psnr_dynamic_minus_static": float(dynamic_eval.get("eval_psnr", 0.0))
        - float(static_eval.get("eval_psnr", 0.0)),
        "delta_eval_ssim_dynamic_minus_static": float(dynamic_eval.get("eval_ssim", 0.0))
        - float(static_eval.get("eval_ssim", 0.0)),
        "speed_ratio_dynamic_over_static": None if static_step_ms <= 0.0 else dynamic_step_ms / static_step_ms,
        "static_step_total_ms_mean": static_step_ms,
        "dynamic_step_total_ms_mean": dynamic_step_ms,
        "warm_speed_ratio_dynamic_over_static": None
        if static_warm_step_ms <= 0.0
        else dynamic_warm_step_ms / static_warm_step_ms,
        "static_warm_step_total_ms_mean": static_warm_step_ms,
        "dynamic_warm_step_total_ms_mean": dynamic_warm_step_ms,
        "geometry_available": geometry is not None,
        "geometry_delta_eval_psnr_minus_static": None
        if geometry is None
        else float(geometry["metrics"].get("eval", {}).get("eval_psnr", 0.0))
        - float(static_eval.get("eval_psnr", 0.0)),
        "geometry_warm_step_total_ms_mean": None
        if geometry is None
        else float(geometry["metrics"].get("warm_timing_excluding_step0", {}).get("step_total_ms_mean", 0.0)),
    }


def planned_summary(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    video = Path(args.video)
    execute_command = [
        sys.executable,
        rel(Path(__file__)),
        "--execute",
        "--run-id",
        str(args.run_id),
        "--output-dir",
        str(output_dir),
        "--video",
        rel(video),
        "--frames",
        str(int(args.frames)),
        "--size",
        str(int(args.size)),
        "--iterations",
        str(int(args.iterations)),
        "--points",
        str(int(args.points)),
        "--num-texel-sites",
        str(int(args.num_texel_sites)),
        "--sv-dof",
        str(int(args.sv_dof)),
        "--dynamic-time-basis-count",
        str(int(args.dynamic_time_basis_count)),
        "--dynamic-center-basis-count",
        str(int(args.dynamic_center_basis_count)),
        "--dynamic-height-basis-count",
        str(int(args.dynamic_height_basis_count)),
        "--seed",
        str(int(args.seed)),
        "--gpu",
        str(args.gpu),
        "--max-gpu-minutes",
        str(int(args.max_gpu_minutes)),
    ]
    if args.skip_official_fixture:
        execute_command.append("--skip-official-fixture")
    if args.fixed_black_background:
        execute_command.append("--fixed-black-background")
    if args.dynamic_geometry:
        execute_command.append("--dynamic-geometry")
    patch_sources = {
        "feature": {
            "path": rel(DYNAMIC_FEATURE_PATCH),
            "sha256": sha256_text(DYNAMIC_FEATURE_PATCH.read_text(encoding="utf-8")),
        },
        "geometry": {
            "path": rel(DYNAMIC_GEOMETRY_PATCH),
            "sha256": sha256_text(DYNAMIC_GEOMETRY_PATCH.read_text(encoding="utf-8")),
        },
    }
    return {
        "schema_version": "powerfoam_cuda_smoke_v1",
        "status": "planned",
        "run_id": args.run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "official_repo_url": args.official_repo_url,
            "official_commit": args.official_commit,
            "dynamic_patch": patch_sources["feature"]["path"],
            "dynamic_patch_sha256": patch_sources["feature"]["sha256"],
            "dynamic_feature_patch": patch_sources["feature"]["path"],
            "dynamic_feature_patch_sha256": patch_sources["feature"]["sha256"],
            "dynamic_geometry_patch": patch_sources["geometry"]["path"],
            "dynamic_geometry_patch_sha256": patch_sources["geometry"]["sha256"],
        },
        "clip": {
            "path": rel(video),
            "sha256": sha256_file(video) if video.exists() else None,
            "frames": int(args.frames),
            "size": int(args.size),
        },
        "settings": {
            "iterations": int(args.iterations),
            "points": int(args.points),
            "num_texel_sites": int(args.num_texel_sites),
            "sv_dof": int(args.sv_dof),
            "gpu": args.gpu,
            "max_gpu_minutes": int(args.max_gpu_minutes),
            "seed": int(args.seed),
            "dynamic_geometry": bool(args.dynamic_geometry),
            "dynamic_time_basis_count": int(args.dynamic_time_basis_count),
            "dynamic_center_basis_count": int(args.dynamic_center_basis_count),
            "dynamic_height_basis_count": int(args.dynamic_height_basis_count),
            "skip_official_fixture": bool(args.skip_official_fixture),
            "fixed_black_background": bool(args.fixed_black_background),
            "random_background": not bool(args.fixed_black_background),
            "output_dir": str(output_dir),
        },
        "planned_commands": [
            command_record(["git", "clone", "--no-checkout", args.official_repo_url, "<workdir>/official_static_cuda"], output_dir),
            command_record(["git", "fetch", "--depth", "1", "origin", args.official_commit], Path("<workdir>/official_static_cuda")),
            command_record(["git", "checkout", "FETCH_HEAD"], Path("<workdir>/official_static_cuda")),
            command_record(
                ["git", "apply", patch_sources["feature"]["path"]],
                Path("<workdir>/dynamic_feature_foam_cuda"),
            ),
            *(
                [
                    command_record(
                        ["git", "apply", patch_sources["geometry"]["path"]],
                        Path("<workdir>/dynamic_geometry_foam_cuda"),
                    )
                ]
                if args.dynamic_geometry
                else []
            ),
            command_record(execute_command, ROOT),
        ],
    }


def execute_smoke(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    start = time.perf_counter()
    timeout_s = int(args.max_gpu_minutes) * 60
    workdir = output_dir / "work"
    repos_dir = workdir / "repos"
    dataset_root = workdir / "datasets"
    repos_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    host = preflight()
    if not host["torch_cuda_available"]:
        raise RuntimeError("torch.cuda.is_available() is false; refusing CUDA smoke execution.")

    dataset_summary = export_dataset(
        video_path=Path(args.video),
        output_dir=dataset_root,
        scene_name=str(args.scene_name),
        frame_count=int(args.frames),
        size=int(args.size),
        camera_angle_x=float(args.camera_angle_x),
        camera_motion=str(args.camera_motion),
        camera_radius=float(args.camera_radius),
        overwrite=True,
    )

    static_repo = repos_dir / "official_static_cuda"
    feature_repo = repos_dir / "dynamic_feature_foam_cuda"
    geometry_repo = repos_dir / "dynamic_geometry_foam_cuda"
    clone_info = clone_official_repo(
        repo_url=str(args.official_repo_url),
        commit=str(args.official_commit),
        dest=static_repo,
        timeout_s=timeout_s,
    )
    feature_info = copy_dynamic_fork(static_repo, feature_repo, patch_kind="feature")
    geometry_info = (
        copy_dynamic_fork(static_repo, geometry_repo, patch_kind="geometry")
        if bool(args.dynamic_geometry)
        else None
    )

    fixture: dict[str, Any] | None = None
    if not bool(args.skip_official_fixture):
        try:
            fixture = run_official_fixture(upstream_root=static_repo, output_dir=output_dir, timeout_s=timeout_s)
        except Exception as exc:
            fixture = {
                "name": "official_cuda_warp_fixture",
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }

    common = {
        "dataset_root": dataset_root,
        "scene_name": args.scene_name,
        "iterations": int(args.iterations),
        "points": int(args.points),
        "num_texel_sites": int(args.num_texel_sites),
        "sv_dof": int(args.sv_dof),
        "seed": int(args.seed),
        "random_background": not bool(args.fixed_black_background),
        "dynamic_time_basis_count": int(args.dynamic_time_basis_count),
        "dynamic_center_basis_count": int(args.dynamic_center_basis_count),
        "dynamic_height_basis_count": int(args.dynamic_height_basis_count),
    }
    run_specs = [
        (
            static_repo,
            base_smoke_settings(run_name="official_static_cuda", dynamic=False, dynamic_patch_kind="feature", **common),
        ),
        (
            feature_repo,
            base_smoke_settings(run_name="dynamic_feature_foam_cuda", dynamic=True, dynamic_patch_kind="feature", **common),
        ),
    ]
    if bool(args.dynamic_geometry):
        run_specs.append(
            (
                geometry_repo,
                base_smoke_settings(
                    run_name="dynamic_geometry_foam_cuda",
                    dynamic=True,
                    dynamic_patch_kind="geometry",
                    **common,
                ),
            )
        )
    runs = []
    for repo, settings in run_specs:
        try:
            runs.append(run_lane(repo=repo, settings=settings, output_dir=output_dir, timeout_s=timeout_s))
        except Exception as exc:
            runs.append({"name": settings["run_name"], "status": "failed", "error": f"{type(exc).__name__}: {exc}"})

    summary = {
        "schema_version": "powerfoam_cuda_smoke_v1",
        "status": "ok"
        if all(run.get("status") == "ok" for run in runs)
        and (fixture is None or fixture.get("status") == "ok")
        else "failed",
        "run_id": args.run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "official_repo_url": args.official_repo_url,
            "official_commit": args.official_commit,
            "dynamic_patch": rel(DYNAMIC_FEATURE_PATCH),
            "dynamic_patch_sha256": feature_info["patch_sha256"],
            "dynamic_feature_patch": rel(DYNAMIC_FEATURE_PATCH),
            "dynamic_feature_patch_sha256": feature_info["patch_sha256"],
            "dynamic_geometry_patch": rel(DYNAMIC_GEOMETRY_PATCH),
            "dynamic_geometry_patch_sha256": None if geometry_info is None else geometry_info["patch_sha256"],
            "dynaworld_root": str(ROOT),
        },
        "host": host,
        "clip": {
            "path": rel(Path(args.video)),
            "sha256": sha256_file(Path(args.video)),
            "frames": int(args.frames),
            "size": int(args.size),
            "dataset": dataset_summary,
        },
        "settings": {
            "iterations": int(args.iterations),
            "points": int(args.points),
            "num_texel_sites": int(args.num_texel_sites),
            "sv_dof": int(args.sv_dof),
            "gpu": args.gpu,
            "max_gpu_minutes": int(args.max_gpu_minutes),
            "seed": int(args.seed),
            "dynamic_geometry": bool(args.dynamic_geometry),
            "dynamic_time_basis_count": int(args.dynamic_time_basis_count),
            "dynamic_center_basis_count": int(args.dynamic_center_basis_count),
            "dynamic_height_basis_count": int(args.dynamic_height_basis_count),
            "skip_official_fixture": bool(args.skip_official_fixture),
            "fixed_black_background": bool(args.fixed_black_background),
            "random_background": not bool(args.fixed_black_background),
            "output_dir": str(output_dir),
        },
        "official_clone": clone_info,
        "dynamic_fork": feature_info,
        "dynamic_feature_fork": feature_info,
        "dynamic_geometry_fork": geometry_info,
        "official_fixture": fixture,
        "runs": runs,
        "comparisons": compare_runs(runs),
        "elapsed_s": time.perf_counter() - start,
    }
    return summary


def write_summary(summary: dict[str, Any], output_dir: Path) -> Path:
    return write_report_json(output_dir / "summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or plan the official CUDA PowerFoam same-clip smoke.")
    parser.add_argument("--execute", action="store_true", help="Actually run CUDA work. Default only writes a plan.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-id", default=now_run_id())
    parser.add_argument("--official-repo-url", default=UPSTREAM_REPO_URL)
    parser.add_argument("--official-commit", default=UPSTREAM_COMMIT)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--scene-name", default="dynaworld_tiny_clip")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--camera-angle-x", type=float, default=0.75)
    parser.add_argument("--camera-motion", choices=["static", "tiny_orbit"], default="static")
    parser.add_argument("--camera-radius", type=float, default=2.0)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--points", type=int, default=512)
    parser.add_argument("--num-texel-sites", type=int, default=4)
    parser.add_argument("--sv-dof", type=int, default=4)
    parser.add_argument("--dynamic-patch-kind", choices=sorted(DYNAMIC_PATCHES), default="feature")
    parser.add_argument("--dynamic-time-basis-count", type=int, default=4)
    parser.add_argument("--dynamic-center-basis-count", type=int, default=4)
    parser.add_argument("--dynamic-height-basis-count", type=int, default=4)
    parser.add_argument("--dynamic-geometry", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--gpu", default="L40S")
    parser.add_argument("--max-gpu-minutes", type=int, default=20)
    parser.add_argument("--skip-official-fixture", action="store_true")
    parser.add_argument("--fixed-black-background", action="store_true")
    args = parser.parse_args()
    if args.dynamic_patch_kind == "geometry":
        args.dynamic_geometry = True

    output_dir = args.output_dir or ROOT / "outputs/powerfoam_cuda_smokes" / str(args.run_id)
    try:
        summary = execute_smoke(args, output_dir) if args.execute else planned_summary(args, output_dir)
        path = write_summary(summary, output_dir)
        print(json.dumps({"summary": str(path), "status": summary["status"]}, indent=2, sort_keys=True))
        if summary["status"] == "failed":
            raise SystemExit(1)
    except Exception as exc:
        failed = {
            "schema_version": "powerfoam_cuda_smoke_v1",
            "status": "failed",
            "run_id": args.run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": f"{type(exc).__name__}: {exc}",
        }
        path = write_summary(failed, output_dir)
        print(json.dumps({"summary": str(path), "status": "failed", "error": failed["error"]}, indent=2, sort_keys=True))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
