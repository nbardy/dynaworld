from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import modal

APP_NAME = "dynaworld-powerfoam-cuda-smoke"


def repo_root() -> Path:
    current = Path(__file__).resolve()
    marker = Path("research_experiments/dynamic_foam/powerfoam_cuda_smoke_runner.py")
    for parent in current.parents:
        if (parent / marker).exists():
            return parent
    modal_root = Path("/root/dynaworld")
    if (modal_root / marker).exists():
        return modal_root
    raise RuntimeError(f"Could not find dynaworld repo root from {current}")


ROOT = repo_root()
REMOTE_ROOT = Path("/root/dynaworld")
RUNNER = ROOT / "research_experiments/dynamic_foam/powerfoam_cuda_smoke_runner.py"
REMOTE_RUNNER = REMOTE_ROOT / "research_experiments/dynamic_foam/powerfoam_cuda_smoke_runner.py"
DEFAULT_VIDEO = ROOT / "test_data/test_video_small_128_4fps.mp4"
REMOTE_VIDEO = REMOTE_ROOT / "test_data/test_video_small_128_4fps.mp4"


CUDA_SMOKE_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0", "libgomp1")
    .pip_install(
        "torch",
        "torchvision",
        "numpy==2.1.2",
        "scipy==1.16.2",
        "pillow==11.0.0",
        "pyyaml==6.0.2",
        "configargparse==1.7.1",
        "matplotlib==3.10.6",
        "tqdm==4.67.1",
        "einops==0.8.1",
        "plyfile==1.1.2",
        "fpsample==0.3.3",
        "pycolmap==3.12.0",
        "open3d==0.19.0",
        "tensorboard==2.20.0",
        "lpips==0.1.4",
        "warp-lang==1.10.0",
        "opencv-python-headless",
    )
    .workdir(str(REMOTE_ROOT))
    .add_local_dir("src/train", str(REMOTE_ROOT / "src/train"), ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_dir(
        "research_experiments/dynamic_foam",
        str(REMOTE_ROOT / "research_experiments/dynamic_foam"),
        ignore=[
            "artifacts/**",
            "**/__pycache__/**",
            "**/*.pyc",
        ],
    )
    .add_local_file(str(DEFAULT_VIDEO), str(REMOTE_VIDEO))
)


app = modal.App(APP_NAME)


def preset_settings(preset: str) -> dict[str, int]:
    presets = {
        "tiny_clip_128_8f_20step": {
            "frames": 8,
            "size": 128,
            "iterations": 20,
            "points": 512,
            "num_texel_sites": 4,
            "sv_dof": 4,
        },
        "micro_clip_64_4f_5step": {
            "frames": 4,
            "size": 64,
            "iterations": 5,
            "points": 256,
            "num_texel_sites": 4,
            "sv_dof": 2,
        },
    }
    if preset not in presets:
        raise ValueError(f"Unknown preset {preset!r}; choices: {sorted(presets)}")
    return dict(presets[preset])


def collect_small_json_files(root: Path, *, max_bytes: int = 2_000_000) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in root.rglob("*.json"):
        if path.stat().st_size <= max_bytes:
            files[str(path.relative_to(root))] = path.read_text(encoding="utf-8")
    return files


def runner_args(
    *,
    execute: bool,
    output_dir: Path,
    run_id: str,
    preset: str,
    max_gpu_minutes: int,
    skip_official_fixture: bool,
    fixed_black_background: bool,
    dynamic_patch_kind: str = "feature",
    dynamic_geometry: bool = False,
) -> list[str]:
    settings = preset_settings(preset)
    args = [
        str(REMOTE_RUNNER if execute else RUNNER),
        "--run-id",
        run_id,
        "--output-dir",
        str(output_dir),
        "--video",
        str(REMOTE_VIDEO if execute else DEFAULT_VIDEO),
        "--max-gpu-minutes",
        str(max_gpu_minutes),
        "--frames",
        str(settings["frames"]),
        "--size",
        str(settings["size"]),
        "--iterations",
        str(settings["iterations"]),
        "--points",
        str(settings["points"]),
        "--num-texel-sites",
        str(settings["num_texel_sites"]),
        "--sv-dof",
        str(settings["sv_dof"]),
        "--gpu",
        "L40S",
    ]
    if dynamic_patch_kind == "geometry":
        dynamic_geometry = True
    if execute:
        args.append("--execute")
    if skip_official_fixture:
        args.append("--skip-official-fixture")
    if fixed_black_background:
        args.append("--fixed-black-background")
    if dynamic_geometry:
        args.append("--dynamic-geometry")
    return args


@app.function(image=CUDA_SMOKE_IMAGE, gpu="L40S", timeout=60 * 30)
def run_remote_smoke(payload: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(payload["output_dir"])
    command = [sys.executable, *runner_args(execute=True, output_dir=output_dir, **payload["runner"])]
    completed = subprocess.run(
        command,
        cwd=REMOTE_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"runner did not write {summary_path}; output:\n{completed.stdout}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "return_code": completed.returncode,
        "stdout_tail": completed.stdout[-8000:],
        "summary": summary,
        "files": collect_small_json_files(output_dir),
    }


def write_returned_files(result: dict[str, Any], local_output_dir: Path, *, copy_official_fixture: bool) -> None:
    local_output_dir.mkdir(parents=True, exist_ok=True)
    (local_output_dir / "modal_return.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for relative, text in result.get("files", {}).items():
        destination = local_output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
        if copy_official_fixture and relative == "fixtures/powerfoam_tiny_height_sv_official_camera_official_v1.json":
            canonical = ROOT / "research_experiments/dynamic_foam/fixtures/powerfoam_tiny_height_sv_official_camera_official_v1.json"
            canonical.parent.mkdir(parents=True, exist_ok=True)
            canonical.write_text(text, encoding="utf-8")


def run_local_plan(
    *,
    local_output_dir: Path,
    run_id: str,
    preset: str,
    max_gpu_minutes: int,
    skip_official_fixture: bool,
    fixed_black_background: bool,
    dynamic_patch_kind: str,
    dynamic_geometry: bool,
) -> None:
    command = [
        sys.executable,
        *runner_args(
            execute=False,
            output_dir=local_output_dir,
            run_id=run_id,
            preset=preset,
            max_gpu_minutes=max_gpu_minutes,
            skip_official_fixture=skip_official_fixture,
            fixed_black_background=fixed_black_background,
            dynamic_patch_kind=dynamic_patch_kind,
            dynamic_geometry=dynamic_geometry,
        ),
    ]
    subprocess.run(command, cwd=ROOT, check=True)


@app.local_entrypoint()
def main(
    execute: bool = False,
    preset: str = "micro_clip_64_4f_5step",
    run_id: str = "latest",
    max_gpu_minutes: int = 8,
    skip_official_fixture: bool = False,
    fixed_black_background: bool = False,
    dynamic_patch_kind: str = "feature",
    dynamic_geometry: bool = False,
    copy_official_fixture: bool = True,
) -> None:
    if dynamic_patch_kind == "geometry":
        dynamic_geometry = True
    local_output_dir = ROOT / "outputs/powerfoam_cuda_smokes" / run_id
    if not execute:
        run_local_plan(
            local_output_dir=local_output_dir,
            run_id=run_id,
            preset=preset,
            max_gpu_minutes=max_gpu_minutes,
            skip_official_fixture=skip_official_fixture,
            fixed_black_background=fixed_black_background,
            dynamic_patch_kind=dynamic_patch_kind,
            dynamic_geometry=dynamic_geometry,
        )
        print(f"planned: {local_output_dir / 'summary.json'}")
        return

    remote_output_dir = Path("/tmp/powerfoam_cuda_smokes") / run_id
    result = run_remote_smoke.remote(
        {
            "output_dir": str(remote_output_dir),
            "runner": {
                "run_id": run_id,
                "preset": preset,
                "max_gpu_minutes": max_gpu_minutes,
                "skip_official_fixture": skip_official_fixture,
                "fixed_black_background": fixed_black_background,
                "dynamic_patch_kind": dynamic_patch_kind,
                "dynamic_geometry": dynamic_geometry,
            },
        }
    )
    write_returned_files(result, local_output_dir, copy_official_fixture=copy_official_fixture)
    print(json.dumps({"output_dir": str(local_output_dir), "status": result["summary"].get("status")}, indent=2))
    if result.get("return_code") != 0:
        raise SystemExit(int(result["return_code"]))
