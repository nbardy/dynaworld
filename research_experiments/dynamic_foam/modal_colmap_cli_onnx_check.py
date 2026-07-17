from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import modal

try:
    from .report_artifacts import write_report_json
except ImportError:  # pragma: no cover - direct script execution/importlib loading
    dynamic_foam_dir = Path(__file__).resolve().parent
    if str(dynamic_foam_dir) not in sys.path:
        sys.path.insert(0, str(dynamic_foam_dir))
    from report_artifacts import write_report_json

APP_NAME = "dynaworld-colmap-cli-onnx-check"


def repo_root() -> Path:
    current = Path(__file__).resolve()
    marker = Path("research_experiments/dynamic_foam/modal_colmap_cli_onnx_check.py")
    for parent in current.parents:
        if (parent / marker).exists():
            return parent
    return Path("/root/dynaworld")


ROOT = repo_root()

COLMAP_CLI_IMAGE = (
    modal.Image.from_registry("colmap/colmap:latest", add_python="3.11")
    .pip_install("numpy==2.1.2", "pillow==11.0.0", "nvidia-cudnn-cu12==9.10.2.21")
    .env(
        {
            "LD_LIBRARY_PATH": (
                "/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:"
                "/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64"
            )
        }
    )
)

app = modal.App(APP_NAME)


def command_env() -> dict[str, str]:
    env = os.environ.copy()
    cudnn_lib = Path("/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib")
    if cudnn_lib.exists():
        env["LD_LIBRARY_PATH"] = f"{cudnn_lib}:{env.get('LD_LIBRARY_PATH', '')}"
    return env


def run_command(command: list[str], *, cwd: Path, timeout_s: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=command_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
        check=False,
    )


@app.function(image=COLMAP_CLI_IMAGE, gpu="L40S", timeout=60 * 10)
def check_remote_colmap_cli_onnx() -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    workdir = Path(tempfile.mkdtemp(prefix="colmap_cli_aliked_onnx_probe_"))
    images_dir = workdir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(23)
    for index in range(2):
        image = (rng.random((128, 128, 3)) * 255.0).astype(np.uint8)
        Image.fromarray(image).save(images_dir / f"probe_{index}.png")

    database_path = workdir / "database.db"
    commands = {
        "colmap_help": ["colmap", "-h"],
        "feature_extractor": [
            "colmap",
            "feature_extractor",
            "--database_path",
            str(database_path),
            "--image_path",
            str(images_dir),
            "--ImageReader.camera_model",
            "PINHOLE",
            "--ImageReader.camera_params",
            "128,128,64,64",
            "--ImageReader.single_camera",
            "0",
            "--FeatureExtraction.type",
            "ALIKED_N16ROT",
            "--FeatureExtraction.use_gpu",
            "1",
            "--AlikedExtraction.max_num_features",
            "256",
        ],
        "exhaustive_matcher": [
            "colmap",
            "exhaustive_matcher",
            "--database_path",
            str(database_path),
            "--FeatureMatching.type",
            "ALIKED_LIGHTGLUE",
            "--FeatureMatching.use_gpu",
            "1",
        ],
    }
    results: dict[str, Any] = {}
    ok = True
    for name, command in commands.items():
        completed = run_command(command, cwd=workdir, timeout_s=60 * 4)
        results[name] = {
            "command": command,
            "return_code": completed.returncode,
            "stdout_tail": completed.stdout[-8000:],
        }
        if completed.returncode != 0:
            ok = False
            break
    return {
        "ok": ok,
        "image": "colmap/colmap:latest",
        "gpu": "L40S",
        "cudnn_package": "nvidia-cudnn-cu12==9.10.2.21",
        "workdir": str(workdir),
        "results": results,
    }


@app.local_entrypoint()
def main(run_id: str = "latest") -> None:
    output_dir = ROOT / "outputs/powerfoam_aliked_geometry" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    result = check_remote_colmap_cli_onnx.remote()
    output = output_dir / "colmap_cli_onnx_check.json"
    write_report_json(output, result)
    print(json.dumps({"colmap_cli_onnx_check_ok": result["ok"], "output": str(output)}, indent=2))
    if not result["ok"]:
        raise SystemExit(1)
