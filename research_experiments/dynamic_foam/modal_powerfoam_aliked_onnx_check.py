from __future__ import annotations

import json
import subprocess
import sys
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

APP_NAME = "dynaworld-powerfoam-aliked-onnx-check"


def repo_root() -> Path:
    current = Path(__file__).resolve()
    marker = Path("research_experiments/dynamic_foam/modal_powerfoam_aliked_onnx_check.py")
    for parent in current.parents:
        if (parent / marker).exists():
            return parent
    return Path("/root/dynaworld")


ROOT = repo_root()

CHECK_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "libgomp1", "libsm6", "libxext6")
    .pip_install("pycolmap-cuda12==4.0.4", "numpy==2.1.2", "pillow==11.0.0")
)

app = modal.App(APP_NAME)


@app.function(image=CHECK_IMAGE, gpu="L40S", timeout=60 * 10)
def check_remote_aliked_onnx() -> dict[str, Any]:
    script = r'''
import json
import tempfile
from pathlib import Path

import numpy as np
import pycolmap
from PIL import Image

required = [
    "FeatureExtractionOptions",
    "FeatureExtractorType",
    "ImageReaderOptions",
    "CameraMode",
    "Device",
    "extract_features",
]
missing = [name for name in required if not hasattr(pycolmap, name)]
if missing:
    raise RuntimeError(
        "pycolmap wheel is missing required builder API: "
        + ", ".join(missing)
        + f"; version={getattr(pycolmap, '__version__', 'unknown')}"
    )

workdir = Path(tempfile.mkdtemp(prefix="pycolmap_aliked_onnx_probe_"))
images_dir = workdir / "images"
images_dir.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(17)
for index in range(2):
    image = (rng.random((128, 128, 3)) * 255.0).astype(np.uint8)
    Image.fromarray(image).save(images_dir / f"probe_{index}.png")

reader = pycolmap.ImageReaderOptions()
reader.camera_model = "PINHOLE"
reader.camera_params = "128,128,64,64"
extraction = pycolmap.FeatureExtractionOptions()
extraction.type = pycolmap.FeatureExtractorType.ALIKED_N16ROT
extraction.use_gpu = True
extraction.max_image_size = 128
device = getattr(pycolmap.Device, "cuda", pycolmap.Device.auto)
pycolmap.extract_features(
    workdir / "database.db",
    images_dir,
    image_names=["probe_0.png", "probe_1.png"],
    camera_mode=pycolmap.CameraMode.PER_IMAGE,
    reader_options=reader,
    extraction_options=extraction,
    device=device,
)
print(json.dumps({
    "pycolmap_version": getattr(pycolmap, "__version__", "unknown"),
    "pycolmap_package": "pycolmap-cuda12==4.0.4",
    "feature_type": "ALIKED_N16ROT",
    "device": str(device),
    "workdir": str(workdir),
}))
'''
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd="/tmp",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60 * 8,
        check=False,
    )
    return {
        "ok": completed.returncode == 0,
        "return_code": completed.returncode,
        "stdout_tail": completed.stdout[-8000:],
    }


@app.local_entrypoint()
def main(run_id: str = "latest") -> None:
    output_dir = ROOT / "outputs/powerfoam_aliked_geometry" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    result = check_remote_aliked_onnx.remote()
    output = output_dir / "onnx_check.json"
    write_report_json(output, result)
    print(json.dumps({"onnx_check_ok": result["ok"], "output": str(output)}, indent=2))
    if not result["ok"]:
        raise SystemExit(1)
