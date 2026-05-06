from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

APP_NAME = "dynaworld-powerfoam-aliked-geometry"
SAMPLE_ID = "deepview_03_Dog_camera_0001_to_camera_0040"
HELDOUT_CAMERA = "camera_0040"
ANCHOR_CAMERA = "camera_0001"
TRAIN_CAMERAS = [
    "camera_0001",
    "camera_0012",
    "camera_0002",
    "camera_0003",
    "camera_0015",
    "camera_0021",
    "camera_0013",
    "camera_0010",
]
PROBE_TRAIN_CAMERAS = ["camera_0001", "camera_0015"]
PROBE_NEAR_TRAIN_CAMERAS = ["camera_0001", "camera_0012", "camera_0002", "camera_0003"]
MATCHER_CHOICES = {"aliked_bruteforce", "aliked_lightglue"}
PROBE_CAMERA_SET_CHOICES = {"wide2", "near4", "full8"}


def repo_root() -> Path:
    current = Path(__file__).resolve()
    marker = Path("research_experiments/dynamic_foam/build_pycolmap_known_pose_point_cloud.py")
    for parent in current.parents:
        if (parent / marker).exists():
            return parent
    modal_root = Path("/root/dynaworld")
    if (modal_root / marker).exists():
        return modal_root
    if current.is_relative_to(Path("/root")):
        return modal_root
    raise RuntimeError(f"Could not find dynaworld repo root from {current}")


ROOT = repo_root()
REMOTE_ROOT = Path("/root/dynaworld")
BASE_CONFIG = (
    ROOT
    / "src/train_configs/"
    "local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux.jsonc"
)
REMOTE_BASE_CONFIG = REMOTE_ROOT / BASE_CONFIG.relative_to(ROOT)
MANIFEST = ROOT / "data/multicam_val/clip_sets/multicam_val_v1_128_4fps_16f/manifest.jsonl"
REMOTE_MANIFEST = REMOTE_ROOT / MANIFEST.relative_to(ROOT)
SCENE_DIR = ROOT / "data/external/deepview_video/extracted/03_Dog/03_Dog"
REMOTE_SCENE_DIR = REMOTE_ROOT / SCENE_DIR.relative_to(ROOT)
BUILDER = REMOTE_ROOT / "research_experiments/dynamic_foam/build_pycolmap_known_pose_point_cloud.py"


def default_aliked_output(matcher_type: str) -> Path:
    return (
        ROOT
        / "research_experiments/dynamic_foam/artifacts/"
        f"deepview_03_dog_8cam_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_aliked_n16rot_{matcher_type}_minucam2.ply"
    )


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def add_scene_files(image: modal.Image, camera_names: list[str]) -> modal.Image:
    image = image.add_local_file(str(SCENE_DIR / "models.json"), str(REMOTE_SCENE_DIR / "models.json"))
    for camera_name in sorted(set(camera_names)):
        image = image.add_local_file(str(SCENE_DIR / f"{camera_name}.mp4"), str(REMOTE_SCENE_DIR / f"{camera_name}.mp4"))
    return image


CHECK_IMAGE = (
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

GEOMETRY_IMAGE = (
    modal.Image.from_registry("colmap/colmap:latest", add_python="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch",
        "numpy==2.1.2",
        "pillow==11.0.0",
        "pycolmap==4.0.4",
        "opencv-python-headless",
        "tqdm==4.67.1",
        "wandb==0.21.3",
        "nvidia-cudnn-cu12==9.10.2.21",
    )
    .env(
        {
            "LD_LIBRARY_PATH": (
                "/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:"
                "/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64"
            )
        }
    )
    .workdir(str(REMOTE_ROOT))
    .add_local_dir("src/train", str(REMOTE_ROOT / "src/train"), ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_dir(
        "research_experiments/dynamic_foam",
        str(REMOTE_ROOT / "research_experiments/dynamic_foam"),
        ignore=["artifacts/**", "**/__pycache__/**", "**/*.pyc"],
    )
    .add_local_file(str(BASE_CONFIG), str(REMOTE_BASE_CONFIG))
    .add_local_file(str(MANIFEST), str(REMOTE_MANIFEST))
)
PROBE_IMAGE = add_scene_files(GEOMETRY_IMAGE, [*TRAIN_CAMERAS, HELDOUT_CAMERA])
FULL_IMAGE = add_scene_files(GEOMETRY_IMAGE, [*TRAIN_CAMERAS, HELDOUT_CAMERA])

app = modal.App(APP_NAME)


def run_command(command: list[str], *, cwd: Path, timeout_s: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
        check=False,
    )


def load_manifest_record(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if str(record.get("sample_id")) == SAMPLE_ID:
                return record
    raise KeyError(f"No manifest record with sample_id={SAMPLE_ID!r} in {path}")


def write_remote_inputs(*, run_dir: Path, train_cameras: list[str], heldout_camera: str) -> Path:
    sys.path.insert(0, str(REMOTE_ROOT / "src/train"))
    from config_utils import load_config_file

    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.jsonl"
    config_path = run_dir / "config.jsonc"
    record = load_manifest_record(REMOTE_MANIFEST)
    record["dataset_scene_dir"] = str(REMOTE_SCENE_DIR)
    record["models_path"] = str(REMOTE_SCENE_DIR / "models.json")
    record["source_video_path"] = str(REMOTE_SCENE_DIR / f"{record['source_camera']}.mp4")
    record["target_video_path"] = str(REMOTE_SCENE_DIR / f"{heldout_camera}.mp4")
    manifest_path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")

    cfg = load_config_file(REMOTE_BASE_CONFIG)
    cfg["data"]["multicam_manifest"] = str(manifest_path)
    cfg["data"]["multicam_sample_id"] = SAMPLE_ID
    cfg["data"]["multicam_split"] = "val"
    cfg["data"]["multicam_train_cameras"] = list(train_cameras)
    cfg["data"]["multicam_heldout_camera"] = str(heldout_camera)
    cfg["data"]["multicam_heldout_cameras"] = None
    cfg["data"]["multicam_anchor_camera"] = ANCHOR_CAMERA
    cfg["data"]["multicam_condition_camera"] = ANCHOR_CAMERA
    config_path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return config_path


def builder_command(
    *,
    config_path: Path,
    output_path: Path,
    train_cameras: list[str],
    heldout_camera: str,
    matcher_type: str,
    target_size: int,
    max_features: int,
    frame_indices: list[int],
    known_pose_guided_verification: bool,
) -> list[str]:
    frame_args = ["--frame-index", str(frame_indices[0])] if len(frame_indices) == 1 else [
        "--frame-indices",
        *(str(index) for index in frame_indices),
    ]
    command = [
        sys.executable,
        str(BUILDER),
        str(config_path),
        "--output",
        str(output_path),
        "--target-size",
        str(target_size),
        *frame_args,
        "--train-cameras",
        *train_cameras,
        "--heldout-camera",
        heldout_camera,
        "--anchor-camera",
        ANCHOR_CAMERA,
        "--condition-camera",
        ANCHOR_CAMERA,
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
    ]
    if known_pose_guided_verification:
        command.append("--known-pose-guided-verification")
    return command


def probe_train_cameras(camera_set: str) -> list[str]:
    camera_set = camera_set.lower()
    if camera_set == "wide2":
        return list(PROBE_TRAIN_CAMERAS)
    if camera_set == "near4":
        return list(PROBE_NEAR_TRAIN_CAMERAS)
    if camera_set == "full8":
        return list(TRAIN_CAMERAS)
    raise ValueError(f"Unknown probe camera set {camera_set!r}; choices: {sorted(PROBE_CAMERA_SET_CHOICES)}")


def parse_frame_indices(raw: str) -> list[int]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("At least one frame index is required.")
    return [int(value) for value in values]


def parse_bool(raw: str | bool) -> bool:
    if isinstance(raw, bool):
        return raw
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected a boolean-like value, got {raw!r}.")


def collect_text_files(files: dict[str, Path], *, max_bytes: int = 8_000_000) -> dict[str, str]:
    collected: dict[str, str] = {}
    for name, path in files.items():
        if path.exists() and path.stat().st_size <= max_bytes:
            collected[name] = path.read_text(encoding="utf-8")
    return collected


@app.function(image=CHECK_IMAGE, gpu="L40S", timeout=60 * 10)
def check_remote_aliked_onnx() -> dict[str, Any]:
    script = r'''
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

def run(command, cwd):
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60 * 4,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(json.dumps({
            "command": command,
            "return_code": completed.returncode,
            "stdout_tail": completed.stdout[-8000:],
        }, indent=2))
    return completed.stdout[-8000:]

workdir = Path(tempfile.mkdtemp(prefix="colmap_cli_aliked_onnx_probe_"))
images_dir = workdir / "images"
images_dir.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(17)
for index in range(2):
    image = (rng.random((128, 128, 3)) * 255.0).astype(np.uint8)
    Image.fromarray(image).save(images_dir / f"probe_{index}.png")

database_path = workdir / "database.db"
feature_stdout = run([
    "colmap", "feature_extractor",
    "--database_path", str(database_path),
    "--image_path", str(images_dir),
    "--ImageReader.camera_model", "PINHOLE",
    "--ImageReader.camera_params", "128,128,64,64",
    "--ImageReader.single_camera", "0",
    "--FeatureExtraction.type", "ALIKED_N16ROT",
    "--FeatureExtraction.use_gpu", "1",
    "--AlikedExtraction.max_num_features", "256",
], workdir)
match_stdout = run([
    "colmap", "exhaustive_matcher",
    "--database_path", str(database_path),
    "--FeatureMatching.type", "ALIKED_LIGHTGLUE",
    "--FeatureMatching.use_gpu", "1",
], workdir)
print(json.dumps({
    "colmap_image": "colmap/colmap:latest",
    "cudnn_package": "nvidia-cudnn-cu12==9.10.2.21",
    "feature_stdout_tail": feature_stdout,
    "match_stdout_tail": match_stdout,
    "workdir": str(workdir),
}))
'''
    completed = run_command([sys.executable, "-c", script], cwd=Path("/tmp"), timeout_s=60 * 8)
    return {
        "ok": completed.returncode == 0,
        "return_code": completed.returncode,
        "stdout_tail": completed.stdout[-8000:],
    }


@app.function(image=PROBE_IMAGE, gpu="L40S", timeout=60 * 20)
def run_remote_probe(payload: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(payload["run_dir"])
    output_path = run_dir / "probe.ply"
    train_cameras = probe_train_cameras(str(payload.get("probe_camera_set", "wide2")))
    config_path = write_remote_inputs(
        run_dir=run_dir,
        train_cameras=train_cameras,
        heldout_camera=HELDOUT_CAMERA,
    )
    command = builder_command(
        config_path=config_path,
        output_path=output_path,
        train_cameras=train_cameras,
        heldout_camera=HELDOUT_CAMERA,
        matcher_type=str(payload.get("probe_matcher_type", "aliked_bruteforce")),
        target_size=int(payload.get("probe_target_size", 128)),
        max_features=int(payload.get("probe_max_features", 500)),
        frame_indices=[int(index) for index in payload.get("probe_frame_indices", [0])],
        known_pose_guided_verification=bool(payload.get("known_pose_guided_verification", False)),
    )
    completed = run_command(command, cwd=REMOTE_ROOT, timeout_s=60 * 18)
    return {
        "ok": completed.returncode == 0,
        "return_code": completed.returncode,
        "command": command,
        "stdout_tail": completed.stdout[-12000:],
        "files": collect_text_files({"probe.ply": output_path, "probe.json": output_path.with_suffix(".json")}),
    }


@app.function(image=FULL_IMAGE, gpu="L40S", timeout=60 * 60)
def run_remote_full(payload: dict[str, Any]) -> dict[str, Any]:
    matcher_type = str(payload["matcher_type"])
    remote_output = REMOTE_ROOT / Path(payload["relative_output"])
    config_path = write_remote_inputs(
        run_dir=Path(payload["run_dir"]),
        train_cameras=TRAIN_CAMERAS,
        heldout_camera=HELDOUT_CAMERA,
    )
    command = builder_command(
        config_path=config_path,
        output_path=remote_output,
        train_cameras=TRAIN_CAMERAS,
        heldout_camera=HELDOUT_CAMERA,
        matcher_type=matcher_type,
        target_size=1024,
        max_features=int(payload["max_features"]),
        frame_indices=[0, 4, 8, 12],
        known_pose_guided_verification=bool(payload.get("known_pose_guided_verification", False)),
    )
    completed = run_command(command, cwd=REMOTE_ROOT, timeout_s=60 * 55)
    return {
        "ok": completed.returncode == 0,
        "return_code": completed.returncode,
        "command": command,
        "stdout_tail": completed.stdout[-16000:],
        "files": collect_text_files(
            {
                str(Path(payload["relative_output"])): remote_output,
                str(Path(payload["relative_output"]).with_suffix(".json")): remote_output.with_suffix(".json"),
            }
        ),
    }


def write_local_result(result: dict[str, Any], output_dir: Path, name: str, *, copy_artifact: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{name}.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    returned = output_dir / "returned_files"
    for relative, text in result.get("files", {}).items():
        if relative.endswith(".json") and text.lstrip().startswith("{"):
            payload = json.loads(text)
            remote_prefix = str(REMOTE_ROOT) + "/"
            if isinstance(payload.get("output"), str) and payload["output"].startswith(remote_prefix):
                payload["output"] = payload["output"][len(remote_prefix) :]
                text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        destination = returned / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
        if copy_artifact and relative.startswith("research_experiments/"):
            canonical = ROOT / relative
            canonical.parent.mkdir(parents=True, exist_ok=True)
            canonical.write_text(text, encoding="utf-8")


def write_plan(
    output_dir: Path,
    *,
    matcher_type: str,
    max_features: int,
    probe_camera_set: str,
    probe_matcher_type: str,
    probe_target_size: int,
    probe_max_features: int,
    probe_frame_indices: list[int],
    known_pose_guided_verification: bool,
) -> None:
    probe_cameras = probe_train_cameras(probe_camera_set)
    plan = {
        "schema_version": "powerfoam_aliked_geometry_modal_plan_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "app": APP_NAME,
        "sample_id": SAMPLE_ID,
        "check": "remote no-data COLMAP CLI ALIKED_N16ROT + ALIKED_LIGHTGLUE ONNX probe",
        "colmap_image": "colmap/colmap:latest",
        "cudnn_package": "nvidia-cudnn-cu12==9.10.2.21",
        "builder_feature_backend": "colmap_cli",
        "gpu": "L40S",
        "probe": {
            "train_cameras": probe_cameras,
            "camera_set": probe_camera_set,
            "heldout_camera": HELDOUT_CAMERA,
            "target_size": int(probe_target_size),
            "frame_indices": probe_frame_indices,
            "matcher_type": probe_matcher_type,
            "max_features": int(probe_max_features),
            "known_pose_guided_verification": bool(known_pose_guided_verification),
        },
        "full": {
            "train_cameras": TRAIN_CAMERAS,
            "heldout_camera": HELDOUT_CAMERA,
            "target_size": 1024,
            "frame_indices": [0, 4, 8, 12],
            "matcher_type": matcher_type,
            "max_features": int(max_features),
            "known_pose_guided_verification": bool(known_pose_guided_verification),
            "output": rel(default_aliked_output(matcher_type)),
        },
        "modal_commands": [
            "modal run research_experiments/dynamic_foam/modal_powerfoam_aliked_onnx_check.py --run-id latest",
            "modal run research_experiments/dynamic_foam/modal_powerfoam_aliked_geometry.py --execute --check-onnx --probe",
            (
                "modal run research_experiments/dynamic_foam/modal_powerfoam_aliked_geometry.py "
                f"--execute --check-onnx --full --matcher-type {matcher_type}"
            ),
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@app.local_entrypoint()
def main(
    execute: bool = False,
    run_id: str = "latest",
    check_onnx: bool = True,
    probe: bool = False,
    full: bool = False,
    matcher_type: str = "aliked_lightglue",
    max_features: int = 12000,
    probe_camera_set: str = "wide2",
    probe_matcher_type: str = "aliked_bruteforce",
    probe_target_size: int = 128,
    probe_max_features: int = 500,
    probe_frame_indices: str = "0",
    known_pose_guided_verification: str = "false",
    copy_artifact: bool = True,
) -> None:
    matcher_type = matcher_type.lower()
    if matcher_type not in MATCHER_CHOICES:
        raise ValueError(f"matcher_type must be one of {sorted(MATCHER_CHOICES)}, got {matcher_type!r}.")
    probe_matcher_type = probe_matcher_type.lower()
    if probe_matcher_type not in MATCHER_CHOICES:
        raise ValueError(f"probe_matcher_type must be one of {sorted(MATCHER_CHOICES)}, got {probe_matcher_type!r}.")
    probe_camera_set = probe_camera_set.lower()
    if probe_camera_set not in PROBE_CAMERA_SET_CHOICES:
        raise ValueError(f"probe_camera_set must be one of {sorted(PROBE_CAMERA_SET_CHOICES)}, got {probe_camera_set!r}.")
    parsed_probe_frame_indices = parse_frame_indices(probe_frame_indices)
    guided_verification = parse_bool(known_pose_guided_verification)
    output_dir = ROOT / "outputs/powerfoam_aliked_geometry" / run_id
    if not execute:
        write_plan(
            output_dir,
            matcher_type=matcher_type,
            max_features=max_features,
            probe_camera_set=probe_camera_set,
            probe_matcher_type=probe_matcher_type,
            probe_target_size=probe_target_size,
            probe_max_features=probe_max_features,
            probe_frame_indices=parsed_probe_frame_indices,
            known_pose_guided_verification=guided_verification,
        )
        print(json.dumps({"planned": str(output_dir / "plan.json")}, indent=2))
        return

    if check_onnx:
        check = check_remote_aliked_onnx.remote()
        write_local_result(check, output_dir, "onnx_check", copy_artifact=False)
        print(json.dumps({"onnx_check_ok": check["ok"], "output": str(output_dir / "onnx_check.json")}, indent=2))
        if not check["ok"]:
            raise SystemExit(1)
    if probe:
        probe_result = run_remote_probe.remote(
            {
                "run_dir": f"/tmp/powerfoam_aliked_geometry/{run_id}/probe",
                "probe_camera_set": probe_camera_set,
                "probe_matcher_type": probe_matcher_type,
                "probe_target_size": int(probe_target_size),
                "probe_max_features": int(probe_max_features),
                "probe_frame_indices": parsed_probe_frame_indices,
                "known_pose_guided_verification": guided_verification,
            }
        )
        write_local_result(probe_result, output_dir, "probe", copy_artifact=False)
        print(json.dumps({"probe_ok": probe_result["ok"], "output": str(output_dir / "probe.json")}, indent=2))
        if not probe_result["ok"]:
            raise SystemExit(1)
    if full:
        relative_output = rel(default_aliked_output(matcher_type))
        full_result = run_remote_full.remote(
            {
                "run_dir": f"/tmp/powerfoam_aliked_geometry/{run_id}/full",
                "matcher_type": matcher_type,
                "relative_output": relative_output,
                "max_features": int(max_features),
                "known_pose_guided_verification": guided_verification,
            }
        )
        write_local_result(full_result, output_dir, "full", copy_artifact=copy_artifact)
        print(json.dumps({"full_ok": full_result["ok"], "output": str(output_dir / "full.json")}, indent=2))
        if not full_result["ok"]:
            raise SystemExit(1)
