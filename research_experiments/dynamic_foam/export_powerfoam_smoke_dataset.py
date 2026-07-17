from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

try:
    from .report_artifacts import PROJECT_ROOT, relative_to_project as rel, write_report_json
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import PROJECT_ROOT, relative_to_project as rel, write_report_json

ROOT = PROJECT_ROOT
BLENDER_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_indices(total_frames: int, frame_count: int) -> list[int]:
    if total_frames <= 0:
        raise ValueError("Video reported no frames.")
    if frame_count <= 0:
        raise ValueError("frame_count must be positive.")
    if frame_count == 1:
        return [0]
    count = min(frame_count, total_frames)
    return [int(round(v)) for v in np.linspace(0, total_frames - 1, count)]


def read_video_frames(video_path: Path, *, frame_count: int, size: int) -> tuple[list[Image.Image], list[int], float]:
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    indices = selected_indices(total, frame_count)
    frames: list[Image.Image] = []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, bgr = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {index} from {video_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        if image.size != (size, size):
            image = image.resize((size, size), Image.Resampling.LANCZOS)
        alpha = Image.new("L", image.size, 255)
        frames.append(Image.merge("RGBA", (*image.split(), alpha)))
    cap.release()
    return frames, indices, fps


def normalized(values: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(values))
    if norm < 1.0e-8:
        raise ValueError(f"Cannot normalize near-zero vector {values.tolist()}")
    return values / norm


def look_at_c2w_opencv(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = normalized(target - eye)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(forward, world_up))) > 0.98:
        world_up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    right = normalized(np.cross(world_up, forward))
    down = normalized(np.cross(forward, right))
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = eye
    return c2w


def frame_camera_transform(frame: int, frame_count: int, *, motion: str, radius: float) -> list[list[float]]:
    target = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    if motion == "static":
        eye = np.array([0.5, 0.5, 0.5 - radius], dtype=np.float32)
    elif motion == "tiny_orbit":
        theta = 2.0 * math.pi * float(frame) / float(max(frame_count, 1))
        eye = np.array(
            [
                0.5 + 0.18 * math.sin(theta),
                0.5 + 0.08 * math.sin(0.5 * theta),
                0.5 - radius + 0.18 * math.cos(theta),
            ],
            dtype=np.float32,
        )
    else:
        raise ValueError(f"Unknown camera motion {motion!r}")
    c2w_opencv = look_at_c2w_opencv(eye, target)
    blender_transform = c2w_opencv @ BLENDER_TO_OPENCV
    return blender_transform.tolist()


def write_transforms(path: Path, frames: list[dict[str, Any]], *, size: int, camera_angle_x: float) -> None:
    payload = {
        "camera_angle_x": float(camera_angle_x),
        "w": int(size),
        "h": int(size),
        "frames": frames,
    }
    write_report_json(path, payload)


def export_dataset(
    *,
    video_path: Path,
    output_dir: Path,
    scene_name: str,
    frame_count: int,
    size: int,
    camera_angle_x: float,
    camera_motion: str,
    camera_radius: float,
    overwrite: bool,
) -> dict[str, Any]:
    scene_dir = output_dir / scene_name
    if scene_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{scene_dir} exists; pass --overwrite to replace it.")
        shutil.rmtree(scene_dir)
    images_dir = scene_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    frames, source_indices, fps = read_video_frames(video_path, frame_count=frame_count, size=size)
    transform_frames: list[dict[str, Any]] = []
    for i, image in enumerate(frames):
        stem = f"{i:06d}"
        image.save(images_dir / f"{stem}.png")
        transform_frames.append(
            {
                "file_path": f"images/{stem}",
                "source_frame_index": int(source_indices[i]),
                "time_index": 0.0 if len(frames) == 1 else float(i) / float(len(frames) - 1),
                "transform_matrix": frame_camera_transform(
                    i,
                    len(frames),
                    motion=camera_motion,
                    radius=camera_radius,
                ),
            }
        )

    write_transforms(scene_dir / "transforms_all.json", transform_frames, size=size, camera_angle_x=camera_angle_x)
    write_transforms(scene_dir / "transforms_train.json", transform_frames, size=size, camera_angle_x=camera_angle_x)
    write_transforms(
        scene_dir / "transforms_test.json",
        transform_frames[-1:],
        size=size,
        camera_angle_x=camera_angle_x,
    )
    summary = {
        "schema_version": "powerfoam_smoke_dataset_v1",
        "scene_dir": rel(scene_dir),
        "scene_name": scene_name,
        "video_path": rel(video_path),
        "video_sha256": sha256_file(video_path),
        "source_fps": fps,
        "source_frame_indices": source_indices,
        "frames": len(frames),
        "size": int(size),
        "camera_angle_x": float(camera_angle_x),
        "camera_motion": camera_motion,
        "camera_radius": float(camera_radius),
        "alpha_format_on_disk": "straight",
        "official_powerfoam_dataset": "blender",
    }
    write_report_json(scene_dir / "dynaworld_smoke_dataset.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a tiny video clip as an official-PowerFoam Blender dataset.")
    parser.add_argument("--video", type=Path, default=ROOT / "test_data/test_video_small_128_4fps.mp4")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene-name", default="dynaworld_tiny_clip")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--camera-angle-x", type=float, default=0.75)
    parser.add_argument("--camera-motion", choices=["static", "tiny_orbit"], default="static")
    parser.add_argument("--camera-radius", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = export_dataset(
        video_path=args.video,
        output_dir=args.output_dir,
        scene_name=str(args.scene_name),
        frame_count=int(args.frames),
        size=int(args.size),
        camera_angle_x=float(args.camera_angle_x),
        camera_motion=str(args.camera_motion),
        camera_radius=float(args.camera_radius),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
