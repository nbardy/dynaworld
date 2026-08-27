"""Prepare the official Deep 3D Mask evaluation clip for canonical loading."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np


OFFICIAL_EVAL_ARCHIVE_ID = "1_9KA20cI_0Bs9ERkT65TPtiom3fdQXD0"
NATIVE_FPS = 120.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_directories(source_root: Path) -> list[Path]:
    frame_dirs = sorted(path for path in (source_root / "frames").iterdir() if path.is_dir())
    if not frame_dirs:
        raise ValueError(f"No frame directories found under {source_root / 'frames'}.")
    indices = [int(path.name) for path in frame_dirs]
    if indices != list(range(indices[0], indices[0] + len(indices))):
        raise ValueError("Deep 3D Mask evaluation frames must be one contiguous native-frame interval.")
    for frame_dir in frame_dirs:
        images = sorted((frame_dir / "images").glob("*.jpg"))
        if [path.stem for path in images] != [f"{camera:03d}" for camera in range(10)]:
            raise ValueError(f"Expected cameras 000..009 in {frame_dir / 'images'}.")
    return frame_dirs


def _validated_poses(frame_dirs: list[Path]) -> np.ndarray:
    reference = np.load(frame_dirs[0] / "poses_bounds.npy")
    if reference.shape != (10, 17):
        raise ValueError(f"Expected ten LLFF pose rows, got {reference.shape}.")
    for frame_dir in frame_dirs[1:]:
        actual = np.load(frame_dir / "poses_bounds.npy")
        if actual.shape != reference.shape or not np.allclose(actual, reference, rtol=0.0, atol=1.0e-9):
            raise ValueError(f"Static-rig poses changed at native frame {frame_dir.name}.")
    return reference


def prepare_eval_scene(source_root: Path, output_dir: Path, *, ffmpeg: str = "ffmpeg") -> Path:
    """Transcode one official eval clip into ten synchronized camera MP4s."""
    frame_dirs = _frame_directories(source_root)
    poses = _validated_poses(frame_dirs)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "poses_bounds.npy", poses)
    start_frame = int(frame_dirs[0].name)
    for camera in range(10):
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                str(NATIVE_FPS),
                "-start_number",
                str(start_frame),
                "-i",
                str(source_root / "frames" / "%05d" / "images" / f"{camera:03d}.jpg"),
                "-frames:v",
                str(len(frame_dirs)),
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_dir / f"cam{camera:02d}.mp4"),
            ],
            check=True,
        )
    background_dir = source_root / "background"
    if background_dir.is_dir():
        shutil.copytree(background_dir, output_dir / "background", dirs_exist_ok=True)
    metadata = {
        "version": "dynaworld_deep3d_mask_prepared/v1",
        "source": "Deep 3D Mask Volume official evaluation archive",
        "source_url": "https://cseweb.ucsd.edu/~viscomp/projects/ICCV21Deep/",
        "source_google_drive_id": OFFICIAL_EVAL_ARCHIVE_ID,
        "license": "MIT",
        "native_frame_start": start_frame,
        "frame_count": len(frame_dirs),
        "fps": NATIVE_FPS,
        "camera_count": 10,
        "source_frame_size": [1080, 1920],
        "poses_sha256": _sha256(output_dir / "poses_bounds.npy"),
    }
    (output_dir / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    prepared = prepare_eval_scene(args.source_root, args.output_dir, ffmpeg=args.ffmpeg)
    print(f"Prepared Deep 3D Mask evaluation scene at {prepared}")


if __name__ == "__main__":
    main()
