from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image


SRC_DIR = Path(__file__).resolve().parents[1]
TRAIN_DIR = SRC_DIR / "train"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

from config_utils import load_config_file  # noqa: E402
from download_utils import download_url  # noqa: E402


def resolve_root(config: dict[str, Any]) -> Path:
    root = Path(config["root_dir"])
    for child in ("raw", "extracted", "metadata"):
        (root / child).mkdir(parents=True, exist_ok=True)
    return root


def download(config: dict[str, Any], root: Path, *, overwrite: bool) -> Path:
    archive = root / "raw" / str(config["archive_name"])
    download_url(
        str(config["download_url"]),
        archive,
        overwrite=overwrite,
        user_agent="dynaworld-dnerf-paper-ingest",
        timeout_seconds=120.0,
    )
    return archive


def extract(config: dict[str, Any], root: Path, *, overwrite: bool) -> Path:
    archive = root / "raw" / str(config["archive_name"])
    if not archive.exists():
        raise FileNotFoundError(f"D-NeRF archive does not exist: {archive}")
    output = root / "extracted"
    marker = output / "data"
    if marker.exists() and not overwrite:
        print(f"Already extracted: {marker}")
        return marker
    if marker.exists():
        shutil.rmtree(marker)
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(output)
    if not marker.exists():
        raise RuntimeError(f"D-NeRF archive did not contain the expected data/ directory: {archive}")
    print(f"Extracted: {marker}")
    return marker


def _frame_image(scene: Path, frame: dict[str, Any]) -> Path:
    image = scene / str(frame["file_path"])
    if image.suffix:
        return image
    return image.with_suffix(".png")


def inspect_split(scene: Path, split: str) -> dict[str, Any]:
    transforms_path = scene / f"transforms_{split}.json"
    if not transforms_path.exists():
        raise FileNotFoundError(f"missing D-NeRF split metadata: {transforms_path}")
    payload = json.loads(transforms_path.read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"D-NeRF split has no frames: {transforms_path}")
    missing_images = []
    times = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise TypeError(f"frame {index} in {transforms_path} is not an object")
        matrix = frame.get("transform_matrix")
        if not isinstance(matrix, list) or len(matrix) != 4 or any(len(row) != 4 for row in matrix):
            raise ValueError(f"frame {index} in {transforms_path} does not have a 4x4 camera transform")
        times.append(float(frame["time"]))
        image = _frame_image(scene, frame)
        if not image.exists():
            missing_images.append(str(image))
    first_image = _frame_image(scene, frames[0])
    with Image.open(first_image) as image:
        image_size = list(image.size)
    return {
        "split": split,
        "transforms_path": str(transforms_path.resolve()),
        "frame_count": len(frames),
        "time_min": min(times),
        "time_max": max(times),
        "unique_time_count": len(set(times)),
        "camera_angle_x": float(payload["camera_angle_x"]),
        "image_size": image_size,
        "missing_images": missing_images,
    }


def inspect(config: dict[str, Any], root: Path) -> dict[str, Any]:
    data_root = root / "extracted" / "data"
    if not data_root.exists():
        raise FileNotFoundError(f"D-NeRF extraction does not exist: {data_root}")
    controlled = set(str(scene) for scene in config["controlled_scenes"])
    available = {path.name: path for path in data_root.iterdir() if path.is_dir()}
    missing_scenes = sorted(controlled - available.keys())
    if missing_scenes:
        raise FileNotFoundError(f"configured D-NeRF scenes are missing: {missing_scenes}")
    scenes = []
    for name in sorted(controlled):
        scene = available[name]
        splits = [inspect_split(scene, split) for split in ("train", "val", "test")]
        scenes.append({"scene": name, "scene_dir": str(scene.resolve()), "splits": splits})
    inventory = {
        "dataset": config["dataset_name"],
        "source_url": config["download_url"],
        "controlled_scenes": sorted(controlled),
        "scenes": scenes,
    }
    output = root / "metadata" / "controlled_scene_inventory.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote inventory: {output}")
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and validate the controlled D-NeRF paper subset.")
    parser.add_argument("stage", choices=("download", "extract", "inspect", "all"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/dataset_configs/dnerf_paper_breadth.jsonc"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config_file(args.config)
    root = resolve_root(config)
    if args.stage in {"download", "all"}:
        download(config, root, overwrite=args.overwrite)
    if args.stage in {"extract", "all"}:
        extract(config, root, overwrite=args.overwrite)
    if args.stage in {"inspect", "all"}:
        inspect(config, root)


if __name__ == "__main__":
    main()
