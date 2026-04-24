from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image


SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR / "train"))

from config_utils import load_config_file  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi"}


def resolve_root(config: dict[str, Any]) -> Path:
    root = Path(config["root_dir"])
    for child in ("raw", "extracted", "logs", "metadata"):
        (root / child).mkdir(parents=True, exist_ok=True)
    return root


def count_files(root: Path, extensions: set[str]) -> int:
    return sum(1 for path in root.rglob("*") if path.is_file() and path.suffix.lower() in extensions)


def camera_dirs(split_dir: Path) -> list[Path]:
    if not split_dir.exists():
        return []
    return sorted(path for path in split_dir.iterdir() if path.is_dir())


def require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Missing required command: {name}")


def inspect_scene(scene_dir: Path) -> dict[str, Any]:
    train_dirs = camera_dirs(scene_dir / "train")
    test_dirs = camera_dirs(scene_dir / "test")
    return {
        "scene": scene_dir.name,
        "scene_dir": str(scene_dir.resolve()),
        "has_calibration": (scene_dir / "calibration.json").exists(),
        "has_rotation_correction": (scene_dir / "rotation_correction.json").exists(),
        "train_camera_count": len(train_dirs),
        "test_camera_count": len(test_dirs),
        "train_cameras": [path.name for path in train_dirs],
        "test_cameras": [path.name for path in test_dirs],
        "image_file_count": count_files(scene_dir, IMAGE_EXTENSIONS),
        "video_file_count": count_files(scene_dir, VIDEO_EXTENSIONS),
        "meta_file_count": sum(1 for path in scene_dir.rglob("*.json") if path.is_file()),
    }


def candidate_scene_dirs(root: Path, scene_glob: str, max_scenes: int) -> list[Path]:
    search_roots = [root / "extracted", root / "raw"]
    scenes: list[Path] = []
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for candidate in sorted(search_root.glob(scene_glob)):
            if not candidate.is_dir():
                continue
            if (candidate / "train").exists() or (candidate / "test").exists() or (candidate / "calibration.json").exists():
                scenes.append(candidate)
            if len(scenes) >= max_scenes:
                return scenes
    return scenes


def inspect(config: dict[str, Any], root: Path) -> None:
    inspect_cfg = config["inspect"]
    scenes = candidate_scene_dirs(root, inspect_cfg["scene_glob"], int(inspect_cfg["max_scenes"]))
    summaries = [inspect_scene(scene) for scene in scenes]
    output_path = root / "metadata" / "scene_inventory.json"
    output_path.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")
    if not summaries:
        print(f"No ViVo scenes found under {root / 'raw'} or {root / 'extracted'}.")
        print("Download one scene manually from the project access link, then place it under one of those folders.")
    for summary in summaries:
        print(
            f"{summary['scene']}: train_cams={summary['train_camera_count']} "
            f"test_cams={summary['test_camera_count']} images={summary['image_file_count']} "
            f"videos={summary['video_file_count']} meta={summary['meta_file_count']}"
        )
    print(f"Wrote inventory: {output_path}")


def image_sequence_dir(camera_dir: Path, source_dir_names: list[str]) -> Path | None:
    for dirname in source_dir_names:
        candidate = camera_dir / dirname
        if candidate.exists() and any(path.suffix.lower() in IMAGE_EXTENSIONS for path in candidate.iterdir()):
            return candidate
    direct_images = [path for path in camera_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    if direct_images and any("colour-image" in path.name or "color-image" in path.name for path in direct_images):
        return camera_dir
    return None


def image_paths(image_dir: Path) -> list[Path]:
    return sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def frame_key(path: Path) -> str:
    # ViVo raw samples use ..._colour-image_<sequence>_<timestamp>.jpg.
    return path.name.split("_colour-image_")[-1].rsplit(".", 1)[0]


def metadata_for_image(image_path: Path) -> Path:
    return image_path.with_name(image_path.name + ".meta.json")


def parse_timestamp_ns(value: str | None) -> int | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    if "." in normalized:
        prefix, suffix = normalized.split(".", 1)
        fractional, timezone = suffix.split("+", 1)
        fractional = (fractional + "000000000")[:9]
        seconds = int(datetime.fromisoformat(prefix + "+00:00").timestamp())
        return seconds * 1_000_000_000 + int(fractional)
    return int(datetime.fromisoformat(normalized).timestamp() * 1_000_000_000)


def camera_rgb_summary(image_dir: Path) -> dict[str, Any]:
    images = image_paths(image_dir)
    if not images:
        raise RuntimeError(f"No RGB images found in {image_dir}")

    dimensions = set()
    for path in [images[0], images[-1]]:
        with Image.open(path) as image:
            dimensions.add(image.size)

    frame_records = []
    frame_rates = set()
    missing_metadata = []
    timestamps_ns = []
    for frame_index, image_path in enumerate(images):
        metadata_path = metadata_for_image(image_path)
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        if not metadata:
            missing_metadata.append(image_path.name)
        frame_rate = metadata.get("recordingFrameRate")
        if frame_rate is not None:
            frame_rates.add(float(frame_rate))
        timestamp_ns = parse_timestamp_ns(metadata.get("captureTimestamp"))
        if timestamp_ns is not None:
            timestamps_ns.append(timestamp_ns)
        frame_records.append(
            {
                "mp4_frame_index": frame_index,
                "image_path": str(image_path.resolve()),
                "metadata_path": str(metadata_path.resolve()) if metadata_path.exists() else None,
                "frame_key": frame_key(image_path),
                "capture_timestamp": metadata.get("captureTimestamp"),
                "sequence_number": metadata.get("sequenceNumber"),
            }
        )

    width, height = next(iter(dimensions))
    inferred_fps = None
    if len(timestamps_ns) >= 2:
        duration_seconds = (timestamps_ns[-1] - timestamps_ns[0]) / 1_000_000_000.0
        if duration_seconds > 0:
            inferred_fps = float((len(timestamps_ns) - 1) / duration_seconds)

    metadata_files = sorted(image_dir.glob("*colour-image*.jpg.meta.json"))
    return {
        "width": width,
        "height": height,
        "dimension_set": sorted([{"width": item[0], "height": item[1]} for item in dimensions], key=lambda item: (item["width"], item["height"])),
        "rgb_frame_count": len(images),
        "rgb_metadata_count": len(metadata_files),
        "missing_metadata_count": len(missing_metadata),
        "missing_metadata_examples": missing_metadata[:5],
        "recording_frame_rates": sorted(frame_rates),
        "inferred_fps_from_timestamps": inferred_fps,
        "first_capture_timestamp": frame_records[0]["capture_timestamp"],
        "last_capture_timestamp": frame_records[-1]["capture_timestamp"],
        "frames": frame_records,
    }


def camera_frame_pattern(image_dir: Path) -> str:
    images = image_paths(image_dir)
    if not images:
        raise RuntimeError(f"No RGB images found in {image_dir}")
    if all(path.stem.isdigit() for path in images):
        return str(image_dir / f"%0{len(images[0].stem)}d{images[0].suffix.lower()}")
    return str(image_dir / f"*{images[0].suffix.lower()}")


def encode_camera_rgb(image_dir: Path, output_path: Path, fps: int, crf: int, preset: str) -> None:
    require_tool("ffmpeg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pattern = camera_frame_pattern(image_dir)
    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-pattern_type",
        "glob",
        "-i",
        pattern,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        preset,
        "-crf",
        str(crf),
        str(output_path),
    ]
    if "%" in pattern and "*" not in pattern:
        command = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            pattern,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            preset,
            "-crf",
            str(crf),
            str(output_path),
        ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        log_path = output_path.with_suffix(".ffmpeg.log")
        log_path.write_text(result.stdout + "\n\n" + result.stderr)
        raise RuntimeError(f"ffmpeg failed for {image_dir}; see {log_path}")


def compact_rgb(config: dict[str, Any], root: Path, scene_name: str | None, delete_heavy: bool) -> None:
    compact_cfg = config["compact_rgb"]
    scenes = candidate_scene_dirs(root, config["inspect"]["scene_glob"], int(config["inspect"]["max_scenes"]))
    if scene_name is not None:
        scenes = [scene for scene in scenes if scene.name == scene_name]
    if not scenes:
        raise RuntimeError("No matching local ViVo scenes found to compact.")

    records = []
    output_root = root / "rgb_mp4"
    output_root.mkdir(parents=True, exist_ok=True)
    for scene in scenes:
        for split_name in ("train", "test"):
            for camera_dir in camera_dirs(scene / split_name):
                image_dir = image_sequence_dir(camera_dir, list(compact_cfg["image_source_dirs"]))
                if image_dir is None:
                    continue
                summary = camera_rgb_summary(image_dir)
                fps_candidates = summary["recording_frame_rates"]
                encode_fps = int(fps_candidates[0]) if len(fps_candidates) == 1 else int(compact_cfg["fps"])
                output_path = output_root / scene.name / split_name / f"{camera_dir.name}.mp4"
                encode_camera_rgb(
                    image_dir,
                    output_path,
                    fps=encode_fps,
                    crf=int(compact_cfg["crf"]),
                    preset=str(compact_cfg["preset"]),
                )
                camera_record = {
                    "scene": scene.name,
                    "split": split_name,
                    "camera": camera_dir.name,
                    "source_dir": str(image_dir.resolve()),
                    "mp4_path": str(output_path.resolve()),
                    "encoded_fps": encode_fps,
                    **{key: value for key, value in summary.items() if key != "frames"},
                }
                records.append(camera_record)

                frame_manifest_path = root / "metadata" / "rgb_mp4_frames" / scene.name / split_name / f"{camera_dir.name}.json"
                frame_manifest_path.parent.mkdir(parents=True, exist_ok=True)
                frame_manifest_path.write_text(
                    json.dumps({**camera_record, "frames": summary["frames"]}, indent=2, sort_keys=True) + "\n"
                )

                if delete_heavy:
                    for dirname in compact_cfg["heavy_dirs"]:
                        heavy_dir = camera_dir / dirname
                        if heavy_dir.exists():
                            shutil.rmtree(heavy_dir)
                    for marker in compact_cfg["heavy_filename_markers"]:
                        for path in camera_dir.glob(f"*{marker}*"):
                            if path.is_file():
                                path.unlink()

    output_path = root / "metadata" / "rgb_mp4_manifest.jsonl"
    with output_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    dimensions = {(record["width"], record["height"]) for record in records}
    encoded_fps_values = {record["encoded_fps"] for record in records}
    if len(dimensions) != 1 or len(encoded_fps_values) != 1:
        print(f"Warning: non-uniform RGB compaction dimensions={dimensions} fps={encoded_fps_values}")
    for record in records:
        if record["rgb_frame_count"] != record["rgb_metadata_count"] or record["missing_metadata_count"]:
            print(
                "Warning: frame/metadata mismatch "
                f"{record['scene']}/{record['split']}/{record['camera']} "
                f"frames={record['rgb_frame_count']} metadata={record['rgb_metadata_count']} "
                f"missing_meta={record['missing_metadata_count']}"
            )
    print(f"Encoded {len(records)} camera RGB videos. Wrote manifest: {output_path}")


def show_info(config: dict[str, Any]) -> None:
    print(json.dumps(config["source"], indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect locally downloaded ViVo dataset scenes.")
    parser.add_argument("stage", choices=("info", "inspect", "compact-rgb"))
    parser.add_argument("--config", type=Path, default=Path("src/dataset_configs/vivo_seed.jsonc"))
    parser.add_argument("--scene", default=None, help="Optional scene name to compact.")
    parser.add_argument(
        "--delete-heavy",
        action="store_true",
        help="After RGB MP4 encoding succeeds, delete configured depth/pointcloud/mask folders.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config_file(args.config)
    root = resolve_root(config)
    if args.stage == "info":
        show_info(config)
    elif args.stage == "inspect":
        inspect(config, root)
    elif args.stage == "compact-rgb":
        compact_rgb(config, root, scene_name=args.scene, delete_heavy=args.delete_heavy)


if __name__ == "__main__":
    main()
