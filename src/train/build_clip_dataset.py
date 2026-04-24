from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image


VIDEO_EXTENSIONS = (".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv")


def import_cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on local video deps.
        raise ImportError("OpenCV is required to build clip datasets. Install opencv-python.") from exc
    return cv2


try:
    BILINEAR = Image.Resampling.BILINEAR
except AttributeError:  # pragma: no cover - old Pillow compatibility.
    BILINEAR = Image.BILINEAR


@dataclass(frozen=True)
class SourceVideo:
    path: Path
    fps: float
    frame_count: int

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.fps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a loader-compatible fixed-length clip dataset from local videos."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more source video files or directories to scan recursively.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/clip_sets/local_100_128_4fps_46f"),
        help="Directory that will receive clips/, manifest.jsonl, and dataset.json.",
    )
    parser.add_argument("--dataset-name", default="local_100_128_4fps_46f")
    parser.add_argument("--target-count", type=int, default=100)
    parser.add_argument("--clip-frames", type=int, default=46)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--target-size", type=int, default=128)
    parser.add_argument(
        "--stride-seconds",
        type=float,
        default=0.0,
        help="Seconds between clip starts. Defaults to non-overlapping clips.",
    )
    parser.add_argument(
        "--start-seconds",
        type=float,
        default=0.0,
        help="Initial offset to skip at the beginning of each source video.",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=list(VIDEO_EXTENSIONS),
        help="Video filename extensions to include when scanning directories.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Probe and plan clips without writing files.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output dataset directory.")
    return parser.parse_args()


def iter_video_paths(inputs: list[str], extensions: tuple[str, ...]) -> list[Path]:
    paths: list[Path] = []
    extension_set = {extension.lower() for extension in extensions}
    for value in inputs:
        path = Path(value).expanduser()
        if path.is_dir():
            paths.extend(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file() and candidate.suffix.lower() in extension_set
            )
        elif path.is_file():
            if path.suffix.lower() in extension_set:
                paths.append(path)
        else:
            raise FileNotFoundError(f"Missing input path: {path}")
    return sorted({path.resolve() for path in paths})


def probe_video(path: Path, cv2: Any) -> SourceVideo | None:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            print(f"Skipping unreadable video: {path}")
            return None
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()

    if fps <= 0.0 or frame_count < 2:
        print(f"Skipping video with unusable metadata: {path} fps={fps} frames={frame_count}")
        return None
    return SourceVideo(path=path, fps=fps, frame_count=frame_count)


def prepare_output_dir(output_dir: Path, overwrite: bool, dry_run: bool) -> None:
    if dry_run:
        return
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    (output_dir / "clips").mkdir(parents=True, exist_ok=True)


def center_crop_resize(frame_rgb, target_size: int) -> Image.Image:
    image = Image.fromarray(frame_rgb)
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    cropped = image.crop((left, top, left + side, top + side))
    return cropped.resize((target_size, target_size), resample=BILINEAR)


def read_frame_at(capture, cv2: Any, frame_index: int):
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame_bgr = capture.read()
    if not ok:
        return None
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def clip_source_frame_indices(source: SourceVideo, start_seconds: float, clip_frames: int, target_fps: float) -> list[int]:
    indices = []
    for index in range(clip_frames):
        timestamp = start_seconds + float(index) / target_fps
        frame_index = int(round(timestamp * source.fps))
        if frame_index >= source.frame_count:
            return []
        indices.append(frame_index)
    return indices


def write_clip(
    *,
    cv2: Any,
    source: SourceVideo,
    output_dir: Path,
    clip_id: str,
    start_seconds: float,
    source_indices: list[int],
    clip_frames: int,
    target_fps: float,
    target_size: int,
) -> dict[str, Any] | None:
    clip_dir = output_dir / "clips" / clip_id
    frames_dir = clip_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(source.path))
    sampled_frames = []
    try:
        for local_index, source_frame_index in enumerate(source_indices):
            frame_rgb = read_frame_at(capture, cv2, source_frame_index)
            if frame_rgb is None:
                print(f"Skipping {clip_id}: failed to read frame {source_frame_index} from {source.path}")
                shutil.rmtree(clip_dir, ignore_errors=True)
                return None
            frame_path = frames_dir / f"frame_{local_index:04d}.png"
            center_crop_resize(frame_rgb, target_size).save(frame_path)
            sampled_frames.append(
                {
                    "path": str(frame_path.resolve()),
                    "timestamp_seconds": float(local_index) / target_fps,
                    "source_timestamp_seconds": float(source_frame_index) / source.fps,
                    "source_frame_index": int(source_frame_index),
                }
            )
    finally:
        capture.release()

    summary = {
        "video": str(source.path.resolve()),
        "frame_sampling": {
            "fps": target_fps,
            "total_frames": clip_frames,
            "sampled_frames": sampled_frames,
        },
    }
    (clip_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return {
        "clip_id": clip_id,
        "sequence_dir": str(clip_dir.resolve()),
        "frames_dir": str(frames_dir.resolve()),
        "frame_source": "summary_sampled",
        "frame_count": clip_frames,
        "fps": target_fps,
        "target_size": target_size,
        "source_path": str(source.path.resolve()),
        "source_fps": source.fps,
        "source_frame_count": source.frame_count,
        "source_start_seconds": start_seconds,
        "source_start_frame": source_indices[0],
        "source_end_frame": source_indices[-1],
    }


def build_dataset(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.target_count < 1:
        raise ValueError("--target-count must be >= 1.")
    if args.clip_frames < 2:
        raise ValueError("--clip-frames must be >= 2.")
    if args.fps <= 0.0:
        raise ValueError("--fps must be positive.")
    if args.target_size < 1:
        raise ValueError("--target-size must be >= 1.")

    cv2 = import_cv2()
    source_paths = iter_video_paths(args.input, tuple(args.extensions))
    sources = [source for path in source_paths if (source := probe_video(path, cv2)) is not None]
    if not sources:
        raise ValueError("No usable source videos found.")

    prepare_output_dir(args.output_dir, overwrite=args.overwrite, dry_run=args.dry_run)

    clip_duration = float(args.clip_frames) / float(args.fps)
    stride_seconds = float(args.stride_seconds) if args.stride_seconds > 0.0 else clip_duration
    entries: list[dict[str, Any]] = []

    for source in sources:
        start_seconds = float(args.start_seconds)
        latest_start = source.duration_seconds - float(args.clip_frames - 1) / float(args.fps)
        while start_seconds <= latest_start + 1.0e-6 and len(entries) < args.target_count:
            clip_id = f"clip_{len(entries):06d}"
            source_indices = clip_source_frame_indices(source, start_seconds, args.clip_frames, args.fps)
            if len(source_indices) == args.clip_frames:
                if args.dry_run:
                    entries.append(
                        {
                            "clip_id": clip_id,
                            "frame_count": args.clip_frames,
                            "fps": args.fps,
                            "target_size": args.target_size,
                            "source_path": str(source.path.resolve()),
                            "source_fps": source.fps,
                            "source_frame_count": source.frame_count,
                            "source_start_seconds": start_seconds,
                            "source_start_frame": source_indices[0],
                            "source_end_frame": source_indices[-1],
                        }
                    )
                else:
                    entry = write_clip(
                        cv2=cv2,
                        source=source,
                        output_dir=args.output_dir,
                        clip_id=clip_id,
                        start_seconds=start_seconds,
                        source_indices=source_indices,
                        clip_frames=args.clip_frames,
                        target_fps=args.fps,
                        target_size=args.target_size,
                    )
                    if entry is not None:
                        entries.append(entry)
            start_seconds += stride_seconds
        if len(entries) >= args.target_count:
            break

    return entries


def write_manifest(args: argparse.Namespace, entries: list[dict[str, Any]]) -> None:
    if args.dry_run:
        return
    manifest_path = args.output_dir / "manifest.jsonl"
    with manifest_path.open("w") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")

    dataset_summary = {
        "dataset_name": args.dataset_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "clip_count": len(entries),
        "target_count": args.target_count,
        "clip_frames": args.clip_frames,
        "fps": args.fps,
        "target_size": args.target_size,
        "manifest_path": str(manifest_path.resolve()),
    }
    (args.output_dir / "dataset.json").write_text(json.dumps(dataset_summary, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    entries = build_dataset(args)
    write_manifest(args, entries)
    print(
        f"Planned {len(entries)} clips "
        f"({args.clip_frames} frames, {args.fps:g} fps, {args.target_size}px) at {args.output_dir}"
    )
    if len(entries) < args.target_count:
        print(f"Warning: requested {args.target_count} clips but only found {len(entries)} usable windows.")


if __name__ == "__main__":
    main()
