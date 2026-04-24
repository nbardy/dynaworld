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


@dataclass(frozen=True)
class ClipPlan:
    source: SourceVideo
    start_seconds: float
    source_indices: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a loader-compatible fixed-length clip dataset from local videos."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=None,
        help="One or more source video files or directories to scan recursively for a single train split.",
    )
    parser.add_argument(
        "--train-input",
        nargs="+",
        default=None,
        help="One or more source video files or directories for the train split.",
    )
    parser.add_argument(
        "--test-input",
        nargs="+",
        default=None,
        help="One or more source video files or directories for the test split.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/clip_sets/local_100_128_4fps_46f"),
        help="Directory that will receive clips/, manifest.jsonl, and dataset.json.",
    )
    parser.add_argument("--dataset-name", default="local_100_128_4fps_46f")
    parser.add_argument("--target-count", type=int, default=100, help="Clip target for --input single-split builds.")
    parser.add_argument("--train-count", type=int, default=None, help="Clip target for --train-input split builds.")
    parser.add_argument("--test-count", type=int, default=0, help="Clip target for --test-input split builds.")
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
    parser.add_argument(
        "--source-schedule",
        choices=("sequential", "round_robin"),
        default="sequential",
        help="How to choose windows across sources. round_robin keeps small split builds balanced across videos.",
    )
    parser.add_argument(
        "--max-clips-per-source",
        type=int,
        default=0,
        help="Maximum clips to take from each source video per split. 0 means unlimited.",
    )
    parser.add_argument(
        "--require-target-count",
        action="store_true",
        help="Fail when the requested train/test counts cannot be built.",
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


def validate_args(args: argparse.Namespace) -> None:
    split_inputs = args.train_input is not None or args.test_input is not None
    if args.input is not None and split_inputs:
        raise ValueError("Use either --input or --train-input/--test-input, not both.")
    if args.input is None and not split_inputs:
        raise ValueError("Pass --input for a train-only build, or --train-input/--test-input for split builds.")
    if args.target_count < 1:
        raise ValueError("--target-count must be >= 1.")
    if args.train_count is not None and args.train_count < 0:
        raise ValueError("--train-count must be >= 0.")
    if args.test_count < 0:
        raise ValueError("--test-count must be >= 0.")
    if args.clip_frames < 2:
        raise ValueError("--clip-frames must be >= 2.")
    if args.fps <= 0.0:
        raise ValueError("--fps must be positive.")
    if args.target_size < 1:
        raise ValueError("--target-size must be >= 1.")
    if args.max_clips_per_source < 0:
        raise ValueError("--max-clips-per-source must be >= 0.")
    if args.train_input is None and args.train_count is not None and args.train_count > 0:
        raise ValueError("--train-count requires --train-input.")
    if args.test_input is None and args.test_count > 0:
        raise ValueError("--test-count requires --test-input.")


def requested_train_count(args: argparse.Namespace) -> int:
    if args.train_input is None:
        return 0
    if args.train_count is None:
        return args.target_count
    return args.train_count


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


def clip_plans_for_source(
    source: SourceVideo,
    *,
    start_seconds: float,
    stride_seconds: float,
    clip_frames: int,
    target_fps: float,
    max_clips_per_source: int,
) -> list[ClipPlan]:
    latest_start = source.duration_seconds - float(clip_frames - 1) / float(target_fps)
    plans: list[ClipPlan] = []
    cursor = float(start_seconds)
    while cursor <= latest_start + 1.0e-6:
        source_indices = clip_source_frame_indices(source, cursor, clip_frames, target_fps)
        if len(source_indices) == clip_frames:
            plans.append(ClipPlan(source=source, start_seconds=cursor, source_indices=source_indices))
            if max_clips_per_source > 0 and len(plans) >= max_clips_per_source:
                break
        cursor += stride_seconds
    return plans


def iter_clip_plans(
    sources: list[SourceVideo],
    *,
    start_seconds: float,
    stride_seconds: float,
    clip_frames: int,
    target_fps: float,
    source_schedule: str,
    max_clips_per_source: int,
) -> list[ClipPlan]:
    plans_by_source = [
        clip_plans_for_source(
            source,
            start_seconds=start_seconds,
            stride_seconds=stride_seconds,
            clip_frames=clip_frames,
            target_fps=target_fps,
            max_clips_per_source=max_clips_per_source,
        )
        for source in sources
    ]
    if source_schedule == "sequential":
        return [plan for source_plans in plans_by_source for plan in source_plans]
    if source_schedule == "round_robin":
        max_windows = max((len(source_plans) for source_plans in plans_by_source), default=0)
        return [
            source_plans[window_index]
            for window_index in range(max_windows)
            for source_plans in plans_by_source
            if window_index < len(source_plans)
        ]
    raise ValueError(f"Unsupported source schedule: {source_schedule}")


def write_clip(
    *,
    cv2: Any,
    source: SourceVideo,
    output_dir: Path,
    clip_id: str,
    split: str,
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
        "split": split,
        "frame_sampling": {
            "fps": target_fps,
            "total_frames": clip_frames,
            "sampled_frames": sampled_frames,
        },
    }
    (clip_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return {
        "clip_id": clip_id,
        "split": split,
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


def build_split_entries(
    args: argparse.Namespace,
    *,
    cv2: Any,
    inputs: list[str],
    split: str,
    target_count: int,
    clip_id_prefix: str,
) -> list[dict[str, Any]]:
    if target_count < 1:
        return []

    source_paths = iter_video_paths(inputs, tuple(args.extensions))
    sources = [source for path in source_paths if (source := probe_video(path, cv2)) is not None]
    if not sources:
        raise ValueError(f"No usable source videos found for split={split!r}.")

    clip_duration = float(args.clip_frames) / float(args.fps)
    stride_seconds = float(args.stride_seconds) if args.stride_seconds > 0.0 else clip_duration
    entries: list[dict[str, Any]] = []
    plans = iter_clip_plans(
        sources,
        start_seconds=float(args.start_seconds),
        stride_seconds=stride_seconds,
        clip_frames=args.clip_frames,
        target_fps=args.fps,
        source_schedule=args.source_schedule,
        max_clips_per_source=args.max_clips_per_source,
    )

    for plan in plans:
        if len(entries) >= target_count:
            break
        clip_id = f"{clip_id_prefix}_{len(entries):06d}"
        if args.dry_run:
            entries.append(
                {
                    "clip_id": clip_id,
                    "split": split,
                    "frame_count": args.clip_frames,
                    "fps": args.fps,
                    "target_size": args.target_size,
                    "source_path": str(plan.source.path.resolve()),
                    "source_fps": plan.source.fps,
                    "source_frame_count": plan.source.frame_count,
                    "source_start_seconds": plan.start_seconds,
                    "source_start_frame": plan.source_indices[0],
                    "source_end_frame": plan.source_indices[-1],
                }
            )
            continue
        entry = write_clip(
            cv2=cv2,
            source=plan.source,
            output_dir=args.output_dir,
            clip_id=clip_id,
            split=split,
            start_seconds=plan.start_seconds,
            source_indices=plan.source_indices,
            clip_frames=args.clip_frames,
            target_fps=args.fps,
            target_size=args.target_size,
        )
        if entry is not None:
            entries.append(entry)

    return entries


def build_dataset(args: argparse.Namespace) -> list[dict[str, Any]]:
    validate_args(args)

    cv2 = import_cv2()
    prepare_output_dir(args.output_dir, overwrite=args.overwrite, dry_run=args.dry_run)

    if args.input is not None:
        return build_split_entries(
            args,
            cv2=cv2,
            inputs=args.input,
            split="train",
            target_count=args.target_count,
            clip_id_prefix="clip",
        )

    entries: list[dict[str, Any]] = []
    if args.train_input is not None:
        entries.extend(
            build_split_entries(
                args,
                cv2=cv2,
                inputs=args.train_input,
                split="train",
                target_count=requested_train_count(args),
                clip_id_prefix="train",
            )
        )
    if args.test_input is not None:
        entries.extend(
            build_split_entries(
                args,
                cv2=cv2,
                inputs=args.test_input,
                split="test",
                target_count=args.test_count,
                clip_id_prefix="test",
            )
        )
    return entries


def write_jsonl(path: Path, entries: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")


def split_counts(entries: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        split = str(entry.get("split", "train"))
        counts[split] = counts.get(split, 0) + 1
    return counts


def split_source_counts(entries: list[dict[str, Any]]) -> dict[str, int]:
    sources_by_split: dict[str, set[str]] = {}
    for entry in entries:
        split = str(entry.get("split", "train"))
        source_path = str(entry.get("source_path", ""))
        sources_by_split.setdefault(split, set()).add(source_path)
    return {split: len(sources) for split, sources in sources_by_split.items()}


def validate_built_counts(args: argparse.Namespace, entries: list[dict[str, Any]]) -> None:
    if not args.require_target_count:
        return
    counts = split_counts(entries)
    if args.input is not None and counts.get("train", 0) != args.target_count:
        raise ValueError(f"Requested {args.target_count} train clips but built {counts.get('train', 0)}.")
    if args.train_input is not None and counts.get("train", 0) != requested_train_count(args):
        raise ValueError(
            f"Requested {requested_train_count(args)} train clips but built {counts.get('train', 0)}."
        )
    if args.test_input is not None and counts.get("test", 0) != args.test_count:
        raise ValueError(f"Requested {args.test_count} test clips but built {counts.get('test', 0)}.")


def write_manifest(args: argparse.Namespace, entries: list[dict[str, Any]]) -> None:
    if args.dry_run:
        return
    manifest_path = args.output_dir / "manifest.jsonl"
    write_jsonl(manifest_path, entries)

    split_manifest_paths = {}
    counts = split_counts(entries)
    for split in sorted(counts):
        split_entries = [entry for entry in entries if entry.get("split", "train") == split]
        split_manifest_path = args.output_dir / f"{split}_manifest.jsonl"
        write_jsonl(split_manifest_path, split_entries)
        split_manifest_paths[split] = str(split_manifest_path.resolve())
    requested_train = requested_train_count(args) if args.train_input is not None else args.target_count
    requested_test = args.test_count if args.test_input is not None else 0
    requested_total = requested_train + requested_test
    source_counts = split_source_counts(entries)

    dataset_summary = {
        "dataset_name": args.dataset_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "clip_count": len(entries),
        "target_count": requested_total,
        "requested_train_count": requested_train,
        "requested_test_count": requested_test,
        "train_count": counts.get("train", 0),
        "test_count": counts.get("test", 0),
        "train_source_count": source_counts.get("train", 0),
        "test_source_count": source_counts.get("test", 0),
        "splits": {
            split: {
                "clip_count": count,
                "source_count": source_counts.get(split, 0),
                "manifest_path": split_manifest_paths[split],
            }
            for split, count in sorted(counts.items())
        },
        "clip_frames": args.clip_frames,
        "fps": args.fps,
        "target_size": args.target_size,
        "source_schedule": args.source_schedule,
        "max_clips_per_source": args.max_clips_per_source,
        "manifest_path": str(manifest_path.resolve()),
    }
    (args.output_dir / "dataset.json").write_text(json.dumps(dataset_summary, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    entries = build_dataset(args)
    validate_built_counts(args, entries)
    write_manifest(args, entries)
    counts = split_counts(entries)
    source_counts = split_source_counts(entries)
    print(
        f"Planned {len(entries)} clips "
        f"({args.clip_frames} frames, {args.fps:g} fps, {args.target_size}px, "
        f"splits={counts}, sources={source_counts}) at {args.output_dir}"
    )
    if args.input is not None and len(entries) < args.target_count:
        print(f"Warning: requested {args.target_count} clips but only found {len(entries)} usable windows.")
    if args.train_input is not None and counts.get("train", 0) < requested_train_count(args):
        print(
            f"Warning: requested {requested_train_count(args)} train clips "
            f"but only found {counts.get('train', 0)} usable windows."
        )
    if args.test_input is not None and counts.get("test", 0) < args.test_count:
        print(f"Warning: requested {args.test_count} test clips but only found {counts.get('test', 0)} usable windows.")


if __name__ == "__main__":
    main()
