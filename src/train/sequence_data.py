"""Single-sequence data loaders for same-view training.

This module owns the single-camera side of ``research_notes/data_contract.md``:
prepared frame clips, direct video windows, and camera-json sequences that train
against the same camera/time window they encode. Calibrated heldout-camera
supervision belongs in ``multicam_video_data.py``.
"""
from __future__ import annotations

import json
import hashlib
from concurrent.futures import Future, ThreadPoolExecutor
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image

from checkpoint_utils import atomic_torch_save, load_torch_checkpoint
from json_io import load_json, load_jsonl_objects

try:
    from camera import CameraSpec
    from runtime_types import ClipBatch, FrameSource, SequenceData
except ImportError:  # pragma: no cover - supports package-style imports in tests.
    from .camera import CameraSpec
    from .runtime_types import ClipBatch, FrameSource, SequenceData

FocalMode = Literal["per_frame", "median"]


def _import_cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on optional local video deps.
        raise ImportError(
            "OpenCV is required for direct video loading. Use frame/camera JSON data or install cv2."
        ) from exc
    return cv2


def infer_video_fps(records: Sequence[Mapping[str, Any]]) -> float:
    timestamps = [record.get("timestamp_seconds") for record in records]
    diffs = []
    for left, right in zip(timestamps[:-1], timestamps[1:]):
        if left is None or right is None:
            continue
        delta = float(right) - float(left)
        if delta > 0:
            diffs.append(delta)
    if not diffs:
        return 1.0
    return float(1.0 / np.median(np.asarray(diffs, dtype=np.float32)))


def normalize_frame_times(frame_times: torch.Tensor | np.ndarray | Sequence[float]) -> torch.Tensor:
    device = frame_times.device if torch.is_tensor(frame_times) else None
    values = torch.as_tensor(frame_times, dtype=torch.float32, device=device)
    if values.numel() < 2:
        return torch.zeros_like(values)

    minimum = values.min()
    maximum = values.max()
    if float((maximum - minimum).detach().cpu()) > 1e-6:
        return (values - minimum) / (maximum - minimum)
    return torch.zeros_like(values)


def build_uniform_frame_times(num_frames: int, fps: float) -> tuple[torch.Tensor, float]:
    if num_frames < 1:
        raise ValueError("Need at least one frame to build timestamps.")
    safe_fps = float(fps) if fps and fps > 0 else 1.0
    values = torch.arange(num_frames, dtype=torch.float32) / safe_fps
    return values.unsqueeze(-1), safe_fps


try:
    BILINEAR = Image.Resampling.BILINEAR
except AttributeError:  # pragma: no cover - old Pillow compatibility.
    BILINEAR = Image.BILINEAR


def _center_square_crop(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def _image_to_tensor(image: Image.Image, target_size: int, image_crop_mode: str = "resize") -> torch.Tensor:
    crop_mode = str(image_crop_mode or "resize").lower()
    if crop_mode in {"center_square", "center_crop", "center"}:
        image = _center_square_crop(image)
    elif crop_mode not in {"resize", "none"}:
        raise ValueError(
            f"Unsupported image_crop_mode={image_crop_mode!r}. "
            "Expected one of: resize, none, center_square."
        )
    resized = image.resize((target_size, target_size), resample=BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _load_frame(path: Path, target_size: int, image_crop_mode: str = "resize") -> torch.Tensor:
    with Image.open(path) as image:
        return _image_to_tensor(image.convert("RGB"), target_size, image_crop_mode=image_crop_mode)


def _frame_cache_path(
    frame_cache_dir: Path | None,
    *,
    video_path: Path,
    target_size: int,
    start_seconds: float,
    duration_seconds: float | None,
    fps: float,
    frame_count: int,
    image_crop_mode: str,
) -> tuple[Path, str] | None:
    if frame_cache_dir is None:
        return None
    resolved_video = video_path.expanduser().resolve()
    video_stat = None
    if resolved_video.exists():
        stat = resolved_video.stat()
        video_stat = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    payload = {
        "version": 1,
        "video_path": str(resolved_video),
        "video_stat": video_stat,
        "target_size": int(target_size),
        "start_seconds": float(start_seconds),
        "duration_seconds": None if duration_seconds is None else float(duration_seconds),
        "fps": float(fps),
        "frame_count": int(frame_count),
        "image_crop_mode": str(image_crop_mode),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    key = hashlib.sha256(encoded).hexdigest()[:24]
    return frame_cache_dir / f"{key}.pt", key


def _load_cached_video_window(
    frame_cache_dir: Path | None,
    *,
    video_path: Path,
    target_size: int,
    start_seconds: float,
    duration_seconds: float | None,
    fps: float,
    frame_count: int,
    frame_source: FrameSource,
    image_crop_mode: str,
) -> SequenceData | None:
    cache_info = _frame_cache_path(
        frame_cache_dir,
        video_path=video_path,
        target_size=target_size,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        fps=fps,
        frame_count=frame_count,
        image_crop_mode=image_crop_mode,
    )
    if cache_info is None:
        return None
    path, key = cache_info
    if not path.exists():
        return None
    try:
        payload = load_torch_checkpoint(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        print(f"[frame-cache] ignoring unreadable cache {path}: {exc}")
        return None
    if not isinstance(payload, Mapping) or payload.get("cache_key") != key:
        print(f"[frame-cache] ignoring stale cache {path}")
        return None
    frames_uint8 = payload.get("frames_uint8")
    frame_times = payload.get("frame_times")
    if not torch.is_tensor(frames_uint8) or frames_uint8.dtype != torch.uint8:
        print(f"[frame-cache] ignoring cache with invalid frames {path}")
        return None
    if not torch.is_tensor(frame_times):
        print(f"[frame-cache] ignoring cache with invalid frame_times {path}")
        return None
    records = tuple(payload.get("records") or ())
    frames = frames_uint8.to(dtype=torch.float32).div(255.0)
    return SequenceData(
        frames=frames,
        frame_times=normalize_frame_times(frame_times.to(dtype=torch.float32)),
        video_fps=float(payload.get("video_fps", fps)),
        frame_source=frame_source,
        image_crop_mode=image_crop_mode,
        records=records,
        source_path=video_path,
        selected_frame_count=int(payload.get("selected_frame_count", frames.shape[0])),
        all_frame_count=int(payload.get("all_frame_count", frames.shape[0])),
    )


def _save_cached_video_window(
    frame_cache_dir: Path | None,
    *,
    video_path: Path,
    target_size: int,
    start_seconds: float,
    duration_seconds: float | None,
    fps: float,
    frame_count: int,
    image_crop_mode: str,
    frames: torch.Tensor,
    frame_times: torch.Tensor,
    video_fps: float,
    records: tuple[Mapping[str, Any], ...],
    selected_frame_count: int,
    all_frame_count: int,
) -> None:
    cache_info = _frame_cache_path(
        frame_cache_dir,
        video_path=video_path,
        target_size=target_size,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        fps=fps,
        frame_count=frame_count,
        image_crop_mode=image_crop_mode,
    )
    if cache_info is None:
        return
    path, key = cache_info
    frames_uint8 = frames.detach().cpu().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
    payload = {
        "cache_key": key,
        "frames_uint8": frames_uint8,
        "frame_times": frame_times.detach().cpu().to(dtype=torch.float32),
        "video_fps": float(video_fps),
        "records": tuple(dict(record) for record in records),
        "selected_frame_count": int(selected_frame_count),
        "all_frame_count": int(all_frame_count),
    }
    atomic_torch_save(payload, path)


def _build_frame_times(
    frame_paths: Sequence[Path],
    metadata: Mapping[str, Any] | None,
) -> tuple[torch.Tensor, float]:
    timestamps = None
    if metadata is not None:
        sampled_frames = metadata.get("frame_sampling", {}).get("sampled_frames", [])
        timestamp_by_path = {}
        for item in sampled_frames:
            path = item.get("path")
            if path is None:
                continue
            timestamp_by_path[str(Path(path).resolve())] = item.get("timestamp_seconds")
        timestamps = [timestamp_by_path.get(str(path.resolve())) for path in frame_paths]

    if timestamps is None or all(timestamp is None for timestamp in timestamps):
        values = torch.arange(len(frame_paths), dtype=torch.float32).unsqueeze(-1)
        return values, 1.0

    values = np.asarray(
        [timestamp if timestamp is not None else index for index, timestamp in enumerate(timestamps)],
        dtype=np.float32,
    )
    video_fps = infer_video_fps([{"timestamp_seconds": value} for value in timestamps])
    return torch.from_numpy(values).unsqueeze(-1), video_fps


def load_sequence_metadata(sequence_dir: Path) -> Mapping[str, Any] | None:
    summary_path = sequence_dir / "summary.json"
    if not summary_path.exists():
        return None
    return load_json(summary_path)


def resolve_frames_dir(sequence_dir: Path, frames_dir: Path | None) -> Path:
    if frames_dir is not None:
        return frames_dir
    return sequence_dir / "frames"


def resolve_video_path(video_path: Path | None, metadata: Mapping[str, Any] | None) -> Path | None:
    if video_path is not None:
        return video_path
    if metadata is None:
        return None
    value = metadata.get("video")
    if not value:
        return None
    return Path(value)


def _resolve_frame_paths(
    frames_dir: Path,
    metadata: Mapping[str, Any] | None,
    frame_source: FrameSource,
) -> tuple[list[Path], FrameSource]:
    if frame_source == "summary_sampled" and metadata is not None:
        sampled_frames = metadata.get("frame_sampling", {}).get("sampled_frames", [])
        sampled_paths = [Path(item["path"]) for item in sampled_frames if item.get("path")]
        existing_paths = [path for path in sampled_paths if path.exists()]
        if len(existing_paths) >= 2:
            return existing_paths, "summary_sampled"

    if frame_source not in {"summary_sampled", "all_frames"}:
        raise ValueError(f"Unsupported frame sequence source: {frame_source}")
    return sorted(frames_dir.glob("*.png")), "all_frames"


def _summarize_sequence_intrinsics(intrinsics: np.ndarray) -> dict[str, float]:
    return {
        "fx_median": float(np.median(intrinsics[:, 0, 0])),
        "fy_median": float(np.median(intrinsics[:, 1, 1])),
        "cx_median": float(np.median(intrinsics[:, 0, 2])),
        "cy_median": float(np.median(intrinsics[:, 1, 2])),
    }


def _resolve_sequence_intrinsics(intrinsics: np.ndarray, focal_mode: FocalMode) -> np.ndarray:
    resolved = intrinsics.copy()
    if focal_mode == "median":
        summary = _summarize_sequence_intrinsics(intrinsics)
        resolved[:, 0, 0] = summary["fx_median"]
        resolved[:, 1, 1] = summary["fy_median"]
        return resolved
    if focal_mode == "per_frame":
        return resolved
    raise ValueError(f"Unsupported camera focal mode: {focal_mode}")


def load_video_sequence(
    video_path: Path,
    target_size: int,
    max_frames: int = 0,
    frame_source: FrameSource = "explicit_video",
    image_crop_mode: str = "resize",
) -> SequenceData:
    cv2 = _import_cv2()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames = []
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(_image_to_tensor(Image.fromarray(frame_rgb), target_size, image_crop_mode=image_crop_mode))
            if max_frames > 0 and len(frames) >= max_frames:
                break
    finally:
        capture.release()

    if not frames:
        raise ValueError(f"Need at least 1 frame in video {video_path}")

    frame_times, video_fps = build_uniform_frame_times(len(frames), fps)
    return SequenceData(
        frames=torch.stack(frames, dim=0),
        frame_times=normalize_frame_times(frame_times),
        video_fps=video_fps,
        frame_source=frame_source,
        image_crop_mode=image_crop_mode,
        source_path=video_path,
        selected_frame_count=len(frames),
        all_frame_count=total_frames if total_frames > 0 else len(frames),
    )


def load_video_window_sequence(
    video_path: Path,
    *,
    target_size: int,
    start_seconds: float,
    duration_seconds: float | None,
    fps: float,
    frame_count: int,
    frame_source: FrameSource = "explicit_video_window",
    image_crop_mode: str = "resize",
    frame_cache_dir: Path | None = None,
) -> SequenceData:
    cached = _load_cached_video_window(
        frame_cache_dir,
        video_path=video_path,
        target_size=target_size,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        fps=fps,
        frame_count=frame_count,
        frame_source=frame_source,
        image_crop_mode=image_crop_mode,
    )
    if cached is not None:
        return cached

    cv2 = _import_cv2()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    safe_fps = float(fps) if fps and fps > 0 else source_fps
    if safe_fps <= 0:
        raise ValueError(f"Need positive fps to sample video window from {video_path}")
    if frame_count <= 0:
        if duration_seconds is None or duration_seconds <= 0:
            raise ValueError("explicit_video_window requires frame_count or positive duration_seconds.")
        frame_count = max(1, int(round(float(duration_seconds) * safe_fps)))

    sample_times = float(start_seconds) + np.arange(int(frame_count), dtype=np.float32) / safe_fps
    if duration_seconds is not None and duration_seconds > 0:
        end_seconds = float(start_seconds) + float(duration_seconds)
        if float(sample_times[-1]) > end_seconds + (0.5 / safe_fps):
            raise ValueError(
                f"Requested {frame_count} frames at fps={safe_fps} exceeds video window "
                f"[{start_seconds}, {end_seconds}] for {video_path}"
            )

    frames = []
    frame_indices = []
    try:
        if source_fps > 0:
            target_indices = [max(0, int(round(float(timestamp) * source_fps))) for timestamp in sample_times]
            if total_frames > 0:
                target_indices = [min(index, total_frames - 1) for index in target_indices]
            first_index = target_indices[0]
            last_index = target_indices[-1]
            capture.set(cv2.CAP_PROP_POS_FRAMES, first_index)
            next_target = 0
            for current_index in range(first_index, last_index + 1):
                ok, frame_bgr = capture.read()
                if not ok:
                    raise ValueError(
                        f"Could not read source frame {current_index} from {video_path} "
                        f"after loading {len(frames)} of {frame_count} requested frame(s)."
                    )
                if current_index != target_indices[next_target]:
                    continue
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_tensor = _image_to_tensor(
                    Image.fromarray(frame_rgb),
                    target_size,
                    image_crop_mode=image_crop_mode,
                )
                while next_target < len(target_indices) and current_index == target_indices[next_target]:
                    frames.append(frame_tensor)
                    frame_indices.append(current_index)
                    next_target += 1
                if next_target >= len(target_indices):
                    break
        else:
            for timestamp in sample_times:
                capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(timestamp)) * 1000.0)
                frame_indices.append(-1)
                ok, frame_bgr = capture.read()
                if not ok:
                    raise ValueError(
                        f"Could not read frame at t={float(timestamp):.3f}s from {video_path} "
                        f"after loading {len(frames)} of {frame_count} requested frame(s)."
                    )
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames.append(
                    _image_to_tensor(
                        Image.fromarray(frame_rgb),
                        target_size,
                        image_crop_mode=image_crop_mode,
                    )
                )
    finally:
        capture.release()
    if len(frames) != int(frame_count):
        raise ValueError(f"Loaded {len(frames)} frame(s) from {video_path}, expected {frame_count}.")

    frame_times = torch.from_numpy(sample_times.astype(np.float32)).unsqueeze(-1)
    records = tuple({"timestamp_seconds": float(value), "frame_index": index} for value, index in zip(sample_times, frame_indices))
    stacked_frames = torch.stack(frames, dim=0)
    _save_cached_video_window(
        frame_cache_dir,
        video_path=video_path,
        target_size=target_size,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        fps=fps,
        frame_count=frame_count,
        image_crop_mode=image_crop_mode,
        frames=stacked_frames,
        frame_times=frame_times,
        video_fps=safe_fps,
        records=records,
        selected_frame_count=len(frames),
        all_frame_count=total_frames if total_frames > 0 else len(frames),
    )
    return SequenceData(
        frames=stacked_frames,
        frame_times=normalize_frame_times(frame_times),
        video_fps=safe_fps,
        frame_source=frame_source,
        image_crop_mode=image_crop_mode,
        records=records,
        source_path=video_path,
        selected_frame_count=len(frames),
        all_frame_count=total_frames if total_frames > 0 else len(frames),
    )


def load_frame_sequence(
    frames_dir: Path,
    metadata: Mapping[str, Any] | None = None,
    *,
    target_size: int,
    max_frames: int = 0,
    frame_source: FrameSource = "all_frames",
    image_crop_mode: str = "resize",
) -> SequenceData:
    all_frame_paths = sorted(frames_dir.glob("*.png"))
    frame_paths, resolved_frame_source = _resolve_frame_paths(frames_dir, metadata, frame_source)
    if max_frames > 0:
        frame_paths = frame_paths[:max_frames]
    if not frame_paths:
        raise ValueError(f"Need at least 1 frame in {frames_dir}")

    frames = [_load_frame(frame_path, target_size, image_crop_mode=image_crop_mode) for frame_path in frame_paths]
    frame_times, video_fps = _build_frame_times(frame_paths, metadata)
    return SequenceData(
        frames=torch.stack(frames, dim=0),
        frame_times=normalize_frame_times(frame_times),
        video_fps=video_fps,
        frame_source=resolved_frame_source,
        image_crop_mode=image_crop_mode,
        frame_paths=tuple(frame_paths),
        source_path=frames_dir,
        selected_frame_count=len(frame_paths),
        all_frame_count=len(all_frame_paths),
    )


def load_camera_sequence(
    camera_json_path: Path,
    target_size: int,
    camera_image_size: int = 224,
    max_frames: int = 0,
    focal_mode: FocalMode = "median",
    image_crop_mode: str = "resize",
    device: torch.device | str | None = None,
) -> SequenceData:
    records = tuple(load_json(camera_json_path))
    if max_frames > 0:
        records = records[:max_frames]
    if len(records) < 2:
        raise ValueError(f"Need at least 2 frame-camera records in {camera_json_path}")

    scale = float(target_size) / float(camera_image_size)
    base_pose = torch.tensor(records[0]["camera_to_world"], dtype=torch.float32)
    base_pose_inv = torch.linalg.inv(base_pose)
    raw_intrinsics = np.stack([np.asarray(record["intrinsics"], dtype=np.float32) for record in records], axis=0)
    intrinsics_per_frame = _resolve_sequence_intrinsics(raw_intrinsics, focal_mode=focal_mode)
    resolved_summary = _summarize_sequence_intrinsics(intrinsics_per_frame)

    frames = []
    cameras = []
    frame_paths = []
    timestamps = []
    for record, intrinsics in zip(records, intrinsics_per_frame):
        frame_path = Path(record["frame_path"])
        if not frame_path.exists():
            raise FileNotFoundError(f"Missing frame referenced by camera JSON: {frame_path}")
        frame_paths.append(frame_path)
        frames.append(_load_frame(frame_path, target_size, image_crop_mode=image_crop_mode))

        pose = torch.tensor(record["camera_to_world"], dtype=torch.float32)
        pose = base_pose_inv @ pose
        cameras.append(
            CameraSpec(
                fx=float(intrinsics[0, 0] * scale),
                fy=float(intrinsics[1, 1] * scale),
                cx=float(intrinsics[0, 2] * scale),
                cy=float(intrinsics[1, 2] * scale),
                camera_to_world=pose,
            )
        )

        timestamp = record.get("timestamp_seconds")
        timestamps.append(float(timestamp) if timestamp is not None else None)

    times_np = np.asarray(
        [timestamp if timestamp is not None else index for index, timestamp in enumerate(timestamps)],
        dtype=np.float32,
    )
    frame_times = torch.from_numpy(times_np).unsqueeze(-1)
    sequence = SequenceData(
        frames=torch.stack(frames, dim=0),
        frame_times=normalize_frame_times(frame_times),
        video_fps=infer_video_fps(records),
        frame_source="camera_json",
        image_crop_mode=image_crop_mode,
        frame_paths=tuple(frame_paths),
        cameras=tuple(cameras),
        records=records,
        intrinsics_summary={
            "focal_mode": focal_mode,
            "raw_fx_median": float(np.median(raw_intrinsics[:, 0, 0])),
            "raw_fy_median": float(np.median(raw_intrinsics[:, 1, 1])),
            "resolved_fx_median": resolved_summary["fx_median"],
            "resolved_fy_median": resolved_summary["fy_median"],
            "resolved_cx_median": resolved_summary["cx_median"],
            "resolved_cy_median": resolved_summary["cy_median"],
            "training_scale": scale,
        },
        source_path=camera_json_path,
        selected_frame_count=len(records),
        all_frame_count=len(records),
    )
    if device is not None:
        return sequence.to(device)
    return sequence


def load_uncalibrated_sequence(
    sequence_dir: Path,
    frames_dir: Path | None,
    video_path: Path | None,
    target_size: int,
    max_frames: int,
    frame_source: FrameSource,
    image_crop_mode: str = "resize",
    device: torch.device | str | None = None,
) -> SequenceData:
    metadata = load_sequence_metadata(sequence_dir)
    if frame_source in {"summary_video", "explicit_video"}:
        resolved_video_path = resolve_video_path(video_path, metadata if frame_source == "summary_video" else None)
        if resolved_video_path is None or not resolved_video_path.exists():
            mode_name = "summary_video" if frame_source == "summary_video" else "explicit_video"
            raise FileNotFoundError(
                f"{mode_name} requested but no usable video path was found. "
                "Pass --video-path explicitly or use --frame-source summary_sampled."
            )

        sequence = load_video_sequence(
            resolved_video_path,
            target_size=target_size,
            max_frames=max_frames,
            frame_source=frame_source,
            image_crop_mode=image_crop_mode,
        )
        expected_frames = metadata.get("frame_sampling", {}).get("total_frames") if metadata is not None else None
        if (
            frame_source == "summary_video"
            and expected_frames is not None
            and max_frames == 0
            and int(expected_frames) != int(sequence.all_frame_count)
        ):
            raise ValueError(
                f"summary_video requested but video frame count {sequence.all_frame_count} does not match "
                f"summary.json frame_sampling.total_frames {expected_frames}. "
                "Use --frame-source summary_sampled when the source video was not pre-downsampled."
            )
    else:
        sequence = load_frame_sequence(
            frames_dir=resolve_frames_dir(sequence_dir, frames_dir),
            metadata=metadata,
            target_size=target_size,
            max_frames=max_frames,
            frame_source=frame_source,
            image_crop_mode=image_crop_mode,
        )

    if device is not None:
        return sequence.to(device)
    return sequence


def _optional_path(value: str | Path | None) -> Path | None:
    return None if value is None else Path(value)


def load_manifest_entries(manifest_path: Path, split: str | None = None) -> list[dict[str, Any]]:
    entries = [
        entry
        for entry in load_jsonl_objects(manifest_path)
        if split is None or str(entry.get("split", "train")) == split
    ]
    if not entries:
        split_text = f" split={split!r}" if split is not None else ""
        raise ValueError(f"No manifest entries found in {manifest_path}{split_text}.")
    return entries


def load_manifest_sequence(
    entry: dict[str, Any],
    *,
    data_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    device: torch.device,
) -> SequenceData:
    sequence_dir = Path(entry["sequence_dir"])
    frames_dir = _optional_path(entry.get("frames_dir"))
    if frames_dir is None:
        frames_dir = resolve_frames_dir(sequence_dir, data_cfg["frames_dir"])
    video_path = _optional_path(entry.get("video_path"))
    frame_source = entry.get("frame_source", data_cfg["frame_source"])
    image_crop_mode = str(entry.get("image_crop_mode", data_cfg.get("image_crop_mode", "resize")))
    max_frames = int(entry.get("max_frames", data_cfg["max_frames"]))
    if frame_source == "explicit_video_window":
        if video_path is None:
            raise ValueError("Manifest entry with frame_source='explicit_video_window' requires video_path.")
        frame_count = int(entry.get("frame_count", max_frames or model_cfg["train_frame_count"]))
        if max_frames > 0:
            frame_count = min(frame_count, max_frames)
        duration_value = entry.get("duration_seconds")
        return load_video_window_sequence(
            video_path,
            target_size=model_cfg["size"],
            start_seconds=float(entry.get("start_seconds", 0.0)),
            duration_seconds=None if duration_value is None else float(duration_value),
            fps=float(entry.get("fps", data_cfg.get("fps", 0.0) or 0.0)),
            frame_count=frame_count,
            frame_source=frame_source,
            image_crop_mode=image_crop_mode,
            frame_cache_dir=_optional_path(entry.get("frame_cache_dir", data_cfg.get("frame_cache_dir"))),
        ).to(device)
    if frame_source == "camera_json":
        camera_json = _optional_path(entry.get("camera_json")) or data_cfg["camera_json"] or (
            sequence_dir / "per_frame_cameras.json"
        )
        return load_camera_sequence(
            camera_json_path=camera_json,
            target_size=model_cfg["size"],
            camera_image_size=int(entry.get("camera_image_size", data_cfg["camera_image_size"])),
            max_frames=max_frames,
            focal_mode=str(entry.get("camera_focal_mode", data_cfg["camera_focal_mode"])),
            image_crop_mode=image_crop_mode,
            device=device,
        )
    return load_uncalibrated_sequence(
        sequence_dir=sequence_dir,
        frames_dir=frames_dir,
        video_path=video_path,
        target_size=model_cfg["size"],
        max_frames=max_frames,
        frame_source=frame_source,
        image_crop_mode=image_crop_mode,
        device=device,
    )


def load_manifest_sequences(
    manifest_path: Path,
    *,
    split: str,
    data_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    device: torch.device,
    limit: int = 0,
) -> list[SequenceData]:
    entries = load_manifest_entries(manifest_path, split=split)
    if limit > 0:
        entries = entries[:limit]
    return [
        load_manifest_sequence(
            entry,
            data_cfg=data_cfg,
            model_cfg=model_cfg,
            device=device,
        )
        for entry in entries
    ]


class ManifestSequenceSampler:
    """Loads and samples same-view manifest sequences without trainer-local cursors."""

    def __init__(
        self,
        *,
        entries: list[dict[str, Any]],
        sequences: list[SequenceData],
        data_cfg: dict[str, Any],
        model_cfg: dict[str, Any],
        device: torch.device | str,
        load_mode: str,
        sample_mode: str,
        prefetch_depth: int = 0,
        prefetch_name: str = "dynaworld-sequence-prefetch",
    ) -> None:
        if not entries:
            raise ValueError("ManifestSequenceSampler requires at least one manifest entry.")
        if not sequences:
            raise ValueError("ManifestSequenceSampler requires at least one loaded sequence.")
        self.entries = entries
        self.sequences = sequences
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.device = torch.device(device)
        self.load_mode = str(load_mode).lower()
        if self.load_mode not in {"eager", "lazy"}:
            raise ValueError("ManifestSequenceSampler load_mode must be one of: eager, lazy.")
        self.sample_mode = str(sample_mode).lower()
        if self.sample_mode not in {"random", "cycle"}:
            raise ValueError("ManifestSequenceSampler sample_mode must be one of: random, cycle.")
        self.prefetch_depth = int(prefetch_depth)
        if self.prefetch_depth < 0:
            raise ValueError("ManifestSequenceSampler prefetch_depth must be >= 0.")
        self.prefetch_name = str(prefetch_name)
        self.cursor = 0
        self._prefetch_executor: ThreadPoolExecutor | None = None
        self._prefetch_futures: list[Future[SequenceData]] = []

    @classmethod
    def from_manifest(
        cls,
        manifest_path: Path,
        *,
        split: str,
        data_cfg: dict[str, Any],
        model_cfg: dict[str, Any],
        device: torch.device | str,
        load_mode: str,
        sample_mode: str,
        prefetch_depth: int = 0,
        prefetch_name: str = "dynaworld-sequence-prefetch",
    ) -> "ManifestSequenceSampler":
        entries = load_manifest_entries(manifest_path, split=split)
        resolved_device = torch.device(device)
        if str(load_mode).lower() == "lazy":
            sequences = [
                load_manifest_sequence(
                    entries[0],
                    data_cfg=data_cfg,
                    model_cfg=model_cfg,
                    device=resolved_device,
                )
            ]
        else:
            sequences = [
                load_manifest_sequence(
                    entry,
                    data_cfg=data_cfg,
                    model_cfg=model_cfg,
                    device=resolved_device,
                )
                for entry in entries
            ]
        return cls(
            entries=entries,
            sequences=sequences,
            data_cfg=data_cfg,
            model_cfg=model_cfg,
            device=resolved_device,
            load_mode=load_mode,
            sample_mode=sample_mode,
            prefetch_depth=prefetch_depth,
            prefetch_name=prefetch_name,
        )

    @property
    def is_lazy(self) -> bool:
        return self.load_mode == "lazy"

    @property
    def sequence_count(self) -> int:
        return len(self.entries) if self.is_lazy else len(self.sequences)

    @property
    def prefetch_enabled(self) -> bool:
        return self._prefetch_executor is not None

    def frame_counts(self, *, fallback_frame_count: int) -> list[int]:
        if self.is_lazy:
            return [int(entry.get("frame_count", fallback_frame_count)) for entry in self.entries]
        return [sequence.frame_count for sequence in self.sequences]

    def validate_min_frame_count(self, minimum_required: int, *, label: str) -> None:
        if self.is_lazy:
            too_short_entries = [
                entry for entry in self.entries if int(entry.get("frame_count", 0)) < minimum_required
            ]
            if not too_short_entries:
                return
            examples = ", ".join(
                str(entry.get("source_path") or entry.get("sequence_dir")) for entry in too_short_entries[:3]
            )
            raise ValueError(
                f"Need at least train_frame_count={minimum_required} frames in every {label}; "
                f"{len(too_short_entries)} entries were too short. Examples: {examples}"
            )
        too_short_sequences = [
            sequence for sequence in self.sequences if sequence.frame_count < minimum_required
        ]
        if not too_short_sequences:
            return
        examples = ", ".join(str(sequence.source_path) for sequence in too_short_sequences[:3])
        raise ValueError(
            f"Need at least train_frame_count={minimum_required} frames in every {label}; "
            f"{len(too_short_sequences)} sequence(s) were too short. Examples: {examples}"
        )

    def next_index(self) -> int:
        count = self.sequence_count
        if count < 1:
            raise ValueError("Cannot sample from an empty manifest sequence sampler.")
        if self.sample_mode == "cycle":
            index = self.cursor % count
            self.cursor += 1
            return index
        return int(torch.randint(count, (1,)).item())

    def _load_entry(self, index: int, *, device: torch.device | str) -> SequenceData:
        if index < 0 or index >= len(self.entries):
            raise IndexError(f"Manifest sequence index {index} is out of range for {len(self.entries)} entries.")
        return load_manifest_sequence(
            self.entries[index],
            data_cfg=self.data_cfg,
            model_cfg=self.model_cfg,
            device=torch.device(device),
        )

    def sequence_at(self, index: int) -> SequenceData:
        if self.is_lazy:
            return self._load_entry(index, device=self.device)
        if index < 0 or index >= len(self.sequences):
            raise IndexError(
                f"Manifest sequence index {index} is out of range for {len(self.sequences)} loaded sequences."
            )
        return self.sequences[index]

    def start_prefetch(self) -> bool:
        if not self.is_lazy or self.prefetch_depth <= 0:
            return False
        if self._prefetch_executor is not None:
            return True
        self._prefetch_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=self.prefetch_name,
        )
        self._submit_prefetch()
        return True

    def _submit_prefetch(self) -> None:
        if self._prefetch_executor is None:
            return
        while len(self._prefetch_futures) < self.prefetch_depth:
            index = self.next_index()
            self._prefetch_futures.append(
                self._prefetch_executor.submit(self._load_entry, index, device=torch.device("cpu"))
            )

    def close_prefetch(self) -> None:
        if self._prefetch_executor is None:
            return
        self._prefetch_executor.shutdown(wait=False, cancel_futures=True)
        self._prefetch_executor = None
        self._prefetch_futures = []

    def _sample_lazy(self) -> SequenceData:
        if self._prefetch_executor is None:
            return self._load_entry(self.next_index(), device=self.device)
        if not self._prefetch_futures:
            self._submit_prefetch()
        future = self._prefetch_futures.pop(0)
        self._submit_prefetch()
        return future.result().to(self.device)

    def sample(self) -> SequenceData:
        if self.is_lazy:
            return self._sample_lazy()
        return self.sequences[self.next_index()]


def select_window_indices(
    num_frames: int,
    window_size: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if num_frames < 1:
        raise ValueError("Need at least one frame to sample a window.")
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}.")

    window = min(window_size, num_frames)
    if window >= num_frames:
        return torch.arange(num_frames, device=device)
    start = torch.randint(0, num_frames - window + 1, (1,), device=device).item()
    return torch.arange(start, start + window, device=device)


def make_clip(sequence: SequenceData, frame_indices: torch.Tensor) -> ClipBatch:
    indices = frame_indices.to(device=sequence.frames.device, dtype=torch.long)
    cameras = None
    if sequence.cameras is not None:
        cameras = tuple(sequence.cameras[index] for index in indices.detach().cpu().tolist())
    return ClipBatch(
        frames=sequence.frames[indices],
        frame_times=sequence.frame_times[indices],
        frame_indices=indices,
        video_fps=sequence.video_fps,
        cameras=cameras,
    )


def prepare_clip(sequence: SequenceData, frame_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return trainer legacy clip tensors from a typed `ClipBatch`.

    The data module owns frame/time/camera slicing. Older trainer call sites
    still consume `(clip_frames, clip_times)` where frames are shaped
    `[1, K, 3, H, W]` and times are shaped `[1, K]`; keep that adapter here
    while newer code can use `make_clip(...)` directly.
    """

    clip = make_clip(sequence, frame_indices)
    return clip.as_video_batch(), clip.as_time_batch(device=frame_indices.device)
