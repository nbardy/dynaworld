from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image

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


def _image_to_tensor(image: Image.Image, target_size: int) -> torch.Tensor:
    resized = image.resize((target_size, target_size), resample=BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _load_frame(path: Path, target_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        return _image_to_tensor(image.convert("RGB"), target_size)


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
    return json.loads(summary_path.read_text())


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
            frames.append(_image_to_tensor(Image.fromarray(frame_rgb), target_size))
            if max_frames > 0 and len(frames) >= max_frames:
                break
    finally:
        capture.release()

    if len(frames) < 2:
        raise ValueError(f"Need at least 2 frames in video {video_path}")

    frame_times, video_fps = build_uniform_frame_times(len(frames), fps)
    return SequenceData(
        frames=torch.stack(frames, dim=0),
        frame_times=normalize_frame_times(frame_times),
        video_fps=video_fps,
        frame_source=frame_source,
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
) -> SequenceData:
    all_frame_paths = sorted(frames_dir.glob("*.png"))
    frame_paths, resolved_frame_source = _resolve_frame_paths(frames_dir, metadata, frame_source)
    if max_frames > 0:
        frame_paths = frame_paths[:max_frames]
    if len(frame_paths) < 2:
        raise ValueError(f"Need at least 2 frames in {frames_dir}")

    frames = [_load_frame(frame_path, target_size) for frame_path in frame_paths]
    frame_times, video_fps = _build_frame_times(frame_paths, metadata)
    return SequenceData(
        frames=torch.stack(frames, dim=0),
        frame_times=normalize_frame_times(frame_times),
        video_fps=video_fps,
        frame_source=resolved_frame_source,
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
    device: torch.device | str | None = None,
) -> SequenceData:
    records = tuple(json.loads(camera_json_path.read_text()))
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
        frames.append(_load_frame(frame_path, target_size))

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
        )

    if device is not None:
        return sequence.to(device)
    return sequence


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
