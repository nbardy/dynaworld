from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


try:
    BILINEAR = Image.Resampling.BILINEAR
except AttributeError:  # pragma: no cover - old Pillow compatibility.
    BILINEAR = Image.BILINEAR


@dataclass(frozen=True)
class MulticamValSample:
    metadata: dict[str, Any]
    source_frames: torch.Tensor
    target_frames: torch.Tensor


def load_multicam_val_manifest(manifest_path: Path, split: str | None = "val") -> list[dict[str, Any]]:
    records = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if not isinstance(record, dict):
                raise ValueError(f"Expected object on {manifest_path}:{line_number}.")
            if split is None or str(record.get("split", "val")) == split:
                records.append(record)
    if not records:
        split_text = f" split={split!r}" if split is not None else ""
        raise ValueError(f"No multi-camera validation records found in {manifest_path}{split_text}.")
    return records


def _load_frame(path: Path, target_size: int | None) -> torch.Tensor:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        if target_size is not None:
            rgb = rgb.resize((target_size, target_size), resample=BILINEAR)
        array = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _load_frame_dir(path: Path, target_size: int | None = None) -> torch.Tensor:
    frames = sorted(path.glob("frame_*.png"))
    if len(frames) < 2:
        raise ValueError(f"Need at least 2 validation frames in {path}")
    return torch.stack([_load_frame(frame, target_size) for frame in frames], dim=0)


def _import_cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on optional local video deps.
        raise ImportError("OpenCV is required to sample multi-camera validation MP4s.") from exc
    return cv2


def _sample_video_frames(
    *,
    video_path: Path,
    start_seconds: float,
    sample_fps: float,
    frame_count: int,
    target_size: int,
) -> torch.Tensor:
    cv2 = _import_cv2()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    native_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if native_fps <= 0.0:
        capture.release()
        raise ValueError(f"Could not infer native FPS for {video_path}")

    frames = []
    try:
        for index in range(frame_count):
            timestamp = start_seconds + float(index) / sample_fps
            native_frame_index = int(round(timestamp * native_fps))
            capture.set(cv2.CAP_PROP_POS_FRAMES, native_frame_index)
            ok, frame_bgr = capture.read()
            if not ok:
                raise RuntimeError(
                    f"Failed to read frame {native_frame_index} at {timestamp:.3f}s from {video_path}"
                )
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            if target_size is not None:
                image = image.resize((target_size, target_size), resample=BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
            frames.append(torch.from_numpy(array).permute(2, 0, 1).contiguous())
    finally:
        capture.release()

    return torch.stack(frames, dim=0)


def _load_record_frames(record: dict[str, Any], *, prefix: str, target_size: int) -> torch.Tensor:
    frames_dir = record.get(f"{prefix}_frames_dir")
    if frames_dir:
        path = Path(frames_dir)
        if path.exists():
            return _load_frame_dir(path, target_size=target_size)

    return _sample_video_frames(
        video_path=Path(record[f"{prefix}_video_path"]),
        start_seconds=float(record[f"{prefix}_start_seconds"]),
        sample_fps=float(record["fps"]),
        frame_count=int(record["frame_count"]),
        target_size=target_size,
    )


def load_multicam_val_sample(
    record: dict[str, Any],
    *,
    target_size: int | None = None,
    device: torch.device | str | None = None,
) -> MulticamValSample:
    resolved_size = int(target_size or record.get("target_size") or 0)
    if resolved_size <= 0:
        raise ValueError("target_size must be provided or present in the multi-camera validation record.")
    source_frames = _load_record_frames(record, prefix="source", target_size=resolved_size)
    target_frames = _load_record_frames(record, prefix="target", target_size=resolved_size)
    if source_frames.shape != target_frames.shape:
        raise ValueError(
            f"Source/target frame shape mismatch for {record.get('sample_id')}: "
            f"{tuple(source_frames.shape)} vs {tuple(target_frames.shape)}"
        )
    if device is not None:
        source_frames = source_frames.to(device)
        target_frames = target_frames.to(device)
    return MulticamValSample(metadata=record, source_frames=source_frames, target_frames=target_frames)
