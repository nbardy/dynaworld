from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

import numpy as np


def _import_cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on local video deps.
        raise ImportError("OpenCV is required to score video motion.") from exc
    return cv2


def _resize_gray(frame_bgr: np.ndarray, target_size: int) -> np.ndarray:
    cv2 = _import_cv2()
    frame_bgr = cv2.resize(frame_bgr, (target_size, target_size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32) / 255.0


def _read_sampled_frames(
    path: Path,
    *,
    max_frames: int,
    target_size: int,
    sample_fps: float | None,
    start_seconds: float,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    cv2 = _import_cv2()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frames: list[np.ndarray] = []
    try:
        if start_seconds > 0.0:
            capture.set(cv2.CAP_PROP_POS_MSEC, start_seconds * 1000.0)
        if sample_fps is not None and sample_fps > 0.0 and source_fps > 0.0:
            source_step = source_fps / float(sample_fps)
            next_sample_index = 0.0
            decoded_index = 0
            max_decode = int(np.ceil(source_step * max_frames)) + 4
            while len(frames) < max_frames and decoded_index < max_decode:
                ok, frame_bgr = capture.read()
                if not ok:
                    break
                if decoded_index + 1.0e-6 >= next_sample_index:
                    frames.append(_resize_gray(frame_bgr, target_size))
                    next_sample_index += source_step
                decoded_index += 1
        elif sample_fps is not None and sample_fps > 0.0:
            for sample_index in range(max_frames):
                seconds = start_seconds + float(sample_index) / float(sample_fps)
                capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000.0)
                ok, frame_bgr = capture.read()
                if not ok:
                    break
                frames.append(_resize_gray(frame_bgr, target_size))
        else:
            while len(frames) < max_frames:
                ok, frame_bgr = capture.read()
                if not ok:
                    break
                frames.append(_resize_gray(frame_bgr, target_size))
    finally:
        capture.release()

    metadata = {
        "path": str(path),
        "source_fps": source_fps,
        "source_frame_count": frame_count,
        "source_width": width,
        "source_height": height,
        "start_seconds": float(start_seconds),
        "sample_fps": None if sample_fps is None else float(sample_fps),
        "frames_scored": len(frames),
    }
    return frames, metadata


def score_video(
    path: Path,
    *,
    max_frames: int,
    target_size: int,
    sample_fps: float | None,
    start_seconds: float,
) -> dict[str, Any]:
    frames, metadata = _read_sampled_frames(
        path,
        max_frames=max_frames,
        target_size=target_size,
        sample_fps=sample_fps,
        start_seconds=start_seconds,
    )
    if len(frames) < 2:
        return {
            **metadata,
            "ok": False,
            "error": "fewer_than_two_frames",
            "mean_absdiff": 0.0,
            "p95_absdiff": 0.0,
            "max_absdiff": 0.0,
        }

    per_pair = np.asarray(
        [np.mean(np.abs(right - left)) for left, right in zip(frames[:-1], frames[1:])],
        dtype=np.float32,
    )
    return {
        **metadata,
        "ok": True,
        "mean_absdiff": float(np.mean(per_pair)),
        "median_absdiff": float(np.median(per_pair)),
        "p95_absdiff": float(np.percentile(per_pair, 95)),
        "max_absdiff": float(np.max(per_pair)),
        "pair_absdiff": [float(value) for value in per_pair],
    }


def iter_paths(patterns: list[str]) -> list[Path]:
    seen: set[Path] = set()
    paths: list[Path] = []
    for pattern in patterns:
        for value in sorted(glob.glob(pattern)):
            path = Path(value)
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            paths.append(path)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank videos by a cheap frame-difference motion proxy.")
    parser.add_argument("--glob", action="append", dest="globs", required=True, help="Video glob. May be repeated.")
    parser.add_argument("--output", type=Path, required=True, help="JSON output path.")
    parser.add_argument("--max-videos", type=int, default=0, help="Optional cap after glob expansion.")
    parser.add_argument("--max-frames", type=int, default=16, help="Number of frames to score per video.")
    parser.add_argument("--target-size", type=int, default=96, help="Square downsample size for scoring.")
    parser.add_argument("--sample-fps", type=float, default=4.0, help="Sample FPS; <=0 means consecutive frames.")
    parser.add_argument("--start-seconds", type=float, default=0.0, help="Start time for every scored video.")
    parser.add_argument("--top", type=int, default=10, help="Number of ranked rows to print.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = iter_paths(args.globs)
    if args.max_videos > 0:
        paths = paths[: args.max_videos]
    sample_fps = None if args.sample_fps <= 0.0 else float(args.sample_fps)

    rows = []
    for path in paths:
        try:
            row = score_video(
                path,
                max_frames=int(args.max_frames),
                target_size=int(args.target_size),
                sample_fps=sample_fps,
                start_seconds=float(args.start_seconds),
            )
        except Exception as exc:  # pragma: no cover - artifact script should record bad local media.
            row = {"path": str(path), "ok": False, "error": repr(exc)}
        rows.append(row)

    ranked = sorted(rows, key=lambda row: float(row.get("mean_absdiff", 0.0)), reverse=True)
    payload = {
        "schema_version": "video_motion_rank_v1",
        "metric": "mean grayscale absdiff between sampled adjacent frames, 0..1",
        "sample": {
            "max_frames": int(args.max_frames),
            "target_size": int(args.target_size),
            "sample_fps": sample_fps,
            "start_seconds": float(args.start_seconds),
        },
        "inputs": {"globs": args.globs, "video_count": len(paths)},
        "ranked": ranked,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for index, row in enumerate(ranked[: int(args.top)], start=1):
        print(
            {
                "rank": index,
                "path": row.get("path"),
                "ok": row.get("ok"),
                "mean_absdiff": row.get("mean_absdiff"),
                "p95_absdiff": row.get("p95_absdiff"),
                "frames_scored": row.get("frames_scored"),
                "source_size": [row.get("source_width"), row.get("source_height")],
                "source_fps": row.get("source_fps"),
            }
        )


if __name__ == "__main__":
    main()
