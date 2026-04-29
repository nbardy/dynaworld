from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR / "train"))

from config_utils import load_config_file  # noqa: E402


def import_cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on local video deps.
        raise ImportError("OpenCV is required for segmentation. Install opencv-python.") from exc
    return cv2


@dataclass(frozen=True)
class Paths:
    root: Path
    candidates: Path
    raw: Path
    segments: Path
    clip_sets: Path
    logs: Path


def resolve_paths(config: dict[str, Any]) -> Paths:
    root = Path(config["root_dir"])
    paths = Paths(
        root=root,
        candidates=root / "candidates",
        raw=root / "raw",
        segments=root / "segments",
        clip_sets=root / "clip_sets",
        logs=root / "logs",
    )
    for path in paths.__dict__.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_command(command: list[str], *, log_path: Path | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if log_path is not None:
        log_path.write_text(
            "COMMAND\n"
            + " ".join(command)
            + "\n\nSTDOUT\n"
            + result.stdout
            + "\n\nSTDERR\n"
            + result.stderr
        )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(command)}")
    return result


def require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Missing required command: {name}")


def yt_dlp_command() -> list[str]:
    executable = shutil.which("yt-dlp")
    if executable is not None:
        return [executable]
    return [sys.executable, "-m", "yt_dlp"]


def canonical_youtube_url(record: dict[str, Any]) -> str | None:
    for key in ("webpage_url", "url"):
        value = record.get(key)
        if isinstance(value, str) and value.startswith("http"):
            return value
    video_id = record.get("id")
    if video_id:
        return f"https://www.youtube.com/watch?v={video_id}"
    return None


def existing_download(raw_dir: Path, video_id: str) -> Path | None:
    matches = sorted(raw_dir.glob(f"{video_id}.*"))
    return matches[-1].resolve() if matches else None


def search(config: dict[str, Any], paths: Paths) -> None:
    search_cfg = config["search"]
    max_results = int(search_cfg["max_results_per_query"])
    seen: set[str] = set()
    candidates: list[dict[str, Any]] = []

    for query in search_cfg["queries"]:
        target = f"ytsearch{max_results}:{query}"
        result = run_command([*yt_dlp_command(), "--dump-single-json", "--flat-playlist", target])
        payload = json.loads(result.stdout)
        for entry in payload.get("entries", []):
            video_id = entry.get("id") or entry.get("url")
            if not video_id or video_id in seen:
                continue
            seen.add(video_id)
            candidates.append(
                {
                    "id": video_id,
                    "query": query,
                    "title": entry.get("title"),
                    "url": canonical_youtube_url(entry),
                    "duration": entry.get("duration"),
                    "uploader": entry.get("uploader"),
                    "view_count": entry.get("view_count"),
                }
            )

    output_path = paths.candidates / "search_results.jsonl"
    write_jsonl(output_path, candidates)
    print(f"Wrote {len(candidates)} search candidates to {output_path}")


def download(config: dict[str, Any], paths: Paths, overwrite: bool) -> None:
    candidates = read_jsonl(paths.candidates / "search_results.jsonl")
    if not candidates:
        raise RuntimeError("No search candidates found. Run the search stage first.")

    download_cfg = config["download"]
    limit = int(download_cfg["limit"])
    max_height = int(download_cfg["max_height"])
    cookies_from_browser = download_cfg.get("cookies_from_browser")
    continue_on_error = bool(download_cfg.get("continue_on_error", False))
    section_start = download_cfg.get("section_start_seconds")
    section_duration = download_cfg.get("section_duration_seconds")
    downloaded: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for index, candidate in enumerate(candidates[:limit]):
        url = canonical_youtube_url(candidate)
        if not url:
            continue
        video_id = str(candidate["id"])
        if not overwrite:
            existing = existing_download(paths.raw, video_id)
            if existing is not None:
                downloaded.append({**candidate, "local_path": str(existing)})
                continue
        output_template = str(paths.raw / "%(id)s.%(ext)s")
        base_command = yt_dlp_command()
        command = [
            *base_command,
            "-f",
            f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/best[height<={max_height}][ext=mp4]/best",
            "--merge-output-format",
            "mp4",
            "--no-playlist",
            "-o",
            output_template,
            url,
        ]
        if overwrite:
            command[len(base_command):len(base_command)] = ["--force-overwrites"]
        if section_start is not None and section_duration is not None:
            start = float(section_start)
            end = start + float(section_duration)
            command[len(base_command):len(base_command)] = [
                "--download-sections",
                f"*{start:.3f}-{end:.3f}",
                "--force-keyframes-at-cuts",
            ]
        if cookies_from_browser:
            command[len(base_command):len(base_command)] = ["--cookies-from-browser", str(cookies_from_browser)]
        log_path = paths.logs / f"download_{index:04d}_{candidate['id']}.log"
        try:
            run_command(command, log_path=log_path)
        except RuntimeError as exc:
            failure = {**candidate, "error": str(exc), "log_path": str(log_path.resolve())}
            failures.append(failure)
            print(f"Skipping failed download {candidate.get('id')}: {exc}")
            if continue_on_error:
                continue
            raise
        match = existing_download(paths.raw, video_id)
        if match is not None:
            downloaded.append({**candidate, "local_path": str(match)})

    output_path = paths.candidates / "downloads.jsonl"
    write_jsonl(output_path, downloaded)
    failure_path = paths.candidates / "download_failures.jsonl"
    write_jsonl(failure_path, failures)
    print(f"Wrote {len(downloaded)} download records to {output_path}")
    if failures:
        print(f"Wrote {len(failures)} download failures to {failure_path}")


def probe_video(path: Path, cv2: Any) -> tuple[float, int]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()
    if fps <= 0.0 or frame_count < 2:
        raise RuntimeError(f"Unusable video metadata: {path}")
    return fps, frame_count


def sample_gray_frames(path: Path, analysis_fps: float, cv2: Any) -> tuple[list[np.ndarray], list[float]]:
    source_fps, frame_count = probe_video(path, cv2)
    stride = max(1, int(round(source_fps / analysis_fps)))
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    times: list[float] = []
    try:
        for frame_index in range(0, frame_count, stride):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
            frames.append(gray)
            times.append(float(frame_index) / source_fps)
    finally:
        capture.release()
    return frames, times


def detect_scene_cuts(frames: list[np.ndarray], times: list[float], threshold: float, cv2: Any) -> list[float]:
    cuts = []
    previous_hist = None
    for frame, timestamp in zip(frames, times):
        hist = cv2.calcHist([frame], [0], None, [64], [0, 256])
        cv2.normalize(hist, hist)
        if previous_hist is not None:
            correlation = float(cv2.compareHist(previous_hist, hist, cv2.HISTCMP_CORREL))
            if correlation < threshold:
                cuts.append(timestamp)
        previous_hist = hist
    return cuts


def motion_score(frames: list[np.ndarray], cv2: Any) -> float:
    if len(frames) < 2:
        return 0.0
    scores = []
    for left, right in zip(frames[:-1], frames[1:]):
        flow = cv2.calcOpticalFlowFarneback(left, right, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, _angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        scores.append(float(np.percentile(magnitude, 75)))
    return float(np.median(np.asarray(scores, dtype=np.float32))) if scores else 0.0


def windows_between_cuts(duration: float, cuts: list[float], target_seconds: float) -> list[tuple[float, float]]:
    boundaries = [0.0] + sorted(cut for cut in cuts if 0.0 < cut < duration) + [duration]
    windows = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end - start >= target_seconds:
            windows.append((start, start + target_seconds))
    return windows


def extract_segment(source: Path, output: Path, start: float, duration: float) -> None:
    require_tool("ffmpeg")
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(source),
        "-t",
        f"{duration:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-an",
        str(output),
    ]
    run_command(command)


def segment(config: dict[str, Any], paths: Paths) -> None:
    cv2 = import_cv2()
    segment_cfg = config["segment"]
    downloads = read_jsonl(paths.candidates / "downloads.jsonl")
    if not downloads:
        raise RuntimeError("No downloads found. Run the download stage first.")

    target_seconds = float(segment_cfg["target_clip_seconds"])
    analysis_fps = float(segment_cfg["analysis_fps"])
    cut_threshold = float(segment_cfg["scene_cut_threshold"])
    min_motion = float(segment_cfg["min_motion_score"])
    max_windows = int(segment_cfg["max_windows_per_video"])
    continue_on_error = bool(segment_cfg.get("continue_on_error", False))
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for download_record in downloads:
        source = Path(download_record["local_path"])
        try:
            source_fps, frame_count = probe_video(source, cv2)
            duration = frame_count / source_fps
            frames, times = sample_gray_frames(source, analysis_fps, cv2)
            cuts = detect_scene_cuts(frames, times, cut_threshold, cv2)
            candidate_windows = windows_between_cuts(duration, cuts, target_seconds)

            scored = []
            for start, end in candidate_windows:
                window_frames = [frame for frame, t in zip(frames, times) if start <= t < end]
                score = motion_score(window_frames, cv2)
                if score >= min_motion:
                    scored.append((score, start, end))
            scored.sort(reverse=True)

            for local_index, (score, start, end) in enumerate(scored[:max_windows]):
                segment_id = f"{source.stem}_seg_{local_index:03d}"
                output = paths.segments / f"{segment_id}.mp4"
                extract_segment(source, output, start, end - start)
                records.append(
                    {
                        "segment_id": segment_id,
                        "path": str(output.resolve()),
                        "source_path": str(source.resolve()),
                        "youtube_id": download_record.get("id"),
                        "title": download_record.get("title"),
                        "start_seconds": start,
                        "end_seconds": end,
                        "duration_seconds": end - start,
                        "motion_score": score,
                        "scene_cut_count_in_source": len(cuts),
                    }
                )
        except RuntimeError as exc:
            failure = {**download_record, "error": str(exc)}
            failures.append(failure)
            print(f"Skipping failed segment source {source}: {exc}")
            if continue_on_error:
                continue
            raise

    output_path = paths.candidates / "segments_manifest.jsonl"
    write_jsonl(output_path, records)
    failure_path = paths.candidates / "segment_failures.jsonl"
    write_jsonl(failure_path, failures)
    print(f"Wrote {len(records)} high-motion segments to {output_path}")
    if failures:
        print(f"Wrote {len(failures)} segment failures to {failure_path}")


def build_clips(config: dict[str, Any], paths: Paths, overwrite: bool) -> None:
    segments = read_jsonl(paths.candidates / "segments_manifest.jsonl")
    if not segments:
        raise RuntimeError("No segments found. Run the segment stage first.")
    clip_cfg = config["clip_dataset"]
    dataset_name = config["dataset_name"]
    output_dir = paths.clip_sets / dataset_name
    train_count = int(clip_cfg.get("train_count", 0))
    test_count = int(clip_cfg.get("test_count", 0))
    if train_count > 0 or test_count > 0:
        needed = train_count + test_count
        if len(segments) < needed:
            raise RuntimeError(f"Need {needed} segments for train/test split, found {len(segments)}.")
        train_segments = segments[:train_count]
        test_segments = segments[train_count:needed]
        command = [
            sys.executable,
            "src/train/build_clip_dataset.py",
            "--train-input",
            *[record["path"] for record in train_segments],
            "--test-input",
            *[record["path"] for record in test_segments],
            "--output-dir",
            str(output_dir),
            "--dataset-name",
            dataset_name,
            "--target-count",
            str(needed),
            "--train-count",
            str(train_count),
            "--test-count",
            str(test_count),
            "--clip-frames",
            str(int(clip_cfg["clip_frames"])),
            "--fps",
            str(float(clip_cfg["fps"])),
            "--target-size",
            str(int(clip_cfg["target_size"])),
            "--max-clips-per-source",
            str(int(clip_cfg.get("max_clips_per_source", 0))),
            "--require-target-count",
        ]
    else:
        command = [
            sys.executable,
            "src/train/build_clip_dataset.py",
            "--input",
            *[record["path"] for record in segments],
            "--output-dir",
            str(output_dir),
            "--dataset-name",
            dataset_name,
            "--target-count",
            str(int(clip_cfg["target_count"])),
            "--clip-frames",
            str(int(clip_cfg["clip_frames"])),
            "--fps",
            str(float(clip_cfg["fps"])),
            "--target-size",
            str(int(clip_cfg["target_size"])),
        ]
        if int(clip_cfg.get("max_clips_per_source", 0)) > 0:
            command.extend(["--max-clips-per-source", str(int(clip_cfg["max_clips_per_source"]))])
    if overwrite:
        command.append("--overwrite")
    run_command(command)


# --- validate-local + cleanup-intermediates ----------------------------------
#
# Network-free walks of the local tree resolved from the same config the rest
# of the pipeline uses. Used by both scene-distinct and high-camera-motion seeds
# (they share this entrypoint).
#
# Source-of-truth (never auto-deleted):
#   candidates/*.jsonl, raw/*.mp4, segments/*.mp4
# Intermediates (safe to delete, derivable from raw + this config):
#   clip_sets/<dataset_name>/, logs/, raw/*.part, raw/*.ytdl, raw/*.tmp


PARTIAL_DOWNLOAD_SUFFIXES = (".part", ".ytdl", ".tmp")


def file_byte_size(path: Path) -> int:
    return path.stat().st_size if path.exists() and path.is_file() else 0


def directory_byte_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            total += entry.stat().st_size
    return total


def ffprobe_video(path: Path) -> dict[str, Any]:
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe is required for validate-local; install ffmpeg.")
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,nb_read_packets,duration",
        "-count_packets",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    payload = json.loads(result.stdout or "{}")
    streams = payload.get("streams") or []
    if not streams:
        raise RuntimeError(f"ffprobe found no video stream in {path}")
    stream = streams[0]
    packets = stream.get("nb_read_packets")
    if packets is not None and int(packets) < 1:
        raise RuntimeError(f"ffprobe reports zero packets in {path}")
    return stream


def validate_local(config: dict[str, Any], paths: Paths, *, sample_probes: int) -> None:
    downloads = read_jsonl(paths.candidates / "downloads.jsonl")
    segments = read_jsonl(paths.candidates / "segments_manifest.jsonl")

    missing_raw: list[dict[str, Any]] = []
    present_raw: list[Path] = []
    for record in downloads:
        local = record.get("local_path")
        if not local:
            missing_raw.append({"id": record.get("id"), "reason": "no local_path in downloads.jsonl"})
            continue
        local_path = Path(local)
        if local_path.exists():
            present_raw.append(local_path)
        else:
            missing_raw.append({"id": record.get("id"), "local_path": local, "reason": "missing on disk"})

    missing_segments: list[dict[str, Any]] = []
    present_segments: list[Path] = []
    for record in segments:
        seg_path = Path(record.get("path", ""))
        if seg_path.exists():
            present_segments.append(seg_path)
        else:
            missing_segments.append({"segment_id": record.get("segment_id"), "path": str(seg_path)})

    partials = [
        entry for entry in paths.raw.iterdir()
        if entry.is_file() and entry.suffix in PARTIAL_DOWNLOAD_SUFFIXES
    ] if paths.raw.exists() else []

    probe_targets = present_raw[: max(0, int(sample_probes))]
    probe_results: list[dict[str, Any]] = []
    for target in probe_targets:
        stream = ffprobe_video(target)
        probe_results.append(
            {
                "path": str(target),
                "codec": stream.get("codec_name"),
                "width": stream.get("width"),
                "height": stream.get("height"),
                "packets": stream.get("nb_read_packets"),
                "duration": stream.get("duration"),
            }
        )

    summary = {
        "root_dir": str(paths.root),
        "downloads_recorded": len(downloads),
        "raw_present": len(present_raw),
        "raw_missing": len(missing_raw),
        "segments_recorded": len(segments),
        "segments_present": len(present_segments),
        "segments_missing": len(missing_segments),
        "raw_partials": [str(p) for p in partials],
        "probes": probe_results,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if missing_raw:
        print(f"WARN: {len(missing_raw)} downloads.jsonl entries have no on-disk raw mp4.")
    if missing_segments:
        print(f"WARN: {len(missing_segments)} segments_manifest.jsonl entries have no on-disk segment mp4.")
    if partials:
        print(f"WARN: {len(partials)} partial yt-dlp files; clean with cleanup-intermediates --execute.")


def cleanup_intermediates(
    config: dict[str, Any],
    paths: Paths,
    *,
    execute: bool,
    include_raw: bool,
    include_segments: bool,
) -> None:
    dataset_name = str(config["dataset_name"])
    targets: list[tuple[str, Path]] = []

    clip_set_dir = paths.clip_sets / dataset_name
    if clip_set_dir.exists():
        targets.append((f"clip_sets/{dataset_name}", clip_set_dir))

    if paths.logs.exists():
        targets.append(("logs/", paths.logs))

    partials: list[Path] = []
    if paths.raw.exists():
        partials = [
            entry for entry in paths.raw.iterdir()
            if entry.is_file() and entry.suffix in PARTIAL_DOWNLOAD_SUFFIXES
        ]
    for partial in partials:
        targets.append((f"raw/{partial.name}", partial))

    if include_segments and paths.segments.exists():
        seg_files = [entry for entry in paths.segments.iterdir() if entry.is_file()]
        for seg_file in seg_files:
            targets.append((f"segments/{seg_file.name}  [SOURCE-OF-TRUTH]", seg_file))

    if include_raw and paths.raw.exists():
        raw_files = [
            entry for entry in paths.raw.iterdir()
            if entry.is_file() and entry.suffix not in PARTIAL_DOWNLOAD_SUFFIXES
        ]
        for raw_file in raw_files:
            targets.append((f"raw/{raw_file.name}  [SOURCE-OF-TRUTH]", raw_file))

    total = 0
    print(
        f"# cleanup-intermediates  root={paths.root}  execute={execute}  "
        f"include_raw={include_raw}  include_segments={include_segments}"
    )
    for label, path in targets:
        size = directory_byte_size(path) if path.is_dir() else file_byte_size(path)
        total += size
        print(f"  - {label:60s} {size:>14d} B  ({size / 1024 / 1024:.2f} MB)")
    print(f"  TOTAL reclaimable: {total} B  ({total / 1024 / 1024:.2f} MB)")

    if not execute:
        print("# DRY RUN -- pass --execute to actually delete these.")
        return

    for label, path in targets:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.is_file():
            path.unlink()
        print(f"  deleted {label}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine YouTube videos for high camera-motion single-path clips.")
    parser.add_argument(
        "stage",
        choices=("search", "download", "segment", "build-clips", "all", "validate-local", "cleanup-intermediates"),
    )
    parser.add_argument("--config", type=Path, default=Path("src/dataset_configs/youtube_high_camera_motion_seed.jsonc"))
    parser.add_argument("--overwrite", action="store_true", help="Overwrite local downloads and the final clip dataset.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="cleanup-intermediates: actually delete (default is dry-run).",
    )
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="cleanup-intermediates: also delete raw/*.mp4 (NOT recoverable without yt-dlp + network).",
    )
    parser.add_argument(
        "--include-segments",
        action="store_true",
        help="cleanup-intermediates: also delete segments/*.mp4 (recoverable from raw via segment stage).",
    )
    parser.add_argument(
        "--sample-probes",
        type=int,
        default=3,
        help="validate-local: how many raw mp4s to ffprobe (default 3).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config_file(args.config)
    paths = resolve_paths(config)
    if args.stage in {"search", "all"}:
        search(config, paths)
    if args.stage in {"download", "all"}:
        download(config, paths, overwrite=args.overwrite)
    if args.stage in {"segment", "all"}:
        segment(config, paths)
    if args.stage in {"build-clips", "all"}:
        build_clips(config, paths, overwrite=args.overwrite)
    if args.stage == "validate-local":
        validate_local(config, paths, sample_probes=args.sample_probes)
    if args.stage == "cleanup-intermediates":
        cleanup_intermediates(
            config,
            paths,
            execute=args.execute,
            include_raw=args.include_raw,
            include_segments=args.include_segments,
        )


if __name__ == "__main__":
    main()
