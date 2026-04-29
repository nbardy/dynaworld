from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


SRC_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR / "train"))

from config_utils import load_config_file  # noqa: E402


@dataclass(frozen=True)
class Paths:
    root: Path
    candidates: Path
    raw: Path
    clip_sets: Path
    logs: Path


def resolve_paths(config: dict[str, Any]) -> Paths:
    root = Path(config["root_dir"])
    paths = Paths(
        root=root,
        candidates=root / "candidates",
        raw=root / "raw",
        clip_sets=root / "clip_sets",
        logs=root / "logs",
    )
    for path in paths.__dict__.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
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


def yt_dlp_command() -> list[str]:
    executable = shutil.which("yt-dlp")
    if executable is not None:
        return [executable]
    return [sys.executable, "-m", "yt_dlp"]


def resolve_input_path(value: str, config_dir: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    for candidate in (REPO_ROOT / path, config_dir / path, Path.cwd() / path):
        if candidate.exists():
            return candidate.resolve()
    return (REPO_ROOT / path).resolve()


def parse_time_seconds(value: str | int | float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    parts = str(value).strip().split(":")
    if not parts:
        return None
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60.0 + float(part)
    return seconds


def youtube_id(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc.endswith("youtu.be"):
        return parsed.path.strip("/")
    if "/shorts/" in parsed.path:
        return parsed.path.split("/shorts/", 1)[1].split("/", 1)[0]
    query_id = parse_qs(parsed.query).get("v", [None])[0]
    if query_id:
        return query_id
    fallback = re.sub(r"[^A-Za-z0-9_-]+", "_", url).strip("_")
    return fallback[:64] or "youtube"


def clean_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return cleaned[:120] or "clip"


def span_clip_id(video_id: str, segment_index: int, start: float | None, end: float | None) -> str:
    if start is None or end is None:
        return clean_id(f"{video_id}_whole_{segment_index:03d}")
    start_ms = int(round(start * 1000.0))
    end_ms = int(round(end * 1000.0))
    return clean_id(f"{video_id}_seg_{segment_index:03d}_s{start_ms:08d}_e{end_ms:08d}")


def span_from_record(
    *,
    record: dict[str, Any],
    source_manifest: str | None,
    source_record_index: int | None,
    segment: dict[str, Any] | None,
    segment_index: int,
    defaults: dict[str, Any],
) -> dict[str, Any]:
    url = str(record["url"])
    video_id = youtube_id(url)
    segment = segment or record
    whole_video = bool(record.get("whole_video", False)) and "start_seconds" not in segment and "start_time" not in segment
    default_duration = float(defaults.get("default_segment_seconds", 8.0))
    start = None if whole_video else parse_time_seconds(segment.get("start_seconds", segment.get("start_time")))
    end = None if whole_video else parse_time_seconds(segment.get("end_seconds", segment.get("end_time")))
    if start is not None and end is None:
        end = start + default_duration
    if start is None and end is not None:
        raise ValueError(f"Span has end without start for url={url}")
    if start is not None and end is not None and end <= start:
        raise ValueError(f"Span end must be greater than start for url={url}: start={start} end={end}")

    clip_id = clean_id(str(record.get("clip_id") or span_clip_id(video_id, segment_index, start, end)))
    output: dict[str, Any] = {
        "clip_id": clip_id,
        "url": url,
        "youtube_id": video_id,
        "title": record.get("title"),
        "source": record.get("source", "user_curated"),
        "notes": record.get("notes"),
        "split": record.get("split", defaults.get("split", "train")),
        "whole_video": whole_video,
        "segment_index": segment_index,
    }
    if source_manifest is not None:
        output["source_manifest"] = source_manifest
    if source_record_index is not None:
        output["source_record_index"] = source_record_index
    if start is not None and end is not None:
        output.update(
            {
                "start_seconds": float(start),
                "end_seconds": float(end),
                "duration_seconds": float(end - start),
                "start_time": segment.get("start_time", record.get("start_time")),
                "end_time": segment.get("end_time", record.get("end_time")),
            }
        )
    return output


def flatten_records(config: dict[str, Any], config_dir: Path) -> list[dict[str, Any]]:
    defaults = dict(config.get("defaults", {}))
    records: list[dict[str, Any]] = []

    for manifest_value in config.get("source_manifests", []):
        manifest_label = str(manifest_value)
        manifest_path = resolve_input_path(manifest_label, config_dir)
        for source_index, record in enumerate(read_jsonl(manifest_path)):
            if record.get("segments"):
                for segment_index, segment in enumerate(record["segments"]):
                    records.append(
                        span_from_record(
                            record=record,
                            source_manifest=manifest_label,
                            source_record_index=source_index,
                            segment=segment,
                            segment_index=segment_index,
                            defaults=defaults,
                        )
                    )
            else:
                records.append(
                    span_from_record(
                        record=record,
                        source_manifest=manifest_label,
                        source_record_index=source_index,
                        segment=None,
                        segment_index=0,
                        defaults=defaults,
                    )
                )

    for index, record in enumerate(config.get("records", [])):
        records.append(
            span_from_record(
                record=record,
                source_manifest=None,
                source_record_index=None,
                segment=None,
                segment_index=index,
                defaults=defaults,
            )
        )

    seen: set[str] = set()
    for record in records:
        clip_id = str(record["clip_id"])
        if clip_id in seen:
            raise ValueError(f"Duplicate clip_id in curated spans: {clip_id}")
        seen.add(clip_id)
    return records


def materialize(config: dict[str, Any], config_dir: Path, paths: Paths) -> None:
    records = flatten_records(config, config_dir)
    output_path = paths.candidates / "curated_spans.jsonl"
    write_jsonl(output_path, records)
    split_counts: dict[str, int] = {}
    for record in records:
        split = str(record.get("split", "train"))
        split_counts[split] = split_counts.get(split, 0) + 1
    print(f"Wrote {len(records)} curated spans to {output_path} splits={split_counts}")


def existing_download(raw_dir: Path, clip_id: str) -> Path | None:
    matches = sorted(raw_dir.glob(f"{clip_id}.*"))
    return matches[-1].resolve() if matches else None


def download(config: dict[str, Any], paths: Paths, overwrite: bool) -> None:
    records = read_jsonl(paths.candidates / "curated_spans.jsonl")
    if not records:
        raise RuntimeError("No curated spans found. Run the materialize stage first.")

    download_cfg = dict(config.get("download", {}))
    max_height = int(download_cfg.get("max_height", 360))
    cookies_from_browser = download_cfg.get("cookies_from_browser")
    continue_on_error = bool(download_cfg.get("continue_on_error", False))
    downloaded: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        clip_id = str(record["clip_id"])
        if not overwrite:
            existing = existing_download(paths.raw, clip_id)
            if existing is not None:
                downloaded.append({**record, "local_path": str(existing)})
                continue

        output_template = str(paths.raw / f"{clip_id}.%(ext)s")
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
            str(record["url"]),
        ]
        if not record.get("whole_video", False):
            command[len(base_command) : len(base_command)] = [
                "--download-sections",
                f"*{float(record['start_seconds']):.3f}-{float(record['end_seconds']):.3f}",
                "--force-keyframes-at-cuts",
            ]
        if overwrite:
            command[len(base_command) : len(base_command)] = ["--force-overwrites"]
        if cookies_from_browser:
            command[len(base_command) : len(base_command)] = ["--cookies-from-browser", str(cookies_from_browser)]

        log_path = paths.logs / f"download_{index:04d}_{clip_id}.log"
        try:
            run_command(command, log_path=log_path)
        except RuntimeError as exc:
            failure = {**record, "error": str(exc), "log_path": str(log_path.resolve())}
            failures.append(failure)
            print(f"Skipping failed curated download {clip_id}: {exc}")
            if continue_on_error:
                continue
            raise
        match = existing_download(paths.raw, clip_id)
        if match is None:
            failure = {**record, "error": "yt-dlp completed but no output file was found", "log_path": str(log_path.resolve())}
            failures.append(failure)
            print(f"Skipping missing output for curated download {clip_id}")
            if continue_on_error:
                continue
            raise RuntimeError(f"No output file was found for {clip_id}")
        downloaded.append({**record, "local_path": str(match)})

    output_path = paths.candidates / "downloads.jsonl"
    write_jsonl(output_path, downloaded)
    failure_path = paths.candidates / "download_failures.jsonl"
    write_jsonl(failure_path, failures)
    print(f"Wrote {len(downloaded)} curated download records to {output_path}")
    if failures:
        print(f"Wrote {len(failures)} curated download failures to {failure_path}")


def build_clips(config: dict[str, Any], paths: Paths, overwrite: bool) -> None:
    downloads = read_jsonl(paths.candidates / "downloads.jsonl")
    if not downloads:
        raise RuntimeError("No curated downloads found. Run the download stage first.")

    clip_cfg = dict(config["clip_dataset"])
    dataset_name = str(config["dataset_name"])
    output_dir = paths.clip_sets / dataset_name
    train_paths = [record["local_path"] for record in downloads if record.get("split", "train") == "train"]
    test_paths = [record["local_path"] for record in downloads if record.get("split", "train") == "test"]
    if not train_paths and not test_paths:
        raise RuntimeError("No train or test downloads available to build clips.")

    total_count = len(train_paths) + len(test_paths)
    command = [
        sys.executable,
        "src/train/build_clip_dataset.py",
        "--output-dir",
        str(output_dir),
        "--dataset-name",
        dataset_name,
        "--target-count",
        str(total_count),
        "--clip-frames",
        str(int(clip_cfg["clip_frames"])),
        "--fps",
        str(float(clip_cfg["fps"])),
        "--target-size",
        str(int(clip_cfg["target_size"])),
        "--max-clips-per-source",
        str(int(clip_cfg.get("max_clips_per_source", 0))),
        "--source-schedule",
        str(clip_cfg.get("source_schedule", "sequential")),
    ]
    if train_paths:
        command.extend(["--train-input", *train_paths, "--train-count", str(len(train_paths))])
    if test_paths:
        command.extend(["--test-input", *test_paths, "--test-count", str(len(test_paths))])
    if bool(clip_cfg.get("require_target_count", False)):
        command.append("--require-target-count")
    if overwrite:
        command.append("--overwrite")
    run_command(command)


# --- validate-local + cleanup-intermediates ----------------------------------
#
# These stages do NOT touch the network. They walk the local tree resolved from
# the same config the rest of the pipeline uses.
#
# Source-of-truth (never auto-deleted):
#   candidates/*.jsonl, raw/*.mp4
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


def validate_local(config: dict[str, Any], config_dir: Path, paths: Paths, *, sample_probes: int) -> None:
    expected_records = flatten_records(config, config_dir)
    candidates_path = paths.candidates / "curated_spans.jsonl"
    materialized = read_jsonl(candidates_path)

    missing_raw: list[dict[str, Any]] = []
    present_raw: list[Path] = []
    for record in expected_records:
        clip_id = str(record["clip_id"])
        # `existing_download` globs `clip_id.*`, which can select a `.part` if a
        # download was interrupted. Filter those out here -- the partial is a
        # separate signal reported below.
        match = existing_download(paths.raw, clip_id)
        if match is not None and match.suffix in PARTIAL_DOWNLOAD_SUFFIXES:
            match = None
        if match is None:
            missing_raw.append({"clip_id": clip_id, "url": record.get("url")})
        else:
            present_raw.append(match)

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
        "config": str(config_dir),
        "root_dir": str(paths.root),
        "expected_spans": len(expected_records),
        "materialized_spans": len(materialized),
        "raw_present": len(present_raw),
        "raw_missing": len(missing_raw),
        "raw_partials": [str(p) for p in partials],
        "probes": probe_results,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if missing_raw:
        print(f"WARN: {len(missing_raw)} curated spans have no matching raw mp4 (rerun yt-dlp download stage).")
    if partials:
        print(f"WARN: {len(partials)} partial yt-dlp files; clean with cleanup-intermediates --execute.")


def cleanup_intermediates(
    config: dict[str, Any],
    paths: Paths,
    *,
    execute: bool,
    include_raw: bool,
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

    raw_targets: list[Path] = []
    if include_raw and paths.raw.exists():
        raw_targets = [
            entry for entry in paths.raw.iterdir()
            if entry.is_file() and entry.suffix not in PARTIAL_DOWNLOAD_SUFFIXES
        ]
        for raw_file in raw_targets:
            targets.append((f"raw/{raw_file.name}  [SOURCE-OF-TRUTH]", raw_file))

    total = 0
    print(f"# cleanup-intermediates  root={paths.root}  execute={execute}  include_raw={include_raw}")
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
    parser = argparse.ArgumentParser(description="Download user-curated YouTube spans as small local test clips.")
    parser.add_argument(
        "stage",
        choices=("materialize", "download", "build-clips", "all", "validate-local", "cleanup-intermediates"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/dataset_configs/youtube_curated_spans_64_4fps_16f.jsonc"),
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing raw downloads and clip datasets.")
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
    config_dir = args.config.resolve().parent
    if args.stage in {"materialize", "all"}:
        materialize(config, config_dir, paths)
    if args.stage in {"download", "all"}:
        download(config, paths, overwrite=args.overwrite)
    if args.stage in {"build-clips", "all"}:
        build_clips(config, paths, overwrite=args.overwrite)
    if args.stage == "validate-local":
        validate_local(config, config_dir, paths, sample_probes=args.sample_probes)
    if args.stage == "cleanup-intermediates":
        cleanup_intermediates(config, paths, execute=args.execute, include_raw=args.include_raw)


if __name__ == "__main__":
    main()
