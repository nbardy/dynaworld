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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download user-curated YouTube spans as small local test clips.")
    parser.add_argument("stage", choices=("materialize", "download", "build-clips", "all"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/dataset_configs/youtube_curated_spans_64_4fps_16f.jsonc"),
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing raw downloads and clip datasets.")
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


if __name__ == "__main__":
    main()
