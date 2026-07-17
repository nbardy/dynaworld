#!/usr/bin/env python3
"""Build the broad single-sequence pretraining manifest.

This script owns the scale/same-view side of the data contract documented in
``research_notes/data_contract.md``. It inventories prepared clips, lazy video
windows, and camera-json sequences without copying media. Calibrated multicam
heldout-camera training still uses ``src/train/multicam_video_data.py``; this
builder only references those manifests for leakage guards unless a config
explicitly opts into multicam rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


try:
    from .script_paths import ensure_train_path, repo_path, repo_text as path_text
except ImportError:  # pragma: no cover - direct script execution
    from script_paths import ensure_train_path, repo_path, repo_text as path_text

ensure_train_path()
from config_utils import load_config_file  # noqa: E402
from train_artifacts import write_json, write_jsonl  # noqa: E402

SCHEMA_VERSION = "dynaworld_single_video_pretrain_manifest_v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path}:{line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object in {path}:{line_number}")
            records.append(payload)
    return records


def stable_hash(text: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{text}".encode("utf-8")).hexdigest()


def parse_frame_rate(value: str | None) -> float:
    if not value:
        return 0.0
    if "/" not in value:
        try:
            return float(value)
        except ValueError:
            return 0.0
    numerator_text, denominator_text = value.split("/", 1)
    try:
        numerator = float(numerator_text)
        denominator = float(denominator_text)
    except ValueError:
        return 0.0
    if denominator == 0:
        return 0.0
    return numerator / denominator


def stable_sample(records: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if limit <= 0 or len(records) <= limit:
        return list(records)
    ordered = sorted(records, key=lambda record: stable_hash(str(record["item_id"]), seed))
    return ordered[:limit]


def make_item_id(label: str, key: str, seed: int) -> str:
    cleaned = "".join(char if char.isalnum() else "_" for char in key).strip("_")
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    suffix = stable_hash(f"{label}:{key}", seed)[:10]
    return f"{label}__{cleaned[:80]}__{suffix}"


def split_allowed(record: dict[str, Any], include_splits: set[str]) -> bool:
    return str(record.get("split", "train")) in include_splits


def collect_heldout_paths(config: dict[str, Any]) -> set[str]:
    heldout: set[str] = set()
    for source in config.get("heldout_reference_manifests", []):
        path = repo_path(source["path"])
        splits = {str(value) for value in source.get("heldout_splits", ["test", "val", "eval", "heldout"])}
        path_keys = tuple(source.get("heldout_path_keys", ["source_path", "video_path", "target_video_path"]))
        list_path_keys = tuple(source.get("heldout_list_path_keys", ["heldout_video_paths", "heldout_source_paths"]))
        for record in read_jsonl(path):
            if splits and str(record.get("split", "train")) not in splits:
                continue
            for key in path_keys:
                value = record.get(key)
                if value:
                    heldout.add(path_text(repo_path(value)))
            for key in list_path_keys:
                for value in record.get(key, []) or []:
                    heldout.add(path_text(repo_path(value)))
    return heldout


def probe_video(path: Path) -> dict[str, Any] | None:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-show_entries",
        "stream=width,height,nb_frames,r_frame_rate,avg_frame_rate,duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    payload = json.loads(result.stdout or "{}")
    streams = [stream for stream in payload.get("streams", []) if stream.get("width") and stream.get("height")]
    stream = streams[0] if streams else {}
    duration = payload.get("format", {}).get("duration") or stream.get("duration")
    if duration is None:
        return None
    avg_rate = stream.get("avg_frame_rate")
    raw_rate = stream.get("r_frame_rate")
    fps = parse_frame_rate(avg_rate) or parse_frame_rate(raw_rate)
    return {
        "duration_seconds": float(duration),
        "width": int(stream.get("width", 0) or 0),
        "height": int(stream.get("height", 0) or 0),
        "nb_frames": stream.get("nb_frames"),
        "r_frame_rate": raw_rate,
        "avg_frame_rate": avg_rate,
        "fps": fps,
    }


def window_starts(duration: float, window_seconds: float, stride_seconds: float, max_windows: int) -> list[float]:
    if duration + 1.0e-6 < window_seconds:
        return []
    starts: list[float] = []
    cursor = 0.0
    latest = max(0.0, duration - window_seconds)
    while cursor <= latest + 1.0e-6:
        starts.append(round(cursor, 6))
        cursor += stride_seconds
        if max_windows > 0 and len(starts) >= max_windows:
            break
    return starts


def resolve_window_fps(window: dict[str, Any], video_info: dict[str, Any]) -> float:
    value = window.get("fps", 0.0)
    if isinstance(value, str) and value.lower() in {"natural", "native", "source"}:
        fps = float(video_info.get("fps", 0.0) or 0.0)
    else:
        fps = float(value or 0.0)
    if fps <= 0:
        raise ValueError(f"Could not resolve positive FPS from window={window} video_info={video_info}")
    return fps


def resolve_window_duration(window: dict[str, Any], fps: float) -> float:
    duration = window.get("duration_seconds")
    if duration is not None:
        return float(duration)
    return float(window["frame_count"]) / float(fps)


def source_limit(source: dict[str, Any], split: str) -> int:
    if split == "train":
        return int(source.get("max_train_records", 0) or 0)
    if split == "eval":
        return int(source.get("max_eval_records", 0) or 0)
    return int(source.get("max_records", 0) or 0)


def add_source_records(
    output: dict[str, list[dict[str, Any]]],
    records: list[dict[str, Any]],
    *,
    source: dict[str, Any],
    seed: int,
) -> None:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_split[str(record.get("split", "train"))].append(record)
    for split, split_records in by_split.items():
        capped = stable_sample(split_records, source_limit(source, split), seed)
        output[split].extend(capped)


def build_frame_clip_records(
    source: dict[str, Any],
    *,
    seed: int,
    heldout_paths: set[str],
) -> list[dict[str, Any]]:
    path = repo_path(source["path"])
    include_splits = {str(value) for value in source.get("include_splits", ["train"])}
    output_split = str(source.get("output_split", "train"))
    records: list[dict[str, Any]] = []
    for index, entry in enumerate(read_jsonl(path)):
        if not split_allowed(entry, include_splits):
            continue
        source_path = entry.get("source_path") or entry.get("video_path")
        normalized_source = path_text(repo_path(source_path)) if source_path else None
        split = "heldout" if normalized_source in heldout_paths and output_split == "train" else output_split
        item = dict(entry)
        item.update(
            {
                "item_id": make_item_id(str(source["label"]), str(entry.get("clip_id", index)), seed),
                "record_type": "frame_clip_sequence",
                "schema_version": SCHEMA_VERSION,
                "split": split,
                "original_split": str(entry.get("split", "train")),
                "source_label": str(source["label"]),
                "source_family": str(source["family"]),
                "source_manifest": path_text(path),
                "current_loader_compatible": True,
                "loader_contract": "train_video_token manifest_path + summary_sampled frames",
            }
        )
        records.append(item)
    return records


def path_split(path: Path, source: dict[str, Any]) -> str:
    if not bool(source.get("split_from_path", False)):
        return str(source.get("split", "train"))
    parts = set(path.parts)
    if "test" in parts or "heldout" in parts or "val" in parts:
        return "eval"
    return "train"


def build_video_window_records(
    source: dict[str, Any],
    *,
    seed: int,
    window: dict[str, Any],
    heldout_paths: set[str],
    probe_videos: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = repo_path(source["root"])
    glob = str(source.get("glob", "**/*.mp4"))
    videos = sorted(path for path in root.glob(glob) if path.is_file())
    records: list[dict[str, Any]] = []
    skipped = Counter()
    for video in videos:
        split = path_split(video, source)
        normalized = path_text(video)
        if normalized in heldout_paths and split == "train":
            split = "heldout"
        if split == "eval" and not bool(source.get("include_eval", False)):
            skipped["eval_path"] += 1
            continue
        if split == "heldout" and not bool(source.get("include_heldout", False)):
            skipped["heldout_path"] += 1
            continue
        if not probe_videos:
            skipped["probe_disabled"] += 1
            continue
        info = probe_video(video)
        if info is None:
            skipped["probe_failed"] += 1
            continue
        record_fps = resolve_window_fps(window, info)
        record_duration = resolve_window_duration(window, record_fps)
        starts = window_starts(
            duration=float(info["duration_seconds"]),
            window_seconds=record_duration,
            stride_seconds=float(source.get("stride_seconds", window["stride_seconds"])),
            max_windows=int(source.get("max_windows_per_video", 0) or 0),
        )
        if not starts:
            skipped["too_short"] += 1
            continue
        for window_index, start_seconds in enumerate(starts):
            key = f"{normalized}:w{window_index:04d}:{start_seconds:.3f}"
            records.append(
                {
                    "item_id": make_item_id(str(source["label"]), key, seed),
                    "record_type": "single_view_video_window",
                    "schema_version": SCHEMA_VERSION,
                    "split": split,
                    "source_label": str(source["label"]),
                    "source_family": str(source["family"]),
                    "video_path": normalized,
                    "source_path": normalized,
                    "sequence_dir": path_text(video.parent),
                    "frame_source": "explicit_video_window",
                    "start_seconds": start_seconds,
                    "duration_seconds": record_duration,
                    "fps": record_fps,
                    "frame_count": int(window["frame_count"]),
                    "target_size": int(window["target_size"]),
                    "image_crop_mode": str(window.get("image_crop_mode", "resize")),
                    "video_duration_seconds": float(info["duration_seconds"]),
                    "source_fps": float(info.get("fps", 0.0) or 0.0),
                    "source_r_frame_rate": info.get("r_frame_rate"),
                    "source_avg_frame_rate": info.get("avg_frame_rate"),
                    "video_width": int(info["width"]),
                    "video_height": int(info["height"]),
                    "current_loader_compatible": True,
                    "loader_contract": "train_video_token manifest_path + explicit_video_window video_path/start_seconds",
                }
            )
    return records, {"video_count": len(videos), "skipped": dict(skipped)}


def record_window_duration(record: dict[str, Any], default_duration: float) -> float:
    for key in ("synchronized_available_seconds", "source_available_seconds", "duration_seconds"):
        value = record.get(key)
        if value is not None:
            return float(value)
    source_video = record.get("source_video")
    if isinstance(source_video, dict) and source_video.get("duration_seconds") is not None:
        return max(0.0, float(source_video["duration_seconds"]) - float(record.get("source_start_seconds", 0.0)))
    return default_duration


def build_multicam_window_records(
    source: dict[str, Any],
    *,
    seed: int,
    window: dict[str, Any],
) -> list[dict[str, Any]]:
    path = repo_path(source["path"])
    include_splits = {str(value) for value in source.get("include_splits", ["train2_holdout1", "val"])}
    output_split = str(source.get("output_split", "train"))
    records: list[dict[str, Any]] = []
    for index, entry in enumerate(read_jsonl(path)):
        if not split_allowed(entry, include_splits):
            continue
        starts = window_starts(
            duration=record_window_duration(entry, float(window["duration_seconds"])),
            window_seconds=float(window["duration_seconds"]),
            stride_seconds=float(source.get("stride_seconds", window["stride_seconds"])),
            max_windows=int(source.get("max_windows_per_record", 1) or 1),
        )
        base_source = float(entry.get("source_start_seconds", 0.0))
        base_target = float(entry.get("target_start_seconds", base_source))
        for window_index, offset in enumerate(starts or [0.0]):
            sample_id = str(entry.get("sample_id", f"record_{index:04d}"))
            item = dict(entry)
            item.update(
                {
                    "item_id": make_item_id(str(source["label"]), f"{sample_id}:w{window_index:04d}", seed),
                    "record_type": "multicam_pair_window",
                    "schema_version": SCHEMA_VERSION,
                    "split": output_split,
                    "original_split": str(entry.get("split", "train")),
                    "source_label": str(source["label"]),
                    "source_family": str(source["family"]),
                    "source_manifest": path_text(path),
                    "sample_id": f"{sample_id}_w{window_index:04d}",
                    "source_start_seconds": round(base_source + offset, 6),
                    "target_start_seconds": round(base_target + offset, 6),
                    "duration_seconds": float(window["duration_seconds"]),
                    "fps": float(window["fps"]),
                    "frame_count": int(window["frame_count"]),
                    "target_size": int(window["target_size"]),
                    "image_crop_mode": str(window.get("image_crop_mode", "resize")),
                    "current_loader_compatible": True,
                    "loader_contract": "multicam_video_data manifest record; trainer still selects one record unless Worker A batches records",
                }
            )
            records.append(item)
    return records


def build_blender_sequence_records(
    source: dict[str, Any],
    *,
    seed: int,
    window: dict[str, Any],
) -> list[dict[str, Any]]:
    root = repo_path(source["root"])
    records: list[dict[str, Any]] = []
    for camera_json in sorted(root.glob(str(source.get("glob", "*/cameras.json")))):
        sequence_dir = camera_json.parent
        frames = sorted(sequence_dir.glob("frame_*.png"))
        if len(frames) < int(window["frame_count"]):
            continue
        key = path_text(sequence_dir)
        records.append(
            {
                "item_id": make_item_id(str(source["label"]), key, seed),
                "record_type": "synthetic_camera_json_sequence",
                "schema_version": SCHEMA_VERSION,
                "split": str(source.get("split", "train")),
                "source_label": str(source["label"]),
                "source_family": str(source["family"]),
                "sequence_dir": path_text(sequence_dir),
                "frames_dir": path_text(sequence_dir),
                "camera_json": path_text(camera_json),
                "frame_source": "camera_json",
                "frame_count": int(window["frame_count"]),
                "available_frame_count": len(frames),
                "target_size": int(window["target_size"]),
                "image_crop_mode": str(window.get("image_crop_mode", "resize")),
                "current_loader_compatible": True,
                "loader_contract": "train_video_token manifest_path + camera_json",
            }
        )
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_split = Counter(str(record.get("split", "train")) for record in records)
    by_source = Counter(str(record.get("source_label", "unknown")) for record in records)
    by_type = Counter(str(record.get("record_type", "unknown")) for record in records)
    compatible = Counter(str(bool(record.get("current_loader_compatible", False))) for record in records)
    return {
        "count": len(records),
        "by_split": dict(sorted(by_split.items())),
        "by_source_label": dict(sorted(by_source.items())),
        "by_record_type": dict(sorted(by_type.items())),
        "current_loader_compatible": {
            "true": compatible.get("True", 0),
            "false": compatible.get("False", 0),
        },
    }


def build(config: dict[str, Any], *, limit: int, eval_limit: int, dry_run: bool, probe_videos: bool) -> dict[str, Any]:
    seed = int(config.get("seed", 0))
    window = dict(config["window"])
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_stats: dict[str, Any] = {}
    heldout_paths = collect_heldout_paths(config)

    for source in config.get("frame_clip_manifests", []):
        records = build_frame_clip_records(source, seed=seed, heldout_paths=heldout_paths)
        add_source_records(output, records, source=source, seed=seed)
        source_stats[str(source["label"])] = summarize(records)

    for source in config.get("single_view_video_roots", []):
        records, stats = build_video_window_records(
            source,
            seed=seed,
            window=window,
            heldout_paths=heldout_paths,
            probe_videos=probe_videos,
        )
        add_source_records(output, records, source=source, seed=seed)
        source_stats[str(source["label"])] = {**summarize(records), **stats}

    for source in config.get("multicam_manifests", []):
        records = build_multicam_window_records(source, seed=seed, window=window)
        add_source_records(output, records, source=source, seed=seed)
        source_stats[str(source["label"])] = summarize(records)

    for source in config.get("blender_sequence_roots", []):
        records = build_blender_sequence_records(source, seed=seed, window=window)
        add_source_records(output, records, source=source, seed=seed)
        source_stats[str(source["label"])] = summarize(records)

    train_target = limit if limit > 0 else int(config.get("target_train_items", 0) or 0)
    eval_target = eval_limit if eval_limit > 0 else int(config.get("target_eval_items", 0) or 0)
    train = stable_sample(output.get("train", []), train_target, seed)
    eval_records = stable_sample(output.get("eval", []), eval_target, seed)
    heldout = output.get("heldout", [])
    all_selected = train + eval_records + heldout

    output_dir = repo_path(config["output_dir"])
    paths = {
        "manifest_path": output_dir / "manifest.jsonl",
        "train_manifest_path": output_dir / "train_manifest.jsonl",
        "eval_manifest_path": output_dir / "eval_manifest.jsonl",
        "heldout_manifest_path": output_dir / "heldout_manifest.jsonl",
        "dataset_path": output_dir / "dataset.json",
    }
    dataset = {
        "dataset_name": str(config["dataset_name"]),
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "target_train_items": int(config.get("target_train_items", 0) or 0),
        "target_eval_items": int(config.get("target_eval_items", 0) or 0),
        "selected_train_items": len(train),
        "selected_eval_items": len(eval_records),
        "heldout_reference_path_count": len(heldout_paths),
        "window": window,
        "paths": {name: path_text(path) for name, path in paths.items()},
        "summary": summarize(all_selected),
        "source_stats": source_stats,
        "schema_notes": [
            "frame_clip_sequence and synthetic_camera_json_sequence records are load_manifest_sequences compatible today.",
            "single_view_video_window records do not copy assets; train_video_token loads video_path from start_seconds at fps.",
            "heldout matching is path-level/camera-level, not scene-level; known test/val target video paths are not emitted as train single-view windows.",
        ],
    }

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(paths["manifest_path"], all_selected, compact=True)
        write_jsonl(paths["train_manifest_path"], train, compact=True)
        write_jsonl(paths["eval_manifest_path"], eval_records, compact=True)
        write_jsonl(paths["heldout_manifest_path"], heldout, compact=True)
        write_json(paths["dataset_path"], dataset)

    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a deterministic Dynaworld single-video pretrain manifest.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/dataset_configs/single_video_pretrain_1k_manifest.jsonc"),
        help="JSONC manifest-builder config.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Override config output_dir.")
    parser.add_argument("--limit", type=int, default=0, help="Limit selected train items after source caps.")
    parser.add_argument("--eval-limit", type=int, default=0, help="Limit selected eval items after source caps.")
    parser.add_argument("--dry-run", action="store_true", help="Print counts without writing manifests.")
    parser.add_argument("--no-probe-videos", action="store_true", help="Skip ffprobe-backed video-window discovery.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config_file(repo_path(args.config))
    if args.output_dir is not None:
        config["output_dir"] = path_text(repo_path(args.output_dir))
    dataset = build(
        config,
        limit=int(args.limit),
        eval_limit=int(args.eval_limit),
        dry_run=bool(args.dry_run),
        probe_videos=not bool(args.no_probe_videos),
    )
    print(json.dumps(dataset["summary"], indent=2, sort_keys=True))
    print(json.dumps({"paths": dataset["paths"], "dry_run": bool(args.dry_run)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
