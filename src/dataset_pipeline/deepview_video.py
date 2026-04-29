from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


SRC_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR / "train"))

from config_utils import load_config_file  # noqa: E402


VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v"}


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def resolve_root(config: dict[str, Any]) -> Path:
    root = resolve_path(config["root_dir"])
    for child in ("raw", "extracted", "metadata", "logs"):
        (root / child).mkdir(parents=True, exist_ok=True)
    return root


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def scene_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    records = config.get("scenes", [])
    if not isinstance(records, list):
        raise RuntimeError("DeepView config expected `scenes` to be a list.")
    return records


def selected_scenes(config: dict[str, Any], *, all_scenes: bool) -> list[dict[str, Any]]:
    records = scene_records(config)
    if all_scenes:
        return records
    selected_names = set(config["download"].get("scene_names") or [])
    selected = [record for record in records if record.get("name") in selected_names]
    missing = sorted(selected_names - {record.get("name") for record in selected})
    if missing:
        raise RuntimeError(f"Configured DeepView scenes were not found: {missing}")
    return selected


def archive_path(root: Path, scene: dict[str, Any]) -> Path:
    return root / "raw" / f"{scene['name']}.zip"


def list_scenes(config: dict[str, Any], root: Path) -> None:
    total_bytes = 0
    for scene in scene_records(config):
        size = int(scene.get("expected_size_bytes") or 0)
        total_bytes += size
        local_path = archive_path(root, scene)
        status = "local" if local_path.exists() else "remote"
        print(f"{scene['name']}\t{size / 1_000_000_000.0:.2f} GB\t{status}\t{scene['url']}")
    print(f"TOTAL\t{total_bytes / 1_000_000_000.0:.2f} GB")


def download_file(url: str, output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        print(f"Already exists: {output_path}")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "dynaworld-deepview-ingest"})
    with urllib.request.urlopen(request) as response, tmp_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    tmp_path.replace(output_path)
    print(f"Downloaded: {output_path}")


def download(config: dict[str, Any], root: Path, *, all_scenes: bool, overwrite: bool | None) -> None:
    should_overwrite = bool(config["download"].get("overwrite", False)) if overwrite is None else overwrite
    records = []
    for scene in selected_scenes(config, all_scenes=all_scenes):
        output_path = archive_path(root, scene)
        download_file(str(scene["url"]), output_path, overwrite=should_overwrite)
        records.append(
            {
                "name": scene["name"],
                "url": scene["url"],
                "expected_size_bytes": scene.get("expected_size_bytes"),
                "local_path": str(output_path.resolve()),
            }
        )
    write_jsonl(root / "metadata" / "selected_scenes.jsonl", records)


def extract_zip(path: Path, output_root: Path, overwrite: bool) -> Path:
    output_dir = output_root / path.stem
    if output_dir.exists():
        if not overwrite:
            print(f"Already extracted: {output_dir}")
            return output_dir
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        archive.extractall(output_dir)
    print(f"Extracted: {output_dir}")
    return output_dir


def extract(config: dict[str, Any], root: Path, *, all_scenes: bool, overwrite: bool | None) -> None:
    should_overwrite = bool(config["extract"].get("overwrite", False)) if overwrite is None else overwrite
    raw_paths = [archive_path(root, scene) for scene in selected_scenes(config, all_scenes=all_scenes)]
    missing = [path for path in raw_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing DeepView raw zip assets: {[str(path) for path in missing]}")
    for path in raw_paths:
        extract_zip(path, root / "extracted", overwrite=should_overwrite)


def ffprobe(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return {"readable": False, "error": result.stderr.strip()}
    payload = json.loads(result.stdout)
    stream = payload.get("streams", [{}])[0]
    duration = float(payload.get("format", {}).get("duration") or 0.0)
    return {
        "readable": True,
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "r_frame_rate": stream.get("r_frame_rate"),
        "nb_frames": stream.get("nb_frames"),
        "duration_seconds": duration,
    }


def deep_find_scene_dirs(root: Path) -> list[Path]:
    extracted = root / "extracted"
    if not extracted.exists():
        return []
    return sorted(path.parent for path in extracted.rglob("models.json") if path.is_file())


def extract_views(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("views", "cameras", "models"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def camera_name_for_video(path: Path) -> str:
    return path.stem


def inspect_scene(scene_dir: Path) -> dict[str, Any]:
    models_path = scene_dir / "models.json"
    models_payload = json.loads(models_path.read_text(encoding="utf-8"))
    views = extract_views(models_payload)
    views_by_name = {str(view.get("name")): view for view in views if view.get("name") is not None}
    videos = []
    for video_path in sorted(path for path in scene_dir.rglob("*") if path.suffix.lower() in VIDEO_EXTENSIONS):
        camera_name = camera_name_for_video(video_path)
        probe = ffprobe(video_path)
        view = views_by_name.get(camera_name)
        videos.append(
            {
                "camera": camera_name,
                "video_path": str(video_path.resolve()),
                "relative_video_path": str(video_path.relative_to(scene_dir)),
                "has_camera_model": view is not None,
                "projection_type": view.get("projection_type") if isinstance(view, dict) else None,
                "radial_distortion": view.get("radial_distortion") if isinstance(view, dict) else None,
                **probe,
            }
        )
    readable = [video for video in videos if video.get("readable")]
    durations = [float(video.get("duration_seconds") or 0.0) for video in readable]
    return {
        "scene": scene_dir.name,
        "scene_dir": str(scene_dir.resolve()),
        "models_path": str(models_path.resolve()),
        "model_camera_count": len(views),
        "video_count": len(videos),
        "videos_with_camera_model": sum(1 for video in videos if video["has_camera_model"]),
        "projection_types": sorted({str(video["projection_type"]) for video in videos if video.get("projection_type")}),
        "median_duration_seconds": median(durations),
        "videos": videos,
    }


def median(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def extracted_dir_for_archive(root: Path, archive: Path) -> Path:
    return root / "extracted" / archive.stem


def archive_extraction_populated(extracted_dir: Path) -> bool:
    # We treat an extraction as "populated" only if the canonical DeepView
    # calibration file is present. That file is written last by the official
    # zip layout and contains the per-camera intrinsics+extrinsics, so its
    # presence is the strongest single-signal that extraction did not abort
    # part-way. Counting raw files would let us reclaim a zip whose extraction
    # crashed mid-way.
    if not extracted_dir.is_dir():
        return False
    return (extracted_dir / "models.json").is_file() or any(extracted_dir.rglob("models.json"))


def cleanup_zips(config: dict[str, Any], root: Path, *, all_scenes: bool, execute: bool) -> None:
    raw_dir = root / "raw"
    archives = sorted(raw_dir.glob("*.zip"))
    selected_names = {scene["name"] for scene in selected_scenes(config, all_scenes=all_scenes)}
    reclaimable: list[tuple[Path, int]] = []
    skipped: list[tuple[Path, str]] = []
    for archive in archives:
        if archive.stem not in selected_names:
            skipped.append((archive, "not in selected scenes (pass --all-scenes if intended)"))
            continue
        extracted = extracted_dir_for_archive(root, archive)
        if not archive_extraction_populated(extracted):
            skipped.append((archive, f"extraction missing or incomplete at {extracted}"))
            continue
        reclaimable.append((archive, archive.stat().st_size))
    total_bytes = sum(size for _, size in reclaimable)
    print(f"DeepView raw archive cleanup ({'EXECUTE' if execute else 'DRY-RUN'}):")
    for archive, size in reclaimable:
        print(f"  reclaim\t{archive}\t{size / 1_000_000.0:.1f} MB")
    for archive, reason in skipped:
        print(f"  skip\t{archive}\t{reason}")
    print(f"  TOTAL reclaimable: {total_bytes} bytes ({total_bytes / 1_000_000.0:.1f} MB) across {len(reclaimable)} archives")
    if execute:
        for archive, _ in reclaimable:
            archive.unlink()
            print(f"  deleted: {archive}")


def inspect(config: dict[str, Any], root: Path) -> None:
    summaries = [inspect_scene(scene_dir) for scene_dir in deep_find_scene_dirs(root)]
    output_path = root / "metadata" / "scene_inventory.json"
    output_path.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_jsonl(root / "metadata" / "scenes.jsonl", summaries)
    if not summaries:
        print(f"No DeepView scenes found under {root / 'extracted'}. Run download/extract first.")
    for summary in summaries:
        print(
            f"{summary['scene']}: videos={summary['video_count']} "
            f"models={summary['model_camera_count']} matched={summary['videos_with_camera_model']} "
            f"projection={summary['projection_types']} median_duration={summary['median_duration_seconds']:.2f}s"
        )
    print(f"Wrote inventory: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and inspect DeepView raw multi-camera video scenes.")
    parser.add_argument("stage", choices=("list-scenes", "download", "extract", "inspect", "cleanup-zips", "all"))
    parser.add_argument("--config", type=Path, default=Path("src/dataset_configs/deepview_video_seed.jsonc"))
    parser.add_argument("--all-scenes", action="store_true", help="Use all 15 scenes instead of the configured seed scenes.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite downloaded archives or extracted folders.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="cleanup-zips only: actually delete reclaimable raw archives. Default is dry-run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config_file(args.config)
    root = resolve_root(config)
    overwrite = True if args.overwrite else None
    if args.stage in {"list-scenes", "all"}:
        list_scenes(config, root)
    if args.stage in {"download", "all"}:
        download(config, root, all_scenes=args.all_scenes, overwrite=overwrite)
    if args.stage in {"extract", "all"}:
        extract(config, root, all_scenes=args.all_scenes, overwrite=overwrite)
    if args.stage in {"inspect", "all"}:
        inspect(config, root)
    if args.stage == "cleanup-zips":
        cleanup_zips(config, root, all_scenes=args.all_scenes, execute=args.execute)


if __name__ == "__main__":
    main()
