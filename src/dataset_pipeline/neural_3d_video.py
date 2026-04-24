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
sys.path.insert(0, str(SRC_DIR / "train"))

from config_utils import load_config_file  # noqa: E402


def import_cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on local video deps.
        raise ImportError("OpenCV is required to inspect Neural 3D Video scenes.") from exc
    return cv2


def resolve_root(config: dict[str, Any]) -> Path:
    root = Path(config["root_dir"])
    for child in ("raw", "extracted", "logs", "metadata"):
        (root / child).mkdir(parents=True, exist_ok=True)
    return root


def github_release_url(config: dict[str, Any]) -> str:
    github = config["github"]
    return (
        "https://api.github.com/repos/"
        f"{github['owner']}/{github['repo']}/releases/tags/{github['tag']}"
    )


def fetch_release(config: dict[str, Any], root: Path) -> dict[str, Any]:
    url = github_release_url(config)
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request) as response:
        payload = json.loads(response.read().decode("utf-8"))
    metadata_path = root / "metadata" / "release.json"
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def release_assets(config: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    metadata_path = root / "metadata" / "release.json"
    if metadata_path.exists():
        payload = json.loads(metadata_path.read_text())
    else:
        payload = fetch_release(config, root)
    assets = payload.get("assets", [])
    if not isinstance(assets, list):
        raise RuntimeError("GitHub release metadata did not contain an assets list.")
    return assets


def selected_assets(config: dict[str, Any], root: Path, all_assets: bool) -> list[dict[str, Any]]:
    assets = release_assets(config, root)
    if all_assets:
        return assets
    names = set(config["download"].get("asset_names") or [])
    if not names:
        return assets
    selected = [asset for asset in assets if asset.get("name") in names]
    missing = sorted(names - {asset.get("name") for asset in selected})
    if missing:
        raise RuntimeError(f"Configured asset names were not found in the release: {missing}")
    return selected


def download_file(url: str, output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        print(f"Already exists: {output_path}")
        return
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "dynaworld-dataset-ingest"})
    with urllib.request.urlopen(request) as response, tmp_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    tmp_path.replace(output_path)
    print(f"Downloaded: {output_path}")


def list_assets(config: dict[str, Any], root: Path) -> None:
    release = fetch_release(config, root)
    assets = release.get("assets", [])
    print(f"Release: {release.get('html_url')}")
    for asset in assets:
        size_gb = float(asset.get("size", 0)) / 1_000_000_000.0
        print(f"{asset.get('name')}\t{size_gb:.2f} GB\t{asset.get('browser_download_url')}")


def download(config: dict[str, Any], root: Path, all_assets: bool, overwrite: bool | None) -> None:
    configured_overwrite = bool(config["download"].get("overwrite", False))
    should_overwrite = configured_overwrite if overwrite is None else overwrite
    assets = selected_assets(config, root, all_assets=all_assets)
    if not assets:
        raise RuntimeError("No release assets selected for download.")
    for asset in assets:
        name = asset.get("name")
        url = asset.get("browser_download_url")
        if not name or not url:
            continue
        download_file(url, root / "raw" / name, should_overwrite)


def extract_zip(path: Path, output_root: Path, overwrite: bool) -> None:
    scene_name = path.name
    for suffix in (".zip", ".ZIP"):
        if scene_name.endswith(suffix):
            scene_name = scene_name[: -len(suffix)]
    output_dir = output_root / scene_name
    if output_dir.exists():
        if not overwrite:
            print(f"Already extracted: {output_dir}")
            return
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        archive.extractall(output_dir)
    print(f"Extracted: {output_dir}")


def repair_flame_salmon_if_present(root: Path) -> Path | None:
    split_root = root / "raw"
    first_split = split_root / "flame_salmon_1_split.zip"
    output = split_root / "flame_salmon_1.zip"
    if output.exists() or not first_split.exists():
        return output if output.exists() else None
    if shutil.which("zip") is None:
        print("Skipping flame_salmon_1 repair because `zip` is not installed.")
        return None
    subprocess.run(
        ["zip", "-F", str(first_split), "--out", str(output)],
        check=True,
        cwd=str(split_root),
    )
    return output


def extract(config: dict[str, Any], root: Path, overwrite: bool | None) -> None:
    configured_overwrite = bool(config["extract"].get("overwrite", False))
    should_overwrite = configured_overwrite if overwrite is None else overwrite
    repaired = repair_flame_salmon_if_present(root)
    archives = sorted((root / "raw").glob("*.zip"))
    if repaired is not None and repaired not in archives:
        archives.append(repaired)
    if not archives:
        raise RuntimeError(f"No zip archives found under {root / 'raw'}")
    for archive in archives:
        if archive.name.startswith("flame_salmon_1_split"):
            continue
        extract_zip(archive, root / "extracted", should_overwrite)


def nested_scene_dirs(root: Path) -> list[Path]:
    scene_dirs = []
    for candidate in sorted((root / "extracted").rglob("*")):
        if candidate.is_dir() and (candidate / "poses_bounds.npy").exists():
            scene_dirs.append(candidate)
    return scene_dirs


def inspect_video(path: Path, cv2: Any) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return {"path": str(path), "readable": False}
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        capture.release()
    return {
        "path": str(path),
        "readable": True,
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": frame_count / fps if fps > 0 else 0.0,
        "width": width,
        "height": height,
    }


def inspect(config: dict[str, Any], root: Path) -> None:
    cv2 = import_cv2()
    summaries = []
    for scene_dir in nested_scene_dirs(root):
        cameras = [inspect_video(path, cv2) for path in sorted(scene_dir.glob("cam*.mp4"))]
        summary = {
            "scene": scene_dir.name,
            "scene_dir": str(scene_dir.resolve()),
            "camera_count": len(cameras),
            "has_cam00_reference": any(Path(camera["path"]).name == "cam00.mp4" for camera in cameras),
            "poses_bounds_path": str((scene_dir / "poses_bounds.npy").resolve()),
            "cameras": cameras,
        }
        summaries.append(summary)

    output_path = root / "metadata" / "scene_inventory.json"
    output_path.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")
    for summary in summaries:
        durations = [camera.get("duration_seconds", 0.0) for camera in summary["cameras"]]
        median_duration = float(np_median(durations)) if durations else 0.0
        print(
            f"{summary['scene']}: cameras={summary['camera_count']} "
            f"cam00={summary['has_cam00_reference']} median_duration={median_duration:.2f}s"
        )
    print(f"Wrote inventory: {output_path}")


def np_median(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and inspect Neural 3D Video release assets.")
    parser.add_argument("stage", choices=("list-assets", "download", "extract", "inspect", "all"))
    parser.add_argument("--config", type=Path, default=Path("src/dataset_configs/neural_3d_video_seed.jsonc"))
    parser.add_argument("--all-assets", action="store_true", help="Download all release assets instead of configured names.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite downloaded archives or extracted folders.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config_file(args.config)
    root = resolve_root(config)
    overwrite = True if args.overwrite else None
    if args.stage in {"list-assets", "all"}:
        list_assets(config, root)
    if args.stage in {"download", "all"}:
        download(config, root, all_assets=args.all_assets, overwrite=overwrite)
    if args.stage in {"extract", "all"}:
        extract(config, root, overwrite=overwrite)
    if args.stage in {"inspect", "all"}:
        inspect(config, root)


if __name__ == "__main__":
    main()
