from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any


SRC_DIR = Path(__file__).resolve().parents[1]
TRAIN_DIR = SRC_DIR / "train"
REPO_ROOT = SRC_DIR.parent
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

from config_utils import load_config_file  # noqa: E402
from download_utils import download_url, fetch_json_url  # noqa: E402


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


def github_release_url(config: dict[str, Any]) -> str:
    github = config["github"]
    return f"https://api.github.com/repos/{github['owner']}/{github['repo']}/releases/tags/{github['tag']}"


def fetch_release(config: dict[str, Any], root: Path) -> dict[str, Any]:
    payload = fetch_json_url(
        github_release_url(config),
        headers={"Accept": "application/vnd.github+json"},
    )
    (root / "metadata" / "release.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def release_assets(config: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    metadata_path = root / "metadata" / "release.json"
    payload = json.loads(metadata_path.read_text()) if metadata_path.exists() else fetch_release(config, root)
    assets = payload.get("assets", [])
    if not isinstance(assets, list):
        raise RuntimeError("GitHub release metadata did not contain an asset list.")
    return assets


def selected_assets(config: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    assets = release_assets(config, root)
    selected_names = set(config["download"].get("asset_names") or [])
    selected = [asset for asset in assets if asset.get("name") in selected_names]
    missing = sorted(selected_names - {asset.get("name") for asset in selected})
    if missing:
        raise RuntimeError(f"Configured assets were not found in release: {missing}")
    return selected


def download_file(url: str, output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        print(f"Already exists: {output_path}")
        return
    download_url(url, output_path, overwrite=True, user_agent="dynaworld-ex4dgs-ingest")


def list_assets(config: dict[str, Any], root: Path) -> None:
    release = fetch_release(config, root)
    print(f"Release: {release.get('html_url')}")
    for asset in release_assets(config, root):
        size_mb = float(asset.get("size", 0)) / 1_000_000.0
        print(f"{asset.get('name')}\t{size_mb:.1f} MB\t{asset.get('browser_download_url')}")


def download(config: dict[str, Any], root: Path, overwrite: bool | None) -> None:
    should_overwrite = bool(config["download"].get("overwrite", False)) if overwrite is None else overwrite
    records = []
    for asset in selected_assets(config, root):
        name = asset.get("name")
        url = asset.get("browser_download_url")
        if not name or not url:
            continue
        output_path = root / "raw" / name
        download_file(url, output_path, overwrite=should_overwrite)
        records.append(
            {
                "name": name,
                "url": url,
                "local_path": str(output_path.resolve()),
                "size": asset.get("size"),
                "content_type": asset.get("content_type"),
            }
        )
    (root / "metadata" / "selected_assets.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")


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


def extract(config: dict[str, Any], root: Path, overwrite: bool | None) -> None:
    should_overwrite = bool(config["extract"].get("overwrite", False)) if overwrite is None else overwrite
    raw_paths = [root / "raw" / name for name in config["download"].get("asset_names", [])]
    missing = [path for path in raw_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing raw zip assets: {[str(path) for path in missing]}")
    for path in raw_paths:
        extract_zip(path, root / "extracted", should_overwrite)


def read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def relative_files(root: Path, pattern: str) -> list[str]:
    return sorted(str(path.relative_to(root)) for path in root.rglob(pattern) if path.is_file())


def inspect(config: dict[str, Any], root: Path) -> None:
    summaries = []
    for asset_name in config["download"].get("asset_names", []):
        name = Path(asset_name).stem
        extracted_dir = root / "extracted" / name
        if not extracted_dir.exists():
            summaries.append({"name": name, "extracted": False})
            continue
        files = [path for path in extracted_dir.rglob("*") if path.is_file()]
        suffix_counts = Counter(path.suffix.lower() or "<none>" for path in files)
        top_level = sorted(path.name for path in extracted_dir.iterdir())[:30]
        cameras = read_json(extracted_dir / "cameras.json")
        mean_metrics = read_json(extracted_dir / "mean_metrics.json")
        all_metrics = read_json(extracted_dir / "all_metrics.json")
        summaries.append(
            {
                "name": name,
                "extracted": True,
                "asset_name": asset_name,
                "file_count": len(files),
                "total_bytes": sum(path.stat().st_size for path in files),
                "suffix_counts": dict(sorted(suffix_counts.items())),
                "top_level_entries": top_level,
                "camera_count": len(cameras) if isinstance(cameras, list) else None,
                "mean_metrics": mean_metrics if isinstance(mean_metrics, dict) else None,
                "all_metrics_keys": sorted(all_metrics.keys()) if isinstance(all_metrics, dict) else None,
                "point_cloud_files": relative_files(extracted_dir, "*.ply"),
                "sample_files": [str(path.relative_to(extracted_dir)) for path in files[:50]],
            }
        )
    output_path = root / "metadata" / "inventory.json"
    output_path.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")
    for summary in summaries:
        print(
            f"{summary['name']}: extracted={summary['extracted']} "
            f"files={summary.get('file_count', 0)} cameras={summary.get('camera_count')} "
            f"suffixes={summary.get('suffix_counts', {})}"
        )
    print(f"Wrote inventory: {output_path}")


def ex4dgs_extraction_populated(extracted_dir: Path) -> bool:
    # Each Ex4DGS asset is an unzipped checkpoint directory. Treat as
    # populated only if the canonical metadata files are present; this is
    # the same loadability check `inspect` already exercises.
    if not extracted_dir.is_dir():
        return False
    return (extracted_dir / "cameras.json").is_file() and (extracted_dir / "mean_metrics.json").is_file()


def cleanup_zips(config: dict[str, Any], root: Path, *, execute: bool) -> None:
    raw_dir = root / "raw"
    asset_names = list(config["download"].get("asset_names", []))
    archives = sorted(raw_dir.glob("*.zip"))
    selected_archive_names = {Path(name).name for name in asset_names}
    reclaimable: list[tuple[Path, int]] = []
    skipped: list[tuple[Path, str]] = []
    for archive in archives:
        if archive.name not in selected_archive_names:
            skipped.append((archive, "not in configured asset_names"))
            continue
        extracted = root / "extracted" / archive.stem
        if not ex4dgs_extraction_populated(extracted):
            skipped.append((archive, f"extraction missing/incomplete at {extracted}"))
            continue
        reclaimable.append((archive, archive.stat().st_size))
    total_bytes = sum(size for _, size in reclaimable)
    print(f"Ex4DGS raw archive cleanup ({'EXECUTE' if execute else 'DRY-RUN'}):")
    for archive, size in reclaimable:
        print(f"  reclaim\t{archive}\t{size / 1_000_000.0:.1f} MB")
    for archive, reason in skipped:
        print(f"  skip\t{archive}\t{reason}")
    print(f"  TOTAL reclaimable: {total_bytes} bytes ({total_bytes / 1_000_000.0:.1f} MB) across {len(reclaimable)} archives")
    if execute:
        for archive, _ in reclaimable:
            archive.unlink()
            print(f"  deleted: {archive}")


def validate_local(config: dict[str, Any], root: Path) -> None:
    asset_names = list(config["download"].get("asset_names", []))
    failures: list[str] = []
    for asset_name in asset_names:
        name = Path(asset_name).stem
        extracted = root / "extracted" / name
        if not ex4dgs_extraction_populated(extracted):
            failures.append(f"{name}: missing cameras.json or mean_metrics.json under {extracted}")
            continue
        cameras = read_json(extracted / "cameras.json")
        if not isinstance(cameras, list) or not cameras:
            failures.append(f"{name}: cameras.json not a non-empty list")
            continue
        ply_files = list(extracted.rglob("*.ply"))
        if not ply_files:
            failures.append(f"{name}: no point-cloud .ply files found")
            continue
        total_bytes = sum(path.stat().st_size for path in extracted.rglob("*") if path.is_file())
        print(f"  ok\t{name}\tcameras={len(cameras)}\tply_files={len(ply_files)}\tbytes={total_bytes}")
    if failures:
        raise RuntimeError("Ex4DGS validate-local failed:\n  " + "\n  ".join(failures))
    print(f"validate-local: {len(asset_names)} bundles look loadable.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and inspect Ex4DGS pretrained model release assets.")
    parser.add_argument(
        "stage",
        choices=("list-assets", "download", "extract", "inspect", "validate-local", "cleanup-zips", "all"),
    )
    parser.add_argument("--config", type=Path, default=Path("src/dataset_configs/ex4dgs_pretrained_val_seed.jsonc"))
    parser.add_argument("--overwrite", action="store_true", help="Overwrite downloaded or extracted assets.")
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
    if args.stage in {"list-assets", "all"}:
        list_assets(config, root)
    if args.stage in {"download", "all"}:
        download(config, root, overwrite=overwrite)
    if args.stage in {"extract", "all"}:
        extract(config, root, overwrite=overwrite)
    if args.stage in {"inspect", "all"}:
        inspect(config, root)
    if args.stage == "validate-local":
        validate_local(config, root)
    if args.stage == "cleanup-zips":
        cleanup_zips(config, root, execute=args.execute)


if __name__ == "__main__":
    main()
