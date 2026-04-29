from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SRC_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR / "train"))

from config_utils import load_config_file  # noqa: E402


@dataclass(frozen=True)
class Paths:
    root: Path
    metadata: Path
    manifests: Path
    clip_sets: Path
    logs: Path


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def resolve_paths(config: dict[str, Any]) -> Paths:
    root = resolve_path(config["root_dir"])
    paths = Paths(
        root=root,
        metadata=root / "metadata",
        manifests=root / "manifests",
        clip_sets=root / "clip_sets",
        logs=root / "logs",
    )
    for path in paths.__dict__.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def clean_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value).strip("_")


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
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "COMMAND\n"
            + " ".join(command)
            + "\n\nSTDOUT\n"
            + result.stdout
            + "\n\nSTDERR\n"
            + result.stderr,
            encoding="utf-8",
        )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(command)}")
    return result


def download_file(url: str, output_path: Path, overwrite: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        print(f"Already exists: {output_path}")
        return
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "dynaworld-dataset-ingest"})
    with urllib.request.urlopen(request) as response, tmp_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    tmp_path.replace(output_path)
    print(f"Downloaded: {output_path}")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def aist_csv_path(config: dict[str, Any], paths: Paths) -> Path:
    url_name = Path(str(config["aist"]["csv_url"])).name
    return paths.metadata / "aist" / url_name


def aist_sequence_key(sequence: dict[str, Any]) -> str:
    return "_".join(
        [
            str(sequence["genre"]),
            str(sequence["situation"]),
            str(sequence["dancer"]),
            str(sequence["music"]),
            str(sequence["choreography"]),
        ]
    )


def row_matches_aist_sequence(row: dict[str, str], sequence: dict[str, Any], camera: str) -> bool:
    return (
        row.get("GENRE") == sequence["genre"]
        and row.get("SITUATION") == sequence["situation"]
        and row.get("DANCER") == sequence["dancer"]
        and row.get("MUSIC") == sequence["music"]
        and row.get("CHOREOGRAPHY") == sequence["choreography"]
        and row.get("CAMERA") == camera
    )


def find_aist_row(rows: list[dict[str, str]], sequence: dict[str, Any], camera: str) -> dict[str, str]:
    for row in rows:
        if row_matches_aist_sequence(row, sequence, camera):
            return row
    raise RuntimeError(f"No AIST row found for sequence={aist_sequence_key(sequence)} camera={camera}")


def download_aist(config: dict[str, Any], paths: Paths, overwrite: bool) -> None:
    aist_cfg = config.get("aist", {})
    if not bool(aist_cfg.get("enabled", False)):
        print("AIST disabled in config.")
        return
    csv_path = aist_csv_path(config, paths)
    download_file(str(aist_cfg["csv_url"]), csv_path, overwrite=False)
    rows = read_csv_rows(csv_path)
    raw_dir = resolve_path(aist_cfg["raw_dir"])
    selected = []
    for sequence in aist_cfg.get("sequences", []):
        cameras = [str(sequence["source_camera"]), *[str(item) for item in sequence["target_cameras"]]]
        for camera in cameras:
            row = find_aist_row(rows, sequence, camera)
            output_path = raw_dir / row["FILE_NAME"]
            download_file(row["URL"], output_path, overwrite=overwrite)
            selected.append(
                {
                    **row,
                    "local_path": str(output_path.resolve()),
                    "sequence_key": aist_sequence_key(sequence),
                    "role": "source" if camera == sequence["source_camera"] else "target",
                }
            )
    write_jsonl(paths.metadata / "aist" / "selected_videos.jsonl", selected)
    print(f"Wrote {len(selected)} AIST selected-video records.")


# AIST++ camera bundle layout (after extraction):
#   <cameras_dir>/cameras/mapping.txt              one line per AIST seq: "<seq_name>  <env_name>"
#   <cameras_dir>/cameras/<env_name>.json          list of 9 dicts: {name, size, matrix, rotation, translation, distortions}
# Schema, axis convention, and the raw archive URL come from
#   https://google.github.io/aistplusplus_dataset/download.html
# rotation/translation are OpenCV-style (rvec, tvec); R = cv2.Rodrigues(rotation) gives
#   x_camera = R @ x_world + translation (i.e. world-to-camera).
# size = [W, H] in pixels (1920x1080 at native resolution).
def aist_camera_zip_path(aist_cfg: dict[str, Any], paths: Paths) -> Path:
    cameras_root = resolve_path(aist_cfg.get("cameras_dir") or paths.metadata / "aist" / "cameras")
    return cameras_root / "cameras.zip"


def aist_cameras_dir(aist_cfg: dict[str, Any], paths: Paths) -> Path:
    cameras_root = resolve_path(aist_cfg.get("cameras_dir") or paths.metadata / "aist" / "cameras")
    return cameras_root / "extracted" / "cameras"


def parse_aist_mapping(mapping_path: Path) -> dict[str, str]:
    if not mapping_path.exists():
        raise FileNotFoundError(f"AIST++ mapping.txt not found: {mapping_path}")
    mapping: dict[str, str] = {}
    for line_number, raw in enumerate(mapping_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"Unexpected mapping line at {mapping_path}:{line_number}: {raw!r}")
        seq_name, env_name = parts
        mapping[seq_name] = env_name
    return mapping


def aist_seq_name(sequence: dict[str, Any]) -> str:
    # AIST++ keys all sequences by the canonical "cAll" video name:
    #   <genre>_<situation>_cAll_<dancer>_<music>_<choreography>
    # We never run inference on cAll itself; it is only the joint multi-view key into mapping.txt.
    return "_".join(
        [
            str(sequence["genre"]),
            str(sequence["situation"]),
            "cAll",
            str(sequence["dancer"]),
            str(sequence["music"]),
            str(sequence["choreography"]),
        ]
    )


def download_aist_cameras(config: dict[str, Any], paths: Paths, overwrite: bool) -> None:
    aist_cfg = config.get("aist", {})
    if not bool(aist_cfg.get("enabled", False)):
        print("AIST disabled in config.")
        return
    cameras_zip_url = str(
        aist_cfg.get(
            "cameras_zip_url",
            "https://github.com/google/aistplusplus_dataset/releases/download/v1.0/cameras.zip",
        )
    )
    zip_path = aist_camera_zip_path(aist_cfg, paths)
    download_file(cameras_zip_url, zip_path, overwrite=overwrite)

    extracted_root = zip_path.parent / "extracted"
    extracted_root.mkdir(parents=True, exist_ok=True)
    cameras_dir = aist_cameras_dir(aist_cfg, paths)
    if not cameras_dir.exists() or overwrite:
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.namelist():
                # Strip the macOS-only AppleDouble metadata; keep payload only.
                if member.startswith("__MACOSX/") or "/._" in member or member.endswith("/"):
                    continue
                archive.extract(member, extracted_root)

    mapping_path = cameras_dir / "mapping.txt"
    mapping = parse_aist_mapping(mapping_path)
    inventory: list[dict[str, Any]] = []
    for sequence in aist_cfg.get("sequences", []):
        seq_name = aist_seq_name(sequence)
        env_name = mapping.get(seq_name)
        if env_name is None:
            raise KeyError(
                f"AIST++ mapping.txt has no entry for sequence {seq_name!r}; "
                f"check genre/situation/dancer/music/choreography in config."
            )
        setting_path = cameras_dir / f"{env_name}.json"
        if not setting_path.exists():
            raise FileNotFoundError(f"Expected AIST++ setting file at {setting_path}")
        params = json.loads(setting_path.read_text(encoding="utf-8"))
        camera_names = [str(item["name"]) for item in params]
        inventory.append(
            {
                "sequence_key": "_".join(
                    [
                        str(sequence["genre"]),
                        str(sequence["situation"]),
                        str(sequence["dancer"]),
                        str(sequence["music"]),
                        str(sequence["choreography"]),
                    ]
                ),
                "aist_seq_name": seq_name,
                "aist_setting_name": env_name,
                "aist_setting_path": str(setting_path.resolve()),
                "available_cameras": camera_names,
            }
        )
    inventory_path = paths.metadata / "aist" / "camera_inventory.jsonl"
    write_jsonl(inventory_path, inventory)
    print(f"Wrote AIST camera inventory to {inventory_path} ({len(inventory)} sequences).")


def ffprobe(path: Path) -> dict[str, Any]:
    result = run_command(
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
        ]
    )
    payload = json.loads(result.stdout)
    stream = payload.get("streams", [{}])[0]
    return {
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "r_frame_rate": stream.get("r_frame_rate"),
        "nb_frames": stream.get("nb_frames"),
        "duration_seconds": float(payload.get("format", {}).get("duration", 0.0)),
    }


def frame_filter(size: int, fps: float) -> str:
    return (
        f"fps={fps},"
        f"scale={size}:{size}:force_original_aspect_ratio=decrease,"
        f"pad={size}:{size}:(ow-iw)/2:(oh-ih)/2"
    )


def extract_frames(
    *,
    video_path: Path,
    output_dir: Path,
    start_seconds: float,
    frame_count: int,
    fps: float,
    target_size: int,
    log_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in output_dir.glob("frame_*.png"):
        old_frame.unlink()
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_seconds:.6f}",
        "-i",
        str(video_path),
        "-frames:v",
        str(frame_count),
        "-vf",
        frame_filter(target_size, fps),
        str(output_dir / "frame_%04d.png"),
    ]
    run_command(command, log_path=log_path)
    frames = sorted(output_dir.glob("frame_*.png"))
    if len(frames) != frame_count:
        raise RuntimeError(f"Expected {frame_count} frames from {video_path}, wrote {len(frames)} to {output_dir}")


def write_preview(
    *,
    source_video: Path,
    target_video: Path,
    output_path: Path,
    source_start_seconds: float,
    target_start_seconds: float,
    duration_seconds: float,
    fps: float,
    preview_size: int,
    log_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vf = frame_filter(preview_size, fps)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{source_start_seconds:.6f}",
        "-t",
        f"{duration_seconds:.6f}",
        "-i",
        str(source_video),
        "-ss",
        f"{target_start_seconds:.6f}",
        "-t",
        f"{duration_seconds:.6f}",
        "-i",
        str(target_video),
        "-filter_complex",
        f"[0:v]{vf},setpts=PTS-STARTPTS[left];[1:v]{vf},setpts=PTS-STARTPTS[right];[left][right]hstack=inputs=2[v]",
        "-map",
        "[v]",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "20",
        str(output_path),
    ]
    run_command(command, log_path=log_path)


def parse_iso_to_seconds(value: str) -> float:
    normalized = value.replace("Z", "+00:00")
    if "." in normalized:
        prefix, suffix = normalized.split(".", 1)
        timezone_sep = "+" if "+" in suffix else "-"
        fractional, tz = suffix.split(timezone_sep, 1)
        fractional = (fractional + "000000")[:6]
        normalized = f"{prefix}.{fractional}{timezone_sep}{tz}"
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def build_sample(
    *,
    dataset_name: str,
    output_root: Path,
    sample_id: str,
    source_video: Path,
    target_video: Path,
    source_camera: str,
    target_camera: str,
    source_start_seconds: float,
    target_start_seconds: float,
    frame_count: int,
    fps: float,
    target_size: int,
    preview_size: int,
    preview_fps: float,
    materialize_metric_frames: bool,
    metadata: dict[str, Any],
    logs_dir: Path,
    overwrite: bool,
) -> dict[str, Any]:
    sample_dir = output_root / "clips" / sample_id
    if sample_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Sample already exists: {sample_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    source_frames = sample_dir / "source" / source_camera
    target_frames = sample_dir / "target" / target_camera
    duration_seconds = frame_count / fps
    if materialize_metric_frames:
        extract_frames(
            video_path=source_video,
            output_dir=source_frames,
            start_seconds=source_start_seconds,
            frame_count=frame_count,
            fps=fps,
            target_size=target_size,
            log_path=logs_dir / f"{sample_id}_source_frames.log",
        )
        extract_frames(
            video_path=target_video,
            output_dir=target_frames,
            start_seconds=target_start_seconds,
            frame_count=frame_count,
            fps=fps,
            target_size=target_size,
            log_path=logs_dir / f"{sample_id}_target_frames.log",
        )
    preview_path = output_root / "previews" / f"{sample_id}.mp4"
    write_preview(
        source_video=source_video,
        target_video=target_video,
        output_path=preview_path,
        source_start_seconds=source_start_seconds,
        target_start_seconds=target_start_seconds,
        duration_seconds=duration_seconds,
        fps=preview_fps,
        preview_size=preview_size,
        log_path=logs_dir / f"{sample_id}_preview.log",
    )
    source_probe = ffprobe(source_video)
    target_probe = ffprobe(target_video)
    record = {
        "sample_id": sample_id,
        "dataset_name": dataset_name,
        "split": "val",
        "source_camera": source_camera,
        "target_camera": target_camera,
        "source_video_path": str(source_video.resolve()),
        "target_video_path": str(target_video.resolve()),
        "source_frames_dir": str(source_frames.resolve()) if materialize_metric_frames else None,
        "target_frames_dir": str(target_frames.resolve()) if materialize_metric_frames else None,
        "preview_path": str(preview_path.resolve()),
        "source_start_seconds": source_start_seconds,
        "target_start_seconds": target_start_seconds,
        "duration_seconds": duration_seconds,
        "frame_count": frame_count,
        "fps": fps,
        "target_size": target_size,
        "preview_fps": preview_fps,
        "materialize_metric_frames": materialize_metric_frames,
        "source_video": source_probe,
        "target_video": target_probe,
        **metadata,
    }
    (sample_dir / "summary.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def aist_pairs(config: dict[str, Any], paths: Paths) -> list[dict[str, Any]]:
    aist_cfg = config.get("aist", {})
    if not bool(aist_cfg.get("enabled", False)):
        return []
    selected_path = paths.metadata / "aist" / "selected_videos.jsonl"
    if not selected_path.exists():
        raise FileNotFoundError("Missing AIST selected videos. Run download-aist first.")
    selected = read_jsonl(selected_path)
    by_key_camera = {(record["sequence_key"], record["CAMERA"]): record for record in selected}

    # AIST++ camera-parameter bundle is required for any multi-view rendering of these clips,
    # so we resolve and validate the per-sequence setting JSON path up front. This matches the
    # DeepView path which carries `models_path` for the same audit-trail reason.
    cameras_dir = aist_cameras_dir(aist_cfg, paths)
    mapping_path = cameras_dir / "mapping.txt"
    cameras_available = mapping_path.exists()
    mapping = parse_aist_mapping(mapping_path) if cameras_available else {}
    raw_dir = resolve_path(aist_cfg["raw_dir"])

    pairs = []
    for sequence in aist_cfg.get("sequences", []):
        sequence_key = aist_sequence_key(sequence)
        full_seq_name = aist_seq_name(sequence)
        source_camera = str(sequence["source_camera"])
        source = by_key_camera[(sequence_key, source_camera)]

        if cameras_available:
            env_name = mapping.get(full_seq_name)
            if env_name is None:
                raise KeyError(
                    f"AIST++ mapping.txt has no entry for {full_seq_name!r}; rerun download-aist-cameras."
                )
            setting_path = cameras_dir / f"{env_name}.json"
            aist_camera_metadata = {
                "aist_seq_name": full_seq_name,
                "aist_setting_name": env_name,
                "aist_setting_path": str(setting_path.resolve()),
                "aist_cameras_dir": str(cameras_dir.resolve()),
                "aist_raw_dir": str(raw_dir.resolve()),
                "camera_model": "opencv_pinhole_radial",
            }
        else:
            aist_camera_metadata = {}

        for target_camera in sequence["target_cameras"]:
            target = by_key_camera[(sequence_key, str(target_camera))]
            pairs.append(
                {
                    "sample_id": clean_id(f"aist_{sequence_key}_{source_camera}_to_{target_camera}"),
                    "dataset": "aist_dance_db",
                    "scene": sequence_key,
                    "source_camera": source_camera,
                    "target_camera": str(target_camera),
                    "source_video": resolve_path(source["local_path"]),
                    "target_video": resolve_path(target["local_path"]),
                    "source_start_seconds": float(aist_cfg.get("start_seconds", 0.0)),
                    "target_start_seconds": float(aist_cfg.get("start_seconds", 0.0)),
                    "metadata": {
                        "source_family": "aist_dance_db",
                        "genre": sequence["genre"],
                        "situation": sequence["situation"],
                        "dancer": sequence["dancer"],
                        "music": sequence["music"],
                        "choreography": sequence["choreography"],
                        **aist_camera_metadata,
                    },
                }
            )
    return pairs


def neural_3d_pairs(config: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = config.get("neural_3d_video", {})
    if not bool(cfg.get("enabled", False)):
        return []
    scene_dir = resolve_path(cfg["scene_dir"])
    source_camera = str(cfg["source_camera"])
    source_video = scene_dir / f"{source_camera}.mp4"
    pairs = []
    for target_camera in cfg["target_cameras"]:
        target_camera = str(target_camera)
        pairs.append(
            {
                "sample_id": clean_id(f"neural3d_{cfg['scene']}_{source_camera}_to_{target_camera}"),
                "dataset": "neural_3d_video",
                "scene": cfg["scene"],
                "source_camera": source_camera,
                "target_camera": target_camera,
                "source_video": source_video,
                "target_video": scene_dir / f"{target_camera}.mp4",
                "source_start_seconds": float(cfg.get("start_seconds", 0.0)),
                "target_start_seconds": float(cfg.get("start_seconds", 0.0)),
                "metadata": {
                    "source_family": "dynamic_nerf",
                    "dataset_scene_dir": str(scene_dir.resolve()),
                },
            }
        )
    return pairs


def deepview_inventory_by_scene(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    inventory_path = resolve_path(config["deepview_video"]["inventory_path"])
    if not inventory_path.exists():
        raise FileNotFoundError("Missing DeepView inventory. Run src/dataset_scripts/deepview_video_seed.sh first.")
    payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError(f"Expected DeepView inventory list at {inventory_path}")
    return {str(record["scene"]): record for record in payload}


def deepview_pairs(config: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = config.get("deepview_video", {})
    if not bool(cfg.get("enabled", False)):
        return []
    by_scene = deepview_inventory_by_scene(config)
    required_duration = float(config["clip_frames"]) / float(config["fps"])
    pairs = []
    for scene_cfg in cfg.get("scenes", []):
        scene = str(scene_cfg["scene"])
        scene_record = by_scene[scene]
        by_camera = {str(video["camera"]): video for video in scene_record.get("videos", [])}
        source_camera = str(scene_cfg["source_camera"])
        source = by_camera[source_camera]
        start_seconds = float(scene_cfg.get("start_seconds", cfg.get("start_seconds", 0.0)))
        for target_camera in scene_cfg["target_cameras"]:
            target_camera = str(target_camera)
            target = by_camera[target_camera]
            for role, video in (("source", source), ("target", target)):
                remaining = float(video.get("duration_seconds") or 0.0) - start_seconds
                if remaining < required_duration:
                    raise RuntimeError(
                        f"DeepView {scene} {role} camera {video['camera']} only has "
                        f"{remaining:.2f}s available after start; need {required_duration:.2f}s."
                    )
            pairs.append(
                {
                    "sample_id": clean_id(f"deepview_{scene}_{source_camera}_to_{target_camera}"),
                    "dataset": "deepview_video",
                    "scene": scene,
                    "source_camera": source_camera,
                    "target_camera": target_camera,
                    "source_video": resolve_path(source["video_path"]),
                    "target_video": resolve_path(target["video_path"]),
                    "source_start_seconds": start_seconds,
                    "target_start_seconds": start_seconds,
                    "metadata": {
                        "source_family": "deepview_light_field_video",
                        "dataset_scene_dir": scene_record["scene_dir"],
                        "models_path": scene_record["models_path"],
                        "camera_model": "fisheye_radial_distortion",
                        "source_projection_type": source.get("projection_type"),
                        "target_projection_type": target.get("projection_type"),
                        "source_radial_distortion": source.get("radial_distortion"),
                        "target_radial_distortion": target.get("radial_distortion"),
                    },
                }
            )
    return pairs


def vivo_record_by_camera(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    manifest_path = resolve_path(config["vivo"]["manifest_path"])
    records = read_jsonl(manifest_path)
    scene = str(config["vivo"]["scene"])
    return {str(record["camera"]): record for record in records if record.get("scene") == scene}


def vivo_pairs(config: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = config.get("vivo", {})
    if not bool(cfg.get("enabled", False)):
        return []
    by_camera = vivo_record_by_camera(config)
    source_camera = str(cfg["source_camera"])
    source = by_camera[source_camera]
    source_first = parse_iso_to_seconds(source["first_capture_timestamp"])
    source_last = parse_iso_to_seconds(source["last_capture_timestamp"])
    pairs = []
    for target_camera in cfg["target_cameras"]:
        target_camera = str(target_camera)
        target = by_camera[target_camera]
        target_first = parse_iso_to_seconds(target["first_capture_timestamp"])
        target_last = parse_iso_to_seconds(target["last_capture_timestamp"])
        common_start = max(source_first, target_first) + float(cfg.get("overlap_margin_seconds", 0.0))
        common_end = min(source_last, target_last)
        duration_seconds = float(config["clip_frames"]) / float(config["fps"])
        if common_end - common_start < duration_seconds:
            raise RuntimeError(
                f"ViVo cameras {source_camera}->{target_camera} do not overlap for {duration_seconds:.2f}s."
            )
        pairs.append(
            {
                "sample_id": clean_id(f"vivo_{cfg['scene']}_{source_camera}_to_{target_camera}"),
                "dataset": "vivo",
                "scene": cfg["scene"],
                "source_camera": source_camera,
                "target_camera": target_camera,
                "source_video": resolve_path(source["mp4_path"]),
                "target_video": resolve_path(target["mp4_path"]),
                "source_start_seconds": common_start - source_first,
                "target_start_seconds": common_start - target_first,
                "metadata": {
                    "source_family": "multicamera_video",
                    "alignment": "capture_timestamp_overlap",
                    "common_start_unix_seconds": common_start,
                    "common_end_unix_seconds": common_end,
                    "source_first_capture_timestamp": source["first_capture_timestamp"],
                    "target_first_capture_timestamp": target["first_capture_timestamp"],
                },
            }
        )
    return pairs


def validate_pair_inputs(pairs: list[dict[str, Any]]) -> None:
    for pair in pairs:
        for key in ("source_video", "target_video"):
            path = Path(pair[key])
            if not path.exists():
                raise FileNotFoundError(f"Missing {key} for {pair['sample_id']}: {path}")


def build(config: dict[str, Any], paths: Paths, overwrite: bool) -> None:
    dataset_name = str(config["dataset_name"])
    output_root = paths.clip_sets / dataset_name
    if output_root.exists() and overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    frame_count = int(config["clip_frames"])
    fps = float(config["fps"])
    target_size = int(config["target_size"])
    preview_size = int(config.get("preview_size", target_size))
    preview_fps = float(config.get("preview_fps", min(30.0, fps)))
    materialize_metric_frames = bool(config.get("materialize_metric_frames", False))
    pairs = [*aist_pairs(config, paths), *neural_3d_pairs(config), *vivo_pairs(config), *deepview_pairs(config)]
    validate_pair_inputs(pairs)

    records = []
    for pair in pairs:
        records.append(
            build_sample(
                dataset_name=dataset_name,
                output_root=output_root,
                sample_id=pair["sample_id"],
                source_video=Path(pair["source_video"]),
                target_video=Path(pair["target_video"]),
                source_camera=str(pair["source_camera"]),
                target_camera=str(pair["target_camera"]),
                source_start_seconds=float(pair["source_start_seconds"]),
                target_start_seconds=float(pair["target_start_seconds"]),
                frame_count=frame_count,
                fps=fps,
                target_size=target_size,
                preview_size=preview_size,
                preview_fps=preview_fps,
                materialize_metric_frames=materialize_metric_frames,
                metadata={
                    "dataset": pair["dataset"],
                    "scene": pair["scene"],
                    **pair.get("metadata", {}),
                },
                logs_dir=paths.logs,
                overwrite=overwrite,
            )
        )

    manifest_path = output_root / "manifest.jsonl"
    write_jsonl(manifest_path, records)
    write_jsonl(paths.manifests / f"{dataset_name}.jsonl", records)
    summary = {
        "dataset_name": dataset_name,
        "clip_count": len(records),
        "split": "val",
        "frame_count": frame_count,
        "fps": fps,
        "preview_fps": preview_fps,
        "target_size": target_size,
        "materialize_metric_frames": materialize_metric_frames,
        "manifest_path": str(manifest_path.resolve()),
        "sources": sorted({record["dataset"] for record in records}),
    }
    (output_root / "dataset.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} multi-camera validation samples to {manifest_path}")


def inspect(config: dict[str, Any], paths: Paths) -> None:
    dataset_name = str(config["dataset_name"])
    manifest_path = paths.clip_sets / dataset_name / "manifest.jsonl"
    records = read_jsonl(manifest_path)
    if not records:
        raise RuntimeError(f"No records found at {manifest_path}. Run build first.")
    print(f"{dataset_name}: {len(records)} validation pairs")
    for record in records:
        print(
            f"{record['sample_id']}: {record['dataset']} {record['scene']} "
            f"{record['source_camera']}->{record['target_camera']} "
            f"frames={record['frame_count']} size={record['target_size']} preview={record['preview_path']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build tiny synchronized multi-camera validation clips.")
    parser.add_argument(
        "stage",
        choices=("download-aist", "download-aist-cameras", "build", "inspect", "all"),
    )
    parser.add_argument("--config", type=Path, default=Path("src/dataset_configs/multicam_val_v1_128_4fps_16f.jsonc"))
    parser.add_argument("--overwrite", action="store_true", help="Overwrite downloaded AIST videos and generated clips.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config_file(args.config)
    paths = resolve_paths(config)
    overwrite = bool(args.overwrite or config.get("overwrite", False))
    if args.stage in {"download-aist", "all"}:
        download_aist(config, paths, overwrite=overwrite)
    if args.stage in {"download-aist-cameras", "all"}:
        download_aist_cameras(config, paths, overwrite=overwrite)
    if args.stage in {"build", "all"}:
        build(config, paths, overwrite=overwrite)
    if args.stage in {"inspect", "all"}:
        inspect(config, paths)


if __name__ == "__main__":
    main()
