from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch
from checkpoint_utils import load_torch_checkpoint
from config_utils import load_config_file
from fast_attn import fast_attn_context, pick_device
from model_factories import build_model_from_config
from multicam_video_data import load_multicam_video_bundle, select_multicam_record
from PIL import Image
from powerfoam_point_cloud import load_point_cloud_xyz_rgb
from sequence_data import (
    load_camera_sequence,
    load_manifest_sequences,
    load_uncalibrated_sequence,
    prepare_clip,
    resolve_frames_dir,
)
from train_artifacts import write_json
from trainer_registry import resolve_config_for_arch

EXPORT_BUNDLE_VERSION = "dynaworld_token_head_bundle/v2"
BROWSER_MULTICAM_BUNDLE_VERSION = "dynaworld_browser_multicam_dataset/v1"


@dataclass(frozen=True)
class _BrowserMulticamExportBundle:
    train_frames: torch.Tensor
    train_K: torch.Tensor
    train_w2c: torch.Tensor
    train_camera_names: list[str]
    heldout_frames: torch.Tensor | None
    heldout_K: torch.Tensor | None
    heldout_w2c: torch.Tensor | None
    heldout_camera_names: list[str] | None
    pose_source: str | None
    anchor_c2w: torch.Tensor | None
    metadata: dict[str, Any]

    @property
    def train_view_count(self) -> int:
        return int(self.train_frames.shape[0])


def _resolve_seed_provenance(
    *,
    report_path: Path | None,
    seed_point_cloud_path: Path,
    train_cameras: list[str],
    heldout_cameras: list[str],
    allow_unverified: bool,
) -> dict[str, Any]:
    if report_path is None:
        if not allow_unverified:
            raise ValueError(
                "Seed initialization requires --seed-provenance-report, or an explicit "
                "--allow-unverified-seed-provenance opt-in for an external point cloud."
            )
        return {
            "method": "external_unverified",
            "source_report": None,
            "source_path": str(seed_point_cloud_path),
            "input_cameras": [],
            "train_only_verified": False,
            "coordinate_frame": "world",
        }

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not load seed provenance report {report_path}: {error}") from error
    if not isinstance(report, dict):
        raise ValueError(f"Seed provenance report {report_path} must contain a JSON object.")

    method = report.get("method")
    if not isinstance(method, str) or not method.strip():
        raise ValueError("Seed provenance report field 'method' must be a non-empty string.")
    input_cameras = report.get("input_cameras")
    if (
        not isinstance(input_cameras, list)
        or any(not isinstance(camera, str) or not camera for camera in input_cameras)
        or len(set(input_cameras)) != len(input_cameras)
    ):
        raise ValueError("Seed provenance report field 'input_cameras' must be a list of unique camera names.")
    train_only_verified = report.get("train_only_verified")
    if not isinstance(train_only_verified, bool):
        raise ValueError("Seed provenance report field 'train_only_verified' must be a boolean.")
    coordinate_frame = report.get("coordinate_frame")
    if not isinstance(coordinate_frame, str) or not coordinate_frame.strip():
        raise ValueError("Seed provenance report field 'coordinate_frame' must be a non-empty string.")

    heldout_overlap = sorted(set(input_cameras) & set(heldout_cameras))
    if heldout_overlap:
        raise ValueError(
            "Seed provenance input cameras overlap canonical heldout cameras: "
            f"{heldout_overlap}."
        )
    non_train_cameras = sorted(set(input_cameras) - set(train_cameras))
    if non_train_cameras:
        raise ValueError(
            "Seed provenance input cameras are not a subset of canonical train cameras: "
            f"{non_train_cameras}."
        )
    if train_only_verified and not input_cameras:
        raise ValueError("Verified train-only seed provenance must declare at least one input camera.")
    if not train_only_verified and not allow_unverified:
        raise ValueError(
            "Seed provenance report is unverified; pass --allow-unverified-seed-provenance "
            "to export it without a train-only claim."
        )

    return {
        "method": method.strip(),
        "source_report": str(report_path),
        "source_path": str(seed_point_cloud_path),
        "input_cameras": input_cameras,
        "train_only_verified": train_only_verified,
        "coordinate_frame": coordinate_frame.strip(),
    }


def _seed_points_in_anchor_frame(
    points: torch.Tensor,
    *,
    bundle: Any,
    seed_provenance: dict[str, Any],
) -> torch.Tensor:
    metadata = bundle.metadata or {}
    anchor_frame = f"{metadata['anchor_camera']}_opencv"
    coordinate_frame = str(seed_provenance["coordinate_frame"])
    if coordinate_frame in {"model", anchor_frame}:
        return points
    if coordinate_frame != "world":
        raise ValueError(
            f"Unsupported seed coordinate frame {coordinate_frame!r}; expected "
            f"'world', 'model', or {anchor_frame!r}."
        )
    if bundle.anchor_c2w is None:
        raise ValueError("The canonical multicam bundle did not provide anchor_c2w for point initialization.")
    world_to_anchor = torch.linalg.inv(bundle.anchor_c2w.detach().cpu())
    points_h = torch.cat([points, torch.ones((points.shape[0], 1), dtype=points.dtype)], dim=1)
    return (points_h @ world_to_anchor.T)[:, :3]


def _browser_camera_filename_component(camera_name: str) -> str:
    component = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(camera_name)).strip("._")
    if not component:
        raise ValueError(f"Camera name {camera_name!r} cannot be represented as a browser atlas filename.")
    return component


def _browser_sparse_frame_record(record: dict[str, Any], frame_index: int) -> dict[str, Any]:
    """Derive one exact synchronized timestamp without changing rig semantics."""
    source_frame_count = int(record["frame_count"])
    if frame_index < 0 or frame_index >= source_frame_count:
        raise ValueError(
            f"Browser frame index {frame_index} is outside source frame_count={source_frame_count}."
        )
    fps = float(record["fps"])
    if fps <= 0.0:
        raise ValueError("Browser sparse frame decode requires a positive manifest fps.")
    if "source_start_seconds" not in record or "target_start_seconds" not in record:
        raise ValueError(
            "Browser sparse frame decode requires synchronized source_start_seconds "
            "and target_start_seconds in the canonical manifest."
        )

    sampled = dict(record)
    time_offset = float(frame_index) / fps
    sampled["source_start_seconds"] = float(record["source_start_seconds"]) + time_offset
    sampled["target_start_seconds"] = float(record["target_start_seconds"]) + time_offset
    sampled["frame_count"] = 1
    sampled["duration_seconds"] = 1.0 / fps
    return sampled


def _load_browser_multicam_sparse_frames(
    *,
    manifest_path: Path,
    sample_id: str,
    split: str,
    target_size: tuple[int, int],
    frame_indices: list[int],
) -> _BrowserMulticamExportBundle:
    """Load only requested frames while delegating every camera contract to the canonical loader."""
    data_cfg = {
        "multicam_manifest": str(manifest_path),
        "multicam_split": split,
        "multicam_sample_id": sample_id,
    }
    record = select_multicam_record(data_cfg)
    if str(record.get("dataset")) != "neural_3d_video":
        raise ValueError(
            "Sparse browser frame decode is currently verified only for synchronized "
            "Neural 3D Video manifests."
        )
    normalized_indices = [int(index) for index in frame_indices]
    if not normalized_indices:
        raise ValueError("Browser sparse frame decode requires at least one frame index.")
    if normalized_indices != sorted(set(normalized_indices)):
        raise ValueError("Browser sparse frame indices must be unique and strictly increasing.")

    partial_bundles: list[Any] = []
    with TemporaryDirectory(prefix="dynaworld_browser_sparse_frames_") as temporary_dir:
        sparse_manifest_path = Path(temporary_dir) / "manifest.jsonl"
        for frame_index in normalized_indices:
            sparse_record = _browser_sparse_frame_record(record, frame_index)
            sparse_manifest_path.write_text(
                json.dumps(sparse_record, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            partial_bundles.append(
                load_multicam_video_bundle(
                    data_cfg={
                        "multicam_manifest": str(sparse_manifest_path),
                        "multicam_split": split,
                        "multicam_sample_id": sample_id,
                    },
                    camera_cfg={"rig_init": "neural_3d_video"},
                    target_size=target_size,
                    device=torch.device("cpu"),
                )
            )

    reference = partial_bundles[0]
    for partial in partial_bundles:
        if partial.frame_count != 1:
            raise RuntimeError("Sparse browser frame decode returned more than one frame.")
        if partial.train_camera_names != reference.train_camera_names:
            raise RuntimeError("Canonical train-camera order changed across sparse frame loads.")
        if partial.heldout_camera_names != reference.heldout_camera_names:
            raise RuntimeError("Canonical heldout-camera order changed across sparse frame loads.")
        if partial.pose_source != reference.pose_source:
            raise RuntimeError("Canonical pose source changed across sparse frame loads.")
        for name, actual, expected in (
            ("train intrinsics", partial.train_K, reference.train_K),
            ("train poses", partial.train_w2c, reference.train_w2c),
            ("heldout intrinsics", partial.heldout_K, reference.heldout_K),
            ("heldout poses", partial.heldout_w2c, reference.heldout_w2c),
            ("anchor pose", partial.anchor_c2w, reference.anchor_c2w),
        ):
            if actual is None or expected is None:
                if actual is not expected:
                    raise RuntimeError(f"Canonical {name} availability changed across sparse frame loads.")
                continue
            if not torch.allclose(actual, expected, rtol=1.0e-5, atol=1.0e-6):
                raise RuntimeError(f"Canonical {name} changed across sparse frame loads.")

    heldout_frames = None
    heldout_w2c = None
    if reference.heldout_frames is not None:
        heldout_frames = torch.cat(
            [partial.heldout_frames for partial in partial_bundles],
            dim=1,
        ).contiguous()
    if reference.heldout_w2c is not None:
        heldout_w2c = torch.cat(
            [partial.heldout_w2c for partial in partial_bundles],
            dim=1,
        ).contiguous()
    return _BrowserMulticamExportBundle(
        train_frames=torch.cat(
            [partial.train_frames for partial in partial_bundles],
            dim=1,
        ).contiguous(),
        train_K=reference.train_K,
        train_w2c=torch.cat(
            [partial.train_w2c for partial in partial_bundles],
            dim=1,
        ).contiguous(),
        train_camera_names=reference.train_camera_names,
        heldout_frames=heldout_frames,
        heldout_K=reference.heldout_K,
        heldout_w2c=heldout_w2c,
        heldout_camera_names=reference.heldout_camera_names,
        pose_source=reference.pose_source,
        anchor_c2w=reference.anchor_c2w,
        metadata={**(reference.metadata or {}), **record},
    )


def _farthest_point_subset(points: torch.Tensor, count: int) -> torch.Tensor:
    if int(points.shape[0]) <= count:
        return torch.arange(int(points.shape[0]))
    scale = (points.max(dim=0).values - points.min(dim=0).values).clamp_min(1.0e-5)
    normalized = (points - points.median(dim=0).values) / scale
    selected = torch.empty(count, dtype=torch.long)
    selected[0] = torch.argmin((normalized * normalized).sum(dim=1))
    min_distance2 = ((normalized - normalized[selected[0]]) ** 2).sum(dim=1)
    for index in range(1, count):
        selected[index] = torch.argmax(min_distance2)
        candidate_distance2 = ((normalized - normalized[selected[index]]) ** 2).sum(dim=1)
        min_distance2 = torch.minimum(min_distance2, candidate_distance2)
    return selected


def _write_browser_frame_atlases(bundle: Any, output_path: Path) -> dict[str, str]:
    frames_by_name = list(zip(bundle.train_camera_names, bundle.train_frames))
    if bundle.heldout_camera_names and bundle.heldout_frames is not None:
        frames_by_name.extend(zip(bundle.heldout_camera_names, bundle.heldout_frames))
    atlas_urls = {}
    used_filenames: set[str] = set()
    for camera_name, frames in frames_by_name:
        atlas = torch.cat([frame.permute(1, 2, 0) for frame in frames], dim=1)
        pixels = (atlas.clamp(0.0, 1.0) * 255.0).round().to(dtype=torch.uint8).cpu().numpy()
        camera_component = _browser_camera_filename_component(str(camera_name))
        atlas_filename = f"{output_path.stem}_{camera_component}.png"
        if atlas_filename in used_filenames:
            raise ValueError(f"Camera names collide on browser atlas filename {atlas_filename!r}.")
        used_filenames.add(atlas_filename)
        atlas_path = output_path.with_name(atlas_filename)
        Image.fromarray(pixels).save(atlas_path, optimize=True)
        atlas_urls[str(camera_name)] = f"./{atlas_path.name}"
    return atlas_urls


def _browser_camera_rows(
    bundle: Any,
    *,
    width: int,
    height: int,
    atlas_urls: dict[str, str],
) -> list[dict[str, Any]]:
    rows = []
    camera_groups = (
        ("train", bundle.train_camera_names, bundle.train_K, bundle.train_w2c),
        ("heldout", bundle.heldout_camera_names, bundle.heldout_K, bundle.heldout_w2c),
    )
    for role, names, intrinsics, world_to_camera in camera_groups:
        if not names or intrinsics is None or world_to_camera is None:
            continue
        for index, name in enumerate(names):
            K = intrinsics[index].detach().cpu()
            camera_poses = world_to_camera[index].detach().cpu()
            reference_pose = camera_poses[0].expand_as(camera_poses)
            if not torch.allclose(camera_poses, reference_pose, rtol=1.0e-5, atol=1.0e-6):
                raise ValueError(
                    "dynaworld_browser_multicam_dataset/v1 supports static camera rigs only; "
                    f"camera {name!r} changes pose across sampled frames."
                )
            rows.append(
                {
                    "name": str(name),
                    "role": role,
                    "frame_atlas_url": atlas_urls[str(name)],
                    "intrinsics": [
                        float(K[0, 0]) / float(width),
                        float(K[1, 1]) / float(height),
                        float(K[0, 2]) / float(width),
                        float(K[1, 2]) / float(height),
                    ],
                    "world_to_camera": world_to_camera[index, 0].detach().cpu().tolist(),
                }
            )
    return rows


def export_browser_multicam_dataset_bundle(
    *,
    manifest_path: Path,
    sample_id: str,
    split: str,
    seed_point_cloud_path: Path,
    output_path: Path,
    target_size: tuple[int, int],
    frame_indices: list[int],
    seed_count: int,
    seed_provenance_report_path: Path | None = None,
    allow_unverified_seed_provenance: bool = False,
    sparse_frame_decode: bool = False,
) -> Path:
    """Serialize a thin browser adapter over the canonical multicam contract."""
    height, width = target_size
    points, colors = load_point_cloud_xyz_rgb(seed_point_cloud_path)
    if int(points.shape[0]) < seed_count:
        raise ValueError(
            f"Seed point cloud has only {int(points.shape[0])} points before visibility filtering; "
            f"the browser bundle requested {seed_count}."
        )
    if sparse_frame_decode:
        bundle = _load_browser_multicam_sparse_frames(
            manifest_path=manifest_path,
            sample_id=sample_id,
            split=split,
            target_size=target_size,
            frame_indices=frame_indices,
        )
    else:
        bundle = load_multicam_video_bundle(
            data_cfg={
                "multicam_manifest": str(manifest_path),
                "multicam_split": split,
                "multicam_sample_id": sample_id,
                "frame_indices": frame_indices,
            },
            camera_cfg={"rig_init": "neural_3d_video"},
            target_size=target_size,
            device=torch.device("cpu"),
        )
    seed_provenance = _resolve_seed_provenance(
        report_path=seed_provenance_report_path,
        seed_point_cloud_path=seed_point_cloud_path,
        train_cameras=list(bundle.train_camera_names),
        heldout_cameras=list(bundle.heldout_camera_names or []),
        allow_unverified=allow_unverified_seed_provenance,
    )
    anchor_points = _seed_points_in_anchor_frame(
        points,
        bundle=bundle,
        seed_provenance=seed_provenance,
    )

    visible = torch.zeros(anchor_points.shape[0], dtype=torch.bool)
    anchor_points_h = torch.cat([anchor_points, torch.ones_like(anchor_points[:, :1])], dim=1)
    for view in range(bundle.train_view_count):
        camera_points = (anchor_points_h @ bundle.train_w2c[view, 0].detach().cpu().T)[:, :3]
        z = camera_points[:, 2]
        K = bundle.train_K[view].detach().cpu()
        u = K[0, 0] * camera_points[:, 0] / z.clamp_min(1.0e-5) + K[0, 2]
        v = K[1, 1] * camera_points[:, 1] / z.clamp_min(1.0e-5) + K[1, 2]
        visible |= (z > 0.1) & (u > -0.05 * width) & (u < 1.05 * width) & (v > -0.05 * height) & (v < 1.05 * height)
    anchor_points = anchor_points[visible]
    colors = colors[visible]
    if int(anchor_points.shape[0]) < seed_count:
        raise ValueError(
            f"Seed point cloud has only {int(anchor_points.shape[0])} train-visible points; "
            f"the browser bundle requested {seed_count}. Build a denser train-only SfM cloud "
            "or lower --dataset-seed-count explicitly."
        )
    selected = _farthest_point_subset(anchor_points, seed_count)
    seeds = torch.cat([anchor_points[selected], colors[selected]], dim=1)

    metadata = bundle.metadata or {}
    fps = float(metadata["fps"])
    start_seconds = float(metadata.get("source_start_seconds", 0.0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atlas_urls = _write_browser_frame_atlases(bundle, output_path)
    payload = {
        "version": BROWSER_MULTICAM_BUNDLE_VERSION,
        "dataset_contract": {
            "manifest": str(manifest_path),
            "sample_id": sample_id,
            "split": split,
            "train_cameras": list(bundle.train_camera_names),
            "heldout_cameras": list(bundle.heldout_camera_names or []),
            "anchor_camera": str(metadata["anchor_camera"]),
            "heldout_usage": "validation_only",
            "pose_source": bundle.pose_source,
            "camera_motion": "static_rig",
            "frame_decode": "sparse_exact" if sparse_frame_decode else "eager_then_select",
        },
        "dataset": str(metadata["dataset"]),
        "scene": str(metadata["scene"]),
        "name": f"{metadata['scene']} calibrated {split}",
        "decode_size": [width, height],
        "frame_count": len(frame_indices),
        "frame_indices": frame_indices,
        "frame_times_seconds": [start_seconds + float(index) / fps for index in frame_indices],
        "cameras": _browser_camera_rows(bundle, width=width, height=height, atlas_urls=atlas_urls),
        "seed_source": str(seed_point_cloud_path),
        "seed_provenance": seed_provenance,
        "seed_coordinate_frame": f"{metadata['anchor_camera']}_opencv",
        "seed_points_xyzrgb": seeds.round(decimals=7).tolist(),
    }
    output_path.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"Wrote browser multicam dataset bundle to {output_path}")
    return output_path


def _load_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    payload = load_torch_checkpoint(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
        state_dict = payload["state_dict"]
    elif isinstance(payload, dict) and payload and all(torch.is_tensor(value) for value in payload.values()):
        state_dict = payload
    else:
        raise ValueError(
            f"Could not find a plain state_dict in {checkpoint_path}. "
            "Expected either a tensor mapping or a mapping with a 'state_dict' entry."
        )
    return {str(name): value for name, value in state_dict.items()}


def export_id_from_config(config: dict[str, Any], *, suffix: str | None = None) -> str:
    export_cfg = config.get("export", {})
    explicit_id = export_cfg.get("id") if isinstance(export_cfg, dict) else None
    if explicit_id:
        return str(explicit_id)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = str(config.get("logging", {}).get("wandb_run_name") or "dynaworld")
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", run_name).strip("-")[:96] or "dynaworld"
    if suffix:
        slug = f"{slug}-{suffix}"
    return f"{timestamp}_{slug}"


def _load_sequence_from_config(
    resolved: dict[str, Any],
    *,
    device: torch.device,
    sequence_index: int,
) -> Any:
    data_cfg = resolved["data"]
    model_cfg = resolved["model"]
    if data_cfg["manifest_path"] is not None:
        sequences = load_manifest_sequences(
            data_cfg["manifest_path"],
            split=data_cfg["split"],
            data_cfg=data_cfg,
            model_cfg=model_cfg,
            device=device,
        )
        if sequence_index < 0 or sequence_index >= len(sequences):
            raise IndexError(
                f"sequence_index={sequence_index} is out of range for {len(sequences)} loaded manifest sequences."
            )
        return sequences[sequence_index]

    if sequence_index != 0:
        raise ValueError("sequence_index is only valid when data.manifest_path is set.")
    if data_cfg["sequence_dir"] is None:
        raise ValueError("config['data']['sequence_dir'] is required when manifest_path is not set.")
    if data_cfg["frame_source"] == "camera_json":
        camera_json_path = data_cfg["camera_json"] or (data_cfg["sequence_dir"] / "per_frame_cameras.json")
        return load_camera_sequence(
            camera_json_path=camera_json_path,
            target_size=model_cfg["size"],
            camera_image_size=data_cfg["camera_image_size"],
            max_frames=data_cfg["max_frames"],
            focal_mode=data_cfg["camera_focal_mode"],
            device=device,
        )
    if data_cfg["frame_source"] == "explicit_video" and data_cfg["video_path"] is None:
        raise ValueError("config['data']['video_path'] is required when frame_source='explicit_video'.")
    frames_dir = resolve_frames_dir(data_cfg["sequence_dir"], data_cfg["frames_dir"])
    return load_uncalibrated_sequence(
        sequence_dir=data_cfg["sequence_dir"],
        frames_dir=frames_dir,
        video_path=data_cfg["video_path"],
        target_size=model_cfg["size"],
        max_frames=data_cfg["max_frames"],
        frame_source=data_cfg["frame_source"],
        device=device,
    )


def _clip_indices(
    *,
    frame_count: int,
    train_frame_count: int,
    window_start: int,
    device: torch.device,
) -> torch.Tensor:
    window = min(int(train_frame_count), int(frame_count))
    if window < 1:
        raise ValueError("window must be at least 1 frame.")
    if window >= frame_count:
        return torch.arange(frame_count, device=device)
    start = max(0, min(int(window_start), frame_count - window))
    return torch.arange(start, start + window, device=device)


def _write_tensor(output_dir: Path, relative_path: str, tensor: torch.Tensor) -> dict[str, Any]:
    path = output_dir / relative_path
    value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    path.write_bytes(value.numpy().tobytes())
    return {
        "path": relative_path,
        "dtype": "float32",
        "shape": list(value.shape),
        "count": int(value.numel()),
    }


def _write_module_tensors(
    output_dir: Path,
    tensors: dict[str, dict[str, Any]],
    prefix: str,
    module: torch.nn.Module,
) -> None:
    for name, value in module.state_dict().items():
        key = f"{prefix}.{name}"
        filename = f"{key.replace('.', '_')}.f32"
        tensors[key] = _write_tensor(output_dir, filename, value)


def _gaussian_head_meta(head: Any) -> dict[str, Any]:
    return {
        "gaussians_per_token": int(head.gaussians_per_token),
        "xy_extent": float(head.xy_extent),
        "z_min": float(head.z_min),
        "z_max": float(head.z_min + head.z_extent),
        "z_extent": float(head.z_extent),
        "scale_init": float(head.scale_init),
    }


def _bounds_from_head_meta(
    *,
    static_meta: dict[str, Any],
    dynamic_meta: dict[str, Any],
    dynamic_motion_extent: float,
    dynamic_time_basis_count: int,
) -> dict[str, list[float]]:
    motion_bound = float(dynamic_motion_extent) * float(dynamic_time_basis_count)
    min_xyz = torch.tensor(
        [
            min(-float(static_meta["xy_extent"]), -float(dynamic_meta["xy_extent"]) - motion_bound),
            min(-float(static_meta["xy_extent"]), -float(dynamic_meta["xy_extent"]) - motion_bound),
            min(float(static_meta["z_min"]), float(dynamic_meta["z_min"]) - motion_bound),
        ],
        dtype=torch.float32,
    )
    max_xyz = torch.tensor(
        [
            max(float(static_meta["xy_extent"]), float(dynamic_meta["xy_extent"]) + motion_bound),
            max(float(static_meta["xy_extent"]), float(dynamic_meta["xy_extent"]) + motion_bound),
            max(float(static_meta["z_max"]), float(dynamic_meta["z_max"]) + motion_bound),
        ],
        dtype=torch.float32,
    )
    center = 0.5 * (min_xyz + max_xyz)
    return {
        "min": [float(value) for value in min_xyz.tolist()],
        "max": [float(value) for value in max_xyz.tolist()],
        "center": [float(value) for value in center.tolist()],
    }


def _load_model_input(
    resolved: dict[str, Any],
    sequence_data: Any,
    clip_frames: torch.Tensor,
    clip_times: torch.Tensor,
    *,
    device: torch.device,
) -> Any:
    backend = str(resolved["model"]["video_encoder_backend"]).lower()
    if backend in {"precomputed", "precomputed_ltx"}:
        if "features" not in resolved:
            raise ValueError(
                "This config uses a precomputed feature backend but has no 'features' section. "
                "Use the precomputed-feature trainer config or add the features block."
            )
        from video_feature_cache import VideoFeatureCache

        feature_cache = VideoFeatureCache(resolved["features"], device)
        return feature_cache.load_or_bake(sequence_data)
    del clip_times
    return clip_frames


def export_browser_bundle_from_model(
    *,
    model: torch.nn.Module,
    resolved: dict[str, Any],
    sequence_data: Any,
    clip_indices: torch.Tensor,
    clip_times: torch.Tensor,
    model_input: Any,
    output_dir: Path,
    config_path: Path | None = None,
    state_dict_path: Path | None,
    export_id: str | None = None,
) -> Path:
    if not bool(getattr(model, "use_static_dynamic_split", False)):
        raise ValueError(
            "The browser bundle exporter currently only supports models with "
            "model.static_tokens + model.dynamic_tokens enabled."
        )
    if not hasattr(model, "static_gaussian_heads") or not hasattr(model, "dynamic_gaussian_heads"):
        raise ValueError("Model is marked as static/dynamic split but missing split Gaussian head modules.")

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), fast_attn_context(clip_times.device):
            video_tokens = model.video_encoder(model_input, frame_times=clip_times)
            fixed_queries = model.refine_queries(video_tokens, decode_time=None).squeeze(0)
            batched_fixed_queries = fixed_queries.unsqueeze(0)
            if hasattr(model, "decoded_static_query_tokens") and hasattr(model, "decoded_dynamic_query_tokens"):
                static_query_tokens = model.decoded_static_query_tokens(batched_fixed_queries).squeeze(0)
                dynamic_query_tokens = model.decoded_dynamic_query_tokens(batched_fixed_queries).squeeze(0)
            else:
                static_query_tokens = fixed_queries[2 : 2 + int(model.static_tokens)]
                dynamic_query_tokens = fixed_queries[2 + int(model.static_tokens) :]
    finally:
        if was_training:
            model.train()

    output_dir.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, dict[str, Any]] = {}
    tensors["refined_queries"] = _write_tensor(output_dir, "refined_queries.f32", fixed_queries)
    tensors["static_query_tokens"] = _write_tensor(output_dir, "static_query_tokens.f32", static_query_tokens)
    tensors["dynamic_query_tokens"] = _write_tensor(output_dir, "dynamic_query_tokens.f32", dynamic_query_tokens)
    _write_module_tensors(output_dir, tensors, "static_gaussian_heads", model.static_gaussian_heads)
    _write_module_tensors(output_dir, tensors, "dynamic_gaussian_heads.base_heads", model.dynamic_gaussian_heads.base_heads)
    _write_module_tensors(output_dir, tensors, "dynamic_gaussian_heads.motion_head", model.dynamic_gaussian_heads.motion_head)
    _write_module_tensors(output_dir, tensors, "dynamic_gaussian_heads.rotation_head", model.dynamic_gaussian_heads.rotation_head)
    _write_module_tensors(output_dir, tensors, "dynamic_gaussian_heads.alpha_head", model.dynamic_gaussian_heads.alpha_head)
    _write_module_tensors(output_dir, tensors, "time_proj", model.time_proj)
    _write_module_tensors(output_dir, tensors, "head_time_proj", model.head_time_proj)

    static_meta = _gaussian_head_meta(model.static_gaussian_heads)
    dynamic_base_meta = _gaussian_head_meta(model.dynamic_gaussian_heads.base_heads)
    bounds = _bounds_from_head_meta(
        static_meta=static_meta,
        dynamic_meta=dynamic_base_meta,
        dynamic_motion_extent=float(model.dynamic_gaussian_heads.motion_extent),
        dynamic_time_basis_count=int(model.dynamic_time_basis_count),
    )
    clip_index_values = clip_indices.detach().cpu().tolist()
    clip_time_values = clip_times.squeeze(0).detach().cpu().tolist()
    manifest = {
        "version": EXPORT_BUNDLE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "export_id": export_id,
        "config_path": None if config_path is None else str(config_path),
        "state_dict_path": None if state_dict_path is None else str(state_dict_path),
        "source": {
            "sequence_index": int(resolved.get("export", {}).get("sequence_index", 0)),
            "sequence_path": str(sequence_data.source_path),
            "frame_count": int(sequence_data.frame_count),
            "frame_source": str(sequence_data.frame_source),
            "clip_indices": [int(value) for value in clip_index_values],
            "clip_times": [float(value) for value in clip_time_values],
            "window_start": int(clip_index_values[0]) if clip_index_values else 0,
        },
        "model": {
            "variant": str(resolved["model"]["variant"]),
            "video_encoder_backend": str(resolved["model"]["video_encoder_backend"]),
            "train_frame_count": int(resolved["model"]["train_frame_count"]),
            "feat_dim": int(model.feat_dim),
            "num_tokens": int(model.num_tokens),
            "gaussians_per_token": int(model.gaussians_per_token),
            "static_tokens": int(model.static_tokens),
            "dynamic_tokens": int(model.dynamic_tokens),
            "token_layout": resolved["model"].get("token_layout"),
            "dynamic_time_basis_count": int(model.dynamic_time_basis_count),
            "dynamic_time_max_frequency": float(model.dynamic_time_max_frequency),
            "image_size": int(model.image_size),
            "bundle_contract": "refined_tokens_plus_decoder_heads",
            "notes": [
                "The viewer decodes static and dynamic splats from saved refined query tokens plus Gaussian head MLP weights.",
                "No decoded Gaussian arrays are saved in this bundle.",
            ],
        },
        "decoder": {
            "static_gaussian_heads": static_meta,
            "dynamic_gaussian_heads": {
                "base_heads": dynamic_base_meta,
                "time_basis_count": int(model.dynamic_gaussian_heads.time_basis_count),
                "motion_extent": float(model.dynamic_gaussian_heads.motion_extent),
                "rotation_radians": float(model.dynamic_gaussian_heads.rotation_radians),
                "alpha_logit_extent": float(model.dynamic_gaussian_heads.alpha_logit_extent),
            },
            "time_proj": {"type": model.time_proj.__class__.__name__},
            "head_time_proj": {"type": model.head_time_proj.__class__.__name__},
        },
        "viewer_defaults": {
            "fov_degrees": 60.0,
            "near": 0.01,
            "far": 100.0,
            "time_domain": [0.0, 1.0],
        },
        "bounds": bounds,
        "counts": {
            "static_gaussians": int(model.static_tokens * model.gaussians_per_token),
            "dynamic_gaussians": int(model.dynamic_tokens * model.gaussians_per_token),
            "total_gaussians": int((model.static_tokens + model.dynamic_tokens) * model.gaussians_per_token),
        },
        "tensors": tensors,
    }
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest, sort_keys=False)
    print(f"Wrote browser bundle manifest to {manifest_path}")
    return manifest_path


def export_browser_bundle(
    *,
    config_path: Path,
    output_dir: Path,
    state_dict_path: Path | None,
    sequence_index: int,
    window_start: int,
) -> Path:
    resolved = resolve_config_for_arch(load_config_file(config_path), config_path)
    device = pick_device()
    print(f"Using device: {device}")

    model = build_model_from_config(resolved).to(device)
    if state_dict_path is not None:
        state_dict = _load_state_dict(state_dict_path)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise ValueError(
                "Checkpoint did not match model strictly.\n"
                f"Missing keys: {sorted(missing)}\n"
                f"Unexpected keys: {sorted(unexpected)}"
            )

    sequence_data = _load_sequence_from_config(resolved, device=device, sequence_index=sequence_index)
    clip_indices = _clip_indices(
        frame_count=sequence_data.frame_count,
        train_frame_count=resolved["model"]["train_frame_count"],
        window_start=window_start,
        device=device,
    )
    clip_frames, clip_times = prepare_clip(sequence_data, clip_indices)
    model_input = _load_model_input(resolved, sequence_data, clip_frames, clip_times, device=device)
    export_id = export_id_from_config(resolved)
    return export_browser_bundle_from_model(
        model=model,
        resolved=resolved,
        sequence_data=sequence_data,
        clip_indices=clip_indices,
        clip_times=clip_times,
        model_input=model_input,
        output_dir=output_dir,
        config_path=config_path,
        state_dict_path=state_dict_path,
        export_id=export_id,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a browser-loadable Dynaworld static/dynamic split bundle. "
            "This saves refined token arrays plus the Gaussian decoder head MLP "
            "weights needed to decode splats in the browser."
        )
    )
    parser.add_argument("config", type=Path, nargs="?", help="Path to the Dynaworld JSONC train config.")
    parser.add_argument("--output-dir", type=Path, help="Directory to write the trained-model browser bundle.")
    parser.add_argument(
        "--state-dict",
        type=Path,
        default=None,
        help="Optional model state_dict/checkpoint path. If omitted, exports the random-init model state.",
    )
    parser.add_argument(
        "--sequence-index",
        type=int,
        default=0,
        help="Sequence index to export when data.manifest_path is set. Default: 0.",
    )
    parser.add_argument(
        "--window-start",
        type=int,
        default=0,
        help="Deterministic start index for the exported clip window. Default: 0.",
    )
    parser.add_argument("--dataset-manifest", type=Path, help="Canonical multicam manifest for a dataset demo bundle.")
    parser.add_argument("--dataset-sample-id", help="Sample id in --dataset-manifest.")
    parser.add_argument("--dataset-split", default="train2_holdout1", help="Manifest split for the dataset bundle.")
    parser.add_argument("--dataset-output", type=Path, help="JSON path for the dataset demo bundle.")
    parser.add_argument("--seed-point-cloud", type=Path, help="SfM/COLMAP point cloud used only for browser initialization.")
    parser.add_argument(
        "--seed-provenance-report",
        type=Path,
        help=(
            "JSON report declaring seed method, input_cameras, and train_only_verified. "
            "Camera claims are checked against the canonical dataset split."
        ),
    )
    parser.add_argument(
        "--allow-unverified-seed-provenance",
        action="store_true",
        help=(
            "Explicitly allow an external seed source whose train-only provenance is not verified. "
            "The bundle will serialize train_only_verified=false."
        ),
    )
    parser.add_argument("--dataset-height", type=int, default=72)
    parser.add_argument("--dataset-width", type=int, default=96)
    parser.add_argument("--dataset-frame-count", type=int, default=8)
    parser.add_argument("--dataset-native-frame-count", type=int, default=300)
    parser.add_argument("--dataset-seed-count", type=int, default=768)
    parser.add_argument(
        "--dataset-sparse-frame-decode",
        action="store_true",
        help=(
            "Decode each requested Neural 3D Video timestamp independently instead of "
            "materializing every source frame up to the final requested index."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset_manifest is not None:
        required = {
            "--dataset-sample-id": args.dataset_sample_id,
            "--dataset-output": args.dataset_output,
            "--seed-point-cloud": args.seed_point_cloud,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise SystemExit(f"Dataset export requires: {', '.join(missing)}")
        if args.dataset_frame_count < 2 or args.dataset_native_frame_count < args.dataset_frame_count:
            raise SystemExit("Dataset frame counts must satisfy native >= sampled >= 2.")
        frame_indices = torch.linspace(
            0,
            args.dataset_native_frame_count - 1,
            args.dataset_frame_count,
        ).round().to(dtype=torch.long).tolist()
        export_browser_multicam_dataset_bundle(
            manifest_path=args.dataset_manifest,
            sample_id=args.dataset_sample_id,
            split=args.dataset_split,
            seed_point_cloud_path=args.seed_point_cloud,
            output_path=args.dataset_output,
            target_size=(args.dataset_height, args.dataset_width),
            frame_indices=frame_indices,
            seed_count=args.dataset_seed_count,
            seed_provenance_report_path=args.seed_provenance_report,
            allow_unverified_seed_provenance=args.allow_unverified_seed_provenance,
            sparse_frame_decode=args.dataset_sparse_frame_decode,
        )
        return
    if args.config is None or args.output_dir is None:
        raise SystemExit("Model export requires config and --output-dir.")
    export_browser_bundle(
        config_path=args.config,
        output_dir=args.output_dir,
        state_dict_path=args.state_dict,
        sequence_index=args.sequence_index,
        window_start=args.window_start,
    )


if __name__ == "__main__":
    main()
