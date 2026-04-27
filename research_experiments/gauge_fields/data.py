from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

from common import resolve_dynaworld_path
from config_utils import path_or_none
from multicam_val_data import load_multicam_val_camera_frames, load_multicam_val_manifest, load_multicam_val_sample
from sequence_data import load_uncalibrated_sequence


@dataclass
class GaugeVideoBundle:
    video: torch.Tensor
    K: torch.Tensor
    w2c: torch.Tensor
    fps: float = 4.0
    source_path: str | None = None
    metadata: dict[str, Any] | None = None
    heldout_video: torch.Tensor | None = None
    heldout_K: torch.Tensor | None = None
    heldout_w2c: torch.Tensor | None = None
    heldout_pose_source: str | None = None
    train_videos: torch.Tensor | None = None
    train_K: torch.Tensor | None = None
    train_w2c: torch.Tensor | None = None
    train_camera_names: list[str] | None = None
    heldout_camera_name: str | None = None


def make_fixed_pinhole_camera(
    num_frames: int,
    H: int,
    W: int,
    fov_degrees: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    fov = math.radians(fov_degrees)
    focal = 0.5 * float(W) / math.tan(0.5 * fov)
    K = torch.tensor(
        [
            [focal, 0.0, float(W) * 0.5],
            [0.0, focal, float(H) * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    w2c = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0).repeat(num_frames, 1, 1)
    return K, w2c


def make_scaled_intrinsics(
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    source_width: float,
    source_height: float,
    target_width: int,
    target_height: int,
    device: torch.device,
) -> torch.Tensor:
    sx = float(target_width) / float(source_width)
    sy = float(target_height) / float(source_height)
    return torch.tensor(
        [
            [float(fx) * sx, 0.0, float(cx) * sx],
            [0.0, float(fy) * sy, float(cy) * sy],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )


def rodrigues_matrix(axis_angle: list[float] | tuple[float, ...], device: torch.device) -> torch.Tensor:
    r = torch.tensor(axis_angle, dtype=torch.float32, device=device)
    theta = torch.linalg.norm(r).clamp_min(1e-8)
    rx, ry, rz = r
    skew = torch.stack(
        [
            torch.stack([r.new_zeros(()), -rz, ry]),
            torch.stack([rz, r.new_zeros(()), -rx]),
            torch.stack([-ry, rx, r.new_zeros(())]),
        ]
    )
    eye = torch.eye(3, dtype=torch.float32, device=device)
    return eye + (torch.sin(theta) / theta) * skew + ((1.0 - torch.cos(theta)) / (theta * theta)) * (skew @ skew)


def deepview_camera_from_models(
    record: dict[str, Any],
    camera_name: str,
    *,
    H: int,
    W: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    models_path = Path(record["models_path"])
    models = json.loads(models_path.read_text(encoding="utf-8"))
    by_name = {str(model["name"]): model for model in models}
    if camera_name not in by_name:
        raise KeyError(f"DeepView camera {camera_name!r} not found in {models_path}.")
    model = by_name[camera_name]

    focal = float(model["focal_length"])
    pixel_aspect = float(model.get("pixel_aspect_ratio", 1.0))
    principal = model["principal_point"]
    K = make_scaled_intrinsics(
        fx=focal,
        fy=focal * pixel_aspect,
        cx=float(principal[0]),
        cy=float(principal[1]),
        source_width=float(model["width"]),
        source_height=float(model["height"]),
        target_width=W,
        target_height=H,
        device=device,
    )

    # DeepView stores a Rodrigues world-to-camera rotation for an OpenGL-style
    # camera. Convert it to the gauge renderer's camera frame: +x right,
    # +y down, +z forward.
    w2c_gl_rot = rodrigues_matrix(model["orientation"], device=device)
    c2w_gl_rot = w2c_gl_rot.T
    gl_to_plus_z = torch.diag(torch.tensor([1.0, -1.0, -1.0], dtype=torch.float32, device=device))
    c2w = torch.eye(4, dtype=torch.float32, device=device)
    c2w[:3, :3] = c2w_gl_rot @ gl_to_plus_z
    c2w[:3, 3] = torch.tensor(model["position"], dtype=torch.float32, device=device)
    return K, c2w


def make_multicam_pair_cameras(
    record: dict[str, Any],
    *,
    T: int,
    H: int,
    W: int,
    fov_degrees: float,
    pose_source: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
    if pose_source not in {"auto", "deepview", "source_proxy"}:
        raise ValueError("camera.multicam_pose_source must be one of: auto, deepview, source_proxy")

    if pose_source in {"auto", "deepview"} and record.get("dataset") == "deepview_video" and record.get("models_path"):
        source_K, source_c2w = deepview_camera_from_models(
            record,
            str(record["source_camera"]),
            H=H,
            W=W,
            device=device,
        )
        target_K, target_c2w = deepview_camera_from_models(
            record,
            str(record["target_camera"]),
            H=H,
            W=W,
            device=device,
        )
        source_w2c = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0).repeat(T, 1, 1)
        target_w2c = torch.linalg.inv(target_c2w) @ source_c2w
        target_w2c = target_w2c.unsqueeze(0).repeat(T, 1, 1)
        return source_K, source_w2c, target_K, target_w2c, "deepview_models_relative_pinhole"

    if pose_source == "deepview":
        raise ValueError(
            f"Requested DeepView camera calibration for non-DeepView record {record.get('sample_id')!r}."
        )

    source_K, source_w2c = make_fixed_pinhole_camera(
        num_frames=T,
        H=H,
        W=W,
        fov_degrees=fov_degrees,
        device=device,
    )
    target_K = source_K.clone()
    target_w2c = source_w2c.clone()
    return source_K, source_w2c, target_K, target_w2c, "source_camera_proxy_uncalibrated"


def make_deepview_multiview_cameras(
    record: dict[str, Any],
    *,
    train_cameras: list[str],
    heldout_camera: str,
    anchor_camera: str,
    T: int,
    H: int,
    W: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
    if record.get("dataset") != "deepview_video" or not record.get("models_path"):
        raise ValueError("Configured multicam_train_cameras currently require a DeepView record with models_path.")

    _, anchor_c2w = deepview_camera_from_models(record, anchor_camera, H=H, W=W, device=device)
    train_K = []
    train_w2c = []
    for camera_name in train_cameras:
        K, c2w = deepview_camera_from_models(record, camera_name, H=H, W=W, device=device)
        rel_w2c = torch.linalg.inv(c2w) @ anchor_c2w
        train_K.append(K)
        train_w2c.append(rel_w2c.unsqueeze(0).repeat(T, 1, 1))

    heldout_K, heldout_c2w = deepview_camera_from_models(record, heldout_camera, H=H, W=W, device=device)
    heldout_w2c = torch.linalg.inv(heldout_c2w) @ anchor_c2w
    return (
        torch.stack(train_K, dim=0),
        torch.stack(train_w2c, dim=0),
        heldout_K,
        heldout_w2c.unsqueeze(0).repeat(T, 1, 1),
        "deepview_models_relative_pinhole",
    )


def load_baseline_video(
    sequence_dir: Path,
    frames_dir: Optional[Path],
    video_path: Optional[Path],
    frame_source: str,
    render_size: int,
    max_frames: int,
    device: torch.device,
) -> torch.Tensor:
    sequence = load_uncalibrated_sequence(
        sequence_dir=sequence_dir,
        frames_dir=frames_dir,
        video_path=video_path,
        target_size=render_size,
        max_frames=max_frames,
        frame_source=frame_source,
        device=device,
    )
    return sequence.frames.permute(0, 2, 3, 1).contiguous()


def select_multicam_record(data_cfg: dict[str, Any]) -> dict[str, Any]:
    manifest_path = resolve_dynaworld_path(data_cfg["multicam_manifest"])
    records = load_multicam_val_manifest(manifest_path, split=data_cfg["multicam_split"])
    sample_id = data_cfg.get("multicam_sample_id")
    if sample_id:
        for record in records:
            if str(record.get("sample_id")) == str(sample_id):
                return record
        raise ValueError(f"multicam_sample_id={sample_id!r} was not found in {manifest_path}.")

    index = int(data_cfg.get("multicam_sample_index", 0))
    if index < 0 or index >= len(records):
        raise IndexError(f"multicam_sample_index={index} out of range for {len(records)} records.")
    return records[index]


def select_configured_frames(video: torch.Tensor, frame_indices: Any) -> torch.Tensor:
    if frame_indices is None:
        return video
    if not isinstance(frame_indices, list) or not frame_indices:
        raise ValueError("data.frame_indices must be a non-empty list of integer frame indices when provided.")
    indices = torch.as_tensor(frame_indices, dtype=torch.long, device=video.device)
    if bool((indices < 0).any()) or bool((indices >= video.shape[0]).any()):
        raise IndexError(f"data.frame_indices {frame_indices!r} out of range for {video.shape[0]} loaded frames.")
    return video[indices].contiguous()


def select_configured_multiview_frames(videos: torch.Tensor, frame_indices: Any) -> torch.Tensor:
    if frame_indices is None:
        return videos
    if not isinstance(frame_indices, list) or not frame_indices:
        raise ValueError("data.frame_indices must be a non-empty list of integer frame indices when provided.")
    indices = torch.as_tensor(frame_indices, dtype=torch.long, device=videos.device)
    if bool((indices < 0).any()) or bool((indices >= videos.shape[1]).any()):
        raise IndexError(f"data.frame_indices {frame_indices!r} out of range for {videos.shape[1]} loaded frames.")
    return videos[:, indices].contiguous()


def deepview_video_path_for_camera(record: dict[str, Any], camera_name: str) -> Path:
    scene_dir = Path(record["dataset_scene_dir"])
    path = scene_dir / f"{camera_name}.mp4"
    if not path.exists():
        raise FileNotFoundError(f"DeepView camera video not found: {path}")
    return path


def load_deepview_camera_video(
    record: dict[str, Any],
    camera_name: str,
    *,
    target_size: int,
    device: torch.device,
) -> torch.Tensor:
    frames = load_multicam_val_camera_frames(
        video_path=deepview_video_path_for_camera(record, camera_name),
        start_seconds=float(record.get("source_start_seconds", record.get("target_start_seconds", 0.0))),
        fps=float(record["fps"]),
        frame_count=int(record["frame_count"]),
        target_size=target_size,
        device=device,
    )
    return frames.permute(0, 2, 3, 1).contiguous()


def load_gauge_video_bundle(
    *,
    data_cfg: dict[str, Any],
    camera_cfg: dict[str, Any],
    render_size: int,
    device: torch.device,
) -> GaugeVideoBundle:
    frame_source = str(data_cfg["frame_source"])
    if frame_source == "multicam_val":
        record = select_multicam_record(data_cfg)
        train_cameras_raw = data_cfg.get("multicam_train_cameras")
        if train_cameras_raw:
            train_cameras = [str(camera) for camera in train_cameras_raw]
            heldout_camera = str(data_cfg.get("multicam_heldout_camera") or record["target_camera"])
            anchor_camera = str(data_cfg.get("multicam_anchor_camera") or train_cameras[0])
            if anchor_camera not in train_cameras:
                raise ValueError("data.multicam_anchor_camera must be one of data.multicam_train_cameras.")

            train_videos = torch.stack(
                [
                    load_deepview_camera_video(record, camera, target_size=render_size, device=device)
                    for camera in train_cameras
                ],
                dim=0,
            )
            heldout_video = load_deepview_camera_video(record, heldout_camera, target_size=render_size, device=device)
            max_frames = int(data_cfg["max_frames"])
            if max_frames > 0:
                train_videos = train_videos[:, :max_frames].contiguous()
                heldout_video = heldout_video[:max_frames].contiguous()
            train_videos = select_configured_multiview_frames(train_videos, data_cfg["frame_indices"])
            heldout_video = select_configured_frames(heldout_video, data_cfg["frame_indices"])

            _, T, H, W, _ = train_videos.shape
            train_K, train_w2c, heldout_K, heldout_w2c, pose_note = make_deepview_multiview_cameras(
                record,
                train_cameras=train_cameras,
                heldout_camera=heldout_camera,
                anchor_camera=anchor_camera,
                T=T,
                H=H,
                W=W,
                device=device,
            )
            return GaugeVideoBundle(
                video=train_videos[0],
                K=train_K[0],
                w2c=train_w2c[0],
                fps=float(record.get("fps", 4.0)),
                source_path=",".join(str(deepview_video_path_for_camera(record, camera)) for camera in train_cameras),
                metadata={
                    **record,
                    "train_cameras": train_cameras,
                    "heldout_camera": heldout_camera,
                    "anchor_camera": anchor_camera,
                },
                heldout_video=heldout_video,
                heldout_K=heldout_K,
                heldout_w2c=heldout_w2c,
                heldout_pose_source=pose_note,
                train_videos=train_videos,
                train_K=train_K,
                train_w2c=train_w2c,
                train_camera_names=train_cameras,
                heldout_camera_name=heldout_camera,
            )

        sample = load_multicam_val_sample(record, target_size=render_size, device=device)
        video = sample.source_frames.permute(0, 2, 3, 1).contiguous()
        heldout_video = sample.target_frames.permute(0, 2, 3, 1).contiguous()
        max_frames = int(data_cfg["max_frames"])
        if max_frames > 0:
            video = video[:max_frames].contiguous()
            heldout_video = heldout_video[:max_frames].contiguous()
        video = select_configured_frames(video, data_cfg["frame_indices"])
        heldout_video = select_configured_frames(heldout_video, data_cfg["frame_indices"])
        if video.shape != heldout_video.shape:
            raise ValueError(
                f"Source/heldout frame shape mismatch for {record.get('sample_id')}: "
                f"{tuple(video.shape)} vs {tuple(heldout_video.shape)}"
            )
        T, H, W, _ = video.shape
        K, w2c, heldout_K, heldout_w2c, pose_note = make_multicam_pair_cameras(
            record,
            T=T,
            H=H,
            W=W,
            fov_degrees=float(camera_cfg["base_fov_degrees"]),
            pose_source=str(camera_cfg["multicam_pose_source"]),
            device=device,
        )
        return GaugeVideoBundle(
            video=video,
            K=K,
            w2c=w2c,
            fps=float(record.get("fps", 4.0)),
            source_path=str(record.get("source_video_path")),
            metadata=record,
            heldout_video=heldout_video,
            heldout_K=heldout_K,
            heldout_w2c=heldout_w2c,
            heldout_pose_source=pose_note,
        )

    sequence_dir = resolve_dynaworld_path(data_cfg["sequence_dir"])
    frames_dir = path_or_none(data_cfg["frames_dir"])
    if frames_dir is not None:
        frames_dir = resolve_dynaworld_path(frames_dir)
    video_path = path_or_none(data_cfg["video_path"])
    if video_path is not None:
        video_path = resolve_dynaworld_path(video_path)
    video = load_baseline_video(
        sequence_dir=sequence_dir,
        frames_dir=frames_dir,
        video_path=video_path,
        frame_source=frame_source,
        render_size=render_size,
        max_frames=int(data_cfg["max_frames"]),
        device=device,
    )
    video = select_configured_frames(video, data_cfg["frame_indices"])
    T, H, W, _ = video.shape
    K, w2c = make_fixed_pinhole_camera(
        num_frames=T,
        H=H,
        W=W,
        fov_degrees=float(camera_cfg["base_fov_degrees"]),
        device=device,
    )
    return GaugeVideoBundle(
        video=video,
        K=K,
        w2c=w2c,
        fps=4.0,
        source_path=str(video_path) if video_path is not None else None,
        metadata=None,
    )


def initialize_material_points_from_first_frame(
    video: torch.Tensor,
    K: torch.Tensor,
    num_elements: int,
    init_depth: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, H, W, _ = video.shape
    device = video.device

    grid_x = max(1, math.ceil(math.sqrt(float(num_elements) * float(W) / float(H))))
    grid_y = max(1, math.ceil(float(num_elements) / float(grid_x)))

    xs = torch.linspace(0.5, float(W) - 0.5, grid_x, device=device)
    ys = torch.linspace(0.5, float(H) - 0.5, grid_y, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    pixels = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)[:num_elements]

    z = torch.full((pixels.shape[0],), init_depth, device=device)
    x = (pixels[:, 0] - K[0, 2]) * z / K[0, 0]
    y = (pixels[:, 1] - K[1, 2]) * z / K[1, 1]
    x0 = torch.stack([x, y, z], dim=-1)

    px = pixels[:, 0].round().long().clamp(0, W - 1)
    py = pixels[:, 1].round().long().clamp(0, H - 1)
    color = video[0, py, px]
    return x0, color


def initialize_material_points_from_multiview_first_frames(
    videos: torch.Tensor,
    K: torch.Tensor,
    w2c: torch.Tensor,
    num_elements: int,
    init_depth: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if videos.ndim != 5:
        raise ValueError(f"Expected videos [V,T,H,W,3], got {tuple(videos.shape)}.")
    view_count, _T, H, W, _C = videos.shape
    device = videos.device
    base = int(num_elements) // int(view_count)
    rem = int(num_elements) % int(view_count)

    points = []
    colors = []
    for view in range(int(view_count)):
        count = base + (1 if view < rem else 0)
        if count <= 0:
            continue
        K_view = K[view, 0] if K.ndim == 4 else (K[view] if K.ndim == 3 else K)
        w2c_view = w2c[view, 0] if w2c.ndim == 4 else w2c[0]

        grid_x = max(1, math.ceil(math.sqrt(float(count) * float(W) / float(H))))
        grid_y = max(1, math.ceil(float(count) / float(grid_x)))
        xs = torch.linspace(0.5, float(W) - 0.5, grid_x, device=device)
        ys = torch.linspace(0.5, float(H) - 0.5, grid_y, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        pixels = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)[:count]

        z = torch.full((pixels.shape[0],), init_depth, device=device)
        x = (pixels[:, 0] - K_view[0, 2]) * z / K_view[0, 0]
        y = (pixels[:, 1] - K_view[1, 2]) * z / K_view[1, 1]
        x_cam = torch.stack([x, y, z], dim=-1)
        xh = torch.cat([x_cam, torch.ones(x_cam.shape[0], 1, device=device)], dim=-1)
        c2w_view = torch.linalg.inv(w2c_view)
        points.append((xh @ c2w_view.T)[:, :3])

        px = pixels[:, 0].round().long().clamp(0, W - 1)
        py = pixels[:, 1].round().long().clamp(0, H - 1)
        colors.append(videos[view, 0, py, px])

    if not points:
        raise ValueError("num_elements must allocate at least one material point.")
    return torch.cat(points, dim=0)[:num_elements].contiguous(), torch.cat(colors, dim=0)[:num_elements].contiguous()
