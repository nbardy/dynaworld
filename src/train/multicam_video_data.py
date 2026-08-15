"""Multicam bundle loaders for novel-view training.

This module owns the calibrated multi-camera side of
``research_notes/data_contract.md``: train/condition cameras go in and heldout
cameras provide novel-view supervision. Broad one-camera scale pretraining
belongs in ``sequence_data.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from camera import CameraSpec, build_look_at_camera_to_world
from json_io import load_json
from multicam_val_data import ImageSizeLike, load_multicam_val_camera_frames, load_multicam_val_manifest
from runtime_types import SequenceData
from sequence_data import normalize_frame_times


@dataclass(frozen=True)
class MulticamVideoFrameSource:
    """Metadata-only logical frame selection for one synchronized MP4."""

    camera_name: str
    video_path: Path
    start_seconds: float
    sample_fps: float
    source_frame_count: int
    selected_frame_indices: tuple[int, ...]
    height: int
    width: int

    def __post_init__(self) -> None:
        if not self.video_path.is_file():
            raise FileNotFoundError(f"camera video does not exist: {self.video_path}")
        if self.video_path.suffix.lower() != ".mp4":
            raise ValueError(f"deferred camera targets require MP4 input: {self.video_path}")
        if self.sample_fps <= 0.0 or self.source_frame_count < 1:
            raise ValueError("camera video metadata requires positive fps and frame count")
        if self.height < 1 or self.width < 1 or not self.selected_frame_indices:
            raise ValueError("camera video metadata requires positive output dimensions and selected frames")
        invalid = next(
            (index for index in self.selected_frame_indices if index < 0 or index >= self.source_frame_count),
            None,
        )
        if invalid is not None:
            raise IndexError(f"selected camera frame {invalid} is outside [0, {self.source_frame_count})")


@dataclass
class MulticamVideoBundle:
    condition_sequence: SequenceData
    train_sequences: tuple[SequenceData, ...]
    train_frames: torch.Tensor
    train_K: torch.Tensor
    train_w2c: torch.Tensor
    train_camera_names: list[str]
    train_lens_models: list[str] | None = None
    train_distortions: torch.Tensor | None = None
    heldout_sequences: tuple[SequenceData, ...] = ()
    heldout_frames: torch.Tensor | None = None
    heldout_K: torch.Tensor | None = None
    heldout_w2c: torch.Tensor | None = None
    heldout_camera_names: list[str] | None = None
    heldout_lens_models: list[str] | None = None
    heldout_distortions: torch.Tensor | None = None
    pose_source: str | None = None
    anchor_c2w: torch.Tensor | None = None
    metadata: dict[str, Any] | None = None
    train_frame_sources: tuple[MulticamVideoFrameSource, ...] = ()
    heldout_frame_sources: tuple[MulticamVideoFrameSource, ...] = ()
    deferred_target_frames: bool = False

    @property
    def train_view_count(self) -> int:
        return int(self.train_frames.shape[0])

    @property
    def frame_count(self) -> int:
        return int(self.train_frames.shape[1])

    @property
    def heldout_view_count(self) -> int:
        if self.heldout_frames is None:
            return 0
        return int(self.heldout_frames.shape[0])

    @property
    def heldout_camera_name(self) -> str | None:
        if not self.heldout_camera_names:
            return None
        return self.heldout_camera_names[0]


CAMXTIME_DATASETS = {"camxtime", "camxtime_full_grid", "camxtime_eval_gt"}
DNERF_DATASETS = {"dnerf"}
CAMXTIME_TRAJECTORY_VIDEOS = {
    "moving_forward",
    "moving_backward",
    "moving_zigzag",
    "moving_bullettime",
    "moving_slowmo",
}


def dnerf_scene_dir(record: dict[str, Any]) -> Path:
    scene_dir = record.get("dnerf_scene_dir") or record.get("dataset_scene_dir")
    if not scene_dir:
        raise ValueError(f"D-NeRF record {record.get('sample_id')!r} is missing its scene directory.")
    return Path(scene_dir)


def dnerf_camera_split(record: dict[str, Any], camera_name: str) -> str:
    mapping = record.get("dnerf_camera_splits")
    if not isinstance(mapping, dict) or camera_name not in mapping:
        raise ValueError(
            f"D-NeRF camera {camera_name!r} has no split mapping on record {record.get('sample_id')!r}."
        )
    return str(mapping[camera_name])


def dnerf_camera_frames(record: dict[str, Any], camera_name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    split = dnerf_camera_split(record, camera_name)
    payload = load_json(dnerf_scene_dir(record) / f"transforms_{split}.json")
    raw_frames = payload.get("frames")
    index_mapping = record.get("dnerf_frame_indices")
    if not isinstance(raw_frames, list) or not isinstance(index_mapping, dict) or camera_name not in index_mapping:
        raise ValueError(f"D-NeRF record {record.get('sample_id')!r} has no indexed frames for {camera_name!r}.")
    frames = [raw_frames[int(index)] for index in index_mapping[camera_name]]
    expected_times = [float(value) for value in record.get("dnerf_times", ())]
    actual_times = [float(frame["time"]) for frame in frames]
    if len(actual_times) != len(expected_times) or any(
        not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-9)
        for actual, expected in zip(actual_times, expected_times)
    ):
        raise ValueError(f"D-NeRF {camera_name!r} frames do not match the declared paired times.")
    return payload, frames


def dnerf_image_path(record: dict[str, Any], frame: dict[str, Any]) -> Path:
    path = dnerf_scene_dir(record) / str(frame["file_path"])
    return path if path.suffix else path.with_suffix(".png")


def load_dnerf_camera_frames(
    record: dict[str, Any],
    camera_name: str,
    *,
    target_size: ImageSizeLike,
    device: torch.device,
    frame_count: int,
) -> torch.Tensor:
    import numpy as np  # noqa: WPS433 -- ingestion-only dependency.
    from PIL import Image  # noqa: WPS433 -- ingestion-only dependency.

    _payload, frames = dnerf_camera_frames(record, camera_name)
    if isinstance(target_size, int):
        height = width = int(target_size)
    elif isinstance(target_size, (tuple, list)) and len(target_size) == 2:
        height, width = (int(value) for value in target_size)
    else:
        raise ValueError("target_size must be an integer or [height, width]")
    background = tuple(int(round(255.0 * float(value))) for value in record.get("dnerf_background", [0, 0, 0]))
    tensors = []
    for frame in frames[: int(frame_count)]:
        with Image.open(dnerf_image_path(record, frame)) as image:
            rgba = image.convert("RGBA")
            canvas = Image.new("RGBA", rgba.size, (*background, 255))
            rgb = Image.alpha_composite(canvas, rgba).convert("RGB")
            rgb = rgb.resize((width, height), resample=Image.Resampling.BILINEAR)
            array = np.asarray(rgb, dtype=np.float32) / 255.0
        tensors.append(torch.from_numpy(array).permute(2, 0, 1).contiguous())
    if len(tensors) != int(frame_count):
        raise ValueError(f"D-NeRF {camera_name!r} has {len(tensors)} frames; expected {frame_count}.")
    return torch.stack(tensors, dim=0).to(device=device)


def dnerf_c2w(frame: dict[str, Any], *, device: torch.device) -> torch.Tensor:
    c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32, device=device)
    if c2w.shape != (4, 4):
        raise ValueError("D-NeRF transform_matrix must be 4x4.")
    return c2w @ torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], device=device))


def make_dnerf_matched_trajectory_cameras(
    record: dict[str, Any],
    *,
    train_cameras: list[str],
    heldout_cameras: list[str],
    frame_positions: list[int],
    H: int,
    W: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
    if len(train_cameras) != 1 or len(heldout_cameras) != 1:
        raise ValueError("The controlled D-NeRF adapter requires one train and one heldout trajectory.")
    train_payload, train_frames = dnerf_camera_frames(record, train_cameras[0])
    heldout_payload, heldout_frames = dnerf_camera_frames(record, heldout_cameras[0])
    train_frames = [train_frames[index] for index in frame_positions]
    heldout_frames = [heldout_frames[index] for index in frame_positions]
    anchor_c2w = dnerf_c2w(train_frames[0], device=device)

    def trajectory(payload: dict[str, Any], frames: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
        K = make_fixed_pinhole_K(
            H=H,
            W=W,
            fov_degrees=math.degrees(float(payload["camera_angle_x"])),
            device=device,
        )
        w2c = torch.stack([torch.linalg.inv(dnerf_c2w(frame, device=device)) @ anchor_c2w for frame in frames])
        return K.unsqueeze(0), w2c.unsqueeze(0)

    train_K, train_w2c = trajectory(train_payload, train_frames)
    heldout_K, heldout_w2c = trajectory(heldout_payload, heldout_frames)
    return (
        train_K,
        train_w2c,
        heldout_K,
        heldout_w2c,
        anchor_c2w,
        "dnerf_matched_time_blender_to_opencv_relative_pinhole",
    )


def make_fixed_pinhole_K(*, H: int, W: int, fov_degrees: float, device: torch.device) -> torch.Tensor:
    fov = math.radians(float(fov_degrees))
    focal = 0.5 * float(W) / math.tan(0.5 * fov)
    return torch.tensor(
        [
            [focal, 0.0, float(W) * 0.5],
            [0.0, focal, float(H) * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )


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
    theta = torch.linalg.norm(r).clamp_min(1.0e-8)
    rx, ry, rz = r
    zero = r.new_zeros(())
    skew = torch.stack(
        [
            torch.stack([zero, -rz, ry]),
            torch.stack([rz, zero, -rx]),
            torch.stack([-ry, rx, zero]),
        ]
    )
    eye = torch.eye(3, dtype=torch.float32, device=device)
    return eye + (torch.sin(theta) / theta) * skew + ((1.0 - torch.cos(theta)) / (theta * theta)) * (skew @ skew)


def deepview_models_by_name(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    models_path = Path(record["models_path"])
    models = load_json(models_path)
    return {str(model["name"]): model for model in models}


def deepview_model_for_camera(record: dict[str, Any], camera_name: str) -> dict[str, Any]:
    by_name = deepview_models_by_name(record)
    if camera_name not in by_name:
        raise KeyError(f"DeepView camera {camera_name!r} not found in {record['models_path']}.")
    return by_name[camera_name]


def deepview_lens_from_model(model: dict[str, Any], *, device: torch.device) -> tuple[str, torch.Tensor | None]:
    projection_type = str(model.get("projection_type", "pinhole")).lower()
    if projection_type == "fisheye":
        coeffs = torch.zeros(4, dtype=torch.float32, device=device)
        raw = torch.as_tensor(model.get("radial_distortion", []), dtype=torch.float32, device=device).flatten()
        if raw.numel() > coeffs.numel():
            raise ValueError(
                f"DeepView fisheye camera {model.get('name')!r} has {raw.numel()} distortion coefficients; "
                "expected at most 4 for CameraSpec(opencv_fisheye)."
            )
        coeffs[: raw.numel()] = raw
        return "opencv_fisheye", coeffs
    if projection_type in {"pinhole", "perspective"}:
        return "pinhole", None
    raise ValueError(
        f"Unsupported DeepView projection_type={projection_type!r} for camera {model.get('name')!r}."
    )


def deepview_lens_metadata(
    record: dict[str, Any],
    camera_names: list[str],
    *,
    device: torch.device,
) -> tuple[list[str], torch.Tensor | None]:
    by_name = deepview_models_by_name(record)
    missing = [camera_name for camera_name in camera_names if camera_name not in by_name]
    if missing:
        raise KeyError(f"DeepView cameras {missing!r} not found in {record['models_path']}.")
    models = [by_name[camera_name] for camera_name in camera_names]
    lens_pairs = [deepview_lens_from_model(model, device=device) for model in models]
    lens_models = [lens_model for lens_model, _distortion in lens_pairs]
    distortions = [distortion for _lens_model, distortion in lens_pairs]
    if all(distortion is None for distortion in distortions):
        return lens_models, None
    padded = [
        torch.zeros(4, dtype=torch.float32, device=device) if distortion is None else distortion
        for distortion in distortions
    ]
    return lens_models, torch.stack(padded, dim=0)


def deepview_camera_from_models(
    record: dict[str, Any],
    camera_name: str,
    *,
    H: int,
    W: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model = deepview_model_for_camera(record, camera_name)
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

    w2c_gl_rot = rodrigues_matrix(model["orientation"], device=device)
    c2w_gl_rot = w2c_gl_rot.T
    gl_to_plus_z = torch.diag(torch.tensor([1.0, -1.0, -1.0], dtype=torch.float32, device=device))
    c2w = torch.eye(4, dtype=torch.float32, device=device)
    c2w[:3, :3] = c2w_gl_rot @ gl_to_plus_z
    c2w[:3, 3] = torch.tensor(model["position"], dtype=torch.float32, device=device)
    return K, c2w


def deepview_video_path_for_camera(record: dict[str, Any], camera_name: str) -> Path:
    if camera_name == str(record.get("source_camera")):
        return Path(record["source_video_path"])
    if camera_name == str(record.get("target_camera")):
        return Path(record["target_video_path"])
    scene_dir = record.get("dataset_scene_dir")
    if not scene_dir:
        raise ValueError(f"Record {record.get('sample_id')!r} does not include dataset_scene_dir.")
    path = Path(scene_dir) / f"{camera_name}.mp4"
    if not path.exists():
        raise FileNotFoundError(f"DeepView camera video not found: {path}")
    return path


def camxtime_scene_dir(record: dict[str, Any]) -> Path:
    scene_dir = record.get("camxtime_scene_dir") or record.get("dataset_scene_dir")
    if not scene_dir:
        raise ValueError(
            f"CamXTime record {record.get('sample_id')!r} is missing camxtime_scene_dir/dataset_scene_dir."
        )
    return Path(scene_dir)


def camxtime_camera_data_path(record: dict[str, Any]) -> Path:
    path = Path(record.get("camxtime_camera_data_path") or camxtime_scene_dir(record) / "camera_data.json")
    if not path.exists():
        raise FileNotFoundError(f"CamXTime camera_data.json not found: {path}")
    return path


def load_camxtime_camera_data(record: dict[str, Any]) -> dict[str, Any]:
    return load_json(camxtime_camera_data_path(record))


def camxtime_intrinsics(camera_data: dict[str, Any]) -> dict[str, Any]:
    if isinstance(camera_data.get("intrinsics"), dict):
        return camera_data["intrinsics"]
    cameras = camera_data.get("cameras")
    if isinstance(cameras, dict) and isinstance(cameras.get("intrinsics"), dict):
        return cameras["intrinsics"]
    raise ValueError("CamXTime camera_data.json is missing intrinsics.")


def camxtime_camera_count(camera_data: dict[str, Any]) -> int:
    if camera_data.get("n_cameras") is not None:
        return int(camera_data["n_cameras"])
    cameras = camera_data.get("cameras")
    if isinstance(cameras, dict) and isinstance(cameras.get("extrinsic"), dict):
        return len(cameras["extrinsic"])
    if isinstance(cameras, dict):
        return len([key for key, value in cameras.items() if isinstance(value, dict) and "intrinsics" not in key])
    raise ValueError("CamXTime camera_data.json is missing cameras.")


def camxtime_camera_index(camera_name: str) -> int:
    stem = Path(str(camera_name)).stem
    if stem.startswith("camera_"):
        return int(stem.split("_", 1)[1])
    return int(stem)


def camxtime_camera_indices(camera_name: str, *, frame_count: int, camera_count: int) -> list[int]:
    if camera_name in CAMXTIME_TRAJECTORY_VIDEOS:
        if frame_count > camera_count:
            raise ValueError(
                f"CamXTime trajectory {camera_name!r} needs {frame_count} camera poses, "
                f"but camera_data.json contains {camera_count}."
            )
        return list(range(frame_count))
    index = camxtime_camera_index(camera_name)
    if not 0 <= index < camera_count:
        raise IndexError(f"CamXTime camera index {index} out of range for {camera_count} cameras.")
    return [index] * int(frame_count)


def camxtime_c2w_for_index(
    camera_data: dict[str, Any],
    camera_index: int,
    *,
    device: torch.device,
    extrinsic_convention: str = "c2w",
    camera_convention: str = "opengl",
) -> torch.Tensor:
    def normalize_camera_axes(c2w: torch.Tensor) -> torch.Tensor:
        convention = str(camera_convention).lower()
        if convention in {"opencv", "opencv_plus_z", "plus_z"}:
            return c2w
        if convention in {"opengl", "blender", "nerf"}:
            # Blender/OpenGL camera poses store +Y as image up and -Z as the
            # view direction. CameraSpec/renderers use OpenCV-style +Y down and
            # +Z forward, so flip the camera-frame Y/Z columns.
            flip = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=torch.float32, device=device))
            return c2w @ flip
        raise ValueError("camxtime_camera_convention must be 'opengl'/'blender' or 'opencv'.")

    cameras = camera_data.get("cameras")
    if not isinstance(cameras, dict):
        raise ValueError("CamXTime camera_data.json is missing cameras.")

    full_grid_key = str(camera_index)
    if full_grid_key in cameras and isinstance(cameras[full_grid_key], dict):
        camera_record = cameras[full_grid_key]
        if "c2w" in camera_record:
            return normalize_camera_axes(torch.tensor(camera_record["c2w"], dtype=torch.float32, device=device))
        if "w2c" in camera_record:
            w2c = torch.tensor(camera_record["w2c"], dtype=torch.float32, device=device)
            return normalize_camera_axes(torch.linalg.inv(w2c))

    extrinsics = cameras.get("extrinsic")
    if isinstance(extrinsics, dict):
        key = f"camera_{camera_index:03d}"
        if key not in extrinsics:
            raise KeyError(f"CamXTime camera {key!r} not present in camera_data.json.")
        matrix = torch.tensor(extrinsics[key], dtype=torch.float32, device=device)
        convention = str(extrinsic_convention).lower()
        if convention == "c2w":
            return normalize_camera_axes(matrix)
        if convention == "w2c":
            return normalize_camera_axes(torch.linalg.inv(matrix))
        raise ValueError("camxtime_extrinsic_convention must be 'c2w' or 'w2c'.")

    raise KeyError(f"CamXTime camera index {camera_index} not present in camera_data.json.")


def camxtime_K_from_camera_data(
    record: dict[str, Any],
    camera_data: dict[str, Any],
    *,
    H: int,
    W: int,
    device: torch.device,
) -> torch.Tensor:
    intr = camxtime_intrinsics(camera_data)
    if "K" in intr:
        matrix = intr["K"]
        fx = float(matrix[0][0])
        fy = float(matrix[1][1])
        cx = float(matrix[0][2])
        cy = float(matrix[1][2])
    else:
        fx = float(intr["fx"])
        fy = float(intr["fy"])
        cx = float(intr["cx"])
        cy = float(intr["cy"])
    source_width = float(record.get("camxtime_source_width") or intr.get("width") or (2.0 * cx))
    source_height = float(record.get("camxtime_source_height") or intr.get("height") or (2.0 * cy))
    return make_scaled_intrinsics(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        source_width=source_width,
        source_height=source_height,
        target_width=W,
        target_height=H,
        device=device,
    )


def camxtime_video_path_for_camera(record: dict[str, Any], camera_name: str) -> Path:
    if camera_name == str(record.get("source_camera")):
        return Path(record["source_video_path"])
    if camera_name == str(record.get("target_camera")):
        return Path(record["target_video_path"])
    scene_dir = camxtime_scene_dir(record)
    candidates = [scene_dir / f"{camera_name}.mp4"]
    if camera_name not in CAMXTIME_TRAJECTORY_VIDEOS:
        try:
            candidates.append(scene_dir / f"camera_{camxtime_camera_index(camera_name):03d}.mp4")
        except ValueError:
            pass
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"CamXTime camera video not found for {camera_name!r}; tried {candidates}.")


def make_camxtime_multiview_cameras(
    record: dict[str, Any],
    *,
    train_cameras: list[str],
    heldout_cameras: list[str],
    anchor_camera: str,
    T: int,
    H: int,
    W: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
    dataset = str(record.get("dataset") or "")
    if dataset not in CAMXTIME_DATASETS:
        raise ValueError(f"camera.rig_init=camxtime requires a CamXTime record; got dataset={dataset!r}.")

    camera_data = load_camxtime_camera_data(record)
    camera_count = camxtime_camera_count(camera_data)
    extrinsic_convention = str(record.get("camxtime_extrinsic_convention", "c2w"))
    camera_convention = str(record.get("camxtime_camera_convention", "opengl"))
    anchor_indices = camxtime_camera_indices(anchor_camera, frame_count=T, camera_count=camera_count)
    anchor_c2w = camxtime_c2w_for_index(
        camera_data,
        anchor_indices[0],
        device=device,
        extrinsic_convention=extrinsic_convention,
        camera_convention=camera_convention,
    )
    K = camxtime_K_from_camera_data(record, camera_data, H=H, W=W, device=device)

    def build_view(camera_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        c2w_sequence = torch.stack(
            [
                camxtime_c2w_for_index(
                    camera_data,
                    index,
                    device=device,
                    extrinsic_convention=extrinsic_convention,
                    camera_convention=camera_convention,
                )
                for index in camxtime_camera_indices(camera_name, frame_count=T, camera_count=camera_count)
            ],
            dim=0,
        )
        rel_w2c = torch.linalg.inv(c2w_sequence) @ anchor_c2w
        return K, rel_w2c

    train_pairs = [build_view(camera_name) for camera_name in train_cameras]
    heldout_pairs = [build_view(camera_name) for camera_name in heldout_cameras]
    return (
        torch.stack([pair[0] for pair in train_pairs], dim=0),
        torch.stack([pair[1] for pair in train_pairs], dim=0),
        torch.stack([pair[0] for pair in heldout_pairs], dim=0),
        torch.stack([pair[1] for pair in heldout_pairs], dim=0),
        f"{dataset}_{camera_convention.lower()}_to_opencv_relative_pinhole",
    )


# ---------------------------------------------------------------------------
# AIST++ camera adapter.
#
# Source bundle layout (downloaded by src/dataset_pipeline/multicam_val.py):
#     <aist_cameras_dir>/mapping.txt              "<seq_name>  <env_name>"
#     <aist_cameras_dir>/<env_name>.json          list of 9 camera dicts.
# Each camera dict has:
#     name        e.g. "c05"
#     size        [W, H], native pixels = [1920, 1080]
#     matrix      3x3 K (OpenCV intrinsic matrix)
#     rotation    OpenCV Rodrigues rvec (3,) such that R = Rodrigues(rvec)
#     translation OpenCV tvec (3,)
#     distortions OpenCV [k1, k2, p1, p2, k3]
#
# OpenCV / AIST++ axis convention:
#     x_camera = R @ x_world + translation              # world-to-camera
# so:
#     w2c = [[R, t], [0, 1]]
#     c2w = inv(w2c) = [[R^T, -R^T @ t], [0, 1]]
# We then express each camera relative to the anchor camera, exactly as we do for
# DeepView, so that the decoded splat world is anchored at the input/condition view.
#
# Translation units: AIST++ tvec is in millimeters (typical OpenCV calibration on a
# multi-camera dance rig where cameras sit ~3-5m from the dancer). For the splat
# renderer, only relative geometry matters because we always feed `rel_w2c`, but the
# overall scene scale still has to roughly match what the model decodes.  We expose a
# rig-side scale knob (`rig.aist_translation_scale`, default 1.0 = AIST native mm)
# rather than silently rescaling.  If you want AIST poses at meter-scale to match
# DeepView, set 0.001.
#
# Lens model: AIST++ cameras have a small radial-only distortion (k1 ~ -0.11)
# plus zero tangential terms. The current AIST path keeps the pinhole
# approximation and preserves raw distortion metadata for a later OpenCV path.
# Edge pixels in c01/c05/c09 carry 1-2px of residual distortion at 1920x1080,
# which scales to <0.2px at 128x128.
#
# Reference: https://google.github.io/aistplusplus_dataset/download.html and the
# AIST++ loader at https://github.com/google/aistplusplus_api/blob/main/aist_plusplus/loader.py
# ---------------------------------------------------------------------------
def aist_camera_from_setting(
    record: dict[str, Any],
    camera_name: str,
    *,
    H: int,
    W: int,
    device: torch.device,
    translation_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    setting_path = record.get("aist_setting_path")
    if not setting_path:
        raise ValueError(
            f"AIST record {record.get('sample_id')!r} is missing aist_setting_path. "
            f"Rebuild the manifest with src/dataset_pipeline/multicam_val.py "
            f"after running the download-aist-cameras stage."
        )
    params = load_json(setting_path)
    by_name = {str(item["name"]): item for item in params}
    if camera_name not in by_name:
        raise KeyError(
            f"AIST camera {camera_name!r} not in {setting_path}. Available: {sorted(by_name)}"
        )
    cam = by_name[camera_name]

    matrix = cam["matrix"]
    fx = float(matrix[0][0])
    fy = float(matrix[1][1])
    cx = float(matrix[0][2])
    cy = float(matrix[1][2])
    source_w, source_h = float(cam["size"][0]), float(cam["size"][1])
    K = make_scaled_intrinsics(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        source_width=source_w,
        source_height=source_h,
        target_width=W,
        target_height=H,
        device=device,
    )

    # AIST++ rotation/translation are OpenCV-convention world-to-camera,
    # so rodrigues_matrix(rvec) IS the world-to-camera rotation directly.
    rotation = rodrigues_matrix(cam["rotation"], device=device)
    translation = torch.tensor(cam["translation"], dtype=torch.float32, device=device) * float(
        translation_scale
    )
    w2c = torch.eye(4, dtype=torch.float32, device=device)
    w2c[:3, :3] = rotation
    w2c[:3, 3] = translation
    c2w = torch.linalg.inv(w2c)
    return K, c2w


def aist_video_path_for_camera(record: dict[str, Any], camera_name: str) -> Path:
    if camera_name == str(record.get("source_camera")):
        return Path(record["source_video_path"])
    if camera_name == str(record.get("target_camera")):
        return Path(record["target_video_path"])
    raw_dir = record.get("aist_raw_dir")
    seq_name = record.get("aist_seq_name")
    if not raw_dir or not seq_name:
        raise ValueError(
            f"Cannot locate AIST video for {camera_name!r} on record "
            f"{record.get('sample_id')!r}: missing aist_raw_dir / aist_seq_name."
        )
    if "cAll" not in seq_name:
        raise ValueError(f"AIST seq_name {seq_name!r} does not contain 'cAll'; cannot substitute view tag.")
    video_name = seq_name.replace("cAll", camera_name) + ".mp4"
    path = Path(raw_dir) / video_name
    if not path.exists():
        raise FileNotFoundError(f"AIST camera video not found: {path}")
    return path


def make_aist_multiview_cameras(
    record: dict[str, Any],
    *,
    train_cameras: list[str],
    heldout_cameras: list[str],
    anchor_camera: str,
    T: int,
    H: int,
    W: int,
    device: torch.device,
    translation_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
    if record.get("dataset") != "aist_dance_db" or not record.get("aist_setting_path"):
        raise ValueError(
            "Configured multicam_train_cameras for an AIST record require aist_setting_path; "
            "rebuild the manifest after running download-aist-cameras."
        )
    _, anchor_c2w = aist_camera_from_setting(
        record, anchor_camera, H=H, W=W, device=device, translation_scale=translation_scale
    )

    train_K = []
    train_w2c = []
    for camera_name in train_cameras:
        K, c2w = aist_camera_from_setting(
            record, camera_name, H=H, W=W, device=device, translation_scale=translation_scale
        )
        # rel_w2c = inv(c2w_other) @ c2w_anchor
        # ==> a world point fixed in the anchor camera's local frame projects through this view.
        rel_w2c = torch.linalg.inv(c2w) @ anchor_c2w
        train_K.append(K)
        train_w2c.append(rel_w2c.unsqueeze(0).repeat(T, 1, 1))

    heldout_K = []
    heldout_w2c = []
    for camera_name in heldout_cameras:
        K, c2w = aist_camera_from_setting(
            record, camera_name, H=H, W=W, device=device, translation_scale=translation_scale
        )
        rel_w2c = torch.linalg.inv(c2w) @ anchor_c2w
        heldout_K.append(K)
        heldout_w2c.append(rel_w2c.unsqueeze(0).repeat(T, 1, 1))

    return (
        torch.stack(train_K, dim=0),
        torch.stack(train_w2c, dim=0),
        torch.stack(heldout_K, dim=0),
        torch.stack(heldout_w2c, dim=0),
        "aist_plusplus_relative_pinhole",
    )


# ---------------------------------------------------------------------------
# Neural 3D Video (Facebook AI dynamic-NeRF release) camera adapter.
#
# Source bundle layout (downloaded by src/dataset_pipeline/neural_3d_video.py):
#     <scene_dir>/cam00.mp4 ... cam20.mp4   (18 cameras for coffee_martini;
#                                            indices are NOT contiguous --
#                                            cam03, cam15, cam17 are missing)
#     <scene_dir>/poses_bounds.npy          LLFF-style pose bundle, shape (N, 17)
#
# Where N == number of mp4s. The release follows the upstream LLFF convention,
# so row k in poses_bounds.npy corresponds to the k-th camera mp4 when the
# mp4 filenames are sorted lexicographically (`sorted(scene_dir.glob("cam*.mp4"))`).
# That means cam04 lands on row 3, cam20 on row 17, etc. This matches the
# loader convention used by Neural_3D_Video, K-Planes, MixVoxels, and 4DGS.
#
# poses_bounds.npy schema (per row of 17 float64):
#     [ 0:15 ] flattened 3x5 "pose-hwf" matrix in LLFF storage order:
#                  down  right  back  t   hwf
#                  ----- 3x4 ------  -- 3x1 --
#         columns 0..2  : camera basis vectors in world space
#         column   3    : world-space camera position (translation, c2w[:3, 3])
#         column   4    : [H, W, focal] in *native* pixel units
#     [15:17 ] [near, far] depth bounds for the scene (LLFF NDC bounds, unused here)
#
# Raw poses_bounds rows use LLFF's documented [down, right, backwards] basis,
# not the post-load NeRF [right, up, backwards] basis. Convert the stored
# columns directly to OpenCV [right, down, forwards]:
#     R_opencv = R_stored[:, [1, 0, 2]] @ diag([+1, +1, -1])
# A sign-only NeRF-to-OpenCV flip is wrong at this ingestion boundary because
# it omits the raw LLFF down/right column swap. Real Coffee Martini epipolar
# matches are the regression signal for this distinction.
#
# Translation (c2w[:3, 3]) is already in world coordinates and does NOT need
# the axis flip -- only the rotation columns do. World scene scale is in the
# scene's native LLFF units (the original Neural 3D Video calibration is in
# meters; bounds for coffee_martini are near=8.83, far=109.78 in those units,
# which look like decimeters once you compare the scene to a 1m table; we keep
# the native scale and expose `n3d_translation_scale` if a future caller wants
# to rescale to meters).
#
# Intrinsics: hwf = [H, W, focal] in *native* pixel units (2028 x 2704 for
# coffee_martini, focal ~1460.75). N3D videos are pre-undistorted to a single
# pinhole, so K is a clean isotropic pinhole with cx=W/2, cy=H/2. We scale
# the focal to match the (target_W, target_H) we are resizing to, exactly as
# we do for AIST and DeepView -- non-uniform resize stretches fx and fy by
# different factors when target_size doesn't match the source aspect.
#
# Anchor-relative pose: same pattern as DeepView/AIST. Express each camera
# relative to the anchor camera so the splat world is anchored at the input
# view; anchor_w2c[0] becomes identity in the relative frame.
#
# Reference: https://github.com/facebookresearch/Neural_3D_Video and the
# upstream LLFF loader at
# https://github.com/Fyusion/LLFF/blob/master/llff/poses/pose_utils.py
# (the pose-hwf packing format originated there).
# ---------------------------------------------------------------------------
def neural_3d_camera_from_poses_bounds(
    record: dict[str, Any],
    camera_name: str,
    *,
    H: int,
    W: int,
    device: torch.device,
    translation_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    scene_dir = record.get("dataset_scene_dir")
    if not scene_dir:
        raise ValueError(
            f"Neural 3D Video record {record.get('sample_id')!r} is missing dataset_scene_dir. "
            f"Rebuild the manifest with src/dataset_pipeline/multicam_val.py."
        )
    scene_path = Path(scene_dir)
    poses_path = scene_path / "poses_bounds.npy"
    if not poses_path.exists():
        raise FileNotFoundError(f"poses_bounds.npy not found at {poses_path}.")

    # numpy is only needed at this ingestion edge; load lazily so the rest of
    # the trainer doesn't pull it in if Neural 3D Video isn't used.
    import numpy as np  # noqa: WPS433 -- intentional local import.

    poses_bounds = np.load(poses_path).astype(np.float32)
    if poses_bounds.ndim != 2 or poses_bounds.shape[1] != 17:
        raise ValueError(
            f"poses_bounds.npy at {poses_path} has shape {poses_bounds.shape}; "
            f"expected (N, 17)."
        )

    # Camera <-> row mapping: sorted mp4 filename order == row order.
    mp4_paths = sorted(scene_path.glob("cam*.mp4"))
    camera_names = [path.stem for path in mp4_paths]
    if len(camera_names) != poses_bounds.shape[0]:
        raise RuntimeError(
            f"Neural 3D Video {scene_path.name}: found {len(camera_names)} mp4s "
            f"but {poses_bounds.shape[0]} poses_bounds rows. The two must agree."
        )
    if camera_name not in camera_names:
        raise KeyError(
            f"Neural 3D Video camera {camera_name!r} not present in {scene_path}. "
            f"Available: {camera_names}"
        )
    row_index = camera_names.index(camera_name)

    pose_hwf = poses_bounds[row_index, :15].reshape(3, 5)
    rotation_llff_stored = pose_hwf[:, :3]                # 3x3, [down, right, back].
    translation = pose_hwf[:, 3]                          # 3,   world-space camera position.
    height_native = float(pose_hwf[0, 4])
    width_native = float(pose_hwf[1, 4])
    focal_native = float(pose_hwf[2, 4])

    # Raw LLFF storage -> OpenCV: [down, right, back] becomes
    # [right, down, forward]. Translation is already world-space and unchanged.
    rotation_opencv = rotation_llff_stored[:, [1, 0, 2]].copy()
    rotation_opencv[:, 2] *= -1.0

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = rotation_opencv
    c2w[:3, 3] = translation * float(translation_scale)
    c2w_t = torch.from_numpy(c2w).to(device=device, dtype=torch.float32)

    # K is isotropic pinhole in native pixels; resize to (W, H).
    K = make_scaled_intrinsics(
        fx=focal_native,
        fy=focal_native,
        cx=0.5 * width_native,
        cy=0.5 * height_native,
        source_width=width_native,
        source_height=height_native,
        target_width=W,
        target_height=H,
        device=device,
    )
    return K, c2w_t


def neural_3d_video_path_for_camera(record: dict[str, Any], camera_name: str) -> Path:
    if camera_name == str(record.get("source_camera")):
        return Path(record["source_video_path"])
    if camera_name == str(record.get("target_camera")):
        return Path(record["target_video_path"])
    scene_dir = record.get("dataset_scene_dir")
    if not scene_dir:
        raise ValueError(
            f"Cannot locate Neural 3D Video video for {camera_name!r} on record "
            f"{record.get('sample_id')!r}: missing dataset_scene_dir."
        )
    path = Path(scene_dir) / f"{camera_name}.mp4"
    if not path.exists():
        raise FileNotFoundError(f"Neural 3D Video camera video not found: {path}")
    return path


def make_neural_3d_multiview_cameras(
    record: dict[str, Any],
    *,
    train_cameras: list[str],
    heldout_cameras: list[str],
    anchor_camera: str,
    T: int,
    H: int,
    W: int,
    device: torch.device,
    translation_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
    if record.get("dataset") != "neural_3d_video" or not record.get("dataset_scene_dir"):
        raise ValueError(
            "Configured multicam_train_cameras for a Neural 3D Video record require "
            "dataset_scene_dir; rebuild the manifest after extracting the scene zip."
        )
    _, anchor_c2w = neural_3d_camera_from_poses_bounds(
        record, anchor_camera, H=H, W=W, device=device, translation_scale=translation_scale
    )

    train_K = []
    train_w2c = []
    for camera_name in train_cameras:
        K, c2w = neural_3d_camera_from_poses_bounds(
            record, camera_name, H=H, W=W, device=device, translation_scale=translation_scale
        )
        rel_w2c = torch.linalg.inv(c2w) @ anchor_c2w
        train_K.append(K)
        train_w2c.append(rel_w2c.unsqueeze(0).repeat(T, 1, 1))

    heldout_K = []
    heldout_w2c = []
    for camera_name in heldout_cameras:
        K, c2w = neural_3d_camera_from_poses_bounds(
            record, camera_name, H=H, W=W, device=device, translation_scale=translation_scale
        )
        rel_w2c = torch.linalg.inv(c2w) @ anchor_c2w
        heldout_K.append(K)
        heldout_w2c.append(rel_w2c.unsqueeze(0).repeat(T, 1, 1))

    return (
        torch.stack(train_K, dim=0),
        torch.stack(train_w2c, dim=0),
        torch.stack(heldout_K, dim=0),
        torch.stack(heldout_w2c, dim=0),
        "neural_3d_llff_opencv_relative_pinhole_v2",
    )


# ---------------------------------------------------------------------------
# ViVo (Bristol VICR multicam human body capture) camera adapter.
#
# Source bundle layout (downloaded manually via the project MS Form; see
# data/REHYDRATE.md and src/dataset_pipeline/vivo.py):
#     <scene_dir>/calibration.json
#     <scene_dir>/{train,test}/<serial>/<...>_colour-image_<seq>_<ts>.jpg(.meta.json)
#     <scene_dir>/rotation_correction.json     # OPTIONAL, present in some scenes only
#
# `calibration.json` schema (per camera serial, e.g. "000236320812"):
#     {
#       "depth_extrinsics":          {"orientation": [9 floats row-major R],
#                                     "translation": [3 floats t]},   # rig WORLD -> depth camera
#       "depth_intrinsics":          {"fx", "fy", "ppx", "ppy", "width", "height", ...},
#       "colour_intrinsics":         {"fx", "fy", "ppx", "ppy", "width", "height",
#                                     "coefficients", "distortion_mode"},
#       "colour_to_depth_extrinsics":{"orientation": [...], "translation": [...]},
#                                                                    # depth -> colour camera
#     }
#
# Per-frame .meta.json schema (NOT used for rig pose, kept for audit only):
#     imageMetadata.intrinsics: {fx, fy, ppx, ppy, distortionMode="OPENCV_8VAL",
#                                distortion: [k1,k2,p1,p2,k3,k4,k5,k6], width, height}
#     imageMetadata.extrinsics: {orientation: [9 floats], translation: [3 floats]}
#     # The per-frame extrinsics are the IMU-anchored sensor pose in the device
#     # body frame (effectively near-identity, with small accelerometer-bias
#     # rotations). They are NOT the rig pose. Use calibration.json for multicam
#     # geometry. Audit: per-frame extrinsics on cam 000236320812 vary by
#     # <0.05 rad across all 501 frames -- a stationary camera with IMU jitter,
#     # not a moving rig.
#
# ViVo extrinsic convention (from the upstream ViVo-DataProcessing repo,
# https://github.com/azzarelli/ViVo-DataProcessing):
#     For each camera C:
#         x_depth(C)  = R_d(C)  @ x_world + t_d(C)            # depth_extrinsics
#         x_colour(C) = R_cd(C) @ x_depth(C) + t_cd(C)        # colour_to_depth_extrinsics
#     Compose to get world -> colour:
#         R_w2c(C) = R_cd(C) @ R_d(C)
#         t_w2c(C) = R_cd(C) @ t_d(C) + t_cd(C)
#     Then c2w = inv([[R_w2c, t_w2c], [0, 1]]).
#
# Orientation packing: the calibration "orientation" field is a 9-element list
# in ROW-MAJOR order. Reshape (3,3) and use directly as R; do NOT transpose.
#
# Translation units: ViVo is in METERS (Femto Bolt sensor calibration). Cameras
# in athlete_rows sit ~1.5-2.5 m from origin and the subject's torso is ~1-2 m
# from each lens. No scaling needed to match Neural 3D / DeepView's meter-ish
# world scale; we still expose `vivo_translation_scale=1.0` for parity with the
# other adapters.
#
# Distortion: colour intrinsics in calibration.json carry no coefficients on
# the Femto Bolt (`distortion_mode=0`, `coefficients=[]`). The per-frame
# meta.json blocks DO carry an OPENCV_8VAL set, but those are RAW sensor
# intrinsics; calibration.json is the rectified post-processed view used by
# the upstream pipeline. We use calibration.json (rectified, distortion-free)
# and treat the colour camera as a clean pinhole.
#
# rotation_correction.json: when present, contains an additional 4x4
# world-frame rigid transform that aligns this scene's rig coordinate frame
# to a canonical capture-room frame across recordings. We DEFER support
# until we have a ViVo record that actually carries one (athlete_rows does
# not). The adapter raises a clear error if `vivo_rotation_correction_path`
# is set on the record, asking the integrator to extend the math at this
# site rather than silently producing wrong poses.
#
# Reference: https://github.com/azzarelli/ViVo-DataProcessing
# ---------------------------------------------------------------------------
def vivo_camera_from_calibration(
    record: dict[str, Any],
    camera_name: str,
    *,
    H: int,
    W: int,
    device: torch.device,
    translation_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    calibration_path = record.get("vivo_calibration_path")
    if not calibration_path:
        raise ValueError(
            f"ViVo record {record.get('sample_id')!r} is missing vivo_calibration_path. "
            f"Build the manifest with src/dataset_pipeline/vivo.py and ensure "
            f"calibration.json is at <scene_dir>/calibration.json."
        )
    if record.get("vivo_rotation_correction_path"):
        # The math here does NOT yet apply rotation_correction.json. Extend it
        # before claiming support, rather than silently producing rig-frame-
        # misaligned cameras across recordings.
        raise NotImplementedError(
            f"ViVo record {record.get('sample_id')!r} carries a "
            f"rotation_correction.json, which the adapter does not yet apply. "
            f"Add the canonical-frame transform composition in "
            f"vivo_camera_from_calibration before using this scene."
        )
    payload = load_json(calibration_path)
    cameras = payload.get("cameras") or {}
    if camera_name not in cameras:
        raise KeyError(
            f"ViVo camera {camera_name!r} not in {calibration_path}. "
            f"Available: {sorted(cameras)}"
        )
    cam = cameras[camera_name]

    colour_intr = cam["colour_intrinsics"]
    fx = float(colour_intr["fx"])
    fy = float(colour_intr["fy"])
    cx = float(colour_intr["ppx"])
    cy = float(colour_intr["ppy"])
    source_w = float(colour_intr["width"])
    source_h = float(colour_intr["height"])
    K = make_scaled_intrinsics(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        source_width=source_w,
        source_height=source_h,
        target_width=W,
        target_height=H,
        device=device,
    )

    # Compose rig -> colour:  R_w2c = R_cd @ R_d,  t_w2c = R_cd @ t_d + t_cd
    depth_ext = cam["depth_extrinsics"]
    cd_ext = cam["colour_to_depth_extrinsics"]
    R_d = torch.tensor(depth_ext["orientation"], dtype=torch.float32, device=device).reshape(3, 3)
    t_d = torch.tensor(depth_ext["translation"], dtype=torch.float32, device=device)
    R_cd = torch.tensor(cd_ext["orientation"], dtype=torch.float32, device=device).reshape(3, 3)
    t_cd = torch.tensor(cd_ext["translation"], dtype=torch.float32, device=device)
    R_w2c = R_cd @ R_d
    t_w2c = (R_cd @ t_d + t_cd) * float(translation_scale)
    w2c = torch.eye(4, dtype=torch.float32, device=device)
    w2c[:3, :3] = R_w2c
    w2c[:3, 3] = t_w2c
    c2w = torch.linalg.inv(w2c)
    return K, c2w


def vivo_video_path_for_camera(record: dict[str, Any], camera_name: str) -> Path:
    if camera_name == str(record.get("source_camera")):
        return Path(record["source_video_path"])
    if camera_name == str(record.get("target_camera")):
        return Path(record["target_video_path"])
    rgb_mp4_root = record.get("vivo_rgb_mp4_root")
    scene = record.get("vivo_scene")
    if not rgb_mp4_root or not scene:
        raise ValueError(
            f"Cannot locate ViVo video for {camera_name!r} on record "
            f"{record.get('sample_id')!r}: missing vivo_rgb_mp4_root / vivo_scene."
        )
    # ViVo lays out compacted mp4s as <root>/<scene>/{train,test}/<serial>.mp4.
    candidates = [
        Path(rgb_mp4_root) / scene / "train" / f"{camera_name}.mp4",
        Path(rgb_mp4_root) / scene / "test" / f"{camera_name}.mp4",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"ViVo camera video not found for {camera_name!r} under "
        f"{Path(rgb_mp4_root) / scene}/(train|test)/. Run "
        f"src/dataset_pipeline/vivo.py compact-rgb first."
    )


def make_vivo_multiview_cameras(
    record: dict[str, Any],
    *,
    train_cameras: list[str],
    heldout_cameras: list[str],
    anchor_camera: str,
    T: int,
    H: int,
    W: int,
    device: torch.device,
    translation_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
    if record.get("dataset") != "vivo" or not record.get("vivo_calibration_path"):
        raise ValueError(
            "Configured multicam_train_cameras for a ViVo record require "
            "vivo_calibration_path; rebuild the manifest with src/dataset_pipeline/vivo.py."
        )
    _, anchor_c2w = vivo_camera_from_calibration(
        record, anchor_camera, H=H, W=W, device=device, translation_scale=translation_scale
    )

    train_K = []
    train_w2c = []
    for camera_name in train_cameras:
        K, c2w = vivo_camera_from_calibration(
            record, camera_name, H=H, W=W, device=device, translation_scale=translation_scale
        )
        rel_w2c = torch.linalg.inv(c2w) @ anchor_c2w
        train_K.append(K)
        train_w2c.append(rel_w2c.unsqueeze(0).repeat(T, 1, 1))

    heldout_K = []
    heldout_w2c = []
    for camera_name in heldout_cameras:
        K, c2w = vivo_camera_from_calibration(
            record, camera_name, H=H, W=W, device=device, translation_scale=translation_scale
        )
        rel_w2c = torch.linalg.inv(c2w) @ anchor_c2w
        heldout_K.append(K)
        heldout_w2c.append(rel_w2c.unsqueeze(0).repeat(T, 1, 1))

    return (
        torch.stack(train_K, dim=0),
        torch.stack(train_w2c, dim=0),
        torch.stack(heldout_K, dim=0),
        torch.stack(heldout_w2c, dim=0),
        "vivo_calibration_relative_pinhole",
    )


def select_multicam_record(data_cfg: dict[str, Any]) -> dict[str, Any]:
    records = load_multicam_val_manifest(Path(data_cfg["multicam_manifest"]), split=data_cfg["multicam_split"])
    sample_id = data_cfg.get("multicam_sample_id")
    if sample_id:
        for record in records:
            if str(record.get("sample_id")) == str(sample_id):
                return record
        raise ValueError(f"No multicam_val record with sample_id={sample_id!r}.")
    sample_index = int(data_cfg.get("multicam_sample_index", 0))
    if sample_index < 0 or sample_index >= len(records):
        raise IndexError(f"multicam_sample_index={sample_index} out of range for {len(records)} records.")
    return records[sample_index]


def _frame_indices_tensor(frame_indices: Any, *, device: torch.device) -> torch.Tensor | None:
    if frame_indices is None:
        return None
    values = torch.as_tensor([int(index) for index in frame_indices], dtype=torch.long, device=device)
    if values.numel() < 1:
        raise ValueError("data.frame_indices must contain at least one index when provided.")
    return values


def select_configured_frames(video: torch.Tensor, frame_indices: Any) -> torch.Tensor:
    indices = _frame_indices_tensor(frame_indices, device=video.device)
    if indices is None:
        return video
    return video.index_select(0, indices).contiguous()


def select_configured_multiview_frames(videos: torch.Tensor, frame_indices: Any) -> torch.Tensor:
    indices = _frame_indices_tensor(frame_indices, device=videos.device)
    if indices is None:
        return videos
    return videos.index_select(1, indices).contiguous()


def requested_camera_frame_count(data_cfg: dict[str, Any], record: dict[str, Any]) -> int:
    record_frame_count = int(record["frame_count"])
    max_frames = int(data_cfg.get("max_frames") or 0)
    if max_frames > 0:
        return min(max_frames, record_frame_count)
    frame_indices = data_cfg.get("frame_indices")
    if frame_indices is None:
        return record_frame_count
    indices = [int(index) for index in frame_indices]
    if not indices:
        raise ValueError("data.frame_indices must contain at least one index when provided.")
    return min(max(indices) + 1, record_frame_count)


def camera_start_seconds(record: dict[str, Any], camera_name: str) -> float:
    if camera_name == str(record.get("source_camera")):
        return float(record.get("source_start_seconds", 0.0))
    if camera_name == str(record.get("target_camera")):
        return float(record.get("target_start_seconds", 0.0))

    dataset = str(record.get("dataset") or "")
    if dataset in {"deepview_video", "aist_dance_db", "neural_3d_video"} | CAMXTIME_DATASETS:
        return float(record.get("source_start_seconds", record.get("target_start_seconds", 0.0)))
    if dataset == "vivo":
        raise ValueError(
            f"ViVo camera {camera_name!r} is not the source or target camera on record "
            f"{record.get('sample_id')!r}; the manifest does not carry its capture-timestamp offset."
        )
    return float(record.get("source_start_seconds", record.get("target_start_seconds", 0.0)))


def video_path_for_camera(record: dict[str, Any], camera_name: str) -> Path:
    dataset = str(record.get("dataset") or "")
    if dataset in DNERF_DATASETS:
        return dnerf_scene_dir(record) / f"transforms_{dnerf_camera_split(record, camera_name)}.json"
    elif dataset == "deepview_video":
        return deepview_video_path_for_camera(record, camera_name)
    elif dataset == "aist_dance_db":
        return aist_video_path_for_camera(record, camera_name)
    elif dataset == "neural_3d_video":
        return neural_3d_video_path_for_camera(record, camera_name)
    elif dataset == "vivo":
        return vivo_video_path_for_camera(record, camera_name)
    elif dataset in CAMXTIME_DATASETS:
        return camxtime_video_path_for_camera(record, camera_name)
    elif camera_name == str(record.get("source_camera")):
        return Path(record["source_video_path"])
    elif camera_name == str(record.get("target_camera")):
        return Path(record["target_video_path"])
    else:
        raise ValueError(
            f"Arbitrary train camera {camera_name!r} requires a DeepView, AIST, Neural 3D Video, ViVo, "
            f"or CamXTime record; "
            f"record dataset={dataset!r}."
        )


def load_camera_video(
    record: dict[str, Any],
    camera_name: str,
    *,
    target_size: ImageSizeLike,
    device: torch.device,
    frame_count: int | None = None,
) -> torch.Tensor:
    if str(record.get("dataset") or "") in DNERF_DATASETS:
        return load_dnerf_camera_frames(
            record,
            camera_name,
            target_size=target_size,
            device=device,
            frame_count=int(frame_count if frame_count is not None else record["frame_count"]),
        ).contiguous()
    video_path = video_path_for_camera(record, camera_name)
    start_seconds = camera_start_seconds(record, camera_name)
    frames = load_multicam_val_camera_frames(
        video_path=video_path,
        start_seconds=start_seconds,
        fps=float(record["fps"]),
        frame_count=int(frame_count if frame_count is not None else record["frame_count"]),
        target_size=target_size,
        device=device,
    )
    return frames.contiguous()


def sequence_source_path_for_camera(record: dict[str, Any], camera_name: str) -> Path | None:
    try:
        return video_path_for_camera(record, camera_name)
    except ValueError:
        if "source_video_path" in record:
            return Path(record["source_video_path"])
        return None


def duplicate_camera_names(camera_names: list[str]) -> list[str]:
    seen = set()
    duplicates = []
    for camera_name in camera_names:
        if camera_name in seen and camera_name not in duplicates:
            duplicates.append(camera_name)
        seen.add(camera_name)
    return duplicates


def validate_multicam_camera_split(
    *,
    train_cameras: list[str],
    heldout_cameras: list[str],
    anchor_camera: str,
    condition_camera: str,
) -> None:
    if not train_cameras:
        raise ValueError("data.multicam_train_cameras must contain at least one camera.")
    if not heldout_cameras:
        raise ValueError("data.multicam_heldout_cameras/data.multicam_heldout_camera must contain at least one camera.")

    duplicate_train = duplicate_camera_names(train_cameras)
    if duplicate_train:
        raise ValueError(f"data.multicam_train_cameras contains duplicates: {duplicate_train}.")
    duplicate_heldout = duplicate_camera_names(heldout_cameras)
    if duplicate_heldout:
        raise ValueError(f"data.multicam_heldout_cameras contains duplicates: {duplicate_heldout}.")

    overlap = sorted(set(train_cameras) & set(heldout_cameras))
    if overlap:
        raise ValueError(f"Multicam train/heldout camera split overlaps: {overlap}.")
    if anchor_camera not in train_cameras:
        raise ValueError("data.multicam_anchor_camera must be one of data.multicam_train_cameras.")
    if condition_camera not in train_cameras:
        raise ValueError("data.multicam_condition_camera must be one of data.multicam_train_cameras.")


def make_deepview_multiview_cameras(
    record: dict[str, Any],
    *,
    train_cameras: list[str],
    heldout_cameras: list[str],
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

    heldout_K = []
    heldout_w2c = []
    for camera_name in heldout_cameras:
        K, c2w = deepview_camera_from_models(record, camera_name, H=H, W=W, device=device)
        rel_w2c = torch.linalg.inv(c2w) @ anchor_c2w
        heldout_K.append(K)
        heldout_w2c.append(rel_w2c.unsqueeze(0).repeat(T, 1, 1))
    lens_models, _distortions = deepview_lens_metadata(record, train_cameras + heldout_cameras, device=device)
    pose_source = (
        "deepview_models_relative_opencv_fisheye"
        if set(lens_models) == {"opencv_fisheye"}
        else "deepview_models_relative_pinhole"
    )
    return (
        torch.stack(train_K, dim=0),
        torch.stack(train_w2c, dim=0),
        torch.stack(heldout_K, dim=0),
        torch.stack(heldout_w2c, dim=0),
        pose_source,
    )


def make_orthogonal_origin_multiview_cameras(
    *,
    view_count: int,
    heldout_count: int,
    T: int,
    H: int,
    W: int,
    radius: float,
    fov_degrees: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
    base_positions = [
        torch.tensor([0.0, 0.0, -float(radius)], dtype=torch.float32, device=device),
        torch.tensor([float(radius), 0.0, 0.0], dtype=torch.float32, device=device),
        torch.tensor([0.0, 0.0, float(radius)], dtype=torch.float32, device=device),
        torch.tensor([-float(radius), 0.0, 0.0], dtype=torch.float32, device=device),
    ]
    if view_count > len(base_positions):
        raise ValueError(f"orthogonal_origin supports at most {len(base_positions)} train views, got {view_count}.")
    K = make_fixed_pinhole_K(H=H, W=W, fov_degrees=fov_degrees, device=device)
    train_K = K.unsqueeze(0).repeat(view_count, 1, 1)
    train_w2c = []
    for index in range(view_count):
        c2w = build_look_at_camera_to_world(base_positions[index])
        train_w2c.append(torch.linalg.inv(c2w).unsqueeze(0).repeat(T, 1, 1))
    heldout_position = torch.tensor([0.0, float(radius), 0.0], dtype=torch.float32, device=device)
    heldout_c2w = build_look_at_camera_to_world(
        heldout_position,
        up=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device),
    )
    heldout_K = K.unsqueeze(0).repeat(heldout_count, 1, 1)
    heldout_w2c = torch.linalg.inv(heldout_c2w).unsqueeze(0).unsqueeze(0).repeat(heldout_count, T, 1, 1)
    return (
        train_K,
        torch.stack(train_w2c, dim=0),
        heldout_K,
        heldout_w2c,
        "orthogonal_origin_pinhole",
    )


def camera_from_K_w2c(
    K: torch.Tensor,
    w2c: torch.Tensor,
    *,
    lens_model: str = "pinhole",
    distortion: torch.Tensor | None = None,
) -> CameraSpec:
    c2w = torch.linalg.inv(w2c)
    return CameraSpec(
        fx=K[0, 0],
        fy=K[1, 1],
        cx=K[0, 2],
        cy=K[1, 2],
        camera_to_world=c2w,
        lens_model=lens_model,  # type: ignore[arg-type]
        distortion=distortion,
    )


def source_relative_cameras_from_K_w2c(
    *,
    source_w2c: torch.Tensor,
    target_K: torch.Tensor,
    target_w2c: torch.Tensor,
    frame_indices: torch.Tensor,
    target_lens_model: str = "pinhole",
    target_distortion: torch.Tensor | None = None,
) -> tuple[CameraSpec, ...]:
    """Return target cameras expressed in the source camera's local frame.

    Stored multicam poses are relative to the configured anchor:
    `w2c_view_anchor = inv(c2w_view) @ c2w_anchor`.  For a source-anchored
    world, the target query is `inv(c2w_target) @ c2w_source`, which is:
    `w2c_target_anchor @ inv(w2c_source_anchor)`.
    """

    if source_w2c.ndim != 3 or target_w2c.ndim != 3:
        raise ValueError(
            f"Expected source_w2c and target_w2c as [T,4,4], got {tuple(source_w2c.shape)} "
            f"and {tuple(target_w2c.shape)}."
        )
    source = source_w2c.index_select(0, frame_indices)
    target = target_w2c.index_select(0, frame_indices)
    target_in_source_w2c = target @ torch.linalg.inv(source)
    return tuple(
        camera_from_K_w2c(
            target_K,
            target_in_source_w2c[index],
            lens_model=target_lens_model,
            distortion=target_distortion,
        )
        for index in range(len(frame_indices))
    )


def cameras_from_K_w2c(
    K: torch.Tensor,
    w2c: torch.Tensor,
    *,
    lens_models: list[str] | None = None,
    distortions: torch.Tensor | None = None,
) -> tuple[tuple[CameraSpec, ...], ...]:
    if K.ndim != 3 or w2c.ndim != 4:
        raise ValueError(f"Expected K [V,3,3] and w2c [V,T,4,4], got {tuple(K.shape)} and {tuple(w2c.shape)}.")
    view_count, frame_count = int(w2c.shape[0]), int(w2c.shape[1])
    if lens_models is not None and len(lens_models) != view_count:
        raise ValueError(f"Expected {view_count} lens models, got {len(lens_models)}.")
    if distortions is not None and int(distortions.shape[0]) != view_count:
        raise ValueError(f"Expected distortions first dim {view_count}, got {tuple(distortions.shape)}.")
    return tuple(
        tuple(
            camera_from_K_w2c(
                K[view],
                w2c[view, frame],
                lens_model="pinhole" if lens_models is None else lens_models[view],
                distortion=None if distortions is None else distortions[view],
            )
            for frame in range(frame_count)
        )
        for view in range(view_count)
    )


def heldout_cameras_from_K_w2c(
    K: torch.Tensor,
    w2c: torch.Tensor,
    *,
    lens_models: list[str] | None = None,
    distortions: torch.Tensor | None = None,
) -> tuple[tuple[CameraSpec, ...], ...]:
    if K.ndim == 2 and w2c.ndim == 3:
        K = K.unsqueeze(0)
        w2c = w2c.unsqueeze(0)
    if K.ndim != 3 or w2c.ndim != 4:
        raise ValueError(f"Expected heldout K [H,3,3] and w2c [H,T,4,4], got {tuple(K.shape)} and {tuple(w2c.shape)}.")
    heldout_count, frame_count = int(w2c.shape[0]), int(w2c.shape[1])
    if lens_models is not None and len(lens_models) != heldout_count:
        raise ValueError(f"Expected {heldout_count} lens models, got {len(lens_models)}.")
    if distortions is not None and int(distortions.shape[0]) != heldout_count:
        raise ValueError(f"Expected distortions first dim {heldout_count}, got {tuple(distortions.shape)}.")
    return tuple(
        tuple(
            camera_from_K_w2c(
                K[view],
                w2c[view, frame],
                lens_model="pinhole" if lens_models is None else lens_models[view],
                distortion=None if distortions is None else distortions[view],
            )
            for frame in range(frame_count)
        )
        for view in range(heldout_count)
    )


def load_multicam_video_bundle(
    *,
    data_cfg: dict[str, Any],
    camera_cfg: dict[str, Any],
    target_size: ImageSizeLike,
    device: torch.device,
    frame_device: torch.device | None = None,
    defer_video_frames: bool = False,
) -> MulticamVideoBundle:
    resolved_frame_device = device if frame_device is None else frame_device
    record = select_multicam_record(data_cfg)
    defer_video_frames = bool(
        defer_video_frames and str(record.get("dataset") or "") not in DNERF_DATASETS
    )
    train_raw = data_cfg.get("multicam_train_cameras") or record.get("train_cameras")
    if train_raw:
        train_cameras = [str(camera) for camera in train_raw]
    else:
        train_cameras = [str(record["source_camera"])]
    heldout_raw = data_cfg.get("multicam_heldout_cameras") or record.get("heldout_cameras")
    if heldout_raw:
        heldout_cameras = [str(camera) for camera in heldout_raw]
    else:
        heldout_cameras = [
            str(data_cfg.get("multicam_heldout_camera") or record.get("heldout_camera") or record["target_camera"])
        ]
    anchor_camera = str(data_cfg.get("multicam_anchor_camera") or record.get("anchor_camera") or train_cameras[0])
    condition_camera = str(data_cfg.get("multicam_condition_camera") or record.get("condition_camera") or anchor_camera)
    validate_multicam_camera_split(
        train_cameras=train_cameras,
        heldout_cameras=heldout_cameras,
        anchor_camera=anchor_camera,
        condition_camera=condition_camera,
    )

    if isinstance(target_size, bool):
        raise ValueError("target_size must be an integer or [height, width]")
    if isinstance(target_size, int):
        target_height = target_width = int(target_size)
    elif isinstance(target_size, (tuple, list)) and len(target_size) == 2:
        target_height, target_width = (int(value) for value in target_size)
    else:
        raise ValueError("target_size must be an integer or [height, width]")
    if target_height < 1 or target_width < 1:
        raise ValueError("target_size dimensions must be positive")

    camera_frame_count = requested_camera_frame_count(data_cfg, record)
    frame_positions = list(range(camera_frame_count))
    if data_cfg.get("frame_indices") is not None:
        selected_positions = [int(index) for index in data_cfg["frame_indices"]]
        if not selected_positions:
            raise ValueError("data.frame_indices must contain at least one index when provided.")
        invalid = next(
            (index for index in selected_positions if index < 0 or index >= camera_frame_count),
            None,
        )
        if invalid is not None:
            raise IndexError(f"data.frame_indices contains {invalid} outside [0, {camera_frame_count})")
        frame_positions = [frame_positions[index] for index in selected_positions]
    T = len(frame_positions)
    H, W = target_height, target_width

    def frame_source(camera_name: str) -> MulticamVideoFrameSource:
        return MulticamVideoFrameSource(
            camera_name=camera_name,
            video_path=video_path_for_camera(record, camera_name),
            start_seconds=camera_start_seconds(record, camera_name),
            sample_fps=float(record["fps"]),
            source_frame_count=camera_frame_count,
            selected_frame_indices=tuple(frame_positions),
            height=H,
            width=W,
        )

    if defer_video_frames:
        train_frame_sources = tuple(frame_source(camera) for camera in train_cameras)
        heldout_frame_sources = tuple(frame_source(camera) for camera in heldout_cameras)
        train_frames = torch.empty(
            (len(train_cameras), T, 3, H, W),
            dtype=torch.float32,
            device="meta",
        )
        heldout_frames = torch.empty(
            (len(heldout_cameras), T, 3, H, W),
            dtype=torch.float32,
            device="meta",
        )
    else:
        train_frame_sources = ()
        heldout_frame_sources = ()
        train_frames = torch.stack(
            [
                load_camera_video(
                    record,
                    camera,
                    target_size=target_size,
                    device=resolved_frame_device,
                    frame_count=camera_frame_count,
                )
                for camera in train_cameras
            ],
            dim=0,
        )
        heldout_frames = torch.stack(
            [
                load_camera_video(
                    record,
                    camera,
                    target_size=target_size,
                    device=resolved_frame_device,
                    frame_count=camera_frame_count,
                )
                for camera in heldout_cameras
            ],
            dim=0,
        )
        max_frames = int(data_cfg.get("max_frames") or 0)
        if max_frames > 0:
            train_frames = train_frames[:, :max_frames].contiguous()
            heldout_frames = heldout_frames[:, :max_frames].contiguous()
        train_frames = select_configured_multiview_frames(
            train_frames,
            data_cfg.get("frame_indices"),
        )
        heldout_frames = select_configured_multiview_frames(
            heldout_frames,
            data_cfg.get("frame_indices"),
        )
        if tuple(train_frames.shape[1:]) != (T, 3, H, W):
            raise ValueError(
                "decoded train video shape does not match requested metadata: "
                f"expected {(T, 3, H, W)}, got {tuple(train_frames.shape[1:])}"
            )
    rig_init = str(camera_cfg.get("rig_init", "deepview")).lower()
    train_lens_models = None
    train_distortions = None
    heldout_lens_models = None
    heldout_distortions = None
    if rig_init == "dnerf":
        if str(record.get("dataset") or "") not in DNERF_DATASETS:
            raise ValueError(f"camera.rig_init=dnerf requires a D-NeRF record; got {record.get('dataset')!r}.")
        train_K, train_w2c, heldout_K, heldout_w2c, anchor_c2w, pose_source = (
            make_dnerf_matched_trajectory_cameras(
                record,
                train_cameras=train_cameras,
                heldout_cameras=heldout_cameras,
                frame_positions=frame_positions,
                H=H,
                W=W,
                device=device,
            )
        )
    elif rig_init == "deepview":
        if record.get("dataset") != "deepview_video":
            raise ValueError(
                f"camera.rig_init=deepview requires a DeepView record; got dataset={record.get('dataset')!r}."
            )
        _, anchor_c2w = deepview_camera_from_models(record, anchor_camera, H=H, W=W, device=device)
        train_K, train_w2c, heldout_K, heldout_w2c, pose_source = make_deepview_multiview_cameras(
            record,
            train_cameras=train_cameras,
            heldout_cameras=heldout_cameras,
            anchor_camera=anchor_camera,
            T=T,
            H=H,
            W=W,
            device=device,
        )
        train_lens_models, train_distortions = deepview_lens_metadata(record, train_cameras, device=device)
        heldout_lens_models, heldout_distortions = deepview_lens_metadata(record, heldout_cameras, device=device)
        if set(train_lens_models + heldout_lens_models) == {"opencv_fisheye"}:
            pose_source = "deepview_models_relative_opencv_fisheye"
    elif rig_init == "aist":
        if record.get("dataset") != "aist_dance_db":
            raise ValueError(
                f"camera.rig_init=aist requires an AIST record; got dataset={record.get('dataset')!r}."
            )
        translation_scale = float(camera_cfg.get("aist_translation_scale", 1.0))
        _, anchor_c2w = aist_camera_from_setting(
            record, anchor_camera, H=H, W=W, device=device, translation_scale=translation_scale
        )
        train_K, train_w2c, heldout_K, heldout_w2c, pose_source = make_aist_multiview_cameras(
            record,
            train_cameras=train_cameras,
            heldout_cameras=heldout_cameras,
            anchor_camera=anchor_camera,
            T=T,
            H=H,
            W=W,
            device=device,
            translation_scale=translation_scale,
        )
    elif rig_init == "neural_3d_video":
        if record.get("dataset") != "neural_3d_video":
            raise ValueError(
                f"camera.rig_init=neural_3d_video requires a Neural 3D Video record; "
                f"got dataset={record.get('dataset')!r}."
            )
        translation_scale = float(camera_cfg.get("n3d_translation_scale", 1.0))
        _, anchor_c2w = neural_3d_camera_from_poses_bounds(
            record, anchor_camera, H=H, W=W, device=device, translation_scale=translation_scale
        )
        train_K, train_w2c, heldout_K, heldout_w2c, pose_source = make_neural_3d_multiview_cameras(
            record,
            train_cameras=train_cameras,
            heldout_cameras=heldout_cameras,
            anchor_camera=anchor_camera,
            T=T,
            H=H,
            W=W,
            device=device,
            translation_scale=translation_scale,
        )
    elif rig_init == "vivo":
        if record.get("dataset") != "vivo":
            raise ValueError(
                f"camera.rig_init=vivo requires a ViVo record; got dataset={record.get('dataset')!r}."
            )
        translation_scale = float(camera_cfg.get("vivo_translation_scale", 1.0))
        _, anchor_c2w = vivo_camera_from_calibration(
            record, anchor_camera, H=H, W=W, device=device, translation_scale=translation_scale
        )
        train_K, train_w2c, heldout_K, heldout_w2c, pose_source = make_vivo_multiview_cameras(
            record,
            train_cameras=train_cameras,
            heldout_cameras=heldout_cameras,
            anchor_camera=anchor_camera,
            T=T,
            H=H,
            W=W,
            device=device,
            translation_scale=translation_scale,
        )
    elif rig_init == "camxtime":
        if str(record.get("dataset") or "") not in CAMXTIME_DATASETS:
            raise ValueError(
                f"camera.rig_init=camxtime requires a CamXTime record; got dataset={record.get('dataset')!r}."
            )
        camera_data = load_camxtime_camera_data(record)
        camera_convention = str(record.get("camxtime_camera_convention", "opengl"))
        anchor_indices = camxtime_camera_indices(
            anchor_camera,
            frame_count=T,
            camera_count=camxtime_camera_count(camera_data),
        )
        anchor_c2w = camxtime_c2w_for_index(
            camera_data,
            anchor_indices[0],
            device=device,
            extrinsic_convention=str(record.get("camxtime_extrinsic_convention", "c2w")),
            camera_convention=camera_convention,
        )
        train_K, train_w2c, heldout_K, heldout_w2c, pose_source = make_camxtime_multiview_cameras(
            record,
            train_cameras=train_cameras,
            heldout_cameras=heldout_cameras,
            anchor_camera=anchor_camera,
            T=T,
            H=H,
            W=W,
            device=device,
        )
    elif rig_init == "orthogonal_origin":
        anchor_c2w = torch.eye(4, dtype=torch.float32, device=device)
        train_K, train_w2c, heldout_K, heldout_w2c, pose_source = make_orthogonal_origin_multiview_cameras(
            view_count=len(train_cameras),
            heldout_count=len(heldout_cameras),
            T=T,
            H=H,
            W=W,
            radius=float(camera_cfg.get("rig_radius", camera_cfg.get("base_radius", 3.0))),
            fov_degrees=float(camera_cfg.get("base_fov_degrees", 60.0)),
            device=device,
        )
    else:
        raise ValueError(
            "camera.rig_init must be one of: dnerf, deepview, aist, neural_3d_video, vivo, camxtime, "
            "orthogonal_origin"
        )

    condition_index = train_cameras.index(condition_camera)
    if rig_init == "dnerf":
        normalized_times = torch.tensor(
            [float(record["dnerf_times"][index]) for index in frame_positions],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(-1)
    else:
        frame_times = torch.arange(T, dtype=torch.float32, device=device).unsqueeze(-1) / float(
            record.get("fps", 4.0)
        )
        normalized_times = normalize_frame_times(frame_times)
    all_frame_count = int(record.get("frame_count", T))
    train_sequences = tuple(
        SequenceData(
            frames=train_frames[index],
            frame_times=normalized_times,
            video_fps=float(record.get("fps", 4.0)),
            frame_source="explicit_video",
            source_path=sequence_source_path_for_camera(record, camera_name),
            selected_frame_count=T,
            all_frame_count=all_frame_count,
        )
        for index, camera_name in enumerate(train_cameras)
    )
    heldout_sequences = tuple(
        SequenceData(
            frames=heldout_frames[index],
            frame_times=normalized_times,
            video_fps=float(record.get("fps", 4.0)),
            frame_source="explicit_video",
            source_path=sequence_source_path_for_camera(record, camera_name),
            selected_frame_count=T,
            all_frame_count=all_frame_count,
        )
        for index, camera_name in enumerate(heldout_cameras)
    )
    condition_sequence = train_sequences[condition_index]
    return MulticamVideoBundle(
        condition_sequence=condition_sequence,
        train_sequences=train_sequences,
        train_frames=train_frames,
        train_K=train_K,
        train_w2c=train_w2c,
        train_camera_names=train_cameras,
        train_lens_models=train_lens_models,
        train_distortions=train_distortions,
        heldout_sequences=heldout_sequences,
        heldout_frames=heldout_frames,
        heldout_K=heldout_K,
        heldout_w2c=heldout_w2c,
        heldout_camera_names=heldout_cameras,
        heldout_lens_models=heldout_lens_models,
        heldout_distortions=heldout_distortions,
        pose_source=pose_source,
        anchor_c2w=anchor_c2w,
        metadata={
            **record,
            "train_cameras": train_cameras,
            "heldout_cameras": heldout_cameras,
            "anchor_camera": anchor_camera,
            "condition_camera": condition_camera,
            "sample_layout": record.get("sample_layout", "synchronized_multicamera"),
            "selected_frame_indices": frame_positions,
            "deferred_target_frames": bool(defer_video_frames),
        },
        train_frame_sources=train_frame_sources,
        heldout_frame_sources=heldout_frame_sources,
        deferred_target_frames=bool(defer_video_frames),
    )
