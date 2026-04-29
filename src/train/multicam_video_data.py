from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from camera import CameraSpec, build_look_at_camera_to_world
from multicam_val_data import load_multicam_val_camera_frames, load_multicam_val_manifest
from runtime_types import SequenceData
from sequence_data import normalize_frame_times


@dataclass
class MulticamVideoBundle:
    condition_sequence: SequenceData
    train_frames: torch.Tensor
    train_K: torch.Tensor
    train_w2c: torch.Tensor
    train_camera_names: list[str]
    heldout_frames: torch.Tensor | None = None
    heldout_K: torch.Tensor | None = None
    heldout_w2c: torch.Tensor | None = None
    heldout_camera_names: list[str] | None = None
    pose_source: str | None = None
    metadata: dict[str, Any] | None = None

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
# Lens model: AIST++ cameras have a small radial-only distortion (k1 ~ -0.11) plus
# zero tangential terms.  The current splat renderer treats CameraSpec as pinhole;
# we keep that approximation here for parity with the DeepView path, and preserve
# the raw distortion coefficients in metadata so a later pass can adopt the OpenCV
# distortion model without touching ingestion.  Edge pixels in c01/c05/c09 carry
# 1-2px of residual distortion at 1920x1080, which scales to <0.2px at 128x128.
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
    params = json.loads(Path(setting_path).read_text(encoding="utf-8"))
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
#     [ 0:15 ] flattened 3x5 "pose-hwf" matrix in column-major LLFF order:
#                  R0  R1  R2  t   hwf
#                  -- 3x4 ----  -- 3x1 --
#         columns 0..2  : LLFF rotation columns (camera basis vectors in world space)
#         column   3    : world-space camera position (translation, c2w[:3, 3])
#         column   4    : [H, W, focal] in *native* pixel units
#     [15:17 ] [near, far] depth bounds for the scene (LLFF NDC bounds, unused here)
#
# LLFF axis convention (the historical NeRF/Mildenhall convention):
#     x_cam = right     (matches OpenCV +X)
#     y_cam = up        (opposite of OpenCV +Y, which points down)
#     z_cam = back      (opposite of OpenCV +Z, which points forward into the scene)
# i.e. LLFF cameras "look down -z" in their own frame, while OpenCV cameras
# "look down +z".
#
# Convert LLFF rotation columns -> OpenCV rotation columns:
#     R_opencv = R_llff @ diag([+1, -1, -1])
# This negates the y and z basis vectors, equivalently swapping LLFF's up/back
# pair to OpenCV's down/forward pair, while keeping the right axis. This is the
# standard `nerf_synthetic`/`run_nerf_helpers.recenter_poses` flip in matrix form.
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
    rotation_llff = pose_hwf[:, :3]                       # 3x3, LLFF basis columns.
    translation = pose_hwf[:, 3]                          # 3,   world-space camera position.
    height_native = float(pose_hwf[0, 4])
    width_native = float(pose_hwf[1, 4])
    focal_native = float(pose_hwf[2, 4])

    # LLFF -> OpenCV rotation: negate the y and z basis vectors (the second
    # and third columns of the rotation matrix). Equivalent to right-multiplying
    # by diag([+1, -1, -1]).
    llff_to_opencv = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    rotation_opencv = rotation_llff @ llff_to_opencv

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
        "neural_3d_llff_relative_pinhole",
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
    payload = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
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


def load_camera_video(record: dict[str, Any], camera_name: str, *, target_size: int, device: torch.device) -> torch.Tensor:
    dataset = str(record.get("dataset") or "")
    if dataset == "deepview_video":
        video_path = deepview_video_path_for_camera(record, camera_name)
    elif dataset == "aist_dance_db":
        video_path = aist_video_path_for_camera(record, camera_name)
    elif dataset == "neural_3d_video":
        video_path = neural_3d_video_path_for_camera(record, camera_name)
    elif dataset == "vivo":
        video_path = vivo_video_path_for_camera(record, camera_name)
    elif camera_name == str(record.get("source_camera")):
        video_path = Path(record["source_video_path"])
    elif camera_name == str(record.get("target_camera")):
        video_path = Path(record["target_video_path"])
    else:
        raise ValueError(
            f"Arbitrary train camera {camera_name!r} requires a DeepView, AIST, Neural 3D Video, or ViVo record; "
            f"record dataset={dataset!r}."
        )

    start_seconds = float(record.get("source_start_seconds", record.get("target_start_seconds", 0.0)))
    frames = load_multicam_val_camera_frames(
        video_path=video_path,
        start_seconds=start_seconds,
        fps=float(record["fps"]),
        frame_count=int(record["frame_count"]),
        target_size=target_size,
        device=device,
    )
    return frames.contiguous()


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
    return (
        torch.stack(train_K, dim=0),
        torch.stack(train_w2c, dim=0),
        torch.stack(heldout_K, dim=0),
        torch.stack(heldout_w2c, dim=0),
        "deepview_models_relative_pinhole",
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


def camera_from_K_w2c(K: torch.Tensor, w2c: torch.Tensor) -> CameraSpec:
    c2w = torch.linalg.inv(w2c)
    return CameraSpec(
        fx=K[0, 0],
        fy=K[1, 1],
        cx=K[0, 2],
        cy=K[1, 2],
        camera_to_world=c2w,
        lens_model="pinhole",
    )


def cameras_from_K_w2c(K: torch.Tensor, w2c: torch.Tensor) -> tuple[tuple[CameraSpec, ...], ...]:
    if K.ndim != 3 or w2c.ndim != 4:
        raise ValueError(f"Expected K [V,3,3] and w2c [V,T,4,4], got {tuple(K.shape)} and {tuple(w2c.shape)}.")
    view_count, frame_count = int(w2c.shape[0]), int(w2c.shape[1])
    return tuple(
        tuple(camera_from_K_w2c(K[view], w2c[view, frame]) for frame in range(frame_count))
        for view in range(view_count)
    )


def heldout_cameras_from_K_w2c(K: torch.Tensor, w2c: torch.Tensor) -> tuple[tuple[CameraSpec, ...], ...]:
    if K.ndim == 2 and w2c.ndim == 3:
        K = K.unsqueeze(0)
        w2c = w2c.unsqueeze(0)
    if K.ndim != 3 or w2c.ndim != 4:
        raise ValueError(f"Expected heldout K [H,3,3] and w2c [H,T,4,4], got {tuple(K.shape)} and {tuple(w2c.shape)}.")
    heldout_count, frame_count = int(w2c.shape[0]), int(w2c.shape[1])
    return tuple(
        tuple(camera_from_K_w2c(K[view], w2c[view, frame]) for frame in range(frame_count))
        for view in range(heldout_count)
    )


def load_multicam_video_bundle(
    *,
    data_cfg: dict[str, Any],
    camera_cfg: dict[str, Any],
    target_size: int,
    device: torch.device,
) -> MulticamVideoBundle:
    record = select_multicam_record(data_cfg)
    train_raw = data_cfg.get("multicam_train_cameras")
    if train_raw:
        train_cameras = [str(camera) for camera in train_raw]
    else:
        train_cameras = [str(record["source_camera"])]
    heldout_raw = data_cfg.get("multicam_heldout_cameras")
    if heldout_raw:
        heldout_cameras = [str(camera) for camera in heldout_raw]
    else:
        heldout_cameras = [str(data_cfg.get("multicam_heldout_camera") or record["target_camera"])]
    anchor_camera = str(data_cfg.get("multicam_anchor_camera") or train_cameras[0])
    condition_camera = str(data_cfg.get("multicam_condition_camera") or anchor_camera)
    if anchor_camera not in train_cameras:
        raise ValueError("data.multicam_anchor_camera must be one of data.multicam_train_cameras.")
    if condition_camera not in train_cameras:
        raise ValueError("data.multicam_condition_camera must be one of data.multicam_train_cameras.")

    train_frames = torch.stack(
        [load_camera_video(record, camera, target_size=target_size, device=device) for camera in train_cameras],
        dim=0,
    )
    heldout_frames = torch.stack(
        [load_camera_video(record, camera, target_size=target_size, device=device) for camera in heldout_cameras],
        dim=0,
    )
    max_frames = int(data_cfg.get("max_frames") or 0)
    if max_frames > 0:
        train_frames = train_frames[:, :max_frames].contiguous()
        heldout_frames = heldout_frames[:, :max_frames].contiguous()
    train_frames = select_configured_multiview_frames(train_frames, data_cfg.get("frame_indices"))
    heldout_frames = select_configured_multiview_frames(heldout_frames, data_cfg.get("frame_indices"))

    _, T, _, H, W = train_frames.shape
    rig_init = str(camera_cfg.get("rig_init", "deepview")).lower()
    if rig_init == "deepview":
        if record.get("dataset") != "deepview_video":
            raise ValueError(
                f"camera.rig_init=deepview requires a DeepView record; got dataset={record.get('dataset')!r}."
            )
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
    elif rig_init == "aist":
        if record.get("dataset") != "aist_dance_db":
            raise ValueError(
                f"camera.rig_init=aist requires an AIST record; got dataset={record.get('dataset')!r}."
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
            translation_scale=float(camera_cfg.get("aist_translation_scale", 1.0)),
        )
    elif rig_init == "neural_3d_video":
        if record.get("dataset") != "neural_3d_video":
            raise ValueError(
                f"camera.rig_init=neural_3d_video requires a Neural 3D Video record; "
                f"got dataset={record.get('dataset')!r}."
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
            translation_scale=float(camera_cfg.get("n3d_translation_scale", 1.0)),
        )
    elif rig_init == "vivo":
        if record.get("dataset") != "vivo":
            raise ValueError(
                f"camera.rig_init=vivo requires a ViVo record; got dataset={record.get('dataset')!r}."
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
            translation_scale=float(camera_cfg.get("vivo_translation_scale", 1.0)),
        )
    elif rig_init == "orthogonal_origin":
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
            "camera.rig_init must be one of: deepview, aist, neural_3d_video, vivo, orthogonal_origin"
        )

    condition_index = train_cameras.index(condition_camera)
    frame_times = torch.arange(T, dtype=torch.float32, device=device).unsqueeze(-1) / float(record.get("fps", 4.0))
    dataset = str(record.get("dataset") or "")
    if dataset == "deepview_video":
        condition_path = deepview_video_path_for_camera(record, condition_camera)
    elif dataset == "aist_dance_db":
        condition_path = aist_video_path_for_camera(record, condition_camera)
    elif dataset == "neural_3d_video":
        condition_path = neural_3d_video_path_for_camera(record, condition_camera)
    elif dataset == "vivo":
        condition_path = vivo_video_path_for_camera(record, condition_camera)
    else:
        condition_path = Path(record["source_video_path"])
    condition_sequence = SequenceData(
        frames=train_frames[condition_index],
        frame_times=normalize_frame_times(frame_times),
        video_fps=float(record.get("fps", 4.0)),
        frame_source="explicit_video",
        source_path=condition_path,
        selected_frame_count=T,
        all_frame_count=int(record.get("frame_count", T)),
    )
    return MulticamVideoBundle(
        condition_sequence=condition_sequence,
        train_frames=train_frames,
        train_K=train_K,
        train_w2c=train_w2c,
        train_camera_names=train_cameras,
        heldout_frames=heldout_frames,
        heldout_K=heldout_K,
        heldout_w2c=heldout_w2c,
        heldout_camera_names=heldout_cameras,
        pose_source=pose_source,
        metadata={
            **record,
            "train_cameras": train_cameras,
            "heldout_cameras": heldout_cameras,
            "anchor_camera": anchor_camera,
            "condition_camera": condition_camera,
        },
    )
