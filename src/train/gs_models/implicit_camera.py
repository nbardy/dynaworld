import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from camera import LensModel, make_camera_like, make_orbit_camera


def build_zero_init_head(in_dim, out_dim):
    head = nn.Sequential(
        nn.Linear(in_dim, in_dim),
        nn.SiLU(),
        nn.Linear(in_dim, out_dim),
    )
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)
    return head


def distortion_dim_for_lens(lens_model: str) -> int:
    if lens_model == "pinhole":
        return 0
    if lens_model == "radial_tangential":
        return 5
    if lens_model == "opencv_fisheye":
        return 4
    raise ValueError(
        f"Unknown lens_model={lens_model!r}. Expected one of: pinhole, radial_tangential, opencv_fisheye."
    )


def _coefficient_buffer(value, size: int, name: str) -> torch.Tensor:
    if size == 0:
        return torch.zeros(0, dtype=torch.float32)
    if value is None:
        return torch.zeros(size, dtype=torch.float32)
    values = torch.as_tensor(value, dtype=torch.float32).flatten()
    if values.numel() == 1:
        return values.expand(size).clone()
    if values.numel() != size:
        raise ValueError(f"{name} must be a scalar or have {size} values, got {values.numel()}.")
    return values


def skew_symmetric(vectors):
    batch_size = vectors.shape[0]
    skew = vectors.new_zeros((batch_size, 3, 3))
    skew[:, 0, 1] = -vectors[:, 2]
    skew[:, 0, 2] = vectors[:, 1]
    skew[:, 1, 0] = vectors[:, 2]
    skew[:, 1, 2] = -vectors[:, 0]
    skew[:, 2, 0] = -vectors[:, 1]
    skew[:, 2, 1] = vectors[:, 0]
    return skew


def axis_angle_to_matrix(axis_angle):
    angles = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
    axes = axis_angle / angles.clamp_min(1e-8)
    skew = skew_symmetric(axes)
    eye = (
        torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype).unsqueeze(0).expand(axis_angle.shape[0], -1, -1)
    )
    sin_term = torch.sin(angles).unsqueeze(-1)
    cos_term = (1.0 - torch.cos(angles)).unsqueeze(-1)
    small_angle = angles.squeeze(-1) < 1e-6
    rotation = eye + sin_term * skew + cos_term * (skew @ skew)
    if torch.any(small_angle):
        small_skew = skew_symmetric(axis_angle[small_angle])
        rotation[small_angle] = eye[small_angle] + small_skew
    return rotation


def compose_camera_with_se3_delta(base_camera, rotation_delta, translation_delta):
    delta_transform = (
        torch.eye(4, device=rotation_delta.device, dtype=rotation_delta.dtype)
        .unsqueeze(0)
        .repeat(rotation_delta.shape[0], 1, 1)
    )
    delta_transform[:, :3, :3] = axis_angle_to_matrix(rotation_delta)
    delta_transform[:, :3, 3] = translation_delta

    base_transform = base_camera.camera_to_world.to(device=rotation_delta.device, dtype=rotation_delta.dtype)
    composed = base_transform.unsqueeze(0) @ delta_transform
    cameras = []
    for index in range(rotation_delta.shape[0]):
        cameras.append(make_camera_like(base_camera, camera_to_world=composed[index]))
    return cameras


def build_cameras_from_optical_axis(base_camera, rotation_delta, translation_delta):
    device = rotation_delta.device
    dtype = rotation_delta.dtype
    base_transform = base_camera.camera_to_world.to(device=device, dtype=dtype)
    base_right = base_transform[:3, 0]
    base_up = base_transform[:3, 1]
    base_forward = base_transform[:3, 2]
    base_center = base_transform[:3, 3]

    yaw = rotation_delta[:, 0:1]
    pitch = rotation_delta[:, 1:2]
    roll = rotation_delta[:, 2:3]
    forward = F.normalize(base_forward.unsqueeze(0) + yaw * base_right + pitch * base_up, dim=-1)

    reference_up = base_up.unsqueeze(0).expand_as(forward)
    right = F.normalize(torch.cross(reference_up, forward, dim=-1), dim=-1)
    up = F.normalize(torch.cross(forward, right, dim=-1), dim=-1)

    roll_cos = torch.cos(roll)
    roll_sin = torch.sin(roll)
    rolled_right = roll_cos * right + roll_sin * up
    rolled_up = -roll_sin * right + roll_cos * up

    local_translation = (
        translation_delta[:, 0:1] * base_right
        + translation_delta[:, 1:2] * base_up
        + translation_delta[:, 2:3] * base_forward
    )
    centers = base_center.unsqueeze(0) + local_translation

    transforms = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(rotation_delta.shape[0], 1, 1)
    transforms[:, :3, 0] = rolled_right
    transforms[:, :3, 1] = rolled_up
    transforms[:, :3, 2] = forward
    transforms[:, :3, 3] = centers

    cameras = []
    for index in range(rotation_delta.shape[0]):
        cameras.append(make_camera_like(base_camera, camera_to_world=transforms[index]))
    return cameras


class GlobalCameraHead(nn.Module):
    def __init__(
        self,
        feat_dim,
        base_fov_degrees=60.0,
        base_radius=3.0,
        max_fov_delta_degrees=15.0,
        max_radius_scale=1.5,
    ):
        super().__init__()
        self.base_fov_radians = math.radians(base_fov_degrees)
        self.base_radius = base_radius
        self.max_fov_delta_radians = math.radians(max_fov_delta_degrees)
        self.max_log_radius_delta = math.log(max_radius_scale)
        self.net = build_zero_init_head(feat_dim, 2)

    def forward(self, camera_token, image_size):
        raw = self.net(camera_token)
        fov = self.base_fov_radians + torch.tanh(raw[..., 0]) * self.max_fov_delta_radians
        radius = self.base_radius * torch.exp(torch.tanh(raw[..., 1]) * self.max_log_radius_delta)
        image_extent = camera_token.new_tensor(float(image_size))
        focal = 0.5 * image_extent / torch.tan(0.5 * fov)
        camera = make_orbit_camera(
            image_size=image_size,
            radius=radius,
            azimuth=camera_token.new_tensor(0.0),
            elevation=camera_token.new_tensor(0.0),
            focal=focal,
            device=camera_token.device,
            dtype=camera_token.dtype,
        )
        camera_state = {
            "fov_radians": fov,
            "fov_degrees": torch.rad2deg(fov),
            "radius": radius,
            "global_residuals": raw,
        }
        return camera, camera_state


class CentralLensCameraHead(nn.Module):
    def __init__(
        self,
        feat_dim,
        lens_model: LensModel = "pinhole",
        base_fov_degrees=60.0,
        base_radius=3.0,
        max_fov_delta_degrees=15.0,
        max_radius_scale=1.5,
        max_aspect_log_delta=0.0,
        max_principal_point_delta=0.0,
        distortion_max_abs=0.0,
        base_distortion=None,
    ):
        super().__init__()
        self.lens_model = lens_model
        self.distortion_dim = distortion_dim_for_lens(lens_model)
        self.base_fov_radians = math.radians(base_fov_degrees)
        self.base_radius = base_radius
        self.max_fov_delta_radians = math.radians(max_fov_delta_degrees)
        self.max_log_radius_delta = math.log(max_radius_scale)
        self.max_aspect_log_delta = float(max_aspect_log_delta)
        self.max_principal_point_delta = float(max_principal_point_delta)
        self.net = build_zero_init_head(feat_dim, 5 + self.distortion_dim)
        self.register_buffer(
            "base_distortion",
            _coefficient_buffer(base_distortion, self.distortion_dim, "base_distortion"),
            persistent=False,
        )
        self.register_buffer(
            "distortion_max_abs",
            _coefficient_buffer(distortion_max_abs, self.distortion_dim, "distortion_max_abs"),
            persistent=False,
        )

    def forward(self, camera_token, image_size):
        raw = self.net(camera_token)
        fov = self.base_fov_radians + torch.tanh(raw[..., 0]) * self.max_fov_delta_radians
        radius = self.base_radius * torch.exp(torch.tanh(raw[..., 1]) * self.max_log_radius_delta)
        aspect = torch.exp(torch.tanh(raw[..., 2]) * self.max_aspect_log_delta)

        image_extent = camera_token.new_tensor(float(image_size))
        center = image_extent * 0.5
        principal_offset = torch.tanh(raw[..., 3:5]) * self.max_principal_point_delta * image_extent
        cx = center + principal_offset[..., 0]
        cy = center + principal_offset[..., 1]
        focal = 0.5 * image_extent / torch.tan(0.5 * fov)
        fx = focal * aspect
        fy = focal / aspect

        base_camera = make_orbit_camera(
            image_size=image_size,
            radius=radius,
            azimuth=camera_token.new_tensor(0.0),
            elevation=camera_token.new_tensor(0.0),
            focal=focal,
            device=camera_token.device,
            dtype=camera_token.dtype,
        )

        distortion = None
        if self.distortion_dim > 0:
            base_distortion = self.base_distortion.to(device=camera_token.device, dtype=camera_token.dtype)
            distortion_max_abs = self.distortion_max_abs.to(device=camera_token.device, dtype=camera_token.dtype)
            distortion = base_distortion + torch.tanh(raw[..., 5 : 5 + self.distortion_dim]) * distortion_max_abs

        camera = make_camera_like(
            base_camera,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            lens_model=self.lens_model,
            distortion=distortion,
        )
        camera_state = {
            "fov_radians": fov,
            "fov_degrees": torch.rad2deg(fov),
            "radius": radius,
            "global_residuals": raw,
        }
        return camera, camera_state


def build_global_camera_head(
    head_type: str,
    *,
    feat_dim,
    lens_model: LensModel = "pinhole",
    base_fov_degrees=60.0,
    base_radius=3.0,
    max_fov_delta_degrees=15.0,
    max_radius_scale=1.5,
    max_aspect_log_delta=0.0,
    max_principal_point_delta=0.0,
    distortion_max_abs=0.0,
    base_distortion=None,
):
    normalized_head_type = str(head_type).lower()
    if normalized_head_type in {"legacy_orbit", "legacy_pinhole", "simple_pinhole"}:
        if lens_model != "pinhole":
            raise ValueError("legacy_orbit global camera head only supports lens_model='pinhole'.")
        return GlobalCameraHead(
            feat_dim=feat_dim,
            base_fov_degrees=base_fov_degrees,
            base_radius=base_radius,
            max_fov_delta_degrees=max_fov_delta_degrees,
            max_radius_scale=max_radius_scale,
        )
    if normalized_head_type in {"central_lens", "central_ray_lens"}:
        return CentralLensCameraHead(
            feat_dim=feat_dim,
            lens_model=lens_model,
            base_fov_degrees=base_fov_degrees,
            base_radius=base_radius,
            max_fov_delta_degrees=max_fov_delta_degrees,
            max_radius_scale=max_radius_scale,
            max_aspect_log_delta=max_aspect_log_delta,
            max_principal_point_delta=max_principal_point_delta,
            distortion_max_abs=distortion_max_abs,
            base_distortion=base_distortion,
        )
    raise ValueError(
        f"Unknown camera global_head={head_type!r}. Expected legacy_orbit or central_lens."
    )


class PathCameraHead(nn.Module):
    def __init__(self, feat_dim, max_rotation_degrees=5.0, max_translation_ratio=0.2):
        super().__init__()
        self.max_rotation_radians = math.radians(max_rotation_degrees)
        self.max_translation_ratio = max_translation_ratio
        self.net = build_zero_init_head(feat_dim, 6)

    def forward(self, path_token, base_radius, decode_time=None):
        raw = self.net(path_token)
        rotation_delta = torch.tanh(raw[:, :3]) * self.max_rotation_radians
        translation_delta = torch.tanh(raw[:, 3:]) * (base_radius * self.max_translation_ratio)
        return rotation_delta, translation_delta, raw


class TimeConditionedPathCameraHead(nn.Module):
    def __init__(self, feat_dim, time_conditioner, max_rotation_degrees=5.0, max_translation_ratio=0.2):
        super().__init__()
        self.max_rotation_radians = math.radians(max_rotation_degrees)
        self.max_translation_ratio = max_translation_ratio
        self.time_conditioner = time_conditioner
        self.net = build_zero_init_head(feat_dim, 6)

    def forward(self, path_token, base_radius, decode_time=None):
        if decode_time is None:
            raise ValueError("decode_time is required for TimeConditionedPathCameraHead.")
        if decode_time.ndim == 1:
            decode_time = decode_time.unsqueeze(-1)
        decode_time = decode_time.to(device=path_token.device, dtype=path_token.dtype)
        raw = self.net(path_token + self.time_conditioner(decode_time))
        rotation_delta = torch.tanh(raw[:, :3]) * self.max_rotation_radians
        translation_delta = torch.tanh(raw[:, 3:]) * (base_radius * self.max_translation_ratio)
        return rotation_delta, translation_delta, raw


class TimeConditionedOpticalAxisCameraHead(nn.Module):
    def __init__(self, feat_dim, time_conditioner, max_rotation_degrees=5.0, max_translation_ratio=0.2):
        super().__init__()
        self.max_rotation_radians = math.radians(max_rotation_degrees)
        self.max_translation_ratio = max_translation_ratio
        self.time_conditioner = time_conditioner
        self.net = build_zero_init_head(feat_dim, 6)

    def forward(self, path_token, base_camera, base_radius, decode_time=None):
        if decode_time is None:
            raise ValueError("decode_time is required for TimeConditionedOpticalAxisCameraHead.")
        if decode_time.ndim == 1:
            decode_time = decode_time.unsqueeze(-1)
        decode_time = decode_time.to(device=path_token.device, dtype=path_token.dtype)
        raw = self.net(path_token + self.time_conditioner(decode_time))
        rotation_delta = torch.tanh(raw[:, :3]) * self.max_rotation_radians
        translation_delta = torch.tanh(raw[:, 3:]) * (base_radius * self.max_translation_ratio)
        cameras = build_cameras_from_optical_axis(base_camera, rotation_delta, translation_delta)
        return cameras, rotation_delta, translation_delta, raw


__all__ = [
    "CentralLensCameraHead",
    "GlobalCameraHead",
    "PathCameraHead",
    "TimeConditionedPathCameraHead",
    "TimeConditionedOpticalAxisCameraHead",
    "axis_angle_to_matrix",
    "build_cameras_from_optical_axis",
    "build_global_camera_head",
    "build_zero_init_head",
    "compose_camera_with_se3_delta",
    "distortion_dim_for_lens",
    "skew_symmetric",
]
