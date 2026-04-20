import math

import torch
import torch.nn as nn
from camera import CameraSpec, make_orbit_camera


def build_zero_init_head(in_dim, out_dim):
    head = nn.Sequential(
        nn.Linear(in_dim, in_dim),
        nn.SiLU(),
        nn.Linear(in_dim, out_dim),
    )
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)
    return head


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
        cameras.append(
            CameraSpec(
                fx=base_camera.fx,
                fy=base_camera.fy,
                cx=base_camera.cx,
                cy=base_camera.cy,
                camera_to_world=composed[index],
            )
        )
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


class PathCameraHead(nn.Module):
    def __init__(self, feat_dim, max_rotation_degrees=5.0, max_translation_ratio=0.2):
        super().__init__()
        self.max_rotation_radians = math.radians(max_rotation_degrees)
        self.max_translation_ratio = max_translation_ratio
        self.net = build_zero_init_head(feat_dim, 6)

    def forward(self, path_token, base_radius):
        raw = self.net(path_token)
        rotation_delta = torch.tanh(raw[:, :3]) * self.max_rotation_radians
        translation_delta = torch.tanh(raw[:, 3:]) * (base_radius * self.max_translation_ratio)
        return rotation_delta, translation_delta, raw


__all__ = [
    "GlobalCameraHead",
    "PathCameraHead",
    "axis_angle_to_matrix",
    "build_zero_init_head",
    "compose_camera_with_se3_delta",
    "skew_symmetric",
]
