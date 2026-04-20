import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from camera import CameraSpec, make_orbit_camera

from .blocks import ConvImageEncoder, TokenAttentionBlock, build_mlp, flatten_hw_features


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


class CanonicalGaussianParameterHeads(nn.Module):
    def __init__(self, feat_dim, gaussians_per_token, scene_extent=1.0):
        super().__init__()
        self.gaussians_per_token = gaussians_per_token
        self.scene_extent = scene_extent
        self.xyz_head = build_mlp(feat_dim, gaussians_per_token * 3)
        self.scale_head = build_mlp(feat_dim, gaussians_per_token * 3)
        self.rot_head = build_mlp(feat_dim, gaussians_per_token * 4)
        self.opacity_head = build_mlp(feat_dim, gaussians_per_token)
        self.rgb_head = build_mlp(feat_dim, gaussians_per_token * 3)

    def _reshape(self, values, channels):
        batch_size, token_count, _ = values.shape
        gaussian_count = token_count * self.gaussians_per_token
        return values.reshape(batch_size, gaussian_count, channels)

    def forward(self, tokens):
        xyz = torch.tanh(self._reshape(self.xyz_head(tokens), 3)) * self.scene_extent
        scales = torch.exp(self._reshape(self.scale_head(tokens), 3)) * 0.05
        quats = F.normalize(self._reshape(self.rot_head(tokens), 4), p=2, dim=-1)
        opacities = torch.sigmoid(self._reshape(self.opacity_head(tokens), 1))
        rgbs = torch.sigmoid(self._reshape(self.rgb_head(tokens), 3))
        return xyz, scales, quats, opacities, rgbs


class GlobalCameraHead(nn.Module):
    def __init__(self, feat_dim, base_fov_degrees=60.0, base_radius=3.0):
        super().__init__()
        self.base_fov_radians = math.radians(base_fov_degrees)
        self.base_radius = base_radius
        self.max_fov_delta_radians = math.radians(15.0)
        self.max_log_radius_delta = math.log(1.5)
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
    def __init__(self, feat_dim):
        super().__init__()
        self.max_rotation_radians = math.radians(5.0)
        self.max_translation_ratio = 0.2
        self.net = build_zero_init_head(feat_dim, 6)

    def forward(self, path_token, base_radius):
        raw = self.net(path_token)
        rotation_delta = torch.tanh(raw[:, :3]) * self.max_rotation_radians
        translation_delta = torch.tanh(raw[:, 3:]) * (base_radius * self.max_translation_ratio)
        return rotation_delta, translation_delta, raw


class DynamicTokenGSImplicitCamera(nn.Module):
    def __init__(self, num_tokens=128, feat_dim=128, gaussians_per_token=4, scene_extent=1.0):
        super().__init__()
        self.num_tokens = num_tokens
        self.total_tokens = num_tokens + 2
        self.feat_dim = feat_dim
        self.gaussians_per_token = gaussians_per_token

        self.encoder = ConvImageEncoder(feat_dim)
        self.tokens = nn.Parameter(torch.randn(1, self.total_tokens, feat_dim))
        self.token_block = TokenAttentionBlock(feat_dim=feat_dim)
        self.gaussian_heads = CanonicalGaussianParameterHeads(
            feat_dim=feat_dim,
            gaussians_per_token=gaussians_per_token,
            scene_extent=scene_extent,
        )
        self.global_camera_head = GlobalCameraHead(feat_dim=feat_dim)
        self.path_camera_head = PathCameraHead(feat_dim=feat_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(1, feat_dim),
            nn.SiLU(),
            nn.Linear(feat_dim, feat_dim),
        )

    def _refine_tokens(self, img, frame_times=None):
        batch_size = img.shape[0]
        if frame_times is None:
            frame_times = torch.zeros((batch_size, 1), device=img.device, dtype=img.dtype)
        else:
            frame_times = frame_times.to(device=img.device, dtype=img.dtype).reshape(batch_size, 1)

        feature_map = self.encoder(img)
        context = flatten_hw_features(feature_map)
        token_offsets = torch.zeros(
            (batch_size, self.total_tokens, self.feat_dim),
            device=img.device,
            dtype=img.dtype,
        )
        token_offsets[:, 1:, :] = self.time_proj(frame_times).unsqueeze(1)
        queries = self.tokens.expand(batch_size, -1, -1) + token_offsets
        return self.token_block(queries, context)

    @torch.no_grad()
    def infer_global_camera_token(self, img, frame_times=None, batch_size=4):
        tokens = []
        for start in range(0, img.shape[0], batch_size):
            end = min(start + batch_size, img.shape[0])
            refined_tokens = self._refine_tokens(
                img[start:end], None if frame_times is None else frame_times[start:end]
            )
            tokens.append(refined_tokens[:, 0, :])
        return torch.cat(tokens, dim=0).mean(dim=0)

    def forward(self, img, frame_times=None, global_camera_token=None):
        refined_tokens = self._refine_tokens(img, frame_times=frame_times)

        if global_camera_token is None:
            global_camera_token = refined_tokens[:, 0, :].mean(dim=0)
        else:
            global_camera_token = global_camera_token.to(device=img.device, dtype=img.dtype)

        splat_tokens = refined_tokens[:, 2:, :]
        path_tokens = refined_tokens[:, 1, :]
        xyz, scales, quats, opacities, rgbs = self.gaussian_heads(splat_tokens)

        base_camera, camera_state = self.global_camera_head(global_camera_token, image_size=img.shape[-1])
        rotation_delta, translation_delta, path_residuals = self.path_camera_head(
            path_tokens, base_radius=camera_state["radius"]
        )
        cameras = compose_camera_with_se3_delta(base_camera, rotation_delta, translation_delta)
        camera_state["rotation_delta"] = rotation_delta
        camera_state["translation_delta"] = translation_delta
        camera_state["path_residuals"] = path_residuals
        return xyz, scales, quats, opacities, rgbs, cameras, camera_state
