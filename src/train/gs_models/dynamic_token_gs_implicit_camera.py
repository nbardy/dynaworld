import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvImageEncoder, TokenAttentionBlock, build_mlp, flatten_hw_features
from .implicit_camera import GlobalCameraHead, PathCameraHead, compose_camera_with_se3_delta


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
