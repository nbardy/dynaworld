import torch
import torch.nn as nn

from runtime_types import CameraState, GaussianSequence

from .blocks import ConvImageEncoder, TokenAttentionBlock, build_mlp, flatten_hw_features
from .dynamic_token_gs_implicit_camera import CanonicalGaussianParameterHeads
from .implicit_camera import GlobalCameraHead, PathCameraHead, compose_camera_with_se3_delta


class DynamicTokenGSSeparatedImplicitCamera(nn.Module):
    """Implicit-camera TokenGS with camera heads branched before world-token attention.

    The image encoder produces an early pooled camera feature. Camera heads read
    that branch directly. Only splat/world tokens enter the token self-attention
    block, so camera/path tokens are not mixed through the world-token stack.
    """

    def __init__(self, num_tokens=128, feat_dim=128, gaussians_per_token=4, scene_extent=1.0):
        super().__init__()
        self.num_tokens = num_tokens
        self.total_tokens = num_tokens
        self.feat_dim = feat_dim
        self.gaussians_per_token = gaussians_per_token

        self.encoder = ConvImageEncoder(feat_dim)
        self.camera_norm = nn.LayerNorm(feat_dim)
        self.camera_proj = build_mlp(feat_dim, feat_dim)
        self.path_proj = build_mlp(feat_dim, feat_dim)
        self.splat_tokens = nn.Parameter(torch.randn(1, num_tokens, feat_dim))
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

    def _encode_image(self, img):
        feature_map = self.encoder(img)
        context = flatten_hw_features(feature_map)
        early_camera_token = self.camera_norm(context.mean(dim=1))
        return context, early_camera_token

    def _camera_tokens(self, early_camera_token, frame_times):
        if frame_times is None:
            frame_times = torch.zeros((early_camera_token.shape[0], 1), device=early_camera_token.device)
        frame_times = frame_times.to(device=early_camera_token.device, dtype=early_camera_token.dtype).reshape(
            early_camera_token.shape[0], 1
        )
        time_offset = self.time_proj(frame_times)
        global_camera_features = self.camera_proj(early_camera_token)
        path_features = self.path_proj(early_camera_token + time_offset)
        return global_camera_features, path_features, time_offset

    def _refine_splat_tokens(self, context, time_offset):
        queries = self.splat_tokens.expand(context.shape[0], -1, -1) + time_offset.unsqueeze(1)
        return self.token_block(queries, context)

    @torch.no_grad()
    def infer_global_camera_token(self, img, frame_times=None, batch_size=4):
        tokens = []
        for start in range(0, img.shape[0], batch_size):
            end = min(start + batch_size, img.shape[0])
            context, early_camera_token = self._encode_image(img[start:end])
            del context
            global_camera_features, _path_features, _time_offset = self._camera_tokens(
                early_camera_token,
                None if frame_times is None else frame_times[start:end],
            )
            tokens.append(global_camera_features)
        return torch.cat(tokens, dim=0).mean(dim=0)

    def forward(self, img, frame_times=None, global_camera_token=None):
        context, early_camera_token = self._encode_image(img)
        global_camera_features, path_features, time_offset = self._camera_tokens(early_camera_token, frame_times)

        if global_camera_token is None:
            global_camera_token = global_camera_features.mean(dim=0)
        else:
            global_camera_token = global_camera_token.to(device=img.device, dtype=img.dtype)

        splat_tokens = self._refine_splat_tokens(context, time_offset)
        xyz, scales, quats, opacities, rgbs = self.gaussian_heads(splat_tokens)

        base_camera, camera_state = self.global_camera_head(global_camera_token, image_size=img.shape[-1])
        rotation_delta, translation_delta, path_residuals = self.path_camera_head(
            path_features, base_radius=camera_state["radius"]
        )
        cameras = compose_camera_with_se3_delta(base_camera, rotation_delta, translation_delta)
        camera_state["rotation_delta"] = rotation_delta
        camera_state["translation_delta"] = translation_delta
        camera_state["path_residuals"] = path_residuals
        return GaussianSequence(
            xyz=xyz,
            scales=scales,
            quats=quats,
            opacities=opacities,
            rgbs=rgbs,
            cameras=tuple(cameras),
            camera_state=CameraState.from_mapping(camera_state),
        )
