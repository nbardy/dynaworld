import math

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, value):
        rms = value.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return value * rms * self.weight


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, value):
        return self.net(value)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = RMSNorm(dim)
        self.ffn = FeedForward(dim, mlp_ratio=mlp_ratio)

    def forward(self, tokens):
        attn_input = self.norm1(tokens)
        attn_output, _ = self.self_attn(attn_input, attn_input, attn_input, need_weights=False)
        tokens = tokens + attn_output
        return tokens + self.ffn(self.norm2(tokens))


class QueryCrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.query_self_norm = RMSNorm(dim)
        self.query_self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.query_cross_norm = RMSNorm(dim)
        self.memory_norm = RMSNorm(dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = FeedForward(dim, mlp_ratio=mlp_ratio)

    def forward(self, queries, memory):
        self_input = self.query_self_norm(queries)
        self_output, _ = self.query_self_attn(self_input, self_input, self_input, need_weights=False)
        queries = queries + self_output
        cross_query = self.query_cross_norm(queries)
        cross_memory = self.memory_norm(memory)
        cross_output, _ = self.cross_attn(cross_query, cross_memory, cross_memory, need_weights=False)
        queries = queries + cross_output
        return queries + self.ffn(self.ffn_norm(queries))


def flatten_video_tokens(tokens_3d):
    batch_size, channels, depth, height, width = tokens_3d.shape
    return tokens_3d.permute(0, 2, 3, 4, 1).reshape(batch_size, depth * height * width, channels)


def unflatten_video_tokens(tokens, grid_shape):
    depth, height, width = grid_shape
    batch_size, _, channels = tokens.shape
    return tokens.reshape(batch_size, depth, height, width, channels).permute(0, 4, 1, 2, 3)


class VideoPatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, dim=128, tubelet_size=(4, 16, 16)):
        super().__init__()
        self.tubelet_size = tubelet_size
        self.proj = nn.Conv3d(
            in_channels=in_channels,
            out_channels=dim,
            kernel_size=tubelet_size,
            stride=tubelet_size,
        )

    def forward(self, video):
        video_3d = video.permute(0, 2, 1, 3, 4)
        tokens_3d = self.proj(video_3d)
        return tokens_3d, tokens_3d.shape[-3:]


class VideoTokenDownsample(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv3d(in_dim, in_dim, kernel_size=3, stride=(1, 2, 2), padding=1, groups=in_dim),
            nn.GELU(),
            nn.Conv3d(in_dim, out_dim, kernel_size=1),
        )

    def forward(self, tokens_3d):
        output = self.downsample(tokens_3d)
        return output, output.shape[-3:]


class VideoTokenUpsample(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_dim, out_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.GELU(),
            nn.Conv3d(out_dim, out_dim, kernel_size=3, padding=1, groups=out_dim),
            nn.GELU(),
            nn.Conv3d(out_dim, out_dim, kernel_size=1),
        )

    def forward(self, tokens_3d, target_shape):
        output = self.upsample(tokens_3d)
        target_depth, target_height, target_width = target_shape
        return output[:, :, :target_depth, :target_height, :target_width]


class VideoEncoder(nn.Module):
    def __init__(
        self,
        clip_length,
        image_size,
        dim=128,
        bottleneck_dim=256,
        num_heads=8,
        mlp_ratio=4.0,
        tubelet_size=(4, 16, 16),
        encoder_self_attn_layers=1,
        bottleneck_self_attn_layers=4,
    ):
        super().__init__()
        self.clip_length = clip_length
        self.image_size = image_size
        self.dim = dim
        self.bottleneck_dim = bottleneck_dim
        self.patch_embed = VideoPatchEmbedding(dim=dim, tubelet_size=tubelet_size)
        self.stage1_blocks = nn.ModuleList(
            [
                TransformerBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
                for _ in range(encoder_self_attn_layers)
            ]
        )
        self.downsample = VideoTokenDownsample(in_dim=dim, out_dim=bottleneck_dim)
        self.bottleneck_blocks = nn.ModuleList(
            [
                TransformerBlock(dim=bottleneck_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
                for _ in range(bottleneck_self_attn_layers)
            ]
        )
        self.upsample = VideoTokenUpsample(in_dim=bottleneck_dim, out_dim=dim)
        stage1_grid = self._compute_stage1_grid()
        stage2_grid = (stage1_grid[0], max(1, stage1_grid[1] // 2), max(1, stage1_grid[2] // 2))
        self.stage1_pos = nn.Parameter(torch.randn(1, stage1_grid[0] * stage1_grid[1] * stage1_grid[2], dim) * 0.02)
        self.stage2_pos = nn.Parameter(
            torch.randn(1, stage2_grid[0] * stage2_grid[1] * stage2_grid[2], bottleneck_dim) * 0.02
        )

    def _compute_stage1_grid(self):
        tubelet_t, tubelet_h, tubelet_w = self.patch_embed.tubelet_size
        if self.clip_length % tubelet_t != 0:
            raise ValueError(f"clip_length={self.clip_length} must be divisible by tubelet temporal size {tubelet_t}")
        if self.image_size % tubelet_h != 0 or self.image_size % tubelet_w != 0:
            raise ValueError(
                f"image_size={self.image_size} must be divisible by tubelet spatial size ({tubelet_h}, {tubelet_w})"
            )
        return self.clip_length // tubelet_t, self.image_size // tubelet_h, self.image_size // tubelet_w

    def forward(self, video):
        stage1_tokens_3d, stage1_shape = self.patch_embed(video)
        stage1_tokens = flatten_video_tokens(stage1_tokens_3d) + self.stage1_pos
        for block in self.stage1_blocks:
            stage1_tokens = block(stage1_tokens)
        stage1_tokens_3d = unflatten_video_tokens(stage1_tokens, stage1_shape)

        stage2_tokens_3d, stage2_shape = self.downsample(stage1_tokens_3d)
        stage2_tokens = flatten_video_tokens(stage2_tokens_3d) + self.stage2_pos
        for block in self.bottleneck_blocks:
            stage2_tokens = block(stage2_tokens)
        stage2_tokens_3d = unflatten_video_tokens(stage2_tokens, stage2_shape)

        upsampled_tokens_3d = self.upsample(stage2_tokens_3d, target_shape=stage1_shape)
        merged_tokens_3d = upsampled_tokens_3d + stage1_tokens_3d
        return flatten_video_tokens(merged_tokens_3d)


class LearnedQueryTokenBank(nn.Module):
    def __init__(self, total_tokens, dim):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(1, total_tokens, dim) * 0.02)

    def forward(self, batch_size):
        return self.tokens.expand(batch_size, -1, -1)


class TimeConditionedGaussianHeads(nn.Module):
    def __init__(self, feat_dim, gaussians_per_token, scene_extent=1.0):
        super().__init__()
        self.gaussians_per_token = gaussians_per_token
        self.scene_extent = scene_extent
        self.xyz_head = build_zero_init_head(feat_dim, gaussians_per_token * 3)
        self.scale_head = build_zero_init_head(feat_dim, gaussians_per_token * 3)
        self.rot_head = build_zero_init_head(feat_dim, gaussians_per_token * 4)
        self.opacity_head = build_zero_init_head(feat_dim, gaussians_per_token)
        self.rgb_head = build_zero_init_head(feat_dim, gaussians_per_token * 3)

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


class DynamicVideoTokenGSImplicitCamera(nn.Module):
    def __init__(
        self,
        clip_length=4,
        image_size=384,
        num_tokens=8,
        feat_dim=128,
        bottleneck_dim=256,
        num_heads=8,
        mlp_ratio=4.0,
        gaussians_per_token=64,
        scene_extent=1.0,
        tubelet_size=(4, 16, 16),
        encoder_self_attn_layers=1,
        bottleneck_self_attn_layers=4,
        cross_attn_layers=1,
        base_fov_degrees=60.0,
        base_radius=3.0,
        max_fov_delta_degrees=15.0,
        max_radius_scale=1.5,
        max_rotation_degrees=5.0,
        max_translation_ratio=0.2,
    ):
        super().__init__()
        self.clip_length = clip_length
        self.image_size = image_size
        self.num_tokens = num_tokens
        self.total_tokens = num_tokens + 2
        self.feat_dim = feat_dim
        self.gaussians_per_token = gaussians_per_token
        self.video_encoder = VideoEncoder(
            clip_length=clip_length,
            image_size=image_size,
            dim=feat_dim,
            bottleneck_dim=bottleneck_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            tubelet_size=tubelet_size,
            encoder_self_attn_layers=encoder_self_attn_layers,
            bottleneck_self_attn_layers=bottleneck_self_attn_layers,
        )
        self.query_tokens = LearnedQueryTokenBank(total_tokens=self.total_tokens, dim=feat_dim)
        self.query_decoder_blocks = nn.ModuleList(
            [
                QueryCrossAttentionBlock(dim=feat_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
                for _ in range(cross_attn_layers)
            ]
        )
        self.gaussian_heads = TimeConditionedGaussianHeads(
            feat_dim=feat_dim,
            gaussians_per_token=gaussians_per_token,
            scene_extent=scene_extent,
        )
        self.global_camera_head = GlobalCameraHead(
            feat_dim=feat_dim,
            base_fov_degrees=base_fov_degrees,
            base_radius=base_radius,
            max_fov_delta_degrees=max_fov_delta_degrees,
            max_radius_scale=max_radius_scale,
        )
        self.path_camera_head = PathCameraHead(
            feat_dim=feat_dim,
            max_rotation_degrees=max_rotation_degrees,
            max_translation_ratio=max_translation_ratio,
        )
        self.time_proj = nn.Sequential(
            nn.Linear(1, feat_dim),
            nn.SiLU(),
            nn.Linear(feat_dim, feat_dim),
        )

    def encode_queries(self, video):
        batch_size = video.shape[0]
        video_tokens = self.video_encoder(video)
        queries = self.query_tokens(batch_size)
        for block in self.query_decoder_blocks:
            queries = block(queries, video_tokens)
        return queries

    def _decode_single_time(self, refined_queries, decode_time):
        time_offset = self.time_proj(decode_time.reshape(1, 1))
        global_camera_token = refined_queries[:, 0, :]
        path_token = refined_queries[:, 1, :] + time_offset
        gs_tokens = refined_queries[:, 2:, :] + time_offset.unsqueeze(1)

        xyz, scales, quats, opacities, rgbs = self.gaussian_heads(gs_tokens)
        base_camera, base_state = self.global_camera_head(global_camera_token.squeeze(0), image_size=self.image_size)
        rotation_delta, translation_delta, path_residuals = self.path_camera_head(
            path_token, base_radius=base_state["radius"]
        )
        camera = compose_camera_with_se3_delta(base_camera, rotation_delta, translation_delta)[0]
        camera_state = {
            "fov_degrees": base_state["fov_degrees"],
            "radius": base_state["radius"],
            "global_residuals": base_state["global_residuals"],
            "rotation_delta": rotation_delta,
            "translation_delta": translation_delta,
            "path_residuals": path_residuals,
        }
        return (
            xyz.squeeze(0),
            scales.squeeze(0),
            quats.squeeze(0),
            opacities.squeeze(0),
            rgbs.squeeze(0),
            camera,
            camera_state,
        )

    def forward(self, video, decode_times):
        if video.ndim != 5:
            raise ValueError(f"Expected video of shape (B, T, C, H, W), got {tuple(video.shape)}")
        if decode_times.ndim != 2:
            raise ValueError(f"Expected decode_times of shape (B, T), got {tuple(decode_times.shape)}")
        if video.shape[0] != 1:
            raise ValueError("DynamicVideoTokenGSImplicitCamera currently expects clip batch size 1.")
        if video.shape[1] != decode_times.shape[1]:
            raise ValueError("decode_times must have one value per frame in the clip.")

        refined_queries = self.encode_queries(video)
        decoded = [
            self._decode_single_time(refined_queries, decode_times[0, index]) for index in range(decode_times.shape[1])
        ]

        xyz = torch.stack([item[0] for item in decoded], dim=0)
        scales = torch.stack([item[1] for item in decoded], dim=0)
        quats = torch.stack([item[2] for item in decoded], dim=0)
        opacities = torch.stack([item[3] for item in decoded], dim=0)
        rgbs = torch.stack([item[4] for item in decoded], dim=0)
        cameras = [item[5] for item in decoded]
        camera_state = {
            "fov_degrees": torch.stack([item[6]["fov_degrees"] for item in decoded]).mean(),
            "radius": torch.stack([item[6]["radius"] for item in decoded]).mean(),
            "global_residuals": torch.stack([item[6]["global_residuals"] for item in decoded]).mean(dim=0),
            "rotation_delta": torch.cat([item[6]["rotation_delta"] for item in decoded], dim=0),
            "translation_delta": torch.cat([item[6]["translation_delta"] for item in decoded], dim=0),
            "path_residuals": torch.cat([item[6]["path_residuals"] for item in decoded], dim=0),
        }
        return xyz, scales, quats, opacities, rgbs, cameras, camera_state
