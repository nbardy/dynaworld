import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def inverse_sigmoid(value: float) -> float:
    value = min(max(float(value), 1.0e-6), 1.0 - 1.0e-6)
    return math.log(value / (1.0 - value))


def inverse_tanh_values(values: torch.Tensor) -> torch.Tensor:
    return torch.atanh(values.clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6))


def build_mlp(
    in_dim,
    out_dim,
    hidden_dim=64,
    hidden_layers=1,
    activation=nn.GELU,
    output_init_std=None,
):
    if hidden_layers < 0:
        raise ValueError(f"hidden_layers must be non-negative, got {hidden_layers}.")

    layers = []
    current_dim = in_dim
    for _ in range(hidden_layers):
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(activation())
        current_dim = hidden_dim

    output = nn.Linear(current_dim, out_dim)
    if output_init_std is not None:
        nn.init.normal_(output.weight, mean=0.0, std=float(output_init_std))
        nn.init.zeros_(output.bias)
    layers.append(output)
    return nn.Sequential(*layers)


def flatten_hw_features(feature_map):
    batch_size, channels, height, width = feature_map.shape
    return feature_map.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)


class ConvImageEncoder(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, feat_dim, 3, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, image):
        return self.net(image)


class TokenAttentionBlock(nn.Module):
    def __init__(self, feat_dim, num_heads=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=num_heads, batch_first=True)
        self.self_norm = nn.LayerNorm(feat_dim)
        self.cross_norm = nn.LayerNorm(feat_dim)

    def forward(self, queries, context):
        self_attended, _ = self.self_attn(query=queries, key=queries, value=queries, need_weights=False)
        queries = self.self_norm(queries + self_attended)
        cross_attended, _ = self.cross_attn(
            query=queries,
            key=context,
            value=context,
            need_weights=False,
        )
        return self.cross_norm(queries + cross_attended)


class GaussianParameterHeads(nn.Module):
    def __init__(
        self,
        feat_dim,
        gaussians_per_token,
        xy_extent=1.5,
        z_min=0.5,
        z_max=2.5,
        scale_init=0.05,
        scale_init_log_jitter=0.0,
        opacity_init=None,
        head_hidden_dim=64,
        head_hidden_layers=1,
        head_output_init_std=None,
        position_init_extent_coverage=0.0,
        rotation_init="random",
    ):
        super().__init__()
        if z_max <= z_min:
            raise ValueError(f"z_max must be greater than z_min, got z_min={z_min}, z_max={z_max}.")
        if scale_init <= 0:
            raise ValueError(f"scale_init must be positive, got {scale_init}.")
        if scale_init_log_jitter < 0:
            raise ValueError(f"scale_init_log_jitter must be non-negative, got {scale_init_log_jitter}.")
        if position_init_extent_coverage < 0:
            raise ValueError(
                f"position_init_extent_coverage must be non-negative, got {position_init_extent_coverage}."
            )
        if position_init_extent_coverage >= 1.0:
            raise ValueError(
                "position_init_extent_coverage must be < 1.0, "
                f"got {position_init_extent_coverage}."
            )
        if opacity_init is not None and not (0.0 < opacity_init < 1.0):
            raise ValueError(f"opacity_init must be in (0, 1), got {opacity_init}.")
        if rotation_init not in {"random", "identity"}:
            raise ValueError(f"rotation_init must be 'random' or 'identity', got {rotation_init!r}.")

        self.gaussians_per_token = gaussians_per_token
        self.xy_extent = float(xy_extent)
        self.z_min = float(z_min)
        self.z_extent = float(z_max) - float(z_min)
        self.scale_init = float(scale_init)
        mlp_kwargs = {
            "hidden_dim": head_hidden_dim,
            "hidden_layers": head_hidden_layers,
            "output_init_std": head_output_init_std,
        }
        self.xyz_head = build_mlp(feat_dim, gaussians_per_token * 3, **mlp_kwargs)
        self.scale_head = build_mlp(feat_dim, gaussians_per_token * 3, **mlp_kwargs)
        self.rot_head = build_mlp(feat_dim, gaussians_per_token * 4, **mlp_kwargs)
        self.opacity_head = build_mlp(feat_dim, gaussians_per_token, **mlp_kwargs)
        self.rgb_head = build_mlp(feat_dim, gaussians_per_token * 3, **mlp_kwargs)
        self._init_output_biases(
            scale_init_log_jitter=float(scale_init_log_jitter),
            opacity_init=opacity_init,
            position_init_extent_coverage=float(position_init_extent_coverage),
            rotation_init=rotation_init,
        )

    def _init_output_biases(
        self,
        scale_init_log_jitter: float,
        opacity_init: float | None,
        position_init_extent_coverage: float,
        rotation_init: str,
    ) -> None:
        if position_init_extent_coverage > 0:
            coverage = float(position_init_extent_coverage)
            z_margin = 0.5 * (1.0 - coverage)
            with torch.no_grad():
                xyz_bias = self.xyz_head[-1].bias.view(self.gaussians_per_token, 3)
                xy_target = torch.empty_like(xyz_bias[:, :2]).uniform_(-coverage, coverage)
                z_target = torch.empty_like(xyz_bias[:, 2:]).uniform_(z_margin, 1.0 - z_margin)
                xyz_bias[:, :2].copy_(inverse_tanh_values(xy_target))
                xyz_bias[:, 2:].copy_(torch.logit(z_target, eps=1.0e-6))
        if scale_init_log_jitter > 0:
            nn.init.uniform_(self.scale_head[-1].bias, -scale_init_log_jitter, scale_init_log_jitter)
        if rotation_init == "identity":
            with torch.no_grad():
                rot_bias = self.rot_head[-1].bias.view(self.gaussians_per_token, 4)
                rot_bias.zero_()
                rot_bias[:, 0] = 1.0
        if opacity_init is not None:
            nn.init.constant_(self.opacity_head[-1].bias, inverse_sigmoid(float(opacity_init)))

    def _reshape(self, values, channels):
        batch_size, token_count, _ = values.shape
        gaussian_count = token_count * self.gaussians_per_token
        return values.reshape(batch_size, gaussian_count, channels)

    def forward(self, tokens):
        xyz_raw = self._reshape(self.xyz_head(tokens), 3)
        xyz = torch.cat(
            [
                torch.tanh(xyz_raw[..., :2]) * self.xy_extent,
                torch.sigmoid(xyz_raw[..., 2:]) * self.z_extent + self.z_min,
            ],
            dim=-1,
        )

        scales = torch.exp(self._reshape(self.scale_head(tokens), 3)) * self.scale_init
        quats = F.normalize(self._reshape(self.rot_head(tokens), 4), p=2, dim=-1)
        opacities = torch.sigmoid(self._reshape(self.opacity_head(tokens), 1))
        rgbs = torch.sigmoid(self._reshape(self.rgb_head(tokens), 3))
        return xyz, scales, quats, opacities, rgbs


class TokenGSBackbone(nn.Module):
    def __init__(
        self,
        num_tokens=128,
        feat_dim=128,
        gaussians_per_token=4,
        xy_extent=1.5,
        z_min=0.5,
        z_max=2.5,
        scale_init=0.05,
        scale_init_log_jitter=0.0,
        opacity_init=None,
        token_init_std=1.0,
        head_hidden_dim=64,
        head_hidden_layers=1,
        head_output_init_std=None,
        position_init_extent_coverage=0.0,
        rotation_init="random",
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.feat_dim = feat_dim
        self.gaussians_per_token = gaussians_per_token

        self.encoder = ConvImageEncoder(feat_dim)
        self.tokens = nn.Parameter(torch.randn(1, num_tokens, feat_dim) * float(token_init_std))
        self.ray_proj = nn.Conv2d(6, feat_dim, kernel_size=1)
        self.token_block = TokenAttentionBlock(feat_dim=feat_dim)
        self.gaussian_heads = GaussianParameterHeads(
            feat_dim=feat_dim,
            gaussians_per_token=gaussians_per_token,
            xy_extent=xy_extent,
            z_min=z_min,
            z_max=z_max,
            scale_init=scale_init,
            scale_init_log_jitter=scale_init_log_jitter,
            opacity_init=opacity_init,
            head_hidden_dim=head_hidden_dim,
            head_hidden_layers=head_hidden_layers,
            head_output_init_std=head_output_init_std,
            position_init_extent_coverage=position_init_extent_coverage,
            rotation_init=rotation_init,
        )

    def encode_grounded_features(self, image, plucker_grid):
        feature_map = self.encoder(image)
        feature_h, feature_w = feature_map.shape[-2:]
        if plucker_grid.shape[-2:] != (feature_h, feature_w):
            plucker_grid = F.interpolate(
                plucker_grid, size=(feature_h, feature_w), mode="bilinear", align_corners=False
            )
        grounded_feature_map = feature_map + self.ray_proj(plucker_grid)
        return flatten_hw_features(grounded_feature_map)

    def predict_gaussians(self, image, plucker_grid, token_offsets=None):
        grounded_features = self.encode_grounded_features(image, plucker_grid)
        queries = self.tokens.expand(image.shape[0], -1, -1)
        if token_offsets is not None:
            queries = queries + token_offsets
        refined_tokens = self.token_block(queries, grounded_features)
        return self.gaussian_heads(refined_tokens)
