import torch
import torch.nn as nn
import torch.nn.functional as F


def build_mlp(in_dim, out_dim, hidden_dim=64, activation=nn.GELU):
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        activation(),
        nn.Linear(hidden_dim, out_dim),
    )


def flatten_hw_features(feature_map):
    batch_size, channels, height, width = feature_map.shape
    return feature_map.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)


def squeeze_batch_outputs(outputs):
    return tuple(tensor.squeeze(0) for tensor in outputs)


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
    def __init__(self, feat_dim, gaussians_per_token):
        super().__init__()
        self.gaussians_per_token = gaussians_per_token
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
        xyz_raw = self._reshape(self.xyz_head(tokens), 3)
        xyz = torch.cat(
            [
                torch.tanh(xyz_raw[..., :2]) * 1.5,
                torch.sigmoid(xyz_raw[..., 2:]) * 2.0 + 0.5,
            ],
            dim=-1,
        )

        scales = torch.exp(self._reshape(self.scale_head(tokens), 3)) * 0.05
        quats = F.normalize(self._reshape(self.rot_head(tokens), 4), p=2, dim=-1)
        opacities = torch.sigmoid(self._reshape(self.opacity_head(tokens), 1))
        rgbs = torch.sigmoid(self._reshape(self.rgb_head(tokens), 3))
        return xyz, scales, quats, opacities, rgbs


class TokenGSBackbone(nn.Module):
    def __init__(self, num_tokens=128, feat_dim=128, gaussians_per_token=4):
        super().__init__()
        self.num_tokens = num_tokens
        self.feat_dim = feat_dim
        self.gaussians_per_token = gaussians_per_token

        self.encoder = ConvImageEncoder(feat_dim)
        self.tokens = nn.Parameter(torch.randn(1, num_tokens, feat_dim))
        self.ray_proj = nn.Conv2d(6, feat_dim, kernel_size=1)
        self.token_block = TokenAttentionBlock(feat_dim=feat_dim)
        self.gaussian_heads = GaussianParameterHeads(
            feat_dim=feat_dim,
            gaussians_per_token=gaussians_per_token,
        )

    def encode_grounded_features(self, image, plucker_grid):
        feature_map = self.encoder(image)
        feature_h, feature_w = feature_map.shape[-2:]
        if plucker_grid.shape[-2:] != (feature_h, feature_w):
            plucker_grid = F.interpolate(plucker_grid, size=(feature_h, feature_w), mode="bilinear", align_corners=False)
        grounded_feature_map = feature_map + self.ray_proj(plucker_grid)
        return flatten_hw_features(grounded_feature_map)

    def predict_gaussians(self, image, plucker_grid, token_offsets=None):
        grounded_features = self.encode_grounded_features(image, plucker_grid)
        queries = self.tokens.expand(image.shape[0], -1, -1)
        if token_offsets is not None:
            queries = queries + token_offsets
        refined_tokens = self.token_block(queries, grounded_features)
        return self.gaussian_heads(refined_tokens)
