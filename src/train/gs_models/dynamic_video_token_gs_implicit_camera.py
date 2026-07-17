import math
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from camera import build_plucker_ray_grid, build_plucker_ray_grid_batch
from runtime_types import CameraState, GaussianSequence

from .blocks import (
    GaussianParameterHeads,
    build_mlp,
    flatten_hw_features,
    inverse_sigmoid,
    inverse_sigmoid_values,
    inverse_tanh_values,
)
from .implicit_camera import (
    PathCameraHead,
    TimeConditionedOpticalAxisCameraHead,
    TimeConditionedPathCameraHead,
    build_global_camera_head,
    compose_camera_with_se3_delta,
)
from .time_conditioning import SinusoidalTimeConditioner, build_time_projector


VJEPA_TORCHHUB_CHECKPOINTS = {
    "vjepa2_1_vit_base_384": (
        "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt",
        "ema_encoder",
    ),
    "vjepa2_1_vit_large_384": (
        "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt",
        "ema_encoder",
    ),
    "vjepa2_1_vit_giant_384": (
        "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitg_384.pt",
        "target_encoder",
    ),
    "vjepa2_1_vit_gigantic_384": (
        "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitG_384.pt",
        "target_encoder",
    ),
}


def _clean_torchhub_backbone_state_dict(state_dict):
    return {
        key.replace("module.", "").replace("backbone.", ""): value
        for key, value in state_dict.items()
    }


def _load_torchhub_encoder_checkpoint(encoder, model_id, checkpoint_url=None, checkpoint_key=None):
    default = VJEPA_TORCHHUB_CHECKPOINTS.get(str(model_id))
    if checkpoint_url is None:
        checkpoint_url = default[0] if default is not None else None
    if checkpoint_key is None:
        checkpoint_key = default[1] if default is not None else "target_encoder"
    if checkpoint_url is None:
        return False

    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")
    if checkpoint_key not in state_dict:
        available = ", ".join(sorted(state_dict.keys()))
        raise KeyError(
            f"Checkpoint for {model_id!r} does not contain encoder key {checkpoint_key!r}. "
            f"Available keys: {available}"
        )
    encoder_state_dict = _clean_torchhub_backbone_state_dict(state_dict[checkpoint_key])
    encoder.load_state_dict(encoder_state_dict, strict=True)
    return True


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


_MPS_MANUAL_CROSS_ATTN_TOKEN_LIMIT = 32768


def _manual_batch_first_mha(mha, query, key, value):
    if not getattr(mha, "batch_first", False):
        raise ValueError("Manual MHA fallback expects batch_first=True.")
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("Manual MHA fallback expects rank-3 query/key/value tensors.")
    if mha.in_proj_weight is None:
        raise ValueError("Manual MHA fallback expects a combined qkv projection.")
    if getattr(mha, "bias_k", None) is not None or getattr(mha, "bias_v", None) is not None:
        raise ValueError("Manual MHA fallback does not support bias_k/bias_v.")
    if getattr(mha, "add_zero_attn", False):
        raise ValueError("Manual MHA fallback does not support add_zero_attn.")

    embed_dim = int(mha.embed_dim)
    num_heads = int(mha.num_heads)
    head_dim = embed_dim // num_heads
    in_proj_bias = mha.in_proj_bias
    q_bias = k_bias = v_bias = None
    if in_proj_bias is not None:
        q_bias = in_proj_bias[:embed_dim]
        k_bias = in_proj_bias[embed_dim : 2 * embed_dim]
        v_bias = in_proj_bias[2 * embed_dim :]

    q_proj = F.linear(query, mha.in_proj_weight[:embed_dim], q_bias)
    k_proj = F.linear(key, mha.in_proj_weight[embed_dim : 2 * embed_dim], k_bias)
    v_proj = F.linear(value, mha.in_proj_weight[2 * embed_dim :], v_bias)

    batch_size, query_tokens, _ = q_proj.shape
    key_tokens = k_proj.shape[1]
    q_proj = q_proj.view(batch_size, query_tokens, num_heads, head_dim).transpose(1, 2)
    k_proj = k_proj.view(batch_size, key_tokens, num_heads, head_dim).transpose(1, 2)
    v_proj = v_proj.view(batch_size, key_tokens, num_heads, head_dim).transpose(1, 2)

    scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) * (head_dim**-0.5)
    weights = torch.softmax(scores, dim=-1)
    if float(mha.dropout) > 0.0:
        weights = F.dropout(weights, p=float(mha.dropout), training=mha.training)
    output = torch.matmul(weights, v_proj).transpose(1, 2).reshape(batch_size, query_tokens, embed_dim)
    return mha.out_proj(output)


def _needs_mps_manual_cross_attn(query, memory):
    return query.device.type == "mps" and int(memory.shape[1]) > _MPS_MANUAL_CROSS_ATTN_TOKEN_LIMIT


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
        if _needs_mps_manual_cross_attn(cross_query, cross_memory):
            cross_output = _manual_batch_first_mha(self.cross_attn, cross_query, cross_memory, cross_memory)
        else:
            cross_output, _ = self.cross_attn(cross_query, cross_memory, cross_memory, need_weights=False)
        queries = queries + cross_output
        return queries + self.ffn(self.ffn_norm(queries))


class PluckerRayTokenConditioner(nn.Module):
    def __init__(self, feat_dim, num_heads=8, mlp_ratio=4.0, ray_grid_size=16):
        super().__init__()
        self.ray_grid_size = int(ray_grid_size)
        if self.ray_grid_size < 1:
            raise ValueError(f"ray_grid_size must be >= 1, got {ray_grid_size}.")
        self.ray_proj = nn.Sequential(
            nn.Conv2d(6, feat_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=feat_dim),
            nn.GELU(),
            nn.Conv2d(feat_dim, feat_dim, kernel_size=1),
        )
        self.ray_cross_attn = QueryCrossAttentionBlock(dim=feat_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)

    def _build_plucker_grid(self, cameras, image_size, device, dtype):
        if isinstance(cameras, (list, tuple)):
            plucker = build_plucker_ray_grid_batch(
                list(cameras),
                image_size=image_size,
                device=device,
                channels_first=True,
            )
        else:
            plucker = build_plucker_ray_grid(
                cameras,
                image_size=image_size,
                device=device,
                channels_first=True,
            )
        return plucker.to(device=device, dtype=dtype)

    def forward(self, tokens, cameras, image_size):
        plucker_grid = self._build_plucker_grid(
            cameras,
            image_size=image_size,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        ray_features = self.ray_proj(plucker_grid)
        if ray_features.shape[-2:] != (self.ray_grid_size, self.ray_grid_size):
            ray_features = F.adaptive_avg_pool2d(ray_features, (self.ray_grid_size, self.ray_grid_size))
        ray_context = flatten_hw_features(ray_features)
        return self.ray_cross_attn(tokens, ray_context)


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
        self.stage1_time_proj = build_time_projector(1, dim)
        self.stage1_span_proj = build_time_projector(2, dim)
        self.stage2_time_proj = build_time_projector(1, bottleneck_dim)
        self.stage2_span_proj = build_time_projector(2, bottleneck_dim)

    def _compute_stage1_grid(self):
        tubelet_t, tubelet_h, tubelet_w = self.patch_embed.tubelet_size
        if self.clip_length % tubelet_t != 0:
            raise ValueError(f"clip_length={self.clip_length} must be divisible by tubelet temporal size {tubelet_t}")
        if self.image_size % tubelet_h != 0 or self.image_size % tubelet_w != 0:
            raise ValueError(
                f"image_size={self.image_size} must be divisible by tubelet spatial size ({tubelet_h}, {tubelet_w})"
            )
        return self.clip_length // tubelet_t, self.image_size // tubelet_h, self.image_size // tubelet_w

    def _default_frame_times(self, batch_size, device, dtype):
        if self.clip_length < 2:
            return torch.zeros((batch_size, self.clip_length), device=device, dtype=dtype)
        return (
            torch.linspace(0.0, 1.0, self.clip_length, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1)
        )

    def _normalize_frame_times(self, frame_times, batch_size, device, dtype):
        if frame_times is None:
            return self._default_frame_times(batch_size, device, dtype)
        frame_times = frame_times.to(device=device, dtype=dtype)
        if frame_times.ndim == 3 and frame_times.shape[-1] == 1:
            frame_times = frame_times.squeeze(-1)
        if frame_times.shape != (batch_size, self.clip_length):
            raise ValueError(
                f"Expected frame_times shape {(batch_size, self.clip_length)}, got {tuple(frame_times.shape)}"
            )
        return frame_times

    def _tubelet_times(self, frame_times, depth):
        tubelet_t = self.patch_embed.tubelet_size[0]
        expected_frames = depth * tubelet_t
        if frame_times.shape[1] != expected_frames:
            raise ValueError(f"Expected {expected_frames} frame times for {depth} tubelets, got {frame_times.shape[1]}")
        return frame_times.reshape(frame_times.shape[0], depth, tubelet_t).mean(dim=-1)

    def _time_tokens(self, frame_times, grid_shape, time_proj, span_proj):
        depth, height, width = grid_shape
        tubelet_times = self._tubelet_times(frame_times, depth)
        time_tokens = time_proj(tubelet_times.unsqueeze(-1))
        time_tokens = time_tokens[:, :, None, None, :].expand(-1, -1, height, width, -1)
        time_tokens = time_tokens.reshape(frame_times.shape[0], depth * height * width, -1)
        span = torch.stack([frame_times[:, 0], frame_times[:, -1]], dim=-1)
        return time_tokens + span_proj(span).unsqueeze(1)

    def forward(self, video, frame_times=None):
        frame_times = self._normalize_frame_times(
            frame_times,
            batch_size=video.shape[0],
            device=video.device,
            dtype=video.dtype,
        )
        stage1_tokens_3d, stage1_shape = self.patch_embed(video)
        stage1_tokens = (
            flatten_video_tokens(stage1_tokens_3d)
            + self.stage1_pos
            + self._time_tokens(frame_times, stage1_shape, self.stage1_time_proj, self.stage1_span_proj)
        )
        for block in self.stage1_blocks:
            stage1_tokens = block(stage1_tokens)
        stage1_tokens_3d = unflatten_video_tokens(stage1_tokens, stage1_shape)

        stage2_tokens_3d, stage2_shape = self.downsample(stage1_tokens_3d)
        stage2_tokens = (
            flatten_video_tokens(stage2_tokens_3d)
            + self.stage2_pos
            + self._time_tokens(frame_times, stage2_shape, self.stage2_time_proj, self.stage2_span_proj)
        )
        for block in self.bottleneck_blocks:
            stage2_tokens = block(stage2_tokens)
        stage2_tokens_3d = unflatten_video_tokens(stage2_tokens, stage2_shape)

        upsampled_tokens_3d = self.upsample(stage2_tokens_3d, target_shape=stage1_shape)
        merged_tokens_3d = upsampled_tokens_3d + stage1_tokens_3d
        return flatten_video_tokens(merged_tokens_3d)


def _first_parameter_dtype(module):
    for parameter in module.parameters():
        return parameter.dtype
    return torch.float32


def _first_parameter_device(module):
    for parameter in module.parameters():
        return parameter.device
    return torch.device("cpu")


def _resolve_vjepa_dtype(dtype_name):
    if dtype_name is None:
        return None
    dtype_name = str(dtype_name).lower()
    if dtype_name in {"none", "null"}:
        return None
    if dtype_name == "auto":
        return torch.float16 if torch.cuda.is_available() else torch.float32
    dtype_by_name = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_name not in dtype_by_name:
        known = ", ".join(sorted(set(dtype_by_name) | {"auto", "none"}))
        raise ValueError(f"Unknown vjepa_dtype={dtype_name!r}. Expected one of: {known}.")
    return dtype_by_name[dtype_name]


def _infer_encoder_feature_dim(module):
    for attr_name in ("hidden_size", "embed_dim", "num_features"):
        value = getattr(module, attr_name, None)
        if value is not None:
            return int(value)
    config = getattr(module, "config", None)
    if config is not None:
        for attr_name in ("hidden_size", "embed_dim", "num_features"):
            value = getattr(config, attr_name, None)
            if value is not None:
                return int(value)
    return None


def _as_feature_tokens(features, feature_dim=None):
    if hasattr(features, "last_hidden_state"):
        features = features.last_hidden_state
    if isinstance(features, (tuple, list)):
        features = features[0]
    if features.ndim == 3:
        return features
    if features.ndim == 5:
        if feature_dim is not None and features.shape[1] == feature_dim:
            return flatten_video_tokens(features)
        if feature_dim is not None and features.shape[-1] == feature_dim:
            batch_size, depth, height, width, channels = features.shape
            return features.reshape(batch_size, depth * height * width, channels)
    raise ValueError(f"Expected V-JEPA features with rank 3 or 5, got shape {tuple(features.shape)}.")


def _safe_module_key(name):
    return str(name).replace(".", "__").replace("-", "_")


def _flatten_precomputed_feature(value, channels):
    if not torch.is_tensor(value):
        raise TypeError(f"Expected cached video feature tensor, got {type(value).__name__}.")
    if value.ndim == 2:
        if value.shape[-1] != channels:
            raise ValueError(f"Expected feature channels={channels}, got shape {tuple(value.shape)}.")
        return value.unsqueeze(0)
    if value.ndim == 3:
        if value.shape[-1] != channels:
            raise ValueError(f"Expected feature channels={channels}, got shape {tuple(value.shape)}.")
        return value
    if value.ndim == 4:
        if value.shape[1] == channels:
            return flatten_hw_features(value)
        if value.shape[-1] == channels:
            batch_size, height, width, _ = value.shape
            return value.reshape(batch_size, height * width, channels)
    if value.ndim == 5:
        if value.shape[1] == channels:
            return flatten_video_tokens(value)
        if value.shape[-1] == channels:
            batch_size, depth, height, width, _ = value.shape
            return value.reshape(batch_size, depth * height * width, channels)
    raise ValueError(
        f"Could not interpret cached feature shape {tuple(value.shape)} with channels={channels}. "
        "Expected [B,N,C], [B,C,H,W], [B,H,W,C], [B,C,T,H,W], or [B,T,H,W,C]."
    )


def _stride_precomputed_tokens(tokens, stride):
    stride = int(stride)
    if stride <= 1:
        return tokens
    return tokens[:, ::stride, :].contiguous()


def _resolve_feature_output_dtype(dtype_name):
    if dtype_name is None:
        return None
    dtype_name = str(dtype_name).lower()
    if dtype_name in {"none", "null"}:
        return None
    dtype_by_name = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if dtype_name not in dtype_by_name:
        known = ", ".join(sorted(set(dtype_by_name) | {"none"}))
        raise ValueError(f"Unknown video_feature_output_dtype={dtype_name!r}. Expected one of: {known}.")
    return dtype_by_name[dtype_name]


class PrecomputedVideoFeatureAdapter(nn.Module):
    """Project cached native feature tensors into the query decoder memory.

    The adapter deliberately keeps each layer at its native token/grid
    resolution. It flattens native maps into memory tokens and lets the Gaussian
    query decoder attend across the resulting multi-scale feature set.
    """

    def __init__(self, output_dim, feature_channels, feature_layers=None, token_stride=1, output_dtype=None):
        super().__init__()
        if not isinstance(feature_channels, Mapping) or not feature_channels:
            raise ValueError("video_feature_channels must be a non-empty mapping for precomputed features.")
        self.token_stride = int(token_stride)
        if self.token_stride < 1:
            raise ValueError(f"token_stride must be >= 1, got {token_stride}.")
        self.output_dtype = _resolve_feature_output_dtype(output_dtype)
        self.feature_channels = {str(name): int(channels) for name, channels in feature_channels.items()}
        self.feature_layers = tuple(str(name) for name in (feature_layers or self.feature_channels.keys()))
        missing = [name for name in self.feature_layers if name not in self.feature_channels]
        if missing:
            raise ValueError(
                f"video_feature_layers contains layer(s) without channel counts: {missing}. "
                f"Known layers: {sorted(self.feature_channels)}"
            )

        self.safe_names = {name: _safe_module_key(name) for name in self.feature_layers}
        self.input_norms = nn.ModuleDict()
        self.output_projs = nn.ModuleDict()
        self.layer_embeddings = nn.ParameterDict()
        for name in self.feature_layers:
            safe_name = self.safe_names[name]
            channels = self.feature_channels[name]
            if channels <= 0:
                raise ValueError(f"Feature channel count for {name!r} must be positive, got {channels}.")
            self.input_norms[safe_name] = nn.LayerNorm(channels)
            self.output_projs[safe_name] = nn.Linear(channels, output_dim)
            self.layer_embeddings[safe_name] = nn.Parameter(torch.zeros(output_dim))

    def forward(self, feature_payload, frame_times=None):
        del frame_times
        if torch.is_tensor(feature_payload):
            if len(self.feature_layers) != 1:
                raise ValueError(
                    "Tensor feature payloads require exactly one configured video_feature_layer; "
                    f"got {self.feature_layers}."
                )
            feature_payload = {self.feature_layers[0]: feature_payload}
        if not isinstance(feature_payload, Mapping):
            raise TypeError(
                "Precomputed video features must be a tensor or a mapping of layer name to tensor, "
                f"got {type(feature_payload).__name__}."
            )

        projected = []
        for name in self.feature_layers:
            if name not in feature_payload:
                raise KeyError(f"Missing cached video feature layer {name!r}.")
            safe_name = self.safe_names[name]
            channels = self.feature_channels[name]
            tokens = _flatten_precomputed_feature(feature_payload[name], channels)
            proj = self.output_projs[safe_name]
            tokens = tokens.to(device=proj.weight.device, dtype=proj.weight.dtype)
            tokens = _stride_precomputed_tokens(tokens, self.token_stride)
            tokens = proj(self.input_norms[safe_name](tokens))
            tokens = tokens + self.layer_embeddings[safe_name].view(1, 1, -1)
            if self.output_dtype is not None:
                tokens = tokens.to(dtype=self.output_dtype)
            projected.append(tokens)

        return torch.cat(projected, dim=1)


class HuggingFaceVJEPAVideoEncoder(nn.Module):
    def __init__(
        self,
        output_dim,
        model_id="facebook/vjepa2-vitl-fpc64-256",
        feature_dim=None,
        freeze=True,
        attn_implementation="sdpa",
        dtype="auto",
    ):
        super().__init__()
        try:
            from transformers import AutoModel, AutoVideoProcessor
        except ImportError as exc:
            raise ImportError(
                "video_encoder_backend='vjepa_hf' requires transformers. "
                "Run with `uv run --group vjepa-hf ...` or install the matching optional dependency."
            ) from exc

        self.model_id = model_id
        self.freeze = bool(freeze)
        self.processor = AutoVideoProcessor.from_pretrained(model_id)
        model_kwargs = {}
        load_dtype = _resolve_vjepa_dtype(dtype)
        if load_dtype is not None:
            model_kwargs["torch_dtype"] = load_dtype
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        try:
            self.encoder = AutoModel.from_pretrained(model_id, **model_kwargs)
        except TypeError:
            if "torch_dtype" not in model_kwargs:
                raise
            model_kwargs["dtype"] = model_kwargs.pop("torch_dtype")
            self.encoder = AutoModel.from_pretrained(model_id, **model_kwargs)

        self.encoder.eval()
        if self.freeze:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)

        hidden_dim = int(feature_dim or _infer_encoder_feature_dim(self.encoder) or 0)
        if hidden_dim <= 0:
            raise ValueError(
                f"Could not infer hidden size for V-JEPA model {model_id!r}; set model.vjepa_feature_dim."
            )
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.encoder.eval()
        return self

    @staticmethod
    def _processor_video(video):
        video = video.detach().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).cpu()
        if video.shape[0] == 1:
            return video[0]
        return [video[index] for index in range(video.shape[0])]

    def _processor_inputs(self, video):
        processed = self.processor(self._processor_video(video), return_tensors="pt")
        model_device = _first_parameter_device(self.encoder)
        model_dtype = _first_parameter_dtype(self.encoder)
        inputs = {}
        for key, value in processed.items():
            value = value.to(model_device)
            if torch.is_floating_point(value):
                value = value.to(dtype=model_dtype)
            inputs[key] = value
        return inputs

    def _extract_features(self, inputs):
        if hasattr(self.encoder, "get_vision_features"):
            try:
                return self.encoder.get_vision_features(**inputs)
            except TypeError:
                return self.encoder.get_vision_features(inputs["pixel_values_videos"])
        try:
            outputs = self.encoder(**inputs, skip_predictor=True)
        except TypeError:
            outputs = self.encoder(**inputs)
        return outputs.last_hidden_state

    def forward(self, video, frame_times=None):
        del frame_times
        inputs = self._processor_inputs(video)
        context = torch.no_grad() if self.freeze else nullcontext()
        with context:
            features = self._extract_features(inputs)
        features = _as_feature_tokens(features, feature_dim=self.input_norm.normalized_shape[0])
        features = features.to(device=self.output_proj.weight.device, dtype=self.output_proj.weight.dtype)
        return self.output_proj(self.input_norm(features))


class TorchHubVJEPAVideoEncoder(nn.Module):
    def __init__(
        self,
        output_dim,
        model_id="vjepa2_1_vit_base_384",
        feature_dim=None,
        freeze=True,
        pretrained=True,
        dtype="auto",
        crop_size=None,
        checkpoint_url=None,
        checkpoint_key=None,
    ):
        super().__init__()
        self.model_id = model_id
        self.freeze = bool(freeze)
        self.crop_size = int(crop_size or (384 if "384" in model_id else 256))
        self.checkpoint_url = checkpoint_url
        self.checkpoint_key = checkpoint_key
        load_weights_after_init = bool(pretrained) and (
            checkpoint_url is not None or str(model_id) in VJEPA_TORCHHUB_CHECKPOINTS
        )
        hub_pretrained = bool(pretrained) and not load_weights_after_init
        try:
            try:
                loaded = torch.hub.load("facebookresearch/vjepa2", model_id, pretrained=hub_pretrained)
            except TypeError:
                loaded = torch.hub.load("facebookresearch/vjepa2", model_id)
        except ImportError as exc:
            raise ImportError(
                "video_encoder_backend='vjepa_torchhub' requires the V-JEPA 2 repo dependencies "
                "(notably timm and einops). Install them before loading the torchhub encoder."
            ) from exc
        self.encoder = loaded[0] if isinstance(loaded, (tuple, list)) else loaded
        if load_weights_after_init:
            _load_torchhub_encoder_checkpoint(
                self.encoder,
                model_id=model_id,
                checkpoint_url=checkpoint_url,
                checkpoint_key=checkpoint_key,
            )
        load_dtype = _resolve_vjepa_dtype(dtype)
        if load_dtype is not None:
            self.encoder.to(dtype=load_dtype)
        self.encoder.eval()
        if self.freeze:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)

        hidden_dim = int(feature_dim or _infer_encoder_feature_dim(self.encoder) or 0)
        if hidden_dim <= 0:
            raise ValueError(
                f"Could not infer hidden size for torchhub V-JEPA model {model_id!r}; "
                "set model.vjepa_feature_dim."
            )
        self.feature_dim = hidden_dim
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 1, 3, 1, 1)
        self.register_buffer("imagenet_mean", mean, persistent=False)
        self.register_buffer("imagenet_std", std, persistent=False)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.encoder.eval()
        return self

    def _preprocess(self, video):
        batch_size, frame_count, channels, height, width = video.shape
        video = video.detach().float().clamp(0.0, 1.0)
        if (height, width) != (self.crop_size, self.crop_size):
            flat = video.reshape(batch_size * frame_count, channels, height, width)
            flat = F.interpolate(
                flat,
                size=(self.crop_size, self.crop_size),
                mode="bilinear",
                align_corners=False,
            )
            video = flat.reshape(batch_size, frame_count, channels, self.crop_size, self.crop_size)
        video = (video - self.imagenet_mean) / self.imagenet_std
        return video.permute(0, 2, 1, 3, 4)

    def forward(self, video, frame_times=None):
        del frame_times
        encoder_device = _first_parameter_device(self.encoder)
        encoder_dtype = _first_parameter_dtype(self.encoder)
        inputs = self._preprocess(video).to(device=encoder_device, dtype=encoder_dtype)
        context = torch.no_grad() if self.freeze else nullcontext()
        with context:
            features = self.encoder(inputs)
        features = _as_feature_tokens(features, feature_dim=self.feature_dim)
        features = features.to(device=self.output_proj.weight.device, dtype=self.output_proj.weight.dtype)
        return self.output_proj(self.input_norm(features))


def build_video_encoder(
    backend,
    *,
    clip_length,
    image_size,
    output_dim,
    bottleneck_dim,
    num_heads,
    mlp_ratio,
    tubelet_size,
    encoder_self_attn_layers,
    bottleneck_self_attn_layers,
    vjepa_model_id,
    vjepa_feature_dim,
    vjepa_freeze,
    vjepa_attn_implementation,
    vjepa_dtype,
    vjepa_pretrained,
    vjepa_crop_size,
    vjepa_checkpoint_url,
    video_feature_layers=None,
    video_feature_channels=None,
    video_feature_token_stride=1,
    video_feature_output_dtype=None,
):
    backend = str(backend).lower()
    if backend == "local":
        return VideoEncoder(
            clip_length=clip_length,
            image_size=image_size,
            dim=output_dim,
            bottleneck_dim=bottleneck_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            tubelet_size=tubelet_size,
            encoder_self_attn_layers=encoder_self_attn_layers,
            bottleneck_self_attn_layers=bottleneck_self_attn_layers,
        )
    if backend == "vjepa_hf":
        if vjepa_model_id is None:
            vjepa_model_id = "facebook/vjepa2-vitl-fpc64-256"
        return HuggingFaceVJEPAVideoEncoder(
            output_dim=output_dim,
            model_id=vjepa_model_id,
            feature_dim=vjepa_feature_dim,
            freeze=vjepa_freeze,
            attn_implementation=vjepa_attn_implementation,
            dtype=vjepa_dtype,
        )
    if backend == "vjepa_torchhub":
        if vjepa_model_id is None or str(vjepa_model_id).startswith("facebook/"):
            vjepa_model_id = "vjepa2_1_vit_base_384"
        if vjepa_feature_dim is None and str(vjepa_model_id) == "vjepa2_1_vit_base_384":
            vjepa_feature_dim = 768
        return TorchHubVJEPAVideoEncoder(
            output_dim=output_dim,
            model_id=vjepa_model_id,
            feature_dim=vjepa_feature_dim,
            freeze=vjepa_freeze,
            pretrained=vjepa_pretrained,
            dtype=vjepa_dtype,
            crop_size=vjepa_crop_size,
            checkpoint_url=vjepa_checkpoint_url,
        )
    if backend in {"precomputed", "precomputed_ltx"}:
        return PrecomputedVideoFeatureAdapter(
            output_dim=output_dim,
            feature_channels=video_feature_channels,
            feature_layers=video_feature_layers,
            token_stride=video_feature_token_stride,
            output_dtype=video_feature_output_dtype,
        )
    raise ValueError(
        f"Unknown video_encoder_backend={backend!r}. "
        "Expected one of: local, vjepa_hf, vjepa_torchhub, precomputed, precomputed_ltx."
    )


class LearnedQueryTokenBank(nn.Module):
    def __init__(self, total_tokens, dim, init_std=0.02):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(1, total_tokens, dim) * float(init_std))

    def forward(self, batch_size):
        return self.tokens.expand(batch_size, -1, -1)


@dataclass(frozen=True)
class QueryTokenSpan:
    name: str
    start: int
    count: int

    @property
    def end(self) -> int:
        return self.start + self.count


@dataclass(frozen=True)
class QueryTokenLayout:
    world_tokens: int
    register_tokens: int
    default_active_detail_level: int
    static_core_span: QueryTokenSpan
    dynamic_core_span: QueryTokenSpan
    static_detail_spans: tuple[QueryTokenSpan, ...]
    dynamic_detail_spans: tuple[QueryTokenSpan, ...]
    detail_register_spans: tuple[QueryTokenSpan, ...]
    total_non_camera_tokens: int

    @property
    def detail_levels(self) -> int:
        return max(len(self.static_detail_spans), len(self.dynamic_detail_spans))

    @property
    def max_decoded_static_tokens(self) -> int:
        return self.decoded_static_tokens(self.detail_levels)

    @property
    def max_decoded_dynamic_tokens(self) -> int:
        return self.decoded_dynamic_tokens(self.detail_levels)

    @property
    def max_decoded_tokens(self) -> int:
        return self.max_decoded_static_tokens + self.max_decoded_dynamic_tokens

    def validate_active_detail_level(self, active_detail_level: int | None = None) -> int:
        if active_detail_level is None:
            active_detail_level = self.default_active_detail_level
        active_detail_level = int(active_detail_level)
        if not 0 <= active_detail_level <= self.detail_levels:
            raise ValueError(
                f"token_layout active_detail_level must be between 0 and {self.detail_levels}, "
                f"got {active_detail_level}."
            )
        return active_detail_level

    def static_spans(self, active_detail_level: int | None = None) -> tuple[QueryTokenSpan, ...]:
        level = self.validate_active_detail_level(active_detail_level)
        return (self.static_core_span,) + self.static_detail_spans[:level]

    def dynamic_spans(self, active_detail_level: int | None = None) -> tuple[QueryTokenSpan, ...]:
        level = self.validate_active_detail_level(active_detail_level)
        return (self.dynamic_core_span,) + self.dynamic_detail_spans[:level]

    def decoded_static_tokens(self, active_detail_level: int | None = None) -> int:
        return sum(span.count for span in self.static_spans(active_detail_level))

    def decoded_dynamic_tokens(self, active_detail_level: int | None = None) -> int:
        return sum(span.count for span in self.dynamic_spans(active_detail_level))

    def gather_static(self, queries, active_detail_level: int | None = None):
        return _gather_token_spans(queries, self.static_spans(active_detail_level))

    def gather_dynamic(self, queries, active_detail_level: int | None = None):
        return _gather_token_spans(queries, self.dynamic_spans(active_detail_level))


def _nonnegative_int(value, field_name):
    value = int(value)
    if value < 0:
        raise ValueError(f"token_layout.{field_name} must be >= 0, got {value}.")
    return value


def _positive_int(value, field_name):
    value = int(value)
    if value < 1:
        raise ValueError(f"token_layout.{field_name} must be >= 1, got {value}.")
    return value


def _detail_counts(layout, field_name):
    values = layout.get(field_name, ())
    if values is None:
        return ()
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"token_layout.{field_name} must be a list of token counts.")
    return tuple(_nonnegative_int(value, field_name) for value in values)


def _detail_register_counts(layout, detail_levels):
    values = _detail_counts(layout, "detail_register_tokens")
    if len(values) > detail_levels:
        raise ValueError(
            "token_layout.detail_register_tokens cannot be longer than the number of detail levels "
            f"({detail_levels}), got {len(values)}."
        )
    return values + (0,) * (detail_levels - len(values))


def _gather_token_spans(queries, spans):
    parts = [queries[:, 2 + span.start : 2 + span.end, :] for span in spans if span.count > 0]
    if not parts:
        raise ValueError("Token layout selected no decoded Gaussian tokens.")
    return torch.cat(parts, dim=1)


def _resolve_query_token_layout(token_layout, *, num_tokens, static_tokens=None, dynamic_tokens=None):
    if token_layout is None:
        return None
    if not isinstance(token_layout, Mapping):
        raise ValueError("token_layout must be a mapping or null.")
    allowed = {
        "world_tokens",
        "register_tokens",
        "static_core_tokens",
        "dynamic_core_tokens",
        "static_detail_tokens",
        "dynamic_detail_tokens",
        "detail_register_tokens",
        "active_detail_level",
    }
    unknown = set(token_layout) - allowed
    if unknown:
        known = ", ".join(sorted(allowed))
        raise ValueError(f"Unknown token_layout key(s): {', '.join(sorted(unknown))}. Known keys: {known}.")

    world_tokens = _nonnegative_int(token_layout.get("world_tokens", 0), "world_tokens")
    register_tokens = _nonnegative_int(token_layout.get("register_tokens", 0), "register_tokens")
    static_core_tokens = _positive_int(token_layout.get("static_core_tokens", 0), "static_core_tokens")
    dynamic_core_tokens = _positive_int(token_layout.get("dynamic_core_tokens", 0), "dynamic_core_tokens")
    static_detail_tokens = _detail_counts(token_layout, "static_detail_tokens")
    dynamic_detail_tokens = _detail_counts(token_layout, "dynamic_detail_tokens")
    detail_levels = max(len(static_detail_tokens), len(dynamic_detail_tokens))
    static_detail_tokens = static_detail_tokens + (0,) * (detail_levels - len(static_detail_tokens))
    dynamic_detail_tokens = dynamic_detail_tokens + (0,) * (detail_levels - len(dynamic_detail_tokens))
    detail_register_tokens = _detail_register_counts(token_layout, detail_levels)
    active_detail_level = token_layout.get("active_detail_level", detail_levels)
    if active_detail_level is None:
        active_detail_level = detail_levels
    active_detail_level = int(active_detail_level)
    if not 0 <= active_detail_level <= detail_levels:
        raise ValueError(
            f"token_layout.active_detail_level must be between 0 and {detail_levels}, got {active_detail_level}."
        )

    static_detail_spans = []
    dynamic_detail_spans = []
    detail_register_spans = []
    cursor = 0
    cursor += world_tokens
    cursor += register_tokens
    static_core = QueryTokenSpan(name="static_core", start=cursor, count=static_core_tokens)
    cursor = static_core.end
    dynamic_core = QueryTokenSpan(name="dynamic_core", start=cursor, count=dynamic_core_tokens)
    cursor = dynamic_core.end
    for level, (register_count, static_count, dynamic_count) in enumerate(
        zip(detail_register_tokens, static_detail_tokens, dynamic_detail_tokens, strict=True)
    ):
        register_span = QueryTokenSpan(name=f"detail_register_{level}", start=cursor, count=register_count)
        detail_register_spans.append(register_span)
        cursor = register_span.end
        static_detail_span = QueryTokenSpan(name=f"static_detail_{level}", start=cursor, count=static_count)
        static_detail_spans.append(static_detail_span)
        cursor = static_detail_span.end
        dynamic_detail_span = QueryTokenSpan(name=f"dynamic_detail_{level}", start=cursor, count=dynamic_count)
        dynamic_detail_spans.append(dynamic_detail_span)
        cursor = dynamic_detail_span.end

    if cursor != int(num_tokens):
        raise ValueError(
            f"token_layout spans must sum to num_tokens={int(num_tokens)}, got {cursor}. "
            "Include world/register, static/dynamic core, and all detail tokens in model.tokens."
        )

    layout = QueryTokenLayout(
        world_tokens=world_tokens,
        register_tokens=register_tokens,
        default_active_detail_level=active_detail_level,
        static_core_span=static_core,
        dynamic_core_span=dynamic_core,
        static_detail_spans=tuple(static_detail_spans),
        dynamic_detail_spans=tuple(dynamic_detail_spans),
        detail_register_spans=tuple(detail_register_spans),
        total_non_camera_tokens=cursor,
    )
    if static_tokens is not None and int(static_tokens) != layout.max_decoded_static_tokens:
        raise ValueError(
            "model.static_tokens must match the full token_layout decoded static capacity "
            f"({layout.max_decoded_static_tokens}), got {static_tokens}."
        )
    if dynamic_tokens is not None and int(dynamic_tokens) != layout.max_decoded_dynamic_tokens:
        raise ValueError(
            "model.dynamic_tokens must match the full token_layout decoded dynamic capacity "
            f"({layout.max_decoded_dynamic_tokens}), got {dynamic_tokens}."
        )
    return layout


@dataclass(frozen=True)
class DynamicGaussianBank:
    xyz0: torch.Tensor
    scales: torch.Tensor
    quats0: torch.Tensor
    opacities0: torch.Tensor
    rgbs: torch.Tensor
    A_mu: torch.Tensor
    A_rot: torch.Tensor
    A_alpha: torch.Tensor


def _temporal_basis(decode_time, basis_count, max_frequency):
    if basis_count < 1:
        raise ValueError(f"basis_count must be >= 1, got {basis_count}.")
    if max_frequency <= 0:
        raise ValueError(f"max_frequency must be positive, got {max_frequency}.")

    decode_time = decode_time.reshape(decode_time.shape[0], 1)
    pair_count = basis_count // 2
    basis_parts = []
    if pair_count:
        log2_max = math.log2(float(max_frequency))
        freqs = torch.pow(
            torch.tensor(2.0, device=decode_time.device, dtype=decode_time.dtype),
            torch.linspace(0.0, log2_max, pair_count, device=decode_time.device, dtype=decode_time.dtype),
        )
        angles = (2.0 * math.pi) * decode_time * freqs.unsqueeze(0)
        basis_parts.extend([torch.sin(angles), torch.cos(angles)])
    if basis_count % 2:
        basis_parts.append(decode_time * 2.0 - 1.0)
    return torch.cat(basis_parts, dim=-1)[..., :basis_count]


def _axis_angle_to_quat(axis_angle):
    angle = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
    half_angle = 0.5 * angle
    angle_sq = angle.square()
    sin_half_over_angle = torch.where(
        angle < 1.0e-6,
        0.5 - angle_sq / 48.0,
        torch.sin(half_angle) / angle.clamp_min(1.0e-12),
    )
    quat = torch.cat([torch.cos(half_angle), axis_angle * sin_half_over_angle], dim=-1)
    return F.normalize(quat, p=2, dim=-1)


def _quat_multiply(lhs, rhs):
    lw, lx, ly, lz = lhs.unbind(dim=-1)
    rw, rx, ry, rz = rhs.unbind(dim=-1)
    return torch.stack(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dim=-1,
    )


class DynamicResidualGaussianBankHead(nn.Module):
    def __init__(
        self,
        feat_dim,
        gaussians_per_token,
        time_basis_count,
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
        rgb_init=None,
        rgb_init_min=0.01,
        rgb_init_max=0.99,
        motion_extent=0.25,
        rotation_degrees=10.0,
        alpha_logit_extent=2.0,
        coeff_output_init_std=1.0e-4,
        feature_dim=3,
    ):
        super().__init__()
        if time_basis_count < 1:
            raise ValueError(f"time_basis_count must be >= 1, got {time_basis_count}.")
        if motion_extent < 0:
            raise ValueError(f"motion_extent must be non-negative, got {motion_extent}.")
        if rotation_degrees < 0:
            raise ValueError(f"rotation_degrees must be non-negative, got {rotation_degrees}.")
        if alpha_logit_extent < 0:
            raise ValueError(f"alpha_logit_extent must be non-negative, got {alpha_logit_extent}.")

        self.gaussians_per_token = int(gaussians_per_token)
        self.time_basis_count = int(time_basis_count)
        self.motion_extent = float(motion_extent)
        self.rotation_radians = math.radians(float(rotation_degrees))
        self.alpha_logit_extent = float(alpha_logit_extent)
        self.feature_dim = int(feature_dim)
        self.base_heads = GaussianParameterHeads(
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
            rgb_init=rgb_init,
            rgb_init_min=rgb_init_min,
            rgb_init_max=rgb_init_max,
            feature_dim=self.feature_dim,
        )
        mlp_kwargs = {
            "hidden_dim": head_hidden_dim,
            "hidden_layers": head_hidden_layers,
            "output_init_std": None,
        }
        coeff_count = gaussians_per_token * time_basis_count
        self.motion_head = build_mlp(feat_dim, coeff_count * 3, **mlp_kwargs)
        self.rotation_head = build_mlp(feat_dim, coeff_count * 3, **mlp_kwargs)
        self.alpha_head = build_mlp(feat_dim, coeff_count, **mlp_kwargs)
        self._init_coeff_head(self.motion_head, coeff_output_init_std)
        self._init_coeff_head(self.rotation_head, coeff_output_init_std)
        self._init_coeff_head(self.alpha_head, coeff_output_init_std)

    @staticmethod
    def _init_coeff_head(head, output_init_std):
        output = head[-1]
        if output_init_std is None:
            nn.init.zeros_(output.bias)
            return
        if float(output_init_std) == 0.0:
            nn.init.zeros_(output.weight)
        else:
            nn.init.normal_(output.weight, mean=0.0, std=float(output_init_std))
        nn.init.zeros_(output.bias)

    def _reshape_coefficients(self, values, channels):
        batch_size, token_count, _ = values.shape
        gaussian_count = token_count * self.gaussians_per_token
        return values.reshape(batch_size, gaussian_count, self.time_basis_count, channels)

    def forward(self, tokens):
        xyz0, scales, quats0, opacities0, rgbs = self.base_heads(tokens)
        A_mu = torch.tanh(self._reshape_coefficients(self.motion_head(tokens), 3)) * self.motion_extent
        A_rot = torch.tanh(self._reshape_coefficients(self.rotation_head(tokens), 3)) * self.rotation_radians
        A_alpha = torch.tanh(self._reshape_coefficients(self.alpha_head(tokens), 1)) * self.alpha_logit_extent
        return DynamicGaussianBank(
            xyz0=xyz0,
            scales=scales,
            quats0=quats0,
            opacities0=opacities0,
            rgbs=rgbs,
            A_mu=A_mu,
            A_rot=A_rot,
            A_alpha=A_alpha,
        )


def _init_free_gaussian_raw_params(
    *,
    gaussian_count: int,
    xy_extent: float,
    z_min: float,
    z_max: float,
    scale_init_log_jitter: float,
    opacity_init: float | None,
    position_init_extent_coverage: float,
    rotation_init: str,
    rgb_init: str | None,
    rgb_init_min: float,
    rgb_init_max: float,
    feature_dim: int = 3,
) -> dict[str, torch.Tensor]:
    if position_init_extent_coverage > 0:
        coverage = float(position_init_extent_coverage)
        z_margin = 0.5 * (1.0 - coverage)
        raw_xy = inverse_tanh_values(torch.empty(gaussian_count, 2).uniform_(-coverage, coverage))
        raw_z = torch.logit(torch.empty(gaussian_count, 1).uniform_(z_margin, 1.0 - z_margin), eps=1.0e-6)
        raw_xyz = torch.cat([raw_xy, raw_z], dim=-1)
    else:
        raw_xyz = torch.zeros(gaussian_count, 3)

    raw_scales = torch.zeros(gaussian_count, 3)
    if scale_init_log_jitter > 0:
        raw_scales.uniform_(-float(scale_init_log_jitter), float(scale_init_log_jitter))

    raw_quats = torch.randn(gaussian_count, 4)
    if rotation_init == "identity":
        raw_quats.zero_()
        raw_quats[:, 0] = 1.0

    raw_opacities = torch.zeros(gaussian_count, 1)
    if opacity_init is not None:
        raw_opacities.fill_(inverse_sigmoid(float(opacity_init)))

    raw_rgbs = torch.zeros(gaussian_count, int(feature_dim))
    # rgb_init only makes sense for true RGB output (feature_dim == 3); for
    # feature splatting the base raw values stay zeroed and the rgb_init
    # request is silently ignored.
    if rgb_init == "uniform" and int(feature_dim) == 3:
        rgb_target = torch.empty_like(raw_rgbs).uniform_(float(rgb_init_min), float(rgb_init_max))
        raw_rgbs.copy_(inverse_sigmoid_values(rgb_target))

    return {
        "xyz": raw_xyz,
        "scales": raw_scales,
        "quats": raw_quats,
        "opacities": raw_opacities,
        "rgbs": raw_rgbs,
    }


class FreeGaussianBankImplicitCamera(nn.Module):
    """Direct learnable Gaussian bank with no video encoder and no token decoder."""

    def __init__(
        self,
        clip_length=4,
        image_size=384,
        num_tokens=8,
        feat_dim=128,
        gaussians_per_token=64,
        scene_extent=1.0,
        xy_extent=None,
        z_min=None,
        z_max=None,
        scale_init=0.05,
        scale_init_log_jitter=0.0,
        opacity_init=None,
        query_token_init_std=0.02,
        position_init_extent_coverage=0.0,
        rotation_init="random",
        rgb_init=None,
        rgb_init_min=0.01,
        rgb_init_max=0.99,
        free_frame_count=None,
        free_time_interpolation="nearest",
        base_fov_degrees=60.0,
        base_radius=3.0,
        max_fov_delta_degrees=15.0,
        max_radius_scale=1.5,
        camera_global_head="legacy_orbit",
        lens_model="pinhole",
        max_aspect_log_delta=0.0,
        max_principal_point_delta=0.0,
        distortion_max_abs=0.0,
        base_distortion=None,
        max_rotation_degrees=5.0,
        max_translation_ratio=0.2,
        **_unused,
    ):
        super().__init__()
        self.clip_length = int(clip_length)
        self.image_size = int(image_size)
        self.num_tokens = int(num_tokens)
        self.gaussians_per_token = int(gaussians_per_token)
        self.gaussian_count = self.num_tokens * self.gaussians_per_token
        self.free_frame_count = 1 if free_frame_count is None else int(free_frame_count)
        if self.free_frame_count < 1:
            raise ValueError(f"free_frame_count must be >= 1, got {self.free_frame_count}.")
        self.free_time_interpolation = str(free_time_interpolation).lower()
        if self.free_time_interpolation != "nearest":
            raise ValueError("FreeGaussianBankImplicitCamera currently supports only nearest time lookup.")
        self.scale_init = float(scale_init)
        if xy_extent is None:
            xy_extent = scene_extent
        if z_min is None:
            z_min = -scene_extent
        if z_max is None:
            z_max = scene_extent
        self.xy_extent = float(xy_extent)
        self.z_min = float(z_min)
        self.z_extent = float(z_max) - float(z_min)

        raw = _init_free_gaussian_raw_params(
            gaussian_count=self.free_frame_count * self.gaussian_count,
            xy_extent=self.xy_extent,
            z_min=self.z_min,
            z_max=float(z_max),
            scale_init_log_jitter=float(scale_init_log_jitter),
            opacity_init=opacity_init,
            position_init_extent_coverage=float(position_init_extent_coverage),
            rotation_init=rotation_init,
            rgb_init=None if rgb_init is None else str(rgb_init).lower(),
            rgb_init_min=float(rgb_init_min),
            rgb_init_max=float(rgb_init_max),
        )
        self.raw_xyz = nn.Parameter(raw["xyz"].reshape(self.free_frame_count, self.gaussian_count, 3))
        self.raw_scales = nn.Parameter(raw["scales"].reshape(self.free_frame_count, self.gaussian_count, 3))
        self.raw_quats = nn.Parameter(raw["quats"].reshape(self.free_frame_count, self.gaussian_count, 4))
        self.raw_opacities = nn.Parameter(raw["opacities"].reshape(self.free_frame_count, self.gaussian_count, 1))
        self.raw_rgbs = nn.Parameter(raw["rgbs"].reshape(self.free_frame_count, self.gaussian_count, 3))

        self.global_camera_token = nn.Parameter(torch.randn(feat_dim) * float(query_token_init_std))
        self.path_camera_token = nn.Parameter(torch.randn(feat_dim) * float(query_token_init_std))
        self.path_time_proj = build_time_projector(1, feat_dim)
        self.global_camera_head = build_global_camera_head(
            camera_global_head,
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
        self.path_camera_head = PathCameraHead(
            feat_dim=feat_dim,
            max_rotation_degrees=max_rotation_degrees,
            max_translation_ratio=max_translation_ratio,
        )

    def _gaussians_from_raw(self, raw_xyz, raw_scales, raw_quats, raw_opacities, raw_rgbs):
        xyz = torch.cat(
            [
                torch.tanh(raw_xyz[..., :2]) * self.xy_extent,
                torch.sigmoid(raw_xyz[..., 2:]) * self.z_extent + self.z_min,
            ],
            dim=-1,
        )
        return (
            xyz,
            torch.exp(raw_scales) * self.scale_init,
            F.normalize(raw_quats, p=2, dim=-1),
            torch.sigmoid(raw_opacities),
            torch.sigmoid(raw_rgbs),
        )

    def _gaussians_at_times(self, decode_times):
        frame_positions = decode_times.reshape(-1).to(device=self.raw_xyz.device, dtype=self.raw_xyz.dtype)
        frame_positions = frame_positions.clamp(0.0, 1.0) * float(self.free_frame_count - 1)
        frame_indices = frame_positions.round().to(dtype=torch.long).clamp(0, self.free_frame_count - 1)
        return self._gaussians_from_raw(
            self.raw_xyz.index_select(0, frame_indices),
            self.raw_scales.index_select(0, frame_indices),
            self.raw_quats.index_select(0, frame_indices),
            self.raw_opacities.index_select(0, frame_indices),
            self.raw_rgbs.index_select(0, frame_indices),
        )

    def _camera_for_time(self, decode_time):
        decode_time = decode_time.reshape(1, 1).to(device=self.raw_xyz.device, dtype=self.raw_xyz.dtype)
        global_camera_token = self.global_camera_token
        path_token = self.path_camera_token + self.path_time_proj(decode_time).squeeze(0)
        base_camera, base_state = self.global_camera_head(global_camera_token, image_size=self.image_size)
        rotation_delta, translation_delta, path_residuals = self.path_camera_head(
            path_token.unsqueeze(0),
            base_radius=base_state["radius"],
        )
        camera = compose_camera_with_se3_delta(base_camera, rotation_delta, translation_delta)[0]
        return camera, CameraState(
            fov_degrees=base_state["fov_degrees"],
            radius=base_state["radius"],
            global_residuals=base_state["global_residuals"],
            rotation_delta=rotation_delta,
            translation_delta=translation_delta,
            path_residuals=path_residuals,
        )

    @staticmethod
    def _merge_frames(gaussian_frames, decoded_cameras):
        xyz, scales, quats, opacities, rgbs = gaussian_frames
        camera_states = [item[1] for item in decoded_cameras]
        camera_state = CameraState(
            fov_degrees=torch.stack([state.fov_degrees for state in camera_states]).mean(),
            radius=torch.stack([state.radius for state in camera_states]).mean(),
            global_residuals=torch.stack([state.global_residuals for state in camera_states]).mean(dim=0),
            rotation_delta=torch.cat([state.rotation_delta for state in camera_states], dim=0),
            translation_delta=torch.cat([state.translation_delta for state in camera_states], dim=0),
            path_residuals=torch.cat([state.path_residuals for state in camera_states], dim=0),
        )
        return GaussianSequence(
            xyz=xyz.contiguous(),
            scales=scales.contiguous(),
            quats=quats.contiguous(),
            opacities=opacities.contiguous(),
            rgbs=rgbs.contiguous(),
            cameras=tuple(item[0] for item in decoded_cameras),
            camera_state=camera_state,
        )

    def forward(self, video, decode_times, input_times=None):
        del video, input_times
        if decode_times.ndim != 2:
            raise ValueError(f"Expected decode_times of shape (B, T), got {tuple(decode_times.shape)}")
        if decode_times.shape[0] != 1:
            raise ValueError("FreeGaussianBankImplicitCamera currently expects clip batch size 1.")
        gaussian_frames = self._gaussians_at_times(decode_times)
        decoded_cameras = [self._camera_for_time(decode_times[:, index]) for index in range(decode_times.shape[1])]
        return self._merge_frames(gaussian_frames, decoded_cameras)


class LinearTimeFreeGaussianBankImplicitCamera(FreeGaussianBankImplicitCamera):
    """Shared free Gaussian bank with linear xyz velocity and opacity slope."""

    def __init__(
        self,
        *args,
        free_velocity_extent=1.0,
        free_velocity_init_std=0.0,
        free_opacity_slope_init_std=0.0,
        free_time_center=0.5,
        **kwargs,
    ):
        kwargs.pop("free_frame_count", None)
        kwargs.pop("free_time_interpolation", None)
        super().__init__(*args, free_frame_count=1, free_time_interpolation="nearest", **kwargs)
        if free_velocity_extent < 0:
            raise ValueError(f"free_velocity_extent must be non-negative, got {free_velocity_extent}.")
        if free_velocity_init_std < 0:
            raise ValueError(f"free_velocity_init_std must be non-negative, got {free_velocity_init_std}.")
        if free_opacity_slope_init_std < 0:
            raise ValueError(
                f"free_opacity_slope_init_std must be non-negative, got {free_opacity_slope_init_std}."
            )
        self.free_velocity_extent = float(free_velocity_extent)
        self.free_time_center = float(free_time_center)
        velocity = torch.zeros(self.gaussian_count, 3)
        if free_velocity_init_std > 0:
            velocity.normal_(mean=0.0, std=float(free_velocity_init_std))
        opacity_slope = torch.zeros(self.gaussian_count, 1)
        if free_opacity_slope_init_std > 0:
            opacity_slope.normal_(mean=0.0, std=float(free_opacity_slope_init_std))
        self.raw_velocity = nn.Parameter(velocity)
        self.raw_opacity_slope = nn.Parameter(opacity_slope)

    def _gaussians_at_times(self, decode_times):
        times = decode_times.reshape(-1).to(device=self.raw_xyz.device, dtype=self.raw_xyz.dtype)
        centered_time = (times - self.free_time_center).reshape(-1, 1, 1)

        base_xyz, base_scales, base_quats, base_opacities, base_rgbs = self._gaussians_from_raw(
            self.raw_xyz.squeeze(0),
            self.raw_scales.squeeze(0),
            self.raw_quats.squeeze(0),
            self.raw_opacities.squeeze(0),
            self.raw_rgbs.squeeze(0),
        )
        del base_opacities

        velocity = torch.cat(
            [
                torch.tanh(self.raw_velocity[..., :2]) * self.free_velocity_extent,
                torch.tanh(self.raw_velocity[..., 2:]) * self.free_velocity_extent,
            ],
            dim=-1,
        )
        raw_opacity = self.raw_opacities.squeeze(0) + centered_time * self.raw_opacity_slope.unsqueeze(0)

        frame_count = times.shape[0]
        return (
            base_xyz.unsqueeze(0) + centered_time * velocity.unsqueeze(0),
            base_scales.unsqueeze(0).expand(frame_count, -1, -1),
            base_quats.unsqueeze(0).expand(frame_count, -1, -1),
            torch.sigmoid(raw_opacity),
            base_rgbs.unsqueeze(0).expand(frame_count, -1, -1),
        )


class ResidualFreeGaussianParameterHeads(nn.Module):
    """Per-token free Gaussian base bank plus bounded residuals from token features."""

    def __init__(
        self,
        *,
        feat_dim: int,
        num_tokens: int,
        gaussians_per_token: int,
        xy_extent=1.5,
        z_min=0.5,
        z_max=2.5,
        scale_init=0.05,
        scale_init_log_jitter=0.0,
        opacity_init=None,
        head_hidden_dim=64,
        head_hidden_layers=1,
        residual_output_init_std=1.0e-3,
        position_init_extent_coverage=0.0,
        rotation_init="random",
        rgb_init=None,
        rgb_init_min=0.01,
        rgb_init_max=0.99,
        residual_xyz_raw_scale=0.5,
        residual_scale_log_scale=0.5,
        residual_rot_raw_scale=0.5,
        residual_opacity_logit_scale=1.0,
        residual_rgb_logit_scale=1.0,
        residual_head_input_norm="rmsnorm",
        feature_dim=3,
    ):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.gaussians_per_token = int(gaussians_per_token)
        self.feature_dim = int(feature_dim)
        if self.feature_dim < 1:
            raise ValueError(f"feature_dim must be >= 1, got {feature_dim}.")
        self.xy_extent = float(xy_extent)
        self.z_min = float(z_min)
        self.z_extent = float(z_max) - float(z_min)
        self.scale_init = float(scale_init)
        self.residual_xyz_raw_scale = float(residual_xyz_raw_scale)
        self.residual_scale_log_scale = float(residual_scale_log_scale)
        self.residual_rot_raw_scale = float(residual_rot_raw_scale)
        self.residual_opacity_logit_scale = float(residual_opacity_logit_scale)
        self.residual_rgb_logit_scale = float(residual_rgb_logit_scale)
        if self.num_tokens < 1:
            raise ValueError(f"num_tokens must be positive, got {num_tokens}.")
        if self.gaussians_per_token < 1:
            raise ValueError(f"gaussians_per_token must be positive, got {gaussians_per_token}.")
        if self.z_extent <= 0:
            raise ValueError(f"z_max must be greater than z_min, got z_min={z_min}, z_max={z_max}.")
        if self.scale_init <= 0:
            raise ValueError(f"scale_init must be positive, got {scale_init}.")

        raw = _init_free_gaussian_raw_params(
            gaussian_count=self.num_tokens * self.gaussians_per_token,
            xy_extent=self.xy_extent,
            z_min=self.z_min,
            z_max=float(z_max),
            scale_init_log_jitter=float(scale_init_log_jitter),
            opacity_init=opacity_init,
            position_init_extent_coverage=float(position_init_extent_coverage),
            rotation_init=rotation_init,
            rgb_init=None if rgb_init is None else str(rgb_init).lower(),
            rgb_init_min=float(rgb_init_min),
            rgb_init_max=float(rgb_init_max),
            feature_dim=self.feature_dim,
        )
        shape3 = (self.num_tokens, self.gaussians_per_token, 3)
        feature_shape = (self.num_tokens, self.gaussians_per_token, self.feature_dim)
        self.base_raw_xyz = nn.Parameter(raw["xyz"].reshape(shape3))
        self.base_raw_scales = nn.Parameter(raw["scales"].reshape(shape3))
        self.base_raw_quats = nn.Parameter(raw["quats"].reshape(self.num_tokens, self.gaussians_per_token, 4))
        self.base_raw_opacities = nn.Parameter(raw["opacities"].reshape(self.num_tokens, self.gaussians_per_token, 1))
        self.base_raw_rgbs = nn.Parameter(raw["rgbs"].reshape(feature_shape))

        norm_name = str(residual_head_input_norm).lower()
        if norm_name in {"none", "false", "off"}:
            self.input_norm = nn.Identity()
        elif norm_name in {"rms", "rmsnorm"}:
            self.input_norm = RMSNorm(feat_dim)
        elif norm_name in {"layer", "layernorm"}:
            self.input_norm = nn.LayerNorm(feat_dim)
        else:
            raise ValueError(
                f"Unknown residual_head_input_norm={residual_head_input_norm!r}. "
                "Expected one of: none, rmsnorm, layernorm."
            )

        mlp_kwargs = {
            "hidden_dim": head_hidden_dim,
            "hidden_layers": head_hidden_layers,
            "output_init_std": residual_output_init_std,
        }
        self.xyz_residual_head = build_mlp(feat_dim, gaussians_per_token * 3, **mlp_kwargs)
        self.scale_residual_head = build_mlp(feat_dim, gaussians_per_token * 3, **mlp_kwargs)
        self.rot_residual_head = build_mlp(feat_dim, gaussians_per_token * 4, **mlp_kwargs)
        self.opacity_residual_head = build_mlp(feat_dim, gaussians_per_token, **mlp_kwargs)
        self.rgb_residual_head = build_mlp(feat_dim, gaussians_per_token * self.feature_dim, **mlp_kwargs)

    def _reshape(self, values, channels):
        return values.reshape(values.shape[0], self.num_tokens, self.gaussians_per_token, channels)

    def _bounded_residual(self, values, channels, scale):
        return torch.tanh(self._reshape(values, channels)) * float(scale)

    def forward(self, tokens):
        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens with shape [B,T,C], got {tuple(tokens.shape)}.")
        if tokens.shape[1] != self.num_tokens:
            raise ValueError(f"Expected {self.num_tokens} tokens, got {tokens.shape[1]}.")
        head_tokens = self.input_norm(tokens)
        raw_xyz = self.base_raw_xyz.unsqueeze(0) + self._bounded_residual(
            self.xyz_residual_head(head_tokens),
            3,
            self.residual_xyz_raw_scale,
        )
        raw_scales = self.base_raw_scales.unsqueeze(0) + self._bounded_residual(
            self.scale_residual_head(head_tokens),
            3,
            self.residual_scale_log_scale,
        )
        raw_quats = self.base_raw_quats.unsqueeze(0) + self._bounded_residual(
            self.rot_residual_head(head_tokens),
            4,
            self.residual_rot_raw_scale,
        )
        raw_opacities = self.base_raw_opacities.unsqueeze(0) + self._bounded_residual(
            self.opacity_residual_head(head_tokens),
            1,
            self.residual_opacity_logit_scale,
        )
        raw_rgbs = self.base_raw_rgbs.unsqueeze(0) + self._bounded_residual(
            self.rgb_residual_head(head_tokens),
            self.feature_dim,
            self.residual_rgb_logit_scale,
        )

        xyz = torch.cat(
            [
                torch.tanh(raw_xyz[..., :2]) * self.xy_extent,
                torch.sigmoid(raw_xyz[..., 2:]) * self.z_extent + self.z_min,
            ],
            dim=-1,
        )
        batch_size = tokens.shape[0]
        # F=3 keeps the legacy sigmoid-RGB path bit-identical; F!=3 emits raw
        # features for a downstream colorize MLP.
        if self.feature_dim == 3:
            rgbs_out = torch.sigmoid(raw_rgbs)
        else:
            rgbs_out = raw_rgbs
        return (
            xyz.reshape(batch_size, self.num_tokens * self.gaussians_per_token, 3),
            (torch.exp(raw_scales) * self.scale_init).reshape(batch_size, self.num_tokens * self.gaussians_per_token, 3),
            F.normalize(raw_quats, p=2, dim=-1).reshape(batch_size, self.num_tokens * self.gaussians_per_token, 4),
            torch.sigmoid(raw_opacities).reshape(batch_size, self.num_tokens * self.gaussians_per_token, 1),
            rgbs_out.reshape(batch_size, self.num_tokens * self.gaussians_per_token, self.feature_dim),
        )


class UnconditionedTokenGSImplicitCamera(nn.Module):
    """Learned tokens decoded to splats with no video or external feature memory."""

    def __init__(
        self,
        clip_length=4,
        image_size=384,
        num_tokens=8,
        feat_dim=128,
        gaussians_per_token=64,
        scene_extent=1.0,
        xy_extent=None,
        z_min=None,
        z_max=None,
        scale_init=0.05,
        scale_init_log_jitter=0.0,
        opacity_init=None,
        query_token_init_std=0.02,
        head_hidden_dim=64,
        head_hidden_layers=1,
        head_output_init_std=None,
        position_init_extent_coverage=0.0,
        rotation_init="random",
        rgb_init=None,
        rgb_init_min=0.01,
        rgb_init_max=0.99,
        video_encoder_backend="none",
        base_fov_degrees=60.0,
        base_radius=3.0,
        max_fov_delta_degrees=15.0,
        max_radius_scale=1.5,
        camera_global_head="legacy_orbit",
        lens_model="pinhole",
        max_aspect_log_delta=0.0,
        max_principal_point_delta=0.0,
        distortion_max_abs=0.0,
        base_distortion=None,
        max_rotation_degrees=5.0,
        max_translation_ratio=0.2,
        static_tokens=None,
        dynamic_tokens=None,
        dynamic_time_basis_count=8,
        dynamic_time_max_frequency=8.0,
        dynamic_motion_extent=None,
        dynamic_rotation_degrees=10.0,
        dynamic_alpha_logit_extent=2.0,
        dynamic_coeff_output_init_std=1.0e-4,
        feature_dim=3,
        **_unused,
    ):
        super().__init__()
        self.clip_length = int(clip_length)
        self.image_size = int(image_size)
        self.num_tokens = int(num_tokens)
        self.feature_dim = int(feature_dim)
        self.static_tokens = None if static_tokens is None else int(static_tokens)
        self.dynamic_tokens = None if dynamic_tokens is None else int(dynamic_tokens)
        self.use_static_dynamic_split = self.static_tokens is not None or self.dynamic_tokens is not None
        if self.use_static_dynamic_split:
            if self.static_tokens is None or self.dynamic_tokens is None:
                raise ValueError("static_tokens and dynamic_tokens must be provided together.")
            if self.static_tokens < 1 or self.dynamic_tokens < 1:
                raise ValueError(
                    f"static_tokens and dynamic_tokens must be positive, "
                    f"got {self.static_tokens} and {self.dynamic_tokens}."
                )
            if self.static_tokens + self.dynamic_tokens != self.num_tokens:
                raise ValueError(
                    f"static_tokens + dynamic_tokens must equal num_tokens={self.num_tokens}, "
                    f"got {self.static_tokens} + {self.dynamic_tokens}."
                )
        self.total_tokens = self.num_tokens + 2
        self.feat_dim = int(feat_dim)
        self.gaussians_per_token = int(gaussians_per_token)
        self.dynamic_time_basis_count = int(dynamic_time_basis_count)
        self.dynamic_time_max_frequency = float(dynamic_time_max_frequency)
        self.video_encoder_backend = str(video_encoder_backend).lower()
        if xy_extent is None:
            xy_extent = scene_extent
        if z_min is None:
            z_min = -scene_extent
        if z_max is None:
            z_max = scene_extent
        self.query_tokens = LearnedQueryTokenBank(
            total_tokens=self.total_tokens,
            dim=feat_dim,
            init_std=query_token_init_std,
        )
        self.time_proj = build_time_projector(1, feat_dim)
        self.head_time_proj = build_time_projector(1, feat_dim)
        gaussian_head_kwargs = {
            "feat_dim": feat_dim,
            "gaussians_per_token": gaussians_per_token,
            "xy_extent": xy_extent,
            "z_min": z_min,
            "z_max": z_max,
            "scale_init": scale_init,
            "scale_init_log_jitter": scale_init_log_jitter,
            "opacity_init": opacity_init,
            "head_hidden_dim": head_hidden_dim,
            "head_hidden_layers": head_hidden_layers,
            "head_output_init_std": head_output_init_std,
            "position_init_extent_coverage": position_init_extent_coverage,
            "rotation_init": rotation_init,
            "rgb_init": rgb_init,
            "rgb_init_min": rgb_init_min,
            "rgb_init_max": rgb_init_max,
            "feature_dim": self.feature_dim,
        }
        if self.use_static_dynamic_split:
            if dynamic_motion_extent is None:
                dynamic_motion_extent = 0.25 * float(scene_extent)
            self.static_gaussian_heads = GaussianParameterHeads(**gaussian_head_kwargs)
            self.dynamic_gaussian_heads = DynamicResidualGaussianBankHead(
                **gaussian_head_kwargs,
                time_basis_count=dynamic_time_basis_count,
                motion_extent=dynamic_motion_extent,
                rotation_degrees=dynamic_rotation_degrees,
                alpha_logit_extent=dynamic_alpha_logit_extent,
                coeff_output_init_std=dynamic_coeff_output_init_std,
            )
        else:
            self.gaussian_heads = GaussianParameterHeads(**gaussian_head_kwargs)
        self.global_camera_head = build_global_camera_head(
            camera_global_head,
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
        self.path_camera_head = PathCameraHead(
            feat_dim=feat_dim,
            max_rotation_degrees=max_rotation_degrees,
            max_translation_ratio=max_translation_ratio,
        )

    def refine_queries(self, batch_size, decode_time=None):
        queries = self.query_tokens(batch_size)
        if decode_time is not None:
            decode_time = decode_time.reshape(batch_size, 1).to(device=queries.device, dtype=queries.dtype)
            query_offsets = torch.zeros_like(queries)
            query_offsets[:, 1:, :] = self.time_proj(decode_time).unsqueeze(1)
            queries = queries + query_offsets
        return queries

    def _decode_single_time(self, decode_time, fixed_global_camera_token):
        refined_queries = self.refine_queries(1, decode_time=decode_time)
        path_token = refined_queries[:, 1, :]
        gs_tokens = refined_queries[:, 2:, :]
        head_time_offset = self.head_time_proj(
            decode_time.reshape(1, 1).to(device=gs_tokens.device, dtype=gs_tokens.dtype)
        )
        path_token = path_token + head_time_offset
        gs_tokens = gs_tokens + head_time_offset.unsqueeze(1)
        xyz, scales, quats, opacities, rgbs = self.gaussian_heads(gs_tokens)
        base_camera, base_state = self.global_camera_head(fixed_global_camera_token.squeeze(0), image_size=self.image_size)
        rotation_delta, translation_delta, path_residuals = self.path_camera_head(
            path_token,
            base_radius=base_state["radius"],
        )
        camera = compose_camera_with_se3_delta(base_camera, rotation_delta, translation_delta)[0]
        return (
            xyz.squeeze(0),
            scales.squeeze(0),
            quats.squeeze(0),
            opacities.squeeze(0),
            rgbs.squeeze(0),
            camera,
            CameraState(
                fov_degrees=base_state["fov_degrees"],
                radius=base_state["radius"],
                global_residuals=base_state["global_residuals"],
                rotation_delta=rotation_delta,
                translation_delta=translation_delta,
                path_residuals=path_residuals,
            ),
        )

    def _eval_dynamic_bank(self, bank, decode_time):
        return DynamicVideoTokenGSImplicitCamera._eval_dynamic_bank(self, bank, decode_time)

    def _decode_static_dynamic_split(self, fixed_queries, decode_times, fixed_global_camera_token):
        static_token_slice = fixed_queries[:, 2 : 2 + self.static_tokens, :]
        dynamic_token_slice = fixed_queries[:, 2 + self.static_tokens :, :]
        static_xyz, static_scales, static_quats, static_opacities, static_rgbs = self.static_gaussian_heads(
            static_token_slice
        )
        dynamic_bank = self.dynamic_gaussian_heads(dynamic_token_slice)

        decoded = []
        dynamic_opacity_frames = []
        for index in range(decode_times.shape[1]):
            dynamic_xyz, dynamic_scales, dynamic_quats, dynamic_opacities, dynamic_rgbs = self._eval_dynamic_bank(
                dynamic_bank,
                decode_times[:, index],
            )
            dynamic_opacity_frames.append(dynamic_opacities.squeeze(0))
            camera, camera_state = DynamicVideoTokenGSImplicitCamera._decode_camera_single_time(
                self,
                self.refine_queries(1, decode_time=decode_times[:, index]),
                decode_times[:, index],
                fixed_global_camera_token,
            )
            decoded.append(
                (
                    torch.cat([static_xyz, dynamic_xyz], dim=1).squeeze(0),
                    torch.cat([static_scales, dynamic_scales], dim=1).squeeze(0),
                    torch.cat([static_quats, dynamic_quats], dim=1).squeeze(0),
                    torch.cat([static_opacities, dynamic_opacities], dim=1).squeeze(0),
                    torch.cat([static_rgbs, dynamic_rgbs], dim=1).squeeze(0),
                    camera,
                    camera_state,
                )
            )

        return DynamicVideoTokenGSImplicitCamera._merge_decoded_frames(
            decoded,
            auxiliary={
                "static_opacities": static_opacities,
                "dynamic_opacities": torch.stack(dynamic_opacity_frames, dim=0),
                "dynamic_A_mu": dynamic_bank.A_mu,
                "dynamic_A_rot": dynamic_bank.A_rot,
                "dynamic_A_alpha": dynamic_bank.A_alpha,
            },
        )

    def forward(self, video, decode_times, input_times=None):
        del video, input_times
        if decode_times.ndim != 2:
            raise ValueError(f"Expected decode_times of shape (B, T), got {tuple(decode_times.shape)}")
        if decode_times.shape[0] != 1:
            raise ValueError("UnconditionedTokenGSImplicitCamera currently expects clip batch size 1.")
        fixed_queries = self.refine_queries(1, decode_time=None)
        fixed_global_camera_token = fixed_queries[:, 0, :]
        if self.use_static_dynamic_split:
            return self._decode_static_dynamic_split(fixed_queries, decode_times, fixed_global_camera_token)
        decoded = [
            self._decode_single_time(decode_times[:, index], fixed_global_camera_token)
            for index in range(decode_times.shape[1])
        ]
        return DynamicVideoTokenGSImplicitCamera._merge_decoded_frames(decoded)


class UnconditionedResidualFreeBankImplicitCamera(UnconditionedTokenGSImplicitCamera):
    """Unconditioned token residuals on top of a per-token free Gaussian base bank."""

    def __init__(
        self,
        *args,
        residual_output_init_std=1.0e-3,
        residual_xyz_raw_scale=0.5,
        residual_scale_log_scale=0.5,
        residual_rot_raw_scale=0.5,
        residual_opacity_logit_scale=1.0,
        residual_rgb_logit_scale=1.0,
        residual_head_input_norm="rmsnorm",
        **kwargs,
    ):
        num_tokens = int(kwargs.get("num_tokens", 8))
        feat_dim = int(kwargs.get("feat_dim", 128))
        gaussians_per_token = int(kwargs.get("gaussians_per_token", 64))
        scene_extent = float(kwargs.get("scene_extent", 1.0))
        xy_extent = kwargs.get("xy_extent", scene_extent)
        z_min = kwargs.get("z_min", -scene_extent)
        z_max = kwargs.get("z_max", scene_extent)
        scale_init = kwargs.get("scale_init", 0.05)
        scale_init_log_jitter = kwargs.get("scale_init_log_jitter", 0.0)
        opacity_init = kwargs.get("opacity_init")
        head_hidden_dim = kwargs.get("head_hidden_dim", 64)
        head_hidden_layers = kwargs.get("head_hidden_layers", 1)
        position_init_extent_coverage = kwargs.get("position_init_extent_coverage", 0.0)
        rotation_init = kwargs.get("rotation_init", "random")
        rgb_init = kwargs.get("rgb_init")
        rgb_init_min = kwargs.get("rgb_init_min", 0.01)
        rgb_init_max = kwargs.get("rgb_init_max", 0.99)
        feature_dim = int(kwargs.get("feature_dim", 3))
        super().__init__(*args, **kwargs)
        self.gaussian_heads = ResidualFreeGaussianParameterHeads(
            feat_dim=feat_dim,
            num_tokens=num_tokens,
            gaussians_per_token=gaussians_per_token,
            xy_extent=xy_extent,
            z_min=z_min,
            z_max=z_max,
            scale_init=scale_init,
            scale_init_log_jitter=scale_init_log_jitter,
            opacity_init=opacity_init,
            head_hidden_dim=head_hidden_dim,
            head_hidden_layers=head_hidden_layers,
            residual_output_init_std=residual_output_init_std,
            position_init_extent_coverage=position_init_extent_coverage,
            rotation_init=rotation_init,
            rgb_init=rgb_init,
            rgb_init_min=rgb_init_min,
            rgb_init_max=rgb_init_max,
            residual_xyz_raw_scale=residual_xyz_raw_scale,
            residual_scale_log_scale=residual_scale_log_scale,
            residual_rot_raw_scale=residual_rot_raw_scale,
            residual_opacity_logit_scale=residual_opacity_logit_scale,
            residual_rgb_logit_scale=residual_rgb_logit_scale,
            residual_head_input_norm=residual_head_input_norm,
            feature_dim=feature_dim,
        )


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
        xy_extent=None,
        z_min=None,
        z_max=None,
        scale_init=0.05,
        scale_init_log_jitter=0.0,
        opacity_init=None,
        query_token_init_std=0.02,
        head_hidden_dim=64,
        head_hidden_layers=1,
        head_output_init_std=None,
        position_init_extent_coverage=0.0,
        rotation_init="random",
        rgb_init=None,
        rgb_init_min=0.01,
        rgb_init_max=0.99,
        video_encoder_backend="local",
        tubelet_size=(4, 16, 16),
        encoder_self_attn_layers=1,
        bottleneck_self_attn_layers=4,
        vjepa_model_id=None,
        vjepa_feature_dim=None,
        vjepa_freeze=True,
        vjepa_attn_implementation="sdpa",
        vjepa_dtype="auto",
        vjepa_pretrained=True,
        vjepa_crop_size=None,
        vjepa_checkpoint_url=None,
        video_feature_layers=None,
        video_feature_channels=None,
        video_feature_token_stride=1,
        video_feature_output_dtype=None,
        cross_attn_layers=1,
        camera_refine_with_decode_time=True,
        base_fov_degrees=60.0,
        base_radius=3.0,
        max_fov_delta_degrees=15.0,
        max_radius_scale=1.5,
        camera_global_head="legacy_orbit",
        lens_model="pinhole",
        max_aspect_log_delta=0.0,
        max_principal_point_delta=0.0,
        distortion_max_abs=0.0,
        base_distortion=None,
        max_rotation_degrees=5.0,
        max_translation_ratio=0.2,
        static_tokens=None,
        dynamic_tokens=None,
        dynamic_time_basis_count=8,
        dynamic_time_max_frequency=8.0,
        dynamic_motion_extent=None,
        dynamic_rotation_degrees=10.0,
        dynamic_alpha_logit_extent=2.0,
        dynamic_coeff_output_init_std=1.0e-4,
        token_layout=None,
        feature_dim=3,
    ):
        super().__init__()
        self.clip_length = clip_length
        self.image_size = image_size
        self.num_tokens = int(num_tokens)
        legacy_static_tokens = None if static_tokens is None else int(static_tokens)
        legacy_dynamic_tokens = None if dynamic_tokens is None else int(dynamic_tokens)
        self.token_layout = _resolve_query_token_layout(
            token_layout,
            num_tokens=self.num_tokens,
            static_tokens=legacy_static_tokens,
            dynamic_tokens=legacy_dynamic_tokens,
        )
        if self.token_layout is None:
            self.static_tokens = legacy_static_tokens
            self.dynamic_tokens = legacy_dynamic_tokens
        else:
            self.active_detail_level = self.token_layout.default_active_detail_level
            self.static_tokens = self.token_layout.decoded_static_tokens(self.active_detail_level)
            self.dynamic_tokens = self.token_layout.decoded_dynamic_tokens(self.active_detail_level)
        self.use_static_dynamic_split = (
            self.token_layout is not None or self.static_tokens is not None or self.dynamic_tokens is not None
        )
        if self.use_static_dynamic_split and self.token_layout is None:
            if self.static_tokens is None or self.dynamic_tokens is None:
                raise ValueError("static_tokens and dynamic_tokens must be provided together.")
            if self.static_tokens < 1 or self.dynamic_tokens < 1:
                raise ValueError(
                    f"static_tokens and dynamic_tokens must be positive, "
                    f"got {self.static_tokens} and {self.dynamic_tokens}."
                )
            if self.static_tokens + self.dynamic_tokens != self.num_tokens:
                raise ValueError(
                    f"static_tokens + dynamic_tokens must equal num_tokens={self.num_tokens}, "
                    f"got {self.static_tokens} + {self.dynamic_tokens}."
                )
        self.total_tokens = self.num_tokens + 2
        self.feat_dim = feat_dim
        self.gaussians_per_token = gaussians_per_token
        self.feature_dim = int(feature_dim)
        self.dynamic_time_basis_count = int(dynamic_time_basis_count)
        self.dynamic_time_max_frequency = float(dynamic_time_max_frequency)
        self.camera_refine_with_decode_time = bool(camera_refine_with_decode_time)
        self.video_encoder_backend = str(video_encoder_backend).lower()
        self.video_encoder = build_video_encoder(
            self.video_encoder_backend,
            clip_length=clip_length,
            image_size=image_size,
            output_dim=feat_dim,
            bottleneck_dim=bottleneck_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            tubelet_size=tubelet_size,
            encoder_self_attn_layers=encoder_self_attn_layers,
            bottleneck_self_attn_layers=bottleneck_self_attn_layers,
            vjepa_model_id=vjepa_model_id,
            vjepa_feature_dim=vjepa_feature_dim,
            vjepa_freeze=vjepa_freeze,
            vjepa_attn_implementation=vjepa_attn_implementation,
            vjepa_dtype=vjepa_dtype,
            vjepa_pretrained=vjepa_pretrained,
            vjepa_crop_size=vjepa_crop_size,
            vjepa_checkpoint_url=vjepa_checkpoint_url,
            video_feature_layers=video_feature_layers,
            video_feature_channels=video_feature_channels,
            video_feature_token_stride=video_feature_token_stride,
            video_feature_output_dtype=video_feature_output_dtype,
        )
        self.query_tokens = LearnedQueryTokenBank(
            total_tokens=self.total_tokens,
            dim=feat_dim,
            init_std=query_token_init_std,
        )
        self.query_decoder_blocks = nn.ModuleList(
            [
                QueryCrossAttentionBlock(dim=feat_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
                for _ in range(cross_attn_layers)
            ]
        )
        if xy_extent is None:
            xy_extent = scene_extent
        if z_min is None:
            z_min = -scene_extent
        if z_max is None:
            z_max = scene_extent
        gaussian_head_kwargs = {
            "feat_dim": feat_dim,
            "gaussians_per_token": gaussians_per_token,
            "xy_extent": xy_extent,
            "z_min": z_min,
            "z_max": z_max,
            "scale_init": scale_init,
            "scale_init_log_jitter": scale_init_log_jitter,
            "opacity_init": opacity_init,
            "head_hidden_dim": head_hidden_dim,
            "head_hidden_layers": head_hidden_layers,
            "head_output_init_std": head_output_init_std,
            "position_init_extent_coverage": position_init_extent_coverage,
            "rotation_init": rotation_init,
            "rgb_init": rgb_init,
            "rgb_init_min": rgb_init_min,
            "rgb_init_max": rgb_init_max,
            "feature_dim": self.feature_dim,
        }
        if self.use_static_dynamic_split:
            if dynamic_motion_extent is None:
                dynamic_motion_extent = 0.25 * float(scene_extent)
            self.static_gaussian_heads = GaussianParameterHeads(**gaussian_head_kwargs)
            self.dynamic_gaussian_heads = DynamicResidualGaussianBankHead(
                **gaussian_head_kwargs,
                time_basis_count=dynamic_time_basis_count,
                motion_extent=dynamic_motion_extent,
                rotation_degrees=dynamic_rotation_degrees,
                alpha_logit_extent=dynamic_alpha_logit_extent,
                coeff_output_init_std=dynamic_coeff_output_init_std,
            )
        else:
            self.gaussian_heads = GaussianParameterHeads(**gaussian_head_kwargs)
        self.global_camera_head = build_global_camera_head(
            camera_global_head,
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
        self.path_camera_head = PathCameraHead(
            feat_dim=feat_dim,
            max_rotation_degrees=max_rotation_degrees,
            max_translation_ratio=max_translation_ratio,
        )
        self.time_proj = build_time_projector(1, feat_dim)
        self.head_time_proj = build_time_projector(1, feat_dim)

    def set_active_detail_level(self, active_detail_level: int | None) -> int | None:
        if self.token_layout is None:
            if active_detail_level is not None:
                raise ValueError("active_detail_level requires model.token_layout.")
            return None
        level = self.token_layout.validate_active_detail_level(active_detail_level)
        self.active_detail_level = level
        self.static_tokens = self.token_layout.decoded_static_tokens(level)
        self.dynamic_tokens = self.token_layout.decoded_dynamic_tokens(level)
        return level

    def active_decoded_token_count(self) -> int:
        if self.token_layout is None:
            return int(self.num_tokens)
        return int(self.static_tokens) + int(self.dynamic_tokens)

    def active_gaussian_count(self) -> int:
        return self.active_decoded_token_count() * int(self.gaussians_per_token)

    def decoded_static_query_tokens(self, fixed_queries):
        if self.token_layout is None:
            return fixed_queries[:, 2 : 2 + self.static_tokens, :]
        return self.token_layout.gather_static(fixed_queries, self.active_detail_level)

    def decoded_dynamic_query_tokens(self, fixed_queries):
        if self.token_layout is None:
            return fixed_queries[:, 2 + self.static_tokens :, :]
        return self.token_layout.gather_dynamic(fixed_queries, self.active_detail_level)

    def refine_queries(self, video_tokens, decode_time=None):
        batch_size = video_tokens.shape[0]
        queries = self.query_tokens(batch_size)
        if decode_time is not None:
            decode_time = decode_time.to(device=video_tokens.device, dtype=video_tokens.dtype).reshape(batch_size, 1)
            query_offsets = torch.zeros_like(queries)
            query_offsets[:, 1:, :] = self.time_proj(decode_time).unsqueeze(1)
            queries = queries + query_offsets
        for block in self.query_decoder_blocks:
            queries = block(queries, video_tokens)
        return queries

    def encode_queries(self, video, frame_times=None):
        return self.refine_queries(self.video_encoder(video, frame_times=frame_times))

    def _decode_single_time(self, refined_queries, decode_time=None, global_camera_token=None):
        if global_camera_token is None:
            global_camera_token = refined_queries[:, 0, :]
        path_token = refined_queries[:, 1, :]
        gs_tokens = refined_queries[:, 2:, :]
        if decode_time is not None:
            decode_time = decode_time.to(device=refined_queries.device, dtype=refined_queries.dtype).reshape(
                refined_queries.shape[0], 1
            )
            head_time_offset = self.head_time_proj(decode_time)
            path_token = path_token + head_time_offset
            gs_tokens = gs_tokens + head_time_offset.unsqueeze(1)

        xyz, scales, quats, opacities, rgbs = self.gaussian_heads(gs_tokens)
        base_camera, base_state = self.global_camera_head(global_camera_token.squeeze(0), image_size=self.image_size)
        rotation_delta, translation_delta, path_residuals = self.path_camera_head(
            path_token, base_radius=base_state["radius"]
        )
        camera = compose_camera_with_se3_delta(base_camera, rotation_delta, translation_delta)[0]
        return (
            xyz.squeeze(0),
            scales.squeeze(0),
            quats.squeeze(0),
            opacities.squeeze(0),
            rgbs.squeeze(0),
            camera,
            CameraState(
                fov_degrees=base_state["fov_degrees"],
                radius=base_state["radius"],
                global_residuals=base_state["global_residuals"],
                rotation_delta=rotation_delta,
                translation_delta=translation_delta,
                path_residuals=path_residuals,
            ),
        )

    def _decode_camera_single_time(self, refined_queries, decode_time=None, global_camera_token=None):
        if global_camera_token is None:
            global_camera_token = refined_queries[:, 0, :]
        path_token = refined_queries[:, 1, :]
        if decode_time is not None:
            decode_time = decode_time.to(device=refined_queries.device, dtype=refined_queries.dtype).reshape(
                refined_queries.shape[0], 1
            )
            path_token = path_token + self.head_time_proj(decode_time)
        base_camera, base_state = self.global_camera_head(global_camera_token.squeeze(0), image_size=self.image_size)
        rotation_delta, translation_delta, path_residuals = self.path_camera_head(
            path_token, base_radius=base_state["radius"]
        )
        camera = compose_camera_with_se3_delta(base_camera, rotation_delta, translation_delta)[0]
        return (
            camera,
            CameraState(
                fov_degrees=base_state["fov_degrees"],
                radius=base_state["radius"],
                global_residuals=base_state["global_residuals"],
                rotation_delta=rotation_delta,
                translation_delta=translation_delta,
                path_residuals=path_residuals,
            ),
        )

    def _eval_dynamic_bank(self, bank, decode_time):
        decode_time = decode_time.to(device=bank.xyz0.device, dtype=bank.xyz0.dtype).reshape(bank.xyz0.shape[0], 1)
        phi = _temporal_basis(
            decode_time,
            basis_count=self.dynamic_time_basis_count,
            max_frequency=self.dynamic_time_max_frequency,
        )
        delta_xyz = torch.einsum("bm,bgmc->bgc", phi, bank.A_mu)
        delta_rot = torch.einsum("bm,bgmc->bgc", phi, bank.A_rot)
        delta_alpha = torch.einsum("bm,bgmc->bgc", phi, bank.A_alpha)
        alpha0_logits = torch.logit(bank.opacities0.clamp(1.0e-6, 1.0 - 1.0e-6))
        quats = F.normalize(_quat_multiply(bank.quats0, _axis_angle_to_quat(delta_rot)), p=2, dim=-1)
        return (
            bank.xyz0 + delta_xyz,
            bank.scales,
            quats,
            torch.sigmoid(alpha0_logits + delta_alpha),
            bank.rgbs,
        )

    @staticmethod
    def _merge_decoded_frames(decoded, auxiliary=None):
        camera_state = CameraState(
            fov_degrees=torch.stack([item[6].fov_degrees for item in decoded]).mean(),
            radius=torch.stack([item[6].radius for item in decoded]).mean(),
            global_residuals=torch.stack([item[6].global_residuals for item in decoded]).mean(dim=0),
            rotation_delta=torch.cat([item[6].rotation_delta for item in decoded], dim=0),
            translation_delta=torch.cat([item[6].translation_delta for item in decoded], dim=0),
            path_residuals=torch.cat([item[6].path_residuals for item in decoded], dim=0),
        )
        return GaussianSequence(
            xyz=torch.stack([item[0] for item in decoded], dim=0),
            scales=torch.stack([item[1] for item in decoded], dim=0),
            quats=torch.stack([item[2] for item in decoded], dim=0),
            opacities=torch.stack([item[3] for item in decoded], dim=0),
            rgbs=torch.stack([item[4] for item in decoded], dim=0),
            cameras=tuple(item[5] for item in decoded),
            camera_state=camera_state,
            auxiliary={} if auxiliary is None else auxiliary,
        )

    def _decode_static_dynamic_split(self, video_tokens, fixed_queries, decode_times, fixed_global_camera_token):
        static_token_slice = self.decoded_static_query_tokens(fixed_queries)
        dynamic_token_slice = self.decoded_dynamic_query_tokens(fixed_queries)
        static_xyz, static_scales, static_quats, static_opacities, static_rgbs = self.static_gaussian_heads(
            static_token_slice
        )
        dynamic_bank = self.dynamic_gaussian_heads(dynamic_token_slice)

        decoded = []
        dynamic_opacity_frames = []
        for index in range(decode_times.shape[1]):
            dynamic_xyz, dynamic_scales, dynamic_quats, dynamic_opacities, dynamic_rgbs = self._eval_dynamic_bank(
                dynamic_bank,
                decode_times[:, index],
            )
            dynamic_opacity_frames.append(dynamic_opacities.squeeze(0))
            camera_queries = (
                self.refine_queries(video_tokens, decode_times[:, index])
                if self.camera_refine_with_decode_time
                else fixed_queries
            )
            camera, camera_state = self._decode_camera_single_time(
                camera_queries,
                decode_times[:, index],
                global_camera_token=fixed_global_camera_token,
            )
            decoded.append(
                (
                    torch.cat([static_xyz, dynamic_xyz], dim=1).squeeze(0),
                    torch.cat([static_scales, dynamic_scales], dim=1).squeeze(0),
                    torch.cat([static_quats, dynamic_quats], dim=1).squeeze(0),
                    torch.cat([static_opacities, dynamic_opacities], dim=1).squeeze(0),
                    torch.cat([static_rgbs, dynamic_rgbs], dim=1).squeeze(0),
                    camera,
                    camera_state,
                )
            )

        auxiliary = {
            "static_opacities": static_opacities,
            "dynamic_opacities": torch.stack(dynamic_opacity_frames, dim=0),
            "dynamic_A_mu": dynamic_bank.A_mu,
            "dynamic_A_rot": dynamic_bank.A_rot,
            "dynamic_A_alpha": dynamic_bank.A_alpha,
        }
        if self.token_layout is not None:
            auxiliary["token_layout_active_detail_level"] = torch.tensor(
                self.active_detail_level,
                device=static_opacities.device,
            )
        return self._merge_decoded_frames(decoded, auxiliary=auxiliary)

    def forward(self, video, decode_times, input_times=None):
        if decode_times.ndim != 2:
            raise ValueError(f"Expected decode_times of shape (B, T), got {tuple(decode_times.shape)}")
        precomputed_input = self.video_encoder_backend in {"precomputed", "precomputed_ltx"}
        if not precomputed_input:
            if video.ndim != 5:
                raise ValueError(f"Expected video of shape (B, T, C, H, W), got {tuple(video.shape)}")
            if video.shape[0] != 1:
                raise ValueError("DynamicVideoTokenGSImplicitCamera currently expects clip batch size 1.")
            if video.shape[1] != decode_times.shape[1]:
                raise ValueError("decode_times must have one value per frame in the clip.")
        if input_times is None:
            input_times = decode_times
        if input_times.ndim != 2:
            raise ValueError(f"Expected input_times of shape (B, T), got {tuple(input_times.shape)}")
        if not precomputed_input and video.shape[1] != input_times.shape[1]:
            raise ValueError("input_times must have one value per input frame in the clip.")

        video_tokens = self.video_encoder(video, frame_times=input_times)
        if video_tokens.shape[0] != 1:
            raise ValueError("DynamicVideoTokenGSImplicitCamera currently expects feature batch size 1.")
        fixed_camera_queries = self.refine_queries(video_tokens, decode_time=None)
        fixed_global_camera_token = fixed_camera_queries[:, 0, :]
        if self.use_static_dynamic_split:
            return self._decode_static_dynamic_split(
                video_tokens,
                fixed_camera_queries,
                decode_times,
                fixed_global_camera_token,
            )

        decoded = [
            self._decode_single_time(
                self.refine_queries(video_tokens, decode_times[:, index]),
                decode_times[:, index],
                global_camera_token=fixed_global_camera_token,
            )
            for index in range(decode_times.shape[1])
        ]
        return self._merge_decoded_frames(decoded)


class ResidualFreeBankVideoTokenGSImplicitCamera(DynamicVideoTokenGSImplicitCamera):
    """Video-conditioned residual decoder on top of per-token free Gaussian base parameters."""

    def __init__(
        self,
        *args,
        residual_output_init_std=1.0e-3,
        residual_xyz_raw_scale=0.5,
        residual_scale_log_scale=0.5,
        residual_rot_raw_scale=0.5,
        residual_opacity_logit_scale=1.0,
        residual_rgb_logit_scale=1.0,
        residual_head_input_norm="rmsnorm",
        **kwargs,
    ):
        num_tokens = int(kwargs.get("num_tokens", 8))
        feat_dim = int(kwargs.get("feat_dim", 128))
        gaussians_per_token = int(kwargs.get("gaussians_per_token", 64))
        scene_extent = float(kwargs.get("scene_extent", 1.0))
        xy_extent = kwargs.get("xy_extent", scene_extent)
        z_min = kwargs.get("z_min", -scene_extent)
        z_max = kwargs.get("z_max", scene_extent)
        scale_init = kwargs.get("scale_init", 0.05)
        scale_init_log_jitter = kwargs.get("scale_init_log_jitter", 0.0)
        opacity_init = kwargs.get("opacity_init")
        head_hidden_dim = kwargs.get("head_hidden_dim", 64)
        head_hidden_layers = kwargs.get("head_hidden_layers", 1)
        position_init_extent_coverage = kwargs.get("position_init_extent_coverage", 0.0)
        rotation_init = kwargs.get("rotation_init", "random")
        rgb_init = kwargs.get("rgb_init")
        rgb_init_min = kwargs.get("rgb_init_min", 0.01)
        rgb_init_max = kwargs.get("rgb_init_max", 0.99)
        feature_dim = int(kwargs.get("feature_dim", 3))
        super().__init__(*args, **kwargs)
        if self.use_static_dynamic_split:
            raise ValueError("Residual free-bank heads are not wired for static/dynamic split yet.")
        self.gaussian_heads = ResidualFreeGaussianParameterHeads(
            feat_dim=feat_dim,
            num_tokens=num_tokens,
            gaussians_per_token=gaussians_per_token,
            xy_extent=xy_extent,
            z_min=z_min,
            z_max=z_max,
            scale_init=scale_init,
            scale_init_log_jitter=scale_init_log_jitter,
            opacity_init=opacity_init,
            head_hidden_dim=head_hidden_dim,
            head_hidden_layers=head_hidden_layers,
            residual_output_init_std=residual_output_init_std,
            position_init_extent_coverage=position_init_extent_coverage,
            rotation_init=rotation_init,
            rgb_init=rgb_init,
            rgb_init_min=rgb_init_min,
            rgb_init_max=rgb_init_max,
            residual_xyz_raw_scale=residual_xyz_raw_scale,
            residual_scale_log_scale=residual_scale_log_scale,
            residual_rot_raw_scale=residual_rot_raw_scale,
            residual_opacity_logit_scale=residual_opacity_logit_scale,
            residual_rgb_logit_scale=residual_rgb_logit_scale,
            residual_head_input_norm=residual_head_input_norm,
            feature_dim=feature_dim,
        )


class DynamicVideoTokenGSKnownCamera(nn.Module):
    """Video-token Gaussian decoder that renders with supplied cameras.

    This is the no-implicit-camera control for DynamicVideoTokenGSImplicitCamera:
    it keeps the same video encoder, query decoder, Gaussian heads, and temporal
    conditioning, but removes the global/path camera tokens and camera heads.
    """

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
        xy_extent=None,
        z_min=None,
        z_max=None,
        scale_init=0.05,
        scale_init_log_jitter=0.0,
        opacity_init=None,
        query_token_init_std=0.02,
        head_hidden_dim=64,
        head_hidden_layers=1,
        head_output_init_std=None,
        position_init_extent_coverage=0.0,
        rotation_init="random",
        rgb_init=None,
        rgb_init_min=0.01,
        rgb_init_max=0.99,
        video_encoder_backend="local",
        tubelet_size=(4, 16, 16),
        encoder_self_attn_layers=1,
        bottleneck_self_attn_layers=4,
        vjepa_model_id=None,
        vjepa_feature_dim=None,
        vjepa_freeze=True,
        vjepa_attn_implementation="sdpa",
        vjepa_dtype="auto",
        vjepa_pretrained=True,
        vjepa_crop_size=None,
        vjepa_checkpoint_url=None,
        video_feature_layers=None,
        video_feature_channels=None,
        video_feature_token_stride=1,
        video_feature_output_dtype=None,
        cross_attn_layers=1,
        feature_dim=3,
        **_unused_camera_kwargs,
    ):
        super().__init__()
        self.clip_length = clip_length
        self.image_size = image_size
        self.num_tokens = int(num_tokens)
        self.total_tokens = self.num_tokens
        self.feat_dim = feat_dim
        self.gaussians_per_token = gaussians_per_token
        self.feature_dim = int(feature_dim)
        self.video_encoder_backend = str(video_encoder_backend).lower()
        self.video_encoder = build_video_encoder(
            self.video_encoder_backend,
            clip_length=clip_length,
            image_size=image_size,
            output_dim=feat_dim,
            bottleneck_dim=bottleneck_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            tubelet_size=tubelet_size,
            encoder_self_attn_layers=encoder_self_attn_layers,
            bottleneck_self_attn_layers=bottleneck_self_attn_layers,
            vjepa_model_id=vjepa_model_id,
            vjepa_feature_dim=vjepa_feature_dim,
            vjepa_freeze=vjepa_freeze,
            vjepa_attn_implementation=vjepa_attn_implementation,
            vjepa_dtype=vjepa_dtype,
            vjepa_pretrained=vjepa_pretrained,
            vjepa_crop_size=vjepa_crop_size,
            vjepa_checkpoint_url=vjepa_checkpoint_url,
            video_feature_layers=video_feature_layers,
            video_feature_channels=video_feature_channels,
            video_feature_token_stride=video_feature_token_stride,
            video_feature_output_dtype=video_feature_output_dtype,
        )
        self.query_tokens = LearnedQueryTokenBank(
            total_tokens=self.total_tokens,
            dim=feat_dim,
            init_std=query_token_init_std,
        )
        self.query_decoder_blocks = nn.ModuleList(
            [
                QueryCrossAttentionBlock(dim=feat_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
                for _ in range(cross_attn_layers)
            ]
        )
        if xy_extent is None:
            xy_extent = scene_extent
        if z_min is None:
            z_min = -scene_extent
        if z_max is None:
            z_max = scene_extent
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
            rgb_init=rgb_init,
            rgb_init_min=rgb_init_min,
            rgb_init_max=rgb_init_max,
            feature_dim=self.feature_dim,
        )
        self.time_proj = build_time_projector(1, feat_dim)
        self.head_time_proj = build_time_projector(1, feat_dim)

    def refine_queries(self, video_tokens, decode_time=None):
        batch_size = video_tokens.shape[0]
        queries = self.query_tokens(batch_size)
        if decode_time is not None:
            decode_time = decode_time.to(device=video_tokens.device, dtype=video_tokens.dtype).reshape(batch_size, 1)
            queries = queries + self.time_proj(decode_time).unsqueeze(1)
        for block in self.query_decoder_blocks:
            queries = block(queries, video_tokens)
        return queries

    def _decode_single_time(self, refined_queries, camera, decode_time=None):
        gs_tokens = refined_queries
        if decode_time is not None:
            decode_time = decode_time.to(device=refined_queries.device, dtype=refined_queries.dtype).reshape(
                refined_queries.shape[0], 1
            )
            gs_tokens = gs_tokens + self.head_time_proj(decode_time).unsqueeze(1)

        xyz, scales, quats, opacities, rgbs = self.gaussian_heads(gs_tokens)
        return (
            xyz.squeeze(0),
            scales.squeeze(0),
            quats.squeeze(0),
            opacities.squeeze(0),
            rgbs.squeeze(0),
            camera,
        )

    @staticmethod
    def _merge_decoded_frames(decoded):
        return GaussianSequence(
            xyz=torch.stack([item[0] for item in decoded], dim=0),
            scales=torch.stack([item[1] for item in decoded], dim=0),
            quats=torch.stack([item[2] for item in decoded], dim=0),
            opacities=torch.stack([item[3] for item in decoded], dim=0),
            rgbs=torch.stack([item[4] for item in decoded], dim=0),
            cameras=tuple(item[5] for item in decoded),
            camera_state=None,
        )

    def forward(self, video, decode_times, cameras, input_times=None):
        if cameras is None:
            raise ValueError("Known-camera video decode requires cameras.")
        if decode_times.ndim != 2:
            raise ValueError(f"Expected decode_times of shape (B, T), got {tuple(decode_times.shape)}")
        if video.ndim != 5:
            raise ValueError(f"Expected video of shape (B, T, C, H, W), got {tuple(video.shape)}")
        if video.shape[0] != 1:
            raise ValueError("DynamicVideoTokenGSKnownCamera currently expects clip batch size 1.")
        if video.shape[1] != decode_times.shape[1]:
            raise ValueError("decode_times must have one value per frame in the clip.")
        if len(cameras) != decode_times.shape[1]:
            raise ValueError(f"Expected {decode_times.shape[1]} cameras, got {len(cameras)}.")
        if input_times is None:
            input_times = decode_times
        if input_times.ndim != 2:
            raise ValueError(f"Expected input_times of shape (B, T), got {tuple(input_times.shape)}")
        if video.shape[1] != input_times.shape[1]:
            raise ValueError("input_times must have one value per input frame in the clip.")

        video_tokens = self.video_encoder(video, frame_times=input_times)
        if video_tokens.shape[0] != 1:
            raise ValueError("DynamicVideoTokenGSKnownCamera currently expects feature batch size 1.")
        decoded = [
            self._decode_single_time(
                self.refine_queries(video_tokens, decode_times[:, index]),
                cameras[index],
                decode_times[:, index],
            )
            for index in range(decode_times.shape[1])
        ]
        return self._merge_decoded_frames(decoded)


class DynamicVideoTokenGSImplicitCameraSinusoidalTime(DynamicVideoTokenGSImplicitCamera):
    def __init__(
        self,
        *args,
        time_fourier_bands=8,
        time_max_frequency=128.0,
        feat_dim=128,
        bottleneck_dim=256,
        max_rotation_degrees=5.0,
        max_translation_ratio=0.2,
        **kwargs,
    ):
        super().__init__(
            *args,
            feat_dim=feat_dim,
            bottleneck_dim=bottleneck_dim,
            max_rotation_degrees=max_rotation_degrees,
            max_translation_ratio=max_translation_ratio,
            **kwargs,
        )
        conditioner_kwargs = {
            "num_bands": time_fourier_bands,
            "max_frequency": time_max_frequency,
        }
        if isinstance(self.video_encoder, VideoEncoder):
            self.video_encoder.stage1_time_proj = SinusoidalTimeConditioner(1, feat_dim, **conditioner_kwargs)
            self.video_encoder.stage1_span_proj = SinusoidalTimeConditioner(2, feat_dim, **conditioner_kwargs)
            self.video_encoder.stage2_time_proj = SinusoidalTimeConditioner(1, bottleneck_dim, **conditioner_kwargs)
            self.video_encoder.stage2_span_proj = SinusoidalTimeConditioner(2, bottleneck_dim, **conditioner_kwargs)
        self.time_proj = SinusoidalTimeConditioner(1, feat_dim, **conditioner_kwargs)
        self.head_time_proj = SinusoidalTimeConditioner(1, feat_dim, **conditioner_kwargs)
        self.path_camera_head = TimeConditionedPathCameraHead(
            feat_dim=feat_dim,
            time_conditioner=SinusoidalTimeConditioner(1, feat_dim, **conditioner_kwargs),
            max_rotation_degrees=max_rotation_degrees,
            max_translation_ratio=max_translation_ratio,
        )

    def _decode_single_time(self, refined_queries, decode_time=None, global_camera_token=None):
        if global_camera_token is None:
            global_camera_token = refined_queries[:, 0, :]
        path_token = refined_queries[:, 1, :]
        gs_tokens = refined_queries[:, 2:, :]
        if decode_time is not None:
            decode_time = decode_time.to(device=refined_queries.device, dtype=refined_queries.dtype).reshape(
                refined_queries.shape[0], 1
            )
            gs_time_offset = self.head_time_proj(decode_time)
            gs_tokens = gs_tokens + gs_time_offset.unsqueeze(1)

        xyz, scales, quats, opacities, rgbs = self.gaussian_heads(gs_tokens)
        base_camera, base_state = self.global_camera_head(global_camera_token.squeeze(0), image_size=self.image_size)
        rotation_delta, translation_delta, path_residuals = self.path_camera_head(
            path_token,
            base_radius=base_state["radius"],
            decode_time=decode_time,
        )
        camera = compose_camera_with_se3_delta(base_camera, rotation_delta, translation_delta)[0]
        return (
            xyz.squeeze(0),
            scales.squeeze(0),
            quats.squeeze(0),
            opacities.squeeze(0),
            rgbs.squeeze(0),
            camera,
            CameraState(
                fov_degrees=base_state["fov_degrees"],
                radius=base_state["radius"],
                global_residuals=base_state["global_residuals"],
                rotation_delta=rotation_delta,
                translation_delta=translation_delta,
                path_residuals=path_residuals,
            ),
        )

    def _decode_camera_single_time(self, refined_queries, decode_time=None, global_camera_token=None):
        if global_camera_token is None:
            global_camera_token = refined_queries[:, 0, :]
        path_token = refined_queries[:, 1, :]
        if decode_time is not None:
            decode_time = decode_time.to(device=refined_queries.device, dtype=refined_queries.dtype).reshape(
                refined_queries.shape[0], 1
            )

        base_camera, base_state = self.global_camera_head(global_camera_token.squeeze(0), image_size=self.image_size)
        rotation_delta, translation_delta, path_residuals = self.path_camera_head(
            path_token,
            base_radius=base_state["radius"],
            decode_time=decode_time,
        )
        camera = compose_camera_with_se3_delta(base_camera, rotation_delta, translation_delta)[0]
        return (
            camera,
            CameraState(
                fov_degrees=base_state["fov_degrees"],
                radius=base_state["radius"],
                global_residuals=base_state["global_residuals"],
                rotation_delta=rotation_delta,
                translation_delta=translation_delta,
                path_residuals=path_residuals,
            ),
        )


class DynamicVideoTokenGSImplicitCameraPoseToPlucker(DynamicVideoTokenGSImplicitCameraSinusoidalTime):
    def __init__(
        self,
        *args,
        time_fourier_bands=8,
        time_max_frequency=128.0,
        feat_dim=128,
        bottleneck_dim=256,
        num_heads=8,
        mlp_ratio=4.0,
        max_rotation_degrees=5.0,
        max_translation_ratio=0.2,
        ray_condition_grid_size=16,
        **kwargs,
    ):
        super().__init__(
            *args,
            time_fourier_bands=time_fourier_bands,
            time_max_frequency=time_max_frequency,
            feat_dim=feat_dim,
            bottleneck_dim=bottleneck_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            max_rotation_degrees=max_rotation_degrees,
            max_translation_ratio=max_translation_ratio,
            **kwargs,
        )
        conditioner_kwargs = {
            "num_bands": time_fourier_bands,
            "max_frequency": time_max_frequency,
        }
        self.path_camera_head = TimeConditionedOpticalAxisCameraHead(
            feat_dim=feat_dim,
            time_conditioner=SinusoidalTimeConditioner(1, feat_dim, **conditioner_kwargs),
            max_rotation_degrees=max_rotation_degrees,
            max_translation_ratio=max_translation_ratio,
        )
        self.plucker_ray_conditioner = PluckerRayTokenConditioner(
            feat_dim=feat_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            ray_grid_size=ray_condition_grid_size,
        )

    def _decode_single_time(self, refined_queries, decode_time=None, global_camera_token=None):
        if global_camera_token is None:
            global_camera_token = refined_queries[:, 0, :]
        path_token = refined_queries[:, 1, :]
        gs_tokens = refined_queries[:, 2:, :]
        if decode_time is None:
            raise ValueError("decode_time is required for DynamicVideoTokenGSImplicitCameraPoseToPlucker.")
        decode_time = decode_time.to(device=refined_queries.device, dtype=refined_queries.dtype).reshape(
            refined_queries.shape[0], 1
        )
        gs_time_offset = self.head_time_proj(decode_time)
        gs_tokens = gs_tokens + gs_time_offset.unsqueeze(1)

        base_camera, base_state = self.global_camera_head(global_camera_token.squeeze(0), image_size=self.image_size)
        cameras, rotation_delta, translation_delta, path_residuals = self.path_camera_head(
            path_token,
            base_camera=base_camera,
            base_radius=base_state["radius"],
            decode_time=decode_time,
        )
        gs_tokens = self.plucker_ray_conditioner(gs_tokens, cameras, image_size=self.image_size)
        xyz, scales, quats, opacities, rgbs = self.gaussian_heads(gs_tokens)
        camera = cameras[0]
        return (
            xyz.squeeze(0),
            scales.squeeze(0),
            quats.squeeze(0),
            opacities.squeeze(0),
            rgbs.squeeze(0),
            camera,
            CameraState(
                fov_degrees=base_state["fov_degrees"],
                radius=base_state["radius"],
                global_residuals=base_state["global_residuals"],
                rotation_delta=rotation_delta,
                translation_delta=translation_delta,
                path_residuals=path_residuals,
            ),
        )
