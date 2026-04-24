import math
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from camera import build_plucker_ray_grid, build_plucker_ray_grid_batch
from runtime_types import CameraState, GaussianSequence

from .blocks import GaussianParameterHeads, build_mlp, flatten_hw_features
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


class PrecomputedVideoFeatureAdapter(nn.Module):
    """Project cached native feature tensors into the query decoder memory.

    The adapter deliberately keeps each layer at its native token/grid
    resolution. It flattens native maps into memory tokens and lets the Gaussian
    query decoder attend across the resulting multi-scale feature set.
    """

    def __init__(self, output_dim, feature_channels, feature_layers=None):
        super().__init__()
        if not isinstance(feature_channels, Mapping) or not feature_channels:
            raise ValueError("video_feature_channels must be a non-empty mapping for precomputed features.")
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
            tokens = proj(self.input_norms[safe_name](tokens))
            tokens = tokens + self.layer_embeddings[safe_name].view(1, 1, -1)
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
                "Install the current V-JEPA-capable build with: "
                "pip install -U git+https://github.com/huggingface/transformers"
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
            crop_size=vjepa_crop_size,
            checkpoint_url=vjepa_checkpoint_url,
        )
    if backend in {"precomputed", "precomputed_ltx"}:
        return PrecomputedVideoFeatureAdapter(
            output_dim=output_dim,
            feature_channels=video_feature_channels,
            feature_layers=video_feature_layers,
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
        motion_extent=0.25,
        rotation_degrees=10.0,
        alpha_logit_extent=2.0,
        coeff_output_init_std=1.0e-4,
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
        cross_attn_layers=1,
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
    ):
        super().__init__()
        self.clip_length = clip_length
        self.image_size = image_size
        self.num_tokens = int(num_tokens)
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
        self.feat_dim = feat_dim
        self.gaussians_per_token = gaussians_per_token
        self.dynamic_time_basis_count = int(dynamic_time_basis_count)
        self.dynamic_time_max_frequency = float(dynamic_time_max_frequency)
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
            camera, camera_state = self._decode_camera_single_time(
                self.refine_queries(video_tokens, decode_times[:, index]),
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

        return self._merge_decoded_frames(
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
