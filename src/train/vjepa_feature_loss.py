from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

from gs_models.dynamic_video_token_gs_implicit_camera import (
    VJEPA_TORCHHUB_CHECKPOINTS,
    _as_feature_tokens,
    _first_parameter_device,
    _first_parameter_dtype,
    _infer_encoder_feature_dim,
    _load_torchhub_encoder_checkpoint,
    _resolve_vjepa_dtype,
)


class TorchHubVJEPAFeatureLoss(nn.Module):
    """Frozen V-JEPA feature distance that keeps gradients to predicted video.

    This differs from the conditioning encoders, which intentionally detach input
    videos and run under no_grad. For a perceptual loss the encoder weights stay
    frozen, but the predicted video path must remain differentiable.
    """

    def __init__(
        self,
        *,
        model_id: str = "vjepa2_1_vit_base_384",
        crop_size: int | None = None,
        dtype: str | None = "auto",
        checkpoint_url: str | None = None,
        checkpoint_key: str | None = None,
        temporal_stride: int = 1,
        normalize_features: bool = True,
        loss_type: str = "mse",
    ) -> None:
        super().__init__()
        self.model_id = str(model_id)
        self.crop_size = int(crop_size or (384 if "384" in self.model_id else 256))
        self.temporal_stride = max(1, int(temporal_stride))
        self.normalize_features = bool(normalize_features)
        self.loss_type = str(loss_type).lower()
        if self.loss_type not in {"mse", "l1", "smooth_l1"}:
            raise ValueError("V-JEPA feature loss_type must be one of: mse, l1, smooth_l1.")

        load_weights_after_init = bool(checkpoint_url is not None or self.model_id in VJEPA_TORCHHUB_CHECKPOINTS)
        hub_pretrained = not load_weights_after_init
        try:
            try:
                loaded = torch.hub.load("facebookresearch/vjepa2", self.model_id, pretrained=hub_pretrained)
            except TypeError:
                loaded = torch.hub.load("facebookresearch/vjepa2", self.model_id)
        except ImportError as exc:
            raise ImportError(
                "V-JEPA feature loss requires the V-JEPA 2 torchhub dependencies "
                "(notably timm and einops)."
            ) from exc
        self.encoder = loaded[0] if isinstance(loaded, (tuple, list)) else loaded
        if load_weights_after_init:
            _load_torchhub_encoder_checkpoint(
                self.encoder,
                model_id=self.model_id,
                checkpoint_url=checkpoint_url,
                checkpoint_key=checkpoint_key,
            )
        load_dtype = _resolve_vjepa_dtype(dtype)
        if load_dtype is not None:
            self.encoder.to(dtype=load_dtype)
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

        feature_dim = _infer_encoder_feature_dim(self.encoder)
        if feature_dim is None:
            raise ValueError(f"Could not infer hidden size for V-JEPA feature loss model {self.model_id!r}.")
        self.feature_dim = int(feature_dim)

        mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 1, 3, 1, 1)
        self.register_buffer("imagenet_mean", mean, persistent=False)
        self.register_buffer("imagenet_std", std, persistent=False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def _video_batch(self, video: torch.Tensor) -> torch.Tensor:
        if video.dim() == 4:
            video = video.unsqueeze(0)
        if video.dim() != 5 or video.shape[2] != 3:
            raise ValueError(f"V-JEPA feature loss expects [T,3,H,W] or [B,T,3,H,W], got {tuple(video.shape)}")
        if self.temporal_stride > 1:
            video = video[:, :: self.temporal_stride]
        return video

    def _preprocess(self, video: torch.Tensor) -> torch.Tensor:
        batch_size, frame_count, channels, height, width = video.shape
        video = video.float().clamp(0.0, 1.0)
        if (height, width) != (self.crop_size, self.crop_size):
            flat = video.reshape(batch_size * frame_count, channels, height, width)
            flat = F.interpolate(
                flat,
                size=(self.crop_size, self.crop_size),
                mode="bilinear",
                align_corners=False,
            )
            video = flat.reshape(batch_size, frame_count, channels, self.crop_size, self.crop_size)
        video = (video - self.imagenet_mean.to(device=video.device, dtype=video.dtype)) / self.imagenet_std.to(
            device=video.device,
            dtype=video.dtype,
        )
        return video.permute(0, 2, 1, 3, 4)

    def encode(self, video: torch.Tensor, *, input_grad: bool) -> torch.Tensor:
        video = self._video_batch(video)
        encoder_device = _first_parameter_device(self.encoder)
        encoder_dtype = _first_parameter_dtype(self.encoder)
        inputs = self._preprocess(video).to(device=encoder_device, dtype=encoder_dtype)
        context = nullcontext() if input_grad else torch.no_grad()
        with context:
            features = self.encoder(inputs)
        tokens = _as_feature_tokens(features, feature_dim=self.feature_dim).float()
        if self.normalize_features:
            tokens = F.normalize(tokens, dim=-1)
        return tokens

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_tokens = self.encode(prediction, input_grad=True)
        target_tokens = self.encode(target.detach(), input_grad=False)
        if pred_tokens.shape != target_tokens.shape:
            raise ValueError(
                "V-JEPA feature token shape mismatch: "
                f"{tuple(pred_tokens.shape)} vs {tuple(target_tokens.shape)}"
            )
        target_tokens = target_tokens.to(device=pred_tokens.device, dtype=pred_tokens.dtype)
        if self.loss_type == "mse":
            return F.mse_loss(pred_tokens, target_tokens)
        if self.loss_type == "l1":
            return F.l1_loss(pred_tokens, target_tokens)
        return F.smooth_l1_loss(pred_tokens, target_tokens)
