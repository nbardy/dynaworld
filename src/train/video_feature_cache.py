from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

from config_utils import serialize_config_value
from gs_models.dynamic_video_token_gs_implicit_camera import (
    TorchHubVJEPAVideoEncoder,
    _as_feature_tokens,
    _first_parameter_dtype,
    _first_parameter_device,
    _resolve_vjepa_dtype,
)
from runtime_types import SequenceData


def _json_bytes(value: Any) -> bytes:
    return json.dumps(serialize_config_value(value), sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )


def _path_fingerprint(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    payload: dict[str, Any] = {"path": str(resolved)}
    if resolved.exists():
        stat = resolved.stat()
        payload.update({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return payload


def _sample_fingerprint(sequence_data: SequenceData) -> dict[str, Any]:
    frame_paths = [_path_fingerprint(path) for path in sequence_data.frame_paths]
    return {
        "source_path": _path_fingerprint(sequence_data.source_path),
        "frame_paths": frame_paths,
        "frame_source": sequence_data.frame_source,
        "frame_count": sequence_data.frame_count,
        "image_size": sequence_data.image_size,
        "selected_frame_count": sequence_data.selected_frame_count,
        "all_frame_count": sequence_data.all_frame_count,
    }


def _feature_fingerprint(feature_cfg: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "extractor",
        "model_id",
        "pipeline",
        "layers",
        "prompt",
        "negative_prompt",
        "height",
        "width",
        "num_frames",
        "num_inference_steps",
        "timesteps",
        "timestep",
        "guidance_scale",
        "guidance_scale_2",
        "conditioning_scale",
        "flow_shift",
        "output_type",
        "torch_dtype",
        "vae_torch_dtype",
        "mask_mode",
        "module_root",
        "max_sequence_length",
        "vjepa_feature_dim",
        "vjepa_freeze",
        "vjepa_attn_implementation",
        "vjepa_pretrained",
        "vjepa_crop_size",
        "vjepa_checkpoint_url",
        "rgb_pyramid_scales",
        "sample_cache_key",
        "cache_version",
    )
    return {key: feature_cfg.get(key) for key in keys if key in feature_cfg}


def sample_cache_key(sequence_data: SequenceData, feature_cfg: Mapping[str, Any]) -> str:
    """Return the per-sample feature-cache key.

    `features.sample_cache_key` is intentionally part of the hash, so changing a
    human-readable key such as "ltx-blocks-4-12-20-v2" busts every sample cache
    even if the source video did not change.
    """

    payload = {
        "sample": _sample_fingerprint(sequence_data),
        "features": _feature_fingerprint(feature_cfg),
    }
    return hashlib.sha256(_json_bytes(payload)).hexdigest()[:24]


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older torch compatibility.
        return torch.load(path, map_location="cpu")


def _dtype_from_name(name: str | None) -> torch.dtype:
    if name is None:
        return torch.float32
    normalized = str(name).lower()
    if normalized in {"float32", "fp32"}:
        return torch.float32
    if normalized in {"float16", "fp16"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"Unknown feature dtype {name!r}. Expected float32, float16, or bfloat16.")


def _move_payload(value: Any, device: torch.device | str) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: _move_payload(inner, device) for key, inner in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_payload(inner, device) for inner in value)
    if isinstance(value, list):
        return [_move_payload(inner, device) for inner in value]
    return value


def _cast_feature_payload(value: Any, dtype: torch.dtype) -> Any:
    if torch.is_tensor(value):
        if torch.is_floating_point(value):
            return value.to(dtype=dtype)
        return value
    if isinstance(value, dict):
        return {key: _cast_feature_payload(inner, dtype) for key, inner in value.items()}
    if isinstance(value, tuple):
        return tuple(_cast_feature_payload(inner, dtype) for inner in value)
    if isinstance(value, list):
        return [_cast_feature_payload(inner, dtype) for inner in value]
    return value


def infer_feature_channels(features: Mapping[str, torch.Tensor]) -> dict[str, int]:
    channels = {}
    for name, value in features.items():
        if not torch.is_tensor(value):
            raise TypeError(f"Expected tensor feature for {name!r}, got {type(value).__name__}.")
        if value.ndim in {2, 3}:
            channels[name] = int(value.shape[-1])
        elif value.ndim in {4, 5}:
            channels[name] = int(value.shape[1])
        else:
            raise ValueError(f"Cannot infer channel count for {name!r} with shape {tuple(value.shape)}.")
    return channels


def _sequence_to_pil_frames(sequence_data: SequenceData) -> list[Image.Image]:
    frames = sequence_data.frames.detach().cpu().float().clamp(0.0, 1.0)
    images = []
    for frame in frames:
        array = frame.permute(1, 2, 0).mul(255.0).round().to(torch.uint8).numpy()
        images.append(Image.fromarray(array, mode="RGB"))
    return images


def _first_tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, Mapping):
        for item in value.values():
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def _single_layer_name(feature_cfg: Mapping[str, Any], default: str) -> str:
    layers = feature_cfg.get("layers")
    if layers is None:
        return default
    if isinstance(layers, str):
        return layers
    if len(layers) < 1:
        return default
    return str(layers[0])


class RGBPyramidFeatureExtractor:
    """Local deterministic extractor for cache/trainer smoke tests."""

    def __init__(self, feature_cfg: Mapping[str, Any]):
        scales = feature_cfg.get("rgb_pyramid_scales") or [1, 2, 4]
        self.scales = [int(scale) for scale in scales]
        if any(scale < 1 for scale in self.scales):
            raise ValueError(f"rgb_pyramid_scales must be positive integers, got {self.scales}.")

    @torch.no_grad()
    def __call__(self, sequence_data: SequenceData) -> dict[str, torch.Tensor]:
        frames = sequence_data.frames.detach().cpu().float().clamp(0.0, 1.0)
        outputs = {}
        for scale in self.scales:
            if scale == 1:
                scaled = frames
            else:
                scaled = F.interpolate(
                    frames,
                    size=(max(1, frames.shape[-2] // scale), max(1, frames.shape[-1] // scale)),
                    mode="bilinear",
                    align_corners=False,
                )
            outputs[f"rgb_x{scale}"] = scaled.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()
        return outputs


class LTXVideoFeatureExtractor:
    """Hook-based LTX feature extractor.

    This intentionally depends on Diffusers only at runtime. The layer paths are
    config-owned because the exact LTX transformer block names are model-version
    dependent.
    """

    def __init__(self, feature_cfg: Mapping[str, Any], device: torch.device | str):
        self.feature_cfg = dict(feature_cfg)
        self.device = torch.device(device)
        self.pipeline = None
        self.pipeline_cls = None

    def _load_pipeline(self):
        if self.pipeline is not None:
            return self.pipeline
        try:
            from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXConditionPipeline
        except ImportError as exc:
            raise ImportError(
                "features.extractor='ltx' requires a Diffusers build with LTXConditionPipeline. "
                "Install diffusers plus its LTX dependencies, then rerun the feature bake."
            ) from exc

        dtype = _dtype_from_name(self.feature_cfg.get("torch_dtype", "float32"))
        model_id = self.feature_cfg.get("model_id", "Lightricks/LTX-Video-0.9.5")
        self.pipeline_cls = LTXConditionPipeline
        self.pipeline = LTXConditionPipeline.from_pretrained(model_id, torch_dtype=dtype)
        self.pipeline.to(self.device)
        if hasattr(self.pipeline, "set_progress_bar_config"):
            self.pipeline.set_progress_bar_config(disable=True)
        self.pipeline.eval() if hasattr(self.pipeline, "eval") else None
        return self.pipeline

    @staticmethod
    def _module_root(pipeline):
        if hasattr(pipeline, "transformer"):
            return pipeline.transformer
        if hasattr(pipeline, "unet"):
            return pipeline.unet
        raise AttributeError("Could not find pipeline.transformer or pipeline.unet for feature hooks.")

    def _resolve_layer(self, root, layer_path: str):
        try:
            return root.get_submodule(layer_path)
        except AttributeError:
            available = [name for name, _module in root.named_modules() if name][:30]
            raise KeyError(
                f"Could not resolve LTX feature layer {layer_path!r}. "
                f"First available module paths: {available}"
            ) from None

    def _call_pipeline(self, pipeline, frames: Sequence[Image.Image]) -> None:
        num_frames = int(self.feature_cfg.get("num_frames") or len(frames))
        height = int(self.feature_cfg.get("height") or frames[0].height)
        width = int(self.feature_cfg.get("width") or frames[0].width)
        kwargs = {
            "video": list(frames),
            "prompt": self.feature_cfg.get("prompt", ""),
            "negative_prompt": self.feature_cfg.get("negative_prompt"),
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "num_inference_steps": int(self.feature_cfg.get("num_inference_steps", 1)),
            "guidance_scale": float(self.feature_cfg.get("guidance_scale", 1.0)),
            "output_type": self.feature_cfg.get("output_type", "latent"),
            "return_dict": True,
        }
        timesteps = self.feature_cfg.get("timesteps")
        timestep = self.feature_cfg.get("timestep")
        if timesteps is not None:
            kwargs["timesteps"] = list(timesteps)
        elif timestep is not None:
            kwargs["timesteps"] = [int(timestep)]
        with torch.no_grad():
            pipeline(**kwargs)

    def __call__(self, sequence_data: SequenceData) -> dict[str, torch.Tensor]:
        pipeline = self._load_pipeline()
        root = self._module_root(pipeline)
        layers = [str(name) for name in self.feature_cfg.get("layers", [])]
        if not layers:
            raise ValueError("features.layers must name at least one LTX module path to cache.")

        features: dict[str, torch.Tensor] = {}
        hooks = []

        def hook_fn(name):
            def hook(_module, _inputs, output):
                tensor = _first_tensor(output)
                if tensor is None:
                    raise TypeError(f"LTX feature hook {name!r} did not return a tensor-like output.")
                features[name] = tensor.detach().cpu()

            return hook

        for name in layers:
            hooks.append(self._resolve_layer(root, name).register_forward_hook(hook_fn(name)))
        try:
            self._call_pipeline(pipeline, _sequence_to_pil_frames(sequence_data))
        finally:
            for hook in hooks:
                hook.remove()

        missing = [name for name in layers if name not in features]
        if missing:
            raise RuntimeError(f"LTX feature bake completed but did not capture layer(s): {missing}")
        return features


class WanVACEVideoFeatureExtractor:
    """Hook-based Wan-VACE feature extractor for editing-conditioned hidden states.

    The first experiment uses the source video as known conditioning everywhere
    via a black VACE mask, then caches selected transformer block activations.
    Layer paths are config-owned because Diffusers Wan/VACE module names are not
    stable enough to bake into the trainer.
    """

    def __init__(self, feature_cfg: Mapping[str, Any], device: torch.device | str):
        self.feature_cfg = dict(feature_cfg)
        self.device = torch.device(device)
        self.pipeline = None

    def _load_pipeline(self):
        if self.pipeline is not None:
            return self.pipeline
        try:
            from diffusers import AutoencoderKLWan, WanVACEPipeline
            from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
        except ImportError:
            try:
                from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
                from diffusers.pipelines.wan.pipeline_wan_vace import WanVACEPipeline
                from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
            except ImportError as exc:
                raise ImportError(
                    "features.extractor='wan_vace' requires a Diffusers build with WanVACEPipeline "
                    "and AutoencoderKLWan. Install latest diffusers plus Wan dependencies, then rerun "
                    "the feature bake."
                ) from exc

        model_id = self.feature_cfg.get("model_id") or "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
        dtype = _dtype_from_name(self.feature_cfg.get("torch_dtype", "float32"))
        vae_dtype = _dtype_from_name(self.feature_cfg.get("vae_torch_dtype", "float32"))

        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=vae_dtype)
        self.pipeline = WanVACEPipeline.from_pretrained(model_id, vae=vae, torch_dtype=dtype)

        flow_shift = self.feature_cfg.get("flow_shift")
        if flow_shift is not None:
            self.pipeline.scheduler = UniPCMultistepScheduler.from_config(
                self.pipeline.scheduler.config,
                flow_shift=float(flow_shift),
            )

        self.pipeline.to(self.device)
        if hasattr(self.pipeline, "set_progress_bar_config"):
            self.pipeline.set_progress_bar_config(disable=True)
        return self.pipeline

    def _module_root(self, pipeline):
        root_name = str(self.feature_cfg.get("module_root") or "transformer")
        if hasattr(pipeline, root_name) and getattr(pipeline, root_name) is not None:
            return getattr(pipeline, root_name)
        available = [
            name
            for name in ("transformer", "transformer_2", "unet")
            if hasattr(pipeline, name) and getattr(pipeline, name) is not None
        ]
        raise AttributeError(
            f"Could not find Wan-VACE module root {root_name!r}. Available roots: {available}."
        )

    def _resolve_layer(self, root, layer_path: str):
        try:
            return root.get_submodule(layer_path)
        except AttributeError:
            available = [name for name, _module in root.named_modules() if name][:30]
            raise KeyError(
                f"Could not resolve Wan-VACE feature layer {layer_path!r}. "
                f"First available module paths: {available}"
            ) from None

    def _known_everywhere_mask(self, width: int, height: int, num_frames: int) -> list[Image.Image]:
        mask_mode = str(self.feature_cfg.get("mask_mode") or "known").lower()
        if mask_mode not in {"known", "known_everywhere", "black"}:
            raise ValueError(
                f"Unsupported Wan-VACE mask_mode={mask_mode!r}. "
                "The feature-cache path currently supports only known/black masks."
            )
        return [Image.new("L", (width, height), 0) for _ in range(num_frames)]

    def _call_pipeline(self, pipeline, frames: Sequence[Image.Image]) -> None:
        num_frames = int(self.feature_cfg.get("num_frames") or len(frames))
        height = int(self.feature_cfg.get("height") or frames[0].height)
        width = int(self.feature_cfg.get("width") or frames[0].width)
        if len(frames) < num_frames:
            raise ValueError(
                f"Wan-VACE feature bake requested {num_frames} frames but sample only has {len(frames)}."
            )
        resized_frames = [frame.resize((width, height)) for frame in list(frames)[:num_frames]]
        mask = self._known_everywhere_mask(width, height, num_frames)
        kwargs: dict[str, Any] = {
            "prompt": self.feature_cfg.get("prompt", ""),
            "negative_prompt": self.feature_cfg.get("negative_prompt"),
            "video": resized_frames,
            "mask": mask,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "num_inference_steps": int(self.feature_cfg.get("num_inference_steps", 1)),
            "guidance_scale": float(self.feature_cfg.get("guidance_scale", 1.0)),
            "conditioning_scale": self.feature_cfg.get("conditioning_scale", 1.0),
            "output_type": self.feature_cfg.get("output_type", "latent"),
            "return_dict": True,
        }
        guidance_scale_2 = self.feature_cfg.get("guidance_scale_2")
        if guidance_scale_2 is not None:
            kwargs["guidance_scale_2"] = float(guidance_scale_2)
        max_sequence_length = self.feature_cfg.get("max_sequence_length")
        if max_sequence_length is not None:
            kwargs["max_sequence_length"] = int(max_sequence_length)
        with torch.no_grad():
            pipeline(**kwargs)

    def __call__(self, sequence_data: SequenceData) -> dict[str, torch.Tensor]:
        pipeline = self._load_pipeline()
        root = self._module_root(pipeline)
        layers = [str(name) for name in self.feature_cfg.get("layers", [])]
        if not layers:
            raise ValueError("features.layers must name at least one Wan-VACE module path to cache.")

        features: dict[str, torch.Tensor] = {}
        hooks = []

        def hook_fn(name):
            def hook(_module, _inputs, output):
                tensor = _first_tensor(output)
                if tensor is None:
                    raise TypeError(f"Wan-VACE feature hook {name!r} did not return a tensor-like output.")
                features[name] = tensor.detach().cpu()

            return hook

        for name in layers:
            hooks.append(self._resolve_layer(root, name).register_forward_hook(hook_fn(name)))
        try:
            self._call_pipeline(pipeline, _sequence_to_pil_frames(sequence_data))
        finally:
            for hook in hooks:
                hook.remove()

        missing = [name for name in layers if name not in features]
        if missing:
            raise RuntimeError(f"Wan-VACE feature bake completed but did not capture layer(s): {missing}")
        return features


class HuggingFaceVJEPAFeatureExtractor:
    """Frozen HF V-JEPA feature extractor for the shared disk cache."""

    def __init__(self, feature_cfg: Mapping[str, Any], device: torch.device | str):
        try:
            from transformers import AutoModel, AutoVideoProcessor
        except ImportError as exc:
            raise ImportError(
                "features.extractor='vjepa_hf' requires transformers with V-JEPA support."
            ) from exc

        self.feature_cfg = dict(feature_cfg)
        self.device = torch.device(device)
        self.model_id = self.feature_cfg.get("model_id") or "facebook/vjepa2-vitl-fpc64-256"
        self.processor = AutoVideoProcessor.from_pretrained(self.model_id)
        model_kwargs = {}
        load_dtype = _resolve_vjepa_dtype(self.feature_cfg.get("torch_dtype", "auto"))
        if load_dtype is not None:
            model_kwargs["torch_dtype"] = load_dtype
        attn_implementation = self.feature_cfg.get("vjepa_attn_implementation", "sdpa")
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        try:
            self.encoder = AutoModel.from_pretrained(self.model_id, **model_kwargs)
        except TypeError:
            if "torch_dtype" not in model_kwargs:
                raise
            model_kwargs["dtype"] = model_kwargs.pop("torch_dtype")
            self.encoder = AutoModel.from_pretrained(self.model_id, **model_kwargs)
        self.encoder.to(self.device)
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.feature_dim = self.feature_cfg.get("vjepa_feature_dim")
        if self.feature_dim is not None:
            self.feature_dim = int(self.feature_dim)
        self.layer_name = _single_layer_name(self.feature_cfg, "vjepa_tokens")

    @staticmethod
    def _processor_video(frames):
        video = frames.detach().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).cpu()
        return video

    def _processor_inputs(self, sequence_data: SequenceData):
        processed = self.processor(
            self._processor_video(sequence_data.frames),
            return_tensors="pt",
        )
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

    @torch.no_grad()
    def __call__(self, sequence_data: SequenceData) -> dict[str, torch.Tensor]:
        features = self._extract_features(self._processor_inputs(sequence_data))
        tokens = _as_feature_tokens(features, feature_dim=self.feature_dim)
        return {self.layer_name: tokens.detach().cpu()}


class TorchHubVJEPAFeatureExtractor:
    """Frozen torchhub V-JEPA 2.x feature extractor for the shared disk cache."""

    def __init__(self, feature_cfg: Mapping[str, Any], device: torch.device | str):
        self.feature_cfg = dict(feature_cfg)
        self.device = torch.device(device)
        self.model_id = self.feature_cfg.get("model_id") or "vjepa2_1_vit_base_384"
        self.feature_dim = self.feature_cfg.get("vjepa_feature_dim")
        if self.feature_dim is not None:
            self.feature_dim = int(self.feature_dim)
        self.layer_name = _single_layer_name(self.feature_cfg, "vjepa_tokens")
        self.encoder_wrapper = TorchHubVJEPAVideoEncoder(
            output_dim=max(1, int(self.feature_dim or 1)),
            model_id=self.model_id,
            feature_dim=self.feature_dim,
            freeze=True,
            pretrained=bool(self.feature_cfg.get("vjepa_pretrained", True)),
            crop_size=self.feature_cfg.get("vjepa_crop_size"),
            checkpoint_url=self.feature_cfg.get("vjepa_checkpoint_url"),
        ).to(self.device)
        self.encoder_wrapper.eval()

    @torch.no_grad()
    def __call__(self, sequence_data: SequenceData) -> dict[str, torch.Tensor]:
        video = sequence_data.frames.unsqueeze(0).to(self.device)
        inputs = self.encoder_wrapper._preprocess(video)
        encoder_device = _first_parameter_device(self.encoder_wrapper.encoder)
        encoder_dtype = _first_parameter_dtype(self.encoder_wrapper.encoder)
        inputs = inputs.to(device=encoder_device, dtype=encoder_dtype)
        features = self.encoder_wrapper.encoder(inputs)
        tokens = _as_feature_tokens(features, feature_dim=self.encoder_wrapper.feature_dim)
        return {self.layer_name: tokens.detach().cpu()}


def build_feature_extractor(feature_cfg: Mapping[str, Any], device: torch.device | str):
    extractor = str(feature_cfg.get("extractor", "ltx")).lower()
    if extractor == "ltx":
        return LTXVideoFeatureExtractor(feature_cfg, device=device)
    if extractor in {"wan_vace", "wan2_1_vace", "vace_wan"}:
        return WanVACEVideoFeatureExtractor(feature_cfg, device=device)
    if extractor == "vjepa_hf":
        return HuggingFaceVJEPAFeatureExtractor(feature_cfg, device=device)
    if extractor in {"vjepa_torchhub", "vjepa2_torchhub", "vjepa2_1_torchhub"}:
        return TorchHubVJEPAFeatureExtractor(feature_cfg, device=device)
    if extractor == "rgb_pyramid":
        return RGBPyramidFeatureExtractor(feature_cfg)
    raise ValueError(
        f"Unknown features.extractor={extractor!r}. "
        "Expected ltx, wan_vace, vjepa_hf, vjepa_torchhub, or rgb_pyramid."
    )


class VideoFeatureCache:
    def __init__(self, feature_cfg: Mapping[str, Any], device: torch.device | str):
        self.feature_cfg = dict(feature_cfg)
        self.device = torch.device(device)
        self.cache_dir = Path(self.feature_cfg["cache_dir"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.force_rebake = bool(self.feature_cfg.get("force_rebake", False))
        self.keep_in_memory = bool(self.feature_cfg.get("keep_in_memory", True))
        self.save_dtype = _dtype_from_name(self.feature_cfg.get("save_dtype", "float16"))
        self.extractor = build_feature_extractor(self.feature_cfg, device=self.device)
        self._memory: dict[str, Any] = {}

    def cache_key(self, sequence_data: SequenceData) -> str:
        return sample_cache_key(sequence_data, self.feature_cfg)

    def cache_path(self, sequence_data: SequenceData) -> Path:
        return self.cache_dir / f"{self.cache_key(sequence_data)}.pt"

    def _load_cached(self, path: Path, key: str) -> Mapping[str, torch.Tensor]:
        payload = _torch_load(path)
        if payload.get("sample_cache_key") != key:
            raise ValueError(f"Feature cache key mismatch in {path}.")
        features = payload["features"]
        if not isinstance(features, Mapping):
            raise TypeError(f"Expected cached features mapping in {path}.")
        return features

    def _save_cached(self, path: Path, key: str, sequence_data: SequenceData, features: Mapping[str, torch.Tensor]):
        payload = {
            "sample_cache_key": key,
            "feature_config": _feature_fingerprint(self.feature_cfg),
            "sample": _sample_fingerprint(sequence_data),
            "features": _cast_feature_payload(dict(features), self.save_dtype),
        }
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp_path)
        tmp_path.replace(path)

    def load_or_bake(self, sequence_data: SequenceData) -> Mapping[str, torch.Tensor]:
        key = self.cache_key(sequence_data)
        if self.keep_in_memory and key in self._memory:
            return self._memory[key]

        path = self.cache_path(sequence_data)
        if path.exists() and not self.force_rebake:
            print(f"[features] cache hit {path}")
            features = self._load_cached(path, key)
        else:
            print(f"[features] cache miss; baking {path}")
            features = self.extractor(sequence_data)
            self._save_cached(path, key, sequence_data, features)

        moved = _move_payload(features, self.device)
        if self.keep_in_memory:
            self._memory[key] = moved
        return moved

    def prebake(self, sequences: Sequence[SequenceData]) -> None:
        seen = set()
        for sequence_data in sequences:
            key = self.cache_key(sequence_data)
            if key in seen:
                continue
            seen.add(key)
            self.load_or_bake(sequence_data)

    def infer_channels(self, sequence_data: SequenceData) -> dict[str, int]:
        features = self.load_or_bake(sequence_data)
        cpu_features = {name: value.detach().cpu() for name, value in features.items()}
        return infer_feature_channels(cpu_features)
