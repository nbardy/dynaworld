from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch.nn as nn

from colorize import FeatureToColor, normalize_view_condition
from gs_models.dynamic_video_token_gs_implicit_camera import (
    DynamicVideoTokenGSKnownCamera,
    DynamicVideoTokenGSImplicitCamera,
    DynamicVideoTokenGSImplicitCameraPoseToPlucker,
    DynamicVideoTokenGSImplicitCameraSinusoidalTime,
    FreeGaussianBankImplicitCamera,
    LinearTimeFreeGaussianBankImplicitCamera,
    ResidualFreeBankVideoTokenGSImplicitCamera,
    UnconditionedResidualFreeBankImplicitCamera,
    UnconditionedTokenGSImplicitCamera,
)


class ModelFactoryConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ColorizeFactoryResult:
    module: FeatureToColor | None
    view_condition: str
    detach_view_condition: bool


MODEL_DEFAULTS: dict[str, Any] = {
    "variant": "learned_time_orbit_path",
    "feature_dim": 3,
    "xy_extent": None,
    "z_min": None,
    "z_max": None,
    "scale_init": 0.05,
    "scale_init_log_jitter": 0.0,
    "opacity_init": None,
    "query_token_init_std": 0.02,
    "head_hidden_dim": 64,
    "head_hidden_layers": 1,
    "head_output_init_std": None,
    "position_init_extent_coverage": 0.0,
    "rotation_init": "random",
    "rgb_init": None,
    "rgb_init_min": 0.01,
    "rgb_init_max": 0.99,
    "video_encoder_backend": "local",
    "vjepa_model_id": None,
    "vjepa_feature_dim": None,
    "vjepa_freeze": True,
    "vjepa_attn_implementation": "sdpa",
    "vjepa_dtype": "auto",
    "vjepa_pretrained": True,
    "vjepa_crop_size": None,
    "vjepa_checkpoint_url": None,
    "video_feature_layers": None,
    "video_feature_channels": None,
    "video_feature_token_stride": 1,
    "video_feature_output_dtype": None,
    "camera_refine_with_decode_time": True,
    "time_fourier_bands": 8,
    "time_max_frequency": 128.0,
    "ray_condition_grid_size": 16,
    "static_tokens": None,
    "dynamic_tokens": None,
    "dynamic_time_basis_count": 8,
    "dynamic_time_max_frequency": 8.0,
    "dynamic_motion_extent": None,
    "dynamic_rotation_degrees": 10.0,
    "dynamic_alpha_logit_extent": 2.0,
    "dynamic_coeff_output_init_std": 1.0e-4,
    "free_frame_count": None,
    "free_time_interpolation": "nearest",
    "free_velocity_extent": 1.0,
    "free_velocity_init_std": 0.0,
    "free_opacity_slope_init_std": 0.0,
    "free_time_center": 0.5,
    "residual_output_init_std": 1.0e-3,
    "residual_xyz_raw_scale": 0.5,
    "residual_scale_log_scale": 0.5,
    "residual_rot_raw_scale": 0.5,
    "residual_opacity_logit_scale": 1.0,
    "residual_rgb_logit_scale": 1.0,
    "residual_head_input_norm": "rmsnorm",
}

CAMERA_DEFAULTS: dict[str, Any] = {
    "global_head": "legacy_orbit",
    "lens_model": "pinhole",
    "base_fov_degrees": 60.0,
    "base_radius": 3.0,
    "max_fov_delta_degrees": 15.0,
    "max_radius_scale": 1.5,
    "max_aspect_log_delta": 0.0,
    "max_principal_point_delta": 0.0,
    "distortion_max_abs": 0.0,
    "base_distortion": None,
    "max_rotation_degrees": 5.0,
    "max_translation_ratio": 0.2,
}

REQUIRED_MODEL_KEYS = frozenset(
    {
        "size",
        "train_frame_count",
        "tokens",
        "model_dim",
        "bottleneck_dim",
        "num_heads",
        "mlp_ratio",
        "gaussians_per_token",
        "scene_extent",
        "tubelet_size_t",
        "patch_compression",
        "encoder_self_attn_layers",
        "bottleneck_self_attn_layers",
        "cross_attn_layers",
    }
)

RIG_CAMERA_KEYS = frozenset(
    {
        "rig_anchor_policy",
        "rig_init",
        "rig_learn_global_se3",
        "rig_learn_per_camera_se3",
        "rig_radius",
        "rig_regularization_weight",
        "rig_rotation_degrees",
        "rig_translation_ratio",
    }
)

KNOWN_MODEL_CONFIG_KEYS = frozenset(MODEL_DEFAULTS) | REQUIRED_MODEL_KEYS | frozenset({"use_static_dynamic_split"})
KNOWN_CAMERA_CONFIG_KEYS = frozenset(CAMERA_DEFAULTS) | RIG_CAMERA_KEYS
COLORIZE_CONFIG_KEYS = frozenset(
    {
        "hidden_dim",
        "activation",
        "pre_norm",
        "weight_init",
        "weight_init_gain",
        "view_condition",
        "detach_view_condition",
    }
)

MODEL_VARIANT_CLASSES: dict[str, type[nn.Module]] = {
    "learned_time_orbit_path": DynamicVideoTokenGSImplicitCamera,
    "free_splats": FreeGaussianBankImplicitCamera,
    "free_linear_time_splats": LinearTimeFreeGaussianBankImplicitCamera,
    "residual_free_bank": ResidualFreeBankVideoTokenGSImplicitCamera,
    "known_camera": DynamicVideoTokenGSKnownCamera,
    "sinusoidal_time_path_mlp": DynamicVideoTokenGSImplicitCameraSinusoidalTime,
    "token_to_pose_to_plucker": DynamicVideoTokenGSImplicitCameraPoseToPlucker,
    "unconditioned_tokens": UnconditionedTokenGSImplicitCamera,
    "unconditioned_residual_free_bank": UnconditionedResidualFreeBankImplicitCamera,
}

KNOWN_CAMERA_VARIANTS = frozenset({"known_camera"})
UNCONDITIONED_VARIANTS = frozenset({"unconditioned_tokens"})
UNCONDITIONED_RESIDUAL_VARIANTS = frozenset({"unconditioned_residual_free_bank"})
FREE_BANK_VARIANTS = frozenset({"free_splats"})
LINEAR_FREE_BANK_VARIANTS = frozenset({"free_linear_time_splats"})
RESIDUAL_VARIANTS = frozenset(
    {
        "residual_free_bank",
        "unconditioned_residual_free_bank",
    }
)

ArgSource = str | tuple[str, str] | Callable[[Mapping[str, Any], Mapping[str, Any]], Any]

BASE_VIDEO_ARGS: dict[str, ArgSource] = {
    "clip_length": "train_frame_count",
    "image_size": "size",
    "num_tokens": "tokens",
    "feature_dim": "feature_dim",
    "feat_dim": "model_dim",
    "bottleneck_dim": "bottleneck_dim",
    "num_heads": "num_heads",
    "mlp_ratio": "mlp_ratio",
    "gaussians_per_token": "gaussians_per_token",
    "scene_extent": "scene_extent",
    "xy_extent": "xy_extent",
    "z_min": "z_min",
    "z_max": "z_max",
    "scale_init": "scale_init",
    "scale_init_log_jitter": "scale_init_log_jitter",
    "opacity_init": "opacity_init",
    "query_token_init_std": "query_token_init_std",
    "head_hidden_dim": "head_hidden_dim",
    "head_hidden_layers": "head_hidden_layers",
    "head_output_init_std": "head_output_init_std",
    "position_init_extent_coverage": "position_init_extent_coverage",
    "rotation_init": "rotation_init",
    "rgb_init": "rgb_init",
    "rgb_init_min": "rgb_init_min",
    "rgb_init_max": "rgb_init_max",
    "video_encoder_backend": "video_encoder_backend",
    "tubelet_size": lambda model, _camera: (
        model["tubelet_size_t"],
        model["patch_compression"],
        model["patch_compression"],
    ),
    "encoder_self_attn_layers": "encoder_self_attn_layers",
    "bottleneck_self_attn_layers": "bottleneck_self_attn_layers",
    "vjepa_model_id": "vjepa_model_id",
    "vjepa_feature_dim": "vjepa_feature_dim",
    "vjepa_freeze": "vjepa_freeze",
    "vjepa_attn_implementation": "vjepa_attn_implementation",
    "vjepa_dtype": "vjepa_dtype",
    "vjepa_pretrained": "vjepa_pretrained",
    "vjepa_crop_size": "vjepa_crop_size",
    "vjepa_checkpoint_url": "vjepa_checkpoint_url",
    "video_feature_layers": "video_feature_layers",
    "video_feature_channels": "video_feature_channels",
    "video_feature_token_stride": "video_feature_token_stride",
    "video_feature_output_dtype": "video_feature_output_dtype",
    "cross_attn_layers": "cross_attn_layers",
    "camera_refine_with_decode_time": "camera_refine_with_decode_time",
}

DYNAMIC_SPLIT_ARGS: dict[str, ArgSource] = {
    "static_tokens": "static_tokens",
    "dynamic_tokens": "dynamic_tokens",
    "dynamic_time_basis_count": "dynamic_time_basis_count",
    "dynamic_time_max_frequency": "dynamic_time_max_frequency",
    "dynamic_motion_extent": "dynamic_motion_extent",
    "dynamic_rotation_degrees": "dynamic_rotation_degrees",
    "dynamic_alpha_logit_extent": "dynamic_alpha_logit_extent",
    "dynamic_coeff_output_init_std": "dynamic_coeff_output_init_std",
}

CAMERA_ARGS: dict[str, ArgSource] = {
    "base_fov_degrees": ("camera", "base_fov_degrees"),
    "base_radius": ("camera", "base_radius"),
    "max_fov_delta_degrees": ("camera", "max_fov_delta_degrees"),
    "max_radius_scale": ("camera", "max_radius_scale"),
    "camera_global_head": ("camera", "global_head"),
    "lens_model": ("camera", "lens_model"),
    "max_aspect_log_delta": ("camera", "max_aspect_log_delta"),
    "max_principal_point_delta": ("camera", "max_principal_point_delta"),
    "distortion_max_abs": ("camera", "distortion_max_abs"),
    "base_distortion": ("camera", "base_distortion"),
    "max_rotation_degrees": ("camera", "max_rotation_degrees"),
    "max_translation_ratio": ("camera", "max_translation_ratio"),
}

FREE_BANK_ARGS: dict[str, ArgSource] = {
    "clip_length": "train_frame_count",
    "image_size": "size",
    "num_tokens": "tokens",
    "feat_dim": "model_dim",
    "gaussians_per_token": "gaussians_per_token",
    "scene_extent": "scene_extent",
    "xy_extent": "xy_extent",
    "z_min": "z_min",
    "z_max": "z_max",
    "scale_init": "scale_init",
    "scale_init_log_jitter": "scale_init_log_jitter",
    "opacity_init": "opacity_init",
    "query_token_init_std": "query_token_init_std",
    "position_init_extent_coverage": "position_init_extent_coverage",
    "rotation_init": "rotation_init",
    "rgb_init": "rgb_init",
    "rgb_init_min": "rgb_init_min",
    "rgb_init_max": "rgb_init_max",
    "free_frame_count": "free_frame_count",
    "free_time_interpolation": "free_time_interpolation",
    **CAMERA_ARGS,
}

UNCONDITIONED_BASE_ARG_NAMES = frozenset(
    {
        "clip_length",
        "image_size",
        "num_tokens",
        "feature_dim",
        "feat_dim",
        "gaussians_per_token",
        "scene_extent",
        "xy_extent",
        "z_min",
        "z_max",
        "scale_init",
        "scale_init_log_jitter",
        "opacity_init",
        "query_token_init_std",
        "head_hidden_dim",
        "head_hidden_layers",
        "head_output_init_std",
        "position_init_extent_coverage",
        "rotation_init",
        "rgb_init",
        "rgb_init_min",
        "rgb_init_max",
        "video_encoder_backend",
    }
)
UNCONDITIONED_ARGS = {key: BASE_VIDEO_ARGS[key] for key in UNCONDITIONED_BASE_ARG_NAMES}

LINEAR_FREE_ARGS: dict[str, ArgSource] = {
    "free_velocity_extent": "free_velocity_extent",
    "free_velocity_init_std": "free_velocity_init_std",
    "free_opacity_slope_init_std": "free_opacity_slope_init_std",
    "free_time_center": "free_time_center",
}
RESIDUAL_ARGS: dict[str, ArgSource] = {
    "residual_output_init_std": "residual_output_init_std",
    "residual_xyz_raw_scale": "residual_xyz_raw_scale",
    "residual_scale_log_scale": "residual_scale_log_scale",
    "residual_rot_raw_scale": "residual_rot_raw_scale",
    "residual_opacity_logit_scale": "residual_opacity_logit_scale",
    "residual_rgb_logit_scale": "residual_rgb_logit_scale",
    "residual_head_input_norm": "residual_head_input_norm",
}
TIME_ARGS: dict[str, ArgSource] = {
    "time_fourier_bands": "time_fourier_bands",
    "time_max_frequency": "time_max_frequency",
}
PLUCKER_ARGS: dict[str, ArgSource] = {"ray_condition_grid_size": "ray_condition_grid_size"}


def _sorted_keys(keys: set[str] | frozenset[str]) -> str:
    return ", ".join(sorted(keys))


def _reject_unknown(values: Mapping[str, Any], *, allowed: frozenset[str], context: str) -> None:
    unknown = set(values) - set(allowed)
    if unknown:
        raise ModelFactoryConfigError(
            f"Unknown {context} config key(s): {_sorted_keys(unknown)}. "
            f"Allowed keys: {_sorted_keys(allowed)}"
        )


def _required(values: Mapping[str, Any], *, required: frozenset[str], context: str) -> None:
    missing = set(required) - set(values) - set(MODEL_DEFAULTS)
    if missing:
        raise ModelFactoryConfigError(f"Missing required {context} config key(s): {_sorted_keys(missing)}")


def normalized_model_section(model_spec: Mapping[str, Any]) -> dict[str, Any]:
    values = dict(model_spec)
    _reject_unknown(values, allowed=KNOWN_MODEL_CONFIG_KEYS, context="model")
    _required(values, required=REQUIRED_MODEL_KEYS, context="model")
    values = {**MODEL_DEFAULTS, **values}
    values["variant"] = str(values["variant"]).lower()
    values["feature_dim"] = int(values["feature_dim"])
    if values["feature_dim"] < 1:
        raise ModelFactoryConfigError(f"model.feature_dim must be >= 1, got {values['feature_dim']}.")
    values["video_encoder_backend"] = str(values["video_encoder_backend"]).lower()
    values["free_time_interpolation"] = str(values["free_time_interpolation"]).lower()
    values["residual_head_input_norm"] = str(values["residual_head_input_norm"]).lower()
    if values["vjepa_feature_dim"] is not None:
        values["vjepa_feature_dim"] = int(values["vjepa_feature_dim"])
    if values["vjepa_crop_size"] is not None:
        values["vjepa_crop_size"] = int(values["vjepa_crop_size"])
    if values["vjepa_checkpoint_url"] is not None:
        values["vjepa_checkpoint_url"] = str(values["vjepa_checkpoint_url"])
    if values["video_feature_layers"] is not None:
        values["video_feature_layers"] = [str(name) for name in values["video_feature_layers"]]
    if values["video_feature_channels"] is not None:
        if not isinstance(values["video_feature_channels"], Mapping):
            raise ModelFactoryConfigError("model.video_feature_channels must map layer names to channel counts.")
        values["video_feature_channels"] = {
            str(name): int(channels) for name, channels in values["video_feature_channels"].items()
        }
    values["video_feature_token_stride"] = int(values["video_feature_token_stride"])
    if values["video_feature_token_stride"] < 1:
        raise ModelFactoryConfigError(
            f"model.video_feature_token_stride must be >= 1, got {values['video_feature_token_stride']}."
        )
    if values["video_feature_output_dtype"] is not None:
        values["video_feature_output_dtype"] = str(values["video_feature_output_dtype"]).lower()
    values["camera_refine_with_decode_time"] = bool(values["camera_refine_with_decode_time"])
    if values["xy_extent"] is None:
        values["xy_extent"] = values["scene_extent"]
    if values["z_min"] is None:
        values["z_min"] = -values["scene_extent"]
    if values["z_max"] is None:
        values["z_max"] = values["scene_extent"]
    if values["static_tokens"] is not None or values["dynamic_tokens"] is not None:
        if values["static_tokens"] is None or values["dynamic_tokens"] is None:
            raise ModelFactoryConfigError("model.static_tokens and model.dynamic_tokens must be provided together.")
        values["use_static_dynamic_split"] = True
    else:
        values["use_static_dynamic_split"] = False
    if values["free_frame_count"] is not None:
        values["free_frame_count"] = int(values["free_frame_count"])
    return values


def normalized_camera_section(camera_spec: Mapping[str, Any]) -> dict[str, Any]:
    values = dict(camera_spec)
    _reject_unknown(values, allowed=KNOWN_CAMERA_CONFIG_KEYS, context="camera")
    values = {**CAMERA_DEFAULTS, **values}
    values["global_head"] = str(values["global_head"])
    values["lens_model"] = str(values["lens_model"]).lower()
    return values


def resolve_model_variant(variant: str) -> str:
    normalized = str(variant).lower()
    if normalized not in MODEL_VARIANT_CLASSES:
        raise ModelFactoryConfigError(
            f"Unknown model.variant={variant!r}. Expected one of: {_sorted_keys(frozenset(MODEL_VARIANT_CLASSES))}."
        )
    return normalized


def model_class_for_variant(variant: str) -> type[nn.Module]:
    return MODEL_VARIANT_CLASSES[resolve_model_variant(variant)]


def _arg_map_for_variant(variant: str) -> dict[str, ArgSource]:
    variant = resolve_model_variant(variant)
    if variant in FREE_BANK_VARIANTS:
        return dict(FREE_BANK_ARGS)
    if variant in LINEAR_FREE_BANK_VARIANTS:
        return {**FREE_BANK_ARGS, **LINEAR_FREE_ARGS}
    if variant in UNCONDITIONED_VARIANTS:
        args = {**UNCONDITIONED_ARGS, **DYNAMIC_SPLIT_ARGS, **CAMERA_ARGS}
    elif variant in UNCONDITIONED_RESIDUAL_VARIANTS:
        args = {**UNCONDITIONED_ARGS, **DYNAMIC_SPLIT_ARGS, **CAMERA_ARGS, **RESIDUAL_ARGS}
    else:
        args = dict(BASE_VIDEO_ARGS)
        if variant not in KNOWN_CAMERA_VARIANTS:
            args.update(DYNAMIC_SPLIT_ARGS)
            args.update(CAMERA_ARGS)
    if variant in {"sinusoidal_time_path_mlp", "token_to_pose_to_plucker"}:
        args.update(TIME_ARGS)
    if variant == "token_to_pose_to_plucker":
        args.update(PLUCKER_ARGS)
    if variant in RESIDUAL_VARIANTS:
        args.update(RESIDUAL_ARGS)
    return args


def allowed_model_kwargs_for_variant(variant: str) -> frozenset[str]:
    return frozenset(_arg_map_for_variant(variant))


def _read_source(source: ArgSource, model: Mapping[str, Any], camera: Mapping[str, Any]) -> Any:
    if callable(source):
        return source(model, camera)
    if isinstance(source, tuple):
        namespace, key = source
        if namespace != "camera":
            raise AssertionError(f"Unknown model-factory source namespace: {namespace!r}")
        return camera[key]
    return model[source]


def validated_model_kwargs(model_spec: Mapping[str, Any], camera_spec: Mapping[str, Any]) -> dict[str, Any]:
    model = normalized_model_section(model_spec)
    camera = normalized_camera_section(camera_spec)
    variant = resolve_model_variant(model["variant"])
    if variant in FREE_BANK_VARIANTS | LINEAR_FREE_BANK_VARIANTS and model["feature_dim"] != 3:
        raise ModelFactoryConfigError(
            "free Gaussian bank variants do not currently honor model.feature_dim != 3; "
            "use a token decoder variant for feature splatting."
        )
    return {
        arg_name: _read_source(source, model, camera)
        for arg_name, source in _arg_map_for_variant(variant).items()
    }


def reject_unknown_kwargs(kwargs: Mapping[str, Any], *, allowed: frozenset[str], context: str) -> None:
    _reject_unknown(kwargs, allowed=allowed, context=f"{context} constructor kwargs")


def build_model_module(
    model_spec: Mapping[str, Any],
    camera_spec: Mapping[str, Any],
    *,
    overrides: Mapping[str, Any] | None = None,
) -> nn.Module:
    model = normalized_model_section(model_spec)
    variant = resolve_model_variant(model["variant"])
    kwargs = validated_model_kwargs(model, camera_spec)
    if overrides:
        reject_unknown_kwargs(
            overrides,
            allowed=allowed_model_kwargs_for_variant(variant),
            context=f"model.variant={variant} overrides",
        )
        kwargs.update(dict(overrides))
    return model_class_for_variant(variant)(**kwargs)


def build_model_from_config(config: Mapping[str, Any]) -> nn.Module:
    return build_model_module(config["model"], config["camera"])


def validated_colorize_kwargs(
    colorize_spec: Mapping[str, Any] | None,
    *,
    feature_dim: int,
) -> tuple[dict[str, Any] | None, str, bool]:
    feature_dim = int(feature_dim)
    if colorize_spec is None:
        if feature_dim != 3:
            raise ModelFactoryConfigError(
                f"model.feature_dim={feature_dim} requires a 'colorize' config section. "
                "Set model.feature_dim=3 for the legacy RGB path or add colorize settings for feature splatting."
            )
        return None, "none", True

    colorize = dict(colorize_spec)
    _reject_unknown(colorize, allowed=COLORIZE_CONFIG_KEYS, context="colorize")
    view_condition = normalize_view_condition(colorize.get("view_condition", "none"))
    kwargs = {
        "feature_dim": feature_dim,
        "hidden_dim": colorize.get("hidden_dim"),
        "activation": str(colorize.get("activation", "sigmoid")),
        "pre_norm": bool(colorize.get("pre_norm", False)),
        "weight_init": str(colorize.get("weight_init", "kaiming")).lower(),
        "weight_init_gain": float(colorize.get("weight_init_gain", 1.0)),
        "view_condition": view_condition,
        "detach_view_condition": bool(colorize.get("detach_view_condition", True)),
    }
    return kwargs, view_condition, kwargs["detach_view_condition"]


def build_colorizer(
    colorize_spec: Mapping[str, Any] | None,
    *,
    feature_dim: int,
) -> ColorizeFactoryResult:
    kwargs, view_condition, detach_view_condition = validated_colorize_kwargs(
        colorize_spec,
        feature_dim=feature_dim,
    )
    return ColorizeFactoryResult(
        module=None if kwargs is None else FeatureToColor(**kwargs),
        view_condition=view_condition,
        detach_view_condition=detach_view_condition,
    )
