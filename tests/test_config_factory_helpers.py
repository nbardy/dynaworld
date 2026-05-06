from __future__ import annotations

import pytest
import torch

from config_utils import load_config_file
from gs_models.dynamic_video_token_gs_implicit_camera import PrecomputedVideoFeatureAdapter
from model_factories import (
    build_colorizer,
    ModelFactoryConfigError,
    model_class_for_variant,
    validated_model_kwargs,
)
from train_video_token_implicit_dynamic import resolve_config


F32_SINGLE_CAM_CONFIG = "src/train_configs/local_mac_unconditioned_tokens_features_F32_alpha_400step.jsonc"
F32_MULTICAM_CONFIG = "src/train_configs/local_mac_ultimate_features_F32_vjepa_multicam_256px_8192splats_alpha.jsonc"


def test_validated_model_kwargs_accepts_raw_f32_single_cam_config() -> None:
    cfg = load_config_file(F32_SINGLE_CAM_CONFIG)

    kwargs = validated_model_kwargs(cfg["model"], cfg["camera"])

    assert kwargs["feature_dim"] == 32
    assert kwargs["video_encoder_backend"] == "none"
    assert model_class_for_variant(cfg["model"]["variant"]).__name__ == "UnconditionedTokenGSImplicitCamera"


def test_validated_model_kwargs_accepts_multicam_rig_keys_without_passing_them_to_model() -> None:
    cfg = load_config_file(F32_MULTICAM_CONFIG)
    cfg["model"]["video_feature_token_stride"] = 4
    cfg["model"]["video_feature_output_dtype"] = "bf16"
    cfg["model"]["camera_refine_with_decode_time"] = False

    kwargs = validated_model_kwargs(cfg["model"], cfg["camera"])

    assert kwargs["feature_dim"] == 32
    assert kwargs["video_encoder_backend"] == "precomputed"
    assert kwargs["video_feature_token_stride"] == 4
    assert kwargs["video_feature_output_dtype"] == "bf16"
    assert kwargs["camera_refine_with_decode_time"] is False
    assert "rig_radius" not in kwargs
    assert "rig_init" not in kwargs


def test_model_factory_rejects_unknown_model_keys() -> None:
    cfg = load_config_file(F32_SINGLE_CAM_CONFIG)
    cfg["model"]["typo_key"] = 123

    with pytest.raises(ModelFactoryConfigError, match="typo_key"):
        validated_model_kwargs(cfg["model"], cfg["camera"])


def test_model_factory_rejects_unknown_camera_keys() -> None:
    cfg = load_config_file(F32_MULTICAM_CONFIG)
    cfg["camera"]["rig_typo_key"] = 123

    with pytest.raises(ModelFactoryConfigError, match="rig_typo_key"):
        validated_model_kwargs(cfg["model"], cfg["camera"])


def test_f32_requires_colorize_section() -> None:
    with pytest.raises(ValueError, match="requires a 'colorize' config section"):
        build_colorizer(None, feature_dim=32)


def test_colorizer_factory_builds_f32_colorizer() -> None:
    cfg = load_config_file(F32_SINGLE_CAM_CONFIG)

    result = build_colorizer(cfg["colorize"], feature_dim=cfg["model"]["feature_dim"])

    assert result.module is not None
    assert result.module.feature_dim == 32
    assert result.module.weight_init == "kaiming"
    assert result.module.weight_init_gain == 4.0
    assert result.view_condition == "none"


def test_precomputed_feature_adapter_strides_tokens_before_projection() -> None:
    adapter = PrecomputedVideoFeatureAdapter(
        output_dim=4,
        feature_channels={"vjepa_tokens": 2},
        feature_layers=["vjepa_tokens"],
        token_stride=2,
    )
    payload = {"vjepa_tokens": torch.arange(12, dtype=torch.float32).reshape(1, 6, 2)}

    projected = adapter(payload)

    assert projected.shape == (1, 3, 4)


def test_precomputed_feature_adapter_can_cast_projected_tokens_to_bf16() -> None:
    adapter = PrecomputedVideoFeatureAdapter(
        output_dim=4,
        feature_channels={"vjepa_tokens": 2},
        feature_layers=["vjepa_tokens"],
        output_dtype="bf16",
    )
    payload = {"vjepa_tokens": torch.arange(12, dtype=torch.float32).reshape(1, 6, 2)}

    projected = adapter(payload)

    assert projected.dtype == torch.bfloat16


def test_resolve_config_accepts_explicit_bf16_amp_dtype() -> None:
    cfg = load_config_file(F32_MULTICAM_CONFIG)
    cfg["train"]["amp"] = True
    cfg["train"]["amp_dtype"] = "bf16"

    resolved = resolve_config(cfg)

    assert resolved["train"]["amp_dtype"] == "bf16"
