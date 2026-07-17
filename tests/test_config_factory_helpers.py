from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from config_utils import load_config_file
from gs_models.dynamic_video_token_gs_implicit_camera import (
    DynamicVideoTokenGSImplicitCamera,
    PrecomputedVideoFeatureAdapter,
)
from model_factories import (
    build_colorizer,
    ModelFactoryConfigError,
    model_class_for_variant,
    validated_model_kwargs,
)
from trainer_registry import resolve_config_for_arch as resolve_config


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


def test_validated_model_kwargs_passes_opt_in_token_layout() -> None:
    cfg = load_config_file(F32_MULTICAM_CONFIG)
    cfg["model"]["tokens"] = 144
    cfg["model"]["token_layout"] = {
        "world_tokens": 8,
        "register_tokens": 8,
        "static_core_tokens": 64,
        "dynamic_core_tokens": 16,
        "static_detail_tokens": [32],
        "dynamic_detail_tokens": [16],
        "active_detail_level": 1,
    }

    kwargs = validated_model_kwargs(cfg["model"], cfg["camera"])

    assert kwargs["num_tokens"] == 144
    assert kwargs["token_layout"]["world_tokens"] == 8
    assert kwargs["static_tokens"] == 96
    assert kwargs["dynamic_tokens"] == 32


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


def test_resolve_config_defaults_train_seed() -> None:
    cfg = load_config_file(F32_MULTICAM_CONFIG)
    cfg["train"].pop("seed", None)

    resolved = resolve_config(cfg)

    assert resolved["train"]["seed"] == 17


def test_resolve_config_accepts_explicit_train_seed() -> None:
    cfg = load_config_file(F32_MULTICAM_CONFIG)
    cfg["train"]["seed"] = 123

    resolved = resolve_config(cfg)

    assert resolved["train"]["seed"] == 123


def test_resolve_config_accepts_manifest_prefetch_and_wandb_mode() -> None:
    cfg = load_config_file(F32_MULTICAM_CONFIG)
    cfg["data"]["train_manifest_prefetch"] = 2
    cfg["logging"]["wandb_mode"] = "online"

    resolved = resolve_config(cfg)

    assert resolved["data"]["train_manifest_prefetch"] == 2
    assert resolved["logging"]["wandb_mode"] == "online"


def test_train_router_accepts_star_uvt_video_overfit_config() -> None:
    path = Path(__file__).resolve().parents[1] / "src" / "train" / "train.py"
    spec = importlib.util.spec_from_file_location("dynaworld_train_entry_star_uvt_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load train module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    entry = module.trainer_entry_for_config(
        "src/train_configs/star_uvt_highmotion_hlaZbH_64f_256_directatomic_200step.jsonc"
    )

    assert entry.module == "star_uvt_video_trainer"
    assert entry.runner == "run_training"


def _tiny_precomputed_token_layout_model(active_detail_level: int) -> DynamicVideoTokenGSImplicitCamera:
    return DynamicVideoTokenGSImplicitCamera(
        clip_length=2,
        image_size=16,
        num_tokens=10,
        feat_dim=8,
        bottleneck_dim=8,
        num_heads=2,
        mlp_ratio=1.0,
        gaussians_per_token=1,
        scene_extent=1.0,
        video_encoder_backend="precomputed",
        video_feature_layers=["vjepa_tokens"],
        video_feature_channels={"vjepa_tokens": 4},
        cross_attn_layers=1,
        static_tokens=None,
        dynamic_tokens=None,
        token_layout={
            "world_tokens": 2,
            "register_tokens": 1,
            "static_core_tokens": 2,
            "dynamic_core_tokens": 1,
            "static_detail_tokens": [2],
            "dynamic_detail_tokens": [2],
            "active_detail_level": active_detail_level,
        },
        feature_dim=32,
    )


def test_token_layout_keeps_world_register_queries_but_decodes_active_core_only() -> None:
    torch.manual_seed(123)
    model = _tiny_precomputed_token_layout_model(active_detail_level=0)
    features = {"vjepa_tokens": torch.randn(1, 6, 4)}
    decode_times = torch.tensor([[0.0, 1.0]])

    sequence = model(features, decode_times)

    assert model.query_tokens(1).shape == (1, 12, 8)
    assert model.static_tokens == 2
    assert model.dynamic_tokens == 1
    assert sequence.xyz.shape == (2, 3, 3)
    assert sequence.rgbs.shape == (2, 3, 32)
    assert sequence.auxiliary["token_layout_active_detail_level"].item() == 0


def test_token_layout_active_detail_level_adds_decoded_detail_tokens() -> None:
    torch.manual_seed(123)
    model = _tiny_precomputed_token_layout_model(active_detail_level=1)
    features = {"vjepa_tokens": torch.randn(1, 6, 4)}
    decode_times = torch.tensor([[0.0, 1.0]])

    sequence = model(features, decode_times)

    assert model.query_tokens(1).shape == (1, 12, 8)
    assert model.static_tokens == 4
    assert model.dynamic_tokens == 3
    assert sequence.xyz.shape == (2, 7, 3)
    assert sequence.rgbs.shape == (2, 7, 32)


def test_token_layout_null_preserves_legacy_static_dynamic_split() -> None:
    model = DynamicVideoTokenGSImplicitCamera(
        clip_length=2,
        image_size=16,
        num_tokens=3,
        feat_dim=8,
        bottleneck_dim=8,
        num_heads=2,
        mlp_ratio=1.0,
        gaussians_per_token=1,
        scene_extent=1.0,
        video_encoder_backend="precomputed",
        video_feature_layers=["vjepa_tokens"],
        video_feature_channels={"vjepa_tokens": 4},
        cross_attn_layers=1,
        static_tokens=2,
        dynamic_tokens=1,
        token_layout=None,
        feature_dim=32,
    )

    sequence = model({"vjepa_tokens": torch.randn(1, 6, 4)}, torch.tensor([[0.0, 1.0]]))

    assert model.query_tokens(1).shape == (1, 5, 8)
    assert sequence.xyz.shape == (2, 3, 3)
