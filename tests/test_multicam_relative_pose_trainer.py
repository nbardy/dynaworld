from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from config_utils import load_config_file
from runtime_types import SequenceData
from train_multicam_relative_pose_implicit_dynamic import (
    MulticamRelativePoseImplicitTrainer,
    first_frame_repeated_sequence,
    normalize_multires_render_probabilities,
    normalize_multires_render_sizes,
    normalize_multires_token_detail_levels,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
JOINT_CONFIG = (
    REPO_ROOT
    / "src/train_configs/"
    "local_mac_multicam_deepview_3cam_train2_test1_vjepa_full_relpose_128_16f_8192splats_goodset_train0006_0014_holdout0005.jsonc"
)
F32_JOINT_CONFIG = (
    REPO_ROOT
    / "src/train_configs/"
    "local_mac_multicam_deepview_3cam_train2_test1_vjepa_full_relpose_features_F32_128_16f_8192splats_goodset_train0006_0014_holdout0005.jsonc"
)
F32_256_JOINT_CONFIG = (
    REPO_ROOT
    / "src/train_configs/"
    "local_mac_multicam_deepview_3cam_train2_test1_vjepa_full_relpose_features_F32_256_16f_8192splats_goodset_train0006_0014_holdout0005.jsonc"
)
F32_256_RELPOSE_OUTPUTINIT_CONFIG = (
    REPO_ROOT
    / "src/train_configs/"
    "local_mac_multicam_deepview_3cam_train2_test1_vjepa_full_relpose_features_F32_256_16f_8192splats_goodset_train0006_0014_holdout0005_alpha1_128_relpose_outputinit012.jsonc"
)
F32_256_RELPOSE_PAIRDELTA_CONFIG = (
    REPO_ROOT
    / "src/train_configs/"
    "local_mac_multicam_deepview_3cam_train2_test1_vjepa_full_relpose_features_F32_256_16f_8192splats_goodset_train0006_0014_holdout0005_alpha1_128_relpose_pairdelta012.jsonc"
)
F32_FAST_RELPOSE_PAIRDELTA_CONFIG = (
    REPO_ROOT
    / "src/train_configs/"
    "local_mac_multicam_deepview_3cam_train2_test1_vjepa_full_relpose_features_F32_multires64_128_256_512_tokenbudget_world4_fast_16f_8192splats_goodset_train0006_0014_holdout0005_alpha1_128_relpose_pairdelta012.jsonc"
)
F32_MULTIRES_RELPOSE_OUTPUTINIT_CONFIG = (
    REPO_ROOT
    / "src/train_configs/"
    "local_mac_multicam_deepview_3cam_train2_test1_vjepa_full_relpose_features_F32_multires64_128_256_512_1024_1920_16f_8192splats_goodset_train0006_0014_holdout0005_alpha1_128_relpose_outputinit012.jsonc"
)
F32_MULTIRES_TOKENBUDGET_RELPOSE_OUTPUTINIT_CONFIG = (
    REPO_ROOT
    / "src/train_configs/"
    "local_mac_multicam_deepview_3cam_train2_test1_vjepa_full_relpose_features_F32_multires64_128_256_512_1024_1920_tokenbudget_world4_16f_8192splats_goodset_train0006_0014_holdout0005_alpha1_128_relpose_outputinit012.jsonc"
)
F32_512_V6_JOINT_CONFIG = (
    REPO_ROOT
    / "src/train_configs/"
    "local_mac_multicam_deepview_3cam_train2_test1_vjepa_full_relpose_features_F32_512_16f_8192splats_v6refined_goodset_train0006_0014_holdout0005.jsonc"
)
OFFSET_ONLY_CONFIG = (
    REPO_ROOT
    / "src/train_configs/"
    "local_mac_multicam_deepview_3cam_train2_test1_vjepa_full_relpose_offsetonly_128_16f_8192splats_goodset_train0006_0014_holdout0005.jsonc"
)


def _trainer_entry_for_config(config_path: Path):
    spec = importlib.util.spec_from_file_location("dynaworld_train_dispatch", REPO_ROOT / "src/train/train.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load src/train/train.py dispatch module.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.trainer_entry_for_config(config_path)


def test_multicam_relative_pose_arch_dispatches_to_trainer() -> None:
    entry = _trainer_entry_for_config(JOINT_CONFIG)

    assert entry.module == "train_multicam_relative_pose_implicit_dynamic"


def test_joint_full_relpose_config_resolves_predicted_heldout_mode() -> None:
    cfg = MulticamRelativePoseImplicitTrainer.resolve_config(load_config_file(JOINT_CONFIG))

    assert cfg["train"]["relpose_output_mode"] == "full"
    assert cfg["train"]["relpose_output_init_std"] == 0.0
    assert cfg["train"]["relpose_pair_delta_init_std"] == 0.0
    assert cfg["train"]["heldout_eval_camera_mode"] == "predicted_relpose"
    assert cfg["train"]["trainable_scope"] == "all"
    assert cfg["train"]["relpose_feature_frame_mode"] == "first_frame"
    assert cfg["data"]["multicam_train_cameras"] == ["camera_0006", "camera_0014"]
    assert cfg["data"]["multicam_heldout_camera"] == "camera_0005"
    assert cfg["train"]["checkpoint_save_path"] == Path(
        "outputs/multicam_relative_pose/full_relpose_goodset_train0006_0014_holdout0005/checkpoint_final.pt"
    )


def test_f32_joint_full_relpose_config_resolves_feature_splatting_path() -> None:
    cfg = MulticamRelativePoseImplicitTrainer.resolve_config(load_config_file(F32_JOINT_CONFIG))

    assert cfg["model"]["feature_dim"] == 32
    assert cfg["colorize"]["pre_norm"] is True
    assert cfg["colorize"]["weight_init"] == "kaiming"
    assert cfg["render"]["fast_mac"]["feature_variant"] == "v5_features"
    assert cfg["render"]["fast_mac"]["feature_background"] == 0.0
    assert cfg["logging"]["feature_pca_log"] is True
    assert "feature-splatting" in cfg["logging"]["wandb_tags"]
    assert cfg["train"]["checkpoint_save_path"] == Path(
        "outputs/multicam_relative_pose/full_relpose_features_F32_goodset_train0006_0014_holdout0005/checkpoint_final.pt"
    )


def test_f32_256_joint_full_relpose_config_keeps_modern_feature_settings() -> None:
    cfg = MulticamRelativePoseImplicitTrainer.resolve_config(load_config_file(F32_256_JOINT_CONFIG))

    assert cfg["model"]["size"] == 256
    assert cfg["render"]["render_size"] == 256
    assert cfg["model"]["feature_dim"] == 32
    assert cfg["features"]["cache_dir"] == Path(
        "data/feature_cache/multicam_deepview_static_dynamic_vjepa2_1_vitb_384_256px"
    )
    assert "256-16f" in cfg["features"]["sample_cache_key"]
    assert cfg["render"]["fast_mac"]["feature_variant"] == "v5_features"
    assert cfg["render"]["fast_mac"]["feature_background"] == 0.0
    assert cfg["logging"]["feature_pca_log"] is True
    assert "256px" in cfg["logging"]["wandb_tags"]
    assert cfg["train"]["relpose_output_init_std"] == 0.0
    assert cfg["train"]["relpose_pair_delta_init_std"] == 0.0
    assert cfg["train"]["checkpoint_save_path"] == Path(
        "outputs/multicam_relative_pose/full_relpose_features_F32_256_goodset_train0006_0014_holdout0005/checkpoint_final.pt"
    )


def test_f32_256_relpose_output_init_config_breaks_identity_init() -> None:
    cfg = MulticamRelativePoseImplicitTrainer.resolve_config(load_config_file(F32_256_RELPOSE_OUTPUTINIT_CONFIG))

    assert cfg["render"]["alpha_threshold"] == 1.0 / 128.0
    assert cfg["render"]["fast_mac"]["alpha_threshold"] == 1.0 / 128.0
    assert cfg["train"]["relpose_output_init_std"] == 0.12
    assert cfg["train"]["relpose_pair_delta_init_std"] == 0.0
    assert cfg["train"]["multires_render_sizes"] is None
    assert "relpose-output-init-012" in cfg["logging"]["wandb_tags"]
    assert cfg["train"]["checkpoint_save_path"] == Path(
        "outputs/multicam_relative_pose/full_relpose_features_F32_256_alpha1_128_relpose_outputinit012_goodset_train0006_0014_holdout0005/checkpoint_final.pt"
    )


def test_f32_256_relpose_pairdelta_config_uses_target_dependent_init() -> None:
    cfg = MulticamRelativePoseImplicitTrainer.resolve_config(load_config_file(F32_256_RELPOSE_PAIRDELTA_CONFIG))

    assert cfg["render"]["alpha_threshold"] == 1.0 / 128.0
    assert cfg["train"]["relpose_output_init_std"] == 0.0
    assert cfg["train"]["relpose_pair_delta_init_std"] == 0.12
    assert cfg["train"]["multires_render_sizes"] is None
    assert "relpose-output-init-000" in cfg["logging"]["wandb_tags"]
    assert "relpose-pair-delta-init-012" in cfg["logging"]["wandb_tags"]
    assert "target-dependent-relpose-init" in cfg["logging"]["wandb_tags"]
    assert cfg["train"]["checkpoint_save_path"] == Path(
        "outputs/multicam_relative_pose/full_relpose_features_F32_256_alpha1_128_relpose_pairdelta012_goodset_train0006_0014_holdout0005/checkpoint_final.pt"
    )


def test_fast_multires_relpose_pairdelta_config_is_the_current_short_run() -> None:
    cfg = MulticamRelativePoseImplicitTrainer.resolve_config(load_config_file(F32_FAST_RELPOSE_PAIRDELTA_CONFIG))

    assert cfg["train"]["relpose_output_init_std"] == 0.0
    assert cfg["train"]["relpose_pair_delta_init_std"] == 0.12
    assert cfg["train"]["multires_render_sizes"] == [64, 128, 256, 512]
    assert cfg["train"]["multires_render_probabilities"] == [0.25, 0.45, 0.25, 0.05]
    assert cfg["train"]["multires_token_detail_levels"] == [0, 0, 1, 2]
    assert "fast-res-capped-512" in cfg["logging"]["wandb_tags"]
    assert "target-dependent-relpose-init" in cfg["logging"]["wandb_tags"]
    assert cfg["train"]["checkpoint_save_path"] == Path(
        "outputs/multicam_relative_pose/full_relpose_features_F32_multires64_128_256_512_tokenbudget_world4_fast_alpha1_128_relpose_pairdelta012_goodset_train0006_0014_holdout0005/checkpoint_final.pt"
    )


def test_f32_multires_relpose_config_preserves_baseline_and_enables_schedule() -> None:
    cfg = MulticamRelativePoseImplicitTrainer.resolve_config(load_config_file(F32_MULTIRES_RELPOSE_OUTPUTINIT_CONFIG))

    assert cfg["model"]["size"] == 256
    assert cfg["render"]["render_size"] == 256
    assert cfg["model"]["feature_dim"] == 32
    assert cfg["render"]["alpha_threshold"] == 1.0 / 128.0
    assert cfg["render"]["fast_mac"]["feature_variant"] == "v5_features"
    assert cfg["render"]["fast_mac"]["alpha_threshold"] == 1.0 / 128.0
    assert cfg["render"]["fast_mac"]["feature_background"] == 0.0
    assert cfg["train"]["multires_render_sizes"] == [64, 128, 256, 512, 1024, 1920]
    assert cfg["train"]["multires_render_probabilities"] == [0.2, 0.4, 0.2, 0.1, 0.05, 0.05]
    assert cfg["train"]["multires_token_detail_levels"] is None
    assert cfg["train"]["frame_sampling"]["mode"] == "contiguous"
    assert cfg["train"]["relpose_output_init_std"] == 0.12
    assert cfg["data"]["multicam_train_cameras"] == ["camera_0006", "camera_0014"]
    assert cfg["data"]["multicam_heldout_camera"] == "camera_0005"
    assert cfg["features"]["cache_dir"] == Path(
        "data/feature_cache/multicam_deepview_static_dynamic_vjepa2_1_vitb_384_256px"
    )
    assert "multires-render" in cfg["logging"]["wandb_tags"]
    assert "multires-64-128-256-512-1024-1920" in cfg["logging"]["wandb_tags"]
    assert "weighted-resolutions" in cfg["logging"]["wandb_tags"]
    assert cfg["train"]["checkpoint_save_path"] == Path(
        "outputs/multicam_relative_pose/full_relpose_features_F32_multires64_128_256_512_1024_1920_alpha1_128_relpose_outputinit012_goodset_train0006_0014_holdout0005/checkpoint_final.pt"
    )


def test_f32_multires_tokenbudget_config_adds_world_tokens_and_detail_schedule() -> None:
    cfg = MulticamRelativePoseImplicitTrainer.resolve_config(
        load_config_file(F32_MULTIRES_TOKENBUDGET_RELPOSE_OUTPUTINIT_CONFIG)
    )

    assert cfg["model"]["size"] == 256
    assert cfg["model"]["feature_dim"] == 32
    assert cfg["model"]["tokens"] == 136
    assert cfg["model"]["static_tokens"] == 96
    assert cfg["model"]["dynamic_tokens"] == 32
    assert cfg["model"]["token_layout"] == {
        "world_tokens": 4,
        "register_tokens": 2,
        "static_core_tokens": 56,
        "dynamic_core_tokens": 16,
        "detail_register_tokens": [1, 1],
        "static_detail_tokens": [24, 16],
        "dynamic_detail_tokens": [8, 8],
        "active_detail_level": 2,
    }
    assert cfg["train"]["multires_render_sizes"] == [64, 128, 256, 512, 1024, 1920]
    assert cfg["train"]["multires_render_probabilities"] == [0.2, 0.4, 0.2, 0.1, 0.05, 0.05]
    assert cfg["train"]["multires_token_detail_levels"] == [0, 0, 1, 2, 2, 2]
    assert cfg["train"]["frame_sampling"]["mode"] == "contiguous"
    assert cfg["render"]["fast_mac"]["feature_variant"] == "v5_features"
    assert cfg["data"]["multicam_train_cameras"] == ["camera_0006", "camera_0014"]
    assert cfg["data"]["multicam_heldout_camera"] == "camera_0005"
    assert "token-layout" in cfg["logging"]["wandb_tags"]
    assert "dynamic-token-budget" in cfg["logging"]["wandb_tags"]
    assert "weighted-resolutions" in cfg["logging"]["wandb_tags"]
    assert cfg["train"]["checkpoint_save_path"] == Path(
        "outputs/multicam_relative_pose/full_relpose_features_F32_multires64_128_256_512_1024_1920_tokenbudget_world4_alpha1_128_relpose_outputinit012_goodset_train0006_0014_holdout0005/checkpoint_final.pt"
    )


def test_multires_render_size_normalization_rejects_ambiguous_values() -> None:
    assert normalize_multires_render_sizes([128, "192", 256]) == [128, 192, 256]
    assert normalize_multires_render_probabilities([20, 40, 40], render_sizes=[64, 128, 256]) == [0.2, 0.4, 0.4]
    assert normalize_multires_token_detail_levels([0, "1", 2], render_sizes=[128, 192, 256]) == [0, 1, 2]

    with pytest.raises(ValueError, match="cannot be empty"):
        normalize_multires_render_sizes([])

    with pytest.raises(ValueError, match="duplicates"):
        normalize_multires_render_sizes([128, 128])

    with pytest.raises(ValueError, match="list"):
        normalize_multires_render_sizes(True)

    with pytest.raises(ValueError, match="same length"):
        normalize_multires_token_detail_levels([0, 1], render_sizes=[128, 192, 256])

    with pytest.raises(ValueError, match="requires"):
        normalize_multires_token_detail_levels([0], render_sizes=None)

    with pytest.raises(ValueError, match="same length"):
        normalize_multires_render_probabilities([0.5, 0.5], render_sizes=[64, 128, 256])

    with pytest.raises(ValueError, match="positive"):
        normalize_multires_render_probabilities([0.0, 0.0], render_sizes=[64, 128])


def test_f32_512_v6_joint_full_relpose_config_uses_refined_feature_rasterizer() -> None:
    cfg = MulticamRelativePoseImplicitTrainer.resolve_config(load_config_file(F32_512_V6_JOINT_CONFIG))

    assert cfg["model"]["size"] == 512
    assert cfg["render"]["render_size"] == 512
    assert cfg["model"]["feature_dim"] == 32
    assert cfg["model"]["video_feature_token_stride"] == 12
    assert cfg["model"]["video_feature_output_dtype"] == "bf16"
    assert cfg["model"]["tokens"] == 256
    assert cfg["model"]["static_tokens"] == 192
    assert cfg["model"]["dynamic_tokens"] == 64
    assert cfg["model"]["gaussians_per_token"] == 32
    assert cfg["model"]["model_dim"] == 64
    assert cfg["model"]["encoder_self_attn_layers"] == 2
    assert cfg["model"]["bottleneck_self_attn_layers"] == 4
    assert cfg["model"]["cross_attn_layers"] == 6
    assert cfg["features"]["cache_dir"] == Path(
        "data/feature_cache/multicam_deepview_static_dynamic_vjepa2_1_vitb_384_512px"
    )
    assert "512-16f" in cfg["features"]["sample_cache_key"]
    assert cfg["render"]["fast_mac"]["feature_variant"] == "v6_refined_features"
    assert cfg["render"]["fast_mac"]["use_active_tiles"] is False
    assert cfg["render"]["fast_mac"]["active_policy"] == "off"
    assert cfg["render"]["fast_mac"]["stop_count_mode"] == "adaptive"
    assert cfg["train"]["temporal_microbatch_size"] == 2
    assert "512px" in cfg["logging"]["wandb_tags"]
    assert "v6-refined-features" in cfg["logging"]["wandb_tags"]
    assert "feature-token-stride-12" in cfg["logging"]["wandb_tags"]
    assert cfg["train"]["checkpoint_save_path"] == Path(
        "outputs/multicam_relative_pose/full_relpose_features_F32_512_v6refined_goodset_train0006_0014_holdout0005/checkpoint_final.pt"
    )


def test_relpose_only_config_requires_cross_pairs_and_checkpoint() -> None:
    cfg = load_config_file(OFFSET_ONLY_CONFIG)
    cfg["train"]["camera_swap_include_cross"] = False

    with pytest.raises(ValueError, match="camera_swap_include_cross"):
        MulticamRelativePoseImplicitTrainer.resolve_config(cfg)

    cfg = load_config_file(OFFSET_ONLY_CONFIG)
    cfg["train"]["checkpoint_load_path"] = None

    with pytest.raises(ValueError, match="checkpoint_load_path"):
        MulticamRelativePoseImplicitTrainer.resolve_config(cfg)


def test_first_frame_repeated_sequence_uses_only_initial_frame() -> None:
    frames = torch.arange(4 * 3 * 2 * 2, dtype=torch.float32).reshape(4, 3, 2, 2)
    frame_times = torch.arange(4, dtype=torch.float32).view(4, 1)
    sequence = SequenceData(
        frames=frames,
        frame_times=frame_times,
        video_fps=4.0,
        frame_source="explicit_video",
        source_path=Path("/tmp/camera_0006.mp4"),
        selected_frame_count=4,
        all_frame_count=16,
    )

    relpose_sequence = first_frame_repeated_sequence(sequence, repeat_count=3, cache_tag="train_0")

    assert relpose_sequence.frame_count == 3
    assert torch.equal(relpose_sequence.frames[0], frames[0])
    assert torch.equal(relpose_sequence.frames[1], frames[0])
    assert torch.equal(relpose_sequence.frames[2], frames[0])
    assert torch.equal(relpose_sequence.frame_times, frame_times[:1].expand(3, 1))
    assert relpose_sequence.source_path is not None
    assert "relpose_first_frame_train_0_3f" in str(relpose_sequence.source_path)
