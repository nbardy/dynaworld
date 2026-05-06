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
    assert cfg["train"]["checkpoint_save_path"] == Path(
        "outputs/multicam_relative_pose/full_relpose_features_F32_256_goodset_train0006_0014_holdout0005/checkpoint_final.pt"
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
