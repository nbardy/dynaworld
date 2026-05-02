from __future__ import annotations

import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src" / "train"))

from config_utils import load_config_file  # noqa: E402
from multicam_video_data import load_multicam_video_bundle  # noqa: E402


SPLIT_CONFIG = ROOT / "src/dataset_configs/multicam_train2_holdout1_5sample_128_4fps_16f.jsonc"


def data_cfg_for_sample(split_cfg: dict, sample: dict) -> dict:
    return {
        "multicam_manifest": split_cfg["manifest_path"],
        "multicam_split": "train2_holdout1",
        "multicam_sample_id": sample["sample_id"],
        "multicam_sample_index": 0,
        "max_frames": int(split_cfg["clip_frames"]),
        "frame_indices": None,
        "multicam_train_cameras": list(sample["train_cameras"]),
        "multicam_heldout_cameras": None,
        "multicam_heldout_camera": str(sample["heldout_camera"]),
        "multicam_anchor_camera": str(sample["anchor_camera"]),
        "multicam_condition_camera": str(sample["condition_camera"]),
    }


def camera_cfg_for_sample(sample: dict) -> dict:
    return {
        "rig_init": str(sample["rig_init"]),
        "base_radius": 3.0,
        "rig_radius": 3.0,
        "aist_translation_scale": 1.0,
        "n3d_translation_scale": 1.0,
    }


def main() -> None:
    split_cfg = load_config_file(SPLIT_CONFIG)
    for index, sample in enumerate(split_cfg["samples"]):
        bundle = load_multicam_video_bundle(
            data_cfg=data_cfg_for_sample(split_cfg, sample),
            camera_cfg=camera_cfg_for_sample(sample),
            target_size=int(split_cfg["target_size"]),
            device=torch.device("cpu"),
        )
        if tuple(bundle.train_frames.shape[:2]) != (2, int(split_cfg["clip_frames"])):
            raise RuntimeError(f"{sample['sample_id']} train_frames has shape {tuple(bundle.train_frames.shape)}")
        if tuple(bundle.heldout_frames.shape[:2]) != (1, int(split_cfg["clip_frames"])):
            raise RuntimeError(f"{sample['sample_id']} heldout_frames has shape {tuple(bundle.heldout_frames.shape)}")
        if list(bundle.train_camera_names) != list(sample["train_cameras"]):
            raise RuntimeError(f"{sample['sample_id']} train cameras mismatched: {bundle.train_camera_names}")
        if list(bundle.heldout_camera_names) != [str(sample["heldout_camera"])]:
            raise RuntimeError(f"{sample['sample_id']} heldout camera mismatched: {bundle.heldout_camera_names}")
        print(
            f"{index + 1}. {sample['sample_id']} "
            f"train={bundle.train_camera_names} heldout={bundle.heldout_camera_names} "
            f"train_shape={tuple(bundle.train_frames.shape)} "
            f"heldout_shape={tuple(bundle.heldout_frames.shape)} "
            f"pose_source={bundle.pose_source}"
        )


if __name__ == "__main__":
    main()
