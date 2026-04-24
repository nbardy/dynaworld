from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from config_utils import apply_defaults, load_config_file, path_or_none
from runtime_types import SequenceData
from train_video_token_implicit_dynamic import Trainer as VideoTokenImplicitTrainer
from video_feature_cache import VideoFeatureCache


FEATURE_OPTION_DEFAULTS = {
    "extractor": "ltx",
    "model_id": None,
    "pipeline": None,
    "layers": None,
    "cache_dir": "data/feature_cache/precomputed",
    "sample_cache_key": "precomputed-features-v1",
    "cache_version": 1,
    "force_rebake": False,
    "keep_in_memory": True,
    "save_dtype": "float16",
    "torch_dtype": "float32",
    "prompt": "",
    "negative_prompt": None,
    "height": None,
    "width": None,
    "num_frames": None,
    "num_inference_steps": 1,
    "timesteps": None,
    "timestep": None,
    "guidance_scale": 1.0,
    "output_type": "latent",
    "vjepa_feature_dim": None,
    "vjepa_freeze": True,
    "vjepa_attn_implementation": "sdpa",
    "vjepa_pretrained": True,
    "vjepa_crop_size": None,
    "vjepa_checkpoint_url": None,
}


class PrecomputedFeatureImplicitTrainer(VideoTokenImplicitTrainer):
    @classmethod
    def resolve_config(cls, config: dict[str, Any]) -> dict[str, Any]:
        raw = deepcopy(config)
        raw.setdefault("model", {})
        raw["model"].setdefault("video_encoder_backend", "precomputed")
        cfg = super().resolve_config(raw)
        feature_cfg = cfg.setdefault("features", {})
        apply_defaults(feature_cfg, FEATURE_OPTION_DEFAULTS)
        feature_cfg["cache_dir"] = path_or_none(feature_cfg["cache_dir"])
        if feature_cfg["cache_dir"] is None:
            raise ValueError("features.cache_dir is required for precomputed feature training.")
        if feature_cfg["layers"] is not None:
            if isinstance(feature_cfg["layers"], str):
                feature_cfg["layers"] = [feature_cfg["layers"]]
            else:
                feature_cfg["layers"] = [str(name) for name in feature_cfg["layers"]]
        if feature_cfg["timesteps"] is not None:
            feature_cfg["timesteps"] = [int(timestep) for timestep in feature_cfg["timesteps"]]
        if feature_cfg["timestep"] is not None:
            feature_cfg["timestep"] = int(feature_cfg["timestep"])
        for key in ("height", "width", "num_frames", "vjepa_crop_size", "vjepa_feature_dim"):
            if feature_cfg[key] is not None:
                feature_cfg[key] = int(feature_cfg[key])
        feature_cfg["num_inference_steps"] = int(feature_cfg["num_inference_steps"])
        feature_cfg["guidance_scale"] = float(feature_cfg["guidance_scale"])
        feature_cfg["sample_cache_key"] = str(feature_cfg["sample_cache_key"])
        feature_cfg["cache_version"] = int(feature_cfg["cache_version"])
        if feature_cfg["vjepa_checkpoint_url"] is not None:
            feature_cfg["vjepa_checkpoint_url"] = str(feature_cfg["vjepa_checkpoint_url"])
        feature_cfg["extractor"] = str(feature_cfg["extractor"]).lower()

        if cfg["model"]["video_encoder_backend"] not in {"precomputed", "precomputed_ltx"}:
            raise ValueError(
                "The precomputed feature trainer requires model.video_encoder_backend='precomputed' "
                "or 'precomputed_ltx'."
            )
        return cfg

    def on_sequences_loaded(self) -> None:
        self.feature_cache = VideoFeatureCache(self.cfg["features"], self.device)
        sequences = list(self.train_sequences)
        sequences.extend(self.eval_sequences)
        self.feature_cache.prebake(sequences)

        if self.model_cfg["video_feature_channels"] is None:
            self.model_cfg["video_feature_channels"] = self.feature_cache.infer_channels(self.train_sequences[0])
        if self.model_cfg["video_feature_layers"] is None:
            self.model_cfg["video_feature_layers"] = list(self.model_cfg["video_feature_channels"])

        print(
            "Precomputed video features: "
            f"layers={self.model_cfg['video_feature_layers']}, "
            f"channels={self.model_cfg['video_feature_channels']}"
        )

    def model_input_for_clip(
        self,
        sequence_data: SequenceData,
        clip_frames: torch.Tensor,
        clip_times: torch.Tensor,
    ) -> Any:
        del clip_frames, clip_times
        return self.feature_cache.load_or_bake(sequence_data)

    def run(self) -> None:
        print(
            "Feature cache: "
            f"extractor={self.cfg['features']['extractor']}, "
            f"model_id={self.cfg['features']['model_id']}, "
            f"sample_cache_key={self.cfg['features']['sample_cache_key']}, "
            f"cache_dir={self.cfg['features']['cache_dir']}"
        )
        super().run()


def run_training(config: dict[str, Any]) -> None:
    PrecomputedFeatureImplicitTrainer(config).run()


def main(config: dict[str, Any] | str | Path) -> None:
    if isinstance(config, (str, Path)):
        run_training(load_config_file(config))
    else:
        run_training(config)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: uv run python src/train/train_precomputed_feature_implicit_dynamic.py "
            "src/train_configs/local_mac_overfit_ltx_feature_implicit_camera_128_4fps_fast_mac_8192splats.jsonc"
        )
    main(sys.argv[1])
