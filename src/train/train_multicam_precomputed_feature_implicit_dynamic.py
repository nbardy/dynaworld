from __future__ import annotations

from pathlib import Path
from typing import Any

from multicam_precomputed_trainer import run_training
from train_cli import ConfigInput, run_config_main


def main(config: ConfigInput = None) -> None:
    run_config_main(
        config,
        run_training,
        usage=(
            "Usage: uv run python src/train/train_multicam_precomputed_feature_implicit_dynamic.py "
            "src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_static_dynamic_96_32_precomputed_vjepa2_1_vitb_384_128_16f_8192splats.jsonc"
        ),
    )


if __name__ == "__main__":
    main()


__all__ = ["main", "run_training"]
