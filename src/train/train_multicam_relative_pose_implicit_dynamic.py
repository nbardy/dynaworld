from __future__ import annotations

from pathlib import Path
from typing import Any

from multicam_relative_pose_trainer import run_training
from train_cli import ConfigInput, run_config_main


def main(config: ConfigInput = None) -> None:
    run_config_main(
        config,
        run_training,
        usage=(
            "Usage: uv run python src/train/train_multicam_relative_pose_implicit_dynamic.py "
            "src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_vjepa_full_relpose_128_16f_8192splats_goodset_train0006_0014_holdout0005.jsonc"
        ),
    )


if __name__ == "__main__":
    main()


__all__ = ["main", "run_training"]
