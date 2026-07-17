from __future__ import annotations

from pathlib import Path
from typing import Any

from precomputed_feature_trainer import run_training
from train_cli import ConfigInput, run_config_main


def main(config: ConfigInput = None) -> None:
    run_config_main(
        config,
        run_training,
        usage=(
            "Usage: uv run python src/train/train_precomputed_feature_implicit_dynamic.py "
            "src/train_configs/local_mac_overfit_ltx_feature_implicit_camera_128_4fps_fast_mac_8192splats.jsonc"
        ),
    )


if __name__ == "__main__":
    main()


__all__ = ["main", "run_training"]
