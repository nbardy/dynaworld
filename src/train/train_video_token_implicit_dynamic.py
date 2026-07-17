from __future__ import annotations

from pathlib import Path
from typing import Any

from token_gs_trainer import run_training
from train_cli import ConfigInput, run_config_main


def main(config: ConfigInput = None) -> None:
    run_config_main(
        config,
        run_training,
        usage=(
            "Usage: uv run python src/train/train_video_token_implicit_dynamic.py "
            "src/train_configs/local_mac_overfit_video_token_full.jsonc"
        ),
    )


if __name__ == "__main__":
    main()


__all__ = ["main", "run_training"]
