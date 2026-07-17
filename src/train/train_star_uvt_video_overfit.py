from __future__ import annotations

from star_uvt_video_trainer import run_training
from train_cli import ConfigInput, run_config_main


def main(config: ConfigInput = None) -> None:
    run_config_main(
        config,
        run_training,
        usage="Usage: PYTHONPATH=src/train uv run python src/train/train_star_uvt_video_overfit.py <config.jsonc>",
    )


if __name__ == "__main__":
    main()


__all__ = ["main", "run_training"]
