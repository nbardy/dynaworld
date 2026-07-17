from __future__ import annotations

from star_uvt_feature_overfit_trainer import run_training
from train_cli import ConfigInput, run_config_main

__all__ = ["main", "run_training"]


def main(config: ConfigInput = None) -> None:
    run_config_main(
        config,
        run_training,
        usage="Usage: python src/train/train_star_uvt_feature_overfit.py <path/to/config.jsonc>",
    )


if __name__ == "__main__":
    main()
