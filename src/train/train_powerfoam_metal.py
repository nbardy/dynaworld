from __future__ import annotations

from powerfoam_metal_trainer import run_training
from train_cli import ConfigInput, run_config_main


def main(config: ConfigInput = None) -> None:
    run_config_main(
        config,
        run_training,
        usage="Usage: python src/train/train_powerfoam_metal.py <path/to/config.jsonc>",
    )


if __name__ == "__main__":
    main()


__all__ = ["main", "run_training"]
