from __future__ import annotations

from powerfoam_direct_trainer import run_training
from train_cli import ConfigInput, run_config_main


def main(config: ConfigInput = None) -> None:
    run_config_main(
        config,
        run_training,
        usage="Usage: PYTHONPATH=src/train uv run python src/train/train_powerfoam_direct.py <config.jsonc>",
    )


if __name__ == "__main__":
    main()


__all__ = ["main", "run_training"]
