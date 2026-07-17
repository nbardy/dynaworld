from __future__ import annotations

from train_cli import run_path_arg
from trainer_registry import (
    EXTERNAL_TRAINER_BY_ARCH,
    TRAINER_BY_ARCH,
    ExternalTrainerEntry,
    TrainerEntry,
    config_arch,
    load_config_and_entry,
    run_config,
    trainer_entry_for_arch,
    trainer_entry_for_config,
)

__all__ = [
    "EXTERNAL_TRAINER_BY_ARCH",
    "TRAINER_BY_ARCH",
    "ExternalTrainerEntry",
    "TrainerEntry",
    "config_arch",
    "load_config_and_entry",
    "run_config",
    "trainer_entry_for_arch",
    "trainer_entry_for_config",
]


def main() -> None:
    run_path_arg(
        run_config,
        usage="Usage: PYTHONPATH=src/train uv run python src/train/train.py <config.jsonc>",
    )


if __name__ == "__main__":
    main()
