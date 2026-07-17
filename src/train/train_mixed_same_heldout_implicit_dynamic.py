from __future__ import annotations

from pathlib import Path
from typing import Any

from mixed_same_heldout_trainer import run_training
from train_cli import ConfigInput, run_config_main


def main(config: ConfigInput = None) -> None:
    run_config_main(
        config,
        run_training,
        usage="Usage: uv run python src/train/train_mixed_same_heldout_implicit_dynamic.py <config.jsonc>",
    )


if __name__ == "__main__":
    main()


__all__ = ["main", "run_training"]
