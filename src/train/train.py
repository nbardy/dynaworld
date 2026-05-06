from __future__ import annotations

import sys
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from config_utils import load_config_file


@dataclass(frozen=True)
class TrainerEntry:
    module: str
    runner: str = "run_training"


TRAINER_BY_ARCH = {
    "tokengs": TrainerEntry("train_video_token_implicit_dynamic"),
    "tokengs_video_implicit_camera": TrainerEntry("train_video_token_implicit_dynamic"),
    "tokengs_video_known_camera": TrainerEntry("train_video_token_implicit_dynamic"),
    "precomputed_feature_implicit_camera": TrainerEntry("train_precomputed_feature_implicit_dynamic"),
    "ltx_feature_implicit_camera": TrainerEntry("train_precomputed_feature_implicit_dynamic"),
    "wan_vace_feature_implicit_camera": TrainerEntry("train_precomputed_feature_implicit_dynamic"),
    "powerfoam_direct": TrainerEntry("train_powerfoam_direct"),
    "powerfoam_metal": TrainerEntry("train_powerfoam_metal"),
    "dynamic_powerfoam_metal": TrainerEntry("train_dynamic_powerfoam_metal"),
    "dynamic_gauge_foam": TrainerEntry("train_dynamic_gauge_foam"),
    "multicam_precomputed_feature_implicit_camera": TrainerEntry(
        "train_multicam_precomputed_feature_implicit_dynamic"
    ),
}


def _config_arch(config: dict[str, Any], config_path: Path) -> str:
    arch = config.get("arch")
    if arch is None:
        raise ValueError(f"Missing top-level 'arch' in {config_path}.")
    return str(arch).lower()


def trainer_entry_for_config(config_path: str | Path) -> TrainerEntry:
    path = Path(config_path)
    config = load_config_file(path)
    arch = _config_arch(config, path)
    if arch not in TRAINER_BY_ARCH:
        expected = ", ".join(sorted(TRAINER_BY_ARCH))
        raise ValueError(f"Unsupported arch={arch!r} in {path}. Expected one of: {expected}.")
    return TRAINER_BY_ARCH[arch]


def run_config(config_path: str | Path) -> None:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    config = load_config_file(path)
    arch = _config_arch(config, path)
    if arch not in TRAINER_BY_ARCH:
        expected = ", ".join(sorted(TRAINER_BY_ARCH))
        raise ValueError(f"Unsupported arch={arch!r} in {path}. Expected one of: {expected}.")
    entry = TRAINER_BY_ARCH[arch]
    runner = getattr(import_module(entry.module), entry.runner)
    runner(config)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: PYTHONPATH=src/train uv run python src/train/train.py <config.jsonc>")
    run_config(sys.argv[1])


if __name__ == "__main__":
    main()
