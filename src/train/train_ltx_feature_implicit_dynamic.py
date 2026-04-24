from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from config_utils import load_config_file
from train_precomputed_feature_implicit_dynamic import PrecomputedFeatureImplicitTrainer


class LTXFeatureImplicitTrainer(PrecomputedFeatureImplicitTrainer):
    """Backward-compatible LTX-named entrypoint for the shared feature trainer."""


def run_training(config: dict[str, Any]) -> None:
    LTXFeatureImplicitTrainer(config).run()


def main(config: dict[str, Any] | str | Path) -> None:
    if isinstance(config, (str, Path)):
        run_training(load_config_file(config))
    else:
        run_training(config)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: uv run python src/train/train_ltx_feature_implicit_dynamic.py "
            "src/train_configs/local_mac_overfit_ltx_feature_implicit_camera_128_4fps_fast_mac_8192splats.jsonc"
        )
    main(sys.argv[1])
