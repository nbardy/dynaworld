from __future__ import annotations

from star_uvt_rendered_feature_rgb_probe_trainer import run_probe
from train_cli import ConfigInput, run_config_main


def main(config: ConfigInput = None) -> None:
    run_config_main(
        config,
        run_probe,
        usage="Usage: python src/train/train_star_uvt_rendered_feature_rgb_probe.py <path/to/config.jsonc>",
    )


if __name__ == "__main__":
    main()


__all__ = ["main", "run_probe"]
