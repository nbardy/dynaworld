from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import Any

from config_utils import load_config_file


DYNAWORLD_ROOT = Path(__file__).resolve().parents[2]

TRAINER_BY_ARCH = {
    "tokengs": "src/train/train_video_token_implicit_dynamic.py",
    "tokengs_video_implicit_camera": "src/train/train_video_token_implicit_dynamic.py",
    "tokengs_video_known_camera": "src/train/train_video_token_implicit_dynamic.py",
    "precomputed_feature_implicit_camera": "src/train/train_precomputed_feature_implicit_dynamic.py",
    "ltx_feature_implicit_camera": "src/train/train_precomputed_feature_implicit_dynamic.py",
    "wan_vace_feature_implicit_camera": "src/train/train_precomputed_feature_implicit_dynamic.py",
    "multicam_precomputed_feature_implicit_camera": (
        "src/train/train_multicam_precomputed_feature_implicit_dynamic.py"
    ),
    "gauge_fields_material_surfel": "research_experiments/gauge_fields/train.py",
    "splat_baseline_static_3dgs": "research_experiments/gauge_fields/train_splat_baseline.py",
    "splat_baseline_free_dynamic_3dgs": "research_experiments/gauge_fields/train_splat_baseline.py",
}


def _config_arch(config: dict[str, Any], config_path: Path) -> str:
    arch = config.get("arch")
    if arch is None:
        raise ValueError(f"Missing top-level 'arch' in {config_path}.")
    return str(arch).lower()


def trainer_script_for_config(config_path: str | Path) -> Path:
    path = Path(config_path)
    config = load_config_file(path)
    arch = _config_arch(config, path)
    if arch not in TRAINER_BY_ARCH:
        expected = ", ".join(sorted(TRAINER_BY_ARCH))
        raise ValueError(f"Unsupported arch={arch!r} in {path}. Expected one of: {expected}.")
    return DYNAWORLD_ROOT / TRAINER_BY_ARCH[arch]


def run_config(config_path: str | Path) -> None:
    config = Path(config_path)
    script = trainer_script_for_config(config)
    if not config.exists():
        raise FileNotFoundError(f"Missing config: {config}")
    if not script.exists():
        raise FileNotFoundError(f"Missing trainer script for {config}: {script}")

    old_argv = sys.argv
    sys.argv = [str(script), str(config)]
    sys.path.insert(0, str(script.parent))
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = old_argv
        try:
            sys.path.remove(str(script.parent))
        except ValueError:
            pass


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: PYTHONPATH=src/train uv run python src/train/train.py <config.jsonc>")
    run_config(sys.argv[1])


if __name__ == "__main__":
    main()
