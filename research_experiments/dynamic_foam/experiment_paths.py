from __future__ import annotations

import sys
from pathlib import Path


DYNAMIC_FOAM_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = DYNAMIC_FOAM_ROOT.parents[1]
TRAIN_SRC = PROJECT_ROOT / "src" / "train"
POWERFOAM_METAL_ROOT = PROJECT_ROOT / "third_party" / "powerfoam-metal"


def ensure_sys_path(*paths: Path) -> None:
    for path in reversed(paths):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def ensure_train_path() -> None:
    ensure_sys_path(TRAIN_SRC)


def relative_to_project(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


__all__ = [
    "DYNAMIC_FOAM_ROOT",
    "POWERFOAM_METAL_ROOT",
    "PROJECT_ROOT",
    "TRAIN_SRC",
    "ensure_sys_path",
    "ensure_train_path",
    "relative_to_project",
]
