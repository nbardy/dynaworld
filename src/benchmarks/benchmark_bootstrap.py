from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT
BENCHMARK_DIR = Path(__file__).resolve().parent
TRAIN_ROOT = ROOT / "src" / "train"
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"


def ensure_sys_path(*paths: str | Path, require_exists: bool = False) -> None:
    for path in paths:
        resolved = Path(path)
        if require_exists and not resolved.exists():
            continue
        path_str = str(resolved)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


ensure_sys_path(ROOT, TRAIN_ROOT)


__all__ = ["BENCHMARK_DIR", "PROJECT_ROOT", "ROOT", "TRAIN_ROOT", "VENV_PYTHON", "ensure_sys_path"]
