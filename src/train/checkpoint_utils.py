from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def atomic_torch_save(payload: Any, path: Path) -> None:
    """Write a torch checkpoint without leaving a truncated target on failure."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    try:
        if tmp_path.exists():
            tmp_path.unlink()
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
