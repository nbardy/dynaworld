from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


def load_torch_checkpoint(
    path: Path,
    *,
    map_location: object = None,
    weights_only: bool | None = None,
) -> Any:
    kwargs: dict[str, Any] = {"map_location": map_location}
    if weights_only is not None:
        kwargs["weights_only"] = bool(weights_only)
    return torch.load(Path(path), **kwargs)


def load_checkpoint_mapping(
    path: Path,
    *,
    map_location: object = None,
    label: str = "Checkpoint",
) -> Mapping[str, Any]:
    payload = load_torch_checkpoint(path, map_location=map_location)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} {path} must contain a mapping payload")
    return payload


def model_state_dict_from_checkpoint(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping) and isinstance(checkpoint.get("model"), Mapping):
        return checkpoint["model"]
    if isinstance(checkpoint, Mapping) and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint
    raise ValueError("Expected a checkpoint dict with a 'model' state_dict or a raw state_dict.")


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
