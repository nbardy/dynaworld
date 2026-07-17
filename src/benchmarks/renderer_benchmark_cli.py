from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class ImageSaveTarget:
    resolutions: set[tuple[int, int]]
    splat_counts: set[int]
    set_index: int


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def normalize_resolution(value: Any) -> tuple[int, int]:
    if isinstance(value, int):
        if value < 1:
            raise ValueError(f"Resolution must be positive, got {value}.")
        return value, value
    if isinstance(value, str):
        if "x" in value:
            left, right = value.lower().split("x", 1)
            return normalize_resolution([int(left), int(right)])
        return normalize_resolution(int(value))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        height, width = int(value[0]), int(value[1])
        if height < 1 or width < 1:
            raise ValueError(f"Resolution must be positive, got {height}x{width}.")
        return height, width
    raise ValueError(f"Expected resolution as int, 'HxW', or [height, width], got {value!r}.")


def parse_csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_csv_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_csv_resolutions(value: str) -> list[tuple[int, int]]:
    return [normalize_resolution(part.strip()) for part in value.split(",") if part.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def apply_save_image_cli_overrides(
    cfg: dict[str, Any],
    *,
    save_images: Path | None,
    no_save_images: bool,
) -> dict[str, Any]:
    if save_images is not None:
        cfg.setdefault("save_images", {})
        cfg["save_images"]["enabled"] = True
        cfg["save_images"]["directory"] = str(save_images)
    if no_save_images:
        cfg.setdefault("save_images", {})
        cfg["save_images"]["enabled"] = False
    return cfg


def torch_dtype_from_name(name: str, *, input_sentinel_error: bool = False) -> torch.dtype:
    if name == "input":
        if input_sentinel_error:
            raise ValueError("'input' is a Taichi precision sentinel, not a torch dtype.")
        raise ValueError("Unknown torch dtype: input")
    try:
        return getattr(torch, name)
    except AttributeError as exc:
        raise ValueError(f"Unknown torch dtype: {name}") from exc


def resolve_project_path(path_value: str | Path, project_root: Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = project_root / path
    return path


def safe_filename_part(value: Any, *, allow_dot: bool = True) -> str:
    text = str(value).strip().replace(" ", "_")
    allowed = {"-", "_"}
    if allow_dot:
        allowed.add(".")
    return "".join(char if char.isalnum() or char in allowed else "_" for char in text)


def resolve_image_save_target(
    save_config: dict[str, Any],
    resolutions: list[tuple[int, int]],
    splat_counts: list[int],
) -> ImageSaveTarget:
    if bool(save_config.get("largest_resolution_only", True)):
        max_area = max(height * width for height, width in resolutions)
        target_resolutions = {(height, width) for height, width in resolutions if height * width == max_area}
    else:
        target_resolutions = set(resolutions)

    if bool(save_config.get("largest_splat_count_only", True)):
        target_splat_counts = {max(splat_counts)}
    else:
        target_splat_counts = set(splat_counts)

    return ImageSaveTarget(
        resolutions=target_resolutions,
        splat_counts=target_splat_counts,
        set_index=int(save_config.get("set_index", 0)),
    )


def row_matches_image_save_target(
    row: dict[str, Any],
    target: ImageSaveTarget,
    *,
    required_status: str | None = None,
) -> bool:
    if required_status is not None and row.get("status") != required_status:
        return False
    return (
        (int(row["height"]), int(row["width"])) in target.resolutions
        and int(row["splat_count"]) in target.splat_counts
        and int(row["set_index"]) == target.set_index
    )


def save_chw_image(output: torch.Tensor, path: Path, *, label: str = "CHW image") -> Path:
    from PIL import Image

    image = output.detach()
    if image.ndim != 3:
        raise ValueError(f"Expected {label} as CHW image, got shape {tuple(image.shape)}.")
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    elif image.shape[0] > 3:
        image = image[:3]
    array = image.clamp(0.0, 1.0).nan_to_num(0.0).permute(1, 2, 0).mul(255.0).round().to(torch.uint8).cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)
    return path


__all__ = [
    "apply_save_image_cli_overrides",
    "deep_merge",
    "ImageSaveTarget",
    "normalize_resolution",
    "parse_csv_floats",
    "parse_csv_ints",
    "parse_csv_resolutions",
    "parse_csv_strings",
    "resolve_image_save_target",
    "resolve_project_path",
    "row_matches_image_save_target",
    "safe_filename_part",
    "save_chw_image",
    "torch_dtype_from_name",
]
