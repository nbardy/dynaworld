from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

BENCHMARK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BENCHMARK_DIR.parents[1]
RAW_METAL_DIR = PROJECT_ROOT / "third_party" / "raw-metal-mlx-gsplat"


class RawMetalUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class RawMetalSettings:
    tile_size: int = 16
    chunk_size: int = 32
    alpha_threshold: float = 1.0 / 255.0
    transmittance_threshold: float = 1.0e-4


def settings_from_config(config: dict[str, Any]) -> RawMetalSettings:
    settings = RawMetalSettings(
        tile_size=int(config.get("tile_size", 16)),
        chunk_size=int(config.get("chunk_size", 32)),
        alpha_threshold=float(config.get("alpha_threshold", 1.0 / 255.0)),
        transmittance_threshold=float(config.get("transmittance_threshold", 1.0e-4)),
    )
    if settings.chunk_size < 1 or settings.chunk_size > 64:
        raise ValueError("raw_metal.chunk_size must be in [1, 64]; the current Metal kernels stage 64 values.")
    if settings.tile_size < 1:
        raise ValueError("raw_metal.tile_size must be positive.")
    return settings


def import_raw_metal():
    if not RAW_METAL_DIR.exists():
        raise RawMetalUnavailable(f"raw Metal experiment directory is missing: {RAW_METAL_DIR}")
    if str(RAW_METAL_DIR) not in sys.path:
        sys.path.insert(0, str(RAW_METAL_DIR))
    try:
        import mlx.core as mx
        from mlx_projected_gaussian_rasterizer import MetalRasterConfig, make_projected_gaussian_rasterizer
    except Exception as exc:
        raise RawMetalUnavailable(
            f"{exc}; run with `uv run --with mlx ...` or install MLX to enable raw_metal"
        ) from exc
    return mx, MetalRasterConfig, make_projected_gaussian_rasterizer


@lru_cache(maxsize=32)
def _cached_rasterizer(
    height: int,
    width: int,
    tile_size: int,
    chunk_size: int,
    alpha_threshold: float,
    transmittance_threshold: float,
    background: tuple[float, float, float],
):
    _mx, MetalRasterConfig, make_projected_gaussian_rasterizer = import_raw_metal()
    cfg = MetalRasterConfig(
        height=height,
        width=width,
        tile_size=tile_size,
        chunk_size=chunk_size,
        alpha_threshold=alpha_threshold,
        transmittance_threshold=transmittance_threshold,
        background=background,
    )
    return make_projected_gaussian_rasterizer(cfg)


def _torch_to_mx_f32(mx, value: torch.Tensor):
    array = value.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    return mx.array(array, mx.float32)


def _mx_to_torch(mx, value, device: torch.device | None = None) -> torch.Tensor:
    mx.eval(value)
    tensor = torch.from_numpy(np.array(value, copy=True))
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor


def _background_tuple(
    background: torch.Tensor | tuple[float, float, float] | list[float],
) -> tuple[float, float, float]:
    if torch.is_tensor(background):
        values = background.detach().to(device="cpu", dtype=torch.float32).reshape(-1).tolist()
    else:
        values = [float(value) for value in background]
    if len(values) != 3:
        raise ValueError(f"raw_metal background must have three channels, got {len(values)}.")
    return float(values[0]), float(values[1]), float(values[2])


def _packed_to_conics_mx(mx, packed):
    means = packed[:, 0:2]
    axis = packed[:, 2:4]
    sigma = packed[:, 4:6]
    opacities = packed[:, 6]

    inv_sigma0 = 1.0 / mx.square(sigma[:, 0])
    inv_sigma1 = 1.0 / mx.square(sigma[:, 1])
    ux = axis[:, 0]
    uy = axis[:, 1]

    conic_a = mx.square(ux) * inv_sigma0 + mx.square(uy) * inv_sigma1
    conic_b = ux * uy * (inv_sigma0 - inv_sigma1)
    conic_c = mx.square(uy) * inv_sigma0 + mx.square(ux) * inv_sigma1
    conics = mx.stack([conic_a, conic_b, conic_c], axis=-1)
    return means, conics, opacities


def render_projected_torch(
    means2d: torch.Tensor,
    conics: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    depths: torch.Tensor,
    height: int,
    width: int,
    background: torch.Tensor | tuple[float, float, float] | list[float],
    config: dict[str, Any],
    output_device: torch.device | None = None,
) -> torch.Tensor:
    mx, _MetalRasterConfig, _make_projected_gaussian_rasterizer = import_raw_metal()
    settings = settings_from_config(config)
    background_values = _background_tuple(background)
    rasterize = _cached_rasterizer(
        int(height),
        int(width),
        settings.tile_size,
        settings.chunk_size,
        settings.alpha_threshold,
        settings.transmittance_threshold,
        background_values,
    )
    image_hwc = rasterize(
        _torch_to_mx_f32(mx, means2d),
        _torch_to_mx_f32(mx, conics),
        _torch_to_mx_f32(mx, colors),
        _torch_to_mx_f32(mx, opacities).reshape((-1,)),
        _torch_to_mx_f32(mx, depths).reshape((-1,)),
    )
    return _mx_to_torch(mx, image_hwc, device=output_device).permute(2, 0, 1).contiguous()


def run_packed_accuracy(
    packed: torch.Tensor,
    depths: torch.Tensor,
    features: torch.Tensor,
    target: torch.Tensor,
    background: torch.Tensor,
    height: int,
    width: int,
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    mx, _MetalRasterConfig, _make_projected_gaussian_rasterizer = import_raw_metal()
    settings = settings_from_config(config)
    background_values = _background_tuple(background)
    rasterize = _cached_rasterizer(
        int(height),
        int(width),
        settings.tile_size,
        settings.chunk_size,
        settings.alpha_threshold,
        settings.transmittance_threshold,
        background_values,
    )

    packed_mx = _torch_to_mx_f32(mx, packed)
    depths_mx = _torch_to_mx_f32(mx, depths).reshape((-1,))
    features_mx = _torch_to_mx_f32(mx, features)
    target_mx = _torch_to_mx_f32(mx, target)

    def loss_fn(packed_arg, features_arg):
        means, conics, opacities = _packed_to_conics_mx(mx, packed_arg)
        image_hwc = rasterize(means, conics, features_arg, opacities, depths_mx)
        image_chw = mx.transpose(image_hwc, (2, 0, 1))
        return mx.mean(mx.square(image_chw - target_mx))

    value, grads = mx.value_and_grad(loss_fn, argnums=(0, 1))(packed_mx, features_mx)
    packed_grad_mx, features_grad_mx = grads

    means, conics, opacities = _packed_to_conics_mx(mx, packed_mx)
    image_hwc = rasterize(means, conics, features_mx, opacities, depths_mx)
    image_chw = mx.transpose(image_hwc, (2, 0, 1))
    mx.eval(value, image_chw, packed_grad_mx, features_grad_mx)

    output = _mx_to_torch(mx, image_chw)
    packed_grad = _mx_to_torch(mx, packed_grad_mx)
    features_grad = _mx_to_torch(mx, features_grad_mx)
    return output, packed_grad, features_grad, float(value.item())
