from __future__ import annotations

from typing import Any

from external_paths import ensure_third_party_path

_POWERFOAM_METAL_ROOT = ensure_third_party_path("powerfoam-metal")


def _raster_config_kwargs(render_cfg: dict[str, Any], *, use_tiled: bool) -> dict[str, Any]:
    kwargs = {
        "near_plane": float(render_cfg["near_plane"]),
        "alpha_threshold": float(render_cfg["alpha_threshold"]),
        "transmittance_threshold": float(render_cfg["transmittance_threshold"]),
        "max_alpha": float(render_cfg["max_alpha"]),
        "eps": float(render_cfg["eps"]),
        "texel_temperature": float(render_cfg["texel_temperature"]),
    }
    if use_tiled:
        kwargs.update(
            {
                "use_tiled": bool(render_cfg["use_tiled"]),
                "tiled_builder": str(render_cfg["tiled_builder"]),
            }
        )
    return kwargs


def make_powerfoam_metal_raster_config(render_cfg: dict[str, Any]) -> Any:
    from torch_powerfoam_metal import FoamRasterConfig

    return FoamRasterConfig(**_raster_config_kwargs(render_cfg, use_tiled=True))


def make_dynamic_powerfoam_metal_raster_config(render_cfg: dict[str, Any]) -> Any:
    ensure_third_party_path("dynamic-powerfoam-metal")
    from torch_dynamic_powerfoam_metal import FoamRasterConfig

    return FoamRasterConfig(**_raster_config_kwargs(render_cfg, use_tiled=False))


__all__ = [
    "make_dynamic_powerfoam_metal_raster_config",
    "make_powerfoam_metal_raster_config",
]
