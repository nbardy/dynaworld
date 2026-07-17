from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from external_paths import ensure_module_path, third_party_path
from renderers.common import MIN_RENDER_DEPTH, project_gaussians_2d, project_gaussians_2d_batch
from renderers.projection import project_gaussians_2d_camera, project_gaussians_2d_camera_batch

FeatureBackground = float | tuple[float, ...]

FAST_MAC_VARIANTS_DIR = third_party_path("fast-mac-gsplat") / "variants"


def _fast_mac_variant_path(name: str) -> Path:
    return FAST_MAC_VARIANTS_DIR / name


FAST_MAC_V5_DIR = _fast_mac_variant_path("v5")
FAST_MAC_V5_SOFTMAX_GS_DIR = _fast_mac_variant_path("v5_softmax_gs")
FAST_MAC_V5_FEATURES_DIR = _fast_mac_variant_path("v5_features")
FAST_MAC_V6_REFINED_DIR = _fast_mac_variant_path("v6_refined")
FAST_MAC_V6_REFINED_FEATURES_DIR = _fast_mac_variant_path("v6_refined_features")
FAST_MAC_V6_REFINED_FEATURES_F32_REDUCE_DIR = _fast_mac_variant_path("v6_refined_features_f32_reduce")
FAST_MAC_V6_REFINED_FEATURES_F32_ACCUM_DIR = _fast_mac_variant_path("v6_refined_features_f32_accum")
FAST_MAC_V6_REFINED_FEATURES_F32_GRADCACHE_DIR = _fast_mac_variant_path("v6_refined_features_f32_gradcache")
FAST_MAC_V6_REFINED_FEATURES_F32_FIXEDBIN_DIR = _fast_mac_variant_path("v6_refined_features_f32_fixedbin")
FAST_MAC_V6_REFINED_FEATURES_F32_ZERO_BG_DIR = _fast_mac_variant_path("v6_refined_features_f32_zero_bg")
FAST_MAC_V9_FEATURES_GRADCACHE_ZERO_BG_DIR = _fast_mac_variant_path("v9_features_gradcache_zero_bg")
FAST_MAC_V10_FEATURES_GRADCACHE_ZERO_BG_HOSTMETA_DIR = _fast_mac_variant_path(
    "v10_features_gradcache_zero_bg_hostmeta"
)
FAST_MAC_V11_FEATURES_GRADCACHE_ZERO_BG_HOSTMETA_FIXEDBIN_DIR = _fast_mac_variant_path(
    "v11_features_gradcache_zero_bg_hostmeta_fixedbin"
)
FAST_MAC_V13A_TEMPORAL_RECOMPUTE_STATE_DIR = _fast_mac_variant_path("v13a_temporal_recompute_state")
FAST_MAC_V13B_RGB_GRAD_HANDOFF_DIR = _fast_mac_variant_path("v13b_rgb_grad_handoff")


def _float_tuple(value: Any, *, field_name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{field_name} must be numeric, got {value!r}.")
    try:
        return tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a numeric sequence, got {value!r}.") from exc


def _normalize_rgb_background(value: Any) -> tuple[float, float, float]:
    background = _float_tuple(value, field_name="fast_mac.background")
    if len(background) != 3:
        raise ValueError(f"fast_mac.background must contain three RGB values, got {value!r}.")
    return background


def _normalize_feature_background(value: Any) -> FeatureBackground:
    if isinstance(value, (int, float)):
        return float(value)
    background = _float_tuple(value, field_name="fast_mac.feature_background")
    if len(background) == 0:
        raise ValueError("fast_mac.feature_background must be a scalar or a non-empty sequence.")
    return background


def _normalize_optional_bool(value: Any, *, field_name: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"none", "null"}:
            return None
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{field_name} must be a boolean or null, got {value!r}.")


def _normalize_choice(value: Any, *, field_name: str, choices: set[str]) -> str:
    normalized = str(value).strip().lower()
    if normalized not in choices:
        expected = ", ".join(sorted(choices))
        raise ValueError(f"{field_name}={value!r} is not supported. Expected one of: {expected}.")
    return normalized


@dataclass(frozen=True)
class FastMacRendererConfig:
    rgb_variant: str = "v5"
    feature_variant: str = "v5_features"
    tile_size: int = 16
    max_fast_pairs: int = 2048
    alpha_threshold: float = 1.0 / 255.0
    transmittance_threshold: float = 1.0e-4
    background: tuple[float, float, float] = (1.0, 1.0, 1.0)
    feature_background: FeatureBackground = 0.0
    enable_overflow_fallback: bool = True
    inputs_sorted_by_depth: bool = True
    batch_strategy: str = "flatten"
    batch_launch_limit_tiles: int = 262144
    batch_launch_limit_gaussians: int = 262144
    use_active_tiles: bool | None = None
    active_policy: str = "off"
    sort_active_tiles_by_count: bool = True
    active_sparse_fraction_threshold: float = 0.45
    active_dense_multiplier: float = 2.0
    stop_count_mode: str = "adaptive"
    stop_count_dense_threshold: int = 64
    backward_state_strategy: str = "save"
    depth_mode: str = "rank_depth"
    softmax_gs_enabled: bool = False
    softmax_gs_beta: float = 0.0
    softmax_gs_gamma: float = 0.0
    softmax_gs_tape_k: int = 0

    @classmethod
    def from_mapping(
        cls,
        values: dict[str, Any] | None,
        *,
        fallback_tile_size: int,
        fallback_alpha_threshold: float,
    ) -> "FastMacRendererConfig":
        values = values or {}
        return cls(
            rgb_variant=_normalize_choice(
                values.get("rgb_variant", values.get("variant", cls.rgb_variant)),
                field_name="fast_mac.rgb_variant",
                choices={"v5", "v5_softmax_gs", "v6_refined"},
            ),
            feature_variant=_normalize_choice(
                values.get("feature_variant", cls.feature_variant),
                field_name="fast_mac.feature_variant",
                choices={
                    "v5_features",
                    "v6_refined_features",
                    "v6_refined_features_f32_reduce",
                    "v6_refined_features_f32_accum",
                    "v6_refined_features_f32_gradcache",
                    "v6_refined_features_f32_fixedbin",
                    "v6_refined_features_f32_zero_bg",
                    "v9_features_gradcache_zero_bg",
                    "v10_features_gradcache_zero_bg_hostmeta",
                    "v11_features_gradcache_zero_bg_hostmeta_fixedbin",
                    "v13a_temporal_recompute_state",
                    "v13b_rgb_grad_handoff",
                },
            ),
            tile_size=int(values.get("tile_size", fallback_tile_size)),
            max_fast_pairs=int(values.get("max_fast_pairs", cls.max_fast_pairs)),
            alpha_threshold=float(values.get("alpha_threshold", fallback_alpha_threshold)),
            transmittance_threshold=float(values.get("transmittance_threshold", cls.transmittance_threshold)),
            background=_normalize_rgb_background(values.get("background", cls.background)),
            feature_background=_normalize_feature_background(
                values.get("feature_background", cls.feature_background)
            ),
            enable_overflow_fallback=bool(values.get("enable_overflow_fallback", cls.enable_overflow_fallback)),
            inputs_sorted_by_depth=bool(values.get("inputs_sorted_by_depth", cls.inputs_sorted_by_depth)),
            batch_strategy=str(values.get("batch_strategy", cls.batch_strategy)),
            batch_launch_limit_tiles=int(values.get("batch_launch_limit_tiles", cls.batch_launch_limit_tiles)),
            batch_launch_limit_gaussians=int(
                values.get("batch_launch_limit_gaussians", cls.batch_launch_limit_gaussians)
            ),
            use_active_tiles=_normalize_optional_bool(
                values.get("use_active_tiles", cls.use_active_tiles),
                field_name="fast_mac.use_active_tiles",
            ),
            active_policy=_normalize_choice(
                values.get("active_policy", cls.active_policy),
                field_name="fast_mac.active_policy",
                choices={"off", "on", "auto"},
            ),
            sort_active_tiles_by_count=bool(
                values.get("sort_active_tiles_by_count", cls.sort_active_tiles_by_count)
            ),
            active_sparse_fraction_threshold=float(
                values.get("active_sparse_fraction_threshold", cls.active_sparse_fraction_threshold)
            ),
            active_dense_multiplier=float(values.get("active_dense_multiplier", cls.active_dense_multiplier)),
            stop_count_mode=_normalize_choice(
                values.get("stop_count_mode", cls.stop_count_mode),
                field_name="fast_mac.stop_count_mode",
                choices={"always", "never", "adaptive"},
            ),
            stop_count_dense_threshold=int(
                values.get("stop_count_dense_threshold", cls.stop_count_dense_threshold)
            ),
            backward_state_strategy=_normalize_choice(
                values.get("backward_state_strategy", cls.backward_state_strategy),
                field_name="fast_mac.backward_state_strategy",
                choices={"save", "recompute"},
            ),
            depth_mode=_normalize_choice(
                values.get("depth_mode", cls.depth_mode),
                field_name="fast_mac.depth_mode",
                choices={"rank_depth", "center_camera_z"},
            ),
            softmax_gs_enabled=bool(values.get("softmax_gs_enabled", cls.softmax_gs_enabled)),
            softmax_gs_beta=float(values.get("softmax_gs_beta", cls.softmax_gs_beta)),
            softmax_gs_gamma=float(values.get("softmax_gs_gamma", cls.softmax_gs_gamma)),
            softmax_gs_tape_k=int(values.get("softmax_gs_tape_k", cls.softmax_gs_tape_k)),
        )


def _ensure_variant_on_path(variant_dir: Path, *, package_name: str, label: str) -> None:
    ensure_module_path(
        package_name,
        variant_dir,
        missing_message=f"fast-mac-gsplat {label} directory not found: {variant_dir}",
    )


def _ensure_fast_mac_v5_on_path() -> None:
    _ensure_variant_on_path(FAST_MAC_V5_DIR, package_name="torch_gsplat_bridge_v5", label="v5")


def _ensure_fast_mac_v5_softmax_gs_on_path() -> None:
    _ensure_variant_on_path(
        FAST_MAC_V5_SOFTMAX_GS_DIR,
        package_name="torch_gsplat_bridge_v5_softmax_gs",
        label="v5_softmax_gs",
    )


def _ensure_fast_mac_v5_features_on_path() -> None:
    _ensure_variant_on_path(
        FAST_MAC_V5_FEATURES_DIR,
        package_name="torch_gsplat_bridge_v5_features",
        label="v5_features",
    )


def _ensure_fast_mac_v6_refined_on_path() -> None:
    _ensure_variant_on_path(
        FAST_MAC_V6_REFINED_DIR,
        package_name="torch_gsplat_bridge_v6",
        label="v6_refined",
    )


def _ensure_fast_mac_v6_refined_features_on_path() -> None:
    _ensure_variant_on_path(
        FAST_MAC_V6_REFINED_FEATURES_DIR,
        package_name="torch_gsplat_bridge_v6_refined_features",
        label="v6_refined_features",
    )


def _ensure_fast_mac_v6_refined_features_f32_reduce_on_path() -> None:
    _ensure_variant_on_path(
        FAST_MAC_V6_REFINED_FEATURES_F32_REDUCE_DIR,
        package_name="torch_gsplat_bridge_v6_refined_features_f32_reduce",
        label="v6_refined_features_f32_reduce",
    )


def _ensure_fast_mac_v6_refined_features_f32_accum_on_path() -> None:
    _ensure_variant_on_path(
        FAST_MAC_V6_REFINED_FEATURES_F32_ACCUM_DIR,
        package_name="torch_gsplat_bridge_v6_refined_features_f32_accum",
        label="v6_refined_features_f32_accum",
    )


def _ensure_fast_mac_v6_refined_features_f32_gradcache_on_path() -> None:
    _ensure_variant_on_path(
        FAST_MAC_V6_REFINED_FEATURES_F32_GRADCACHE_DIR,
        package_name="torch_gsplat_bridge_v6_refined_features_f32_gradcache",
        label="v6_refined_features_f32_gradcache",
    )


def _ensure_fast_mac_v6_refined_features_f32_fixedbin_on_path() -> None:
    _ensure_variant_on_path(
        FAST_MAC_V6_REFINED_FEATURES_F32_FIXEDBIN_DIR,
        package_name="torch_gsplat_bridge_v6_refined_features_f32_fixedbin",
        label="v6_refined_features_f32_fixedbin",
    )


def _ensure_fast_mac_v6_refined_features_f32_zero_bg_on_path() -> None:
    _ensure_variant_on_path(
        FAST_MAC_V6_REFINED_FEATURES_F32_ZERO_BG_DIR,
        package_name="torch_gsplat_bridge_v6_refined_features_f32_zero_bg",
        label="v6_refined_features_f32_zero_bg",
    )


def _ensure_fast_mac_v9_features_gradcache_zero_bg_on_path() -> None:
    _ensure_variant_on_path(
        FAST_MAC_V9_FEATURES_GRADCACHE_ZERO_BG_DIR,
        package_name="torch_gsplat_bridge_v9_features_gradcache_zero_bg",
        label="v9_features_gradcache_zero_bg",
    )


def _ensure_fast_mac_v10_features_gradcache_zero_bg_hostmeta_on_path() -> None:
    _ensure_variant_on_path(
        FAST_MAC_V10_FEATURES_GRADCACHE_ZERO_BG_HOSTMETA_DIR,
        package_name="torch_gsplat_bridge_v10_features_gradcache_zero_bg_hostmeta",
        label="v10_features_gradcache_zero_bg_hostmeta",
    )


def _ensure_fast_mac_v11_features_gradcache_zero_bg_hostmeta_fixedbin_on_path() -> None:
    _ensure_variant_on_path(
        FAST_MAC_V11_FEATURES_GRADCACHE_ZERO_BG_HOSTMETA_FIXEDBIN_DIR,
        package_name="torch_gsplat_bridge_v11_features_gradcache_zero_bg_hostmeta_fixedbin",
        label="v11_features_gradcache_zero_bg_hostmeta_fixedbin",
    )


def _ensure_fast_mac_v13a_temporal_recompute_state_on_path() -> None:
    _ensure_variant_on_path(
        FAST_MAC_V13A_TEMPORAL_RECOMPUTE_STATE_DIR,
        package_name="torch_gsplat_bridge_v13a_temporal_recompute_state",
        label="v13a_temporal_recompute_state",
    )


def _ensure_fast_mac_v13b_rgb_grad_handoff_on_path() -> None:
    _ensure_variant_on_path(
        FAST_MAC_V13B_RGB_GRAD_HANDOFF_DIR,
        package_name="torch_gsplat_bridge_v13b_rgb_grad_handoff",
        label="v13b_rgb_grad_handoff",
    )


def _make_v5_config(config: FastMacRendererConfig, height: int, width: int):
    _ensure_fast_mac_v5_on_path()
    from torch_gsplat_bridge_v5 import RasterConfig

    return RasterConfig(
        height=height,
        width=width,
        tile_size=config.tile_size,
        max_fast_pairs=config.max_fast_pairs,
        alpha_threshold=config.alpha_threshold,
        transmittance_threshold=config.transmittance_threshold,
        background=config.background,
        enable_overflow_fallback=config.enable_overflow_fallback,
        inputs_sorted_by_depth=config.inputs_sorted_by_depth,
        batch_strategy=config.batch_strategy,
        batch_launch_limit_tiles=config.batch_launch_limit_tiles,
        batch_launch_limit_gaussians=config.batch_launch_limit_gaussians,
    )


def _make_v5_softmax_gs_config(config: FastMacRendererConfig, height: int, width: int):
    _ensure_fast_mac_v5_softmax_gs_on_path()
    from torch_gsplat_bridge_v5_softmax_gs import RasterConfig

    return RasterConfig(
        height=height,
        width=width,
        tile_size=config.tile_size,
        max_fast_pairs=config.max_fast_pairs,
        alpha_threshold=config.alpha_threshold,
        transmittance_threshold=config.transmittance_threshold,
        background=config.background,
        enable_overflow_fallback=config.enable_overflow_fallback,
        inputs_sorted_by_depth=config.inputs_sorted_by_depth,
        batch_strategy=config.batch_strategy,
        batch_launch_limit_tiles=config.batch_launch_limit_tiles,
        batch_launch_limit_gaussians=config.batch_launch_limit_gaussians,
        softmax_gs_enabled=config.softmax_gs_enabled,
        softmax_gs_beta=config.softmax_gs_beta,
        softmax_gs_gamma=config.softmax_gs_gamma,
        softmax_gs_tape_k=config.softmax_gs_tape_k,
    )


def _feature_raster_config_cls(config: FastMacRendererConfig):
    if config.feature_variant == "v5_features":
        _ensure_fast_mac_v5_features_on_path()
        from torch_gsplat_bridge_v5_features import RasterConfig as FeatureRasterConfig

        return FeatureRasterConfig
    if config.feature_variant == "v6_refined_features":
        _ensure_fast_mac_v6_refined_features_on_path()
        from torch_gsplat_bridge_v6_refined_features import RasterConfig as FeatureRasterConfig

        return FeatureRasterConfig
    if config.feature_variant == "v6_refined_features_f32_reduce":
        _ensure_fast_mac_v6_refined_features_f32_reduce_on_path()
        from torch_gsplat_bridge_v6_refined_features_f32_reduce import RasterConfig as FeatureRasterConfig

        return FeatureRasterConfig
    if config.feature_variant == "v6_refined_features_f32_accum":
        _ensure_fast_mac_v6_refined_features_f32_accum_on_path()
        from torch_gsplat_bridge_v6_refined_features_f32_accum import RasterConfig as FeatureRasterConfig

        return FeatureRasterConfig
    if config.feature_variant == "v6_refined_features_f32_gradcache":
        _ensure_fast_mac_v6_refined_features_f32_gradcache_on_path()
        from torch_gsplat_bridge_v6_refined_features_f32_gradcache import RasterConfig as FeatureRasterConfig

        return FeatureRasterConfig
    if config.feature_variant == "v6_refined_features_f32_fixedbin":
        _ensure_fast_mac_v6_refined_features_f32_fixedbin_on_path()
        from torch_gsplat_bridge_v6_refined_features_f32_fixedbin import RasterConfig as FeatureRasterConfig

        return FeatureRasterConfig
    if config.feature_variant == "v6_refined_features_f32_zero_bg":
        _ensure_fast_mac_v6_refined_features_f32_zero_bg_on_path()
        from torch_gsplat_bridge_v6_refined_features_f32_zero_bg import RasterConfig as FeatureRasterConfig

        return FeatureRasterConfig
    if config.feature_variant == "v9_features_gradcache_zero_bg":
        _ensure_fast_mac_v9_features_gradcache_zero_bg_on_path()
        from torch_gsplat_bridge_v9_features_gradcache_zero_bg import RasterConfig as FeatureRasterConfig

        return FeatureRasterConfig
    if config.feature_variant == "v10_features_gradcache_zero_bg_hostmeta":
        _ensure_fast_mac_v10_features_gradcache_zero_bg_hostmeta_on_path()
        from torch_gsplat_bridge_v10_features_gradcache_zero_bg_hostmeta import (
            RasterConfig as FeatureRasterConfig,
        )

        return FeatureRasterConfig
    if config.feature_variant == "v11_features_gradcache_zero_bg_hostmeta_fixedbin":
        _ensure_fast_mac_v11_features_gradcache_zero_bg_hostmeta_fixedbin_on_path()
        from torch_gsplat_bridge_v11_features_gradcache_zero_bg_hostmeta_fixedbin import (
            RasterConfig as FeatureRasterConfig,
        )

        return FeatureRasterConfig
    if config.feature_variant == "v13a_temporal_recompute_state":
        _ensure_fast_mac_v13a_temporal_recompute_state_on_path()
        from torch_gsplat_bridge_v13a_temporal_recompute_state import RasterConfig as FeatureRasterConfig

        return FeatureRasterConfig
    if config.feature_variant == "v13b_rgb_grad_handoff":
        _ensure_fast_mac_v13b_rgb_grad_handoff_on_path()
        from torch_gsplat_bridge_v13b_rgb_grad_handoff import RasterConfig as FeatureRasterConfig

        return FeatureRasterConfig
    raise ValueError(f"Unsupported fast_mac.feature_variant={config.feature_variant!r}.")


def _make_feature_config(config: FastMacRendererConfig, height: int, width: int, feature_dim: int):
    FeatureRasterConfig = _feature_raster_config_cls(config)
    if isinstance(config.feature_background, (int, float)):
        background = (config.feature_background,)
    elif len(config.feature_background) in (1, feature_dim):
        background = config.feature_background
    else:
        raise ValueError(
            "fast_mac.feature_background must be a scalar or contain "
            f"feature_dim={feature_dim} values; got {len(config.feature_background)}."
        )
    kwargs = {
        "height": height,
        "width": width,
        "tile_size": config.tile_size,
        "max_fast_pairs": config.max_fast_pairs,
        "alpha_threshold": config.alpha_threshold,
        "transmittance_threshold": config.transmittance_threshold,
        "background": background,
        "enable_overflow_fallback": config.enable_overflow_fallback,
        "inputs_sorted_by_depth": config.inputs_sorted_by_depth,
        "batch_strategy": config.batch_strategy,
        "batch_launch_limit_tiles": config.batch_launch_limit_tiles,
        "batch_launch_limit_gaussians": config.batch_launch_limit_gaussians,
    }
    if config.feature_variant in {
        "v6_refined_features",
        "v6_refined_features_f32_reduce",
        "v6_refined_features_f32_accum",
        "v6_refined_features_f32_gradcache",
        "v6_refined_features_f32_fixedbin",
        "v6_refined_features_f32_zero_bg",
        "v9_features_gradcache_zero_bg",
        "v10_features_gradcache_zero_bg_hostmeta",
        "v11_features_gradcache_zero_bg_hostmeta_fixedbin",
        "v13a_temporal_recompute_state",
        "v13b_rgb_grad_handoff",
    }:
        kwargs.update(
            {
                "use_active_tiles": config.use_active_tiles,
                "active_policy": config.active_policy,
                "sort_active_tiles_by_count": config.sort_active_tiles_by_count,
                "active_sparse_fraction_threshold": config.active_sparse_fraction_threshold,
                "active_dense_multiplier": config.active_dense_multiplier,
                "stop_count_mode": config.stop_count_mode,
                "stop_count_dense_threshold": config.stop_count_dense_threshold,
            }
        )
    if config.feature_variant == "v13a_temporal_recompute_state":
        kwargs["backward_state_strategy"] = config.backward_state_strategy
    return FeatureRasterConfig(**kwargs)


def _rasterize_features_projected(
    means2d: torch.Tensor,
    conics: torch.Tensor,
    colors: torch.Tensor,
    projected_opacities: torch.Tensor,
    depths: torch.Tensor,
    config: FastMacRendererConfig,
    height: int,
    width: int,
    feature_dim: int,
):
    if config.feature_variant == "v5_features":
        _ensure_fast_mac_v5_features_on_path()
        from torch_gsplat_bridge_v5_features import rasterize_projected_gaussians

        return rasterize_projected_gaussians(
            means2d,
            conics,
            colors,
            projected_opacities,
            depths,
            _make_feature_config(config, height, width, feature_dim),
        )
    if config.feature_variant == "v6_refined_features":
        _ensure_fast_mac_v6_refined_features_on_path()
        from torch_gsplat_bridge_v6_refined_features import rasterize_projected_gaussians

        return rasterize_projected_gaussians(
            means2d,
            conics,
            colors,
            projected_opacities,
            depths,
            _make_feature_config(config, height, width, feature_dim),
        )
    if config.feature_variant == "v6_refined_features_f32_reduce":
        _ensure_fast_mac_v6_refined_features_f32_reduce_on_path()
        from torch_gsplat_bridge_v6_refined_features_f32_reduce import rasterize_projected_gaussians

        return rasterize_projected_gaussians(
            means2d,
            conics,
            colors,
            projected_opacities,
            depths,
            _make_feature_config(config, height, width, feature_dim),
        )
    if config.feature_variant == "v6_refined_features_f32_accum":
        _ensure_fast_mac_v6_refined_features_f32_accum_on_path()
        from torch_gsplat_bridge_v6_refined_features_f32_accum import rasterize_projected_gaussians

        return rasterize_projected_gaussians(
            means2d,
            conics,
            colors,
            projected_opacities,
            depths,
            _make_feature_config(config, height, width, feature_dim),
        )
    if config.feature_variant == "v6_refined_features_f32_gradcache":
        _ensure_fast_mac_v6_refined_features_f32_gradcache_on_path()
        from torch_gsplat_bridge_v6_refined_features_f32_gradcache import rasterize_projected_gaussians

        return rasterize_projected_gaussians(
            means2d,
            conics,
            colors,
            projected_opacities,
            depths,
            _make_feature_config(config, height, width, feature_dim),
        )
    if config.feature_variant == "v6_refined_features_f32_fixedbin":
        _ensure_fast_mac_v6_refined_features_f32_fixedbin_on_path()
        from torch_gsplat_bridge_v6_refined_features_f32_fixedbin import rasterize_projected_gaussians

        return rasterize_projected_gaussians(
            means2d,
            conics,
            colors,
            projected_opacities,
            depths,
            _make_feature_config(config, height, width, feature_dim),
        )
    if config.feature_variant == "v6_refined_features_f32_zero_bg":
        _ensure_fast_mac_v6_refined_features_f32_zero_bg_on_path()
        from torch_gsplat_bridge_v6_refined_features_f32_zero_bg import rasterize_projected_gaussians

        return rasterize_projected_gaussians(
            means2d,
            conics,
            colors,
            projected_opacities,
            depths,
            _make_feature_config(config, height, width, feature_dim),
        )
    if config.feature_variant == "v9_features_gradcache_zero_bg":
        _ensure_fast_mac_v9_features_gradcache_zero_bg_on_path()
        from torch_gsplat_bridge_v9_features_gradcache_zero_bg import rasterize_projected_gaussians

        return rasterize_projected_gaussians(
            means2d,
            conics,
            colors,
            projected_opacities,
            depths,
            _make_feature_config(config, height, width, feature_dim),
        )
    if config.feature_variant == "v10_features_gradcache_zero_bg_hostmeta":
        _ensure_fast_mac_v10_features_gradcache_zero_bg_hostmeta_on_path()
        from torch_gsplat_bridge_v10_features_gradcache_zero_bg_hostmeta import rasterize_projected_gaussians

        return rasterize_projected_gaussians(
            means2d,
            conics,
            colors,
            projected_opacities,
            depths,
            _make_feature_config(config, height, width, feature_dim),
        )
    if config.feature_variant == "v11_features_gradcache_zero_bg_hostmeta_fixedbin":
        _ensure_fast_mac_v11_features_gradcache_zero_bg_hostmeta_fixedbin_on_path()
        from torch_gsplat_bridge_v11_features_gradcache_zero_bg_hostmeta_fixedbin import (
            rasterize_projected_gaussians,
        )

        return rasterize_projected_gaussians(
            means2d,
            conics,
            colors,
            projected_opacities,
            depths,
            _make_feature_config(config, height, width, feature_dim),
        )
    if config.feature_variant == "v13a_temporal_recompute_state":
        _ensure_fast_mac_v13a_temporal_recompute_state_on_path()
        from torch_gsplat_bridge_v13a_temporal_recompute_state import rasterize_projected_gaussians

        return rasterize_projected_gaussians(
            means2d,
            conics,
            colors,
            projected_opacities,
            depths,
            _make_feature_config(config, height, width, feature_dim),
        )
    if config.feature_variant == "v13b_rgb_grad_handoff":
        _ensure_fast_mac_v13b_rgb_grad_handoff_on_path()
        from torch_gsplat_bridge_v13b_rgb_grad_handoff import rasterize_projected_gaussians

        return rasterize_projected_gaussians(
            means2d,
            conics,
            colors,
            projected_opacities,
            depths,
            _make_feature_config(config, height, width, feature_dim),
        )
    raise ValueError(f"Unsupported fast_mac.feature_variant={config.feature_variant!r}.")


def _make_v6_refined_config(config: FastMacRendererConfig, height: int, width: int):
    _ensure_fast_mac_v6_refined_on_path()
    from torch_gsplat_bridge_v6 import RasterConfig

    return RasterConfig(
        height=height,
        width=width,
        tile_size=config.tile_size,
        max_fast_pairs=config.max_fast_pairs,
        alpha_threshold=config.alpha_threshold,
        transmittance_threshold=config.transmittance_threshold,
        background=config.background,
        enable_overflow_fallback=config.enable_overflow_fallback,
        batch_strategy=config.batch_strategy,
        batch_launch_limit_tiles=config.batch_launch_limit_tiles,
        batch_launch_limit_gaussians=config.batch_launch_limit_gaussians,
        use_active_tiles=config.use_active_tiles,
        active_policy=config.active_policy,
        sort_active_tiles_by_count=config.sort_active_tiles_by_count,
        active_sparse_fraction_threshold=config.active_sparse_fraction_threshold,
        active_dense_multiplier=config.active_dense_multiplier,
        stop_count_mode=config.stop_count_mode,
        stop_count_dense_threshold=config.stop_count_dense_threshold,
    )


def _rasterize_rgb_projected(
    means2d: torch.Tensor,
    conics: torch.Tensor,
    colors: torch.Tensor,
    projected_opacities: torch.Tensor,
    depths: torch.Tensor,
    config: FastMacRendererConfig,
    height: int,
    width: int,
) -> torch.Tensor:
    if config.rgb_variant == "v5":
        _ensure_fast_mac_v5_on_path()
        from torch_gsplat_bridge_v5 import rasterize_projected_gaussians

        return rasterize_projected_gaussians(
            means2d,
            conics,
            colors,
            projected_opacities,
            depths,
            _make_v5_config(config, height, width),
        )
    if config.rgb_variant == "v5_softmax_gs":
        _ensure_fast_mac_v5_softmax_gs_on_path()
        from torch_gsplat_bridge_v5_softmax_gs import rasterize_projected_gaussians

        if config.softmax_gs_enabled and config.depth_mode != "center_camera_z":
            raise ValueError("fast_mac.depth_mode='center_camera_z' is required when Softmax-GS is enabled.")
        return rasterize_projected_gaussians(
            means2d,
            conics,
            colors,
            projected_opacities,
            depths,
            _make_v5_softmax_gs_config(config, height, width),
        )
    if config.rgb_variant == "v6_refined":
        _ensure_fast_mac_v6_refined_on_path()
        from torch_gsplat_bridge_v6 import rasterize_projected_gaussians

        return rasterize_projected_gaussians(
            means2d,
            conics,
            colors,
            projected_opacities,
            depths,
            _make_v6_refined_config(config, height, width),
        )
    raise ValueError(f"Unsupported fast_mac.rgb_variant={config.rgb_variant!r}.")


def _conics_from_inv_cov(inv_cov2d: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            inv_cov2d[..., 0, 0],
            0.5 * (inv_cov2d[..., 0, 1] + inv_cov2d[..., 1, 0]),
            inv_cov2d[..., 1, 1],
        ],
        dim=-1,
    ).contiguous()


def _rank_depths(
    gaussian_count: int, *, batch_size: int | None, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    depths = torch.arange(gaussian_count, device=device, dtype=dtype)
    if gaussian_count > 1:
        depths = depths / float(gaussian_count - 1)
    if batch_size is None:
        return depths.contiguous()
    return depths.view(1, -1).expand(batch_size, -1).contiguous()


def describe_fast_mac_depth_signal(config: FastMacRendererConfig) -> dict[str, object]:
    return {
        "kind": config.depth_mode,
        "softmax_gs_ready": config.depth_mode == "center_camera_z",
    }


def _camera_centers(means3d: torch.Tensor, camera_to_world: torch.Tensor | None) -> torch.Tensor:
    if camera_to_world is None:
        return means3d
    camera_to_world = camera_to_world.to(device=means3d.device, dtype=means3d.dtype)
    rotation_cw = camera_to_world[:3, :3]
    translation = camera_to_world[:3, 3]
    return (means3d - translation.unsqueeze(0)) @ rotation_cw


def _center_camera_depths(means3d: torch.Tensor, camera_to_world: torch.Tensor | None) -> torch.Tensor:
    centers = _camera_centers(means3d, camera_to_world)
    depths = centers[:, 2]
    sorted_idx = torch.argsort(depths, descending=False, stable=True)
    return depths[sorted_idx].contiguous()


def _camera_centers_batch(means3d: torch.Tensor, camera_to_world: torch.Tensor | None) -> torch.Tensor:
    if camera_to_world is None:
        return means3d
    camera_to_world = camera_to_world.to(device=means3d.device, dtype=means3d.dtype)
    if camera_to_world.ndim == 2:
        camera_to_world = camera_to_world.unsqueeze(0).expand(means3d.shape[0], -1, -1)
    rotation_cw = camera_to_world[:, :3, :3]
    translation = camera_to_world[:, :3, 3]
    return (means3d - translation[:, None, :]) @ rotation_cw


def _center_camera_depths_batch(means3d: torch.Tensor, camera_to_world: torch.Tensor | None) -> torch.Tensor:
    centers = _camera_centers_batch(means3d, camera_to_world)
    depths = centers[..., 2]
    sorted_idx = torch.argsort(depths, dim=1, descending=False, stable=True)
    return torch.gather(depths, dim=1, index=sorted_idx).contiguous()


def _camera_to_world_from_camera(camera) -> torch.Tensor | None:
    if camera is None:
        return None
    return camera.camera_to_world


def _camera_to_world_from_cameras(cameras, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
    if cameras is None:
        return None
    return torch.stack(
        [camera.camera_to_world.to(device=device, dtype=dtype) for camera in cameras],
        dim=0,
    )


def project_for_fast_mac(
    means3d: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: torch.Tensor,
    rgbs: torch.Tensor,
    fx,
    fy,
    cx,
    cy,
    *,
    camera=None,
    projection_mode: str = "legacy_pinhole",
    camera_to_world: torch.Tensor | None = None,
    near_plane: float = MIN_RENDER_DEPTH,
    depth_mode: str = "rank_depth",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if projection_mode == "camera_model":
        if camera is None:
            raise ValueError("camera_model projection requires a CameraSpec.")
        means2d, inv_cov2d, _cov2d, opacities, colors = project_gaussians_2d_camera(
            means3d,
            scales,
            quats,
            opacities,
            rgbs,
            camera,
            near_plane=near_plane,
        )
    elif projection_mode == "legacy_pinhole":
        means2d, inv_cov2d, _cov2d, opacities, colors = project_gaussians_2d(
            means3d,
            scales,
            quats,
            opacities,
            rgbs,
            fx,
            fy,
            cx,
            cy,
            camera_to_world=camera_to_world,
            near_plane=near_plane,
        )
    else:
        raise ValueError(f"Unknown projection_mode: {projection_mode}")
    if depth_mode == "rank_depth":
        depths = _rank_depths(means2d.shape[0], batch_size=None, device=means2d.device, dtype=means2d.dtype)
    elif depth_mode == "center_camera_z":
        depth_camera_to_world = camera_to_world if projection_mode == "legacy_pinhole" else _camera_to_world_from_camera(camera)
        depths = _center_camera_depths(means3d, depth_camera_to_world)
    else:
        raise ValueError(f"Unknown fast_mac depth_mode: {depth_mode}")
    return (
        means2d.contiguous(),
        _conics_from_inv_cov(inv_cov2d),
        colors.contiguous(),
        opacities.squeeze(-1).contiguous(),
        depths,
    )


def project_for_fast_mac_batch(
    means3d: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: torch.Tensor,
    rgbs: torch.Tensor,
    fx,
    fy,
    cx,
    cy,
    *,
    cameras=None,
    projection_mode: str = "legacy_pinhole",
    camera_to_world: torch.Tensor | None = None,
    near_plane: float = MIN_RENDER_DEPTH,
    depth_mode: str = "rank_depth",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if projection_mode == "camera_model":
        if cameras is None:
            raise ValueError("camera_model batch projection requires CameraSpec values.")
        means2d, inv_cov2d, _cov2d, opacities, colors = project_gaussians_2d_camera_batch(
            means3d,
            scales,
            quats,
            opacities,
            rgbs,
            cameras,
            near_plane=near_plane,
        )
    elif projection_mode == "legacy_pinhole":
        means2d, inv_cov2d, _cov2d, opacities, colors = project_gaussians_2d_batch(
            means3d,
            scales,
            quats,
            opacities,
            rgbs,
            fx,
            fy,
            cx,
            cy,
            camera_to_world=camera_to_world,
            near_plane=near_plane,
        )
    else:
        raise ValueError(f"Unknown projection_mode: {projection_mode}")
    if depth_mode == "rank_depth":
        depths = _rank_depths(
            means2d.shape[1],
            batch_size=means2d.shape[0],
            device=means2d.device,
            dtype=means2d.dtype,
        )
    elif depth_mode == "center_camera_z":
        if projection_mode == "legacy_pinhole":
            depth_camera_to_world = camera_to_world
        else:
            depth_camera_to_world = _camera_to_world_from_cameras(cameras, device=means3d.device, dtype=means3d.dtype)
        depths = _center_camera_depths_batch(means3d, depth_camera_to_world)
    else:
        raise ValueError(f"Unknown fast_mac depth_mode: {depth_mode}")
    return (
        means2d.contiguous(),
        _conics_from_inv_cov(inv_cov2d),
        colors.contiguous(),
        opacities.squeeze(-1).contiguous(),
        depths,
    )


def render_fast_mac_3dgs(
    means3d: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: torch.Tensor,
    rgbs: torch.Tensor,
    height: int,
    width: int,
    fx,
    fy,
    cx,
    cy,
    *,
    camera=None,
    projection_mode: str = "legacy_pinhole",
    camera_to_world: torch.Tensor | None = None,
    near_plane: float = MIN_RENDER_DEPTH,
    config: FastMacRendererConfig,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    feature_dim = rgbs.shape[-1]
    means2d, conics, colors, projected_opacities, depths = project_for_fast_mac(
        means3d.float(),
        scales.float(),
        quats.float(),
        opacities.float(),
        rgbs.float(),
        fx,
        fy,
        cx,
        cy,
        camera=camera,
        projection_mode=projection_mode,
        camera_to_world=camera_to_world.float() if camera_to_world is not None else None,
        near_plane=near_plane,
        depth_mode=config.depth_mode,
    )
    # F=3 -> selected RGB variant, output clamped to [0,1] for direct loss. Returns (features, None).
    # F!=3 -> v5_features (raw F-channel feature map + accumulated alpha mask).
    if feature_dim == 3:
        image_hwc = _rasterize_rgb_projected(
            means2d,
            conics,
            colors,
            projected_opacities,
            depths,
            config,
            height,
            width,
        )
        return image_hwc.clamp(0.0, 1.0).permute(2, 0, 1).contiguous(), None
    rasterize_out = _rasterize_features_projected(
        means2d,
        conics,
        colors,
        projected_opacities,
        depths,
        config,
        height,
        width,
        feature_dim,
    )
    if isinstance(rasterize_out, tuple):
        image_hwf, alpha_hw = rasterize_out
    else:
        image_hwf, alpha_hw = rasterize_out, None
    return image_hwf.permute(2, 0, 1).contiguous(), alpha_hw


def render_fast_mac_3dgs_batch(
    means3d: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: torch.Tensor,
    rgbs: torch.Tensor,
    height: int,
    width: int,
    fx,
    fy,
    cx,
    cy,
    *,
    cameras=None,
    projection_mode: str = "legacy_pinhole",
    camera_to_world: torch.Tensor | None = None,
    near_plane: float = MIN_RENDER_DEPTH,
    config: FastMacRendererConfig,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    feature_dim = rgbs.shape[-1]
    means2d, conics, colors, projected_opacities, depths = project_for_fast_mac_batch(
        means3d.float(),
        scales.float(),
        quats.float(),
        opacities.float(),
        rgbs.float(),
        fx,
        fy,
        cx,
        cy,
        cameras=cameras,
        projection_mode=projection_mode,
        camera_to_world=camera_to_world.float() if camera_to_world is not None else None,
        near_plane=near_plane,
        depth_mode=config.depth_mode,
    )
    # Returns (features, alpha_mask). Alpha is None for the F=3 RGB path
    # and a tensor of shape [B, H, W] for the F!=3 v5_features path.
    if feature_dim == 3:
        image_bhwc = _rasterize_rgb_projected(
            means2d,
            conics,
            colors,
            projected_opacities,
            depths,
            config,
            height,
            width,
        )
        return image_bhwc.clamp(0.0, 1.0).permute(0, 3, 1, 2).contiguous(), None
    rasterize_out = _rasterize_features_projected(
        means2d,
        conics,
        colors,
        projected_opacities,
        depths,
        config,
        height,
        width,
        feature_dim,
    )
    if isinstance(rasterize_out, tuple):
        image_bhwf, alpha_bhw = rasterize_out
    else:
        image_bhwf, alpha_bhw = rasterize_out, None
    return image_bhwf.permute(0, 3, 1, 2).contiguous(), alpha_bhw


__all__ = [
    "FastMacRendererConfig",
    "describe_fast_mac_depth_signal",
    "project_for_fast_mac",
    "project_for_fast_mac_batch",
    "render_fast_mac_3dgs",
    "render_fast_mac_3dgs_batch",
]
