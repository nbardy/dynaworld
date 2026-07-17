from __future__ import annotations

from collections.abc import Mapping
from typing import Any


FEATURE_GRADCACHE_CAP = 64

FEATURE_RENDER_MODE_ORDER = (
    "feature_direct_atomic",
    "feature_direct_gradcache",
    "feature_direct_gradcache_cached_bins",
    "feature_direct_gradcache_reduce",
    "feature_direct_gradcache_reduce_vec4",
    "feature_direct_fixedbin",
)
FEATURE_RENDER_MODES = frozenset(FEATURE_RENDER_MODE_ORDER)
FEATURE_RENDER_BACKWARD_MODES = {
    "feature_direct_atomic": "direct_atomic",
    "feature_direct_fixedbin": "direct_atomic",
    "feature_direct_gradcache": "gradcache",
    "feature_direct_gradcache_cached_bins": "gradcache_cached_bins",
    "feature_direct_gradcache_reduce": "gradcache_reduce_feature_grad",
    "feature_direct_gradcache_reduce_vec4": "gradcache_reduce_feature_grad_vec4",
}
_FEATURE_GRADCACHE_RENDER_MODES = frozenset(
    (
        "feature_direct_gradcache",
        "feature_direct_gradcache_cached_bins",
        "feature_direct_gradcache_reduce",
        "feature_direct_gradcache_reduce_vec4",
    )
)
_FEATURE_GRADCACHE_CAP_REQUIRED_MODES = frozenset(
    (
        "feature_direct_gradcache_cached_bins",
        "feature_direct_gradcache_reduce",
        "feature_direct_gradcache_reduce_vec4",
    )
)


def _validate_feature_render_mode(render_mode: str) -> str:
    if render_mode not in FEATURE_RENDER_MODES:
        expected = ", ".join(sorted(FEATURE_RENDER_MODES))
        raise ValueError(f"feature render mode must be one of: {expected}; got {render_mode!r}")
    return render_mode


def backward_mode_for_feature_render_mode(
    render_mode: str,
    feature_dim: int,
    *,
    cap_plain_gradcache: bool = True,
) -> str:
    """Return the Metal backward mode for a requested feature render mode.

    `cap_plain_gradcache=False` preserves the current trainer dispatch, where
    plain gradcache is allowed through even though reporting treats it as a
    cap-limited mode.
    """

    mode = _validate_feature_render_mode(render_mode)
    requires_cap = mode in _FEATURE_GRADCACHE_CAP_REQUIRED_MODES or (
        cap_plain_gradcache and mode == "feature_direct_gradcache"
    )
    if requires_cap and int(feature_dim) > FEATURE_GRADCACHE_CAP:
        return "direct_atomic"
    return FEATURE_RENDER_BACKWARD_MODES[mode]


def effective_feature_render_mode_for_report(render_mode: str, feature_dim: int) -> str:
    mode = _validate_feature_render_mode(render_mode)
    if mode in _FEATURE_GRADCACHE_RENDER_MODES and int(feature_dim) <= FEATURE_GRADCACHE_CAP:
        return mode
    return "feature_direct_atomic"


def feature_render_mode_fallback_required(
    render_mode: str,
    feature_dim: int,
    *,
    tile_stats: Mapping[str, Any] | None = None,
) -> bool:
    mode = _validate_feature_render_mode(render_mode)
    fixedbin_overflow = (
        mode == "feature_direct_fixedbin"
        and tile_stats is not None
        and int(tile_stats.get("overflow_tile_count", 0)) != 0
    )
    gradcache_over_cap = mode in _FEATURE_GRADCACHE_RENDER_MODES and int(feature_dim) > FEATURE_GRADCACHE_CAP
    return bool(fixedbin_overflow or gradcache_over_cap)


__all__ = [
    "FEATURE_GRADCACHE_CAP",
    "FEATURE_RENDER_BACKWARD_MODES",
    "FEATURE_RENDER_MODE_ORDER",
    "FEATURE_RENDER_MODES",
    "backward_mode_for_feature_render_mode",
    "effective_feature_render_mode_for_report",
    "feature_render_mode_fallback_required",
]
