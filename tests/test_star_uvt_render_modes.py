from __future__ import annotations

import pytest

from star_uvt_render_modes import (
    FEATURE_RENDER_MODE_ORDER,
    backward_mode_for_feature_render_mode,
    effective_feature_render_mode_for_report,
    feature_render_mode_fallback_required,
)


def test_feature_render_mode_order_is_the_matrix_default() -> None:
    assert FEATURE_RENDER_MODE_ORDER == (
        "feature_direct_atomic",
        "feature_direct_gradcache",
        "feature_direct_gradcache_cached_bins",
        "feature_direct_gradcache_reduce",
        "feature_direct_gradcache_reduce_vec4",
        "feature_direct_fixedbin",
    )


@pytest.mark.parametrize(
    ("mode", "expected"),
    (
        ("feature_direct_atomic", "direct_atomic"),
        ("feature_direct_fixedbin", "direct_atomic"),
        ("feature_direct_gradcache", "gradcache"),
        ("feature_direct_gradcache_cached_bins", "gradcache_cached_bins"),
        ("feature_direct_gradcache_reduce", "gradcache_reduce_feature_grad"),
        ("feature_direct_gradcache_reduce_vec4", "gradcache_reduce_feature_grad_vec4"),
    ),
)
def test_backward_mode_for_feature_render_mode_maps_supported_modes(mode: str, expected: str) -> None:
    assert backward_mode_for_feature_render_mode(mode, 32) == expected


def test_backward_mode_for_feature_render_mode_preserves_trainer_plain_gradcache_dispatch() -> None:
    assert backward_mode_for_feature_render_mode("feature_direct_gradcache", 128) == "direct_atomic"
    assert (
        backward_mode_for_feature_render_mode(
            "feature_direct_gradcache",
            128,
            cap_plain_gradcache=False,
        )
        == "gradcache"
    )
    assert (
        backward_mode_for_feature_render_mode(
            "feature_direct_gradcache_reduce_vec4",
            128,
            cap_plain_gradcache=False,
        )
        == "direct_atomic"
    )


def test_report_metadata_matches_feature_gradcache_cap_and_fixedbin_overflow() -> None:
    assert (
        effective_feature_render_mode_for_report("feature_direct_gradcache_reduce_vec4", 32)
        == "feature_direct_gradcache_reduce_vec4"
    )
    assert effective_feature_render_mode_for_report("feature_direct_gradcache", 128) == "feature_direct_atomic"
    assert effective_feature_render_mode_for_report("feature_direct_fixedbin", 32) == "feature_direct_atomic"

    assert not feature_render_mode_fallback_required(
        "feature_direct_fixedbin",
        32,
        tile_stats={"overflow_tile_count": 0},
    )
    assert feature_render_mode_fallback_required(
        "feature_direct_fixedbin",
        32,
        tile_stats={"overflow_tile_count": 1},
    )
    assert feature_render_mode_fallback_required("feature_direct_gradcache_cached_bins", 128)
