from __future__ import annotations

import pytest

from star_uvt_projective_interval_backend import (
    projective_cell_interval_backend_config_from_cfg,
    require_projective_interval_atlas_producer,
    resolve_projective_interval_backend_settings,
)
from star_uvt_render_configs import star_uvt_render_configs_from_cfg


def _base_cfg() -> dict:
    return {
        "data": {
            "max_frames": 7,
            "target_size": 48,
        },
        "feature_uvt": {
            "feature_dim": 16,
            "alpha_threshold": 0.125,
            "max_alpha": 0.875,
            "tile_t": 3,
            "tile_capacity": 192,
        },
    }


def test_star_uvt_render_configs_from_cfg_builds_matching_feature_and_uvt_configs() -> None:
    cfg = _base_cfg()

    feature_config, uvt_config = star_uvt_render_configs_from_cfg(cfg)

    assert feature_config.frames == 7
    assert feature_config.height == 48
    assert feature_config.width == 48
    assert feature_config.feature_dim == 16
    assert feature_config.alpha_threshold == 0.125
    assert feature_config.max_alpha == 0.875

    assert uvt_config.frames == feature_config.frames
    assert uvt_config.height == feature_config.height
    assert uvt_config.width == feature_config.width
    assert uvt_config.tile_t == 3
    assert uvt_config.tile_capacity == 192
    assert uvt_config.alpha_threshold == feature_config.alpha_threshold
    assert uvt_config.max_alpha == feature_config.max_alpha


def test_projective_interval_backend_defaults_to_feature_render_domain() -> None:
    cfg = _base_cfg()

    section = resolve_projective_interval_backend_settings(cfg)
    backend_config = projective_cell_interval_backend_config_from_cfg(cfg)

    assert section["enabled"] is False
    assert backend_config.enabled is False
    assert backend_config.image_width == 48
    assert backend_config.image_height == 48
    assert backend_config.tile_size == 16
    assert backend_config.sigma_px == 1.0
    assert backend_config.support_guard_padding == 0.0
    assert backend_config.support_guard_policy == "fixed"
    assert backend_config.support_guard_bisect_steps == 8
    assert backend_config.support_stale_overshoot_epsilon == 0.0
    assert backend_config.support_stale_tail_alpha_epsilon == 0.0
    assert backend_config.support_uv_padding == backend_config.uv_padding
    assert backend_config.refresh_policy == "cadence"
    assert backend_config.refresh_every == 1
    assert backend_config.fallback_render_mode == "error"
    assert backend_config.max_fallback_fraction == 0.20


def test_projective_interval_backend_accepts_explicit_refresh_and_budget_policy() -> None:
    cfg = _base_cfg()
    cfg["feature_uvt"]["projective_interval"] = {
        "enabled": True,
        "sigma_px": 1.5,
        "image_width": 64,
        "image_height": 32,
        "tile_size": 8,
        "uv_padding": 2.0,
        "support_guard_padding": 3.5,
        "support_guard_policy": "budgeted",
        "support_guard_bisect_steps": 5,
        "support_stale_overshoot_epsilon": 0.125,
        "support_stale_tail_alpha_epsilon": 2.5e-4,
        "depth_padding": 0.25,
        "depth_epsilon": 0.01,
        "refresh_policy": "measured",
        "refresh_every": 4,
        "check_visibility": False,
        "allow_ambiguous_fallback": True,
        "fallback_render_mode": "mixed",
        "enforce_complexity_budget": True,
        "max_interval_to_dense_trace_sample_ratio": 0.5,
        "max_fallback_fraction": 0.1,
        "max_cells_per_active_set_group": 3,
    }

    backend_config = projective_cell_interval_backend_config_from_cfg(cfg)

    assert backend_config.enabled is True
    assert backend_config.sigma_px == 1.5
    assert backend_config.image_width == 64
    assert backend_config.image_height == 32
    assert backend_config.tile_size == 8
    assert backend_config.uv_padding == 2.0
    assert backend_config.support_guard_padding == 3.5
    assert backend_config.support_guard_policy == "budgeted"
    assert backend_config.support_guard_bisect_steps == 5
    assert backend_config.support_stale_overshoot_epsilon == 0.125
    assert backend_config.support_stale_tail_alpha_epsilon == 2.5e-4
    assert backend_config.support_uv_padding == 5.5
    assert backend_config.depth_padding == 0.25
    assert backend_config.depth_epsilon == 0.01
    assert backend_config.refresh_policy == "measured"
    assert backend_config.refresh_every == 4
    assert backend_config.check_visibility is False
    assert backend_config.allow_ambiguous_fallback is True
    assert backend_config.fallback_render_mode == "mixed"
    assert backend_config.enforce_complexity_budget is True
    assert backend_config.max_interval_to_dense_trace_sample_ratio == 0.5
    assert backend_config.max_fallback_fraction == 0.1
    assert backend_config.max_cells_per_active_set_group == 3

    cfg = _base_cfg()
    cfg["feature_uvt"]["projective_interval"] = {"support_guard_policy": "local_budgeted"}

    assert projective_cell_interval_backend_config_from_cfg(cfg).support_guard_policy == "local_budgeted"

    cfg = _base_cfg()
    cfg["feature_uvt"]["projective_interval"] = {"support_guard_policy": "trace_budgeted"}

    assert projective_cell_interval_backend_config_from_cfg(cfg).support_guard_policy == "trace_budgeted"

    cfg = _base_cfg()
    cfg["feature_uvt"]["projective_interval"] = {"support_guard_policy": "slack_budgeted"}

    assert projective_cell_interval_backend_config_from_cfg(cfg).support_guard_policy == "slack_budgeted"


def test_projective_interval_backend_rejects_invalid_policy_values() -> None:
    cfg = _base_cfg()
    cfg["feature_uvt"]["projective_interval"] = {"tile_size": 0}

    with pytest.raises(ValueError, match="tile_size"):
        resolve_projective_interval_backend_settings(cfg)

    cfg = _base_cfg()
    cfg["feature_uvt"]["projective_interval"] = {"fallback_render_mode": "banana"}

    with pytest.raises(ValueError, match="fallback_render_mode"):
        resolve_projective_interval_backend_settings(cfg)

    cfg = _base_cfg()
    cfg["feature_uvt"]["projective_interval"] = {"refresh_policy": "banana"}

    with pytest.raises(ValueError, match="refresh_policy"):
        resolve_projective_interval_backend_settings(cfg)

    cfg = _base_cfg()
    cfg["feature_uvt"]["projective_interval"] = {"support_guard_padding": -1.0}

    with pytest.raises(ValueError, match="support_guard_padding"):
        resolve_projective_interval_backend_settings(cfg)

    cfg = _base_cfg()
    cfg["feature_uvt"]["projective_interval"] = {"support_guard_policy": "banana"}

    with pytest.raises(ValueError, match="support_guard_policy"):
        resolve_projective_interval_backend_settings(cfg)

    cfg = _base_cfg()
    cfg["feature_uvt"]["projective_interval"] = {"support_guard_bisect_steps": -1}

    with pytest.raises(ValueError, match="support_guard_bisect_steps"):
        resolve_projective_interval_backend_settings(cfg)

    cfg = _base_cfg()
    cfg["feature_uvt"]["projective_interval"] = {"support_stale_overshoot_epsilon": -0.1}

    with pytest.raises(ValueError, match="support_stale_overshoot_epsilon"):
        resolve_projective_interval_backend_settings(cfg)

    cfg = _base_cfg()
    cfg["feature_uvt"]["projective_interval"] = {"support_stale_tail_alpha_epsilon": -0.1}

    with pytest.raises(ValueError, match="support_stale_tail_alpha_epsilon"):
        resolve_projective_interval_backend_settings(cfg)


def test_projective_interval_enabled_requires_explicit_atlas_producer() -> None:
    cfg = _base_cfg()
    cfg["feature_uvt"]["projective_interval"] = {"enabled": True}

    with pytest.raises(RuntimeError, match="ProjectiveTraceCellTraceAtlas producer"):
        require_projective_interval_atlas_producer(
            cfg,
            owner="unit_test_trainer",
            producer_available=False,
        )

    backend_config = require_projective_interval_atlas_producer(
        cfg,
        owner="unit_test_trainer",
        producer_available=True,
    )
    assert backend_config.enabled is True


def test_projective_interval_disabled_passes_without_atlas_producer() -> None:
    cfg = _base_cfg()

    backend_config = require_projective_interval_atlas_producer(
        cfg,
        owner="unit_test_trainer",
        producer_available=False,
    )

    assert backend_config.enabled is False
