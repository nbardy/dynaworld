from __future__ import annotations

import pytest

from render_dispatch import (
    decoded_token_count_from_model_config,
    pick_renderer_mode_from_config,
    token_layout_detail_levels,
    token_summary_from_model_config,
)


def _base_cfg() -> dict[str, object]:
    return {
        "model": {
            "tokens": 8,
            "gaussians_per_token": 2,
            "use_static_dynamic_split": False,
            "static_tokens": 3,
            "dynamic_tokens": 5,
            "token_layout": None,
        },
        "render": {
            "renderer": "auto",
            "render_size": 4,
            "auto_dense_limit": 512,
        },
    }


def _token_layout() -> dict[str, object]:
    return {
        "static_core_tokens": 2,
        "dynamic_core_tokens": 3,
        "static_detail_tokens": [5, 7],
        "dynamic_detail_tokens": [11],
        "active_detail_level": 1,
    }


def test_pick_renderer_mode_from_config_uses_effective_gaussian_count() -> None:
    cfg = _base_cfg()

    mode, effective_gaussians = pick_renderer_mode_from_config(cfg)

    assert mode == "dense"
    assert effective_gaussians == 16


def test_pick_renderer_mode_from_config_honors_token_layout_detail_level() -> None:
    cfg = _base_cfg()
    cfg["model"]["token_layout"] = _token_layout()

    mode, effective_gaussians = pick_renderer_mode_from_config(cfg, active_detail_level=2)

    assert decoded_token_count_from_model_config(cfg["model"], active_detail_level=0) == 5
    assert decoded_token_count_from_model_config(cfg["model"], active_detail_level=1) == 21
    assert decoded_token_count_from_model_config(cfg["model"], active_detail_level=2) == 28
    assert token_layout_detail_levels(cfg["model"]) == 2
    assert effective_gaussians == 56
    assert mode == "tiled"


def test_decoded_token_count_rejects_invalid_detail_level() -> None:
    cfg = _base_cfg()
    cfg["model"]["token_layout"] = _token_layout()

    with pytest.raises(ValueError, match="active_detail_level"):
        decoded_token_count_from_model_config(cfg["model"], active_detail_level=3)


def test_token_summary_matches_legacy_and_layout_shapes() -> None:
    cfg = _base_cfg()

    assert token_summary_from_model_config(cfg["model"]) == "8 3DGS tokens"

    cfg["model"]["use_static_dynamic_split"] = True
    assert token_summary_from_model_config(cfg["model"]) == "3 static + 5 dynamic 3DGS tokens"

    cfg["model"]["token_layout"] = _token_layout()
    assert token_summary_from_model_config(cfg["model"]) == (
        "21 active decoded 3DGS tokens inside 8 total non-camera query tokens"
    )
