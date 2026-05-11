from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "benchmarks" / "temporal_raster_overlap_profile.py"
SPEC = importlib.util.spec_from_file_location("temporal_raster_overlap_profile", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
overlap_profile = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = overlap_profile
SPEC.loader.exec_module(overlap_profile)


def test_synthetic_overlap_profile_reports_temporal_churn() -> None:
    static_state = overlap_profile.make_synthetic_projected_state(
        frames=4,
        gaussians=64,
        height=64,
        width=64,
        feature_dim=8,
        radius_px=2.0,
        radius_jitter_px=0.0,
        motion_px=0.0,
        noise_px=0.0,
        opacity=0.9,
        seed=3,
    )
    moving_state = overlap_profile.make_synthetic_projected_state(
        frames=4,
        gaussians=64,
        height=64,
        width=64,
        feature_dim=8,
        radius_px=2.0,
        radius_jitter_px=0.0,
        motion_px=18.0,
        noise_px=0.0,
        opacity=0.9,
        seed=3,
    )

    static_metrics = overlap_profile.profile_projected_state(
        static_state,
        height=64,
        width=64,
        tile_size=8,
        alpha_threshold=1.0 / 128.0,
    )
    moving_metrics = overlap_profile.profile_projected_state(
        moving_state,
        height=64,
        width=64,
        tile_size=8,
        alpha_threshold=1.0 / 128.0,
    )

    assert static_metrics["gaussian_tile_pair_adjacent_jaccard_mean"] == 1.0
    assert moving_metrics["gaussian_tile_pair_adjacent_jaccard_mean"] < 1.0
    assert moving_metrics["input_mode"] == "synthetic_projected_approximation"
