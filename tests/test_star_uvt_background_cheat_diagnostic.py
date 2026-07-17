from __future__ import annotations

from research_experiments.star_uvt_feature_tubes.background_cheat_diagnostic import run_diagnostic


def _row(rows: list[dict], mode: str, alpha: float) -> dict:
    matches = [row for row in rows if row["mode"] == mode and abs(float(row["alpha"]) - alpha) < 1.0e-9]
    assert len(matches) == 1
    return matches[0]


def test_rgb_background_after_colorizer_gates_colorizer_gradients_by_alpha() -> None:
    rows = run_diagnostic((0.0, 0.02, 1.0))["rows"]

    rgb_alpha0 = _row(rows, "rgb_background_after_colorizer", 0.0)
    rgb_alpha002 = _row(rows, "rgb_background_after_colorizer", 0.02)
    rgb_alpha1 = _row(rows, "rgb_background_after_colorizer", 1.0)

    assert rgb_alpha0["colorizer_grad_l2"] == 0.0
    assert rgb_alpha0["feature_grad_l2"] == 0.0
    assert rgb_alpha002["colorizer_grad_l2"] > 0.0
    assert rgb_alpha1["colorizer_grad_l2"] > rgb_alpha002["colorizer_grad_l2"]


def test_feature_background_before_colorizer_trains_colorizer_at_empty_alpha() -> None:
    rows = run_diagnostic((0.0,))["rows"]

    rgb_alpha0 = _row(rows, "rgb_background_after_colorizer", 0.0)
    feature_bg_alpha0 = _row(rows, "feature_background_before_colorizer", 0.0)

    assert rgb_alpha0["colorizer_grad_l2"] == 0.0
    assert feature_bg_alpha0["colorizer_grad_l2"] > 0.0
    assert feature_bg_alpha0["feature_grad_l2"] == 0.0
