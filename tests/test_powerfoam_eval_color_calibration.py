from __future__ import annotations

import torch

from powerfoam_eval_color import (
    apply_eval_color_calibration,
    fit_eval_color_calibration,
    serialize_eval_color_calibration,
)


def test_train_fit_channel_affine_recovers_channel_exposure() -> None:
    rendered = torch.linspace(0.05, 0.65, steps=2 * 3 * 3 * 3, dtype=torch.float32).reshape(2, 3, 3, 3)
    scale = torch.tensor([0.8, 1.1, 0.9], dtype=torch.float32).view(1, 3, 1, 1)
    bias = torch.tensor([0.03, -0.02, 0.04], dtype=torch.float32).view(1, 3, 1, 1)
    target = (rendered * scale + bias).clamp(0.0, 1.0)

    calibration = fit_eval_color_calibration(
        {"eval_color_calibration": "train_fit_channel_affine"},
        rendered,
        target,
    )
    corrected = apply_eval_color_calibration(rendered, calibration)

    assert calibration is not None
    assert calibration["mode"] == "train_fit_channel_affine"
    torch.testing.assert_close(corrected, target, atol=2.0e-6, rtol=1.0e-6)


def test_train_fit_rgb_matrix_affine_recovers_cross_channel_exposure() -> None:
    rendered = torch.linspace(0.04, 0.62, steps=2 * 3 * 3 * 3, dtype=torch.float32).reshape(2, 3, 3, 3)
    flat = rendered.permute(0, 2, 3, 1).reshape(-1, 3)
    design = torch.cat([flat, torch.ones(flat.shape[0], 1)], dim=1)
    transform = torch.tensor(
        [
            [0.80, 0.05, 0.00],
            [0.02, 0.90, 0.03],
            [0.01, 0.04, 0.85],
            [0.03, -0.01, 0.02],
        ],
        dtype=torch.float32,
    )
    target = (design @ transform).reshape(2, 3, 3, 3).permute(0, 3, 1, 2).clamp(0.0, 1.0)

    calibration = fit_eval_color_calibration(
        {"eval_color_calibration": "train_fit_rgb_matrix_affine"},
        rendered,
        target,
    )
    corrected = apply_eval_color_calibration(rendered, calibration)

    assert calibration is not None
    assert calibration["mode"] == "train_fit_rgb_matrix_affine"
    torch.testing.assert_close(corrected, target, atol=2.0e-6, rtol=1.0e-6)


def test_eval_color_calibration_serializes_train_fit_provenance() -> None:
    calibration = {
        "mode": "train_fit_rgb_matrix_affine",
        "transform": torch.eye(4, 3, dtype=torch.float32),
    }

    payload = serialize_eval_color_calibration(
        calibration,
        step=7,
        train_frame_indices=torch.tensor([0, 1, 1, 2], dtype=torch.long),
        heldout_frame_indices=torch.tensor([3, 3], dtype=torch.long),
    )

    assert payload is not None
    assert payload["mode"] == "train_fit_rgb_matrix_affine"
    assert payload["step"] == 7
    assert payload["fit_scope"] == "train_render_to_train_target"
    assert payload["heldout_blind"] is True
    assert payload["train_frame_indices"] == {"count": 4, "unique": [0, 1, 2]}
    assert payload["heldout_frame_indices"] == {"count": 2, "unique": [3]}
