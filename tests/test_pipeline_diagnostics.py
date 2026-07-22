from __future__ import annotations

import pytest
import torch

from pipeline.diagnostics import (
    ReconstructionEvalAccumulator,
    camera_state_payload,
    camera_state_summary_metrics,
    decoded_temporal_payload,
    decoded_temporal_payload_from_sequence,
    reconstruction_eval_metrics,
    reconstruction_l1_mse_metrics,
)
from runtime_types import CameraState, GaussianSequence


def test_reconstruction_l1_mse_metrics_uses_prefix_and_full_clip_mean() -> None:
    prediction = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]])
    target = torch.zeros_like(prediction)

    metrics = reconstruction_l1_mse_metrics(prediction, target, prefix="eval")

    assert metrics == {
        "eval_l1": pytest.approx(1.5),
        "eval_mse": pytest.approx(3.5),
    }


def test_reconstruction_eval_metrics_adds_psnr_ssim_and_clamps_window() -> None:
    prediction = torch.zeros(2, 3, 4, 4)
    target = torch.zeros_like(prediction)
    cfg = {
        "losses": {
            "ssim_window_size": 12,
            "ssim_c1": 0.0001,
            "ssim_c2": 0.0009,
        }
    }

    metrics = reconstruction_eval_metrics(prediction, target, cfg, prefix="heldout_eval")

    assert metrics["heldout_eval_l1"] == pytest.approx(0.0)
    assert metrics["heldout_eval_mse"] == pytest.approx(0.0)
    assert metrics["heldout_eval_psnr"] == pytest.approx(120.0)
    assert metrics["heldout_eval_ssim"] == pytest.approx(1.0)


def test_streamed_reconstruction_metrics_match_full_clip() -> None:
    generator = torch.Generator().manual_seed(17)
    prediction = torch.rand(5, 3, 9, 7, generator=generator)
    target = torch.rand(5, 3, 9, 7, generator=generator)
    cfg = {"losses": {"ssim_window_size": 11, "ssim_c1": 0.0001, "ssim_c2": 0.0009}}
    expected = reconstruction_eval_metrics(prediction, target, cfg, prefix="eval")
    accumulator = ReconstructionEvalAccumulator(cfg, "eval")
    accumulator.update(prediction[:2], target[:2])
    accumulator.update(prediction[2:], target[2:])
    assert accumulator.metrics() == pytest.approx(expected, abs=1.0e-6)


def test_decoded_temporal_payload_from_sequence_matches_buffer_contract() -> None:
    decoded = GaussianSequence(
        xyz=torch.arange(3 * 2 * 3, dtype=torch.float32).reshape(3, 2, 3),
        scales=torch.arange(3 * 2 * 3, dtype=torch.float32).reshape(3, 2, 3) * 0.1,
        quats=torch.zeros(3, 2, 4),
        opacities=torch.linspace(0.1, 0.9, 6, dtype=torch.float32).reshape(3, 2),
        rgbs=torch.arange(3 * 2 * 4, dtype=torch.float32).reshape(3, 2, 4) * 0.01,
    )
    buffers = {
        "xyz": [decoded.xyz[index].detach().cpu() for index in range(decoded.frame_count)],
        "scales": [decoded.scales[index].detach().cpu() for index in range(decoded.frame_count)],
        "opacities": [decoded.opacities[index].detach().cpu() for index in range(decoded.frame_count)],
        "rgbs": [decoded.rgbs[index].detach().cpu() for index in range(decoded.frame_count)],
    }

    assert decoded_temporal_payload_from_sequence(decoded) == decoded_temporal_payload(buffers)


def test_camera_state_payload_preserves_train_and_eval_key_shapes() -> None:
    state = CameraState(
        fov_degrees=torch.tensor(60.0),
        radius=torch.tensor(3.0),
        global_residuals=torch.zeros(1),
        rotation_delta=torch.tensor([[0.0, 0.0, 0.0], [0.0, torch.pi / 6.0, 0.0]]),
        translation_delta=torch.tensor([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]]),
    )

    summary = camera_state_summary_metrics(state)
    train_payload = camera_state_payload(state)
    eval_payload = camera_state_payload(state, key_prefix="Camera/Eval")

    assert summary["fov_degrees"] == pytest.approx(60.0)
    assert summary["radius"] == pytest.approx(3.0)
    assert summary["rotation_delta_mean_degrees"] == pytest.approx(15.0)
    assert summary["translation_delta_mean"] == pytest.approx(2.5)
    assert train_payload == {
        "Camera/FOVDegrees": pytest.approx(60.0),
        "Camera/Radius": pytest.approx(3.0),
        "Camera/RotationDeltaMeanDegrees": pytest.approx(15.0),
        "Camera/TranslationDeltaMean": pytest.approx(2.5),
    }
    assert eval_payload == {
        "Camera/EvalFOVDegrees": pytest.approx(60.0),
        "Camera/EvalRadius": pytest.approx(3.0),
        "Camera/EvalRotationDeltaMeanDegrees": pytest.approx(15.0),
        "Camera/EvalTranslationDeltaMean": pytest.approx(2.5),
    }
