from __future__ import annotations

from typing import Any

import powerfoam_eval_artifacts as eval_artifacts
import torch
from paper_training_protocol import normalize_image_size, resize_video_frames
from torch import nn


class RecordingTargetProvider:
    height = 1
    width = 1

    def __init__(self, sample_count: int, *, values: torch.Tensor | None = None) -> None:
        self.sample_count = int(sample_count)
        self.values = values
        self.calls: list[tuple[list[int], str, int | None, int | None]] = []

    def select(
        self,
        indices: torch.Tensor,
        *,
        device: torch.device,
        height: int | None = None,
        width: int | None = None,
    ) -> torch.Tensor:
        self.calls.append((indices.tolist(), str(device), height, width))
        target_height = self.height if height is None else int(height)
        target_width = self.width if width is None else int(width)
        values = (
            indices.to(dtype=torch.float32).div(10.0)
            if self.values is None
            else self.values.index_select(0, indices.to(dtype=torch.long))
        ).reshape(-1, 1, 1, 1)
        return values.expand(-1, 3, target_height, target_width).to(device=device)


class RecordingRayProvider:
    def __init__(self) -> None:
        self.calls: list[list[int]] = []

    def select(self, indices: torch.Tensor) -> torch.Tensor:
        self.calls.append(indices.tolist())
        rays = torch.zeros((indices.numel(), 1, 1, 6), dtype=torch.float32)
        rays[:, 0, 0, 0] = indices.to(dtype=torch.float32).div(10.0)
        return rays


class ProviderEvalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.contrib_ema = torch.zeros((3, 1), dtype=torch.float32)
        self.point_error_ema = torch.zeros((3, 1), dtype=torch.float32)

    def forward(self, frame_indices: torch.Tensor, *, rays: torch.Tensor | None = None):
        assert rays is not None
        values = rays[:, 0, 0, 0].reshape(-1, 1, 1, 1)
        rendered = values.expand(-1, 3, 1, 1) + self.anchor * 0.0
        return rendered, torch.ones((frame_indices.numel(), 1, 1), dtype=rendered.dtype)

    def aux_metrics(
        self,
        frame_indices: torch.Tensor,
        targets: torch.Tensor,
        *,
        rays: torch.Tensor | None,
    ) -> dict[str, float]:
        assert rays is not None
        expected = rays[:, 0, 0, 0].reshape(-1, 1, 1, 1).expand_as(targets)
        assert torch.equal(targets, expected)
        return {
            "aux_mean_contrib": float(frame_indices.float().mean()),
            "aux_max_contrib": float(frame_indices.max()),
            "aux_mean_point_error": 0.0,
            "aux_max_point_error": 0.0,
            "aux_mean_contrib_ema": 0.0,
            "aux_mean_point_error_ema": 0.0,
            "aux_visible_fraction": 1.0,
            "aux_visible_cell_frame_events": float(frame_indices.numel()),
            "aux_possible_cell_frame_events": float(frame_indices.numel()),
            "aux_mean_visible_cells_per_frame": 1.0,
            "aux_mean_normal_distance": 0.0,
            "aux_mean_normal_norm": 1.0,
            "aux_median_depth_valid_fraction": 1.0,
            "aux_mean_median_depth": 1.0,
        }

    def parameter_drift_metrics(self) -> dict[str, float]:
        return {}


def test_training_target_selection_uses_provider_before_resize_and_transfer() -> None:
    from powerfoam_metal_trainer import (
        powerfoam_target_dataset_shape,
        select_powerfoam_training_targets,
    )

    provider = RecordingTargetProvider(sample_count=6)
    sample_count, loaded_size = powerfoam_target_dataset_shape(None, provider)
    selected = select_powerfoam_training_targets(
        None,
        provider,
        torch.tensor([5, 1, 3]),
        image_size=normalize_image_size((2, 3)),
        loaded_image_size=normalize_image_size((4, 5)),
        device=torch.device("cpu"),
    )

    assert sample_count == 6
    assert loaded_size == normalize_image_size((1, 1))
    assert provider.calls == [([5, 1, 3], "cpu", 2, 3)]
    assert selected.shape == (3, 3, 2, 3)
    assert torch.equal(selected[:, 0, 0, 0], torch.tensor([0.5, 0.1, 0.3]))


def test_training_target_selection_preserves_legacy_eager_path() -> None:
    from powerfoam_metal_trainer import select_powerfoam_training_targets

    targets = torch.arange(4 * 3 * 4 * 6, dtype=torch.float32).reshape(4, 3, 4, 6)
    selected = select_powerfoam_training_targets(
        targets,
        None,
        torch.tensor([3, 0]),
        image_size=normalize_image_size((2, 3)),
        loaded_image_size=normalize_image_size((4, 6)),
        device=torch.device("cpu"),
    )

    expected = resize_video_frames(
        targets[[3, 0]],
        normalize_image_size((2, 3)),
    )
    assert torch.equal(selected, expected)


def test_paper_memory_policy_reports_provider_and_initialization_residency() -> None:
    from powerfoam_metal_trainer import powerfoam_memory_policy

    target_residency = {"train": {"source_kind": "path_backed_images", "resident_bytes": 0}}
    init_residency = {
        "enabled": True,
        "resident_bytes": 192,
        "shares_train_target_storage": False,
    }
    policy = powerfoam_memory_policy(
        {
            "sample_target_provider": object(),
            "sample_ray_provider": object(),
            "target_residency": target_residency,
            "init_frames_resident_bytes": 192,
            "init_frames_residency": init_residency,
        },
        {
            "train": {"frames_per_step": 3},
            "logging": {"eval_media_max_frames": 5},
        },
    )

    assert policy["targets"] == "selected_target_provider"
    assert policy["rays"] == "sampled_on_demand"
    assert policy["evaluation_chunk_frames"] == 3
    assert policy["target_residency"] is target_residency
    assert policy["init_frames_residency"] is init_residency


def test_streamed_provider_metrics_use_global_sample_normalization() -> None:
    target_provider = RecordingTargetProvider(
        sample_count=5,
        values=torch.tensor([0.0, 0.1, 0.2, 0.3, 1.0]),
    )
    result = eval_artifacts._stream_eval_split(
        ProviderEvalModel(),
        None,
        torch.tensor([0, 1, 2, 0, 1]),
        None,
        RecordingRayProvider(),
        {
            "render": {"background_mode": "fixed", "background": [0.0, 0.0, 0.0]},
            "losses": {"ssim_window_size": 1, "ssim_c1": 0.0001, "ssim_c2": 0.0009},
            "train": {"frames_per_step": 2},
            "logging": {"eval_media_max_frames": 2},
        },
        target_provider=target_provider,
        prefix="eval",
        include_lpips=False,
    )

    assert abs(result.metrics["eval_l1"] - 0.12) < 1.0e-7
    assert abs(result.metrics["eval_mse"] - 0.072) < 1.0e-7
    assert [call[0] for call in target_provider.calls] == [[0, 1], [2, 3], [4]]


def test_artifact_call_graph_streams_train_heldout_aux_and_media_targets(
    tmp_path,
    monkeypatch,
) -> None:
    train_targets = RecordingTargetProvider(sample_count=5)
    heldout_targets = RecordingTargetProvider(sample_count=3)
    train_rays = RecordingRayProvider()
    heldout_rays = RecordingRayProvider()
    saved: dict[str, Any] = {}

    def capture_media(
        _output_dir,
        _step,
        renders,
        targets,
        _alphas,
        **kwargs,
    ) -> None:
        saved["train_shapes"] = (tuple(renders.shape), tuple(targets.shape))
        saved["heldout_shapes"] = (
            tuple(kwargs["heldout_renders"].shape),
            tuple(kwargs["heldout_targets"].shape),
        )

    monkeypatch.setattr(eval_artifacts, "save_rgb_alpha_eval_media", capture_media)
    monkeypatch.setattr(eval_artifacts, "log_wandb_run_payload_lazy", lambda *_args, **_kwargs: None)
    cfg = {
        "paper_protocol": {"enabled": True},
        "render": {
            "background_mode": "fixed",
            "background": [0.0, 0.0, 0.0],
            "eval_color_calibration": "none",
        },
        "losses": {
            "ssim_window_size": 1,
            "ssim_c1": 0.0001,
            "ssim_c2": 0.0009,
        },
        "train": {"frames_per_step": 2, "steps": 2},
        "logging": {
            "eval_media_max_frames": 3,
            "video_log_every": 100,
            "always_log_last_step": False,
        },
        "video_fps": 30.0,
    }
    metrics = eval_artifacts.log_powerfoam_artifacts(
        ProviderEvalModel(),
        None,
        cfg,
        0,
        tmp_path,
        None,
        frame_indices=torch.tensor([0, 1, 2, 0, 1]),
        ray_provider=train_rays,
        target_provider=train_targets,
        heldout_frame_indices=torch.tensor([0, 1, 2]),
        heldout_ray_provider=heldout_rays,
        heldout_target_provider=heldout_targets,
    )

    assert metrics["eval_l1"] == 0.0
    assert metrics["eval_mse"] == 0.0
    assert metrics["heldout_eval_l1"] == 0.0
    assert metrics["heldout_eval_mse"] == 0.0
    assert [call[0] for call in train_targets.calls] == [
        [0, 1],
        [2, 3],
        [4],
        [0, 1],
        [2, 3],
        [4],
    ]
    assert [call[0] for call in heldout_targets.calls] == [[0, 1], [2]]
    assert train_rays.calls == [[0, 1], [2, 3], [4], [0, 1], [2, 3], [4]]
    assert heldout_rays.calls == [[0, 1], [2]]
    assert saved["train_shapes"] == ((3, 3, 1, 1), (3, 3, 1, 1))
    assert saved["heldout_shapes"] == ((3, 3, 1, 1), (3, 3, 1, 1))
