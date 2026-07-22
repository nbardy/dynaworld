from __future__ import annotations

from types import SimpleNamespace

import torch

import powerfoam_training_data as training_data
from camera import CameraSpec
from powerfoam_eval_artifacts import _stream_aux_metrics


def _camera() -> CameraSpec:
    return CameraSpec(
        fx=4.0,
        fy=4.0,
        cx=2.0,
        cy=2.0,
        camera_to_world=torch.eye(4),
    )


def test_paper_multicam_data_keeps_rays_lazy(monkeypatch) -> None:
    frames = torch.zeros((2, 3, 3, 4, 4), dtype=torch.float32)
    heldout = torch.zeros((1, 3, 3, 4, 4), dtype=torch.float32)
    bundle = SimpleNamespace(
        train_frames=frames,
        heldout_frames=heldout,
        train_K=torch.eye(3).repeat(2, 1, 1),
        heldout_K=torch.eye(3).repeat(1, 1, 1),
        train_w2c=torch.eye(4).repeat(2, 3, 1, 1),
        heldout_w2c=torch.eye(4).repeat(1, 3, 1, 1),
        train_lens_models=None,
        heldout_lens_models=None,
        train_distortions=None,
        heldout_distortions=None,
        train_view_count=2,
        heldout_view_count=1,
        frame_count=3,
        condition_sequence=SimpleNamespace(frames=frames[0], video_fps=30.0),
        metadata={"sample_id": "fixture"},
        train_camera_names=["a", "b"],
        heldout_camera_names=["c"],
        pose_source="fixture",
        anchor_c2w=None,
    )
    train_cameras = ((_camera(),) * 3,) * 2
    heldout_cameras = ((_camera(),) * 3,)
    load_kwargs = {}

    def load_bundle(**kwargs):
        load_kwargs.update(kwargs)
        return bundle

    monkeypatch.setattr(training_data, "load_multicam_video_bundle", load_bundle)
    monkeypatch.setattr(training_data, "cameras_from_K_w2c", lambda *_args, **_kwargs: train_cameras)
    monkeypatch.setattr(
        training_data, "heldout_cameras_from_K_w2c", lambda *_args, **_kwargs: heldout_cameras
    )

    def forbidden_full_grid(*_args, **_kwargs):
        raise AssertionError("paper data must not materialize the full view-time ray grid")

    monkeypatch.setattr(training_data, "powerfoam_rays_from_camera_grid", forbidden_full_grid)
    data = training_data.load_powerfoam_training_data(
        {
            "render": {"render_size": 4, "image_size": [4, 4]},
            "data": {"frame_source": "multicam_val"},
            "camera": {},
            "paper_protocol": {"enabled": True},
        },
        torch.device("cpu"),
    )

    assert data["targets"].shape == (6, 3, 4, 4)
    assert load_kwargs["frame_device"] == torch.device("cpu")
    assert data["sample_rays"] is None
    assert data["sample_ray_provider"].sample_count == 6
    assert data["heldout_rays"] is None
    assert data["heldout_ray_provider"].sample_count == 3


def test_streamed_aux_metrics_visit_every_sample_in_bounded_ray_chunks() -> None:
    class Provider:
        def __init__(self) -> None:
            self.calls: list[list[int]] = []

        def select(self, indices: torch.Tensor) -> torch.Tensor:
            self.calls.append(indices.tolist())
            return torch.zeros((indices.numel(), 1, 1, 6), dtype=torch.float32)

    class Model:
        def __init__(self) -> None:
            self.contrib_ema = torch.zeros((3, 2), dtype=torch.float32)
            self.point_error_ema = torch.zeros((3, 2), dtype=torch.float32)

        def aux_metrics(self, frame_indices, targets, rays):
            assert frame_indices.numel() <= 2
            assert rays.shape[0] == frame_indices.numel()
            for frame in frame_indices.tolist():
                self.contrib_ema[frame] += 1.0
                self.point_error_ema[frame] += 2.0
            mean_frame = float(frame_indices.float().mean())
            return {
                "aux_mean_contrib": mean_frame,
                "aux_max_contrib": float(frame_indices.max()),
                "aux_mean_point_error": 2.0 * mean_frame,
                "aux_max_point_error": 2.0 * float(frame_indices.max()),
                "aux_mean_contrib_ema": 0.0,
                "aux_mean_point_error_ema": 0.0,
                "aux_visible_fraction": 0.5,
                "aux_visible_cell_frame_events": float(frame_indices.numel()),
                "aux_possible_cell_frame_events": float(2 * frame_indices.numel()),
                "aux_mean_visible_cells_per_frame": 1.0,
                "aux_mean_normal_distance": 0.25,
                "aux_mean_normal_norm": 1.0,
                "aux_median_depth_valid_fraction": 0.5,
                "aux_mean_median_depth": mean_frame + 3.0,
            }

    provider = Provider()
    model = Model()
    frame_indices = torch.tensor([0, 1, 2, 0, 1], dtype=torch.long)
    metrics = _stream_aux_metrics(
        model,
        torch.zeros((5, 3, 1, 1)),
        frame_indices,
        None,
        provider,
        {"train": {"frames_per_step": 2}},
    )

    assert provider.calls == [[0, 1], [2, 3], [4]]
    assert metrics["aux_visible_cell_frame_events"] == 5.0
    assert metrics["aux_possible_cell_frame_events"] == 10.0
    assert metrics["aux_visible_fraction"] == 0.5
    assert metrics["aux_mean_contrib"] == 0.8
    assert metrics["aux_max_contrib"] == 2.0
    assert abs(metrics["aux_mean_contrib_ema"] - 1.8) < 1.0e-6
