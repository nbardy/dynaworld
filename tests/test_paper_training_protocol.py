from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from multicam_val_data import _load_frame
from paper_training_protocol import (
    PaperCostTracker,
    SpacetimeEpochSampler,
    normalize_paper_stages,
    paper_stage_for_step,
    resize_ray_grids,
    resize_video_frames,
    scale_intrinsics,
)
from paper_training_types import ImageSize


def test_spacetime_epoch_sampler_covers_each_pair_once_and_is_reproducible() -> None:
    kwargs = {
        "view_count": 3,
        "frame_indices": range(5),
        "batch_size": 4,
        "same_time_count": 3,
        "local_time_count": 1,
        "local_time_radius": 2,
        "seed": 17,
    }
    first = SpacetimeEpochSampler(**kwargs)
    second = SpacetimeEpochSampler(**kwargs)

    batches = []
    while not batches or not batches[-1].completes_epoch:
        left = first.next_batch()
        right = second.next_batch()
        assert left == right
        batches.append(left)

    samples = [sample for batch in batches for sample in batch.samples]
    assert len(samples) == 15
    assert len(set(samples)) == 15
    assert set(samples) == {
        type(samples[0])(view_index=view, frame_index=frame)
        for view in range(3)
        for frame in range(5)
    }
    assert all(len(batch.samples) == 4 for batch in batches[:-1])
    assert len(batches[-1].samples) == 3
    assert len({sample.frame_index for sample in batches[0].samples[:3]}) == 1


def test_paper_stage_schedule_requires_contiguous_monotonic_progression() -> None:
    stages = normalize_paper_stages(
        [
            {
                "label": "coarse",
                "until_step": 10,
                "image_size": [96, 128],
                "primitive_count": 256,
                "frames_per_step": 4,
            },
            {
                "label": "fine",
                "until_step": 20,
                "image_size": [192, 256],
                "primitive_count": 1024,
                "frames_per_step": 2,
                "lr_multiplier": 0.25,
            },
        ],
        total_steps=20,
        default_image_size=ImageSize(192, 256),
        default_primitive_count=1024,
        default_frames_per_step=1,
    )

    assert paper_stage_for_step(stages, 0).label == "coarse"
    assert paper_stage_for_step(stages, 10).label == "fine"
    assert stages[-1].as_dict()["width"] == 256

    with pytest.raises(ValueError, match="non-decreasing"):
        normalize_paper_stages(
            [
                {"until_step": 5, "image_size": [128, 128], "primitive_count": 8},
                {"until_step": 10, "image_size": [64, 64], "primitive_count": 8},
            ],
            total_steps=10,
            default_image_size=ImageSize(128, 128),
            default_primitive_count=8,
            default_frames_per_step=1,
        )


def test_aspect_preserving_resize_scales_frames_rays_and_intrinsics() -> None:
    frames = torch.arange(2 * 3 * 8 * 12, dtype=torch.float32).reshape(2, 3, 8, 12)
    resized = resize_video_frames(frames, ImageSize(4, 6))
    assert resized.shape == (2, 3, 4, 6)

    rays = torch.zeros(2, 8, 12, 6)
    rays[..., 3] = 1.0
    resized_rays = resize_ray_grids(rays, ImageSize(4, 6))
    assert resized_rays.shape == (2, 4, 6, 6)
    assert torch.allclose(resized_rays[..., 3:].norm(dim=-1), torch.ones(2, 4, 6))

    K = torch.tensor([[[12.0, 0.0, 6.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]]])
    scaled = scale_intrinsics(K, source=ImageSize(8, 12), target=ImageSize(4, 6))
    assert torch.allclose(scaled[0], torch.tensor([[6.0, 0.0, 3.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]]))


def test_multicam_frame_loader_accepts_rectangular_target(tmp_path: Path) -> None:
    path = tmp_path / "frame.png"
    Image.new("RGB", (12, 8), (10, 20, 30)).save(path)

    frame = _load_frame(path, (4, 6))

    assert frame.shape == (3, 4, 6)


def test_paper_cost_tracker_separates_target_and_rasterized_pixels() -> None:
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    loss = model(torch.ones(1, 3)).sum()
    loss.backward()
    optimizer.step()
    stage = normalize_paper_stages(
        None,
        total_steps=1,
        default_image_size=ImageSize(4, 6),
        default_primitive_count=2,
        default_frames_per_step=2,
    )[0]
    tracker = PaperCostTracker()
    tracker.record(stage=stage, target_frames=2, rasterized_frames=8)

    snapshot = tracker.snapshot(model=model, optimizer=optimizer, elapsed_s=0.5)

    assert snapshot.target_pixels == 48
    assert snapshot.rasterized_pixels == 192
    assert snapshot.parameter_count == 8
    assert snapshot.optimizer_state_bytes > 0
