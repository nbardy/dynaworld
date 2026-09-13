from __future__ import annotations

import math

import pytest
import torch

import paper_multicam_targets
from multicam_video_data import MulticamVideoFrameSource
from paper_multicam_targets import load_grouped_frozen_target_frames


@pytest.mark.parametrize("chunk_frames,cache_frames", [(1, 8), (3, 8), (8, 8), (16, 8), (1, 1)])
def test_grouped_targets_preserve_selected_order_and_bounded_decode(
    tmp_path, monkeypatch, chunk_frames, cache_frames,
) -> None:
    selected = (1, 4, 4, 7, 9, 11, 12, 15, 20, 21, 23, 28, 32, 34, 39, 41, 44, 49, 53)
    video_path = tmp_path / "camera.mp4"
    video_path.touch()

    def decode(*, sample_indices, **kwargs):
        # A real provider with deterministic external I/O: reject unrelated
        # frames and excessive batches, then expose each frame's exact identity.
        assert set(sample_indices) <= set(selected)
        assert len(sample_indices) <= max(chunk_frames, cache_frames)
        return torch.tensor(sample_indices, dtype=torch.float32).view(-1, 1, 1, 1).expand(-1, 3, 2, 2).clone()

    monkeypatch.setattr(paper_multicam_targets, "load_multicam_val_selected_camera_frames", decode)
    provider = paper_multicam_targets.PaperMulticamTargetProvider(
        (MulticamVideoFrameSource("camera", video_path, 0.0, 30.0, 64, tuple(range(64)), 2, 2),),
        cache_capacity_frames=cache_frames,
    )
    for _ in range(2):
        outputs = [
            load_grouped_frozen_target_frames(
                provider, selected, start=start,
                stop=min(start + chunk_frames, len(selected)), chunk_frames=chunk_frames,
            )
            for start in range(0, len(selected), chunk_frames)
        ]
        assert all(value.device.type == "cpu" and value.shape[0] <= chunk_frames for value in outputs)
        assert torch.equal(torch.cat(outputs)[:, 0, 0, 0], torch.tensor(selected, dtype=torch.float32))
    accounting = provider.accounting()
    assert accounting["peak_cache_resident_frames"] <= cache_frames
    assert accounting["peak_request_frame_count"] <= max(chunk_frames, cache_frames)
    grouped_frames = max(chunk_frames, (cache_frames // chunk_frames) * chunk_frames)
    assert accounting["decode_call_count"] <= 2 * math.ceil(len(selected) / grouped_frames)
