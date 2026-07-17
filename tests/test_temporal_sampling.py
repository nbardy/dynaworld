from __future__ import annotations

import pytest
import torch

import sequence_data as sequence_data_module
from clip_sampling import sample_clip_batch
from camera import CameraSpec
from runtime_types import SequenceData
from sequence_data import ManifestSequenceSampler
from temporal_sampling import (
    TEMPORAL_DILATION_OFFSETS,
    normalize_frame_sampling_config,
    select_frame_indices,
    temporal_dilation_indices_for_center,
    validate_frame_sampling_config,
)
from multicam_precomputed_trainer import MulticamPrecomputedFeatureImplicitTrainer
from token_gs_trainer import Trainer


def test_normalize_frame_sampling_accepts_string_and_typo_alias() -> None:
    assert normalize_frame_sampling_config("random")["mode"] == "random"
    assert normalize_frame_sampling_config("contigous")["mode"] == "contiguous"
    assert normalize_frame_sampling_config(None)["mode"] == "contiguous"


def test_temporal_dilation_wraps_around_sequence_bounds() -> None:
    indices = temporal_dilation_indices_for_center(10, center=0, offsets=(-16, -8, -1, 0, 1, 8, 16))

    assert torch.equal(indices.cpu(), torch.tensor([4, 2, 9, 0, 1, 8, 6]))


def test_temporal_dilation_default_offsets_match_power_of_two_span() -> None:
    assert TEMPORAL_DILATION_OFFSETS == (-16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16)
    indices = select_frame_indices(64, len(TEMPORAL_DILATION_OFFSETS), {"mode": "temporal-dilation"})

    assert indices.shape == (len(TEMPORAL_DILATION_OFFSETS),)
    assert int(indices.min()) >= 0
    assert int(indices.max()) < 64


def test_temporal_dilation_requires_train_frame_count_to_match_offsets() -> None:
    config = normalize_frame_sampling_config({"mode": "temporal-dilation"})

    with pytest.raises(ValueError, match="model.train_frame_count=16"):
        validate_frame_sampling_config(config, sample_count=16)


def test_random_frame_sampling_returns_sorted_unique_subset() -> None:
    torch.manual_seed(3)
    indices = select_frame_indices(20, 6, {"mode": "random"})

    assert indices.shape == (6,)
    assert torch.equal(indices, torch.sort(indices).values)
    assert torch.unique(indices).numel() == 6


def test_contiguous_frame_sampling_preserves_legacy_full_window_behavior() -> None:
    indices = select_frame_indices(4, 16, {"mode": "contiguous"})

    assert torch.equal(indices.cpu(), torch.arange(4))


def test_sample_clip_batch_returns_typed_clip_with_camera_slice() -> None:
    camera = CameraSpec(
        fx=1.0,
        fy=1.0,
        cx=0.5,
        cy=0.5,
        camera_to_world=torch.eye(4),
    )
    sequence = _sequence(frame_count=5, cameras=tuple(camera for _ in range(5)))

    clip = sample_clip_batch(
        sequence,
        train_frame_count=3,
        frame_sampling={"mode": "contiguous"},
        device=torch.device("cpu"),
    )

    assert clip.frame_count == 3
    assert clip.as_video_batch().shape == (1, 3, 3, 1, 1)
    assert clip.as_time_batch().shape == (1, 3)
    assert clip.cameras == sequence.cameras[:3]


def test_base_trainer_sample_clip_uses_frame_sampling_config() -> None:
    trainer = object.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.model_cfg = {"train_frame_count": len(TEMPORAL_DILATION_OFFSETS)}
    trainer.train_cfg = {"frame_sampling": {"mode": "temporal-dilation"}}
    trainer.train_sequences = [_sequence(frame_count=40)]
    trainer.train_sequence_sampler = None

    _sequence_data, typed_clip = Trainer.sample_train_clip_batch(trainer)
    _legacy_sequence_data, clip_frames, clip_times = Trainer.sample_clip(trainer)

    assert typed_clip.frame_count == len(TEMPORAL_DILATION_OFFSETS)
    assert clip_frames.shape[1] == len(TEMPORAL_DILATION_OFFSETS)
    assert clip_times.shape[1] == len(TEMPORAL_DILATION_OFFSETS)


def test_lazy_manifest_prefetch_preserves_cycle_order(monkeypatch) -> None:
    def fake_load_manifest_sequence(entry, *, data_cfg, model_cfg, device):
        del data_cfg, model_cfg
        value = float(entry["value"])
        return SequenceData(
            frames=torch.full((4, 3, 1, 1), value, dtype=torch.float32, device=device),
            frame_times=torch.linspace(0.0, 1.0, 4, device=device).view(4, 1),
            video_fps=30.0,
            frame_source="test",
            selected_frame_count=4,
            all_frame_count=4,
        )

    monkeypatch.setattr(sequence_data_module, "load_manifest_sequence", fake_load_manifest_sequence)
    sampler = ManifestSequenceSampler(
        entries=[{"value": 0}, {"value": 1}, {"value": 2}],
        sequences=[_sequence(frame_count=4)],
        data_cfg={},
        model_cfg={},
        device=torch.device("cpu"),
        load_mode="lazy",
        sample_mode="cycle",
        prefetch_depth=2,
    )

    sampler.start_prefetch()
    try:
        first = sampler.sample()
        second = sampler.sample()
    finally:
        sampler.close_prefetch()

    assert float(first.frames[0, 0, 0, 0]) == 0.0
    assert float(second.frames[0, 0, 0, 0]) == 1.0


def test_multicam_trainer_sample_clip_uses_frame_sampling_config() -> None:
    trainer = object.__new__(MulticamPrecomputedFeatureImplicitTrainer)
    trainer.device = torch.device("cpu")
    trainer.model_cfg = {"train_frame_count": 6}
    trainer.train_cfg = {"frame_sampling": {"mode": "random"}}
    trainer.sequence_data = _sequence(frame_count=20)
    trainer.sample_views = lambda: [0, 1]

    _sequence_data, typed_clip, typed_views = MulticamPrecomputedFeatureImplicitTrainer.sample_multicam_clip_batch(trainer)
    _legacy_sequence_data, clip_indices, clip_frames, _clip_times, views = (
        MulticamPrecomputedFeatureImplicitTrainer.sample_multicam_clip(trainer)
    )

    assert typed_clip.frame_count == 6
    assert typed_views == [0, 1]
    assert clip_indices.shape == (6,)
    assert torch.unique(clip_indices).numel() == 6
    assert clip_frames.shape[1] == 6
    assert views == [0, 1]


def _sequence(frame_count: int, cameras=None) -> SequenceData:
    return SequenceData(
        frames=torch.arange(frame_count * 3, dtype=torch.float32).reshape(frame_count, 3, 1, 1),
        frame_times=torch.linspace(0.0, 1.0, frame_count).view(frame_count, 1),
        video_fps=30.0,
        frame_source="test",
        cameras=cameras,
        selected_frame_count=frame_count,
        all_frame_count=frame_count,
    )
