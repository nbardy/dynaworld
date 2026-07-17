from __future__ import annotations

import importlib.util
import sys
from contextlib import nullcontext
from pathlib import Path

import torch

from mixed_data_scheduler import MixedStepBatch, sample_novel_view_batch, sample_same_view_batch
from mixed_same_heldout_trainer import (
    MixedBackwardResult,
    MixedSameHeldoutPrecomputedFeatureTrainer,
)
from multicam_video_data import MulticamVideoBundle
from runtime_types import SequenceData


def test_mixed_same_heldout_arch_dispatches_to_trainer(tmp_path) -> None:
    config_path = tmp_path / "mixed.jsonc"
    config_path.write_text('{"arch": "mixed_same_heldout_precomputed_feature_implicit_camera"}')

    entry = _trainer_entry_for_config(config_path)

    assert entry.module == "mixed_same_heldout_trainer"


def _trainer_entry_for_config(config_path: Path):
    spec = importlib.util.spec_from_file_location("dynaworld_train_dispatch", Path("src/train/train.py"))
    if spec is None or spec.loader is None:
        raise AssertionError("Could not load src/train/train.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.trainer_entry_for_config(config_path)


def test_mixed_trainer_samples_scheduler_batches_with_explicit_loss_names() -> None:
    trainer = object.__new__(MixedSameHeldoutPrecomputedFeatureTrainer)
    trainer.device = torch.device("cpu")
    trainer.model_cfg = {"train_frame_count": 3}
    trainer.train_cfg = {
        "frame_sampling": {"mode": "contiguous"},
        "mixed_schedule_mode": "both",
        "same_view_weight": 0.75,
        "novel_view_weight": 0.25,
        "train_views_per_step": 1,
        "heldout_views_per_step": 1,
    }
    trainer.same_view_sampler = _FakeSequenceSampler([_sequence(5, value=11.0)])
    trainer.multicam_bundle = _bundle(train_views=2, heldout_views=1, frame_count=5)

    batch = MixedSameHeldoutPrecomputedFeatureTrainer.sample_mixed_step_batch(trainer, 1)

    assert batch.loss_names() == ("same_view_recon", "heldout_view_recon")
    assert batch.same_view is not None
    assert batch.same_view.weight == 0.75
    assert batch.same_view.clip.frame_count == 3
    assert batch.novel_view is not None
    assert batch.novel_view.weight == 0.25
    assert len(batch.novel_view.train_views) == 1
    assert batch.novel_view.heldout_views == (0,)


def test_mixed_trainer_step_preserves_separate_recon_terms() -> None:
    same_sequence = _sequence(4, value=2.0)
    bundle = _bundle(train_views=1, heldout_views=1, frame_count=4)
    same_batch = sample_same_view_batch(
        same_sequence,
        train_frame_count=2,
        frame_sampling={"mode": "contiguous"},
        device=torch.device("cpu"),
        weight=0.5,
    )
    novel_batch = sample_novel_view_batch(
        bundle,
        train_frame_count=2,
        frame_sampling={"mode": "contiguous"},
        device=torch.device("cpu"),
        weight=0.25,
    )
    trainer = object.__new__(MixedSameHeldoutPrecomputedFeatureTrainer)
    trainer.device = torch.device("cpu")
    trainer.sequence_data = bundle.condition_sequence
    trainer.optimizer = _FakeOptimizer()
    trainer.profile_section = lambda _name: nullcontext()
    trainer.reset_profile_timing = lambda: None
    trainer.finish_profile_timing = lambda: None
    trainer.sample_mixed_step_batch = lambda _step: MixedStepBatch(same_view=same_batch, novel_view=novel_batch)
    trainer._backward_same_view_batch = lambda _batch, **_kwargs: MixedBackwardResult(
        loss_name=same_batch.loss_name,
        sequence=same_batch.sequence,
        recon_loss=torch.tensor(2.0),
        weighted_recon_loss=torch.tensor(1.0),
        bank_rate_loss=torch.tensor(0.1),
        camera_motion_loss=torch.tensor(0.01),
        camera_temporal_loss=torch.tensor(0.02),
        camera_global_loss=torch.tensor(0.03),
        preview_render=None,
        preview_features=None,
        clip_frames=same_batch.clip.as_video_batch(),
        bank_rate_terms={"same_view_same_alpha": torch.tensor(0.4)},
    )
    trainer._backward_novel_view_batch = lambda _batch, **_kwargs: MixedBackwardResult(
        loss_name=novel_batch.loss_name,
        sequence=novel_batch.condition_sequence,
        recon_loss=torch.tensor(3.0),
        weighted_recon_loss=torch.tensor(0.75),
        bank_rate_loss=torch.tensor(0.2),
        camera_motion_loss=torch.tensor(0.0),
        camera_temporal_loss=torch.tensor(0.0),
        camera_global_loss=torch.tensor(0.0),
        preview_render=None,
        preview_features=None,
        clip_frames=novel_batch.clip.as_video_batch(),
        bank_rate_terms={"heldout_view_heldout_alpha": torch.tensor(0.5)},
    )

    result = MixedSameHeldoutPrecomputedFeatureTrainer.step(trainer, keep_preview=False)

    assert trainer.optimizer.zero_grad_calls == 1
    assert trainer.optimizer.step_calls == 1
    assert torch.isclose(result.recon_loss, torch.tensor(1.75))
    assert torch.isclose(result.loss, torch.tensor(2.11))
    assert torch.isclose(result.aux_loss_terms["same_view_recon"], torch.tensor(2.0))
    assert torch.isclose(result.aux_loss_terms["same_view_recon_weighted"], torch.tensor(1.0))
    assert torch.isclose(result.aux_loss_terms["heldout_view_recon"], torch.tensor(3.0))
    assert torch.isclose(result.aux_loss_terms["heldout_view_recon_weighted"], torch.tensor(0.75))
    assert "same_view_same_alpha" in result.bank_rate_terms
    assert "heldout_view_heldout_alpha" in result.bank_rate_terms


class _FakeOptimizer:
    def __init__(self) -> None:
        self.zero_grad_calls = 0
        self.step_calls = 0

    def zero_grad(self, set_to_none: bool = True) -> None:
        del set_to_none
        self.zero_grad_calls += 1

    def step(self) -> None:
        self.step_calls += 1


class _FakeSequenceSampler:
    def __init__(self, sequences: list[SequenceData]) -> None:
        self.sequences = sequences
        self.sequence_count = len(sequences)

    def sample(self) -> SequenceData:
        return self.sequences[0]


def _sequence(frame_count: int, *, value: float = 0.0) -> SequenceData:
    return SequenceData(
        frames=torch.full((frame_count, 3, 1, 1), value, dtype=torch.float32),
        frame_times=torch.linspace(0.0, 1.0, frame_count).view(frame_count, 1),
        video_fps=4.0,
        frame_source="explicit_video",
        selected_frame_count=frame_count,
        all_frame_count=frame_count,
    )


def _bundle(*, train_views: int, heldout_views: int, frame_count: int) -> MulticamVideoBundle:
    train_sequences = tuple(_sequence(frame_count, value=float(view)) for view in range(train_views))
    heldout_sequences = tuple(
        _sequence(frame_count, value=float(100 + view)) for view in range(heldout_views)
    )
    return MulticamVideoBundle(
        condition_sequence=train_sequences[0],
        train_sequences=train_sequences,
        train_frames=torch.stack([sequence.frames for sequence in train_sequences], dim=0),
        train_K=torch.eye(3).repeat(train_views, 1, 1),
        train_w2c=torch.eye(4).repeat(train_views, frame_count, 1, 1),
        train_camera_names=[f"train_{view}" for view in range(train_views)],
        heldout_sequences=heldout_sequences,
        heldout_frames=torch.stack([sequence.frames for sequence in heldout_sequences], dim=0),
        heldout_K=torch.eye(3).repeat(heldout_views, 1, 1),
        heldout_w2c=torch.eye(4).repeat(heldout_views, frame_count, 1, 1),
        heldout_camera_names=[f"heldout_{view}" for view in range(heldout_views)],
    )
