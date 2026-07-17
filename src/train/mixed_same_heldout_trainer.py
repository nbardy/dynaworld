from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from config_utils import apply_defaults
from mixed_data_scheduler import (
    MixedStepBatch,
    NovelViewBatch,
    SameViewBatch,
    sample_mixed_step_batch as build_mixed_step_batch,
)
from multicam_precomputed_trainer import MulticamPrecomputedFeatureImplicitTrainer
from pipeline.losses import build_bank_rate_loss as _build_bank_rate_loss_impl
from pipeline.losses import build_camera_loss as _build_camera_loss_impl
from runtime_types import SequenceData, StepResult, build_step_result
from sequence_data import ManifestSequenceSampler


TRAIN_MIXED_DEFAULTS = {
    "mixed_schedule_mode": "alternate",
    "same_view_weight": 1.0,
    "novel_view_weight": 1.0,
    "heldout_views_per_step": 1,
}


@dataclass(frozen=True)
class MixedBackwardResult:
    loss_name: str
    sequence: SequenceData
    recon_loss: torch.Tensor
    weighted_recon_loss: torch.Tensor
    bank_rate_loss: torch.Tensor
    camera_motion_loss: torch.Tensor
    camera_temporal_loss: torch.Tensor
    camera_global_loss: torch.Tensor
    preview_render: torch.Tensor | None
    preview_features: torch.Tensor | None
    clip_frames: torch.Tensor
    bank_rate_terms: dict[str, torch.Tensor]
    aux_loss_terms: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class MixedStepAccumulator:
    weighted_recon_loss: torch.Tensor
    bank_rate_loss: torch.Tensor
    camera_motion_loss: torch.Tensor
    camera_temporal_loss: torch.Tensor
    camera_global_loss: torch.Tensor
    preview_render: torch.Tensor | None = None
    preview_features: torch.Tensor | None = None
    clip_frames: torch.Tensor | None = None
    sequence: SequenceData | None = None
    bank_rate_terms: dict[str, torch.Tensor] = field(default_factory=dict)
    aux_loss_terms: dict[str, torch.Tensor] = field(default_factory=dict)

    def add(self, result: MixedBackwardResult) -> None:
        self.weighted_recon_loss = self.weighted_recon_loss + result.weighted_recon_loss.detach()
        self.bank_rate_loss = self.bank_rate_loss + result.bank_rate_loss.detach()
        self.camera_motion_loss = self.camera_motion_loss + result.camera_motion_loss.detach()
        self.camera_temporal_loss = self.camera_temporal_loss + result.camera_temporal_loss.detach()
        self.camera_global_loss = self.camera_global_loss + result.camera_global_loss.detach()
        self.bank_rate_terms.update(result.bank_rate_terms)
        self.aux_loss_terms[result.loss_name] = result.recon_loss.detach()
        self.aux_loss_terms[f"{result.loss_name}_weighted"] = result.weighted_recon_loss.detach()
        self.aux_loss_terms.update(result.aux_loss_terms)
        if self.preview_render is None:
            self.preview_render = result.preview_render
            self.preview_features = result.preview_features
            self.clip_frames = result.clip_frames
            self.sequence = result.sequence

    @property
    def total_loss(self) -> torch.Tensor:
        return (
            self.weighted_recon_loss
            + self.camera_motion_loss
            + self.camera_temporal_loss
            + self.camera_global_loss
            + self.bank_rate_loss
        )


class MixedSameHeldoutPrecomputedFeatureTrainer(MulticamPrecomputedFeatureImplicitTrainer):
    """Smoke bridge for same-view scale batches plus heldout-camera supervision."""

    @classmethod
    def resolve_config(cls, config: dict[str, Any]) -> dict[str, Any]:
        cfg = super().resolve_config(config)
        apply_defaults(cfg["train"], TRAIN_MIXED_DEFAULTS)
        cfg["train"]["mixed_schedule_mode"] = str(cfg["train"]["mixed_schedule_mode"]).lower()
        if cfg["train"]["mixed_schedule_mode"] not in {"alternate", "both"}:
            raise ValueError("train.mixed_schedule_mode must be one of: alternate, both.")
        cfg["train"]["same_view_weight"] = float(cfg["train"]["same_view_weight"])
        cfg["train"]["novel_view_weight"] = float(cfg["train"]["novel_view_weight"])
        cfg["train"]["heldout_views_per_step"] = int(cfg["train"]["heldout_views_per_step"])
        if cfg["train"]["same_view_weight"] < 0.0:
            raise ValueError("train.same_view_weight must be >= 0.")
        if cfg["train"]["novel_view_weight"] < 0.0:
            raise ValueError("train.novel_view_weight must be >= 0.")
        if cfg["train"]["heldout_views_per_step"] < 0:
            raise ValueError("train.heldout_views_per_step must be >= 0.")
        if cfg["data"]["manifest_path"] is None:
            raise ValueError("Mixed same-view/heldout training requires data.manifest_path for same-view data.")
        if cfg["train"]["camera_swap_pairs_per_step"] > 0:
            raise ValueError("Mixed same-view/heldout training does not support camera_swap_pairs_per_step yet.")
        return cfg

    def load_train_sequences(self) -> list[SequenceData]:
        self.same_view_lazy_entries: list[dict[str, Any]] = []
        multicam_sequences = super().load_train_sequences()
        load_mode = self.data_cfg["train_manifest_load_mode"]
        self.same_view_sampler = ManifestSequenceSampler.from_manifest(
            self.data_cfg["manifest_path"],
            split=self.data_cfg["split"],
            data_cfg=self.data_cfg,
            model_cfg=self.model_cfg,
            device=self.device,
            load_mode=load_mode,
            sample_mode=self.data_cfg["train_manifest_sample_mode"] if load_mode == "lazy" else "random",
            prefetch_depth=0,
            prefetch_name="dynaworld-same-view-prefetch",
        )
        self.same_view_lazy_entries = self.same_view_sampler.entries if self.same_view_sampler.is_lazy else []
        self.same_view_sequences = self.same_view_sampler.sequences
        return list(multicam_sequences) + list(self.same_view_sequences)

    def validate_train_sequences(self) -> None:
        super().validate_train_sequences()
        if not self.same_view_sequences:
            raise ValueError("Mixed same-view/heldout training loaded no same-view sequences.")
        minimum_required = int(self.model_cfg["train_frame_count"])
        label = "same-view lazy manifest entry" if self.same_view_sampler.is_lazy else "same-view train sequence"
        self.same_view_sampler.validate_min_frame_count(minimum_required, label=label)

    def sample_same_view_sequence(self) -> SequenceData:
        return self.same_view_sampler.sample()

    def sample_mixed_step_batch(self, step: int) -> MixedStepBatch:
        return build_mixed_step_batch(
            step=step,
            schedule_mode=self.train_cfg["mixed_schedule_mode"],
            same_view_sequence=self.sample_same_view_sequence,
            multicam_bundle=self.multicam_bundle,
            train_frame_count=int(self.model_cfg["train_frame_count"]),
            frame_sampling=self.train_cfg["frame_sampling"],
            device=self.device,
            same_view_weight=float(self.train_cfg["same_view_weight"]),
            novel_view_weight=float(self.train_cfg["novel_view_weight"]),
            train_views_per_step=int(self.train_cfg["train_views_per_step"]),
            heldout_views_per_step=int(self.train_cfg["heldout_views_per_step"]),
        )

    def step(self, keep_preview: bool = False) -> StepResult:
        mixed_step = int(getattr(self, "_mixed_step_index", 0)) + 1
        self._mixed_step_index = mixed_step
        zero = self.sequence_data.frames.new_zeros(())
        accum = MixedStepAccumulator(
            weighted_recon_loss=zero,
            bank_rate_loss=zero,
            camera_motion_loss=zero,
            camera_temporal_loss=zero,
            camera_global_loss=zero,
        )

        with self.train_step_context():
            with self.profile_section("sample_clip"):
                mixed_batch = self.sample_mixed_step_batch(mixed_step)
            if mixed_batch.same_view is not None:
                accum.add(
                    self._backward_same_view_batch(
                        mixed_batch.same_view,
                        keep_preview=keep_preview,
                        keep_preview_media=accum.preview_render is None,
                    )
                )
            if mixed_batch.novel_view is not None:
                accum.add(
                    self._backward_novel_view_batch(
                        mixed_batch.novel_view,
                        keep_preview=keep_preview,
                        keep_preview_media=accum.preview_render is None,
                    )
                )
            self.optimizer_step()
        if accum.clip_frames is None or accum.sequence is None:
            raise ValueError("Mixed step produced no active same-view or novel-view batch.")
        return build_step_result(
            sequence_data=accum.sequence,
            clip_frames=accum.clip_frames,
            preview_render=accum.preview_render,
            preview_features=accum.preview_features,
            camera_state=None,
            loss=accum.total_loss.detach(),
            recon_loss=accum.weighted_recon_loss.detach(),
            camera_motion_loss=accum.camera_motion_loss,
            camera_temporal_loss=accum.camera_temporal_loss,
            camera_global_loss=accum.camera_global_loss,
            bank_rate_loss=accum.bank_rate_loss,
            bank_rate_terms=accum.bank_rate_terms,
            aux_loss_terms=accum.aux_loss_terms,
        )

    def _merge_bank_rate_terms(
        self,
        prefix: str,
        terms: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return {f"{prefix}_{key}": value.detach() for key, value in terms.items()}

    def _backward_same_view_batch(
        self,
        batch: SameViewBatch,
        *,
        keep_preview: bool,
        keep_preview_media: bool,
    ):
        clip_frames = batch.clip.as_video_batch()
        clip_times = batch.clip.as_time_batch(device=self.device)
        with self.profile_section("model_input"):
            model_input = self.model_input_for_clip(batch.sequence, clip_frames, clip_times)
        with self.profile_section("forward_decode"):
            decoded = self.forward_clip(model_input, clip_times)
        if decoded.camera_state is None:
            raise ValueError("Same-view mixed training requires decoded camera_state.")
        with self.profile_section("regularizers"):
            camera_loss, camera_motion_loss, camera_temporal_loss, camera_global_loss = _build_camera_loss_impl(
                clip_times,
                decoded.camera_state,
                self.loss_cfg,
            )
            bank_rate_loss, bank_rate_terms = _build_bank_rate_loss_impl(decoded, self.loss_cfg)
        recon_loss, preview_render, preview_features, aux_loss_terms = self.recon_backward(
            clip_frames,
            decoded,
            camera_loss + bank_rate_loss,
            keep_preview and keep_preview_media,
            loss_scale=float(batch.weight),
        )
        prefixed_aux = {f"same_view_{key}": value.detach() for key, value in aux_loss_terms.items()}
        weight = float(batch.weight)
        return MixedBackwardResult(
            loss_name=batch.loss_name,
            sequence=batch.sequence,
            recon_loss=recon_loss,
            weighted_recon_loss=recon_loss * weight,
            bank_rate_loss=bank_rate_loss * weight,
            camera_motion_loss=camera_motion_loss * weight,
            camera_temporal_loss=camera_temporal_loss * weight,
            camera_global_loss=camera_global_loss * weight,
            preview_render=preview_render,
            preview_features=preview_features,
            clip_frames=clip_frames,
            bank_rate_terms=self._merge_bank_rate_terms("same_view", bank_rate_terms),
            aux_loss_terms=prefixed_aux,
        )

    def _backward_novel_view_batch(
        self,
        batch: NovelViewBatch,
        *,
        keep_preview: bool,
        keep_preview_media: bool,
    ):
        clip_frames = batch.clip.as_video_batch()
        clip_times = batch.clip.as_time_batch(device=self.device)
        with self.profile_section("forward_decode"):
            decoded = self._decode_clip(batch.condition_sequence, clip_frames, clip_times)
        with self.profile_section("regularizers"):
            bank_rate_loss, bank_rate_terms = _build_bank_rate_loss_impl(decoded, self.loss_cfg)
            rig_loss = self.rig_regularization_loss()
        heldout_loss, preview_render, preview_features = self.heldout_recon_loss(
            decoded,
            clip_indices=batch.clip.frame_indices,
            views=list(batch.heldout_views),
            phase="train",
            keep_preview=keep_preview and keep_preview_media,
        )
        loss = (heldout_loss + bank_rate_loss + rig_loss) * float(batch.weight)
        with self.profile_section("backward"):
            loss.backward()
        zero = clip_frames.new_zeros(())
        weight = float(batch.weight)
        return MixedBackwardResult(
            loss_name=batch.loss_name,
            sequence=batch.condition_sequence,
            recon_loss=heldout_loss,
            weighted_recon_loss=heldout_loss * weight,
            bank_rate_loss=(bank_rate_loss + rig_loss) * weight,
            camera_motion_loss=zero,
            camera_temporal_loss=zero,
            camera_global_loss=zero,
            preview_render=preview_render,
            preview_features=preview_features,
            clip_frames=clip_frames,
            bank_rate_terms=self._merge_bank_rate_terms("heldout_view", bank_rate_terms),
        )

    def scalar_payload(self, result: StepResult) -> dict[str, Any]:
        payload = super().scalar_payload(result)
        payload["Mixed/ScheduleMode"] = 1.0 if self.train_cfg["mixed_schedule_mode"] == "both" else 0.0
        payload["Mixed/SameViewWeight"] = float(self.train_cfg["same_view_weight"])
        payload["Mixed/NovelViewWeight"] = float(self.train_cfg["novel_view_weight"])
        payload["Mixed/HeldoutViewsPerStep"] = int(self.train_cfg["heldout_views_per_step"])
        payload["Mixed/SameViewSequenceCount"] = self.same_view_sampler.sequence_count
        return payload


def run_training(config: dict[str, Any]) -> None:
    MixedSameHeldoutPrecomputedFeatureTrainer(config).run()


__all__ = [
    "MixedBackwardResult",
    "MixedSameHeldoutPrecomputedFeatureTrainer",
    "MixedStepAccumulator",
    "TRAIN_MIXED_DEFAULTS",
    "run_training",
]
