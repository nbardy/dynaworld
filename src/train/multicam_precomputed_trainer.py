from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from clip_sampling import sample_clip_batch
from camera_swap_sampling import (
    CameraSwapPair,
    camera_swap_pair_counts,
    sample_train_camera_swap_pairs,
)
from mixed_data_scheduler import sample_view_indices
from camera_rig import LearnableCameraRig
from config_utils import apply_defaults
from multicam_video_data import (
    cameras_from_K_w2c,
    heldout_cameras_from_K_w2c,
    load_multicam_video_bundle,
    source_relative_cameras_from_K_w2c,
)
from objective.types import BackgroundSample, RenderedView, RunPhase
from pipeline.diagnostics import decoded_temporal_payload_from_sequence
from pipeline.losses import build_bank_rate_loss as _build_bank_rate_loss_impl
from pipeline.validation_media import (
    multicam_validation_video_payload,
)
from relative_pose import (
    RelativePoseCrossAttentionHead,
    compose_cameras_with_se3_residual,
    compose_transform_with_se3_residual,
    se3_cycle_loss,
    se3_residual_identity_loss,
)
from rendering import resize_images
from runtime_types import CameraState, ClipBatch, SequenceData, StepResult, build_step_result
from sequence_data import prepare_clip
from precomputed_feature_trainer import PrecomputedFeatureImplicitTrainer


DATA_MULTICAM_DEFAULTS = {
    "frame_indices": None,
    "multicam_manifest": "data/multicam_val/clip_sets/multicam_val_v1_128_4fps_16f/manifest.jsonl",
    "multicam_split": "val",
    "multicam_sample_id": None,
    "multicam_sample_index": 0,
    "multicam_train_cameras": None,
    "multicam_heldout_cameras": None,
    "multicam_heldout_camera": None,
    "multicam_anchor_camera": None,
    "multicam_condition_camera": None,
}


CAMERA_RIG_DEFAULTS = {
    "rig_init": "deepview",
    "rig_radius": None,
    "rig_learn_global_se3": True,
    "rig_learn_per_camera_se3": True,
    "rig_anchor_policy": "soft",
    "rig_rotation_degrees": 15.0,
    "rig_translation_ratio": 0.2,
    "rig_regularization_weight": 1.0e-4,
}


TRAIN_MULTICAM_DEFAULTS = {
    "train_views_per_step": 0,
    "camera_rig_lr": None,
    "camera_swap_mode": "external_rig",
    "camera_swap_pairs_per_step": 0,
    "camera_swap_include_self": True,
    "camera_swap_include_cross": True,
    "camera_swap_self_pair_probability": None,
    "relpose_layers": 1,
    "relpose_query_count": 1,
    "relpose_mlp_ratio": 2.0,
    "relpose_head_hidden_dim": None,
    "relpose_query_init_std": 0.02,
    "relpose_output_init_std": 0.0,
    "relpose_pair_delta_init_std": 0.0,
    "relpose_max_rotation_degrees": 5.0,
    "relpose_max_translation_ratio": 0.05,
    "relpose_identity_loss_weight": 1.0,
    "relpose_cycle_loss_weight": 0.1,
}


class MulticamPrecomputedFeatureImplicitTrainer(PrecomputedFeatureImplicitTrainer):
    @classmethod
    def resolve_config(cls, config: dict[str, Any]) -> dict[str, Any]:
        cfg = super().resolve_config(config)
        apply_defaults(cfg["data"], DATA_MULTICAM_DEFAULTS)
        apply_defaults(cfg["camera"], CAMERA_RIG_DEFAULTS)
        apply_defaults(cfg["train"], TRAIN_MULTICAM_DEFAULTS)
        cfg["data"]["multicam_manifest"] = Path(cfg["data"]["multicam_manifest"])
        cfg["data"]["multicam_split"] = str(cfg["data"]["multicam_split"])
        if cfg["data"]["multicam_sample_id"] is not None:
            cfg["data"]["multicam_sample_id"] = str(cfg["data"]["multicam_sample_id"])
        for key in ("multicam_train_cameras", "multicam_heldout_cameras"):
            if cfg["data"][key] is not None:
                cfg["data"][key] = [str(camera) for camera in cfg["data"][key]]
        for key in ("multicam_heldout_camera", "multicam_anchor_camera", "multicam_condition_camera"):
            if cfg["data"][key] is not None:
                cfg["data"][key] = str(cfg["data"][key])
        if cfg["camera"]["rig_radius"] is None:
            cfg["camera"]["rig_radius"] = float(cfg["camera"]["base_radius"])
        cfg["camera"]["rig_init"] = str(cfg["camera"]["rig_init"]).lower()
        cfg["camera"]["rig_anchor_policy"] = str(cfg["camera"]["rig_anchor_policy"]).lower()
        cfg["camera"]["rig_learn_global_se3"] = bool(cfg["camera"]["rig_learn_global_se3"])
        cfg["camera"]["rig_learn_per_camera_se3"] = bool(cfg["camera"]["rig_learn_per_camera_se3"])
        cfg["camera"]["rig_radius"] = float(cfg["camera"]["rig_radius"])
        cfg["camera"]["rig_rotation_degrees"] = float(cfg["camera"]["rig_rotation_degrees"])
        cfg["camera"]["rig_translation_ratio"] = float(cfg["camera"]["rig_translation_ratio"])
        cfg["camera"]["rig_regularization_weight"] = float(cfg["camera"]["rig_regularization_weight"])
        cfg["train"]["train_views_per_step"] = int(cfg["train"]["train_views_per_step"])
        if cfg["train"]["camera_rig_lr"] is not None:
            cfg["train"]["camera_rig_lr"] = float(cfg["train"]["camera_rig_lr"])
        cfg["train"]["camera_swap_mode"] = str(cfg["train"]["camera_swap_mode"]).lower()
        if cfg["train"]["camera_swap_mode"] not in {"external_rig", "oracle_relative", "learned_residual"}:
            raise ValueError("train.camera_swap_mode must be one of: external_rig, oracle_relative, learned_residual")
        cfg["train"]["camera_swap_pairs_per_step"] = int(cfg["train"]["camera_swap_pairs_per_step"])
        cfg["train"]["camera_swap_include_self"] = bool(cfg["train"]["camera_swap_include_self"])
        cfg["train"]["camera_swap_include_cross"] = bool(cfg["train"]["camera_swap_include_cross"])
        if cfg["train"]["camera_swap_self_pair_probability"] is not None:
            cfg["train"]["camera_swap_self_pair_probability"] = float(cfg["train"]["camera_swap_self_pair_probability"])
        cfg["train"]["relpose_layers"] = int(cfg["train"]["relpose_layers"])
        cfg["train"]["relpose_query_count"] = int(cfg["train"]["relpose_query_count"])
        cfg["train"]["relpose_mlp_ratio"] = float(cfg["train"]["relpose_mlp_ratio"])
        if cfg["train"]["relpose_head_hidden_dim"] is not None:
            cfg["train"]["relpose_head_hidden_dim"] = int(cfg["train"]["relpose_head_hidden_dim"])
        cfg["train"]["relpose_query_init_std"] = float(cfg["train"]["relpose_query_init_std"])
        cfg["train"]["relpose_output_init_std"] = float(cfg["train"]["relpose_output_init_std"])
        if cfg["train"]["relpose_output_init_std"] < 0.0:
            raise ValueError("train.relpose_output_init_std must be >= 0.")
        cfg["train"]["relpose_pair_delta_init_std"] = float(cfg["train"]["relpose_pair_delta_init_std"])
        if cfg["train"]["relpose_pair_delta_init_std"] < 0.0:
            raise ValueError("train.relpose_pair_delta_init_std must be >= 0.")
        cfg["train"]["relpose_max_rotation_degrees"] = float(cfg["train"]["relpose_max_rotation_degrees"])
        cfg["train"]["relpose_max_translation_ratio"] = float(cfg["train"]["relpose_max_translation_ratio"])
        cfg["train"]["relpose_identity_loss_weight"] = float(cfg["train"]["relpose_identity_loss_weight"])
        cfg["train"]["relpose_cycle_loss_weight"] = float(cfg["train"]["relpose_cycle_loss_weight"])
        return cfg

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.best_heldout_eval_psnr: float | None = None
        self.best_heldout_eval_ssim_at_best_psnr: float | None = None
        self.best_heldout_eval_step: int | None = None
        self.last_camera_swap_counts: dict[str, int] | None = None
        self.relpose_head = None
        rig_params = [parameter for parameter in self.camera_rig.parameters() if parameter.requires_grad]
        if self.train_cfg["camera_swap_mode"] == "external_rig" and rig_params:
            self.optimizer.add_param_group(
                {
                    "params": rig_params,
                    "lr": self.train_cfg["camera_rig_lr"] or self.train_cfg["lr"],
                }
            )
        if self.train_cfg["camera_swap_mode"] == "learned_residual":
            if self.model_cfg["video_encoder_backend"] not in {"precomputed", "precomputed_ltx"}:
                raise ValueError("camera_swap_mode='learned_residual' requires precomputed video features.")
            self.relpose_head = RelativePoseCrossAttentionHead(
                dim=int(self.model_cfg["model_dim"]),
                num_heads=int(self.model_cfg["num_heads"]),
                layers=int(self.train_cfg["relpose_layers"]),
                query_count=int(self.train_cfg["relpose_query_count"]),
                mlp_ratio=float(self.train_cfg["relpose_mlp_ratio"]),
                hidden_dim=self.train_cfg["relpose_head_hidden_dim"],
                query_init_std=float(self.train_cfg["relpose_query_init_std"]),
                output_init_std=float(self.train_cfg["relpose_output_init_std"]),
                pair_delta_init_std=float(self.train_cfg["relpose_pair_delta_init_std"]),
                max_rotation_degrees=float(self.train_cfg["relpose_max_rotation_degrees"]),
                max_translation=float(self.cfg["camera"]["rig_radius"])
                * float(self.train_cfg["relpose_max_translation_ratio"]),
            ).to(self.device)
            self.optimizer.add_param_group({"params": self.relpose_head.parameters(), "lr": self.train_cfg["lr"]})

    def load_train_sequences(self):
        self.multicam_bundle = load_multicam_video_bundle(
            data_cfg=self.data_cfg,
            camera_cfg=self.cfg["camera"],
            target_size=int(self.model_cfg["size"]),
            device=self.device,
        )
        if str(self.train_cfg["camera_swap_mode"]) in {"oracle_relative", "learned_residual"}:
            return list(self.multicam_bundle.train_sequences)
        return [self.multicam_bundle.condition_sequence]

    def load_eval_sequences(self):
        return []

    def on_sequences_loaded(self) -> None:
        super().on_sequences_loaded()
        train_cameras = cameras_from_K_w2c(self.multicam_bundle.train_K, self.multicam_bundle.train_w2c)
        heldout_cameras = None
        if self.multicam_bundle.heldout_K is not None and self.multicam_bundle.heldout_w2c is not None:
            heldout_cameras = heldout_cameras_from_K_w2c(
                self.multicam_bundle.heldout_K,
                self.multicam_bundle.heldout_w2c,
            )
        self.camera_rig = LearnableCameraRig(
            train_cameras,
            heldout_cameras=heldout_cameras,
            learn_global_se3=bool(self.cfg["camera"]["rig_learn_global_se3"]),
            learn_per_camera_se3=bool(self.cfg["camera"]["rig_learn_per_camera_se3"]),
            anchor_policy=str(self.cfg["camera"]["rig_anchor_policy"]),
            max_rotation_degrees=float(self.cfg["camera"]["rig_rotation_degrees"]),
            max_translation_ratio=float(self.cfg["camera"]["rig_translation_ratio"]),
            radius=float(self.cfg["camera"]["rig_radius"]),
        ).to(self.device)
        print(
            "Multicam rig: "
            f"views={self.multicam_bundle.train_camera_names}, "
            f"heldout={self.multicam_bundle.heldout_camera_names}, "
            f"pose_source={self.multicam_bundle.pose_source}, "
            f"init={self.cfg['camera']['rig_init']}, "
            f"anchor_policy={self.cfg['camera']['rig_anchor_policy']}"
        )

    def sample_views(self) -> list[int]:
        return list(
            sample_view_indices(
                self.multicam_bundle.train_view_count,
                int(self.train_cfg["train_views_per_step"]),
                device=self.device,
            )
        )

    def sample_camera_swap_pairs(self) -> tuple[CameraSwapPair, ...]:
        return sample_train_camera_swap_pairs(
            self.multicam_bundle.train_view_count,
            pairs_per_step=int(self.train_cfg["camera_swap_pairs_per_step"]),
            include_self=bool(self.train_cfg["camera_swap_include_self"]),
            include_cross=bool(self.train_cfg["camera_swap_include_cross"]),
            self_pair_probability=self.train_cfg["camera_swap_self_pair_probability"],
            train_camera_names=self.multicam_bundle.train_camera_names,
        )

    def sample_multicam_clip_batch(self) -> tuple[SequenceData, ClipBatch, list[int]]:
        sequence_data = self.sequence_data
        clip = sample_clip_batch(
            sequence_data,
            train_frame_count=int(self.model_cfg["train_frame_count"]),
            frame_sampling=self.train_cfg["frame_sampling"],
            device=self.device,
        )
        return sequence_data, clip, self.sample_views()

    def sample_multicam_clip(self):
        sequence_data, clip, views = self.sample_multicam_clip_batch()
        return (
            sequence_data,
            clip.frame_indices,
            clip.as_video_batch(),
            clip.as_time_batch(device=self.device),
            views,
        )

    def _decode_clip(self, sequence_data, clip_frames: torch.Tensor, clip_times: torch.Tensor):
        model_input = self.model_input_for_clip(sequence_data, clip_frames, clip_times)
        return self.forward_clip(model_input, clip_times)

    def _decode_source_view(self, source_view: int, clip_indices: torch.Tensor):
        sequence_data = self.multicam_bundle.train_sequences[int(source_view)]
        clip_frames, clip_times = prepare_clip(sequence_data, clip_indices)
        return sequence_data, clip_frames, clip_times, self._decode_clip(sequence_data, clip_frames, clip_times)

    def projected_feature_memory_for_train_view(self, view: int, clip_indices: torch.Tensor) -> torch.Tensor:
        sequence_data = self.multicam_bundle.train_sequences[int(view)]
        clip_frames, clip_times = prepare_clip(sequence_data, clip_indices)
        model_input = self.model_input_for_clip(sequence_data, clip_frames, clip_times)
        with self.autocast_context():
            return self.model.video_encoder(model_input, frame_times=clip_times)

    def frame_times_for_indices(self, clip_indices: torch.Tensor) -> torch.Tensor:
        denominator = max(self.sequence_data.frame_count - 1, 1)
        return clip_indices.to(dtype=torch.float32) / float(denominator)

    def render_view_clip(
        self,
        decoded,
        *,
        view: int,
        clip_indices: torch.Tensor,
        phase: RunPhase,
        background: BackgroundSample | None = None,
    ) -> RenderedView:
        cameras = self.camera_rig.cameras_for_view(view, clip_indices)
        target = self.make_target_view(
            view_id=f"train_view_{view}",
            frames=self.multicam_bundle.train_frames[int(view), clip_indices],
            frame_indices=clip_indices,
            frame_times=self.frame_times_for_indices(clip_indices),
            cameras=cameras,
            role="train",
            camera_owner="external_rig",
            camera_name=self.multicam_bundle.train_camera_names[int(view)],
            metrics_prefix=f"TrainView{view}",
            log_media=int(view) == 0,
        )
        return self.rgb_objective.render_view(
            decoded,
            target,
            phase=phase,
            background=background,
        )

    def camera_swap_target_frames(self, pair: CameraSwapPair, clip_indices: torch.Tensor) -> torch.Tensor:
        if pair.target_set == "train":
            return self.multicam_bundle.train_frames[int(pair.target_view), clip_indices]
        if pair.target_set == "heldout" and self.multicam_bundle.heldout_frames is not None:
            return self.multicam_bundle.heldout_frames[int(pair.target_view), clip_indices]
        raise ValueError(f"Unsupported camera-swap target set {pair.target_set!r}.")

    def source_relative_cameras_for_pair(self, pair: CameraSwapPair, clip_indices: torch.Tensor):
        if pair.source_set != "train":
            raise ValueError("Camera-swap source worlds must come from train cameras.")
        if pair.query_set == "train":
            target_K = self.multicam_bundle.train_K[int(pair.query_view)]
            target_w2c = self.multicam_bundle.train_w2c[int(pair.query_view)]
        elif (
            pair.query_set == "heldout"
            and self.multicam_bundle.heldout_K is not None
            and self.multicam_bundle.heldout_w2c is not None
        ):
            target_K = self.multicam_bundle.heldout_K[int(pair.query_view)]
            target_w2c = self.multicam_bundle.heldout_w2c[int(pair.query_view)]
        else:
            raise ValueError(f"Unsupported camera-swap query set {pair.query_set!r}.")
        return source_relative_cameras_from_K_w2c(
            source_w2c=self.multicam_bundle.train_w2c[int(pair.source_view)],
            target_K=target_K,
            target_w2c=target_w2c,
            frame_indices=clip_indices,
        )

    def render_camera_swap_pair(
        self,
        decoded,
        *,
        pair: CameraSwapPair,
        clip_indices: torch.Tensor,
        phase: RunPhase,
        background: BackgroundSample | None = None,
        log_media: bool = False,
        residual: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> RenderedView:
        target_view = int(pair.target_view)
        cameras = self.source_relative_cameras_for_pair(pair, clip_indices)
        if residual is not None:
            cameras = compose_cameras_with_se3_residual(cameras, residual[0], residual[1])
        if pair.target_set == "train":
            target_name = self.multicam_bundle.train_camera_names[target_view]
            target_frames = self.multicam_bundle.train_frames[target_view, clip_indices]
            role = "train"
            camera_role = "target"
            metrics_prefix = f"SwapSrc{pair.source_view}_Tgt{target_view}_{target_name}"
        elif pair.target_set == "heldout" and self.multicam_bundle.heldout_frames is not None:
            heldout_names = self.multicam_bundle.heldout_camera_names or []
            target_name = heldout_names[target_view] if target_view < len(heldout_names) else f"heldout_{target_view}"
            target_frames = self.multicam_bundle.heldout_frames[target_view, clip_indices]
            role = "heldout"
            camera_role = "heldout"
            metrics_prefix = f"Heldout{target_view}_{target_name}"
        else:
            raise ValueError(f"Unsupported camera-swap target set {pair.target_set!r}.")
        source_name = self.multicam_bundle.train_camera_names[int(pair.source_view)]
        target = self.make_target_view(
            view_id=f"swap_source_{pair.source_view}_target_{target_view}",
            frames=target_frames,
            frame_indices=clip_indices,
            frame_times=self.frame_times_for_indices(clip_indices),
            cameras=cameras,
            role=role,
            camera_role=camera_role,
            camera_owner="external_rig",
            camera_name=target_name,
            metrics_prefix=f"{metrics_prefix}_from_{source_name}",
            log_media=log_media,
        )
        return self.rgb_objective.render_view(
            decoded,
            target,
            phase=phase,
            background=background,
        )

    def relpose_residual_for_pair(
        self,
        pair: CameraSwapPair,
        *,
        clip_indices: torch.Tensor,
        memory_cache: dict[int, torch.Tensor],
    ) -> tuple[tuple[torch.Tensor, torch.Tensor] | None, torch.Tensor | None]:
        if self.relpose_head is None or not pair.is_train_cross_view:
            return None, None
        source_view = int(pair.source_view)
        target_view = int(pair.target_view)
        if source_view not in memory_cache:
            memory_cache[source_view] = self.projected_feature_memory_for_train_view(source_view, clip_indices)
        if target_view not in memory_cache:
            memory_cache[target_view] = self.projected_feature_memory_for_train_view(target_view, clip_indices)
        rotation, translation = self.relpose_head(memory_cache[source_view], memory_cache[target_view])
        loss = se3_residual_identity_loss(rotation, translation)
        return (rotation, translation), loss

    def source_relative_camera_to_world_tensor(
        self,
        pair: CameraSwapPair,
        clip_indices: torch.Tensor,
        residual: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> torch.Tensor:
        cameras = self.source_relative_cameras_for_pair(pair, clip_indices)
        transforms = torch.stack([camera.camera_to_world for camera in cameras], dim=0)
        if residual is None:
            return transforms
        return compose_transform_with_se3_residual(transforms, residual[0], residual[1])

    def relpose_cycle_loss_for_pair(
        self,
        pair: CameraSwapPair,
        *,
        clip_indices: torch.Tensor,
        residual: tuple[torch.Tensor, torch.Tensor] | None,
        memory_cache: dict[int, torch.Tensor],
    ) -> torch.Tensor | None:
        if self.relpose_head is None or not pair.is_train_cross_view:
            return None
        reverse_pair = CameraSwapPair(
            source_set="train",
            source_view=int(pair.target_view),
            query_set="train",
            query_view=int(pair.source_view),
            target_set="train",
            target_view=int(pair.source_view),
            source_name=pair.target_name,
            query_name=pair.source_name,
            target_name=pair.source_name,
        )
        reverse_residual, _reverse_identity_loss = self.relpose_residual_for_pair(
            reverse_pair,
            clip_indices=clip_indices,
            memory_cache=memory_cache,
        )
        forward_c2w = self.source_relative_camera_to_world_tensor(pair, clip_indices, residual)
        reverse_c2w = self.source_relative_camera_to_world_tensor(reverse_pair, clip_indices, reverse_residual)
        return se3_cycle_loss(forward_c2w, reverse_c2w)

    def render_heldout_view_clip(
        self,
        decoded,
        *,
        view: int,
        clip_indices: torch.Tensor,
        phase: RunPhase,
        background: BackgroundSample | None = None,
    ) -> RenderedView:
        if self.multicam_bundle.heldout_frames is None:
            raise ValueError("Heldout render requested but no heldout frames are loaded.")
        cameras = self.camera_rig.heldout_cameras_for(view, clip_indices)
        camera_names = self.multicam_bundle.heldout_camera_names or []
        camera_name = camera_names[view] if view < len(camera_names) else f"heldout_{view}"
        target = self.make_target_view(
            view_id=f"heldout_view_{view}",
            frames=self.multicam_bundle.heldout_frames[int(view), clip_indices],
            frame_indices=clip_indices,
            frame_times=self.frame_times_for_indices(clip_indices),
            cameras=cameras,
            role="heldout",
            camera_role="heldout",
            camera_owner="external_rig",
            camera_name=camera_name,
            metrics_prefix=f"Heldout{view}_{camera_name}",
            log_media=True,
        )
        return self.rgb_objective.render_view(
            decoded,
            target,
            phase=phase,
            background=background,
        )

    def _recon_loss_for_views(
        self,
        decoded,
        *,
        clip_indices: torch.Tensor,
        views: list[int],
        phase: RunPhase,
        keep_preview: bool,
        target_frames: torch.Tensor,
        render_fn: Callable[..., RenderedView],
        context: str,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if not views:
            raise ValueError(f"{context} reconstruction requires at least one view.")
        recon_loss = target_frames.new_zeros(())
        preview_render = None
        preview_features = None
        background = self.rgb_objective.sample_background(
            phase=phase,
            like=target_frames[int(views[0]), clip_indices],
            frame_count=len(clip_indices),
        )
        for view in views:
            with self.profile_section("render_view_total"):
                rendered = render_fn(
                    decoded,
                    view=int(view),
                    clip_indices=clip_indices,
                    phase=phase,
                    background=background,
                )
            view_loss, preview_render, preview_features = self._rendered_view_recon_loss(
                rendered,
                context=context,
                keep_preview=keep_preview,
                preview_render=preview_render,
                preview_features=preview_features,
            )
            recon_loss = recon_loss + view_loss
        return recon_loss / float(max(len(views), 1)), preview_render, preview_features

    def _rendered_view_recon_loss(
        self,
        rendered: RenderedView,
        *,
        context: str,
        keep_preview: bool,
        preview_render: torch.Tensor | None,
        preview_features: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        self.rgb_objective.require_alpha_for_feature_background(rendered, context=context)
        with self.profile_section("recon_loss"):
            recon_loss = self.rgb_objective.reconstruction_loss(rendered)
        if keep_preview and preview_render is None:
            preview_render = rendered.rgb[0].detach()
            if self.feature_pca_log:
                preview_features = rendered.features[0].detach()
        return recon_loss, preview_render, preview_features

    def multicam_recon_loss(
        self,
        decoded,
        *,
        clip_indices: torch.Tensor,
        views: list[int],
        phase: RunPhase,
        keep_preview: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        return self._recon_loss_for_views(
            decoded,
            clip_indices=clip_indices,
            views=views,
            phase=phase,
            keep_preview=keep_preview,
            target_frames=self.multicam_bundle.train_frames,
            render_fn=self.render_view_clip,
            context="multicam training",
        )

    def heldout_recon_loss(
        self,
        decoded,
        *,
        clip_indices: torch.Tensor,
        views: list[int],
        phase: RunPhase,
        keep_preview: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if self.multicam_bundle.heldout_frames is None:
            raise ValueError("Heldout reconstruction requested but no heldout frames are loaded.")
        return self._recon_loss_for_views(
            decoded,
            clip_indices=clip_indices,
            views=views,
            phase=phase,
            keep_preview=keep_preview,
            target_frames=self.multicam_bundle.heldout_frames,
            render_fn=self.render_heldout_view_clip,
            context="heldout training",
        )

    def camera_swap_recon_loss(
        self,
        *,
        clip_indices: torch.Tensor,
        pairs: tuple[CameraSwapPair, ...],
        phase: RunPhase,
        keep_preview: bool,
    ):
        if not pairs:
            raise ValueError("camera_swap_recon_loss requires at least one pair.")

        recon_loss = self.multicam_bundle.train_frames.new_zeros(())
        bank_rate_loss = self.multicam_bundle.train_frames.new_zeros(())
        bank_rate_terms_accum: dict[str, torch.Tensor] = {}
        preview_render = None
        preview_features = None
        first_camera_state = None
        first_clip_frames = None
        first_sequence_data = None
        relpose_identity_loss = self.multicam_bundle.train_frames.new_zeros(())
        relpose_identity_count = 0
        relpose_cycle_loss = self.multicam_bundle.train_frames.new_zeros(())
        relpose_cycle_count = 0
        relpose_memory_cache: dict[int, torch.Tensor] = {}

        pairs_by_source: dict[int, list[CameraSwapPair]] = {}
        for pair in pairs:
            pairs_by_source.setdefault(int(pair.source_view), []).append(pair)

        rendered_count = 0
        for source_index, source_pairs in pairs_by_source.items():
            with self.profile_section("camera_swap/source_decode"):
                sequence_data, clip_frames, clip_times, decoded = self._decode_source_view(source_index, clip_indices)
            if first_sequence_data is None:
                first_sequence_data = sequence_data
                first_clip_frames = clip_frames
                first_camera_state = decoded.camera_state
            with self.profile_section("camera_swap/bank_rate"):
                source_bank_loss, source_bank_terms = _build_bank_rate_loss_impl(decoded, self.loss_cfg)
            bank_rate_loss = bank_rate_loss + source_bank_loss
            for key, value in source_bank_terms.items():
                bank_rate_terms_accum[key] = bank_rate_terms_accum.get(key, value.new_zeros(())) + value

            for pair_index, pair in enumerate(source_pairs):
                with self.profile_section("camera_swap/background"):
                    target_frames = self.camera_swap_target_frames(pair, clip_indices)
                    background = self.rgb_objective.sample_background(
                        phase=phase,
                        like=target_frames,
                        frame_count=len(clip_indices),
                    )
                with self.profile_section("camera_swap/relpose_predict"):
                    residual, residual_loss = self.relpose_residual_for_pair(
                        pair,
                        clip_indices=clip_indices,
                        memory_cache=relpose_memory_cache,
                    )
                if residual_loss is not None:
                    relpose_identity_loss = relpose_identity_loss + residual_loss
                    relpose_identity_count += 1
                with self.profile_section("camera_swap/relpose_cycle"):
                    cycle_loss = self.relpose_cycle_loss_for_pair(
                        pair,
                        clip_indices=clip_indices,
                        residual=residual,
                        memory_cache=relpose_memory_cache,
                    )
                if cycle_loss is not None:
                    relpose_cycle_loss = relpose_cycle_loss + cycle_loss
                    relpose_cycle_count += 1
                with self.profile_section("render_view_total"):
                    rendered = self.render_camera_swap_pair(
                        decoded,
                        pair=pair,
                        clip_indices=clip_indices,
                        phase=phase,
                        background=background,
                        log_media=keep_preview and preview_render is None and pair_index == 0,
                        residual=residual,
                    )
                view_loss, preview_render, preview_features = self._rendered_view_recon_loss(
                    rendered,
                    context="camera-swap training",
                    keep_preview=keep_preview,
                    preview_render=preview_render,
                    preview_features=preview_features,
                )
                recon_loss = recon_loss + view_loss
                rendered_count += 1

        source_count = float(max(len(pairs_by_source), 1))
        bank_rate_terms = {key: value / source_count for key, value in bank_rate_terms_accum.items()}
        bank_rate_loss = bank_rate_loss / source_count
        if relpose_identity_count:
            relpose_identity_loss = relpose_identity_loss / float(relpose_identity_count)
            relpose_weighted = relpose_identity_loss * float(self.train_cfg["relpose_identity_loss_weight"])
            bank_rate_loss = bank_rate_loss + relpose_weighted
            bank_rate_terms["relpose_identity_loss"] = relpose_identity_loss
            bank_rate_terms["relpose_identity_loss_weighted"] = relpose_weighted
        if relpose_cycle_count:
            relpose_cycle_loss = relpose_cycle_loss / float(relpose_cycle_count)
            relpose_cycle_weighted = relpose_cycle_loss * float(self.train_cfg["relpose_cycle_loss_weight"])
            bank_rate_loss = bank_rate_loss + relpose_cycle_weighted
            bank_rate_terms["relpose_cycle_loss"] = relpose_cycle_loss
            bank_rate_terms["relpose_cycle_loss_weighted"] = relpose_cycle_weighted
        return (
            recon_loss / float(rendered_count),
            bank_rate_loss,
            bank_rate_terms,
            preview_render,
            preview_features,
            first_camera_state,
            first_clip_frames,
            first_sequence_data,
        )

    def rig_regularization_loss(self) -> torch.Tensor:
        return float(self.cfg["camera"]["rig_regularization_weight"]) * self.camera_rig.regularization_loss()

    def _step_result(
        self,
        *,
        sequence_data: SequenceData,
        clip_frames: torch.Tensor,
        preview_render: torch.Tensor | None,
        preview_features: torch.Tensor | None,
        camera_state: CameraState | None,
        loss: torch.Tensor,
        recon_loss: torch.Tensor,
        bank_rate_loss: torch.Tensor,
        bank_rate_terms: dict[str, torch.Tensor],
    ) -> StepResult:
        return build_step_result(
            sequence_data=sequence_data,
            clip_frames=clip_frames,
            preview_render=preview_render,
            preview_features=preview_features,
            camera_state=camera_state,
            loss=loss,
            recon_loss=recon_loss,
            bank_rate_loss=bank_rate_loss,
            bank_rate_terms=bank_rate_terms,
        )

    @torch.no_grad()
    def initial_step_result(self) -> StepResult:
        with self.model_eval_mode():
            if self.train_cfg["camera_swap_mode"] in {"oracle_relative", "learned_residual"}:
                clip_indices = self.initial_clip_indices()
                pairs = self.sample_camera_swap_pairs()
                self.last_camera_swap_counts = camera_swap_pair_counts(pairs)
                (
                    recon_loss,
                    bank_rate_loss,
                    bank_rate_terms,
                    preview_render,
                    preview_features,
                    camera_state,
                    clip_frames,
                    sequence_data,
                ) = self.camera_swap_recon_loss(
                    clip_indices=clip_indices,
                    pairs=pairs,
                    phase="eval",
                    keep_preview=True,
                )
                loss = recon_loss + bank_rate_loss
                return self._step_result(
                    sequence_data=sequence_data,
                    clip_frames=clip_frames,
                    preview_render=preview_render,
                    preview_features=preview_features,
                    camera_state=camera_state,
                    loss=loss,
                    recon_loss=recon_loss,
                    bank_rate_loss=bank_rate_loss,
                    bank_rate_terms=bank_rate_terms,
                )

            sequence_data = self.sequence_data
            clip_indices, clip_frames, clip_times = self.initial_clip_for_sequence(sequence_data)
            decoded = self._decode_clip(sequence_data, clip_frames, clip_times)
            bank_rate_loss, bank_rate_terms = _build_bank_rate_loss_impl(decoded, self.loss_cfg)
            rig_loss = self.rig_regularization_loss()
            recon_loss, preview_render, preview_features = self.multicam_recon_loss(
                decoded,
                clip_indices=clip_indices,
                views=list(range(self.multicam_bundle.train_view_count)),
                phase="eval",
                keep_preview=True,
            )
            loss = recon_loss + bank_rate_loss + rig_loss
            return self._step_result(
                sequence_data=sequence_data,
                clip_frames=clip_frames,
                preview_render=preview_render,
                preview_features=preview_features,
                camera_state=decoded.camera_state,
                loss=loss,
                recon_loss=recon_loss,
                bank_rate_loss=bank_rate_loss + rig_loss,
                bank_rate_terms=bank_rate_terms,
            )

    def step(self, keep_preview: bool = False) -> StepResult:
        with self.train_step_context():
            if self.train_cfg["camera_swap_mode"] in {"oracle_relative", "learned_residual"}:
                with self.profile_section("sample_clip"):
                    clip = sample_clip_batch(
                        self.sequence_data,
                        train_frame_count=int(self.model_cfg["train_frame_count"]),
                        frame_sampling=self.train_cfg["frame_sampling"],
                        device=self.device,
                    )
                    clip_indices = clip.frame_indices
                    pairs = self.sample_camera_swap_pairs()
                    self.last_camera_swap_counts = camera_swap_pair_counts(pairs)
                (
                    recon_loss,
                    bank_rate_loss,
                    bank_rate_terms,
                    preview_render,
                    preview_features,
                    camera_state,
                    clip_frames,
                    sequence_data,
                ) = self.camera_swap_recon_loss(
                    clip_indices=clip_indices,
                    pairs=pairs,
                    phase="train",
                    keep_preview=keep_preview,
                )
                rig_loss = clip_frames.new_zeros(())
                loss = recon_loss + bank_rate_loss + rig_loss
                with self.profile_section("backward"):
                    loss.backward()
                self.optimizer_step()
                return self._step_result(
                    sequence_data=sequence_data,
                    clip_frames=clip_frames,
                    preview_render=preview_render,
                    preview_features=preview_features,
                    camera_state=camera_state,
                    loss=loss,
                    recon_loss=recon_loss,
                    bank_rate_loss=bank_rate_loss + rig_loss,
                    bank_rate_terms=bank_rate_terms,
                )

            with self.profile_section("sample_clip"):
                sequence_data, clip, views = self.sample_multicam_clip_batch()
                clip_indices = clip.frame_indices
                clip_frames = clip.as_video_batch()
                clip_times = clip.as_time_batch(device=self.device)
            with self.profile_section("forward_decode"):
                decoded = self._decode_clip(sequence_data, clip_frames, clip_times)
            with self.profile_section("regularizers"):
                bank_rate_loss, bank_rate_terms = _build_bank_rate_loss_impl(decoded, self.loss_cfg)
                rig_loss = self.rig_regularization_loss()
            recon_loss, preview_render, preview_features = self.multicam_recon_loss(
                decoded,
                clip_indices=clip_indices,
                views=views,
                phase="train",
                keep_preview=keep_preview,
            )
            loss = recon_loss + bank_rate_loss + rig_loss
            with self.profile_section("backward"):
                loss.backward()
            self.optimizer_step()
        return self._step_result(
            sequence_data=sequence_data,
            clip_frames=clip_frames,
            preview_render=preview_render,
            preview_features=preview_features,
            camera_state=decoded.camera_state,
            loss=loss,
            recon_loss=recon_loss,
            bank_rate_loss=bank_rate_loss + rig_loss,
            bank_rate_terms=bank_rate_terms,
        )

    def scalar_payload(self, result: StepResult) -> dict[str, Any]:
        payload = super().scalar_payload(result)
        payload.update(self.camera_rig.metrics())
        payload["TrainViewCount"] = self.multicam_bundle.train_view_count
        payload["TrainViewsPerStep"] = int(self.train_cfg["train_views_per_step"])
        payload["CameraSwap/OracleRelativeMode"] = 1.0 if self.train_cfg["camera_swap_mode"] == "oracle_relative" else 0.0
        payload["CameraSwap/LearnedResidualMode"] = 1.0 if self.train_cfg["camera_swap_mode"] == "learned_residual" else 0.0
        payload["CameraSwap/PairsPerStepConfig"] = int(self.train_cfg["camera_swap_pairs_per_step"])
        if self.train_cfg["camera_swap_mode"] in {"oracle_relative", "learned_residual"}:
            counts = self.last_camera_swap_counts or camera_swap_pair_counts(self.sample_camera_swap_pairs())
            payload.update({f"CameraSwap/{key}Pairs": value for key, value in counts.items()})
        payload["Rig/RegularizationWeight"] = float(self.cfg["camera"]["rig_regularization_weight"])
        return payload

    def update_best_heldout_payload(self, payload: dict[str, Any], step: int | None) -> None:
        psnr_values = [
            float(value)
            for key, value in payload.items()
            if key.startswith("Heldout") and key.endswith("/Eval/PSNR")
        ]
        if not psnr_values:
            return

        mean_psnr = sum(psnr_values) / len(psnr_values)
        payload["Heldout/Eval/PSNRMean"] = mean_psnr

        ssim_values = [
            float(value)
            for key, value in payload.items()
            if key.startswith("Heldout") and key.endswith("/Eval/SSIM")
        ]
        mean_ssim = None
        if ssim_values:
            mean_ssim = sum(ssim_values) / len(ssim_values)
            payload["Heldout/Eval/SSIMMean"] = mean_ssim

        if self.best_heldout_eval_psnr is None or mean_psnr > self.best_heldout_eval_psnr:
            self.best_heldout_eval_psnr = mean_psnr
            self.best_heldout_eval_ssim_at_best_psnr = mean_ssim
            self.best_heldout_eval_step = step

        payload["Heldout/BestEvalPSNR"] = self.best_heldout_eval_psnr
        if self.best_heldout_eval_ssim_at_best_psnr is not None:
            payload["Heldout/BestEvalSSIMAtBestPSNR"] = self.best_heldout_eval_ssim_at_best_psnr
        if self.best_heldout_eval_step is not None:
            payload["Heldout/BestEvalStep"] = self.best_heldout_eval_step

    def multicam_validation_payload_from_renders(
        self,
        *,
        train_rendered,
        heldout_rendered,
        decoded_metrics: dict[str, Any],
        step: int | None,
    ) -> dict[str, Any]:
        train_targets = [
            resize_images(self.multicam_bundle.train_frames[view], self.render_size).detach().cpu()
            for view in range(len(train_rendered))
        ]
        heldout_targets: list[torch.Tensor] = []
        if heldout_rendered and self.multicam_bundle.heldout_frames is not None:
            heldout_targets = [
                resize_images(self.multicam_bundle.heldout_frames[view], self.render_size).detach().cpu()
                for view in range(len(heldout_rendered))
            ]
        payload, self.gt_video_logged = multicam_validation_video_payload(
            self.cfg,
            train_rendered=train_rendered,
            heldout_rendered=heldout_rendered,
            train_targets=train_targets,
            heldout_targets=heldout_targets,
            heldout_camera_names=self.multicam_bundle.heldout_camera_names or [],
            decoded_metrics=decoded_metrics,
            camera_rig_metrics=self.camera_rig.metrics(),
            gt_video_logged=self.gt_video_logged,
            fps=self.sequence_data.video_fps,
        )
        self.update_best_heldout_payload(payload, step)
        return payload

    @torch.no_grad()
    def render_full_external_views(self):
        sequence_data = self.sequence_data
        frame_indices = torch.arange(0, sequence_data.frame_count, device=self.device)
        clip_frames, clip_times = prepare_clip(sequence_data, frame_indices)
        decoded = self._decode_clip(sequence_data, clip_frames, clip_times)
        train_rendered = []
        for view in range(self.multicam_bundle.train_view_count):
            train_rendered.append(
                self.render_view_clip(
                    decoded,
                    view=view,
                    clip_indices=frame_indices,
                    phase="eval",
                )
            )
            train_rendered[-1] = train_rendered[-1].detach_cpu()
        heldout_rendered = []
        if self.multicam_bundle.heldout_frames is not None:
            for view in range(self.multicam_bundle.heldout_view_count):
                heldout_rendered.append(
                    self.render_heldout_view_clip(
                        decoded,
                        view=view,
                        clip_indices=frame_indices,
                        phase="eval",
                    )
                )
                heldout_rendered[-1] = heldout_rendered[-1].detach_cpu()
        return train_rendered, heldout_rendered, decoded_temporal_payload_from_sequence(decoded)

    @torch.no_grad()
    def render_full_oracle_relative_views(self):
        source_view = 0
        sequence_data = self.multicam_bundle.train_sequences[source_view]
        frame_indices = torch.arange(0, sequence_data.frame_count, device=self.device)
        clip_frames, clip_times = prepare_clip(sequence_data, frame_indices)
        decoded = self._decode_clip(sequence_data, clip_frames, clip_times)
        source_name = self.multicam_bundle.train_camera_names[source_view]
        train_rendered = []
        relpose_memory_cache: dict[int, torch.Tensor] = {}
        for view in range(self.multicam_bundle.train_view_count):
            pair = CameraSwapPair(
                source_set="train",
                source_view=source_view,
                query_set="train",
                query_view=view,
                target_set="train",
                target_view=view,
                source_name=source_name,
                query_name=self.multicam_bundle.train_camera_names[view],
                target_name=self.multicam_bundle.train_camera_names[view],
            )
            residual, _residual_loss = self.relpose_residual_for_pair(
                pair,
                clip_indices=frame_indices,
                memory_cache=relpose_memory_cache,
            )
            train_rendered.append(
                self.render_camera_swap_pair(
                    decoded,
                    pair=pair,
                    clip_indices=frame_indices,
                    phase="eval",
                    residual=residual,
                ).detach_cpu()
            )
        heldout_rendered = []
        if self.multicam_bundle.heldout_frames is not None:
            heldout_names = self.multicam_bundle.heldout_camera_names or []
            for view in range(self.multicam_bundle.heldout_view_count):
                heldout_name = heldout_names[view] if view < len(heldout_names) else f"heldout_{view}"
                pair = CameraSwapPair(
                    source_set="train",
                    source_view=source_view,
                    query_set="heldout",
                    query_view=view,
                    target_set="heldout",
                    target_view=view,
                    source_name=source_name,
                    query_name=heldout_name,
                    target_name=heldout_name,
                )
                heldout_rendered.append(
                    self.render_camera_swap_pair(
                        decoded,
                        pair=pair,
                        clip_indices=frame_indices,
                        phase="eval",
                    ).detach_cpu()
                )
        return train_rendered, heldout_rendered, decoded_temporal_payload_from_sequence(decoded)

    def validation_video_payload(self, step: int | None = None) -> dict[str, Any]:
        if self.train_cfg["camera_swap_mode"] in {"oracle_relative", "learned_residual"}:
            train_rendered, heldout_rendered, decoded_metrics = self.render_full_oracle_relative_views()
        else:
            train_rendered, heldout_rendered, decoded_metrics = self.render_full_external_views()
        return self.multicam_validation_payload_from_renders(
            train_rendered=train_rendered,
            heldout_rendered=heldout_rendered,
            decoded_metrics=decoded_metrics,
            step=step,
        )

    def export_browser_bundle(self) -> None:
        if self.export_cfg["enabled"]:
            raise ValueError("Browser export is not wired for multicam external-camera training yet.")
        return None


def run_training(config: dict[str, Any]) -> None:
    MulticamPrecomputedFeatureImplicitTrainer(config).run()


__all__ = [
    "CAMERA_RIG_DEFAULTS",
    "DATA_MULTICAM_DEFAULTS",
    "TRAIN_MULTICAM_DEFAULTS",
    "MulticamPrecomputedFeatureImplicitTrainer",
    "run_training",
]
