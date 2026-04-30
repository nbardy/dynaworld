from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

from camera_rig import LearnableCameraRig
from config_utils import apply_defaults, load_config_file
from multicam_video_data import (
    cameras_from_K_w2c,
    heldout_cameras_from_K_w2c,
    load_multicam_video_bundle,
)
from objective import BackgroundSample, RenderedView, RunPhase
from pipeline.validation_media import (
    alpha_to_grayscale_video,
    multicam_validation_video_payload,
    render_diagnostics_payload,
)
from rendering import resize_images
from train_precomputed_feature_implicit_dynamic import PrecomputedFeatureImplicitTrainer
from train_video_token_implicit_dynamic import (
    StepResult,
    decoded_temporal_payload,
    eval_metric_payload,
    prepare_clip,
    select_window_indices,
    temporal_similarity_payload,
)


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
}


def _prefix_eval_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    payload = {}
    for key, value in metrics.items():
        if key.startswith("Eval/"):
            payload[f"{prefix}/{key}"] = value
        else:
            payload[f"{prefix}/{key}"] = value
    return payload


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
        return cfg

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        rig_params = [parameter for parameter in self.camera_rig.parameters() if parameter.requires_grad]
        if rig_params:
            self.optimizer.add_param_group(
                {
                    "params": rig_params,
                    "lr": self.train_cfg["camera_rig_lr"] or self.train_cfg["lr"],
                }
            )

    def load_train_sequences(self):
        self.multicam_bundle = load_multicam_video_bundle(
            data_cfg=self.data_cfg,
            camera_cfg=self.cfg["camera"],
            target_size=int(self.model_cfg["size"]),
            device=self.device,
        )
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
        view_count = self.multicam_bundle.train_view_count
        views_per_step = int(self.train_cfg["train_views_per_step"])
        if views_per_step <= 0 or views_per_step >= view_count:
            return list(range(view_count))
        return torch.randperm(view_count, device=self.device)[:views_per_step].detach().cpu().tolist()

    def sample_multicam_clip(self):
        sequence_data = self.sequence_data
        clip_indices = select_window_indices(
            sequence_data.frame_count,
            self.model_cfg["train_frame_count"],
            device=self.device,
        )
        clip_frames, clip_times = prepare_clip(sequence_data, clip_indices)
        return sequence_data, clip_indices, clip_frames, clip_times, self.sample_views()

    def _decode_clip(self, sequence_data, clip_frames: torch.Tensor, clip_times: torch.Tensor):
        model_input = self.model_input_for_clip(sequence_data, clip_frames, clip_times)
        return self.forward_clip(model_input, clip_times)

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

    def multicam_recon_loss(
        self,
        decoded,
        *,
        clip_indices: torch.Tensor,
        views: list[int],
        phase: RunPhase,
        keep_preview: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        recon_loss = self.multicam_bundle.train_frames.new_zeros(())
        preview_render = None
        preview_features = None
        background = self.rgb_objective.sample_background(
            phase=phase,
            like=self.multicam_bundle.train_frames[int(views[0]), clip_indices],
            frame_count=len(clip_indices),
        )
        for view in views:
            rendered = self.render_view_clip(
                decoded,
                view=int(view),
                clip_indices=clip_indices,
                phase=phase,
                background=background,
            )
            if phase == "train" and self.feature_dim != 3 and rendered.alpha is None:
                raise ValueError(
                    "F-channel multicam training requires alpha-aware render output so random-background "
                    "composition is active. Got alpha=None; check renderer='fast_mac' and v5_features build."
                )
            recon_loss = recon_loss + self.rgb_objective.reconstruction_loss(rendered)
            if keep_preview and preview_render is None:
                preview_render = rendered.rgb[0].detach()
                if self.feature_pca_log:
                    preview_features = rendered.features[0].detach()
        return recon_loss / float(max(len(views), 1)), preview_render, preview_features

    def rig_regularization_loss(self) -> torch.Tensor:
        return float(self.cfg["camera"]["rig_regularization_weight"]) * self.camera_rig.regularization_loss()

    @torch.no_grad()
    def initial_step_result(self) -> StepResult:
        was_training = self.model.training
        self.model.eval()
        try:
            sequence_data = self.sequence_data
            clip_length = int(self.model_cfg["train_frame_count"])
            clip_indices = torch.arange(0, clip_length, device=self.device)
            clip_frames, clip_times = prepare_clip(sequence_data, clip_indices)
            decoded = self._decode_clip(sequence_data, clip_frames, clip_times)
            bank_rate_loss, bank_rate_terms = self.build_bank_rate_loss(decoded)
            rig_loss = self.rig_regularization_loss()
            recon_loss, preview_render, preview_features = self.multicam_recon_loss(
                decoded,
                clip_indices=clip_indices,
                views=list(range(self.multicam_bundle.train_view_count)),
                phase="eval",
                keep_preview=True,
            )
            zero = clip_frames.new_zeros(())
            loss = recon_loss + bank_rate_loss + rig_loss
            return StepResult(
                source_path=sequence_data.source_path,
                sequence_frame_count=sequence_data.frame_count,
                clip_frames=clip_frames,
                preview_render=preview_render,
                preview_features=preview_features,
                camera_state=decoded.camera_state,
                loss=loss.detach(),
                recon_loss=recon_loss.detach(),
                camera_motion_loss=zero,
                camera_temporal_loss=zero,
                camera_global_loss=zero,
                bank_rate_loss=(bank_rate_loss + rig_loss).detach(),
                bank_rate_terms={key: value.detach() for key, value in bank_rate_terms.items()},
            )
        finally:
            if was_training:
                self.model.train()

    def step(self, keep_preview: bool = False) -> StepResult:
        self.optimizer.zero_grad(set_to_none=True)
        sequence_data, clip_indices, clip_frames, clip_times, views = self.sample_multicam_clip()
        decoded = self._decode_clip(sequence_data, clip_frames, clip_times)
        bank_rate_loss, bank_rate_terms = self.build_bank_rate_loss(decoded)
        rig_loss = self.rig_regularization_loss()
        recon_loss, preview_render, preview_features = self.multicam_recon_loss(
            decoded,
            clip_indices=clip_indices,
            views=views,
            phase="train",
            keep_preview=keep_preview,
        )
        loss = recon_loss + bank_rate_loss + rig_loss
        loss.backward()
        self.optimizer.step()
        zero = clip_frames.new_zeros(())
        return StepResult(
            source_path=sequence_data.source_path,
            sequence_frame_count=sequence_data.frame_count,
            clip_frames=clip_frames,
            preview_render=preview_render,
            preview_features=preview_features,
            camera_state=decoded.camera_state,
            loss=loss.detach(),
            recon_loss=recon_loss.detach(),
            camera_motion_loss=zero,
            camera_temporal_loss=zero,
            camera_global_loss=zero,
            bank_rate_loss=(bank_rate_loss + rig_loss).detach(),
            bank_rate_terms={key: value.detach() for key, value in bank_rate_terms.items()},
        )

    def scalar_payload(self, result: StepResult) -> dict[str, Any]:
        payload = super().scalar_payload(result)
        payload.update(self.camera_rig.metrics())
        payload["TrainViewCount"] = self.multicam_bundle.train_view_count
        payload["TrainViewsPerStep"] = int(self.train_cfg["train_views_per_step"])
        payload["Rig/RegularizationWeight"] = float(self.cfg["camera"]["rig_regularization_weight"])
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
        return train_rendered, heldout_rendered, decoded_temporal_payload(
            {
                "xyz": [decoded.xyz[index].detach().cpu() for index in range(decoded.frame_count)],
                "scales": [decoded.scales[index].detach().cpu() for index in range(decoded.frame_count)],
                "opacities": [decoded.opacities[index].detach().cpu() for index in range(decoded.frame_count)],
                "rgbs": [decoded.rgbs[index].detach().cpu() for index in range(decoded.frame_count)],
            }
        )

    def add_render_diagnostics(
        self,
        payload: dict[str, Any],
        *,
        prefix: str,
        rendered: RenderedView,
        target: torch.Tensor,
    ) -> None:
        payload.update(
            render_diagnostics_payload(
                self.cfg,
                prefix=prefix,
                target=target,
                pred=rendered.rgb,
                alpha=rendered.alpha,
                features=rendered.features if self.feature_pca_log else None,
                fps=self.sequence_data.video_fps,
            )
        )

    def validation_video_payload(self) -> dict[str, Any]:
        train_rendered, heldout_rendered, decoded_metrics = self.render_full_external_views()
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
            eval_metric_fn=eval_metric_payload,
            temporal_metric_fn=temporal_similarity_payload,
            prefix_eval_metrics_fn=_prefix_eval_metrics,
            camera_rig_metrics=self.camera_rig.metrics(),
            gt_video_logged=self.gt_video_logged,
            fps=self.sequence_data.video_fps,
        )
        return payload

    def export_browser_bundle(self) -> None:
        if self.export_cfg["enabled"]:
            raise ValueError("Browser export is not wired for multicam external-camera training yet.")
        return None


def run_training(config: dict[str, Any]) -> None:
    MulticamPrecomputedFeatureImplicitTrainer(config).run()


def main(config: dict[str, Any] | str | Path) -> None:
    if isinstance(config, (str, Path)):
        run_training(load_config_file(config))
    else:
        run_training(config)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: uv run python src/train/train_multicam_precomputed_feature_implicit_dynamic.py "
            "src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_static_dynamic_96_32_precomputed_vjepa2_1_vitb_384_128_16f_8192splats.jsonc"
        )
    main(sys.argv[1])
