from __future__ import annotations

import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import wandb

from camera_swap_sampling import (
    CameraSwapPair,
    build_heldout_camera_swap_pairs,
)
from checkpoint_utils import atomic_torch_save
from config_utils import apply_defaults, load_config_file, path_or_none, serialize_config_value
from objective.types import BackgroundSample, RenderedView, RunPhase
from pipeline.diagnostics import decoded_temporal_payload
from pipeline.losses import build_bank_rate_loss as _build_bank_rate_loss_impl
from pipeline.render import prepare_clip
from pipeline.validation_media import multicam_validation_video_payload
from relative_pose import (
    cameras_with_se3_transform,
    se3_cycle_loss,
    se3_residual_identity_loss,
    se3_residual_matrix,
    se3_transform_l2_loss,
)
from rendering import build_or_reuse_grid, resize_images
from runtime_types import SequenceData
from train_multicam_precomputed_feature_implicit_dynamic import MulticamPrecomputedFeatureImplicitTrainer
from train_video_token_implicit_dynamic import (
    _render_preview_image_impl,
    pick_renderer_mode_from_config,
)


RELATIVE_POSE_TRAIN_DEFAULTS = {
    "relpose_output_mode": "full",
    "heldout_eval_camera_mode": "predicted_relpose",
    "trainable_scope": "all",
    "relpose_feature_frame_mode": "first_frame",
    "relpose_pose_loss_weight": 1.0,
    "multires_render_sizes": None,
    "multires_render_probabilities": None,
    "multires_token_detail_levels": None,
    "checkpoint_load_path": None,
    "checkpoint_save_path": None,
}


@dataclass(frozen=True)
class FullRelativePosePrediction:
    rotation: torch.Tensor
    translation: torch.Tensor
    camera_to_world: torch.Tensor
    camera_to_world_per_frame: torch.Tensor
    oracle_camera_to_world: torch.Tensor
    target_template_cameras: tuple[Any, ...]


def normalize_multires_render_sizes(value: Any) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError("train.multires_render_sizes must be null or a list of positive integer render sizes.")
    if not value:
        raise ValueError("train.multires_render_sizes cannot be empty when provided.")
    sizes: list[int] = []
    for item in value:
        if isinstance(item, bool):
            raise ValueError("train.multires_render_sizes entries must be positive integers, not booleans.")
        size = int(item)
        if size < 1:
            raise ValueError(f"train.multires_render_sizes entries must be >= 1, got {item!r}.")
        sizes.append(size)
    if len(set(sizes)) != len(sizes):
        raise ValueError(f"train.multires_render_sizes must not contain duplicates, got {sizes!r}.")
    return sizes


def normalize_multires_render_probabilities(value: Any, *, render_sizes: list[int] | None) -> list[float] | None:
    if value is None:
        return None
    if render_sizes is None:
        raise ValueError("train.multires_render_probabilities requires train.multires_render_sizes.")
    if not isinstance(value, (list, tuple)):
        raise ValueError("train.multires_render_probabilities must be null or a list of non-negative weights.")
    if len(value) != len(render_sizes):
        raise ValueError(
            "train.multires_render_probabilities must have the same length as train.multires_render_sizes "
            f"({len(render_sizes)}), got {len(value)}."
        )
    weights: list[float] = []
    for item in value:
        if isinstance(item, bool):
            raise ValueError("train.multires_render_probabilities entries must be numeric weights, not booleans.")
        weight = float(item)
        if weight < 0.0:
            raise ValueError(f"train.multires_render_probabilities entries must be >= 0, got {item!r}.")
        weights.append(weight)
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("train.multires_render_probabilities must contain at least one positive weight.")
    return [weight / total for weight in weights]


def normalize_multires_token_detail_levels(value: Any, *, render_sizes: list[int] | None) -> list[int] | None:
    if value is None:
        return None
    if render_sizes is None:
        raise ValueError("train.multires_token_detail_levels requires train.multires_render_sizes.")
    if not isinstance(value, (list, tuple)):
        raise ValueError("train.multires_token_detail_levels must be null or a list of non-negative integers.")
    if len(value) != len(render_sizes):
        raise ValueError(
            "train.multires_token_detail_levels must have the same length as train.multires_render_sizes "
            f"({len(render_sizes)}), got {len(value)}."
        )
    levels: list[int] = []
    for item in value:
        if isinstance(item, bool):
            raise ValueError("train.multires_token_detail_levels entries must be integers, not booleans.")
        level = int(item)
        if level < 0:
            raise ValueError(f"train.multires_token_detail_levels entries must be >= 0, got {item!r}.")
        levels.append(level)
    return levels


def model_token_layout_detail_levels(model_cfg: dict[str, Any]) -> int:
    token_layout = model_cfg.get("token_layout")
    if token_layout is None:
        return 0
    return max(
        len(token_layout.get("static_detail_tokens") or []),
        len(token_layout.get("dynamic_detail_tokens") or []),
    )


def first_frame_repeated_sequence(sequence_data: SequenceData, *, repeat_count: int, cache_tag: str) -> SequenceData:
    if int(repeat_count) < 1:
        raise ValueError(f"repeat_count must be >= 1, got {repeat_count}.")
    if sequence_data.frame_count < 1:
        raise ValueError("Cannot build first-frame relpose features from an empty sequence.")
    frames = sequence_data.frames[:1].expand(int(repeat_count), -1, -1, -1).contiguous()
    frame_times = sequence_data.frame_times[:1].expand(int(repeat_count), *sequence_data.frame_times.shape[1:]).contiguous()
    cameras = None
    if sequence_data.cameras is not None:
        cameras = tuple(sequence_data.cameras[0] for _ in range(int(repeat_count)))
    records = ()
    if sequence_data.records:
        records = tuple(sequence_data.records[0] for _ in range(int(repeat_count)))
    source_prefix = str(sequence_data.source_path or "unknown_source")
    return SequenceData(
        frames=frames,
        frame_times=frame_times,
        video_fps=sequence_data.video_fps,
        frame_source=sequence_data.frame_source,
        frame_paths=(),
        cameras=cameras,
        records=records,
        intrinsics_summary=sequence_data.intrinsics_summary,
        source_path=Path(f"{source_prefix}#relpose_first_frame_{cache_tag}_{int(repeat_count)}f"),
        selected_frame_count=int(repeat_count),
        all_frame_count=sequence_data.all_frame_count,
    )


class MulticamRelativePoseImplicitTrainer(MulticamPrecomputedFeatureImplicitTrainer):
    """No-VGGT multicam trainer that predicts a full source-relative query pose.

    The source world decoder only receives the source camera's features. The
    relative-pose head receives source and query-camera feature memories and
    predicts the query camera-to-world transform in the source camera's local
    world. Heldout cameras are excluded from training pairs but can be consumed
    as query features during evaluation.
    """

    @classmethod
    def resolve_config(cls, config: dict[str, Any]) -> dict[str, Any]:
        cfg = super().resolve_config(config)
        apply_defaults(cfg["train"], RELATIVE_POSE_TRAIN_DEFAULTS)
        if cfg["train"]["camera_swap_mode"] != "learned_residual":
            raise ValueError(
                "multicam_relative_pose_implicit_camera requires train.camera_swap_mode='learned_residual' "
                "so the inherited train loop uses camera-swap pairs."
            )
        cfg["train"]["relpose_output_mode"] = str(cfg["train"]["relpose_output_mode"]).lower()
        if cfg["train"]["relpose_output_mode"] != "full":
            raise ValueError("train.relpose_output_mode must be 'full' for multicam_relative_pose_implicit_camera.")
        cfg["train"]["heldout_eval_camera_mode"] = str(cfg["train"]["heldout_eval_camera_mode"]).lower()
        if cfg["train"]["heldout_eval_camera_mode"] not in {"predicted_relpose", "calibrated"}:
            raise ValueError("train.heldout_eval_camera_mode must be one of: predicted_relpose, calibrated.")
        cfg["train"]["trainable_scope"] = str(cfg["train"]["trainable_scope"]).lower()
        if cfg["train"]["trainable_scope"] not in {"all", "relpose_only"}:
            raise ValueError("train.trainable_scope must be one of: all, relpose_only.")
        cfg["train"]["relpose_feature_frame_mode"] = str(cfg["train"]["relpose_feature_frame_mode"]).lower()
        if cfg["train"]["relpose_feature_frame_mode"] not in {"first_frame", "clip"}:
            raise ValueError("train.relpose_feature_frame_mode must be one of: first_frame, clip.")
        cfg["train"]["relpose_pose_loss_weight"] = float(cfg["train"]["relpose_pose_loss_weight"])
        cfg["train"]["multires_render_sizes"] = normalize_multires_render_sizes(cfg["train"]["multires_render_sizes"])
        cfg["train"]["multires_render_probabilities"] = normalize_multires_render_probabilities(
            cfg["train"]["multires_render_probabilities"],
            render_sizes=cfg["train"]["multires_render_sizes"],
        )
        cfg["train"]["multires_token_detail_levels"] = normalize_multires_token_detail_levels(
            cfg["train"]["multires_token_detail_levels"],
            render_sizes=cfg["train"]["multires_render_sizes"],
        )
        if cfg["train"]["multires_token_detail_levels"] is not None:
            if cfg["model"].get("token_layout") is None:
                raise ValueError("train.multires_token_detail_levels requires model.token_layout.")
            max_detail_level = model_token_layout_detail_levels(cfg["model"])
            invalid = [level for level in cfg["train"]["multires_token_detail_levels"] if level > max_detail_level]
            if invalid:
                raise ValueError(
                    "train.multires_token_detail_levels cannot exceed model.token_layout detail levels "
                    f"({max_detail_level}), got {invalid!r}."
                )
        cfg["train"]["checkpoint_load_path"] = path_or_none(cfg["train"]["checkpoint_load_path"])
        cfg["train"]["checkpoint_save_path"] = path_or_none(cfg["train"]["checkpoint_save_path"])
        if cfg["train"]["trainable_scope"] == "relpose_only":
            if not cfg["train"]["camera_swap_include_cross"]:
                raise ValueError("train.trainable_scope='relpose_only' requires camera_swap_include_cross=true.")
            if cfg["train"]["checkpoint_load_path"] is None:
                raise ValueError("train.trainable_scope='relpose_only' requires train.checkpoint_load_path.")
        return cfg

    def __init__(self, config: dict[str, Any]) -> None:
        self.relpose_first_frame_sequences: dict[tuple[str, int], SequenceData] = {}
        super().__init__(config)
        self.base_render_size = int(self.render_cfg["render_size"])
        self.multires_render_sizes = tuple(self.train_cfg["multires_render_sizes"] or ())
        self.multires_render_probabilities = tuple(self.train_cfg["multires_render_probabilities"] or ())
        self.multires_render_probability_by_size = (
            dict(zip(self.multires_render_sizes, self.multires_render_probabilities, strict=True))
            if self.multires_render_probabilities
            else {}
        )
        self._multires_render_probability_tensor = (
            torch.tensor(self.multires_render_probabilities, dtype=torch.float32, device=self.device)
            if self.multires_render_probabilities
            else None
        )
        self.multires_token_detail_levels = tuple(self.train_cfg["multires_token_detail_levels"] or ())
        self.multires_token_detail_by_size = (
            dict(zip(self.multires_render_sizes, self.multires_token_detail_levels, strict=True))
            if self.multires_token_detail_levels
            else {}
        )
        self._dense_grid_by_render_size = {int(self.render_size): self.dense_grid}
        self.last_train_render_size = int(self.render_size)
        self.last_train_token_detail_level = self.current_token_detail_level()
        if self.relpose_head is None:
            raise ValueError("MulticamRelativePoseImplicitTrainer requires a constructed relpose_head.")
        if self.multires_render_sizes:
            print(
                "Multires render schedule enabled: "
                f"train sizes={list(self.multires_render_sizes)}, "
                f"probabilities={list(self.multires_render_probabilities) or 'uniform'}, "
                f"validation size={self.base_render_size}"
            )
        if self.multires_token_detail_by_size:
            print(f"Multires token-detail schedule enabled: {self.multires_token_detail_by_size}")
        self._load_checkpoint_if_configured()
        self._configure_trainable_scope()

    def load_eval_sequences(self):
        return list(self.multicam_bundle.heldout_sequences)

    def on_sequences_loaded(self) -> None:
        release_after_prebake = bool(self.cfg["features"]["release_extractor_after_prebake"])
        if self.train_cfg["relpose_feature_frame_mode"] == "first_frame":
            self.cfg["features"]["release_extractor_after_prebake"] = False
        super().on_sequences_loaded()
        if self.train_cfg["relpose_feature_frame_mode"] != "first_frame":
            return
        self._prepare_relpose_first_frame_sequences()
        if self.model_cfg["video_encoder_backend"] != "none":
            self.feature_cache.prebake(self.relpose_first_frame_sequences.values())
            if release_after_prebake:
                if self.feature_cache.force_rebake:
                    print("[features] disabling force_rebake after relpose first-frame prebake before releasing extractor")
                    self.feature_cache.force_rebake = False
                self.feature_cache.release_extractor()
        self.cfg["features"]["release_extractor_after_prebake"] = release_after_prebake

    def _prepare_relpose_first_frame_sequences(self) -> None:
        repeat_count = int(self.model_cfg["train_frame_count"])
        sequences: dict[tuple[str, int], SequenceData] = {}
        for view, sequence_data in enumerate(self.multicam_bundle.train_sequences):
            sequences[("train", int(view))] = first_frame_repeated_sequence(
                sequence_data,
                repeat_count=repeat_count,
                cache_tag=f"train_{int(view)}",
            )
        for view, sequence_data in enumerate(self.multicam_bundle.heldout_sequences):
            sequences[("heldout", int(view))] = first_frame_repeated_sequence(
                sequence_data,
                repeat_count=repeat_count,
                cache_tag=f"heldout_{int(view)}",
            )
        self.relpose_first_frame_sequences = sequences

    def _load_checkpoint_if_configured(self) -> None:
        checkpoint_path = self.train_cfg["checkpoint_load_path"]
        if checkpoint_path is None:
            return
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing relative-pose checkpoint: {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location=self.device)
        if not isinstance(payload, dict):
            raise TypeError(f"Expected checkpoint dict at {checkpoint_path}, got {type(payload).__name__}.")
        if "model" not in payload or "relpose_head" not in payload:
            raise KeyError("Relative-pose checkpoint must contain 'model' and 'relpose_head' state dicts.")
        self.model.load_state_dict(payload["model"])
        relpose_load = self.relpose_head.load_state_dict(payload["relpose_head"], strict=False)
        allowed_missing = {"pair_delta_norm.weight", "pair_delta_norm.bias", "pair_delta_output.weight"}
        missing = set(relpose_load.missing_keys)
        unexpected = set(relpose_load.unexpected_keys)
        if unexpected:
            raise KeyError(f"Unexpected relative-pose checkpoint keys: {sorted(unexpected)}")
        if missing - allowed_missing:
            raise KeyError(f"Missing relative-pose checkpoint keys: {sorted(missing)}")
        if missing:
            print(f"Initialized new relative-pose pair-delta keys not present in checkpoint: {sorted(missing)}")
        if self.colorize is not None and "colorizer" in payload:
            self.colorize.load_state_dict(payload["colorizer"])
        if hasattr(self, "camera_rig") and "camera_rig" in payload:
            self.camera_rig.load_state_dict(payload["camera_rig"])
        print(f"Loaded relative-pose checkpoint: {checkpoint_path}")

    def _configure_trainable_scope(self) -> None:
        if self.train_cfg["trainable_scope"] == "all":
            return
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        if self.colorize is not None:
            for parameter in self.colorize.parameters():
                parameter.requires_grad_(False)
        for parameter in self.camera_rig.parameters():
            parameter.requires_grad_(False)
        for parameter in self.relpose_head.parameters():
            parameter.requires_grad_(True)
        self.optimizer = torch.optim.Adam(
            [parameter for parameter in self.relpose_head.parameters() if parameter.requires_grad],
            lr=self.train_cfg["lr"],
            fused=self.device.type in {"cuda", "mps"},
        )
        print("Trainable scope: relpose_only (model/colorizer/camera_rig frozen).")

    def checkpoint_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model.state_dict(),
            "relpose_head": self.relpose_head.state_dict(),
            "camera_rig": self.camera_rig.state_dict(),
            "config": serialize_config_value(self.cfg),
            "step": int(self.train_cfg["steps"]),
        }
        if self.colorize is not None:
            payload["colorizer"] = self.colorize.state_dict()
        return payload

    def save_checkpoint_if_configured(self) -> None:
        checkpoint_path = self.train_cfg["checkpoint_save_path"]
        if checkpoint_path is None:
            return
        atomic_torch_save(self.checkpoint_payload(), checkpoint_path)
        print(f"Saved relative-pose checkpoint: {checkpoint_path}")

    def run(self) -> None:
        super().run()
        self.save_checkpoint_if_configured()

    def _dense_grid_for_render_size(self, render_size: int) -> torch.Tensor:
        size = int(render_size)
        if not hasattr(self, "_dense_grid_by_render_size"):
            self._dense_grid_by_render_size = {}
        if size not in self._dense_grid_by_render_size:
            self._dense_grid_by_render_size[size] = build_or_reuse_grid(size, size, self.device)
        return self._dense_grid_by_render_size[size]

    def _activate_render_size(self, render_size: int) -> None:
        size = int(render_size)
        self.cfg["render"]["render_size"] = size
        self.render_size = size
        self.dense_grid = self._dense_grid_for_render_size(size)
        self.renderer_mode, self.effective_gaussians = pick_renderer_mode_from_config(
            self.cfg,
            active_detail_level=self.current_token_detail_level(),
        )

    @contextmanager
    def temporary_render_size(self, render_size: int):
        previous_size = int(self.render_size)
        previous_grid = self.dense_grid
        previous_renderer_mode = self.renderer_mode
        previous_effective_gaussians = self.effective_gaussians
        self._activate_render_size(int(render_size))
        try:
            yield
        finally:
            self.cfg["render"]["render_size"] = previous_size
            self.render_size = previous_size
            self.dense_grid = previous_grid
            self.renderer_mode = previous_renderer_mode
            self.effective_gaussians = previous_effective_gaussians

    def sample_train_render_size(self) -> int:
        if not self.multires_render_sizes:
            return self.base_render_size
        if self._multires_render_probability_tensor is None:
            index = int(torch.randint(len(self.multires_render_sizes), (1,), device=self.device).item())
        else:
            index = int(torch.multinomial(self._multires_render_probability_tensor, 1).item())
        return int(self.multires_render_sizes[index])

    def current_token_detail_level(self) -> int | None:
        return getattr(self.model, "active_detail_level", None) if hasattr(self, "model") else None

    @contextmanager
    def temporary_token_detail_level(self, active_detail_level: int | None):
        if active_detail_level is None:
            yield
            return
        if not hasattr(self.model, "set_active_detail_level"):
            raise ValueError("train.multires_token_detail_levels requires a model with set_active_detail_level().")
        previous_level = self.current_token_detail_level()
        previous_renderer_mode = self.renderer_mode
        previous_effective_gaussians = self.effective_gaussians
        self.model.set_active_detail_level(int(active_detail_level))
        self.renderer_mode, self.effective_gaussians = pick_renderer_mode_from_config(
            self.cfg,
            active_detail_level=int(active_detail_level),
        )
        try:
            yield
        finally:
            self.model.set_active_detail_level(previous_level)
            self.renderer_mode = previous_renderer_mode
            self.effective_gaussians = previous_effective_gaussians

    def sample_train_token_detail_level(self, render_size: int) -> int | None:
        return self.multires_token_detail_by_size.get(int(render_size))

    def step(self, keep_preview: bool = False):
        render_size = self.sample_train_render_size()
        token_detail_level = self.sample_train_token_detail_level(render_size)
        self.last_train_render_size = render_size
        self.last_train_token_detail_level = token_detail_level
        with self.temporary_render_size(render_size), self.temporary_token_detail_level(token_detail_level):
            result = super().step(keep_preview=keep_preview)
        setattr(result, "render_size", render_size)
        setattr(result, "token_detail_level", token_detail_level)
        if hasattr(self.model, "active_decoded_token_count") and token_detail_level is not None:
            previous_level = self.current_token_detail_level()
            self.model.set_active_detail_level(token_detail_level)
            try:
                setattr(result, "active_decoded_tokens", self.model.active_decoded_token_count())
            finally:
                self.model.set_active_detail_level(previous_level)
        return result

    def sequence_for_camera_set(self, camera_set: str, view: int):
        key = (str(camera_set), int(view))
        if self.train_cfg["relpose_feature_frame_mode"] == "first_frame" and key in self.relpose_first_frame_sequences:
            return self.relpose_first_frame_sequences[key]
        if camera_set == "train":
            return self.multicam_bundle.train_sequences[int(view)]
        if camera_set == "heldout":
            return self.multicam_bundle.heldout_sequences[int(view)]
        raise ValueError(f"Unsupported camera set {camera_set!r}.")

    def projected_feature_memory_for_relpose_view(
        self,
        camera_set: str,
        view: int,
    ) -> torch.Tensor:
        sequence_data = self.sequence_for_camera_set(camera_set, view)
        clip_indices = torch.arange(0, sequence_data.frame_count, device=self.device)
        clip_frames, clip_times = prepare_clip(sequence_data, clip_indices)
        model_input = self.model_input_for_clip(sequence_data, clip_frames, clip_times)
        with self.autocast_context():
            return self.model.video_encoder(model_input, frame_times=clip_times)

    def full_relative_pose_for_pair(
        self,
        pair: CameraSwapPair,
        *,
        clip_indices: torch.Tensor,
        memory_cache: dict[tuple[str, int], torch.Tensor],
    ) -> FullRelativePosePrediction:
        if pair.source_set != "train":
            raise ValueError("Full relative-pose source worlds must come from train cameras.")
        source_key = ("train", int(pair.source_view))
        query_key = (pair.query_set, int(pair.query_view))
        if source_key not in memory_cache:
            with self.profile_section("relpose/feature_memory"):
                memory_cache[source_key] = self.projected_feature_memory_for_relpose_view("train", int(pair.source_view))
        if query_key not in memory_cache:
            with self.profile_section("relpose/feature_memory"):
                memory_cache[query_key] = self.projected_feature_memory_for_relpose_view(pair.query_set, int(pair.query_view))
        with self.profile_section("relpose/head"):
            rotation, translation = self.relpose_head(memory_cache[source_key], memory_cache[query_key])
        with self.profile_section("relpose/camera_math"):
            camera_to_world = se3_residual_matrix(rotation, translation)
            target_template_cameras = self.source_relative_cameras_for_pair(pair, clip_indices)
            oracle_camera_to_world = torch.stack(
                [
                    camera.camera_to_world.to(device=camera_to_world.device, dtype=camera_to_world.dtype)
                    for camera in target_template_cameras
                ],
                dim=0,
            )
            camera_to_world_per_frame = camera_to_world.expand(len(target_template_cameras), -1, -1)
        return FullRelativePosePrediction(
            rotation=rotation,
            translation=translation,
            camera_to_world=camera_to_world,
            camera_to_world_per_frame=camera_to_world_per_frame,
            oracle_camera_to_world=oracle_camera_to_world,
            target_template_cameras=target_template_cameras,
        )

    def relpose_cycle_loss_for_prediction(
        self,
        pair: CameraSwapPair,
        prediction: FullRelativePosePrediction,
        *,
        clip_indices: torch.Tensor,
        memory_cache: dict[tuple[str, int], torch.Tensor],
    ) -> torch.Tensor | None:
        if not pair.is_train_cross_view:
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
        reverse_prediction = self.full_relative_pose_for_pair(
            reverse_pair,
            clip_indices=clip_indices,
            memory_cache=memory_cache,
        )
        return se3_cycle_loss(prediction.camera_to_world, reverse_prediction.camera_to_world)

    def render_camera_swap_pair_with_cameras(
        self,
        decoded,
        *,
        pair: CameraSwapPair,
        cameras: tuple[Any, ...],
        clip_indices: torch.Tensor,
        phase: RunPhase,
        background: BackgroundSample | None = None,
        log_media: bool = False,
        camera_owner: str = "model",
    ) -> RenderedView:
        target_view = int(pair.target_view)
        if pair.target_set == "train":
            target_name = self.multicam_bundle.train_camera_names[target_view]
            target_frames = self.multicam_bundle.train_frames[target_view, clip_indices]
            role = "train"
            camera_role = "target"
            metrics_prefix = f"PredRelPoseSrc{pair.source_view}_Tgt{target_view}_{target_name}"
        elif pair.target_set == "heldout" and self.multicam_bundle.heldout_frames is not None:
            heldout_names = self.multicam_bundle.heldout_camera_names or []
            target_name = heldout_names[target_view] if target_view < len(heldout_names) else f"heldout_{target_view}"
            target_frames = self.multicam_bundle.heldout_frames[target_view, clip_indices]
            role = "heldout"
            camera_role = "heldout"
            metrics_prefix = f"PredRelPoseHeldout{target_view}_{target_name}"
        else:
            raise ValueError(f"Unsupported camera-swap target set {pair.target_set!r}.")
        source_name = self.multicam_bundle.train_camera_names[int(pair.source_view)]
        target = self.make_target_view(
            view_id=f"relpose_source_{pair.source_view}_target_{target_view}",
            frames=target_frames,
            frame_indices=clip_indices,
            frame_times=self.frame_times_for_indices(clip_indices),
            cameras=cameras,
            role=role,
            camera_role=camera_role,
            camera_owner=camera_owner,
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
        relpose_pose_loss = self.multicam_bundle.train_frames.new_zeros(())
        relpose_pose_count = 0
        relpose_identity_loss = self.multicam_bundle.train_frames.new_zeros(())
        relpose_identity_count = 0
        relpose_cycle_loss = self.multicam_bundle.train_frames.new_zeros(())
        relpose_cycle_count = 0
        relpose_memory_cache: dict[tuple[str, int], torch.Tensor] = {}

        pairs_by_source: dict[int, list[CameraSwapPair]] = {}
        for pair in pairs:
            pairs_by_source.setdefault(int(pair.source_view), []).append(pair)

        rendered_count = 0
        for source_index, source_pairs in pairs_by_source.items():
            with self.profile_section("camera_swap/source_decode"):
                sequence_data, clip_frames, _clip_times, decoded = self._decode_source_view(source_index, clip_indices)
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
                    prediction = self.full_relative_pose_for_pair(
                        pair,
                        clip_indices=clip_indices,
                        memory_cache=relpose_memory_cache,
                    )
                with self.profile_section("camera_swap/relpose_loss"):
                    pose_loss = se3_transform_l2_loss(
                        prediction.camera_to_world_per_frame,
                        prediction.oracle_camera_to_world,
                    )
                relpose_pose_loss = relpose_pose_loss + pose_loss
                relpose_pose_count += 1
                if pair.is_self_reconstruction:
                    relpose_identity_loss = relpose_identity_loss + se3_residual_identity_loss(
                        prediction.rotation,
                        prediction.translation,
                    )
                    relpose_identity_count += 1
                with self.profile_section("camera_swap/relpose_cycle"):
                    cycle_loss = self.relpose_cycle_loss_for_prediction(
                        pair,
                        prediction,
                        clip_indices=clip_indices,
                        memory_cache=relpose_memory_cache,
                    )
                if cycle_loss is not None:
                    relpose_cycle_loss = relpose_cycle_loss + cycle_loss
                    relpose_cycle_count += 1
                with self.profile_section("camera_swap/camera_apply"):
                    cameras = cameras_with_se3_transform(
                        prediction.target_template_cameras,
                        prediction.camera_to_world,
                    )
                with self.profile_section("render_view_total"):
                    rendered = self.render_camera_swap_pair_with_cameras(
                        decoded,
                        pair=pair,
                        cameras=cameras,
                        clip_indices=clip_indices,
                        phase=phase,
                        background=background,
                        log_media=keep_preview and preview_render is None and pair_index == 0,
                    )
                if phase == "train" and self.feature_dim != 3 and rendered.alpha is None:
                    raise ValueError(
                        "F-channel camera-swap training requires alpha-aware render output so random-background "
                        "composition is active. Got alpha=None; check renderer='fast_mac' and v5_features build."
                    )
                with self.profile_section("recon_loss"):
                    recon_loss = recon_loss + self.rgb_objective.reconstruction_loss(rendered)
                rendered_count += 1
                if keep_preview and preview_render is None:
                    preview_render = rendered.rgb[0].detach()
                    if self.feature_pca_log:
                        preview_features = rendered.features[0].detach()

        source_count = float(max(len(pairs_by_source), 1))
        bank_rate_terms = {key: value / source_count for key, value in bank_rate_terms_accum.items()}
        bank_rate_loss = bank_rate_loss / source_count
        if relpose_pose_count:
            relpose_pose_loss = relpose_pose_loss / float(relpose_pose_count)
            relpose_pose_weighted = relpose_pose_loss * float(self.train_cfg["relpose_pose_loss_weight"])
            bank_rate_loss = bank_rate_loss + relpose_pose_weighted
            bank_rate_terms["relpose_full_pose_loss"] = relpose_pose_loss
            bank_rate_terms["relpose_full_pose_loss_weighted"] = relpose_pose_weighted
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

    @torch.no_grad()
    def render_full_predicted_relative_views(self):
        source_view = 0
        sequence_data = self.multicam_bundle.train_sequences[source_view]
        frame_indices = torch.arange(0, sequence_data.frame_count, device=self.device)
        clip_frames, clip_times = prepare_clip(sequence_data, frame_indices)
        decoded = self._decode_clip(sequence_data, clip_frames, clip_times)
        source_name = self.multicam_bundle.train_camera_names[source_view]
        relpose_memory_cache: dict[tuple[str, int], torch.Tensor] = {}

        train_rendered = []
        for view in range(self.multicam_bundle.train_view_count):
            target_name = self.multicam_bundle.train_camera_names[view]
            pair = CameraSwapPair(
                source_set="train",
                source_view=source_view,
                query_set="train",
                query_view=view,
                target_set="train",
                target_view=view,
                source_name=source_name,
                query_name=target_name,
                target_name=target_name,
            )
            prediction = self.full_relative_pose_for_pair(
                pair,
                clip_indices=frame_indices,
                memory_cache=relpose_memory_cache,
            )
            train_rendered.append(
                self.render_camera_swap_pair_with_cameras(
                    decoded,
                    pair=pair,
                    cameras=cameras_with_se3_transform(
                        prediction.target_template_cameras,
                        prediction.camera_to_world,
                    ),
                    clip_indices=frame_indices,
                    phase="eval",
                ).detach_cpu()
            )

        heldout_rendered = []
        if self.multicam_bundle.heldout_frames is not None:
            heldout_pairs = build_heldout_camera_swap_pairs(
                self.multicam_bundle.train_view_count,
                self.multicam_bundle.heldout_view_count,
                train_camera_names=self.multicam_bundle.train_camera_names,
                heldout_camera_names=self.multicam_bundle.heldout_camera_names or None,
            )
            for pair in heldout_pairs:
                if int(pair.source_view) != source_view:
                    continue
                if self.train_cfg["heldout_eval_camera_mode"] == "calibrated":
                    cameras = self.source_relative_cameras_for_pair(pair, frame_indices)
                    camera_owner = "external_rig"
                else:
                    prediction = self.full_relative_pose_for_pair(
                        pair,
                        clip_indices=frame_indices,
                        memory_cache=relpose_memory_cache,
                    )
                    cameras = cameras_with_se3_transform(
                        prediction.target_template_cameras,
                        prediction.camera_to_world,
                    )
                    camera_owner = "model"
                heldout_rendered.append(
                    self.render_camera_swap_pair_with_cameras(
                        decoded,
                        pair=pair,
                        cameras=cameras,
                        clip_indices=frame_indices,
                        phase="eval",
                        camera_owner=camera_owner,
                    ).detach_cpu()
                )

        return train_rendered, heldout_rendered, decoded_temporal_payload(
            {
                "xyz": [decoded.xyz[index].detach().cpu() for index in range(decoded.frame_count)],
                "scales": [decoded.scales[index].detach().cpu() for index in range(decoded.frame_count)],
                "opacities": [decoded.opacities[index].detach().cpu() for index in range(decoded.frame_count)],
                "rgbs": [decoded.rgbs[index].detach().cpu() for index in range(decoded.frame_count)],
            }
        )

    def validation_video_payload(self, step: int | None = None) -> dict[str, Any]:
        base_render_size = getattr(self, "base_render_size", int(self.render_cfg["render_size"]))
        if int(self.render_size) != int(base_render_size):
            with self.temporary_render_size(base_render_size):
                return self.validation_video_payload(step=step)
        train_rendered, heldout_rendered, decoded_metrics = self.render_full_predicted_relative_views()
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

    def scalar_payload(self, result) -> dict[str, Any]:
        payload = super().scalar_payload(result)
        payload["RenderSize"] = int(getattr(result, "render_size", self.render_size))
        payload["Render/BaseSize"] = int(getattr(self, "base_render_size", self.render_size))
        payload["Render/MultiResEnabled"] = 1.0 if self.multires_render_sizes else 0.0
        if self.multires_render_sizes:
            payload["Render/MultiResChoiceCount"] = len(self.multires_render_sizes)
        if self.multires_render_probability_by_size:
            payload["Render/MultiResProbability"] = float(
                self.multires_render_probability_by_size[int(getattr(result, "render_size", self.render_size))]
            )
        token_detail_level = getattr(result, "token_detail_level", None)
        if token_detail_level is not None:
            payload["Tokens/ActiveDetailLevel"] = int(token_detail_level)
            payload["Tokens/ActiveDecodedTokens"] = int(getattr(result, "active_decoded_tokens", 0))
        payload["RelPose/FullPoseMode"] = 1.0
        payload["RelPose/HeldoutPredictedEval"] = (
            1.0 if self.train_cfg["heldout_eval_camera_mode"] == "predicted_relpose" else 0.0
        )
        payload["RelPose/RelposeOnlyTrainable"] = 1.0 if self.train_cfg["trainable_scope"] == "relpose_only" else 0.0
        payload["RelPose/PoseLossWeight"] = float(self.train_cfg["relpose_pose_loss_weight"])
        return payload

    def val_log(self, step: int, result) -> None:
        should_log_scalars = self.should_log_scalars(step)
        should_log_images = self.should_log_images(step)
        should_log_videos = self.should_log_videos(step)
        if not (should_log_scalars or should_log_images or should_log_videos):
            return

        result_render_size = int(getattr(result, "render_size", self.render_size))
        with self.temporary_render_size(result_render_size):
            payload = self.scalar_payload(result)
            if should_log_images:
                payload["Render_GT_vs_Pred"] = _render_preview_image_impl(self.cfg, result, step)
                if self.feature_pca_log:
                    if result.preview_features is None:
                        raise ValueError(
                            "feature_pca_log is on but preview_features was not retained for the training step."
                        )
                    from feature_pca_viz import feature_pca_to_rgb

                    pca_rgb = feature_pca_to_rgb(result.preview_features)
                    pca_image = (
                        pca_rgb.detach().cpu().clamp(0, 1).permute(1, 2, 0) * 255.0
                    ).to(torch.uint8).numpy()
                    payload["media/feature_pca_image"] = wandb.Image(pca_image, caption=f"Step {step}")
        if should_log_videos:
            with self.temporary_render_size(self.base_render_size):
                payload.update(self.validation_video_payload(step=step))
        wandb.log(payload, step=step)


def run_training(config: dict[str, Any]) -> None:
    MulticamRelativePoseImplicitTrainer(config).run()


def main(config: dict[str, Any] | str | Path) -> None:
    if isinstance(config, (str, Path)):
        run_training(load_config_file(config))
    else:
        run_training(config)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: uv run python src/train/train_multicam_relative_pose_implicit_dynamic.py "
            "src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_vjepa_full_relpose_128_16f_8192splats_goodset_train0006_0014_holdout0005.jsonc"
        )
    main(sys.argv[1])
