from __future__ import annotations

from typing import Any

from config_utils import require_config_keys, require_config_sections
from star_uvt_config_keys import (
    REQUIRED_STAR_UVT_COLORIZE_KEYS,
    REQUIRED_STAR_UVT_DATA_KEYS,
    REQUIRED_STAR_UVT_LOGGING_KEYS,
    REQUIRED_STAR_UVT_OUTPUT_KEYS,
    require_star_uvt_colorize_config,
    require_star_uvt_data_config,
    require_star_uvt_logging_config,
    require_star_uvt_output_config,
)
from star_uvt_feature_targets import FEATURE_TARGET_GRID_ADAPTERS
from star_uvt_feature_rendering import ALPHA_BACKGROUND_SAMPLE_SCOPES, ALPHA_BACKGROUND_STRATEGIES
from star_uvt_projective_interval_backend import resolve_projective_interval_backend_settings
from star_uvt_render_modes import FEATURE_RENDER_MODES
from star_uvt_schedules import _feature_target_weight_schedule
from star_uvt_sparse_visual_sampling import (
    NATIVE_PIXEL_SPARSE_VISUAL_LOSS_VJP_MODES,
    NATIVE_SPARSE_VISUAL_LOSS_VJP_MODES,
    NATIVE_TARGET_AREA_SPARSE_VISUAL_LOSS_VJP_MODES,
    SPARSE_VISUAL_COMPOSITIONS,
    SPARSE_VISUAL_LOSS_BASES,
    SPARSE_VISUAL_LOSS_VJP_MODES,
    SPARSE_VISUAL_PATCH_PIXEL_SOURCES,
    SPARSE_VISUAL_PIXEL_SOURCES,
)
from star_uvt_visibility_support import (
    SUPPORT_BIRTH_SPLIT_FEATURE_INIT_MODES,
    SUPPORT_BIRTH_SPLIT_CENTER_STRATEGIES,
    SUPPORT_BIRTH_SPLIT_SHAPES,
    SUPPORT_BIRTH_SPLIT_TARGET_POINT_SOURCES,
    SUPPORT_BIRTH_SPLIT_TUBE_ALLOCATIONS,
    SUPPORT_BIRTH_SPLIT_TUBE_SELECTIONS,
)


REQUIRED_SECTIONS = ("data", "train", "feature_uvt", "colorize", "output", "logging")
REQUIRED_DATA_KEYS = REQUIRED_STAR_UVT_DATA_KEYS
REQUIRED_TRAIN_KEYS = (
    "steps",
    "lr",
    "device",
    "seed",
    "frame_chunk_size",
    "require_loss_decrease",
    "require_gradient_flow",
    "require_no_tile_overflow",
)
REQUIRED_FEATURE_UVT_KEYS = (
    "tube_count",
    "feature_dim",
    "tile_t",
    "tile_capacity",
    "alpha_threshold",
    "max_alpha",
)
REQUIRED_COLORIZE_KEYS = REQUIRED_STAR_UVT_COLORIZE_KEYS
REQUIRED_OUTPUT_KEYS = REQUIRED_STAR_UVT_OUTPUT_KEYS
REQUIRED_LOGGING_KEYS = REQUIRED_STAR_UVT_LOGGING_KEYS
REQUIRED_FEATURE_TARGET_KEYS = (
    "enabled",
    "layer",
    "loss_type",
    "loss_weight",
    "rgb_loss_weight",
    "channel_adapter",
    "temporal_spatial_adapter",
    "normalization",
)
FEATURE_TARGET_LOSSES = {"mse", "l1", "smooth_l1"}
FEATURE_TARGET_CHANNEL_ADAPTERS = {"error", "truncate_or_pad", "repeat_truncate"}
FEATURE_TARGET_NORMALIZATIONS = {"none", "channel_standardize"}
FEATURE_TARGET_MATERIALIZATIONS = {"dense", "chunked", "cached_chunks", "target_grid"}
FEATURE_TARGET_IMAGE_VJP_MODES = {
    "autograd",
    "analytic",
    "analytic_sparse_pixels",
    "analytic_sparse_grid",
    "analytic_sparse_grid_forward",
    "analytic_sparse_grid_forward_batched",
}
REQUIRED_SPARSE_VISUAL_KEYS = (
    "enabled",
    "loss_weight",
    "pixel_source",
    "sample_grid_shape",
)
DENSE_ALPHA_BACKWARD_MODES = {
    "direct_atomic_skip_feature_grad",
    "gradcache_skip_feature_grad",
}
DENSE_ALPHA_RENDER_MODES = {
    "dense_f32",
    "sparse_f1",
}
REQUIRED_VISIBILITY_PROXY_KEYS = (
    "enabled",
    "loss_weight",
    "target_top_fraction",
    "max_points",
    "grid_stride",
    "frame_stride",
    "center_weight",
    "support_weight",
    "support_epsilon",
    "scale_px",
    "temperature",
    "velocity_penalty",
)
REQUIRED_SUPPORT_BIRTH_SPLIT_KEYS = (
    "enabled",
    "target_point_source",
    "target_top_fraction",
    "max_points",
    "grid_stride",
    "frame_stride",
    "reallocate_tubes",
    "support_radius_px",
    "support_shape",
    "support_radius_along_px",
    "support_radius_across_px",
    "support_precision_radius_px",
    "temporal_radius_frames",
    "opacity",
    "tube_selection",
    "center_strategy",
    "center_count",
    "tube_allocation",
    "feature_init_mode",
    "target_alpha_loss_weight",
    "target_alpha_target",
    "target_alpha_max_points",
    "target_area_loss_weight",
    "target_area_patch_shape",
    "target_area_max_points",
    "target_area_vjp_mode",
    "target_area_composition",
    "prefix_alpha_loss_weight",
    "prefix_alpha_target",
    "prefix_alpha_max_points",
    "tile_overflow_repair_enabled",
    "tile_overflow_repair_max_drops",
    "tile_overflow_repair_guard_refs",
    "tile_overflow_repair_opacity",
)



def resolve_config(config: dict[str, Any]) -> dict[str, Any]:
    require_config_sections(config, REQUIRED_SECTIONS)
    require_star_uvt_data_config(config)
    require_config_keys("train", config["train"], REQUIRED_TRAIN_KEYS)
    require_config_keys("feature_uvt", config["feature_uvt"], REQUIRED_FEATURE_UVT_KEYS)
    require_star_uvt_colorize_config(config)
    require_star_uvt_output_config(config)
    require_star_uvt_logging_config(config)
    train = config["train"]
    train.setdefault("resume_checkpoint", None)
    train.setdefault("resume_optimizer", True)
    train.setdefault("resume_colorizer", True)
    if bool(train["resume_optimizer"]) and not bool(train["resume_colorizer"]):
        raise ValueError("train.resume_optimizer requires train.resume_colorizer=true")
    train.setdefault("global_step_offset", 0)
    if int(train["global_step_offset"]) < 0:
        raise ValueError("train.global_step_offset must be non-negative")
    train.setdefault("lr_schedule", None)
    train.setdefault("trace_global_steps", [])
    if train["trace_global_steps"] is None:
        train["trace_global_steps"] = []
    if not isinstance(train["trace_global_steps"], list):
        raise TypeError("train.trace_global_steps must be a list of non-negative global step ids")
    train["trace_global_steps"] = [int(step) for step in train["trace_global_steps"]]
    if any(step < 0 for step in train["trace_global_steps"]):
        raise ValueError("train.trace_global_steps entries must be non-negative")
    config["output"].setdefault("checkpoint", None)
    config["output"].setdefault("rgb_probe_contact_sheet", None)
    config["output"].setdefault("rgb_probe_side_by_side_video", None)
    config["output"].setdefault("rgb_probe_side_by_side_fps", config["output"]["side_by_side_fps"])
    config["colorize"].setdefault("init_checkpoint", None)
    if (
        config["colorize"]["init_checkpoint"] is not None
        and train["resume_checkpoint"] is not None
        and bool(train["resume_optimizer"])
    ):
        raise ValueError("colorize.init_checkpoint with train.resume_checkpoint requires train.resume_optimizer=false")
    feature_uvt = config["feature_uvt"]
    feature_uvt.setdefault("render_mode", "feature_direct_atomic")
    if str(feature_uvt["render_mode"]) not in FEATURE_RENDER_MODES:
        expected = ", ".join(sorted(FEATURE_RENDER_MODES))
        raise ValueError(f"feature_uvt.render_mode must be one of: {expected}")
    resolve_projective_interval_backend_settings(config)
    alpha_background = config.setdefault("alpha_background", {})
    alpha_background.setdefault("train_strategy", "fixed_black_after_colorizer")
    alpha_background.setdefault("eval_strategy", "fixed_black_after_colorizer")
    alpha_background.setdefault("sample_scope", "step")
    if str(alpha_background["train_strategy"]) not in ALPHA_BACKGROUND_STRATEGIES:
        expected = ", ".join(sorted(ALPHA_BACKGROUND_STRATEGIES))
        raise ValueError(f"alpha_background.train_strategy must be one of: {expected}")
    if str(alpha_background["eval_strategy"]) not in ALPHA_BACKGROUND_STRATEGIES:
        expected = ", ".join(sorted(ALPHA_BACKGROUND_STRATEGIES))
        raise ValueError(f"alpha_background.eval_strategy must be one of: {expected}")
    if str(alpha_background["sample_scope"]) not in ALPHA_BACKGROUND_SAMPLE_SCOPES:
        expected = ", ".join(sorted(ALPHA_BACKGROUND_SAMPLE_SCOPES))
        raise ValueError(f"alpha_background.sample_scope must be one of: {expected}")
    feature_target = config.setdefault("feature_target", {"enabled": False})
    feature_target.setdefault("enabled", False)
    if bool(feature_target["enabled"]):
        if "features" not in config:
            raise KeyError("feature_target.enabled=true requires a top-level features config section")
        require_config_keys("feature_target", feature_target, REQUIRED_FEATURE_TARGET_KEYS)
        if str(feature_target["loss_type"]) not in FEATURE_TARGET_LOSSES:
            expected = ", ".join(sorted(FEATURE_TARGET_LOSSES))
            raise ValueError(f"feature_target.loss_type must be one of: {expected}")
        if str(feature_target["channel_adapter"]) not in FEATURE_TARGET_CHANNEL_ADAPTERS:
            expected = ", ".join(sorted(FEATURE_TARGET_CHANNEL_ADAPTERS))
            raise ValueError(f"feature_target.channel_adapter must be one of: {expected}")
        if str(feature_target["temporal_spatial_adapter"]) not in FEATURE_TARGET_GRID_ADAPTERS:
            expected = ", ".join(sorted(FEATURE_TARGET_GRID_ADAPTERS))
            raise ValueError(f"feature_target.temporal_spatial_adapter must be one of: {expected}")
        if str(feature_target["normalization"]) not in FEATURE_TARGET_NORMALIZATIONS:
            expected = ", ".join(sorted(FEATURE_TARGET_NORMALIZATIONS))
            raise ValueError(f"feature_target.normalization must be one of: {expected}")
        feature_target.setdefault("materialization", "dense")
        if str(feature_target["materialization"]) not in FEATURE_TARGET_MATERIALIZATIONS:
            expected = ", ".join(sorted(FEATURE_TARGET_MATERIALIZATIONS))
            raise ValueError(f"feature_target.materialization must be one of: {expected}")
        feature_target.setdefault("image_vjp_mode", "autograd")
        if str(feature_target["image_vjp_mode"]) not in FEATURE_TARGET_IMAGE_VJP_MODES:
            expected = ", ".join(sorted(FEATURE_TARGET_IMAGE_VJP_MODES))
            raise ValueError(f"feature_target.image_vjp_mode must be one of: {expected}")
        if (
            feature_target.get("materialization_chunk_size") is not None
            and int(feature_target["materialization_chunk_size"]) <= 0
        ):
            raise ValueError("feature_target.materialization_chunk_size must be positive")
        feature_target.setdefault("rgb_probe_checkpoint", None)
        feature_target.setdefault("rgb_grid_loss_weight", 0.0)
        feature_target.setdefault("rgb_probe_loss_weight", 0.0)
        feature_target.setdefault("rgb_probe_target_rgb_adapter", feature_target["temporal_spatial_adapter"])
        rgb_grid_loss_weight = float(feature_target["rgb_grid_loss_weight"])
        rgb_probe_loss_weight = float(feature_target["rgb_probe_loss_weight"])
        if rgb_grid_loss_weight < 0.0:
            raise ValueError("feature_target.rgb_grid_loss_weight must be non-negative")
        if rgb_probe_loss_weight < 0.0:
            raise ValueError("feature_target.rgb_probe_loss_weight must be non-negative")
        if str(feature_target["rgb_probe_target_rgb_adapter"]) not in FEATURE_TARGET_GRID_ADAPTERS:
            expected = ", ".join(sorted(FEATURE_TARGET_GRID_ADAPTERS))
            raise ValueError(f"feature_target.rgb_probe_target_rgb_adapter must be one of: {expected}")
        if feature_target["rgb_probe_checkpoint"] is None and rgb_probe_loss_weight > 0.0:
            raise ValueError("feature_target.rgb_probe_loss_weight requires rgb_probe_checkpoint")
        if feature_target["rgb_probe_checkpoint"] is not None and str(feature_target["materialization"]) != "target_grid":
            raise ValueError("feature_target.rgb_probe_checkpoint requires materialization=target_grid")
        if rgb_grid_loss_weight > 0.0 and str(feature_target["materialization"]) != "target_grid":
            raise ValueError("feature_target.rgb_grid_loss_weight requires materialization=target_grid")
        weight_schedule = _feature_target_weight_schedule(config)
        if any(stage.rgb_grid_loss_weight > 0.0 for stage in weight_schedule) and str(
            feature_target["materialization"]
        ) != "target_grid":
            raise ValueError("feature_target.weight_schedule rgb_grid_loss_weight requires materialization=target_grid")
        if feature_target["rgb_probe_checkpoint"] is None and any(
            stage.rgb_probe_loss_weight > 0.0 for stage in weight_schedule
        ):
            raise ValueError("feature_target.weight_schedule rgb_probe_loss_weight requires rgb_probe_checkpoint")
        if str(feature_target["image_vjp_mode"]) in {
            "analytic",
            "analytic_sparse_pixels",
            "analytic_sparse_grid",
            "analytic_sparse_grid_forward",
            "analytic_sparse_grid_forward_batched",
        }:
            if str(feature_target["materialization"]) != "target_grid":
                raise ValueError("analytic feature_target.image_vjp_mode requires materialization=target_grid")
            if str(feature_target["loss_type"]) != "mse":
                raise ValueError("analytic feature_target.image_vjp_mode currently requires loss_type=mse")
            if any(stage.rgb_loss_weight > 0.0 for stage in weight_schedule):
                raise ValueError("analytic feature_target.image_vjp_mode does not support RGB reconstruction loss")
            if (
                str(feature_target["image_vjp_mode"]) in {"analytic_sparse_grid", "analytic_sparse_grid_forward"}
                and str(feature_target["temporal_spatial_adapter"]) != "trilinear"
            ):
                raise ValueError("analytic_sparse_grid image VJP modes currently require temporal_spatial_adapter=trilinear")
    sparse_visual = config.setdefault("sparse_visual", {"enabled": False})
    sparse_visual.setdefault("enabled", False)
    if bool(sparse_visual["enabled"]):
        require_config_keys("sparse_visual", sparse_visual, REQUIRED_SPARSE_VISUAL_KEYS)
        if float(sparse_visual["loss_weight"]) <= 0.0:
            raise ValueError("sparse_visual.loss_weight must be positive when enabled")
        if str(sparse_visual["pixel_source"]) not in SPARSE_VISUAL_PIXEL_SOURCES:
            expected = ", ".join(sorted(SPARSE_VISUAL_PIXEL_SOURCES))
            raise ValueError(f"sparse_visual.pixel_source must be one of: {expected}")
        sparse_visual.setdefault("loss_basis", "pixel")
        if str(sparse_visual["loss_basis"]) not in SPARSE_VISUAL_LOSS_BASES:
            expected = ", ".join(sorted(SPARSE_VISUAL_LOSS_BASES))
            raise ValueError(f"sparse_visual.loss_basis must be one of: {expected}")
        sparse_visual.setdefault("composition", "black")
        if str(sparse_visual["composition"]) not in SPARSE_VISUAL_COMPOSITIONS:
            expected = ", ".join(sorted(SPARSE_VISUAL_COMPOSITIONS))
            raise ValueError(f"sparse_visual.composition must be one of: {expected}")
        sparse_visual.setdefault("loss_vjp_mode", "autograd")
        if str(sparse_visual["loss_vjp_mode"]) not in SPARSE_VISUAL_LOSS_VJP_MODES:
            expected = ", ".join(sorted(SPARSE_VISUAL_LOSS_VJP_MODES))
            raise ValueError(f"sparse_visual.loss_vjp_mode must be one of: {expected}")
        sparse_visual.setdefault("alpha_loss_weight", 0.0)
        sparse_visual.setdefault("alpha_target", 1.0)
        sparse_visual.setdefault("black_hole_loss_weight", 0.0)
        if float(sparse_visual["alpha_loss_weight"]) < 0.0:
            raise ValueError("sparse_visual.alpha_loss_weight must be non-negative")
        if not 0.0 <= float(sparse_visual["alpha_target"]) <= 1.0:
            raise ValueError("sparse_visual.alpha_target must be in [0, 1]")
        if float(sparse_visual["black_hole_loss_weight"]) < 0.0:
            raise ValueError("sparse_visual.black_hole_loss_weight must be non-negative")
        if (
            float(sparse_visual["alpha_loss_weight"]) > 0.0
            and str(sparse_visual["loss_vjp_mode"]) in NATIVE_SPARSE_VISUAL_LOSS_VJP_MODES
        ):
            raise ValueError("sparse_visual.alpha_loss_weight currently requires non-native sparse_visual.loss_vjp_mode")
        if (
            float(sparse_visual["black_hole_loss_weight"]) > 0.0
            and str(sparse_visual["loss_vjp_mode"]) in NATIVE_SPARSE_VISUAL_LOSS_VJP_MODES
        ):
            raise ValueError(
                "sparse_visual.black_hole_loss_weight currently requires non-native sparse_visual.loss_vjp_mode"
            )
        if (
            str(sparse_visual["composition"]) != "black"
            and str(sparse_visual["loss_vjp_mode"]) in NATIVE_SPARSE_VISUAL_LOSS_VJP_MODES
        ):
            raise ValueError("sparse_visual.composition=target_background requires non-native sparse_visual.loss_vjp_mode")
        if str(sparse_visual["loss_vjp_mode"]) in NATIVE_PIXEL_SPARSE_VISUAL_LOSS_VJP_MODES:
            if str(sparse_visual["loss_basis"]) != "pixel":
                raise ValueError("native pixel sparse_visual.loss_vjp_mode currently requires loss_basis=pixel")
            if float(sparse_visual["loss_weight"]) != 1.0:
                raise ValueError("native pixel sparse_visual.loss_vjp_mode currently requires loss_weight=1.0")
        if str(sparse_visual["loss_vjp_mode"]) in NATIVE_TARGET_AREA_SPARSE_VISUAL_LOSS_VJP_MODES:
            if str(sparse_visual["loss_basis"]) != "target_area_mean":
                raise ValueError(
                    "native target-area sparse_visual.loss_vjp_mode currently requires loss_basis=target_area_mean"
                )
            if float(sparse_visual["loss_weight"]) != 1.0:
                raise ValueError("native target-area sparse_visual.loss_vjp_mode currently requires loss_weight=1.0")
        if str(sparse_visual["loss_basis"]) in {"patch_mean", "target_area_mean"} and str(
            sparse_visual["pixel_source"]
        ) not in SPARSE_VISUAL_PATCH_PIXEL_SOURCES:
            raise ValueError(f"sparse_visual.loss_basis={sparse_visual['loss_basis']} requires a patch pixel_source")
        sample_grid_shape = sparse_visual["sample_grid_shape"]
        if (
            not isinstance(sample_grid_shape, list | tuple)
            or len(sample_grid_shape) != 3
            or any(int(item) <= 0 for item in sample_grid_shape)
        ):
            raise ValueError("sparse_visual.sample_grid_shape must be [frames, height, width]")
        max_shape = (
            int(config["data"]["max_frames"]),
            int(config["data"]["target_size"]),
            int(config["data"]["target_size"]),
        )
        if any(int(requested) > int(limit) for requested, limit in zip(sample_grid_shape, max_shape, strict=True)):
            raise ValueError("sparse_visual.sample_grid_shape cannot exceed [max_frames, target_size, target_size]")
        if str(sparse_visual["pixel_source"]) in SPARSE_VISUAL_PATCH_PIXEL_SOURCES:
            patch_shape = sparse_visual.setdefault("patch_shape", [1, 1])
            if (
                not isinstance(patch_shape, list | tuple)
                or len(patch_shape) != 2
                or any(int(item) <= 0 for item in patch_shape)
            ):
                raise ValueError("sparse_visual.patch_shape must be [height, width]")
            if int(patch_shape[0]) > int(config["data"]["target_size"]) or int(patch_shape[1]) > int(
                config["data"]["target_size"]
            ):
                raise ValueError("sparse_visual.patch_shape cannot exceed target_size")
            phase_shape = sparse_visual.setdefault("patch_phase_shape", [1, 1])
            if (
                not isinstance(phase_shape, list | tuple)
                or len(phase_shape) != 2
                or any(int(item) <= 0 for item in phase_shape)
            ):
                raise ValueError("sparse_visual.patch_phase_shape must be [height, width]")
            if str(sparse_visual["pixel_source"]) == "stratified_patch_grid_phase":
                sample_h, sample_w = int(sample_grid_shape[1]), int(sample_grid_shape[2])
                target_size = int(config["data"]["target_size"])
                if target_size % sample_h != 0 or target_size % sample_w != 0:
                    raise ValueError("stratified_patch_grid_phase requires target_size divisible by grid shape")
                cell_h = target_size // sample_h
                cell_w = target_size // sample_w
                if int(patch_shape[0]) * int(phase_shape[0]) > cell_h or int(patch_shape[1]) * int(
                    phase_shape[1]
                ) > cell_w:
                    raise ValueError(
                        "sparse_visual.patch_shape * patch_phase_shape must fit within one stratified cell"
                    )
    dense_alpha = config.setdefault("dense_alpha", {"enabled": False})
    dense_alpha.setdefault("enabled", False)
    if bool(dense_alpha["enabled"]):
        dense_alpha.setdefault("loss_weight", 1.0)
        dense_alpha.setdefault("alpha_target", 1.0)
        dense_alpha.setdefault("backward_mode", "gradcache_skip_feature_grad")
        dense_alpha.setdefault("render_mode", "dense_f32")
        if float(dense_alpha["loss_weight"]) <= 0.0:
            raise ValueError("dense_alpha.loss_weight must be positive when enabled")
        if not 0.0 <= float(dense_alpha["alpha_target"]) <= 1.0:
            raise ValueError("dense_alpha.alpha_target must be in [0, 1]")
        if str(dense_alpha["backward_mode"]) not in DENSE_ALPHA_BACKWARD_MODES:
            expected = ", ".join(sorted(DENSE_ALPHA_BACKWARD_MODES))
            raise ValueError(f"dense_alpha.backward_mode must be one of: {expected}")
        if str(dense_alpha["render_mode"]) not in DENSE_ALPHA_RENDER_MODES:
            expected = ", ".join(sorted(DENSE_ALPHA_RENDER_MODES))
            raise ValueError(f"dense_alpha.render_mode must be one of: {expected}")
    visibility_proxy = config.setdefault("visibility_proxy", {"enabled": False})
    visibility_proxy.setdefault("enabled", False)
    if bool(visibility_proxy["enabled"]):
        visibility_proxy.setdefault("loss_weight", 1.0)
        visibility_proxy.setdefault("target_top_fraction", 0.02)
        visibility_proxy.setdefault("max_points", 4096)
        visibility_proxy.setdefault("grid_stride", 8)
        visibility_proxy.setdefault("frame_stride", 1)
        visibility_proxy.setdefault("center_weight", 1.0)
        visibility_proxy.setdefault("support_weight", 0.0)
        visibility_proxy.setdefault("support_epsilon", 1.0e-4)
        visibility_proxy.setdefault("scale_px", 64.0)
        visibility_proxy.setdefault("temperature", 0.75)
        visibility_proxy.setdefault("velocity_penalty", 0.0025)
        require_config_keys("visibility_proxy", visibility_proxy, REQUIRED_VISIBILITY_PROXY_KEYS)
        if float(visibility_proxy["loss_weight"]) <= 0.0:
            raise ValueError("visibility_proxy.loss_weight must be positive when enabled")
        if not 0.0 < float(visibility_proxy["target_top_fraction"]) <= 1.0:
            raise ValueError("visibility_proxy.target_top_fraction must be in (0, 1]")
        if int(visibility_proxy["max_points"]) <= 0:
            raise ValueError("visibility_proxy.max_points must be positive")
        if int(visibility_proxy["grid_stride"]) <= 0:
            raise ValueError("visibility_proxy.grid_stride must be positive")
        if int(visibility_proxy["frame_stride"]) <= 0:
            raise ValueError("visibility_proxy.frame_stride must be positive")
        if float(visibility_proxy["center_weight"]) < 0.0:
            raise ValueError("visibility_proxy.center_weight must be non-negative")
        if float(visibility_proxy["support_weight"]) < 0.0:
            raise ValueError("visibility_proxy.support_weight must be non-negative")
        if float(visibility_proxy["center_weight"]) <= 0.0 and float(visibility_proxy["support_weight"]) <= 0.0:
            raise ValueError("visibility_proxy requires center_weight or support_weight to be positive")
        if float(visibility_proxy["support_epsilon"]) <= 0.0:
            raise ValueError("visibility_proxy.support_epsilon must be positive")
        if float(visibility_proxy["scale_px"]) <= 0.0:
            raise ValueError("visibility_proxy.scale_px must be positive")
        if float(visibility_proxy["temperature"]) <= 0.0:
            raise ValueError("visibility_proxy.temperature must be positive")
        if float(visibility_proxy["velocity_penalty"]) < 0.0:
            raise ValueError("visibility_proxy.velocity_penalty must be non-negative")
    support_birth_split = config.setdefault("support_birth_split", {"enabled": False})
    support_birth_split.setdefault("enabled", False)
    if bool(support_birth_split["enabled"]):
        support_birth_split.setdefault("target_point_source", "top_brightness")
        support_birth_split.setdefault("target_top_fraction", 0.02)
        support_birth_split.setdefault("max_points", 2048)
        support_birth_split.setdefault("grid_stride", 8)
        support_birth_split.setdefault("frame_stride", 1)
        support_birth_split.setdefault("reallocate_tubes", 32)
        support_birth_split.setdefault("support_radius_px", 64.0)
        support_birth_split.setdefault("support_shape", "isotropic")
        support_birth_split.setdefault("support_radius_along_px", float(support_birth_split["support_radius_px"]))
        support_birth_split.setdefault("support_radius_across_px", float(support_birth_split["support_radius_px"]))
        support_birth_split.setdefault("support_precision_radius_px", float(support_birth_split["support_radius_px"]))
        support_birth_split.setdefault("temporal_radius_frames", float(config["data"]["max_frames"]))
        support_birth_split.setdefault("opacity", min(0.8, float(config["feature_uvt"]["max_alpha"]) * 0.9))
        support_birth_split.setdefault("tube_selection", "lowest_opacity")
        support_birth_split.setdefault("center_strategy", "global_line")
        support_birth_split.setdefault("center_count", 1)
        support_birth_split.setdefault("tube_allocation", "proportional")
        support_birth_split.setdefault("feature_init_mode", "preserve")
        support_birth_split.setdefault("target_alpha_loss_weight", 0.0)
        support_birth_split.setdefault("target_alpha_target", min(1.0, float(config["feature_uvt"]["max_alpha"])))
        support_birth_split.setdefault("target_alpha_max_points", int(support_birth_split["max_points"]))
        support_birth_split.setdefault("target_area_loss_weight", 0.0)
        support_birth_split.setdefault("target_area_patch_shape", [2, 2])
        support_birth_split.setdefault("target_area_max_points", int(support_birth_split["max_points"]))
        support_birth_split.setdefault("target_area_vjp_mode", "manual_hidden64_star_only")
        support_birth_split.setdefault("target_area_composition", "black")
        support_birth_split.setdefault("prefix_alpha_loss_weight", 0.0)
        support_birth_split.setdefault("prefix_alpha_target", min(1.0, float(config["feature_uvt"]["max_alpha"])))
        support_birth_split.setdefault("prefix_alpha_max_points", int(support_birth_split["max_points"]))
        support_birth_split.setdefault("tile_overflow_repair_enabled", False)
        support_birth_split.setdefault("tile_overflow_repair_max_drops", int(support_birth_split["reallocate_tubes"]))
        support_birth_split.setdefault("tile_overflow_repair_guard_refs", 0)
        support_birth_split.setdefault("tile_overflow_repair_opacity", 1.0e-5)
        require_config_keys("support_birth_split", support_birth_split, REQUIRED_SUPPORT_BIRTH_SPLIT_KEYS)
        if train["resume_checkpoint"] is not None and bool(train["resume_optimizer"]):
            raise ValueError("support_birth_split with train.resume_checkpoint requires train.resume_optimizer=false")
        if str(support_birth_split["target_point_source"]) not in SUPPORT_BIRTH_SPLIT_TARGET_POINT_SOURCES:
            expected = ", ".join(sorted(SUPPORT_BIRTH_SPLIT_TARGET_POINT_SOURCES))
            raise ValueError(f"support_birth_split.target_point_source must be one of: {expected}")
        if not 0.0 < float(support_birth_split["target_top_fraction"]) <= 1.0:
            raise ValueError("support_birth_split.target_top_fraction must be in (0, 1]")
        if int(support_birth_split["max_points"]) <= 0:
            raise ValueError("support_birth_split.max_points must be positive")
        if int(support_birth_split["grid_stride"]) <= 0:
            raise ValueError("support_birth_split.grid_stride must be positive")
        if int(support_birth_split["frame_stride"]) <= 0:
            raise ValueError("support_birth_split.frame_stride must be positive")
        reallocate_tubes = int(support_birth_split["reallocate_tubes"])
        if reallocate_tubes <= 0:
            raise ValueError("support_birth_split.reallocate_tubes must be positive")
        if reallocate_tubes > int(config["feature_uvt"]["tube_count"]):
            raise ValueError("support_birth_split.reallocate_tubes cannot exceed feature_uvt.tube_count")
        if float(support_birth_split["support_radius_px"]) <= 0.0:
            raise ValueError("support_birth_split.support_radius_px must be positive")
        if str(support_birth_split["support_shape"]) not in SUPPORT_BIRTH_SPLIT_SHAPES:
            expected = ", ".join(sorted(SUPPORT_BIRTH_SPLIT_SHAPES))
            raise ValueError(f"support_birth_split.support_shape must be one of: {expected}")
        if float(support_birth_split["support_radius_along_px"]) <= 0.0:
            raise ValueError("support_birth_split.support_radius_along_px must be positive")
        if float(support_birth_split["support_radius_across_px"]) <= 0.0:
            raise ValueError("support_birth_split.support_radius_across_px must be positive")
        if float(support_birth_split["support_precision_radius_px"]) <= 0.0:
            raise ValueError("support_birth_split.support_precision_radius_px must be positive")
        if float(support_birth_split["temporal_radius_frames"]) <= 0.0:
            raise ValueError("support_birth_split.temporal_radius_frames must be positive")
        if not 0.0 < float(support_birth_split["opacity"]) < float(config["feature_uvt"]["max_alpha"]):
            raise ValueError("support_birth_split.opacity must be in (0, feature_uvt.max_alpha)")
        if str(support_birth_split["tube_selection"]) not in SUPPORT_BIRTH_SPLIT_TUBE_SELECTIONS:
            expected = ", ".join(sorted(SUPPORT_BIRTH_SPLIT_TUBE_SELECTIONS))
            raise ValueError(f"support_birth_split.tube_selection must be one of: {expected}")
        if str(support_birth_split["center_strategy"]) not in SUPPORT_BIRTH_SPLIT_CENTER_STRATEGIES:
            expected = ", ".join(sorted(SUPPORT_BIRTH_SPLIT_CENTER_STRATEGIES))
            raise ValueError(f"support_birth_split.center_strategy must be one of: {expected}")
        if int(support_birth_split["center_count"]) <= 0:
            raise ValueError("support_birth_split.center_count must be positive")
        if int(support_birth_split["center_count"]) > reallocate_tubes:
            raise ValueError("support_birth_split.center_count cannot exceed reallocate_tubes")
        if str(support_birth_split["tube_allocation"]) not in SUPPORT_BIRTH_SPLIT_TUBE_ALLOCATIONS:
            expected = ", ".join(sorted(SUPPORT_BIRTH_SPLIT_TUBE_ALLOCATIONS))
            raise ValueError(f"support_birth_split.tube_allocation must be one of: {expected}")
        if str(support_birth_split["feature_init_mode"]) not in SUPPORT_BIRTH_SPLIT_FEATURE_INIT_MODES:
            expected = ", ".join(sorted(SUPPORT_BIRTH_SPLIT_FEATURE_INIT_MODES))
            raise ValueError(f"support_birth_split.feature_init_mode must be one of: {expected}")
        if str(support_birth_split["feature_init_mode"]) != "preserve" and not bool(feature_target["enabled"]):
            raise ValueError("support_birth_split.feature_init_mode target modes require feature_target.enabled=true")
        if float(support_birth_split["target_alpha_loss_weight"]) < 0.0:
            raise ValueError("support_birth_split.target_alpha_loss_weight must be non-negative")
        if not 0.0 <= float(support_birth_split["target_alpha_target"]) <= float(config["feature_uvt"]["max_alpha"]):
            raise ValueError("support_birth_split.target_alpha_target must be in [0, feature_uvt.max_alpha]")
        if int(support_birth_split["target_alpha_max_points"]) <= 0:
            raise ValueError("support_birth_split.target_alpha_max_points must be positive")
        if float(support_birth_split["target_area_loss_weight"]) < 0.0:
            raise ValueError("support_birth_split.target_area_loss_weight must be non-negative")
        patch_shape = support_birth_split["target_area_patch_shape"]
        if (
            not isinstance(patch_shape, list | tuple)
            or len(patch_shape) != 2
            or any(int(item) <= 0 for item in patch_shape)
        ):
            raise ValueError("support_birth_split.target_area_patch_shape must contain two positive integers")
        if int(support_birth_split["target_area_max_points"]) <= 0:
            raise ValueError("support_birth_split.target_area_max_points must be positive")
        if str(support_birth_split["target_area_vjp_mode"]) not in SPARSE_VISUAL_LOSS_VJP_MODES:
            expected = ", ".join(sorted(SPARSE_VISUAL_LOSS_VJP_MODES))
            raise ValueError(f"support_birth_split.target_area_vjp_mode must be one of: {expected}")
        if str(support_birth_split["target_area_vjp_mode"]) in NATIVE_SPARSE_VISUAL_LOSS_VJP_MODES:
            raise ValueError(
                "support_birth_split.target_area_vjp_mode currently requires non-native sparse visual VJP"
            )
        if str(support_birth_split["target_area_composition"]) not in SPARSE_VISUAL_COMPOSITIONS:
            expected = ", ".join(sorted(SPARSE_VISUAL_COMPOSITIONS))
            raise ValueError(f"support_birth_split.target_area_composition must be one of: {expected}")
        if float(support_birth_split["prefix_alpha_loss_weight"]) < 0.0:
            raise ValueError("support_birth_split.prefix_alpha_loss_weight must be non-negative")
        if not 0.0 <= float(support_birth_split["prefix_alpha_target"]) <= 1.0:
            raise ValueError("support_birth_split.prefix_alpha_target must be in [0, 1]")
        if int(support_birth_split["prefix_alpha_max_points"]) <= 0:
            raise ValueError("support_birth_split.prefix_alpha_max_points must be positive")
        if int(support_birth_split["tile_overflow_repair_max_drops"]) < 0:
            raise ValueError("support_birth_split.tile_overflow_repair_max_drops must be non-negative")
        if int(support_birth_split["tile_overflow_repair_max_drops"]) > reallocate_tubes:
            raise ValueError("support_birth_split.tile_overflow_repair_max_drops cannot exceed reallocate_tubes")
        if int(support_birth_split["tile_overflow_repair_guard_refs"]) < 0:
            raise ValueError("support_birth_split.tile_overflow_repair_guard_refs must be non-negative")
        if int(support_birth_split["tile_overflow_repair_guard_refs"]) >= int(config["feature_uvt"]["tile_capacity"]):
            raise ValueError("support_birth_split.tile_overflow_repair_guard_refs must be smaller than tile_capacity")
        if not 0.0 < float(support_birth_split["tile_overflow_repair_opacity"]) < float(
            config["feature_uvt"]["max_alpha"]
        ):
            raise ValueError("support_birth_split.tile_overflow_repair_opacity must be in (0, feature_uvt.max_alpha)")
    return config



__all__ = [
    "resolve_config",
]
