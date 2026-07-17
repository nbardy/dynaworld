from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import torch
from torch import nn

from colorize import FeatureToColor
from config_utils import path_or_none
from star_uvt_checkpoints import (
    load_feature_to_rgb_probe as _load_feature_to_rgb_probe,
    load_star_training_checkpoint as _load_training_checkpoint,
    optimizer_lrs as _optimizer_lrs,
    save_star_training_checkpoint as _save_training_checkpoint,
    set_optimizer_lr as _set_optimizer_lr,
)
from star_uvt_colorizers import build_feature_colorizer
from star_uvt_common import (
    grad_norms as _grad_norms,
    load_colorizer_init_checkpoint as _load_colorizer_init_checkpoint,
    load_training_sequence as _load_training_sequence,
    target_grid_slice_for_render_chunk as _target_grid_slice_for_render_chunk,
)
from star_uvt_feature_targets import (
    FeatureTargetTensor,
    _adapt_feature_target_grid,
    _adapt_feature_target_grid_chunk,
    _adapt_render_to_feature_target,
    _adapt_rgb_to_grid,
    _feature_target_channel_stats,
    _feature_tensor_to_tchw,
    _load_cached_feature_target,
    _normalize_feature_target,
    _normalize_feature_target_with_stats,
    _upsample_grid_rgb,
)
from star_uvt_feature_config import resolve_config
from star_uvt_feature_rendering import (
    _compose_alpha_background_rgb,
    _render_rgb_chunks,
)
from star_uvt_feature_losses import (
    _feature_target_loss,
    _manual_batched_sparse_target_grid_loss_and_vjp,
    _manual_sparse_target_grid_loss_and_vjp,
    _manual_target_grid_loss_and_vjp,
    _pack_sparse_image_vjp,
)
from star_uvt_render_modes import (
    backward_mode_for_feature_render_mode,
    effective_feature_render_mode_for_report,
    feature_render_mode_fallback_required,
)
from star_uvt_feature_tube_model import _inv_softplus
from star_uvt_projective_interval_backend import (
    make_projective_cell_interval_live_atlas_from_uvt_tubes,
    make_projective_cell_interval_trainer_state_from_uvt_tubes,
    require_projective_interval_atlas_producer,
)
from star_uvt_render_configs import star_uvt_render_configs_from_cfg
from star_uvt_runtime import (
    DYNAWORLD_ROOT,
    STAR_UVT_ROOT,
    ensure_star_uvt_on_path as _ensure_star_uvt_on_path,
    psnr_from_loss as _psnr,
    resolve_device as _resolve_device,
    sync_device as _sync_device,
)
from star_uvt_models import build_feature_tube_model
from star_uvt_outputs import log_star_uvt_row_outputs, write_prediction_media, write_row_json_and_print
from star_uvt_schedules import (
    _feature_target_enabled,
    _feature_target_weight_schedule,
    _feature_target_weight_schedule_json,
    _feature_target_weights_for_step,
    _optimizer_lr_for_step,
    _optimizer_lr_schedule,
    _optimizer_lr_schedule_json,
    _rgb_loss_weight,
)
from star_uvt_sparse_grid import (
    _sparse_target_grid_pixel_ids,
)
from star_uvt_tile_stats import _tile_load_stats
from star_uvt_sparse_visual_losses import (
    _add_param_grad,
    _compose_sparse_visual_rgb,
    _gather_sparse_visual_rgb_values,
    _hidden64_colorizer_layers,
    _native_target_area_backward_mode,
    _sparse_visual_alpha_loss_and_grad,
    _sparse_visual_black_hole_loss_and_grad,
    _sparse_visual_rgb_loss_and_grads,
    _sparse_visual_target_area_cell_ids,
    _sparse_visual_target_area_cells,
)
from star_uvt_sparse_visual_sampling import (
    NATIVE_SPARSE_VISUAL_LOSS_VJP_MODES,
    NATIVE_TARGET_AREA_SPARSE_VISUAL_LOSS_VJP_MODES,
    _sparse_visual_local_frame_ids_for_chunk,
    _sparse_visual_loss_sample_count,
    _sparse_visual_patch_phase_for_step,
    _sparse_visual_pixel_ids_for_chunk,
)
from star_uvt_timing import mean_timing_ms, timing_trace_summary_ms
from star_uvt_visibility_support import (
    _apply_support_birth_split,
    _support_birth_split_repair_tile_overflow_ids,
    _support_birth_split_sample_grid,
    _support_birth_split_sample_target_grid_features,
    _support_birth_split_sampled_tile_load,
    _support_birth_split_set_tube_opacity,
    _support_birth_split_target_points,
    _support_birth_split_target_patch_pixel_ids_for_chunk,
    _support_birth_split_target_pixel_ids_for_chunk,
    _visibility_proxy_loss,
    _visibility_proxy_target_points,
)
from train_logging import finish_wandb_run, init_wandb_run


@dataclass(frozen=True)
class _ProjectiveIntervalFeatureRender:
    feature_image: torch.Tensor
    alpha: torch.Tensor
    timing_ms: dict[str, float] | None = None


@dataclass
class _ProjectiveIntervalFeatureRenderCache:
    state: Any | None = None
    last_rebuild_step: int | None = None
    rebuild_count: int = 0
    live_update_count: int = 0
    alpha_render_count: int = 0
    staleness_check_count: int = 0
    stale_refresh_count: int = 0
    support_rebin_count: int = 0
    visibility_stratify_count: int = 0
    fallback_mark_count: int = 0
    last_support_margin_missing_tile_pairs: int = 0
    last_support_margin_min_slack_px: float = 0.0
    last_support_margin_max_overshoot_px: float = 0.0
    min_support_margin_min_slack_px: float | None = None
    max_support_margin_max_overshoot_px: float = 0.0
    last_support_tail_alpha_bound: float = 0.0
    max_support_tail_alpha_bound: float = 0.0


def _projective_interval_cache_should_rebuild(
    cache: _ProjectiveIntervalFeatureRenderCache | None,
    *,
    step_index: int,
    refresh_every: int,
    refresh_policy: str,
) -> bool:
    if int(refresh_every) < 1:
        raise ValueError("projective interval refresh_every must be positive")
    if refresh_policy not in {"cadence", "measured"}:
        raise ValueError("projective interval refresh_policy must be one of: cadence, measured")
    if cache is None or cache.state is None or cache.last_rebuild_step is None:
        return True
    if refresh_policy == "measured":
        return False
    return int(step_index) - int(cache.last_rebuild_step) >= int(refresh_every)


def _projective_interval_times(frames: int, device: torch.device) -> torch.Tensor:
    return (
        torch.arange(int(frames), dtype=torch.float32, device=device) - 0.5 * float(int(frames) - 1)
    ).contiguous()


def _lock_projective_interval_spatial_precision(model: nn.Module, sigma_px: float) -> float:
    if not hasattr(model, "raw_precision") or not hasattr(model, "min_precision"):
        raise RuntimeError("projective interval producer requires FeatureScreenTimeTubeModel precision parameters")
    raw_precision = model.raw_precision
    if raw_precision.dim() != 2 or int(raw_precision.shape[1]) < 3:
        raise RuntimeError("projective interval producer expects raw_precision with shape [N,3]")
    target_precision = 1.0 / (float(sigma_px) * float(sigma_px))
    min_precision = float(model.min_precision)
    if target_precision <= min_precision:
        raise RuntimeError("projective interval sigma_px is too large for the model min_precision")
    raw_target = _inv_softplus(
        torch.full(
            (),
            target_precision - min_precision,
            dtype=raw_precision.dtype,
            device=raw_precision.device,
        )
    )
    with torch.no_grad():
        raw_precision[:, 0].fill_(raw_target)
        raw_precision[:, 1].fill_(raw_target)
    if raw_precision.requires_grad:
        mask = torch.ones_like(raw_precision.detach())
        mask[:, 0:2] = 0.0
        raw_precision.register_hook(lambda grad, mask=mask: grad * mask)
    if hasattr(model, "raw_spatial_correlation"):
        raw_spatial_correlation = model.raw_spatial_correlation
        with torch.no_grad():
            raw_spatial_correlation.zero_()
        if raw_spatial_correlation.requires_grad:
            raw_spatial_correlation.register_hook(lambda grad: torch.zeros_like(grad))
    return target_precision


def _refresh_projective_interval_cache_if_stale(
    cache: _ProjectiveIntervalFeatureRenderCache,
    state: Any,
) -> None:
    refresh = state.refresh(force=False)
    first_check = cache.staleness_check_count == 0
    cache.staleness_check_count += 1
    cache.last_support_margin_missing_tile_pairs = int(refresh.support_margin_before.missing_tile_pairs)
    cache.last_support_margin_min_slack_px = float(refresh.support_margin_before.min_boundary_slack_px)
    cache.last_support_margin_max_overshoot_px = float(refresh.support_margin_before.max_boundary_overshoot_px)
    cache.last_support_tail_alpha_bound = float(getattr(refresh, "support_tail_alpha_bound_before", 0.0))
    cache.min_support_margin_min_slack_px = (
        float(refresh.support_margin_before.min_boundary_slack_px)
        if first_check or cache.min_support_margin_min_slack_px is None
        else min(
            float(cache.min_support_margin_min_slack_px),
            float(refresh.support_margin_before.min_boundary_slack_px),
        )
    )
    cache.max_support_margin_max_overshoot_px = max(
        float(cache.max_support_margin_max_overshoot_px),
        float(refresh.support_margin_before.max_boundary_overshoot_px),
    )
    cache.max_support_tail_alpha_bound = max(
        float(cache.max_support_tail_alpha_bound),
        float(cache.last_support_tail_alpha_bound),
    )
    changed = bool(refresh.rebinned or refresh.visibility_stratified or refresh.fallback_marked)
    if changed:
        cache.stale_refresh_count += 1
    if bool(refresh.rebinned):
        cache.support_rebin_count += 1
    if bool(refresh.visibility_stratified):
        cache.visibility_stratify_count += 1
    if bool(refresh.fallback_marked):
        cache.fallback_mark_count += 1


def _render_projective_interval_feature_tubes_autograd(
    *,
    ma: torch.Tensor,
    q_uvt: torch.Tensor,
    depth0: torch.Tensor,
    depth_beta: torch.Tensor,
    opacity: torch.Tensor,
    feature: torch.Tensor,
    cfg: dict[str, Any],
    feature_config: Any,
    uvt_config: Any,
    times: torch.Tensor,
    cache: _ProjectiveIntervalFeatureRenderCache | None = None,
    global_step: int | None = None,
    refresh_every: int = 1,
    refresh_policy: str = "cadence",
    collect_timing: bool = False,
) -> _ProjectiveIntervalFeatureRender:
    if int(feature.shape[1]) != 3:
        raise RuntimeError("projective interval feature trainer route currently requires feature_uvt.feature_dim=3")
    timing_ms: dict[str, float] | None = {} if collect_timing else None
    timing_device = feature.device

    def _start_timing() -> float:
        _sync_device(timing_device)
        return time.perf_counter()

    def _finish_timing(key: str, started: float) -> None:
        _sync_device(timing_device)
        if timing_ms is not None:
            timing_ms[key] = (time.perf_counter() - started) * 1000.0

    step_index = 0 if global_step is None else int(global_step)
    rebuild = _projective_interval_cache_should_rebuild(
        cache,
        step_index=step_index,
        refresh_every=refresh_every,
        refresh_policy=refresh_policy,
    )
    stage_t0 = _start_timing() if collect_timing else 0.0
    if rebuild:
        feature_state = make_projective_cell_interval_trainer_state_from_uvt_tubes(
            ma,
            q_uvt,
            depth0,
            depth_beta,
            opacity,
            feature,
            times,
            cfg,
            feature_config=feature_config,
            uvt_config=uvt_config,
        )
        if cache is not None:
            cache.state = feature_state
            cache.last_rebuild_step = step_index
            cache.rebuild_count += 1
    else:
        feature_state = cache.state
        feature_state.atlas = make_projective_cell_interval_live_atlas_from_uvt_tubes(
            ma,
            q_uvt,
            depth0,
            depth_beta,
            opacity,
            feature,
            cfg,
            reference_atlas=feature_state.atlas,
            feature_config=feature_config,
        )
        cache.live_update_count += 1
        _refresh_projective_interval_cache_if_stale(cache, feature_state)
    if collect_timing:
        _finish_timing("feature_state_update_ms", stage_t0)
    stage_t0 = _start_timing() if collect_timing else 0.0
    feature_thwc = feature_state.render()
    if collect_timing:
        _finish_timing("feature_render_ms", stage_t0)
    alpha_color = torch.ones((int(feature.shape[0]), 3), dtype=feature.dtype, device=feature.device)
    stage_t0 = _start_timing() if collect_timing else 0.0
    if cache is None:
        alpha_state = make_projective_cell_interval_trainer_state_from_uvt_tubes(
            ma,
            q_uvt,
            depth0,
            depth_beta,
            opacity,
            alpha_color,
            times,
            cfg,
            feature_config=feature_config,
            uvt_config=uvt_config,
        )
        if collect_timing:
            _finish_timing("alpha_state_update_ms", stage_t0)
        stage_t0 = _start_timing() if collect_timing else 0.0
        alpha_thwc = alpha_state.render()
    else:
        feature_state.atlas = make_projective_cell_interval_live_atlas_from_uvt_tubes(
            ma,
            q_uvt,
            depth0,
            depth_beta,
            opacity,
            alpha_color,
            cfg,
            reference_atlas=feature_state.atlas,
            feature_config=feature_config,
        )
        cache.alpha_render_count += 1
        if collect_timing:
            _finish_timing("alpha_state_update_ms", stage_t0)
        stage_t0 = _start_timing() if collect_timing else 0.0
        alpha_thwc = feature_state.render()
    if collect_timing:
        _finish_timing("alpha_render_ms", stage_t0)
        if timing_ms is not None:
            timing_ms["projective_interval_render_ms"] = sum(float(value) for value in timing_ms.values())
    return _ProjectiveIntervalFeatureRender(
        feature_image=feature_thwc.permute(0, 3, 1, 2).contiguous(),
        alpha=alpha_thwc[..., 0].contiguous(),
        timing_ms=timing_ms,
    )


def _render_rgb_probe_chunks(
    *,
    model: nn.Module,
    rgb_probe: FeatureToColor,
    target_feature: FeatureTargetTensor,
    render_uvt_feature_tubes: Any,
    shift_ma_for_frame_chunk: Any,
    chunked_uvt_config: Any,
    uvt_config: Any,
    frames: int,
    height: int,
    width: int,
    chunk_size: int,
    adapter: str,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    if target_feature.source is None:
        raise RuntimeError("RGB probe media requires target-grid feature_target source")
    outputs: list[torch.Tensor] = []
    _sync_device(device)
    started = time.perf_counter()
    for frame_start in range(0, frames, chunk_size):
        chunk_frames = min(chunk_size, frames - frame_start)
        ma, q_uvt, depth0, depth_beta, opacity, feature = model.tensors()
        if chunk_frames == frames:
            render = render_uvt_feature_tubes(
                ma,
                q_uvt,
                depth0.detach(),
                depth_beta.detach(),
                opacity,
                feature,
                uvt_config,
            )
        else:
            ma_chunk = shift_ma_for_frame_chunk(
                ma,
                global_frames=uvt_config.frames,
                frame_start=frame_start,
                chunk_frames=chunk_frames,
            )
            render = render_uvt_feature_tubes(
                ma_chunk,
                q_uvt,
                depth0.detach(),
                depth_beta.detach(),
                opacity,
                feature,
                chunked_uvt_config(uvt_config, chunk_frames=chunk_frames),
            )
        target_start, target_frames = _target_grid_slice_for_render_chunk(
            target_frames=int(target_feature.source.shape[0]),
            render_frames=frames,
            frame_start=frame_start,
            chunk_frames=chunk_frames,
        )
        target_shape = (
            target_frames,
            int(target_feature.source.shape[1]),
            int(target_feature.source.shape[2]),
            int(target_feature.source.shape[3]),
        )
        rendered_grid = _adapt_render_to_feature_target(
            render.feature_image,
            target_shape=target_shape,
            mode=target_feature.grid_mode,
        )
        grid_rgb = rgb_probe(rendered_grid)
        full_rgb = _upsample_grid_rgb(
            grid_rgb,
            target_shape=(chunk_frames, height, width),
            mode=adapter,
        )
        outputs.append(full_rgb.permute(0, 2, 3, 1).detach().cpu())
    _sync_device(device)
    return torch.cat(outputs, dim=0).contiguous(), (time.perf_counter() - started) * 1000.0


def _support_prefix_quadratic(q_uvt: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    return (
        q_uvt[..., 0] * delta[..., 0] * delta[..., 0]
        + 2.0 * q_uvt[..., 1] * delta[..., 0] * delta[..., 1]
        + 2.0 * q_uvt[..., 2] * delta[..., 0] * delta[..., 2]
        + q_uvt[..., 3] * delta[..., 1] * delta[..., 1]
        + 2.0 * q_uvt[..., 4] * delta[..., 1] * delta[..., 2]
        + q_uvt[..., 5] * delta[..., 2] * delta[..., 2]
    )


def _support_prefix_alpha_loss(
    model: nn.Module,
    target_points: torch.Tensor,
    selected_tube_ids: torch.Tensor,
    *,
    alpha_target: float,
    total_loss_elems: int,
    alpha_threshold: float,
    max_alpha: float,
    transmittance_threshold: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if int(target_points.shape[0]) <= 0:
        raise ValueError("support_birth_split prefix alpha loss requires target points")
    if int(selected_tube_ids.numel()) <= 0:
        raise ValueError("support_birth_split prefix alpha loss requires selected tube ids")
    if int(total_loss_elems) <= 0:
        raise ValueError("support_birth_split prefix alpha total_loss_elems must be positive")
    ma, q_uvt, depth0, depth_beta, opacity, _feature = model.tensors()
    points = target_points.to(device=ma.device, dtype=ma.dtype)
    delta = points.unsqueeze(1) - ma.unsqueeze(0)
    qv = _support_prefix_quadratic(q_uvt.unsqueeze(0), delta)
    alpha = opacity.unsqueeze(0) * torch.exp(torch.clamp(-0.5 * qv, min=-80.0, max=0.0))
    alpha = torch.clamp(alpha, min=0.0, max=float(max_alpha))
    alpha = torch.where(alpha >= float(alpha_threshold), alpha, torch.zeros_like(alpha))
    with torch.no_grad():
        depth = depth0.unsqueeze(0) + ((points.unsqueeze(1) - ma.unsqueeze(0)) * depth_beta.unsqueeze(0)).sum(dim=-1)
        depth_for_sort = torch.where(alpha > 0.0, depth, torch.full_like(depth, float("inf")))
        order = torch.argsort(depth_for_sort, dim=1, stable=True)
        selected_mask_by_id = torch.zeros((int(ma.shape[0]),), dtype=torch.bool, device=ma.device)
        selected_mask_by_id.index_fill_(0, selected_tube_ids.to(device=ma.device, dtype=torch.int64), True)
        ordered_selected = selected_mask_by_id.index_select(0, order.reshape(-1)).reshape_as(order)
    ordered_alpha = alpha.gather(1, order)
    trans_after = torch.cumprod((1.0 - ordered_alpha).clamp_min(0.0), dim=1)
    prefix = torch.cat((torch.ones_like(trans_after[:, :1]), trans_after[:, :-1]), dim=1)
    weight = prefix * ordered_alpha
    if float(transmittance_threshold) > 0.0:
        weight = torch.where(prefix > float(transmittance_threshold), weight, torch.zeros_like(weight))
    selected_weight = torch.where(ordered_selected, weight, torch.zeros_like(weight))
    selected_weight_sum = selected_weight.sum(dim=1)
    final_alpha = weight.sum(dim=1).clamp(0.0, 1.0)
    diff = selected_weight_sum - float(alpha_target)
    loss = diff.square().sum() / float(total_loss_elems)
    metrics = {
        "selected_weight_mean": float(selected_weight_sum.detach().mean().cpu().item()),
        "selected_weight_share_mean": float(
            (selected_weight_sum.detach() / final_alpha.detach().clamp_min(1.0e-8)).mean().cpu().item()
        ),
        "final_alpha_mean": float(final_alpha.detach().mean().cpu().item()),
    }
    return loss, metrics


def _assert_requirements(row: dict[str, Any], cfg: dict[str, Any]) -> None:
    if bool(cfg["train"]["require_loss_decrease"]) and not bool(row["loss_decreased"]):
        raise AssertionError(f"loss did not decrease: {row['start_loss']} -> {row['end_loss']}")
    if bool(cfg["train"]["require_no_tile_overflow"]) and int(row["tile_overflow_sum"]) != 0:
        raise AssertionError(f"tile overflow is nonzero: {row['tile_overflow_sum']}")
    if bool(cfg["train"]["require_gradient_flow"]):
        required = (
            "raw_feature_grad_seen",
            "center_uv_grad_seen",
            "center_t_grad_seen",
            "velocity_uv_grad_seen",
            "raw_precision_grad_seen",
            "raw_opacity_grad_seen",
        )
        if bool(row["colorizer_grad_required"]):
            required = (*required, "colorizer_grad_seen")
        missing = [key for key in required if not bool(row[key])]
        if missing:
            raise AssertionError(f"missing gradient flow: {', '.join(missing)}")


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    cfg = resolve_config(config)
    _ensure_star_uvt_on_path()
    from torch_gsplat_bridge_star_uvt.feature_rasterize import (
        bin_uvt_feature_tubes,
        chunked_uvt_config,
        direct_atomic_feature_backward,
        direct_atomic_feature_backward_cached_bins,
        direct_atomic_feature_sparse_pixels_backward_cached_bins,
        direct_hidden_sigmoid_mse_sparse_pixels_backward_cached_bins,
        direct_hidden_sigmoid_target_area_backward_cached_bins,
        render_uvt_feature_alpha_all_pixels_with_bins,
        render_uvt_feature_sparse_pixels_with_bins,
        render_uvt_feature_tubes,
        render_uvt_feature_tubes_autograd,
        render_uvt_feature_tubes_autograd_frame_chunk,
        shift_ma_for_frame_chunk,
        sparse_hidden_sigmoid_target_area_forward_sums_cached_bins,
    )

    device = _resolve_device(str(cfg["train"]["device"]))
    if device.type != "mps":
        raise RuntimeError("STAR UVT feature Metal training currently requires MPS")
    torch.manual_seed(int(cfg["train"]["seed"]))
    feature_config, uvt_config = star_uvt_render_configs_from_cfg(cfg)
    feature_dim = int(cfg["feature_uvt"]["feature_dim"])
    projective_interval_requested = bool(cfg["feature_uvt"].get("projective_interval", {}).get("enabled", False))
    if projective_interval_requested and feature_dim != 3:
        raise RuntimeError("projective interval trainer route currently requires feature_uvt.feature_dim=3")
    projective_interval_backend = require_projective_interval_atlas_producer(
        cfg,
        feature_config,
        owner="star_uvt_feature_overfit_trainer",
        producer_available=True,
    )
    requested_render_mode = str(cfg["feature_uvt"]["render_mode"])
    backward_mode = backward_mode_for_feature_render_mode(
        requested_render_mode,
        feature_dim,
        cap_plain_gradcache=False,
    )
    sequence = _load_training_sequence(cfg, device)
    target_rgb = sequence.frames.contiguous()
    target_thwc = target_rgb.permute(0, 2, 3, 1).contiguous()
    visibility_proxy_cfg = cfg.get("visibility_proxy", {})
    visibility_proxy_enabled = bool(visibility_proxy_cfg.get("enabled", False))
    visibility_proxy_target_points: torch.Tensor | None = None
    if visibility_proxy_enabled:
        visibility_proxy_target_points = _visibility_proxy_target_points(
            target_rgb,
            target_top_fraction=float(visibility_proxy_cfg["target_top_fraction"]),
            max_points=int(visibility_proxy_cfg["max_points"]),
            grid_stride=int(visibility_proxy_cfg["grid_stride"]),
            frame_stride=int(visibility_proxy_cfg["frame_stride"]),
            device=device,
        )
    support_birth_split_cfg = cfg.get("support_birth_split", {})
    support_birth_split_enabled = bool(support_birth_split_cfg.get("enabled", False))
    support_birth_split_target_points: torch.Tensor | None = None
    support_birth_split_target_meta: dict[str, Any] = {}
    support_birth_split_alpha_sample_ms = 0.0
    feature_target_meta: dict[str, Any] | None = None
    target_feature: FeatureTargetTensor | None = None
    feature_target_load_ms = 0.0
    rgb_probe: FeatureToColor | None = None
    rgb_probe_meta: dict[str, Any] | None = None
    rgb_grid_target: torch.Tensor | None = None
    rgb_grid_target_shape: list[int] | None = None
    rgb_probe_target: torch.Tensor | None = None
    rgb_probe_target_shape: list[int] | None = None
    base_rgb_probe_loss_weight = 0.0
    rgb_probe_adapter = "trilinear"
    if _feature_target_enabled(cfg):
        _sync_device(device)
        feature_target_t0 = time.perf_counter()
        target_feature = _load_cached_feature_target(
            cfg=cfg,
            sequence_data=sequence,
            device=device,
            frames=feature_config.frames,
            height=feature_config.height,
            width=feature_config.width,
            feature_dim=feature_config.feature_dim,
        )
        _sync_device(device)
        feature_target_load_ms = (time.perf_counter() - feature_target_t0) * 1000.0
        feature_target_meta = target_feature.meta
        base_rgb_probe_loss_weight = float(cfg["feature_target"].get("rgb_probe_loss_weight", 0.0))
        rgb_probe_adapter = str(cfg["feature_target"].get("rgb_probe_target_rgb_adapter", target_feature.grid_mode))
        if target_feature.materialization == "target_grid" and target_feature.source is not None:
            rgb_grid_target = _adapt_rgb_to_grid(
                target_rgb,
                target_shape=(
                    int(target_feature.source.shape[0]),
                    int(target_feature.source.shape[2]),
                    int(target_feature.source.shape[3]),
                ),
                mode=rgb_probe_adapter,
            ).detach()
            rgb_grid_target_shape = list(rgb_grid_target.shape)
        rgb_probe, rgb_probe_meta = _load_feature_to_rgb_probe(
            cfg,
            device=device,
            feature_dim=feature_config.feature_dim,
        )
        if rgb_probe is not None:
            if target_feature.materialization != "target_grid" or target_feature.source is None:
                raise RuntimeError("RGB probe requires target-grid feature target source")
            if rgb_grid_target is None:
                raise RuntimeError("RGB probe requires target-grid RGB target")
            rgb_probe_target = rgb_grid_target
            rgb_probe_target_shape = list(rgb_probe_target.shape)
    model = build_feature_tube_model(cfg, feature_config, device=device)
    colorizer = build_feature_colorizer(cfg["colorize"], feature_dim=feature_config.feature_dim, device=device)
    configured_lr = float(cfg["train"]["lr"])
    lr_schedule = _optimizer_lr_schedule(cfg)
    optimizer = torch.optim.Adam([*model.parameters(), *colorizer.parameters()], lr=configured_lr)
    resume_checkpoint = path_or_none(cfg["train"].get("resume_checkpoint"))
    resume_state: dict[str, Any] = {
        "path": None,
        "loaded": False,
        "colorizer_loaded": False,
        "optimizer_loaded": False,
        "optimizer_lrs_loaded": [],
        "steps": None,
    }
    colorizer_init_state: dict[str, Any] = {
        "path": None,
        "loaded": False,
    }
    if resume_checkpoint is not None:
        resume_state = _load_training_checkpoint(
            resume_checkpoint,
            model=model,
            colorizer=colorizer,
            optimizer=optimizer,
            device=device,
            resume_optimizer=bool(cfg["train"]["resume_optimizer"]),
            resume_colorizer=bool(cfg["train"]["resume_colorizer"]),
        )
        _set_optimizer_lr(optimizer, configured_lr)
    colorizer_init_checkpoint = path_or_none(cfg["colorize"].get("init_checkpoint"))
    if colorizer_init_checkpoint is not None:
        colorizer_init_state = _load_colorizer_init_checkpoint(
            colorizer_init_checkpoint,
            colorizer=colorizer,
            device=device,
        )
    support_birth_split_state: dict[str, Any] = {"enabled": False}
    if support_birth_split_enabled:
        sampled_alpha: torch.Tensor | None = None
        sampled_residual: torch.Tensor | None = None
        sampled_tile_load: torch.Tensor | None = None
        if str(support_birth_split_cfg["target_point_source"]) != "top_brightness":
            _sync_device(device)
            alpha_sample_t0 = time.perf_counter()
            _frame_ids, y_ids, x_ids, pixel_ids = _support_birth_split_sample_grid(
                frames=feature_config.frames,
                height=feature_config.height,
                width=feature_config.width,
                frame_stride=int(support_birth_split_cfg["frame_stride"]),
                grid_stride=int(support_birth_split_cfg["grid_stride"]),
                device=device,
            )
            ma, q_uvt, depth0, depth_beta, opacity, _feature = model.tensors()
            needs_residual_target = "residual" in str(support_birth_split_cfg["target_point_source"])
            if needs_residual_target and getattr(colorizer, "view_condition", "none") != "none":
                raise RuntimeError("support_birth_split residual target sources require colorize.view_condition='none'")
            feature_for_sample = _feature if needs_residual_target else torch.zeros(
                (ma.shape[0], 1),
                dtype=torch.float32,
                device=device,
            )
            alpha_render = render_uvt_feature_sparse_pixels_with_bins(
                ma,
                q_uvt,
                depth0.detach(),
                depth_beta.detach(),
                opacity,
                feature_for_sample,
                pixel_ids,
                uvt_config,
            )
            sampled_alpha = alpha_render.alpha_values.reshape(
                -1,
                int(y_ids.numel()),
                int(x_ids.numel()),
            ).contiguous()
            if needs_residual_target:
                with torch.no_grad():
                    target_values = _gather_sparse_visual_rgb_values(target_rgb, pixel_ids)
                    pred_values = _compose_sparse_visual_rgb(
                        alpha_render.feature_values.detach(),
                        alpha_render.alpha_values.detach(),
                        colorizer,
                        composition="black",
                    )
                    sampled_residual = (pred_values - target_values).abs().mean(dim=1).reshape_as(sampled_alpha)
            sampled_tile_load = _support_birth_split_sampled_tile_load(
                alpha_render.tile_counts,
                frames=feature_config.frames,
                height=feature_config.height,
                width=feature_config.width,
                frame_stride=int(support_birth_split_cfg["frame_stride"]),
                grid_stride=int(support_birth_split_cfg["grid_stride"]),
                tile_x=int(uvt_config.tile_x),
                tile_y=int(uvt_config.tile_y),
                tile_t=int(uvt_config.tile_t),
            )
            _sync_device(device)
            support_birth_split_alpha_sample_ms = (time.perf_counter() - alpha_sample_t0) * 1000.0
        support_birth_split_target_points, support_birth_split_target_meta = _support_birth_split_target_points(
            target_rgb,
            target_point_source=str(support_birth_split_cfg["target_point_source"]),
            target_top_fraction=float(support_birth_split_cfg["target_top_fraction"]),
            max_points=int(support_birth_split_cfg["max_points"]),
            grid_stride=int(support_birth_split_cfg["grid_stride"]),
            frame_stride=int(support_birth_split_cfg["frame_stride"]),
            device=device,
            sampled_alpha=sampled_alpha,
            sampled_residual=sampled_residual,
            sampled_tile_load=sampled_tile_load,
            tile_capacity=int(cfg["feature_uvt"]["tile_capacity"]),
            footprint_radius_px=float(support_birth_split_cfg["support_radius_px"]),
        )
        if support_birth_split_target_points is None:
            raise RuntimeError("support_birth_split enabled but target points are missing")
        support_birth_split_target_features: torch.Tensor | None = None
        if str(support_birth_split_cfg["feature_init_mode"]) != "preserve":
            if target_feature is None or target_feature.materialization != "target_grid" or target_feature.source is None:
                raise RuntimeError("support_birth_split target feature init requires target-grid feature target source")
            support_birth_split_target_features = _support_birth_split_sample_target_grid_features(
                target_feature.source,
                support_birth_split_target_points,
                frames=feature_config.frames,
                height=feature_config.height,
                width=feature_config.width,
                mode=target_feature.grid_mode,
            )
        support_birth_split_state = _apply_support_birth_split(
            model,
            support_birth_split_target_points,
            reallocate_tubes=int(support_birth_split_cfg["reallocate_tubes"]),
            support_radius_px=float(support_birth_split_cfg["support_radius_px"]),
            support_shape=str(support_birth_split_cfg["support_shape"]),
            support_radius_along_px=float(support_birth_split_cfg["support_radius_along_px"]),
            support_radius_across_px=float(support_birth_split_cfg["support_radius_across_px"]),
            support_precision_radius_px=float(support_birth_split_cfg["support_precision_radius_px"]),
            temporal_radius_frames=float(support_birth_split_cfg["temporal_radius_frames"]),
            opacity=float(support_birth_split_cfg["opacity"]),
            max_alpha=float(cfg["feature_uvt"]["max_alpha"]),
            tube_selection=str(support_birth_split_cfg["tube_selection"]),
            center_strategy=str(support_birth_split_cfg["center_strategy"]),
            center_count=int(support_birth_split_cfg["center_count"]),
            tube_allocation=str(support_birth_split_cfg["tube_allocation"]),
            target_point_features=support_birth_split_target_features,
            feature_init_mode=str(support_birth_split_cfg["feature_init_mode"]),
        )
        support_birth_split_state["target_point_meta"] = support_birth_split_target_meta
        support_birth_split_state["alpha_sample_ms"] = support_birth_split_alpha_sample_ms
        if bool(support_birth_split_cfg["tile_overflow_repair_enabled"]):
            ma, q_uvt, depth0, depth_beta, opacity, feature = model.tensors()
            repair_render = render_uvt_feature_tubes(
                ma,
                q_uvt,
                depth0.detach(),
                depth_beta.detach(),
                opacity,
                feature,
                uvt_config,
                return_bins=True,
            )
            if repair_render.tile_tube_ids is None:
                raise RuntimeError("support_birth_split tile-overflow repair requires tile_tube_ids")
            selected_tube_ids = torch.tensor(
                support_birth_split_state["selected_tube_ids"],
                dtype=torch.int64,
                device=repair_render.tile_counts.device,
            )
            repair_state = _support_birth_split_repair_tile_overflow_ids(
                repair_render.tile_counts,
                repair_render.tile_tube_ids,
                selected_tube_ids,
                tile_capacity=int(cfg["feature_uvt"]["tile_capacity"]),
                max_drops=int(support_birth_split_cfg["tile_overflow_repair_max_drops"]),
                guard_refs=int(support_birth_split_cfg["tile_overflow_repair_guard_refs"]),
            )
            dropped_tube_ids = torch.tensor(
                repair_state["dropped_tube_ids"],
                dtype=torch.int64,
                device=model.raw_opacity.device,
            )
            if int(dropped_tube_ids.numel()) > 0:
                _support_birth_split_set_tube_opacity(
                    model,
                    dropped_tube_ids,
                    opacity=float(support_birth_split_cfg["tile_overflow_repair_opacity"]),
                )
                ma, q_uvt, depth0, depth_beta, opacity, feature = model.tensors()
                repaired_render = render_uvt_feature_tubes(
                    ma,
                    q_uvt,
                    depth0.detach(),
                    depth_beta.detach(),
                    opacity,
                    feature,
                    uvt_config,
                    return_bins=True,
                )
                repaired_stats = _tile_load_stats(
                    tile_counts=[repaired_render.tile_counts],
                    tile_overflow=[repaired_render.tile_overflow],
                    tile_unstable=[repaired_render.tile_unstable],
                    tile_capacity=int(cfg["feature_uvt"]["tile_capacity"]),
                )
                repair_state["post_repair_overflow_tile_count"] = int(repaired_stats["overflow_tile_count"])
                repair_state["post_repair_max_tile_count"] = int(repaired_stats["max_tile_count"])
            support_birth_split_state["tile_overflow_repair"] = repair_state

    support_target_alpha_loss_weight = (
        float(support_birth_split_cfg.get("target_alpha_loss_weight", 0.0)) if support_birth_split_enabled else 0.0
    )
    support_target_alpha_enabled = support_birth_split_enabled and support_target_alpha_loss_weight > 0.0
    support_target_alpha_target = (
        float(support_birth_split_cfg.get("target_alpha_target", 1.0)) if support_target_alpha_enabled else 1.0
    )
    support_target_alpha_points: torch.Tensor | None = None
    if support_target_alpha_enabled:
        if support_birth_split_target_points is None:
            raise RuntimeError("support_birth_split target alpha loss requires support birth target points")
        support_target_alpha_points = support_birth_split_target_points
        support_target_alpha_max_points = int(support_birth_split_cfg.get("target_alpha_max_points", 0))
        if support_target_alpha_max_points <= 0:
            raise ValueError("support_birth_split.target_alpha_max_points must be positive")
        if int(support_target_alpha_points.shape[0]) > support_target_alpha_max_points:
            select = (
                torch.linspace(
                    0,
                    int(support_target_alpha_points.shape[0]) - 1,
                    support_target_alpha_max_points,
                    device=support_target_alpha_points.device,
                )
                .round()
                .to(torch.int64)
            )
            support_target_alpha_points = support_target_alpha_points.index_select(0, select)
    total_support_target_alpha_loss_elems = 1 if support_target_alpha_points is None else int(
        support_target_alpha_points.shape[0]
    )
    support_target_area_loss_weight = (
        float(support_birth_split_cfg.get("target_area_loss_weight", 0.0)) if support_birth_split_enabled else 0.0
    )
    support_target_area_enabled = support_birth_split_enabled and support_target_area_loss_weight > 0.0
    support_target_area_points: torch.Tensor | None = None
    support_target_area_patch_shape = tuple(
        int(item) for item in support_birth_split_cfg.get("target_area_patch_shape", (2, 2))
    )
    support_target_area_vjp_mode = str(support_birth_split_cfg.get("target_area_vjp_mode", "manual_hidden64_star_only"))
    support_target_area_composition = str(support_birth_split_cfg.get("target_area_composition", "black"))
    if support_target_area_enabled:
        if support_birth_split_target_points is None:
            raise RuntimeError("support_birth_split target area loss requires support birth target points")
        support_target_area_points = support_birth_split_target_points
        support_target_area_max_points = int(support_birth_split_cfg.get("target_area_max_points", 0))
        if support_target_area_max_points <= 0:
            raise ValueError("support_birth_split.target_area_max_points must be positive")
        if int(support_target_area_points.shape[0]) > support_target_area_max_points:
            select = (
                torch.linspace(
                    0,
                    int(support_target_area_points.shape[0]) - 1,
                    support_target_area_max_points,
                    device=support_target_area_points.device,
                )
                .round()
                .to(torch.int64)
            )
            support_target_area_points = support_target_area_points.index_select(0, select)
    total_support_target_area_loss_elems = 1 if support_target_area_points is None else int(
        support_target_area_points.shape[0] * 3
    )
    support_prefix_alpha_loss_weight = (
        float(support_birth_split_cfg.get("prefix_alpha_loss_weight", 0.0)) if support_birth_split_enabled else 0.0
    )
    support_prefix_alpha_enabled = support_birth_split_enabled and support_prefix_alpha_loss_weight > 0.0
    support_prefix_alpha_target = (
        float(support_birth_split_cfg.get("prefix_alpha_target", 1.0)) if support_prefix_alpha_enabled else 1.0
    )
    support_prefix_alpha_points: torch.Tensor | None = None
    support_prefix_alpha_selected_ids: torch.Tensor | None = None
    if support_prefix_alpha_enabled:
        if support_birth_split_target_points is None:
            raise RuntimeError("support_birth_split prefix alpha loss requires support birth target points")
        selected_ids = support_birth_split_state.get("selected_tube_ids", [])
        support_prefix_alpha_selected_ids = torch.tensor(
            [int(item) for item in selected_ids],
            dtype=torch.int64,
            device=device,
        )
        if int(support_prefix_alpha_selected_ids.numel()) <= 0:
            raise RuntimeError("support_birth_split prefix alpha loss requires selected_tube_ids")
        support_prefix_alpha_points = support_birth_split_target_points
        support_prefix_alpha_max_points = int(support_birth_split_cfg.get("prefix_alpha_max_points", 0))
        if support_prefix_alpha_max_points <= 0:
            raise ValueError("support_birth_split.prefix_alpha_max_points must be positive")
        if int(support_prefix_alpha_points.shape[0]) > support_prefix_alpha_max_points:
            select = (
                torch.linspace(
                    0,
                    int(support_prefix_alpha_points.shape[0]) - 1,
                    support_prefix_alpha_max_points,
                    device=support_prefix_alpha_points.device,
                )
                .round()
                .to(torch.int64)
            )
            support_prefix_alpha_points = support_prefix_alpha_points.index_select(0, select)
    total_support_prefix_alpha_loss_elems = 1 if support_prefix_alpha_points is None else int(
        support_prefix_alpha_points.shape[0]
    )

    projective_interval_runtime_enabled = bool(projective_interval_backend.enabled)
    projective_interval_spatial_precision: float | None = None
    projective_interval_times = (
        _projective_interval_times(feature_config.frames, device) if projective_interval_runtime_enabled else None
    )
    projective_interval_cache = (
        _ProjectiveIntervalFeatureRenderCache() if projective_interval_runtime_enabled else None
    )
    if projective_interval_runtime_enabled:
        projective_interval_spatial_precision_locked = not projective_interval_backend.allow_anisotropic_spatial_precision
        if projective_interval_spatial_precision_locked:
            projective_interval_spatial_precision = _lock_projective_interval_spatial_precision(
                model,
                projective_interval_backend.sigma_px,
            )
    else:
        projective_interval_spatial_precision_locked = False

    losses: list[float] = []
    rgb_losses: list[float] = []
    feature_target_losses: list[float] = []
    rgb_grid_losses: list[float] = []
    rgb_probe_losses: list[float] = []
    dense_alpha_losses: list[float] = []
    sparse_visual_losses: list[float] = []
    sparse_visual_alpha_losses: list[float] = []
    sparse_visual_black_hole_losses: list[float] = []
    support_target_alpha_losses: list[float] = []
    support_target_area_losses: list[float] = []
    support_prefix_alpha_losses: list[float] = []
    support_prefix_alpha_selected_weight_means: list[float] = []
    support_prefix_alpha_selected_share_means: list[float] = []
    support_prefix_alpha_final_alpha_means: list[float] = []
    visibility_proxy_losses: list[float] = []
    timings: list[dict[str, float]] = []
    final_grad_norms: dict[str, float] = {}
    frame_chunk_size = cfg["train"]["frame_chunk_size"]
    chunk_size = feature_config.frames if frame_chunk_size is None else int(frame_chunk_size)
    if chunk_size <= 0:
        raise ValueError("train.frame_chunk_size must be null or positive")
    chunk_size = min(chunk_size, feature_config.frames)
    total_loss_elems = target_rgb.numel()
    total_feature_loss_elems = 0 if target_feature is None else int(target_feature.numel)
    total_rgb_grid_loss_elems = 0 if rgb_grid_target is None else int(rgb_grid_target.numel())
    total_rgb_probe_loss_elems = 0 if rgb_probe_target is None else int(rgb_probe_target.numel())
    dense_alpha_cfg = cfg.get("dense_alpha", {})
    dense_alpha_enabled = bool(dense_alpha_cfg.get("enabled", False))
    dense_alpha_loss_weight = float(dense_alpha_cfg.get("loss_weight", 0.0)) if dense_alpha_enabled else 0.0
    dense_alpha_target = float(dense_alpha_cfg.get("alpha_target", 1.0)) if dense_alpha_enabled else 1.0
    dense_alpha_backward_mode = str(dense_alpha_cfg.get("backward_mode", "gradcache_skip_feature_grad"))
    dense_alpha_render_mode = str(dense_alpha_cfg.get("render_mode", "dense_f32"))
    total_dense_alpha_loss_elems = int(feature_config.frames * feature_config.height * feature_config.width)
    visibility_proxy_loss_weight = (
        float(visibility_proxy_cfg.get("loss_weight", 0.0)) if visibility_proxy_enabled else 0.0
    )
    visibility_proxy_center_weight = float(visibility_proxy_cfg.get("center_weight", 1.0))
    visibility_proxy_support_weight = float(visibility_proxy_cfg.get("support_weight", 0.0))
    visibility_proxy_support_epsilon = float(visibility_proxy_cfg.get("support_epsilon", 1.0e-4))
    visibility_proxy_scale_px = float(visibility_proxy_cfg.get("scale_px", 64.0))
    visibility_proxy_temperature = float(visibility_proxy_cfg.get("temperature", 0.75))
    visibility_proxy_velocity_penalty = float(visibility_proxy_cfg.get("velocity_penalty", 0.0025))
    sparse_visual_cfg = cfg.get("sparse_visual", {})
    sparse_visual_enabled = bool(sparse_visual_cfg.get("enabled", False))
    sparse_visual_loss_weight = float(sparse_visual_cfg.get("loss_weight", 0.0)) if sparse_visual_enabled else 0.0
    sparse_visual_alpha_loss_weight = (
        float(sparse_visual_cfg.get("alpha_loss_weight", 0.0)) if sparse_visual_enabled else 0.0
    )
    sparse_visual_alpha_target = float(sparse_visual_cfg.get("alpha_target", 1.0)) if sparse_visual_enabled else 1.0
    sparse_visual_black_hole_loss_weight = (
        float(sparse_visual_cfg.get("black_hole_loss_weight", 0.0)) if sparse_visual_enabled else 0.0
    )
    sparse_visual_composition = str(sparse_visual_cfg.get("composition", "black"))
    sparse_visual_pixel_source = str(sparse_visual_cfg.get("pixel_source", "stratified_grid"))
    sparse_visual_loss_basis = str(sparse_visual_cfg.get("loss_basis", "pixel"))
    sparse_visual_loss_vjp_mode = str(sparse_visual_cfg.get("loss_vjp_mode", "autograd"))
    native_sparse_visual_vjp_enabled = sparse_visual_loss_vjp_mode in NATIVE_SPARSE_VISUAL_LOSS_VJP_MODES
    native_target_area_sparse_visual_vjp_enabled = (
        sparse_visual_loss_vjp_mode in NATIVE_TARGET_AREA_SPARSE_VISUAL_LOSS_VJP_MODES
    )
    sparse_visual_grid_shape = tuple(int(item) for item in sparse_visual_cfg.get("sample_grid_shape", (0, 0, 0)))
    sparse_visual_patch_shape = tuple(int(item) for item in sparse_visual_cfg.get("patch_shape", (1, 1)))
    sparse_visual_patch_phase_shape = tuple(int(item) for item in sparse_visual_cfg.get("patch_phase_shape", (1, 1)))
    alpha_background_train_strategy = str(cfg["alpha_background"]["train_strategy"])
    alpha_background_eval_strategy = str(cfg["alpha_background"]["eval_strategy"])
    alpha_background_sample_scope = str(cfg["alpha_background"]["sample_scope"])
    total_sparse_visual_loss_elems = 1
    total_sparse_visual_alpha_loss_elems = 1
    total_sparse_visual_black_hole_loss_elems = 1
    if sparse_visual_enabled:
        expected_sparse_visual_loss_samples = 0
        expected_sparse_visual_alpha_samples = 0
        for frame_start in range(0, feature_config.frames, chunk_size):
            chunk_frames = min(chunk_size, feature_config.frames - frame_start)
            pixel_count = int(
                _sparse_visual_pixel_ids_for_chunk(
                    pixel_source=sparse_visual_pixel_source,
                    chunk_frames=chunk_frames,
                    height=feature_config.height,
                    width=feature_config.width,
                    render_frames=feature_config.frames,
                    frame_start=frame_start,
                    sample_grid_shape=sparse_visual_grid_shape,
                    patch_shape=sparse_visual_patch_shape,
                    device=device,
                ).numel()
            )
            expected_sparse_visual_alpha_samples += pixel_count
            expected_sparse_visual_loss_samples += _sparse_visual_loss_sample_count(
                pixel_count,
                loss_basis=sparse_visual_loss_basis,
                patch_shape=sparse_visual_patch_shape,
            )
        total_sparse_visual_loss_elems = max(expected_sparse_visual_loss_samples * 3, 1)
        total_sparse_visual_alpha_loss_elems = max(expected_sparse_visual_alpha_samples, 1)
        total_sparse_visual_black_hole_loss_elems = max(expected_sparse_visual_loss_samples, 1)
    base_rgb_loss_weight = _rgb_loss_weight(cfg)
    base_rgb_grid_loss_weight = (
        float(cfg["feature_target"].get("rgb_grid_loss_weight", 0.0)) if _feature_target_enabled(cfg) else 0.0
    )
    base_feature_loss_weight = float(cfg["feature_target"]["loss_weight"]) if _feature_target_enabled(cfg) else 0.0
    weight_schedule = _feature_target_weight_schedule(cfg)
    sparse_visual_trains_colorizer = sparse_visual_enabled and "star_only" not in sparse_visual_loss_vjp_mode
    support_target_area_trains_colorizer = (
        support_target_area_enabled and "star_only" not in support_target_area_vjp_mode
    )
    colorizer_grad_required = (
        any(stage.rgb_loss_weight > 0.0 or stage.rgb_grid_loss_weight > 0.0 for stage in weight_schedule)
        or sparse_visual_trains_colorizer
        or support_target_area_trains_colorizer
    )
    rgb_grid_loss_required = any(stage.rgb_grid_loss_weight > 0.0 for stage in weight_schedule)
    rgb_probe_loss_required = any(stage.rgb_probe_loss_weight > 0.0 for stage in weight_schedule)
    feature_loss_type = str(cfg["feature_target"]["loss_type"]) if _feature_target_enabled(cfg) else "mse"
    feature_target_image_vjp_mode = (
        str(cfg["feature_target"].get("image_vjp_mode", "autograd")) if _feature_target_enabled(cfg) else "autograd"
    )
    if projective_interval_runtime_enabled:
        if chunk_size != feature_config.frames:
            raise RuntimeError("projective interval trainer route currently requires train.frame_chunk_size=null")
        if feature_target_image_vjp_mode != "autograd":
            raise RuntimeError("projective interval trainer route currently requires feature_target.image_vjp_mode=autograd")
    sparse_image_vjp_enabled = feature_target_image_vjp_mode in {
        "analytic_sparse_pixels",
        "analytic_sparse_grid",
        "analytic_sparse_grid_forward",
        "analytic_sparse_grid_forward_batched",
    }
    global_step_offset = int(cfg["train"]["global_step_offset"])
    step_rgb_loss_weights: list[float] = []
    step_feature_target_loss_weights: list[float] = []
    step_rgb_grid_loss_weights: list[float] = []
    step_rgb_probe_loss_weights: list[float] = []
    step_lrs: list[float] = []
    step_global_steps: list[int] = []
    sparse_pixel_counts: list[int] = []
    sparse_pixel_fractions: list[float] = []
    sparse_visual_pixel_counts: list[int] = []
    sparse_visual_loss_sample_counts: list[int] = []
    sparse_visual_alpha_sample_counts: list[int] = []
    support_target_alpha_sample_counts: list[int] = []
    support_target_area_sample_counts: list[int] = []
    support_prefix_alpha_sample_counts: list[int] = []
    sparse_visual_pixel_fractions: list[float] = []
    sparse_visual_patch_phases: list[list[int]] = []
    trace_global_steps = set(int(step) for step in cfg["train"]["trace_global_steps"])
    chunk_traces: list[dict[str, Any]] = []

    run = init_wandb_run(cfg)
    try:
        for _step in range(int(cfg["train"]["steps"])):
            global_step = global_step_offset + _step
            weight_stage = _feature_target_weights_for_step(weight_schedule, global_step)
            lr_stage = _optimizer_lr_for_step(lr_schedule, global_step)
            _set_optimizer_lr(optimizer, lr_stage.lr)
            rgb_loss_weight = weight_stage.rgb_loss_weight
            feature_loss_weight = weight_stage.loss_weight
            rgb_grid_loss_weight = weight_stage.rgb_grid_loss_weight
            rgb_probe_loss_weight = weight_stage.rgb_probe_loss_weight
            step_global_steps.append(global_step)
            step_rgb_loss_weights.append(rgb_loss_weight)
            step_feature_target_loss_weights.append(feature_loss_weight)
            step_rgb_grid_loss_weights.append(rgb_grid_loss_weight)
            step_rgb_probe_loss_weights.append(rgb_probe_loss_weight)
            step_lrs.append(lr_stage.lr)
            optimizer.zero_grad(set_to_none=True)
            t0 = time.perf_counter()
            loss_value = 0.0
            rgb_loss_value = 0.0
            feature_target_loss_value = 0.0
            rgb_grid_loss_value = 0.0
            rgb_probe_loss_value = 0.0
            render_forward_ms = 0.0
            colorize_loss_ms = 0.0
            feature_target_ms = 0.0
            rgb_grid_loss_ms = 0.0
            rgb_probe_loss_ms = 0.0
            sparse_pack_ms = 0.0
            dense_alpha_render_ms = 0.0
            dense_alpha_loss_ms = 0.0
            dense_alpha_backward_ms = 0.0
            dense_alpha_loss_value = 0.0
            sparse_visual_render_ms = 0.0
            sparse_visual_loss_ms = 0.0
            sparse_visual_backward_ms = 0.0
            sparse_visual_loss_value = 0.0
            sparse_visual_alpha_loss_value = 0.0
            sparse_visual_black_hole_loss_value = 0.0
            support_target_alpha_loss_value = 0.0
            support_target_area_loss_value = 0.0
            support_prefix_alpha_loss_value = 0.0
            support_prefix_alpha_selected_weight_mean_value = 0.0
            support_prefix_alpha_selected_share_mean_value = 0.0
            support_prefix_alpha_final_alpha_mean_value = 0.0
            visibility_proxy_ms = 0.0
            visibility_proxy_loss_value = 0.0
            step_sparse_pixel_count = 0
            step_sparse_total_pixels = 0
            step_sparse_visual_pixel_count = 0
            step_sparse_visual_loss_sample_count = 0
            step_sparse_visual_alpha_sample_count = 0
            step_support_target_alpha_sample_count = 0
            step_support_target_area_sample_count = 0
            step_support_prefix_alpha_sample_count = 0
            step_sparse_visual_total_pixels = 0
            sparse_visual_patch_phase = _sparse_visual_patch_phase_for_step(
                pixel_source=sparse_visual_pixel_source,
                global_step=global_step,
                patch_phase_shape=sparse_visual_patch_phase_shape,
            )
            backward_ms = 0.0
            support_target_alpha_ms = 0.0
            support_target_area_ms = 0.0
            support_prefix_alpha_ms = 0.0
            last_backward_end = t0
            trace_chunks: list[dict[str, Any]] | None = [] if global_step in trace_global_steps else None
            use_batched_sparse_forward = feature_target_image_vjp_mode == "analytic_sparse_grid_forward_batched"
            for frame_start in range(0, feature_config.frames, chunk_size):
                if use_batched_sparse_forward:
                    if target_feature is None:
                        raise RuntimeError("analytic_sparse_grid_forward_batched requires feature target")
                    if rgb_loss_weight > 0.0:
                        raise RuntimeError("analytic_sparse_grid_forward_batched does not support RGB loss")
                    batched_chunks: list[dict[str, Any]] = []
                    frame_starts: list[int] = []
                    chunk_frame_counts: list[int] = []
                    for batch_frame_start in range(0, feature_config.frames, chunk_size):
                        batch_chunk_frames = min(chunk_size, feature_config.frames - batch_frame_start)
                        target_feature_chunk_for_sparse = target_feature.chunk(batch_frame_start, batch_chunk_frames)
                        sparse_forward_pixel_ids = _sparse_target_grid_pixel_ids(
                            input_shape=(
                                int(batch_chunk_frames),
                                int(feature_config.feature_dim),
                                int(feature_config.height),
                                int(feature_config.width),
                            ),
                            target_shape=tuple(int(item) for item in target_feature_chunk_for_sparse.shape),
                            mode=target_feature.grid_mode,
                            device=device,
                        )
                        ma, q_uvt, depth0, depth_beta, opacity, feature = model.tensors()
                        if batch_chunk_frames == feature_config.frames:
                            render_inputs = (
                                ma,
                                q_uvt,
                                depth0.detach(),
                                depth_beta.detach(),
                                opacity,
                                feature,
                            )
                            render_config = uvt_config
                        else:
                            ma_chunk = shift_ma_for_frame_chunk(
                                ma,
                                global_frames=feature_config.frames,
                                frame_start=batch_frame_start,
                                chunk_frames=batch_chunk_frames,
                            )
                            render_inputs = (
                                ma_chunk,
                                q_uvt,
                                depth0.detach(),
                                depth_beta.detach(),
                                opacity,
                                feature,
                            )
                            render_config = chunked_uvt_config(uvt_config, chunk_frames=batch_chunk_frames)
                        _sync_device(device)
                        chunk_t0 = time.perf_counter()
                        render = render_uvt_feature_sparse_pixels_with_bins(
                            *render_inputs,
                            sparse_forward_pixel_ids,
                            render_config,
                        )
                        _sync_device(device)
                        chunk_t1 = time.perf_counter()
                        render_forward_ms += (chunk_t1 - chunk_t0) * 1000.0
                        target_start, target_frames = _target_grid_slice_for_render_chunk(
                            target_frames=int(target_feature.source.shape[0]),
                            render_frames=feature_config.frames,
                            frame_start=batch_frame_start,
                            chunk_frames=batch_chunk_frames,
                        )
                        probe_target_start: int | None = None
                        probe_target_frames: int | None = None
                        if rgb_probe_target is not None:
                            probe_target_start, probe_target_frames = _target_grid_slice_for_render_chunk(
                                target_frames=int(rgb_probe_target.shape[0]),
                                render_frames=feature_config.frames,
                                frame_start=batch_frame_start,
                                chunk_frames=batch_chunk_frames,
                            )
                        batched_chunks.append(
                            {
                                "frame_start": batch_frame_start,
                                "chunk_frames": batch_chunk_frames,
                                "target_start": target_start,
                                "target_frames": target_frames,
                                "probe_target_start": probe_target_start,
                                "probe_target_frames": probe_target_frames,
                                "render_forward_ms": (chunk_t1 - chunk_t0) * 1000.0,
                                "render_inputs": render_inputs,
                                "render_config": render_config,
                                "render": render,
                            }
                        )
                        frame_starts.append(batch_frame_start)
                        chunk_frame_counts.append(batch_chunk_frames)
                    _sync_device(device)
                    loss_t0 = time.perf_counter()
                    batched_vjp = _manual_batched_sparse_target_grid_loss_and_vjp(
                        [chunk["render"].feature_values for chunk in batched_chunks],
                        target_feature=target_feature,
                        colorizer=colorizer,
                        rgb_grid_target=rgb_grid_target,
                        rgb_probe=rgb_probe,
                        rgb_probe_target=rgb_probe_target,
                        feature_config=feature_config,
                        frame_starts=frame_starts,
                        chunk_frames=chunk_frame_counts,
                        feature_loss_type=feature_loss_type,
                        feature_loss_weight=feature_loss_weight,
                        rgb_grid_loss_weight=rgb_grid_loss_weight,
                        rgb_probe_loss_weight=rgb_probe_loss_weight,
                        total_feature_loss_elems=total_feature_loss_elems,
                        total_rgb_grid_loss_elems=total_rgb_grid_loss_elems,
                        total_rgb_probe_loss_elems=total_rgb_probe_loss_elems,
                        device=device,
                    )
                    _sync_device(device)
                    loss_t1 = time.perf_counter()
                    loss_value += float(batched_vjp.loss.detach().cpu().item())
                    feature_target_loss_value += batched_vjp.feature_target_loss
                    rgb_grid_loss_value += batched_vjp.rgb_grid_loss
                    rgb_probe_loss_value += batched_vjp.rgb_probe_loss
                    feature_target_ms += batched_vjp.feature_target_ms + batched_vjp.image_vjp_ms
                    rgb_grid_loss_ms += batched_vjp.rgb_grid_loss_ms
                    rgb_probe_loss_ms += batched_vjp.rgb_probe_loss_ms
                    colorize_loss_ms += (loss_t1 - loss_t0) * 1000.0
                    for chunk, sparse_pack in zip(batched_chunks, batched_vjp.sparse_packs, strict=True):
                        render = chunk["render"]
                        render_inputs = chunk["render_inputs"]
                        render_config = chunk["render_config"]
                        if render.tile_tube_ids is None or render.tile_depths is None:
                            raise RuntimeError("analytic_sparse_grid_forward_batched requires render bins")
                        step_sparse_pixel_count += sparse_pack.pixel_count
                        step_sparse_total_pixels += sparse_pack.total_pixels
                        _sync_device(device)
                        chunk_t2 = time.perf_counter()
                        grad_ma, grad_q, grad_opacity, grad_feature, _tile_unstable = (
                            direct_atomic_feature_sparse_pixels_backward_cached_bins(
                                *render_inputs,
                                sparse_pack.pixel_ids,
                                sparse_pack.grad_feature_values,
                                sparse_pack.grad_alpha_values,
                                render.tile_counts,
                                render.tile_tube_ids,
                                render.tile_depths,
                                render.tile_unstable,
                                render_config,
                            )
                        )
                        torch.autograd.backward(
                            (render_inputs[0], render_inputs[1], render_inputs[4], render_inputs[5]),
                            (grad_ma, grad_q, grad_opacity, grad_feature),
                        )
                        _sync_device(device)
                        chunk_t3 = time.perf_counter()
                        backward_ms += (chunk_t3 - chunk_t2) * 1000.0
                        last_backward_end = chunk_t3
                        if trace_chunks is not None:
                            trace_chunks.append(
                                {
                                    "frame_start": int(chunk["frame_start"]),
                                    "chunk_frames": int(chunk["chunk_frames"]),
                                    "target_start": chunk["target_start"],
                                    "target_frames": chunk["target_frames"],
                                    "probe_target_start": chunk["probe_target_start"],
                                    "probe_target_frames": chunk["probe_target_frames"],
                                    "weighted_loss": None,
                                    "rgb_loss": 0.0,
                                    "feature_target_loss": None,
                                    "rgb_grid_loss": batched_vjp.rgb_grid_loss,
                                    "rgb_probe_loss": None,
                                    "render_forward_ms": float(chunk["render_forward_ms"]),
                                    "colorize_loss_ms": 0.0,
                                    "feature_target_ms": 0.0,
                                    "rgb_grid_loss_ms": batched_vjp.rgb_grid_loss_ms,
                                    "rgb_probe_loss_ms": 0.0,
                                    "sparse_pack_ms": 0.0,
                                    "dense_alpha_render_ms": 0.0,
                                    "dense_alpha_loss_ms": 0.0,
                                    "dense_alpha_backward_ms": 0.0,
                                    "sparse_pixel_count": sparse_pack.pixel_count,
                                    "sparse_pixel_fraction": (
                                        0.0
                                        if sparse_pack.total_pixels <= 0
                                        else float(sparse_pack.pixel_count) / float(sparse_pack.total_pixels)
                                    ),
                                    "backward_ms": (chunk_t3 - chunk_t2) * 1000.0,
                                }
                            )
                    break
                chunk_frames = min(chunk_size, feature_config.frames - frame_start)
                chunk_feature_target_ms = 0.0
                chunk_rgb_grid_loss_ms = 0.0
                chunk_rgb_probe_loss_ms = 0.0
                chunk_sparse_pack_ms = 0.0
                chunk_sparse_pixel_count: int | None = None
                chunk_sparse_pixel_fraction: float | None = None
                chunk_rgb_loss_value = 0.0
                chunk_feature_target_loss_value = 0.0
                chunk_rgb_grid_loss_value = 0.0
                chunk_rgb_probe_loss_value = 0.0
                chunk_target_start: int | None = None
                chunk_target_frames: int | None = None
                chunk_probe_target_start: int | None = None
                chunk_probe_target_frames: int | None = None
                chunk_projective_interval_timing_ms: dict[str, float] | None = None
                ma, q_uvt, depth0, depth_beta, opacity, feature = model.tensors()
                _sync_device(device)
                chunk_t0 = time.perf_counter()
                use_sparse_forward = feature_target_image_vjp_mode == "analytic_sparse_grid_forward"
                use_analytic_image_vjp = (
                    feature_target_image_vjp_mode
                    in {"analytic", "analytic_sparse_pixels", "analytic_sparse_grid", "analytic_sparse_grid_forward"}
                    and target_feature is not None
                    and (
                        feature_loss_weight > 0.0
                        or rgb_grid_loss_weight > 0.0
                        or rgb_probe_loss_weight > 0.0
                    )
                )
                if use_analytic_image_vjp:
                    if rgb_loss_weight > 0.0:
                        raise RuntimeError("analytic feature_target.image_vjp_mode does not support RGB loss")
                    if chunk_frames == feature_config.frames:
                        render_inputs = (
                            ma,
                            q_uvt,
                            depth0.detach(),
                            depth_beta.detach(),
                            opacity,
                            feature,
                        )
                        render_config = uvt_config
                    else:
                        ma_chunk = shift_ma_for_frame_chunk(
                            ma,
                            global_frames=feature_config.frames,
                            frame_start=frame_start,
                            chunk_frames=chunk_frames,
                        )
                        render_inputs = (
                            ma_chunk,
                            q_uvt,
                            depth0.detach(),
                            depth_beta.detach(),
                            opacity,
                            feature,
                        )
                        render_config = chunked_uvt_config(uvt_config, chunk_frames=chunk_frames)
                    if use_sparse_forward:
                        target_feature_chunk_for_sparse = target_feature.chunk(frame_start, chunk_frames)
                        sparse_forward_pixel_ids = _sparse_target_grid_pixel_ids(
                            input_shape=(
                                int(chunk_frames),
                                int(feature_config.feature_dim),
                                int(feature_config.height),
                                int(feature_config.width),
                            ),
                            target_shape=tuple(int(item) for item in target_feature_chunk_for_sparse.shape),
                            mode=target_feature.grid_mode,
                            device=device,
                        )
                        render = render_uvt_feature_sparse_pixels_with_bins(
                            *render_inputs,
                            sparse_forward_pixel_ids,
                            render_config,
                        )
                    else:
                        render = render_uvt_feature_tubes(
                            *render_inputs,
                            render_config,
                            return_bins=sparse_image_vjp_enabled,
                        )
                elif projective_interval_runtime_enabled:
                    if projective_interval_times is None:
                        raise RuntimeError("projective interval times were not initialized")
                    render = _render_projective_interval_feature_tubes_autograd(
                        ma=ma,
                        q_uvt=q_uvt,
                        depth0=depth0.detach(),
                        depth_beta=depth_beta.detach(),
                        opacity=opacity,
                        feature=feature,
                        cfg=cfg,
                        feature_config=feature_config,
                        uvt_config=uvt_config,
                        times=projective_interval_times,
                        cache=projective_interval_cache,
                        global_step=global_step,
                        refresh_every=projective_interval_backend.refresh_every,
                        refresh_policy=projective_interval_backend.refresh_policy,
                        collect_timing=trace_chunks is not None,
                    )
                    chunk_projective_interval_timing_ms = render.timing_ms
                elif chunk_frames == feature_config.frames:
                    render = render_uvt_feature_tubes_autograd(
                        ma,
                        q_uvt,
                        depth0.detach(),
                        depth_beta.detach(),
                        opacity,
                        feature,
                        uvt_config,
                        backward_mode=backward_mode,
                    )
                else:
                    render = render_uvt_feature_tubes_autograd_frame_chunk(
                        ma,
                        q_uvt,
                        depth0.detach(),
                        depth_beta.detach(),
                        opacity,
                        feature,
                        uvt_config,
                        frame_start=frame_start,
                        chunk_frames=chunk_frames,
                        backward_mode=backward_mode,
                    )
                _sync_device(device)
                chunk_t1 = time.perf_counter()
                loss_base = render.feature_values if use_sparse_forward else render.feature_image
                loss = loss_base.new_zeros(())
                if use_analytic_image_vjp:
                    if target_feature is None:
                        raise RuntimeError("analytic image VJP requires feature target")
                    if use_sparse_forward:
                        manual_vjp = _manual_sparse_target_grid_loss_and_vjp(
                            render.feature_values,
                            target_feature=target_feature,
                            colorizer=colorizer,
                            rgb_grid_target=rgb_grid_target,
                            rgb_probe=rgb_probe,
                            rgb_probe_target=rgb_probe_target,
                            feature_config=feature_config,
                            frame_start=frame_start,
                            chunk_frames=chunk_frames,
                            feature_loss_type=feature_loss_type,
                            feature_loss_weight=feature_loss_weight,
                            rgb_grid_loss_weight=rgb_grid_loss_weight,
                            rgb_probe_loss_weight=rgb_probe_loss_weight,
                            total_feature_loss_elems=total_feature_loss_elems,
                            total_rgb_grid_loss_elems=total_rgb_grid_loss_elems,
                            total_rgb_probe_loss_elems=total_rgb_probe_loss_elems,
                            device=device,
                        )
                    else:
                        manual_vjp = _manual_target_grid_loss_and_vjp(
                            render.feature_image,
                            target_feature=target_feature,
                            colorizer=colorizer,
                            rgb_grid_target=rgb_grid_target,
                            rgb_probe=rgb_probe,
                            rgb_probe_target=rgb_probe_target,
                            feature_config=feature_config,
                            frame_start=frame_start,
                            chunk_frames=chunk_frames,
                            feature_loss_type=feature_loss_type,
                            feature_loss_weight=feature_loss_weight,
                            rgb_grid_loss_weight=rgb_grid_loss_weight,
                            rgb_probe_loss_weight=rgb_probe_loss_weight,
                            total_feature_loss_elems=total_feature_loss_elems,
                            total_rgb_grid_loss_elems=total_rgb_grid_loss_elems,
                            total_rgb_probe_loss_elems=total_rgb_probe_loss_elems,
                            device=device,
                            image_vjp_mode=feature_target_image_vjp_mode,
                        )
                    loss = manual_vjp.loss
                    chunk_feature_target_loss_value = manual_vjp.feature_target_loss
                    chunk_rgb_grid_loss_value = manual_vjp.rgb_grid_loss
                    chunk_rgb_probe_loss_value = manual_vjp.rgb_probe_loss
                    feature_target_loss_value += chunk_feature_target_loss_value
                    rgb_grid_loss_value += chunk_rgb_grid_loss_value
                    rgb_probe_loss_value += chunk_rgb_probe_loss_value
                    chunk_feature_target_ms = manual_vjp.feature_target_ms + manual_vjp.image_vjp_ms
                    chunk_rgb_grid_loss_ms = manual_vjp.rgb_grid_loss_ms
                    chunk_rgb_probe_loss_ms = manual_vjp.rgb_probe_loss_ms
                    feature_target_ms += chunk_feature_target_ms
                    rgb_grid_loss_ms += chunk_rgb_grid_loss_ms
                    rgb_probe_loss_ms += chunk_rgb_probe_loss_ms
                    chunk_target_start = manual_vjp.target_start
                    chunk_target_frames = manual_vjp.target_frames
                    chunk_probe_target_start = manual_vjp.probe_target_start
                    chunk_probe_target_frames = manual_vjp.probe_target_frames
                elif rgb_loss_weight > 0.0:
                    rgb = _compose_alpha_background_rgb(
                        render.feature_image,
                        render.alpha,
                        colorizer,
                        strategy=alpha_background_train_strategy,
                        sample_scope=alpha_background_sample_scope,
                    )
                    target_chunk = target_rgb[frame_start : frame_start + chunk_frames]
                    rgb_loss = (rgb - target_chunk).square().sum() / float(total_loss_elems)
                    loss = loss + rgb_loss_weight * rgb_loss
                    chunk_rgb_loss_value = float(rgb_loss.detach().cpu().item())
                    rgb_loss_value += chunk_rgb_loss_value
                rendered_feature_chunk: torch.Tensor | None = None
                target_feature_chunk: torch.Tensor | None = None
                if (
                    not use_analytic_image_vjp
                    and target_feature is not None
                    and (feature_loss_weight > 0.0 or rgb_grid_loss_weight > 0.0 or rgb_probe_loss_weight > 0.0)
                ):
                    target_t0 = time.perf_counter()
                    if target_feature.materialization == "target_grid" and target_feature.source is not None:
                        chunk_target_start, chunk_target_frames = _target_grid_slice_for_render_chunk(
                            target_frames=int(target_feature.source.shape[0]),
                            render_frames=feature_config.frames,
                            frame_start=frame_start,
                            chunk_frames=chunk_frames,
                        )
                    target_feature_chunk = target_feature.chunk(frame_start, chunk_frames)
                    rendered_feature_chunk = (
                        _adapt_render_to_feature_target(
                            render.feature_image,
                            target_shape=tuple(int(item) for item in target_feature_chunk.shape),
                            mode=target_feature.grid_mode,
                        )
                        if target_feature.materialization == "target_grid"
                        else render.feature_image
                    )
                    _sync_device(device)
                    chunk_feature_target_ms = (time.perf_counter() - target_t0) * 1000.0
                    feature_target_ms += chunk_feature_target_ms
                if not use_analytic_image_vjp and target_feature is not None and rgb_grid_loss_weight > 0.0:
                    if rendered_feature_chunk is None or rgb_grid_target is None:
                        raise RuntimeError("RGB-grid colorizer loss missing rendered or target chunk")
                    rgb_grid_t0 = time.perf_counter()
                    target_start, target_frames = _target_grid_slice_for_render_chunk(
                        target_frames=int(rgb_grid_target.shape[0]),
                        render_frames=feature_config.frames,
                        frame_start=frame_start,
                        chunk_frames=chunk_frames,
                    )
                    target_rgb_grid_chunk = rgb_grid_target[target_start : target_start + target_frames]
                    rgb_grid = colorizer(rendered_feature_chunk)
                    rgb_grid_loss = (
                        (rgb_grid - target_rgb_grid_chunk).square().sum()
                        / float(total_rgb_grid_loss_elems)
                    )
                    loss = loss + rgb_grid_loss_weight * rgb_grid_loss
                    chunk_rgb_grid_loss_value = float(rgb_grid_loss.detach().cpu().item())
                    rgb_grid_loss_value += chunk_rgb_grid_loss_value
                    _sync_device(device)
                    chunk_rgb_grid_loss_ms = (time.perf_counter() - rgb_grid_t0) * 1000.0
                    rgb_grid_loss_ms += chunk_rgb_grid_loss_ms
                if not use_analytic_image_vjp and target_feature is not None and feature_loss_weight > 0.0:
                    if rendered_feature_chunk is None or target_feature_chunk is None:
                        raise RuntimeError("feature target loss missing rendered or target chunk")
                    feature_target_loss = _feature_target_loss(
                        rendered_feature_chunk,
                        target_feature_chunk,
                        feature_loss_type,
                    ) / float(total_feature_loss_elems)
                    loss = loss + feature_loss_weight * feature_target_loss
                    chunk_feature_target_loss_value = float(feature_target_loss.detach().cpu().item())
                    feature_target_loss_value += chunk_feature_target_loss_value
                if not use_analytic_image_vjp and rgb_probe is not None and rgb_probe_loss_weight > 0.0:
                    if rendered_feature_chunk is None or rgb_probe_target is None:
                        raise RuntimeError("RGB probe loss missing rendered target-grid chunk")
                    rgb_probe_t0 = time.perf_counter()
                    target_start, target_frames = _target_grid_slice_for_render_chunk(
                        target_frames=int(rgb_probe_target.shape[0]),
                        render_frames=feature_config.frames,
                        frame_start=frame_start,
                        chunk_frames=chunk_frames,
                    )
                    chunk_probe_target_start = target_start
                    chunk_probe_target_frames = target_frames
                    target_rgb_probe_chunk = rgb_probe_target[target_start : target_start + target_frames]
                    rgb_probe_pred = rgb_probe(rendered_feature_chunk)
                    rgb_probe_loss = (
                        (rgb_probe_pred - target_rgb_probe_chunk).square().sum()
                        / float(total_rgb_probe_loss_elems)
                    )
                    loss = loss + rgb_probe_loss_weight * rgb_probe_loss
                    chunk_rgb_probe_loss_value = float(rgb_probe_loss.detach().cpu().item())
                    rgb_probe_loss_value += chunk_rgb_probe_loss_value
                    _sync_device(device)
                    chunk_rgb_probe_loss_ms = (time.perf_counter() - rgb_probe_t0) * 1000.0
                    rgb_probe_loss_ms += chunk_rgb_probe_loss_ms
                chunk_loss_value = float(loss.detach().cpu().item())
                loss_value += chunk_loss_value
                _sync_device(device)
                chunk_t2 = time.perf_counter()
                if use_analytic_image_vjp:
                    if sparse_image_vjp_enabled:
                        if render.tile_tube_ids is None or render.tile_depths is None:
                            raise RuntimeError("sparse image VJP requires render bins")
                        if feature_target_image_vjp_mode in {"analytic_sparse_grid", "analytic_sparse_grid_forward"}:
                            if manual_vjp.sparse_pack is None:
                                raise RuntimeError("analytic_sparse_grid image VJP did not produce sparse image gradients")
                            sparse_pack = manual_vjp.sparse_pack
                        else:
                            if manual_vjp.grad_feature_image is None:
                                raise RuntimeError("analytic_sparse_pixels did not produce feature-image gradients")
                            grad_alpha = torch.zeros_like(render.alpha)
                            _sync_device(device)
                            pack_t0 = time.perf_counter()
                            sparse_pack = _pack_sparse_image_vjp(
                                manual_vjp.grad_feature_image.contiguous(),
                                grad_alpha.contiguous(),
                            )
                            _sync_device(device)
                            chunk_sparse_pack_ms = (time.perf_counter() - pack_t0) * 1000.0
                            sparse_pack_ms += chunk_sparse_pack_ms
                        step_sparse_pixel_count += sparse_pack.pixel_count
                        step_sparse_total_pixels += sparse_pack.total_pixels
                        chunk_sparse_pixel_count = sparse_pack.pixel_count
                        chunk_sparse_pixel_fraction = (
                            0.0
                            if sparse_pack.total_pixels <= 0
                            else float(sparse_pack.pixel_count) / float(sparse_pack.total_pixels)
                        )
                        grad_ma, grad_q, grad_opacity, grad_feature, _tile_unstable = (
                            direct_atomic_feature_sparse_pixels_backward_cached_bins(
                                *render_inputs,
                                sparse_pack.pixel_ids,
                                sparse_pack.grad_feature_values,
                                sparse_pack.grad_alpha_values,
                                render.tile_counts,
                                render.tile_tube_ids,
                                render.tile_depths,
                                render.tile_unstable,
                                render_config,
                            )
                        )
                    else:
                        if manual_vjp.grad_feature_image is None:
                            raise RuntimeError("analytic image VJP did not produce feature-image gradients")
                        grad_alpha = torch.zeros_like(render.alpha)
                        grad_ma, grad_q, grad_opacity, grad_feature, _tile_unstable = direct_atomic_feature_backward(
                            *render_inputs,
                            manual_vjp.grad_feature_image.contiguous(),
                            grad_alpha.contiguous(),
                            render_config,
                            backward_mode=backward_mode,
                        )
                    torch.autograd.backward(
                        (render_inputs[0], render_inputs[1], render_inputs[4], render_inputs[5]),
                        (grad_ma, grad_q, grad_opacity, grad_feature),
                    )
                else:
                    loss.backward()
                _sync_device(device)
                chunk_t3 = time.perf_counter()
                render_forward_ms += (chunk_t1 - chunk_t0) * 1000.0
                colorize_loss_ms += (chunk_t2 - chunk_t1) * 1000.0
                backward_ms += (chunk_t3 - chunk_t2) * 1000.0
                last_backward_end = chunk_t3
                if trace_chunks is not None:
                    trace_chunks.append(
                        {
                            "frame_start": frame_start,
                            "chunk_frames": chunk_frames,
                            "target_start": chunk_target_start,
                            "target_frames": chunk_target_frames,
                            "probe_target_start": chunk_probe_target_start,
                            "probe_target_frames": chunk_probe_target_frames,
                            "weighted_loss": chunk_loss_value,
                            "rgb_loss": chunk_rgb_loss_value,
                            "feature_target_loss": chunk_feature_target_loss_value,
                            "rgb_grid_loss": chunk_rgb_grid_loss_value,
                            "rgb_probe_loss": chunk_rgb_probe_loss_value,
                            "render_forward_ms": (chunk_t1 - chunk_t0) * 1000.0,
                            "colorize_loss_ms": (chunk_t2 - chunk_t1) * 1000.0,
                            "feature_target_ms": chunk_feature_target_ms,
                            "rgb_grid_loss_ms": chunk_rgb_grid_loss_ms,
                            "rgb_probe_loss_ms": chunk_rgb_probe_loss_ms,
                            "sparse_pack_ms": chunk_sparse_pack_ms,
                            "dense_alpha_render_ms": 0.0,
                            "dense_alpha_loss_ms": 0.0,
                            "dense_alpha_backward_ms": 0.0,
                            "projective_interval_timing_ms": chunk_projective_interval_timing_ms,
                            "sparse_pixel_count": chunk_sparse_pixel_count,
                            "sparse_pixel_fraction": chunk_sparse_pixel_fraction,
                            "backward_ms": (chunk_t3 - chunk_t2) * 1000.0,
                        }
                    )
            if dense_alpha_enabled:
                for frame_start in range(0, feature_config.frames, chunk_size):
                    chunk_frames = min(chunk_size, feature_config.frames - frame_start)
                    ma, q_uvt, depth0, depth_beta, opacity, feature = model.tensors()
                    if chunk_frames == feature_config.frames:
                        render_inputs = (
                            ma,
                            q_uvt,
                            depth0.detach(),
                            depth_beta.detach(),
                            opacity,
                            feature,
                        )
                        render_config = uvt_config
                    else:
                        ma_chunk = shift_ma_for_frame_chunk(
                            ma,
                            global_frames=feature_config.frames,
                            frame_start=frame_start,
                            chunk_frames=chunk_frames,
                        )
                        render_inputs = (
                            ma_chunk,
                            q_uvt,
                            depth0.detach(),
                            depth_beta.detach(),
                            opacity,
                            feature,
                        )
                        render_config = chunked_uvt_config(uvt_config, chunk_frames=chunk_frames)
                    _sync_device(device)
                    alpha_t0 = time.perf_counter()
                    if dense_alpha_render_mode == "sparse_f1":
                        dense_render = render_uvt_feature_alpha_all_pixels_with_bins(
                            render_inputs[0],
                            render_inputs[1],
                            render_inputs[2],
                            render_inputs[3],
                            render_inputs[4],
                            render_config,
                        )
                    else:
                        dense_render = render_uvt_feature_tubes(
                            *render_inputs,
                            render_config,
                        )
                    _sync_device(device)
                    alpha_t1 = time.perf_counter()
                    alpha_diff = dense_render.alpha - dense_alpha_target
                    alpha_loss = alpha_diff.square().sum() / float(total_dense_alpha_loss_elems)
                    grad_alpha = (
                        (dense_alpha_loss_weight * 2.0 / float(total_dense_alpha_loss_elems)) * alpha_diff
                    ).contiguous()
                    chunk_dense_alpha_loss = float(alpha_loss.detach().cpu().item())
                    dense_alpha_loss_value += chunk_dense_alpha_loss
                    loss_value += dense_alpha_loss_weight * chunk_dense_alpha_loss
                    _sync_device(device)
                    alpha_t2 = time.perf_counter()
                    if dense_alpha_render_mode == "sparse_f1":
                        dummy_feature = torch.zeros((render_inputs[0].shape[0], 1), dtype=torch.float32, device=device)
                        grad_feature_image = torch.zeros(
                            (
                                int(render_config.frames),
                                1,
                                int(render_config.height),
                                int(render_config.width),
                            ),
                            dtype=torch.float32,
                            device=device,
                        )
                        grad_ma, grad_q, grad_opacity, _grad_feature, _tile_unstable = (
                            direct_atomic_feature_backward_cached_bins(
                                render_inputs[0],
                                render_inputs[1],
                                render_inputs[2],
                                render_inputs[3],
                                render_inputs[4],
                                dummy_feature,
                                grad_feature_image,
                                grad_alpha,
                                dense_render.tile_counts,
                                dense_render.tile_tube_ids,
                                dense_render.tile_depths,
                                dense_render.tile_unstable,
                                render_config,
                                backward_mode=dense_alpha_backward_mode,
                            )
                        )
                        torch.autograd.backward(
                            (render_inputs[0], render_inputs[1], render_inputs[4]),
                            (grad_ma, grad_q, grad_opacity),
                        )
                    else:
                        grad_feature_image = torch.zeros_like(dense_render.feature_image)
                        grad_ma, grad_q, grad_opacity, grad_feature, _tile_unstable = direct_atomic_feature_backward(
                            *render_inputs,
                            grad_feature_image,
                            grad_alpha,
                            render_config,
                            backward_mode=dense_alpha_backward_mode,
                        )
                        torch.autograd.backward(
                            (render_inputs[0], render_inputs[1], render_inputs[4], render_inputs[5]),
                            (grad_ma, grad_q, grad_opacity, grad_feature),
                        )
                    _sync_device(device)
                    alpha_t3 = time.perf_counter()
                    dense_alpha_render_ms += (alpha_t1 - alpha_t0) * 1000.0
                    dense_alpha_loss_ms += (alpha_t2 - alpha_t1) * 1000.0
                    dense_alpha_backward_ms += (alpha_t3 - alpha_t2) * 1000.0
                    backward_ms += (alpha_t3 - alpha_t2) * 1000.0
                    last_backward_end = alpha_t3
                    if trace_chunks is not None:
                        trace_chunks.append(
                            {
                                "frame_start": frame_start,
                                "chunk_frames": chunk_frames,
                                "target_start": None,
                                "target_frames": None,
                                "probe_target_start": None,
                                "probe_target_frames": None,
                                "weighted_loss": dense_alpha_loss_weight * chunk_dense_alpha_loss,
                                "rgb_loss": 0.0,
                                "feature_target_loss": 0.0,
                                "rgb_grid_loss": 0.0,
                                "rgb_probe_loss": 0.0,
                                "dense_alpha_loss": chunk_dense_alpha_loss,
                                "render_forward_ms": 0.0,
                                "colorize_loss_ms": 0.0,
                                "feature_target_ms": 0.0,
                                "rgb_grid_loss_ms": 0.0,
                                "rgb_probe_loss_ms": 0.0,
                                "sparse_pack_ms": 0.0,
                                "dense_alpha_render_ms": (alpha_t1 - alpha_t0) * 1000.0,
                                "dense_alpha_loss_ms": (alpha_t2 - alpha_t1) * 1000.0,
                                "dense_alpha_backward_ms": (alpha_t3 - alpha_t2) * 1000.0,
                                "sparse_pixel_count": None,
                                "sparse_pixel_fraction": None,
                                "backward_ms": (alpha_t3 - alpha_t2) * 1000.0,
                            }
                        )
            if sparse_visual_enabled:
                for frame_start in range(0, feature_config.frames, chunk_size):
                    chunk_frames = min(chunk_size, feature_config.frames - frame_start)
                    pixel_ids = _sparse_visual_pixel_ids_for_chunk(
                        pixel_source=sparse_visual_pixel_source,
                        chunk_frames=chunk_frames,
                        height=feature_config.height,
                        width=feature_config.width,
                        render_frames=feature_config.frames,
                        frame_start=frame_start,
                        sample_grid_shape=sparse_visual_grid_shape,
                        patch_shape=sparse_visual_patch_shape,
                        patch_phase=sparse_visual_patch_phase,
                        patch_phase_shape=sparse_visual_patch_phase_shape,
                        device=device,
                    )
                    if int(pixel_ids.numel()) == 0:
                        continue
                    local_frame_ids = _sparse_visual_local_frame_ids_for_chunk(
                        render_frames=feature_config.frames,
                        frame_start=frame_start,
                        chunk_frames=chunk_frames,
                        sample_grid_shape=sparse_visual_grid_shape,
                        device=device,
                    )
                    ma, q_uvt, depth0, depth_beta, opacity, feature = model.tensors()
                    if chunk_frames == feature_config.frames:
                        render_inputs = (
                            ma,
                            q_uvt,
                            depth0.detach(),
                            depth_beta.detach(),
                            opacity,
                            feature,
                        )
                        render_config = uvt_config
                    else:
                        ma_chunk = shift_ma_for_frame_chunk(
                            ma,
                            global_frames=feature_config.frames,
                            frame_start=frame_start,
                            chunk_frames=chunk_frames,
                        )
                        render_inputs = (
                            ma_chunk,
                            q_uvt,
                            depth0.detach(),
                            depth_beta.detach(),
                            opacity,
                            feature,
                        )
                        render_config = chunked_uvt_config(uvt_config, chunk_frames=chunk_frames)
                    _sync_device(device)
                    visual_t0 = time.perf_counter()
                    target_rgb_chunk = target_rgb[frame_start : frame_start + chunk_frames]
                    if native_target_area_sparse_visual_vjp_enabled:
                        conv1, _gelu, conv2 = _hidden64_colorizer_layers(colorizer)
                        if colorizer.activation != "sigmoid":
                            raise RuntimeError("native target-area sparse_visual VJP requires colorize.activation='sigmoid'")
                        with torch.no_grad():
                            visual_bins = bin_uvt_feature_tubes(
                                *render_inputs[:5],
                                render_config,
                                feature_dim=int(render_inputs[5].shape[1]),
                            )
                        _sync_device(device)
                        visual_t1 = time.perf_counter()
                        local_frame_count = int(local_frame_ids.numel())
                        target_cells = _sparse_visual_target_area_cells(
                            target_rgb_chunk,
                            local_frame_ids,
                            sample_grid_shape=sparse_visual_grid_shape,
                        )
                        cell_ids = _sparse_visual_target_area_cell_ids(
                            local_frame_count,
                            sample_grid_shape=sparse_visual_grid_shape,
                            patch_shape=sparse_visual_patch_shape,
                            device=device,
                        )
                        patch_area = int(sparse_visual_patch_shape[0]) * int(sparse_visual_patch_shape[1])
                        target_forward = sparse_hidden_sigmoid_target_area_forward_sums_cached_bins(
                            *render_inputs,
                            pixel_ids,
                            cell_ids,
                            conv1.weight[:, :, 0, 0].detach().contiguous(),
                            conv1.bias.detach().contiguous(),
                            conv2.weight[:, :, 0, 0].detach().contiguous(),
                            conv2.bias.detach().contiguous(),
                            visual_bins.tile_counts,
                            visual_bins.tile_tube_ids,
                            visual_bins.tile_depths,
                            visual_bins.tile_unstable,
                            render_config,
                            cell_count=int(target_cells.shape[0]),
                        )
                        diff = target_forward.pred_sums / float(patch_area) - target_cells
                        visual_loss = (diff.square().sum() / float(total_sparse_visual_loss_elems)).detach()
                        cell_grad_rgb = (
                            (float(sparse_visual_loss_weight) * 2.0 / float(total_sparse_visual_loss_elems))
                            * diff
                            / float(patch_area)
                        ).contiguous()
                        _sync_device(device)
                        visual_t2 = time.perf_counter()
                        hidden_backward = direct_hidden_sigmoid_target_area_backward_cached_bins(
                            *render_inputs,
                            pixel_ids,
                            cell_ids,
                            cell_grad_rgb,
                            conv1.weight[:, :, 0, 0].detach().contiguous(),
                            conv1.bias.detach().contiguous(),
                            conv2.weight[:, :, 0, 0].detach().contiguous(),
                            conv2.bias.detach().contiguous(),
                            visual_bins.tile_counts,
                            visual_bins.tile_tube_ids,
                            visual_bins.tile_depths,
                            target_forward.tile_unstable,
                            render_config,
                            backward_mode=_native_target_area_backward_mode(sparse_visual_loss_vjp_mode),
                        )
                        grad_ma = hidden_backward.grad_ma
                        grad_q = hidden_backward.grad_q_uvt
                        grad_opacity = hidden_backward.grad_opacity
                        grad_feature = hidden_backward.grad_feature
                        if "colorizer" in sparse_visual_loss_vjp_mode:
                            if (
                                hidden_backward.grad_hidden_weight is None
                                or hidden_backward.grad_hidden_bias is None
                                or hidden_backward.grad_output_weight is None
                                or hidden_backward.grad_output_bias is None
                            ):
                                raise RuntimeError("native target-area colorizer VJP did not return colorizer gradients")
                            _add_param_grad(conv1.weight, hidden_backward.grad_hidden_weight.view_as(conv1.weight))
                            _add_param_grad(conv1.bias, hidden_backward.grad_hidden_bias.view_as(conv1.bias))
                            _add_param_grad(conv2.weight, hidden_backward.grad_output_weight.view_as(conv2.weight))
                            _add_param_grad(conv2.bias, hidden_backward.grad_output_bias.view_as(conv2.bias))
                    else:
                        with torch.no_grad():
                            visual_render = render_uvt_feature_sparse_pixels_with_bins(
                                *render_inputs,
                                pixel_ids,
                                render_config,
                            )
                        feature_values = visual_render.feature_values.detach()
                        alpha_values = visual_render.alpha_values.detach()
                        _sync_device(device)
                        visual_t1 = time.perf_counter()
                        target_values = (
                            _gather_sparse_visual_rgb_values(
                                target_rgb_chunk,
                                pixel_ids,
                            )
                            if (
                                sparse_visual_loss_basis != "target_area_mean"
                                or sparse_visual_composition == "target_background"
                            )
                            else None
                        )
                        if visual_render.tile_tube_ids is None or visual_render.tile_depths is None:
                            raise RuntimeError("sparse_visual requires render bins")
                        if native_sparse_visual_vjp_enabled:
                            if target_values is None:
                                raise RuntimeError("native sparse_visual VJP requires pixel target values")
                            conv1, _gelu, conv2 = _hidden64_colorizer_layers(colorizer)
                            if colorizer.activation != "sigmoid":
                                raise RuntimeError("native sparse_visual VJP requires colorize.activation='sigmoid'")
                            visual_t2 = visual_t1
                            hidden_backward = direct_hidden_sigmoid_mse_sparse_pixels_backward_cached_bins(
                                *render_inputs,
                                pixel_ids,
                                target_values.contiguous(),
                                conv1.weight[:, :, 0, 0].detach().contiguous(),
                                conv1.bias.detach().contiguous(),
                                conv2.weight[:, :, 0, 0].detach().contiguous(),
                                conv2.bias.detach().contiguous(),
                                visual_render.tile_counts,
                                visual_render.tile_tube_ids,
                                visual_render.tile_depths,
                                visual_render.tile_unstable,
                                render_config,
                                total_loss_elems=total_sparse_visual_loss_elems,
                            )
                            visual_loss = hidden_backward.loss.detach()
                            grad_ma = hidden_backward.grad_ma
                            grad_q = hidden_backward.grad_q_uvt
                            grad_opacity = hidden_backward.grad_opacity
                            grad_feature = hidden_backward.grad_feature
                        else:
                            visual_loss, grad_feature_values, grad_alpha_values = _sparse_visual_rgb_loss_and_grads(
                                feature_values,
                                alpha_values,
                                target_values,
                                colorizer,
                                total_loss_elems=total_sparse_visual_loss_elems,
                                loss_weight=sparse_visual_loss_weight,
                                loss_basis=sparse_visual_loss_basis,
                                sample_grid_shape=sparse_visual_grid_shape,
                                patch_shape=sparse_visual_patch_shape,
                                target_rgb_chunk=target_rgb_chunk,
                                local_frame_ids=local_frame_ids,
                                vjp_mode=sparse_visual_loss_vjp_mode,
                                composition=sparse_visual_composition,
                            )
                            if sparse_visual_alpha_loss_weight > 0.0:
                                alpha_loss, alpha_grad = _sparse_visual_alpha_loss_and_grad(
                                    alpha_values,
                                    target=sparse_visual_alpha_target,
                                    total_loss_elems=total_sparse_visual_alpha_loss_elems,
                                    loss_weight=sparse_visual_alpha_loss_weight,
                                )
                                grad_alpha_values = grad_alpha_values + alpha_grad
                                sparse_visual_alpha_loss_value += float(alpha_loss.detach().cpu().item())
                            if sparse_visual_black_hole_loss_weight > 0.0:
                                black_hole_loss, black_hole_grad = _sparse_visual_black_hole_loss_and_grad(
                                    alpha_values,
                                    target_values,
                                    total_loss_elems=total_sparse_visual_black_hole_loss_elems,
                                    loss_weight=sparse_visual_black_hole_loss_weight,
                                    loss_basis=sparse_visual_loss_basis,
                                    sample_grid_shape=sparse_visual_grid_shape,
                                    patch_shape=sparse_visual_patch_shape,
                                    target_rgb_chunk=target_rgb_chunk,
                                    local_frame_ids=local_frame_ids,
                                )
                                grad_alpha_values = grad_alpha_values + black_hole_grad
                                sparse_visual_black_hole_loss_value += float(
                                    black_hole_loss.detach().cpu().item()
                                )
                            _sync_device(device)
                            visual_t2 = time.perf_counter()
                            grad_ma, grad_q, grad_opacity, grad_feature, _tile_unstable = (
                                direct_atomic_feature_sparse_pixels_backward_cached_bins(
                                    *render_inputs,
                                    pixel_ids,
                                    grad_feature_values,
                                    grad_alpha_values,
                                    visual_render.tile_counts,
                                    visual_render.tile_tube_ids,
                                    visual_render.tile_depths,
                                    visual_render.tile_unstable,
                                    render_config,
                                )
                            )
                    torch.autograd.backward(
                        (render_inputs[0], render_inputs[1], render_inputs[4], render_inputs[5]),
                        (grad_ma, grad_q, grad_opacity, grad_feature),
                    )
                    _sync_device(device)
                    visual_t3 = time.perf_counter()
                    sparse_visual_render_ms += (visual_t1 - visual_t0) * 1000.0
                    sparse_visual_loss_ms += (visual_t2 - visual_t1) * 1000.0
                    sparse_visual_backward_ms += (visual_t3 - visual_t2) * 1000.0
                    backward_ms += (visual_t3 - visual_t1) * 1000.0
                    last_backward_end = visual_t3
                    chunk_sparse_visual_loss = float(visual_loss.detach().cpu().item())
                    sparse_visual_loss_value += chunk_sparse_visual_loss
                    loss_value += sparse_visual_loss_weight * chunk_sparse_visual_loss
                    if sparse_visual_alpha_loss_weight > 0.0:
                        loss_value += sparse_visual_alpha_loss_weight * float(alpha_loss.detach().cpu().item())
                    if sparse_visual_black_hole_loss_weight > 0.0:
                        loss_value += sparse_visual_black_hole_loss_weight * float(
                            black_hole_loss.detach().cpu().item()
                        )
                    step_sparse_visual_pixel_count += int(pixel_ids.numel())
                    step_sparse_visual_loss_sample_count += _sparse_visual_loss_sample_count(
                        int(pixel_ids.numel()),
                        loss_basis=sparse_visual_loss_basis,
                        patch_shape=sparse_visual_patch_shape,
                    )
                    step_sparse_visual_alpha_sample_count += int(pixel_ids.numel())
                    step_sparse_visual_total_pixels += int(chunk_frames * feature_config.height * feature_config.width)
            if support_target_area_enabled:
                if support_target_area_points is None:
                    raise RuntimeError("support_birth_split target area loss requires target points")
                for frame_start in range(0, feature_config.frames, chunk_size):
                    chunk_frames = min(chunk_size, feature_config.frames - frame_start)
                    pixel_ids, target_area_cell_count = _support_birth_split_target_patch_pixel_ids_for_chunk(
                        support_target_area_points,
                        frames=feature_config.frames,
                        height=feature_config.height,
                        width=feature_config.width,
                        frame_start=frame_start,
                        chunk_frames=chunk_frames,
                        patch_shape=support_target_area_patch_shape,
                        device=device,
                    )
                    if int(pixel_ids.numel()) == 0:
                        continue
                    ma, q_uvt, depth0, depth_beta, opacity, feature = model.tensors()
                    if chunk_frames == feature_config.frames:
                        render_inputs = (
                            ma,
                            q_uvt,
                            depth0.detach(),
                            depth_beta.detach(),
                            opacity,
                            feature,
                        )
                        render_config = uvt_config
                    else:
                        ma_chunk = shift_ma_for_frame_chunk(
                            ma,
                            global_frames=feature_config.frames,
                            frame_start=frame_start,
                            chunk_frames=chunk_frames,
                        )
                        render_inputs = (
                            ma_chunk,
                            q_uvt,
                            depth0.detach(),
                            depth_beta.detach(),
                            opacity,
                            feature,
                        )
                        render_config = chunked_uvt_config(uvt_config, chunk_frames=chunk_frames)
                    _sync_device(device)
                    area_t0 = time.perf_counter()
                    target_rgb_chunk = target_rgb[frame_start : frame_start + chunk_frames]
                    with torch.no_grad():
                        area_render = render_uvt_feature_sparse_pixels_with_bins(
                            *render_inputs,
                            pixel_ids,
                            render_config,
                        )
                    _sync_device(device)
                    area_t1 = time.perf_counter()
                    target_values = _gather_sparse_visual_rgb_values(target_rgb_chunk, pixel_ids)
                    if area_render.tile_tube_ids is None or area_render.tile_depths is None:
                        raise RuntimeError("support_birth_split target area loss requires render bins")
                    area_loss, grad_feature_values, grad_alpha_values = _sparse_visual_rgb_loss_and_grads(
                        area_render.feature_values.detach(),
                        area_render.alpha_values.detach(),
                        target_values,
                        colorizer,
                        total_loss_elems=total_support_target_area_loss_elems,
                        loss_weight=support_target_area_loss_weight,
                        loss_basis="patch_mean",
                        sample_grid_shape=(target_area_cell_count, 1, 1),
                        patch_shape=support_target_area_patch_shape,
                        vjp_mode=support_target_area_vjp_mode,
                        composition=support_target_area_composition,
                    )
                    support_target_area_loss_value += float(area_loss.detach().cpu().item())
                    loss_value += support_target_area_loss_weight * float(area_loss.detach().cpu().item())
                    _sync_device(device)
                    area_t2 = time.perf_counter()
                    grad_ma, grad_q, grad_opacity, grad_feature, _tile_unstable = (
                        direct_atomic_feature_sparse_pixels_backward_cached_bins(
                            *render_inputs,
                            pixel_ids,
                            grad_feature_values,
                            grad_alpha_values,
                            area_render.tile_counts,
                            area_render.tile_tube_ids,
                            area_render.tile_depths,
                            area_render.tile_unstable,
                            render_config,
                        )
                    )
                    torch.autograd.backward(
                        (render_inputs[0], render_inputs[1], render_inputs[4], render_inputs[5]),
                        (grad_ma, grad_q, grad_opacity, grad_feature),
                    )
                    _sync_device(device)
                    area_t3 = time.perf_counter()
                    support_target_area_ms += (area_t3 - area_t0) * 1000.0
                    backward_ms += (area_t3 - area_t2) * 1000.0
                    last_backward_end = area_t3
                    step_support_target_area_sample_count += int(target_area_cell_count)
                    if trace_chunks is not None:
                        trace_chunks.append(
                            {
                                "frame_start": frame_start,
                                "chunk_frames": chunk_frames,
                                "target_start": None,
                                "target_frames": None,
                                "probe_target_start": None,
                                "probe_target_frames": None,
                                "weighted_loss": support_target_area_loss_weight
                                * float(area_loss.detach().cpu().item()),
                                "rgb_loss": 0.0,
                                "feature_target_loss": 0.0,
                                "rgb_grid_loss": 0.0,
                                "rgb_probe_loss": 0.0,
                                "support_target_area_loss": float(area_loss.detach().cpu().item()),
                                "render_forward_ms": 0.0,
                                "colorize_loss_ms": 0.0,
                                "feature_target_ms": 0.0,
                                "rgb_grid_loss_ms": 0.0,
                                "rgb_probe_loss_ms": 0.0,
                                "sparse_pack_ms": 0.0,
                                "dense_alpha_render_ms": 0.0,
                                "dense_alpha_loss_ms": 0.0,
                                "dense_alpha_backward_ms": 0.0,
                                "support_target_area_ms": (area_t3 - area_t0) * 1000.0,
                                "sparse_pixel_count": int(pixel_ids.numel()),
                                "sparse_pixel_fraction": float(pixel_ids.numel())
                                / float(chunk_frames * feature_config.height * feature_config.width),
                                "backward_ms": (area_t3 - area_t2) * 1000.0,
                            }
                        )
            if support_prefix_alpha_enabled:
                if support_prefix_alpha_points is None or support_prefix_alpha_selected_ids is None:
                    raise RuntimeError("support_birth_split prefix alpha loss requires target points and selected ids")
                _sync_device(device)
                prefix_t0 = time.perf_counter()
                prefix_alpha_loss, prefix_alpha_metrics = _support_prefix_alpha_loss(
                    model,
                    support_prefix_alpha_points,
                    support_prefix_alpha_selected_ids,
                    alpha_target=support_prefix_alpha_target,
                    total_loss_elems=total_support_prefix_alpha_loss_elems,
                    alpha_threshold=float(uvt_config.alpha_threshold),
                    max_alpha=float(uvt_config.max_alpha),
                    transmittance_threshold=float(uvt_config.transmittance_threshold),
                )
                weighted_prefix_alpha_loss = support_prefix_alpha_loss_weight * prefix_alpha_loss
                weighted_prefix_alpha_loss.backward()
                _sync_device(device)
                prefix_t1 = time.perf_counter()
                support_prefix_alpha_ms = (prefix_t1 - prefix_t0) * 1000.0
                backward_ms += support_prefix_alpha_ms
                last_backward_end = prefix_t1
                support_prefix_alpha_loss_value = float(prefix_alpha_loss.detach().cpu().item())
                support_prefix_alpha_selected_weight_mean_value = float(prefix_alpha_metrics["selected_weight_mean"])
                support_prefix_alpha_selected_share_mean_value = float(prefix_alpha_metrics["selected_weight_share_mean"])
                support_prefix_alpha_final_alpha_mean_value = float(prefix_alpha_metrics["final_alpha_mean"])
                loss_value += float(weighted_prefix_alpha_loss.detach().cpu().item())
                step_support_prefix_alpha_sample_count += int(support_prefix_alpha_points.shape[0])
                if trace_chunks is not None:
                    trace_chunks.append(
                        {
                            "frame_start": None,
                            "chunk_frames": None,
                            "target_start": None,
                            "target_frames": None,
                            "probe_target_start": None,
                            "probe_target_frames": None,
                            "weighted_loss": float(weighted_prefix_alpha_loss.detach().cpu().item()),
                            "rgb_loss": 0.0,
                            "feature_target_loss": 0.0,
                            "rgb_grid_loss": 0.0,
                            "rgb_probe_loss": 0.0,
                            "support_prefix_alpha_loss": support_prefix_alpha_loss_value,
                            "support_prefix_alpha_selected_weight_mean": (
                                support_prefix_alpha_selected_weight_mean_value
                            ),
                            "support_prefix_alpha_selected_share_mean": (
                                support_prefix_alpha_selected_share_mean_value
                            ),
                            "support_prefix_alpha_final_alpha_mean": support_prefix_alpha_final_alpha_mean_value,
                            "render_forward_ms": 0.0,
                            "colorize_loss_ms": 0.0,
                            "feature_target_ms": 0.0,
                            "rgb_grid_loss_ms": 0.0,
                            "rgb_probe_loss_ms": 0.0,
                            "sparse_pack_ms": 0.0,
                            "dense_alpha_render_ms": 0.0,
                            "dense_alpha_loss_ms": 0.0,
                            "dense_alpha_backward_ms": 0.0,
                            "support_prefix_alpha_ms": support_prefix_alpha_ms,
                            "sparse_pixel_count": int(support_prefix_alpha_points.shape[0]),
                            "sparse_pixel_fraction": float(support_prefix_alpha_points.shape[0])
                            / float(feature_config.frames * feature_config.height * feature_config.width),
                            "backward_ms": support_prefix_alpha_ms,
                        }
                    )
            if support_target_alpha_enabled:
                if support_target_alpha_points is None:
                    raise RuntimeError("support_birth_split target alpha loss requires target points")
                for frame_start in range(0, feature_config.frames, chunk_size):
                    chunk_frames = min(chunk_size, feature_config.frames - frame_start)
                    pixel_ids = _support_birth_split_target_pixel_ids_for_chunk(
                        support_target_alpha_points,
                        frames=feature_config.frames,
                        height=feature_config.height,
                        width=feature_config.width,
                        frame_start=frame_start,
                        chunk_frames=chunk_frames,
                        device=device,
                    )
                    if int(pixel_ids.numel()) == 0:
                        continue
                    ma, q_uvt, depth0, depth_beta, opacity, feature = model.tensors()
                    if chunk_frames == feature_config.frames:
                        render_inputs = (
                            ma,
                            q_uvt,
                            depth0.detach(),
                            depth_beta.detach(),
                            opacity,
                            feature,
                        )
                        render_config = uvt_config
                    else:
                        ma_chunk = shift_ma_for_frame_chunk(
                            ma,
                            global_frames=feature_config.frames,
                            frame_start=frame_start,
                            chunk_frames=chunk_frames,
                        )
                        render_inputs = (
                            ma_chunk,
                            q_uvt,
                            depth0.detach(),
                            depth_beta.detach(),
                            opacity,
                            feature,
                        )
                        render_config = chunked_uvt_config(uvt_config, chunk_frames=chunk_frames)
                    _sync_device(device)
                    alpha_t0 = time.perf_counter()
                    with torch.no_grad():
                        alpha_render = render_uvt_feature_sparse_pixels_with_bins(
                            *render_inputs,
                            pixel_ids,
                            render_config,
                        )
                    _sync_device(device)
                    alpha_t1 = time.perf_counter()
                    alpha_loss, grad_alpha_values = _sparse_visual_alpha_loss_and_grad(
                        alpha_render.alpha_values.detach(),
                        target=support_target_alpha_target,
                        total_loss_elems=total_support_target_alpha_loss_elems,
                        loss_weight=support_target_alpha_loss_weight,
                    )
                    grad_feature_values = torch.zeros_like(alpha_render.feature_values)
                    support_target_alpha_loss_value += float(alpha_loss.detach().cpu().item())
                    loss_value += support_target_alpha_loss_weight * float(alpha_loss.detach().cpu().item())
                    _sync_device(device)
                    alpha_t2 = time.perf_counter()
                    if alpha_render.tile_tube_ids is None or alpha_render.tile_depths is None:
                        raise RuntimeError("support_birth_split target alpha loss requires render bins")
                    grad_ma, grad_q, grad_opacity, grad_feature, _tile_unstable = (
                        direct_atomic_feature_sparse_pixels_backward_cached_bins(
                            *render_inputs,
                            pixel_ids,
                            grad_feature_values,
                            grad_alpha_values,
                            alpha_render.tile_counts,
                            alpha_render.tile_tube_ids,
                            alpha_render.tile_depths,
                            alpha_render.tile_unstable,
                            render_config,
                        )
                    )
                    torch.autograd.backward(
                        (render_inputs[0], render_inputs[1], render_inputs[4], render_inputs[5]),
                        (grad_ma, grad_q, grad_opacity, grad_feature),
                    )
                    _sync_device(device)
                    alpha_t3 = time.perf_counter()
                    support_target_alpha_ms += (alpha_t3 - alpha_t0) * 1000.0
                    backward_ms += (alpha_t3 - alpha_t2) * 1000.0
                    last_backward_end = alpha_t3
                    step_support_target_alpha_sample_count += int(pixel_ids.numel())
                    if trace_chunks is not None:
                        trace_chunks.append(
                            {
                                "frame_start": frame_start,
                                "chunk_frames": chunk_frames,
                                "target_start": None,
                                "target_frames": None,
                                "probe_target_start": None,
                                "probe_target_frames": None,
                                "weighted_loss": support_target_alpha_loss_weight
                                * float(alpha_loss.detach().cpu().item()),
                                "rgb_loss": 0.0,
                                "feature_target_loss": 0.0,
                                "rgb_grid_loss": 0.0,
                                "rgb_probe_loss": 0.0,
                                "support_target_alpha_loss": float(alpha_loss.detach().cpu().item()),
                                "render_forward_ms": 0.0,
                                "colorize_loss_ms": 0.0,
                                "feature_target_ms": 0.0,
                                "rgb_grid_loss_ms": 0.0,
                                "rgb_probe_loss_ms": 0.0,
                                "sparse_pack_ms": 0.0,
                                "dense_alpha_render_ms": 0.0,
                                "dense_alpha_loss_ms": 0.0,
                                "dense_alpha_backward_ms": 0.0,
                                "support_target_alpha_ms": (alpha_t3 - alpha_t0) * 1000.0,
                                "sparse_pixel_count": int(pixel_ids.numel()),
                                "sparse_pixel_fraction": float(pixel_ids.numel())
                                / float(chunk_frames * feature_config.height * feature_config.width),
                                "backward_ms": (alpha_t3 - alpha_t2) * 1000.0,
                            }
                        )
            if visibility_proxy_enabled:
                if visibility_proxy_target_points is None:
                    raise RuntimeError("visibility_proxy enabled but target points are missing")
                _sync_device(device)
                visibility_t0 = time.perf_counter()
                visibility_proxy_loss = _visibility_proxy_loss(
                    model,
                    visibility_proxy_target_points,
                    center_weight=visibility_proxy_center_weight,
                    support_weight=visibility_proxy_support_weight,
                    support_epsilon=visibility_proxy_support_epsilon,
                    max_alpha=feature_config.max_alpha,
                    scale_px=visibility_proxy_scale_px,
                    temperature=visibility_proxy_temperature,
                    velocity_penalty=visibility_proxy_velocity_penalty,
                )
                weighted_visibility_proxy_loss = visibility_proxy_loss_weight * visibility_proxy_loss
                weighted_visibility_proxy_loss.backward()
                _sync_device(device)
                visibility_t1 = time.perf_counter()
                visibility_proxy_ms = (visibility_t1 - visibility_t0) * 1000.0
                backward_ms += visibility_proxy_ms
                last_backward_end = visibility_t1
                visibility_proxy_loss_value = float(visibility_proxy_loss.detach().cpu().item())
                loss_value += float(weighted_visibility_proxy_loss.detach().cpu().item())
            optimizer.step()
            _sync_device(device)
            t4 = time.perf_counter()
            sparse_pixel_counts.append(step_sparse_pixel_count)
            sparse_pixel_fractions.append(
                0.0
                if step_sparse_total_pixels <= 0
                else float(step_sparse_pixel_count) / float(step_sparse_total_pixels)
            )
            if sparse_visual_enabled:
                sparse_visual_patch_phases.append([int(sparse_visual_patch_phase[0]), int(sparse_visual_patch_phase[1])])
                sparse_visual_pixel_counts.append(step_sparse_visual_pixel_count)
                sparse_visual_loss_sample_counts.append(step_sparse_visual_loss_sample_count)
                sparse_visual_alpha_sample_counts.append(step_sparse_visual_alpha_sample_count)
                sparse_visual_pixel_fractions.append(
                    0.0
                    if step_sparse_visual_total_pixels <= 0
                    else float(step_sparse_visual_pixel_count) / float(step_sparse_visual_total_pixels)
                )
            if trace_chunks is not None:
                chunk_traces.append(
                    {
                        "global_step": global_step,
                        "loss": loss_value,
                        "feature_target_loss": feature_target_loss_value,
                        "rgb_grid_loss": rgb_grid_loss_value,
                        "rgb_probe_loss": rgb_probe_loss_value,
                        "dense_alpha_loss": dense_alpha_loss_value,
                        "sparse_visual_loss": sparse_visual_loss_value,
                        "sparse_visual_alpha_loss": sparse_visual_alpha_loss_value,
                        "sparse_visual_black_hole_loss": sparse_visual_black_hole_loss_value,
                        "support_target_alpha_loss": support_target_alpha_loss_value,
                        "support_target_area_loss": support_target_area_loss_value,
                        "visibility_proxy_loss": visibility_proxy_loss_value,
                        "timing_ms": {
                            "render_forward_ms": render_forward_ms,
                            "colorize_loss_ms": colorize_loss_ms,
                            "feature_target_ms": feature_target_ms,
                            "rgb_grid_loss_ms": rgb_grid_loss_ms,
                            "rgb_probe_loss_ms": rgb_probe_loss_ms,
                            "sparse_pack_ms": sparse_pack_ms,
                            "dense_alpha_render_ms": dense_alpha_render_ms,
                            "dense_alpha_loss_ms": dense_alpha_loss_ms,
                            "dense_alpha_backward_ms": dense_alpha_backward_ms,
                            "support_target_area_ms": support_target_area_ms,
                            "support_target_alpha_ms": support_target_alpha_ms,
                            "support_prefix_alpha_ms": support_prefix_alpha_ms,
                            "visibility_proxy_ms": visibility_proxy_ms,
                            "backward_ms": backward_ms,
                            "optimizer_ms": (t4 - last_backward_end) * 1000.0,
                            "step_ms": (t4 - t0) * 1000.0,
                        },
                        "chunks": trace_chunks,
                    }
                )
            losses.append(loss_value)
            if rgb_loss_weight > 0.0:
                rgb_losses.append(rgb_loss_value)
            if target_feature is not None and feature_loss_weight > 0.0:
                feature_target_losses.append(feature_target_loss_value)
            if target_feature is not None and rgb_grid_loss_weight > 0.0:
                rgb_grid_losses.append(rgb_grid_loss_value)
            if rgb_probe is not None and rgb_probe_loss_weight > 0.0:
                rgb_probe_losses.append(rgb_probe_loss_value)
            if dense_alpha_enabled:
                dense_alpha_losses.append(dense_alpha_loss_value)
            if sparse_visual_enabled:
                sparse_visual_losses.append(sparse_visual_loss_value)
            if sparse_visual_enabled and sparse_visual_alpha_loss_weight > 0.0:
                sparse_visual_alpha_losses.append(sparse_visual_alpha_loss_value)
            if sparse_visual_enabled and sparse_visual_black_hole_loss_weight > 0.0:
                sparse_visual_black_hole_losses.append(sparse_visual_black_hole_loss_value)
            if support_target_alpha_enabled:
                support_target_alpha_losses.append(support_target_alpha_loss_value)
                support_target_alpha_sample_counts.append(step_support_target_alpha_sample_count)
            if support_target_area_enabled:
                support_target_area_losses.append(support_target_area_loss_value)
                support_target_area_sample_counts.append(step_support_target_area_sample_count)
            if support_prefix_alpha_enabled:
                support_prefix_alpha_losses.append(support_prefix_alpha_loss_value)
                support_prefix_alpha_sample_counts.append(step_support_prefix_alpha_sample_count)
                support_prefix_alpha_selected_weight_means.append(support_prefix_alpha_selected_weight_mean_value)
                support_prefix_alpha_selected_share_means.append(support_prefix_alpha_selected_share_mean_value)
                support_prefix_alpha_final_alpha_means.append(support_prefix_alpha_final_alpha_mean_value)
            if visibility_proxy_enabled:
                visibility_proxy_losses.append(visibility_proxy_loss_value)
            timings.append(
                {
                    "render_forward_ms": render_forward_ms,
                    "colorize_loss_ms": colorize_loss_ms,
                    "feature_target_ms": feature_target_ms,
                    "rgb_grid_loss_ms": rgb_grid_loss_ms,
                    "rgb_probe_loss_ms": rgb_probe_loss_ms,
                    "sparse_pack_ms": sparse_pack_ms,
                    "dense_alpha_render_ms": dense_alpha_render_ms,
                    "dense_alpha_loss_ms": dense_alpha_loss_ms,
                    "dense_alpha_backward_ms": dense_alpha_backward_ms,
                    "sparse_visual_render_ms": sparse_visual_render_ms,
                    "sparse_visual_loss_ms": sparse_visual_loss_ms,
                    "sparse_visual_backward_ms": sparse_visual_backward_ms,
                    "support_target_area_ms": support_target_area_ms,
                    "support_target_alpha_ms": support_target_alpha_ms,
                    "support_prefix_alpha_ms": support_prefix_alpha_ms,
                    "visibility_proxy_ms": visibility_proxy_ms,
                    "backward_ms": backward_ms,
                    "optimizer_ms": (t4 - last_backward_end) * 1000.0,
                    "step_ms": (t4 - t0) * 1000.0,
                }
            )
            final_grad_norms = _grad_norms(model, colorizer)

        with torch.no_grad():
            tile_counts: list[torch.Tensor] = []
            tile_overflow: list[torch.Tensor] = []
            tile_unstable: list[torch.Tensor] = []
            for frame_start in range(0, feature_config.frames, chunk_size):
                chunk_frames = min(chunk_size, feature_config.frames - frame_start)
                ma, q_uvt, depth0, depth_beta, opacity, feature = model.tensors()
                if chunk_frames == feature_config.frames:
                    aux = render_uvt_feature_tubes(
                        ma,
                        q_uvt,
                        depth0.detach(),
                        depth_beta.detach(),
                        opacity,
                        feature,
                        uvt_config,
                    )
                else:
                    ma_chunk = shift_ma_for_frame_chunk(
                        ma,
                        global_frames=uvt_config.frames,
                        frame_start=frame_start,
                        chunk_frames=chunk_frames,
                    )
                    aux = render_uvt_feature_tubes(
                        ma_chunk,
                        q_uvt,
                        depth0.detach(),
                        depth_beta.detach(),
                        opacity,
                        feature,
                        chunked_uvt_config(uvt_config, chunk_frames=chunk_frames),
                    )
                tile_counts.append(aux.tile_counts)
                tile_overflow.append(aux.tile_overflow)
                tile_unstable.append(aux.tile_unstable)
            tile_stats = _tile_load_stats(
                tile_counts=tile_counts,
                tile_overflow=tile_overflow,
                tile_unstable=tile_unstable,
                tile_capacity=int(cfg["feature_uvt"]["tile_capacity"]),
            )

        contact_sheet = path_or_none(cfg["output"]["contact_sheet"])
        side_by_side_video = path_or_none(cfg["output"]["side_by_side_video"])
        rgb_probe_contact_sheet = path_or_none(cfg["output"].get("rgb_probe_contact_sheet"))
        rgb_probe_side_by_side_video = path_or_none(cfg["output"].get("rgb_probe_side_by_side_video"))
        checkpoint = path_or_none(cfg["output"].get("checkpoint"))
        media_render_ms: float | None = None
        rgb_probe_media_render_ms: float | None = None
        final_full_rgb_loss: float | None = None
        target_thwc_cpu = target_thwc.detach().cpu()
        if contact_sheet is not None or side_by_side_video is not None:
            with torch.no_grad():
                if projective_interval_runtime_enabled:
                    if projective_interval_times is None:
                        raise RuntimeError("projective interval times were not initialized")
                    _sync_device(device)
                    media_t0 = time.perf_counter()
                    ma, q_uvt, depth0, depth_beta, opacity, feature = model.tensors()
                    render = _render_projective_interval_feature_tubes_autograd(
                        ma=ma,
                        q_uvt=q_uvt,
                        depth0=depth0.detach(),
                        depth_beta=depth_beta.detach(),
                        opacity=opacity,
                        feature=feature,
                        cfg=cfg,
                        feature_config=feature_config,
                        uvt_config=uvt_config,
                        times=projective_interval_times,
                        refresh_every=projective_interval_backend.refresh_every,
                        refresh_policy=projective_interval_backend.refresh_policy,
                    )
                    rgb = _compose_alpha_background_rgb(
                        render.feature_image,
                        render.alpha,
                        colorizer,
                        strategy=alpha_background_eval_strategy,
                        sample_scope=alpha_background_sample_scope,
                    )
                    final_rgb_thwc = rgb.permute(0, 2, 3, 1).detach().cpu()
                    _sync_device(device)
                    media_render_ms = (time.perf_counter() - media_t0) * 1000.0
                else:
                    final_rgb_thwc, media_render_ms = _render_rgb_chunks(
                        model=model,
                        colorizer=colorizer,
                        render_uvt_feature_tubes=render_uvt_feature_tubes,
                        shift_ma_for_frame_chunk=shift_ma_for_frame_chunk,
                        chunked_uvt_config=chunked_uvt_config,
                        uvt_config=uvt_config,
                        frames=feature_config.frames,
                        chunk_size=chunk_size,
                        device=device,
                        alpha_background_strategy=alpha_background_eval_strategy,
                        alpha_background_sample_scope=alpha_background_sample_scope,
                    )
                final_rgb_tchw = final_rgb_thwc.permute(0, 3, 1, 2).to(device=device, dtype=target_rgb.dtype)
                final_full_rgb_loss = float((final_rgb_tchw - target_rgb).square().mean().detach().cpu().item())
            write_prediction_media(
                target_thwc=target_thwc_cpu,
                pred_thwc=final_rgb_thwc,
                output_cfg=cfg["output"],
                data_cfg=cfg["data"],
            )
        if (
            rgb_probe is not None
            and target_feature is not None
            and (rgb_probe_contact_sheet is not None or rgb_probe_side_by_side_video is not None)
        ):
            with torch.no_grad():
                final_rgb_probe_thwc, rgb_probe_media_render_ms = _render_rgb_probe_chunks(
                    model=model,
                    rgb_probe=rgb_probe,
                    target_feature=target_feature,
                    render_uvt_feature_tubes=render_uvt_feature_tubes,
                    shift_ma_for_frame_chunk=shift_ma_for_frame_chunk,
                    chunked_uvt_config=chunked_uvt_config,
                    uvt_config=uvt_config,
                    frames=feature_config.frames,
                    height=feature_config.height,
                    width=feature_config.width,
                    chunk_size=chunk_size,
                    adapter=rgb_probe_adapter,
                    device=device,
                )
            write_prediction_media(
                target_thwc=target_thwc_cpu,
                pred_thwc=final_rgb_probe_thwc,
                output_cfg=cfg["output"],
                data_cfg=cfg["data"],
                contact_sheet_key="rgb_probe_contact_sheet",
                side_by_side_video_key="rgb_probe_side_by_side_video",
                side_by_side_fps_key="rgb_probe_side_by_side_fps",
            )

        timing_keys = (
            "render_forward_ms",
            "colorize_loss_ms",
            "feature_target_ms",
            "rgb_grid_loss_ms",
            "rgb_probe_loss_ms",
            "sparse_pack_ms",
            "dense_alpha_render_ms",
            "dense_alpha_loss_ms",
            "dense_alpha_backward_ms",
            "sparse_visual_render_ms",
            "sparse_visual_loss_ms",
            "sparse_visual_backward_ms",
            "support_target_area_ms",
            "support_target_alpha_ms",
            "support_prefix_alpha_ms",
            "visibility_proxy_ms",
            "backward_ms",
            "optimizer_ms",
            "step_ms",
        )
        mean_timing = mean_timing_ms(timings, timing_keys)
        timing_trace_summary = timing_trace_summary_ms(timings, timing_keys)
        start_loss = losses[0] if losses else None
        end_loss = losses[-1] if losses else None
        start_rgb_loss = rgb_losses[0] if rgb_losses else None
        end_rgb_loss = rgb_losses[-1] if rgb_losses else None
        start_feature_target_loss = feature_target_losses[0] if feature_target_losses else None
        end_feature_target_loss = feature_target_losses[-1] if feature_target_losses else None
        start_rgb_grid_loss = rgb_grid_losses[0] if rgb_grid_losses else None
        end_rgb_grid_loss = rgb_grid_losses[-1] if rgb_grid_losses else None
        start_rgb_probe_loss = rgb_probe_losses[0] if rgb_probe_losses else None
        end_rgb_probe_loss = rgb_probe_losses[-1] if rgb_probe_losses else None
        start_dense_alpha_loss = dense_alpha_losses[0] if dense_alpha_losses else None
        end_dense_alpha_loss = dense_alpha_losses[-1] if dense_alpha_losses else None
        start_sparse_visual_loss = sparse_visual_losses[0] if sparse_visual_losses else None
        end_sparse_visual_loss = sparse_visual_losses[-1] if sparse_visual_losses else None
        start_sparse_visual_alpha_loss = sparse_visual_alpha_losses[0] if sparse_visual_alpha_losses else None
        end_sparse_visual_alpha_loss = sparse_visual_alpha_losses[-1] if sparse_visual_alpha_losses else None
        start_sparse_visual_black_hole_loss = (
            sparse_visual_black_hole_losses[0] if sparse_visual_black_hole_losses else None
        )
        end_sparse_visual_black_hole_loss = (
            sparse_visual_black_hole_losses[-1] if sparse_visual_black_hole_losses else None
        )
        start_support_target_alpha_loss = (
            support_target_alpha_losses[0] if support_target_alpha_losses else None
        )
        end_support_target_alpha_loss = support_target_alpha_losses[-1] if support_target_alpha_losses else None
        start_support_target_area_loss = (
            support_target_area_losses[0] if support_target_area_losses else None
        )
        end_support_target_area_loss = support_target_area_losses[-1] if support_target_area_losses else None
        start_support_prefix_alpha_loss = (
            support_prefix_alpha_losses[0] if support_prefix_alpha_losses else None
        )
        end_support_prefix_alpha_loss = support_prefix_alpha_losses[-1] if support_prefix_alpha_losses else None
        start_visibility_proxy_loss = visibility_proxy_losses[0] if visibility_proxy_losses else None
        end_visibility_proxy_loss = visibility_proxy_losses[-1] if visibility_proxy_losses else None
        row: dict[str, Any] = {
            "gate": "star_uvt_feature_firstclass_overfit",
            "target_source": str(cfg["data"]["video_path"]),
            "frames": feature_config.frames,
            "size": feature_config.height,
            "tubes": int(cfg["feature_uvt"]["tube_count"]),
            "feature_dim": feature_config.feature_dim,
            "colorize_hidden_dim": cfg["colorize"]["hidden_dim"],
            "colorize_activation": str(cfg["colorize"]["activation"]),
            "colorize_pre_norm": bool(cfg["colorize"]["pre_norm"]),
            "colorize_weight_init": str(cfg["colorize"]["weight_init"]),
            "colorize_weight_init_gain": float(cfg["colorize"]["weight_init_gain"]),
            "colorize_init_checkpoint": colorizer_init_state["path"],
            "colorize_init_loaded": bool(colorizer_init_state["loaded"]),
            "feature_target_enabled": _feature_target_enabled(cfg),
            "feature_target": feature_target_meta,
            "feature_target_load_ms": feature_target_load_ms,
            "feature_target_loss_type": feature_loss_type if _feature_target_enabled(cfg) else None,
            "feature_target_image_vjp_mode": feature_target_image_vjp_mode,
            "sparse_image_vjp_enabled": sparse_image_vjp_enabled,
            "sparse_pixel_counts": sparse_pixel_counts,
            "sparse_pixel_fractions": sparse_pixel_fractions,
            "mean_sparse_pixel_count": (
                None if not sparse_pixel_counts else sum(sparse_pixel_counts) / float(len(sparse_pixel_counts))
            ),
            "mean_sparse_pixel_fraction": (
                None if not sparse_pixel_fractions else sum(sparse_pixel_fractions) / float(len(sparse_pixel_fractions))
            ),
            "dense_alpha_enabled": dense_alpha_enabled,
            "dense_alpha_loss_weight": dense_alpha_loss_weight,
            "dense_alpha_target": dense_alpha_target if dense_alpha_enabled else None,
            "dense_alpha_backward_mode": dense_alpha_backward_mode if dense_alpha_enabled else None,
            "dense_alpha_render_mode": dense_alpha_render_mode if dense_alpha_enabled else None,
            "dense_alpha_total_loss_elems": total_dense_alpha_loss_elems if dense_alpha_enabled else None,
            "sparse_visual_enabled": sparse_visual_enabled,
            "sparse_visual_loss_weight": sparse_visual_loss_weight,
            "sparse_visual_alpha_loss_weight": sparse_visual_alpha_loss_weight,
            "sparse_visual_alpha_target": sparse_visual_alpha_target if sparse_visual_enabled else None,
            "sparse_visual_black_hole_loss_weight": sparse_visual_black_hole_loss_weight,
            "sparse_visual_composition": sparse_visual_composition if sparse_visual_enabled else None,
            "sparse_visual_loss_basis": sparse_visual_loss_basis if sparse_visual_enabled else None,
            "sparse_visual_loss_vjp_mode": sparse_visual_loss_vjp_mode if sparse_visual_enabled else None,
            "sparse_visual_pixel_source": sparse_visual_pixel_source if sparse_visual_enabled else None,
            "sparse_visual_sample_grid_shape": list(sparse_visual_grid_shape) if sparse_visual_enabled else None,
            "sparse_visual_patch_shape": list(sparse_visual_patch_shape) if sparse_visual_enabled else None,
            "sparse_visual_patch_phase_shape": (
                list(sparse_visual_patch_phase_shape) if sparse_visual_enabled else None
            ),
            "sparse_visual_patch_phases": sparse_visual_patch_phases,
            "sparse_visual_pixel_counts": sparse_visual_pixel_counts,
            "sparse_visual_loss_sample_counts": sparse_visual_loss_sample_counts,
            "sparse_visual_alpha_sample_counts": sparse_visual_alpha_sample_counts,
            "sparse_visual_pixel_fractions": sparse_visual_pixel_fractions,
            "mean_sparse_visual_pixel_count": (
                None
                if not sparse_visual_pixel_counts
                else sum(sparse_visual_pixel_counts) / float(len(sparse_visual_pixel_counts))
            ),
            "mean_sparse_visual_loss_sample_count": (
                None
                if not sparse_visual_loss_sample_counts
                else sum(sparse_visual_loss_sample_counts) / float(len(sparse_visual_loss_sample_counts))
            ),
            "mean_sparse_visual_alpha_sample_count": (
                None
                if not sparse_visual_alpha_sample_counts
                else sum(sparse_visual_alpha_sample_counts) / float(len(sparse_visual_alpha_sample_counts))
            ),
            "mean_sparse_visual_pixel_fraction": (
                None
                if not sparse_visual_pixel_fractions
                else sum(sparse_visual_pixel_fractions) / float(len(sparse_visual_pixel_fractions))
            ),
            "visibility_proxy_enabled": visibility_proxy_enabled,
            "visibility_proxy_loss_weight": visibility_proxy_loss_weight,
            "visibility_proxy_center_weight": (
                visibility_proxy_center_weight if visibility_proxy_enabled else None
            ),
            "visibility_proxy_support_weight": (
                visibility_proxy_support_weight if visibility_proxy_enabled else None
            ),
            "visibility_proxy_support_epsilon": (
                visibility_proxy_support_epsilon if visibility_proxy_enabled else None
            ),
            "visibility_proxy_target_top_fraction": (
                float(visibility_proxy_cfg.get("target_top_fraction", 0.0)) if visibility_proxy_enabled else None
            ),
            "visibility_proxy_max_points": (
                int(visibility_proxy_cfg.get("max_points", 0)) if visibility_proxy_enabled else None
            ),
            "visibility_proxy_grid_stride": (
                int(visibility_proxy_cfg.get("grid_stride", 0)) if visibility_proxy_enabled else None
            ),
            "visibility_proxy_frame_stride": (
                int(visibility_proxy_cfg.get("frame_stride", 0)) if visibility_proxy_enabled else None
            ),
            "visibility_proxy_scale_px": visibility_proxy_scale_px if visibility_proxy_enabled else None,
            "visibility_proxy_temperature": visibility_proxy_temperature if visibility_proxy_enabled else None,
            "visibility_proxy_velocity_penalty": visibility_proxy_velocity_penalty if visibility_proxy_enabled else None,
            "visibility_proxy_target_point_count": (
                None if visibility_proxy_target_points is None else int(visibility_proxy_target_points.shape[0])
            ),
            "support_birth_split_enabled": support_birth_split_enabled,
            "support_birth_split": support_birth_split_state,
            "support_birth_split_target_point_source": (
                str(support_birth_split_cfg.get("target_point_source", "top_brightness"))
                if support_birth_split_enabled
                else None
            ),
            "support_birth_split_alpha_sample_ms": (
                support_birth_split_alpha_sample_ms if support_birth_split_enabled else None
            ),
            "support_birth_split_target_top_fraction": (
                float(support_birth_split_cfg.get("target_top_fraction", 0.0)) if support_birth_split_enabled else None
            ),
            "support_birth_split_max_points": (
                int(support_birth_split_cfg.get("max_points", 0)) if support_birth_split_enabled else None
            ),
            "support_birth_split_grid_stride": (
                int(support_birth_split_cfg.get("grid_stride", 0)) if support_birth_split_enabled else None
            ),
            "support_birth_split_frame_stride": (
                int(support_birth_split_cfg.get("frame_stride", 0)) if support_birth_split_enabled else None
            ),
            "support_birth_split_support_shape": (
                str(support_birth_split_cfg.get("support_shape", "isotropic")) if support_birth_split_enabled else None
            ),
            "support_birth_split_support_radius_along_px": (
                float(support_birth_split_cfg.get("support_radius_along_px", 0.0))
                if support_birth_split_enabled
                else None
            ),
            "support_birth_split_support_radius_across_px": (
                float(support_birth_split_cfg.get("support_radius_across_px", 0.0))
                if support_birth_split_enabled
                else None
            ),
            "support_birth_split_support_precision_radius_px": (
                float(support_birth_split_cfg.get("support_precision_radius_px", 0.0))
                if support_birth_split_enabled
                else None
            ),
            "support_birth_split_center_strategy": (
                str(support_birth_split_cfg.get("center_strategy", "global_line"))
                if support_birth_split_enabled
                else None
            ),
            "support_birth_split_center_count": (
                int(support_birth_split_cfg.get("center_count", 1)) if support_birth_split_enabled else None
            ),
            "support_birth_split_tube_allocation": (
                str(support_birth_split_cfg.get("tube_allocation", "proportional"))
                if support_birth_split_enabled
                else None
            ),
            "support_birth_split_target_point_count": (
                None
                if support_birth_split_target_points is None
                else int(support_birth_split_target_points.shape[0])
            ),
            "support_target_alpha_enabled": support_target_alpha_enabled,
            "support_target_alpha_loss_weight": support_target_alpha_loss_weight,
            "support_target_alpha_target": support_target_alpha_target if support_target_alpha_enabled else None,
            "support_target_alpha_max_points": (
                int(support_birth_split_cfg.get("target_alpha_max_points", 0))
                if support_target_alpha_enabled
                else None
            ),
            "support_target_alpha_target_point_count": (
                None if support_target_alpha_points is None else int(support_target_alpha_points.shape[0])
            ),
            "support_target_alpha_sample_counts": support_target_alpha_sample_counts,
            "mean_support_target_alpha_sample_count": (
                None
                if not support_target_alpha_sample_counts
                else sum(support_target_alpha_sample_counts) / float(len(support_target_alpha_sample_counts))
            ),
            "support_target_area_enabled": support_target_area_enabled,
            "support_target_area_loss_weight": support_target_area_loss_weight,
            "support_target_area_patch_shape": (
                list(support_target_area_patch_shape) if support_target_area_enabled else None
            ),
            "support_target_area_max_points": (
                int(support_birth_split_cfg.get("target_area_max_points", 0))
                if support_target_area_enabled
                else None
            ),
            "support_target_area_vjp_mode": support_target_area_vjp_mode if support_target_area_enabled else None,
            "support_target_area_composition": (
                support_target_area_composition if support_target_area_enabled else None
            ),
            "support_target_area_target_point_count": (
                None if support_target_area_points is None else int(support_target_area_points.shape[0])
            ),
            "support_target_area_sample_counts": support_target_area_sample_counts,
            "mean_support_target_area_sample_count": (
                None
                if not support_target_area_sample_counts
                else sum(support_target_area_sample_counts) / float(len(support_target_area_sample_counts))
            ),
            "support_prefix_alpha_enabled": support_prefix_alpha_enabled,
            "support_prefix_alpha_loss_weight": support_prefix_alpha_loss_weight,
            "support_prefix_alpha_target": support_prefix_alpha_target if support_prefix_alpha_enabled else None,
            "support_prefix_alpha_max_points": (
                int(support_birth_split_cfg.get("prefix_alpha_max_points", 0))
                if support_prefix_alpha_enabled
                else None
            ),
            "support_prefix_alpha_target_point_count": (
                None if support_prefix_alpha_points is None else int(support_prefix_alpha_points.shape[0])
            ),
            "support_prefix_alpha_sample_counts": support_prefix_alpha_sample_counts,
            "mean_support_prefix_alpha_sample_count": (
                None
                if not support_prefix_alpha_sample_counts
                else sum(support_prefix_alpha_sample_counts) / float(len(support_prefix_alpha_sample_counts))
            ),
            "support_prefix_alpha_selected_weight_means": support_prefix_alpha_selected_weight_means,
            "support_prefix_alpha_selected_share_means": support_prefix_alpha_selected_share_means,
            "support_prefix_alpha_final_alpha_means": support_prefix_alpha_final_alpha_means,
            "feature_target_loss_weight": base_feature_loss_weight,
            "rgb_loss_weight": base_rgb_loss_weight,
            "rgb_grid_loss_weight": base_rgb_grid_loss_weight,
            "rgb_grid_target_rgb_adapter": rgb_probe_adapter,
            "rgb_grid_target_rgb_shape": rgb_grid_target_shape,
            "rgb_probe_enabled": rgb_probe is not None,
            "rgb_probe": rgb_probe_meta,
            "rgb_probe_loss_weight": base_rgb_probe_loss_weight,
            "rgb_probe_target_rgb_adapter": rgb_probe_adapter,
            "rgb_probe_target_rgb_shape": rgb_probe_target_shape,
            "feature_target_weight_schedule": _feature_target_weight_schedule_json(weight_schedule),
            "optimizer_lr_schedule": _optimizer_lr_schedule_json(lr_schedule),
            "global_step_offset": global_step_offset,
            "start_global_step": global_step_offset,
            "end_global_step": global_step_offset + int(cfg["train"]["steps"]),
            "step_global_steps": step_global_steps,
            "step_feature_target_loss_weights": step_feature_target_loss_weights,
            "step_rgb_loss_weights": step_rgb_loss_weights,
            "step_rgb_grid_loss_weights": step_rgb_grid_loss_weights,
            "step_rgb_probe_loss_weights": step_rgb_probe_loss_weights,
            "step_lrs": step_lrs,
            "chunk_trace_global_steps": sorted(trace_global_steps),
            "resume_checkpoint": resume_state["path"],
            "resume_loaded": bool(resume_state["loaded"]),
            "resume_colorizer": bool(cfg["train"]["resume_colorizer"]),
            "resume_colorizer_loaded": bool(resume_state["colorizer_loaded"]),
            "resume_optimizer": bool(cfg["train"]["resume_optimizer"]),
            "resume_optimizer_loaded": bool(resume_state["optimizer_loaded"]),
            "resume_optimizer_lrs_loaded": resume_state["optimizer_lrs_loaded"],
            "resume_checkpoint_steps": resume_state["steps"],
            "resume_behavior": "warm_start_local_steps",
            "checkpoint": None if checkpoint is None else str(checkpoint),
            "steps": int(cfg["train"]["steps"]),
            "lr": configured_lr,
            "optimizer_lrs": _optimizer_lrs(optimizer),
            "frame_chunk_size": chunk_size,
            "tile_t": int(cfg["feature_uvt"]["tile_t"]),
            "tile_capacity": int(cfg["feature_uvt"]["tile_capacity"]),
            "alpha_threshold": float(cfg["feature_uvt"]["alpha_threshold"]),
            "max_alpha": float(cfg["feature_uvt"]["max_alpha"]),
            "requested_render_mode": requested_render_mode,
            "kernel_backward_mode": backward_mode,
            "projective_interval_enabled": bool(projective_interval_backend.enabled),
            "projective_interval_runtime_enabled": bool(projective_interval_runtime_enabled),
            "projective_interval_fallback_render_mode": projective_interval_backend.fallback_render_mode,
            "projective_interval_tile_size": int(projective_interval_backend.tile_size),
            "projective_interval_refresh_policy": projective_interval_backend.refresh_policy,
            "projective_interval_refresh_every": int(projective_interval_backend.refresh_every),
            "projective_interval_sigma_px": float(projective_interval_backend.sigma_px),
            "projective_interval_uv_padding": float(projective_interval_backend.uv_padding),
            "projective_interval_support_guard_padding": float(projective_interval_backend.support_guard_padding),
            "projective_interval_support_guard_policy": projective_interval_backend.support_guard_policy,
            "projective_interval_support_guard_bisect_steps": int(
                projective_interval_backend.support_guard_bisect_steps
            ),
            "projective_interval_support_stale_overshoot_epsilon": float(
                projective_interval_backend.support_stale_overshoot_epsilon
            ),
            "projective_interval_support_stale_tail_alpha_epsilon": float(
                projective_interval_backend.support_stale_tail_alpha_epsilon
            ),
            "projective_interval_effective_support_uv_padding": (
                None
                if projective_interval_cache is None or projective_interval_cache.state is None
                else float(projective_interval_cache.state.support_uv_padding)
            ),
            "projective_interval_spatial_precision_locked": bool(projective_interval_spatial_precision_locked),
            "projective_interval_locked_spatial_precision": projective_interval_spatial_precision,
            "projective_interval_alpha_render_mode": "white_trace" if projective_interval_runtime_enabled else None,
            "projective_interval_cache_rebuilds": (
                None if projective_interval_cache is None else int(projective_interval_cache.rebuild_count)
            ),
            "projective_interval_cache_live_updates": (
                None if projective_interval_cache is None else int(projective_interval_cache.live_update_count)
            ),
            "projective_interval_cache_alpha_renders": (
                None if projective_interval_cache is None else int(projective_interval_cache.alpha_render_count)
            ),
            "projective_interval_cache_staleness_checks": (
                None if projective_interval_cache is None else int(projective_interval_cache.staleness_check_count)
            ),
            "projective_interval_cache_stale_refreshes": (
                None if projective_interval_cache is None else int(projective_interval_cache.stale_refresh_count)
            ),
            "projective_interval_cache_support_rebins": (
                None if projective_interval_cache is None else int(projective_interval_cache.support_rebin_count)
            ),
            "projective_interval_cache_last_support_missing_tile_pairs": (
                None
                if projective_interval_cache is None
                else int(projective_interval_cache.last_support_margin_missing_tile_pairs)
            ),
            "projective_interval_cache_last_support_min_slack_px": (
                None
                if projective_interval_cache is None
                else float(projective_interval_cache.last_support_margin_min_slack_px)
            ),
            "projective_interval_cache_last_support_max_overshoot_px": (
                None
                if projective_interval_cache is None
                else float(projective_interval_cache.last_support_margin_max_overshoot_px)
            ),
            "projective_interval_cache_min_support_min_slack_px": (
                None
                if (
                    projective_interval_cache is None
                    or projective_interval_cache.min_support_margin_min_slack_px is None
                )
                else float(projective_interval_cache.min_support_margin_min_slack_px)
            ),
            "projective_interval_cache_max_support_max_overshoot_px": (
                None
                if projective_interval_cache is None
                else float(projective_interval_cache.max_support_margin_max_overshoot_px)
            ),
            "projective_interval_cache_last_support_tail_alpha_bound": (
                None if projective_interval_cache is None else float(projective_interval_cache.last_support_tail_alpha_bound)
            ),
            "projective_interval_cache_max_support_tail_alpha_bound": (
                None if projective_interval_cache is None else float(projective_interval_cache.max_support_tail_alpha_bound)
            ),
            "projective_interval_cache_visibility_stratifications": (
                None
                if projective_interval_cache is None
                else int(projective_interval_cache.visibility_stratify_count)
            ),
            "projective_interval_cache_fallback_marks": (
                None if projective_interval_cache is None else int(projective_interval_cache.fallback_mark_count)
            ),
            "requested_fixedbin_is_direct_atomic_alias": bool(
                requested_render_mode == "feature_direct_fixedbin" and backward_mode == "direct_atomic"
            ),
            "effective_render_mode": effective_feature_render_mode_for_report(requested_render_mode, feature_dim),
            "mode_fallback_required": feature_render_mode_fallback_required(
                requested_render_mode,
                feature_dim,
                tile_stats=tile_stats,
            ),
            "media_render_ms": media_render_ms,
            "final_full_rgb_loss": final_full_rgb_loss,
            "final_full_rgb_psnr": None if final_full_rgb_loss is None else _psnr(float(final_full_rgb_loss)),
            "rgb_probe_media_render_ms": rgb_probe_media_render_ms,
            "alpha_background_train_strategy": alpha_background_train_strategy,
            "alpha_background_eval_strategy": alpha_background_eval_strategy,
            "alpha_background_sample_scope": alpha_background_sample_scope,
            "contact_sheet": None if contact_sheet is None else str(contact_sheet),
            "contact_sheet_frames": int(cfg["output"]["contact_sheet_frames"]),
            "contact_sheet_mode": str(cfg["output"]["contact_sheet_mode"]),
            "side_by_side_video": None if side_by_side_video is None else str(side_by_side_video),
            "side_by_side_fps": cfg["output"]["side_by_side_fps"],
            "rgb_probe_contact_sheet": None
            if rgb_probe_contact_sheet is None
            else str(rgb_probe_contact_sheet),
            "rgb_probe_side_by_side_video": None
            if rgb_probe_side_by_side_video is None
            else str(rgb_probe_side_by_side_video),
            "rgb_probe_side_by_side_fps": cfg["output"]["rgb_probe_side_by_side_fps"],
            "start_loss": start_loss,
            "end_loss": end_loss,
            "start_psnr": None if start_loss is None else _psnr(float(start_loss)),
            "end_psnr": None if end_loss is None else _psnr(float(end_loss)),
            "start_rgb_loss": start_rgb_loss,
            "end_rgb_loss": end_rgb_loss,
            "start_rgb_psnr": None if start_rgb_loss is None else _psnr(float(start_rgb_loss)),
            "end_rgb_psnr": None if end_rgb_loss is None else _psnr(float(end_rgb_loss)),
            "start_feature_target_loss": start_feature_target_loss,
            "end_feature_target_loss": end_feature_target_loss,
            "start_rgb_grid_loss": start_rgb_grid_loss,
            "end_rgb_grid_loss": end_rgb_grid_loss,
            "start_rgb_grid_psnr": None if start_rgb_grid_loss is None else _psnr(float(start_rgb_grid_loss)),
            "end_rgb_grid_psnr": None if end_rgb_grid_loss is None else _psnr(float(end_rgb_grid_loss)),
            "start_rgb_probe_loss": start_rgb_probe_loss,
            "end_rgb_probe_loss": end_rgb_probe_loss,
            "start_rgb_probe_psnr": None if start_rgb_probe_loss is None else _psnr(float(start_rgb_probe_loss)),
            "end_rgb_probe_psnr": None if end_rgb_probe_loss is None else _psnr(float(end_rgb_probe_loss)),
            "start_dense_alpha_loss": start_dense_alpha_loss,
            "end_dense_alpha_loss": end_dense_alpha_loss,
            "start_dense_alpha_psnr": None if start_dense_alpha_loss is None else _psnr(float(start_dense_alpha_loss)),
            "end_dense_alpha_psnr": None if end_dense_alpha_loss is None else _psnr(float(end_dense_alpha_loss)),
            "start_sparse_visual_loss": start_sparse_visual_loss,
            "end_sparse_visual_loss": end_sparse_visual_loss,
            "start_sparse_visual_psnr": (
                None if start_sparse_visual_loss is None else _psnr(float(start_sparse_visual_loss))
            ),
            "end_sparse_visual_psnr": None if end_sparse_visual_loss is None else _psnr(float(end_sparse_visual_loss)),
            "start_sparse_visual_alpha_loss": start_sparse_visual_alpha_loss,
            "end_sparse_visual_alpha_loss": end_sparse_visual_alpha_loss,
            "start_sparse_visual_alpha_psnr": (
                None if start_sparse_visual_alpha_loss is None else _psnr(float(start_sparse_visual_alpha_loss))
            ),
            "end_sparse_visual_alpha_psnr": (
                None if end_sparse_visual_alpha_loss is None else _psnr(float(end_sparse_visual_alpha_loss))
            ),
            "start_sparse_visual_black_hole_loss": start_sparse_visual_black_hole_loss,
            "end_sparse_visual_black_hole_loss": end_sparse_visual_black_hole_loss,
            "start_sparse_visual_black_hole_psnr": (
                None
                if start_sparse_visual_black_hole_loss is None
                else _psnr(float(start_sparse_visual_black_hole_loss))
            ),
            "end_sparse_visual_black_hole_psnr": (
                None if end_sparse_visual_black_hole_loss is None else _psnr(float(end_sparse_visual_black_hole_loss))
            ),
            "start_support_target_alpha_loss": start_support_target_alpha_loss,
            "end_support_target_alpha_loss": end_support_target_alpha_loss,
            "start_support_target_alpha_psnr": (
                None if start_support_target_alpha_loss is None else _psnr(float(start_support_target_alpha_loss))
            ),
            "end_support_target_alpha_psnr": (
                None if end_support_target_alpha_loss is None else _psnr(float(end_support_target_alpha_loss))
            ),
            "start_support_target_area_loss": start_support_target_area_loss,
            "end_support_target_area_loss": end_support_target_area_loss,
            "start_support_target_area_psnr": (
                None if start_support_target_area_loss is None else _psnr(float(start_support_target_area_loss))
            ),
            "end_support_target_area_psnr": (
                None if end_support_target_area_loss is None else _psnr(float(end_support_target_area_loss))
            ),
            "start_support_prefix_alpha_loss": start_support_prefix_alpha_loss,
            "end_support_prefix_alpha_loss": end_support_prefix_alpha_loss,
            "start_support_prefix_alpha_psnr": (
                None if start_support_prefix_alpha_loss is None else _psnr(float(start_support_prefix_alpha_loss))
            ),
            "end_support_prefix_alpha_psnr": (
                None if end_support_prefix_alpha_loss is None else _psnr(float(end_support_prefix_alpha_loss))
            ),
            "start_visibility_proxy_loss": start_visibility_proxy_loss,
            "end_visibility_proxy_loss": end_visibility_proxy_loss,
            "support_target_alpha_loss_decreased": bool(
                support_target_alpha_losses and support_target_alpha_losses[-1] < support_target_alpha_losses[0]
            ),
            "support_target_area_loss_decreased": bool(
                support_target_area_losses and support_target_area_losses[-1] < support_target_area_losses[0]
            ),
            "support_prefix_alpha_loss_decreased": bool(
                support_prefix_alpha_losses and support_prefix_alpha_losses[-1] < support_prefix_alpha_losses[0]
            ),
            "visibility_proxy_loss_decreased": bool(
                visibility_proxy_losses and visibility_proxy_losses[-1] < visibility_proxy_losses[0]
            ),
            "rgb_loss_decreased": bool(rgb_losses and rgb_losses[-1] < rgb_losses[0]),
            "feature_target_loss_decreased": bool(
                feature_target_losses and feature_target_losses[-1] < feature_target_losses[0]
            ),
            "rgb_grid_loss_decreased": bool(rgb_grid_losses and rgb_grid_losses[-1] < rgb_grid_losses[0]),
            "rgb_probe_loss_decreased": bool(rgb_probe_losses and rgb_probe_losses[-1] < rgb_probe_losses[0]),
            "dense_alpha_loss_decreased": bool(
                dense_alpha_losses and dense_alpha_losses[-1] < dense_alpha_losses[0]
            ),
            "sparse_visual_loss_decreased": bool(
                sparse_visual_losses and sparse_visual_losses[-1] < sparse_visual_losses[0]
            ),
            "sparse_visual_alpha_loss_decreased": bool(
                sparse_visual_alpha_losses and sparse_visual_alpha_losses[-1] < sparse_visual_alpha_losses[0]
            ),
            "sparse_visual_black_hole_loss_decreased": bool(
                sparse_visual_black_hole_losses
                and sparse_visual_black_hole_losses[-1] < sparse_visual_black_hole_losses[0]
            ),
            "loss_decreased": bool(losses and losses[-1] < losses[0]),
            "losses": losses,
            "rgb_losses": rgb_losses,
            "feature_target_losses": feature_target_losses,
            "rgb_grid_losses": rgb_grid_losses,
            "rgb_probe_losses": rgb_probe_losses,
            "dense_alpha_losses": dense_alpha_losses,
            "sparse_visual_losses": sparse_visual_losses,
            "sparse_visual_alpha_losses": sparse_visual_alpha_losses,
            "sparse_visual_black_hole_losses": sparse_visual_black_hole_losses,
            "support_target_alpha_losses": support_target_alpha_losses,
            "support_target_area_losses": support_target_area_losses,
            "support_prefix_alpha_losses": support_prefix_alpha_losses,
            "visibility_proxy_losses": visibility_proxy_losses,
            "mean_timing_ms": mean_timing,
            "timing_trace_summary_ms": timing_trace_summary,
            "step_timings_ms": timings,
            "chunk_traces": chunk_traces,
            "last_timing_ms": timings[-1] if timings else None,
            "grad_norms": final_grad_norms,
            "raw_feature_grad_seen": final_grad_norms.get("model.raw_feature", 0.0) > 0.0,
            "center_uv_grad_seen": final_grad_norms.get("model.center_uv", 0.0) > 0.0,
            "center_t_grad_seen": final_grad_norms.get("model.center_t", 0.0) > 0.0,
            "velocity_uv_grad_seen": final_grad_norms.get("model.velocity_uv", 0.0) > 0.0,
            "raw_precision_grad_seen": final_grad_norms.get("model.raw_precision", 0.0) > 0.0,
            "raw_spatial_correlation_grad_seen": final_grad_norms.get("model.raw_spatial_correlation", 0.0) > 0.0,
            "raw_opacity_grad_seen": final_grad_norms.get("model.raw_opacity", 0.0) > 0.0,
            "colorizer_grad_seen": any(
                key.startswith("colorizer.") and value > 0.0 for key, value in final_grad_norms.items()
            ),
            "colorizer_grad_required": bool(colorizer_grad_required),
            "tile_stats": tile_stats,
            "tile_overflow_sum": int(tile_stats["overflow_tile_count"]),
            "tile_unstable_sum": int(tile_stats["unstable_tile_count"]),
            "fixedbin_eligible": bool(tile_stats["fixedbin_eligible"]),
        }
        row["pass"] = bool(
            row["loss_decreased"]
            and row["raw_feature_grad_seen"]
            and row["center_uv_grad_seen"]
            and row["center_t_grad_seen"]
            and row["velocity_uv_grad_seen"]
            and row["raw_precision_grad_seen"]
            and row["raw_opacity_grad_seen"]
            and (not row["colorizer_grad_required"] or row["colorizer_grad_seen"])
            and (not rgb_grid_loss_required or row["rgb_grid_loss_decreased"])
            and (not rgb_probe_loss_required or row["rgb_probe_loss_decreased"])
            and (not dense_alpha_enabled or row["dense_alpha_loss_decreased"])
            and (not sparse_visual_enabled or row["sparse_visual_loss_decreased"])
            and (sparse_visual_alpha_loss_weight <= 0.0 or row["sparse_visual_alpha_loss_decreased"])
            and (sparse_visual_black_hole_loss_weight <= 0.0 or row["sparse_visual_black_hole_loss_decreased"])
            and (not support_target_alpha_enabled or row["support_target_alpha_loss_decreased"])
            and (not support_target_area_enabled or row["support_target_area_loss_decreased"])
            and (not support_prefix_alpha_enabled or row["support_prefix_alpha_loss_decreased"])
            and (not visibility_proxy_enabled or row["visibility_proxy_loss_decreased"])
            and row["tile_overflow_sum"] == 0
        )
        if checkpoint is not None:
            _save_training_checkpoint(
                checkpoint,
                model=model,
                colorizer=colorizer,
                optimizer=optimizer,
                cfg=cfg,
                row=row,
            )
        if run is not None:
            log_star_uvt_row_outputs(
                row,
                cfg,
                metric_prefix="star_uvt_feature",
                image_outputs=(
                    ("contact_sheet", "media/contact_sheet"),
                    ("rgb_probe_contact_sheet", "media/rgb_probe_contact_sheet"),
                ),
                video_outputs=(
                    ("side_by_side_video", "media/side_by_side_video"),
                    ("rgb_probe_side_by_side_video", "media/rgb_probe_side_by_side_video"),
                ),
            )
        write_row_json_and_print(row, cfg["output"]["out_json"])
        _assert_requirements(row, cfg)
        return row
    finally:
        finish_wandb_run(run)
