from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from config_utils import path_or_none
from star_uvt_checkpoints import (
    load_star_model_from_training_checkpoint as _load_star_model_from_checkpoint,
    save_rendered_feature_rgb_probe_checkpoint,
)
from star_uvt_colorizers import build_feature_colorizer, set_module_trainable
from star_uvt_common import (
    grad_norms as _grad_norms,
    load_colorizer_init_checkpoint as _load_colorizer_init_checkpoint,
    load_training_sequence as _load_training_sequence,
)
from star_uvt_feature_rendering import _render_rgb_chunks
from star_uvt_models import build_feature_tube_model
from star_uvt_outputs import log_star_uvt_row_outputs, write_prediction_media, write_row_json_and_print
from star_uvt_rendered_feature_probe_config import resolve_config
from star_uvt_rendered_feature_probe_objective import (
    _pixel_ids_for_chunk,
    compose_sparse_rgb,
    gather_target_rgb_values,
    sparse_rgb_loss_and_grads,
)
from star_uvt_render_configs import star_uvt_render_configs_from_cfg
from star_uvt_runtime import (
    ensure_star_uvt_on_path as _ensure_star_uvt_on_path,
    psnr_from_loss as _psnr,
    resolve_device as _resolve_device,
    sync_device as _sync_device,
)
from star_uvt_timing import mean_timing_ms
from train_logging import finish_wandb_run, init_wandb_run


def run_probe(config: dict[str, Any]) -> dict[str, Any]:
    cfg = resolve_config(config)
    _ensure_star_uvt_on_path()
    from torch_gsplat_bridge_star_uvt.feature_rasterize import (
        chunked_uvt_config,
        direct_atomic_feature_sparse_pixels_backward_cached_bins,
        render_uvt_feature_sparse_pixels_with_bins,
        render_uvt_feature_tubes,
        shift_ma_for_frame_chunk,
    )

    device = _resolve_device(str(cfg["probe"]["device"]))
    if device.type != "mps":
        raise RuntimeError("Rendered STAR UVT sparse-pixel probe currently requires MPS")
    torch.manual_seed(int(cfg["probe"]["seed"]))
    sequence = _load_training_sequence(cfg, device)
    target_rgb = sequence.frames.contiguous()
    target_thwc = target_rgb.permute(0, 2, 3, 1).detach().cpu().contiguous()
    frames = int(cfg["data"]["max_frames"])
    size = int(cfg["data"]["target_size"])
    feature_config, uvt_config = star_uvt_render_configs_from_cfg(cfg)
    feature_dim = int(feature_config.feature_dim)
    model = build_feature_tube_model(cfg, feature_config, device=device, seed_section="probe")
    train_star_model = bool(cfg["probe"]["train_star_model"])
    train_colorizer = bool(cfg["probe"]["train_colorizer"])
    resume_state = _load_star_model_from_checkpoint(
        Path(cfg["probe"]["resume_checkpoint"]),
        model=model,
        device=device,
        freeze_model=not train_star_model,
    )
    colorizer = build_feature_colorizer(cfg["colorize"], feature_dim=feature_dim, device=device)
    colorizer_init_state: dict[str, Any] = {"path": None, "loaded": False}
    colorizer_init_checkpoint = path_or_none(cfg["probe"]["colorizer_init_checkpoint"])
    if colorizer_init_checkpoint is not None:
        colorizer_init_state = _load_colorizer_init_checkpoint(
            colorizer_init_checkpoint,
            colorizer=colorizer,
            device=device,
        )
    set_module_trainable(colorizer, train_colorizer)
    optimizer_params = [param for param in (*model.parameters(), *colorizer.parameters()) if param.requires_grad]
    if not optimizer_params:
        raise RuntimeError("probe has no trainable parameters")
    optimizer = torch.optim.Adam(optimizer_params, lr=float(cfg["probe"]["lr"]))
    chunk_size = min(int(cfg["probe"]["frame_chunk_size"]), frames)
    sample_grid_shape = tuple(int(item) for item in cfg["probe"]["sample_grid_shape"])
    sample_grid_adapter = str(cfg["probe"]["sample_grid_adapter"])
    pixel_source = str(cfg["probe"]["pixel_source"])
    expected_step_pixels = 0
    for frame_start in range(0, frames, chunk_size):
        chunk_frames = min(chunk_size, frames - frame_start)
        expected_step_pixels += int(
            _pixel_ids_for_chunk(
                pixel_source=pixel_source,
                chunk_frames=chunk_frames,
                feature_dim=feature_dim,
                height=size,
                width=size,
                render_frames=frames,
                frame_start=frame_start,
                sample_grid_shape=sample_grid_shape,
                sample_grid_adapter=sample_grid_adapter,
                device=device,
            ).numel()
        )
    total_loss_elems = max(expected_step_pixels * 3, 1)
    losses: list[float] = []
    timings: list[dict[str, float]] = []
    sample_pixel_counts: list[int] = []
    last_grad_norms: dict[str, float] = {}
    run = init_wandb_run(cfg)
    try:
        for _step in range(int(cfg["probe"]["steps"])):
            optimizer.zero_grad(set_to_none=True)
            step_loss = target_rgb.new_zeros(())
            step_pixels = 0
            render_forward_ms = 0.0
            colorize_loss_ms = 0.0
            local_backward_ms = 0.0
            native_backward_ms = 0.0
            _sync_device(device)
            step_t0 = time.perf_counter()
            for frame_start in range(0, frames, chunk_size):
                chunk_frames = min(chunk_size, frames - frame_start)
                pixel_ids = _pixel_ids_for_chunk(
                    pixel_source=pixel_source,
                    chunk_frames=chunk_frames,
                    feature_dim=feature_dim,
                    height=size,
                    width=size,
                    render_frames=frames,
                    frame_start=frame_start,
                    sample_grid_shape=sample_grid_shape,
                    sample_grid_adapter=sample_grid_adapter,
                    device=device,
                )
                if int(pixel_ids.numel()) == 0:
                    continue
                ma, q_uvt, depth0, depth_beta, opacity, feature = model.tensors()
                if chunk_frames == frames:
                    render_inputs = (ma, q_uvt, depth0.detach(), depth_beta.detach(), opacity, feature)
                    render_config = uvt_config
                else:
                    ma_chunk = shift_ma_for_frame_chunk(
                        ma,
                        global_frames=frames,
                        frame_start=frame_start,
                        chunk_frames=chunk_frames,
                    )
                    render_inputs = (ma_chunk, q_uvt, depth0.detach(), depth_beta.detach(), opacity, feature)
                    render_config = chunked_uvt_config(uvt_config, chunk_frames=chunk_frames)
                _sync_device(device)
                render_t0 = time.perf_counter()
                with torch.no_grad():
                    render = render_uvt_feature_sparse_pixels_with_bins(*render_inputs, pixel_ids, render_config)
                feature_values = render.feature_values.detach()
                alpha_values = render.alpha_values.detach()
                _sync_device(device)
                render_t1 = time.perf_counter()
                target_values = gather_target_rgb_values(
                    target_rgb[frame_start : frame_start + chunk_frames],
                    pixel_ids,
                )
                loss_t0 = time.perf_counter()
                if train_star_model:
                    chunk_loss, grad_feature_values, grad_alpha_values = sparse_rgb_loss_and_grads(
                        feature_values,
                        alpha_values,
                        target_values,
                        colorizer,
                        total_loss_elems=total_loss_elems,
                    )
                    _sync_device(device)
                    loss_t1 = time.perf_counter()
                    if render.tile_tube_ids is None or render.tile_depths is None:
                        raise RuntimeError("native sparse visual VJP requires render bins")
                    grad_ma, grad_q, grad_opacity, grad_feature, _tile_unstable = (
                        direct_atomic_feature_sparse_pixels_backward_cached_bins(
                            *render_inputs,
                            pixel_ids,
                            grad_feature_values,
                            grad_alpha_values,
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
                    native_t1 = time.perf_counter()
                    local_backward_ms += (loss_t1 - loss_t0) * 1000.0
                    native_backward_ms += (native_t1 - loss_t1) * 1000.0
                    step_loss = step_loss + chunk_loss
                    colorize_loss_ms += (loss_t1 - loss_t0) * 1000.0
                else:
                    pred_values = compose_sparse_rgb(feature_values, alpha_values, colorizer)
                    chunk_loss_sum = (pred_values - target_values).square().sum()
                    step_loss = step_loss + chunk_loss_sum
                    _sync_device(device)
                    loss_t1 = time.perf_counter()
                step_pixels += int(pixel_ids.numel())
                render_forward_ms += (render_t1 - render_t0) * 1000.0
            _sync_device(device)
            loss_t1 = time.perf_counter()
            if train_star_model:
                loss = step_loss
                backward_t1 = loss_t1
            else:
                loss = step_loss / float(max(step_pixels * 3, 1))
                colorize_loss_ms += (loss_t1 - step_t0) * 1000.0 - render_forward_ms
                loss.backward()
                _sync_device(device)
                backward_t1 = time.perf_counter()
            optimizer.step()
            last_grad_norms = _grad_norms(model, colorizer)
            _sync_device(device)
            opt_t1 = time.perf_counter()
            losses.append(float(loss.detach().cpu().item()))
            sample_pixel_counts.append(step_pixels)
            timings.append(
                {
                    "render_forward_ms": render_forward_ms,
                    "colorize_loss_ms": colorize_loss_ms,
                    "local_backward_ms": local_backward_ms,
                    "native_backward_ms": native_backward_ms,
                    "backward_ms": (
                        local_backward_ms + native_backward_ms
                        if train_star_model
                        else (backward_t1 - loss_t1) * 1000.0
                    ),
                    "optimizer_ms": (opt_t1 - backward_t1) * 1000.0,
                    "step_ms": (opt_t1 - step_t0) * 1000.0,
                }
            )

        colorizer.eval()
        with torch.no_grad():
            pred_thwc, media_render_ms = _render_rgb_chunks(
                model=model,
                colorizer=colorizer,
                render_uvt_feature_tubes=render_uvt_feature_tubes,
                shift_ma_for_frame_chunk=shift_ma_for_frame_chunk,
                chunked_uvt_config=chunked_uvt_config,
                uvt_config=uvt_config,
                frames=frames,
                chunk_size=chunk_size,
                device=device,
                alpha_background_strategy="fixed_black_after_colorizer",
                alpha_background_sample_scope="step",
            )
            pred_tchw = pred_thwc.permute(0, 3, 1, 2).to(device=device, dtype=target_rgb.dtype)
            final_full_loss = float((pred_tchw - target_rgb).square().mean().detach().cpu().item())

        contact_sheet, side_by_side_video = write_prediction_media(
            target_thwc=target_thwc,
            pred_thwc=pred_thwc,
            output_cfg=cfg["output"],
            data_cfg=cfg["data"],
        )

        checkpoint = path_or_none(cfg["output"]["checkpoint"])
        if checkpoint is not None:
            save_rendered_feature_rgb_probe_checkpoint(
                checkpoint,
                model=model,
                colorizer=colorizer,
                optimizer=optimizer,
                cfg=cfg,
                resume_state=resume_state,
                colorizer_init_state=colorizer_init_state,
                train_star_model=train_star_model,
                sparse_sample_loss=losses[-1] if losses else None,
                full_loss=final_full_loss,
            )

        row: dict[str, Any] = {
            "gate": "star_uvt_rendered_feature_rgb_probe",
            "target_source": str(cfg["data"]["video_path"]),
            "resume_state": resume_state,
            "frames": frames,
            "size": size,
            "tubes": int(cfg["feature_uvt"]["tube_count"]),
            "feature_dim": feature_dim,
            "steps": int(cfg["probe"]["steps"]),
            "lr": float(cfg["probe"]["lr"]),
            "device": str(device),
            "frame_chunk_size": chunk_size,
            "pixel_source": pixel_source,
            "train_star_model": train_star_model,
            "train_colorizer": train_colorizer,
            "colorizer_init_checkpoint": colorizer_init_state["path"],
            "colorizer_init_loaded": bool(colorizer_init_state["loaded"]),
            "sample_grid_shape": list(sample_grid_shape),
            "sample_grid_adapter": sample_grid_adapter,
            "mean_sample_pixel_count": (
                sum(sample_pixel_counts) / float(len(sample_pixel_counts)) if sample_pixel_counts else 0.0
            ),
            "mean_sample_pixel_fraction": (
                (sum(sample_pixel_counts) / float(len(sample_pixel_counts))) / float(frames * size * size)
                if sample_pixel_counts
                else 0.0
            ),
            "colorize_hidden_dim": cfg["colorize"]["hidden_dim"],
            "colorize_activation": str(cfg["colorize"]["activation"]),
            "colorize_pre_norm": bool(cfg["colorize"]["pre_norm"]),
            "colorize_weight_init": str(cfg["colorize"]["weight_init"]),
            "colorize_weight_init_gain": float(cfg["colorize"]["weight_init_gain"]),
            "start_sparse_sample_loss": losses[0] if losses else None,
            "end_sparse_sample_loss": losses[-1] if losses else None,
            "start_sparse_sample_psnr": None if not losses else _psnr(losses[0]),
            "end_sparse_sample_psnr": None if not losses else _psnr(losses[-1]),
            "final_full_loss": final_full_loss,
            "final_full_psnr": _psnr(final_full_loss),
            "loss_decreased": bool(losses and losses[-1] < losses[0]),
            "losses": losses,
            "mean_timing_ms": mean_timing_ms(timings),
            "last_timing_ms": timings[-1] if timings else None,
            "grad_norms": last_grad_norms,
            "model_grad_seen": any(
                key.startswith("model.") and value > 0.0 for key, value in last_grad_norms.items()
            ),
            "colorizer_grad_seen": any(
                key.startswith("colorizer.") and value > 0.0 for key, value in last_grad_norms.items()
            ),
            "media_render_ms": media_render_ms,
            "checkpoint": None if checkpoint is None else str(checkpoint),
            "contact_sheet": None if contact_sheet is None else str(contact_sheet),
            "side_by_side_video": None if side_by_side_video is None else str(side_by_side_video),
            "wandb_run_id": None if run is None else run.id,
            "wandb_run_name": cfg["logging"]["wandb_run_name"],
            "wandb_mode": cfg["logging"]["wandb_mode"],
            "pass": bool(losses and losses[-1] < losses[0]),
        }
        if run is not None:
            log_star_uvt_row_outputs(row, cfg, metric_prefix="star_uvt_rendered_feature_rgb_probe")
        write_row_json_and_print(row, cfg["output"]["out_json"])
        if bool(cfg["probe"]["require_loss_decrease"]) and not row["loss_decreased"]:
            raise AssertionError(
                f"sparse sample loss did not decrease: {row['start_sparse_sample_loss']} -> "
                f"{row['end_sparse_sample_loss']}"
            )
        return row
    finally:
        finish_wandb_run(run)


__all__ = ["run_probe"]
