from __future__ import annotations

import time
from typing import Any

import torch

from config_utils import path_or_none
from star_uvt_checkpoints import save_feature_rgb_probe_checkpoint
from star_uvt_colorizers import build_feature_colorizer
from star_uvt_common import load_training_sequence as _load_training_sequence
from star_uvt_feature_rgb_probe_config import resolve_config
from star_uvt_feature_targets import (
    _load_cached_feature_target,
    adapt_rgb_to_grid,
    mean_rgb_grid_loss,
    upsample_grid_rgb,
)
from star_uvt_outputs import log_star_uvt_row_outputs, write_prediction_media, write_row_json_and_print
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
    device = _resolve_device(str(cfg["probe"]["device"]))
    torch.manual_seed(int(cfg["probe"]["seed"]))
    sequence = _load_training_sequence(cfg, device)
    frames = int(cfg["data"]["max_frames"])
    size = int(cfg["data"]["target_size"])
    feature_dim = int(cfg["feature_uvt"]["feature_dim"])
    _sync_device(device)
    target_t0 = time.perf_counter()
    target_feature = _load_cached_feature_target(
        cfg=cfg,
        sequence_data=sequence,
        device=device,
        frames=frames,
        height=size,
        width=size,
        feature_dim=feature_dim,
    )
    _sync_device(device)
    feature_target_load_ms = (time.perf_counter() - target_t0) * 1000.0
    if target_feature.source is None:
        raise RuntimeError("target-grid RGB probe expected a source target grid")

    target_grid = target_feature.source.detach().contiguous()
    target_grid_shape = (
        int(target_grid.shape[0]),
        int(target_grid.shape[2]),
        int(target_grid.shape[3]),
    )
    rgb_adapter = str(cfg["probe"]["target_rgb_adapter"])
    target_grid_rgb = adapt_rgb_to_grid(sequence.frames, target_shape=target_grid_shape, mode=rgb_adapter)
    target_full_rgb = sequence.frames.contiguous()

    colorizer = build_feature_colorizer(cfg["colorize"], feature_dim=feature_dim, device=device)
    optimizer = torch.optim.Adam(colorizer.parameters(), lr=float(cfg["probe"]["lr"]))

    losses: list[float] = []
    timings: list[dict[str, float]] = []
    run = init_wandb_run(cfg)
    try:
        for _step in range(int(cfg["probe"]["steps"])):
            optimizer.zero_grad(set_to_none=True)
            _sync_device(device)
            t0 = time.perf_counter()
            pred = colorizer(target_grid)
            loss = mean_rgb_grid_loss(pred, target_grid_rgb)
            _sync_device(device)
            t1 = time.perf_counter()
            loss.backward()
            _sync_device(device)
            t2 = time.perf_counter()
            optimizer.step()
            _sync_device(device)
            t3 = time.perf_counter()
            losses.append(float(loss.detach().cpu().item()))
            timings.append(
                {
                    "forward_loss_ms": (t1 - t0) * 1000.0,
                    "backward_ms": (t2 - t1) * 1000.0,
                    "optimizer_ms": (t3 - t2) * 1000.0,
                    "step_ms": (t3 - t0) * 1000.0,
                }
            )

        with torch.no_grad():
            _sync_device(device)
            media_t0 = time.perf_counter()
            pred_grid_rgb = colorizer(target_grid).detach()
            pred_full_rgb = upsample_grid_rgb(
                pred_grid_rgb,
                target_shape=(frames, size, size),
                mode=rgb_adapter,
            )
            _sync_device(device)
            media_render_ms = (time.perf_counter() - media_t0) * 1000.0
            grid_loss = float(mean_rgb_grid_loss(pred_grid_rgb, target_grid_rgb).detach().cpu().item())
            full_loss = float(mean_rgb_grid_loss(pred_full_rgb, target_full_rgb).detach().cpu().item())

        target_full_thwc = target_full_rgb.permute(0, 2, 3, 1).detach().cpu()
        pred_full_thwc = pred_full_rgb.permute(0, 2, 3, 1).detach().cpu()
        contact_sheet, side_by_side_video = write_prediction_media(
            target_thwc=target_full_thwc,
            pred_thwc=pred_full_thwc,
            output_cfg=cfg["output"],
            data_cfg=cfg["data"],
        )

        timing_keys = ("forward_loss_ms", "backward_ms", "optimizer_ms", "step_ms")
        mean_timing = mean_timing_ms(timings, timing_keys)
        checkpoint = path_or_none(cfg["output"]["checkpoint"])
        if checkpoint is not None:
            save_feature_rgb_probe_checkpoint(
                checkpoint,
                colorizer=colorizer,
                cfg=cfg,
                feature_target_meta=target_feature.meta,
                target_grid_shape=target_grid.shape,
                target_rgb_shape=target_grid_rgb.shape,
                grid_loss=grid_loss,
                full_loss=full_loss,
            )

        row: dict[str, Any] = {
            "gate": "star_uvt_target_grid_feature_to_rgb_probe",
            "target_source": str(cfg["data"]["video_path"]),
            "frames": frames,
            "size": size,
            "feature_dim": feature_dim,
            "steps": int(cfg["probe"]["steps"]),
            "lr": float(cfg["probe"]["lr"]),
            "device": str(device),
            "target_rgb_adapter": rgb_adapter,
            "colorize_hidden_dim": cfg["colorize"]["hidden_dim"],
            "colorize_activation": str(cfg["colorize"]["activation"]),
            "colorize_pre_norm": bool(cfg["colorize"]["pre_norm"]),
            "colorize_weight_init": str(cfg["colorize"]["weight_init"]),
            "colorize_weight_init_gain": float(cfg["colorize"]["weight_init_gain"]),
            "feature_target": target_feature.meta,
            "feature_target_load_ms": feature_target_load_ms,
            "target_grid_shape": list(target_grid.shape),
            "target_grid_rgb_shape": list(target_grid_rgb.shape),
            "target_full_rgb_shape": list(target_full_rgb.shape),
            "start_grid_loss": losses[0] if losses else None,
            "end_grid_loss": losses[-1] if losses else None,
            "start_grid_psnr": None if not losses else _psnr(losses[0]),
            "end_grid_psnr": None if not losses else _psnr(losses[-1]),
            "final_grid_loss": grid_loss,
            "final_grid_psnr": _psnr(grid_loss),
            "final_full_loss": full_loss,
            "final_full_psnr": _psnr(full_loss),
            "loss_decreased": bool(losses and losses[-1] < losses[0]),
            "losses": losses,
            "mean_timing_ms": mean_timing,
            "last_timing_ms": timings[-1] if timings else None,
            "media_render_ms": media_render_ms,
            "checkpoint": None if checkpoint is None else str(checkpoint),
            "contact_sheet": None if contact_sheet is None else str(contact_sheet),
            "side_by_side_video": None if side_by_side_video is None else str(side_by_side_video),
            "wandb_run_id": None if run is None else run.id,
            "wandb_run_name": cfg["logging"]["wandb_run_name"],
            "wandb_mode": cfg["logging"]["wandb_mode"],
            "pass": bool(
                losses
                and losses[-1] < losses[0]
                and grid_loss < losses[0]
            ),
        }
        if run is not None:
            log_star_uvt_row_outputs(row, cfg, metric_prefix="star_uvt_feature_rgb_probe")
        write_row_json_and_print(row, cfg["output"]["out_json"])
        if bool(cfg["probe"]["require_loss_decrease"]) and not row["loss_decreased"]:
            raise AssertionError(f"loss did not decrease: {row['start_grid_loss']} -> {row['end_grid_loss']}")
        return row
    finally:
        finish_wandb_run(run)


__all__ = ["run_probe"]
