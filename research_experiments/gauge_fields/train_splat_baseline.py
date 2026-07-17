from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from common import (
    DYNAWORLD_ROOT,
    prefix_metrics,
    resolve_device,
    robust_l1,
    save_preview_strip,
    save_side_by_side_mp4,
    scalar_background,
    video_metrics,
    write_checkpoint,
    write_json,
    resolve_dynaworld_path,
)
from camera import CameraSpec
from config_utils import apply_defaults, load_config_file, resolved_config, serialize_config_value
from data import initialize_material_points_from_first_frame, load_gauge_video_bundle
from rendering import render_gaussian_frame
from runtime_types import GaussianFrame


DATA_DEFAULTS = {
    "sequence_dir": "test_data",
    "frames_dir": None,
    "video_path": "test_data/test_video_small_128_4fps.mp4",
    "frame_source": "explicit_video",
    "max_frames": 0,
    "frame_indices": None,
    "multicam_manifest": "data/multicam_val/clip_sets/multicam_val_v1_128_4fps_16f/manifest.jsonl",
    "multicam_split": "val",
    "multicam_sample_id": None,
    "multicam_sample_index": 0,
    "multicam_train_cameras": None,
    "multicam_heldout_camera": None,
    "multicam_anchor_camera": None,
}

MODEL_DEFAULTS = {
    "support_mode": "free_dynamic_3dgs",
    "splat_mode": "per_frame",
    "num_splats": 2048,
    "init_depth": 0.5,
    "init_scale": 0.035,
    "scale_init_log_jitter": 0.0,
    "init_alpha_logit": 0.0,
    "init_xyz_noise": 0.001,
    "init_quat_noise": 0.0,
    "log_scale_min": -12.0,
    "log_scale_max": 4.0,
}

CAMERA_DEFAULTS = {
    "lens_model": "pinhole",
    "base_fov_degrees": 60.0,
    "multicam_pose_source": "auto",
}

RENDER_DEFAULTS = {
    "render_size": 128,
    "background": [1.0, 1.0, 1.0],
    "near_plane": 1e-3,
    "renderer": "dense",
    "tile_size": 8,
    "bound_scale": 3.0,
    "alpha_threshold": 1.0 / 255.0,
    "camera_projection": "legacy_pinhole",
}

TRAIN_DEFAULTS = {
    "steps": 250,
    "lr": 2e-3,
    "device": "auto",
    "seed": 0,
    "frames_per_step": 1,
    "train_frame_count": 16,
}

LOSS_DEFAULTS = {
    "rgb_weight": 1.0,
    "scale_weight": 1e-4,
    "temporal_smooth_weight": 1e-3,
}

LOGGING_DEFAULTS = {
    "log_every": 25,
    "log_to_wandb": False,
    "wandb_project": "dynaworld",
    "wandb_run_name": "splat-baseline-free-dynamic-3dgs",
    "wandb_tags": ["splat-baseline", "free-dynamic-3dgs"],
    "wandb_mode": "online",
    "output_dir": "outputs/gauge_fields/splat_baseline_free_dynamic_3dgs",
}


def logit(value: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    value = value.clamp(eps, 1.0 - eps)
    return torch.log(value) - torch.log1p(-value)


def camera_from_K_w2c(K: torch.Tensor, w2c: torch.Tensor) -> CameraSpec:
    c2w = torch.linalg.inv(w2c)
    return CameraSpec(
        fx=K[0, 0],
        fy=K[1, 1],
        cx=K[0, 2],
        cy=K[1, 2],
        camera_to_world=c2w,
        lens_model="pinhole",
    )


def select_K_for_view_time(K: torch.Tensor, *, view: int, t: int, view_count: int) -> torch.Tensor:
    if K.ndim == 4:
        return K[view, t]
    if K.ndim == 3:
        if K.shape[0] == view_count:
            return K[view]
        return K[t]
    return K


def select_w2c_for_view_time(w2c: torch.Tensor, *, view: int, t: int) -> torch.Tensor:
    if w2c.ndim == 4:
        return w2c[view, t]
    return w2c[t]


def camera_list_from_K_w2c(K: torch.Tensor, w2c: torch.Tensor) -> list[CameraSpec]:
    return [camera_from_K_w2c(K[t] if K.ndim == 3 else K, w2c[t]) for t in range(w2c.shape[0])]


class FreeDynamic3DGS(nn.Module):
    def __init__(
        self,
        *,
        init_xyz: torch.Tensor,
        init_rgb: torch.Tensor,
        num_frames: int,
        splat_mode: str,
        init_scale: float,
        scale_init_log_jitter: float,
        init_alpha_logit: float,
        init_xyz_noise: float,
        init_quat_noise: float,
        log_scale_min: float,
        log_scale_max: float,
    ) -> None:
        super().__init__()
        if splat_mode not in {"static", "per_frame"}:
            raise ValueError("model.splat_mode must be one of: static, per_frame")
        self.splat_mode = splat_mode
        self.T = int(num_frames)
        self.log_scale_min = float(log_scale_min)
        self.log_scale_max = float(log_scale_max)
        frame_count = self.T if splat_mode == "per_frame" else 1

        xyz = init_xyz.unsqueeze(0).repeat(frame_count, 1, 1)
        if float(init_xyz_noise) > 0.0:
            xyz = xyz + float(init_xyz_noise) * torch.randn_like(xyz)
        self.xyz = nn.Parameter(xyz)

        base_log_scale = math.log(float(init_scale))
        log_scales = torch.full((frame_count, init_xyz.shape[0], 3), base_log_scale, device=init_xyz.device)
        if float(scale_init_log_jitter) > 0.0:
            log_scales = log_scales + float(scale_init_log_jitter) * torch.randn_like(log_scales)
        self.log_scales = nn.Parameter(log_scales)

        quat = torch.zeros(frame_count, init_xyz.shape[0], 4, device=init_xyz.device)
        quat[..., 0] = 1.0
        if float(init_quat_noise) > 0.0:
            quat = quat + float(init_quat_noise) * torch.randn_like(quat)
        self.raw_quats = nn.Parameter(quat)
        self.opacity_logits = nn.Parameter(
            torch.full((frame_count, init_xyz.shape[0], 1), float(init_alpha_logit), device=init_xyz.device)
        )
        self.rgb_logits = nn.Parameter(logit(init_rgb).unsqueeze(0).repeat(frame_count, 1, 1))

    def parameter_index(self, t: int) -> int:
        return 0 if self.splat_mode == "static" else int(t)

    def frame(self, t: int) -> GaussianFrame:
        index = self.parameter_index(t)
        return GaussianFrame(
            xyz=self.xyz[index],
            scales=torch.exp(self.log_scales[index].clamp(self.log_scale_min, self.log_scale_max)).clamp_min(1e-6),
            quats=torch.nn.functional.normalize(self.raw_quats[index], p=2, dim=-1),
            opacities=torch.sigmoid(self.opacity_logits[index]),
            rgbs=torch.sigmoid(self.rgb_logits[index]),
        )

    def temporal_smoothness_loss(self) -> torch.Tensor:
        if self.splat_mode == "static" or self.xyz.shape[0] < 2:
            return self.xyz.new_zeros(())
        xyz = (self.xyz[1:] - self.xyz[:-1]).square().mean()
        scales = (self.log_scales[1:] - self.log_scales[:-1]).square().mean()
        opacity = (self.opacity_logits[1:] - self.opacity_logits[:-1]).square().mean()
        rgb = (self.rgb_logits[1:] - self.rgb_logits[:-1]).square().mean()
        return xyz + 0.1 * scales + 0.01 * opacity + 0.01 * rgb

    def scale_loss(self) -> torch.Tensor:
        return torch.exp(self.log_scales.clamp(self.log_scale_min, self.log_scale_max)).mean()

    @torch.no_grad()
    def metrics(self) -> dict[str, float]:
        scales = torch.exp(self.log_scales.clamp(self.log_scale_min, self.log_scale_max))
        return {
            "model_splat_scale_mean": float(scales.mean().detach().cpu()),
            "model_splat_scale_p95": float(torch.quantile(scales.detach().reshape(-1).cpu(), 0.95)),
            "model_splat_opacity_mean": float(torch.sigmoid(self.opacity_logits).mean().detach().cpu()),
            "model_splat_temporal_smooth": float(self.temporal_smoothness_loss().detach().cpu()),
        }


@dataclass
class SplatRenderConfig:
    height: int
    width: int
    renderer: str
    tile_size: int
    bound_scale: float
    alpha_threshold: float
    near_plane: float
    camera_projection: str


@torch.no_grad()
def render_splat_sequence(
    model: FreeDynamic3DGS,
    cameras: list[CameraSpec],
    cfg: SplatRenderConfig,
) -> dict[str, torch.Tensor]:
    rgbs = []
    alphas = []
    for t, camera in enumerate(cameras):
        image = render_gaussian_frame(
            model.frame(t),
            camera,
            height=cfg.height,
            width=cfg.width,
            mode=cfg.renderer,
            tile_size=cfg.tile_size,
            bound_scale=cfg.bound_scale,
            alpha_threshold=cfg.alpha_threshold,
            near_plane=cfg.near_plane,
            camera_projection=cfg.camera_projection,
        )
        rgbs.append(image.permute(1, 2, 0).contiguous())
        alphas.append(torch.ones(cfg.height, cfg.width, device=image.device, dtype=image.dtype))
    return {"rgb": torch.stack(rgbs, dim=0), "alpha": torch.stack(alphas, dim=0)}


def splat_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = resolved_config(config, sections=("data", "model", "camera", "render", "train", "losses", "logging"))
    apply_defaults(cfg["data"], DATA_DEFAULTS)
    apply_defaults(cfg["model"], MODEL_DEFAULTS)
    apply_defaults(cfg["camera"], CAMERA_DEFAULTS)
    apply_defaults(cfg["render"], RENDER_DEFAULTS)
    apply_defaults(cfg["train"], TRAIN_DEFAULTS)
    apply_defaults(cfg["losses"], LOSS_DEFAULTS)
    apply_defaults(cfg["logging"], LOGGING_DEFAULTS)
    cfg["model"]["splat_mode"] = str(cfg["model"]["splat_mode"])
    cfg["model"]["support_mode"] = str(cfg["model"].get("support_mode") or f"{cfg['model']['splat_mode']}_3dgs")
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a direct 3DGS splat baseline on the gauge-field data bundle.")
    parser.add_argument("config")
    parser.add_argument("--device", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-wandb", action="store_true", help="Accepted for parity; this script logs local artifacts only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_dynaworld_path(args.config)
    cfg = splat_config(load_config_file(config_path))
    if args.device is not None:
        cfg["train"]["device"] = args.device
    if args.steps is not None:
        cfg["train"]["steps"] = args.steps
    if args.output_dir is not None:
        cfg["logging"]["output_dir"] = args.output_dir
    if args.no_wandb:
        cfg["logging"]["log_to_wandb"] = False

    torch.manual_seed(int(cfg["train"]["seed"]))
    device = resolve_device(str(cfg["train"]["device"]))
    render_size = int(cfg["render"]["render_size"])
    bundle = load_gauge_video_bundle(
        data_cfg=cfg["data"],
        camera_cfg=cfg["camera"],
        render_size=render_size,
        device=device,
    )
    video = bundle.video
    T, H, W, _ = video.shape
    train_video = bundle.train_videos if bundle.train_videos is not None else video.unsqueeze(0)
    train_K = bundle.train_K if bundle.train_K is not None else bundle.K
    train_w2c = bundle.train_w2c if bundle.train_w2c is not None else bundle.w2c
    view_count = int(train_video.shape[0])
    init_xyz, init_rgb = initialize_material_points_from_first_frame(
        video=video,
        K=bundle.K,
        num_elements=int(cfg["model"]["num_splats"]),
        init_depth=float(cfg["model"]["init_depth"]),
    )
    model = FreeDynamic3DGS(
        init_xyz=init_xyz,
        init_rgb=init_rgb,
        num_frames=T,
        splat_mode=str(cfg["model"]["splat_mode"]),
        init_scale=float(cfg["model"]["init_scale"]),
        scale_init_log_jitter=float(cfg["model"]["scale_init_log_jitter"]),
        init_alpha_logit=float(cfg["model"]["init_alpha_logit"]),
        init_xyz_noise=float(cfg["model"]["init_xyz_noise"]),
        init_quat_noise=float(cfg["model"]["init_quat_noise"]),
        log_scale_min=float(cfg["model"]["log_scale_min"]),
        log_scale_max=float(cfg["model"]["log_scale_max"]),
    ).to(device)

    source_cameras = camera_list_from_K_w2c(bundle.K, bundle.w2c)
    heldout_cameras = None
    if bundle.heldout_K is not None and bundle.heldout_w2c is not None:
        heldout_cameras = camera_list_from_K_w2c(bundle.heldout_K, bundle.heldout_w2c)

    render_cfg = SplatRenderConfig(
        height=H,
        width=W,
        renderer=str(cfg["render"]["renderer"]),
        tile_size=int(cfg["render"]["tile_size"]),
        bound_scale=float(cfg["render"]["bound_scale"]),
        alpha_threshold=float(cfg["render"]["alpha_threshold"]),
        near_plane=float(cfg["render"]["near_plane"]),
        camera_projection=str(cfg["render"]["camera_projection"]),
    )
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["train"]["lr"]))

    logs = []
    steps = int(cfg["train"]["steps"])
    batch_size = int(cfg["train"]["frames_per_step"])
    log_every = int(cfg["logging"]["log_every"])
    print(
        "3DGS splat baseline "
        f"config={config_path} video={bundle.source_path} frames={T}/{cfg['data']['max_frames'] or 'all'} size={H}x{W} "
        f"splats={cfg['model']['num_splats']} mode={cfg['model']['splat_mode']} steps={steps} device={device}"
    )
    if bundle.train_camera_names:
        print(
            "Train cameras "
            f"anchor={bundle.metadata.get('anchor_camera') if bundle.metadata else None} "
            f"views={','.join(bundle.train_camera_names)}"
        )
    if bundle.heldout_video is not None:
        print(
            "Held-out camera eval "
            f"sample={bundle.metadata.get('sample_id') if bundle.metadata else None} "
            f"dataset={bundle.metadata.get('dataset') if bundle.metadata else None} "
            f"source={bundle.metadata.get('source_camera') if bundle.metadata else None} "
            f"target={bundle.heldout_camera_name or (bundle.metadata.get('target_camera') if bundle.metadata else None)} "
            f"pose_source={bundle.heldout_pose_source}"
        )

    train_start = time.perf_counter()
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        frames = torch.randint(0, T, (batch_size,), device=device)
        views = torch.randint(0, view_count, (batch_size,), device=device)
        total = video.new_zeros(())
        rgb_meter = 0.0
        for v, t in zip(views.tolist(), frames.tolist()):
            camera = camera_from_K_w2c(
                select_K_for_view_time(train_K, view=int(v), t=int(t), view_count=view_count),
                select_w2c_for_view_time(train_w2c, view=int(v), t=int(t)),
            )
            image = render_gaussian_frame(
                model.frame(int(t)),
                camera,
                height=H,
                width=W,
                mode=render_cfg.renderer,
                tile_size=render_cfg.tile_size,
                bound_scale=render_cfg.bound_scale,
                alpha_threshold=render_cfg.alpha_threshold,
                near_plane=render_cfg.near_plane,
                camera_projection=render_cfg.camera_projection,
            ).permute(1, 2, 0)
            rgb_l = robust_l1(image - train_video[int(v), int(t)])
            rgb_meter += float(rgb_l.detach())
            total = total + float(cfg["losses"]["rgb_weight"]) * rgb_l
        total = total / float(batch_size)
        total = total + float(cfg["losses"]["scale_weight"]) * model.scale_loss()
        total = total + float(cfg["losses"]["temporal_smooth_weight"]) * model.temporal_smoothness_loss()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % log_every == 0 or step == steps - 1:
            log = {
                "step": step,
                "loss": float(total.detach().cpu()),
                "rgb_l1": rgb_meter / float(batch_size),
                "scale": float(model.scale_loss().detach().cpu()),
                "temporal_smooth": float(model.temporal_smoothness_loss().detach().cpu()),
            }
            logs.append(log)
            print(log)
    train_loop_elapsed_s = time.perf_counter() - train_start

    rendered = render_splat_sequence(model, source_cameras, render_cfg)
    metrics = {
        **video_metrics(rendered["rgb"], video),
        **model.metrics(),
        "train_loop_elapsed_s": float(train_loop_elapsed_s),
    }
    if bundle.train_camera_names:
        metrics["train_camera_count"] = float(len(bundle.train_camera_names))
    heldout_rendered = None
    if bundle.heldout_video is not None and heldout_cameras is not None:
        heldout_rendered = render_splat_sequence(model, heldout_cameras, render_cfg)
        metrics.update(
            prefix_metrics(
                "heldout",
                video_metrics(heldout_rendered["rgb"], bundle.heldout_video),
            )
        )
        metrics["heldout_pose_is_calibrated"] = float(bundle.heldout_pose_source == "deepview_models_relative_pinhole")
    print({"final": metrics})

    output_dir = resolve_dynaworld_path(cfg["logging"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", serialize_config_value(cfg))
    write_json(output_dir / "logs.json", logs)
    write_json(output_dir / "metrics.json", metrics)
    write_checkpoint(
        output_dir / "checkpoint.pt",
        {
            "model": model.state_dict(),
            "config": serialize_config_value(cfg),
            "K": bundle.K.detach().cpu(),
            "w2c": bundle.w2c.detach().cpu(),
            "heldout_K": bundle.heldout_K.detach().cpu() if bundle.heldout_K is not None else None,
            "heldout_w2c": bundle.heldout_w2c.detach().cpu() if bundle.heldout_w2c is not None else None,
            "train_K": train_K.detach().cpu() if isinstance(train_K, torch.Tensor) else None,
            "train_w2c": train_w2c.detach().cpu() if isinstance(train_w2c, torch.Tensor) else None,
            "train_camera_names": bundle.train_camera_names,
            "heldout_camera_name": bundle.heldout_camera_name,
            "heldout_pose_source": bundle.heldout_pose_source,
            "metrics": metrics,
        },
    )
    save_preview_strip(output_dir / "preview.png", target=video, rendered=rendered["rgb"], alpha=rendered["alpha"])
    save_side_by_side_mp4(output_dir / "side_by_side.mp4", target=video, rendered=rendered["rgb"], fps=bundle.fps)
    if bundle.heldout_video is not None and heldout_rendered is not None:
        save_preview_strip(
            output_dir / "heldout_preview.png",
            target=bundle.heldout_video,
            rendered=heldout_rendered["rgb"],
            alpha=heldout_rendered["alpha"],
        )
        save_side_by_side_mp4(
            output_dir / "heldout_side_by_side.mp4",
            target=bundle.heldout_video,
            rendered=heldout_rendered["rgb"],
            fps=bundle.fps,
        )
    print(f"Wrote 3DGS splat baseline outputs to {output_dir}")


if __name__ == "__main__":
    main()
