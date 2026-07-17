from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from vjepa_benchmark_common import (
    ROOT,
    apply_video_benchmark_shape,
    effective_splat_count,
    quiet_training_logging,
    set_total_splat_count,
    sync_torch_device as sync,
    timed,
    timing_stats,
)

from camera import CameraSpec
from config_utils import load_config_file
from pipeline.render import gaussian_sequence_slice
from renderers.fast_mac import (
    FastMacRendererConfig,
    _rasterize_features_projected,
    _rasterize_rgb_projected,
    project_for_fast_mac_batch,
)
from rendering import _resolve_camera_projection_mode, camera_for_viewport
from runtime_types import GaussianSequence
from train_artifacts import write_jsonl
from train_logging import finish_wandb_run, set_default_wandb_mode
from trainer_registry import instantiate_trainer_for_config


DEFAULT_CONFIGS = [
    ROOT / "src/train_configs/local_mac_compare_free_splats_16f_implicit_camera_128_fast_mac_8192splats.jsonc",
    ROOT / "src/train_configs/local_mac_compare_unconditioned_tokens_16f_implicit_camera_128_fast_mac_8192splats.jsonc",
]


def prepare_config(
    base_cfg: dict[str, Any],
    *,
    render_size: int,
    clip_length: int,
    splat_count: int | None,
    amp_mode: str,
) -> dict[str, Any]:
    cfg = deepcopy(base_cfg)
    apply_video_benchmark_shape(cfg, render_size=render_size, clip_length=clip_length, steps=1)
    set_total_splat_count(cfg, splat_count)
    if amp_mode == "off":
        cfg["train"]["amp"] = False
    else:
        cfg["train"]["amp"] = True
        cfg["train"]["amp_dtype"] = amp_mode
    quiet_training_logging(cfg)
    cfg["logging"]["wandb_run_name"] = (
        f"fast-mac-phase-profile-{cfg['model']['variant']}-r{render_size}-f{clip_length}-"
        f"g{effective_splat_count(cfg)}-{amp_mode}"
    )
    return cfg


def _camera_scalar_vector(cameras: tuple[CameraSpec, ...], field_name: str, device: torch.device) -> torch.Tensor:
    values = []
    for camera in cameras:
        value = getattr(camera, field_name)
        if isinstance(value, torch.Tensor):
            value = value.detach()
        values.append(float(value))
    return torch.tensor(values, device=device, dtype=torch.float32)


def _viewport_cameras(cfg: dict[str, Any], cameras: tuple[CameraSpec, ...]) -> tuple[CameraSpec, ...]:
    return tuple(
        camera_for_viewport(
            camera,
            source_height=int(cfg["model"]["size"]),
            source_width=int(cfg["model"]["size"]),
            target_height=int(cfg["render"]["render_size"]),
            target_width=int(cfg["render"]["render_size"]),
        )
        for camera in cameras
    )


def _detach_sequence(sequence: GaussianSequence) -> GaussianSequence:
    return GaussianSequence(
        xyz=sequence.xyz.detach(),
        scales=sequence.scales.detach(),
        quats=sequence.quats.detach(),
        opacities=sequence.opacities.detach(),
        rgbs=sequence.rgbs.detach(),
        cameras=sequence.cameras,
        camera_state=sequence.camera_state,
        auxiliary=sequence.auxiliary,
    )


def project_fast_mac_inputs(
    cfg: dict[str, Any],
    sequence: GaussianSequence,
    cameras: tuple[CameraSpec, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = sequence.xyz.device
    render_cameras = _viewport_cameras(cfg, cameras)
    return project_for_fast_mac_batch(
        sequence.xyz.float(),
        sequence.scales.float(),
        sequence.quats.float(),
        sequence.opacities.float(),
        sequence.rgbs.float(),
        _camera_scalar_vector(render_cameras, "fx", device),
        _camera_scalar_vector(render_cameras, "fy", device),
        _camera_scalar_vector(render_cameras, "cx", device),
        _camera_scalar_vector(render_cameras, "cy", device),
        cameras=render_cameras,
        projection_mode=_resolve_camera_projection_mode(render_cameras, cfg["render"].get("camera_projection")),
        camera_to_world=torch.stack(
            [camera.camera_to_world.to(device=device, dtype=torch.float32) for camera in render_cameras],
            dim=0,
        ),
        near_plane=float(cfg["render"]["near_plane"]),
    )


def rasterize_projected(
    cfg: dict[str, Any],
    projected: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    feature_dim: int,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
    means2d, conics, colors, opacities, depths = projected
    render_size = int(cfg["render"]["render_size"])
    config = FastMacRendererConfig.from_mapping(
        cfg["render"]["fast_mac"],
        fallback_tile_size=int(cfg["render"]["tile_size"]),
        fallback_alpha_threshold=float(cfg["render"]["alpha_threshold"]),
    )
    if feature_dim == 3:
        return _rasterize_rgb_projected(
            means2d,
            conics,
            colors,
            opacities,
            depths,
            config,
            render_size,
            render_size,
        )

    return _rasterize_features_projected(
        means2d,
        conics,
        colors,
        opacities,
        depths,
        config,
        render_size,
        render_size,
        feature_dim,
    )


def raster_loss(rasterized: torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]) -> torch.Tensor:
    if isinstance(rasterized, tuple):
        image, alpha = rasterized
        loss = image.square().mean()
        if alpha is not None:
            loss = loss + alpha.square().mean()
        return loss
    return rasterized.square().mean()


def projected_requires_grad(
    projected: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    means2d, conics, colors, opacities, depths = projected
    return (
        means2d.detach().clone().requires_grad_(True),
        conics.detach().clone().requires_grad_(True),
        colors.detach().clone().requires_grad_(True),
        opacities.detach().clone().requires_grad_(True),
        depths.detach().clone(),
    )


def low_precision_acceptance(
    cfg: dict[str, Any],
    projected: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    feature_dim: int,
    device: torch.device,
) -> dict[str, dict[str, str]]:
    results: dict[str, dict[str, str]] = {}
    for name, dtype in (("float16", torch.float16), ("bfloat16", torch.bfloat16)):
        try:
            casted = tuple(t.detach().to(dtype=dtype).contiguous() for t in projected)
            sync(device)
            rasterize_projected(cfg, casted, feature_dim=feature_dim)
            sync(device)
        except Exception as exc:  # noqa: BLE001 - this is an acceptance probe.
            results[name] = {"status": "rejected", "error": f"{type(exc).__name__}: {exc}"}
        else:
            results[name] = {"status": "accepted", "error": ""}
    return results


def sample_decode(trainer) -> tuple[dict[str, float], Any, torch.Tensor, torch.Tensor, GaussianSequence]:
    timings: dict[str, float] = {}
    name, elapsed, sampled = timed("sample_clip", trainer.device, trainer.sample_clip)
    timings[name] = elapsed
    sequence_data, clip_frames, clip_times = sampled
    name, elapsed, model_input = timed(
        "model_input",
        trainer.device,
        lambda: trainer.model_input_for_clip(sequence_data, clip_frames, clip_times),
    )
    timings[name] = elapsed
    name, elapsed, decoded = timed(
        "forward_decode",
        trainer.device,
        lambda: trainer.forward_clip(model_input, clip_times),
    )
    timings[name] = elapsed
    return timings, sequence_data, clip_frames, clip_times, decoded


def profile_step(trainer) -> dict[str, float]:
    timings, _sequence_data, clip_frames, _clip_times, decoded = sample_decode(trainer)
    if decoded.cameras is None:
        raise ValueError("Implicit-camera decode must include cameras.")

    trainer.optimizer.zero_grad(set_to_none=True)
    frame_count = len(decoded.cameras)
    chunk_sequence = gaussian_sequence_slice(decoded, 0, frame_count)
    chunk_indices = torch.arange(0, frame_count, device=clip_frames.device)
    chunk_times = chunk_indices.to(dtype=torch.float32) / float(max(frame_count - 1, 1))
    target = trainer.make_target_view(
        view_id="fast_mac_phase_profile",
        frames=clip_frames[0, :frame_count],
        frame_indices=chunk_indices,
        frame_times=chunk_times,
        cameras=tuple(decoded.cameras),
        role="train",
    )
    name, elapsed, background = timed(
        "background_sample",
        trainer.device,
        lambda: trainer.rgb_objective.sample_background(
            phase="train",
            like=clip_frames,
            frame_count=frame_count,
        ),
    )
    timings[name] = elapsed

    detached_sequence = _detach_sequence(chunk_sequence)
    name, elapsed, projected = timed(
        "fastmac_project_forward",
        trainer.device,
        lambda: project_fast_mac_inputs(trainer.cfg, detached_sequence, tuple(decoded.cameras)),
    )
    timings[name] = elapsed
    feature_dim = int(detached_sequence.rgbs.shape[-1])
    name, elapsed, _rasterized = timed(
        "fastmac_raster_forward_projected",
        trainer.device,
        lambda: rasterize_projected(trainer.cfg, projected, feature_dim=feature_dim),
    )
    timings[name] = elapsed

    projected_for_grad = projected_requires_grad(projected)
    name, elapsed, rasterized_for_grad = timed(
        "fastmac_raster_forward_projected_grad",
        trainer.device,
        lambda: rasterize_projected(trainer.cfg, projected_for_grad, feature_dim=feature_dim),
    )
    timings[name] = elapsed
    raster_backward_loss = raster_loss(rasterized_for_grad)
    name, elapsed, _ = timed(
        "fastmac_raster_backward_projected",
        trainer.device,
        raster_backward_loss.backward,
    )
    timings[name] = elapsed

    name, elapsed, rendered = timed(
        "objective_render_forward",
        trainer.device,
        lambda: trainer.rgb_objective.render_view(
            chunk_sequence,
            target,
            phase="train",
            background=background,
        ),
    )
    timings[name] = elapsed
    name, elapsed, losses = timed(
        "recon_loss_compute",
        trainer.device,
        lambda: trainer.rgb_objective.reconstruction_loss_per_image(rendered),
    )
    timings[name] = elapsed
    loss = losses.mean()
    name, elapsed, _ = timed("full_backward_render_to_model", trainer.device, loss.backward)
    timings[name] = elapsed
    name, elapsed, _ = timed("optimizer_step", trainer.device, trainer.optimizer.step)
    timings[name] = elapsed
    timings["loss"] = float(loss.detach().cpu())
    return timings


def profile_low_precision_probe(trainer) -> dict[str, dict[str, str]]:
    _timings, _sequence_data, _clip_frames, _clip_times, decoded = sample_decode(trainer)
    if decoded.cameras is None:
        raise ValueError("Implicit-camera decode must include cameras.")
    detached = _detach_sequence(decoded)
    projected = project_fast_mac_inputs(trainer.cfg, detached, tuple(decoded.cameras))
    return low_precision_acceptance(
        trainer.cfg,
        projected,
        feature_dim=int(detached.rgbs.shape[-1]),
        device=trainer.device,
    )


def run_case(
    config_path: Path,
    *,
    render_size: int,
    clip_length: int,
    splat_count: int | None,
    amp_mode: str,
    warmup: int,
    steps: int,
) -> dict[str, Any]:
    cfg = prepare_config(
        load_config_file(config_path),
        render_size=render_size,
        clip_length=clip_length,
        splat_count=splat_count,
        amp_mode=amp_mode,
    )
    trainer = instantiate_trainer_for_config(cfg, config_path)
    rows: list[dict[str, float]] = []
    try:
        for _ in range(warmup):
            profile_step(trainer)
        low_precision = profile_low_precision_probe(trainer)
        for _ in range(steps):
            rows.append(profile_step(trainer))
    finally:
        finish_wandb_run()

    timing_keys = [key for key in rows[0] if key != "loss"] if rows else []
    result = {
        "config": str(config_path),
        "model_variant": str(trainer.model_cfg["variant"]),
        "amp_mode": amp_mode,
        "device": str(trainer.device),
        "renderer": str(trainer.renderer_mode),
        "render_size": int(render_size),
        "clip_length": int(clip_length),
        "gaussians": int(trainer.effective_gaussians),
        "steps": int(steps),
        "warmup": int(warmup),
        "timings": {key: timing_stats([row[key] for row in rows]) for key in timing_keys},
        "last_loss": rows[-1]["loss"] if rows else None,
        "low_precision_projected_input": low_precision,
    }
    return result


def print_result(result: dict[str, Any]) -> None:
    print(
        "CASE "
        f"variant={result['model_variant']} amp={result['amp_mode']} "
        f"size={result['render_size']} clip={result['clip_length']} splats={result['gaussians']}"
    )
    for key, stats in result["timings"].items():
        print(
            f"  {key}: mean={stats['mean']:.6f}s median={stats['median']:.6f}s "
            f"min={stats['min']:.6f}s max={stats['max']:.6f}s"
        )
    print(f"  low_precision_projected_input={result['low_precision_projected_input']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile fast-mac render phases on TokenGS/free-splat configs.")
    parser.add_argument("--config", type=Path, action="append", default=None)
    parser.add_argument("--render-size", type=int, default=128)
    parser.add_argument("--clip-length", type=int, default=16)
    parser.add_argument("--splat-count", type=int, default=8192)
    parser.add_argument("--amp-mode", choices=("off", "fp16", "bf16"), action="append", default=None)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    args = parser.parse_args()

    if args.steps < 1 or args.warmup < 0:
        raise SystemExit("--steps must be >= 1 and --warmup must be >= 0")

    set_default_wandb_mode("disabled", silent=None)
    config_paths = args.config or DEFAULT_CONFIGS
    amp_modes = args.amp_mode or ["off"]
    results = []
    for config_path in config_paths:
        for amp_mode in amp_modes:
            result = run_case(
                config_path,
                render_size=args.render_size,
                clip_length=args.clip_length,
                splat_count=args.splat_count,
                amp_mode=amp_mode,
                warmup=args.warmup,
                steps=args.steps,
            )
            print_result(result)
            results.append(result)

    if args.output_jsonl is not None:
        write_jsonl(args.output_jsonl, results)
        print(f"wrote {args.output_jsonl}")


if __name__ == "__main__":
    main()
