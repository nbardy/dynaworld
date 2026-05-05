from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")

ROOT = Path(__file__).resolve().parents[2]
TRAIN_ROOT = ROOT / "src" / "train"
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

from config_utils import load_config_file  # noqa: E402
from objective.types import RasterizedView  # noqa: E402
from pipeline.losses import build_bank_rate_loss, build_camera_loss  # noqa: E402
from pipeline.render import _viewport_cameras, gaussian_sequence_slice  # noqa: E402
from renderers.fast_mac import (  # noqa: E402
    FastMacRendererConfig,
    _rasterize_features_projected,
    _rasterize_rgb_projected,
    project_for_fast_mac_batch,
)
from rendering import _camera_scalar_vector, _resolve_camera_projection_mode  # noqa: E402
from runtime_types import GaussianSequence  # noqa: E402
from train_multicam_precomputed_feature_implicit_dynamic import (  # noqa: E402
    MulticamPrecomputedFeatureImplicitTrainer,
)
from train_precomputed_feature_implicit_dynamic import PrecomputedFeatureImplicitTrainer  # noqa: E402
from train_video_token_implicit_dynamic import trainer_class_for_config  # noqa: E402


TRAIN_STEP_PHASES = (
    "sample",
    "encode",
    "project",
    "raster_forward",
    "loss",
    "autograd_backward_total",
    "optimizer",
)
BACKWARD_BREAKDOWN_PHASES = (
    "loss_colorize_backward_probe",
    "raster_backward_probe",
    "project_backward_probe",
    "model_backward_probe",
    "regularizer_backward_probe",
)


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


class PhaseTimer:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.elapsed_ms: dict[str, float] = defaultdict(float)

    @contextmanager
    def measure(self, phase: str):
        sync_device(self.device)
        start = time.perf_counter()
        try:
            yield
        finally:
            sync_device(self.device)
            self.elapsed_ms[phase] += (time.perf_counter() - start) * 1000.0


@dataclass(frozen=True)
class RasterGraph:
    features: torch.Tensor
    alpha: torch.Tensor | None
    projected: tuple[torch.Tensor, ...]
    projection_inputs: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class ChunkRecord:
    loss: torch.Tensor
    raster: RasterGraph


def trainer_for_config(config: dict[str, Any]):
    if "multicam_manifest" in config.get("data", {}) or "train_views_per_step" in config.get("train", {}):
        return MulticamPrecomputedFeatureImplicitTrainer(config)
    backend = str(config.get("model", {}).get("video_encoder_backend", "")).lower()
    if "features" in config or backend in {"precomputed", "precomputed_ltx"}:
        return PrecomputedFeatureImplicitTrainer(config)
    return trainer_class_for_config(config)(config)


def fast_mac_project_and_rasterize(
    trainer,
    sequence: GaussianSequence,
    cameras: tuple[Any, ...],
    timer: PhaseTimer,
) -> RasterGraph:
    if trainer.renderer_mode != "fast_mac":
        raise ValueError(
            f"trainer_phase_benchmark currently splits project/raster only for renderer='fast_mac'; "
            f"resolved renderer_mode={trainer.renderer_mode!r}."
        )
    if sequence.frame_count != len(cameras):
        raise ValueError(f"Expected {sequence.frame_count} cameras, got {len(cameras)}.")

    cfg = trainer.cfg
    height = int(cfg["render"]["render_size"])
    width = int(cfg["render"]["render_size"])
    render_cameras = _viewport_cameras(cameras, input_size=int(cfg["model"]["size"]), render_size=height)
    device = sequence.xyz.device
    projection_mode = _resolve_camera_projection_mode(render_cameras, cfg["render"]["camera_projection"])
    fx = _camera_scalar_vector(render_cameras, "fx", device)
    fy = _camera_scalar_vector(render_cameras, "fy", device)
    cx = _camera_scalar_vector(render_cameras, "cx", device)
    cy = _camera_scalar_vector(render_cameras, "cy", device)
    camera_to_world = torch.stack(
        [camera.camera_to_world.to(device=device, dtype=torch.float32) for camera in render_cameras],
        dim=0,
    )

    xyz = sequence.xyz.float()
    scales = sequence.scales.float()
    quats = sequence.quats.float()
    opacities = sequence.opacities.float()
    colors_in = sequence.rgbs.float()
    with timer.measure("project"):
        means2d, conics, colors, projected_opacities, depths = project_for_fast_mac_batch(
            xyz,
            scales,
            quats,
            opacities,
            colors_in,
            fx,
            fy,
            cx,
            cy,
            cameras=render_cameras,
            projection_mode=projection_mode,
            camera_to_world=camera_to_world,
            near_plane=cfg["render"]["near_plane"],
        )

    fast_mac_config = FastMacRendererConfig.from_mapping(
        cfg["render"]["fast_mac"],
        fallback_tile_size=cfg["render"]["tile_size"],
        fallback_alpha_threshold=cfg["render"]["alpha_threshold"],
    )
    feature_dim = int(colors.shape[-1])
    with timer.measure("raster_forward"):
        if feature_dim == 3:
            image_bhwc = _rasterize_rgb_projected(
                means2d,
                conics,
                colors,
                projected_opacities,
                depths,
                fast_mac_config,
                height,
                width,
            )
            features = image_bhwc.clamp(0.0, 1.0).permute(0, 3, 1, 2).contiguous()
            alpha = None
        else:
            rasterize_out = _rasterize_features_projected(
                means2d,
                conics,
                colors,
                projected_opacities,
                depths,
                fast_mac_config,
                height,
                width,
                feature_dim,
            )
            image_bhwf, alpha = rasterize_out
            features = image_bhwf.permute(0, 3, 1, 2).contiguous()
    return RasterGraph(
        features=features,
        alpha=alpha,
        projected=(means2d, conics, colors, projected_opacities, depths),
        projection_inputs=(xyz, scales, quats, opacities, colors_in, fx, fy, cx, cy, camera_to_world),
    )


def singlecam_sample_and_encode(trainer, timer: PhaseTimer):
    with timer.measure("sample"):
        sampled = trainer.sample_clip()
    if len(sampled) == 4:
        sequence_data, clip_frames, clip_times, clip_cameras = sampled
        with timer.measure("encode"):
            decoded = trainer.forward_known_clip(clip_frames, clip_times, clip_cameras)
        frame_count = int(clip_frames.shape[1])
        targets = [
            trainer.make_target_view(
                view_id="benchmark_train_clip",
                frames=clip_frames[0],
                frame_indices=torch.arange(frame_count, device=trainer.device),
                frame_times=clip_times.reshape(-1),
                cameras=clip_cameras,
                role="train",
                camera_owner="external_rig",
            )
        ]
        return sequence_data, clip_frames, clip_times, decoded, targets

    sequence_data, clip_frames, clip_times = sampled
    with timer.measure("encode"):
        model_input = trainer.model_input_for_clip(sequence_data, clip_frames, clip_times)
        decoded = trainer.forward_clip(model_input, clip_times)
    if decoded.cameras is None:
        raise ValueError("Implicit-camera decode did not produce cameras.")
    frame_count = int(clip_frames.shape[1])
    targets = [
        trainer.make_target_view(
            view_id="benchmark_train_clip",
            frames=clip_frames[0],
            frame_indices=torch.arange(frame_count, device=trainer.device),
            frame_times=clip_times.reshape(-1),
            cameras=tuple(decoded.cameras),
            role="train",
            camera_owner="model",
        )
    ]
    return sequence_data, clip_frames, clip_times, decoded, targets


def multicam_sample_and_encode(trainer, timer: PhaseTimer):
    with timer.measure("sample"):
        sequence_data, clip_indices, clip_frames, clip_times, views = trainer.sample_multicam_clip()
    with timer.measure("encode"):
        decoded = trainer._decode_clip(sequence_data, clip_frames, clip_times)
    targets = []
    for view in views:
        view_i = int(view)
        targets.append(
            trainer.make_target_view(
                view_id=f"benchmark_train_view_{view_i}",
                frames=trainer.multicam_bundle.train_frames[view_i, clip_indices],
                frame_indices=clip_indices,
                frame_times=trainer.frame_times_for_indices(clip_indices),
                cameras=trainer.camera_rig.cameras_for_view(view_i, clip_indices),
                role="train",
                camera_owner="external_rig",
                camera_name=trainer.multicam_bundle.train_camera_names[view_i],
            )
        )
    return sequence_data, clip_frames, clip_times, decoded, targets


def build_regularizer_loss(trainer, decoded: GaussianSequence, clip_times: torch.Tensor, *, multicam: bool):
    bank_rate_loss, _bank_rate_terms = build_bank_rate_loss(decoded, trainer.loss_cfg)
    if multicam:
        return bank_rate_loss + trainer.rig_regularization_loss()
    if decoded.camera_state is None:
        return bank_rate_loss
    camera_loss, _motion, _temporal, _global = build_camera_loss(
        clip_times,
        decoded.camera_state,
        trainer.loss_cfg,
    )
    return camera_loss + bank_rate_loss


def iter_target_chunks(trainer, decoded: GaussianSequence, target, *, use_microbatch: bool):
    frame_count = target.frame_count
    chunk_size = trainer.temporal_recon_chunk_size(frame_count) if use_microbatch else frame_count
    for chunk_start in range(0, frame_count, chunk_size):
        chunk_end = min(chunk_start + chunk_size, frame_count)
        chunk_sequence = gaussian_sequence_slice(decoded, chunk_start, chunk_end)
        chunk_target = trainer.make_target_view(
            view_id=target.view_id,
            frames=target.frames[chunk_start:chunk_end],
            frame_indices=target.frame_indices[chunk_start:chunk_end],
            frame_times=target.frame_times.reshape(-1)[chunk_start:chunk_end],
            cameras=tuple(target.cameras[chunk_start:chunk_end]),
            role=target.role,
            camera_owner=target.camera_owner,
            camera_name=target.camera_name,
            metrics_prefix=target.metrics_prefix,
        )
        yield chunk_start, chunk_end, chunk_sequence, chunk_target


def build_step_graph(trainer, *, multicam: bool, timer: PhaseTimer) -> tuple[list[ChunkRecord], torch.Tensor]:
    if multicam:
        _sequence_data, clip_frames, clip_times, decoded, targets = multicam_sample_and_encode(trainer, timer)
    else:
        _sequence_data, clip_frames, clip_times, decoded, targets = singlecam_sample_and_encode(trainer, timer)

    use_microbatch = not multicam
    total_frames = sum(target.frame_count for target in targets)
    with timer.measure("loss"):
        regularizer_loss = build_regularizer_loss(trainer, decoded, clip_times, multicam=multicam)
        background = trainer.rgb_objective.sample_background(
            phase="train",
            like=targets[0].frames,
            frame_count=targets[0].frame_count,
        )

    chunk_records = []
    for target in targets:
        for _chunk_start, _chunk_end, chunk_sequence, chunk_target in iter_target_chunks(
            trainer,
            decoded,
            target,
            use_microbatch=use_microbatch,
        ):
            raster_graph = fast_mac_project_and_rasterize(
                trainer,
                chunk_sequence,
                tuple(chunk_target.cameras),
                timer,
            )
            with timer.measure("loss"):
                rasterized = RasterizedView(
                    view=chunk_target,
                    features=raster_graph.features,
                    alpha=raster_graph.alpha,
                    cameras=chunk_target.cameras,
                    view_dirs=trainer.view_dirs_for_features(raster_graph.features, tuple(chunk_target.cameras)),
                )
                rendered = trainer.rgb_objective.compose_rasterized(
                    rasterized,
                    phase="train",
                    background=background,
                    retain_target=True,
                )
                chunk_loss = trainer.rgb_objective.reconstruction_loss_per_image(rendered).sum() / float(
                    max(total_frames, 1)
                )
            chunk_records.append(ChunkRecord(loss=chunk_loss, raster=raster_graph))
    return chunk_records, regularizer_loss


def benchmark_one_step(trainer, *, multicam: bool) -> dict[str, float]:
    trainer.model.train()
    if trainer.colorize is not None:
        trainer.colorize.train()
    trainer.optimizer.zero_grad(set_to_none=True)
    timer = PhaseTimer(trainer.device)
    chunk_records, regularizer_loss = build_step_graph(trainer, multicam=multicam, timer=timer)

    if multicam:
        with timer.measure("autograd_backward_total"):
            (sum((record.loss for record in chunk_records), regularizer_loss)).backward()
    else:
        for index, record in enumerate(chunk_records):
            is_last = index == len(chunk_records) - 1
            backward_loss = record.loss + (regularizer_loss if is_last else 0.0)
            with timer.measure("autograd_backward_total"):
                backward_loss.backward(retain_graph=not is_last)

    with timer.measure("optimizer"):
        trainer.optimizer.step()

    return {phase: float(timer.elapsed_ms.get(phase, 0.0)) for phase in TRAIN_STEP_PHASES}


def _requires_grad_items(tensors: tuple[torch.Tensor | None, ...]) -> tuple[torch.Tensor, ...]:
    return tuple(tensor for tensor in tensors if tensor is not None and tensor.requires_grad)


def _zero_missing_grads(
    tensors: tuple[torch.Tensor, ...],
    grads: tuple[torch.Tensor | None, ...],
) -> tuple[torch.Tensor, ...]:
    return tuple(torch.zeros_like(tensor) if grad is None else grad for tensor, grad in zip(tensors, grads))


def _backward_targets(
    tensors: tuple[torch.Tensor, ...],
    grads: tuple[torch.Tensor | None, ...],
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    pairs = [(tensor, grad) for tensor, grad in zip(tensors, grads) if grad is not None and tensor.requires_grad]
    return tuple(tensor for tensor, _grad in pairs), tuple(grad for _tensor, grad in pairs)


def benchmark_backward_breakdown(trainer, *, multicam: bool) -> dict[str, float]:
    trainer.model.train()
    if trainer.colorize is not None:
        trainer.colorize.train()
    trainer.optimizer.zero_grad(set_to_none=True)
    timer = PhaseTimer(trainer.device)
    chunk_records, regularizer_loss = build_step_graph(trainer, multicam=multicam, timer=timer)
    colorizer_params = (
        tuple(param for param in trainer.colorize.parameters() if param.requires_grad)
        if trainer.colorize is not None
        else ()
    )

    for record in chunk_records:
        raster_outputs = _requires_grad_items((record.raster.features, record.raster.alpha))
        if not raster_outputs:
            continue

        loss_grad_targets = raster_outputs + colorizer_params
        with timer.measure("loss_colorize_backward_probe"):
            loss_grads = torch.autograd.grad(
                record.loss,
                loss_grad_targets,
                retain_graph=True,
                allow_unused=True,
            )
        raster_output_grads = _zero_missing_grads(raster_outputs, loss_grads[: len(raster_outputs)])

        projected = _requires_grad_items(record.raster.projected)
        with timer.measure("raster_backward_probe"):
            projected_grads_raw = torch.autograd.grad(
                raster_outputs,
                projected,
                grad_outputs=raster_output_grads,
                retain_graph=True,
                allow_unused=True,
            )
        projected_grads = _zero_missing_grads(projected, projected_grads_raw)

        projection_inputs = _requires_grad_items(record.raster.projection_inputs)
        with timer.measure("project_backward_probe"):
            projection_input_grads_raw = torch.autograd.grad(
                projected,
                projection_inputs,
                grad_outputs=projected_grads,
                retain_graph=True,
                allow_unused=True,
            )

        model_targets, model_grads = _backward_targets(projection_inputs, projection_input_grads_raw)
        if model_targets:
            with timer.measure("model_backward_probe"):
                torch.autograd.backward(model_targets, grad_tensors=model_grads, retain_graph=True)

    if regularizer_loss.requires_grad:
        with timer.measure("regularizer_backward_probe"):
            regularizer_loss.backward(retain_graph=True)
    trainer.optimizer.zero_grad(set_to_none=True)
    return {phase: float(timer.elapsed_ms.get(phase, 0.0)) for phase in BACKWARD_BREAKDOWN_PHASES}


def summarize(samples: list[dict[str, float]], phases: tuple[str, ...]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for phase in phases:
        values = [sample[phase] for sample in samples]
        out[phase] = {
            "mean_ms": float(statistics.mean(values)),
            "median_ms": float(statistics.median(values)),
            "min_ms": float(min(values)),
            "max_ms": float(max(values)),
        }
    total_values = [sum(sample[phase] for phase in phases) for sample in samples]
    out["total"] = {
        "mean_ms": float(statistics.mean(total_values)),
        "median_ms": float(statistics.median(total_values)),
        "min_ms": float(min(total_values)),
        "max_ms": float(max(total_values)),
    }
    return out


def print_table(summary: dict[str, dict[str, float]], phases: tuple[str, ...], *, title: str | None = None) -> None:
    total = summary["total"]["mean_ms"]
    if title is not None:
        print(title)
    print("| phase | mean_ms | median_ms | pct_total |")
    print("|---|---:|---:|---:|")
    for phase in (*phases, "total"):
        row = summary[phase]
        pct = 100.0 * row["mean_ms"] / total if total > 0 else 0.0
        print(f"| {phase} | {row['mean_ms']:.3f} | {row['median_ms']:.3f} | {pct:.1f}% |")


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-phase fast_mac trainer bottleneck benchmark.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument(
        "--backward-breakdown",
        action="store_true",
        help=(
            "Run detached VJP probes for loss/colorizer, raster, projection, model, and regularizer "
            "backward. These probes are a diagnostic breakdown, not one optimizer-step timing."
        ),
    )
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    config = load_config_file(args.config)
    trainer = trainer_for_config(config)
    multicam = isinstance(trainer, MulticamPrecomputedFeatureImplicitTrainer)
    samples: list[dict[str, float]] = []
    try:
        for _ in range(max(0, args.warmup)):
            benchmark_one_step(trainer, multicam=multicam)
        for _ in range(max(1, args.iters)):
            samples.append(benchmark_one_step(trainer, multicam=multicam))
        summary = summarize(samples, TRAIN_STEP_PHASES)
        breakdown_samples: list[dict[str, float]] = []
        breakdown_summary = None
        if args.backward_breakdown:
            for _ in range(max(1, args.iters)):
                breakdown_samples.append(benchmark_backward_breakdown(trainer, multicam=multicam))
            breakdown_summary = summarize(breakdown_samples, BACKWARD_BREAKDOWN_PHASES)
        payload = {
            "config": str(args.config),
            "renderer_mode": trainer.renderer_mode,
            "trainer": type(trainer).__name__,
            "warmup": int(args.warmup),
            "iters": int(args.iters),
            "samples": samples,
            "summary": summary,
            "backward_breakdown_samples": breakdown_samples,
            "backward_breakdown_summary": breakdown_summary,
            "backward_breakdown_note": (
                "Breakdown rows are separate detached VJP probes and are not expected to sum exactly "
                "to autograd_backward_total."
            ),
        }
        print_table(summary, TRAIN_STEP_PHASES, title="Training step phases")
        if breakdown_summary is not None:
            print()
            print_table(breakdown_summary, BACKWARD_BREAKDOWN_PHASES, title="Backward breakdown probes")
            print("Note: breakdown probes are separate VJPs and will not sum exactly to autograd_backward_total.")
        if args.json_output is not None:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            print(f"wrote {args.json_output}")
    finally:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
