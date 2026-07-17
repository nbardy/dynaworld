from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

import torch

from benchmark_bootstrap import ROOT

from config_utils import load_config_file
from benchmark_compare import seed_everything
from benchmark_memory import run_with_memory_sampling
from fixed_render_graph import (
    FixedRenderCase,
    PhaseTimer,
    RasterGraph,
    clone_sequence_for_fixed_render,
    fast_mac_project_and_rasterize,
    iter_target_chunks,
    multicam_sample_and_encode,
    prepare_fixed_render_case,
    singlecam_sample_and_encode,
)
from objective.loss import resize_target_for_render
from objective.types import BackgroundSample, RasterizedView
from objective.v12a_fused_l1 import fused_no_norm_l1_mean_loss
from pipeline.losses import build_bank_rate_loss, build_camera_loss
from runtime_types import GaussianSequence
from train_artifacts import write_json
from trainer_capabilities import trainer_uses_multicam_phase
from train_logging import finish_wandb_run, set_default_wandb_mode
from trainer_registry import instantiate_trainer_for_config

set_default_wandb_mode("disabled", silent=True)


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
FIXED_RENDER_PHASES = (
    "project",
    "raster_forward",
    "loss",
    "autograd_backward_total",
)


@dataclass(frozen=True)
class ChunkRecord:
    loss: torch.Tensor
    raster: RasterGraph


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


def _v12a_fused_l1_chunk_loss(
    trainer,
    raster_graph: RasterGraph,
    target,
    background: BackgroundSample,
    *,
    total_frames: int,
) -> torch.Tensor:
    if trainer.colorize is None:
        raise ValueError("--v12a-fused-l1 requires a feature colorizer.")
    if background.rgb is None:
        raise ValueError("--v12a-fused-l1 requires an explicit RGB background.")
    target_rgb = resize_target_for_render(target, render_size=int(raster_graph.features.shape[-1]))
    loss_mean = fused_no_norm_l1_mean_loss(
        features_nchw=raster_graph.features,
        alpha_nhw=raster_graph.alpha,
        target_rgb=target_rgb,
        background_rgb=background.rgb,
        colorizer=trainer.colorize,
    )
    return loss_mean * (float(raster_graph.features.shape[0]) / float(max(total_frames, 1)))


def build_step_graph(
    trainer,
    *,
    multicam: bool,
    timer: PhaseTimer,
    use_v12a_fused_l1: bool = False,
) -> tuple[list[ChunkRecord], torch.Tensor]:
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
                if use_v12a_fused_l1:
                    chunk_loss = _v12a_fused_l1_chunk_loss(
                        trainer,
                        raster_graph,
                        chunk_target,
                        background,
                        total_frames=total_frames,
                    )
                else:
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


def benchmark_fixed_render_case(
    trainer,
    fixed_case: FixedRenderCase,
    *,
    freeze_colors: bool,
    backward_mode: str = "batched",
    use_v12a_fused_l1: bool = False,
) -> dict[str, float]:
    if trainer.colorize is not None:
        trainer.colorize.train()
    trainer.optimizer.zero_grad(set_to_none=True)
    timer = PhaseTimer(trainer.device)
    losses: list[torch.Tensor] = []
    for fixed_chunk in fixed_case.chunks:
        chunk_background = fixed_chunk.background if fixed_chunk.background is not None else fixed_case.background
        chunk_sequence = clone_sequence_for_fixed_render(fixed_chunk.sequence, freeze_colors=freeze_colors)
        raster_graph = fast_mac_project_and_rasterize(
            trainer,
            chunk_sequence,
            tuple(fixed_chunk.target.cameras),
            timer,
        )
        with timer.measure("loss"):
            if use_v12a_fused_l1:
                chunk_loss = _v12a_fused_l1_chunk_loss(
                    trainer,
                    raster_graph,
                    fixed_chunk.target,
                    chunk_background,
                    total_frames=fixed_case.total_frames,
                )
            else:
                rasterized = RasterizedView(
                    view=fixed_chunk.target,
                    features=raster_graph.features,
                    alpha=raster_graph.alpha,
                    cameras=fixed_chunk.target.cameras,
                    view_dirs=trainer.view_dirs_for_features(raster_graph.features, tuple(fixed_chunk.target.cameras)),
                )
                rendered = trainer.rgb_objective.compose_rasterized(
                    rasterized,
                    phase="train",
                    background=chunk_background,
                    retain_target=True,
                )
                chunk_loss = (
                    trainer.rgb_objective.reconstruction_loss_per_image(rendered).sum()
                    / float(max(fixed_case.total_frames, 1))
                )
        if backward_mode == "chunked":
            with timer.measure("autograd_backward_total"):
                chunk_loss.backward()
            del chunk_loss, raster_graph, rasterized, rendered, chunk_sequence
        else:
            losses.append(chunk_loss)

    if backward_mode == "batched":
        with timer.measure("autograd_backward_total"):
            sum(losses).backward()
    elif backward_mode != "chunked":
        raise ValueError(f"Unsupported fixed render backward mode: {backward_mode!r}.")
    trainer.optimizer.zero_grad(set_to_none=True)
    return {phase: float(timer.elapsed_ms.get(phase, 0.0)) for phase in FIXED_RENDER_PHASES}


def benchmark_one_step(trainer, *, multicam: bool, use_v12a_fused_l1: bool = False) -> dict[str, float]:
    trainer.model.train()
    if trainer.colorize is not None:
        trainer.colorize.train()
    trainer.optimizer.zero_grad(set_to_none=True)
    timer = PhaseTimer(trainer.device)
    chunk_records, regularizer_loss = build_step_graph(
        trainer,
        multicam=multicam,
        timer=timer,
        use_v12a_fused_l1=use_v12a_fused_l1,
    )

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


def summarize_sampled_memory(samples: list[dict[str, float | int]]) -> dict[str, int] | None:
    sampled = [sample for sample in samples if int(sample.get("memory_sample_count", 0)) > 0]
    if not sampled:
        return None
    return {
        "sampled_peak_current_allocated_bytes": int(
            max(sample["sampled_peak_current_allocated_bytes"] for sample in sampled)
        ),
        "sampled_peak_driver_allocated_bytes": int(
            max(sample["sampled_peak_driver_allocated_bytes"] for sample in sampled)
        ),
        "memory_sample_count_total": int(sum(int(sample["memory_sample_count"]) for sample in sampled)),
        "max_end_current_allocated_bytes": int(max(sample.get("end_current_allocated_bytes", 0) for sample in sampled)),
        "max_end_driver_allocated_bytes": int(max(sample.get("end_driver_allocated_bytes", 0) for sample in sampled)),
    }


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
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for model init, sampling, and background.")
    parser.add_argument(
        "--backward-breakdown",
        action="store_true",
        help=(
            "Run detached VJP probes for loss/colorizer, raster, projection, model, and regularizer "
            "backward. These probes are a diagnostic breakdown, not one optimizer-step timing."
        ),
    )
    parser.add_argument(
        "--fixed-render-graph",
        action="store_true",
        help=(
            "Sample/decode one clip once, then repeatedly time project/raster/loss/backward on detached "
            "Gaussian leaves. This isolates renderer/loss timing from sample, encode, model, and optimizer jitter."
        ),
    )
    parser.add_argument(
        "--fixed-render-freeze-colors",
        action="store_true",
        help="With --fixed-render-graph, keep decoded splat features/colors detached to isolate no-color-grad paths.",
    )
    parser.add_argument(
        "--fixed-render-temporal-chunk-size",
        type=int,
        default=0,
        help=(
            "With --fixed-render-graph, split each render target into frame chunks of this size during the "
            "fixed-render benchmark. Default 0 keeps the existing full-target chunks."
        ),
    )
    parser.add_argument(
        "--fixed-render-backward-mode",
        choices=("batched", "chunked"),
        default="batched",
        help=(
            "With --fixed-render-graph, either accumulate all chunk losses and backprop once, or backprop each "
            "chunk immediately to probe render/loss microbatch memory pressure."
        ),
    )
    parser.add_argument(
        "--v12a-fused-l1",
        action="store_true",
        help=(
            "Use the opt-in v12a fused no-norm colorize+alpha-compose+L1 autograd path for the "
            "reconstruction loss. Requires colorize.pre_norm=false, hidden_dim=null, sigmoid, and "
            "view_condition=none."
        ),
    )
    parser.add_argument(
        "--memory-sample-interval-ms",
        type=float,
        default=0.0,
        help=(
            "When >0, sample MPS/CUDA allocation counters in a background thread while each measured "
            "iteration runs. Default off to preserve historical timing comparability."
        ),
    )
    parser.add_argument(
        "--memory-clear-cache",
        action="store_true",
        help="Clear Python and device caches before each measured iteration when memory sampling is enabled.",
    )
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        seed_everything(int(args.seed))
    if int(args.fixed_render_temporal_chunk_size) < 0:
        raise ValueError("--fixed-render-temporal-chunk-size must be >= 0.")
    if not args.fixed_render_graph and (
        int(args.fixed_render_temporal_chunk_size) > 0 or args.fixed_render_backward_mode != "batched"
    ):
        raise ValueError(
            "--fixed-render-temporal-chunk-size and --fixed-render-backward-mode require --fixed-render-graph."
        )

    config = load_config_file(args.config)
    trainer = instantiate_trainer_for_config(config, args.config)
    multicam = trainer_uses_multicam_phase(trainer)
    samples: list[dict[str, float]] = []
    try:
        temporal_chunk_size = int(args.fixed_render_temporal_chunk_size)
        fixed_case = (
            prepare_fixed_render_case(
                trainer,
                multicam=multicam,
                temporal_chunk_size=temporal_chunk_size if temporal_chunk_size > 0 else None,
            )
            if args.fixed_render_graph
            else None
        )
        phases = FIXED_RENDER_PHASES if fixed_case is not None else TRAIN_STEP_PHASES
        for _ in range(max(0, args.warmup)):
            if fixed_case is not None:
                benchmark_fixed_render_case(
                    trainer,
                    fixed_case,
                    freeze_colors=bool(args.fixed_render_freeze_colors),
                    backward_mode=str(args.fixed_render_backward_mode),
                    use_v12a_fused_l1=bool(args.v12a_fused_l1),
                )
            else:
                benchmark_one_step(trainer, multicam=multicam, use_v12a_fused_l1=bool(args.v12a_fused_l1))
        for _ in range(max(1, args.iters)):
            def run_measured_iteration():
                if fixed_case is not None:
                    return benchmark_fixed_render_case(
                        trainer,
                        fixed_case,
                        freeze_colors=bool(args.fixed_render_freeze_colors),
                        backward_mode=str(args.fixed_render_backward_mode),
                        use_v12a_fused_l1=bool(args.v12a_fused_l1),
                    )
                return benchmark_one_step(
                    trainer,
                    multicam=multicam,
                    use_v12a_fused_l1=bool(args.v12a_fused_l1),
                )

            if float(args.memory_sample_interval_ms) > 0.0:
                samples.append(
                    run_with_memory_sampling(
                        trainer.device,
                        interval_ms=float(args.memory_sample_interval_ms),
                        clear_cache=bool(args.memory_clear_cache),
                        fn=run_measured_iteration,
                    )
                )
                continue
            if fixed_case is not None:
                samples.append(
                    benchmark_fixed_render_case(
                        trainer,
                        fixed_case,
                        freeze_colors=bool(args.fixed_render_freeze_colors),
                        backward_mode=str(args.fixed_render_backward_mode),
                        use_v12a_fused_l1=bool(args.v12a_fused_l1),
                    )
                )
            else:
                samples.append(
                    benchmark_one_step(
                        trainer,
                        multicam=multicam,
                        use_v12a_fused_l1=bool(args.v12a_fused_l1),
                    )
                )
        summary = summarize(samples, phases)
        memory_summary = summarize_sampled_memory(samples)
        breakdown_samples: list[dict[str, float]] = []
        breakdown_summary = None
        if args.backward_breakdown and fixed_case is not None:
            print("Skipping --backward-breakdown because --fixed-render-graph already detaches the render graph.")
        elif args.backward_breakdown:
            for _ in range(max(1, args.iters)):
                breakdown_samples.append(benchmark_backward_breakdown(trainer, multicam=multicam))
            breakdown_summary = summarize(breakdown_samples, BACKWARD_BREAKDOWN_PHASES)
        payload = {
            "config": str(args.config),
            "renderer_mode": trainer.renderer_mode,
            "trainer": type(trainer).__name__,
            "warmup": int(args.warmup),
            "iters": int(args.iters),
            "fixed_render_graph": bool(args.fixed_render_graph),
            "fixed_render_freeze_colors": bool(args.fixed_render_freeze_colors),
            "fixed_render_temporal_chunk_size": (
                fixed_case.temporal_chunk_size if fixed_case is not None else None
            ),
            "fixed_render_backward_mode": str(args.fixed_render_backward_mode) if fixed_case is not None else None,
            "v12a_fused_l1": bool(args.v12a_fused_l1),
            "fixed_render_setup_phases_ms": fixed_case.setup_phases_ms if fixed_case is not None else None,
            "fixed_render_chunk_count": len(fixed_case.chunks) if fixed_case is not None else None,
            "fixed_render_note": (
                "Fixed-render rows reuse one sampled clip and detached Gaussian leaves; they intentionally exclude "
                "model backward, regularizers, optimizer, and sample/encode timing. Chunked backward mode "
                "backprops each fixed render chunk immediately as a benchmark-only microbatch probe."
                if fixed_case is not None
                else None
            ),
            "memory_sample_interval_ms": float(args.memory_sample_interval_ms),
            "memory_clear_cache": bool(args.memory_clear_cache),
            "sampled_memory_summary": memory_summary,
            "samples": samples,
            "summary": summary,
            "backward_breakdown_samples": breakdown_samples,
            "backward_breakdown_summary": breakdown_summary,
            "backward_breakdown_note": (
                "Breakdown rows are separate detached VJP probes and are not expected to sum exactly "
                "to autograd_backward_total."
            ),
        }
        print_table(
            summary,
            phases,
            title="Fixed render graph phases" if fixed_case is not None else "Training step phases",
        )
        if fixed_case is not None:
            print(
                f"Fixed render setup phases: {fixed_case.setup_phases_ms}; chunks={len(fixed_case.chunks)}; "
                f"temporal_chunk_size={fixed_case.temporal_chunk_size}; "
                f"backward_mode={args.fixed_render_backward_mode}"
            )
            print("Note: fixed-render totals exclude sample/encode/model backward/regularizer/optimizer.")
        if memory_summary is not None:
            print(
                "Sampled memory: "
                f"peak_current={memory_summary['sampled_peak_current_allocated_bytes']} bytes, "
                f"peak_driver={memory_summary['sampled_peak_driver_allocated_bytes']} bytes, "
                f"samples={memory_summary['memory_sample_count_total']}"
            )
        if breakdown_summary is not None:
            print()
            print_table(breakdown_summary, BACKWARD_BREAKDOWN_PHASES, title="Backward breakdown probes")
            print("Note: breakdown probes are separate VJPs and will not sum exactly to autograd_backward_total.")
        if args.json_output is not None:
            write_json(args.json_output, payload)
            print(f"wrote {args.json_output}")
    finally:
        finish_wandb_run()


if __name__ == "__main__":
    main()
