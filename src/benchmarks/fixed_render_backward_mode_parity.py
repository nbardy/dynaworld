from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

import benchmark_bootstrap
from benchmark_compare import grad_diff_stats, max_tensor_diff, seed_everything, tensor_diff_stats
from benchmark_gradients import module_parameter_grads, sequence_leaf_grads
from fixed_render_cases import prepare_heldout_fixed_render_case
from config_utils import load_config_file
from objective.types import BackgroundSample, RasterizedView
from pipeline.render import gaussian_sequence_slice
from runtime_types import GaussianSequence
from trainer_capabilities import trainer_uses_multicam_phase
from trainer_registry import instantiate_trainer_for_config
from fixed_render_graph import (
    FixedRenderCase,
    FixedRenderChunk,
    PhaseTimer,
    background_for_chunk,
    clone_sequence_for_fixed_render,
    fast_mac_project_and_rasterize,
    prepare_fixed_render_case,
)
from train_artifacts import write_json
from train_logging import finish_wandb_run


@dataclass(frozen=True)
class FullChunkSpec:
    key: str
    frame_indices: tuple[int, ...]
    sequence: GaussianSequence
    target: Any
    background: BackgroundSample


def _chunk_key(chunk: FixedRenderChunk) -> str:
    camera_name = getattr(chunk.target, "camera_name", None)
    return f"{chunk.target.view_id}|{camera_name or ''}"


def _full_specs(fixed_case: FixedRenderCase) -> dict[str, FullChunkSpec]:
    specs = {}
    for chunk in fixed_case.chunks:
        background = chunk.background if chunk.background is not None else fixed_case.background
        if background is None:
            raise ValueError("Fixed render chunk has no background sample.")
        key = _chunk_key(chunk)
        if key in specs:
            raise ValueError(f"Duplicate full chunk key {key!r}; cannot aggregate chunked gradients.")
        specs[key] = FullChunkSpec(
            key=key,
            frame_indices=tuple(int(value) for value in chunk.target.frame_indices.detach().cpu().tolist()),
            sequence=chunk.sequence,
            target=chunk.target,
            background=background,
        )
    return specs


def _split_fixed_case(trainer, fixed_case: FixedRenderCase, *, temporal_chunk_size: int) -> FixedRenderCase:
    chunks = []
    for full_chunk in fixed_case.chunks:
        background = full_chunk.background if full_chunk.background is not None else fixed_case.background
        if background is None:
            raise ValueError("Fixed render chunk has no background sample.")
        frame_count = int(full_chunk.target.frame_count)
        chunk_size = min(max(1, int(temporal_chunk_size)), frame_count)
        for chunk_start in range(0, frame_count, chunk_size):
            chunk_end = min(chunk_start + chunk_size, frame_count)
            chunk_sequence = gaussian_sequence_slice(full_chunk.sequence, chunk_start, chunk_end)
            chunk_target = trainer.make_target_view(
                view_id=full_chunk.target.view_id,
                frames=full_chunk.target.frames[chunk_start:chunk_end],
                frame_indices=full_chunk.target.frame_indices[chunk_start:chunk_end],
                frame_times=full_chunk.target.frame_times.reshape(-1)[chunk_start:chunk_end],
                cameras=tuple(full_chunk.target.cameras[chunk_start:chunk_end]),
                role=full_chunk.target.role,
                camera_role=full_chunk.target.camera_role,
                camera_owner=full_chunk.target.camera_owner,
                camera_name=full_chunk.target.camera_name,
                metrics_prefix=full_chunk.target.metrics_prefix,
                log_media=bool(getattr(full_chunk.target, "log_media", False)),
            )
            chunks.append(
                FixedRenderChunk(
                    sequence=chunk_sequence,
                    target=chunk_target,
                    background=background_for_chunk(background, chunk_start=chunk_start, chunk_end=chunk_end),
                )
            )
    return FixedRenderCase(
        chunks=tuple(chunks),
        background=fixed_case.background,
        total_frames=fixed_case.total_frames,
        setup_phases_ms=fixed_case.setup_phases_ms,
        temporal_chunk_size=int(temporal_chunk_size),
    )


def _render_loss(trainer, sequence: GaussianSequence, chunk: FixedRenderChunk, fixed_case: FixedRenderCase):
    timer = PhaseTimer(trainer.device)
    background = chunk.background if chunk.background is not None else fixed_case.background
    raster_graph = fast_mac_project_and_rasterize(trainer, sequence, tuple(chunk.target.cameras), timer)
    rasterized = RasterizedView(
        view=chunk.target,
        features=raster_graph.features,
        alpha=raster_graph.alpha,
        cameras=chunk.target.cameras,
        view_dirs=trainer.view_dirs_for_features(raster_graph.features, tuple(chunk.target.cameras)),
    )
    rendered = trainer.rgb_objective.compose_rasterized(
        rasterized,
        phase="train",
        background=background,
        retain_target=True,
    )
    return trainer.rgb_objective.reconstruction_loss_per_image(rendered).sum() / float(
        max(fixed_case.total_frames, 1)
    )


def _zero_like_full_grad(specs: dict[str, FullChunkSpec]) -> dict[str, dict[str, torch.Tensor]]:
    return {
        key: {name: torch.zeros_like(tensor) for name, tensor in _sequence_tensors(spec.sequence).items()}
        for key, spec in specs.items()
    }


def _sequence_tensors(sequence: GaussianSequence) -> dict[str, torch.Tensor]:
    return {
        "xyz": sequence.xyz,
        "scales": sequence.scales,
        "quats": sequence.quats,
        "opacities": sequence.opacities,
        "rgbs": sequence.rgbs,
    }


def _accumulate_chunk_grads(
    out: dict[str, dict[str, torch.Tensor]],
    specs: dict[str, FullChunkSpec],
    chunk: FixedRenderChunk,
    grads: dict[str, torch.Tensor],
) -> None:
    key = _chunk_key(chunk)
    spec = specs[key]
    positions_by_frame = {frame_index: pos for pos, frame_index in enumerate(spec.frame_indices)}
    positions = [positions_by_frame[int(value)] for value in chunk.target.frame_indices.detach().cpu().tolist()]
    index = torch.tensor(positions, device=next(iter(out[key].values())).device)
    for name, grad in grads.items():
        if int(grad.shape[0]) != len(positions):
            raise ValueError(f"Gradient {name} first dim {grad.shape[0]} does not match chunk length {len(positions)}.")
        out[key][name].index_add_(0, index, grad)


def _run_backward_mode(
    trainer,
    fixed_case: FixedRenderCase,
    *,
    specs: dict[str, FullChunkSpec],
    mode: str,
) -> dict[str, Any]:
    trainer.optimizer.zero_grad(set_to_none=True)
    if trainer.colorize is not None:
        trainer.colorize.zero_grad(set_to_none=True)
        trainer.colorize.train()
    total_loss = 0.0
    aggregated_grads = _zero_like_full_grad(specs)
    losses = []
    sequences: list[tuple[FixedRenderChunk, GaussianSequence]] = []
    for chunk in fixed_case.chunks:
        sequence = clone_sequence_for_fixed_render(chunk.sequence, freeze_colors=False)
        loss = _render_loss(trainer, sequence, chunk, fixed_case)
        total_loss += float(loss.detach().cpu())
        if mode == "chunked":
            loss.backward()
            _accumulate_chunk_grads(aggregated_grads, specs, chunk, sequence_leaf_grads(sequence, missing="zero", clone=True))
        elif mode == "batched":
            losses.append(loss)
            sequences.append((chunk, sequence))
        else:
            raise ValueError(f"Unsupported mode {mode!r}.")
    if mode == "batched":
        sum(losses).backward()
        for chunk, sequence in sequences:
            _accumulate_chunk_grads(aggregated_grads, specs, chunk, sequence_leaf_grads(sequence, missing="zero", clone=True))
    colorize_grads = module_parameter_grads(trainer.colorize)
    trainer.optimizer.zero_grad(set_to_none=True)
    if trainer.colorize is not None:
        trainer.colorize.zero_grad(set_to_none=True)
    return {
        "loss": total_loss,
        "sequence_grads": aggregated_grads,
        "colorize_grads": colorize_grads,
    }


def _nested_grad_diff(
    base: dict[str, dict[str, torch.Tensor]],
    candidate: dict[str, dict[str, torch.Tensor]],
) -> dict[str, dict[str, dict[str, Any]]]:
    rows = {}
    for key in sorted(set(base) | set(candidate)):
        rows[key] = {}
        for name in sorted(set(base.get(key, {})) | set(candidate.get(key, {}))):
            rows[key][name] = tensor_diff_stats(base.get(key, {}).get(name), candidate.get(key, {}).get(name))
    return rows


def _flat_max_nested(diff: dict[str, dict[str, dict[str, Any]]]) -> float:
    return max((float(row.get("max_abs") or 0.0) for by_name in diff.values() for row in by_name.values()), default=0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare batched and chunked fixed-render backward gradients.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target", choices=("train", "heldout"), default="train")
    parser.add_argument("--temporal-chunk-size", type=int, default=8)
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    if int(args.temporal_chunk_size) < 1:
        raise ValueError("--temporal-chunk-size must be >= 1.")

    seed_everything(int(args.seed))
    trainer = instantiate_trainer_for_config(load_config_file(args.config), args.config)
    multicam = trainer_uses_multicam_phase(trainer)
    if args.target == "heldout":
        full_case = prepare_heldout_fixed_render_case(trainer)
    else:
        full_case = prepare_fixed_render_case(trainer, multicam=multicam)
    chunked_case = _split_fixed_case(trainer, full_case, temporal_chunk_size=int(args.temporal_chunk_size))
    specs = _full_specs(full_case)

    seed_everything(int(args.seed))
    batched = _run_backward_mode(trainer, full_case, specs=specs, mode="batched")
    seed_everything(int(args.seed))
    chunked = _run_backward_mode(trainer, chunked_case, specs=specs, mode="chunked")

    sequence_grad_diff = _nested_grad_diff(batched["sequence_grads"], chunked["sequence_grads"])
    colorize_grad_diff = grad_diff_stats(batched["colorize_grads"], chunked["colorize_grads"])
    payload = {
        "config": str(args.config),
        "seed": int(args.seed),
        "target": str(args.target),
        "temporal_chunk_size": int(args.temporal_chunk_size),
        "full_chunk_count": len(full_case.chunks),
        "chunked_chunk_count": len(chunked_case.chunks),
        "batched_loss": batched["loss"],
        "chunked_loss": chunked["loss"],
        "loss_abs_diff": abs(float(batched["loss"]) - float(chunked["loss"])),
        "max_sequence_grad_abs_diff": _flat_max_nested(sequence_grad_diff),
        "max_colorize_grad_abs_diff": max_tensor_diff(colorize_grad_diff),
        "sequence_grad_diff": sequence_grad_diff,
        "colorize_grad_diff": colorize_grad_diff,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.json_output is not None:
        write_json(args.json_output, payload)

    finish_wandb_run()


if __name__ == "__main__":
    main()
