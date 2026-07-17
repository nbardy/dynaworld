from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

import benchmark_bootstrap
from benchmark_compare import grad_diff_stats, max_tensor_diff, seed_everything, tensor_diff_stats
from fixed_render_cases import prepare_heldout_fixed_render_case
from config_utils import load_config_file
from benchmark_gradients import module_parameter_grads, sequence_leaf_grads
from objective.types import RasterizedView
from train_artifacts import write_json
from train_logging import finish_wandb_run
from trainer_capabilities import trainer_uses_multicam_phase
from trainer_registry import instantiate_trainer_for_config
from fixed_render_graph import (
    PhaseTimer,
    clone_sequence_for_fixed_render,
    fast_mac_project_and_rasterize,
    prepare_fixed_render_case,
)


def _render_chunk(trainer, fixed_case, chunk_index: int, *, check_gradients: bool):
    fixed_chunk = fixed_case.chunks[chunk_index]
    timer = PhaseTimer(trainer.device)
    trainer.optimizer.zero_grad(set_to_none=True)
    if check_gradients:
        sequence = clone_sequence_for_fixed_render(fixed_chunk.sequence, freeze_colors=False)
        grad_context = torch.enable_grad()
    else:
        sequence = fixed_chunk.sequence
        grad_context = torch.no_grad()
    with grad_context:
        raster_graph = fast_mac_project_and_rasterize(
            trainer,
            sequence,
            tuple(fixed_chunk.target.cameras),
            timer,
        )
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
            background=fixed_case.background,
            retain_target=True,
        )
        per_image = trainer.rgb_objective.reconstruction_loss_per_image(rendered)
        loss = per_image.sum() / float(max(fixed_case.total_frames, 1))
        if check_gradients:
            loss.backward()
    return {
        "features": raster_graph.features.detach(),
        "alpha": None if raster_graph.alpha is None else raster_graph.alpha.detach(),
        "rgb": rendered.rgb.detach(),
        "loss_sum": float(loss.detach().item()),
        "sequence_grads": sequence_leaf_grads(sequence) if check_gradients else {},
        "colorize_grads": module_parameter_grads(trainer.colorize) if check_gradients else {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare fixed-render outputs for two fast_mac feature variants.")
    parser.add_argument("baseline_config", type=Path)
    parser.add_argument("candidate_config", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target", choices=("train", "heldout"), default="train")
    parser.add_argument("--check-gradients", action="store_true")
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    seed_everything(int(args.seed))
    baseline = instantiate_trainer_for_config(load_config_file(args.baseline_config), args.baseline_config)
    if args.target == "heldout":
        fixed_case = prepare_heldout_fixed_render_case(baseline)
    else:
        fixed_case = prepare_fixed_render_case(
            baseline,
            multicam=trainer_uses_multicam_phase(baseline),
        )

    seed_everything(int(args.seed))
    candidate = instantiate_trainer_for_config(load_config_file(args.candidate_config), args.candidate_config)
    if baseline.colorize is not None and candidate.colorize is not None:
        candidate.colorize.load_state_dict(baseline.colorize.state_dict())

    chunk_rows = []
    loss_baseline = 0.0
    loss_candidate = 0.0
    max_feature = 0.0
    max_alpha = 0.0
    max_rgb = 0.0
    max_sequence_grad = 0.0
    max_colorize_grad = 0.0
    for chunk_index in range(len(fixed_case.chunks)):
        base = _render_chunk(baseline, fixed_case, chunk_index, check_gradients=bool(args.check_gradients))
        cand = _render_chunk(candidate, fixed_case, chunk_index, check_gradients=bool(args.check_gradients))
        feature_diff = tensor_diff_stats(base["features"], cand["features"])
        alpha_diff = tensor_diff_stats(base["alpha"], cand["alpha"])
        rgb_diff = tensor_diff_stats(base["rgb"], cand["rgb"])
        sequence_grad_diff = (
            grad_diff_stats(base["sequence_grads"], cand["sequence_grads"]) if args.check_gradients else {}
        )
        colorize_grad_diff = (
            grad_diff_stats(base["colorize_grads"], cand["colorize_grads"]) if args.check_gradients else {}
        )
        loss_baseline += base["loss_sum"]
        loss_candidate += cand["loss_sum"]
        max_feature = max(max_feature, float(feature_diff.get("max_abs") or 0.0))
        max_alpha = max(max_alpha, float(alpha_diff.get("max_abs") or 0.0))
        max_rgb = max(max_rgb, float(rgb_diff.get("max_abs") or 0.0))
        max_sequence_grad = max(max_sequence_grad, max_tensor_diff(sequence_grad_diff))
        max_colorize_grad = max(max_colorize_grad, max_tensor_diff(colorize_grad_diff))
        chunk_rows.append(
            {
                "chunk_index": chunk_index,
                "feature_diff": feature_diff,
                "alpha_diff": alpha_diff,
                "rgb_diff": rgb_diff,
                "baseline_loss": base["loss_sum"],
                "candidate_loss": cand["loss_sum"],
                "loss_abs_diff": abs(base["loss_sum"] - cand["loss_sum"]),
                "sequence_grad_diff": sequence_grad_diff,
                "colorize_grad_diff": colorize_grad_diff,
            }
        )

    payload = {
        "baseline_config": str(args.baseline_config),
        "candidate_config": str(args.candidate_config),
        "seed": int(args.seed),
        "target": str(args.target),
        "check_gradients": bool(args.check_gradients),
        "chunk_count": len(fixed_case.chunks),
        "setup_phases_ms": fixed_case.setup_phases_ms,
        "baseline_loss": loss_baseline,
        "candidate_loss": loss_candidate,
        "loss_abs_diff": abs(loss_baseline - loss_candidate),
        "max_feature_abs_diff": max_feature,
        "max_alpha_abs_diff": max_alpha,
        "max_rgb_abs_diff": max_rgb,
        "max_sequence_grad_abs_diff": max_sequence_grad,
        "max_colorize_grad_abs_diff": max_colorize_grad,
        "chunks": chunk_rows,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.json_output is not None:
        write_json(args.json_output, payload)

    finish_wandb_run()


if __name__ == "__main__":
    main()
