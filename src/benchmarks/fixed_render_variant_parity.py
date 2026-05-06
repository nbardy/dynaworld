from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch

from config_utils import load_config_file
from objective.types import RasterizedView
from trainer_phase_benchmark import (
    FixedRenderCase,
    FixedRenderChunk,
    PhaseTimer,
    fast_mac_project_and_rasterize,
    prepare_fixed_render_case,
    trainer_for_config,
)
from runtime_types import GaussianSequence


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _diff_stats(a: torch.Tensor | None, b: torch.Tensor | None) -> dict[str, Any]:
    if a is None or b is None:
        return {
            "both_none": a is None and b is None,
            "max_abs": None,
            "mean_abs": None,
            "shape": None,
        }
    if tuple(a.shape) != tuple(b.shape):
        return {
            "both_none": False,
            "shape_mismatch": [list(a.shape), list(b.shape)],
            "max_abs": None,
            "mean_abs": None,
            "shape": None,
        }
    delta = (a.detach() - b.detach()).abs()
    return {
        "both_none": False,
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
        "shape": list(a.shape),
    }


def _detach_sequence(sequence: GaussianSequence) -> GaussianSequence:
    return GaussianSequence(
        xyz=sequence.xyz.detach(),
        scales=sequence.scales.detach(),
        quats=sequence.quats.detach(),
        opacities=sequence.opacities.detach(),
        rgbs=sequence.rgbs.detach(),
        cameras=sequence.cameras,
        camera_state=None,
        auxiliary=sequence.auxiliary,
    )


def _prepare_heldout_fixed_render_case(trainer) -> FixedRenderCase:
    if trainer.__class__.__name__ != "MulticamPrecomputedFeatureImplicitTrainer":
        raise ValueError("--target heldout is only supported for multicam trainers.")
    if trainer.multicam_bundle.heldout_frames is None:
        raise ValueError("--target heldout requested, but the config has no heldout frames.")
    setup_timer = PhaseTimer(trainer.device)
    with setup_timer.measure("sample"):
        sequence_data, clip_indices, clip_frames, clip_times, _views = trainer.sample_multicam_clip()
    with setup_timer.measure("encode"):
        decoded = trainer._decode_clip(sequence_data, clip_frames, clip_times)
    background = trainer.rgb_objective.sample_background(
        phase="train",
        like=trainer.multicam_bundle.heldout_frames[0, clip_indices],
        frame_count=len(clip_indices),
    )
    chunks = []
    heldout_count = int(trainer.multicam_bundle.heldout_frames.shape[0])
    for view in range(heldout_count):
        camera_names = trainer.multicam_bundle.heldout_camera_names or []
        camera_name = camera_names[view] if view < len(camera_names) else f"heldout_{view}"
        target = trainer.make_target_view(
            view_id=f"heldout_view_{view}",
            frames=trainer.multicam_bundle.heldout_frames[int(view), clip_indices],
            frame_indices=clip_indices,
            frame_times=trainer.frame_times_for_indices(clip_indices),
            cameras=trainer.camera_rig.heldout_cameras_for(view, clip_indices),
            role="heldout",
            camera_role="heldout",
            camera_owner="external_rig",
            camera_name=camera_name,
            metrics_prefix=f"Heldout{view}_{camera_name}",
        )
        chunks.append(FixedRenderChunk(sequence=_detach_sequence(decoded), target=target))
    return FixedRenderCase(
        chunks=tuple(chunks),
        background=background.detach() if torch.is_tensor(background) else background,
        total_frames=heldout_count * int(len(clip_indices)),
        setup_phases_ms={phase: float(setup_timer.elapsed_ms.get(phase, 0.0)) for phase in ("sample", "encode")},
    )


def _clone_sequence_for_grad(sequence: GaussianSequence) -> GaussianSequence:
    def leaf(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().clone().requires_grad_(True)

    return GaussianSequence(
        xyz=leaf(sequence.xyz),
        scales=leaf(sequence.scales),
        quats=leaf(sequence.quats),
        opacities=leaf(sequence.opacities),
        rgbs=leaf(sequence.rgbs),
        cameras=sequence.cameras,
        camera_state=None,
        auxiliary=sequence.auxiliary,
    )


def _sequence_grads(sequence: GaussianSequence) -> dict[str, torch.Tensor | None]:
    return {
        "xyz": sequence.xyz.grad,
        "scales": sequence.scales.grad,
        "quats": sequence.quats.grad,
        "opacities": sequence.opacities.grad,
        "rgbs": sequence.rgbs.grad,
    }


def _colorize_grads(trainer) -> dict[str, torch.Tensor | None]:
    if trainer.colorize is None:
        return {}
    return {
        name: None if param.grad is None else param.grad.detach().clone()
        for name, param in trainer.colorize.named_parameters()
    }


def _grad_diff_stats(
    base: dict[str, torch.Tensor | None],
    candidate: dict[str, torch.Tensor | None],
) -> dict[str, dict[str, Any]]:
    keys = sorted(set(base) | set(candidate))
    return {key: _diff_stats(base.get(key), candidate.get(key)) for key in keys}


def _max_diff(diff_by_key: dict[str, dict[str, Any]]) -> float:
    return max((float(value.get("max_abs") or 0.0) for value in diff_by_key.values()), default=0.0)


def _render_chunk(trainer, fixed_case, chunk_index: int, *, check_gradients: bool):
    fixed_chunk = fixed_case.chunks[chunk_index]
    timer = PhaseTimer(trainer.device)
    trainer.optimizer.zero_grad(set_to_none=True)
    if check_gradients:
        sequence = _clone_sequence_for_grad(fixed_chunk.sequence)
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
        "sequence_grads": _sequence_grads(sequence) if check_gradients else {},
        "colorize_grads": _colorize_grads(trainer) if check_gradients else {},
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

    _seed_everything(int(args.seed))
    baseline = trainer_for_config(load_config_file(args.baseline_config))
    if args.target == "heldout":
        fixed_case = _prepare_heldout_fixed_render_case(baseline)
    else:
        fixed_case = prepare_fixed_render_case(
            baseline,
            multicam=baseline.__class__.__name__ == "MulticamPrecomputedFeatureImplicitTrainer",
        )

    _seed_everything(int(args.seed))
    candidate = trainer_for_config(load_config_file(args.candidate_config))
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
        feature_diff = _diff_stats(base["features"], cand["features"])
        alpha_diff = _diff_stats(base["alpha"], cand["alpha"])
        rgb_diff = _diff_stats(base["rgb"], cand["rgb"])
        sequence_grad_diff = (
            _grad_diff_stats(base["sequence_grads"], cand["sequence_grads"]) if args.check_gradients else {}
        )
        colorize_grad_diff = (
            _grad_diff_stats(base["colorize_grads"], cand["colorize_grads"]) if args.check_gradients else {}
        )
        loss_baseline += base["loss_sum"]
        loss_candidate += cand["loss_sum"]
        max_feature = max(max_feature, float(feature_diff.get("max_abs") or 0.0))
        max_alpha = max(max_alpha, float(alpha_diff.get("max_abs") or 0.0))
        max_rgb = max(max_rgb, float(rgb_diff.get("max_abs") or 0.0))
        max_sequence_grad = max(max_sequence_grad, _max_diff(sequence_grad_diff))
        max_colorize_grad = max(max_colorize_grad, _max_diff(colorize_grad_diff))
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
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    import wandb

    wandb.finish()


if __name__ == "__main__":
    main()
