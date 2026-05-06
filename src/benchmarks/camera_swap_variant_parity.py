from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
TRAIN_ROOT = ROOT / "src" / "train"
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

from camera_swap_sampling import CameraSwapPair  # noqa: E402
from config_utils import load_config_file  # noqa: E402
from fixed_render_variant_parity import _diff_stats  # noqa: E402
from train_multicam_precomputed_feature_implicit_dynamic import MulticamPrecomputedFeatureImplicitTrainer  # noqa: E402
from trainer_phase_benchmark import trainer_for_config  # noqa: E402


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _named_grads(trainer) -> dict[str, torch.Tensor | None]:
    modules = {
        "model": trainer.model,
        "colorize": trainer.colorize,
        "relpose_head": getattr(trainer, "relpose_head", None),
        "camera_rig": getattr(trainer, "camera_rig", None),
    }
    grads: dict[str, torch.Tensor | None] = {}
    for prefix, module in modules.items():
        if module is None:
            continue
        for name, param in module.named_parameters():
            grads[f"{prefix}.{name}"] = None if param.grad is None else param.grad.detach().clone()
    return grads


def _grad_diff_stats(
    base: dict[str, torch.Tensor | None],
    candidate: dict[str, torch.Tensor | None],
) -> dict[str, dict[str, Any]]:
    keys = sorted(set(base) | set(candidate))
    return {key: _diff_stats(base.get(key), candidate.get(key)) for key in keys}


def _top_diffs(diff: dict[str, dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    rows = [
        {
            "name": key,
            "max_abs": float(value.get("max_abs") or 0.0),
            "mean_abs": value.get("mean_abs"),
            "shape": value.get("shape"),
            "both_none": value.get("both_none", False),
        }
        for key, value in diff.items()
    ]
    rows.sort(key=lambda row: float(row["max_abs"]), reverse=True)
    return rows[: int(limit)]


def _max_diff_excluding(diff: dict[str, dict[str, Any]], excluded_prefixes: tuple[str, ...]) -> float:
    return max(
        (
            float(value.get("max_abs") or 0.0)
            for key, value in diff.items()
            if not any(key.startswith(prefix) for prefix in excluded_prefixes)
        ),
        default=0.0,
    )


def _pair_payload(pairs: tuple[CameraSwapPair, ...]) -> list[dict[str, Any]]:
    return [
        {
            "source_set": pair.source_set,
            "source_view": int(pair.source_view),
            "query_set": pair.query_set,
            "query_view": int(pair.query_view),
            "target_set": pair.target_set,
            "target_view": int(pair.target_view),
            "source_name": pair.source_name,
            "query_name": pair.query_name,
            "target_name": pair.target_name,
        }
        for pair in pairs
    ]


def _feature_variant(cfg: dict[str, Any]) -> str:
    return str(cfg["render"]["fast_mac"]["feature_variant"])


def _config_without_feature_variant(cfg: dict[str, Any]) -> dict[str, Any]:
    cloned = copy.deepcopy(cfg)
    cloned["render"]["fast_mac"]["feature_variant"] = "<feature_variant>"
    return cloned


def _set_feature_variant(trainer: MulticamPrecomputedFeatureImplicitTrainer, feature_variant: str) -> str:
    previous = str(trainer.cfg["render"]["fast_mac"]["feature_variant"])
    trainer.cfg["render"]["fast_mac"]["feature_variant"] = str(feature_variant)
    return previous


def _run_graph(
    trainer: MulticamPrecomputedFeatureImplicitTrainer,
    *,
    seed: int,
    clip_indices: torch.Tensor,
    pairs: tuple[CameraSwapPair, ...],
    feature_variant: str,
) -> dict[str, Any]:
    _seed_everything(seed)
    trainer.optimizer.zero_grad(set_to_none=True)
    previous_variant = _set_feature_variant(trainer, feature_variant)
    try:
        (
            recon_loss,
            bank_rate_loss,
            bank_rate_terms,
            _preview_render,
            _preview_features,
            _camera_state,
            _clip_frames,
            _sequence_data,
        ) = trainer.camera_swap_recon_loss(
            clip_indices=clip_indices,
            pairs=pairs,
            phase="train",
            keep_preview=False,
        )
        loss = recon_loss + bank_rate_loss
        loss.backward()
        grads = _named_grads(trainer)
    finally:
        _set_feature_variant(trainer, previous_variant)
        trainer.optimizer.zero_grad(set_to_none=True)
    return {
        "loss": float(loss.detach().cpu()),
        "recon_loss": float(recon_loss.detach().cpu()),
        "bank_rate_loss": float(bank_rate_loss.detach().cpu()),
        "bank_rate_terms": {key: float(value.detach().cpu()) for key, value in bank_rate_terms.items()},
        "grads": grads,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare camera-swap graph loss and grads across render variants.")
    parser.add_argument("baseline_config", type=Path)
    parser.add_argument("candidate_config", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-diff-limit", type=int, default=20)
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    baseline_cfg = load_config_file(args.baseline_config)
    candidate_cfg = load_config_file(args.candidate_config)
    if _config_without_feature_variant(baseline_cfg) != _config_without_feature_variant(candidate_cfg):
        raise ValueError(
            "camera_swap_variant_parity expects configs to differ only in render.fast_mac.feature_variant."
        )
    baseline_variant = _feature_variant(baseline_cfg)
    candidate_variant = _feature_variant(candidate_cfg)

    _seed_everything(int(args.seed))
    trainer = trainer_for_config(baseline_cfg)
    if not isinstance(trainer, MulticamPrecomputedFeatureImplicitTrainer):
        raise ValueError("camera_swap_variant_parity requires a multicam trainer config.")

    clip_length = int(trainer.model_cfg["train_frame_count"])
    clip_indices = torch.arange(0, clip_length, device=trainer.device)
    _seed_everything(int(args.seed))
    pairs = trainer.sample_camera_swap_pairs()

    baseline_out = _run_graph(
        trainer,
        seed=int(args.seed),
        clip_indices=clip_indices,
        pairs=pairs,
        feature_variant=baseline_variant,
    )
    candidate_out = _run_graph(
        trainer,
        seed=int(args.seed),
        clip_indices=clip_indices,
        pairs=pairs,
        feature_variant=candidate_variant,
    )
    grad_diff = _grad_diff_stats(baseline_out["grads"], candidate_out["grads"])
    max_grad = max((float(value.get("max_abs") or 0.0) for value in grad_diff.values()), default=0.0)
    max_grad_excluding_input_norms = _max_diff_excluding(
        grad_diff,
        excluded_prefixes=("model.video_encoder.input_norms.",),
    )
    payload = {
        "baseline_config": str(args.baseline_config),
        "candidate_config": str(args.candidate_config),
        "baseline_feature_variant": baseline_variant,
        "candidate_feature_variant": candidate_variant,
        "seed": int(args.seed),
        "clip_length": clip_length,
        "pairs": _pair_payload(pairs),
        "baseline_loss": baseline_out["loss"],
        "candidate_loss": candidate_out["loss"],
        "loss_abs_diff": abs(float(baseline_out["loss"]) - float(candidate_out["loss"])),
        "baseline_recon_loss": baseline_out["recon_loss"],
        "candidate_recon_loss": candidate_out["recon_loss"],
        "recon_loss_abs_diff": abs(float(baseline_out["recon_loss"]) - float(candidate_out["recon_loss"])),
        "baseline_bank_rate_loss": baseline_out["bank_rate_loss"],
        "candidate_bank_rate_loss": candidate_out["bank_rate_loss"],
        "bank_rate_loss_abs_diff": abs(
            float(baseline_out["bank_rate_loss"]) - float(candidate_out["bank_rate_loss"])
        ),
        "max_param_grad_abs_diff": max_grad,
        "max_param_grad_abs_diff_excluding_video_input_norms": max_grad_excluding_input_norms,
        "video_input_norm_caveat": (
            "On MPS this full camera-swap graph shows large LayerNorm input_norm parameter "
            "gradient diffs even for stable-vs-stable controls; use the excluding-input-norm "
            "field when checking renderer-specific drift."
        ),
        "top_param_grad_diffs": _top_diffs(grad_diff, limit=int(args.top_diff_limit)),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    import wandb

    wandb.finish()


if __name__ == "__main__":
    main()
