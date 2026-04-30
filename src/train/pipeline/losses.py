"""Pipeline-level loss-construction helpers.

These gradient-bearing helpers read learnable tensors out of `CameraState` and
`GaussianSequence.auxiliary` and return scalars that participate in backward.
Trainer state is passed explicitly:

* `loss_cfg` is the normalized `cfg["losses"]` dict. No silent defaults are
  applied at use sites; a missing required key should raise loudly.
* `recon_backward_strategy` is `"batched" | "microbatch" | "framewise"` from
  `cfg["train"]`.
* `temporal_microbatch_size` is the normalized integer from `cfg["train"]`.

Diagnostics and eval-payload helpers live in `pipeline.diagnostics`.
"""

from __future__ import annotations

from typing import Any, Literal

import torch

from runtime_types import CameraState, GaussianSequence

ReconBackwardStrategy = Literal["batched", "microbatch", "framewise"]


def compute_camera_losses(
    clip_times: torch.Tensor,
    camera_state: CameraState,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (motion, temporal, global) raw camera regularizer scalars.

    No weights applied — caller composes via `build_camera_loss`. This split
    lets logging code surface the unweighted value so the W&B trace is
    invariant to weight changes across runs.

    `motion`   — squared rotation_delta and unit-radius-normalized translation_delta.
    `temporal` — squared frame-to-frame difference of the [rot|trans] motion features
                 (zeros for single-frame clips).
    `global`   — squared global-residual head output.
    """
    camera_motion_loss = (
        torch.cat(
            [
                camera_state.rotation_delta,
                camera_state.translation_delta / camera_state.radius.clamp_min(1e-6),
            ],
            dim=-1,
        )
        .pow(2)
        .mean()
    )

    if clip_times.shape[1] > 1:
        motion_features = camera_state.motion_features()
        camera_temporal_loss = (motion_features[1:] - motion_features[:-1]).pow(2).mean()
    else:
        camera_temporal_loss = camera_motion_loss.new_tensor(0.0)

    camera_global_loss = camera_state.global_residuals.pow(2).mean()
    return camera_motion_loss, camera_temporal_loss, camera_global_loss


def build_camera_loss(
    clip_times: torch.Tensor,
    camera_state: CameraState,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Weighted sum of the three camera regularizers + the unweighted parts.

    Returns (camera_loss, motion, temporal, global) so the caller can log all
    four. The first is what flows into backward; the others are diagnostics.
    """
    camera_motion_loss, camera_temporal_loss, camera_global_loss = compute_camera_losses(
        clip_times,
        camera_state,
    )
    camera_loss = (
        loss_cfg["camera_motion_weight"] * camera_motion_loss
        + loss_cfg["camera_temporal_weight"] * camera_temporal_loss
        + loss_cfg["camera_global_weight"] * camera_global_loss
    )
    return camera_loss, camera_motion_loss, camera_temporal_loss, camera_global_loss


# Bank-rate auxiliary keys come from the residual_free_bank model variants.
# A decode that does not contain these keys is the typed "no bank" case; we
# return a zero loss and zero-valued terms rather than raising — the caller
# logs the terms unconditionally and the model variant flag stays implicit.
_BANK_RATE_REQUIRED_KEYS: frozenset[str] = frozenset(
    {
        "static_opacities",
        "dynamic_opacities",
        "dynamic_A_mu",
        "dynamic_A_rot",
        "dynamic_A_alpha",
    }
)


def build_bank_rate_loss(
    decoded: GaussianSequence,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Weighted bank-rate regularization for residual-free-bank model variants.

    Returns (rate_loss, terms) where `terms` always contains the five named
    fields (static_alpha, dynamic_alpha, dynamic_motion, dynamic_rotation,
    dynamic_alpha_time). Models without the bank produce all-zero terms and a
    zero loss — see `_BANK_RATE_REQUIRED_KEYS` comment above.
    """
    zero = decoded.xyz.new_tensor(0.0)
    terms: dict[str, torch.Tensor] = {
        "static_alpha": zero,
        "dynamic_alpha": zero,
        "dynamic_motion": zero,
        "dynamic_rotation": zero,
        "dynamic_alpha_time": zero,
    }
    auxiliary = decoded.auxiliary
    if not _BANK_RATE_REQUIRED_KEYS.issubset(auxiliary.keys()):
        return zero, terms

    terms = {
        "static_alpha": auxiliary["static_opacities"].mean(),
        "dynamic_alpha": auxiliary["dynamic_opacities"].mean(),
        "dynamic_motion": auxiliary["dynamic_A_mu"].abs().mean(),
        "dynamic_rotation": auxiliary["dynamic_A_rot"].abs().mean(),
        "dynamic_alpha_time": auxiliary["dynamic_A_alpha"].abs().mean(),
    }
    rate_loss = (
        float(loss_cfg["static_alpha_rate_weight"]) * terms["static_alpha"]
        + float(loss_cfg["dynamic_alpha_rate_weight"]) * terms["dynamic_alpha"]
        + float(loss_cfg["dynamic_motion_rate_weight"]) * terms["dynamic_motion"]
        + float(loss_cfg["dynamic_rotation_rate_weight"]) * terms["dynamic_rotation"]
        + float(loss_cfg["dynamic_alpha_time_rate_weight"]) * terms["dynamic_alpha_time"]
    )
    return rate_loss, terms


def temporal_recon_chunk_size(
    frame_count: int,
    *,
    recon_backward_strategy: ReconBackwardStrategy,
    temporal_microbatch_size: int,
) -> int:
    """How many frames to bundle per `backward()` in the recon loop.

    `batched`    -> one chunk = the whole clip (one backward call).
    `framewise`  -> one frame per chunk (memory-light, slow).
    `microbatch` -> capped at `temporal_microbatch_size`, but never larger
                    than `frame_count` so short clips don't pad with zeros.

    Caller is responsible for validating `recon_backward_strategy` against
    the literal set above; the dispatch here is exhaustive on that union.
    """
    match recon_backward_strategy:
        case "batched":
            return frame_count
        case "framewise":
            return 1
        case "microbatch":
            return min(temporal_microbatch_size, frame_count)
