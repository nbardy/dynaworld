"""Pipeline-level diagnostic and eval-payload helpers.

These functions produce W&B-style scalar dicts from validation tensors. None of
them carry gradient. They are pure: dispatch is by tensor shape, and any
"missing data" path is a typed absence (empty dict or `None` buffers) rather
than a silent fallback to a fabricated value.

The split between this file and `pipeline.losses` is by gradient flow:

* `pipeline.diagnostics` handles eval/validation scalars on the `torch.no_grad`
  side, with no learnable parameters touched.
* `pipeline.losses` handles loss-construction helpers that participate in
  backward.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from losses import reconstruction_loss_per_image, ssim_per_image
from runtime_types import CameraState, GaussianSequence

# Decoded-temporal diagnostics walk the same set of GaussianSequence fields
# every call. Centralizing the tuple here keeps init / fill / payload in
# lockstep — adding a field is a one-line edit.
DECODED_TEMPORAL_FIELDS: tuple[str, ...] = ("xyz", "scales", "opacities", "rgbs")


def eval_metric_payload(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss_cfg: dict[str, Any],
) -> dict[str, float]:
    """Per-image L1/MSE/SSIM/PSNR for a rendered eval clip vs. its GT.

    `loss_cfg` is the trainer's `cfg["losses"]` dict; we read the SSIM window
    constants directly so this stays in sync with the training-side spec.
    """
    prediction = prediction.float()
    target = target.float()
    delta = prediction - target
    l1 = delta.abs().flatten(1).mean()
    mse = delta.square().flatten(1).mean()
    ssim = ssim_per_image(
        prediction,
        target,
        window_size=loss_cfg["ssim_window_size"],
        c1=float(loss_cfg["ssim_c1"]),
        c2=float(loss_cfg["ssim_c2"]),
    ).mean()
    dssim = (1.0 - ssim) * 0.5
    recon_loss = reconstruction_loss_per_image(prediction, target, loss_cfg).mean()
    psnr = -10.0 * math.log10(max(float(mse.item()), 1.0e-12))
    return {
        "Eval/Loss": float(recon_loss.item()),
        "Eval/L1": float(l1.item()),
        "Eval/MSE": float(mse.item()),
        "Eval/SSIM": float(ssim.item()),
        "Eval/DSSIM": float(dssim.item()),
        "Eval/PSNR": psnr,
    }


def temporal_similarity_payload(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss_cfg: dict[str, Any],
) -> dict[str, float]:
    """Frame-to-frame and frame-to-first temporal similarity scalars.

    Returns an empty dict for clips with <2 frames — that is the typed absence
    of "temporal" content, not a silent zero. Callers compose the dict with
    `|` so an empty dict is a no-op.
    """
    if prediction.shape[0] < 2:
        return {}

    prediction = prediction.float()
    target = target.float()
    pred_adj_l1 = (prediction[1:] - prediction[:-1]).abs().flatten(1).mean().mean()
    gt_adj_l1 = (target[1:] - target[:-1]).abs().flatten(1).mean().mean()
    pred_to_first_l1 = (prediction[1:] - prediction[:1]).abs().flatten(1).mean().mean()
    gt_to_first_l1 = (target[1:] - target[:1]).abs().flatten(1).mean().mean()
    pred_adj_ssim = ssim_per_image(
        prediction[1:],
        prediction[:-1],
        window_size=loss_cfg["ssim_window_size"],
        c1=float(loss_cfg["ssim_c1"]),
        c2=float(loss_cfg["ssim_c2"]),
    ).mean()
    gt_adj_ssim = ssim_per_image(
        target[1:],
        target[:-1],
        window_size=loss_cfg["ssim_window_size"],
        c1=float(loss_cfg["ssim_c1"]),
        c2=float(loss_cfg["ssim_c2"]),
    ).mean()
    return {
        "Eval/TemporalPredAdjacentL1": float(pred_adj_l1.item()),
        "Eval/TemporalGTAdjacentL1": float(gt_adj_l1.item()),
        "Eval/TemporalAdjacentL1Ratio": float((pred_adj_l1 / gt_adj_l1.clamp_min(1.0e-8)).item()),
        "Eval/TemporalPredToFirstL1": float(pred_to_first_l1.item()),
        "Eval/TemporalGTToFirstL1": float(gt_to_first_l1.item()),
        "Eval/TemporalToFirstL1Ratio": float((pred_to_first_l1 / gt_to_first_l1.clamp_min(1.0e-8)).item()),
        "Eval/TemporalPredAdjacentSSIM": float(pred_adj_ssim.item()),
        "Eval/TemporalGTAdjacentSSIM": float(gt_adj_ssim.item()),
    }


def init_decoded_frame_buffers(num_frames: int) -> dict[str, list[torch.Tensor | None]]:
    """Allocate per-frame slots for streaming decoded-temporal diagnostics.

    The eval loop streams chunks of frames at a time; this buffer collects the
    first-seen value per frame so the temporal stats see exactly one tensor
    per frame even if multiple windows overlap.
    """
    return {field_name: [None] * num_frames for field_name in DECODED_TEMPORAL_FIELDS}


def fill_decoded_frame_buffers(
    buffers: dict[str, list[torch.Tensor | None]],
    decoded: GaussianSequence,
    clip_indices: torch.Tensor,
) -> None:
    """Stream a decoded chunk into the per-frame buffer (CPU, detached).

    Idempotent: a frame slot is filled exactly once, even if the same frame
    index appears in multiple chunks. First-write-wins matches the existing
    monolith semantics — do NOT overwrite without a config-controlled policy.
    """
    frame_indices = clip_indices.detach().cpu().tolist()
    for local_index, frame_index in enumerate(frame_indices):
        if buffers["xyz"][frame_index] is not None:
            continue
        for field_name in DECODED_TEMPORAL_FIELDS:
            field_value = getattr(decoded, field_name)
            buffers[field_name][frame_index] = field_value[local_index].detach().cpu()


def decoded_temporal_payload(buffers: dict[str, list[torch.Tensor | None]]) -> dict[str, float]:
    """Adjacency / first-frame deltas across decoded gaussian fields.

    Returns an empty dict if any frame slot is unfilled or the clip is too
    short — typed absence, no silent zero. Callers must ensure all slots are
    filled (the trainer drives this via `fill_decoded_frame_buffers` over the
    full eval clip) before reading.
    """
    if not buffers or any(item is None for item in buffers["xyz"]):
        return {}
    xyz = torch.stack([item for item in buffers["xyz"] if item is not None], dim=0).float()
    scales = torch.stack([item for item in buffers["scales"] if item is not None], dim=0).float()
    opacities = torch.stack([item for item in buffers["opacities"] if item is not None], dim=0).float()
    rgbs = torch.stack([item for item in buffers["rgbs"] if item is not None], dim=0).float()
    if xyz.shape[0] < 2:
        return {}

    xyz_adjacent_l2 = torch.linalg.norm(xyz[1:] - xyz[:-1], dim=-1).mean()
    xyz_to_first_l2 = torch.linalg.norm(xyz[1:] - xyz[:1], dim=-1).mean()
    scale_adjacent_l1 = (scales[1:] - scales[:-1]).abs().mean()
    opacity_adjacent_l1 = (opacities[1:] - opacities[:-1]).abs().mean()
    opacity_to_first_l1 = (opacities[1:] - opacities[:1]).abs().mean()
    rgb_adjacent_l1 = (rgbs[1:] - rgbs[:-1]).abs().mean()
    rgb_to_first_l1 = (rgbs[1:] - rgbs[:1]).abs().mean()
    return {
        "Eval/DecodedXYZAdjacentL2": float(xyz_adjacent_l2.item()),
        "Eval/DecodedXYZToFirstL2": float(xyz_to_first_l2.item()),
        "Eval/DecodedScaleAdjacentL1": float(scale_adjacent_l1.item()),
        "Eval/DecodedOpacityAdjacentL1": float(opacity_adjacent_l1.item()),
        "Eval/DecodedOpacityToFirstL1": float(opacity_to_first_l1.item()),
        "Eval/DecodedRGBAdjacentL1": float(rgb_adjacent_l1.item()),
        "Eval/DecodedRGBToFirstL1": float(rgb_to_first_l1.item()),
    }


def camera_temporal_payload(camera_state: CameraState) -> dict[str, float]:
    """Frame-to-frame and frame-to-first deltas of predicted camera offsets.

    Returns an empty dict for single-frame clips — typed absence again. Only
    valid when `camera_state` is the implicit-camera model output; callers are
    responsible for not invoking this on known-camera paths.
    """
    if camera_state.rotation_delta.shape[0] < 2:
        return {}
    rotation_adjacent = torch.linalg.norm(camera_state.rotation_delta[1:] - camera_state.rotation_delta[:-1], dim=-1)
    translation_adjacent = torch.linalg.norm(
        camera_state.translation_delta[1:] - camera_state.translation_delta[:-1],
        dim=-1,
    )
    rotation_to_first = torch.linalg.norm(camera_state.rotation_delta[1:] - camera_state.rotation_delta[:1], dim=-1)
    translation_to_first = torch.linalg.norm(
        camera_state.translation_delta[1:] - camera_state.translation_delta[:1],
        dim=-1,
    )
    return {
        "Camera/EvalAdjacentRotationDeltaDegrees": float(torch.rad2deg(rotation_adjacent).mean().item()),
        "Camera/EvalAdjacentTranslationDelta": float(translation_adjacent.mean().item()),
        "Camera/EvalToFirstRotationDeltaDegrees": float(torch.rad2deg(rotation_to_first).mean().item()),
        "Camera/EvalToFirstTranslationDelta": float(translation_to_first.mean().item()),
    }
