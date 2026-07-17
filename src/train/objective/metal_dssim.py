from __future__ import annotations

import torch

from external_paths import ensure_module_path, third_party_path

FAST_MAC_V12A_DIR = (
    third_party_path("fast-mac-gsplat")
    / "variants"
    / "v12a_fused_colorize_l1_no_norm"
)


def _ensure_v12a_on_path() -> None:
    package_name = "torch_gsplat_bridge_v12a_fused_colorize_l1_no_norm"
    ensure_module_path(
        package_name,
        FAST_MAC_V12A_DIR,
        missing_message=f"fast-mac v12a directory not found: {FAST_MAC_V12A_DIR}",
    )


def _metal_dssim_forward_grad(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    window_size: int,
    c1: float,
    c2: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    _ensure_v12a_on_path()
    from torch_gsplat_bridge_v12a_fused_colorize_l1_no_norm import dssim_forward_grad

    return dssim_forward_grad(
        prediction,
        target,
        window_size=window_size,
        c1=c1,
        c2=c2,
    )


class _MetalDSSIMMean(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        prediction: torch.Tensor,
        target: torch.Tensor,
        window_size: int,
        c1: float,
        c2: float,
    ) -> torch.Tensor:
        loss_per_image, grad_prediction = _metal_dssim_forward_grad(
            prediction.contiguous(),
            target.contiguous(),
            window_size=window_size,
            c1=c1,
            c2=c2,
        )
        ctx.save_for_backward(grad_prediction.contiguous())
        return loss_per_image.mean()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (grad_prediction,) = ctx.saved_tensors
        scale = grad_output.to(device=grad_prediction.device, dtype=grad_prediction.dtype)
        return grad_prediction * scale, None, None, None, None


def metal_dssim_mean(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    window_size: int,
    c1: float,
    c2: float,
) -> torch.Tensor:
    if prediction.device.type != "mps":
        raise ValueError("losses.dssim_backend='metal' requires MPS prediction tensors.")
    if target.device.type != "mps":
        raise ValueError("losses.dssim_backend='metal' requires MPS target tensors.")
    if prediction.dtype != torch.float32 or target.dtype != torch.float32:
        raise ValueError("losses.dssim_backend='metal' currently requires float32 prediction and target tensors.")
    if prediction.shape != target.shape:
        raise ValueError(f"prediction and target shape mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}")
    if prediction.dim() != 4:
        raise ValueError(f"Metal DSSIM expects [K,C,H,W], got {tuple(prediction.shape)}")
    return _MetalDSSIMMean.apply(
        prediction.contiguous(),
        target.contiguous(),
        int(window_size),
        float(c1),
        float(c2),
    )
