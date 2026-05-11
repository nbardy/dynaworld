from __future__ import annotations

import sys
from pathlib import Path

import torch


FAST_MAC_V12A_DIR = (
    Path(__file__).resolve().parents[3]
    / "third_party"
    / "fast-mac-gsplat"
    / "variants"
    / "v12a_fused_colorize_l1_no_norm"
)


def _ensure_v12a_on_path() -> None:
    package_name = "torch_gsplat_bridge_v12a_fused_colorize_l1_no_norm"
    if not FAST_MAC_V12A_DIR.exists():
        raise RuntimeError(f"fast-mac v12a directory not found: {FAST_MAC_V12A_DIR}")
    existing_module = sys.modules.get(package_name)
    if existing_module is not None:
        origin_raw = getattr(existing_module, "__file__", None)
        if origin_raw is not None:
            origin = Path(origin_raw).resolve()
            if FAST_MAC_V12A_DIR.resolve() not in origin.parents:
                raise RuntimeError(
                    f"{package_name!r} is already imported from {origin}, not requested v12a directory."
                )
    if str(FAST_MAC_V12A_DIR) not in sys.path:
        sys.path.insert(0, str(FAST_MAC_V12A_DIR))


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
