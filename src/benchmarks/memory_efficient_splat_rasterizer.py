from __future__ import annotations

"""
Torch-only memory-efficient differentiable Gaussian rasterizer.

Key ideas:
- No dense [G, H, W] materialization.
- Tight opacity-aware per-splat bounding boxes.
- Exact front-to-back alpha compositing.
- Custom backward that recomputes local alpha patches and uses a reverse scan.
- Saves only final transmittance + per-splat metadata, trading compute for memory.

This is designed as a practical baseline for small/full-frame training on commodity
hardware (including MPS), not as a replacement for a fused CUDA renderer.

Parameterization:
Each 2D Gaussian uses a conic / precision matrix
    Q = [[a, b], [b, c]]
with per-pixel exponent
    power = -0.5 * (a*dx^2 + 2*b*dx*dy + c*dy^2)
    alpha = opacity * exp(power)

Inputs are a single view/frame. Batch over frames outside for now.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


def _stable_argsort_depth(depths: Tensor, front_to_back: bool = True) -> Tensor:
    """Stable argsort when available; otherwise falls back to regular argsort."""
    descending = not front_to_back
    try:
        return torch.argsort(depths, dim=0, descending=descending, stable=True)
    except TypeError:
        return torch.argsort(depths, dim=0, descending=descending)


@dataclass
class RasterizeConfig:
    alpha_threshold: float = 1.0 / 255.0
    max_alpha: float = 0.99
    eps: float = 1e-8
    front_to_back: bool = True
    pixel_center_offset: float = 0.5
    use_fp16_patches: bool = False
    exact_ellipse_mask: bool = True


class _GaussianRasterizeFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means2d: Tensor,  # [G, 2]
        conics: Tensor,  # [G, 3] = [a, b, c]
        colors: Tensor,  # [G, C]
        opacities: Tensor,  # [G]
        depths: Tensor,  # [G]
        background: Tensor,  # [C]
        image_height: int,
        image_width: int,
        alpha_threshold: float,
        max_alpha: float,
        eps: float,
        front_to_back: bool,
        pixel_center_offset: float,
        use_fp16_patches: bool,
        exact_ellipse_mask: bool,
    ) -> Tensor:
        device = means2d.device
        dtype = means2d.dtype
        G = means2d.shape[0]
        C = colors.shape[1]

        order = _stable_argsort_depth(depths, front_to_back=front_to_back)
        means_s = means2d.index_select(0, order)
        conics_s = conics.index_select(0, order)
        colors_s = colors.index_select(0, order)
        opacities_s = opacities.index_select(0, order)
        depths_s = depths.index_select(0, order)

        # Output buffers kept in fp32 for stability.
        accum = torch.zeros((image_height, image_width, C), device=device, dtype=torch.float32)
        trans = torch.ones((image_height, image_width, 1), device=device, dtype=torch.float32)

        # Precompute pixel centers.
        xs_full = torch.arange(image_width, device=device, dtype=torch.float32) + float(pixel_center_offset)
        ys_full = torch.arange(image_height, device=device, dtype=torch.float32) + float(pixel_center_offset)

        # Saved metadata.
        bboxes = torch.full((G, 4), -1, device=device, dtype=torch.int32)  # x0, x1, y0, y1 (x1/y1 exclusive)
        taus = torch.zeros((G,), device=device, dtype=torch.float32)
        valid = torch.zeros((G,), device=device, dtype=torch.bool)

        patch_dtype = torch.float16 if use_fp16_patches and device.type != "cpu" else torch.float32

        for i in range(G):
            opacity = float(opacities_s[i].detach().item())
            if opacity <= alpha_threshold:
                continue

            a, b, c = conics_s[i]
            a_f = float(a.detach().item())
            b_f = float(b.detach().item())
            c_f = float(c.detach().item())
            det = a_f * c_f - b_f * b_f
            if det <= eps or a_f <= 0.0 or c_f <= 0.0:
                continue

            tau_f = -2.0 * math.log(max(alpha_threshold / max(opacity, eps), eps))
            if tau_f <= 0.0:
                continue

            # Tight opacity-aware axis-aligned bounds for Q x <= tau.
            half_x = math.sqrt(tau_f * c_f / det)
            half_y = math.sqrt(tau_f * a_f / det)
            mx, my = means_s[i]
            mx_f = float(mx.detach().item())
            my_f = float(my.detach().item())

            x0 = max(0, int(math.floor(mx_f - half_x - pixel_center_offset) + 1))
            x1 = min(image_width, int(math.ceil(mx_f + half_x - pixel_center_offset) + 1))
            y0 = max(0, int(math.floor(my_f - half_y - pixel_center_offset) + 1))
            y1 = min(image_height, int(math.ceil(my_f + half_y - pixel_center_offset) + 1))

            if x0 >= x1 or y0 >= y1:
                continue

            x = xs_full[x0:x1].to(patch_dtype) - mx.to(patch_dtype)
            y = ys_full[y0:y1].to(patch_dtype) - my.to(patch_dtype)
            dx = x[None, :]
            dy = y[:, None]

            a_p = a.to(patch_dtype)
            b_p = b.to(patch_dtype)
            c_p = c.to(patch_dtype)
            q = a_p * dx * dx + 2.0 * b_p * dx * dy + c_p * dy * dy
            if exact_ellipse_mask:
                mask = q <= torch.tensor(tau_f, device=device, dtype=patch_dtype)
            else:
                mask = torch.ones_like(q, dtype=torch.bool)

            if not bool(mask.any().item()):
                continue

            alpha_unc = opacities_s[i].to(patch_dtype) * torch.exp(-0.5 * q)
            alpha = torch.where(mask, alpha_unc, torch.zeros_like(alpha_unc))
            alpha = torch.clamp(alpha, min=0.0, max=max_alpha)
            if float(alpha.max().detach().item()) <= 0.0:
                continue

            alpha_f = alpha.to(torch.float32).unsqueeze(-1)
            patch_t = trans[y0:y1, x0:x1]
            patch_w = patch_t * alpha_f
            accum[y0:y1, x0:x1] += patch_w * colors_s[i].to(torch.float32).view(1, 1, C)
            trans[y0:y1, x0:x1] = patch_t * (1.0 - alpha_f)

            bboxes[i, 0] = x0
            bboxes[i, 1] = x1
            bboxes[i, 2] = y0
            bboxes[i, 3] = y1
            taus[i] = tau_f
            valid[i] = True

        out = accum + trans * background.to(torch.float32).view(1, 1, C)
        out = out.to(dtype)

        ctx.save_for_backward(
            means_s,
            conics_s,
            colors_s,
            opacities_s,
            depths_s,
            background,
            order,
            bboxes,
            taus,
            valid,
            trans,
        )
        ctx.image_height = image_height
        ctx.image_width = image_width
        ctx.alpha_threshold = alpha_threshold
        ctx.max_alpha = max_alpha
        ctx.eps = eps
        ctx.front_to_back = front_to_back
        ctx.pixel_center_offset = pixel_center_offset
        ctx.use_fp16_patches = use_fp16_patches
        ctx.exact_ellipse_mask = exact_ellipse_mask
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        (
            means_s,
            conics_s,
            colors_s,
            opacities_s,
            depths_s,
            background,
            order,
            bboxes,
            taus,
            valid,
            trans_final,
        ) = ctx.saved_tensors

        device = means_s.device
        dtype = means_s.dtype
        G = means_s.shape[0]
        C = colors_s.shape[1]
        H = ctx.image_height
        W = ctx.image_width
        eps = ctx.eps
        max_alpha = ctx.max_alpha
        pixel_center_offset = ctx.pixel_center_offset
        use_fp16_patches = ctx.use_fp16_patches
        exact_ellipse_mask = ctx.exact_ellipse_mask

        grad_out_f = grad_out.to(torch.float32)
        bg_f = background.to(torch.float32)

        # Reverse-mode state for front-to-back alpha compositing.
        # gA is constant and equal to dL/d(output_color).
        gA = grad_out_f
        gT = (grad_out_f * bg_f.view(1, 1, C)).sum(dim=-1, keepdim=True)
        T_post = trans_final.clone()

        grad_means_s = torch.zeros_like(means_s, dtype=torch.float32)
        grad_conics_s = torch.zeros_like(conics_s, dtype=torch.float32)
        grad_colors_s = torch.zeros_like(colors_s, dtype=torch.float32)
        grad_opacities_s = torch.zeros_like(opacities_s, dtype=torch.float32)
        grad_background = (grad_out_f * trans_final).sum(dim=(0, 1)).to(background.dtype)

        xs_full = torch.arange(W, device=device, dtype=torch.float32) + float(pixel_center_offset)
        ys_full = torch.arange(H, device=device, dtype=torch.float32) + float(pixel_center_offset)
        patch_dtype = torch.float16 if use_fp16_patches and device.type != "cpu" else torch.float32

        for i in range(G - 1, -1, -1):
            if not bool(valid[i].item()):
                continue
            x0, x1, y0, y1 = [int(v.item()) for v in bboxes[i]]
            if x0 < 0 or y0 < 0 or x0 >= x1 or y0 >= y1:
                continue

            mx, my = means_s[i]
            a, b, c = conics_s[i]
            opacity = opacities_s[i]
            color = colors_s[i].to(torch.float32)
            tau = taus[i]

            x = xs_full[x0:x1].to(patch_dtype) - mx.to(patch_dtype)
            y = ys_full[y0:y1].to(patch_dtype) - my.to(patch_dtype)
            dx = x[None, :]
            dy = y[:, None]
            a_p = a.to(patch_dtype)
            b_p = b.to(patch_dtype)
            c_p = c.to(patch_dtype)
            q = a_p * dx * dx + 2.0 * b_p * dx * dy + c_p * dy * dy

            if exact_ellipse_mask:
                mask = q <= tau.to(patch_dtype)
            else:
                mask = torch.ones_like(q, dtype=torch.bool)

            alpha_unc = opacity.to(patch_dtype) * torch.exp(-0.5 * q)
            alpha = torch.where(mask, alpha_unc, torch.zeros_like(alpha_unc))
            saturated = alpha >= max_alpha
            alpha = torch.clamp(alpha, min=0.0, max=max_alpha)

            alpha_f = alpha.to(torch.float32).unsqueeze(-1)
            one_minus_alpha = torch.clamp(1.0 - alpha_f, min=eps)

            T_post_patch = T_post[y0:y1, x0:x1]
            gT_patch = gT[y0:y1, x0:x1]
            gA_patch = gA[y0:y1, x0:x1]

            # Recover prefix transmittance exactly.
            T_prev_patch = T_post_patch / one_minus_alpha

            # Local gradients.
            grad_colors_s[i] = (gA_patch * (T_prev_patch * alpha_f)).sum(dim=(0, 1))

            grad_alpha = (gA_patch * (T_prev_patch * color.view(1, 1, C))).sum(dim=-1, keepdim=True)
            grad_alpha = grad_alpha - gT_patch * T_prev_patch

            unsat = (~saturated).to(torch.float32).unsqueeze(-1)
            mask_f = mask.to(torch.float32).unsqueeze(-1)
            grad_alpha = grad_alpha * unsat * mask_f

            exp_term = torch.exp(-0.5 * q.to(torch.float32)).unsqueeze(-1)
            grad_opacities_s[i] = (grad_alpha * exp_term).sum()

            # q = a dx^2 + 2b dxdy + c dy^2
            # alpha_unc = opacity * exp(-0.5 q)
            grad_q = grad_alpha * (-0.5) * alpha_unc.to(torch.float32).unsqueeze(-1) * unsat * mask_f

            dx_f = dx.to(torch.float32).unsqueeze(-1)
            dy_f = dy.to(torch.float32).unsqueeze(-1)

            grad_means_s[i, 0] = (grad_q * (-2.0 * a.to(torch.float32) * dx_f - 2.0 * b.to(torch.float32) * dy_f)).sum()
            grad_means_s[i, 1] = (grad_q * (-2.0 * b.to(torch.float32) * dx_f - 2.0 * c.to(torch.float32) * dy_f)).sum()

            grad_conics_s[i, 0] = (grad_q * (dx_f * dx_f)).sum()
            grad_conics_s[i, 1] = (grad_q * (2.0 * dx_f * dy_f)).sum()
            grad_conics_s[i, 2] = (grad_q * (dy_f * dy_f)).sum()

            # Propagate reverse scan state.
            gT_prev_patch = (gA_patch * (alpha_f * color.view(1, 1, C))).sum(dim=-1, keepdim=True)
            gT_prev_patch = gT_prev_patch + gT_patch * one_minus_alpha
            gT[y0:y1, x0:x1] = gT_prev_patch
            T_post[y0:y1, x0:x1] = T_prev_patch

        # Unsort to original order.
        inv_order = torch.empty_like(order)
        inv_order[order] = torch.arange(G, device=device, dtype=order.dtype)

        grad_means = grad_means_s.index_select(0, inv_order).to(dtype)
        grad_conics = grad_conics_s.index_select(0, inv_order).to(dtype)
        grad_colors = grad_colors_s.index_select(0, inv_order).to(colors_s.dtype)
        grad_opacities = grad_opacities_s.index_select(0, inv_order).to(opacities_s.dtype)

        # Sorting is piecewise-constant in depth; return no gradient through depths/order.
        grad_depths = None

        return (
            grad_means,
            grad_conics,
            grad_colors,
            grad_opacities,
            grad_depths,
            grad_background,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class MemoryEfficientGaussianRasterizer(nn.Module):
    """nn.Module wrapper around the custom autograd Function."""

    def __init__(self, config: Optional[RasterizeConfig] = None):
        super().__init__()
        self.config = config or RasterizeConfig()

    def forward(
        self,
        means2d: Tensor,
        conics: Tensor,
        colors: Tensor,
        opacities: Tensor,
        depths: Tensor,
        image_size: Tuple[int, int],
        background: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            means2d: [G, 2] projected means in pixel space.
            conics: [G, 3] precision matrix coefficients [a, b, c].
            colors: [G, C], typically C=3.
            opacities: [G].
            depths: [G], used only for visibility ordering.
            image_size: (H, W).
            background: [C], defaults to zeros.
        Returns:
            image: [H, W, C]
        """
        H, W = image_size
        if background is None:
            background = torch.zeros(colors.shape[-1], device=colors.device, dtype=colors.dtype)
        cfg = self.config
        return _GaussianRasterizeFn.apply(
            means2d,
            conics,
            colors,
            opacities,
            depths,
            background,
            int(H),
            int(W),
            float(cfg.alpha_threshold),
            float(cfg.max_alpha),
            float(cfg.eps),
            bool(cfg.front_to_back),
            float(cfg.pixel_center_offset),
            bool(cfg.use_fp16_patches),
            bool(cfg.exact_ellipse_mask),
        )


def rasterize_gaussians_torch(
    means2d: Tensor,
    conics: Tensor,
    colors: Tensor,
    opacities: Tensor,
    depths: Tensor,
    image_size: Tuple[int, int],
    background: Optional[Tensor] = None,
    config: Optional[RasterizeConfig] = None,
) -> Tensor:
    rasterizer = MemoryEfficientGaussianRasterizer(config)
    return rasterizer(
        means2d=means2d,
        conics=conics,
        colors=colors,
        opacities=opacities,
        depths=depths,
        image_size=image_size,
        background=background,
    )


def _naive_dense_reference(
    means2d: Tensor,
    conics: Tensor,
    colors: Tensor,
    opacities: Tensor,
    depths: Tensor,
    image_size: Tuple[int, int],
    background: Optional[Tensor] = None,
    max_alpha: float = 0.99,
    pixel_center_offset: float = 0.5,
):
    """Reference implementation for small correctness tests only."""
    H, W = image_size
    device = means2d.device
    dtype = means2d.dtype
    if background is None:
        background = torch.zeros(colors.shape[-1], device=device, dtype=dtype)
    order = _stable_argsort_depth(depths, front_to_back=True)
    means2d = means2d.index_select(0, order)
    conics = conics.index_select(0, order)
    colors = colors.index_select(0, order)
    opacities = opacities.index_select(0, order)

    xs = torch.arange(W, device=device, dtype=dtype) + pixel_center_offset
    ys = torch.arange(H, device=device, dtype=dtype) + pixel_center_offset
    xx = xs[None, :].expand(H, W)
    yy = ys[:, None].expand(H, W)

    img = torch.zeros(H, W, colors.shape[-1], device=device, dtype=dtype)
    T = torch.ones(H, W, 1, device=device, dtype=dtype)
    for i in range(means2d.shape[0]):
        dx = xx - means2d[i, 0]
        dy = yy - means2d[i, 1]
        a, b, c = conics[i]
        q = a * dx * dx + 2 * b * dx * dy + c * dy * dy
        alpha = torch.clamp(opacities[i] * torch.exp(-0.5 * q), min=0.0, max=max_alpha)
        alpha = alpha.unsqueeze(-1)
        img = img + T * alpha * colors[i].view(1, 1, -1)
        T = T * (1.0 - alpha)
    return img + T * background.view(1, 1, -1)


if __name__ == "__main__":
    # Tiny self-check against dense autograd for a small case.
    torch.manual_seed(0)
    device = "cpu"
    G, H, W = 6, 16, 16

    means = torch.rand(G, 2, device=device) * torch.tensor([W, H], device=device)
    means.requires_grad_()

    raw = torch.randn(G, 3, device=device)
    # Build positive-definite conics.
    a = raw[:, 0].square() + 0.5
    b = raw[:, 1] * 0.05
    c = raw[:, 2].square() + 0.5
    conics = torch.stack([a, b, c], dim=-1).requires_grad_()

    colors = torch.rand(G, 3, device=device, requires_grad=True)
    opacities = (0.6 * torch.sigmoid(torch.randn(G, device=device))).requires_grad_()
    depths = torch.rand(G, device=device)
    bg = torch.rand(3, device=device)

    img_ref = _naive_dense_reference(means, conics, colors, opacities, depths, (H, W), bg)
    loss_ref = img_ref.square().mean()
    grads_ref = torch.autograd.grad(loss_ref, (means, conics, colors, opacities), retain_graph=False)

    cfg = RasterizeConfig(alpha_threshold=1e-4, exact_ellipse_mask=False)
    img_fast = rasterize_gaussians_torch(means, conics, colors, opacities, depths, (H, W), bg, cfg)
    loss_fast = img_fast.square().mean()
    grads_fast = torch.autograd.grad(loss_fast, (means, conics, colors, opacities), retain_graph=False)

    print("max |img diff|:", (img_ref - img_fast).abs().max().item())
    names = ["means", "conics", "colors", "opacities"]
    for name, ga, gb in zip(names, grads_ref, grads_fast):
        print(name, "max |grad diff|:", (ga - gb).abs().max().item())
