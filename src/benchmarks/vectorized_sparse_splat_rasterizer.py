from __future__ import annotations

"""
Torch-only vectorized sparse differentiable Gaussian rasterizer.

This version is designed to be much friendlier to torch.compile and autograd
than a splat-by-splat Python loop. The core idea is:

1. Sort Gaussians globally by depth (front-to-back).
2. Compute tight opacity-aware AABBs for each Gaussian, detached from grad.
3. Build a sparse worklist of covered pixels inside each Gaussian's AABB.
4. Stably sort the worklist by pixel id, which preserves depth order inside
   every pixel segment.
5. Perform exact front-to-back alpha compositing with a segmented prefix scan
   over that sparse worklist.

Key properties:
- No dense [G, H, W] tensors.
- Work scales with the *covered* pixels sum_i area(bbox_i), not G*H*W.
- Core compositing math is vectorized and differentiable in plain PyTorch.
- Optional activation checkpointing to trade extra compute for lower memory.
- Optional torch.compile wrapper for the differentiable core.

Notes:
- Sorting, bbox truncation, and worklist construction are treated as detached
  visibility/culling logic, so gradients do not flow through support changes.
- This is exact front-to-back alpha compositing for the chosen sparse support.
- The conservative support is the tight opacity-aware AABB. Optionally, an
  exact ellipse mask is applied inside the box.
"""

from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint

__all__ = [
    "SparseRasterConfig",
    "VectorizedSparseGaussianRasterizer",
    "rasterize_gaussians_sparse_torch",
]


def _stable_argsort(x: Tensor, descending: bool = False) -> Tensor:
    try:
        return torch.argsort(x, dim=0, descending=descending, stable=True)
    except TypeError:
        return torch.argsort(x, dim=0, descending=descending)


@dataclass
class SparseRasterConfig:
    alpha_threshold: float = 1.0 / 255.0
    max_alpha: float = 0.99
    eps: float = 1e-8
    front_to_back: bool = True
    pixel_center_offset: float = 0.5
    exact_ellipse_mask: bool = True
    use_checkpoint: bool = True
    compile_core: bool = False
    compile_dynamic: bool = True
    compile_mode: str = "default"
    accumulation_dtype: str = "float32"  # "float32" or "float16"/"bfloat16"


class _Core(nn.Module):
    def __init__(self, config: SparseRasterConfig):
        super().__init__()
        self.cfg = config

    def forward(
        self,
        means2d_s: Tensor,  # [G,2], already depth-sorted
        conics_s: Tensor,  # [G,3]
        colors_s: Tensor,  # [G,C]
        opacities_s: Tensor,  # [G]
        background: Tensor,  # [C]
        taus_s: Tensor,  # [G], detached support threshold per Gaussian
        work_g: Tensor,  # [N], gaussian id in sorted order for each sample
        work_p: Tensor,  # [N], flat pixel id for each sample, sorted stably by pixel
        work_x: Tensor,  # [N], pixel center x coordinate
        work_y: Tensor,  # [N], pixel center y coordinate
        seg_start_idx: Tensor,  # [N], start index of the pixel segment for each sample
        seg_end_idx: Tensor,  # [M], last sample index for each pixel segment
        image_height: int,
        image_width: int,
    ) -> Tensor:
        cfg = self.cfg
        device = means2d_s.device
        C = colors_s.shape[-1]
        out_dtype = means2d_s.dtype
        accum_dtype = getattr(torch, cfg.accumulation_dtype)
        num_pixels = image_height * image_width

        if work_g.numel() == 0:
            return background.to(accum_dtype).view(1, 1, C).expand(image_height, image_width, C).to(out_dtype)

        means = means2d_s.index_select(0, work_g).to(accum_dtype)  # [N,2]
        conics = conics_s.index_select(0, work_g).to(accum_dtype)  # [N,3]
        colors = colors_s.index_select(0, work_g).to(accum_dtype)  # [N,C]
        opacity = opacities_s.index_select(0, work_g).to(accum_dtype)  # [N]
        tau = taus_s.index_select(0, work_g).to(accum_dtype)  # [N]

        dx = work_x.to(accum_dtype) - means[:, 0]
        dy = work_y.to(accum_dtype) - means[:, 1]
        a, b, c = conics.unbind(dim=-1)
        q = a * dx.square() + 2.0 * b * dx * dy + c * dy.square()

        alpha = opacity * torch.exp(-0.5 * q)
        if cfg.exact_ellipse_mask:
            alpha = torch.where(q <= tau, alpha, torch.zeros_like(alpha))
        alpha = torch.clamp(alpha, min=0.0, max=cfg.max_alpha)

        # Exact segmented front-to-back compositing on the sparse worklist.
        one_minus = torch.clamp(1.0 - alpha, min=cfg.eps)
        log_one = torch.log(one_minus)
        cs = torch.cumsum(log_one, dim=0)  # inclusive cumsum over the whole worklist
        zero = torch.zeros(1, device=device, dtype=accum_dtype)
        cs_prev = torch.cat([zero, cs[:-1]], dim=0)
        base_prev = cs_prev.index_select(0, seg_start_idx)
        prefix_log = cs_prev - base_prev
        weights = alpha * torch.exp(prefix_log)  # [N]

        flat_rgb = torch.zeros((num_pixels, C), device=device, dtype=accum_dtype)
        flat_rgb.index_add_(0, work_p, weights.unsqueeze(-1) * colors)

        # Final transmittance per pixel: product over all (1 - alpha) in the segment.
        seg_final_log = cs.index_select(0, seg_end_idx) - base_prev.index_select(0, seg_end_idx)
        seg_final_T = torch.exp(seg_final_log)  # [M]
        flat_T = torch.ones((num_pixels,), device=device, dtype=accum_dtype)
        flat_T.index_copy_(0, work_p.index_select(0, seg_end_idx), seg_final_T)

        out = flat_rgb + flat_T.unsqueeze(-1) * background.to(accum_dtype).view(1, C)
        out = out.view(image_height, image_width, C)
        return out.to(out_dtype)


class VectorizedSparseGaussianRasterizer(nn.Module):
    """
    Vectorized sparse exact-alpha rasterizer.

    The expensive differentiable part is written as a single tensor program over
    a compact sparse worklist. Worklist construction uses detached integer logic.
    """

    def __init__(self, config: Optional[SparseRasterConfig] = None):
        super().__init__()
        self.config = config or SparseRasterConfig()
        self.core = _Core(self.config)
        self._compiled_core = None
        if self.config.compile_core and hasattr(torch, "compile"):
            self._compiled_core = torch.compile(
                self.core,
                dynamic=self.config.compile_dynamic,
                mode=self.config.compile_mode,
                fullgraph=False,
            )

    @staticmethod
    def _compute_support_and_order(
        means2d: Tensor,
        conics: Tensor,
        opacities: Tensor,
        depths: Tensor,
        image_height: int,
        image_width: int,
        alpha_threshold: float,
        eps: float,
        front_to_back: bool,
        pixel_center_offset: float,
    ):
        """
        Detached support computation.

        Returns Gaussians sorted by depth and tight opacity-aware AABBs:
            x0, x1, y0, y1, tau, valid
        where x1/y1 are exclusive.
        """
        device = means2d.device
        with torch.no_grad():
            order = _stable_argsort(depths, descending=not front_to_back)
            means_s = means2d.index_select(0, order)
            conics_s = conics.index_select(0, order)
            opacities_s = opacities.index_select(0, order)

            a = conics_s[:, 0].float()
            b = conics_s[:, 1].float()
            c = conics_s[:, 2].float()
            mx = means_s[:, 0].float()
            my = means_s[:, 1].float()
            op = opacities_s.float()

            det = a * c - b * b
            valid = (op > alpha_threshold) & (a > 0.0) & (c > 0.0) & (det > eps)

            tau = torch.zeros_like(op)
            safe_ratio = torch.clamp(alpha_threshold / torch.clamp(op, min=eps), min=eps)
            tau_valid = -2.0 * torch.log(safe_ratio)
            tau = torch.where(valid, tau_valid, tau)
            valid = valid & (tau > 0.0)

            safe_det = torch.clamp(det, min=eps)
            half_x = torch.sqrt(torch.clamp(tau * c / safe_det, min=0.0))
            half_y = torch.sqrt(torch.clamp(tau * a / safe_det, min=0.0))

            pco = float(pixel_center_offset)
            x0 = torch.floor(mx - half_x - pco).to(torch.int64) + 1
            x1 = torch.ceil(mx + half_x - pco).to(torch.int64) + 1
            y0 = torch.floor(my - half_y - pco).to(torch.int64) + 1
            y1 = torch.ceil(my + half_y - pco).to(torch.int64) + 1

            x0 = x0.clamp_(0, image_width)
            x1 = x1.clamp_(0, image_width)
            y0 = y0.clamp_(0, image_height)
            y1 = y1.clamp_(0, image_height)
            valid = valid & (x0 < x1) & (y0 < y1)

            bbox = torch.stack([x0, x1, y0, y1], dim=-1)
            return order, bbox, tau, valid

    @staticmethod
    def _build_sparse_worklist(
        bbox_s: Tensor,  # [G,4] in sorted order
        valid_s: Tensor,  # [G]
        image_width: int,
        pixel_center_offset: float,
    ):
        """
        Detached sparse worklist.

        Builds one sample per pixel inside each valid AABB, grouped first by
        Gaussian (already depth-sorted), then stably sorted by flat pixel id.
        Stable sort preserves front-to-back depth order within each pixel.

        Returns:
            work_g, work_p, work_x, work_y, seg_start_idx, seg_end_idx
        """
        device = bbox_s.device
        with torch.no_grad():
            x0 = bbox_s[:, 0]
            x1 = bbox_s[:, 1]
            y0 = bbox_s[:, 2]
            y1 = bbox_s[:, 3]
            widths = torch.where(valid_s, x1 - x0, torch.zeros_like(x0))
            heights = torch.where(valid_s, y1 - y0, torch.zeros_like(y0))
            areas = widths * heights
            total = int(areas.sum().item())

            if total == 0:
                empty_long = torch.empty(0, dtype=torch.long, device=device)
                empty_float = torch.empty(0, dtype=torch.float32, device=device)
                return empty_long, empty_long, empty_float, empty_float, empty_long, empty_long

            g_ids = torch.repeat_interleave(
                torch.arange(bbox_s.shape[0], device=device, dtype=torch.long),
                areas.to(torch.long),
            )
            area_offsets = torch.cumsum(areas.to(torch.long), dim=0) - areas.to(torch.long)
            rep_offsets = torch.repeat_interleave(area_offsets, areas.to(torch.long))
            rep_x0 = torch.repeat_interleave(x0.to(torch.long), areas.to(torch.long))
            rep_y0 = torch.repeat_interleave(y0.to(torch.long), areas.to(torch.long))
            rep_w = torch.repeat_interleave(widths.to(torch.long), areas.to(torch.long))

            local = torch.arange(total, device=device, dtype=torch.long) - rep_offsets
            local_x = torch.remainder(local, rep_w)
            local_y = torch.div(local, rep_w, rounding_mode="floor")

            px = rep_x0 + local_x
            py = rep_y0 + local_y
            flat_pixel = py * image_width + px

            perm = _stable_argsort(flat_pixel, descending=False)
            work_g = g_ids.index_select(0, perm)
            work_p = flat_pixel.index_select(0, perm)
            work_x = px.index_select(0, perm).to(torch.float32) + float(pixel_center_offset)
            work_y = py.index_select(0, perm).to(torch.float32) + float(pixel_center_offset)

            n = work_p.numel()
            idx = torch.arange(n, device=device, dtype=torch.long)
            start_flag = torch.empty(n, device=device, dtype=torch.bool)
            start_flag[0] = True
            start_flag[1:] = work_p[1:] != work_p[:-1]
            seg_start = torch.where(start_flag, idx, torch.zeros_like(idx))
            seg_start = torch.cummax(seg_start, dim=0).values

            end_flag = torch.empty(n, device=device, dtype=torch.bool)
            end_flag[-1] = True
            end_flag[:-1] = work_p[:-1] != work_p[1:]
            seg_end = idx[end_flag]

            return work_g, work_p, work_x, work_y, seg_start, seg_end

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
            means2d:   [G, 2] projected means in pixel space.
            conics:    [G, 3] precision matrix coefficients [a, b, c].
            colors:    [G, C].
            opacities: [G].
            depths:    [G], for visibility ordering only.
            image_size:(H, W)
            background:[C], defaults to zeros.

        Returns:
            image: [H, W, C]
        """
        cfg = self.config
        H, W = int(image_size[0]), int(image_size[1])
        if background is None:
            background = torch.zeros(colors.shape[-1], device=colors.device, dtype=colors.dtype)

        order, bbox_s, tau_s, valid_s = self._compute_support_and_order(
            means2d=means2d,
            conics=conics,
            opacities=opacities,
            depths=depths,
            image_height=H,
            image_width=W,
            alpha_threshold=float(cfg.alpha_threshold),
            eps=float(cfg.eps),
            front_to_back=bool(cfg.front_to_back),
            pixel_center_offset=float(cfg.pixel_center_offset),
        )

        means_s = means2d.index_select(0, order)
        conics_s = conics.index_select(0, order)
        colors_s = colors.index_select(0, order)
        opacities_s = opacities.index_select(0, order)

        work_g, work_p, work_x, work_y, seg_start, seg_end = self._build_sparse_worklist(
            bbox_s=bbox_s,
            valid_s=valid_s,
            image_width=W,
            pixel_center_offset=float(cfg.pixel_center_offset),
        )

        core = self._compiled_core if self._compiled_core is not None else self.core

        # The worklist is constant w.r.t. gradients. The differentiable core can
        # be checkpointed to avoid storing large N-sized activations.
        if cfg.use_checkpoint and any(t.requires_grad for t in (means_s, conics_s, colors_s, opacities_s, background)):
            fn = partial(
                core,
                taus_s=tau_s,
                work_g=work_g,
                work_p=work_p,
                work_x=work_x,
                work_y=work_y,
                seg_start_idx=seg_start,
                seg_end_idx=seg_end,
                image_height=H,
                image_width=W,
            )
            out = checkpoint(
                fn,
                means_s,
                conics_s,
                colors_s,
                opacities_s,
                background,
                use_reentrant=False,
            )
        else:
            out = core(
                means2d_s=means_s,
                conics_s=conics_s,
                colors_s=colors_s,
                opacities_s=opacities_s,
                background=background,
                taus_s=tau_s,
                work_g=work_g,
                work_p=work_p,
                work_x=work_x,
                work_y=work_y,
                seg_start_idx=seg_start,
                seg_end_idx=seg_end,
                image_height=H,
                image_width=W,
            )
        return out


def rasterize_gaussians_sparse_torch(
    means2d: Tensor,
    conics: Tensor,
    colors: Tensor,
    opacities: Tensor,
    depths: Tensor,
    image_size: Tuple[int, int],
    background: Optional[Tensor] = None,
    config: Optional[SparseRasterConfig] = None,
) -> Tensor:
    return VectorizedSparseGaussianRasterizer(config)(
        means2d=means2d,
        conics=conics,
        colors=colors,
        opacities=opacities,
        depths=depths,
        image_size=image_size,
        background=background,
    )


# ------------------------------ tiny reference ------------------------------ #


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
    H, W = image_size
    device = means2d.device
    dtype = means2d.dtype
    if background is None:
        background = torch.zeros(colors.shape[-1], device=device, dtype=dtype)
    order = _stable_argsort(depths, descending=False)
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
    torch.manual_seed(0)
    device = "cpu"
    G, H, W = 8, 20, 20

    means = torch.rand(G, 2, device=device) * torch.tensor([W, H], device=device)
    means.requires_grad_()

    raw = torch.randn(G, 3, device=device)
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

    cfg = SparseRasterConfig(alpha_threshold=1e-4, exact_ellipse_mask=False, use_checkpoint=False)
    rast = VectorizedSparseGaussianRasterizer(cfg)
    img_fast = rast(means, conics, colors, opacities, depths, (H, W), bg)
    loss_fast = img_fast.square().mean()
    grads_fast = torch.autograd.grad(loss_fast, (means, conics, colors, opacities), retain_graph=False)

    print("max |img diff|:", (img_ref - img_fast).abs().max().item())
    names = ["means", "conics", "colors", "opacities"]
    for name, ga, gb in zip(names, grads_ref, grads_fast):
        print(name, "max |grad diff|:", (ga - gb).abs().max().item())
