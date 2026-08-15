"""Thin native-Metal bridge for retained-depth SPD(4) optical transfer."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import torch
from torch import Tensor


SOURCE_PATH = Path(__file__).with_name("retained_fiber_transfer.metal")


@dataclass(frozen=True)
class RetainedFiberTileCertificate:
    """Detached tile-level decision for fast ordering versus fiber fallback."""

    fallback_tiles: Tensor
    fallback_mask: Tensor
    active_counts: Tensor
    reason_bits: Tensor
    minimum_pair_separation: Tensor


class RetainedFiberMetal:
    def __init__(self, source_path: Path = SOURCE_PATH) -> None:
        self.source_path = source_path
        self._library = None

    def compile(self):
        if self._library is None:
            if not hasattr(torch.mps, "compile_shader"):
                raise RuntimeError("this PyTorch build does not provide torch.mps.compile_shader")
            self._library = torch.mps.compile_shader(self.source_path.read_text())
        return self._library

    @staticmethod
    def _validate(
        ma: Tensor,
        q_uvt: Tensor,
        depth0: Tensor,
        depth_beta: Tensor,
        depth_variance: Tensor,
        optical_thickness: Tensor,
        color: Tensor,
        times: Tensor,
        *,
        height: int,
        width: int,
        depth_samples: int,
        sigma_extent: float,
    ) -> None:
        atom_count = int(ma.shape[0])
        expected = {
            "ma": (atom_count, 3),
            "q_uvt": (atom_count, 6),
            "depth0": (atom_count,),
            "depth_beta": (atom_count, 3),
            "depth_variance": (atom_count,),
            "optical_thickness": (atom_count,),
            "color": (atom_count, 3),
        }
        values = {
            "ma": ma,
            "q_uvt": q_uvt,
            "depth0": depth0,
            "depth_beta": depth_beta,
            "depth_variance": depth_variance,
            "optical_thickness": optical_thickness,
            "color": color,
        }
        for name, value in values.items():
            if tuple(value.shape) != expected[name]:
                raise ValueError(f"{name} must have shape {expected[name]}")
        if times.ndim != 1 or times.numel() == 0:
            raise ValueError("times must have shape [F] with F > 0")
        tensors = (*values.values(), times)
        if any(value.device.type != "mps" for value in tensors):
            raise ValueError("retained-fiber Metal inputs must all be MPS tensors")
        if any(value.dtype != torch.float32 for value in tensors):
            raise ValueError("retained-fiber Metal inputs must all be float32")
        if any(not value.is_contiguous() for value in tensors):
            raise ValueError("retained-fiber Metal inputs must be contiguous")
        if not all(bool(torch.isfinite(value).all().item()) for value in tensors):
            raise ValueError("retained-fiber Metal inputs must be finite")
        if bool(torch.any(depth_variance <= 0.0).item()):
            raise ValueError("depth_variance must be strictly positive")
        if bool(torch.any(optical_thickness < 0.0).item()):
            raise ValueError("optical_thickness must be nonnegative")
        if height <= 0 or width <= 0:
            raise ValueError("height and width must be positive")
        if depth_samples <= 0 or depth_samples > 64:
            raise ValueError("depth_samples must lie in [1,64]")
        if not (float(sigma_extent) > 0.0):
            raise ValueError("sigma_extent must be positive")

    def forward(
        self,
        ma: Tensor,
        q_uvt: Tensor,
        depth0: Tensor,
        depth_beta: Tensor,
        depth_variance: Tensor,
        optical_thickness: Tensor,
        color: Tensor,
        times: Tensor,
        *,
        height: int,
        width: int,
        depth_samples: int = 64,
        sigma_extent: float = 6.0,
        background: tuple[float, float, float] = (0.0, 0.0, 0.0),
        fallback_mask: Tensor | None = None,
        alpha_threshold: float = 0.0,
    ) -> Tensor:
        self._validate(
            ma,
            q_uvt,
            depth0,
            depth_beta,
            depth_variance,
            optical_thickness,
            color,
            times,
            height=height,
            width=width,
            depth_samples=depth_samples,
            sigma_extent=sigma_extent,
        )
        if not math.isfinite(alpha_threshold) or not 0.0 <= alpha_threshold < 1.0:
            raise ValueError("alpha_threshold must be finite and lie in [0,1)")
        expected_mask = (int(times.numel()), int(height), int(width))
        if fallback_mask is None:
            fallback_mask = torch.ones(
                expected_mask,
                dtype=torch.int32,
                device="mps",
            )
        elif (
            tuple(fallback_mask.shape) != expected_mask
            or fallback_mask.device.type != "mps"
            or fallback_mask.dtype != torch.int32
            or not fallback_mask.is_contiguous()
        ):
            raise ValueError(
                f"fallback_mask must be contiguous MPS int32 with shape {expected_mask}"
            )
        output = torch.empty(
            (int(times.numel()), int(height), int(width), 3),
            dtype=torch.float32,
            device="mps",
        )
        self.compile().retained_fiber_forward(
            fallback_mask,
            output,
            ma,
            q_uvt,
            depth0,
            depth_beta,
            depth_variance,
            optical_thickness,
            color,
            times,
            int(ma.shape[0]),
            int(times.numel()),
            int(height),
            int(width),
            int(depth_samples),
            float(sigma_extent),
            list(background),
            float(alpha_threshold),
        )
        return output

    def vjp(
        self,
        grad_output: Tensor,
        ma: Tensor,
        q_uvt: Tensor,
        depth0: Tensor,
        depth_beta: Tensor,
        depth_variance: Tensor,
        optical_thickness: Tensor,
        color: Tensor,
        times: Tensor,
        *,
        height: int,
        width: int,
        depth_samples: int = 64,
        sigma_extent: float = 6.0,
        background: tuple[float, float, float] = (0.0, 0.0, 0.0),
        fallback_mask: Tensor | None = None,
        alpha_threshold: float = 0.0,
    ) -> dict[str, Tensor]:
        self._validate(
            ma,
            q_uvt,
            depth0,
            depth_beta,
            depth_variance,
            optical_thickness,
            color,
            times,
            height=height,
            width=width,
            depth_samples=depth_samples,
            sigma_extent=sigma_extent,
        )
        if not math.isfinite(alpha_threshold) or not 0.0 <= alpha_threshold < 1.0:
            raise ValueError("alpha_threshold must be finite and lie in [0,1)")
        expected_mask = (int(times.numel()), int(height), int(width))
        if fallback_mask is None:
            fallback_mask = torch.ones(
                expected_mask,
                dtype=torch.int32,
                device="mps",
            )
        elif (
            tuple(fallback_mask.shape) != expected_mask
            or fallback_mask.device.type != "mps"
            or fallback_mask.dtype != torch.int32
            or not fallback_mask.is_contiguous()
        ):
            raise ValueError(
                f"fallback_mask must be contiguous MPS int32 with shape {expected_mask}"
            )
        expected_output = (int(times.numel()), int(height), int(width), 3)
        if tuple(grad_output.shape) != expected_output:
            raise ValueError(f"grad_output must have shape {expected_output}")
        if (
            grad_output.device.type != "mps"
            or grad_output.dtype != torch.float32
            or not grad_output.is_contiguous()
        ):
            raise ValueError("grad_output must be contiguous MPS float32")
        gradients = {
            "ma": torch.zeros_like(ma),
            "q_uvt": torch.zeros_like(q_uvt),
            "depth0": torch.zeros_like(depth0),
            "depth_beta": torch.zeros_like(depth_beta),
            "depth_variance": torch.zeros_like(depth_variance),
            "optical_thickness": torch.zeros_like(optical_thickness),
            "color": torch.zeros_like(color),
        }
        self.compile().retained_fiber_vjp(
            fallback_mask,
            gradients["ma"],
            gradients["q_uvt"],
            gradients["depth0"],
            gradients["depth_beta"],
            gradients["depth_variance"],
            gradients["optical_thickness"],
            gradients["color"],
            grad_output,
            ma,
            q_uvt,
            depth0,
            depth_beta,
            depth_variance,
            optical_thickness,
            color,
            times,
            int(ma.shape[0]),
            int(times.numel()),
            int(height),
            int(width),
            int(depth_samples),
            float(sigma_extent),
            list(background),
            float(alpha_threshold),
        )
        return gradients

    def certify_tiles(
        self,
        ma: Tensor,
        q_uvt: Tensor,
        depth0: Tensor,
        depth_beta: Tensor,
        depth_variance: Tensor,
        optical_thickness: Tensor,
        *,
        frames: int,
        height: int,
        width: int,
        tile_x: int,
        tile_y: int,
        tile_t: int,
        alpha_threshold: float,
        sigma_multiplier: float = 3.0,
        required_gap: float = 0.0,
        depth_fit_error: Tensor | None = None,
    ) -> RetainedFiberTileCertificate:
        """Certify a fixed hard order or request retained-fiber tile fallback.

        The certificate is conservative: it first intersects each primitive's
        optical-support AABB with the tile, then proves every potentially
        overlapping pair's conditional-depth confidence bands are separated
        over their common AABB. Any invalid record, active-set overflow, or
        ambiguous pair requests fallback for the complete tile-time cell.
        """

        atom_count = int(ma.shape[0])
        expected = {
            "ma": (atom_count, 3),
            "q_uvt": (atom_count, 6),
            "depth0": (atom_count,),
            "depth_beta": (atom_count, 3),
            "depth_variance": (atom_count,),
            "optical_thickness": (atom_count,),
        }
        values = {
            "ma": ma,
            "q_uvt": q_uvt,
            "depth0": depth0,
            "depth_beta": depth_beta,
            "depth_variance": depth_variance,
            "optical_thickness": optical_thickness,
        }
        for name, value in values.items():
            if tuple(value.shape) != expected[name]:
                raise ValueError(f"{name} must have shape {expected[name]}")
            if (
                value.device.type != "mps"
                or value.dtype != torch.float32
                or not value.is_contiguous()
            ):
                raise ValueError(f"{name} must be contiguous MPS float32")
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"{name} must be finite")
        if bool(torch.any(depth_variance <= 0.0).item()):
            raise ValueError("depth_variance must be strictly positive")
        if bool(torch.any(optical_thickness < 0.0).item()):
            raise ValueError("optical_thickness must be nonnegative")
        if depth_fit_error is None:
            depth_fit_error = torch.zeros_like(depth_variance)
        elif (
            tuple(depth_fit_error.shape) != (atom_count,)
            or depth_fit_error.device.type != "mps"
            or depth_fit_error.dtype != torch.float32
            or not depth_fit_error.is_contiguous()
        ):
            raise ValueError(
                "depth_fit_error must be contiguous MPS float32 with shape [N]"
            )
        if not bool(torch.isfinite(depth_fit_error).all().item()) or bool(
            torch.any(depth_fit_error < 0.0).item()
        ):
            raise ValueError("depth_fit_error must be finite and nonnegative")
        if min(frames, height, width, tile_x, tile_y, tile_t) <= 0:
            raise ValueError("frame/image/tile dimensions must be positive")
        if not math.isfinite(alpha_threshold) or not 0.0 < alpha_threshold < 1.0:
            raise ValueError("alpha_threshold must be finite and lie in (0,1)")
        if not math.isfinite(sigma_multiplier) or sigma_multiplier < 0.0:
            raise ValueError("sigma_multiplier must be finite and nonnegative")
        if not math.isfinite(required_gap) or required_gap < 0.0:
            raise ValueError("required_gap must be finite and nonnegative")

        tiles_x = (int(width) + int(tile_x) - 1) // int(tile_x)
        tiles_y = (int(height) + int(tile_y) - 1) // int(tile_y)
        tiles_t = (int(frames) + int(tile_t) - 1) // int(tile_t)
        tile_shape = (tiles_t, tiles_y, tiles_x)
        fallback_tiles = torch.empty(tile_shape, dtype=torch.int32, device="mps")
        active_counts = torch.empty_like(fallback_tiles)
        reason_bits = torch.empty_like(fallback_tiles)
        minimum_pair_separation = torch.empty(
            tile_shape,
            dtype=torch.float32,
            device="mps",
        )
        self.compile().retained_fiber_certify_tiles(
            fallback_tiles,
            active_counts,
            reason_bits,
            minimum_pair_separation,
            ma,
            q_uvt,
            depth0,
            depth_beta,
            depth_variance,
            optical_thickness,
            depth_fit_error,
            atom_count,
            int(frames),
            int(height),
            int(width),
            int(tile_x),
            int(tile_y),
            int(tile_t),
            tiles_x,
            tiles_y,
            tiles_t,
            float(alpha_threshold),
            float(sigma_multiplier),
            float(required_gap),
        )
        fallback_mask = (
            fallback_tiles.repeat_interleave(int(tile_t), dim=0)
            .repeat_interleave(int(tile_y), dim=1)
            .repeat_interleave(int(tile_x), dim=2)[:frames, :height, :width]
            .contiguous()
        )
        return RetainedFiberTileCertificate(
            fallback_tiles=fallback_tiles,
            fallback_mask=fallback_mask,
            active_counts=active_counts,
            reason_bits=reason_bits,
            minimum_pair_separation=minimum_pair_separation,
        )


_DEFAULT_METAL = RetainedFiberMetal()


class _RetainedFiberMetalFunction(torch.autograd.Function):
    """Autograd boundary for the native retained-fiber forward/VJP pair."""

    @staticmethod
    def forward(
        ctx,
        ma: Tensor,
        q_uvt: Tensor,
        depth0: Tensor,
        depth_beta: Tensor,
        depth_variance: Tensor,
        optical_thickness: Tensor,
        color: Tensor,
        times: Tensor,
        fallback_mask: Tensor,
        height: int,
        width: int,
        depth_samples: int,
        sigma_extent: float,
        background: tuple[float, float, float],
        alpha_threshold: float,
    ) -> Tensor:
        if times.requires_grad:
            raise ValueError(
                "retained-fiber Metal treats frame times as compiled camera "
                "coordinates and does not provide time-coordinate gradients"
            )
        ctx.save_for_backward(
            ma,
            q_uvt,
            depth0,
            depth_beta,
            depth_variance,
            optical_thickness,
            color,
            times,
            fallback_mask,
        )
        ctx.height = int(height)
        ctx.width = int(width)
        ctx.depth_samples = int(depth_samples)
        ctx.sigma_extent = float(sigma_extent)
        ctx.background = tuple(float(value) for value in background)
        ctx.alpha_threshold = float(alpha_threshold)
        return _DEFAULT_METAL.forward(
            ma,
            q_uvt,
            depth0,
            depth_beta,
            depth_variance,
            optical_thickness,
            color,
            times,
            height=ctx.height,
            width=ctx.width,
            depth_samples=ctx.depth_samples,
            sigma_extent=ctx.sigma_extent,
            background=ctx.background,
            fallback_mask=fallback_mask,
            alpha_threshold=ctx.alpha_threshold,
        )

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (
            ma,
            q_uvt,
            depth0,
            depth_beta,
            depth_variance,
            optical_thickness,
            color,
            times,
            fallback_mask,
        ) = ctx.saved_tensors
        gradients = _DEFAULT_METAL.vjp(
            grad_output.contiguous(),
            ma,
            q_uvt,
            depth0,
            depth_beta,
            depth_variance,
            optical_thickness,
            color,
            times,
            height=ctx.height,
            width=ctx.width,
            depth_samples=ctx.depth_samples,
            sigma_extent=ctx.sigma_extent,
            background=ctx.background,
            fallback_mask=fallback_mask,
            alpha_threshold=ctx.alpha_threshold,
        )
        return (
            gradients["ma"],
            gradients["q_uvt"],
            gradients["depth0"],
            gradients["depth_beta"],
            gradients["depth_variance"],
            gradients["optical_thickness"],
            gradients["color"],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def render_retained_fiber_metal(
    ma: Tensor,
    q_uvt: Tensor,
    depth0: Tensor,
    depth_beta: Tensor,
    depth_variance: Tensor,
    optical_thickness: Tensor,
    color: Tensor,
    times: Tensor,
    *,
    height: int,
    width: int,
    depth_samples: int = 64,
    sigma_extent: float = 6.0,
    background: tuple[float, float, float] = (0.0, 0.0, 0.0),
    fallback_mask: Tensor | None = None,
    alpha_threshold: float = 0.0,
) -> Tensor:
    """Differentiable native-Metal retained-fiber render.

    The depth integration bounds remain a compiled, detached decision in both
    the reference and Metal implementations. Gradients are returned for every
    atom field, including conditional depth variance, but not for the frame
    coordinates or the bound-selection operation.
    """

    if fallback_mask is None:
        fallback_mask = torch.ones(
            (int(times.numel()), int(height), int(width)),
            dtype=torch.int32,
            device=ma.device,
        )
    return _RetainedFiberMetalFunction.apply(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        depth_variance,
        optical_thickness,
        color,
        times,
        fallback_mask,
        int(height),
        int(width),
        int(depth_samples),
        float(sigma_extent),
        tuple(float(value) for value in background),
        float(alpha_threshold),
    )


__all__ = [
    "RetainedFiberMetal",
    "RetainedFiberTileCertificate",
    "SOURCE_PATH",
    "render_retained_fiber_metal",
]
