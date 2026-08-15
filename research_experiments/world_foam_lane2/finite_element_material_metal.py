"""Thin ``torch.mps.compile_shader`` bridge for the material microkernel.

This is intentionally not an autograd Function or renderer integration.  It
exists only for tiny fixed-tape forward/VJP parity and branch-count smokes.
"""

from __future__ import annotations

from pathlib import Path

import torch

try:
    from .finite_element_material_transfer import BranchStatus, branch_status_counts
except ImportError:  # pragma: no cover - supports direct script-style loading.
    from finite_element_material_transfer import BranchStatus, branch_status_counts


SOURCE_PATH = Path(__file__).with_name("finite_element_material_transfer.metal")


class FiniteElementMaterialMetal:
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
    def _validate_batch(
        controls: torch.Tensor,
        lengths: torch.Tensor,
        color_front: torch.Tensor,
        color_back: torch.Tensor,
        modes: torch.Tensor,
    ) -> int:
        count = int(lengths.numel())
        if lengths.shape != (count,):
            raise ValueError("lengths must have shape [N]")
        expected_vec = (count, 3)
        if controls.shape != expected_vec or color_front.shape != expected_vec or color_back.shape != expected_vec:
            raise ValueError(f"controls and colors must have shape {expected_vec}")
        if modes.shape != (count,):
            raise ValueError(f"modes must have shape {(count,)}")
        tensors = (controls, lengths, color_front, color_back, modes)
        if any(value.device.type != "mps" for value in tensors):
            raise ValueError("Metal material inputs must all be MPS tensors")
        if any(value.dtype != torch.float32 for value in tensors[:-1]) or modes.dtype != torch.int32:
            raise ValueError("Metal material floats must be float32 and modes must be int32")
        if any(not value.is_contiguous() for value in tensors):
            raise ValueError("Metal material inputs must be contiguous")
        return count

    @staticmethod
    def _validate_cotangents(
        count: int,
        grad_tau: torch.Tensor,
        grad_beta: torch.Tensor,
        grad_m: torch.Tensor,
    ) -> None:
        if (
            grad_tau.shape != (count,)
            or grad_beta.shape != (count,)
            or grad_m.shape != (count, 3)
        ):
            raise ValueError("VJP cotangents have incompatible shape")
        cotangents = (grad_tau, grad_beta, grad_m)
        if any(value.dtype != torch.float32 for value in cotangents):
            raise ValueError("VJP cotangents must all be float32")
        if any(value.device.type != "mps" for value in cotangents):
            raise ValueError("VJP cotangents must all be MPS tensors")
        if not all(bool(torch.isfinite(value).all().item()) for value in cotangents):
            raise ValueError("VJP cotangents must be finite")

    @staticmethod
    def _raise_on_invalid(status: torch.Tensor) -> None:
        values = status.detach().cpu().to(torch.int64)
        invalid = torch.nonzero(
            (values & int(BranchStatus.INVALID_INPUT)) != 0,
            as_tuple=False,
        ).reshape(-1)
        if invalid.numel():
            rows = [int(value) for value in invalid.tolist()]
            raise FloatingPointError(
                f"Metal material evaluator rejected rows {rows}; "
                "an explicit host fallback is required"
            )

    def forward(
        self,
        controls: torch.Tensor,
        lengths: torch.Tensor,
        color_front: torch.Tensor,
        color_back: torch.Tensor,
        modes: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        count = self._validate_batch(controls, lengths, color_front, color_back, modes)
        tau = torch.empty_like(lengths)
        beta = torch.empty_like(lengths)
        m = torch.empty_like(color_front)
        density_bounds = torch.empty(
            (count, 2), dtype=controls.dtype, device=controls.device
        )
        status = torch.empty_like(modes)
        self.compile().worldfoam_material_forward(
            tau,
            beta,
            m,
            density_bounds,
            status,
            controls,
            lengths,
            color_front,
            color_back,
            modes,
            count,
        )
        self._raise_on_invalid(status)
        return {
            "tau": tau,
            "beta": beta,
            "m": m,
            "density_bounds": density_bounds,
            "status": status,
        }

    def vjp(
        self,
        controls: torch.Tensor,
        lengths: torch.Tensor,
        color_front: torch.Tensor,
        color_back: torch.Tensor,
        modes: torch.Tensor,
        grad_tau: torch.Tensor,
        grad_beta: torch.Tensor,
        grad_m: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        count = self._validate_batch(controls, lengths, color_front, color_back, modes)
        self._validate_cotangents(count, grad_tau, grad_beta, grad_m)
        grad_controls = torch.empty_like(controls)
        grad_color_front = torch.empty_like(color_front)
        grad_color_back = torch.empty_like(color_back)
        grad_length = torch.empty_like(lengths)
        status = torch.empty_like(modes)
        self.compile().worldfoam_material_vjp(
            grad_controls,
            grad_color_front,
            grad_color_back,
            grad_length,
            status,
            controls,
            lengths,
            color_front,
            color_back,
            modes,
            grad_tau.contiguous(),
            grad_beta.contiguous(),
            grad_m.contiguous(),
            count,
        )
        self._raise_on_invalid(status)
        return {
            "density_controls": grad_controls,
            "color_front": grad_color_front,
            "color_back": grad_color_back,
            "length": grad_length,
            "status": status,
        }

    @staticmethod
    def count_branches(status: torch.Tensor) -> dict[str, int]:
        return branch_status_counts(status)


__all__ = ["FiniteElementMaterialMetal", "SOURCE_PATH"]
