"""Canonical P0 WorldFoam material decode, encode, and manual VJP.

The production-adjacent material trainer and the fixed-site paper lifecycle
must use exactly the same physical parameterization.  This module is kept
stateless and deliberately separate from optimizer, checkpoint, native-token,
and replay lifecycles so those systems cannot grow competing softplus or
sigmoid conventions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class WorldFoamMaterialParameterization:
    """Physical material decode and its exact manual parameter-space VJP."""

    density_beta: float = 1.0
    density_threshold: float = 20.0
    minimum_density: float = 0.0

    def assert_valid(self) -> None:
        if (
            not math.isfinite(self.density_beta)
            or self.density_beta <= 0.0
            or not math.isfinite(self.density_threshold)
            or self.density_threshold <= 0.0
            or not math.isfinite(self.minimum_density)
            or self.minimum_density < 0.0
        ):
            raise ValueError("softplus density parameters must be finite and physical")

    @torch.no_grad()
    def decode_density_(
        self,
        destination: torch.Tensor,
        raw_density: torch.Tensor,
    ) -> None:
        self.assert_valid()
        destination.copy_(
            F.softplus(
                raw_density,
                beta=self.density_beta,
                threshold=self.density_threshold,
            )
        )
        destination.add_(self.minimum_density)

    @torch.no_grad()
    def encode_density(
        self,
        physical_density: torch.Tensor,
    ) -> torch.Tensor:
        """Invert the declared thresholded softplus without changing tiny seeds.

        PyTorch's thresholded softplus is linear when ``beta * raw`` exceeds
        ``threshold``.  Below that branch, the stable inverse is

        ``y + log(1 - exp(-beta*y)) / beta``.

        No epsilon clamp is allowed: an unrepresentable positive gap fails
        closed instead of silently becoming a different physical density.
        """

        self.assert_valid()
        density_above_minimum = physical_density - self.minimum_density
        if bool(torch.any(density_above_minimum <= 0.0).item()):
            raise ValueError("physical density must exceed minimum_density")
        beta_density = density_above_minimum * self.density_beta
        nonlinear_inverse = density_above_minimum + (
            torch.log(-torch.expm1(-beta_density)) / self.density_beta
        )
        result = torch.where(
            beta_density > self.density_threshold,
            density_above_minimum,
            nonlinear_inverse,
        )
        if not bool(torch.isfinite(result).all().item()):
            raise ValueError(
                "physical density gap is too small for a finite raw softplus value"
            )
        return result.contiguous()

    @torch.no_grad()
    def decode_color_(
        self,
        destination: torch.Tensor,
        raw_color: torch.Tensor,
    ) -> None:
        torch.sigmoid(raw_color, out=destination)

    @torch.no_grad()
    def density_vjp_(
        self,
        destination: torch.Tensor,
        raw_density: torch.Tensor,
        grad_density: torch.Tensor,
    ) -> None:
        """Apply the exact derivative of PyTorch's thresholded softplus."""

        self.assert_valid()
        destination.copy_(raw_density).mul_(self.density_beta)
        linear_branch = destination > self.density_threshold
        destination.sigmoid_()
        destination.masked_fill_(linear_branch, 1.0)
        destination.mul_(grad_density)

    @torch.no_grad()
    def color_vjp_(
        self,
        destination: torch.Tensor,
        physical_color: torch.Tensor,
        grad_color: torch.Tensor,
    ) -> None:
        """Apply ``sigmoid(raw) * (1-sigmoid(raw))`` without a tape."""

        destination.copy_(physical_color).neg_().add_(1.0)
        destination.mul_(physical_color).mul_(grad_color)


__all__ = ["WorldFoamMaterialParameterization"]
