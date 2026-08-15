"""Closed-loop diagnostics for the ray-fiber optical connection.

Rendering an open camera ray is parallel transport.  This module uses the
word *holonomy* only for an actual closed ray-time rectangle and records the
orientation explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .affine import (
    AffineGenerator,
    AffineTransfer,
    AffineTransferTangent,
    generator_exponential,
    identity_transfer,
    inverse,
    scan,
)


@dataclass(frozen=True)
class RectangleHolonomyReport:
    """Exact group commutator and its small-area curvature comparison."""

    orientation: str
    holonomy: AffineTransfer
    predicted_curvature: AffineGenerator
    area_scaled_holonomy: AffineTransferTangent
    curvature_error: AffineTransferTangent


def generator_bracket(
    left: AffineGenerator,
    right: AffineGenerator,
) -> AffineGenerator:
    """Return ``[left,right]`` for ``R_{>0} semidirect R^3``."""

    left.validate("left generator")
    right.validate("right generator")
    if left.scalar.dtype != right.scalar.dtype:
        raise TypeError("bracket generators must share a dtype")
    if left.scalar.device != right.scalar.device:
        raise ValueError("bracket generators must share a device")
    left_scalar, right_scalar = torch.broadcast_tensors(
        left.scalar,
        right.scalar,
    )
    left_source, right_source = torch.broadcast_tensors(
        left.source,
        right.source,
    )
    return AffineGenerator(
        scalar=torch.zeros_like(left_scalar),
        source=(
            left_scalar.unsqueeze(-1) * right_source
            - right_scalar.unsqueeze(-1) * left_source
        ),
    )


def positive_rectangle_holonomy(
    *,
    depth_generator: AffineGenerator,
    time_generator: AffineGenerator,
    depth_extent: torch.Tensor,
    time_extent: torch.Tensor,
) -> RectangleHolonomyReport:
    """Evaluate ``exp(dt A_t) exp(dz A_z) exp(-dt A_t) exp(-dz A_z)``.

    With this declared orientation, the leading small-area term is
    ``[A_t,A_z] dt dz``.  Reversing the loop reverses that sign.
    """

    depth_generator.validate("depth generator")
    time_generator.validate("time generator")
    if depth_extent.ndim != 0 or time_extent.ndim != 0:
        raise ValueError("rectangle extents must be scalar tensors")
    if depth_extent.dtype != depth_generator.scalar.dtype or time_extent.dtype != depth_extent.dtype:
        raise TypeError("rectangle tensors and generators must share a dtype")
    if depth_extent.device != depth_generator.scalar.device or time_extent.device != depth_extent.device:
        raise ValueError("rectangle tensors and generators must share a device")
    if not bool((depth_extent > 0.0) & (time_extent > 0.0)):
        raise ValueError("rectangle extents must be positive")

    depth_step = generator_exponential(depth_generator, depth_extent)
    time_step = generator_exponential(time_generator, time_extent)
    holonomy = scan(
        (
            time_step,
            depth_step,
            inverse(time_step),
            inverse(depth_step),
        )
    )
    identity = identity_transfer(depth_extent)
    area = depth_extent * time_extent
    scaled = AffineTransferTangent(
        beta=(holonomy.beta - identity.beta) / area,
        moment=(holonomy.moment - identity.moment) / area,
    )
    curvature = generator_bracket(time_generator, depth_generator)
    return RectangleHolonomyReport(
        orientation="+t,+z,-t,-z; leading [A_t,A_z] dt dz",
        holonomy=holonomy,
        predicted_curvature=curvature,
        area_scaled_holonomy=scaled,
        curvature_error=AffineTransferTangent(
            beta=scaled.beta - curvature.scalar,
            moment=scaled.moment - curvature.source,
        ),
    )


__all__ = [
    "RectangleHolonomyReport",
    "generator_bracket",
    "positive_rectangle_holonomy",
]
