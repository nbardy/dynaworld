"""CPU proof oracle for translated optical-depth measures.

For a P0 owner word, segment ``r`` occupies the cumulative optical-depth
interval ``[K[r-1], K[r])`` and carries constant RGB density ``color[r]``.
Concatenation translates the rear word by the front word's total optical
depth.  The affine transfer ``(beta, m)`` is the exponentially weighted
Laplace image of that order-explicit object.

This is deliberately an ``O(R^2)`` certificate oracle.  It materializes only
small CPU proof fixtures, is not imported by a trainer or shader, and is not a
proposed replacement for the four-scalar runtime transfer.  The formulation
is a project-level mathematical lens; this module makes no literature-novelty
claim.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

DTYPE = torch.float64


@dataclass(frozen=True)
class TranslatedOpticalDepthMeasure:
    """Piecewise-constant vector measure encoded by ordered interval widths."""

    optical_depths: torch.Tensor
    colors: torch.Tensor

    @property
    def run_count(self) -> int:
        return int(self.optical_depths.numel())

    @property
    def channel_count(self) -> int:
        return int(self.colors.shape[1])

    @property
    def kappa(self) -> torch.Tensor:
        return self.optical_depths.sum()

    def support_intervals(self) -> torch.Tensor:
        """Return the ordered ``[start,end]`` support intervals in opacity space."""

        if self.run_count == 0:
            return torch.empty((0, 2), dtype=DTYPE)
        ends = torch.cumsum(self.optical_depths, dim=0)
        starts = torch.cat((torch.zeros(1, dtype=DTYPE), ends[:-1]))
        return torch.stack((starts, ends), dim=1)


@dataclass(frozen=True)
class AffineTransfer:
    beta: torch.Tensor
    moment: torch.Tensor

    def as_tensor(self) -> torch.Tensor:
        return torch.cat((self.beta.reshape(1), self.moment))


@dataclass(frozen=True)
class P0WordVJP:
    density: torch.Tensor
    color: torch.Tensor
    length: torch.Tensor


@dataclass(frozen=True)
class LaplaceTransferErrorBounds:
    """Norm bounds induced by exponentially weighted total variation."""

    beta_absolute_error_bound: torch.Tensor
    moment_l2_error_bound: torch.Tensor


def make_translated_measure(
    optical_depths: torch.Tensor,
    colors: torch.Tensor,
) -> TranslatedOpticalDepthMeasure:
    """Build a finite CPU proof object; zero-width intervals are retained."""

    depths = torch.as_tensor(optical_depths, dtype=DTYPE)
    color_values = torch.as_tensor(colors, dtype=DTYPE)
    if depths.device.type != "cpu" or color_values.device.type != "cpu":
        raise ValueError("translated-measure oracle accepts CPU tensors only")
    if depths.ndim != 1:
        raise ValueError("optical_depths must have shape [R]")
    if color_values.ndim != 2 or color_values.shape[0] != depths.numel():
        raise ValueError("colors must have shape [R,C]")
    if color_values.shape[1] < 1:
        raise ValueError("colors must contain at least one channel")
    if not bool(torch.isfinite(depths).all().detach()) or not bool(torch.isfinite(color_values).all().detach()):
        raise ValueError("translated-measure inputs must be finite")
    if bool((depths < 0.0).any().detach()):
        raise ValueError("optical-depth widths must be nonnegative")
    return TranslatedOpticalDepthMeasure(depths, color_values)


def concatenate_measures(
    front: TranslatedOpticalDepthMeasure,
    rear: TranslatedOpticalDepthMeasure,
) -> TranslatedOpticalDepthMeasure:
    """Semidirect concatenation; rear support is shifted by ``front.kappa``."""

    if front.channel_count != rear.channel_count:
        raise ValueError("front and rear measures must have the same channel count")
    return make_translated_measure(
        torch.cat((front.optical_depths, rear.optical_depths)),
        torch.cat((front.colors, rear.colors), dim=0),
    )


def laplace_transfer(measure: TranslatedOpticalDepthMeasure) -> AffineTransfer:
    """Map the order-explicit proof object to its sufficient affine quotient."""

    intervals = measure.support_intervals()
    if measure.run_count == 0:
        return AffineTransfer(torch.ones((), dtype=DTYPE), torch.zeros(measure.channel_count, dtype=DTYPE))
    prefix_beta = torch.exp(-intervals[:, 0])
    interval_mass = prefix_beta * (-torch.expm1(-measure.optical_depths))
    return AffineTransfer(
        beta=torch.exp(-measure.kappa),
        moment=(interval_mass[:, None] * measure.colors).sum(dim=0),
    )


def laplace_tangent(
    measure: TranslatedOpticalDepthMeasure,
    optical_depth_tangent: torch.Tensor,
    color_tangent: torch.Tensor,
) -> AffineTransfer:
    """Apply the distributional tangent, including all boundary masses.

    At a zero-width interval this is a directional derivative only.  A
    physically feasible one-sided direction has nonnegative width tangent at
    that active boundary; topology-changing two-sided derivatives are not
    claimed.
    """

    depth_dot, color_dot = _validate_measure_tangent(
        measure,
        optical_depth_tangent,
        color_tangent,
    )
    if measure.run_count == 0:
        return AffineTransfer(torch.zeros((), dtype=DTYPE), torch.zeros(measure.channel_count, dtype=DTYPE))

    intervals = measure.support_intervals()
    boundaries = intervals[:, 1]
    boundary_dot = torch.cumsum(depth_dot, dim=0)
    interval_mass = torch.exp(-intervals[:, 0]) * (-torch.expm1(-measure.optical_depths))
    continuous_color_part = (interval_mass[:, None] * color_dot).sum(dim=0)
    internal_boundary_part = (
        (measure.colors[:-1] - measure.colors[1:]) * (torch.exp(-boundaries[:-1]) * boundary_dot[:-1])[:, None]
    ).sum(dim=0)
    terminal_boundary_part = measure.colors[-1] * torch.exp(-measure.kappa) * boundary_dot[-1]
    transfer = laplace_transfer(measure)
    return AffineTransfer(
        beta=-transfer.beta * boundary_dot[-1],
        moment=continuous_color_part + internal_boundary_part + terminal_boundary_part,
    )


def weighted_total_variation_distance(
    first: TranslatedOpticalDepthMeasure,
    second: TranslatedOpticalDepthMeasure,
) -> torch.Tensor:
    """Exact weighted-TV distance for two piecewise-constant primal measures.

    Measures are extended by zero on ``[0,infinity)`` and RGB/vector density
    uses the Euclidean norm.  The proof-only implementation builds the union of
    all cumulative-opacity boundaries and is intentionally not a runtime path.
    """

    if first.channel_count != second.channel_count:
        raise ValueError("weighted-TV measures must have the same channel count")
    boundaries = torch.unique(
        torch.cat(
            (
                torch.zeros(1, dtype=DTYPE),
                first.support_intervals().reshape(-1),
                second.support_intervals().reshape(-1),
            )
        ),
        sorted=True,
    )
    if boundaries.numel() < 2:
        return torch.zeros((), dtype=DTYPE)
    starts = boundaries[:-1]
    ends = boundaries[1:]
    positive = ends > starts
    starts = starts[positive]
    ends = ends[positive]
    if starts.numel() == 0:
        return torch.zeros((), dtype=DTYPE)
    midpoints = 0.5 * (starts + ends)
    density_difference = _measure_density_at(first, midpoints) - _measure_density_at(
        second,
        midpoints,
    )
    exponential_mass = torch.exp(-starts) * (
        -torch.expm1(-(ends - starts))
    )
    return torch.sum(
        exponential_mass
        * torch.linalg.vector_norm(density_difference, dim=1)
    )


def laplace_transfer_error_bounds(
    first: TranslatedOpticalDepthMeasure,
    second: TranslatedOpticalDepthMeasure,
) -> LaplaceTransferErrorBounds:
    """Return the paper's attenuation and affine-moment certificate bounds."""

    if first.channel_count != second.channel_count:
        raise ValueError("transfer-error measures must have the same channel count")
    return LaplaceTransferErrorBounds(
        beta_absolute_error_bound=(
            torch.exp(-torch.minimum(first.kappa, second.kappa))
            * torch.abs(first.kappa - second.kappa)
        ),
        moment_l2_error_bound=weighted_total_variation_distance(first, second),
    )


def laplace_tangent_weighted_variation_upper_bound(
    measure: TranslatedOpticalDepthMeasure,
    optical_depth_tangent: torch.Tensor,
    color_tangent: torch.Tensor,
) -> torch.Tensor:
    """Bound the Laplace moment tangent by weighted vector-measure variation.

    Coincident zero-width boundaries are kept as separate atomic terms, so the
    result may overestimate their cancellable variation while remaining sound.
    """

    depth_dot, color_dot = _validate_measure_tangent(
        measure,
        optical_depth_tangent,
        color_tangent,
    )
    if measure.run_count == 0:
        return torch.zeros((), dtype=DTYPE)
    intervals = measure.support_intervals()
    boundaries = intervals[:, 1]
    boundary_dot = torch.cumsum(depth_dot, dim=0)
    interval_mass = torch.exp(-intervals[:, 0]) * (
        -torch.expm1(-measure.optical_depths)
    )
    continuous = torch.sum(
        interval_mass * torch.linalg.vector_norm(color_dot, dim=1)
    )
    internal = torch.sum(
        torch.exp(-boundaries[:-1])
        * torch.abs(boundary_dot[:-1])
        * torch.linalg.vector_norm(
            measure.colors[:-1] - measure.colors[1:],
            dim=1,
        )
    )
    terminal = (
        torch.exp(-measure.kappa)
        * torch.abs(boundary_dot[-1])
        * torch.linalg.vector_norm(measure.colors[-1])
    )
    return continuous + internal + terminal


def opacity_tail_primal_error_bound(
    front: TranslatedOpticalDepthMeasure,
    rear: TranslatedOpticalDepthMeasure,
    background: torch.Tensor,
) -> torch.Tensor:
    """Bound the rendered-color error from replacing ``front+rear`` by front."""

    background_f64 = _validate_background(
        background,
        channel_count=front.channel_count,
        name="background",
    )
    if rear.channel_count != front.channel_count:
        raise ValueError("front and rear measures must have the same channel count")
    rear_color_bound = (
        torch.max(torch.linalg.vector_norm(rear.colors, dim=1))
        if rear.run_count
        else torch.zeros((), dtype=DTYPE)
    )
    front_transfer = laplace_transfer(front)
    rear_transfer = laplace_transfer(rear)
    rear_alpha = -torch.expm1(-rear.kappa)
    return (
        front_transfer.beta
        * rear_alpha
        * (rear_color_bound + torch.linalg.vector_norm(background_f64))
    )


def opacity_tail_directional_error_bound(
    front: TranslatedOpticalDepthMeasure,
    rear: TranslatedOpticalDepthMeasure,
    background: torch.Tensor,
    *,
    front_optical_depth_tangent: torch.Tensor,
    rear_optical_depth_tangent: torch.Tensor,
    rear_color_tangent: torch.Tensor,
    background_tangent: torch.Tensor | None = None,
) -> torch.Tensor:
    """Bound one fixed-split directional derivative of the discarded tail."""

    if rear.channel_count != front.channel_count:
        raise ValueError("front and rear measures must have the same channel count")
    background_f64 = _validate_background(
        background,
        channel_count=front.channel_count,
        name="background",
    )
    background_dot = _validate_background(
        (
            torch.zeros_like(background_f64)
            if background_tangent is None
            else background_tangent
        ),
        channel_count=front.channel_count,
        name="background_tangent",
    )
    front_depth_dot, _ = _validate_measure_tangent(
        front,
        front_optical_depth_tangent,
        torch.zeros_like(front.colors),
    )
    rear_depth_dot, rear_color_dot = _validate_measure_tangent(
        rear,
        rear_optical_depth_tangent,
        rear_color_tangent,
    )
    front_transfer = laplace_transfer(front)
    rear_transfer = laplace_transfer(rear)
    rear_color_bound = (
        torch.max(torch.linalg.vector_norm(rear.colors, dim=1))
        if rear.run_count
        else torch.zeros((), dtype=DTYPE)
    )
    rear_alpha = -torch.expm1(-rear.kappa)
    return front_transfer.beta * (
        laplace_tangent_weighted_variation_upper_bound(
            rear,
            rear_depth_dot,
            rear_color_dot,
        )
        + rear_transfer.beta
        * torch.abs(rear_depth_dot.sum())
        * torch.linalg.vector_norm(background_f64)
        + rear_alpha * torch.linalg.vector_norm(background_dot)
        + torch.abs(front_depth_dot.sum())
        * rear_alpha
        * (rear_color_bound + torch.linalg.vector_norm(background_f64))
    )


def p0_word_transfer(
    density: torch.Tensor,
    color: torch.Tensor,
    length: torch.Tensor,
) -> AffineTransfer:
    """Evaluate a constant-material word through the measure certificate."""

    density_f64, length_f64, color_f64 = _validate_p0_word(density, color, length)
    return laplace_transfer(make_translated_measure(density_f64 * length_f64, color_f64))


def p0_word_vjp(
    density: torch.Tensor,
    color: torch.Tensor,
    length: torch.Tensor,
    *,
    grad_beta: torch.Tensor | float,
    grad_moment: torch.Tensor,
) -> P0WordVJP:
    """Proof-only VJP obtained by probing the distributional tangent.

    The basis loop is intentionally quadratic in word length.  Production uses
    the constant-state prefix identity in ``compiled_transfer_adjoint.py``.
    """

    density_f64, length_f64, color_f64 = _validate_p0_word(density, color, length)
    beta_bar = torch.as_tensor(grad_beta, dtype=DTYPE)
    moment_bar = torch.as_tensor(grad_moment, dtype=DTYPE)
    if beta_bar.device.type != "cpu" or moment_bar.device.type != "cpu":
        raise ValueError("translated-measure oracle accepts CPU tensors only")
    if beta_bar.ndim != 0 or moment_bar.shape != (color_f64.shape[1],):
        raise ValueError("grad_beta must be scalar and grad_moment must have shape [C]")
    if not bool(torch.isfinite(beta_bar).detach()) or not bool(torch.isfinite(moment_bar).all().detach()):
        raise ValueError("transfer cotangents must be finite")

    measure = make_translated_measure(density_f64 * length_f64, color_f64)
    tau_bar_rows = []
    for run_id in range(measure.run_count):
        basis = torch.zeros_like(measure.optical_depths)
        basis[run_id] = 1.0
        tangent = laplace_tangent(measure, basis, torch.zeros_like(color_f64))
        tau_bar_rows.append(beta_bar * tangent.beta + torch.dot(moment_bar, tangent.moment))
    tau_bar = torch.stack(tau_bar_rows) if tau_bar_rows else torch.empty(0, dtype=DTYPE)

    intervals = measure.support_intervals()
    interval_mass = (
        torch.exp(-intervals[:, 0]) * (-torch.expm1(-measure.optical_depths))
        if measure.run_count
        else torch.empty(0, dtype=DTYPE)
    )
    return P0WordVJP(
        density=length_f64 * tau_bar,
        color=interval_mass[:, None] * moment_bar,
        length=density_f64 * tau_bar,
    )


def two_segment_commutator_formula(
    front_optical_depth: torch.Tensor | float,
    front_color: torch.Tensor,
    rear_optical_depth: torch.Tensor | float,
    rear_color: torch.Tensor,
) -> torch.Tensor:
    """Exact ``m(front,rear) - m(rear,front)`` for constant-color segments."""

    tau_front = torch.as_tensor(front_optical_depth, dtype=DTYPE)
    tau_rear = torch.as_tensor(rear_optical_depth, dtype=DTYPE)
    color_front = torch.as_tensor(front_color, dtype=DTYPE)
    color_rear = torch.as_tensor(rear_color, dtype=DTYPE)
    if any(value.device.type != "cpu" for value in (tau_front, tau_rear, color_front, color_rear)):
        raise ValueError("translated-measure oracle accepts CPU tensors only")
    if tau_front.ndim or tau_rear.ndim or color_front.ndim != 1 or color_front.shape != color_rear.shape:
        raise ValueError("commutator expects scalar optical depths and equal color vectors")
    if bool((torch.stack((tau_front, tau_rear)) < 0.0).any().detach()):
        raise ValueError("optical depths must be nonnegative")
    if not bool(torch.isfinite(torch.cat((tau_front.reshape(1), tau_rear.reshape(1), color_front, color_rear))).all()):
        raise ValueError("commutator inputs must be finite")
    alpha_front = -torch.expm1(-tau_front)
    alpha_rear = -torch.expm1(-tau_rear)
    return alpha_front * alpha_rear * (color_front - color_rear)


def _validate_p0_word(
    density: torch.Tensor,
    color: torch.Tensor,
    length: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    density_f64 = torch.as_tensor(density, dtype=DTYPE)
    length_f64 = torch.as_tensor(length, dtype=DTYPE)
    color_f64 = torch.as_tensor(color, dtype=DTYPE)
    if density_f64.device.type != "cpu" or length_f64.device.type != "cpu" or color_f64.device.type != "cpu":
        raise ValueError("translated-measure oracle accepts CPU tensors only")
    if density_f64.ndim != 1 or length_f64.shape != density_f64.shape:
        raise ValueError("density and length must have shape [R]")
    if color_f64.ndim != 2 or color_f64.shape[0] != density_f64.numel():
        raise ValueError("color must have shape [R,C]")
    if not all(bool(torch.isfinite(value).all().detach()) for value in (density_f64, length_f64, color_f64)):
        raise ValueError("P0 word inputs must be finite")
    if bool((density_f64 < 0.0).any().detach()) or bool((length_f64 < 0.0).any().detach()):
        raise ValueError("density and physical length must be nonnegative")
    return density_f64, length_f64, color_f64


def _measure_density_at(
    measure: TranslatedOpticalDepthMeasure,
    points: torch.Tensor,
) -> torch.Tensor:
    result = torch.zeros((points.numel(), measure.channel_count), dtype=DTYPE)
    if measure.run_count == 0 or points.numel() == 0:
        return result
    indices = torch.searchsorted(
        torch.cumsum(measure.optical_depths, dim=0),
        points,
        right=False,
    )
    valid = indices < measure.run_count
    result[valid] = measure.colors[indices[valid]]
    return result


def _validate_measure_tangent(
    measure: TranslatedOpticalDepthMeasure,
    optical_depth_tangent: torch.Tensor,
    color_tangent: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    depth_dot = torch.as_tensor(optical_depth_tangent, dtype=DTYPE)
    color_dot = torch.as_tensor(color_tangent, dtype=DTYPE)
    if depth_dot.shape != measure.optical_depths.shape:
        raise ValueError("optical_depth_tangent must have shape [R]")
    if color_dot.shape != measure.colors.shape:
        raise ValueError("color_tangent must have shape [R,C]")
    if depth_dot.device.type != "cpu" or color_dot.device.type != "cpu":
        raise ValueError("translated-measure oracle accepts CPU tensors only")
    if not bool(torch.isfinite(depth_dot).all().detach()) or not bool(
        torch.isfinite(color_dot).all().detach()
    ):
        raise ValueError("translated-measure tangents must be finite")
    return depth_dot, color_dot


def _validate_background(
    value: torch.Tensor,
    *,
    channel_count: int,
    name: str,
) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=DTYPE)
    if result.device.type != "cpu" or result.shape != (channel_count,):
        raise ValueError(f"{name} must be a CPU vector with shape [C]")
    if not bool(torch.isfinite(result).all().detach()):
        raise ValueError(f"{name} must be finite")
    return result
