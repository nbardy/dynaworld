"""Fixed-segment material laws for the Gaussian-FEM WorldFoam experiment.

This module deliberately stops at the material boundary:

    one physical ray segment -> TransferElement(beta, m)

It does not traverse cells, sort visibility, or render an image.  All six
material modes use three density-control slots with the following meanings:

* M0/M1: ``(sigma, unused, unused)``.
* M2: positive Bernstein-P1 ``(sigma_0, sigma_1, unused)``.
* M3: positive Bernstein-P2 ``(sigma_0, sigma_1, sigma_2)``.
* M4: negative-log P1 ``q(x) = b*x + c``, slots ``(b, c, unused)``.
* M5: convex negative-log P2 ``q(x) = a*x*x + b*x + c``, slots
  ``(a, b, c)`` with ``a >= 0``.

Here ``x`` is normalized segment distance in ``[0, 1]`` and extinction is
``sigma(x) = exp(-q(x))`` in M4/M5.  Thus physical length is carried exactly
once as the ray-fiber Jacobian.

M0 and M2--M5 use constant RGB.  M1 alone uses affine RGB between
``color_front`` and ``color_back`` and integrates it against constant
extinction.  The VJP is explicit; autograd is used only by tests as an
independent oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, IntFlag
import math
from typing import Iterable

import torch


class MaterialMode(IntEnum):
    M0_P0_CONSTANT = 0
    M1_P0_AFFINE_RGB = 1
    M2_POSITIVE_BERNSTEIN_P1 = 2
    M3_POSITIVE_BERNSTEIN_P2 = 3
    M4_LOG_P1 = 4
    M5_CONVEX_LOG_P2 = 5


class BranchStatus(IntFlag):
    """Bit flags reported by both the CPU reference and Metal microkernel."""

    DIRECT = 0
    SMALL_TAU_SERIES = 1 << 0
    LOG_LINEAR_SERIES = 1 << 1
    LOG_QUADRATIC_SERIES = 1 << 2
    LOG_QUADRATIC_ERF = 1 << 3
    LOG_QUADRATIC_TAIL = 1 << 4
    INVALID_INPUT = 1 << 30


@dataclass(frozen=True)
class TransferElement:
    beta: torch.Tensor
    m: torch.Tensor


@dataclass(frozen=True)
class MaterialTransfer:
    element: TransferElement
    tau: torch.Tensor
    density_bounds: torch.Tensor
    branch_status: BranchStatus


@dataclass(frozen=True)
class MaterialVJP:
    density_controls: torch.Tensor
    color_front: torch.Tensor
    color_back: torch.Tensor
    length: torch.Tensor
    branch_status: BranchStatus


_SMALL_TAU = 1.0e-4
_LINEAR_SERIES_LIMIT = 0.75
_QUADRATIC_SERIES_LIMIT = 2.0e-2


def _require_segment_inputs(
    mode: MaterialMode,
    density_controls: torch.Tensor,
    length: torch.Tensor,
    color_front: torch.Tensor,
    color_back: torch.Tensor,
) -> None:
    if density_controls.shape != (3,):
        raise ValueError(f"density_controls must have shape (3,), got {tuple(density_controls.shape)}")
    if color_front.shape != (3,) or color_back.shape != (3,):
        raise ValueError("color_front and color_back must each have shape (3,)")
    if length.ndim != 0:
        raise ValueError("length must be a zero-dimensional scalar tensor")
    tensors = (density_controls, length, color_front, color_back)
    if not all(bool(torch.isfinite(value).all().detach()) for value in tensors):
        raise ValueError("segment inputs must be finite")
    if bool((length < 0).detach()):
        raise ValueError("physical segment length must be nonnegative")
    if mode in (MaterialMode.M0_P0_CONSTANT, MaterialMode.M1_P0_AFFINE_RGB):
        positive = density_controls[:1]
    elif mode == MaterialMode.M2_POSITIVE_BERNSTEIN_P1:
        positive = density_controls[:2]
    elif mode == MaterialMode.M3_POSITIVE_BERNSTEIN_P2:
        positive = density_controls
    else:
        positive = density_controls[:0]
    if positive.numel() and bool((positive < 0).any().detach()):
        raise ValueError("direct-density Bernstein controls must be nonnegative")
    if mode == MaterialMode.M5_CONVEX_LOG_P2 and bool((density_controls[0] < 0).detach()):
        raise ValueError("M5 requires convex negative-log density: quadratic coefficient a >= 0")


def _linear_moment(order: int, b: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Return integral_0^1 x**order exp(-b*x) dx.

    A convergent Taylor series avoids cancellation around b=0.  Away from
    zero, integration-by-parts recurrence is cheaper and stable for the modest
    coefficient range used by this isolated reference.
    """

    b_value = abs(float(b.detach()))
    if b_value < _LINEAR_SERIES_LIMIT:
        term = torch.ones_like(b)
        result = term / float(order + 1)
        for k in range(1, 32):
            term = term * (-b) / float(k)
            result = result + term / float(order + k + 1)
        return result, True

    exp_neg_b = torch.exp(-b)
    moment = (1.0 - exp_neg_b) / b
    for n in range(1, order + 1):
        moment = (float(n) * moment - exp_neg_b) / b
    return moment, False


def _log_linear_moments(
    b: torch.Tensor,
    c: torch.Tensor,
    maximum_order: int,
) -> tuple[list[torch.Tensor], bool]:
    """Return ``integral x**n exp(-(b*x+c)) dx`` without split overflow.

    Computing an unscaled ``exp(-b)`` and multiplying by ``exp(-c)`` later is
    invalid in float32: individually overflowing factors can have a benign
    product.  This routine works directly with the two endpoint densities.
    """

    if maximum_order < 0:
        raise ValueError("maximum_order must be nonnegative")
    if abs(float(b.detach())) < _LINEAR_SERIES_LIMIT:
        scale = torch.exp(-c)
        moments = []
        for order in range(maximum_order + 1):
            value, _ = _linear_moment(order, b)
            moments.append(scale * value)
        return moments, True

    density0 = torch.exp(-c)
    density1 = torch.exp(-(b + c))
    moments = [(density0 - density1) / b]
    for order in range(1, maximum_order + 1):
        moments.append((float(order) * moments[-1] - density1) / b)
    return moments, False


def _log_quadratic_moments(
    controls: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, BranchStatus]:
    """Return I_n = integral x**n exp(-(a*x*x+b*x+c)) dx, n=0,1,2."""

    a, b, c = controls.unbind()
    a_value = float(a.detach())

    if a_value < _QUADRATIC_SERIES_LIMIT:
        # exp(-a x^2) power series.  At a<0.02, eight terms are below
        # float64 roundoff; moments remain explicit analytic sums.
        linear_moments, used_linear_series = _log_linear_moments(b, c, 18)
        sums = [torch.zeros_like(a) for _ in range(3)]
        factor = torch.ones_like(a)
        for k in range(9):
            for n in range(3):
                sums[n] = sums[n] + factor * linear_moments[n + 2 * k]
            factor = factor * (-a) / float(k + 1)
        status = BranchStatus.LOG_QUADRATIC_SERIES
        if used_linear_series:
            status |= BranchStatus.LOG_LINEAR_SERIES
        return sums[0], sums[1], sums[2], status

    sqrt_a = torch.sqrt(a)
    u0 = b / (2.0 * sqrt_a)
    u1 = sqrt_a + u0
    exponent = -c + b.square() / (4.0 * a)
    u0_value = float(u0.detach())
    u1_value = float(u1.detach())
    exponent_value = float(exponent.detach())
    straddles_zero = u0_value <= 0.0 <= u1_value
    # The local Metal erf approximation has a small *absolute* error.  It is
    # therefore safe for a sign-straddling difference, whose magnitude is not
    # a same-sign tail subtraction, but not merely because both arguments are
    # below an arbitrary absolute threshold.  Same-sign arguments always use
    # the scaled-tail form below.
    safe_erf = straddles_zero and -80.0 <= exponent_value <= 80.0
    if safe_erf:
        prefactor = torch.exp(exponent) * (math.sqrt(math.pi) / (2.0 * sqrt_a))
        i0 = prefactor * (torch.erf(u1) - torch.erf(u0))
        f0 = torch.exp(-c)
        f1 = torch.exp(-(a + b + c))
        i1 = (f0 - f1 - b * i0) / (2.0 * a)
        i2 = (i0 - b * i1 - f1) / (2.0 * a)
        # Approximate-erf Metal can lose enough tail precision to make a
        # moment negative. Use the stable scaled-tail branch in that regime.
        if all(float(value.detach()) >= 0.0 and math.isfinite(float(value.detach())) for value in (i0, i1, i2)):
            return i0, i1, i2, BranchStatus.LOG_QUADRATIC_ERF

    prefactor = math.sqrt(math.pi) / (2.0 * sqrt_a)
    q0 = c
    q1 = a + b + c
    if u0_value > 0.0:
        i0 = prefactor * (
            torch.exp(-q0) * torch.special.erfcx(u0)
            - torch.exp(-q1) * torch.special.erfcx(u1)
        )
    elif u1_value < 0.0:
        i0 = prefactor * (
            torch.exp(-q1) * torch.special.erfcx(-u1)
            - torch.exp(-q0) * torch.special.erfcx(-u0)
        )
    else:
        raise FloatingPointError(
            "M5 straddling-peak exponent is outside the float32-safe domain"
        )
    f0 = torch.exp(-q0)
    f1 = torch.exp(-q1)
    i1 = (f0 - f1 - b * i0) / (2.0 * a)
    i2 = (i0 - b * i1 - f1) / (2.0 * a)
    moments = (i0, i1, i2)
    if not all(
        math.isfinite(float(value.detach())) and float(value.detach()) >= 0.0
        for value in moments
    ):
        raise FloatingPointError(
            "M5 scaled-tail moments left the accepted numerical domain"
        )
    return i0, i1, i2, BranchStatus.LOG_QUADRATIC_TAIL


def _tau_and_derivatives(
    mode: MaterialMode,
    density_controls: torch.Tensor,
    length: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, BranchStatus]:
    """Return tau, d(tau)/d(controls), d(tau)/d(length), status."""

    zeros = torch.zeros_like(density_controls)
    if mode in (MaterialMode.M0_P0_CONSTANT, MaterialMode.M1_P0_AFFINE_RGB):
        average_sigma = density_controls[0]
        derivative = torch.stack((length, zeros[1], zeros[2]))
        return length * average_sigma, derivative, average_sigma, BranchStatus.DIRECT
    if mode == MaterialMode.M2_POSITIVE_BERNSTEIN_P1:
        average_sigma = 0.5 * (density_controls[0] + density_controls[1])
        derivative = torch.stack((0.5 * length, 0.5 * length, zeros[2]))
        return length * average_sigma, derivative, average_sigma, BranchStatus.DIRECT
    if mode == MaterialMode.M3_POSITIVE_BERNSTEIN_P2:
        average_sigma = density_controls.sum() / 3.0
        derivative = torch.ones_like(density_controls) * (length / 3.0)
        return length * average_sigma, derivative, average_sigma, BranchStatus.DIRECT
    if mode == MaterialMode.M4_LOG_P1:
        b, c = density_controls[:2]
        moments, used_series = _log_linear_moments(b, c, 1)
        i0, i1 = moments
        derivative = torch.stack((-length * i1, -length * i0, zeros[2]))
        status = BranchStatus.LOG_LINEAR_SERIES if used_series else BranchStatus.DIRECT
        return length * i0, derivative, i0, status
    if mode == MaterialMode.M5_CONVEX_LOG_P2:
        i0, i1, i2, status = _log_quadratic_moments(density_controls)
        derivative = -length * torch.stack((i2, i1, i0))
        return length * i0, derivative, i0, status
    raise ValueError(f"unknown material mode {mode}")


def _density_bounds(
    mode: MaterialMode,
    density_controls: torch.Tensor,
) -> torch.Tensor:
    """Return conservative ``[minimum, maximum]`` extinction on ``[0, 1]``."""

    if mode in (MaterialMode.M0_P0_CONSTANT, MaterialMode.M1_P0_AFFINE_RGB):
        return density_controls[:1].expand(2)
    if mode == MaterialMode.M2_POSITIVE_BERNSTEIN_P1:
        return torch.stack(
            (density_controls[:2].min(), density_controls[:2].max())
        )
    if mode == MaterialMode.M3_POSITIVE_BERNSTEIN_P2:
        # The Bernstein convex-hull property makes these certified bounds.
        # They need not be the tight range of the quadratic.
        return torch.stack((density_controls.min(), density_controls.max()))
    if mode == MaterialMode.M4_LOG_P1:
        b, c = density_controls[:2]
        endpoint_density = torch.stack((torch.exp(-c), torch.exp(-(b + c))))
        return torch.stack((endpoint_density.min(), endpoint_density.max()))
    if mode == MaterialMode.M5_CONVEX_LOG_P2:
        a, b, c = density_controls
        q0 = c
        q1 = a + b + c
        # A convex quadratic reaches its maximum at an endpoint and its
        # minimum at an endpoint or the clipped stationary point.  Avoid a
        # speculative division by zero in the exactly linear case.
        safe_a = torch.where(a > 0.0, a, torch.ones_like(a))
        stationary_x = torch.clamp(-b / (2.0 * safe_a), 0.0, 1.0)
        stationary_q = a * stationary_x.square() + b * stationary_x + c
        q_min = torch.minimum(torch.minimum(q0, q1), stationary_q)
        q_max = torch.maximum(q0, q1)
        return torch.stack((torch.exp(-q_max), torch.exp(-q_min)))
    raise ValueError(f"unknown material mode {mode}")


def _alpha_from_tau(tau: torch.Tensor) -> tuple[torch.Tensor, bool]:
    # torch.expm1 is the independent CPU reference for the local Metal series.
    return -torch.expm1(-tau), abs(float(tau.detach())) < _SMALL_TAU


def _affine_color_weights(
    tau: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Return front/back weights and their tau derivatives for M1."""

    tau_value = abs(float(tau.detach()))
    beta = torch.exp(-tau)
    if tau_value < _SMALL_TAU:
        w1 = torch.zeros_like(tau)
        dw1 = torch.zeros_like(tau)
        term = tau
        dterm = torch.ones_like(tau)
        # w1 = sum_k (-1)^k tau^(k+1)/(k! (k+2)).
        for k in range(12):
            coefficient = (-1.0 if k % 2 else 1.0) / (math.factorial(k) * (k + 2))
            w1 = w1 + coefficient * term
            dw1 = dw1 + coefficient * dterm
            term = term * tau
            dterm = float(k + 2) * term / tau if tau_value > 0.0 else torch.zeros_like(tau)
        alpha = -torch.expm1(-tau)
        dalpha = beta
        return alpha - w1, w1, dalpha - dw1, dw1, True

    numerator = 1.0 - (1.0 + tau) * beta
    w1 = numerator / tau
    dw1 = (tau.square() * beta - numerator) / tau.square()
    alpha = 1.0 - beta
    return alpha - w1, w1, beta - dw1, dw1, False


def evaluate_material_segment(
    mode: MaterialMode | int,
    density_controls: torch.Tensor,
    length: torch.Tensor,
    color_front: torch.Tensor,
    color_back: torch.Tensor | None = None,
) -> MaterialTransfer:
    """Evaluate one fixed physical segment.

    ``color_back`` is used only by M1 and defaults to ``color_front``.  Inputs
    may require gradients; validation does not detach the actual computation.
    """

    mode = MaterialMode(mode)
    color_back = color_front if color_back is None else color_back
    _require_segment_inputs(mode, density_controls, length, color_front, color_back)
    tau, _, _, status = _tau_and_derivatives(mode, density_controls, length)
    density_bounds = _density_bounds(mode, density_controls)
    beta = torch.exp(-tau)
    if mode == MaterialMode.M1_P0_AFFINE_RGB:
        w0, w1, _, _, used_series = _affine_color_weights(tau)
        m = w0 * color_front + w1 * color_back
    else:
        alpha, used_series = _alpha_from_tau(tau)
        m = alpha * color_front
    if used_series:
        status |= BranchStatus.SMALL_TAU_SERIES
    if not all(
        bool(torch.isfinite(value).all().detach())
        for value in (tau, beta, m, density_bounds)
    ):
        raise FloatingPointError("material evaluation overflowed its accepted numerical domain")
    return MaterialTransfer(
        TransferElement(beta=beta, m=m),
        tau=tau,
        density_bounds=density_bounds,
        branch_status=status,
    )


def material_segment_vjp(
    mode: MaterialMode | int,
    density_controls: torch.Tensor,
    length: torch.Tensor,
    color_front: torch.Tensor,
    color_back: torch.Tensor | None = None,
    *,
    grad_tau: torch.Tensor | float = 0.0,
    grad_beta: torch.Tensor | float = 0.0,
    grad_m: torch.Tensor | None = None,
) -> MaterialVJP:
    """Explicit VJP for ``(tau, beta, m)`` with fixed segment topology."""

    mode = MaterialMode(mode)
    color_is_aliased = color_back is None
    color_back = color_front if color_back is None else color_back
    _require_segment_inputs(mode, density_controls, length, color_front, color_back)
    tau, dtau_controls, dtau_length, status = _tau_and_derivatives(mode, density_controls, length)
    beta = torch.exp(-tau)
    grad_tau = torch.as_tensor(grad_tau, dtype=tau.dtype, device=tau.device)
    grad_beta = torch.as_tensor(grad_beta, dtype=tau.dtype, device=tau.device)
    grad_m = torch.zeros_like(color_front) if grad_m is None else grad_m

    if mode == MaterialMode.M1_P0_AFFINE_RGB:
        w0, w1, dw0, dw1, used_series = _affine_color_weights(tau)
        d_m_d_tau = dw0 * color_front + dw1 * color_back
        effective_tau = grad_tau - beta * grad_beta + torch.dot(grad_m, d_m_d_tau)
        grad_color_front = w0 * grad_m
        grad_color_back = w1 * grad_m
        if color_is_aliased:
            # ``color_back=None`` means both endpoint uses refer to the one
            # supplied tensor.  The VJP of that aliased input is the sum of
            # both endpoint contributions; there is no second input gradient.
            grad_color_front = grad_color_front + grad_color_back
            grad_color_back = torch.zeros_like(grad_color_back)
    else:
        alpha, used_series = _alpha_from_tau(tau)
        effective_tau = grad_tau - beta * grad_beta + beta * torch.dot(grad_m, color_front)
        grad_color_front = alpha * grad_m
        grad_color_back = torch.zeros_like(color_back)
    if used_series:
        status |= BranchStatus.SMALL_TAU_SERIES
    result = MaterialVJP(
        density_controls=effective_tau * dtau_controls,
        color_front=grad_color_front,
        color_back=grad_color_back,
        length=effective_tau * dtau_length,
        branch_status=status,
    )
    if not all(
        bool(torch.isfinite(value).all().detach())
        for value in (
            result.density_controls,
            result.color_front,
            result.color_back,
            result.length,
        )
    ):
        raise FloatingPointError("material VJP overflowed its accepted numerical domain")
    return result


def branch_status_counts(statuses: Iterable[int | BranchStatus] | torch.Tensor) -> dict[str, int]:
    """Count explicit numerical branches for benchmark/parity reports."""

    if isinstance(statuses, torch.Tensor):
        values = [int(value) for value in statuses.detach().cpu().reshape(-1).tolist()]
    else:
        values = [int(value) for value in statuses]
    result = {"total": len(values), "direct": sum(value == 0 for value in values)}
    for flag in BranchStatus:
        if flag == BranchStatus.DIRECT:
            continue
        result[flag.name.lower()] = sum(bool(value & int(flag)) for value in values)
    return result


__all__ = [
    "BranchStatus",
    "MaterialMode",
    "MaterialTransfer",
    "MaterialVJP",
    "TransferElement",
    "branch_status_counts",
    "evaluate_material_segment",
    "material_segment_vjp",
]
