"""Differentiable affine optical algebra in WorldFoam repository order.

An element ``T(beta, moment)`` acts on radiance behind it as

``q -> moment + beta * q``.

The repository scans near to far, so ``compose(front, back)`` means
``front(back(q))``.  All functions in this file preserve that order.  The
module is intentionally independent of the production WorldFoam code and
contains no callbacks, global caches, or device transfers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


RGB_CHANNELS = 3


@dataclass(frozen=True)
class AffineTransfer:
    """Affine-group element with ``beta [...]`` and ``moment [...,3]``."""

    beta: torch.Tensor
    moment: torch.Tensor

    def validate(self, name: str = "transfer") -> None:
        _require_scalar_vector_pair(name, self.beta, self.moment)

    def as_vector(self) -> torch.Tensor:
        """Return the four-coordinate representation ``[..., beta,m_rgb]``."""

        self.validate()
        return torch.cat((self.beta.unsqueeze(-1), self.moment), dim=-1)

    @staticmethod
    def from_vector(vector: torch.Tensor) -> "AffineTransfer":
        _require_four_vector("vector", vector)
        return AffineTransfer(beta=vector[..., 0], moment=vector[..., 1:])


@dataclass(frozen=True)
class AffineGenerator:
    """Lie-algebra element ``X(scalar, source)``.

    Its homogeneous matrix has ``scalar * I_3`` in the linear block and
    ``source`` in the affine column.  A physical P0 optical generator is
    ``X(-lambda, eta)``.
    """

    scalar: torch.Tensor
    source: torch.Tensor

    def validate(self, name: str = "generator") -> None:
        _require_scalar_vector_pair(name, self.scalar, self.source)

    def as_vector(self) -> torch.Tensor:
        self.validate()
        return torch.cat((self.scalar.unsqueeze(-1), self.source), dim=-1)

    @staticmethod
    def from_vector(vector: torch.Tensor) -> "AffineGenerator":
        _require_four_vector("vector", vector)
        return AffineGenerator(scalar=vector[..., 0], source=vector[..., 1:])


@dataclass(frozen=True)
class AffineTransferTangent:
    """Signed tangent ``(d beta, d moment)`` at an affine transfer."""

    beta: torch.Tensor
    moment: torch.Tensor

    def validate(self, name: str = "transfer tangent") -> None:
        _require_scalar_vector_pair(name, self.beta, self.moment)

    def as_vector(self) -> torch.Tensor:
        self.validate()
        return torch.cat((self.beta.unsqueeze(-1), self.moment), dim=-1)

    @staticmethod
    def from_vector(vector: torch.Tensor) -> "AffineTransferTangent":
        _require_four_vector("vector", vector)
        return AffineTransferTangent(
            beta=vector[..., 0],
            moment=vector[..., 1:],
        )


@dataclass(frozen=True)
class PhysicalConeReport:
    """Tensor-valued fail-closed report for bounded-RGB optical transfer."""

    maximum_violation: torch.Tensor
    beta_positive_violation: torch.Tensor
    beta_contraction_violation: torch.Tensor
    moment_lower_violation: torch.Tensor
    moment_upper_violation: torch.Tensor
    finite: torch.Tensor
    passed: torch.Tensor


@dataclass(frozen=True)
class AffineGroupReport:
    """Nonsingularity and conditioning report for a group-completion value."""

    minimum_beta: torch.Tensor
    maximum_inverse_beta: torch.Tensor
    homogeneous_condition_number: torch.Tensor
    finite: torch.Tensor
    passed: torch.Tensor


def identity_transfer(reference: torch.Tensor, *, batch_shape: tuple[int, ...] = ()) -> AffineTransfer:
    """Construct an identity on ``reference`` dtype/device without casting."""

    _require_float_tensor("reference", reference)
    return AffineTransfer(
        beta=torch.ones(batch_shape, dtype=reference.dtype, device=reference.device),
        moment=torch.zeros(
            batch_shape + (RGB_CHANNELS,),
            dtype=reference.dtype,
            device=reference.device,
        ),
    )


def zero_tangent(reference: torch.Tensor, *, batch_shape: tuple[int, ...] = ()) -> AffineTransferTangent:
    """Construct a zero affine tangent on ``reference`` dtype/device."""

    _require_float_tensor("reference", reference)
    return AffineTransferTangent(
        beta=torch.zeros(batch_shape, dtype=reference.dtype, device=reference.device),
        moment=torch.zeros(
            batch_shape + (RGB_CHANNELS,),
            dtype=reference.dtype,
            device=reference.device,
        ),
    )


def compose(front: AffineTransfer, back: AffineTransfer) -> AffineTransfer:
    """Return ``front o back`` in the executable near-to-far scan order."""

    front.validate("front")
    back.validate("back")
    _require_same_dtype_device("front", front.beta, "back", back.beta)
    front_beta, back_beta = torch.broadcast_tensors(front.beta, back.beta)
    front_moment, back_moment = torch.broadcast_tensors(
        front.moment,
        back.moment,
    )
    beta = front_beta * back_beta
    moment = front_moment + front_beta.unsqueeze(-1) * back_moment
    return AffineTransfer(beta=beta, moment=moment)


def scan(transfers: tuple[AffineTransfer, ...] | list[AffineTransfer]) -> AffineTransfer:
    """Compose a nonempty near-to-far transfer word."""

    if not transfers:
        raise ValueError("scan requires at least one transfer")
    result = transfers[0]
    result.validate("transfers[0]")
    for transfer in transfers[1:]:
        result = compose(result, transfer)
    return result


def inverse(transfer: AffineTransfer) -> AffineTransfer:
    """Return the affine-group inverse; no physical-cone claim is made."""

    transfer.validate()
    inverse_beta = torch.reciprocal(transfer.beta)
    return AffineTransfer(
        beta=inverse_beta,
        moment=-inverse_beta.unsqueeze(-1) * transfer.moment,
    )


def apply(transfer: AffineTransfer, rear_radiance: torch.Tensor) -> torch.Tensor:
    """Apply a transfer to rear radiance ``[...,3]``."""

    transfer.validate()
    _require_rgb_tensor("rear_radiance", rear_radiance)
    _require_same_dtype_device(
        "transfer",
        transfer.beta,
        "rear_radiance",
        rear_radiance,
    )
    beta, _ = torch.broadcast_tensors(transfer.beta, rear_radiance[..., 0])
    moment, radiance = torch.broadcast_tensors(transfer.moment, rear_radiance)
    return moment + beta.unsqueeze(-1) * radiance


def scale_generator(scale: torch.Tensor, generator: AffineGenerator) -> AffineGenerator:
    generator.validate()
    _require_float_tensor("scale", scale)
    _require_same_dtype_device("scale", scale, "generator", generator.scalar)
    scalar, scale_broadcast = torch.broadcast_tensors(generator.scalar, scale)
    source, _ = torch.broadcast_tensors(
        generator.source,
        scale_broadcast.unsqueeze(-1),
    )
    return AffineGenerator(
        scalar=scale_broadcast * scalar,
        source=scale_broadcast.unsqueeze(-1) * source,
    )


def add_generators(left: AffineGenerator, right: AffineGenerator) -> AffineGenerator:
    left.validate("left generator")
    right.validate("right generator")
    _require_same_dtype_device("left", left.scalar, "right", right.scalar)
    left_scalar, right_scalar = torch.broadcast_tensors(left.scalar, right.scalar)
    left_source, right_source = torch.broadcast_tensors(left.source, right.source)
    return AffineGenerator(
        scalar=left_scalar + right_scalar,
        source=left_source + right_source,
    )


def subtract_generators(left: AffineGenerator, right: AffineGenerator) -> AffineGenerator:
    return add_generators(left, scale_generator(-torch.ones_like(right.scalar), right))


def generator_exponential(generator: AffineGenerator, length: torch.Tensor) -> AffineTransfer:
    """Evaluate ``exp(length * X)`` with a stable identity limit."""

    generator.validate()
    _require_float_tensor("length", length)
    _require_same_dtype_device("generator", generator.scalar, "length", length)
    scalar, length = torch.broadcast_tensors(generator.scalar, length)
    source, _ = torch.broadcast_tensors(
        generator.source,
        scalar.unsqueeze(-1),
    )
    exponent = scalar * length
    exprel, _ = _exprel_and_derivative(exponent)
    return AffineTransfer(
        beta=torch.exp(exponent),
        moment=(length * exprel).unsqueeze(-1) * source,
    )


def segment_time_derivative(
    generator: AffineGenerator,
    generator_rate: AffineGenerator,
    length: torch.Tensor,
    length_rate: torch.Tensor,
) -> AffineTransferTangent:
    """Exact derivative of ``exp(length(t) * generator(t))``.

    This uses the closed affine-group exponential, not autograd or finite
    differences, and remains valid when the source changes color so long as
    the instantaneous P0 generator and its Eulerian time derivative are the
    supplied tensors.
    """

    generator.validate("generator")
    generator_rate.validate("generator_rate")
    _require_float_tensor("length", length)
    _require_float_tensor("length_rate", length_rate)
    _require_same_dtype_device("generator", generator.scalar, "generator_rate", generator_rate.scalar)
    _require_same_dtype_device("generator", generator.scalar, "length", length)
    _require_same_dtype_device("generator", generator.scalar, "length_rate", length_rate)

    scalar, scalar_rate, length, length_rate = torch.broadcast_tensors(
        generator.scalar,
        generator_rate.scalar,
        length,
        length_rate,
    )
    source, source_rate = torch.broadcast_tensors(
        generator.source,
        generator_rate.source,
    )
    source, _ = torch.broadcast_tensors(source, scalar.unsqueeze(-1))
    source_rate, _ = torch.broadcast_tensors(source_rate, scalar.unsqueeze(-1))

    exponent = scalar * length
    exponent_rate = scalar_rate * length + scalar * length_rate
    exprel, exprel_derivative = _exprel_and_derivative(exponent)
    source_scale = length * exprel
    source_scale_rate = (
        length_rate * exprel
        + length * exprel_derivative * exponent_rate
    )
    beta = torch.exp(exponent)
    return AffineTransferTangent(
        beta=beta * exponent_rate,
        moment=(
            source_scale_rate.unsqueeze(-1) * source
            + source_scale.unsqueeze(-1) * source_rate
        ),
    )


def compose_jets(
    front: AffineTransfer,
    front_tangent: AffineTransferTangent,
    back: AffineTransfer,
    back_tangent: AffineTransferTangent,
) -> tuple[AffineTransfer, AffineTransferTangent]:
    """Associative first-jet product for repo-order transfer composition."""

    front.validate("front")
    back.validate("back")
    front_tangent.validate("front_tangent")
    back_tangent.validate("back_tangent")
    _require_same_dtype_device(
        "front",
        front.beta,
        "front_tangent",
        front_tangent.beta,
    )
    _require_same_dtype_device(
        "back",
        back.beta,
        "back_tangent",
        back_tangent.beta,
    )
    value = compose(front, back)
    beta = (
        front_tangent.beta * back.beta
        + front.beta * back_tangent.beta
    )
    moment = (
        front_tangent.moment
        + front_tangent.beta.unsqueeze(-1) * back.moment
        + front.beta.unsqueeze(-1) * back_tangent.moment
    )
    return value, AffineTransferTangent(beta=beta, moment=moment)


def add_tangents(
    left: AffineTransferTangent,
    right: AffineTransferTangent,
) -> AffineTransferTangent:
    left.validate("left tangent")
    right.validate("right tangent")
    _require_same_dtype_device("left", left.beta, "right", right.beta)
    return AffineTransferTangent(
        beta=left.beta + right.beta,
        moment=left.moment + right.moment,
    )


def subtract_tangents(
    left: AffineTransferTangent,
    right: AffineTransferTangent,
) -> AffineTransferTangent:
    left.validate("left tangent")
    right.validate("right tangent")
    _require_same_dtype_device("left", left.beta, "right", right.beta)
    return AffineTransferTangent(
        beta=left.beta - right.beta,
        moment=left.moment - right.moment,
    )


def scale_tangent(scale: torch.Tensor, tangent: AffineTransferTangent) -> AffineTransferTangent:
    tangent.validate()
    _require_float_tensor("scale", scale)
    _require_same_dtype_device("scale", scale, "tangent", tangent.beta)
    return AffineTransferTangent(
        beta=scale * tangent.beta,
        moment=scale.unsqueeze(-1) * tangent.moment,
    )


def right_generator_action(
    transfer: AffineTransfer,
    generator: AffineGenerator,
) -> AffineTransferTangent:
    """Return the tangent matrix ``transfer * generator``."""

    transfer.validate()
    generator.validate()
    _require_same_dtype_device(
        "transfer",
        transfer.beta,
        "generator",
        generator.scalar,
    )
    return AffineTransferTangent(
        beta=transfer.beta * generator.scalar,
        moment=transfer.beta.unsqueeze(-1) * generator.source,
    )


def left_generator_action(
    generator: AffineGenerator,
    transfer: AffineTransfer,
) -> AffineTransferTangent:
    """Return the tangent matrix ``generator * transfer``."""

    transfer.validate()
    generator.validate()
    _require_same_dtype_device(
        "generator",
        generator.scalar,
        "transfer",
        transfer.beta,
    )
    return AffineTransferTangent(
        beta=generator.scalar * transfer.beta,
        moment=(
            generator.scalar.unsqueeze(-1) * transfer.moment
            + generator.source
        ),
    )


def generator_sandwich(
    prefix: AffineTransfer,
    generator: AffineGenerator,
    suffix: AffineTransfer,
) -> AffineTransferTangent:
    """Return ``prefix * generator * suffix`` in repo order.

    The suffix moment is deliberately present in the result.  This is the
    order-sensitive sentinel from the corrected curvature theorem.
    """

    prefix.validate("prefix")
    suffix.validate("suffix")
    generator.validate()
    _require_same_dtype_device(
        "prefix",
        prefix.beta,
        "generator",
        generator.scalar,
    )
    _require_same_dtype_device(
        "prefix",
        prefix.beta,
        "suffix",
        suffix.beta,
    )
    beta = prefix.beta * generator.scalar * suffix.beta
    moment = prefix.beta.unsqueeze(-1) * (
        generator.source
        + generator.scalar.unsqueeze(-1) * suffix.moment
    )
    return AffineTransferTangent(beta=beta, moment=moment)


def tangent_sandwich(
    left: AffineTransfer,
    tangent: AffineTransferTangent,
    right: AffineTransfer,
) -> AffineTransferTangent:
    """Return ``left * tangent * right`` for a signed affine tangent."""

    left.validate("left")
    tangent.validate()
    right.validate("right")
    _require_same_dtype_device("left", left.beta, "tangent", tangent.beta)
    _require_same_dtype_device("left", left.beta, "right", right.beta)
    beta = left.beta * tangent.beta * right.beta
    moment = left.beta.unsqueeze(-1) * (
        tangent.moment + tangent.beta.unsqueeze(-1) * right.moment
    )
    return AffineTransferTangent(beta=beta, moment=moment)


def homogeneous_matrix(transfer: AffineTransfer) -> torch.Tensor:
    """Convert ``T(beta,m)`` to a differentiable homogeneous matrix."""

    transfer.validate()
    eye = torch.eye(
        RGB_CHANNELS,
        dtype=transfer.beta.dtype,
        device=transfer.beta.device,
    )
    linear = transfer.beta[..., None, None] * eye
    top = torch.cat((linear, transfer.moment.unsqueeze(-1)), dim=-1)
    bottom = torch.cat(
        (
            torch.zeros_like(transfer.moment),
            torch.ones_like(transfer.beta).unsqueeze(-1),
        ),
        dim=-1,
    ).unsqueeze(-2)
    return torch.cat((top, bottom), dim=-2)


def physical_cone_report(
    transfer: AffineTransfer,
    *,
    tolerance: float = 0.0,
    minimum_beta: float = 0.0,
) -> PhysicalConeReport:
    """Check ``0<beta<=1`` and ``0<=m_c<=1-beta`` without projection."""

    transfer.validate()
    if tolerance < 0.0:
        raise ValueError("tolerance must be nonnegative")
    if minimum_beta < 0.0:
        raise ValueError("minimum_beta must be nonnegative")
    zero = torch.zeros((), dtype=transfer.beta.dtype, device=transfer.beta.device)
    beta_positive = torch.relu(minimum_beta - transfer.beta)
    beta_contraction = torch.relu(transfer.beta - 1.0)
    moment_lower = torch.relu(-transfer.moment)
    moment_upper = torch.relu(
        transfer.moment - (1.0 - transfer.beta).unsqueeze(-1)
    )
    maximum = torch.amax(
        torch.cat(
            (
                beta_positive.reshape(-1),
                beta_contraction.reshape(-1),
                moment_lower.reshape(-1),
                moment_upper.reshape(-1),
                zero.reshape(-1),
            )
        )
    )
    finite = torch.isfinite(transfer.beta).all() & torch.isfinite(transfer.moment).all()
    return PhysicalConeReport(
        maximum_violation=maximum,
        beta_positive_violation=torch.amax(beta_positive),
        beta_contraction_violation=torch.amax(beta_contraction),
        moment_lower_violation=torch.amax(moment_lower),
        moment_upper_violation=torch.amax(moment_upper),
        finite=finite,
        passed=finite & (maximum <= tolerance) & torch.all(transfer.beta > minimum_beta),
    )


def affine_group_report(
    transfer: AffineTransfer,
    *,
    minimum_beta: float = 0.0,
) -> AffineGroupReport:
    """Check the unrestricted ``beta>0`` affine group completion."""

    transfer.validate()
    if minimum_beta < 0.0:
        raise ValueError("minimum_beta must be nonnegative")
    finite = torch.isfinite(transfer.beta).all() & torch.isfinite(transfer.moment).all()
    min_beta = torch.amin(transfer.beta)
    inverse_beta = torch.reciprocal(transfer.beta)
    condition = torch.linalg.cond(homogeneous_matrix(transfer))
    return AffineGroupReport(
        minimum_beta=min_beta,
        maximum_inverse_beta=torch.amax(torch.abs(inverse_beta)),
        homogeneous_condition_number=torch.amax(condition),
        finite=finite & torch.isfinite(condition).all(),
        passed=finite & torch.all(transfer.beta > minimum_beta) & torch.isfinite(condition).all(),
    )


def _exprel_and_derivative(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``expm1(x)/x`` and its derivative with a finite small-x branch."""

    threshold = torch.finfo(value.dtype).eps ** 0.25
    small = torch.abs(value) <= threshold
    safe = torch.where(small, torch.ones_like(value), value)
    regular = torch.expm1(value) / safe
    regular_derivative = ((value - 1.0) * torch.exp(value) + 1.0) / safe.square()
    square = value.square()
    cube = square * value
    fourth = square.square()
    fifth = fourth * value
    series = (
        1.0
        + 0.5 * value
        + square / 6.0
        + cube / 24.0
        + fourth / 120.0
        + fifth / 720.0
    )
    derivative_series = (
        0.5
        + value / 3.0
        + square / 8.0
        + cube / 30.0
        + fourth / 144.0
        + fifth / 840.0
    )
    return (
        torch.where(small, series, regular),
        torch.where(small, derivative_series, regular_derivative),
    )


def _require_four_vector(name: str, tensor: torch.Tensor) -> None:
    _require_float_tensor(name, tensor)
    if tensor.ndim < 1 or tensor.shape[-1] != RGB_CHANNELS + 1:
        raise ValueError(f"{name} must have shape [...,4]")


def _require_rgb_tensor(name: str, tensor: torch.Tensor) -> None:
    _require_float_tensor(name, tensor)
    if tensor.ndim < 1 or tensor.shape[-1] != RGB_CHANNELS:
        raise ValueError(f"{name} must have shape [...,3]")


def _require_scalar_vector_pair(
    name: str,
    scalar: torch.Tensor,
    vector: torch.Tensor,
) -> None:
    _require_float_tensor(f"{name}.scalar", scalar)
    _require_rgb_tensor(f"{name}.vector", vector)
    if vector.shape[:-1] != scalar.shape:
        raise ValueError(
            f"{name} scalar must have shape [...] and vector shape [...,3]"
        )
    _require_same_dtype_device(
        f"{name}.scalar",
        scalar,
        f"{name}.vector",
        vector,
    )


def _require_float_tensor(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if tensor.dtype not in {torch.float32, torch.float64}:
        raise TypeError(f"{name} must use float32 or float64")


def _require_same_dtype_device(
    left_name: str,
    left: torch.Tensor,
    right_name: str,
    right: torch.Tensor,
) -> None:
    if left.dtype != right.dtype:
        raise TypeError(f"{left_name} and {right_name} must share a dtype")
    if left.device != right.device:
        raise ValueError(f"{left_name} and {right_name} must share a device")


__all__ = [
    "AffineGenerator",
    "AffineGroupReport",
    "AffineTransfer",
    "AffineTransferTangent",
    "PhysicalConeReport",
    "add_generators",
    "add_tangents",
    "affine_group_report",
    "apply",
    "compose",
    "compose_jets",
    "generator_exponential",
    "generator_sandwich",
    "homogeneous_matrix",
    "identity_transfer",
    "inverse",
    "left_generator_action",
    "physical_cone_report",
    "right_generator_action",
    "scale_generator",
    "scale_tangent",
    "scan",
    "segment_time_derivative",
    "subtract_generators",
    "subtract_tangents",
    "tangent_sandwich",
    "zero_tangent",
]
