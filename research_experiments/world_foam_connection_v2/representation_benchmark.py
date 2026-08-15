"""Equal-family, end-to-end probe comparison for ``U``, ``U_tilde``, ``K_F``.

Unlike the generic temporal-atlas helper, this module certifies every
candidate only after reconstructing the same physical transfer ``U``.  The
endpoint transports and the ``K_F`` base value are retained and charged.
The result is still a finite probe-grid certificate, not a continuous proof.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .temporal_atlas import (
    AtlasKind,
    physical_chart_to_transfer,
    transfer_to_physical_chart,
    transfer_to_unrestricted_log_chart,
    unrestricted_log_chart_to_transfer,
)


@dataclass(frozen=True)
class RepresentationProbeSeries:
    """Exact values sampled from one stable chart on a shared time grid."""

    times: torch.Tensor
    physical_transfer: torch.Tensor
    near_endpoint_transport: torch.Tensor
    far_endpoint_transport: torch.Tensor
    flow_corrected_transfer: torch.Tensor
    transported_curvature_source: torch.Tensor

    @property
    def probe_count(self) -> int:
        return int(self.times.numel())

    def validate(self) -> None:
        if self.times.ndim != 1 or self.times.numel() < 3:
            raise ValueError("representation comparison needs at least three times")
        if self.times.dtype not in {torch.float32, torch.float64}:
            raise TypeError("representation times must use float32 or float64")
        if not bool(torch.all(torch.isfinite(self.times))):
            raise ValueError("representation times must be finite")
        if not bool(torch.all(self.times[1:] > self.times[:-1])):
            raise ValueError("representation times must be strictly increasing")
        for name, value in (
            ("physical_transfer", self.physical_transfer),
            ("near_endpoint_transport", self.near_endpoint_transport),
            ("far_endpoint_transport", self.far_endpoint_transport),
            ("flow_corrected_transfer", self.flow_corrected_transfer),
            ("transported_curvature_source", self.transported_curvature_source),
        ):
            if value.shape != (self.probe_count, 4):
                raise ValueError(f"{name} must have shape [N,4]")
            if value.dtype != self.times.dtype:
                raise TypeError(f"{name} must share the time dtype")
            if value.device != self.times.device:
                raise ValueError(f"{name} must share the time device")
            if not bool(torch.all(torch.isfinite(value))):
                raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class RepresentationCertificate:
    """End-to-end finite-grid certificate and complete retained-state receipt."""

    kind: AtlasKind
    variant: str
    probe_count: int
    node_count: int
    selected_probe_indices: tuple[int, ...]
    maximum_relative_physical_error: float
    maximum_absolute_physical_error: float
    maximum_relative_secant_error: float
    primal_tolerance: float
    secant_tolerance: float
    atlas_payload_bytes: int
    shared_flow_payload_bytes: int
    total_retained_bytes: int
    compile_ordered_word_work: int
    compile_flow_run_evaluations: int
    retained_endpoint_node_values: int
    probe_reconstruction_compositions: int
    probe_cone_checks: int
    probe_group_checks: int
    minimum_group_beta: float
    maximum_physical_cone_violation: float
    group_passed: bool
    physical_cone_passed: bool
    probe_grid_only: bool
    selected_parameter_tangents_certified: bool
    complete_work_accounting: bool
    probe_primal_secant_verified: bool
    canonical_primal_tangent_verified: bool
    promotion_eligible: bool


@dataclass(frozen=True)
class RepresentationCompilation:
    kind: AtlasKind
    knots: torch.Tensor
    stored_values: torch.Tensor
    base_group_transfer: torch.Tensor | None
    reconstructed_physical_transfer: torch.Tensor
    certificate: RepresentationCertificate


def _interpolate(
    knots: torch.Tensor,
    values: torch.Tensor,
    query_times: torch.Tensor,
) -> torch.Tensor:
    flat = query_times.reshape(-1)
    right = torch.searchsorted(knots, flat, right=True)
    left = torch.clamp(right - 1, 0, knots.numel() - 2)
    right = left + 1
    alpha = (flat - knots[left]) / (knots[right] - knots[left])
    result = (
        (1.0 - alpha[:, None]) * values[left]
        + alpha[:, None] * values[right]
    )
    return result.reshape(query_times.shape + (values.shape[-1],))


def _integrate_piecewise_linear(
    knots: torch.Tensor,
    values: torch.Tensor,
    query_times: torch.Tensor,
) -> torch.Tensor:
    flat = query_times.reshape(-1)
    right = torch.searchsorted(knots, flat, right=True)
    left = torch.clamp(right - 1, 0, knots.numel() - 2)
    right = left + 1
    widths = knots[1:] - knots[:-1]
    segments = 0.5 * widths[:, None] * (values[:-1] + values[1:])
    cumulative = torch.cat(
        (torch.zeros_like(segments[:1]), torch.cumsum(segments, dim=0)),
        dim=0,
    )
    local_width = flat - knots[left]
    alpha = local_width / (knots[right] - knots[left])
    local_end = (
        (1.0 - alpha[:, None]) * values[left]
        + alpha[:, None] * values[right]
    )
    local = 0.5 * local_width[:, None] * (values[left] + local_end)
    return (cumulative[left] + local).reshape(
        query_times.shape + (values.shape[-1],)
    )


def _compose_raw(front: torch.Tensor, back: torch.Tensor) -> torch.Tensor:
    front, back = torch.broadcast_tensors(front, back)
    return torch.cat(
        (
            front[..., :1] * back[..., :1],
            front[..., 1:] + front[..., :1] * back[..., 1:],
        ),
        dim=-1,
    )


def _inverse_raw(transfer: torch.Tensor) -> torch.Tensor:
    inverse_beta = torch.reciprocal(transfer[..., :1])
    return torch.cat(
        (inverse_beta, -inverse_beta * transfer[..., 1:]),
        dim=-1,
    )


def _stored_probe_values(
    series: RepresentationProbeSeries,
    kind: AtlasKind,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if kind is AtlasKind.PHYSICAL_U:
        return transfer_to_physical_chart(series.physical_transfer), None
    near = transfer_to_unrestricted_log_chart(series.near_endpoint_transport)
    far = transfer_to_unrestricted_log_chart(series.far_endpoint_transport)
    if kind is AtlasKind.GROUP_U_TILDE:
        corrected = transfer_to_unrestricted_log_chart(
            series.flow_corrected_transfer
        )
        return torch.cat((corrected, near, far), dim=-1), None
    if kind is AtlasKind.SIGNED_K_F:
        return (
            torch.cat(
                (series.transported_curvature_source, near, far),
                dim=-1,
            ),
            series.flow_corrected_transfer[0],
        )
    raise ValueError(f"unsupported atlas kind {kind!r}")


def _decode_reconstruction(
    *,
    kind: AtlasKind,
    knots: torch.Tensor,
    stored_values: torch.Tensor,
    base_group_transfer: torch.Tensor | None,
    query_times: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    represented = _interpolate(knots, stored_values, query_times)
    if kind is AtlasKind.PHYSICAL_U:
        physical = physical_chart_to_transfer(represented)
        return physical, physical[..., 0], bool(torch.all(physical[..., 0] > 0.0))

    near = unrestricted_log_chart_to_transfer(represented[..., 4:8])
    far = unrestricted_log_chart_to_transfer(represented[..., 8:12])
    if kind is AtlasKind.GROUP_U_TILDE:
        corrected = unrestricted_log_chart_to_transfer(represented[..., :4])
    else:
        if base_group_transfer is None:
            raise ValueError("K_F reconstruction requires a base group transfer")
        corrected = base_group_transfer + _integrate_piecewise_linear(
            knots,
            stored_values[..., :4],
            query_times,
        )
    all_group_beta = torch.cat(
        (corrected[..., :1], near[..., :1], far[..., :1]),
        dim=-1,
    )
    group_passed = bool(torch.all(all_group_beta > 0.0))
    physical = _compose_raw(_compose_raw(_inverse_raw(near), corrected), far)
    return physical, all_group_beta, group_passed


def _physical_cone_violation(transfer: torch.Tensor) -> torch.Tensor:
    beta = transfer[..., :1]
    moment = transfer[..., 1:]
    return torch.amax(
        torch.cat(
            (
                torch.relu(-beta).reshape(-1),
                torch.relu(beta - 1.0).reshape(-1),
                torch.relu(-moment).reshape(-1),
                torch.relu(moment - (1.0 - beta)).reshape(-1),
                transfer.new_zeros((1,)),
            )
        )
    )


def _errors(
    predicted: torch.Tensor,
    target: torch.Tensor,
    times: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    absolute = torch.linalg.vector_norm(predicted - target, dim=-1)
    relative = absolute / torch.clamp(
        torch.linalg.vector_norm(target, dim=-1),
        min=1.0,
    )
    widths = times[1:] - times[:-1]
    predicted_secant = (predicted[1:] - predicted[:-1]) / widths[:, None]
    target_secant = (target[1:] - target[:-1]) / widths[:, None]
    secant_absolute = torch.linalg.vector_norm(
        predicted_secant - target_secant,
        dim=-1,
    )
    secant_relative = secant_absolute / torch.clamp(
        torch.linalg.vector_norm(target_secant, dim=-1),
        min=1.0,
    )
    return absolute, relative, secant_absolute, secant_relative


def compile_equal_family_representation(
    series: RepresentationProbeSeries,
    *,
    kind: AtlasKind,
    primal_tolerance: float,
    secant_tolerance: float,
    run_count: int,
    shared_flow_payload_bytes: int,
    variant: str | None = None,
    maximum_nodes: int | None = None,
    cone_tolerance: float = 1.0e-9,
) -> RepresentationCompilation:
    """Greedily compile one ABI under the same reconstructed-``U`` gates.

    The secant gate checks temporal behavior on the declared probe intervals;
    it is not a substitute for selected model-parameter JVP certification.
    """

    series.validate()
    if not isinstance(kind, AtlasKind):
        raise TypeError("kind must be AtlasKind")
    if variant is None:
        variant = {
            AtlasKind.PHYSICAL_U: "A0_direct_U",
            AtlasKind.GROUP_U_TILDE: "A1_group_U_tilde",
            AtlasKind.SIGNED_K_F: "A2_signed_K_F",
        }[kind]
    if not variant:
        raise ValueError("representation variant must be nonempty")
    if not primal_tolerance > 0.0 or not secant_tolerance > 0.0:
        raise ValueError("representation tolerances must be positive")
    if isinstance(run_count, bool) or not isinstance(run_count, int) or run_count < 1:
        raise ValueError("run_count must be a positive integer")
    if shared_flow_payload_bytes < 0:
        raise ValueError("shared_flow_payload_bytes must be nonnegative")
    if cone_tolerance < 0.0:
        raise ValueError("cone_tolerance must be nonnegative")
    if maximum_nodes is None:
        maximum_nodes = series.probe_count
    if (
        isinstance(maximum_nodes, bool)
        or not isinstance(maximum_nodes, int)
        or maximum_nodes < 2
        or maximum_nodes > series.probe_count
    ):
        raise ValueError("maximum_nodes must lie in [2, probe_count]")

    stored_probe_values, base = _stored_probe_values(series, kind)
    selected = torch.zeros(
        (series.probe_count,),
        dtype=torch.bool,
        device=series.times.device,
    )
    selected[0] = True
    selected[-1] = True

    while True:
        indices = torch.nonzero(selected, as_tuple=False).flatten()
        knots = series.times[indices]
        stored = stored_probe_values[indices]
        predicted, group_betas, group_passed = _decode_reconstruction(
            kind=kind,
            knots=knots,
            stored_values=stored,
            base_group_transfer=base,
            query_times=series.times,
        )
        absolute, relative, _, secant_relative = _errors(
            predicted,
            series.physical_transfer,
            series.times,
        )
        combined = relative / primal_tolerance
        interval_combined = secant_relative / secant_tolerance
        for interval_index in range(series.probe_count - 1):
            combined[interval_index] = torch.maximum(
                combined[interval_index],
                interval_combined[interval_index],
            )
            combined[interval_index + 1] = torch.maximum(
                combined[interval_index + 1],
                interval_combined[interval_index],
            )
        cone_violation = _physical_cone_violation(predicted)
        physical_passed = bool(cone_violation <= cone_tolerance)
        approximation_passed = bool(torch.all(combined <= 1.0))
        probe_verified = (
            approximation_passed and group_passed and physical_passed
        )
        if probe_verified or int(indices.numel()) >= maximum_nodes:
            atlas_payload = int(
                knots.numel() * knots.element_size()
                + stored.numel() * stored.element_size()
                + (0 if base is None else base.numel() * base.element_size())
            )
            word_multiplier = 2 if kind is AtlasKind.SIGNED_K_F else 1
            certificate = RepresentationCertificate(
                kind=kind,
                variant=variant,
                probe_count=series.probe_count,
                node_count=int(indices.numel()),
                selected_probe_indices=tuple(int(index) for index in indices),
                maximum_relative_physical_error=float(torch.amax(relative).detach().cpu()),
                maximum_absolute_physical_error=float(torch.amax(absolute).detach().cpu()),
                maximum_relative_secant_error=float(torch.amax(secant_relative).detach().cpu()),
                primal_tolerance=primal_tolerance,
                secant_tolerance=secant_tolerance,
                atlas_payload_bytes=atlas_payload,
                shared_flow_payload_bytes=shared_flow_payload_bytes,
                total_retained_bytes=atlas_payload + shared_flow_payload_bytes,
                compile_ordered_word_work=(
                    word_multiplier * int(indices.numel()) * run_count
                ),
                compile_flow_run_evaluations=(
                    0
                    if shared_flow_payload_bytes == 0
                    else int(indices.numel()) * run_count
                ),
                retained_endpoint_node_values=(
                    0
                    if kind is AtlasKind.PHYSICAL_U
                    else 2 * int(indices.numel())
                ),
                probe_reconstruction_compositions=(
                    0
                    if kind is AtlasKind.PHYSICAL_U
                    else 2 * series.probe_count
                ),
                probe_cone_checks=series.probe_count,
                probe_group_checks=(
                    series.probe_count
                    if kind is AtlasKind.PHYSICAL_U
                    else 3 * series.probe_count
                ),
                minimum_group_beta=float(torch.amin(group_betas).detach().cpu()),
                maximum_physical_cone_violation=float(cone_violation.detach().cpu()),
                group_passed=group_passed,
                physical_cone_passed=physical_passed,
                probe_grid_only=True,
                selected_parameter_tangents_certified=False,
                complete_work_accounting=False,
                probe_primal_secant_verified=probe_verified,
                canonical_primal_tangent_verified=False,
                promotion_eligible=False,
            )
            return RepresentationCompilation(
                kind=kind,
                knots=knots,
                stored_values=stored,
                base_group_transfer=base,
                reconstructed_physical_transfer=predicted,
                certificate=certificate,
            )

        scores = torch.where(
            selected,
            torch.full_like(combined, -1.0),
            combined,
        )
        selected[int(torch.argmax(scores))] = True


__all__ = [
    "RepresentationCertificate",
    "RepresentationCompilation",
    "RepresentationProbeSeries",
    "compile_equal_family_representation",
]
