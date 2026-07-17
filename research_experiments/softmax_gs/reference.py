from __future__ import annotations

from dataclasses import dataclass, replace

import torch


@dataclass(frozen=True)
class SoftmaxGSDebugRow:
    index: int
    input_transmittance: torch.Tensor
    output_transmittance: torch.Tensor
    original_absorbance: torch.Tensor
    effective_absorbance: torch.Tensor
    past_absorbance: torch.Tensor
    effective_past_absorbance: torch.Tensor
    softmax_current_weight: torch.Tensor
    decay: torch.Tensor


@dataclass(frozen=True)
class SoftmaxGSTapeRow:
    index: int
    input_transmittance: torch.Tensor
    output_transmittance: torch.Tensor
    input_absorbance: torch.Tensor
    effective_absorbance: torch.Tensor
    effective_past_absorbance: torch.Tensor
    contribution_weight: torch.Tensor
    final_contribution_weight: torch.Tensor
    prefix_weight_scale: torch.Tensor
    past_exponent: torch.Tensor
    output_past_exponent: torch.Tensor


@dataclass(frozen=True)
class SoftmaxGSBoundedTape:
    selected_indices: torch.Tensor
    selected_weights: torch.Tensor
    residual_weight: torch.Tensor
    final_alpha: torch.Tensor
    selected_rows: tuple[SoftmaxGSTapeRow, ...]


def vanilla_alpha_over(
    absorbance: torch.Tensor,
    features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Composite already-sorted per-splat absorbance and features."""

    _validate_inputs(absorbance, features)
    transmittance = torch.ones((), dtype=absorbance.dtype, device=absorbance.device)
    out = torch.zeros(features.shape[-1], dtype=features.dtype, device=features.device)
    for k in range(absorbance.shape[0]):
        a_cur = absorbance[k].clamp(0.0, 1.0)
        out = out + transmittance.to(features.dtype) * a_cur.to(features.dtype) * features[k]
        transmittance = transmittance * (1.0 - a_cur)
    return out, 1.0 - transmittance


def vanilla_alpha_over_weights(absorbance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-splat contribution weights and final alpha for alpha-over."""

    if absorbance.ndim != 1:
        raise ValueError("absorbance must be a 1D tensor for one pixel/ray")
    transmittance = torch.ones((), dtype=absorbance.dtype, device=absorbance.device)
    weights = []
    for k in range(absorbance.shape[0]):
        a_cur = absorbance[k].clamp(0.0, 1.0)
        weight = transmittance * a_cur
        weights.append(weight)
        transmittance = transmittance * (1.0 - a_cur)
    return torch.stack(weights) if weights else torch.empty_like(absorbance), 1.0 - transmittance


def softmax_gs_composite(
    absorbance: torch.Tensor,
    exponent: torch.Tensor,
    depth: torch.Tensor,
    features: torch.Tensor,
    *,
    beta: float | torch.Tensor,
    gamma: float | torch.Tensor,
    enabled: bool = True,
    eps: float = 1.0e-7,
    return_debug: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, tuple[SoftmaxGSDebugRow, ...]]:
    """Softmax-GS per-ray forward reference.

    Inputs are the already-evaluated values for one pixel/ray in front-to-back
    order. `absorbance` is the paper's per-splat `a_k`; `exponent` is `p_k`,
    usually `-0.5 * Mahalanobis2D`; `depth` is the depth signal used by the
    `gamma` decay. The implementation follows the paper's sequential
    past-versus-current approximation and keeps original alpha-over
    transmittance.
    """

    _validate_inputs(absorbance, features)
    if exponent.shape != absorbance.shape:
        raise ValueError("exponent must have the same shape as absorbance")
    if depth.shape != absorbance.shape:
        raise ValueError("depth must have the same shape as absorbance")
    if not enabled:
        return vanilla_alpha_over(absorbance, features)

    beta_t = _as_per_item(beta, absorbance)
    gamma_t = _as_per_item(gamma, absorbance)
    transmittance = torch.ones((), dtype=absorbance.dtype, device=absorbance.device)
    color = torch.zeros(features.shape[-1], dtype=features.dtype, device=features.device)
    past_depth = torch.zeros((), dtype=depth.dtype, device=depth.device)
    past_exponent = torch.zeros((), dtype=exponent.dtype, device=exponent.device)
    debug_rows: list[SoftmaxGSDebugRow] = []

    for k in range(absorbance.shape[0]):
        a_input = absorbance[k].clamp(0.0, 1.0 - eps)
        a_cur = a_input
        effective_past = torch.zeros_like(a_cur)
        w_cur = torch.ones_like(a_cur)
        decay = torch.zeros_like(a_cur)
        input_transmittance = transmittance

        if k > 0:
            original_transmittance = transmittance * (1.0 - a_cur)
            a_past = 1.0 - transmittance
            w_cur = torch.sigmoid(beta_t[k] * (exponent[k] - past_exponent))
            w_past = 1.0 - w_cur
            soft_cur = w_cur * a_cur
            soft_past = w_past * a_past

            soft_denom = (soft_past + soft_cur).clamp_min(eps)
            tilde_past = soft_past * (1.0 - original_transmittance) / soft_denom
            # Algebraic solution of the paper's order-invariance equations.
            # This uses the uncorrected soft past term in the denominator.
            tilde_cur_denom = (soft_cur + soft_past * original_transmittance).clamp_min(eps)
            tilde_cur = soft_cur * (1.0 - original_transmittance) / tilde_cur_denom

            decay = torch.exp(-gamma_t[k].clamp_min(0.0) * torch.abs(depth[k] - past_depth))
            effective_past = decay * tilde_past + (1.0 - decay) * a_past
            a_cur = decay * tilde_cur + (1.0 - decay) * a_cur
            effective_past, a_cur = _rescale_pair_for_transmittance(
                effective_past,
                a_cur,
                original_transmittance,
                eps=eps,
            )

            transmittance = 1.0 - effective_past
            color_scale = effective_past / a_past.clamp_min(eps)
            color = color * color_scale.to(color.dtype)

        color = color + transmittance.to(features.dtype) * a_cur.to(features.dtype) * features[k]
        contribution_absorbance = a_cur * transmittance
        denom = (1.0 - transmittance + contribution_absorbance).clamp_min(eps)
        past_depth = (past_depth * (1.0 - transmittance) + depth[k] * contribution_absorbance) / denom
        past_exponent = (
            past_exponent * (1.0 - transmittance) + exponent[k] * contribution_absorbance
        ) / denom
        transmittance = transmittance * (1.0 - a_cur)

        if return_debug:
            debug_rows.append(
                SoftmaxGSDebugRow(
                    index=k,
                    input_transmittance=input_transmittance,
                    output_transmittance=transmittance,
                    original_absorbance=a_input,
                    effective_absorbance=a_cur,
                    past_absorbance=(1.0 - input_transmittance),
                    effective_past_absorbance=effective_past,
                    softmax_current_weight=w_cur,
                    decay=decay,
                )
            )

    final_alpha = 1.0 - transmittance
    if return_debug:
        return color, final_alpha, tuple(debug_rows)
    return color, final_alpha


def softmax_gs_contribution_tape(
    absorbance: torch.Tensor,
    exponent: torch.Tensor,
    depth: torch.Tensor,
    *,
    beta: float | torch.Tensor,
    gamma: float | torch.Tensor,
    enabled: bool = True,
    eps: float = 1.0e-7,
) -> tuple[torch.Tensor, torch.Tensor, tuple[SoftmaxGSTapeRow, ...]]:
    """Return final per-splat color weights plus a compact forward tape.

    The weights satisfy `output = weights @ features` for any feature matrix
    with matching K. This is the first native-backward contract: color
    gradients can be computed from these weights, and later geometry/opacity
    reverse kernels must preserve the same tape semantics.
    """

    if absorbance.ndim != 1:
        raise ValueError("absorbance must be a 1D tensor for one pixel/ray")
    if exponent.shape != absorbance.shape:
        raise ValueError("exponent must have the same shape as absorbance")
    if depth.shape != absorbance.shape:
        raise ValueError("depth must have the same shape as absorbance")
    if not enabled:
        weights, final_alpha = vanilla_alpha_over_weights(absorbance)
        rows = tuple(
            SoftmaxGSTapeRow(
                index=k,
                input_transmittance=torch.ones_like(final_alpha) - weights[:k].sum(),
                output_transmittance=torch.ones_like(final_alpha) - weights[: k + 1].sum(),
                input_absorbance=absorbance[k].clamp(0.0, 1.0),
                effective_absorbance=absorbance[k].clamp(0.0, 1.0),
                effective_past_absorbance=torch.zeros_like(final_alpha),
                contribution_weight=weights[k],
                final_contribution_weight=weights[k],
                prefix_weight_scale=torch.ones_like(final_alpha),
                past_exponent=torch.zeros_like(final_alpha),
                output_past_exponent=torch.zeros_like(final_alpha),
            )
            for k in range(absorbance.shape[0])
        )
        return weights, final_alpha, rows

    beta_t = _as_per_item(beta, absorbance)
    gamma_t = _as_per_item(gamma, absorbance)
    transmittance = torch.ones((), dtype=absorbance.dtype, device=absorbance.device)
    past_depth = torch.zeros((), dtype=depth.dtype, device=depth.device)
    past_exponent = torch.zeros((), dtype=exponent.dtype, device=exponent.device)
    weights = torch.zeros_like(absorbance)
    rows: list[SoftmaxGSTapeRow] = []

    for k in range(absorbance.shape[0]):
        input_transmittance = transmittance
        input_past_exponent = past_exponent
        a_input = absorbance[k].clamp(0.0, 1.0 - eps)
        a_cur = a_input
        effective_past = torch.zeros_like(a_cur)
        prefix_scale = torch.ones_like(a_cur)

        if k > 0:
            original_transmittance = transmittance * (1.0 - a_cur)
            a_past = 1.0 - transmittance
            w_cur = torch.sigmoid(beta_t[k] * (exponent[k] - past_exponent))
            soft_cur = w_cur * a_cur
            soft_past = (1.0 - w_cur) * a_past
            soft_denom = (soft_past + soft_cur).clamp_min(eps)
            tilde_past = soft_past * (1.0 - original_transmittance) / soft_denom
            tilde_cur = soft_cur * (1.0 - original_transmittance) / (
                soft_cur + soft_past * original_transmittance
            ).clamp_min(eps)

            decay = torch.exp(-gamma_t[k].clamp_min(0.0) * torch.abs(depth[k] - past_depth))
            effective_past = decay * tilde_past + (1.0 - decay) * a_past
            a_cur = decay * tilde_cur + (1.0 - decay) * a_cur
            effective_past, a_cur = _rescale_pair_for_transmittance(
                effective_past,
                a_cur,
                original_transmittance,
                eps=eps,
            )
            transmittance = 1.0 - effective_past
            prefix_scale = effective_past / a_past.clamp_min(eps)
            weights = weights * prefix_scale

        contribution_weight = transmittance * a_cur
        weights = weights.clone()
        weights[k] = contribution_weight
        denom = (1.0 - transmittance + contribution_weight).clamp_min(eps)
        past_depth = (past_depth * (1.0 - transmittance) + depth[k] * contribution_weight) / denom
        past_exponent = (
            past_exponent * (1.0 - transmittance) + exponent[k] * contribution_weight
        ) / denom
        transmittance = transmittance * (1.0 - a_cur)

        rows.append(
            SoftmaxGSTapeRow(
                index=k,
                input_transmittance=input_transmittance,
                output_transmittance=transmittance,
                input_absorbance=a_input,
                effective_absorbance=a_cur,
                effective_past_absorbance=effective_past,
                contribution_weight=contribution_weight,
                final_contribution_weight=contribution_weight,
                prefix_weight_scale=prefix_scale,
                past_exponent=input_past_exponent,
                output_past_exponent=past_exponent,
            )
        )

    final_rows = tuple(
        replace(row, final_contribution_weight=weights[row.index])
        for row in rows
    )
    return weights, 1.0 - transmittance, final_rows


def softmax_gs_bounded_contribution_tape(
    absorbance: torch.Tensor,
    exponent: torch.Tensor,
    depth: torch.Tensor,
    *,
    beta: float | torch.Tensor,
    gamma: float | torch.Tensor,
    k_limit: int,
    enabled: bool = True,
    eps: float = 1.0e-7,
) -> SoftmaxGSBoundedTape:
    """Return the exact top-K final contribution weights plus omitted mass.

    This is the bounded tape contract we want the shader to lower toward. The
    selected weights are exact entries from the full contribution tape, chosen
    by largest final contribution weight and returned in front-to-back order.
    For features in ``[0, 1]``, omitting the residual contributors bounds every
    output-channel error by ``residual_weight``.
    """

    if int(k_limit) < 1:
        raise ValueError(f"k_limit must be >= 1, got {k_limit}.")
    weights, final_alpha, rows = softmax_gs_contribution_tape(
        absorbance,
        exponent,
        depth,
        beta=beta,
        gamma=gamma,
        enabled=enabled,
        eps=eps,
    )
    if weights.numel() == 0:
        empty_indices = torch.empty((0,), dtype=torch.long, device=absorbance.device)
        return SoftmaxGSBoundedTape(
            selected_indices=empty_indices,
            selected_weights=weights,
            residual_weight=final_alpha,
            final_alpha=final_alpha,
            selected_rows=(),
        )

    selected_count = min(int(k_limit), int(weights.numel()))
    top_indices = torch.topk(weights, selected_count).indices
    selected_indices = torch.sort(top_indices).values
    selected_weights = weights.index_select(0, selected_indices)
    residual_weight = (final_alpha - selected_weights.sum()).clamp_min(0.0)
    selected_rows = tuple(rows[int(index)] for index in selected_indices.detach().cpu().tolist())
    return SoftmaxGSBoundedTape(
        selected_indices=selected_indices,
        selected_weights=selected_weights,
        residual_weight=residual_weight,
        final_alpha=final_alpha,
        selected_rows=selected_rows,
    )


def _rescale_pair_for_transmittance(
    past_absorbance: torch.Tensor,
    current_absorbance: torch.Tensor,
    target_transmittance: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    pair_product = past_absorbance * current_absorbance
    pair_sum = past_absorbance + current_absorbance
    if bool((pair_product <= eps).detach().cpu()):
        return past_absorbance, current_absorbance
    discriminant = (pair_sum.square() - 4.0 * (1.0 - target_transmittance) * pair_product).clamp_min(eps)
    scale = 2.0 * (1.0 - target_transmittance) / (pair_sum + torch.sqrt(discriminant)).clamp_min(eps)
    return scale * past_absorbance, scale * current_absorbance


def _as_per_item(value: float | torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=like.dtype, device=like.device)
    if tensor.ndim == 0:
        return tensor.expand_as(like)
    if tensor.shape != like.shape:
        raise ValueError("beta/gamma tensors must be scalar or match absorbance shape")
    return tensor


def _validate_inputs(absorbance: torch.Tensor, features: torch.Tensor) -> None:
    if absorbance.ndim != 1:
        raise ValueError("absorbance must be a 1D tensor for one pixel/ray")
    if features.ndim != 2:
        raise ValueError("features must have shape [K, F]")
    if features.shape[0] != absorbance.shape[0]:
        raise ValueError("features and absorbance must agree on K")
