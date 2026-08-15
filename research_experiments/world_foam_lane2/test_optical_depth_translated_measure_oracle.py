from __future__ import annotations

import pytest
import torch
from finite_element_material_transfer import (
    MaterialMode,
    evaluate_material_segment,
    material_segment_vjp,
)
from optical_depth_translated_measure_oracle import (
    DTYPE,
    concatenate_measures,
    laplace_tangent,
    laplace_tangent_weighted_variation_upper_bound,
    laplace_transfer,
    laplace_transfer_error_bounds,
    make_translated_measure,
    opacity_tail_directional_error_bound,
    opacity_tail_primal_error_bound,
    p0_word_transfer,
    p0_word_vjp,
    two_segment_commutator_formula,
)
from transfer_lie_chart import affine_transfer_compose


def _measure(depths: list[float], colors: list[list[float]]):
    return make_translated_measure(torch.tensor(depths, dtype=DTYPE), torch.tensor(colors, dtype=DTYPE))


def _existing_affine_p0_forward_and_vjp(
    density: torch.Tensor,
    color: torch.Tensor,
    length: torch.Tensor,
    grad_transfer: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Independent existing affine/material path with its explicit segment VJP."""

    total = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=DTYPE)
    prefixes = []
    segments = []
    controls = []
    for run_id in range(density.numel()):
        control = torch.stack((density[run_id], density.new_zeros(()), density.new_zeros(())))
        material = evaluate_material_segment(
            MaterialMode.M0_P0_CONSTANT,
            control,
            length[run_id],
            color[run_id],
        )
        segment = torch.cat((material.element.beta.reshape(1), material.element.m))
        prefixes.append(total)
        segments.append(segment)
        controls.append(control)
        total = affine_transfer_compose(total, segment)

    density_bar = torch.zeros_like(density)
    color_bar = torch.zeros_like(color)
    length_bar = torch.zeros_like(length)
    beta_bar = grad_transfer[0]
    moment_bar = grad_transfer[1:]
    for run_id in reversed(range(density.numel())):
        prefix = prefixes[run_id]
        segment = segments[run_id]
        segment_beta_bar = prefix[0] * beta_bar
        segment_moment_bar = prefix[0] * moment_bar
        segment_vjp = material_segment_vjp(
            MaterialMode.M0_P0_CONSTANT,
            controls[run_id],
            length[run_id],
            color[run_id],
            grad_beta=segment_beta_bar,
            grad_m=segment_moment_bar,
        )
        density_bar[run_id] = segment_vjp.density_controls[0]
        color_bar[run_id] = segment_vjp.color_front
        length_bar[run_id] = segment_vjp.length
        beta_bar = segment[0] * beta_bar + torch.dot(segment[1:], moment_bar)
    return total, density_bar, color_bar, length_bar


def test_semidirect_concatenation_is_associative_and_translates_rear_support() -> None:
    a = _measure([0.2, 0.7], [[0.8, 0.1, 0.3], [0.2, 0.9, 0.4]])
    b = _measure([0.5], [[0.1, 0.4, 0.9]])
    c = _measure([0.3, 0.4], [[0.7, 0.2, 0.5], [0.3, 0.8, 0.1]])

    left = concatenate_measures(concatenate_measures(a, b), c)
    right = concatenate_measures(a, concatenate_measures(b, c))
    torch.testing.assert_close(left.kappa, right.kappa, atol=0.0, rtol=0.0)
    torch.testing.assert_close(left.support_intervals(), right.support_intervals(), atol=0.0, rtol=0.0)
    torch.testing.assert_close(left.colors, right.colors, atol=0.0, rtol=0.0)

    joined = concatenate_measures(a, b)
    translated_b_support = b.support_intervals() + a.kappa
    torch.testing.assert_close(joined.support_intervals()[a.run_count :], translated_b_support, atol=0.0, rtol=0.0)

    identity = make_translated_measure(torch.empty(0, dtype=DTYPE), torch.empty((0, 3), dtype=DTYPE))
    for joined_identity in (concatenate_measures(identity, a), concatenate_measures(a, identity)):
        torch.testing.assert_close(joined_identity.support_intervals(), a.support_intervals(), atol=0.0, rtol=0.0)
        torch.testing.assert_close(joined_identity.colors, a.colors, atol=0.0, rtol=0.0)


def test_laplace_image_is_affine_transfer_homomorphism() -> None:
    front = _measure([0.17, 0.61], [[0.9, 0.2, 0.1], [0.1, 0.7, 0.8]])
    rear = _measure([0.33, 0.29], [[0.4, 0.9, 0.2], [0.7, 0.1, 0.6]])

    actual = laplace_transfer(concatenate_measures(front, rear)).as_tensor()
    expected = affine_transfer_compose(
        laplace_transfer(front).as_tensor(),
        laplace_transfer(rear).as_tensor(),
    )
    torch.testing.assert_close(actual, expected, atol=2.0e-15, rtol=2.0e-15)


def test_two_segment_noncommutativity_formula_is_exact() -> None:
    tau_a = torch.tensor(0.73, dtype=DTYPE)
    tau_b = torch.tensor(0.41, dtype=DTYPE)
    color_a = torch.tensor([0.9, 0.1, 0.3], dtype=DTYPE)
    color_b = torch.tensor([0.2, 0.8, 0.6], dtype=DTYPE)
    a = make_translated_measure(tau_a.reshape(1), color_a.reshape(1, 3))
    b = make_translated_measure(tau_b.reshape(1), color_b.reshape(1, 3))

    transfer_ab = laplace_transfer(concatenate_measures(a, b))
    transfer_ba = laplace_transfer(concatenate_measures(b, a))
    torch.testing.assert_close(transfer_ab.beta, transfer_ba.beta, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        transfer_ab.moment - transfer_ba.moment,
        two_segment_commutator_formula(tau_a, color_a, tau_b, color_b),
        atol=2.0e-15,
        rtol=2.0e-15,
    )


def test_distributional_boundary_tangent_matches_autograd_and_finite_difference() -> None:
    depths = torch.tensor([0.31, 0.77, 0.42], dtype=DTYPE)
    colors = torch.tensor(
        [[0.82, 0.16, 0.11], [0.12, 0.71, 0.91], [0.46, 0.31, 0.77]],
        dtype=DTYPE,
    )
    depth_dot = torch.tensor([0.23, -0.19, 0.37], dtype=DTYPE)
    color_dot = torch.tensor(
        [[-0.11, 0.17, 0.08], [0.21, -0.07, 0.04], [0.09, 0.13, -0.16]],
        dtype=DTYPE,
    )
    analytic = laplace_tangent(make_translated_measure(depths, colors), depth_dot, color_dot).as_tensor()

    def evaluate(depth_values: torch.Tensor, color_values: torch.Tensor) -> torch.Tensor:
        return laplace_transfer(make_translated_measure(depth_values, color_values)).as_tensor()

    _, autograd_jvp = torch.autograd.functional.jvp(
        evaluate,
        (depths, colors),
        (depth_dot, color_dot),
        create_graph=False,
        strict=True,
    )
    epsilon = 1.0e-6
    finite_difference = (
        evaluate(depths + epsilon * depth_dot, colors + epsilon * color_dot)
        - evaluate(depths - epsilon * depth_dot, colors - epsilon * color_dot)
    ) / (2.0 * epsilon)
    torch.testing.assert_close(analytic, autograd_jvp, atol=3.0e-15, rtol=3.0e-15)
    torch.testing.assert_close(analytic, finite_difference, atol=2.0e-10, rtol=2.0e-10)


def test_weighted_total_variation_certifies_transfer_and_tangent_errors() -> None:
    first = _measure(
        [0.23, 0.51, 0.19],
        [[0.81, 0.12, 0.34], [0.15, 0.73, 0.91], [0.42, 0.27, 0.65]],
    )
    second = _measure(
        [0.17, 0.38, 0.44, 0.21],
        [
            [0.76, 0.18, 0.29],
            [0.24, 0.64, 0.83],
            [0.37, 0.31, 0.71],
            [0.69, 0.08, 0.22],
        ],
    )
    bounds = laplace_transfer_error_bounds(first, second)
    first_transfer = laplace_transfer(first)
    second_transfer = laplace_transfer(second)
    assert abs((first_transfer.beta - second_transfer.beta).item()) <= (
        bounds.beta_absolute_error_bound.item() + 1.0e-14
    )
    assert torch.linalg.vector_norm(
        first_transfer.moment - second_transfer.moment
    ).item() <= bounds.moment_l2_error_bound.item() + 1.0e-14

    depth_dot = torch.tensor([0.31, -0.16, 0.22], dtype=DTYPE)
    color_dot = torch.tensor(
        [[0.11, -0.08, 0.04], [-0.17, 0.21, 0.06], [0.09, 0.13, -0.12]],
        dtype=DTYPE,
    )
    tangent = laplace_tangent(first, depth_dot, color_dot)
    tangent_bound = laplace_tangent_weighted_variation_upper_bound(
        first,
        depth_dot,
        color_dot,
    )
    assert torch.linalg.vector_norm(tangent.moment).item() <= (
        tangent_bound.item() + 1.0e-14
    )


def test_opacity_tail_primal_and_fixed_split_directional_bounds_are_sound() -> None:
    front = _measure(
        [1.7, 2.1],
        [[0.82, 0.13, 0.31], [0.21, 0.74, 0.56]],
    )
    rear = _measure(
        [0.43, 0.87, 0.36],
        [[0.17, 0.91, 0.28], [0.66, 0.22, 0.79], [0.39, 0.58, 0.14]],
    )
    background = torch.tensor([0.06, 0.09, 0.12], dtype=DTYPE)
    background_dot = torch.tensor([0.03, -0.02, 0.01], dtype=DTYPE)
    front_depth_dot = torch.tensor([0.19, -0.07], dtype=DTYPE)
    rear_depth_dot = torch.tensor([-0.11, 0.23, 0.08], dtype=DTYPE)
    front_color_dot = torch.tensor(
        [[0.07, -0.04, 0.02], [-0.03, 0.09, 0.05]],
        dtype=DTYPE,
    )
    rear_color_dot = torch.tensor(
        [[-0.12, 0.08, 0.04], [0.05, -0.06, 0.11], [0.09, 0.03, -0.07]],
        dtype=DTYPE,
    )

    full = concatenate_measures(front, rear)
    full_transfer = laplace_transfer(full)
    prefix_transfer = laplace_transfer(front)
    full_color = full_transfer.moment + full_transfer.beta * background
    prefix_color = prefix_transfer.moment + prefix_transfer.beta * background
    primal_bound = opacity_tail_primal_error_bound(front, rear, background)
    assert torch.linalg.vector_norm(full_color - prefix_color).item() <= (
        primal_bound.item() + 1.0e-14
    )

    full_tangent = laplace_tangent(
        full,
        torch.cat((front_depth_dot, rear_depth_dot)),
        torch.cat((front_color_dot, rear_color_dot)),
    )
    prefix_tangent = laplace_tangent(
        front,
        front_depth_dot,
        front_color_dot,
    )
    full_color_tangent = (
        full_tangent.moment
        + full_tangent.beta * background
        + full_transfer.beta * background_dot
    )
    prefix_color_tangent = (
        prefix_tangent.moment
        + prefix_tangent.beta * background
        + prefix_transfer.beta * background_dot
    )
    directional_bound = opacity_tail_directional_error_bound(
        front,
        rear,
        background,
        front_optical_depth_tangent=front_depth_dot,
        rear_optical_depth_tangent=rear_depth_dot,
        rear_color_tangent=rear_color_dot,
        background_tangent=background_dot,
    )
    assert torch.linalg.vector_norm(
        full_color_tangent - prefix_color_tangent
    ).item() <= directional_bound.item() + 1.0e-14


def test_new_certificates_handle_empty_and_zero_width_words() -> None:
    identity = make_translated_measure(
        torch.empty(0, dtype=DTYPE),
        torch.empty((0, 3), dtype=DTYPE),
    )
    zero_width = _measure(
        [0.0, 0.0],
        [[0.92, 0.11, 0.37], [0.18, 0.76, 0.54]],
    )
    bounds = laplace_transfer_error_bounds(identity, zero_width)
    assert bounds.beta_absolute_error_bound.item() == 0.0
    assert bounds.moment_l2_error_bound.item() == 0.0
    assert laplace_tangent_weighted_variation_upper_bound(
        identity,
        torch.empty(0, dtype=DTYPE),
        torch.empty((0, 3), dtype=DTYPE),
    ).item() == 0.0

    front = _measure([0.7], [[0.4, 0.5, 0.6]])
    prefix_transfer = laplace_transfer(front)
    background = torch.tensor([0.03, 0.06, 0.09], dtype=DTYPE)
    assert opacity_tail_primal_error_bound(
        front,
        identity,
        background,
    ).item() == 0.0
    assert opacity_tail_directional_error_bound(
        front,
        identity,
        background,
        front_optical_depth_tangent=torch.tensor([0.2], dtype=DTYPE),
        rear_optical_depth_tangent=torch.empty(0, dtype=DTYPE),
        rear_color_tangent=torch.empty((0, 3), dtype=DTYPE),
    ).item() == 0.0

    tiny_rear = _measure(
        [1.0e-18],
        [[0.71, 0.24, 0.53]],
    )
    tiny_measure_bounds = laplace_transfer_error_bounds(identity, tiny_rear)
    tiny_transfer = laplace_transfer(tiny_rear)
    tiny_moment = torch.linalg.vector_norm(tiny_transfer.moment).item()
    assert tiny_moment > 0.0
    assert tiny_moment <= tiny_measure_bounds.moment_l2_error_bound.item()

    tiny_tail_bound = opacity_tail_primal_error_bound(
        front,
        tiny_rear,
        background,
    )
    tiny_full_transfer = laplace_transfer(concatenate_measures(front, tiny_rear))
    tiny_primal_error = torch.linalg.vector_norm(
        tiny_full_transfer.moment
        + tiny_full_transfer.beta * background
        - prefix_transfer.moment
        - prefix_transfer.beta * background
    )
    assert tiny_tail_bound.item() > 0.0
    # The certificate bounds the exact discarded tail.  At 1e-18 optical
    # depth, recomputing the concatenated prefix and subtracting it is below
    # float64 ulp scale, so separately account for that oracle roundoff rather
    # than pretending it is part of the analytic tail bound.
    rendered_scale = torch.linalg.vector_norm(
        prefix_transfer.moment + prefix_transfer.beta * background
    )
    oracle_roundoff = (
        4.0 * torch.finfo(DTYPE).eps * torch.maximum(rendered_scale, rendered_scale.new_ones(()))
    )
    assert tiny_primal_error.item() <= (
        tiny_tail_bound + oracle_roundoff
    ).item()


def test_p0_forward_and_vjp_match_existing_affine_material_path() -> None:
    density = torch.tensor([0.43, 0.81, 0.27, 0.66], dtype=DTYPE)
    length = torch.tensor([0.52, 1.13, 0.37, 0.91], dtype=DTYPE)
    color = torch.tensor(
        [[0.91, 0.16, 0.08], [0.10, 0.66, 0.93], [0.38, 0.84, 0.25], [0.72, 0.31, 0.57]],
        dtype=DTYPE,
    )
    grad_transfer = torch.tensor([0.37, -0.29, 0.51, 0.18], dtype=DTYPE)

    expected = _existing_affine_p0_forward_and_vjp(density, color, length, grad_transfer)
    actual_transfer = p0_word_transfer(density, color, length).as_tensor()
    actual_vjp = p0_word_vjp(
        density,
        color,
        length,
        grad_beta=grad_transfer[0],
        grad_moment=grad_transfer[1:],
    )
    torch.testing.assert_close(actual_transfer, expected[0], atol=3.0e-15, rtol=3.0e-15)
    torch.testing.assert_close(actual_vjp.density, expected[1], atol=3.0e-15, rtol=3.0e-15)
    torch.testing.assert_close(actual_vjp.color, expected[2], atol=3.0e-15, rtol=3.0e-15)
    torch.testing.assert_close(actual_vjp.length, expected[3], atol=3.0e-15, rtol=3.0e-15)


def test_zero_width_and_zero_opacity_have_primal_no_op_and_one_sided_tangents() -> None:
    density = torch.tensor([0.8, 0.0, 0.5], dtype=DTYPE)
    length = torch.tensor([0.0, 0.9, 0.7], dtype=DTYPE)
    color = torch.tensor(
        [[0.95, 0.08, 0.21], [0.12, 0.88, 0.43], [0.34, 0.29, 0.81]],
        dtype=DTYPE,
    )
    grad_transfer = torch.tensor([0.37, 0.51, -0.23, 0.42], dtype=DTYPE)

    full = p0_word_transfer(density, color, length).as_tensor()
    reduced = p0_word_transfer(density[2:], color[2:], length[2:]).as_tensor()
    torch.testing.assert_close(full, reduced, atol=0.0, rtol=0.0)

    vjp = p0_word_vjp(
        density,
        color,
        length,
        grad_beta=grad_transfer[0],
        grad_moment=grad_transfer[1:],
    )
    assert vjp.density[0].item() == 0.0
    assert vjp.length[1].item() == 0.0
    assert abs(vjp.length[0].item()) > 1.0e-3
    assert abs(vjp.density[1].item()) > 1.0e-3

    def objective(density_values: torch.Tensor, length_values: torch.Tensor) -> torch.Tensor:
        return torch.dot(p0_word_transfer(density_values, color, length_values).as_tensor(), grad_transfer)

    epsilon = 1.0e-7
    base = objective(density, length)
    length_right = length.clone()
    length_right[0] += epsilon
    density_right = density.clone()
    density_right[1] += epsilon
    torch.testing.assert_close(
        (objective(density, length_right) - base) / epsilon, vjp.length[0], atol=8.0e-8, rtol=8.0e-8
    )
    torch.testing.assert_close(
        (objective(density_right, length) - base) / epsilon,
        vjp.density[1],
        atol=8.0e-8,
        rtol=8.0e-8,
    )

    with pytest.raises(ValueError, match="nonnegative"):
        p0_word_transfer(density, color, length - torch.tensor([epsilon, 0.0, 0.0], dtype=DTYPE))
    with pytest.raises(ValueError, match="nonnegative"):
        p0_word_transfer(density - torch.tensor([0.0, epsilon, 0.0], dtype=DTYPE), color, length)
