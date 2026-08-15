from __future__ import annotations

import os
from pathlib import Path
import re

import pytest
import torch

from research_experiments.world_foam_lane2.finite_element_material_metal import (
    FiniteElementMaterialMetal,
    SOURCE_PATH,
)
from research_experiments.world_foam_lane2.run_finite_element_material_gate import (
    build_cpu_report,
)
from research_experiments.world_foam_lane2.finite_element_material_transfer import (
    BranchStatus,
    MaterialMode,
    branch_status_counts,
    evaluate_material_segment,
    material_segment_vjp,
)


DTYPE = torch.float64


def _inputs(mode: MaterialMode, *, branch: str = "ordinary"):
    if branch == "tiny_tau":
        controls = [2.0e-5, 0.0, 0.0]
    elif mode in (MaterialMode.M0_P0_CONSTANT, MaterialMode.M1_P0_AFFINE_RGB):
        controls = [0.7, 0.0, 0.0]
    elif mode == MaterialMode.M2_POSITIVE_BERNSTEIN_P1:
        controls = [0.25, 1.1, 0.0]
    elif mode == MaterialMode.M3_POSITIVE_BERNSTEIN_P2:
        controls = [0.2, 1.4, 0.5]
    elif mode == MaterialMode.M4_LOG_P1:
        if branch == "series":
            controls = [1.0e-7, -0.2, 0.0]
        elif branch == "scaled_endpoints":
            controls = [-100.0, 100.0, 0.0]
        else:
            controls = [0.9, -0.2, 0.0]
    elif branch == "series":
        controls = [1.0e-4, 0.3, -0.1]
    elif branch == "tail":
        controls = [3.0, 28.0, 1.0]
    elif branch == "sharp_interior":
        controls = [1000.0, -1000.0, 250.0]
    else:
        controls = [0.8, -0.35, 0.1]
    return (
        torch.tensor(controls, dtype=DTYPE),
        torch.tensor(1.3, dtype=DTYPE),
        torch.tensor([0.2, 0.7, 0.4], dtype=DTYPE),
        torch.tensor([0.8, 0.1, 0.6], dtype=DTYPE),
    )


def _objective(
    mode: MaterialMode,
    values: list[torch.Tensor],
    grad_tau: torch.Tensor,
    grad_beta: torch.Tensor,
    grad_m: torch.Tensor,
) -> torch.Tensor:
    forward = evaluate_material_segment(mode, *values)
    return (
        grad_tau * forward.tau
        + grad_beta * forward.element.beta
        + torch.dot(grad_m, forward.element.m)
    )


def _central_difference_vjp(
    mode: MaterialMode,
    values: tuple[torch.Tensor, ...],
    grad_tau: torch.Tensor,
    grad_beta: torch.Tensor,
    grad_m: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    gradients = []
    for argument_index, argument in enumerate(values):
        gradient = torch.empty_like(argument)
        for scalar_index in range(argument.numel()):
            plus = [value.clone() for value in values]
            minus = [value.clone() for value in values]
            center = float(argument.reshape(-1)[scalar_index])
            step = 1.0e-6 * max(1.0, abs(center))
            plus[argument_index].reshape(-1)[scalar_index] += step
            minus[argument_index].reshape(-1)[scalar_index] -= step
            gradient.reshape(-1)[scalar_index] = (
                _objective(mode, plus, grad_tau, grad_beta, grad_m)
                - _objective(mode, minus, grad_tau, grad_beta, grad_m)
            ) / (2.0 * step)
        gradients.append(gradient)
    return tuple(gradients)


def _dense_reference(mode, controls, length, color_front, color_back):
    x = torch.linspace(0.0, 1.0, 200_001, dtype=DTYPE)
    if mode in (MaterialMode.M0_P0_CONSTANT, MaterialMode.M1_P0_AFFINE_RGB):
        sigma = torch.ones_like(x) * controls[0]
    elif mode == MaterialMode.M2_POSITIVE_BERNSTEIN_P1:
        sigma = controls[0] * (1.0 - x) + controls[1] * x
    elif mode == MaterialMode.M3_POSITIVE_BERNSTEIN_P2:
        sigma = (
            controls[0] * (1.0 - x).square()
            + 2.0 * controls[1] * x * (1.0 - x)
            + controls[2] * x.square()
        )
    elif mode == MaterialMode.M4_LOG_P1:
        sigma = torch.exp(-(controls[0] * x + controls[1]))
    else:
        sigma = torch.exp(-(controls[0] * x.square() + controls[1] * x + controls[2]))
    tau = length * torch.trapezoid(sigma, x)
    beta = torch.exp(-tau)
    if mode == MaterialMode.M1_P0_AFFINE_RGB:
        color = color_front[None, :] * (1.0 - x[:, None]) + color_back[None, :] * x[:, None]
        # Constant-density transfer from the front endpoint.
        integrand = (
            length
            * sigma[:, None]
            * torch.exp(-length * controls[0] * x)[:, None]
            * color
        )
        m = torch.trapezoid(integrand, x, dim=0)
    else:
        m = (1.0 - beta) * color_front
    return tau, beta, m


@pytest.mark.parametrize("mode", list(MaterialMode))
def test_all_modes_match_independent_dense_integral(mode: MaterialMode) -> None:
    controls, length, color_front, color_back = _inputs(mode)
    actual = evaluate_material_segment(mode, controls, length, color_front, color_back)
    tau, beta, m = _dense_reference(mode, controls, length, color_front, color_back)
    assert torch.allclose(actual.tau, tau, rtol=2e-10, atol=2e-10)
    assert torch.allclose(actual.element.beta, beta, rtol=2e-10, atol=2e-10)
    assert torch.allclose(actual.element.m, m, rtol=2e-10, atol=2e-10)
    x = torch.linspace(0.0, 1.0, 200_001, dtype=DTYPE)
    if mode in (MaterialMode.M0_P0_CONSTANT, MaterialMode.M1_P0_AFFINE_RGB):
        sigma = torch.ones_like(x) * controls[0]
    elif mode == MaterialMode.M2_POSITIVE_BERNSTEIN_P1:
        sigma = controls[0] * (1.0 - x) + controls[1] * x
    elif mode == MaterialMode.M3_POSITIVE_BERNSTEIN_P2:
        sigma = (
            controls[0] * (1.0 - x).square()
            + 2.0 * controls[1] * x * (1.0 - x)
            + controls[2] * x.square()
        )
    elif mode == MaterialMode.M4_LOG_P1:
        sigma = torch.exp(-(controls[0] * x + controls[1]))
    else:
        sigma = torch.exp(-(controls[0] * x.square() + controls[1] * x + controls[2]))
    assert actual.density_bounds[0] <= sigma.min() + 1.0e-12
    assert actual.density_bounds[1] >= sigma.max() - 1.0e-12


@pytest.mark.parametrize(
    ("mode", "branch"),
    [
        (MaterialMode.M0_P0_CONSTANT, "ordinary"),
        (MaterialMode.M1_P0_AFFINE_RGB, "ordinary"),
        (MaterialMode.M2_POSITIVE_BERNSTEIN_P1, "ordinary"),
        (MaterialMode.M3_POSITIVE_BERNSTEIN_P2, "ordinary"),
        (MaterialMode.M4_LOG_P1, "series"),
        (MaterialMode.M4_LOG_P1, "ordinary"),
        (MaterialMode.M5_CONVEX_LOG_P2, "series"),
        (MaterialMode.M5_CONVEX_LOG_P2, "ordinary"),
        (MaterialMode.M5_CONVEX_LOG_P2, "tail"),
    ],
)
def test_explicit_vjp_matches_autograd(mode: MaterialMode, branch: str) -> None:
    controls, length, color_front, color_back = [
        value.clone().requires_grad_(True) for value in _inputs(mode, branch=branch)
    ]
    grad_tau = torch.tensor(0.31, dtype=DTYPE)
    grad_beta = torch.tensor(-0.23, dtype=DTYPE)
    grad_m = torch.tensor([0.2, -0.4, 0.5], dtype=DTYPE)
    forward = evaluate_material_segment(mode, controls, length, color_front, color_back)
    loss = (
        grad_tau * forward.tau
        + grad_beta * forward.element.beta
        + torch.dot(grad_m, forward.element.m)
    )
    expected_raw = torch.autograd.grad(
        loss,
        (controls, color_front, color_back, length),
        allow_unused=True,
    )
    expected = tuple(
        torch.zeros_like(value) if gradient is None else gradient
        for gradient, value in zip(
            expected_raw,
            (controls, color_front, color_back, length),
            strict=True,
        )
    )
    actual = material_segment_vjp(
        mode,
        controls,
        length,
        color_front,
        color_back,
        grad_tau=grad_tau,
        grad_beta=grad_beta,
        grad_m=grad_m,
    )
    for got, want in zip(
        (actual.density_controls, actual.color_front, actual.color_back, actual.length),
        expected,
        strict=True,
    ):
        assert torch.allclose(got, want, rtol=3e-9, atol=3e-10)


@pytest.mark.parametrize("mode", list(MaterialMode))
def test_explicit_vjp_matches_independent_central_difference(
    mode: MaterialMode,
) -> None:
    values = _inputs(mode)
    grad_tau = torch.tensor(0.31, dtype=DTYPE)
    grad_beta = torch.tensor(-0.23, dtype=DTYPE)
    grad_m = torch.tensor([0.2, -0.4, 0.5], dtype=DTYPE)
    finite_difference = _central_difference_vjp(
        mode,
        values,
        grad_tau,
        grad_beta,
        grad_m,
    )
    controls, length, color_front, color_back = values
    explicit = material_segment_vjp(
        mode,
        controls,
        length,
        color_front,
        color_back,
        grad_tau=grad_tau,
        grad_beta=grad_beta,
        grad_m=grad_m,
    )
    for actual, expected in zip(
        (
            explicit.density_controls,
            explicit.length,
            explicit.color_front,
            explicit.color_back,
        ),
        finite_difference,
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, rtol=2.0e-7, atol=2.0e-9)


def test_nonzero_tiny_tau_record_exercises_series_in_forward_and_vjp() -> None:
    values = _inputs(MaterialMode.M0_P0_CONSTANT, branch="tiny_tau")
    forward = evaluate_material_segment(MaterialMode.M0_P0_CONSTANT, *values)
    vjp = material_segment_vjp(
        MaterialMode.M0_P0_CONSTANT,
        *values,
        grad_tau=0.31,
        grad_beta=-0.23,
        grad_m=torch.tensor([0.2, -0.4, 0.5], dtype=DTYPE),
    )
    assert 0.0 < forward.tau < 1.0e-4
    assert forward.branch_status & BranchStatus.SMALL_TAU_SERIES
    assert vjp.branch_status & BranchStatus.SMALL_TAU_SERIES


def test_cpu_gate_records_independent_vjp_and_nonzero_tiny_tau_coverage() -> None:
    report, metal_inputs = build_cpu_report()
    tiny_tau_records = [
        record
        for record in report["records"]
        if record["branch_fixture"] == "tiny_tau"
    ]
    assert report["gate"]["passed"]
    assert report["max_finite_difference_vjp_normalized_error"] <= 1.0e-7
    assert report["branch_counts"]["small_tau_series"] == 1
    assert len(tiny_tau_records) == 1
    assert 0.0 < tiny_tau_records[0]["tau"] < 1.0e-4
    assert len(metal_inputs["modes"]) == report["segment_count"]


def test_small_tau_affine_color_is_finite_and_has_correct_zero_limit() -> None:
    controls, length, color_front, color_back = _inputs(MaterialMode.M1_P0_AFFINE_RGB)
    controls[0] = 0.0
    forward = evaluate_material_segment(
        MaterialMode.M1_P0_AFFINE_RGB, controls, length, color_front, color_back
    )
    vjp = material_segment_vjp(
        MaterialMode.M1_P0_AFFINE_RGB,
        controls,
        length,
        color_front,
        color_back,
        grad_m=torch.ones(3, dtype=DTYPE),
    )
    assert forward.tau == 0.0
    assert forward.element.beta == 1.0
    assert torch.equal(forward.element.m, torch.zeros(3, dtype=DTYPE))
    assert all(torch.isfinite(value).all() for value in vars(vjp).values() if isinstance(value, torch.Tensor))
    assert forward.branch_status & BranchStatus.SMALL_TAU_SERIES


def test_positive_and_convexity_controls_are_enforced() -> None:
    controls, length, color_front, color_back = _inputs(MaterialMode.M3_POSITIVE_BERNSTEIN_P2)
    controls[1] = -0.01
    with pytest.raises(ValueError, match="nonnegative"):
        evaluate_material_segment(
            MaterialMode.M3_POSITIVE_BERNSTEIN_P2,
            controls,
            length,
            color_front,
            color_back,
        )

    controls, length, color_front, color_back = _inputs(MaterialMode.M5_CONVEX_LOG_P2)
    controls[0] = -0.01
    with pytest.raises(ValueError, match="convex"):
        evaluate_material_segment(
            MaterialMode.M5_CONVEX_LOG_P2,
            controls,
            length,
            color_front,
            color_back,
        )


def test_log_quadratic_branches_and_status_counters_are_explicit() -> None:
    statuses = []
    for branch in ("series", "ordinary", "tail"):
        controls, length, color_front, color_back = _inputs(
            MaterialMode.M5_CONVEX_LOG_P2, branch=branch
        )
        statuses.append(
            evaluate_material_segment(
                MaterialMode.M5_CONVEX_LOG_P2,
                controls,
                length,
                color_front,
                color_back,
            ).branch_status
        )
    counts = branch_status_counts(statuses)
    assert counts["total"] == 3
    assert counts["log_quadratic_series"] == 1
    assert counts["log_quadratic_erf"] == 1
    assert counts["log_quadratic_tail"] == 1
    assert counts["invalid_input"] == 0


@pytest.mark.parametrize("curvature", [1000.0, 10_000.0])
def test_sharp_interior_log_quadratic_peak_uses_sign_aware_erf(
    curvature: float,
) -> None:
    # q(x) = a (x-1/2)^2 is narrow but entirely legal. A fixed GL16 rule
    # under-resolves this family, so the sign-straddling analytic branch is
    # required.
    controls = torch.tensor(
        [curvature, -curvature, 0.25 * curvature],
        dtype=DTYPE,
    )
    length = torch.ones((), dtype=DTYPE)
    color = torch.tensor([0.2, 0.4, 0.8], dtype=DTYPE)
    actual = evaluate_material_segment(
        MaterialMode.M5_CONVEX_LOG_P2,
        controls,
        length,
        color,
        color,
    )
    expected_tau = (
        torch.sqrt(torch.tensor(torch.pi / curvature, dtype=DTYPE))
        * torch.erf(torch.tensor(0.5 * curvature**0.5, dtype=DTYPE))
    )
    torch.testing.assert_close(actual.tau, expected_tau, rtol=2.0e-13, atol=1.0e-15)
    assert actual.branch_status & BranchStatus.LOG_QUADRATIC_ERF


def test_log_linear_endpoint_scaling_avoids_split_exponential_overflow() -> None:
    # The old implementation formed exp(100) and exp(-100) separately even
    # though the physical endpoint densities are exp(-100) and 1.
    controls = torch.tensor([-100.0, 100.0, 0.0], dtype=DTYPE)
    length = torch.tensor(1.3, dtype=DTYPE)
    color = torch.tensor([0.2, 0.4, 0.8], dtype=DTYPE)
    actual = evaluate_material_segment(
        MaterialMode.M4_LOG_P1,
        controls,
        length,
        color,
        color,
    )
    expected_tau = length * (torch.exp(torch.tensor(-100.0, dtype=DTYPE)) - 1.0) / -100.0
    torch.testing.assert_close(actual.tau, expected_tau, rtol=2.0e-13, atol=1.0e-15)
    vjp = material_segment_vjp(
        MaterialMode.M4_LOG_P1,
        controls,
        length,
        color,
        color,
        grad_tau=1.0,
    )
    assert all(
        torch.isfinite(value).all()
        for value in (
            vjp.density_controls,
            vjp.color_front,
            vjp.color_back,
            vjp.length,
        )
    )


def test_m1_none_color_back_accumulates_the_aliased_endpoint_vjp() -> None:
    controls, length, color_front, _ = _inputs(MaterialMode.M1_P0_AFFINE_RGB)
    color_front = color_front.clone().requires_grad_(True)
    grad_m = torch.tensor([0.2, -0.4, 0.5], dtype=DTYPE)
    forward = evaluate_material_segment(
        MaterialMode.M1_P0_AFFINE_RGB,
        controls,
        length,
        color_front,
    )
    expected = torch.autograd.grad(
        torch.dot(grad_m, forward.element.m),
        color_front,
    )[0]
    actual = material_segment_vjp(
        MaterialMode.M1_P0_AFFINE_RGB,
        controls,
        length,
        color_front,
        grad_m=grad_m,
    )
    torch.testing.assert_close(actual.color_front, expected)
    assert torch.equal(actual.color_back, torch.zeros_like(actual.color_back))


def test_same_sign_m5_arguments_use_accurate_scaled_tail_moments() -> None:
    # These arguments are individually modest but same-sign.  Subtracting two
    # approximate erf values amplified absolute approximation error in the
    # coefficient VJP, so this case must use the scaled-tail identity.
    controls = torch.tensor([0.05, 1.0, 5.0], dtype=DTYPE)
    length = torch.tensor(1.3, dtype=DTYPE)
    color = torch.tensor([0.2, 0.4, 0.8], dtype=DTYPE)
    actual = evaluate_material_segment(
        MaterialMode.M5_CONVEX_LOG_P2,
        controls,
        length,
        color,
        color,
    )
    expected_tau, expected_beta, expected_m = _dense_reference(
        MaterialMode.M5_CONVEX_LOG_P2,
        controls,
        length,
        color,
        color,
    )
    torch.testing.assert_close(actual.tau, expected_tau, rtol=2.0e-10, atol=2.0e-12)
    torch.testing.assert_close(
        actual.element.beta,
        expected_beta,
        rtol=2.0e-10,
        atol=2.0e-12,
    )
    torch.testing.assert_close(
        actual.element.m,
        expected_m,
        rtol=2.0e-10,
        atol=2.0e-12,
    )
    assert actual.branch_status & BranchStatus.LOG_QUADRATIC_TAIL


def test_reference_rejects_vector_shaped_length_instead_of_failing_later() -> None:
    controls, _, color_front, color_back = _inputs(MaterialMode.M0_P0_CONSTANT)
    with pytest.raises(ValueError, match="zero-dimensional"):
        evaluate_material_segment(
            MaterialMode.M0_P0_CONSTANT,
            controls,
            torch.ones(1, dtype=DTYPE),
            color_front,
            color_back,
        )


def test_forward_and_vjp_both_reject_nonfinite_numerical_domain() -> None:
    controls = torch.tensor([0.0, -1000.0, 0.0], dtype=DTYPE)
    length = torch.ones((), dtype=DTYPE)
    color = torch.ones(3, dtype=DTYPE)
    with pytest.raises(FloatingPointError, match="overflowed"):
        evaluate_material_segment(
            MaterialMode.M4_LOG_P1,
            controls,
            length,
            color,
            color,
        )
    with pytest.raises(FloatingPointError, match="overflowed"):
        material_segment_vjp(
            MaterialMode.M4_LOG_P1,
            controls,
            length,
            color,
            color,
            grad_tau=1.0,
        )


def test_metal_source_has_parameterized_forward_vjp_and_local_special_functions() -> None:
    source = SOURCE_PATH.read_text()
    assert "kernel void worldfoam_material_forward" in source
    assert "kernel void worldfoam_material_vjp" in source
    assert "quadratic_moments" in source
    assert "BRANCH_LOG_QUADRATIC_TAIL" in source
    assert re.search(r"\berf\s*\(", source) is None
    assert re.search(r"\bexpm1\s*\(", source) is None


def test_metal_wrapper_lazily_calls_runtime_compiler(monkeypatch) -> None:
    compiled = []
    sentinel = object()

    def fake_compile(source: str):
        compiled.append(source)
        return sentinel

    monkeypatch.setattr(torch.mps, "compile_shader", fake_compile)
    wrapper = FiniteElementMaterialMetal()
    assert wrapper.compile() is sentinel
    assert wrapper.compile() is sentinel
    assert len(compiled) == 1
    assert "worldfoam_material_forward" in compiled[0]


def test_metal_wrapper_never_silently_accepts_invalid_rows() -> None:
    status = torch.tensor(
        [0, int(BranchStatus.INVALID_INPUT)],
        dtype=torch.int32,
    )
    with pytest.raises(FloatingPointError, match=r"rows \[1\]"):
        FiniteElementMaterialMetal._raise_on_invalid(status)


def test_metal_vjp_cotangent_validator_rejects_dtype_and_device_mismatch() -> None:
    with pytest.raises(ValueError, match="float32"):
        FiniteElementMaterialMetal._validate_cotangents(
            1,
            torch.ones(1, dtype=torch.float64),
            torch.ones(1, dtype=torch.float64),
            torch.ones(1, 3, dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="MPS tensors"):
        FiniteElementMaterialMetal._validate_cotangents(
            1,
            torch.ones(1, dtype=torch.float32),
            torch.ones(1, dtype=torch.float32),
            torch.ones(1, 3, dtype=torch.float32),
        )


@pytest.mark.skipif(
    not torch.backends.mps.is_available() or os.environ.get("WORLD_FOAM_RUN_MPS_COMPILE") != "1",
    reason="opt-in tiny Metal compile smoke; no publication-scale MPS work",
)
def test_metal_source_runtime_compiles_when_opted_in() -> None:
    library = torch.mps.compile_shader(Path(SOURCE_PATH).read_text())
    assert hasattr(library, "worldfoam_material_forward")
    assert hasattr(library, "worldfoam_material_vjp")


@pytest.mark.skipif(
    not torch.backends.mps.is_available() or os.environ.get("WORLD_FOAM_RUN_MPS_PARITY") != "1",
    reason="opt-in sub-kilobyte fixed-segment Metal parity; never a training/paper run",
)
def test_tiny_metal_forward_and_vjp_match_cpu_reference() -> None:
    cases = [
        (MaterialMode.M0_P0_CONSTANT, "ordinary"),
        (MaterialMode.M0_P0_CONSTANT, "tiny_tau"),
        (MaterialMode.M1_P0_AFFINE_RGB, "ordinary"),
        (MaterialMode.M2_POSITIVE_BERNSTEIN_P1, "ordinary"),
        (MaterialMode.M3_POSITIVE_BERNSTEIN_P2, "ordinary"),
        (MaterialMode.M4_LOG_P1, "series"),
        (MaterialMode.M4_LOG_P1, "ordinary"),
        (MaterialMode.M4_LOG_P1, "scaled_endpoints"),
        (MaterialMode.M5_CONVEX_LOG_P2, "series"),
        (MaterialMode.M5_CONVEX_LOG_P2, "ordinary"),
        (MaterialMode.M5_CONVEX_LOG_P2, "tail"),
        (MaterialMode.M5_CONVEX_LOG_P2, "sharp_interior"),
    ]
    cpu_inputs = [_inputs(mode, branch=branch) for mode, branch in cases]
    controls = torch.stack([item[0].float() for item in cpu_inputs])
    lengths = torch.stack([item[1].float() for item in cpu_inputs])
    color_front = torch.stack([item[2].float() for item in cpu_inputs])
    color_back = torch.stack([item[3].float() for item in cpu_inputs])
    modes = torch.tensor([int(mode) for mode, _ in cases], dtype=torch.int32)
    grad_tau = torch.linspace(-0.2, 0.3, len(cases))
    grad_beta = torch.linspace(0.4, -0.1, len(cases))
    grad_m = torch.linspace(-0.5, 0.6, len(cases) * 3).reshape(len(cases), 3)

    wrapper = FiniteElementMaterialMetal()
    actual_forward = wrapper.forward(
        controls.to("mps"),
        lengths.to("mps"),
        color_front.to("mps"),
        color_back.to("mps"),
        modes.to("mps"),
    )
    actual_vjp = wrapper.vjp(
        controls.to("mps"),
        lengths.to("mps"),
        color_front.to("mps"),
        color_back.to("mps"),
        modes.to("mps"),
        grad_tau.to("mps"),
        grad_beta.to("mps"),
        grad_m.to("mps"),
    )

    expected_forward = [
        evaluate_material_segment(mode, *values)
        for (mode, _), values in zip(cases, cpu_inputs, strict=True)
    ]
    expected_vjp = [
        material_segment_vjp(
            mode,
            *values,
            grad_tau=grad_tau[index],
            grad_beta=grad_beta[index],
            grad_m=grad_m[index].double(),
        )
        for index, ((mode, _), values) in enumerate(zip(cases, cpu_inputs, strict=True))
    ]
    expected = {
        "tau": torch.stack([value.tau.float() for value in expected_forward]),
        "beta": torch.stack([value.element.beta.float() for value in expected_forward]),
        "m": torch.stack([value.element.m.float() for value in expected_forward]),
        "density_bounds": torch.stack(
            [value.density_bounds.float() for value in expected_forward]
        ),
        "density_controls": torch.stack([value.density_controls.float() for value in expected_vjp]),
        "color_front": torch.stack([value.color_front.float() for value in expected_vjp]),
        "color_back": torch.stack([value.color_back.float() for value in expected_vjp]),
        "length": torch.stack([value.length.float() for value in expected_vjp]),
    }
    for key in ("tau", "beta", "m", "density_bounds"):
        assert torch.allclose(actual_forward[key].cpu(), expected[key], rtol=2e-4, atol=2e-5), key
    for key in ("density_controls", "color_front", "color_back", "length"):
        assert torch.allclose(actual_vjp[key].cpu(), expected[key], rtol=4e-4, atol=4e-5), key
    assert torch.equal(actual_forward["status"].cpu(), actual_vjp["status"].cpu())
    tiny_tau_status = int(actual_forward["status"][1].cpu())
    assert tiny_tau_status & int(BranchStatus.SMALL_TAU_SERIES)


@pytest.mark.skipif(
    not torch.backends.mps.is_available() or os.environ.get("WORLD_FOAM_RUN_MPS_PARITY") != "1",
    reason="opt-in sub-kilobyte fixed-segment Metal rejection; never a training/paper run",
)
def test_tiny_metal_shader_invalid_row_is_rejected_by_host() -> None:
    controls = torch.tensor([[-0.1, 0.0, 0.0]], dtype=torch.float32, device="mps")
    lengths = torch.ones(1, dtype=torch.float32, device="mps")
    color_front = torch.ones(1, 3, dtype=torch.float32, device="mps")
    color_back = torch.zeros(1, 3, dtype=torch.float32, device="mps")
    modes = torch.tensor(
        [int(MaterialMode.M0_P0_CONSTANT)],
        dtype=torch.int32,
        device="mps",
    )
    wrapper = FiniteElementMaterialMetal()
    with pytest.raises(FloatingPointError, match=r"rejected rows \[0\]"):
        wrapper.forward(controls, lengths, color_front, color_back, modes)
    with pytest.raises(FloatingPointError, match=r"rejected rows \[0\]"):
        wrapper.vjp(
            controls,
            lengths,
            color_front,
            color_back,
            modes,
            torch.ones_like(lengths),
            torch.ones_like(lengths),
            torch.ones_like(color_front),
        )
