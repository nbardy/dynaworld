from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

STAR_UVT_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "fast-mac-gsplat" / "variants" / "star_uvt_v0"
if str(STAR_UVT_ROOT) not in sys.path:
    sys.path.insert(0, str(STAR_UVT_ROOT))

from torch_gsplat_bridge_star_uvt import (  # noqa: E402
    eval_projective_trace,
    eval_projective_trace_polynomial_fit,
    eval_projective_trace_torch,
    fit_projective_trace_polynomial,
    has_projective_trace_metal,
    split_projective_trace_windows,
)


def _sample_coeffs(device: torch.device | str = "cpu") -> torch.Tensor:
    return torch.tensor(
        [
            [12.0, 1.5, -0.25, 8.0, -0.75, 0.125, 2.0, 0.10, 0.025],
            [-3.0, 4.0, 0.50, 6.0, 0.25, -0.375, 1.0, -0.40, 0.050],
            [1.0, 0.0, 0.0, -2.0, 1.0, 0.0, 0.0, 0.02, 0.0],
        ],
        dtype=torch.float32,
        device=device,
    ).contiguous()


def test_eval_projective_trace_torch_matches_manual_rational_formula() -> None:
    coeffs = _sample_coeffs()
    times = torch.tensor([-1.0, 0.0, 0.5, 2.0], dtype=torch.float32)

    out = eval_projective_trace_torch(coeffs, times, eps=1.0e-4)

    t = times[2]
    hu = coeffs[0, 0] + coeffs[0, 1] * t + coeffs[0, 2] * t * t
    hv = coeffs[0, 3] + coeffs[0, 4] * t + coeffs[0, 5] * t * t
    hz = coeffs[0, 6] + coeffs[0, 7] * t + coeffs[0, 8] * t * t
    assert out[0, 2, 0] == pytest.approx(float(hu / hz))
    assert out[0, 2, 1] == pytest.approx(float(hv / hz))
    assert out[0, 2, 2] == pytest.approx(float(hz))
    assert out[0, 2, 3] == 1.0


def test_eval_projective_trace_torch_marks_denominator_boundaries() -> None:
    coeffs = _sample_coeffs()
    times = torch.tensor([0.0], dtype=torch.float32)

    out = eval_projective_trace_torch(coeffs, times, eps=1.0e-4)

    assert out[2, 0, 0] == 0.0
    assert out[2, 0, 1] == 0.0
    assert out[2, 0, 2] == 0.0
    assert out[2, 0, 3] == 0.0


def test_eval_projective_trace_uses_torch_fallback_on_cpu() -> None:
    coeffs = _sample_coeffs()
    times = torch.linspace(-1.0, 1.0, 5, dtype=torch.float32)

    assert torch.allclose(eval_projective_trace(coeffs, times), eval_projective_trace_torch(coeffs, times))


def test_eval_projective_trace_metal_matches_torch_if_available() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if not has_projective_trace_metal():
        pytest.skip("projective trace Metal op unavailable")

    coeffs = _sample_coeffs("mps")
    times = torch.linspace(-1.0, 1.0, 7, dtype=torch.float32, device="mps").contiguous()

    metal = eval_projective_trace(coeffs, times).cpu()
    ref = eval_projective_trace_torch(coeffs.cpu(), times.cpu())

    assert torch.allclose(metal, ref, atol=1.0e-5, rtol=1.0e-5)


def test_fit_projective_trace_polynomial_is_exact_for_affine_screen_trace() -> None:
    coeffs = torch.tensor(
        [[2.0, 0.5, 0.0, -3.0, 0.25, 0.0, 2.0, 0.0, 0.0]],
        dtype=torch.float32,
    ).contiguous()
    times = torch.linspace(-1.0, 1.0, 9, dtype=torch.float32).contiguous()

    fit = fit_projective_trace_polynomial(coeffs, times, degree=1)
    pred = eval_projective_trace_polynomial_fit(fit, times)
    ref = eval_projective_trace_torch(coeffs, times)[:, :, :3]

    assert torch.allclose(pred, ref, atol=1.0e-5, rtol=1.0e-5)
    assert fit.residual_max_uv[0] == pytest.approx(0.0, abs=1.0e-5)
    assert fit.residual_max_depth[0] == pytest.approx(0.0, abs=1.0e-5)
    assert fit.valid_fraction[0] == 1.0
    assert fit.denominator_min_abs[0] == pytest.approx(2.0)


def test_fit_projective_trace_polynomial_reports_curvature_residual() -> None:
    coeffs = torch.tensor(
        [[0.0, 0.0, 1.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0]],
        dtype=torch.float32,
    ).contiguous()
    times = torch.linspace(-1.0, 1.0, 9, dtype=torch.float32).contiguous()

    affine_fit = fit_projective_trace_polynomial(coeffs, times, degree=1)
    quadratic_fit = fit_projective_trace_polynomial(coeffs, times, degree=2)

    assert affine_fit.residual_max_uv[0] > 0.1
    assert quadratic_fit.residual_max_uv[0] == pytest.approx(0.0, abs=1.0e-5)


def test_fit_projective_trace_polynomial_marks_underconstrained_traces() -> None:
    coeffs = torch.tensor(
        [[1.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 1.0, 0.0]],
        dtype=torch.float32,
    ).contiguous()
    times = torch.tensor([0.0, 1.0], dtype=torch.float32).contiguous()

    fit = fit_projective_trace_polynomial(coeffs, times, degree=2, eps=1.0e-4)

    assert fit.valid_count[0] == 1
    assert fit.valid_fraction[0] == pytest.approx(0.5)
    assert torch.isinf(fit.residual_max_uv[0])


def test_split_projective_trace_windows_accepts_one_affine_window() -> None:
    coeffs = torch.tensor(
        [[2.0, 0.5, 0.0, -3.0, 0.25, 0.0, 2.0, 0.0, 0.0]],
        dtype=torch.float32,
    ).contiguous()
    times = torch.linspace(-1.0, 1.0, 17, dtype=torch.float32).contiguous()

    windows = split_projective_trace_windows(coeffs, times, degree=1, max_residual_uv=1.0e-4)

    assert len(windows) == 1
    assert windows[0].accepted
    assert windows[0].start == 0
    assert windows[0].stop == times.numel()


def test_split_projective_trace_windows_splits_curved_trace_until_local() -> None:
    coeffs = torch.tensor(
        [[0.0, 0.0, 1.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0]],
        dtype=torch.float32,
    ).contiguous()
    times = torch.linspace(-1.0, 1.0, 17, dtype=torch.float32).contiguous()

    loose = split_projective_trace_windows(coeffs, times, degree=1, max_residual_uv=2.0)
    tight = split_projective_trace_windows(coeffs, times, degree=1, max_residual_uv=0.05)

    assert len(loose) == 1
    assert len(tight) > 1
    assert all(window.accepted for window in tight)


def test_split_projective_trace_windows_marks_denominator_boundary() -> None:
    coeffs = torch.tensor(
        [[1.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 1.0, 0.0]],
        dtype=torch.float32,
    ).contiguous()
    times = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32).contiguous()

    windows = split_projective_trace_windows(
        coeffs,
        times,
        degree=1,
        max_residual_uv=1.0e-4,
        min_denominator_abs=1.0e-4,
    )

    assert len(windows) == 2
    assert any(not window.accepted for window in windows)
    assert any("invalid_samples" in window.reason for window in windows)
