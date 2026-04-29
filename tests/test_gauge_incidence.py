from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "train"))
sys.path.insert(0, str(ROOT / "research_experiments" / "gauge_fields"))

from incidence import (  # noqa: E402
    compact_poly_ellipsoid_optical_depth,
    ray_gaussian_line_optical_depth,
    validate_incidence_mode,
)
from train import MaterialSurfelField, gauge_config  # noqa: E402
from cheat_probe_material_gauge import (  # noqa: E402
    probe_graph_expansion,
    probe_neighborhood_support_shuffle,
    probe_xmap_shuffle,
)


class ProbeArgs:
    sample_fraction = 1.0
    seed = 0
    graph_expansion_alpha_logit = -8.0


def test_mass_normalized_isotropic_whole_line_matches_closed_form() -> None:
    sigma = 0.2
    mass = 0.7
    offset = 0.3
    mu = torch.tensor([[0.0, 0.0, 0.0]])
    cov = torch.eye(3).unsqueeze(0) * (sigma * sigma)
    origin = torch.tensor([[0.0, offset, 0.0]])
    ray_dirs = torch.tensor([[0.0, 0.0, 1.0]])

    tau = ray_gaussian_line_optical_depth(
        ray_origin=origin[0],
        ray_dirs=ray_dirs,
        s0=-20.0,
        s1=20.0,
        mu=mu,
        cov3=cov,
        strength=torch.tensor([mass]),
        mass_normalized=True,
    )[0, 0]

    expected = mass / (2.0 * math.pi * sigma * sigma) * math.exp(
        -(offset * offset) / (2.0 * sigma * sigma)
    )
    assert torch.allclose(tau, torch.tensor(expected), rtol=2e-4, atol=2e-4)


def test_peak_density_line_integral_matches_numeric_quadrature() -> None:
    mu = torch.tensor([[0.1, -0.2, 0.3]])
    cov = torch.tensor(
        [
            [
                [0.12, 0.01, 0.0],
                [0.01, 0.08, 0.02],
                [0.0, 0.02, 0.10],
            ]
        ],
        dtype=torch.float32,
    )
    origin = torch.tensor([0.3, -0.5, -0.7])
    ray_dirs = torch.tensor([[0.2, 0.1, 1.0]])
    ray_dirs = ray_dirs / ray_dirs.norm(dim=-1, keepdim=True)
    rho = torch.tensor([0.9])
    s0, s1 = -1.5, 2.0

    closed = ray_gaussian_line_optical_depth(
        ray_origin=origin,
        ray_dirs=ray_dirs,
        s0=s0,
        s1=s1,
        mu=mu,
        cov3=cov,
        strength=rho,
        mass_normalized=False,
    )[0, 0]

    samples = torch.linspace(s0, s1, 4096)
    pts = origin[None, :] + samples[:, None] * ray_dirs[0][None, :]
    inv_cov = torch.linalg.inv(cov[0])
    diff = pts - mu[0][None, :]
    exponent = -0.5 * torch.einsum("bi,ij,bj->b", diff, inv_cov, diff)
    numeric = torch.trapz(rho[0] * torch.exp(exponent), samples)

    assert torch.allclose(closed, numeric, rtol=2e-3, atol=2e-4)


def test_ray_gaussian_tau_is_rigid_invariant() -> None:
    mu = torch.tensor([[0.15, -0.2, 0.4]])
    cov = torch.tensor([[[0.08, 0.01, 0.0], [0.01, 0.05, 0.01], [0.0, 0.01, 0.07]]])
    origin = torch.tensor([0.2, -0.1, -1.0])
    ray_dirs = torch.tensor([[0.1, -0.05, 1.0]])
    ray_dirs = ray_dirs / ray_dirs.norm(dim=-1, keepdim=True)
    mass = torch.tensor([0.6])

    base = ray_gaussian_line_optical_depth(
        ray_origin=origin,
        ray_dirs=ray_dirs,
        s0=0.0,
        s1=3.0,
        mu=mu,
        cov3=cov,
        strength=mass,
        mass_normalized=True,
    )

    theta = torch.tensor(0.7)
    R = torch.tensor(
        [
            [torch.cos(theta), -torch.sin(theta), 0.0],
            [torch.sin(theta), torch.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    t = torch.tensor([0.4, -0.3, 0.2])
    transformed = ray_gaussian_line_optical_depth(
        ray_origin=origin @ R.T + t,
        ray_dirs=ray_dirs @ R.T,
        s0=0.0,
        s1=3.0,
        mu=mu @ R.T + t,
        cov3=R @ cov @ R.T,
        strength=mass,
        mass_normalized=True,
    )

    assert torch.allclose(base, transformed, rtol=2e-5, atol=2e-6)


def test_compact_poly_ellipsoid_matches_numeric_quadrature() -> None:
    mu = torch.tensor([[0.1, -0.2, 0.7]], dtype=torch.float32)
    precision = torch.tensor(
        [
            [
                [2.4, 0.2, -0.1],
                [0.2, 1.8, 0.15],
                [-0.1, 0.15, 1.3],
            ]
        ],
        dtype=torch.float32,
    )
    origin = torch.tensor([0.05, -0.1, -0.3], dtype=torch.float32)
    ray_dirs = torch.tensor([[0.08, -0.04, 1.0]], dtype=torch.float32)
    ray_dirs = ray_dirs / ray_dirs.norm(dim=-1, keepdim=True)
    beta = torch.tensor([0.9], dtype=torch.float32)
    s0, s1 = 0.0, 3.0

    closed = compact_poly_ellipsoid_optical_depth(
        ray_origin=origin,
        ray_dirs=ray_dirs,
        s0=s0,
        s1=s1,
        mu=mu,
        precision3=precision,
        beta=beta,
        power=2,
    )[0, 0]

    samples = torch.linspace(s0, s1, 8192)
    pts = origin[None, :] + samples[:, None] * ray_dirs[0][None, :]
    diff = pts - mu[0][None, :]
    r2 = torch.einsum("bi,ij,bj->b", diff, precision[0], diff)
    density = beta[0] * (1.0 - r2).clamp_min(0.0).pow(2)
    numeric = torch.trapz(density, samples)

    assert torch.allclose(closed, numeric, rtol=2e-3, atol=2e-4)


def test_compact_poly_ellipsoid_radial_gauge_is_invariant() -> None:
    mu = torch.tensor([[0.2, -0.1, 3.0]], dtype=torch.float32)
    precision = torch.tensor(
        [
            [
                [7.0, 0.6, 0.2],
                [0.6, 5.5, -0.3],
                [0.2, -0.3, 4.0],
            ]
        ],
        dtype=torch.float32,
    )
    beta = torch.tensor([0.75], dtype=torch.float32)
    dirs = torch.tensor(
        [
            [-0.08, -0.05, 1.0],
            [0.0, 0.0, 1.0],
            [0.06, -0.02, 1.0],
            [0.09, 0.04, 1.0],
        ],
        dtype=torch.float32,
    )
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    origin = torch.zeros(3, dtype=torch.float32)
    base = compact_poly_ellipsoid_optical_depth(
        ray_origin=origin,
        ray_dirs=dirs,
        s0=0.0,
        s1=20.0,
        mu=mu,
        precision3=precision,
        beta=beta,
        power=2,
    )

    scale = 1.4
    shifted = compact_poly_ellipsoid_optical_depth(
        ray_origin=origin,
        ray_dirs=dirs,
        s0=0.0,
        s1=20.0,
        mu=scale * mu,
        precision3=(scale ** -2) * precision,
        beta=(scale ** -1) * beta,
        power=2,
    )

    assert torch.allclose(base, shifted, rtol=2e-5, atol=2e-6)


def test_compact_poly_ellipsoid_removes_projected_covariance_null_direction() -> None:
    mu = torch.tensor([0.2, -0.1, 3.0], dtype=torch.float32)
    cov = torch.tensor(
        [
            [0.12, 0.015, 0.01],
            [0.015, 0.09, -0.012],
            [0.01, -0.012, 0.07],
        ],
        dtype=torch.float32,
    )
    precision = torch.linalg.inv(cov)
    axis = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    cov_null = mu[:, None] @ axis[None, :] + axis[:, None] @ mu[None, :]

    x, y, z = mu
    jac = torch.tensor(
        [
            [1.0 / z, 0.0, -x / (z * z)],
            [0.0, 1.0 / z, -y / (z * z)],
        ],
        dtype=torch.float32,
    )
    projected_before = jac @ cov @ jac.T
    projected_after = jac @ (cov + 1.0e-3 * cov_null) @ jac.T
    assert torch.allclose(projected_before, projected_after, atol=2e-8)

    precision_delta = -precision @ cov_null @ precision
    rays = torch.tensor(
        [
            [-0.02, -0.04, 1.0],
            [0.03, -0.03, 1.0],
            [0.08, 0.00, 1.0],
            [0.12, 0.04, 1.0],
        ],
        dtype=torch.float32,
    )
    rays = rays / rays.norm(dim=-1, keepdim=True)
    kwargs = {
        "ray_origin": torch.zeros(3, dtype=torch.float32),
        "ray_dirs": rays,
        "s0": 0.0,
        "s1": 20.0,
        "mu": mu[None, :],
        "beta": torch.tensor([0.8], dtype=torch.float32),
        "power": 2,
    }
    base = compact_poly_ellipsoid_optical_depth(precision3=precision[None, :, :], **kwargs)
    perturbed = compact_poly_ellipsoid_optical_depth(
        precision3=(precision + 1.0e-5 * precision_delta)[None, :, :],
        **kwargs,
    )

    assert (perturbed - base).abs().max() > 1.0e-6


def test_gauge_config_accepts_incidence_mode_default_and_overrides() -> None:
    cfg = gauge_config(
        {
            "data": {},
            "model": {},
            "camera": {},
            "render": {},
            "train": {},
            "losses": {},
            "logging": {},
        }
    )
    assert cfg["render"]["incidence_mode"] == "projected_conic"
    cfg["model"]["support_mode"] = "derived_support_metric"
    assert gauge_config(cfg)["model"]["support_mode"] == "derived_support_metric"
    assert validate_incidence_mode("ray_gaussian_line_mass") == "ray_gaussian_line_mass"


def test_derived_support_metric_uses_transported_neighbor_covariance() -> None:
    x0 = torch.tensor(
        [
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.5, 0.0],
        ],
        dtype=torch.float32,
    )
    model = MaterialSurfelField(
        init_x0=x0,
        num_frames=1,
        num_basis=0,
        support_mode="derived_support_metric",
        support_knn_k=2,
        derived_support_scale=0.1,
        derived_support_floor=0.01,
        derived_support_normalize_trace=True,
    )
    cov = model.world_support_covariance(model.positions(0))
    assert cov.shape == (4, 3, 3)
    assert torch.isfinite(cov).all()
    assert torch.allclose(cov, cov.transpose(-1, -2))
    eig = torch.linalg.eigvalsh(cov)
    assert torch.all(eig > 0)


def test_neighborhood_support_shuffle_changes_derived_support_graph() -> None:
    x0 = torch.tensor(
        [
            [-1.0, 0.0, 0.0],
            [-0.2, 0.0, 0.0],
            [0.4, 0.1, 0.0],
            [1.2, 0.0, 0.0],
            [0.2, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    model = MaterialSurfelField(
        init_x0=x0,
        num_frames=1,
        num_basis=0,
        support_mode="derived_support_metric",
        support_knn_k=2,
    )
    before = model.support_knn_idx.clone()

    probe_neighborhood_support_shuffle(model, ProbeArgs())

    assert model.support_knn_idx.shape == before.shape
    assert not torch.equal(model.support_knn_idx, before)
    assert set(map(tuple, model.support_knn_idx.tolist())) == set(map(tuple, before.tolist()))


def test_graph_expansion_adds_low_opacity_midpoint_support_anchors() -> None:
    x0 = torch.tensor(
        [
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.5, 0.0],
        ],
        dtype=torch.float32,
    )
    model = MaterialSurfelField(
        init_x0=x0,
        num_frames=1,
        num_basis=0,
        support_mode="derived_support_metric",
        support_knn_k=2,
    )
    expanded = probe_graph_expansion(model, ProbeArgs())

    assert expanded.N == 8
    assert torch.allclose(
        expanded.raw_alpha[4:],
        torch.full_like(expanded.raw_alpha[4:], ProbeArgs.graph_expansion_alpha_logit),
    )
    assert expanded.support_knn_idx.shape == (8, 2)


def test_xmap_shuffle_preserves_positions_when_coefficients_span_frames() -> None:
    x0 = torch.tensor(
        [
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    model = MaterialSurfelField(
        init_x0=x0,
        num_frames=2,
        num_basis=2,
        support_mode="derived_support_metric",
        support_knn_k=2,
    )
    with torch.no_grad():
        model.nr_coeff.copy_(torch.eye(2))
        model.nr_basis[:, 0, :] = torch.tensor([0.05, 0.0, 0.0])
        model.nr_basis[:, 1, :] = torch.tensor([0.0, 0.05, 0.0])
    positions_before = torch.stack([model.positions(t).detach().clone() for t in range(model.T)], dim=0)
    x0_before = model.x0.detach().clone()

    probe_xmap_shuffle(model, ProbeArgs())

    positions_after = torch.stack([model.positions(t).detach().clone() for t in range(model.T)], dim=0)
    assert torch.allclose(positions_after, positions_before, atol=1e-6)
    assert not torch.allclose(model.x0, x0_before)
