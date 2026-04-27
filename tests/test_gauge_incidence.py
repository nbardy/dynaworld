from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "train"))
sys.path.insert(0, str(ROOT / "research_experiments" / "gauge_fields"))

from incidence import ray_gaussian_line_optical_depth, validate_incidence_mode  # noqa: E402
from train import MaterialSurfelField, gauge_config  # noqa: E402


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
