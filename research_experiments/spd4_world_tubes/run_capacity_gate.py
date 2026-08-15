#!/usr/bin/env python3
"""Controlled CPU capacity gate for legacy tubes versus native SPD(4) atoms.

The target is one tilted, full-rank spacetime Gaussian observed through three
static affine camera charts.  Both candidates start from nearly the same
fronto-parallel footprint.  A rank certificate checks that the camera suite
identifies all six conditional spatial covariance degrees of freedom before
the raster optimization is run.

This is a representation-capacity fixture, not a public-scene quality result.
It is matched by atom count and raster workload, not by parameter count.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


ROOT = Path(__file__).resolve().parents[2]
STAR_ROOT = (
    ROOT / "third_party" / "fast-mac-gsplat" / "variants" / "star_uvt_v0"
)
TRAIN_SRC = ROOT / "src" / "train"
for source_root in (STAR_ROOT, TRAIN_SRC):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from research_project.benchmarks.multicam_heldout_compare import (  # noqa: E402
    WorldTubeModel,
    project_world_tube_sequence,
    render_projected_sequence,
)
from research_project.trainer_harness.spd4_world_atom import (  # noqa: E402
    SPD4WorldAtomModel,
)
from torch_gsplat_bridge_star_uvt import UVTRenderConfig  # noqa: E402


SCHEMA_VERSION = 1
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts"
    / "foundation_gates"
    / "spd4_native_multiview_capacity_cpu.json"
)


@dataclass(frozen=True)
class FitResult:
    representation: str
    trainable_scalars: int
    initial_mse: float
    final_mse: float
    final_psnr_db: float
    per_view_mse: list[float]
    elapsed_s: float
    finite_state: bool
    loss_trace: list[dict[str, float | int]]


def _inverse_softplus(value: Tensor) -> Tensor:
    return value + torch.log(-torch.expm1(-value))


def _base_tensors() -> tuple[Tensor, Tensor, Tensor]:
    return (
        torch.tensor([[0.0, 0.0, 3.0]], dtype=torch.float32),
        torch.tensor([[0.85, 0.25, 0.10]], dtype=torch.float32),
        torch.tensor([0.0], dtype=torch.float32),
    )


def _new_spd4(*, init_precision_z: float | None) -> SPD4WorldAtomModel:
    x0, color, t0 = _base_tensors()
    return SPD4WorldAtomModel(
        init_x0=x0,
        init_color=color,
        init_t0=t0,
        frames=5,
        init_precision_xy=25.0,
        init_precision_z=init_precision_z,
        init_lambda_t=0.55,
        init_opacity=0.75,
        min_spatial_scale=1.0e-4,
        min_lambda_t=1.0e-5,
        tilt_reg_weight=0.0,
        depth_tilt_reg_weight=0.0,
        position_reg_weight=0.0,
    )


def _target_spd4() -> SPD4WorldAtomModel:
    model = _new_spd4(init_precision_z=None)
    with torch.no_grad():
        desired_diagonal = torch.tensor(
            [[0.17, 0.12, 0.32]], dtype=torch.float32
        )
        model.raw_spatial_scale.copy_(
            _inverse_softplus(desired_diagonal - model.min_spatial_scale)
        )
        model.spatial_cholesky_offdiag.copy_(
            torch.tensor([[0.07, 0.20, -0.06]], dtype=torch.float32)
        )
        model.space_time_tilt.copy_(
            torch.tensor([[0.055, -0.030, 0.045]], dtype=torch.float32)
        )
    return model


def _legacy_candidate() -> WorldTubeModel:
    x0, color, t0 = _base_tensors()
    return WorldTubeModel(
        init_x0=x0,
        init_color=color,
        init_t0=t0,
        frames=5,
        init_precision_xy=25.0,
        init_lambda_t=0.55,
        init_opacity=0.75,
        min_precision_xy=1.0e-5,
        min_lambda_t=1.0e-5,
        velocity_reg_weight=0.0,
        depth_velocity_reg_weight=0.0,
        position_reg_weight=0.0,
    )


def _camera(angle_radians: float) -> tuple[Tensor, Tensor]:
    x0, _, _ = _base_tensors()
    cosine = math.cos(angle_radians)
    sine = math.sin(angle_radians)
    rotation = torch.tensor(
        (
            (cosine, 0.0, sine),
            (0.0, 1.0, 0.0),
            (-sine, 0.0, cosine),
        ),
        dtype=torch.float32,
    )
    translation = torch.tensor((0.0, 0.0, 3.0), dtype=torch.float32) - (
        rotation @ x0[0]
    )
    world_to_camera = torch.eye(4, dtype=torch.float32)
    world_to_camera[:3, :3] = rotation
    world_to_camera[:3, 3] = translation
    intrinsics = torch.tensor(
        ((42.0, 0.0, 16.0), (0.0, 42.0, 16.0), (0.0, 0.0, 1.0)),
        dtype=torch.float32,
    )
    return intrinsics, world_to_camera


def _camera_suite() -> list[tuple[Tensor, Tensor]]:
    return [_camera(angle) for angle in (-0.65, 0.0, 0.58)]


def _render_views(
    model: WorldTubeModel | SPD4WorldAtomModel,
    cameras: list[tuple[Tensor, Tensor]],
    config: UVTRenderConfig,
) -> list[Tensor]:
    return [
        render_projected_sequence(
            project_world_tube_sequence(model, intrinsics, world_to_camera, config),
            config,
            backend="dense",
        ).rgb
        for intrinsics, world_to_camera in cameras
    ]


def _mean_mse(predictions: list[Tensor], targets: list[Tensor]) -> Tensor:
    return torch.stack(
        [
            (prediction - target).square().mean()
            for prediction, target in zip(predictions, targets, strict=True)
        ]
    ).mean()


def _fit(
    model: WorldTubeModel | SPD4WorldAtomModel,
    targets: list[Tensor],
    cameras: list[tuple[Tensor, Tensor]],
    config: UVTRenderConfig,
    *,
    steps: int,
    learning_rate: float,
) -> FitResult:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    trace_steps = {0, steps - 1}
    trace_steps.update(range(24, steps, 25))
    loss_trace: list[dict[str, float | int]] = []
    started_at = time.perf_counter()
    initial_mse = math.nan
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        predictions = _render_views(model, cameras, config)
        loss = _mean_mse(predictions, targets)
        if step == 0:
            initial_mse = float(loss.detach())
        if not bool(torch.isfinite(loss).detach()):
            raise RuntimeError(
                f"{model.representation_name} produced non-finite loss at step {step}"
            )
        loss.backward()
        optimizer.step()
        if step in trace_steps:
            loss_trace.append({"step": step + 1, "mse": float(loss.detach())})
    elapsed_s = time.perf_counter() - started_at

    with torch.no_grad():
        predictions = _render_views(model, cameras, config)
        per_view_mse = [
            float((prediction - target).square().mean())
            for prediction, target in zip(predictions, targets, strict=True)
        ]
        final_mse = sum(per_view_mse) / len(per_view_mse)
    finite_state = all(
        bool(torch.isfinite(value).all()) for value in model.state_dict().values()
    )
    return FitResult(
        representation=model.representation_name,
        trainable_scalars=sum(parameter.numel() for parameter in model.parameters()),
        initial_mse=initial_mse,
        final_mse=final_mse,
        final_psnr_db=-10.0 * math.log10(max(final_mse, 1.0e-12)),
        per_view_mse=per_view_mse,
        elapsed_s=elapsed_s,
        finite_state=finite_state,
        loss_trace=loss_trace,
    )


def _packed_spatial_covariance(covariance: Tensor) -> Tensor:
    return torch.stack(
        (
            covariance[0, 0],
            covariance[0, 1],
            covariance[0, 2],
            covariance[1, 1],
            covariance[1, 2],
            covariance[2, 2],
        )
    )


def _unpack_spatial_covariance(packed: Tensor) -> Tensor:
    xx, xy, xz, yy, yz, zz = packed
    return torch.stack(
        (
            torch.stack((xx, xy, xz)),
            torch.stack((xy, yy, yz)),
            torch.stack((xz, yz, zz)),
        )
    )


def _conditional_covariance_identifiability(
    target: SPD4WorldAtomModel,
    cameras: list[tuple[Tensor, Tensor]],
) -> dict[str, Any]:
    """Certify that the chosen charts observe all six spatial covariance DOFs."""

    target_batch = target.batch()
    center = target_batch.x0[0].detach().double()
    basis = torch.eye(6, dtype=torch.float64)
    design_rows: list[Tensor] = []
    for intrinsics_f32, world_to_camera_f32 in cameras:
        intrinsics = intrinsics_f32.double()
        world_to_camera = world_to_camera_f32.double()
        rotation = world_to_camera[:3, :3]
        center_cam = rotation @ center + world_to_camera[:3, 3]
        x, y, z = center_cam
        pixel_jacobian = torch.stack(
            (
                torch.stack(
                    (
                        intrinsics[0, 0] / z,
                        z.new_tensor(0.0),
                        -intrinsics[0, 0] * x / z.square(),
                    )
                ),
                torch.stack(
                    (
                        z.new_tensor(0.0),
                        intrinsics[1, 1] / z,
                        -intrinsics[1, 1] * y / z.square(),
                    )
                ),
            )
        )
        screen_from_world = pixel_jacobian @ rotation
        view_rows = []
        for packed_basis in basis:
            projected = (
                screen_from_world
                @ _unpack_spatial_covariance(packed_basis)
                @ screen_from_world.T
            )
            view_rows.append(
                torch.stack((projected[0, 0], projected[0, 1], projected[1, 1]))
            )
        design_rows.append(torch.stack(view_rows, dim=-1))
    design = torch.cat(design_rows, dim=0)
    target_cholesky = (
        target_batch.conditional_spatial_cholesky[0].detach().double()
    )
    target_covariance = target_cholesky @ target_cholesky.T
    target_observations = design @ _packed_spatial_covariance(target_covariance)

    legacy_columns = design[:, (0, 3)]
    legacy_solution = torch.linalg.lstsq(
        legacy_columns, target_observations
    ).solution
    legacy_residual = legacy_columns @ legacy_solution - target_observations
    full_solution = torch.linalg.lstsq(design, target_observations).solution
    full_residual = design @ full_solution - target_observations
    singular_values = torch.linalg.svdvals(design)
    return {
        "camera_count": len(cameras),
        "symmetric_spatial_dof": 6,
        "design_rank": int(torch.linalg.matrix_rank(design)),
        "design_singular_values": singular_values.tolist(),
        "target_conditional_spatial_covariance": target_covariance.tolist(),
        "best_legacy_diagonal_xy_variances": legacy_solution.tolist(),
        "best_legacy_observation_rmse": float(
            legacy_residual.square().mean().sqrt()
        ),
        "full_spd_observation_rmse": float(full_residual.square().mean().sqrt()),
        "interpretation": (
            "rank six identifies the full symmetric conditional spatial "
            "covariance; the legacy model is restricted to the xx/yy columns "
            "with zero depth width and zero spatial cross-covariances"
        ),
    }


def run_gate(
    *,
    steps: int,
    learning_rate: float,
    spd4_init_precision_z: float,
) -> dict[str, Any]:
    if steps < 1:
        raise ValueError("steps must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if spd4_init_precision_z <= 0.0:
        raise ValueError("spd4_init_precision_z must be positive")

    torch.manual_seed(17)
    torch.use_deterministic_algorithms(True)
    config = UVTRenderConfig(
        height=32,
        width=32,
        frames=5,
        tile_x=8,
        tile_y=8,
        tile_t=5,
        tile_capacity=64,
    )
    cameras = _camera_suite()
    target = _target_spd4()
    with torch.no_grad():
        targets = [value.detach() for value in _render_views(target, cameras, config)]

    certificate = _conditional_covariance_identifiability(target, cameras)
    legacy = _fit(
        _legacy_candidate(),
        targets,
        cameras,
        config,
        steps=steps,
        learning_rate=learning_rate,
    )
    spd4 = _fit(
        _new_spd4(init_precision_z=spd4_init_precision_z),
        targets,
        cameras,
        config,
        steps=steps,
        learning_rate=learning_rate,
    )
    improvement_ratio = legacy.final_mse / max(spd4.final_mse, 1.0e-20)
    psnr_gain_db = spd4.final_psnr_db - legacy.final_psnr_db
    initial_loss_ratio = spd4.initial_mse / legacy.initial_mse
    checks = {
        "camera_design_is_full_rank": certificate["design_rank"] == 6,
        "target_is_outside_legacy_covariance_class": (
            certificate["best_legacy_observation_rmse"] > 1.0e-3
        ),
        "initial_losses_are_matched_within_5_percent": (
            0.95 <= initial_loss_ratio <= 1.05
        ),
        "spd4_reaches_near_exact_fit": spd4.final_mse < 1.0e-8,
        "legacy_retains_structural_residual": legacy.final_mse > 1.0e-5,
        "spd4_gains_at_least_20_db": psnr_gain_db > 20.0,
        "all_states_finite": legacy.finite_state and spd4.finite_state,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if all(checks.values()) else "fail",
        "scope": "synthetic_cpu_representation_capacity",
        "claim_limits": {
            "public_scene_quality": False,
            "systems_speed": False,
            "metal_parity": False,
            "matched_atom_count": True,
            "matched_parameter_count": False,
            "alpha_semantics": "legacy_peak_splat",
            "moving_camera": False,
        },
        "fixture": {
            "seed": 17,
            "steps": steps,
            "learning_rate": learning_rate,
            "image_size": [32, 32],
            "frames": 5,
            "camera_angles_radians": [-0.65, 0.0, 0.58],
            "atoms_per_candidate": 1,
            "spd4_init_precision_z": spd4_init_precision_z,
            "target_space_time_tilt": [0.055, -0.030, 0.045],
        },
        "mathematical_certificate": certificate,
        "fits": {
            "legacy_tube": asdict(legacy),
            "full_spd4": asdict(spd4),
        },
        "comparison": {
            "initial_loss_ratio_spd4_over_legacy": initial_loss_ratio,
            "final_mse_improvement_ratio": improvement_ratio,
            "psnr_gain_db": psnr_gain_db,
            "parameter_ratio_spd4_over_legacy": (
                spd4.trainable_scalars / legacy.trainable_scalars
            ),
        },
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.015)
    parser.add_argument(
        "--spd4-init-precision-z",
        type=float,
        default=2500.0,
        help="Near-planar depth initialization used to control initial footprint.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = run_gate(
        steps=args.steps,
        learning_rate=args.learning_rate,
        spd4_init_precision_z=args.spd4_init_precision_z,
    )
    output_path = args.out if args.out.is_absolute() else ROOT / args.out
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote SPD(4) capacity gate to {output_path}")
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
