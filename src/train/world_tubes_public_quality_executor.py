"""Real STAR-UVT executor for the frozen G4 public-quality ablation.

This adapter intentionally exposes only the production MPS route.  The row
worker owns public targets, calibrated rays, the exact sample schedule, and
evaluation.  This module owns the trainable World Tubes representation and the
loaded-and-hashed STAR-UVT Metal forward/backward implementation.  A build
receipt is not inferred from the current source tree.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import sys
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

from gaussian_public_quality_executor import (
    GaussianPublicQualitySession,
    executor_capability,
    load_fresh_representation_seed,
    representation_seed_identity,
    run_gaussian_public_quality_runtime_smoke,
)
from worldfoam_native4d_public_quality_row import RowContext


ROOT = Path(__file__).resolve().parents[2]
STAR_VARIANT = (
    ROOT / "third_party" / "fast-mac-gsplat" / "variants" / "star_uvt_v0"
)
STAR_COMPARE = (
    STAR_VARIANT
    / "research_project"
    / "benchmarks"
    / "multicam_heldout_compare.py"
)
_RUNTIME_MODULE_NAME = "_dynaworld_g4_world_tubes_runtime"
_SHA256_HEX = frozenset("0123456789abcdef")

WORLD_TUBES_HYPERPARAMETERS = {
    "representation": "legacy_tube",
    "optimizer": "adam",
    "base_lr": 0.03,
    "init_precision_xy": 30.0,
    "init_lambda_t": 0.35,
    "init_opacity": 0.35,
    "min_precision_xy": 1.0e-5,
    "min_lambda_t": 1.0e-5,
    "velocity_reg_weight": 1.0e-4,
    "depth_velocity_reg_weight": 0.0,
    "position_reg_weight": 1.0e-6,
    "alpha_mode": "peak_splat",
    "amplitude_convention": "fiber_integrated",
    "camera_projection": "dataset_lens",
    "camera_sequence_mode": "selected_time_gauged_uvt_replay",
    "gauge_projection_math": (
        "dataset_lens_pixel_jacobian_depth_marginalized_uvt"
    ),
    "cross_time_projective_atlas_compiled_here": False,
    "cross_time_scaling_evidence_from_this_row": False,
    "projective_variable_camera_evidence_lane": (
        "schema_v2_variable_camera_closure_and_frozen_world_scaling"
    ),
    "render_backend": "metal_tile",
    "reduction_mode": "index_add",
    "sample_emission_mode": "direct_atomic",
    "tile_x": 8,
    "tile_y": 8,
    "tile_t": 2,
    "tile_capacity": 128,
    "background": [0.0, 0.0, 0.0],
    "near_plane": 0.25,
    "far_plane": 128.0,
    "temporal_seed_mapping": "source_frame_or_sequence_midpoint_broad_support_v1",
    "unknown_source_time_sentinel": -1,
    "unknown_source_time_fallback": "sequence_midpoint_broad_support",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(_SHA256_HEX)
    )


def _load_star_runtime() -> ModuleType:
    cached = sys.modules.get(_RUNTIME_MODULE_NAME)
    if cached is not None:
        return cached
    if not STAR_COMPARE.is_file():
        raise FileNotFoundError(f"STAR-UVT production runtime is missing: {STAR_COMPARE}")
    if str(STAR_VARIANT) not in sys.path:
        sys.path.insert(0, str(STAR_VARIANT))
    spec = importlib.util.spec_from_file_location(_RUNTIME_MODULE_NAME, STAR_COMPARE)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load STAR-UVT production runtime: {STAR_COMPARE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_RUNTIME_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_RUNTIME_MODULE_NAME, None)
        raise
    return module


def _attest_star_native(runtime: ModuleType) -> dict[str, Any]:
    import torch
    from paper_training_protocol import (
        paper_runtime_source_tree_identity,
        validate_paper_runtime_source_tree_identity,
    )

    if not all(
        callable(getattr(runtime, name, None))
        for name in (
            "project_world_tube_sequence",
            "render_projected_sequence",
            "WorldTubeModel",
        )
    ):
        raise RuntimeError("STAR-UVT Python production route is incomplete")
    package = STAR_VARIANT / "torch_gsplat_bridge_star_uvt"
    candidates = [
        package / f"_C{suffix}"
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
        if (package / f"_C{suffix}").is_file()
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "STAR-UVT requires exactly one native binary for the active Python ABI"
        )
    binary = candidates[0].resolve()
    # These C++ files register TORCH_LIBRARY operators and intentionally do not
    # expose PyInit__C.  Importing them as Python modules is therefore the wrong
    # attestation; loading the library and checking the operators is exact.
    torch.ops.load_library(str(binary))
    if not all(
        hasattr(torch.ops.star_uvt_v0, name)
        for name in ("render", "direct_atomic_backward")
    ):
        raise RuntimeError("STAR-UVT production forward/backward operators are missing")
    source_files = sorted(
        (
            *(candidate for candidate in (STAR_VARIANT / "csrc").rglob("*") if candidate.is_file()),
            STAR_VARIANT / "setup.py",
        )
    )
    source_digest = hashlib.sha256()
    for source_path in source_files:
        source_digest.update(str(source_path.relative_to(STAR_VARIANT)).encode("utf-8"))
        source_digest.update(_file_sha256(source_path).encode("ascii"))
    identity = {
        "module": "torch_gsplat_bridge_star_uvt._C",
        "path": str(binary),
        "bytes": int(binary.stat().st_size),
        "sha256": _file_sha256(binary),
        "runtime_source_tree": paper_runtime_source_tree_identity(
            STAR_VARIANT / "csrc" / "metal"
        ),
        "source_tree_sha256": source_digest.hexdigest(),
        "source_file_count": len(source_files),
        "build_source_bound": False,
        "attestation_scope": "loaded_binary_and_current_source_observation_only",
    }
    required = {
        "module",
        "path",
        "bytes",
        "sha256",
        "runtime_source_tree",
        "source_tree_sha256",
        "source_file_count",
        "build_source_bound",
        "attestation_scope",
    }
    if set(identity) != required:
        raise ValueError("STAR-UVT native identity keys changed")
    binary = Path(str(identity["path"])).resolve()
    expected_package = (STAR_VARIANT / "torch_gsplat_bridge_star_uvt").resolve()
    try:
        binary.relative_to(expected_package)
    except ValueError as error:
        raise ValueError("STAR-UVT loaded a binary outside the production variant") from error
    if (
        identity["module"] != "torch_gsplat_bridge_star_uvt._C"
        or not binary.is_file()
        or binary.suffix != ".so"
        or identity["bytes"] != binary.stat().st_size
        or identity["bytes"] < 1
        or identity["sha256"] != _file_sha256(binary)
        or not _valid_sha256(identity["sha256"])
        or not _valid_sha256(identity["source_tree_sha256"])
        or not isinstance(identity["source_file_count"], int)
        or identity["source_file_count"] < 1
        or identity["build_source_bound"] is not False
        or identity["attestation_scope"]
        != "loaded_binary_and_current_source_observation_only"
    ):
        raise ValueError("STAR-UVT loaded binary/current source observation failed")
    source_tree = identity["runtime_source_tree"]
    if not isinstance(source_tree, Mapping):
        raise ValueError("STAR-UVT runtime source-tree identity is missing")
    validate_paper_runtime_source_tree_identity(source_tree)
    if Path(str(source_tree.get("root", ""))).resolve() != (
        STAR_VARIANT / "csrc" / "metal"
    ).resolve():
        raise ValueError("STAR-UVT runtime source-tree root changed")
    return identity


def _selected_time_config(runtime: ModuleType, *, height: int, width: int) -> Any:
    return runtime.UVTRenderConfig(
        height=int(height),
        width=int(width),
        frames=1,
        tile_x=WORLD_TUBES_HYPERPARAMETERS["tile_x"],
        tile_y=WORLD_TUBES_HYPERPARAMETERS["tile_y"],
        tile_t=WORLD_TUBES_HYPERPARAMETERS["tile_t"],
        tile_capacity=WORLD_TUBES_HYPERPARAMETERS["tile_capacity"],
        background=tuple(WORLD_TUBES_HYPERPARAMETERS["background"]),
        max_alpha=runtime.max_alpha_for_mode(
            WORLD_TUBES_HYPERPARAMETERS["alpha_mode"]
        ),
        alpha_mode=WORLD_TUBES_HYPERPARAMETERS["alpha_mode"],
        amplitude_convention=WORLD_TUBES_HYPERPARAMETERS[
            "amplitude_convention"
        ],
    )


class WorldTubesPublicQualityExecutor:
    def __init__(self, *, context: RowContext) -> None:
        if context.request.route != "world_tubes":
            raise ValueError("World Tubes executor received another route")
        self._context = context
        self._runtime: ModuleType | None = None
        self._native_identity: dict[str, Any] | None = None

    def _require_context(self, context: RowContext) -> None:
        if context is not self._context or context.request.route != "world_tubes":
            raise ValueError("World Tubes executor context changed")

    def _attest(self) -> tuple[ModuleType, dict[str, Any]]:
        if self._runtime is None or self._native_identity is None:
            self._runtime = _load_star_runtime()
            self._native_identity = _attest_star_native(self._runtime)
        return self._runtime, self._native_identity

    def capability(self, context: RowContext) -> Mapping[str, Any]:
        import torch

        self._require_context(context)
        if sys.platform != "darwin" or not torch.backends.mps.is_available():
            raise RuntimeError(
                "World Tubes production capability requires available macOS MPS"
            )
        self._attest()
        return executor_capability(context)

    def open_session(self, context: RowContext, dataset: Any) -> Any:
        import torch

        self._require_context(context)
        runtime, native_identity = self._attest()
        if not torch.backends.mps.is_available():
            raise RuntimeError("World Tubes G4 execution requires available MPS")
        frame_count = int(context.protocol.dataset.frame_count)
        site_count = int(context.config["public_protocol"]["primitive_count"])
        compiler = context.scene_receipt["compiler"]
        if (
            frame_count != 300
            or site_count != 1024
            or context.protocol.final_stage.primitive_count != site_count
        ):
            raise ValueError("World Tubes G4 representation dimensions changed")
        if (
            float(compiler["near"]) != WORLD_TUBES_HYPERPARAMETERS["near_plane"]
            or float(compiler["far"]) != WORLD_TUBES_HYPERPARAMETERS["far_plane"]
        ):
            raise ValueError("World Tubes near/far support differs from the frozen compiler")

        seed = load_fresh_representation_seed(
            dataset,
            expected_site_count=site_count,
        )
        device = torch.device("mps")
        torch.manual_seed(int(context.request.seed))
        positions = seed["positions0_f32_cpu"].to(device=device).contiguous()
        colors = seed["colors_f32_cpu"].to(device=device).contiguous()
        source_frames_cpu = seed["source_frame_indices_i64_cpu"]
        if bool(torch.any(source_frames_cpu >= frame_count).item()):
            raise ValueError("representation source frame lies outside the frozen sequence")
        known_source_time = source_frames_cpu >= 0
        half_duration = 0.5 * float(frame_count - 1)
        broad_support_lambda = 1.0 / max(half_duration, 1.0) ** 2
        if broad_support_lambda <= WORLD_TUBES_HYPERPARAMETERS["min_lambda_t"]:
            raise ValueError("unknown-source-time broad support violates minimum precision")
        t0_cpu = torch.where(
            known_source_time,
            source_frames_cpu.to(dtype=torch.float32) - half_duration,
            torch.zeros(site_count, dtype=torch.float32, device="cpu"),
        ).contiguous()
        lambda_t_cpu = torch.where(
            known_source_time,
            torch.full(
                (site_count,),
                WORLD_TUBES_HYPERPARAMETERS["init_lambda_t"],
                dtype=torch.float32,
                device="cpu",
            ),
            torch.full(
                (site_count,),
                broad_support_lambda,
                dtype=torch.float32,
                device="cpu",
            ),
        ).contiguous()
        temporal_initialization = {
            "source_time_provenance": seed["source_time_provenance"],
            "known_source_frame_count": int(known_source_time.sum().item()),
            "fallback_source_frame_count": int((~known_source_time).sum().item()),
            "fallback_centered_time": 0.0,
            "fallback_temporal_precision": broad_support_lambda,
        }
        model = runtime.WorldTubeModel(
            init_x0=positions,
            init_color=colors,
            init_t0=t0_cpu.to(device=device),
            frames=frame_count,
            init_precision_xy=WORLD_TUBES_HYPERPARAMETERS["init_precision_xy"],
            init_lambda_t=lambda_t_cpu.to(device=device),
            init_opacity=WORLD_TUBES_HYPERPARAMETERS["init_opacity"],
            min_precision_xy=WORLD_TUBES_HYPERPARAMETERS["min_precision_xy"],
            min_lambda_t=WORLD_TUBES_HYPERPARAMETERS["min_lambda_t"],
            velocity_reg_weight=WORLD_TUBES_HYPERPARAMETERS[
                "velocity_reg_weight"
            ],
            depth_velocity_reg_weight=WORLD_TUBES_HYPERPARAMETERS[
                "depth_velocity_reg_weight"
            ],
            position_reg_weight=WORLD_TUBES_HYPERPARAMETERS[
                "position_reg_weight"
            ],
            alpha_mode=WORLD_TUBES_HYPERPARAMETERS["alpha_mode"],
            amplitude_convention=WORLD_TUBES_HYPERPARAMETERS[
                "amplitude_convention"
            ],
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=WORLD_TUBES_HYPERPARAMETERS["base_lr"],
        )

        def render_image(request: Any, camera: Any) -> Any:
            config = _selected_time_config(
                runtime,
                height=request.image_height,
                width=request.image_width,
            )
            K = torch.zeros((3, 3), dtype=torch.float32, device=device)
            K[0, 0] = camera.fx
            K[1, 1] = camera.fy
            K[0, 2] = camera.cx
            K[1, 2] = camera.cy
            K[2, 2] = 1.0
            w2c = torch.linalg.inv(camera.camera_to_world).contiguous()
            with torch.no_grad():
                batch = model.batch()
                global_time = float(request.frame_index) - 0.5 * float(
                    frame_count - 1
                )
                selected_world = batch.x0 + batch.velocity * (
                    global_time - batch.t0
                )[:, None]
                selected_camera = (
                    selected_world @ w2c[:3, :3].T + w2c[:3, 3]
                )
                support = (
                    torch.isfinite(selected_camera[:, 2])
                    & (
                        selected_camera[:, 2]
                        >= WORLD_TUBES_HYPERPARAMETERS["near_plane"]
                    )
                    & (
                        selected_camera[:, 2]
                        <= WORLD_TUBES_HYPERPARAMETERS["far_plane"]
                    )
                )
            projected = runtime.project_world_tube_sequence(
                model,
                K,
                w2c,
                config,
                camera_projection=WORLD_TUBES_HYPERPARAMETERS[
                    "camera_projection"
                ],
                lens_model=camera.lens_model,
                distortion=camera.distortion,
                full_frames=frame_count,
                frame_start=int(request.frame_index),
            )
            projected = replace(
                projected,
                depth0=torch.where(
                    support,
                    projected.depth0,
                    torch.full_like(
                        projected.depth0,
                        WORLD_TUBES_HYPERPARAMETERS["far_plane"],
                    ),
                ),
                depth_beta=torch.where(
                    support[:, None],
                    projected.depth_beta,
                    torch.zeros_like(projected.depth_beta),
                ),
                opacity=torch.where(
                    support,
                    projected.opacity,
                    torch.zeros_like(projected.opacity),
                ),
            )
            return runtime.render_projected_sequence(
                projected,
                config,
                backend=WORLD_TUBES_HYPERPARAMETERS["render_backend"],
                reduction_mode=WORLD_TUBES_HYPERPARAMETERS["reduction_mode"],
                sample_emission_mode=WORLD_TUBES_HYPERPARAMETERS[
                    "sample_emission_mode"
                ],
            ).rgb[0]

        route_contract = {
            "schema_version": 1,
            "executor_source_path": str(Path(__file__).resolve()),
            "executor_source_sha256": _file_sha256(Path(__file__).resolve()),
            "star_runtime_source_path": str(STAR_COMPARE.resolve()),
            "star_runtime_source_sha256": _file_sha256(STAR_COMPARE),
            "native_extension_identity": native_identity,
            "hyperparameters": dict(WORLD_TUBES_HYPERPARAMETERS),
            "model_metadata": model.representation_metadata(),
            "full_frame_count": frame_count,
            "site_count": site_count,
            "temporal_initialization": temporal_initialization,
            "seed_identity": representation_seed_identity(seed),
        }
        return GaussianPublicQualitySession(
            context=context,
            dataset=dataset,
            model=model,
            optimizer=optimizer,
            device=device,
            seed_identity=representation_seed_identity(seed),
            route_contract=route_contract,
            base_lr=WORLD_TUBES_HYPERPARAMETERS["base_lr"],
            render_image=render_image,
            regularization=model.regularization,
            set_active_count=model.set_active_tube_count,
        )


def create_public_quality_executor(
    *,
    context: RowContext,
) -> WorldTubesPublicQualityExecutor:
    return WorldTubesPublicQualityExecutor(context=context)


def run_public_quality_runtime_smoke(
    *,
    context: RowContext,
    dataset: Any,
) -> Mapping[str, Any]:
    """Exercise real STAR-UVT train/update/checkpoint/heldout wiring once."""

    executor = create_public_quality_executor(context=context)
    return run_gaussian_public_quality_runtime_smoke(
        context=context,
        dataset=dataset,
        executor=executor,
        executor_source=Path(__file__),
    )


__all__ = [
    "WORLD_TUBES_HYPERPARAMETERS",
    "WorldTubesPublicQualityExecutor",
    "create_public_quality_executor",
    "run_public_quality_runtime_smoke",
]
