"""Real fast-mac dynamic-3DGS executor for the frozen G4 ablation.

The model is the existing per-frame ``FreeDynamic3DGS`` baseline and every
sampled/held-out image goes through the production fast-mac v5 Metal renderer.
There is no dense, CPU, procedural, or reduced-pixel fallback.
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
TRAIN_SRC = ROOT / "src" / "train"
GAUGE_EXPERIMENTS = ROOT / "research_experiments" / "gauge_fields"
DYNAMIC_RUNTIME_SOURCE = GAUGE_EXPERIMENTS / "train_splat_baseline.py"
RENDERING_SOURCE = TRAIN_SRC / "rendering.py"
FAST_MAC_WRAPPER_SOURCE = TRAIN_SRC / "renderers" / "fast_mac.py"
FAST_MAC_VARIANT = (
    ROOT / "third_party" / "fast-mac-gsplat" / "variants" / "v5"
)
_RUNTIME_MODULE_NAME = "_dynaworld_g4_dynamic_3dgs_runtime"
_SHA256_HEX = frozenset("0123456789abcdef")

DYNAMIC_3DGS_HYPERPARAMETERS = {
    "representation": "free_dynamic_3dgs_per_frame",
    "optimizer": "adam",
    "base_lr": 0.002,
    "init_scale": 0.035,
    "scale_init_log_jitter": 0.0,
    "init_alpha_logit": 0.0,
    "init_xyz_noise": 0.0,
    "init_quat_noise": 0.0,
    "log_scale_min": -12.0,
    "log_scale_max": 4.0,
    "scale_reg_weight": 1.0e-4,
    "temporal_smooth_weight": 1.0e-3,
    "renderer": "fast_mac",
    "rgb_variant": "v5",
    "background": [0.0, 0.0, 0.0],
    "camera_projection": "camera_model",
    "tile_size": 16,
    "bound_scale": 3.0,
    "alpha_threshold": 1.0 / 255.0,
    "near_plane": 0.25,
    "far_plane": 128.0,
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


def _load_dynamic_runtime() -> ModuleType:
    cached = sys.modules.get(_RUNTIME_MODULE_NAME)
    if cached is not None:
        return cached
    for path in (TRAIN_SRC, GAUGE_EXPERIMENTS):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    if not DYNAMIC_RUNTIME_SOURCE.is_file():
        raise FileNotFoundError(
            f"dynamic-3DGS production runtime is missing: {DYNAMIC_RUNTIME_SOURCE}"
        )
    spec = importlib.util.spec_from_file_location(
        _RUNTIME_MODULE_NAME,
        DYNAMIC_RUNTIME_SOURCE,
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"cannot load dynamic-3DGS production runtime: {DYNAMIC_RUNTIME_SOURCE}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[_RUNTIME_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_RUNTIME_MODULE_NAME, None)
        raise
    return module


def _attest_fast_mac_native() -> dict[str, Any]:
    import torch
    from paper_training_protocol import (
        paper_runtime_source_tree_identity,
        validate_paper_runtime_source_tree_identity,
    )

    package = FAST_MAC_VARIANT / "torch_gsplat_bridge_v5"
    candidates = [
        package / f"_C{suffix}"
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
        if (package / f"_C{suffix}").is_file()
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "fast-mac v5 requires exactly one native binary for the active Python ABI"
        )
    binary = candidates[0].resolve()
    # v5 is a TORCH_LIBRARY registration library, not a CPython PyInit module.
    torch.ops.load_library(str(binary))
    if not all(
        hasattr(torch.ops.gsplat_metal_v5, name)
        for name in ("bin", "render_fast_forward_state", "render_fast_backward_saved")
    ):
        raise RuntimeError("fast-mac v5 production forward/backward operators are missing")
    identity = {
        "module": "torch_gsplat_bridge_v5._C",
        "path": str(binary),
        "bytes": int(binary.stat().st_size),
        "sha256": _file_sha256(binary),
        "runtime_source_tree": paper_runtime_source_tree_identity(
            FAST_MAC_VARIANT / "csrc" / "metal"
        ),
        "build_source_bound": False,
        "attestation_scope": "loaded_binary_and_current_source_observation_only",
    }
    required = {
        "module",
        "path",
        "bytes",
        "sha256",
        "runtime_source_tree",
        "build_source_bound",
        "attestation_scope",
    }
    if set(identity) != required:
        raise ValueError("fast-mac v5 native identity keys changed")
    binary = Path(str(identity["path"])).resolve()
    expected_package = (FAST_MAC_VARIANT / "torch_gsplat_bridge_v5").resolve()
    try:
        binary.relative_to(expected_package)
    except ValueError as error:
        raise ValueError("fast-mac loaded a binary outside the v5 variant") from error
    if (
        identity["module"] != "torch_gsplat_bridge_v5._C"
        or not binary.is_file()
        or binary.suffix != ".so"
        or identity["bytes"] != binary.stat().st_size
        or identity["bytes"] < 1
        or identity["sha256"] != _file_sha256(binary)
        or not _valid_sha256(identity["sha256"])
        or identity["build_source_bound"] is not False
        or identity["attestation_scope"]
        != "loaded_binary_and_current_source_observation_only"
    ):
        raise ValueError("fast-mac v5 loaded-binary observation failed")
    source_tree = identity["runtime_source_tree"]
    if not isinstance(source_tree, Mapping):
        raise ValueError("fast-mac v5 runtime source-tree identity is missing")
    validate_paper_runtime_source_tree_identity(source_tree)
    if Path(str(source_tree.get("root", ""))).resolve() != (
        FAST_MAC_VARIANT / "csrc" / "metal"
    ).resolve():
        raise ValueError("fast-mac v5 runtime source-tree root changed")
    return identity


class Dynamic3DGSPublicQualityExecutor:
    def __init__(self, *, context: RowContext) -> None:
        if context.request.route != "dynamic_3dgs":
            raise ValueError("dynamic-3DGS executor received another route")
        self._context = context
        self._native_identity: dict[str, Any] | None = None

    def _require_context(self, context: RowContext) -> None:
        if context is not self._context or context.request.route != "dynamic_3dgs":
            raise ValueError("dynamic-3DGS executor context changed")

    def _attest(self) -> dict[str, Any]:
        if self._native_identity is None:
            self._native_identity = _attest_fast_mac_native()
        return self._native_identity

    def capability(self, context: RowContext) -> Mapping[str, Any]:
        import torch

        self._require_context(context)
        if sys.platform != "darwin" or not torch.backends.mps.is_available():
            raise RuntimeError(
                "dynamic-3DGS production capability requires available macOS MPS"
            )
        self._attest()
        return executor_capability(context)

    def open_session(self, context: RowContext, dataset: Any) -> Any:
        import torch

        self._require_context(context)
        native_identity = self._attest()
        runtime = _load_dynamic_runtime()
        if not torch.backends.mps.is_available():
            raise RuntimeError("dynamic-3DGS G4 execution requires available MPS")
        frame_count = int(context.protocol.dataset.frame_count)
        site_count = int(context.config["public_protocol"]["primitive_count"])
        compiler = context.scene_receipt["compiler"]
        if (
            frame_count != 300
            or site_count != 1024
            or context.protocol.final_stage.primitive_count != site_count
        ):
            raise ValueError("dynamic-3DGS G4 representation dimensions changed")
        if (
            float(compiler["near"]) != DYNAMIC_3DGS_HYPERPARAMETERS["near_plane"]
            or float(compiler["far"]) != DYNAMIC_3DGS_HYPERPARAMETERS["far_plane"]
        ):
            raise ValueError("dynamic-3DGS near/far support differs from the frozen compiler")

        seed = load_fresh_representation_seed(
            dataset,
            expected_site_count=site_count,
        )
        device = torch.device("mps")
        torch.manual_seed(int(context.request.seed))
        positions = seed["positions0_f32_cpu"].to(device=device).contiguous()
        colors = seed["colors_f32_cpu"].to(device=device).contiguous()
        model = runtime.FreeDynamic3DGS(
            init_xyz=positions,
            init_rgb=colors,
            num_frames=frame_count,
            splat_mode="per_frame",
            init_scale=DYNAMIC_3DGS_HYPERPARAMETERS["init_scale"],
            scale_init_log_jitter=DYNAMIC_3DGS_HYPERPARAMETERS[
                "scale_init_log_jitter"
            ],
            init_alpha_logit=DYNAMIC_3DGS_HYPERPARAMETERS["init_alpha_logit"],
            init_xyz_noise=DYNAMIC_3DGS_HYPERPARAMETERS["init_xyz_noise"],
            init_quat_noise=DYNAMIC_3DGS_HYPERPARAMETERS["init_quat_noise"],
            log_scale_min=DYNAMIC_3DGS_HYPERPARAMETERS["log_scale_min"],
            log_scale_max=DYNAMIC_3DGS_HYPERPARAMETERS["log_scale_max"],
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=DYNAMIC_3DGS_HYPERPARAMETERS["base_lr"],
        )

        def render_image(request: Any, camera: Any) -> Any:
            frame = model.frame(int(request.frame_index))
            camera_space_depth = (
                frame.xyz - camera.camera_to_world[:3, 3]
            ) @ camera.camera_to_world[:3, :3]
            support = (
                torch.isfinite(camera_space_depth[:, 2])
                & (
                    camera_space_depth[:, 2]
                    >= DYNAMIC_3DGS_HYPERPARAMETERS["near_plane"]
                )
                & (
                    camera_space_depth[:, 2]
                    <= DYNAMIC_3DGS_HYPERPARAMETERS["far_plane"]
                )
            )
            frame = replace(
                frame,
                opacities=torch.where(
                    support[:, None],
                    frame.opacities,
                    torch.zeros_like(frame.opacities),
                ),
            )
            return runtime.render_gaussian_frame(
                frame,
                camera,
                height=int(request.image_height),
                width=int(request.image_width),
                mode=DYNAMIC_3DGS_HYPERPARAMETERS["renderer"],
                tile_size=DYNAMIC_3DGS_HYPERPARAMETERS["tile_size"],
                bound_scale=DYNAMIC_3DGS_HYPERPARAMETERS["bound_scale"],
                alpha_threshold=DYNAMIC_3DGS_HYPERPARAMETERS[
                    "alpha_threshold"
                ],
                near_plane=DYNAMIC_3DGS_HYPERPARAMETERS["near_plane"],
                fast_mac_options={
                    "rgb_variant": DYNAMIC_3DGS_HYPERPARAMETERS["rgb_variant"],
                    "background": list(
                        DYNAMIC_3DGS_HYPERPARAMETERS["background"]
                    ),
                },
                camera_projection=DYNAMIC_3DGS_HYPERPARAMETERS[
                    "camera_projection"
                ],
            ).permute(1, 2, 0).contiguous()

        def regularization() -> Any:
            return (
                DYNAMIC_3DGS_HYPERPARAMETERS["scale_reg_weight"]
                * model.scale_loss()
                + DYNAMIC_3DGS_HYPERPARAMETERS["temporal_smooth_weight"]
                * model.temporal_smoothness_loss()
            )

        route_contract = {
            "schema_version": 1,
            "executor_source_path": str(Path(__file__).resolve()),
            "executor_source_sha256": _file_sha256(Path(__file__).resolve()),
            "dynamic_runtime_source_path": str(DYNAMIC_RUNTIME_SOURCE.resolve()),
            "dynamic_runtime_source_sha256": _file_sha256(DYNAMIC_RUNTIME_SOURCE),
            "rendering_source_path": str(RENDERING_SOURCE.resolve()),
            "rendering_source_sha256": _file_sha256(RENDERING_SOURCE),
            "fast_mac_wrapper_source_path": str(FAST_MAC_WRAPPER_SOURCE.resolve()),
            "fast_mac_wrapper_source_sha256": _file_sha256(
                FAST_MAC_WRAPPER_SOURCE
            ),
            "native_extension_identity": native_identity,
            "hyperparameters": dict(DYNAMIC_3DGS_HYPERPARAMETERS),
            "full_frame_count": frame_count,
            "site_count_per_frame": site_count,
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
            base_lr=DYNAMIC_3DGS_HYPERPARAMETERS["base_lr"],
            render_image=render_image,
            regularization=regularization,
            set_active_count=model.set_active_splat_count,
        )


def create_public_quality_executor(
    *,
    context: RowContext,
) -> Dynamic3DGSPublicQualityExecutor:
    return Dynamic3DGSPublicQualityExecutor(context=context)


def run_public_quality_runtime_smoke(
    *,
    context: RowContext,
    dataset: Any,
) -> Mapping[str, Any]:
    """Exercise real fast-mac train/update/checkpoint/heldout wiring once."""

    executor = create_public_quality_executor(context=context)
    return run_gaussian_public_quality_runtime_smoke(
        context=context,
        dataset=dataset,
        executor=executor,
        executor_source=Path(__file__),
    )


__all__ = [
    "DYNAMIC_3DGS_HYPERPARAMETERS",
    "Dynamic3DGSPublicQualityExecutor",
    "create_public_quality_executor",
    "run_public_quality_runtime_smoke",
]
