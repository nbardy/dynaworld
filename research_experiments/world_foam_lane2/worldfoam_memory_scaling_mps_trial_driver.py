"""Real-MPS trial driver for the fixed-site WorldFoam memory experiment.

This module deliberately keeps its import surface stdlib-only.  The producer
can inspect ``WORLDFOAM_MEMORY_SCALING_DRIVER_CAPABILITIES`` before importing
PyTorch, loading the native extension, or spawning any heavy trials.

The runnable path executes the production fixed-site material coordinator with
the exact native module supplied by the producer.  It does not turn current
MPS allocator counters into exact peak measurements: this PyTorch/MPS build
exposes no authoritative high-water API and no Metal register/spill or
kernel-private-scratch counter.  The producer samples the public counters under
a hard allocation ceiling, while the native attestation labels unobservable
kernel-private properties explicitly.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any


DRIVER_PROTOCOL = "worldfoam-memory-scaling-trial-driver-v1"
OPTIMIZER_LIFECYCLE_PROTOCOL = (
    "worldfoam-memory-scaling-fixed-site-optimizer-lifecycle-v4"
)
WORLDFOAM_MEMORY_SCALING_DRIVER_CAPABILITIES = {
    "schema_version": 3,
    "driver_protocol": "worldfoam-memory-scaling-trial-driver-v1",
    "supported_backends": ("mps",),
    "selected_pixel_target_access": {
        "implemented": True,
        "access_mode": "direct_pixels",
        "full_frame_materialization_count": 0,
        "preserves_request_order_and_duplicates": True,
        "source_budget_enforced_before_allocation": True,
        "contract": "PowerFoamSelectedPixelRead/v1",
    },
}

_NATIVE_FORWARD = "kinetic_precompiled_length_p0_lie_node_forward_launch_only"
_NATIVE_FULL_VJP = "kinetic_precompiled_length_p0_lie_node_vjp_accumulate_launch_only"
_NATIVE_MATERIAL_VJP = (
    "kinetic_precompiled_length_p0_lie_material_node_vjp_accumulate_launch_only"
)
_NATIVE_SAMPLE_PREPARE = "prepare_kinetic_ragged_p0_lie_sample_block"
_NATIVE_SAMPLE_LAUNCH = (
    "kinetic_ragged_p0_lie_sample_accumulate_loss_only_launch_only"
)
_NATIVE_OP_NAMES = (
    _NATIVE_FORWARD,
    _NATIVE_FULL_VJP,
    _NATIVE_MATERIAL_VJP,
    _NATIVE_SAMPLE_PREPARE,
    _NATIVE_SAMPLE_LAUNCH,
)


def _sha256_payload(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _endpoint_including_frame_indices(
    *,
    dataset_frame_count: int,
    requested_frame_count: int,
) -> tuple[int, ...]:
    """Choose a deterministic dense subset without changing the dataset grid."""

    _positive_int(dataset_frame_count, name="dataset_frame_count")
    _positive_int(requested_frame_count, name="requested_frame_count")
    if requested_frame_count > dataset_frame_count:
        raise ValueError("requested_frame_count cannot exceed dataset_frame_count")
    if requested_frame_count == 1:
        return (0,)
    indices = tuple(
        sample_index * (dataset_frame_count - 1) // (requested_frame_count - 1)
        for sample_index in range(requested_frame_count)
    )
    if (
        len(set(indices)) != requested_frame_count
        or indices[0] != 0
        or indices[-1] != dataset_frame_count - 1
    ):
        raise ArithmeticError("endpoint-including frame selection lost coverage")
    return indices


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _numeric_rows(
    value: Any,
    *,
    name: str,
    row_width: int,
) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise TypeError(f"{name} must be a nonempty row sequence")
    rows: list[tuple[float, ...]] = []
    for row_index, row in enumerate(value):
        if (
            not isinstance(row, Sequence)
            or isinstance(row, (str, bytes))
            or len(row) != row_width
        ):
            raise ValueError(f"{name}[{row_index}] must have width {row_width}")
        rows.append(
            tuple(
                _finite_float(item, name=f"{name}[{row_index}][{column_index}]")
                for column_index, item in enumerate(row)
            )
        )
    return tuple(rows)


def _install_runtime_paths() -> Path:
    root = Path(__file__).resolve().parents[2]
    for path in (root / "src" / "train", Path(__file__).resolve().parent):
        label = str(path)
        if label not in sys.path:
            sys.path.insert(0, label)
    return root


class _NativeCallAudit:
    """Count successful calls while preserving the producer's module identity."""

    def __init__(self, native_ops: Any) -> None:
        self.native_ops = native_ops
        self.counts = {name: 0 for name in _NATIVE_OP_NAMES}
        self._originals: dict[str, Any] = {}

    def __enter__(self) -> _NativeCallAudit:
        for name in _NATIVE_OP_NAMES:
            original = getattr(self.native_ops, name, None)
            if not callable(original):
                raise TypeError(f"native_ops lacks required callable {name}")
            self._originals[name] = original

            @functools.wraps(original)
            def audited(*args: Any, __name: str = name, __op: Any = original, **kwargs: Any):
                result = __op(*args, **kwargs)
                self.counts[__name] += 1
                return result

            setattr(self.native_ops, name, audited)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for name, original in self._originals.items():
            setattr(self.native_ops, name, original)


def run_worldfoam_fixed_site_optimizer_lifecycle(
    *,
    provider: Any,
    artifact_store: Any,
    batch: Any,
    step_policy: Any,
    initial_site_rgba_f32_cpu: Any,
    background_rgb_f32_cpu: Any,
    device: Any,
    native_ops: Any,
    backend_provenance: str,
    device_completion_fence: Any,
    device_completion_fence_provenance: str,
    initializer_generation_digest: str,
    source_material_seed_digest: str,
    target_generation_id: str,
    maximum_material_state_logical_tensor_bytes: int,
    color_learning_rate: float = 0.01,
    density_learning_rate: float = 0.01,
) -> Mapping[str, Any]:
    """Run two exact optimizer steps plus in-process checkpoint continuation.

    This backend-neutral helper is the CPU-proxy seam for the real MPS v4
    protocol.  The canonical parameters and manual SGD remain on CPU; each
    coordinator call receives one generation-bound device material snapshot
    and returns only one fenced CPU gradient/loss receipt.
    """

    _install_runtime_paths()
    import torch
    from paper_kinetic_fixed_site_material_device_bridge import (
        apply_paper_kinetic_fixed_site_material_device_gradient_receipt,
        seal_paper_kinetic_fixed_site_material_device_gradient_receipt,
        snapshot_paper_kinetic_fixed_site_material_to_device,
    )
    from paper_kinetic_fixed_site_material_state import (
        PaperKineticFixedSiteMaterialParameterization,
        PaperKineticFixedSiteMaterialSGDPolicy,
        checkpoint_paper_kinetic_fixed_site_material_state,
        prepare_paper_kinetic_fixed_site_material_state,
        restore_paper_kinetic_fixed_site_material_state,
    )
    from paper_kinetic_fixed_site_material_step import (
        PaperKineticFixedSiteMaterialOnlyGenerationPolicy,
        prepare_paper_kinetic_fixed_site_material_step_state,
        run_paper_kinetic_fixed_site_material_only_step,
    )
    from paper_kinetic_world_initializer import (
        prepare_paper_kinetic_p0_material_initialization,
    )

    resolved_device = torch.device(device)
    if resolved_device.type not in {"cpu", "mps", "cuda"}:
        raise ValueError("optimizer lifecycle supports CPU, MPS, or CUDA")
    if tuple(initial_site_rgba_f32_cpu.shape) != (provider.world.site_count, 4):
        raise ValueError("initial P0 material shape does not match the provider world")
    if initial_site_rgba_f32_cpu.device.type != "cpu":
        raise ValueError("initial P0 material must be CPU-owned")
    if tuple(background_rgb_f32_cpu.shape) != (3,) or (
        background_rgb_f32_cpu.device.type != "cpu"
    ):
        raise ValueError("optimizer lifecycle background must be CPU float32 [3]")
    _positive_int(
        maximum_material_state_logical_tensor_bytes,
        name="maximum_material_state_logical_tensor_bytes",
    )
    for name, value in (
        ("initializer_generation_digest", initializer_generation_digest),
        ("source_material_seed_digest", source_material_seed_digest),
    ):
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"{name} must be a SHA-256 digest")
    if not isinstance(target_generation_id, str) or not target_generation_id:
        raise ValueError("target_generation_id must be nonempty")

    initialization = prepare_paper_kinetic_p0_material_initialization(
        initial_site_rgba_f32_cpu,
        provider.world.sites,
        initializer_generation_digest=initializer_generation_digest,
        source_material_seed_digest=source_material_seed_digest,
    )
    material_state = prepare_paper_kinetic_fixed_site_material_state(
        initialization,
        provider.world,
        parameterization=PaperKineticFixedSiteMaterialParameterization(),
        optimizer_policy=PaperKineticFixedSiteMaterialSGDPolicy(
            color_learning_rate=_finite_float(
                color_learning_rate,
                name="color_learning_rate",
            ),
            density_learning_rate=_finite_float(
                density_learning_rate,
                name="density_learning_rate",
            ),
        ),
        device="cpu",
        maximum_material_state_logical_tensor_bytes=(
            maximum_material_state_logical_tensor_bytes
        ),
    )
    coordinator_state = prepare_paper_kinetic_fixed_site_material_step_state(
        provider,
        artifact_store,
        device=resolved_device,
    )
    background_generation_id = _sha256_payload(
        tuple(float(value) for value in background_rgb_f32_cpu.tolist())
    )

    def execute_step(current_material_state: Any, current_coordinator: Any) -> dict[str, Any]:
        step_index = int(current_material_state.step_index)
        material_generation_before = current_material_state.material_generation_id
        snapshot = snapshot_paper_kinetic_fixed_site_material_to_device(
            current_material_state,
            background_rgb_f32_cpu=background_rgb_f32_cpu,
            background_generation_id=background_generation_id,
            device=resolved_device,
            device_completion_fence=device_completion_fence,
            device_completion_fence_provenance=(
                device_completion_fence_provenance
            ),
        )
        snapshot_accounting = snapshot.accounting()
        result = run_paper_kinetic_fixed_site_material_only_step(
            current_coordinator,
            provider,
            batch,
            policy=step_policy,
            generation_policy=(
                PaperKineticFixedSiteMaterialOnlyGenerationPolicy(
                    step_index=step_index,
                    material_generation_id=material_generation_before,
                    background_generation_id=background_generation_id,
                    target_generation_id=target_generation_id,
                )
            ),
            global_site_rgba_f32=snapshot.site_rgba_f32_device,
            background_rgb_f32=snapshot.background_rgb_f32_device,
            native_ops=native_ops,
            backend_provenance=backend_provenance,
            device_completion_fence=device_completion_fence,
            device_completion_fence_provenance=(
                device_completion_fence_provenance
            ),
        )
        result.assert_current()
        loss_rgb_mean = float(result.loss_rgb_mean)
        coordinator_result_generation_digest = result.generation_digest
        optimizer_authorization_generation_digest = (
            result.authorization.generation_digest
        )
        coordinator_accounting = dict(result.accounting)
        gradient_receipt = (
            seal_paper_kinetic_fixed_site_material_device_gradient_receipt(
                current_material_state,
                snapshot,
                result,
                device_completion_fence=device_completion_fence,
                device_completion_fence_provenance=(
                    device_completion_fence_provenance
                ),
            )
        )
        bridge_accounting_before_apply = gradient_receipt.accounting()
        optimizer_commit_generation_digest = (
            gradient_receipt.optimizer_commit_generation_digest
        )
        device_gradient_receipt_generation_digest = (
            gradient_receipt.generation_digest
        )
        # The result owns the coordinator accumulator and its device bars.
        # Release that owner before the bridge reports post-apply residency.
        del result
        material_state_identity = id(current_material_state)
        step_receipt = (
            apply_paper_kinetic_fixed_site_material_device_gradient_receipt(
                current_material_state,
                gradient_receipt,
            )
        )
        if (
            id(current_material_state) != material_state_identity
            or current_material_state.step_index != step_index + 1
            or step_receipt.material_generation_id_before
            != material_generation_before
            or step_receipt.material_generation_id_after
            != current_material_state.material_generation_id
            or step_receipt.authorization_generation_digest
            != gradient_receipt.optimizer_commit_generation_digest
        ):
            raise ArithmeticError("fixed-site optimizer lifecycle mutation receipt changed")
        return {
            "step_index": step_index,
            "loss_rgb_mean": loss_rgb_mean,
            "material_generation_id_before": material_generation_before,
            "material_generation_id_after": (
                current_material_state.material_generation_id
            ),
            "coordinator_result_generation_digest": (
                coordinator_result_generation_digest
            ),
            "optimizer_authorization_generation_digest": (
                optimizer_authorization_generation_digest
            ),
            "optimizer_commit_generation_digest": (
                optimizer_commit_generation_digest
            ),
            "device_gradient_receipt_generation_digest": (
                device_gradient_receipt_generation_digest
            ),
            "material_step_receipt_generation_digest": (
                step_receipt.generation_digest
            ),
            "snapshot_accounting": snapshot_accounting,
            "coordinator_accounting": coordinator_accounting,
            "bridge_accounting_before_apply": (
                bridge_accounting_before_apply
            ),
            "bridge_accounting_after_apply": gradient_receipt.accounting(),
            "snapshot_accounting_after_apply": snapshot.accounting(),
            "parameter_mutation_count": 1,
            "material_state_identity_preserved": True,
            "step_receipt_loss": float(step_receipt.loss),
        }

    first_step = execute_step(material_state, coordinator_state)
    checkpoint = checkpoint_paper_kinetic_fixed_site_material_state(material_state)
    checkpoint.assert_current()
    second_step = execute_step(material_state, coordinator_state)
    restarted_state = restore_paper_kinetic_fixed_site_material_state(
        checkpoint,
        world=provider.world,
        device="cpu",
        maximum_material_state_logical_tensor_bytes=(
            maximum_material_state_logical_tensor_bytes
        ),
    )
    restarted_coordinator = prepare_paper_kinetic_fixed_site_material_step_state(
        provider,
        artifact_store,
        device=resolved_device,
        resume_material_state=restarted_state,
    )
    restarted_second_step = execute_step(
        restarted_state,
        restarted_coordinator,
    )
    if not second_step["loss_rgb_mean"] < first_step["loss_rgb_mean"]:
        raise ArithmeticError("two-step optimizer lifecycle loss was not monotone")
    if (
        restarted_second_step["loss_rgb_mean"] != second_step["loss_rgb_mean"]
        or restarted_state.material_generation_id
        != material_state.material_generation_id
        or restarted_state.step_index != material_state.step_index
        or not torch.equal(restarted_state.raw_color_f32, material_state.raw_color_f32)
        or not torch.equal(
            restarted_state.raw_density_f32,
            material_state.raw_density_f32,
        )
        or not torch.equal(
            restarted_state.site_rgba_f32,
            material_state.site_rgba_f32,
        )
        or restarted_second_step["optimizer_commit_generation_digest"]
        != second_step["optimizer_commit_generation_digest"]
        or restarted_second_step["material_step_receipt_generation_digest"]
        != second_step["material_step_receipt_generation_digest"]
    ):
        raise ArithmeticError(
            "in-process checkpoint continuation was not exact"
        )

    material_accounting = material_state.accounting(
        requested_frame_count=len(batch.samples)
    )
    peak_bridge_bytes = max(
        int(record["bridge_accounting_before_apply"]["live_bridge_tensor_bytes"])
        for record in (first_step, second_step, restarted_second_step)
    )
    return {
        "protocol": OPTIMIZER_LIFECYCLE_PROTOCOL,
        "requested_frame_count": len(batch.samples),
        "minimum_optimizer_step_count": 2,
        "executed_optimizer_step_count": 2,
        "in_process_checkpoint_replay_optimizer_step_count": 1,
        "losses_rgb_mean": (
            first_step["loss_rgb_mean"],
            second_step["loss_rgb_mean"],
        ),
        "strict_monotone_loss": True,
        "in_process_checkpoint_continuation_exact": True,
        "fresh_process_restart_verified": False,
        "checkpoint_generation_digest": checkpoint.generation_digest,
        "checkpoint_tensor_bytes": checkpoint.checkpoint_tensor_bytes,
        "final_material_generation_id": material_state.material_generation_id,
        "in_process_checkpoint_final_material_generation_id": (
            restarted_state.material_generation_id
        ),
        "material_state_accounting": material_accounting,
        "peak_bridge_tensor_bytes": peak_bridge_bytes,
        "steps": (first_step, second_step),
        "in_process_checkpoint_second_step": restarted_second_step,
        "persistent_frame_tensor_bytes": 0,
        "persistent_sample_tensor_bytes": 0,
        "persistent_target_tensor_bytes": 0,
        "persistent_prediction_tensor_bytes": 0,
        "optimizer_history_tensor_bytes": 0,
        "device_lifecycle_executed": True,
        "native_runtime_verified": False,
        "native_runtime_attestation_required_for_promotion": True,
        "cpu_proxy_verified": resolved_device.type == "cpu",
    }


def run_worldfoam_memory_scaling_trial(context: Mapping[str, Any]) -> Mapping[str, Any]:
    """Execute one cold, dense, fixed-site material-only trial on real MPS."""

    context = _mapping(context, name="trial context")
    if context.get("backend") != "mps":
        raise ValueError("this driver supports only the MPS backend")
    if context.get("material_only_scope") is not True:
        raise ValueError("the driver requires the producer's material-only scope")
    if context.get("require_real_native") is not True:
        raise ValueError("the driver refuses a producer that does not require real native ops")
    frame_count = _positive_int(context.get("frame_count"), name="frame_count")
    repeat_index = context.get("repeat_index")
    if isinstance(repeat_index, bool) or not isinstance(repeat_index, int) or repeat_index < 0:
        raise ValueError("repeat_index must be a nonnegative integer")
    config = _mapping(context.get("trial_config"), name="trial_config")
    if config.get("driver_protocol") != DRIVER_PROTOCOL:
        raise ValueError("trial config driver_protocol is missing or stale")
    admitted_frames = tuple(
        _positive_int(value, name="supported_frame_counts entry")
        for value in config.get("supported_frame_counts", ())
    )
    if frame_count not in admitted_frames:
        raise ValueError("frame_count is outside the checked-in trial matrix")
    if tuple(sorted(set(admitted_frames))) != admitted_frames:
        raise ValueError("supported_frame_counts must be unique and increasing")
    # Every row uses one identical physical dataset/camera/compiler grid and
    # changes only the requested observation density. Equating provider frames
    # with requested frames would contaminate the claimed F sweep with
    # F-dependent structural metadata and cold-certification work.
    provider_frame_count = _positive_int(
        config.get("provider_frame_count"),
        name="provider_frame_count",
    )
    if admitted_frames[-1] > provider_frame_count:
        raise ValueError("the requested frame matrix exceeds provider_frame_count")
    selected_frame_indices = _endpoint_including_frame_indices(
        dataset_frame_count=provider_frame_count,
        requested_frame_count=frame_count,
    )

    _install_runtime_paths()
    import torch
    from camera import CameraSpec
    from kinetic_compiled_cpu_artifact_store import (
        PaperKineticCompiledCpuArtifactStore,
        PaperKineticCompiledCpuArtifactStorePolicy,
    )
    from kinetic_dense_cached_native_material_request import (
        MPS_DEVICE_COMPLETION_FENCE_PROVENANCE,
        PaperKineticDenseCachedNativeMemoryPolicy,
        synchronize_mps_device_completion_fence,
    )
    from kinetic_power_word_compiler import AffineKineticPowerSites
    from paper_kinetic_active_track_program_factory import (
        PaperKineticActiveP0TrackProgramFactoryConfig,
        prepare_paper_kinetic_active_p0_track_program_factory,
    )
    from paper_kinetic_fixed_site_material_step import (
        STEP_PROVENANCE,
        PaperKineticFixedSiteMaterialOnlyGenerationPolicy,
        PaperKineticFixedSiteMaterialOnlyStepPolicy,
        prepare_paper_kinetic_fixed_site_material_step_state,
        run_paper_kinetic_fixed_site_material_only_step,
    )
    from paper_kinetic_lazy_program_bundles import (
        PaperKineticWorldInitializationRequest,
        prepare_paper_kinetic_lazy_program_bundle_provider,
    )
    from paper_kinetic_replayable_observations import (
        OBSERVATION_IDENTITY_LOGICAL_BYTES,
        TRACK_ID_LOGICAL_BYTES,
        PaperKineticDenseObservationMemoryPolicy,
    )
    from paper_training_types import SpacetimeBatch, SpacetimeSample
    from powerfoam_training_data import (
        PowerFoamRayProvider,
        PowerFoamSelectedPixelRead,
        PowerFoamTargetProvider,
    )

    if not bool(torch.backends.mps.is_available()):
        raise RuntimeError("MPS is unavailable")
    device = torch.device("mps")
    native_ops = context.get("native_ops")
    if native_ops is None or native_ops.__name__ != context.get("native_ops_module"):
        raise ValueError("producer native_ops identity/name changed")
    attest = getattr(native_ops, "assert_kinetic_memory_light_compiled_abi_registered", None)
    if not callable(attest):
        raise TypeError("native_ops lacks compiled ABI attestation")
    attest()
    extension_path = Path(str(context.get("native_extension_path", ""))).resolve()
    if not extension_path.is_file() or extension_path.suffix != ".so":
        raise ValueError("producer native extension path is missing or not a shared library")
    if _file_sha256(extension_path) != context.get("native_extension_sha256"):
        raise ValueError("producer native extension digest changed")

    image = _mapping(config.get("image"), name="image")
    height = _positive_int(image.get("height"), name="image.height")
    width = _positive_int(image.get("width"), name="image.width")
    scene = _mapping(config.get("scene"), name="scene")
    positions0 = _numeric_rows(scene.get("positions0"), name="scene.positions0", row_width=3)
    velocities = _numeric_rows(scene.get("velocities"), name="scene.velocities", row_width=3)
    weight_coefficients = _numeric_rows(
        scene.get("weight_coefficients"),
        name="scene.weight_coefficients",
        row_width=2,
    )
    material_rgba = _numeric_rows(
        scene.get("site_rgba"),
        name="scene.site_rgba",
        row_width=4,
    )
    site_count = len(positions0)
    if not (
        len(velocities) == len(weight_coefficients) == len(material_rgba) == site_count
    ):
        raise ValueError("scene geometry/material rows must have the same site count")
    if any(
        not all(0.0 <= channel <= 1.0 for channel in row[:3]) or row[3] <= 0.0
        for row in material_rgba
    ):
        raise ValueError("site_rgba requires colors in [0,1] and positive density")

    dataset_generation_digest = _sha256_payload(
        {
            "driver_protocol": DRIVER_PROTOCOL,
            "dataset_generation_id": config.get("dataset_generation_id"),
            "image": {"height": height, "width": width},
            "scene": {
                "positions0": positions0,
                "velocities": velocities,
                "weight_coefficients": weight_coefficients,
            },
        }
    )
    initializer_generation_digest = _sha256_payload(
        {
            "provenance": "worldfoam-memory-scaling-configured-kinetic-world-v1",
            "dataset_generation_digest": dataset_generation_digest,
            "positions0": positions0,
            "velocities": velocities,
            "weight_coefficients": weight_coefficients,
        }
    )

    class _ConfiguredWorldInitializer:
        provenance = "worldfoam-memory-scaling-configured-kinetic-world-v1"
        generation_digest = initializer_generation_digest

        def initialize_world(
            self,
            request: PaperKineticWorldInitializationRequest,
        ) -> AffineKineticPowerSites:
            request.assert_self_consistent()
            if (
                request.dataset_generation_digest != dataset_generation_digest
                or request.view_count != 1
                or request.frame_count != provider_frame_count
                or request.height != height
                or request.width != width
                or request.initializer_generation_digest != self.generation_digest
            ):
                raise ValueError("configured world received a foreign provider request")
            return AffineKineticPowerSites(
                positions0=torch.tensor(positions0, dtype=torch.float64, device="cpu"),
                velocities=torch.tensor(velocities, dtype=torch.float64, device="cpu"),
                weight_coefficients=torch.tensor(
                    weight_coefficients,
                    dtype=torch.float64,
                    device="cpu",
                ),
            )

    class _ProceduralNonresidentTargetSource:
        view_count = 1

        def __init__(self) -> None:
            self.frame_count = provider_frame_count
            self.height = height
            self.width = width

        def select_view_frames(
            self,
            view_indices: tuple[int, ...],
            frame_indices: tuple[int, ...],
        ):
            if not view_indices or len(view_indices) != len(frame_indices):
                raise ValueError("procedural target selection is empty or ragged")
            result = torch.empty(
                (len(view_indices), 3, height, width),
                dtype=torch.float32,
                device="cpu",
            )
            denominator = max(provider_frame_count - 1, 1)
            for row_index, (view_index, frame_index) in enumerate(
                zip(view_indices, frame_indices, strict=True)
            ):
                if view_index != 0 or not 0 <= frame_index < provider_frame_count:
                    raise IndexError("procedural target selection left its grid")
                phase = float(frame_index) / float(denominator)
                result[row_index, 0].fill_(0.15 + 0.35 * phase)
                result[row_index, 1].fill_(0.55 - 0.20 * phase)
                result[row_index, 2].fill_(0.25 + 0.10 * phase)
            return result

        def select_view_frame_pixels_cpu(
            self,
            view_indices: tuple[int, ...],
            frame_indices: tuple[int, ...],
            pixel_indices: tuple[int, ...],
            *,
            maximum_source_decode_tensor_bytes: int,
        ) -> PowerFoamSelectedPixelRead:
            if (
                not view_indices
                or len(view_indices) != len(frame_indices)
                or len(view_indices) != len(pixel_indices)
            ):
                raise ValueError("procedural selected-pixel request is empty or ragged")
            required_peak_bytes = len(frame_indices) * 32
            if required_peak_bytes > int(maximum_source_decode_tensor_bytes):
                raise MemoryError(
                    "procedural selected-pixel read exceeds its source-decode budget"
                )
            if any(view_index != 0 for view_index in view_indices):
                raise IndexError("procedural selected-pixel request left its view grid")
            if any(
                not 0 <= frame_index < provider_frame_count
                for frame_index in frame_indices
            ):
                raise IndexError("procedural selected-pixel request left its frame grid")
            if any(
                not 0 <= pixel_index < height * width
                for pixel_index in pixel_indices
            ):
                raise IndexError("procedural selected-pixel request left its pixel grid")
            denominator = max(provider_frame_count - 1, 1)
            frame_values = torch.tensor(
                frame_indices,
                dtype=torch.float32,
                device="cpu",
            )
            phase = frame_values / float(denominator)
            result = torch.stack(
                (
                    0.15 + 0.35 * phase,
                    0.55 - 0.20 * phase,
                    0.25 + 0.10 * phase,
                ),
                dim=1,
            ).contiguous()
            return PowerFoamSelectedPixelRead.seal(
                result,
                selection_mode="direct_pixels",
                source_provenance=(
                    "worldfoam_memory_scaling_driver/"
                    "deterministic_procedural_direct_pixels_v1"
                ),
                source_visible_peak_logical_tensor_bytes_upper_bound=(
                    required_peak_bytes
                ),
            )

        def residency(self) -> dict[str, Any]:
            return {
                "source_kind": "deterministic_procedural_nonresident_memory_fixture",
                "source_device": "cpu",
                "logical_bytes": provider_frame_count * 3 * height * width * 4,
                "resident_bytes": 0,
                "full_source_resident": False,
                "disk_lazy_decode": False,
            }

    camera_cfg = _mapping(config.get("camera"), name="camera")
    camera_to_world = torch.eye(4, dtype=torch.float64, device="cpu")
    translation = camera_cfg.get("translation", (0.0, 0.0, 0.0))
    translation_rows = _numeric_rows((translation,), name="camera.translation", row_width=3)
    camera_to_world[:3, 3] = torch.tensor(translation_rows[0], dtype=torch.float64)
    camera = CameraSpec(
        fx=_finite_float(camera_cfg.get("fx"), name="camera.fx"),
        fy=_finite_float(camera_cfg.get("fy"), name="camera.fy"),
        cx=_finite_float(camera_cfg.get("cx"), name="camera.cx"),
        cy=_finite_float(camera_cfg.get("cy"), name="camera.cy"),
        camera_to_world=camera_to_world,
        lens_model="pinhole",
        distortion=None,
    )
    target_provider = PowerFoamTargetProvider(
        source=_ProceduralNonresidentTargetSource(),
        device=device,
    )
    ray_provider = PowerFoamRayProvider(
        cameras=(tuple(camera for _ in range(provider_frame_count)),),
        height=height,
        width=width,
        device=device,
    )
    frame_times = tuple(
        float(frame_index) / float(provider_frame_count - 1)
        for frame_index in range(provider_frame_count)
    )

    compiler = _mapping(config.get("compiler"), name="compiler")
    factory = prepare_paper_kinetic_active_p0_track_program_factory(
        PaperKineticActiveP0TrackProgramFactoryConfig(
            near=_finite_float(compiler.get("near"), name="compiler.near"),
            far=_finite_float(compiler.get("far"), name="compiler.far"),
            node_count=_positive_int(compiler.get("node_count"), name="compiler.node_count"),
            maximum_sites_per_track_compile=_positive_int(
                compiler.get("maximum_sites_per_track_compile"),
                name="compiler.maximum_sites_per_track_compile",
            ),
            maximum_charts_per_track=_positive_int(
                compiler.get("maximum_charts_per_track"),
                name="compiler.maximum_charts_per_track",
            ),
            maximum_owner_runs_per_chart=_positive_int(
                compiler.get("maximum_owner_runs_per_chart"),
                name="compiler.maximum_owner_runs_per_chart",
            ),
            rank_selection_provenance=str(compiler.get("rank_selection_provenance", "")),
        )
    )
    streaming = _mapping(config.get("streaming"), name="streaming")
    maximum_tracks_per_request = _positive_int(
        streaming.get("maximum_tracks_per_request"),
        name="streaming.maximum_tracks_per_request",
    )
    provider = prepare_paper_kinetic_lazy_program_bundle_provider(
        dataset_generation_digest=dataset_generation_digest,
        target_provider=target_provider,
        ray_provider=ray_provider,
        frame_times=frame_times,
        height=height,
        width=width,
        maximum_tracks_per_bundle=maximum_tracks_per_request,
        maximum_observations_per_bundle=_positive_int(
            streaming.get("maximum_observations_per_bundle"),
            name="streaming.maximum_observations_per_bundle",
        ),
        maximum_rows_per_native_block=_positive_int(
            streaming.get("maximum_rows_per_native_block"),
            name="streaming.maximum_rows_per_native_block",
        ),
        world_initializer=_ConfiguredWorldInitializer(),
        program_factory=factory,
    )

    budgets = _mapping(config.get("budgets_bytes"), name="budgets_bytes")
    maximum_artifact_bytes = _positive_int(
        budgets.get("maximum_artifact_accounted_bytes"),
        name="budgets_bytes.maximum_artifact_accounted_bytes",
    )
    store = PaperKineticCompiledCpuArtifactStore(
        PaperKineticCompiledCpuArtifactStorePolicy(
            maximum_entries=_positive_int(
                streaming.get("artifact_store_maximum_entries"),
                name="streaming.artifact_store_maximum_entries",
            ),
            maximum_resident_accounted_bytes=_positive_int(
                budgets.get("artifact_store_maximum_resident_accounted_bytes"),
                name="budgets_bytes.artifact_store_maximum_resident_accounted_bytes",
            ),
        )
    )
    state = prepare_paper_kinetic_fixed_site_material_step_state(
        provider,
        store,
        device=device,
    )

    retained_frame_limit = _positive_int(
        streaming.get("maximum_retained_frame_metadata_count"),
        name="streaming.maximum_retained_frame_metadata_count",
    )
    chunk_observation_count = _positive_int(
        streaming.get("maximum_chunk_observation_count"),
        name="streaming.maximum_chunk_observation_count",
    )
    observation_policy = PaperKineticDenseObservationMemoryPolicy(
        maximum_persistent_observation_count=0,
        maximum_persistent_observation_logical_bytes=0,
        maximum_retained_frame_metadata_count=retained_frame_limit,
        maximum_retained_frame_metadata_logical_bytes=retained_frame_limit * 3 * 8,
        maximum_live_generated_observation_count=1,
        maximum_live_generated_observation_logical_bytes=(
            OBSERVATION_IDENTITY_LOGICAL_BYTES
        ),
        maximum_request_track_count=maximum_tracks_per_request,
        maximum_request_track_logical_bytes=(
            maximum_tracks_per_request * TRACK_ID_LOGICAL_BYTES
        ),
        maximum_chunk_observation_count=chunk_observation_count,
        maximum_chunk_observation_logical_bytes=(
            chunk_observation_count * OBSERVATION_IDENTITY_LOGICAL_BYTES
        ),
    )
    request_policy = PaperKineticDenseCachedNativeMemoryPolicy(
        maximum_lane_resident_logical_tensor_bytes=_positive_int(
            budgets.get("maximum_lane_resident_logical_tensor_bytes"),
            name="budgets_bytes.maximum_lane_resident_logical_tensor_bytes",
        ),
        maximum_active_node_and_union_bar_tensor_bytes=_positive_int(
            budgets.get("maximum_active_node_and_union_bar_tensor_bytes"),
            name="budgets_bytes.maximum_active_node_and_union_bar_tensor_bytes",
        ),
        maximum_decoded_frame_scratch_tensor_bytes=_positive_int(
            budgets.get("maximum_decoded_frame_scratch_tensor_bytes"),
            name="budgets_bytes.maximum_decoded_frame_scratch_tensor_bytes",
        ),
        maximum_chunk_target_tensor_bytes=_positive_int(
            budgets.get("maximum_chunk_target_tensor_bytes"),
            name="budgets_bytes.maximum_chunk_target_tensor_bytes",
        ),
        maximum_target_decode_bridge_peak_logical_tensor_bytes=_positive_int(
            budgets.get("maximum_target_decode_bridge_peak_logical_tensor_bytes"),
            name="budgets_bytes.maximum_target_decode_bridge_peak_logical_tensor_bytes",
        ),
        maximum_sample_materialization_logical_tensor_bytes=_positive_int(
            budgets.get("maximum_sample_materialization_logical_tensor_bytes"),
            name="budgets_bytes.maximum_sample_materialization_logical_tensor_bytes",
        ),
        maximum_sample_launch_tensor_bytes=_positive_int(
            budgets.get("maximum_sample_launch_tensor_bytes"),
            name="budgets_bytes.maximum_sample_launch_tensor_bytes",
        ),
        maximum_request_geometry_bar_tensor_bytes=1,
        maximum_geometry_bridge_visible_peak_logical_tensor_bytes=1,
    )
    step_policy = PaperKineticFixedSiteMaterialOnlyStepPolicy(
        observation_memory_policy=observation_policy,
        request_memory_policy=request_policy,
        maximum_world_site_count=site_count,
        maximum_material_state_logical_tensor_bytes=_positive_int(
            budgets.get("maximum_material_state_logical_tensor_bytes"),
            name="budgets_bytes.maximum_material_state_logical_tensor_bytes",
        ),
        maximum_material_checkpoint_logical_tensor_bytes=_positive_int(
            budgets.get("maximum_material_checkpoint_logical_tensor_bytes"),
            name="budgets_bytes.maximum_material_checkpoint_logical_tensor_bytes",
        ),
        maximum_step_accumulator_logical_tensor_bytes=_positive_int(
            budgets.get("maximum_step_accumulator_logical_tensor_bytes"),
            name="budgets_bytes.maximum_step_accumulator_logical_tensor_bytes",
        ),
        maximum_tracks_per_request=maximum_tracks_per_request,
        maximum_artifact_accounted_bytes=maximum_artifact_bytes,
        maximum_samples_per_launch=_positive_int(
            streaming.get("maximum_samples_per_launch"),
            name="streaming.maximum_samples_per_launch",
        ),
        cone_tolerance=_finite_float(
            compiler.get("cone_tolerance"),
            name="compiler.cone_tolerance",
        ),
    )

    raw_color_values = tuple(
        tuple(math.log(max(min(channel, 1.0 - 1.0e-4), 1.0e-4) / (1.0 - max(min(channel, 1.0 - 1.0e-4), 1.0e-4))) for channel in row[:3])
        for row in material_rgba
    )
    raw_density_values = tuple(math.log(math.expm1(row[3])) for row in material_rgba)
    raw_color_f32 = torch.tensor(raw_color_values, dtype=torch.float32, device=device)
    raw_density_f32 = torch.tensor(raw_density_values, dtype=torch.float32, device=device)
    site_rgba_f32 = torch.tensor(material_rgba, dtype=torch.float32, device=device)
    raw_color_grad_f32 = torch.zeros((site_count, 3), dtype=torch.float32, device=device)
    raw_density_grad_f32 = torch.zeros((site_count,), dtype=torch.float32, device=device)
    background = config.get("background_rgb")
    background_rows = _numeric_rows((background,), name="background_rgb", row_width=3)
    background_rgb_f32 = torch.tensor(
        background_rows[0],
        dtype=torch.float32,
        device=device,
    )
    material_tensors = (
        raw_color_f32,
        raw_density_f32,
        site_rgba_f32,
        raw_color_grad_f32,
        raw_density_grad_f32,
    )
    persistent_bytes = sum(
        int(tensor.numel()) * int(tensor.element_size())
        for tensor in material_tensors
    )
    material_generation_id = _sha256_payload(
        {
            "driver_protocol": DRIVER_PROTOCOL,
            "world_generation_digest": provider.world.generation_digest,
            "site_rgba": material_rgba,
            "layout": "fixed-site-p0-rgb-density-manual-sgd-no-history",
        }
    )
    material_state_accounting = {
        "provenance": "worldfoam-memory-scaling-live-mps-fixed-site-material-v1",
        "material_generation_id": material_generation_id,
        "world_generation_digest": provider.world.generation_digest,
        "sites_content_digest": provider.world.sites_content_digest,
        "site_count": site_count,
        "optimizer": "manual_sgd",
        "optimizer_history_tensor_bytes": 0,
        "persistent_parameter_tensor_bytes": 16 * site_count,
        "persistent_physical_snapshot_tensor_bytes": 16 * site_count,
        "persistent_raw_gradient_buffer_tensor_bytes": 16 * site_count,
        "total_persistent_tensor_bytes": persistent_bytes,
        "persistent_scalar_count_per_site": 12,
        "requested_frame_count": frame_count,
        "dataset_frame_count": provider_frame_count,
        "requested_frame_subset_kind": "endpoint_including_even_index_v1",
        "requested_frame_indices_sha256": _sha256_payload(
            selected_frame_indices
        ),
        "frame_dependent_parameter_bytes": 0,
        "persistent_frame_tensor_bytes": 0,
        "persistent_sample_tensor_bytes": 0,
        "persistent_target_tensor_bytes": 0,
        "persistent_prediction_tensor_bytes": 0,
        "material_layout": "rgb_then_density",
        "material_temporal_basis": "P0",
        "geometry_trainable": False,
        "material_state_device_scope": "driver-owned-live-mps-fenced-read-only-step",
        "accelerator_optimizer_update_supported": False,
        "allocator_peak_measured": False,
    }
    if persistent_bytes != 48 * site_count:
        raise ArithmeticError("actual MPS fixed-site material buffers changed layout")

    batch = SpacetimeBatch(
        samples=tuple(
            SpacetimeSample(view_index=0, frame_index=frame_index)
            for frame_index in selected_frame_indices
        ),
        epoch=0,
        batch_index=repeat_index,
        completes_epoch=True,
    )
    generation_policy = PaperKineticFixedSiteMaterialOnlyGenerationPolicy(
        step_index=0,
        material_generation_id=material_generation_id,
        background_generation_id=_sha256_payload(background_rows[0]),
        target_generation_id=_sha256_payload(
            {
                "source": "deterministic-procedural-nonresident",
                "provider_frame_count": provider_frame_count,
                "selected_frame_indices": selected_frame_indices,
                "height": height,
                "width": width,
            }
        ),
    )

    backend_provenance = (
        "mps-real-compiled-worldfoam-lane2-fused-slab-v0/"
        + str(context.get("native_extension_sha256"))
    )
    with _NativeCallAudit(native_ops) as native_audit:
        result = run_paper_kinetic_fixed_site_material_only_step(
            state,
            provider,
            batch,
            policy=step_policy,
            generation_policy=generation_policy,
            global_site_rgba_f32=site_rgba_f32,
            background_rgb_f32=background_rgb_f32,
            native_ops=native_ops,
            backend_provenance=backend_provenance,
            device_completion_fence=synchronize_mps_device_completion_fence,
            device_completion_fence_provenance=(
                MPS_DEVICE_COMPLETION_FENCE_PROVENANCE
            ),
        )
        result.assert_current()
        step_accounting = dict(result.accounting)

    expected_native_calls = {
        _NATIVE_FORWARD: int(step_accounting["node_forward_launch_count"]),
        _NATIVE_FULL_VJP: 0,
        _NATIVE_MATERIAL_VJP: int(
            step_accounting["native_material_vjp_launch_count"]
        ),
        _NATIVE_SAMPLE_PREPARE: int(step_accounting["sample_launch_count"]),
        _NATIVE_SAMPLE_LAUNCH: int(step_accounting["sample_launch_count"]),
    }
    if native_audit.counts != expected_native_calls:
        raise RuntimeError(
            "observed native call counts do not match coordinator receipts: "
            f"observed={native_audit.counts}, expected={expected_native_calls}"
        )
    if (
        step_accounting.get("provenance") != STEP_PROVENANCE
        or step_accounting.get("device_type") != "mps"
        or step_accounting.get("backend_provenance") != backend_provenance
        or step_accounting.get("device_completion_fence_provenance")
        != MPS_DEVICE_COMPLETION_FENCE_PROVENANCE
        or int(step_accounting.get("native_lane_fence_call_count", 0)) < 1
        or int(step_accounting.get("total_step_completion_fence_call_count", 0)) < 1
        or step_accounting.get("optimizer_step_executed") is not False
        or step_accounting.get("parameter_mutation_count") != 0
        or step_accounting.get("full_geometry") is not False
        or step_accounting.get("selected_pixel_read_mode") != "direct_pixels"
        or step_accounting.get("selected_pixel_read_acceptance_capable") is not True
        or step_accounting.get(
            "target_source_decode_budget_enforced_before_allocation"
        )
        is not True
        or step_accounting.get("full_frame_target_materialization_count") != 0
        or step_accounting.get("full_frame_fallback_observation_count") != 0
        or step_accounting.get("direct_selected_pixel_observation_count")
        != step_accounting.get("streamed_observation_count")
    ):
        raise RuntimeError("coordinator result does not prove the real material-only MPS path")
    if step_accounting.get("material_state_logical_tensor_bytes") != persistent_bytes:
        raise RuntimeError("coordinator and actual MPS material layout disagree")
    maximum_node_count = max(int(value) for value in step_accounting["chart_node_ranks"])
    persistent_geometry_bytes = int(provider.world.sites.parameter_bytes)
    if persistent_geometry_bytes != int(
        step_accounting["persistent_world_geometry_tensor_bytes"]
    ):
        raise RuntimeError("coordinator and driver geometry bytes disagree")

    kernel_resource_attestation = asdict(
        native_ops.kinetic_memory_light_selected_kernel_resource_attestation(
            site_rgba_f32
        )
    )
    runtime_measurements = {
        "fake_native_backend": False,
        "cold_compile_included": (
            int(step_accounting["cold_artifact_acquisition_count"]) > 0
            and int(step_accounting["artifact_store_warm_hit_count"]) == 0
        ),
        "native_runtime_verified": True,
        "production_coordinator_integrated": True,
        "native_runtime_measurement_provenance": (
            "attested-extension-plus-observed-op-counts-matched-to-sealed-coordinator-receipts-v1"
        ),
        "native_call_counts": dict(native_audit.counts),
        "selected_kernel_resource_attestation": kernel_resource_attestation,
        "driver_capability_manifest": (
            WORLDFOAM_MEMORY_SCALING_DRIVER_CAPABILITIES
        ),
    }
    synchronize_mps_device_completion_fence()
    return {
        "native_ops_used": native_ops,
        "step_accounting": step_accounting,
        "material_state_accounting": material_state_accounting,
        "runtime_measurements": runtime_measurements,
        "maximum_node_count": maximum_node_count,
        "persistent_world_geometry_tensor_bytes": persistent_geometry_bytes,
    }


__all__ = [
    "DRIVER_PROTOCOL",
    "OPTIMIZER_LIFECYCLE_PROTOCOL",
    "WORLDFOAM_MEMORY_SCALING_DRIVER_CAPABILITIES",
    "run_worldfoam_fixed_site_optimizer_lifecycle",
    "run_worldfoam_memory_scaling_trial",
]
