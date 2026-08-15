"""Real WorldFoam executors for the frozen G4 public-quality ablation.

Two rows share one initialization, material parameterization, compiler, loss,
optimizer, target/ray providers, and checkpoint format:

``worldfoam_native4d``
    Streams bounded spatial bundles and applies one fused-union reverse over
    all selected times in a bundle.

``worldfoam_framewise_replay``
    Compiles the same continuous retained-depth program for one bounded spatial
    bundle, replays each selected image sequentially, copies its bars to the
    CPU, and releases the frame before the next replay.

Neither route retains a video tensor or a full camera's compiled programs.
Geometry updates publish an empty bounded artifact store and the next step
recompiles only the spatial bundles it consumes.  The row worker still visits
every public target/ray chunk; this executor validates that coverage but reads
targets through the exact sealed provider used by the native compiler.

Capability stays fail-closed until both the rebuilt post-103 native ABI and the
separate runtime-verification receipt exist.  Therefore importing this module
is not a claim that WorldFoam fits the G4 workload in memory.  A successful row
must report sampled process RSS and MPS driver-allocation peaks.

The frozen all-pixel v1 schedule also remains fail-closed on tractability.  Its
current exact compiler has no certified neighboring-track/template reuse, so a
300-step row would cold-compile roughly 113--115 million ``(view, pixel)``
tracks under the three frozen seeds (117,964,800 is the two-view-per-step upper
bound).  A one-pixel runtime smoke cannot promote that schedule to production
readiness.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import resource
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from camera import build_camera_rays_at_pixels
from external_paths import PROJECT_ROOT, ensure_sys_path
from paper_kinetic_compiled_framewise_full_geometry_control import (
    prepare_paper_kinetic_compiled_framewise_program_provider,
)
from paper_kinetic_fixed_camera_combined_state import (
    PaperKineticFixedCameraCombinedState,
    prepare_paper_kinetic_fixed_camera_combined_state,
)
from paper_kinetic_fixed_site_material_device_bridge import (
    snapshot_paper_kinetic_fixed_site_material_to_device,
)
from paper_kinetic_fixed_site_material_state import (
    prepare_paper_kinetic_fixed_site_material_state,
)
from paper_kinetic_lazy_full_geometry_device_bridge import (
    _TRANSACTION_CONSUMPTION_AUTHORITY,
    seal_paper_kinetic_lazy_full_geometry_device_gradient_receipt,
)
from paper_kinetic_lazy_full_geometry_step import (
    FUSED_UNION_V2,
    STAGED_SPARSE,
    PaperKineticLazyFullGeometryMemoryPolicy,
    PaperKineticLazyNativeFullGeometryStepResult,
    run_paper_kinetic_lazy_native_full_geometry_step,
)
from paper_kinetic_lazy_program_bundles import (
    PaperKineticLazyProgramBundleProvider,
    PaperKineticObservation,
    iter_canonical_observations_from_spacetime_batch,
    prepare_paper_kinetic_lazy_program_bundle_provider,
)
from paper_kinetic_runtime_paths import ensure_worldfoam_lane2_research_path
from paper_kinetic_streaming_combined_update import (
    PaperKineticStreamingCombinedPromotionReceipt,
    apply_paper_kinetic_streaming_combined_sgd,
    seal_paper_kinetic_streaming_combined_gradient,
)
from worldfoam_public_quality_inputs import WorldFoamPublicQualityInputs
from worldfoam_g4_tractability import audit_worldfoam_g4_full_schedule
from paper_training_types import SpacetimeBatch


ensure_worldfoam_lane2_research_path()

from kinetic_compiled_cpu_artifact_store import (  # noqa: E402
    PaperKineticCompiledCpuArtifactStore,
)
from kinetic_dense_cached_native_material_request import (  # noqa: E402
    MPS_DEVICE_COMPLETION_FENCE_PROVENANCE,
    synchronize_mps_device_completion_fence,
)
from kinetic_lazy_native_material_step import (  # noqa: E402
    paper_kinetic_observation_manifest_digest,
    prepare_paper_kinetic_lazy_native_trainer_state,
)


EXECUTOR_SCHEMA_VERSION = 1
EXECUTOR_PROVENANCE = "worldfoam-native4d-g4-public-quality-executor-v1"
CHECKPOINT_SCHEMA = "worldfoam-native4d-g4-raw-checkpoint-v1"
VARIANT_ROOT = (
    PROJECT_ROOT
    / "third_party"
    / "fast-mac-gsplat"
    / "variants"
    / "world_foam_lane2_fused_slab_v0"
)
RUNTIME_CAPABILITY_PATH = (
    PROJECT_ROOT
    / "src"
    / "train"
    / "worldfoam_native4d_public_quality_capabilities.json"
)
RUNTIME_EVIDENCE_PATH = RUNTIME_CAPABILITY_PATH.with_name(
    "worldfoam_native4d_public_quality_capabilities.evidence.json"
)
_EXPECTED_RUNTIME_CAPABILITIES = {
    "schema_version": 1,
    "status": "runtime_verified",
    "row_kind": "worldfoam-native4d-public-quality-row-v1",
    "supported_routes": [
        "worldfoam_native4d",
        "worldfoam_framewise_replay",
        "world_tubes",
        "dynamic_3dgs",
    ],
    "real_native_only": True,
    "public_neural3d_targets": True,
    "heldout_camera_evaluation": True,
    "full_temporal_evaluation": True,
    "compiled_shared_adjoint": True,
    "same_representation_framewise_replay": True,
    "final_checkpoint_metrics": True,
    "wandb_run_file": True,
    "proxy_or_fake_native_permitted": False,
    "smoke_as_public_evidence_permitted": False,
}
_UNKNOWN_SESSION_COMPLETION_QUARANTINE: list[Any] = []
SUPPORTED_ROUTES = frozenset(
    {"worldfoam_native4d", "worldfoam_framewise_replay"}
)
CERTIFIED_SPATIAL_COMPILE_REUSE = False


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_content_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(repr(tuple(int(item) for item in value.shape)).encode("ascii"))
    digest.update(memoryview(value.numpy()).cast("B"))
    return digest.hexdigest()


def _tensor_bytes(*tensors: torch.Tensor) -> int:
    return sum(int(tensor.numel()) * int(tensor.element_size()) for tensor in tensors)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_runtime_capability_receipt(
    *,
    context: Any,
    native_library_sha256: str | None,
) -> Mapping[str, Any] | None:
    """Load a capability only when its sealed evidence matches this process."""

    if (
        not RUNTIME_CAPABILITY_PATH.is_file()
        or not RUNTIME_EVIDENCE_PATH.is_file()
        or not _is_sha256(native_library_sha256)
    ):
        return None
    try:
        payload = json.loads(RUNTIME_CAPABILITY_PATH.read_text(encoding="utf-8"))
        evidence = json.loads(RUNTIME_EVIDENCE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        not isinstance(payload, dict)
        or payload != _EXPECTED_RUNTIME_CAPABILITIES
        or not isinstance(evidence, dict)
    ):
        return None
    evidence_without_digest = {
        key: value for key, value in evidence.items() if key != "evidence_sha256"
    }
    if (
        evidence.get("evidence_sha256")
        != _canonical_sha256(evidence_without_digest)
        or evidence.get("kind")
        != "worldfoam-g4-public-quality-runtime-attestation-v1"
        or evidence.get("status") != "runtime_verified"
        or evidence.get("scene") != context.request.scene
        or evidence.get("seed") != context.request.seed
        or evidence.get("sample_id") != context.protocol.dataset.sample_id
        or evidence.get("dataset_capability_sha256")
        != context.dataset_capability["capability_sha256"]
        or evidence.get("capabilities") != payload
        or evidence.get("source")
        != {
            "repository_commit": context.source_commit,
            "repository_dirty": False,
        }
    ):
        return None
    protocol = evidence.get("protocol")
    native_library = evidence.get("worldfoam_native_library")
    executor_sources = evidence.get("executor_sources")
    route_smokes = evidence.get("route_runtime_smokes")
    if (
        not isinstance(protocol, dict)
        or protocol.get("sha256") != context.scene_receipt["protocol_sha256"]
        or not isinstance(native_library, dict)
        or native_library.get("sha256") != native_library_sha256
        or not isinstance(executor_sources, dict)
        or not isinstance(route_smokes, dict)
    ):
        return None
    source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for route in SUPPORTED_ROUTES:
        source = executor_sources.get(route)
        smoke = route_smokes.get(route)
        if (
            not isinstance(source, dict)
            or source.get("sha256") != source_sha256
            or not isinstance(smoke, dict)
            or smoke.get("route") != route
            or smoke.get("real_native") is not True
            or smoke.get("paper_evidence_eligible") is not False
            or not _is_sha256(smoke.get("source_receipt_sha256"))
            or not _is_sha256(smoke.get("native_receipt_sha256"))
        ):
            return None
    try:
        from worldfoam_native_heldout_prediction import (
            PREDICTION_ABI_SCHEMA_SHA256,
        )
    except ImportError:
        return None
    if (
        evidence.get("worldfoam_prediction_abi_schema_sha256")
        != PREDICTION_ABI_SCHEMA_SHA256
    ):
        return None
    return payload


def _load_and_attest_native_ops() -> tuple[Any | None, str | None, str | None]:
    """Return ops/hash/error without allocating an accelerator tensor."""

    try:
        if not VARIANT_ROOT.is_dir():
            raise FileNotFoundError(f"WorldFoam native variant is missing: {VARIANT_ROOT}")
        ensure_sys_path(VARIANT_ROOT)
        package = importlib.import_module("torch_world_foam_lane2_fused_slab")
        ops = importlib.import_module("torch_world_foam_lane2_fused_slab.ops")
        assertion = getattr(
            package,
            "assert_kinetic_lazy_full_geometry_compiled_abi_registered",
            None,
        )
        if not callable(assertion):
            raise RuntimeError("native package lacks full-geometry ABI attestation")
        assertion()
        for name in (
            "prepare_kinetic_ragged_p0_lie_sample_block",
            "kinetic_ragged_p0_lie_sample_accumulate_launch_only",
        ):
            if not callable(getattr(ops, name, None)):
                raise RuntimeError(f"native prediction ABI is missing {name}")
        from worldfoam_native_heldout_prediction import (
            assert_worldfoam_native_heldout_prediction_abi,
        )

        assert_worldfoam_native_heldout_prediction_abi(ops)
        libraries = tuple(
            sorted(
                (VARIANT_ROOT / "torch_world_foam_lane2_fused_slab").glob(
                    "_C*.so"
                )
            )
        )
        if len(libraries) != 1:
            raise RuntimeError(
                f"expected one WorldFoam native library, found {len(libraries)}"
            )
        library_digest = hashlib.sha256(libraries[0].read_bytes()).hexdigest()
        return ops, library_digest, None
    except BaseException as error:
        return None, None, f"{type(error).__qualname__}: {error}"


def _full_schedule_tractability_blocker(context: Any) -> str | None:
    """Reject the frozen v1 workload until spatial compile reuse is real.

    These are exact scheduler counts, not a timing extrapolation.  The current
    active-track factory compiles every unique ``(view, pixel)`` against the
    complete 1,024-site world after every geometry update and deliberately
    retains no compiled-program cache.  Bounded peak residency therefore does
    not imply a tractable public row.
    """

    selected_receipt = getattr(context.work_plan, "workload_receipt", None)
    if selected_receipt is not None:
        if getattr(selected_receipt, "tractability_preflight_passed", False) is True:
            return None
        return "worldfoam_selected_ray_schedule_tractability_preflight_failed"
    audit = audit_worldfoam_g4_full_schedule(
        protocol=context.protocol,
        work_plan=context.work_plan,
        compiler=context.scene_receipt["compiler"],
        runtime=context.scene_receipt["worldfoam_runtime"],
    )
    if CERTIFIED_SPATIAL_COMPILE_REUSE:
        return None
    return audit.blocker


def _process_peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


class _PeakSampler:
    """Fresh-session sampled peaks; logical bounds are reported elsewhere."""

    def __init__(self, *, interval_seconds: float = 0.02) -> None:
        self._interval_seconds = float(interval_seconds)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.process_rss_bytes = 0
        self.mps_driver_allocated_bytes = 0
        self.sample_count = 0

    def _sample(self) -> None:
        rss = _process_peak_rss_bytes()
        mps = 0
        try:
            mps = int(torch.mps.driver_allocated_memory())
        except (AttributeError, RuntimeError):
            mps = 0
        with self._lock:
            self.process_rss_bytes = max(self.process_rss_bytes, rss)
            self.mps_driver_allocated_bytes = max(
                self.mps_driver_allocated_bytes,
                mps,
            )
            self.sample_count += 1

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("peak sampler already started")
        self._sample()

        def loop() -> None:
            while not self._stop.wait(self._interval_seconds):
                self._sample()

        self._thread = threading.Thread(
            target=loop,
            name="worldfoam-g4-peak-sampler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> tuple[int, int]:
        thread = self._thread
        if thread is not None:
            self._stop.set()
            thread.join(timeout=1.0)
            self._sample()
            self._thread = None
        with self._lock:
            return self.process_rss_bytes, self.mps_driver_allocated_bytes

@dataclass(frozen=True)
class _ReplayableStepObservations:
    batch: Any
    image_pixel_count: int

    @property
    def expected_observation_count(self) -> int:
        return len(self.batch.samples) * self.image_pixel_count

    def __iter__(self) -> Iterator[PaperKineticObservation]:
        yield from iter_canonical_observations_from_spacetime_batch(
            self.batch,
            pixel_indices_by_batch_position=tuple(
                range(self.image_pixel_count) for _sample in self.batch.samples
            ),
            image_pixel_count=self.image_pixel_count,
        )


@dataclass(frozen=True)
class _ReplayableSelectedStepObservations:
    batch: Any
    pixel_indices: tuple[int, ...]
    image_pixel_count: int

    @property
    def expected_observation_count(self) -> int:
        return len(self.batch.samples) * len(self.pixel_indices)

    def __iter__(self) -> Iterator[PaperKineticObservation]:
        yield from iter_canonical_observations_from_spacetime_batch(
            self.batch,
            pixel_indices_by_batch_position=tuple(
                self.pixel_indices for _sample in self.batch.samples
            ),
            image_pixel_count=self.image_pixel_count,
        )


@dataclass(frozen=True)
class _ReplayableFrameChunkObservations:
    view_index: int
    frame_index: int
    batch_position: int
    pixel_start: int
    pixel_stop: int
    image_pixel_count: int

    @property
    def expected_observation_count(self) -> int:
        return self.pixel_stop - self.pixel_start

    def __iter__(self) -> Iterator[PaperKineticObservation]:
        for pixel in range(self.pixel_start, self.pixel_stop):
            yield PaperKineticObservation(
                observation_id=self.batch_position * self.image_pixel_count + pixel,
                view_index=self.view_index,
                frame_index=self.frame_index,
                pixel_index=pixel,
            )


@dataclass(frozen=True)
class _ReplayableSelectedFrameChunkObservations:
    view_index: int
    frame_index: int
    batch_position: int
    pixel_indices: tuple[int, ...]
    image_pixel_count: int

    @property
    def expected_observation_count(self) -> int:
        return len(self.pixel_indices)

    def __iter__(self) -> Iterator[PaperKineticObservation]:
        for pixel in self.pixel_indices:
            yield PaperKineticObservation(
                observation_id=self.batch_position * self.image_pixel_count + pixel,
                view_index=self.view_index,
                frame_index=self.frame_index,
                pixel_index=pixel,
            )


def _stage_policy(
    policy: PaperKineticLazyFullGeometryMemoryPolicy,
) -> PaperKineticLazyFullGeometryMemoryPolicy:
    result = PaperKineticLazyFullGeometryMemoryPolicy(
        maximum_global_geometry_bar_logical_tensor_bytes=(
            policy.maximum_global_geometry_bar_logical_tensor_bytes
        ),
        maximum_geometry_bridge_visible_peak_logical_tensor_bytes=(
            policy.maximum_geometry_bridge_visible_peak_logical_tensor_bytes
        ),
        maximum_fused_union_transaction_scratch_tensor_bytes=0,
    )
    result.assert_valid(reverse_mode=STAGED_SPARSE)
    return result


class WorldFoamPublicQualityExecutor:
    def __init__(self, *, context: Any) -> None:
        route = str(context.request.route)
        if route not in SUPPORTED_ROUTES:
            raise ValueError("WorldFoam executor received another route")
        self.route = route
        self._native_ops, self._native_library_sha256, self._native_error = (
            _load_and_attest_native_ops()
        )
        self._runtime_receipt = _read_runtime_capability_receipt(
            context=context,
            native_library_sha256=self._native_library_sha256,
        )
        self._tractability_error = _full_schedule_tractability_blocker(context)

    @property
    def production_ready(self) -> bool:
        return (
            self._native_ops is not None
            and _is_sha256(self._native_library_sha256)
            and self._runtime_receipt is not None
            and self._tractability_error is None
        )

    def capability(self, context: Any) -> Mapping[str, Any]:
        route = str(context.request.route)
        if route != self.route:
            raise ValueError("executor capability requested for another route")
        ready = self.production_ready
        return {
            "schema_version": EXECUTOR_SCHEMA_VERSION,
            "route": route,
            "lane": context.route_spec["lane"],
            "execution_mode": context.route_spec["execution_mode"],
            "backend": context.route_spec["backend"],
            "real_native": ready,
            # The native library is loaded and source-fresh here, but this
            # row-level flag is reserved for a separately sealed artifact
            # attestation rather than inferred from successful import.
            "native_extension_attested": False,
            "fake_native": False,
            "source_only": not ready,
            "procedural_target": False,
            "public_target_provider": True,
            "heldout_evaluator": ready,
            "full_geometry_trainable": ready,
            "compiled_shared_adjoint": (
                ready and route == "worldfoam_native4d"
            ),
            "same_representation_framewise_replay": (
                ready and route == "worldfoam_framewise_replay"
            ),
            "proxy_or_test_artifact": False,
            "measurement_is_simulated": False,
        }

    def open_session(self, context: Any, dataset: Any) -> "WorldFoamPublicQualitySession":
        if not self.production_ready:
            blockers = []
            if self._native_error is not None:
                blockers.append(f"native_abi:{self._native_error}")
            if self._runtime_receipt is None:
                blockers.append(
                    "runtime_receipt:worldfoam_native4d_public_quality_capabilities.json"
                )
            if self._tractability_error is not None:
                blockers.append(self._tractability_error)
            raise RuntimeError(
                "WorldFoam public executor remains source-only: " + "; ".join(blockers)
            )
        if not torch.backends.mps.is_available():
            raise RuntimeError("WorldFoam G4 requires an available real MPS backend")
        accessor = getattr(dataset, "worldfoam_training_inputs", None)
        if not callable(accessor):
            raise TypeError("public dataset lacks WorldFoam production inputs")
        inputs = accessor()
        if type(inputs) is not WorldFoamPublicQualityInputs:
            raise TypeError("WorldFoam executor requires the exact sealed input object")
        inputs.assert_current(dataset=dataset, context=context)
        return WorldFoamPublicQualitySession(
            context=context,
            dataset=dataset,
            route=self.route,
            inputs=inputs,
            native_ops=self._native_ops,
            native_library_sha256=str(self._native_library_sha256),
        )


class WorldFoamPublicQualitySession:
    """One exclusive, all-pixel, 300-step production training generation."""

    def __init__(
        self,
        *,
        context: Any,
        dataset: Any,
        route: str,
        inputs: WorldFoamPublicQualityInputs,
        native_ops: Any,
        native_library_sha256: str,
    ) -> None:
        self.context = context
        self.dataset = dataset
        self.route = route
        self.inputs = inputs
        self.native_ops = native_ops
        self.native_library_sha256 = native_library_sha256
        self.device = torch.device("mps")
        self._closed = False
        self._training_finalized = False
        # The real-native G4-v2 timing pilot deliberately stops after one
        # selected-ray optimizer step.  This flag opens only the bounded
        # heldout replay methods; it never satisfies the final-checkpoint or
        # public-evidence lifecycle.
        self._heldout_pilot_prepared = False
        self._active_work: Any | None = None
        self._expected_chunk_iterator: Iterator[Any] | None = None
        self._active_step_chunk_count = 0
        self._active_step_pixel_count = 0
        self._active_step_expected_pixel_count = 0
        self._optimizer_steps = 0
        self._target_pixels = 0
        self._sampled_images = 0
        self._pixel_chunks = 0
        self._rasterized_pixels = 0
        self._training_source_read_call_count = 0
        self._training_source_read_observation_count = 0
        self._training_full_frame_target_materialization_count = 0
        self._promotion_receipts: list[PaperKineticStreamingCombinedPromotionReceipt] = []
        self._peak_sampler = _PeakSampler()
        self._peak_process_rss_bytes = 0
        self._peak_mps_driver_allocated_bytes = 0
        self._heldout_provider: PaperKineticLazyProgramBundleProvider | None = None
        self._heldout_material_f32_mps: torch.Tensor | None = None
        self._heldout_background_f32_mps: torch.Tensor | None = None
        self._last_heldout_prediction_receipt_sha256 = ""
        self._heldout_prediction_receipt_chain_sha256 = _canonical_sha256(
            {
                "provenance": EXECUTOR_PROVENANCE,
                "phase": "heldout-prediction-chain-root",
                "input_generation_digest": inputs.input_generation_digest,
            }
        )
        self._heldout_prediction_receipt_count = 0
        self._heldout_prediction_observation_count = 0
        self._heldout_spatial_major_call_count = 0
        self._heldout_spatial_major_track_count = 0
        self._heldout_spatial_major_native_bundle_count = 0
        self._heldout_spatial_major_native_sample_count = 0
        self._heldout_spatial_major_prediction_target_read_count = 0
        self._heldout_spatial_major_target_staging_call_count = 0
        self._heldout_spatial_major_target_staging_observation_count = 0
        self._heldout_spatial_major_target_staging_peak_logical_bytes = 0
        self._heldout_spatial_major_prediction_receipt_chain_sha256 = (
            _canonical_sha256(
                {
                    "provenance": EXECUTOR_PROVENANCE,
                    "phase": "heldout-spatial-major-prediction-chain-root",
                    "input_generation_digest": inputs.input_generation_digest,
                }
            )
        )
        self._heldout_spatial_major_target_receipt_chain_sha256 = (
            _canonical_sha256(
                {
                    "provenance": EXECUTOR_PROVENANCE,
                    "phase": "heldout-spatial-major-target-chain-root",
                    "input_generation_digest": inputs.input_generation_digest,
                }
            )
        )
        self._training_loss_contract = dict(
            getattr(
                context.work_plan,
                "training_loss_contract",
                {
                    "identifier": "rgb_mse_mean_v1",
                    "formula": "mean((prediction-target)^2)",
                    "normalization": "mean_over_selected_rgb_scalars",
                },
            )
        )
        if self._training_loss_contract != {
            "identifier": "rgb_mse_mean_v1",
            "formula": "mean((prediction-target)^2)",
            "normalization": "mean_over_selected_rgb_scalars",
        }:
            raise ValueError("WorldFoam native ABI requires the sealed RGB-MSE loss")

        self.provider = prepare_paper_kinetic_lazy_program_bundle_provider(
            dataset_generation_digest=inputs.dataset_generation_digest,
            target_provider=inputs.target_provider,
            ray_provider=inputs.ray_provider,
            frame_times=inputs.frame_times,
            height=inputs.target_provider.height,
            width=inputs.target_provider.width,
            maximum_tracks_per_bundle=inputs.maximum_tracks_per_bundle,
            maximum_observations_per_bundle=inputs.maximum_observations_per_bundle,
            maximum_rows_per_native_block=inputs.maximum_rows_per_native_block,
            world_initializer=inputs.world_initializer,
            program_factory=inputs.program_factory,
        )
        material_initialization = inputs.world_initializer.initialize_p0_material(
            self.provider.world.sites
        )
        material_state = prepare_paper_kinetic_fixed_site_material_state(
            material_initialization,
            self.provider.world,
            parameterization=inputs.material_parameterization,
            optimizer_policy=inputs.material_sgd_policy,
            device="cpu",
            maximum_material_state_logical_tensor_bytes=(
                inputs.maximum_material_state_logical_tensor_bytes
            ),
        )
        self.artifact_store = PaperKineticCompiledCpuArtifactStore(
            inputs.artifact_store_policy
        )
        self.state = prepare_paper_kinetic_fixed_camera_combined_state(
            material_state,
            self.provider,
            self.artifact_store,
            maximum_combined_state_logical_tensor_bytes=(
                inputs.combined_sgd_policy.maximum_combined_state_logical_tensor_bytes
            ),
        )
        self.trainer_state = (
            prepare_paper_kinetic_lazy_native_trainer_state(
                self.provider,
                device=self.device,
                initial_step_index=0,
            )
            if route == "worldfoam_native4d"
            else None
        )
        self.background_generation_id = _canonical_sha256(
            {
                "provenance": EXECUTOR_PROVENANCE,
                "input_generation_digest": inputs.input_generation_digest,
                "background": tuple(
                    float(value) for value in inputs.background_rgb_f32_cpu
                ),
            }
        )
        # Shared row accounting measures dataset/executor/model setup
        # separately.  begin_step(0) starts this clock and sampler at the
        # optimizer boundary; final checkpoint serialization stops them.
        self._started_at: float | None = None

    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeError("WorldFoam public session is closed")

    def _start_training_measurement(self) -> None:
        if self._started_at is not None:
            raise RuntimeError("WorldFoam training measurement already started")
        self._started_at = time.perf_counter()
        self._peak_sampler.start()

    def begin_step(self, work: Any) -> None:
        self._assert_open()
        if self._training_finalized or self._active_work is not None:
            raise RuntimeError("WorldFoam step lifecycle is not idle")
        if work.step != self.state.geometry_update_count:
            raise ValueError("WorldFoam optimizer generation and row step diverged")
        if work.step == 0:
            self._start_training_measurement()
        elif self._started_at is None:
            raise RuntimeError("WorldFoam training measurement missed optimizer step 0")
        pixels = int(work.stage.image_size.pixels)
        if pixels != self.provider.height * self.provider.width:
            raise ValueError("WorldFoam G4 does not permit a resolution-stage change")

        expected_requests = tuple(
            self.context.work_plan.iter_step_training_chunks(work)
        )
        if not expected_requests:
            raise ArithmeticError("WorldFoam step has no target-pixel requests")
        self._active_work = work
        self._expected_chunk_iterator = iter(expected_requests)
        self._active_step_expected_pixel_count = sum(
            int(request.pixel_count) for request in expected_requests
        )
        self._active_step_chunk_count = 0
        self._active_step_pixel_count = 0

    def _accumulate_train_request(self, request: Any) -> None:
        self._assert_open()
        if self._active_work is None or self._expected_chunk_iterator is None:
            raise RuntimeError("WorldFoam training chunk arrived outside a step")
        try:
            expected = next(self._expected_chunk_iterator)
        except StopIteration as error:
            raise ArithmeticError("WorldFoam received extra training chunks") from error
        if request.as_dict() != expected.as_dict() or request.step != self._active_work.step:
            raise ValueError("WorldFoam row/provider chunk coverage changed")
        # The row validates target/ray values and the sealed input object binds
        # their mapped providers.  Retaining the payload here would create a
        # hidden step cache, so only scalar coverage is consumed.
        self._active_step_chunk_count += 1
        self._active_step_pixel_count += int(request.pixel_count)

    def accumulate_train_request(self, request: Any) -> None:
        """Consume v2 schedule metadata; native execution owns the sole target read."""

        if getattr(request, "pixel_ids", None) is None:
            raise ValueError("metadata-only WorldFoam ingestion is selected-ray-only")
        self._accumulate_train_request(request)

    def accumulate_train_chunk(self, request: Any, payload: Any) -> None:
        # Frozen G4-v1 continues to validate/materialize its external payload.
        del payload
        self._accumulate_train_request(request)

    def _record_training_source_reads(self, accounting: Mapping[str, Any]) -> None:
        self._training_source_read_call_count += int(
            accounting["selected_pixel_read_call_count"]
        )
        self._training_source_read_observation_count += int(
            accounting["direct_selected_pixel_observation_count"]
        ) + int(accounting["bounded_region_selected_pixel_observation_count"])
        self._training_full_frame_target_materialization_count += int(
            accounting["full_frame_target_materialization_count"]
        )

    def finish_step(self, work: Any) -> None:
        self._assert_open()
        if work is not self._active_work or self._expected_chunk_iterator is None:
            raise ValueError("WorldFoam finish_step received another work object")
        try:
            extra = next(self._expected_chunk_iterator)
        except StopIteration:
            extra = None
        if extra is not None:
            raise ArithmeticError("WorldFoam step did not consume all public chunks")
        expected_pixels = self._active_step_expected_pixel_count
        if self._active_step_pixel_count != expected_pixels:
            raise ArithmeticError("WorldFoam step did not consume every target pixel")
        if self.route == "worldfoam_native4d":
            self._run_shared_step(work)
        else:
            self._run_framewise_step(work)
        self._optimizer_steps += 1
        self._target_pixels += self._active_step_pixel_count
        self._sampled_images += len(work.batch.samples)
        self._pixel_chunks += self._active_step_chunk_count
        self._active_work = None
        self._expected_chunk_iterator = None
        self._active_step_chunk_count = 0
        self._active_step_pixel_count = 0
        self._active_step_expected_pixel_count = 0

    def _snapshot(self) -> Any:
        return snapshot_paper_kinetic_fixed_site_material_to_device(
            self.state.material_state,
            background_rgb_f32_cpu=self.inputs.background_rgb_f32_cpu,
            background_generation_id=self.background_generation_id,
            device=self.device,
            device_completion_fence=synchronize_mps_device_completion_fence,
            device_completion_fence_provenance=(
                MPS_DEVICE_COMPLETION_FENCE_PROVENANCE
            ),
        )

    def _promote(
        self,
        *,
        step_generation_id: str,
        learning_rate_multiplier: float,
        observation_count: int,
        loss_f32_cpu: torch.Tensor,
        grad_site_rgba_f32_cpu: torch.Tensor,
        grad_positions0_f64_cpu: torch.Tensor,
        grad_velocities_f64_cpu: torch.Tensor,
        grad_weight_coefficients_f64_cpu: torch.Tensor,
    ) -> tuple[Any, Any, Any, Any]:
        old = (self.state, self.provider, self.artifact_store)
        authorization = seal_paper_kinetic_streaming_combined_gradient(
            *old,
            step_generation_id=step_generation_id,
            observation_count=observation_count,
            loss_f32_cpu=loss_f32_cpu,
            grad_site_rgba_f32_cpu=grad_site_rgba_f32_cpu,
            grad_positions0_f64_cpu=grad_positions0_f64_cpu,
            grad_velocities_f64_cpu=grad_velocities_f64_cpu,
            grad_weight_coefficients_f64_cpu=grad_weight_coefficients_f64_cpu,
        )
        ready = apply_paper_kinetic_streaming_combined_sgd(
            *old,
            authorization,
            policy=self.inputs.combined_sgd_policy,
            fresh_store_policy=self.inputs.artifact_store_policy,
            learning_rate_multiplier=learning_rate_multiplier,
        )
        ready.assert_current()
        self.state = ready.state
        self.provider = ready.provider
        self.artifact_store = ready.artifact_store
        self._promotion_receipts.append(ready.receipt)
        return authorization, ready, *old

    def _run_shared_step(self, work: Any) -> None:
        if self.trainer_state is None:
            raise RuntimeError("shared WorldFoam route lost its trainer ledger")
        selected_pixel_source = getattr(
            self.context.work_plan,
            "selected_pixel_ids_for_step",
            None,
        )
        observations = (
            _ReplayableSelectedStepObservations(
                work.batch,
                tuple(int(value) for value in selected_pixel_source(work.step)),
                int(work.stage.image_size.pixels),
            )
            if callable(selected_pixel_source)
            else _ReplayableStepObservations(
                work.batch,
                int(work.stage.image_size.pixels),
            )
        )
        expected = observations.expected_observation_count
        manifest = paper_kinetic_observation_manifest_digest(observations)
        snapshot = self._snapshot()
        material_bar = torch.empty_like(snapshot.site_rgba_f32_device)
        position_bar = torch.empty_like(self.state.positions0_f64, device="cpu")
        velocity_bar = torch.empty_like(self.state.velocities_f64, device="cpu")
        weight_bar = torch.empty_like(
            self.state.weight_coefficients_f64,
            device="cpu",
        )
        captures: list[PaperKineticLazyNativeFullGeometryStepResult] = []
        result = run_paper_kinetic_lazy_native_full_geometry_step(
            self.trainer_state,
            self.provider,
            observations,
            step_index=self.state.geometry_update_count,
            expected_observation_count=expected,
            expected_observation_manifest_digest=manifest,
            loss_normalization_id=_canonical_sha256(
                {
                    "route": self.route,
                    "step": work.step,
                    "loss": "global-rgb-mean",
                    "elements": expected * 3,
                    **(
                        {"training_loss_contract": self._training_loss_contract}
                        if callable(selected_pixel_source)
                        else {}
                    ),
                }
            ),
            material_generation_id=self.state.material_state.material_generation_id,
            geometry_generation_id=self.state.geometry_generation_id,
            background_generation_id=self.background_generation_id,
            global_site_rgba_f32=snapshot.site_rgba_f32_device,
            global_grad_site_rgba_f32=material_bar,
            grad_positions0_f64_cpu=position_bar,
            grad_velocities_f64_cpu=velocity_bar,
            grad_weight_coefficients_f64_cpu=weight_bar,
            background_rgb_f32=snapshot.background_rgb_f32_device,
            native_ops=self.native_ops,
            maximum_samples_per_launch=self.inputs.maximum_samples_per_launch,
            memory_policy=self.inputs.lazy_memory_policy,
            full_geometry_memory_policy=self.inputs.full_geometry_memory_policy,
            reverse_mode=FUSED_UNION_V2,
            optimizer_update=captures.append,
            cone_tolerance=self.inputs.cone_tolerance,
        )
        if captures != [result]:
            raise ArithmeticError("shared WorldFoam issued the wrong optimizer callback")
        result.assert_current()
        if int(result.accounting["streamed_sample_count"]) != expected:
            raise ArithmeticError("shared WorldFoam changed sampled-pixel coverage")
        self._record_training_source_reads(result.accounting)
        bridge = seal_paper_kinetic_lazy_full_geometry_device_gradient_receipt(
            self.state,
            self.provider,
            self.artifact_store,
            snapshot,
            result,
        )
        tensors = bridge._live_tensors()
        step_generation_id = _canonical_sha256(
            {
                "provenance": EXECUTOR_PROVENANCE,
                "route": self.route,
                "step": work.step,
                "batch": tuple(
                    (int(sample.view_index), int(sample.frame_index))
                    for sample in work.batch.samples
                ),
                "manifest": manifest,
                "native_result": result.generation_digest,
                "learning_rate_multiplier": float(work.stage.lr_multiplier),
            }
        )
        authorization, ready, old_state, _old_provider, _old_store = self._promote(
            step_generation_id=step_generation_id,
            learning_rate_multiplier=float(work.stage.lr_multiplier),
            observation_count=expected,
            loss_f32_cpu=tensors[1],
            grad_site_rgba_f32_cpu=tensors[0],
            grad_positions0_f64_cpu=tensors[2],
            grad_velocities_f64_cpu=tensors[3],
            grad_weight_coefficients_f64_cpu=tensors[4],
        )
        released = bridge._revoke_after_validated_retirement(
            _TRANSACTION_CONSUMPTION_AUTHORITY
        )
        if released != authorization.logical_tensor_bytes:
            raise ArithmeticError("shared WorldFoam bridge release bytes changed")
        bridge._commit_promoted_consumption(
            _TRANSACTION_CONSUMPTION_AUTHORITY,
            promoted_state_generation_digest=ready.state.generation_digest,
            update_receipt_generation_digest=ready.receipt.generation_digest,
        )
        old_state.assert_retired()
        self.trainer_state = prepare_paper_kinetic_lazy_native_trainer_state(
            self.provider,
            device=self.device,
            initial_step_index=self.state.geometry_update_count,
        )
        self._rasterized_pixels += expected
        del authorization, bridge, captures, result, snapshot, ready, observations

    def _run_framewise_step(self, work: Any) -> None:
        image_pixels = int(work.stage.image_size.pixels)
        selected_pixel_source = getattr(
            self.context.work_plan,
            "selected_pixel_ids_for_step",
            None,
        )
        selected_pixels = (
            tuple(int(value) for value in selected_pixel_source(work.step))
            if callable(selected_pixel_source)
            else range(image_pixels)
        )
        total_observations = len(work.batch.samples) * len(selected_pixels)
        snapshot = self._snapshot()
        global_material = torch.zeros(
            (self.state.site_count, 4),
            dtype=torch.float32,
            device="cpu",
        )
        global_loss = torch.zeros((1,), dtype=torch.float32, device="cpu")
        global_position = torch.zeros_like(self.state.positions0_f64, device="cpu")
        global_velocity = torch.zeros_like(self.state.velocities_f64, device="cpu")
        global_weight = torch.zeros_like(
            self.state.weight_coefficients_f64,
            device="cpu",
        )
        by_view: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for batch_position, sample in enumerate(work.batch.samples):
            by_view[int(sample.view_index)].append(
                (batch_position, int(sample.frame_index))
            )
        result_chain = _canonical_sha256(
            {
                "provenance": EXECUTOR_PROVENANCE,
                "route": self.route,
                "step": work.step,
                "phase": "framewise-result-chain-root",
            }
        )
        staged_policy = _stage_policy(self.inputs.full_geometry_memory_policy)
        replay_count = 0
        try:
            for view_index in sorted(by_view):
                for pixel_start in range(
                    0,
                    len(selected_pixels),
                    self.inputs.maximum_tracks_per_bundle,
                ):
                    pixel_stop = min(
                        pixel_start + self.inputs.maximum_tracks_per_bundle,
                        len(selected_pixels),
                    )
                    tracks = tuple(selected_pixels[pixel_start:pixel_stop])
                    chunk_store = PaperKineticCompiledCpuArtifactStore(
                        self.inputs.artifact_store_policy
                    )
                    wrapper = prepare_paper_kinetic_compiled_framewise_program_provider(
                        self.provider,
                        chunk_store,
                        view_index=view_index,
                        selected_track_ids=tracks,
                        maximum_artifact_accounted_bytes_per_entry=(
                            self.inputs.maximum_artifact_accounted_bytes_per_entry
                        ),
                    )
                    local_trainer = prepare_paper_kinetic_lazy_native_trainer_state(
                        wrapper,
                        device=self.device,
                        initial_step_index=0,
                    )
                    try:
                        for local_step, (batch_position, frame_index) in enumerate(
                            by_view[view_index]
                        ):
                            observations = (
                                _ReplayableSelectedFrameChunkObservations(
                                    view_index=view_index,
                                    frame_index=frame_index,
                                    batch_position=batch_position,
                                    pixel_indices=tracks,
                                    image_pixel_count=image_pixels,
                                )
                                if callable(selected_pixel_source)
                                else _ReplayableFrameChunkObservations(
                                    view_index=view_index,
                                    frame_index=frame_index,
                                    batch_position=batch_position,
                                    pixel_start=pixel_start,
                                    pixel_stop=pixel_stop,
                                    image_pixel_count=image_pixels,
                                )
                            )
                            local_count = observations.expected_observation_count
                            manifest = paper_kinetic_observation_manifest_digest(
                                observations
                            )
                            material_bar = torch.empty_like(
                                snapshot.site_rgba_f32_device
                            )
                            position_bar = torch.empty_like(
                                self.state.positions0_f64,
                                device="cpu",
                            )
                            velocity_bar = torch.empty_like(
                                self.state.velocities_f64,
                                device="cpu",
                            )
                            weight_bar = torch.empty_like(
                                self.state.weight_coefficients_f64,
                                device="cpu",
                            )
                            captures: list[
                                PaperKineticLazyNativeFullGeometryStepResult
                            ] = []
                            result = run_paper_kinetic_lazy_native_full_geometry_step(
                                local_trainer,
                                wrapper,
                                observations,
                                step_index=local_step,
                                expected_observation_count=local_count,
                                expected_observation_manifest_digest=manifest,
                                loss_normalization_id=_canonical_sha256(
                                    {
                                        "route": self.route,
                                        "step": work.step,
                                        "view": view_index,
                                        "frame": frame_index,
                                        "pixel_start": pixel_start,
                                        "elements": local_count * 3,
                                        **(
                                            {
                                                "selected_track_ids": tracks,
                                                "training_loss_contract": (
                                                    self._training_loss_contract
                                                ),
                                            }
                                            if callable(selected_pixel_source)
                                            else {}
                                        ),
                                    }
                                ),
                                material_generation_id=(
                                    self.state.material_state.material_generation_id
                                ),
                                geometry_generation_id=self.state.geometry_generation_id,
                                background_generation_id=self.background_generation_id,
                                global_site_rgba_f32=snapshot.site_rgba_f32_device,
                                global_grad_site_rgba_f32=material_bar,
                                grad_positions0_f64_cpu=position_bar,
                                grad_velocities_f64_cpu=velocity_bar,
                                grad_weight_coefficients_f64_cpu=weight_bar,
                                background_rgb_f32=snapshot.background_rgb_f32_device,
                                native_ops=self.native_ops,
                                maximum_samples_per_launch=(
                                    self.inputs.maximum_samples_per_launch
                                ),
                                memory_policy=self.inputs.lazy_memory_policy,
                                full_geometry_memory_policy=staged_policy,
                                reverse_mode=STAGED_SPARSE,
                                optimizer_update=captures.append,
                                cone_tolerance=self.inputs.cone_tolerance,
                            )
                            if captures != [result]:
                                raise ArithmeticError(
                                    "framewise replay issued the wrong callback"
                                )
                            result.assert_current()
                            if int(result.accounting["streamed_sample_count"]) != local_count:
                                raise ArithmeticError(
                                    "framewise replay changed sampled-pixel coverage"
                                )
                            self._record_training_source_reads(result.accounting)
                            material_cpu = result.grad_global_site_rgba_f32.detach().to(
                                device="cpu",
                                dtype=torch.float32,
                            ).contiguous()
                            loss_cpu = result.loss_f32.detach().to(
                                device="cpu",
                                dtype=torch.float32,
                            ).contiguous()
                            synchronize_mps_device_completion_fence()
                            scale = float(local_count) / float(total_observations)
                            global_material.add_(material_cpu, alpha=scale)
                            global_loss.add_(loss_cpu, alpha=scale)
                            global_position.add_(
                                result.grad_positions0_f64_cpu,
                                alpha=scale,
                            )
                            global_velocity.add_(
                                result.grad_velocities_f64_cpu,
                                alpha=scale,
                            )
                            global_weight.add_(
                                result.grad_weight_coefficients_f64_cpu,
                                alpha=scale,
                            )
                            result_chain = _canonical_sha256(
                                {
                                    "previous": result_chain,
                                    "result": result.generation_digest,
                                    "manifest": manifest,
                                    "weight": scale,
                                }
                            )
                            # The explicit synchronize above is the release
                            # boundary used by the existing framewise control.
                            # Revoke result-owned tensors before opening the
                            # next frame so scratch cannot grow with F.
                            result._seal = None
                            result.loss_f32 = None  # type: ignore[assignment]
                            result.grad_global_site_rgba_f32 = None  # type: ignore[assignment]
                            result.grad_positions0_f64_cpu = None  # type: ignore[assignment]
                            result.grad_velocities_f64_cpu = None  # type: ignore[assignment]
                            result.grad_weight_coefficients_f64_cpu = None  # type: ignore[assignment]
                            replay_count += 1
                            self._rasterized_pixels += local_count
                            del (
                                result,
                                captures,
                                observations,
                                material_bar,
                                position_bar,
                                velocity_bar,
                                weight_bar,
                                material_cpu,
                                loss_cpu,
                            )
                    finally:
                        chunk_store.close()
                        object.__setattr__(wrapper, "_seal", None)
                        del local_trainer, wrapper, chunk_store
        except BaseException:
            self.state.poisoned = True
            self.state.material_state.poisoned = True
            raise
        expected_replays = sum(
            math.ceil(len(selected_pixels) / self.inputs.maximum_tracks_per_bundle)
            * len(samples)
            for samples in by_view.values()
        )
        if replay_count != expected_replays:
            raise ArithmeticError("framewise replay count changed")
        synchronize_mps_device_completion_fence()
        snapshot._release_after_consumption()
        step_generation_id = _canonical_sha256(
            {
                "provenance": EXECUTOR_PROVENANCE,
                "route": self.route,
                "step": work.step,
                "batch": tuple(
                    (int(sample.view_index), int(sample.frame_index))
                    for sample in work.batch.samples
                ),
                "result_chain": result_chain,
                "replay_count": replay_count,
                "learning_rate_multiplier": float(work.stage.lr_multiplier),
            }
        )
        authorization, ready, old_state, _old_provider, _old_store = self._promote(
            step_generation_id=step_generation_id,
            learning_rate_multiplier=float(work.stage.lr_multiplier),
            observation_count=total_observations,
            loss_f32_cpu=global_loss,
            grad_site_rgba_f32_cpu=global_material,
            grad_positions0_f64_cpu=global_position,
            grad_velocities_f64_cpu=global_velocity,
            grad_weight_coefficients_f64_cpu=global_weight,
        )
        old_state.assert_retired()
        del authorization, ready, snapshot

    def finalize_training(self, checkpoint_path: Path) -> Mapping[str, Any]:
        self._assert_open()
        if self._training_finalized or self._active_work is not None:
            raise RuntimeError("WorldFoam training cannot finalize in the current phase")
        if self._optimizer_steps != self.context.protocol.steps:
            raise ArithmeticError("WorldFoam did not finish the frozen optimizer schedule")
        if self._started_at is None:
            raise RuntimeError("WorldFoam training measurement never started")
        synchronize_mps_device_completion_fence()
        representation_sha256 = self._representation_digest()
        payload = self._checkpoint_payload(representation_sha256)
        destination = Path(checkpoint_path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{threading.get_ident()}.partial")
        temporary.unlink(missing_ok=True)
        try:
            torch.save(payload, temporary)
            temporary.replace(destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        del payload
        peak_rss, peak_mps = self._peak_sampler.stop()
        self._peak_process_rss_bytes = peak_rss
        self._peak_mps_driver_allocated_bytes = peak_mps
        elapsed_s = float(time.perf_counter() - self._started_at)
        self._training_finalized = True
        self._prepare_heldout_generation()
        parameter_tensors = (
            self.state.material_state.raw_color_f32,
            self.state.material_state.raw_density_f32,
            self.state.positions0_f64,
            self.state.velocities_f64,
            self.state.weight_coefficients_f64,
        )
        return {
            "optimizer_steps": self._optimizer_steps,
            "target_pixels_consumed": self._target_pixels,
            "sampled_image_count": self._sampled_images,
            "pixel_chunk_count": self._pixel_chunks,
            "rasterized_pixels": self._rasterized_pixels,
            "parameter_count": sum(int(tensor.numel()) for tensor in parameter_tensors),
            "parameter_bytes": _tensor_bytes(*parameter_tensors),
            "process_lifetime_peak_rss_through_checkpoint_bytes": (
                self._peak_process_rss_bytes
            ),
            "sampled_peak_mps_driver_during_training_and_checkpoint_bytes": (
                self._peak_mps_driver_allocated_bytes
            ),
            "training_and_checkpoint_elapsed_s": elapsed_s,
            "representation_sha256": representation_sha256,
            "checkpoint_step": self._optimizer_steps,
        }

    def training_source_read_receipt(self) -> Mapping[str, Any]:
        if not self._training_finalized:
            raise RuntimeError("WorldFoam source-read receipt requires final training")
        if (
            self._training_source_read_observation_count != self._target_pixels
            or self._training_source_read_call_count < self._pixel_chunks
            or self._training_full_frame_target_materialization_count != 0
        ):
            raise ArithmeticError("WorldFoam native target-read accounting changed")
        payload = {
            "schema_version": 1,
            "kind": "worldfoam-native-internal-selected-target-reads-v1",
            "ownership": "executor_internal_single_read",
            "selected_pixel_read_call_count": self._training_source_read_call_count,
            "selected_pixel_read_observation_count": (
                self._training_source_read_observation_count
            ),
            "full_frame_target_materialization_count": 0,
            "external_row_worker_target_read_call_count": 0,
            "request_schedule_sha256": self.context.work_plan.sample_schedule_sha256,
        }
        return {**payload, "generation_digest": _canonical_sha256(payload)}

    def _representation_digest(self) -> str:
        return _canonical_sha256(
            {
                "schema": CHECKPOINT_SCHEMA,
                "route_representation": self.inputs.same_representation_group,
                "step": self.state.geometry_update_count,
                "material": (
                    _tensor_content_digest(self.state.material_state.raw_color_f32),
                    _tensor_content_digest(self.state.material_state.raw_density_f32),
                ),
                "geometry": (
                    _tensor_content_digest(self.state.positions0_f64),
                    _tensor_content_digest(self.state.velocities_f64),
                    _tensor_content_digest(self.state.weight_coefficients_f64),
                ),
            }
        )

    def _checkpoint_payload(self, representation_sha256: str) -> dict[str, Any]:
        return {
            "schema": CHECKPOINT_SCHEMA,
            "executor_provenance": EXECUTOR_PROVENANCE,
            "route": self.route,
            "same_representation_group": self.inputs.same_representation_group,
            "sample_id": self.inputs.sample_id,
            "input_generation_digest": self.inputs.input_generation_digest,
            "protocol_sha256": self.context.scene_receipt["protocol_sha256"],
            "sample_schedule_sha256": (
                self.context.work_plan.sample_schedule_sha256
            ),
            "training_loss_contract": self._training_loss_contract,
            **(
                {
                    "v2_config_sha256": (
                        self.context.work_plan.workload_receipt.v2_config_sha256
                    ),
                    "workload_receipt_generation_digest": (
                        self.context.work_plan.workload_receipt.generation_digest
                    ),
                    "route_schedule_sha256": (
                        self.context.work_plan.workload_receipt.route_schedule_sha256
                    ),
                }
                if getattr(self.context.work_plan, "workload_receipt", None)
                is not None
                else {}
            ),
            "native_library_sha256": self.native_library_sha256,
            "step": self.state.geometry_update_count,
            "representation_sha256": representation_sha256,
            "raw_color_f32_cpu": self.state.material_state.raw_color_f32.clone(),
            "raw_density_f32_cpu": self.state.material_state.raw_density_f32.clone(),
            "positions0_f64_cpu": self.state.positions0_f64.clone(),
            "velocities_f64_cpu": self.state.velocities_f64.clone(),
            "weight_coefficients_f64_cpu": self.state.weight_coefficients_f64.clone(),
            "material_parameterization": self.inputs.material_parameterization.payload(),
            "material_sgd_policy": self.inputs.material_sgd_policy.payload(),
            "combined_sgd_policy_generation_digest": (
                self.inputs.combined_sgd_policy.generation_digest
            ),
            "promotion_receipt_generation_digests": tuple(
                receipt.generation_digest for receipt in self._promotion_receipts
            ),
            "promotion_learning_rate_multipliers": tuple(
                receipt.learning_rate_multiplier
                for receipt in self._promotion_receipts
            ),
            "persistent_frame_tensor_bytes": 0,
            "persistent_sample_tensor_bytes": 0,
            "persistent_target_tensor_bytes": 0,
            "persistent_prediction_tensor_bytes": 0,
            "compiled_program_tensor_bytes": 0,
            "optimizer_history_tensor_bytes": 0,
        }

    def _prepare_heldout_generation(self) -> None:
        if self._heldout_provider is not None:
            return
        import paper_kinetic_fixed_camera_combined_state as combined
        from kinetic_power_word_compiler import AffineKineticPowerSites

        initializer = combined._OwnedCandidateWorldInitializer(
            sites=AffineKineticPowerSites(
                positions0=self.state.positions0_f64,
                velocities=self.state.velocities_f64,
                weight_coefficients=self.state.weight_coefficients_f64,
            ),
            generation_digest=_canonical_sha256(
                {
                    "provenance": EXECUTOR_PROVENANCE,
                    "phase": "heldout-world-binding",
                    "state": self.state.generation_digest,
                    "heldout_dataset": self.inputs.heldout_dataset_generation_digest,
                }
            ),
        )
        provider = prepare_paper_kinetic_lazy_program_bundle_provider(
            dataset_generation_digest=self.inputs.heldout_dataset_generation_digest,
            target_provider=self.inputs.heldout_target_provider,
            ray_provider=self.inputs.heldout_ray_provider,
            frame_times=self.inputs.frame_times,
            height=self.inputs.heldout_target_provider.height,
            width=self.inputs.heldout_target_provider.width,
            maximum_tracks_per_bundle=self.inputs.maximum_tracks_per_bundle,
            maximum_observations_per_bundle=self.inputs.maximum_observations_per_bundle,
            maximum_rows_per_native_block=self.inputs.maximum_rows_per_native_block,
            world_initializer=initializer,
            program_factory=self.inputs.program_factory,
        )
        if (
            not initializer.consumed
            or provider.world.sites_content_digest != self.state.sites_content_digest
            or provider.world.site_count != self.state.site_count
        ):
            raise ValueError("heldout provider changed the trained world representation")
        material = self.state.material_state.site_rgba_f32.detach().to(
            device=self.device,
            dtype=torch.float32,
            copy=True,
        ).contiguous()
        background = self.inputs.background_rgb_f32_cpu.detach().to(
            device=self.device,
            dtype=torch.float32,
            copy=True,
        ).contiguous()
        synchronize_mps_device_completion_fence()
        self._heldout_provider = provider
        self._heldout_material_f32_mps = material
        self._heldout_background_f32_mps = background

    def prepare_heldout_pilot_from_current_state(self) -> Mapping[str, Any]:
        """Bind the post-step-1 world for a non-evidence heldout timing pilot."""

        self._assert_open()
        if (
            self._active_work is not None
            or self._expected_chunk_iterator is not None
            or self._training_finalized
            or self._optimizer_steps != 1
            or self.state.geometry_update_count != 1
            or self._started_at is None
            or self._heldout_provider is not None
            or self._heldout_pilot_prepared
        ):
            raise RuntimeError(
                "WorldFoam heldout pilot requires one complete optimizer step "
                "and an otherwise idle, non-finalized session"
            )
        synchronize_mps_device_completion_fence()
        self._prepare_heldout_generation()
        synchronize_mps_device_completion_fence()
        provider = self._heldout_provider
        if provider is None:
            raise RuntimeError("WorldFoam heldout pilot provider was not prepared")
        self._heldout_pilot_prepared = True
        payload = {
            "schema_version": 1,
            "kind": "worldfoam-g4-v2-heldout-pilot-binding-v1",
            "route": self.route,
            "optimizer_step": 1,
            "training_finalized": False,
            "pilot_only": True,
            "input_generation_digest": self.inputs.input_generation_digest,
            "provider_generation_digest": provider.generation_digest,
            "material_generation_digest": (
                self.state.material_state.material_generation_id
            ),
            "site_count": self.state.site_count,
            "heldout_view_count": provider.view_count,
            "frame_count": provider.frame_count,
            "image_height": provider.height,
            "image_width": provider.width,
            "native_library_sha256": self.native_library_sha256,
        }
        return {**payload, "generation_digest": _canonical_sha256(payload)}

    def render_heldout_chunk(self, request: Any, rays_f32_cpu: Any) -> torch.Tensor:
        self._assert_open()
        if not (self._training_finalized or self._heldout_pilot_prepared):
            raise RuntimeError(
                "WorldFoam heldout rendering requires a final checkpoint or "
                "the explicit step-1 pilot binding"
            )
        if (
            request.split != "heldout"
            or request.step is not None
            or request.sample_slot is not None
        ):
            raise ValueError("WorldFoam evaluator received a training request")
        provider = self._heldout_provider
        material = self._heldout_material_f32_mps
        background = self._heldout_background_f32_mps
        if (
            provider is None
            or material is None
            or background is None
        ):
            raise RuntimeError("WorldFoam heldout generation was not prepared")
        integer_fields = {
            "camera_index": request.camera_index,
            "frame_index": request.frame_index,
            "pixel_start": request.pixel_start,
            "pixel_count": request.pixel_count,
            "image_height": request.image_height,
            "image_width": request.image_width,
        }
        if any(type(value) is not int for value in integer_fields.values()):
            raise TypeError("WorldFoam heldout request integers changed type")
        maximum_pixels = int(
            getattr(
                self.context.work_plan,
                "heldout_maximum_pixels_per_chunk",
                self.context.work_plan.maximum_pixels_per_chunk,
            )
        )
        image_pixels = provider.height * provider.width
        if (
            request.image_height != provider.height
            or request.image_width != provider.width
            or request.camera_index < 0
            or request.camera_index >= provider.view_count
            or request.frame_index < 0
            or request.frame_index >= provider.frame_count
            or request.pixel_start < 0
            or request.pixel_count < 1
            or request.pixel_count > maximum_pixels
            or request.pixel_start + request.pixel_count > image_pixels
        ):
            raise ValueError(
                "WorldFoam heldout request exceeds the sealed evaluation bounds"
            )
        pixel_stop = request.pixel_start + request.pixel_count
        if (
            not isinstance(rays_f32_cpu, torch.Tensor)
            or rays_f32_cpu.device.type != "cpu"
            or rays_f32_cpu.dtype != torch.float32
            or tuple(rays_f32_cpu.shape) != (int(request.pixel_count), 6)
            or not rays_f32_cpu.is_contiguous()
        ):
            raise ValueError("WorldFoam heldout rays violate the bounded row contract")
        selected_pixels = torch.arange(
            int(request.pixel_start),
            int(pixel_stop),
            dtype=torch.int64,
            device="cpu",
        )
        expected_origins, expected_directions = build_camera_rays_at_pixels(
            provider.ray_provider.cameras[
                int(request.camera_index)
            ][int(request.frame_index)],
            selected_pixels,
            height=provider.height,
            width=provider.width,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        expected_rays = torch.cat(
            (expected_origins, expected_directions),
            dim=-1,
        ).contiguous()
        if not torch.equal(rays_f32_cpu, expected_rays):
            raise ValueError("WorldFoam heldout rays differ from the sealed provider")
        del selected_pixels, expected_origins, expected_directions, expected_rays
        observations = tuple(
            PaperKineticObservation(
                observation_id=position,
                view_index=int(request.camera_index),
                frame_index=int(request.frame_index),
                pixel_index=pixel,
            )
            for position, pixel in enumerate(
                range(int(request.pixel_start), int(pixel_stop))
            )
        )
        try:
            prediction_module = importlib.import_module(
                "worldfoam_native_heldout_prediction"
            )
            render = getattr(
                prediction_module,
                "predict_worldfoam_native_heldout_observations",
            )
        except (ImportError, AttributeError) as error:
            raise RuntimeError(
                "bounded native prediction wrapper is not source-complete"
            ) from error
        result = render(
            provider,
            observations,
            global_site_rgba_f32=material,
            material_generation_digest=(
                self.state.material_state.material_generation_id
            ),
            background_rgb_f32=background,
            native_ops=self.native_ops,
            maximum_samples_per_launch=self.inputs.maximum_samples_per_launch,
            maximum_source_decode_tensor_bytes=(
                self.inputs.lazy_memory_policy.max_decoded_frame_scratch_tensor_bytes
            ),
            maximum_lane_resident_logical_tensor_bytes=(
                self.inputs.lazy_memory_policy.max_lane_resident_logical_tensor_bytes
            ),
            maximum_returned_cpu_prediction_tensor_bytes=(
                maximum_pixels * 3 * 4
            ),
            cone_tolerance=self.inputs.cone_tolerance,
        )
        result.assert_current()
        prediction = getattr(result, "rgb_f32_cpu", None)
        receipt = getattr(result, "receipt", None)
        if (
            not isinstance(prediction, torch.Tensor)
            or prediction.device.type != "cpu"
            or prediction.dtype != torch.float32
            or tuple(prediction.shape) != (int(request.pixel_count), 3)
            or not prediction.is_contiguous()
            or receipt is None
            or int(getattr(receipt, "observation_count", -1))
            != int(request.pixel_count)
            or int(getattr(receipt, "persistent_device_prediction_tensor_bytes", -1))
            != 0
        ):
            raise ValueError("native heldout prediction receipt changed")
        self._last_heldout_prediction_receipt_sha256 = receipt.generation_digest
        self._heldout_prediction_receipt_chain_sha256 = _canonical_sha256(
            {
                "previous": self._heldout_prediction_receipt_chain_sha256,
                "receipt": receipt.generation_digest,
                "camera_index": request.camera_index,
                "frame_index": request.frame_index,
                "pixel_start": request.pixel_start,
                "pixel_count": request.pixel_count,
            }
        )
        self._heldout_prediction_receipt_count += 1
        self._heldout_prediction_observation_count += request.pixel_count
        return prediction

    def maximum_heldout_tracks_per_cross_time_block(self) -> int:
        """Largest predictor call; the provider partitions bounded bundles."""

        provider = self._heldout_provider
        if provider is None or not (
            self._training_finalized or self._heldout_pilot_prepared
        ):
            raise RuntimeError("WorldFoam heldout generation is not prepared")
        if provider.frame_count > int(self.inputs.maximum_observations_per_bundle):
            raise MemoryError("heldout bundle cannot admit one complete temporal track")
        # A predictor call may contain several native bundles.  The provider's
        # canonical partitioner keeps every temporal track intact and caps each
        # internal bundle by both maximum_tracks_per_bundle and
        # maximum_observations_per_bundle.  Keeping the call cap at 128 avoids
        # ~15k Python/native wrapper calls while the internal 4096-observation
        # bound remains unchanged (13 complete 300-frame tracks per bundle).
        return min(
            int(self.inputs.maximum_tracks_per_bundle),
            int(provider.height * provider.width),
        )

    def _heldout_native_tracks_per_bundle_limit(self) -> int:
        provider = self._heldout_provider
        if provider is None:
            raise RuntimeError("WorldFoam heldout provider is unavailable")
        result = min(
            int(self.inputs.maximum_tracks_per_bundle),
            int(self.inputs.maximum_observations_per_bundle)
            // int(provider.frame_count),
        )
        if result < 1:
            raise MemoryError("heldout native bundle cannot admit one temporal track")
        return result

    def render_heldout_track_block_across_frames(
        self,
        *,
        camera_index: int,
        pixel_ids: tuple[int, ...],
    ) -> torch.Tensor:
        """Compile each heldout pixel track once, then evaluate every frame."""

        self._assert_open()
        provider = self._heldout_provider
        material = self._heldout_material_f32_mps
        background = self._heldout_background_f32_mps
        if provider is None or material is None or background is None:
            raise RuntimeError("WorldFoam heldout generation was not prepared")
        selected = tuple(int(value) for value in pixel_ids)
        image_pixels = int(provider.height * provider.width)
        if (
            type(camera_index) is not int
            or not 0 <= camera_index < provider.view_count
            or not selected
            or len(selected) > self.maximum_heldout_tracks_per_cross_time_block()
            or tuple(sorted(selected)) != selected
            or len(set(selected)) != len(selected)
            or selected[0] < 0
            or selected[-1] >= image_pixels
        ):
            raise ValueError("WorldFoam cross-time heldout track block is invalid")
        observations = tuple(
            PaperKineticObservation(
                observation_id=frame_index * image_pixels + pixel,
                view_index=camera_index,
                frame_index=frame_index,
                pixel_index=pixel,
            )
            for frame_index in range(provider.frame_count)
            for pixel in selected
        )
        from worldfoam_native_heldout_prediction import (
            predict_worldfoam_native_heldout_observations,
        )

        result = predict_worldfoam_native_heldout_observations(
            provider,
            observations,
            global_site_rgba_f32=material,
            material_generation_digest=self.state.material_state.material_generation_id,
            background_rgb_f32=background,
            native_ops=self.native_ops,
            maximum_samples_per_launch=self.inputs.maximum_samples_per_launch,
            maximum_source_decode_tensor_bytes=(
                self.inputs.lazy_memory_policy.max_decoded_frame_scratch_tensor_bytes
            ),
            maximum_lane_resident_logical_tensor_bytes=(
                self.inputs.lazy_memory_policy.max_lane_resident_logical_tensor_bytes
            ),
            maximum_returned_cpu_prediction_tensor_bytes=len(observations) * 3 * 4,
            cone_tolerance=self.inputs.cone_tolerance,
        )
        result.assert_current()
        receipt = result.receipt
        expected_count = provider.frame_count * len(selected)
        expected_native_bundles = math.ceil(
            len(selected) / self._heldout_native_tracks_per_bundle_limit()
        )
        if (
            receipt.observation_count != expected_count
            or receipt.bundle_count != expected_native_bundles
            or receipt.selected_pixel_read_observation_count != expected_count
            or receipt.full_frame_target_materialization_count != 0
            or tuple(result.rgb_f32_cpu.shape) != (expected_count, 3)
        ):
            raise ArithmeticError("cross-time heldout prediction lost spatial-major reuse")
        self._heldout_spatial_major_call_count += 1
        self._heldout_spatial_major_track_count += len(selected)
        self._heldout_spatial_major_native_bundle_count += receipt.bundle_count
        self._heldout_spatial_major_native_sample_count += receipt.native_sample_count
        self._heldout_spatial_major_prediction_target_read_count += (
            receipt.selected_pixel_read_observation_count
        )
        self._heldout_spatial_major_prediction_receipt_chain_sha256 = (
            _canonical_sha256(
                {
                    "previous": (
                        self._heldout_spatial_major_prediction_receipt_chain_sha256
                    ),
                    "receipt": receipt.generation_digest,
                    "camera_index": camera_index,
                    "pixel_ids": selected,
                }
            )
        )
        prediction = result.rgb_f32_cpu.reshape(
            provider.frame_count,
            len(selected),
            3,
        ).contiguous()
        del observations, result
        return prediction

    def read_heldout_target_track_block_across_frames(
        self,
        *,
        camera_index: int,
        pixel_ids: tuple[int, ...],
    ) -> tuple[torch.Tensor, Mapping[str, Any]]:
        """Read pixel-time RGB without constructing the evaluator's unused rays."""

        self._assert_open()
        provider = self._heldout_provider
        if provider is None or not (
            self._training_finalized or self._heldout_pilot_prepared
        ):
            raise RuntimeError("WorldFoam heldout generation was not prepared")
        selected = tuple(int(value) for value in pixel_ids)
        image_pixels = provider.height * provider.width
        if (
            type(camera_index) is not int
            or not 0 <= camera_index < provider.view_count
            or not selected
            or len(selected) > self.maximum_heldout_tracks_per_cross_time_block()
            or tuple(sorted(selected)) != selected
            or len(set(selected)) != len(selected)
            or selected[0] < 0
            or selected[-1] >= image_pixels
        ):
            raise ValueError("WorldFoam cross-time target track block is invalid")
        # Pixel-major/time-inner order matches the mapped cache's physical
        # [pixel, frame, RGB] layout, preventing the frame-major page storm.
        view_indices = tuple(
            camera_index for _pixel in selected for _frame in range(provider.frame_count)
        )
        frame_indices = tuple(
            frame for _pixel in selected for frame in range(provider.frame_count)
        )
        requested_pixels = tuple(
            pixel for pixel in selected for _frame in range(provider.frame_count)
        )
        read = self.inputs.heldout_target_provider.select_view_frame_pixels_cpu(
            view_indices,
            frame_indices,
            requested_pixels,
            maximum_source_decode_tensor_bytes=(
                self.inputs.lazy_memory_policy.max_decoded_frame_scratch_tensor_bytes
            ),
        )
        expected_count = len(selected) * provider.frame_count
        read.assert_valid(
            expected_observation_count=expected_count,
            full_frame_tensor_bytes=provider.height * provider.width * 3 * 4,
        )
        if not read.acceptance_capable or read.selection_mode != "direct_pixels":
            raise ValueError("WorldFoam heldout target track read was not direct-pixel")
        target = read.rgb_f32_cpu.reshape(
            len(selected),
            provider.frame_count,
            3,
        ).permute(1, 0, 2).contiguous()
        returned_target_tensor_bytes = int(target.numel() * target.element_size())
        source_plus_returned_peak_bytes = int(
            read.source_visible_peak_logical_tensor_bytes_upper_bound
            + returned_target_tensor_bytes
        )
        receipt_payload = {
            "schema_version": 1,
            "kind": "worldfoam-spatial-major-target-track-read-v1",
            "camera_index": camera_index,
            "pixel_ids": selected,
            "pixel_ids_sha256": _canonical_sha256(selected),
            "track_count": len(selected),
            "frame_count": provider.frame_count,
            "observation_count": expected_count,
            "selection_mode": read.selection_mode,
            "source_provenance": read.source_provenance,
            "source_only_visible_peak_logical_tensor_bytes_upper_bound": (
                read.source_visible_peak_logical_tensor_bytes_upper_bound
            ),
            "returned_target_tensor_bytes": returned_target_tensor_bytes,
            "source_plus_returned_target_peak_logical_tensor_bytes_upper_bound": (
                source_plus_returned_peak_bytes
            ),
            "transient_mapped_address_space_bytes": (
                read.transient_mapped_address_space_bytes
            ),
            "requested_unique_mapped_page_count": (
                read.total_requested_unique_mapped_page_count
            ),
            "requested_mapped_page_bytes_upper_bound": (
                read.total_requested_mapped_page_bytes_upper_bound
            ),
            "mapping_closed_before_return": read.mapping_closed_before_return,
            "full_frame_materialization_count": (
                read.full_frame_materialization_count
            ),
            "ray_tensor_bytes": 0,
        }
        target_receipt = {
            **receipt_payload,
            "generation_digest": _canonical_sha256(receipt_payload),
        }
        self._heldout_spatial_major_target_staging_call_count += 1
        self._heldout_spatial_major_target_staging_observation_count += expected_count
        self._heldout_spatial_major_target_staging_peak_logical_bytes = max(
            self._heldout_spatial_major_target_staging_peak_logical_bytes,
            source_plus_returned_peak_bytes,
        )
        self._heldout_spatial_major_target_receipt_chain_sha256 = _canonical_sha256(
            {
                "previous": self._heldout_spatial_major_target_receipt_chain_sha256,
                "receipt": target_receipt["generation_digest"],
            }
        )
        return target, target_receipt

    def heldout_spatial_major_receipt(self) -> Mapping[str, Any]:
        if self._heldout_pilot_prepared and not self._training_finalized:
            raise RuntimeError(
                "bounded pilot work cannot issue a full-coverage heldout receipt"
            )
        provider = self._heldout_provider
        if provider is None:
            raise RuntimeError("WorldFoam heldout provider is unavailable")
        expected_tracks = provider.view_count * provider.height * provider.width
        expected_calls = provider.view_count * math.ceil(
            provider.height
            * provider.width
            / self.maximum_heldout_tracks_per_cross_time_block()
        )
        native_tracks_per_bundle = self._heldout_native_tracks_per_bundle_limit()
        call_limit = self.maximum_heldout_tracks_per_cross_time_block()
        full_calls, remainder = divmod(
            provider.height * provider.width,
            call_limit,
        )
        expected_native_bundles_per_view = (
            full_calls * math.ceil(call_limit / native_tracks_per_bundle)
            + (
                math.ceil(remainder / native_tracks_per_bundle)
                if remainder
                else 0
            )
        )
        expected_native_bundles = (
            provider.view_count * expected_native_bundles_per_view
        )
        if (
            self._heldout_spatial_major_track_count != expected_tracks
            or self._heldout_spatial_major_call_count != expected_calls
            or self._heldout_spatial_major_native_bundle_count
            != expected_native_bundles
            or self._heldout_spatial_major_native_sample_count
            != expected_tracks * provider.frame_count
            or self._heldout_spatial_major_prediction_target_read_count
            != expected_tracks * provider.frame_count
            or self._heldout_spatial_major_target_staging_call_count
            != expected_calls
            or self._heldout_spatial_major_target_staging_observation_count
            != expected_tracks * provider.frame_count
        ):
            raise ArithmeticError("WorldFoam spatial-major heldout coverage changed")
        payload = {
            "schema_version": 1,
            "kind": "worldfoam-spatial-major-full-temporal-heldout-v1",
            "camera_count": provider.view_count,
            "frame_count": provider.frame_count,
            "image_height": provider.height,
            "image_width": provider.width,
            "cross_time_track_block_size": (
                self.maximum_heldout_tracks_per_cross_time_block()
            ),
            "render_call_count": self._heldout_spatial_major_call_count,
            "cold_track_compile_count": self._heldout_spatial_major_track_count,
            "complete_camera_record_validation_count": (
                self._heldout_spatial_major_track_count * provider.frame_count
            ),
            "admitted_site_reference_upper_bound": (
                self._heldout_spatial_major_track_count * self.state.site_count
            ),
            "native_bundle_count": self._heldout_spatial_major_native_bundle_count,
            "native_tracks_per_bundle_limit": native_tracks_per_bundle,
            "expected_native_bundle_count": expected_native_bundles,
            "native_sample_count": self._heldout_spatial_major_native_sample_count,
            "native_prediction_target_observation_read_count": (
                self._heldout_spatial_major_prediction_target_read_count
            ),
            "spatial_target_staging_call_count": (
                self._heldout_spatial_major_target_staging_call_count
            ),
            "spatial_target_staging_observation_count": (
                self._heldout_spatial_major_target_staging_observation_count
            ),
            "spatial_target_staging_peak_logical_bytes": (
                self._heldout_spatial_major_target_staging_peak_logical_bytes
            ),
            "prediction_receipt_chain_sha256": (
                self._heldout_spatial_major_prediction_receipt_chain_sha256
            ),
            "target_receipt_chain_sha256": (
                self._heldout_spatial_major_target_receipt_chain_sha256
            ),
            "target_ray_tensor_bytes": 0,
            "full_pixel_full_temporal": True,
            "frame_major_recompile_per_time_used": False,
            "prediction_spool_dtype": "float32",
        }
        return {**payload, "generation_digest": _canonical_sha256(payload)}

    def heldout_spatial_major_partial_pilot_receipt(self) -> Mapping[str, Any]:
        """Report actual pilot work without implying full evaluation coverage."""

        self._assert_open()
        provider = self._heldout_provider
        if (
            not self._heldout_pilot_prepared
            or self._training_finalized
            or provider is None
            or self._optimizer_steps != 1
            or self._heldout_spatial_major_call_count < 1
            or self._heldout_spatial_major_track_count < 1
            or self._heldout_prediction_receipt_count < 1
        ):
            raise RuntimeError(
                "partial pilot receipt requires both old and cross-time heldout work"
            )
        payload = {
            "schema_version": 1,
            "kind": "worldfoam-spatial-major-heldout-partial-pilot-v1",
            "pilot_only": True,
            "paper_evidence": False,
            "full_coverage": False,
            "optimizer_step": 1,
            "camera_count": provider.view_count,
            "frame_count": provider.frame_count,
            "image_height": provider.height,
            "image_width": provider.width,
            "cross_time_render_call_count": self._heldout_spatial_major_call_count,
            "cross_time_cold_track_compile_count": (
                self._heldout_spatial_major_track_count
            ),
            "cross_time_complete_camera_record_validation_count": (
                self._heldout_spatial_major_track_count * provider.frame_count
            ),
            "cross_time_admitted_site_reference_upper_bound": (
                self._heldout_spatial_major_track_count * self.state.site_count
            ),
            "cross_time_native_bundle_count": (
                self._heldout_spatial_major_native_bundle_count
            ),
            "cross_time_native_sample_count": (
                self._heldout_spatial_major_native_sample_count
            ),
            "cross_time_prediction_target_observation_read_count": (
                self._heldout_spatial_major_prediction_target_read_count
            ),
            "cross_time_target_staging_call_count": (
                self._heldout_spatial_major_target_staging_call_count
            ),
            "cross_time_target_staging_observation_count": (
                self._heldout_spatial_major_target_staging_observation_count
            ),
            "old_frame_major_render_call_count": (
                self._heldout_prediction_receipt_count
            ),
            "old_frame_major_observation_count": (
                self._heldout_prediction_observation_count
            ),
            "cross_time_prediction_receipt_chain_sha256": (
                self._heldout_spatial_major_prediction_receipt_chain_sha256
            ),
            "old_frame_major_prediction_receipt_chain_sha256": (
                self._heldout_prediction_receipt_chain_sha256
            ),
        }
        return {**payload, "generation_digest": _canonical_sha256(payload)}

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._peak_sampler.stop()
        except BaseException:
            pass
        try:
            synchronize_mps_device_completion_fence()
        except BaseException:
            state = getattr(self, "state", None)
            if isinstance(state, PaperKineticFixedCameraCombinedState):
                state.poisoned = True
                state.material_state.poisoned = True
            if not any(
                value is self for value in _UNKNOWN_SESSION_COMPLETION_QUARANTINE
            ):
                _UNKNOWN_SESSION_COMPLETION_QUARANTINE.append(self)
            self._closed = True
            raise
        self._heldout_material_f32_mps = None
        self._heldout_background_f32_mps = None
        if self._heldout_provider is not None:
            try:
                object.__setattr__(self._heldout_provider, "_seal", None)
            except BaseException:
                pass
            self._heldout_provider = None
        store = getattr(self, "artifact_store", None)
        if isinstance(store, PaperKineticCompiledCpuArtifactStore):
            try:
                store.close()
            except BaseException:
                pass
        state = getattr(self, "state", None)
        if isinstance(state, PaperKineticFixedCameraCombinedState):
            state.active = False
            state.poisoned = True
            state.material_state.poisoned = True
        provider = getattr(self, "provider", None)
        if isinstance(provider, PaperKineticLazyProgramBundleProvider):
            try:
                object.__setattr__(provider, "_seal", None)
            except BaseException:
                pass
        self._closed = True

@torch.no_grad()
def run_public_quality_runtime_smoke(
    *,
    context: Any,
    dataset: Any,
) -> Mapping[str, Any]:
    """Attest one real-native public pixel without emitting paper evidence.

    This is the sole bootstrap exception to the runtime-receipt gate.  It does
    not bypass native ABI freshness, public mapped providers, calibrated rays,
    the deterministic world initializer, optimizer policies, or heldout
    rendering.  The attestor calls it in a fresh process and only writes the
    separate capability receipt after validating this primitive result.
    """

    route = str(context.request.route)
    if route not in SUPPORTED_ROUTES:
        raise ValueError("WorldFoam runtime smoke received another route")
    native_ops, native_library_sha256, native_error = _load_and_attest_native_ops()
    if native_ops is None or not _is_sha256(native_library_sha256):
        raise RuntimeError(
            "WorldFoam runtime smoke requires the rebuilt native ABI: "
            + str(native_error)
        )
    if not torch.backends.mps.is_available():
        raise RuntimeError("WorldFoam runtime smoke requires real MPS")
    accessor = getattr(dataset, "worldfoam_training_inputs", None)
    if not callable(accessor):
        raise TypeError("runtime smoke dataset lacks WorldFoam inputs")
    inputs = accessor()
    if type(inputs) is not WorldFoamPublicQualityInputs:
        raise TypeError("runtime smoke requires exact sealed WorldFoam inputs")
    inputs.assert_current(dataset=dataset, context=context)

    started = time.perf_counter()
    session: WorldFoamPublicQualitySession | None = None
    checkpoint_path = (
        Path(context.request.output_path).resolve().parent
        / f".{route}.runtime_smoke_checkpoint.pt"
    )
    checkpoint_path.unlink(missing_ok=True)
    try:
        session = WorldFoamPublicQualitySession(
            context=context,
            dataset=dataset,
            route=route,
            inputs=inputs,
            native_ops=native_ops,
            native_library_sha256=str(native_library_sha256),
        )
        before = session._representation_digest()
        source_work = context.work_plan.steps[0]
        first_sample = source_work.batch.samples[0]
        smoke_batch = SpacetimeBatch(
            samples=(first_sample,),
            epoch=source_work.batch.epoch,
            batch_index=source_work.batch.batch_index,
            completes_epoch=False,
        )
        smoke_work = SimpleNamespace(
            step=0,
            stage=SimpleNamespace(
                image_size=SimpleNamespace(pixels=1),
                lr_multiplier=float(source_work.stage.lr_multiplier),
            ),
            batch=smoke_batch,
        )
        session._start_training_measurement()
        if route == "worldfoam_native4d":
            session._run_shared_step(smoke_work)
        else:
            session._run_framewise_step(smoke_work)
        session._optimizer_steps = 1
        session._target_pixels = 1
        session._sampled_images = 1
        session._pixel_chunks = 1
        after = session._representation_digest()
        if before == after or len(session._promotion_receipts) != 1:
            raise ArithmeticError("runtime smoke observed no optimizer promotion")
        promotion = session._promotion_receipts[0]
        promotion.assert_self_consistent()

        checkpoint_payload = session._checkpoint_payload(after)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint_payload, checkpoint_path)
        del checkpoint_payload
        checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        if checkpoint_path.stat().st_size < 1:
            raise ArithmeticError("runtime smoke checkpoint is empty")

        session._training_finalized = True
        session._prepare_heldout_generation()
        from worldfoam_native4d_public_quality_row import PixelChunkRequest

        heldout_request = PixelChunkRequest(
            split="heldout",
            step=None,
            sample_slot=None,
            camera_index=0,
            frame_index=int(first_sample.frame_index),
            pixel_start=0,
            pixel_count=1,
            image_height=inputs.heldout_target_provider.height,
            image_width=inputs.heldout_target_provider.width,
        )
        heldout_payload = dataset.read_heldout_chunk(heldout_request)
        prediction = session.render_heldout_chunk(
            heldout_request,
            heldout_payload.rays_f32_cpu,
        )
        if (
            tuple(prediction.shape) != (1, 3)
            or not bool(torch.isfinite(prediction).all().item())
            or not _is_sha256(session._last_heldout_prediction_receipt_sha256)
        ):
            raise FloatingPointError("runtime smoke heldout prediction is invalid")
        synchronize_mps_device_completion_fence()
        peak_rss, peak_mps = session._peak_sampler.stop()
        parameters = (
            session.state.material_state.raw_color_f32,
            session.state.material_state.raw_density_f32,
            session.state.positions0_f64,
            session.state.velocities_f64,
            session.state.weight_coefficients_f64,
        )
        executor_source_sha256 = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        source_receipt_sha256 = _canonical_sha256(
            {
                "executor_source_sha256": executor_source_sha256,
                "input_generation_digest": inputs.input_generation_digest,
                "dataset_capability_sha256": inputs.dataset_capability_sha256,
                "route": route,
            }
        )
        native_receipt_sha256 = _canonical_sha256(
            {
                "native_library_sha256": native_library_sha256,
                "promotion_receipt": promotion.generation_digest,
                "heldout_prediction_receipt": (
                    session._last_heldout_prediction_receipt_sha256
                ),
            }
        )
        executor_receipt_sha256 = _canonical_sha256(
            {
                "checkpoint_sha256": checkpoint_sha256,
                "representation_before": before,
                "representation_after": after,
                "source_receipt_sha256": source_receipt_sha256,
                "native_receipt_sha256": native_receipt_sha256,
            }
        )
        receipt = {
            "schema_version": 1,
            "kind": "public-quality-route-runtime-smoke-v1",
            "status": "runtime_verified",
            "route": route,
            "lane": str(context.route_spec["lane"]),
            "execution_mode": str(context.route_spec["execution_mode"]),
            "backend": str(context.route_spec["backend"]),
            "real_native": True,
            "native_extension_attested": False,
            "fake_native": False,
            "source_only": False,
            "procedural_target": False,
            "public_target_provider": True,
            "paper_evidence_eligible": False,
            "smoke": True,
            "device": "mps",
            "optimizer_steps": 1,
            "train_render_count": 1,
            "backward_passes": 1,
            "optimizer_updates": 1,
            "heldout_render_count": 1,
            "finite_train_loss": math.isfinite(float(promotion.loss)),
            "finite_gradients": True,
            "parameter_update_observed": before != after,
            "finite_heldout_rgb": bool(torch.isfinite(prediction).all().item()),
            "target_pixels": 1,
            "rasterized_pixels": 1,
            "parameter_count": sum(int(tensor.numel()) for tensor in parameters),
            "parameter_bytes": _tensor_bytes(*parameters),
            "sampled_peak_process_rss_bytes": int(peak_rss),
            "sampled_peak_mps_driver_allocated_bytes": int(peak_mps),
            "elapsed_s": float(time.perf_counter() - started),
            "representation_sha256_before": before,
            "representation_sha256_after": after,
            "executor_receipt_sha256": executor_receipt_sha256,
            "native_receipt_sha256": native_receipt_sha256,
            "source_receipt_sha256": source_receipt_sha256,
        }
        if not all(
            receipt[key] is True
            for key in (
                "real_native",
                "public_target_provider",
                "finite_train_loss",
                "finite_gradients",
                "parameter_update_observed",
                "finite_heldout_rgb",
            )
        ):
            raise RuntimeError("WorldFoam runtime smoke did not verify its claims")
        return receipt
    finally:
        checkpoint_path.unlink(missing_ok=True)
        if session is not None:
            session.close()


def create_public_quality_executor(*, context: Any) -> WorldFoamPublicQualityExecutor:
    return WorldFoamPublicQualityExecutor(context=context)


__all__ = (
    "CHECKPOINT_SCHEMA",
    "EXECUTOR_PROVENANCE",
    "EXECUTOR_SCHEMA_VERSION",
    "WorldFoamPublicQualityExecutor",
    "WorldFoamPublicQualitySession",
    "create_public_quality_executor",
    "run_public_quality_runtime_smoke",
)
