"""One exact-coverage fixed-site material step without parameter mutation.

This source-level coordinator joins the replayable dense observation source,
the byte-bounded structural artifact store, the dense native request
transaction, and the optimizer-authorization capability.  It deliberately
stops before the fixed-site material state update.  A successful result keeps
only the authorization, its exact accumulator, and its exact replay receipt;
request objects, artifacts, target chunks, native lanes, and the replay source
remain outside the returned capability.

The state is caller-owned because a failed device fence may quarantine native
references on the accumulator.  Successful steps leave no step-local object
on the state.  A partial/device-progress failure poisons the state and retains
the source/session/accumulator until process restart.
"""

from __future__ import annotations

import hashlib
import math
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import torch
from paper_kinetic_runtime_paths import ensure_worldfoam_lane2_research_path

ensure_worldfoam_lane2_research_path()

from kinetic_compiled_cpu_artifact_store import (  # noqa: E402
    PaperKineticCompiledCpuArtifactStore,
    compile_paper_kinetic_compiled_cpu_artifact,
)
from kinetic_dense_cached_native_material_request import (  # noqa: E402
    MPS_DEVICE_COMPLETION_FENCE_PROVENANCE,
    PaperKineticDenseCachedNativeMemoryPolicy,
    PaperKineticDenseOptimizerAuthorization,
    PaperKineticDenseStepGradientAccumulator,
    authorize_paper_kinetic_dense_optimizer_step,
    consume_paper_kinetic_dense_request_delta,
    prepare_paper_kinetic_dense_chunk_target_loader,
    fail_stop_paper_kinetic_dense_step,
    prepare_paper_kinetic_dense_step_gradient_accumulator,
    run_paper_kinetic_dense_cached_native_request,
    synchronize_mps_device_completion_fence,
)
from paper_kinetic_lazy_program_bundles import (  # noqa: E402
    PaperKineticLazyProgramBundleProvider,
)
from paper_kinetic_fixed_site_material_state import (  # noqa: E402
    PaperKineticFixedSiteMaterialState,
)
from paper_kinetic_replayable_observations import (  # noqa: E402
    TRACK_ID_LOGICAL_BYTES,
    PaperKineticDenseObservationMemoryPolicy,
    PaperKineticDenseObservationReplayReceipt,
    PaperKineticDenseObservationReplaySession,
    PaperKineticReplayableDenseObservationSource,
    prepare_paper_kinetic_replayable_dense_observation_source,
)
from paper_training_types import SpacetimeBatch  # noqa: E402


STEP_PROVENANCE = "paper-kinetic-fixed-site-material-only-step-v2"
STEP_STATUS = "source_integrated/native_runtime_unverified"
GLOBAL_RGB_MEAN_LOSS_NORMALIZATION_ID = "global-rgb-mean-v1"
ACTIVE_MATERIAL_MODEL_FORMULA = (
    "sum_q(16*S_q+32*Q_q*J_q)+16*U+16*max(S_q)+4*n_q+16"
)

_STATE_SEAL = object()
_RESULT_SEAL = object()
_LOCK_TYPE = type(threading.Lock())


@dataclass
class _StepStructuralAccounting:
    """O(1)-metadata fold of sealed compiler and native launch receipts."""

    compiler_provenance: str = ""
    physical_interval_digest: str = ""
    artifact_signature_chain_digest: str = ""
    camera_path_signature_chain_digest: str = ""
    artifact_count: int = 0
    event_count: int = 0
    track_chart_row_count: int = 0
    word_entry_count: int = 0
    fallback_count: int = 0
    active_native_block_count: int = 0
    node_forward_launch_count: int = 0
    node_forward_thread_count: int = 0
    node_forward_interaction_count: int = 0
    material_word_vjp_interaction_count: int = 0
    active_material_exact_model_bytes: int = 0
    chart_node_ranks: set[int] = field(default_factory=set)

    def add(self, accounting: Mapping[str, Any]) -> None:
        compiler = str(accounting.get("compiler_provenance", ""))
        interval = str(accounting.get("physical_interval_digest", ""))
        artifact_signature = str(
            accounting.get("artifact_structural_signature_sha256", "")
        )
        camera_signature = str(
            accounting.get("compiled_camera_path_signature_sha256", "")
        )
        if (
            not compiler
            or len(interval) != 64
            or len(artifact_signature) != 64
            or len(camera_signature) != 64
        ):
            raise ValueError("material request omitted sealed structural provenance")
        if self.compiler_provenance and self.compiler_provenance != compiler:
            raise ValueError("one material step mixed structural compilers")
        if self.physical_interval_digest and self.physical_interval_digest != interval:
            raise ValueError("one material step mixed physical ray-time intervals")
        self.compiler_provenance = compiler
        self.physical_interval_digest = interval
        for key in (
            "event_count",
            "track_chart_row_count",
            "word_entry_count",
            "fallback_count",
            "active_native_block_count",
            "node_forward_launch_count",
            "node_forward_thread_count",
            "node_forward_interaction_count",
            "material_word_vjp_interaction_count",
        ):
            value = accounting.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"material request structural {key} is invalid")
            setattr(self, key, getattr(self, key) + value)
        exact_model_bytes = accounting.get("active_material_exact_model_bytes")
        if (
            isinstance(exact_model_bytes, bool)
            or not isinstance(exact_model_bytes, int)
            or exact_model_bytes < 1
        ):
            raise ValueError("material request exact active-model bytes are invalid")
        self.active_material_exact_model_bytes = max(
            self.active_material_exact_model_bytes,
            exact_model_bytes,
        )
        ranks = accounting.get("chart_node_ranks")
        if (
            not isinstance(ranks, tuple)
            or not ranks
            or any(
                isinstance(rank, bool)
                or not isinstance(rank, int)
                or rank < 1
                for rank in ranks
            )
        ):
            raise ValueError("material request chart-node ranks are invalid")
        self.chart_node_ranks.update(ranks)
        self.artifact_signature_chain_digest = _digest_parts(
            "paper-kinetic-step-artifact-signature-chain-v1",
            self.artifact_count,
            self.artifact_signature_chain_digest,
            artifact_signature,
        )
        self.camera_path_signature_chain_digest = _digest_parts(
            "paper-kinetic-step-compiled-camera-signature-chain-v1",
            self.artifact_count,
            self.camera_path_signature_chain_digest,
            camera_signature,
        )
        self.artifact_count += 1

    def report(
        self,
        source: PaperKineticReplayableDenseObservationSource,
        *,
        expected_artifact_count: int,
    ) -> dict[str, Any]:
        if (
            self.artifact_count != expected_artifact_count
            or self.artifact_count < 1
            or not self.compiler_provenance
            or len(self.physical_interval_digest) != 64
            or len(self.camera_path_signature_chain_digest) != 64
            or not self.chart_node_ranks
            or self.track_chart_row_count < 1
            or self.word_entry_count < 1
            or self.active_native_block_count < 1
            or self.node_forward_launch_count != self.active_native_block_count
            or self.node_forward_thread_count < 1
            or self.node_forward_interaction_count < 1
            or self.material_word_vjp_interaction_count < 1
            or self.node_forward_thread_count
            > self.node_forward_interaction_count
            or self.node_forward_interaction_count
            != self.material_word_vjp_interaction_count
            or self.active_material_exact_model_bytes < 1
        ):
            raise ArithmeticError("material step structural accounting is incomplete")
        ranks = tuple(sorted(self.chart_node_ranks))
        report: dict[str, Any] = {
            "compiler_provenance": self.compiler_provenance,
            "active_material_model_formula": ACTIVE_MATERIAL_MODEL_FORMULA,
            "world_generation_digest": source.provider.world.sites_content_digest,
            "camera_generation_digest": self.camera_path_signature_chain_digest,
            "physical_interval_digest": self.physical_interval_digest,
            # The factory generation seals its rank, near/far, exact-root, and
            # numerical tolerance policy.  This is a policy identity, not a
            # measured error certificate.
            "tolerance_policy_digest": source.provider.factory_generation_digest,
            "event_count": self.event_count,
            "track_chart_row_count": self.track_chart_row_count,
            "word_entry_count": self.word_entry_count,
            "fallback_count": self.fallback_count,
            "active_native_block_count": self.active_native_block_count,
            "node_forward_launch_count": self.node_forward_launch_count,
            "node_forward_thread_count": self.node_forward_thread_count,
            "node_forward_interaction_count": self.node_forward_interaction_count,
            "material_word_vjp_interaction_count": (
                self.material_word_vjp_interaction_count
            ),
            "active_material_exact_model_bytes": (
                self.active_material_exact_model_bytes
            ),
            "chart_node_ranks": ranks,
        }
        report["structural_signature_sha256"] = _digest_parts(
            "paper-kinetic-fixed-site-step-structure-v1",
            self.artifact_signature_chain_digest,
            tuple(sorted(report.items())),
        )
        return report


@dataclass(frozen=True)
class PaperKineticFixedSiteMaterialOnlyStepPolicy:
    """Explicit source, artifact, sample, and request memory bounds."""

    observation_memory_policy: PaperKineticDenseObservationMemoryPolicy
    request_memory_policy: PaperKineticDenseCachedNativeMemoryPolicy
    maximum_world_site_count: int
    maximum_material_state_logical_tensor_bytes: int
    maximum_material_checkpoint_logical_tensor_bytes: int
    maximum_step_accumulator_logical_tensor_bytes: int
    maximum_tracks_per_request: int
    maximum_artifact_accounted_bytes: int
    maximum_samples_per_launch: int
    cone_tolerance: float

    def assert_valid(self) -> None:
        if not isinstance(
            self.observation_memory_policy,
            PaperKineticDenseObservationMemoryPolicy,
        ):
            raise TypeError("material step requires a dense observation memory policy")
        if not isinstance(
            self.request_memory_policy,
            PaperKineticDenseCachedNativeMemoryPolicy,
        ):
            raise TypeError("material step requires a dense native request memory policy")
        self.observation_memory_policy.assert_valid()
        self.request_memory_policy.assert_valid()
        for name, value in (
            ("maximum_world_site_count", self.maximum_world_site_count),
            (
                "maximum_material_state_logical_tensor_bytes",
                self.maximum_material_state_logical_tensor_bytes,
            ),
            (
                "maximum_material_checkpoint_logical_tensor_bytes",
                self.maximum_material_checkpoint_logical_tensor_bytes,
            ),
            (
                "maximum_step_accumulator_logical_tensor_bytes",
                self.maximum_step_accumulator_logical_tensor_bytes,
            ),
            ("maximum_tracks_per_request", self.maximum_tracks_per_request),
            (
                "maximum_artifact_accounted_bytes",
                self.maximum_artifact_accounted_bytes,
            ),
            ("maximum_samples_per_launch", self.maximum_samples_per_launch),
        ):
            _require_positive_int(value, name=name)
        if not math.isfinite(self.cone_tolerance) or self.cone_tolerance <= 0.0:
            raise ValueError("material step cone_tolerance must be finite and positive")

    @property
    def generation_digest(self) -> str:
        self.assert_valid()
        return _digest_parts(
            STEP_PROVENANCE,
            "step-policy",
            tuple(self.observation_memory_policy.__dict__.items()),
            tuple(self.request_memory_policy.__dict__.items()),
            self.maximum_world_site_count,
            self.maximum_material_state_logical_tensor_bytes,
            self.maximum_material_checkpoint_logical_tensor_bytes,
            self.maximum_step_accumulator_logical_tensor_bytes,
            self.maximum_tracks_per_request,
            self.maximum_artifact_accounted_bytes,
            self.maximum_samples_per_launch,
            self.cone_tolerance,
        )


@dataclass(frozen=True)
class PaperKineticFixedSiteMaterialOnlyGenerationPolicy:
    """Exact logical-step and immutable-input generation identifiers."""

    step_index: int
    material_generation_id: str
    background_generation_id: str
    target_generation_id: str

    def assert_valid(self) -> None:
        if (
            isinstance(self.step_index, bool)
            or not isinstance(self.step_index, int)
            or self.step_index < 0
        ):
            raise ValueError("step_index must be a nonnegative integer")
        for name, value in (
            ("material_generation_id", self.material_generation_id),
            ("background_generation_id", self.background_generation_id),
            ("target_generation_id", self.target_generation_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")

    @property
    def step_generation_id(self) -> str:
        self.assert_valid()
        return _digest_parts(
            STEP_PROVENANCE,
            "logical-step",
            self.step_index,
            self.material_generation_id,
            self.background_generation_id,
            self.target_generation_id,
        )

    @property
    def generation_digest(self) -> str:
        self.assert_valid()
        return _digest_parts(
            STEP_PROVENANCE,
            "generation-policy",
            self.step_index,
            self.step_generation_id,
            self.material_generation_id,
            self.background_generation_id,
            self.target_generation_id,
        )


@dataclass
class PaperKineticFixedSiteMaterialStepState:
    """Nonreentrant store-bound state and durable failed-fence lifetime root."""

    provider_generation_digest: str
    artifact_store: PaperKineticCompiledCpuArtifactStore = field(repr=False)
    device: torch.device
    authorized_step_count: int
    last_step_generation_id: str
    last_authorized_material_generation_id: str
    active_step_generation_id: str
    poisoned: bool
    restart_required: bool
    failure_type: str
    failure_message: str
    failure_fail_stop_completed: bool
    failure_lifetime_root_roles: tuple[str, ...]
    _provider_identity: int = field(repr=False)
    _artifact_store_identity: int = field(repr=False)
    _failed_source: PaperKineticReplayableDenseObservationSource | None = field(
        default=None,
        repr=False,
    )
    _failed_session: PaperKineticDenseObservationReplaySession | None = field(
        default=None,
        repr=False,
    )
    _failed_accumulator: PaperKineticDenseStepGradientAccumulator | None = field(
        default=None,
        repr=False,
    )
    _failed_lifetime_roots: tuple[Any, ...] = field(default=(), repr=False)
    provenance: str = STEP_PROVENANCE
    _execution_lock: threading.Lock = field(
        default_factory=threading.Lock,
        repr=False,
    )
    _seal: object = field(default=None, repr=False)

    @property
    def failed_accumulator(self) -> PaperKineticDenseStepGradientAccumulator | None:
        return self._failed_accumulator

    @property
    def failed_replay_session(self) -> PaperKineticDenseObservationReplaySession | None:
        return self._failed_session

    def assert_current(
        self,
        provider: PaperKineticLazyProgramBundleProvider,
    ) -> None:
        if not isinstance(provider, PaperKineticLazyProgramBundleProvider):
            raise TypeError("material step state requires its kinetic provider")
        provider.assert_warm_current()
        failure_objects = (
            self._failed_source,
            self._failed_session,
            self._failed_accumulator,
        )
        if (
            self._seal is not _STATE_SEAL
            or self.provenance != STEP_PROVENANCE
            or id(provider) != self._provider_identity
            or provider.generation_digest != self.provider_generation_digest
            or not isinstance(
                self.artifact_store,
                PaperKineticCompiledCpuArtifactStore,
            )
            or id(self.artifact_store) != self._artifact_store_identity
            or not isinstance(self.device, torch.device)
            or self.device.type not in {"cpu", "mps"}
            or self.authorized_step_count < 0
            or bool(self.authorized_step_count)
            != bool(self.last_step_generation_id)
            or bool(self.authorized_step_count)
            != bool(self.last_authorized_material_generation_id)
            or bool(self.active_step_generation_id) and self.poisoned
            or self.restart_required != self.poisoned
            or bool(self.failure_type) != self.poisoned
            or bool(self.failure_message) != self.poisoned
            or not isinstance(self.failure_fail_stop_completed, bool)
            or self.failure_fail_stop_completed and not self.poisoned
            or len(self.failure_lifetime_root_roles)
            != len(self._failed_lifetime_roots)
            or len(set(self.failure_lifetime_root_roles))
            != len(self.failure_lifetime_root_roles)
            or self.poisoned != all(value is not None for value in failure_objects)
            or not self.poisoned and any(value is not None for value in failure_objects)
            or not self.poisoned and self._failed_lifetime_roots
            or not isinstance(self._execution_lock, _LOCK_TYPE)
        ):
            raise ValueError("fixed-site material step state changed")
        if self.poisoned:
            if (
                not isinstance(
                    self._failed_source,
                    PaperKineticReplayableDenseObservationSource,
                )
                or not isinstance(
                    self._failed_session,
                    PaperKineticDenseObservationReplaySession,
                )
                or not isinstance(
                    self._failed_accumulator,
                    PaperKineticDenseStepGradientAccumulator,
                )
                or self._failed_source.provider is not provider
                or self._failed_session.source is not self._failed_source
                or self._failed_accumulator._source_identity
                != id(self._failed_source)
                or self._failed_accumulator._session_identity
                != id(self._failed_session)
            ):
                raise ValueError("fixed-site material failure quarantine changed")
            if self.failure_fail_stop_completed and (
                not self._failed_accumulator.poisoned
                or self._failed_accumulator.optimizer_authorized
            ):
                raise ValueError("fixed-site material fail-stop seal changed")

    def accounting(self) -> Mapping[str, int | str | bool]:
        return MappingProxyType(
            {
                "provenance": self.provenance,
                "authorized_step_count": self.authorized_step_count,
                "step_completion_semantics": (
                    "authorization_only_external_optimizer_apply_required"
                ),
                "active": bool(self.active_step_generation_id),
                "poisoned": self.poisoned,
                "restart_required": self.restart_required,
                "failure_fail_stop_completed": self.failure_fail_stop_completed,
                "failed_source_retained": self._failed_source is not None,
                "failed_session_retained": self._failed_session is not None,
                "failed_accumulator_retained": self._failed_accumulator is not None,
                "failed_request_lifetime_root_count": len(
                    self._failed_lifetime_roots
                ),
                "successful_step_local_object_count_retained": 0,
            }
        )


class PaperKineticFixedSiteMaterialStepPartialFailure(RuntimeError):
    """Partial/device progress failed and is durably rooted on ``state``."""

    def __init__(
        self,
        state: PaperKineticFixedSiteMaterialStepState,
        cause: BaseException,
    ) -> None:
        super().__init__(
            "fixed-site material step failed after partial/device progress; "
            "the state is poisoned and process restart is required: "
            f"{type(cause).__qualname__}: {cause}"
        )
        self.state = state


@dataclass(frozen=True)
class PaperKineticFixedSiteMaterialStepResult:
    """Optimizer capability plus only the two objects required to validate it."""

    authorization: PaperKineticDenseOptimizerAuthorization = field(repr=False)
    accumulator: PaperKineticDenseStepGradientAccumulator = field(repr=False)
    replay_receipt: PaperKineticDenseObservationReplayReceipt = field(repr=False)
    loss_rgb_mean: float
    accounting: Mapping[str, Any]
    generation_digest: str
    _authorization_identity: int = field(repr=False)
    _accumulator_identity: int = field(repr=False)
    _replay_receipt_identity: int = field(repr=False)
    provenance: str = STEP_PROVENANCE
    runtime_status: str = STEP_STATUS
    parameter_mutation_count: int = 0
    retained_authorization_capability_object_count: int = 3
    retained_source_count: int = 0
    retained_session_count: int = 0
    retained_request_count: int = 0
    retained_artifact_count: int = 0
    retained_target_count: int = 0
    retained_native_lane_count: int = 0
    native_runtime_verified: bool = False
    allocator_peak_measured: bool = False
    _seal: object = field(default=None, repr=False)

    def assert_current(self) -> None:
        if not isinstance(self.authorization, PaperKineticDenseOptimizerAuthorization):
            raise TypeError("material step result lost its optimizer authorization")
        if not isinstance(self.accumulator, PaperKineticDenseStepGradientAccumulator):
            raise TypeError("material step result lost its gradient accumulator")
        if not isinstance(
            self.replay_receipt,
            PaperKineticDenseObservationReplayReceipt,
        ):
            raise TypeError("material step result lost its replay receipt")
        self.authorization.assert_current(self.accumulator, self.replay_receipt)
        geometry_values = (
            self.authorization.grad_positions0_f64,
            self.authorization.grad_velocities_f64,
            self.authorization.grad_weight_coefficients_f64,
            self.authorization.grad_track_ray_coefficients_f64,
        )
        if (
            self._seal is not _RESULT_SEAL
            or self.provenance != STEP_PROVENANCE
            or self.runtime_status != STEP_STATUS
            or id(self.authorization) != self._authorization_identity
            or id(self.accumulator) != self._accumulator_identity
            or id(self.replay_receipt) != self._replay_receipt_identity
            or self.authorization.full_geometry
            or self.accumulator.full_geometry
            or self.authorization.ray_bar_keys
            or any(value is not None for value in geometry_values)
            or not math.isfinite(self.loss_rgb_mean)
            or self.loss_rgb_mean < 0.0
            or self.parameter_mutation_count != 0
            or self.retained_authorization_capability_object_count != 3
            or any(
                value != 0
                for value in (
                    self.retained_source_count,
                    self.retained_session_count,
                    self.retained_request_count,
                    self.retained_artifact_count,
                    self.retained_target_count,
                    self.retained_native_lane_count,
                )
            )
            or self.native_runtime_verified
            or self.allocator_peak_measured
            or not isinstance(self.accounting, MappingProxyType)
            or self.accounting.get("loss_normalization_id")
            != GLOBAL_RGB_MEAN_LOSS_NORMALIZATION_ID
            or self.accounting.get("global_rgb_mean_application_count") != 1
            or self.accounting.get(
                "accumulator_initialization_fence_call_count"
            ) != 1
            or self.accounting.get("parameter_mutation_count") != 0
            or self.accounting.get("optimizer_step_executed") is not False
            or self.accounting.get("coordinator_completion_semantics")
            != "authorization_only_external_optimizer_apply_required"
            or self.accounting.get("built_in_bounded_target_decoder") is not True
            or self.accounting.get("arbitrary_external_target_loader") is not False
            or self.accounting.get(
                "target_source_decode_budget_enforced_before_allocation"
            )
            is not True
            or any(
                self.accounting.get(key) != 0
                for key in (
                    "persistent_frame_tensor_bytes",
                    "persistent_sample_tensor_bytes",
                    "persistent_target_tensor_bytes",
                    "persistent_prediction_tensor_bytes",
                    "reachable_autograd_tensor_count",
                )
            )
            or self.accounting.get("step_accumulator_retains_frame_axis") is not False
            or self.accounting.get("autograd_graph_retained") is not False
            or self.accounting.get("transferred_target_payload_bytes")
            != self.replay_receipt.observation_count * 12
            or int(self.accounting.get("sample_node_interaction_count", 0))
            < self.replay_receipt.observation_count
            or int(self.accounting.get("peak_sample_launch_node_count", 0)) < 1
            or self.accounting.get("autograd_saved_tensor_peak_measured") is not False
            or self.accounting.get(
                "sample_materialization_source_visible_logical_tensors_accounted"
            )
            is not True
            or self.accounting.get(
                "sample_materialization_float64_scratch_measured"
            )
            is not False
            or self.generation_digest != _result_digest(self)
        ):
            raise ValueError("fixed-site material step result changed")


def prepare_paper_kinetic_fixed_site_material_step_state(
    provider: PaperKineticLazyProgramBundleProvider,
    artifact_store: PaperKineticCompiledCpuArtifactStore,
    *,
    device: torch.device | str,
    resume_material_state: PaperKineticFixedSiteMaterialState | None = None,
) -> PaperKineticFixedSiteMaterialStepState:
    """Bind a coordinator, optionally resuming sealed material-step history."""

    if not isinstance(provider, PaperKineticLazyProgramBundleProvider):
        raise TypeError("material step state requires a kinetic provider")
    if not isinstance(artifact_store, PaperKineticCompiledCpuArtifactStore):
        raise TypeError("material step state requires a bounded artifact store")
    provider.assert_current()
    artifact_store.report()
    resolved_device = torch.device(device)
    if resolved_device.type not in {"cpu", "mps"}:
        raise NotImplementedError(
            "fixed-site material steps support CPU or canonical-fenced MPS; "
            "other asynchronous devices require an explicit fence contract"
        )
    authorized_step_count = 0
    last_step_generation_id = ""
    last_authorized_material_generation_id = ""
    if resume_material_state is not None:
        if not isinstance(
            resume_material_state,
            PaperKineticFixedSiteMaterialState,
        ):
            raise TypeError("coordinator resume requires a fixed-site material state")
        resume_material_state.assert_current()
        if (
            resume_material_state.world_generation_digest
            != provider.world.generation_digest
            or resume_material_state.sites_content_digest
            != provider.world.sites_content_digest
            or resume_material_state.site_count != provider.world.site_count
            or resume_material_state.device != resolved_device
        ):
            raise ValueError(
                "coordinator resume material state is foreign to its world/device"
            )
        authorized_step_count = resume_material_state.step_index
        last_step_generation_id = resume_material_state.last_step_generation_id
        # The last authorization consumed the material generation that became
        # the live state's parent after its optimizer update.  The current
        # material generation is intentionally eligible for the next step.
        last_authorized_material_generation_id = (
            resume_material_state.generation_parent_digest
        )
    result = PaperKineticFixedSiteMaterialStepState(
        provider_generation_digest=provider.generation_digest,
        artifact_store=artifact_store,
        device=resolved_device,
        authorized_step_count=authorized_step_count,
        last_step_generation_id=last_step_generation_id,
        last_authorized_material_generation_id=(
            last_authorized_material_generation_id
        ),
        active_step_generation_id="",
        poisoned=False,
        restart_required=False,
        failure_type="",
        failure_message="",
        failure_fail_stop_completed=False,
        failure_lifetime_root_roles=(),
        _provider_identity=id(provider),
        _artifact_store_identity=id(artifact_store),
        _seal=_STATE_SEAL,
    )
    result.assert_current(provider)
    return result


@torch.no_grad()
def run_paper_kinetic_fixed_site_material_only_step(
    state: PaperKineticFixedSiteMaterialStepState,
    provider: PaperKineticLazyProgramBundleProvider,
    batch: SpacetimeBatch,
    *,
    policy: PaperKineticFixedSiteMaterialOnlyStepPolicy,
    generation_policy: PaperKineticFixedSiteMaterialOnlyGenerationPolicy,
    global_site_rgba_f32: torch.Tensor,
    background_rgb_f32: torch.Tensor,
    native_ops: Any,
    backend_provenance: str,
    device_completion_fence: Callable[[], None],
    device_completion_fence_provenance: str,
) -> PaperKineticFixedSiteMaterialStepResult:
    """Authorize one exact dense material step without updating parameters."""

    if not isinstance(state, PaperKineticFixedSiteMaterialStepState):
        raise TypeError("material-only step requires its caller-owned state")
    acquired_lock = state._execution_lock.acquire(blocking=False)
    if not acquired_lock:
        raise RuntimeError("fixed-site material step state is already active")
    source: PaperKineticReplayableDenseObservationSource | None = None
    session: PaperKineticDenseObservationReplaySession | None = None
    accumulator: PaperKineticDenseStepGradientAccumulator | None = None
    active_request: Any = None
    active_artifact: Any = None
    active_request_result: Any = None
    execution_started = False
    unsafe_device_fence_failure = False
    accumulator_initialization_fence_call_count = 0
    try:
        state.assert_current(provider)
        if state.poisoned:
            raise RuntimeError("fixed-site material step state requires process restart")
        _validate_step_policy(
            state,
            provider,
            batch,
            policy=policy,
            generation_policy=generation_policy,
            global_site_rgba_f32=global_site_rgba_f32,
            background_rgb_f32=background_rgb_f32,
            backend_provenance=backend_provenance,
            device_completion_fence=device_completion_fence,
            device_completion_fence_provenance=(
                device_completion_fence_provenance
            ),
        )
        state.active_step_generation_id = generation_policy.step_generation_id
        material_signature = _tensor_signature(global_site_rgba_f32)
        background_signature = _tensor_signature(background_rgb_f32)
        state.artifact_store.report()
        source = prepare_paper_kinetic_replayable_dense_observation_source(
            provider,
            batch,
            memory_policy=policy.observation_memory_policy,
        )
        session = source.open_session()
        accumulator = prepare_paper_kinetic_dense_step_gradient_accumulator(
            source,
            session,
            step_generation_id=generation_policy.step_generation_id,
            loss_normalization_id=GLOBAL_RGB_MEAN_LOSS_NORMALIZATION_ID,
            material_generation_id=generation_policy.material_generation_id,
            background_generation_id=generation_policy.background_generation_id,
            global_site_rgba_f32=global_site_rgba_f32,
            background_rgb_f32=background_rgb_f32,
            device=state.device,
            full_geometry=False,
        )
        # Fresh accumulator tensors are device work.  Establish a completion
        # proof before any post-allocation invariant can fail or the objects can
        # leave this step.  A failing fence is handled without any further
        # tensor inspection or mutation.
        execution_started = True
        try:
            returned = device_completion_fence()
        except BaseException:
            unsafe_device_fence_failure = True
            raise
        if returned is not None:
            raise TypeError("accumulator initialization fence must return None")
        accumulator_initialization_fence_call_count = 1
        if (
            accumulator.logical_tensor_bytes
            > policy.maximum_step_accumulator_logical_tensor_bytes
        ):
            raise ArithmeticError("material step accumulator exceeded its preflight")
        expected_requests_per_view = (
            source.image_pixel_count + policy.maximum_tracks_per_request - 1
        ) // policy.maximum_tracks_per_request
        expected_request_count = (
            source.selected_view_count * expected_requests_per_view
        )
        counters: dict[str, int] = {
            "request_count": 0,
            "cold_artifact_count": 0,
            "warm_artifact_count": 0,
            "artifact_store_eviction_count": 0,
            "artifact_store_evicted_accounted_bytes": 0,
            "artifact_store_cold_compiled_track_count": 0,
            "artifact_store_avoided_compile_track_count": 0,
            "streamed_observation_count": 0,
            "replay_chunk_count": 0,
            "sample_launch_count": 0,
            "sample_node_interaction_count": 0,
            "transferred_target_payload_bytes": 0,
            "native_material_vjp_launch_count": 0,
            "request_commit_fence_call_count": 0,
            "native_lane_fence_call_count": 0,
            "selected_pixel_read_call_count": 0,
            "mapped_selected_pixel_read_call_count": 0,
            "mapping_closed_before_return_count": 0,
            "cumulative_requested_mapped_page_count": 0,
            "cumulative_requested_mapped_page_bytes_upper_bound": 0,
            "direct_selected_pixel_observation_count": 0,
            "bounded_region_selected_pixel_observation_count": 0,
            "full_frame_fallback_observation_count": 0,
            "full_frame_target_materialization_count": 0,
            "bounded_region_target_materialization_count": 0,
            "decoded_frame_count": 0,
            "accumulator_initialization_fence_call_count": (
                accumulator_initialization_fence_call_count
            ),
        }
        peaks: dict[str, int] = {
            "lane_resident_logical_tensor_bytes_upper_bound": 0,
            "active_request_logical_tensor_bytes_upper_bound": 0,
            "peak_target_decode_bridge_logical_tensor_bytes": 0,
            "peak_sample_launch_tensor_bytes": 0,
            "peak_sample_launch_node_count": 0,
            "peak_cpu_decoded_frame_tensor_bytes": 0,
            "peak_bounded_region_materialization_tensor_bytes": 0,
            "peak_source_visible_target_read_logical_tensor_bytes_upper_bound": 0,
            "peak_transient_mapped_address_space_bytes": 0,
            "peak_requested_unique_mapped_page_count": 0,
            "peak_mapped_page_size_bytes": 0,
            "peak_requested_mapped_page_bytes_upper_bound": 0,
            "peak_cpu_chunk_target_tensor_bytes": 0,
            "peak_device_chunk_target_tensor_bytes": 0,
            "peak_sample_materialization_logical_tensor_bytes_upper_bound": 0,
            "peak_interpolation_evaluator_scratch_logical_tensor_bytes_upper_bound": 0,
            "maximum_interpolation_rows_per_subchunk": 0,
            "effective_maximum_samples_per_launch": 0,
            "peak_native_prepared_sample_scratch_tensor_bytes": 0,
            "peak_public_sample_launch_logical_tensor_bytes": 0,
            "peak_chunk_dispatch_identity_logical_bytes": 0,
            "maximum_active_block_commit_scratch_tensor_bytes": 0,
            "request_delta_logical_tensor_bytes": 0,
        }
        structural_accounting = _StepStructuralAccounting()
        for view_index in source.canonical_view_indices:
            for track_start in range(
                0,
                source.image_pixel_count,
                policy.maximum_tracks_per_request,
            ):
                track_end = min(
                    track_start + policy.maximum_tracks_per_request,
                    source.image_pixel_count,
                )
                request = source.prepare_track_request(
                    view_index=view_index,
                    track_ids=tuple(range(track_start, track_end)),
                )
                active_request = request
                acquisition = state.artifact_store.acquire(
                    provider,
                    view_index=view_index,
                    track_ids=request.track_ids,
                    maximum_artifact_accounted_bytes=(
                        policy.maximum_artifact_accounted_bytes
                    ),
                    compile_artifact=lambda key: (
                        compile_paper_kinetic_compiled_cpu_artifact(provider, key)
                    ),
                )
                artifact = acquisition.artifact
                active_artifact = artifact

                built_in_target_loader = (
                    prepare_paper_kinetic_dense_chunk_target_loader(
                        source,
                        request,
                        device=state.device,
                        target_generation_id=(
                            f"{generation_policy.target_generation_id}:"
                            f"{request.generation_digest}"
                        ),
                        maximum_decoded_frame_scratch_tensor_bytes=(
                            policy.request_memory_policy.maximum_decoded_frame_scratch_tensor_bytes
                        ),
                        maximum_chunk_target_tensor_bytes=(
                            policy.request_memory_policy.maximum_chunk_target_tensor_bytes
                        ),
                        maximum_target_decode_bridge_peak_logical_tensor_bytes=(
                            policy.request_memory_policy.maximum_target_decode_bridge_peak_logical_tensor_bytes
                        ),
                    )
                )

                request_result = run_paper_kinetic_dense_cached_native_request(
                    source,
                    session,
                    request,
                    artifact,
                    accumulator,
                    step_generation_id=generation_policy.step_generation_id,
                    loss_normalization_id=GLOBAL_RGB_MEAN_LOSS_NORMALIZATION_ID,
                    material_generation_id=generation_policy.material_generation_id,
                    background_generation_id=(
                        generation_policy.background_generation_id
                    ),
                    global_site_rgba_f32=global_site_rgba_f32,
                    background_rgb_f32=background_rgb_f32,
                    native_ops=native_ops,
                    backend_provenance=backend_provenance,
                    maximum_samples_per_launch=policy.maximum_samples_per_launch,
                    memory_policy=policy.request_memory_policy,
                    load_chunk_targets=built_in_target_loader,
                    device_completion_fence=device_completion_fence,
                    device_completion_fence_provenance=(
                        device_completion_fence_provenance
                    ),
                    cone_tolerance=policy.cone_tolerance,
                )
                active_request_result = request_result
                commit_receipt = consume_paper_kinetic_dense_request_delta(
                    accumulator,
                    source,
                    session,
                    request,
                    artifact,
                    request_result.delta,
                    device_completion_fence=device_completion_fence,
                    device_completion_fence_provenance=(
                        device_completion_fence_provenance
                    ),
                )
                _accumulate_request_accounting(
                    counters,
                    peaks,
                    request_result.accounting,
                    telemetry=request_result.telemetry,
                    acquisition=acquisition,
                    structural_accounting=structural_accounting,
                    commit_fence_call_count=(
                        commit_receipt.device_completion_fence_call_count
                    ),
                    expected_global_loss_element_count=(
                        source.observation_count * 3
                    ),
                )
                active_request = None
                active_artifact = None
                active_request_result = None
                del commit_receipt, request_result, artifact, acquisition, request

        if (
            counters["request_count"] != expected_request_count
            or counters["streamed_observation_count"] != source.observation_count
        ):
            raise ArithmeticError("material step request partition changed coverage")
        replay_receipt = session.seal()
        if (
            replay_receipt.request_count != expected_request_count
            or replay_receipt.observation_count != source.observation_count
        ):
            raise ArithmeticError("material step replay seal changed exact coverage")
        authorization = authorize_paper_kinetic_dense_optimizer_step(
            accumulator,
            source,
            session,
            replay_receipt,
        )
        authorization.assert_current(accumulator, replay_receipt)
        if _tensor_signature(global_site_rgba_f32) != material_signature:
            raise ValueError("material step mutated the physical material snapshot")
        if _tensor_signature(background_rgb_f32) != background_signature:
            raise ValueError("material step mutated the background snapshot")
        loss_rgb_mean = float(authorization.loss_f32.detach().cpu().item())
        if not math.isfinite(loss_rgb_mean) or loss_rgb_mean < 0.0:
            raise FloatingPointError("material step produced a non-finite RGB mean")
        store_after = state.artifact_store.report()
        accounting = MappingProxyType(
            _step_accounting(
                source,
                accumulator,
                replay_receipt,
                policy=policy,
                generation_policy=generation_policy,
                counters=counters,
                peaks=peaks,
                structural_accounting=structural_accounting,
                store_after=store_after,
                loss_rgb_mean=loss_rgb_mean,
                backend_provenance=backend_provenance,
                device_completion_fence_provenance=(
                    device_completion_fence_provenance
                ),
            )
        )
        provisional = PaperKineticFixedSiteMaterialStepResult(
            authorization=authorization,
            accumulator=accumulator,
            replay_receipt=replay_receipt,
            loss_rgb_mean=loss_rgb_mean,
            accounting=accounting,
            generation_digest="",
            _authorization_identity=id(authorization),
            _accumulator_identity=id(accumulator),
            _replay_receipt_identity=id(replay_receipt),
            _seal=_RESULT_SEAL,
        )
        result = PaperKineticFixedSiteMaterialStepResult(
            **{
                **provisional.__dict__,
                "generation_digest": _result_digest(provisional),
            }
        )
        result.assert_current()
        state.authorized_step_count += 1
        state.last_step_generation_id = generation_policy.step_generation_id
        state.last_authorized_material_generation_id = (
            generation_policy.material_generation_id
        )
        state.active_step_generation_id = ""
        state.assert_current(provider)
        return result
    except BaseException as error:
        partial_progress = bool(
            execution_started
            or session is not None
            and (session.emitted_observation_count or session.sealed)
            or accumulator is not None
            and (
                accumulator.consumed_request_count
                or accumulator.pending_delta_generation_digest
                or accumulator.optimizer_authorized
                or accumulator.poisoned
            )
        )
        if partial_progress and source is not None and session is not None and accumulator is not None:
            _retain_failed_step(
                state,
                source,
                session,
                accumulator,
                error,
                lifetime_roots=(
                    ("request", active_request),
                    ("artifact", active_artifact),
                    ("request_result_and_delta", active_request_result),
                ),
                attempt_fail_stop=not unsafe_device_fence_failure,
            )
            raise PaperKineticFixedSiteMaterialStepPartialFailure(
                state,
                error,
            ) from error
        state.active_step_generation_id = ""
        raise
    finally:
        state._execution_lock.release()


def _validate_step_policy(
    state: PaperKineticFixedSiteMaterialStepState,
    provider: PaperKineticLazyProgramBundleProvider,
    batch: SpacetimeBatch,
    *,
    policy: PaperKineticFixedSiteMaterialOnlyStepPolicy,
    generation_policy: PaperKineticFixedSiteMaterialOnlyGenerationPolicy,
    global_site_rgba_f32: torch.Tensor,
    background_rgb_f32: torch.Tensor,
    backend_provenance: str,
    device_completion_fence: Callable[[], None],
    device_completion_fence_provenance: str,
) -> None:
    if not isinstance(policy, PaperKineticFixedSiteMaterialOnlyStepPolicy):
        raise TypeError("material step requires its explicit step policy")
    if not isinstance(
        generation_policy,
        PaperKineticFixedSiteMaterialOnlyGenerationPolicy,
    ):
        raise TypeError("material step requires its explicit generation policy")
    if not isinstance(batch, SpacetimeBatch):
        raise TypeError("material step requires a SpacetimeBatch")
    policy.assert_valid()
    generation_policy.assert_valid()
    if (
        generation_policy.step_index != state.authorized_step_count
        or generation_policy.step_generation_id == state.last_step_generation_id
        or generation_policy.material_generation_id
        == state.last_authorized_material_generation_id
    ):
        raise ValueError("material step generation was already authorized")
    site_count = provider.world.site_count
    material_state_bytes = 12 * site_count * 4
    material_checkpoint_bytes = 4 * site_count * 4
    step_accumulator_bytes = (4 * site_count + 1) * 4
    if site_count > policy.maximum_world_site_count:
        raise MemoryError("material step world site count exceeds its explicit bound")
    if (
        material_state_bytes
        > policy.maximum_material_state_logical_tensor_bytes
    ):
        raise MemoryError("fixed-site material state exceeds its explicit byte bound")
    if (
        material_checkpoint_bytes
        > policy.maximum_material_checkpoint_logical_tensor_bytes
    ):
        raise MemoryError("fixed-site material checkpoint exceeds its explicit byte bound")
    if (
        step_accumulator_bytes
        > policy.maximum_step_accumulator_logical_tensor_bytes
    ):
        raise MemoryError("material step accumulator exceeds its explicit byte bound")
    request_track_bytes = (
        policy.maximum_tracks_per_request * TRACK_ID_LOGICAL_BYTES
    )
    if (
        policy.maximum_tracks_per_request > provider.maximum_tracks_per_bundle
        or policy.maximum_tracks_per_request
        > policy.observation_memory_policy.maximum_request_track_count
        or request_track_bytes
        > policy.observation_memory_policy.maximum_request_track_logical_bytes
    ):
        raise ValueError("material step track partition exceeds a source/compiler bound")
    if (
        policy.maximum_artifact_accounted_bytes
        > state.artifact_store.policy.maximum_resident_accounted_bytes
    ):
        raise ValueError("material step artifact bound exceeds its store policy")
    if not isinstance(backend_provenance, str) or not backend_provenance.strip():
        raise ValueError("backend_provenance must be nonempty")
    if not callable(device_completion_fence):
        raise TypeError("material step requires a completion fence")
    if (
        not isinstance(device_completion_fence_provenance, str)
        or not device_completion_fence_provenance.strip()
    ):
        raise ValueError("device completion fence provenance must be nonempty")
    if state.device.type == "mps" and (
        device_completion_fence is not synchronize_mps_device_completion_fence
        or device_completion_fence_provenance
        != MPS_DEVICE_COMPLETION_FENCE_PROVENANCE
    ):
        raise ValueError("MPS material steps require the canonical completion fence")
    _require_f32_tensor(
        global_site_rgba_f32,
        name="global_site_rgba_f32",
        shape=(provider.world.site_count, 4),
        device=state.device,
    )
    _require_f32_tensor(
        background_rgb_f32,
        name="background_rgb_f32",
        shape=(3,),
        device=state.device,
    )
    if global_site_rgba_f32.requires_grad or background_rgb_f32.requires_grad:
        raise ValueError("material step snapshots must be explicit non-autograd tensors")
    if not bool(torch.isfinite(global_site_rgba_f32).all().item()) or not bool(
        torch.isfinite(background_rgb_f32).all().item()
    ):
        raise ValueError("material step snapshots must be finite")
    if _same_storage(global_site_rgba_f32, background_rgb_f32):
        raise ValueError("material and background snapshots must not alias")


def _accumulate_request_accounting(
    counters: dict[str, int],
    peaks: dict[str, int],
    accounting: Mapping[str, Any],
    *,
    telemetry: Any,
    acquisition: Any,
    structural_accounting: _StepStructuralAccounting,
    commit_fence_call_count: int,
    expected_global_loss_element_count: int,
) -> None:
    if (
        accounting.get("full_geometry_vjp_integrated") is not False
        or accounting.get("geometry_row_vjp_call_count") != 0
        or accounting.get("native_full_geometry_vjp_launch_count") != 0
        or accounting.get("caller_bars_mutated_by_request") is not False
        or accounting.get("optimizer_authorization_requires_full_manifest_seal")
        is not True
        or accounting.get(
            "sample_materialization_source_visible_logical_tensors_accounted"
        )
        is not True
        or accounting.get(
            "target_source_decode_budget_enforced_before_allocation"
        )
        is not True
    ):
        raise ValueError("material coordinator received a non-material request result")
    telemetry.assert_current()
    if (
        telemetry.reverse_mode != "material_only"
        or telemetry.loss_normalization_id
        != GLOBAL_RGB_MEAN_LOSS_NORMALIZATION_ID
        or telemetry.global_loss_element_count
        != expected_global_loss_element_count
        or telemetry.loss_scale
        != 1.0 / float(expected_global_loss_element_count)
    ):
        raise ValueError("material request changed the one global RGB mean")
    if accounting.get("expected_observation_count", 0) < 1:
        raise ArithmeticError("material request reported empty observation coverage")
    counters["request_count"] += 1
    if acquisition.cache_status == "cold_compiled":
        counters["cold_artifact_count"] += 1
    elif acquisition.cache_status == "warm_hit":
        counters["warm_artifact_count"] += 1
    else:
        raise ValueError("material step received an unknown artifact cache status")
    counters["artifact_store_eviction_count"] += int(
        acquisition.evicted_entry_count
    )
    counters["artifact_store_evicted_accounted_bytes"] += int(
        acquisition.evicted_accounted_bytes
    )
    counters["artifact_store_cold_compiled_track_count"] += int(
        acquisition.cold_compiled_track_count
    )
    counters["artifact_store_avoided_compile_track_count"] += int(
        acquisition.avoided_compile_track_count
    )
    for destination, source_key in (
        ("streamed_observation_count", "streamed_observation_count"),
        ("replay_chunk_count", "replay_chunk_count"),
        ("sample_launch_count", "sample_launch_count"),
        ("sample_node_interaction_count", "sample_node_interaction_count"),
        (
            "transferred_target_payload_bytes",
            "transferred_target_payload_bytes",
        ),
        (
            "native_material_vjp_launch_count",
            "native_material_word_vjp_launch_count",
        ),
        ("native_lane_fence_call_count", "native_lane_fence_count"),
        ("selected_pixel_read_call_count", "selected_pixel_read_call_count"),
        (
            "mapped_selected_pixel_read_call_count",
            "mapped_selected_pixel_read_call_count",
        ),
        (
            "mapping_closed_before_return_count",
            "mapping_closed_before_return_count",
        ),
        (
            "cumulative_requested_mapped_page_count",
            "cumulative_requested_mapped_page_count",
        ),
        (
            "cumulative_requested_mapped_page_bytes_upper_bound",
            "cumulative_requested_mapped_page_bytes_upper_bound",
        ),
        (
            "direct_selected_pixel_observation_count",
            "direct_selected_pixel_observation_count",
        ),
        (
            "bounded_region_selected_pixel_observation_count",
            "bounded_region_selected_pixel_observation_count",
        ),
        (
            "full_frame_fallback_observation_count",
            "full_frame_fallback_observation_count",
        ),
        (
            "full_frame_target_materialization_count",
            "full_frame_target_materialization_count",
        ),
        (
            "bounded_region_target_materialization_count",
            "bounded_region_target_materialization_count",
        ),
        ("decoded_frame_count", "decoded_frame_count"),
    ):
        counters[destination] += int(accounting[source_key])
    counters["request_commit_fence_call_count"] += commit_fence_call_count
    structural_accounting.add(accounting)
    for key in peaks:
        peaks[key] = max(peaks[key], int(accounting[key]))
    if int(accounting["step_accumulator_logical_tensor_bytes"]) < 1:
        raise ArithmeticError("material request lost the whole-step accumulator")


def _step_accounting(
    source: PaperKineticReplayableDenseObservationSource,
    accumulator: PaperKineticDenseStepGradientAccumulator,
    replay_receipt: PaperKineticDenseObservationReplayReceipt,
    *,
    policy: PaperKineticFixedSiteMaterialOnlyStepPolicy,
    generation_policy: PaperKineticFixedSiteMaterialOnlyGenerationPolicy,
    counters: Mapping[str, int],
    peaks: Mapping[str, int],
    structural_accounting: _StepStructuralAccounting,
    store_after: Any,
    loss_rgb_mean: float,
    backend_provenance: str,
    device_completion_fence_provenance: str,
) -> dict[str, Any]:
    structure_report = structural_accounting.report(
        source,
        expected_artifact_count=counters["request_count"],
    )
    if (
        structure_report["active_native_block_count"]
        != counters["native_material_vjp_launch_count"]
    ):
        raise ArithmeticError("material step structural/native VJP counts disagree")
    target_residency = source.provider.target_provider.residency()
    target_source_resident_tensor_bytes = int(
        target_residency.get("resident_bytes", -1)
    )
    target_source_raw_storage_bytes = int(
        target_residency.get("raw_storage_bytes", 0)
    )
    target_source_maximum_mapped_payload_bytes = int(
        target_residency.get("maximum_mapped_payload_bytes", 0)
    )
    target_source_maximum_total_payload_verification_bytes = int(
        target_residency.get("maximum_total_payload_verification_bytes", 0)
    )
    target_source_construction_verification_bytes = int(
        target_residency.get("construction_payload_verification_bytes", 0)
    )
    full_video_target_tensor_retained = bool(
        target_residency.get("full_source_resident")
    )
    if (
        target_source_resident_tensor_bytes < 0
        or target_source_raw_storage_bytes < 0
        or target_source_maximum_mapped_payload_bytes < 0
        or target_source_maximum_total_payload_verification_bytes < 0
        or target_source_construction_verification_bytes < 0
    ):
        raise ValueError("target provider omitted resident-byte accounting")
    selected_pixel_observation_count = (
        counters["direct_selected_pixel_observation_count"]
        + counters["bounded_region_selected_pixel_observation_count"]
        + counters["full_frame_fallback_observation_count"]
    )
    if selected_pixel_observation_count != counters["streamed_observation_count"]:
        raise ArithmeticError("material step selected-pixel coverage changed")
    if counters["selected_pixel_read_call_count"] != counters["replay_chunk_count"]:
        raise ArithmeticError("material step changed one selected-pixel read per chunk")
    if (
        counters["decoded_frame_count"]
        != counters["full_frame_target_materialization_count"]
    ):
        raise ArithmeticError("material step full-frame accounting aliases disagree")
    if (
        counters["direct_selected_pixel_observation_count"]
        == counters["streamed_observation_count"]
    ):
        selected_pixel_read_mode = "direct_pixels"
    elif (
        counters["bounded_region_selected_pixel_observation_count"]
        == counters["streamed_observation_count"]
    ):
        selected_pixel_read_mode = "certified_bounded_region"
    elif (
        counters["full_frame_fallback_observation_count"]
        == counters["streamed_observation_count"]
    ):
        selected_pixel_read_mode = "full_frame_fallback"
    else:
        selected_pixel_read_mode = "mixed"
    selected_pixel_read_acceptance_capable = (
        selected_pixel_read_mode
        in {"direct_pixels", "certified_bounded_region"}
        and counters["full_frame_target_materialization_count"] == 0
        and counters["full_frame_fallback_observation_count"] == 0
    )
    return {
        "provenance": STEP_PROVENANCE,
        "runtime_status": STEP_STATUS,
        "source_generation_digest": source.generation_digest,
        "compact_manifest_digest": source.compact_manifest_digest,
        # Provider/cache identity may include the requested observation grid.
        # Keep it diagnostic and distinct from structure_report's semantic
        # world_generation_digest, which is the sites-content identity used by
        # the cross-F causal gate.
        "provider_world_generation_digest": accumulator.world_generation_digest,
        "world_sites_content_digest": accumulator.world_sites_content_digest,
        "step_policy_generation_digest": policy.generation_digest,
        "generation_policy_digest": generation_policy.generation_digest,
        "step_index": generation_policy.step_index,
        "step_generation_id": generation_policy.step_generation_id,
        "material_generation_id": generation_policy.material_generation_id,
        "background_generation_id": generation_policy.background_generation_id,
        "target_generation_id": generation_policy.target_generation_id,
        "loss_normalization_id": GLOBAL_RGB_MEAN_LOSS_NORMALIZATION_ID,
        "global_rgb_mean_application_count": 1,
        "global_loss_element_count": source.observation_count * 3,
        "loss_scale": accumulator.loss_scale,
        "loss_rgb_mean": loss_rgb_mean,
        "selected_view_count": source.selected_view_count,
        "selected_frame_count": source.selected_frame_count,
        "image_height": source.provider.height,
        "image_width": source.provider.width,
        "image_pixel_count": source.image_pixel_count,
        "exact_observation_count": replay_receipt.observation_count,
        "exact_request_count": replay_receipt.request_count,
        "replay_chunk_count": counters["replay_chunk_count"],
        "streamed_observation_count": counters["streamed_observation_count"],
        "sample_launch_count": counters["sample_launch_count"],
        "sample_node_interaction_count": counters[
            "sample_node_interaction_count"
        ],
        "transferred_target_payload_bytes": counters[
            "transferred_target_payload_bytes"
        ],
        "native_material_vjp_launch_count": counters[
            "native_material_vjp_launch_count"
        ],
        "native_full_geometry_vjp_launch_count": 0,
        "full_geometry": False,
        "geometry_bar_tensor_bytes": 0,
        "parameter_mutation_count": 0,
        "optimizer_step_executed": False,
        "coordinator_completion_semantics": (
            "authorization_only_external_optimizer_apply_required"
        ),
        "built_in_bounded_target_decoder": True,
        "arbitrary_external_target_loader": False,
        "full_dense_observation_replay": True,
        "sample_and_target_payloads_streamed": True,
        "target_source_decode_budget_enforced_before_allocation": True,
        "canonical_partition_order": "view_major_then_contiguous_pixel_interval",
        "maximum_tracks_per_request": policy.maximum_tracks_per_request,
        "maximum_samples_per_launch": policy.maximum_samples_per_launch,
        "maximum_target_observations_per_chunk": (
            source.effective_chunk_observation_capacity
        ),
        "maximum_sample_materialization_logical_tensor_bytes": (
            policy.request_memory_policy.maximum_sample_materialization_logical_tensor_bytes
        ),
        "sample_materialization_source_visible_logical_tensors_accounted": True,
        "sample_materialization_float64_scratch_measured": False,
        "world_site_count": source.provider.world.site_count,
        "persistent_world_geometry_tensor_bytes": (
            source.provider.world.sites.parameter_bytes
        ),
        "target_source_resident_tensor_bytes": (
            target_source_resident_tensor_bytes
        ),
        "target_source_kind": str(
            target_residency.get("source_kind", "unknown")
        ),
        "target_source_manifest_sha256": str(
            target_residency.get("manifest_sha256", "")
        ),
        "target_source_logical_frame_map_sha256": str(
            target_residency.get("logical_frame_map_sha256", "")
        ),
        "target_source_raw_storage_bytes": target_source_raw_storage_bytes,
        "target_source_maximum_mapped_payload_bytes": (
            target_source_maximum_mapped_payload_bytes
        ),
        "target_source_maximum_total_payload_verification_bytes": (
            target_source_maximum_total_payload_verification_bytes
        ),
        "target_source_construction_payload_verification_bytes": (
            target_source_construction_verification_bytes
        ),
        "target_source_construction_full_payload_scan": bool(
            target_residency.get("construction_full_payload_scan", False)
        ),
        "target_source_requested_page_coverage_is_not_residency_measurement": bool(
            target_residency.get(
                "requested_page_coverage_is_not_residency_measurement",
                False,
            )
        ),
        "target_source_system_page_cache_peak_measured": bool(
            target_residency.get("system_page_cache_peak_measured", False)
        ),
        "material_state_logical_tensor_bytes": (
            12 * source.provider.world.site_count * 4
        ),
        "material_checkpoint_logical_tensor_bytes": (
            4 * source.provider.world.site_count * 4
        ),
        "material_step_accumulator_preflight_logical_tensor_bytes": (
            (4 * source.provider.world.site_count + 1) * 4
        ),
        "maximum_world_site_count": policy.maximum_world_site_count,
        "maximum_material_state_logical_tensor_bytes": (
            policy.maximum_material_state_logical_tensor_bytes
        ),
        "maximum_material_checkpoint_logical_tensor_bytes": (
            policy.maximum_material_checkpoint_logical_tensor_bytes
        ),
        "maximum_step_accumulator_logical_tensor_bytes": (
            policy.maximum_step_accumulator_logical_tensor_bytes
        ),
        "maximum_artifact_accounted_bytes": (
            policy.maximum_artifact_accounted_bytes
        ),
        "artifact_store_lookup_count": counters["request_count"],
        "artifact_store_cold_compile_count": counters["cold_artifact_count"],
        "artifact_store_warm_hit_count": counters["warm_artifact_count"],
        "artifact_store_eviction_count": counters[
            "artifact_store_eviction_count"
        ],
        "artifact_store_evicted_accounted_bytes": counters[
            "artifact_store_evicted_accounted_bytes"
        ],
        "artifact_store_cold_compiled_track_count": counters[
            "artifact_store_cold_compiled_track_count"
        ],
        "artifact_store_avoided_compile_track_count": counters[
            "artifact_store_avoided_compile_track_count"
        ],
        "artifact_store_step_metrics_derived_from_acquisition_receipts": True,
        "canonical_artifact_working_set_count": replay_receipt.request_count,
        "artifact_store_maximum_entry_count": store_after.maximum_entries,
        "artifact_working_set_fits_entry_bound": (
            replay_receipt.request_count <= store_after.maximum_entries
        ),
        "artifact_working_set_conservatively_fits_byte_bound": (
            replay_receipt.request_count * policy.maximum_artifact_accounted_bytes
            <= store_after.maximum_resident_accounted_bytes
        ),
        "artifact_store_current_entry_count": store_after.current_entry_count,
        "artifact_store_current_resident_accounted_bytes": (
            store_after.current_resident_accounted_bytes
        ),
        "artifact_store_peak_resident_accounted_bytes": (
            store_after.peak_resident_accounted_bytes
        ),
        "cold_artifact_acquisition_count": counters["cold_artifact_count"],
        "warm_artifact_acquisition_count": counters["warm_artifact_count"],
        "request_commit_fence_call_count": counters[
            "request_commit_fence_call_count"
        ],
        "accumulator_initialization_fence_call_count": counters[
            "accumulator_initialization_fence_call_count"
        ],
        "native_lane_fence_call_count": counters["native_lane_fence_call_count"],
        "total_step_completion_fence_call_count": (
            counters["accumulator_initialization_fence_call_count"]
            + counters["request_commit_fence_call_count"]
            + counters["native_lane_fence_call_count"]
        ),
        "device_completion_fence_provenance": (
            device_completion_fence_provenance
        ),
        "backend_provenance": backend_provenance,
        "device_type": accumulator.grad_site_rgba_f32.device.type,
        "selected_pixel_read_mode": selected_pixel_read_mode,
        "selected_pixel_read_acceptance_capable": (
            selected_pixel_read_acceptance_capable
        ),
        "selected_pixel_read_call_count": counters[
            "selected_pixel_read_call_count"
        ],
        "mapped_selected_pixel_read_call_count": counters[
            "mapped_selected_pixel_read_call_count"
        ],
        "mapping_closed_before_return_count": counters[
            "mapping_closed_before_return_count"
        ],
        "cumulative_requested_mapped_page_count": counters[
            "cumulative_requested_mapped_page_count"
        ],
        "cumulative_requested_mapped_page_bytes_upper_bound": counters[
            "cumulative_requested_mapped_page_bytes_upper_bound"
        ],
        "all_selected_pixel_mappings_closed_before_return": (
            counters["mapping_closed_before_return_count"]
            == counters["mapped_selected_pixel_read_call_count"]
        ),
        "direct_selected_pixel_observation_count": counters[
            "direct_selected_pixel_observation_count"
        ],
        "bounded_region_selected_pixel_observation_count": counters[
            "bounded_region_selected_pixel_observation_count"
        ],
        "full_frame_fallback_observation_count": counters[
            "full_frame_fallback_observation_count"
        ],
        "full_frame_target_materialization_count": counters[
            "full_frame_target_materialization_count"
        ],
        "bounded_region_target_materialization_count": counters[
            "bounded_region_target_materialization_count"
        ],
        "decoded_frame_count": counters["decoded_frame_count"],
        "source_retained_frame_metadata_logical_bytes": (
            source.retained_frame_metadata_logical_bytes
        ),
        "step_accumulator_logical_tensor_bytes": accumulator.logical_tensor_bytes,
        "peak_ray_payload_logical_tensor_bytes": 0,
        **dict(peaks),
        "maximum_simultaneously_decoded_target_frame_count": (
            1 if counters["decoded_frame_count"] > 0 else 0
        ),
        "persistent_frame_tensor_bytes": 0,
        "persistent_sample_tensor_bytes": 0,
        "persistent_target_tensor_bytes": 0,
        "persistent_prediction_tensor_bytes": 0,
        "step_accumulator_retains_frame_axis": False,
        "reachable_autograd_tensor_count": 0,
        "autograd_graph_retained": False,
        "autograd_saved_tensor_peak_measured": False,
        "authorization_capability_objects_retained": 3,
        "source_retained_by_result": False,
        "session_retained_by_result": False,
        "request_retained_by_result": False,
        "artifact_retained_by_result": False,
        "target_retained_by_result": False,
        "native_lane_retained_by_result": False,
        "native_runtime_verified": False,
        "allocator_peak_measured": False,
        "artifact_compile_scratch_budget_enforced": False,
        "artifact_compile_scratch_peak_measured": False,
        "whole_step_python_object_peak_measured": False,
        "structural_node_word_work_invariance_requires_cross_row_verification": True,
        "full_video_target_tensor_retained": full_video_target_tensor_retained,
        **structure_report,
    }


def _retain_failed_step(
    state: PaperKineticFixedSiteMaterialStepState,
    source: PaperKineticReplayableDenseObservationSource,
    session: PaperKineticDenseObservationReplaySession,
    accumulator: PaperKineticDenseStepGradientAccumulator,
    error: BaseException,
    *,
    lifetime_roots: tuple[tuple[str, Any], ...],
    attempt_fail_stop: bool,
) -> None:
    fail_stop_completed = False
    if not attempt_fail_stop:
        error.add_note(
            "accumulator completion fence failed; skipped all tensor-mutating "
            "whole-step fail-stop work and retained exact lifetime roots"
        )
    else:
        try:
            fail_stop_paper_kinetic_dense_step(accumulator, source, session)
        except BaseException as fail_stop_error:
            error.add_note(
                "dense whole-step fail-stop also failed; retained state still "
                "requires process restart: "
                f"{type(fail_stop_error).__qualname__}: {fail_stop_error}"
            )
        else:
            fail_stop_completed = True
    state.active_step_generation_id = ""
    state.poisoned = True
    state.restart_required = True
    state.failure_type = type(error).__qualname__
    state.failure_message = str(error) or type(error).__qualname__
    state.failure_fail_stop_completed = fail_stop_completed
    retained = tuple(
        (role, value) for role, value in lifetime_roots if value is not None
    )
    state.failure_lifetime_root_roles = tuple(role for role, _ in retained)
    state._failed_lifetime_roots = tuple(value for _, value in retained)
    state._failed_source = source
    state._failed_session = session
    state._failed_accumulator = accumulator


def _require_f32_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, ...],
    device: torch.device,
) -> None:
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.dtype != torch.float32
        or tensor.device != device
        or tuple(tensor.shape) != shape
        or not tensor.is_contiguous()
    ):
        raise ValueError(
            f"{name} must be contiguous float32 {shape} on {device}"
        )


def _same_storage(first: torch.Tensor, second: torch.Tensor) -> bool:
    return first.untyped_storage().data_ptr() == second.untyped_storage().data_ptr()


def _tensor_signature(tensor: torch.Tensor) -> tuple[object, ...]:
    return (
        id(tensor),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
        bool(tensor.requires_grad),
        int(tensor._version),
    )


def _result_digest(result: PaperKineticFixedSiteMaterialStepResult) -> str:
    return _digest_parts(
        STEP_PROVENANCE,
        result.runtime_status,
        result.authorization.generation_digest,
        result.accumulator.generation_digest,
        result.replay_receipt.generation_digest,
        result.loss_rgb_mean,
        tuple(result.accounting.items()),
        result.parameter_mutation_count,
        result.retained_authorization_capability_object_count,
        result.retained_source_count,
        result.retained_session_count,
        result.retained_request_count,
        result.retained_artifact_count,
        result.retained_target_count,
        result.retained_native_lane_count,
        result.native_runtime_verified,
        result.allocator_peak_measured,
    )


def _digest_parts(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = repr(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _require_positive_int(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


__all__ = [
    "GLOBAL_RGB_MEAN_LOSS_NORMALIZATION_ID",
    "STEP_PROVENANCE",
    "STEP_STATUS",
    "PaperKineticFixedSiteMaterialOnlyGenerationPolicy",
    "PaperKineticFixedSiteMaterialOnlyStepPolicy",
    "PaperKineticFixedSiteMaterialStepPartialFailure",
    "PaperKineticFixedSiteMaterialStepResult",
    "PaperKineticFixedSiteMaterialStepState",
    "prepare_paper_kinetic_fixed_site_material_step_state",
    "run_paper_kinetic_fixed_site_material_only_step",
]
