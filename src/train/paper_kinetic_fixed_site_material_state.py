"""Fixed-site CPU material state and manual SGD for native-4D WorldFoam.

This module consumes a sealed ``PaperKineticDenseOptimizerAuthorization`` only
for the material-only lane.  Geometry, kinetic weights, rays, and compiled
programs remain immutable.  Persistent device state is exactly:

* raw sigmoid RGB ``[S,3]``;
* raw softplus density ``[S]``;
* physical ``[R,G,B,density]`` snapshot ``[S,4]`` used by native replay; and
* preallocated raw RGB/density gradient buffers of the same shapes.

There is no frame, sample, target, prediction, or optimizer-history tensor.
The optimizer is explicit SGD without momentum, so a restart checkpoint needs
only the two raw parameter tensors plus scalar policy/provenance.  Physical
material and scratch buffers are deterministically reconstructed.

Authorization is revalidated before any buffer or parameter mutation.  The
authorization must reference the exact live physical snapshot and its declared
material generation.  A successful update changes that snapshot, making the
point-in-time authorization stale by construction.

This first lifecycle is deliberately CPU-only. Its invariant checks use
explicit scalar reductions and it does not yet own an accelerator completion
fence/quarantine contract. Native replay may produce CPU fake-native bars for
this source gate; real accelerator material updates require a separately
fenced version rather than silently admitting asynchronous device tensors.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import torch
from paper_kinetic_runtime_paths import ensure_worldfoam_lane2_research_path

ensure_worldfoam_lane2_research_path()

from kinetic_dense_cached_native_material_request import (  # noqa: E402
    PaperKineticDenseOptimizerAuthorization,
    PaperKineticDenseStepGradientAccumulator,
)
from material_parameterization import (  # noqa: E402
    WorldFoamMaterialParameterization,
)
from paper_kinetic_world_initializer import (  # noqa: E402
    PaperKineticP0MaterialInitialization,
)
from paper_kinetic_lazy_program_bundles import (  # noqa: E402
    PaperKineticWorldSnapshot,
)
from paper_kinetic_replayable_observations import (  # noqa: E402
    PaperKineticDenseObservationReplayReceipt,
)


STATE_PROVENANCE = "paper-kinetic-fixed-site-material-state-v1"
STEP_RECEIPT_PROVENANCE = "paper-kinetic-fixed-site-material-sgd-step-v1"
CHECKPOINT_PROVENANCE = "paper-kinetic-fixed-site-material-checkpoint-v1"
CHECKPOINT_SCHEMA = "paper_kinetic_fixed_site_material_checkpoint_v1"
FLOAT32_LOGICAL_BYTES = 4

_STATE_SEAL = object()
_RECEIPT_SEAL = object()
_CHECKPOINT_SEAL = object()


@dataclass(frozen=True)
class PaperKineticFixedSiteMaterialParameterization:
    density_beta: float = 1.0
    density_threshold: float = 20.0
    minimum_density: float = 0.0
    color_epsilon: float = 1.0e-4

    def assert_valid(self) -> None:
        self.runtime_parameterization.assert_valid()
        if (
            not math.isfinite(self.color_epsilon)
            or not 0.0 < self.color_epsilon < 0.5
        ):
            raise ValueError("fixed-site material parameterization is invalid")

    @property
    def runtime_parameterization(self) -> WorldFoamMaterialParameterization:
        return WorldFoamMaterialParameterization(
            density_beta=self.density_beta,
            density_threshold=self.density_threshold,
            minimum_density=self.minimum_density,
        )

    @property
    def generation_digest(self) -> str:
        self.assert_valid()
        return _digest_parts(
            STATE_PROVENANCE,
            "parameterization",
            self.density_beta,
            self.density_threshold,
            self.minimum_density,
            self.color_epsilon,
        )

    def payload(self) -> dict[str, float]:
        self.assert_valid()
        return {
            "density_beta": self.density_beta,
            "density_threshold": self.density_threshold,
            "minimum_density": self.minimum_density,
            "color_epsilon": self.color_epsilon,
        }


@dataclass(frozen=True)
class PaperKineticFixedSiteMaterialSGDPolicy:
    color_learning_rate: float
    density_learning_rate: float
    maximum_absolute_raw_color_value: float = 15.0
    # The production initializer currently defaults to physical density 64.
    # Softplus is linear there, so the raw guard must admit that honest seed.
    maximum_absolute_raw_density_value: float = 128.0

    def assert_valid(self) -> None:
        if (
            not math.isfinite(self.color_learning_rate)
            or self.color_learning_rate <= 0.0
            or not math.isfinite(self.density_learning_rate)
            or self.density_learning_rate <= 0.0
            or not math.isfinite(self.maximum_absolute_raw_color_value)
            or self.maximum_absolute_raw_color_value <= 0.0
            or not math.isfinite(self.maximum_absolute_raw_density_value)
            or self.maximum_absolute_raw_density_value <= 0.0
        ):
            raise ValueError("fixed-site material SGD policy is invalid")

    @property
    def generation_digest(self) -> str:
        self.assert_valid()
        return _digest_parts(
            STATE_PROVENANCE,
            "manual-sgd-policy",
            self.color_learning_rate,
            self.density_learning_rate,
            self.maximum_absolute_raw_color_value,
            self.maximum_absolute_raw_density_value,
            "no_momentum",
            "no_weight_decay",
        )

    def payload(self) -> dict[str, float | str]:
        self.assert_valid()
        return {
            "optimizer": "manual_sgd",
            "color_learning_rate": self.color_learning_rate,
            "density_learning_rate": self.density_learning_rate,
            "maximum_absolute_raw_color_value": (
                self.maximum_absolute_raw_color_value
            ),
            "maximum_absolute_raw_density_value": (
                self.maximum_absolute_raw_density_value
            ),
            "momentum": 0.0,
            "weight_decay": 0.0,
        }


@dataclass
class PaperKineticFixedSiteMaterialState:
    """Mutable O(S) live material state; geometry is intentionally absent."""

    world_generation_digest: str
    sites_content_digest: str
    p0_material_seed_generation_digest: str
    parameterization: PaperKineticFixedSiteMaterialParameterization
    optimizer_policy: PaperKineticFixedSiteMaterialSGDPolicy
    raw_color_f32: torch.Tensor = field(repr=False)
    raw_density_f32: torch.Tensor = field(repr=False)
    site_rgba_f32: torch.Tensor = field(repr=False)
    raw_color_grad_f32: torch.Tensor = field(repr=False)
    raw_density_grad_f32: torch.Tensor = field(repr=False)
    initialization_content_digest: str
    generation_parent_digest: str
    last_authorization_generation_digest: str
    last_step_generation_id: str
    step_index: int
    material_generation_id: str
    restart_checkpoint_generation_digest: str
    tensor_signatures: tuple[tuple[object, ...], ...] = field(repr=False)
    poisoned: bool
    provenance: str = STATE_PROVENANCE
    persistent_frame_tensor_bytes: int = 0
    persistent_sample_tensor_bytes: int = 0
    persistent_target_tensor_bytes: int = 0
    persistent_prediction_tensor_bytes: int = 0
    geometry_trainable: bool = False
    _seal: object = field(default=None, repr=False)

    @property
    def site_count(self) -> int:
        return int(self.raw_density_f32.numel())

    @property
    def device(self) -> torch.device:
        return self.site_rgba_f32.device

    def _tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.raw_color_f32,
            self.raw_density_f32,
            self.site_rgba_f32,
            self.raw_color_grad_f32,
            self.raw_density_grad_f32,
        )

    @property
    def persistent_parameter_tensor_bytes(self) -> int:
        return _tensor_bytes(self.raw_color_f32, self.raw_density_f32)

    @property
    def persistent_physical_snapshot_tensor_bytes(self) -> int:
        return _tensor_bytes(self.site_rgba_f32)

    @property
    def persistent_raw_gradient_buffer_tensor_bytes(self) -> int:
        return _tensor_bytes(self.raw_color_grad_f32, self.raw_density_grad_f32)

    @property
    def total_persistent_tensor_bytes(self) -> int:
        return _tensor_bytes(*self._tensors())

    def assert_current(self) -> None:
        self.parameterization.assert_valid()
        self.optimizer_policy.assert_valid()
        if (
            self._seal is not _STATE_SEAL
            or self.provenance != STATE_PROVENANCE
            or self.poisoned
            or self.geometry_trainable
            or self.site_count < 1
            or isinstance(self.step_index, bool)
            or not isinstance(self.step_index, int)
            or self.step_index < 0
            or self.persistent_frame_tensor_bytes != 0
            or self.persistent_sample_tensor_bytes != 0
            or self.persistent_target_tensor_bytes != 0
            or self.persistent_prediction_tensor_bytes != 0
        ):
            raise ValueError("fixed-site material state seal/mode changed or is poisoned")
        for name, value in (
            ("world_generation_digest", self.world_generation_digest),
            ("sites_content_digest", self.sites_content_digest),
            (
                "p0_material_seed_generation_digest",
                self.p0_material_seed_generation_digest,
            ),
            ("initialization_content_digest", self.initialization_content_digest),
            ("material_generation_id", self.material_generation_id),
        ):
            _require_sha256(value, name=name)
        if self.generation_parent_digest:
            _require_sha256(
                self.generation_parent_digest,
                name="generation_parent_digest",
            )
        if bool(self.generation_parent_digest) != bool(self.step_index):
            raise ValueError("fixed-site material parent history changed")
        if self.last_authorization_generation_digest:
            _require_sha256(
                self.last_authorization_generation_digest,
                name="last_authorization_generation_digest",
            )
        if bool(self.last_authorization_generation_digest) != bool(self.step_index):
            raise ValueError("fixed-site material step/authorization history changed")
        if (
            not isinstance(self.last_step_generation_id, str)
            or bool(self.last_step_generation_id) != bool(self.step_index)
        ):
            raise ValueError("fixed-site material step identity history changed")
        if self.restart_checkpoint_generation_digest:
            _require_sha256(
                self.restart_checkpoint_generation_digest,
                name="restart_checkpoint_generation_digest",
            )
        _validate_state_tensors(self)
        if tuple(_tensor_signature(tensor) for tensor in self._tensors()) != self.tensor_signatures:
            raise ValueError("fixed-site material tensor identity/content version changed")
        if self.material_generation_id != _state_generation_digest(self):
            raise ValueError("fixed-site material generation changed")

    def accounting(self, *, requested_frame_count: int) -> dict[str, Any]:
        _require_positive_int(requested_frame_count, name="requested_frame_count")
        self.assert_current()
        return {
            "provenance": self.provenance,
            "material_generation_id": self.material_generation_id,
            "material_generation_semantics": (
                "live_version_chain_with_tensor_identity_and_version_guard; "
                "numeric_content_is_bound_at_checkpoint"
            ),
            "world_generation_digest": self.world_generation_digest,
            "sites_content_digest": self.sites_content_digest,
            "site_count": self.site_count,
            "step_index": self.step_index,
            "optimizer": "manual_sgd",
            "optimizer_history_tensor_bytes": 0,
            "persistent_parameter_tensor_bytes": self.persistent_parameter_tensor_bytes,
            "persistent_physical_snapshot_tensor_bytes": (
                self.persistent_physical_snapshot_tensor_bytes
            ),
            "persistent_raw_gradient_buffer_tensor_bytes": (
                self.persistent_raw_gradient_buffer_tensor_bytes
            ),
            "total_persistent_tensor_bytes": self.total_persistent_tensor_bytes,
            "persistent_scalar_count_per_site": 12,
            "requested_frame_count": requested_frame_count,
            "frame_dependent_parameter_bytes": 0,
            "persistent_frame_tensor_bytes": 0,
            "persistent_sample_tensor_bytes": 0,
            "persistent_target_tensor_bytes": 0,
            "persistent_prediction_tensor_bytes": 0,
            "material_layout": "rgb_then_density",
            "material_temporal_basis": "P0",
            "raw_color_parameterization": "sigmoid",
            "raw_density_parameterization": "thresholded_softplus_plus_minimum",
            "manual_parameter_chain_rule": True,
            "color_seed_clamped_to_finite_logit": True,
            "geometry_trainable": False,
            "material_state_device_scope": (
                "cpu_only_until_fenced_optimizer_update"
            ),
            "accelerator_optimizer_update_supported": False,
            "restart_checkpoint_generation_digest": (
                self.restart_checkpoint_generation_digest
            ),
            "allocator_peak_measured": False,
            "bounded_elementwise_transient_tensor_bytes_measured": False,
        }


@dataclass(frozen=True)
class PaperKineticFixedSiteMaterialStepReceipt:
    step_index: int
    step_generation_id: str
    authorization_generation_digest: str
    material_generation_id_before: str
    material_generation_id_after: str
    loss: float
    raw_color_gradient_norm: float
    raw_density_gradient_norm: float
    generation_digest: str
    provenance: str = STEP_RECEIPT_PROVENANCE
    geometry_updated: bool = False
    persistent_tensor_bytes: int = 0
    _seal: object = field(default=None, repr=False)

    def assert_current(self, state: PaperKineticFixedSiteMaterialState) -> None:
        state.assert_current()
        _require_sha256(
            self.authorization_generation_digest,
            name="authorization_generation_digest",
        )
        _require_sha256(
            self.material_generation_id_before,
            name="material_generation_id_before",
        )
        _require_sha256(
            self.material_generation_id_after,
            name="material_generation_id_after",
        )
        if (
            self._seal is not _RECEIPT_SEAL
            or self.provenance != STEP_RECEIPT_PROVENANCE
            or self.step_index != state.step_index
            or self.step_index < 1
            or not isinstance(self.step_generation_id, str)
            or not self.step_generation_id.strip()
            or self.step_generation_id != state.last_step_generation_id
            or self.material_generation_id_before
            != state.generation_parent_digest
            or self.material_generation_id_after != state.material_generation_id
            or self.authorization_generation_digest
            != state.last_authorization_generation_digest
            or self.geometry_updated
            or self.persistent_tensor_bytes != 0
            or not all(
                math.isfinite(value)
                for value in (
                    self.loss,
                    self.raw_color_gradient_norm,
                    self.raw_density_gradient_norm,
                )
            )
            or self.generation_digest != _step_receipt_digest(self)
        ):
            raise ValueError("fixed-site material step receipt changed")


@dataclass(frozen=True)
class PaperKineticFixedSiteMaterialCheckpoint:
    world_generation_digest: str
    sites_content_digest: str
    p0_material_seed_generation_digest: str
    parameterization: PaperKineticFixedSiteMaterialParameterization
    optimizer_policy: PaperKineticFixedSiteMaterialSGDPolicy
    raw_color_f32_cpu: torch.Tensor = field(repr=False)
    raw_density_f32_cpu: torch.Tensor = field(repr=False)
    initialization_content_digest: str
    generation_parent_digest: str
    last_authorization_generation_digest: str
    last_step_generation_id: str
    step_index: int
    material_generation_id: str
    raw_color_content_digest: str
    raw_density_content_digest: str
    generation_digest: str
    provenance: str = CHECKPOINT_PROVENANCE
    schema: str = CHECKPOINT_SCHEMA
    persistent_frame_tensor_bytes: int = 0
    persistent_sample_tensor_bytes: int = 0
    persistent_target_tensor_bytes: int = 0
    persistent_prediction_tensor_bytes: int = 0
    optimizer_history_tensor_bytes: int = 0
    _seal: object = field(default=None, repr=False)

    @property
    def site_count(self) -> int:
        return int(self.raw_density_f32_cpu.numel())

    @property
    def checkpoint_tensor_bytes(self) -> int:
        return _tensor_bytes(self.raw_color_f32_cpu, self.raw_density_f32_cpu)

    def assert_current(self) -> None:
        self.parameterization.assert_valid()
        self.optimizer_policy.assert_valid()
        if (
            self._seal is not _CHECKPOINT_SEAL
            or self.provenance != CHECKPOINT_PROVENANCE
            or self.schema != CHECKPOINT_SCHEMA
            or self.site_count < 1
            or isinstance(self.step_index, bool)
            or not isinstance(self.step_index, int)
            or self.step_index < 0
            or self.persistent_frame_tensor_bytes != 0
            or self.persistent_sample_tensor_bytes != 0
            or self.persistent_target_tensor_bytes != 0
            or self.persistent_prediction_tensor_bytes != 0
            or self.optimizer_history_tensor_bytes != 0
        ):
            raise ValueError("fixed-site material checkpoint seal/schema changed")
        for name, value in (
            ("world_generation_digest", self.world_generation_digest),
            ("sites_content_digest", self.sites_content_digest),
            (
                "p0_material_seed_generation_digest",
                self.p0_material_seed_generation_digest,
            ),
            ("initialization_content_digest", self.initialization_content_digest),
            ("material_generation_id", self.material_generation_id),
            ("raw_color_content_digest", self.raw_color_content_digest),
            ("raw_density_content_digest", self.raw_density_content_digest),
        ):
            _require_sha256(value, name=name)
        if self.generation_parent_digest:
            _require_sha256(
                self.generation_parent_digest,
                name="generation_parent_digest",
            )
        if bool(self.generation_parent_digest) != bool(self.step_index):
            raise ValueError("fixed-site material checkpoint parent history changed")
        if self.last_authorization_generation_digest:
            _require_sha256(
                self.last_authorization_generation_digest,
                name="last_authorization_generation_digest",
            )
        if bool(self.last_authorization_generation_digest) != bool(self.step_index):
            raise ValueError("fixed-site material checkpoint authorization history changed")
        if (
            not isinstance(self.last_step_generation_id, str)
            or bool(self.last_step_generation_id) != bool(self.step_index)
        ):
            raise ValueError("fixed-site material checkpoint step history changed")
        _require_cpu_f32(
            self.raw_color_f32_cpu,
            name="checkpoint raw_color_f32_cpu",
            shape=(self.site_count, 3),
        )
        _require_cpu_f32(
            self.raw_density_f32_cpu,
            name="checkpoint raw_density_f32_cpu",
            shape=(self.site_count,),
        )
        _require_distinct_storage(self.raw_color_f32_cpu, self.raw_density_f32_cpu)
        if bool(
            torch.any(
                self.raw_color_f32_cpu.abs()
                > self.optimizer_policy.maximum_absolute_raw_color_value
            ).item()
        ) or bool(
            torch.any(
                self.raw_density_f32_cpu.abs()
                > self.optimizer_policy.maximum_absolute_raw_density_value
            ).item()
        ):
            raise ValueError("fixed-site material checkpoint exceeds raw-value bound")
        if (
            _tensor_content_digest(self.raw_color_f32_cpu)
            != self.raw_color_content_digest
            or _tensor_content_digest(self.raw_density_f32_cpu)
            != self.raw_density_content_digest
            or self.material_generation_id
            != _material_generation_digest_from_fields(
                world_generation_digest=self.world_generation_digest,
                sites_content_digest=self.sites_content_digest,
                p0_material_seed_generation_digest=(
                    self.p0_material_seed_generation_digest
                ),
                parameterization=self.parameterization,
                optimizer_policy=self.optimizer_policy,
                initialization_content_digest=self.initialization_content_digest,
                generation_parent_digest=self.generation_parent_digest,
                last_authorization_generation_digest=(
                    self.last_authorization_generation_digest
                ),
                last_step_generation_id=self.last_step_generation_id,
                step_index=self.step_index,
            )
            or self.generation_digest != _checkpoint_digest(self)
        ):
            raise ValueError("fixed-site material checkpoint tensor/generation changed")

    def payload(self) -> dict[str, Any]:
        self.assert_current()
        return {
            "schema": self.schema,
            "provenance": self.provenance,
            "world_generation_digest": self.world_generation_digest,
            "sites_content_digest": self.sites_content_digest,
            "p0_material_seed_generation_digest": (
                self.p0_material_seed_generation_digest
            ),
            "parameterization": self.parameterization.payload(),
            "optimizer_policy": self.optimizer_policy.payload(),
            "raw_color_f32_cpu": self.raw_color_f32_cpu.clone(),
            "raw_density_f32_cpu": self.raw_density_f32_cpu.clone(),
            "initialization_content_digest": self.initialization_content_digest,
            "generation_parent_digest": self.generation_parent_digest,
            "last_authorization_generation_digest": (
                self.last_authorization_generation_digest
            ),
            "last_step_generation_id": self.last_step_generation_id,
            "step_index": self.step_index,
            "material_generation_id": self.material_generation_id,
            "raw_color_content_digest": self.raw_color_content_digest,
            "raw_density_content_digest": self.raw_density_content_digest,
            "generation_digest": self.generation_digest,
            "persistent_frame_tensor_bytes": 0,
            "persistent_sample_tensor_bytes": 0,
            "persistent_target_tensor_bytes": 0,
            "persistent_prediction_tensor_bytes": 0,
            "optimizer_history_tensor_bytes": 0,
        }


@torch.no_grad()
def prepare_paper_kinetic_fixed_site_material_state(
    initialization: PaperKineticP0MaterialInitialization,
    world: PaperKineticWorldSnapshot,
    *,
    parameterization: PaperKineticFixedSiteMaterialParameterization,
    optimizer_policy: PaperKineticFixedSiteMaterialSGDPolicy,
    device: torch.device | str,
    maximum_material_state_logical_tensor_bytes: int,
) -> PaperKineticFixedSiteMaterialState:
    """Create fixed-site raw parameters and reusable physical/scratch buffers."""

    if not isinstance(initialization, PaperKineticP0MaterialInitialization):
        raise TypeError("fixed-site material state requires P0 material initialization")
    if not isinstance(world, PaperKineticWorldSnapshot):
        raise TypeError("fixed-site material state requires a sealed world snapshot")
    resolved_device = torch.device(device)
    if resolved_device.type != "cpu":
        raise NotImplementedError(
            "fixed-site material state is CPU-only until optimizer updates "
            "own a completion fence"
        )
    parameterization.assert_valid()
    optimizer_policy.assert_valid()
    _preflight_material_state_logical_bytes(
        initialization.site_count,
        maximum_material_state_logical_tensor_bytes=(
            maximum_material_state_logical_tensor_bytes
        ),
    )
    world.assert_current()
    initialization.assert_current(world.sites)
    physical_seed = initialization.site_rgba_f32.to(
        device=resolved_device,
        dtype=torch.float32,
    )
    color_seed = physical_seed[:, :3].clamp(
        parameterization.color_epsilon,
        1.0 - parameterization.color_epsilon,
    )
    raw_color = (torch.log(color_seed) - torch.log1p(-color_seed)).contiguous()
    raw_density = parameterization.runtime_parameterization.encode_density(
        physical_seed[:, 3]
    )
    if bool(
        torch.any(
            raw_color.abs() > optimizer_policy.maximum_absolute_raw_color_value
        ).item()
    ) or bool(
        torch.any(
            raw_density.abs() > optimizer_policy.maximum_absolute_raw_density_value
        ).item()
    ):
        raise ValueError("initial raw material exceeds the SGD raw-value bound")
    site_rgba = torch.empty(
        (initialization.site_count, 4),
        device=resolved_device,
        dtype=torch.float32,
    )
    raw_color_grad = torch.zeros_like(raw_color)
    raw_density_grad = torch.zeros_like(raw_density)
    _decode_physical_(
        site_rgba,
        raw_color,
        raw_density,
        parameterization=parameterization,
    )
    if not bool(
        torch.allclose(
            site_rgba[:, 3],
            physical_seed[:, 3],
            rtol=2.0e-6,
            atol=0.0,
        )
    ):
        raise ValueError("initial density does not round-trip through its parameterization")
    initialization_content_digest = _digest_parts(
        STATE_PROVENANCE,
        "initial-raw-content",
        _tensor_content_digest(raw_color),
        _tensor_content_digest(raw_density),
    )
    provisional = PaperKineticFixedSiteMaterialState(
        world_generation_digest=world.generation_digest,
        # The initializer uses its own content-digest namespace to bind a P0
        # seed to raw site tensors.  The live optimizer/coordinator contract,
        # however, must carry the provider world's canonical digest: that is
        # the identity emitted by geometry authorizations and checkpoints.
        # ``initialization.assert_current(world.sites)`` above already proves
        # that the two differently-namespaced digests describe the same sites.
        sites_content_digest=world.sites_content_digest,
        p0_material_seed_generation_digest=(
            initialization.material_seed_generation_digest
        ),
        parameterization=parameterization,
        optimizer_policy=optimizer_policy,
        raw_color_f32=raw_color,
        raw_density_f32=raw_density,
        site_rgba_f32=site_rgba,
        raw_color_grad_f32=raw_color_grad,
        raw_density_grad_f32=raw_density_grad,
        initialization_content_digest=initialization_content_digest,
        generation_parent_digest="",
        last_authorization_generation_digest="",
        last_step_generation_id="",
        step_index=0,
        material_generation_id="",
        restart_checkpoint_generation_digest="",
        tensor_signatures=(),
        poisoned=False,
        _seal=_STATE_SEAL,
    )
    provisional.material_generation_id = _state_generation_digest(provisional)
    provisional.tensor_signatures = tuple(
        _tensor_signature(tensor) for tensor in provisional._tensors()
    )
    provisional.assert_current()
    return provisional


@torch.no_grad()
def apply_paper_kinetic_fixed_site_material_sgd_step(
    state: PaperKineticFixedSiteMaterialState,
    authorization: PaperKineticDenseOptimizerAuthorization,
    accumulator: PaperKineticDenseStepGradientAccumulator,
    replay_receipt: PaperKineticDenseObservationReplayReceipt,
) -> PaperKineticFixedSiteMaterialStepReceipt:
    """Validate one complete material authorization, then mutate once."""

    if not isinstance(state, PaperKineticFixedSiteMaterialState):
        raise TypeError("material SGD step requires fixed-site material state")
    if not isinstance(authorization, PaperKineticDenseOptimizerAuthorization):
        raise TypeError("material SGD step requires PaperKineticDenseOptimizerAuthorization")
    state.assert_current()

    # This is deliberately first: point-in-time accumulator/material/background
    # provenance must be proven before mode checks or scratch mutation.
    authorization.assert_current(accumulator, replay_receipt)
    if (
        authorization.full_geometry
        or accumulator.full_geometry
        or authorization.ray_bar_keys
        or authorization.grad_positions0_f64 is not None
        or authorization.grad_velocities_f64 is not None
        or authorization.grad_weight_coefficients_f64 is not None
        or authorization.grad_track_ray_coefficients_f64 is not None
    ):
        raise ValueError("fixed-site material SGD rejects geometry authorizations")
    if accumulator._material_tensor_ref is not state.site_rgba_f32:
        raise ValueError("optimizer authorization does not reference the live material snapshot")
    if accumulator.material_generation_id != state.material_generation_id:
        raise ValueError("optimizer authorization names a stale material generation")
    if (
        accumulator.world_generation_digest != state.world_generation_digest
        or accumulator.world_sites_content_digest != state.sites_content_digest
    ):
        raise ValueError("optimizer authorization belongs to a different world snapshot")
    if (
        authorization.generation_digest
        == state.last_authorization_generation_digest
    ):
        raise ValueError("optimizer authorization was already consumed by this state")
    _require_sha256(
        authorization.generation_digest,
        name="authorization generation_digest",
    )
    if (
        not isinstance(authorization.step_generation_id, str)
        or not authorization.step_generation_id.strip()
    ):
        raise ValueError("optimizer authorization step_generation_id must be nonempty")
    if authorization.step_generation_id == state.last_step_generation_id:
        raise ValueError("optimizer authorization reuses the previous logical step identity")
    _validate_authorized_material_bars(state, authorization)
    return _apply_validated_paper_kinetic_fixed_site_material_bars(
        state,
        grad_site_rgba_f32=authorization.grad_site_rgba_f32,
        loss_f32=authorization.loss_f32,
        authorization_generation_digest=authorization.generation_digest,
        step_generation_id=authorization.step_generation_id,
    )


@torch.no_grad()
def _apply_validated_paper_kinetic_fixed_site_material_bars(
    state: PaperKineticFixedSiteMaterialState,
    *,
    grad_site_rgba_f32: torch.Tensor,
    loss_f32: torch.Tensor,
    authorization_generation_digest: str,
    step_generation_id: str,
) -> PaperKineticFixedSiteMaterialStepReceipt:
    """Apply authorized CPU bars through the sole fixed-site SGD path.

    Both the dense replay authorization above and the sealed device-gradient
    bridge validate their point-in-time capabilities before entering here.
    Mutation, parameter chain rule, and receipt construction remain shared.
    """

    if not isinstance(state, PaperKineticFixedSiteMaterialState):
        raise TypeError("material SGD step requires fixed-site material state")
    state.assert_current()
    _require_sha256(
        authorization_generation_digest,
        name="authorization generation_digest",
    )
    if not isinstance(step_generation_id, str) or not step_generation_id.strip():
        raise ValueError("optimizer authorization step_generation_id must be nonempty")
    if authorization_generation_digest == state.last_authorization_generation_digest:
        raise ValueError("optimizer authorization was already consumed by this state")
    if step_generation_id == state.last_step_generation_id:
        raise ValueError("optimizer authorization reuses the previous logical step identity")
    _validate_material_bar_tensors(
        state,
        grad_site_rgba_f32=grad_site_rgba_f32,
        loss_f32=loss_f32,
    )
    loss = float(loss_f32.item())

    material_before = state.material_generation_id
    mutation_started = False
    try:
        runtime_parameterization = state.parameterization.runtime_parameterization
        runtime_parameterization.color_vjp_(
            state.raw_color_grad_f32,
            state.site_rgba_f32[:, :3],
            grad_site_rgba_f32[:, :3],
        )
        runtime_parameterization.density_vjp_(
            state.raw_density_grad_f32,
            state.raw_density_f32,
            grad_site_rgba_f32[:, 3],
        )
        if not all(
            bool(torch.isfinite(tensor).all().item())
            for tensor in (state.raw_color_grad_f32, state.raw_density_grad_f32)
        ):
            raise FloatingPointError("manual material chain rule produced non-finite bars")

        color_gradient_norm = float(
            torch.linalg.vector_norm(state.raw_color_grad_f32).item()
        )
        density_gradient_norm = float(
            torch.linalg.vector_norm(state.raw_density_grad_f32).item()
        )
        state.raw_color_grad_f32.mul_(
            -state.optimizer_policy.color_learning_rate
        ).add_(state.raw_color_f32)
        state.raw_density_grad_f32.mul_(
            -state.optimizer_policy.density_learning_rate
        ).add_(state.raw_density_f32)
        if not all(
            bool(torch.isfinite(tensor).all().item())
            for tensor in (state.raw_color_grad_f32, state.raw_density_grad_f32)
        ):
            raise FloatingPointError("manual material SGD candidate is non-finite")
        if bool(
            torch.any(
                state.raw_color_grad_f32.abs()
                > state.optimizer_policy.maximum_absolute_raw_color_value
            ).item()
        ) or bool(
            torch.any(
                state.raw_density_grad_f32.abs()
                > state.optimizer_policy.maximum_absolute_raw_density_value
            ).item()
        ):
            raise ValueError("manual material SGD candidate exceeds raw-value bound")

        mutation_started = True
        state.raw_color_f32.copy_(state.raw_color_grad_f32)
        state.raw_density_f32.copy_(state.raw_density_grad_f32)
        _decode_physical_(
            state.site_rgba_f32,
            state.raw_color_f32,
            state.raw_density_f32,
            parameterization=state.parameterization,
        )
        state.raw_color_grad_f32.zero_()
        state.raw_density_grad_f32.zero_()
        state.generation_parent_digest = material_before
        state.last_authorization_generation_digest = (
            authorization_generation_digest
        )
        state.last_step_generation_id = step_generation_id
        state.step_index += 1
        state.material_generation_id = _state_generation_digest(state)
        state.tensor_signatures = tuple(
            _tensor_signature(tensor) for tensor in state._tensors()
        )
        state.assert_current()

        provisional = PaperKineticFixedSiteMaterialStepReceipt(
            step_index=state.step_index,
            step_generation_id=step_generation_id,
            authorization_generation_digest=authorization_generation_digest,
            material_generation_id_before=material_before,
            material_generation_id_after=state.material_generation_id,
            loss=loss,
            raw_color_gradient_norm=color_gradient_norm,
            raw_density_gradient_norm=density_gradient_norm,
            generation_digest="",
            _seal=_RECEIPT_SEAL,
        )
        result = replace(
            provisional,
            generation_digest=_step_receipt_digest(provisional),
        )
        result.assert_current(state)
        return result
    except BaseException as error:
        if mutation_started:
            state.poisoned = True
        else:
            try:
                state.raw_color_grad_f32.zero_()
                state.raw_density_grad_f32.zero_()
                state.tensor_signatures = tuple(
                    _tensor_signature(tensor) for tensor in state._tensors()
                )
                state.assert_current()
            except BaseException as cleanup_error:
                state.poisoned = True
                error.add_note(
                    "fixed-site material pre-commit scratch cleanup failed; "
                    "state was poisoned: "
                    f"{type(cleanup_error).__qualname__}: {cleanup_error}"
                )
        raise


@torch.no_grad()
def checkpoint_paper_kinetic_fixed_site_material_state(
    state: PaperKineticFixedSiteMaterialState,
) -> PaperKineticFixedSiteMaterialCheckpoint:
    state.assert_current()
    raw_color = state.raw_color_f32.detach().to(device="cpu").clone().contiguous()
    raw_density = state.raw_density_f32.detach().to(device="cpu").clone().contiguous()
    provisional = PaperKineticFixedSiteMaterialCheckpoint(
        world_generation_digest=state.world_generation_digest,
        sites_content_digest=state.sites_content_digest,
        p0_material_seed_generation_digest=(
            state.p0_material_seed_generation_digest
        ),
        parameterization=state.parameterization,
        optimizer_policy=state.optimizer_policy,
        raw_color_f32_cpu=raw_color,
        raw_density_f32_cpu=raw_density,
        initialization_content_digest=state.initialization_content_digest,
        generation_parent_digest=state.generation_parent_digest,
        last_authorization_generation_digest=(
            state.last_authorization_generation_digest
        ),
        last_step_generation_id=state.last_step_generation_id,
        step_index=state.step_index,
        material_generation_id=state.material_generation_id,
        raw_color_content_digest=_tensor_content_digest(raw_color),
        raw_density_content_digest=_tensor_content_digest(raw_density),
        generation_digest="",
        _seal=_CHECKPOINT_SEAL,
    )
    result = replace(
        provisional,
        generation_digest=_checkpoint_digest(provisional),
    )
    result.assert_current()
    return result


def paper_kinetic_fixed_site_material_checkpoint_from_payload(
    payload: Mapping[str, Any],
    *,
    expected_world_site_count: int,
    maximum_checkpoint_logical_tensor_bytes: int,
) -> PaperKineticFixedSiteMaterialCheckpoint:
    required = {
        "schema",
        "provenance",
        "world_generation_digest",
        "sites_content_digest",
        "p0_material_seed_generation_digest",
        "parameterization",
        "optimizer_policy",
        "raw_color_f32_cpu",
        "raw_density_f32_cpu",
        "initialization_content_digest",
        "generation_parent_digest",
        "last_authorization_generation_digest",
        "last_step_generation_id",
        "step_index",
        "material_generation_id",
        "raw_color_content_digest",
        "raw_density_content_digest",
        "generation_digest",
        "persistent_frame_tensor_bytes",
        "persistent_sample_tensor_bytes",
        "persistent_target_tensor_bytes",
        "persistent_prediction_tensor_bytes",
        "optimizer_history_tensor_bytes",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise ValueError("fixed-site material checkpoint payload keys changed")
    _require_positive_int(
        expected_world_site_count,
        name="expected_world_site_count",
    )
    _preflight_checkpoint_logical_bytes(
        expected_world_site_count,
        maximum_checkpoint_logical_tensor_bytes=(
            maximum_checkpoint_logical_tensor_bytes
        ),
    )
    parameterization_payload = payload["parameterization"]
    optimizer_payload = payload["optimizer_policy"]
    if not isinstance(parameterization_payload, Mapping) or set(parameterization_payload) != {
        "density_beta",
        "density_threshold",
        "minimum_density",
        "color_epsilon",
    }:
        raise ValueError("fixed-site material checkpoint parameterization changed")
    if not isinstance(optimizer_payload, Mapping) or set(optimizer_payload) != {
        "optimizer",
        "color_learning_rate",
        "density_learning_rate",
        "maximum_absolute_raw_color_value",
        "maximum_absolute_raw_density_value",
        "momentum",
        "weight_decay",
    }:
        raise ValueError("fixed-site material checkpoint optimizer policy changed")
    if (
        optimizer_payload["optimizer"] != "manual_sgd"
        or float(optimizer_payload["momentum"]) != 0.0
        or float(optimizer_payload["weight_decay"]) != 0.0
    ):
        raise ValueError("fixed-site material checkpoint is not stateless manual SGD")
    _require_nonnegative_int(payload["step_index"], name="step_index")
    _preflight_checkpoint_payload_tensor(
        payload["raw_color_f32_cpu"],
        name="raw_color_f32_cpu",
        shape=(expected_world_site_count, 3),
    )
    _preflight_checkpoint_payload_tensor(
        payload["raw_density_f32_cpu"],
        name="raw_density_f32_cpu",
        shape=(expected_world_site_count,),
    )
    result = PaperKineticFixedSiteMaterialCheckpoint(
        world_generation_digest=str(payload["world_generation_digest"]),
        sites_content_digest=str(payload["sites_content_digest"]),
        p0_material_seed_generation_digest=str(
            payload["p0_material_seed_generation_digest"]
        ),
        parameterization=PaperKineticFixedSiteMaterialParameterization(
            density_beta=float(parameterization_payload["density_beta"]),
            density_threshold=float(parameterization_payload["density_threshold"]),
            minimum_density=float(parameterization_payload["minimum_density"]),
            color_epsilon=float(parameterization_payload["color_epsilon"]),
        ),
        optimizer_policy=PaperKineticFixedSiteMaterialSGDPolicy(
            color_learning_rate=float(optimizer_payload["color_learning_rate"]),
            density_learning_rate=float(
                optimizer_payload["density_learning_rate"]
            ),
            maximum_absolute_raw_color_value=float(
                optimizer_payload["maximum_absolute_raw_color_value"]
            ),
            maximum_absolute_raw_density_value=float(
                optimizer_payload["maximum_absolute_raw_density_value"]
            ),
        ),
        raw_color_f32_cpu=_owned_checkpoint_payload_tensor(
            payload["raw_color_f32_cpu"],
            name="raw_color_f32_cpu",
        ),
        raw_density_f32_cpu=_owned_checkpoint_payload_tensor(
            payload["raw_density_f32_cpu"],
            name="raw_density_f32_cpu",
        ),
        initialization_content_digest=str(payload["initialization_content_digest"]),
        generation_parent_digest=str(payload["generation_parent_digest"]),
        last_authorization_generation_digest=str(
            payload["last_authorization_generation_digest"]
        ),
        last_step_generation_id=str(payload["last_step_generation_id"]),
        step_index=payload["step_index"],
        material_generation_id=str(payload["material_generation_id"]),
        raw_color_content_digest=str(payload["raw_color_content_digest"]),
        raw_density_content_digest=str(payload["raw_density_content_digest"]),
        generation_digest=str(payload["generation_digest"]),
        provenance=str(payload["provenance"]),
        schema=str(payload["schema"]),
        persistent_frame_tensor_bytes=int(payload["persistent_frame_tensor_bytes"]),
        persistent_sample_tensor_bytes=int(payload["persistent_sample_tensor_bytes"]),
        persistent_target_tensor_bytes=int(payload["persistent_target_tensor_bytes"]),
        persistent_prediction_tensor_bytes=int(
            payload["persistent_prediction_tensor_bytes"]
        ),
        optimizer_history_tensor_bytes=int(payload["optimizer_history_tensor_bytes"]),
        _seal=_CHECKPOINT_SEAL,
    )
    result.assert_current()
    return result


@torch.no_grad()
def restore_paper_kinetic_fixed_site_material_state(
    checkpoint: PaperKineticFixedSiteMaterialCheckpoint,
    *,
    world: PaperKineticWorldSnapshot,
    device: torch.device | str,
    maximum_material_state_logical_tensor_bytes: int,
) -> PaperKineticFixedSiteMaterialState:
    if not isinstance(checkpoint, PaperKineticFixedSiteMaterialCheckpoint):
        raise TypeError("fixed-site material restart requires a sealed checkpoint")
    if not isinstance(world, PaperKineticWorldSnapshot):
        raise TypeError("fixed-site material restart requires a sealed world snapshot")
    resolved_device = torch.device(device)
    if resolved_device.type != "cpu":
        raise NotImplementedError(
            "fixed-site material restore is CPU-only until optimizer updates "
            "own a completion fence"
        )
    if checkpoint.site_count != world.site_count:
        raise ValueError("fixed-site material checkpoint belongs to a different world")
    _preflight_material_state_logical_bytes(
        world.site_count,
        maximum_material_state_logical_tensor_bytes=(
            maximum_material_state_logical_tensor_bytes
        ),
    )
    checkpoint.assert_current()
    world.assert_current()
    if (
        checkpoint.world_generation_digest != world.generation_digest
        or checkpoint.sites_content_digest != world.sites_content_digest
    ):
        raise ValueError("fixed-site material checkpoint belongs to a different world")
    raw_color = checkpoint.raw_color_f32_cpu.detach().clone().contiguous()
    raw_density = checkpoint.raw_density_f32_cpu.detach().clone().contiguous()
    site_rgba = torch.empty(
        (checkpoint.site_count, 4),
        dtype=torch.float32,
        device=resolved_device,
    )
    color_grad = torch.zeros_like(raw_color)
    density_grad = torch.zeros_like(raw_density)
    _decode_physical_(
        site_rgba,
        raw_color,
        raw_density,
        parameterization=checkpoint.parameterization,
    )
    result = PaperKineticFixedSiteMaterialState(
        world_generation_digest=checkpoint.world_generation_digest,
        sites_content_digest=checkpoint.sites_content_digest,
        p0_material_seed_generation_digest=(
            checkpoint.p0_material_seed_generation_digest
        ),
        parameterization=checkpoint.parameterization,
        optimizer_policy=checkpoint.optimizer_policy,
        raw_color_f32=raw_color,
        raw_density_f32=raw_density,
        site_rgba_f32=site_rgba,
        raw_color_grad_f32=color_grad,
        raw_density_grad_f32=density_grad,
        initialization_content_digest=checkpoint.initialization_content_digest,
        generation_parent_digest=checkpoint.generation_parent_digest,
        last_authorization_generation_digest=(
            checkpoint.last_authorization_generation_digest
        ),
        last_step_generation_id=checkpoint.last_step_generation_id,
        step_index=checkpoint.step_index,
        material_generation_id=checkpoint.material_generation_id,
        restart_checkpoint_generation_digest=checkpoint.generation_digest,
        tensor_signatures=(),
        poisoned=False,
        _seal=_STATE_SEAL,
    )
    result.tensor_signatures = tuple(
        _tensor_signature(tensor) for tensor in result._tensors()
    )
    result.assert_current()
    return result


def _validate_state_tensors(state: PaperKineticFixedSiteMaterialState) -> None:
    device = state.device
    if device.type != "cpu":
        raise ValueError("fixed-site material state requires CPU tensors")
    _require_f32(
        state.raw_color_f32,
        name="raw_color_f32",
        shape=(state.site_count, 3),
        device=device,
    )
    _require_f32(
        state.raw_density_f32,
        name="raw_density_f32",
        shape=(state.site_count,),
        device=device,
    )
    _require_f32(
        state.site_rgba_f32,
        name="site_rgba_f32",
        shape=(state.site_count, 4),
        device=device,
    )
    _require_f32(
        state.raw_color_grad_f32,
        name="raw_color_grad_f32",
        shape=(state.site_count, 3),
        device=device,
    )
    _require_f32(
        state.raw_density_grad_f32,
        name="raw_density_grad_f32",
        shape=(state.site_count,),
        device=device,
    )
    _require_distinct_storage(*state._tensors())
    if any(tensor.requires_grad for tensor in state._tensors()):
        raise ValueError("manual fixed-site material state forbids autograd tensors")
    if not all(bool(torch.isfinite(tensor).all().item()) for tensor in state._tensors()):
        raise ValueError("fixed-site material state contains non-finite tensors")
    if bool(torch.any((state.site_rgba_f32[:, :3] <= 0.0) | (state.site_rgba_f32[:, :3] >= 1.0)).item()):
        raise ValueError("decoded sigmoid RGB must lie strictly inside (0,1)")
    if bool(
        torch.any(
            state.site_rgba_f32[:, 3]
            <= state.parameterization.minimum_density
        ).item()
    ):
        raise ValueError("decoded density must exceed minimum_density")
    if bool(torch.any(state.raw_color_grad_f32 != 0.0).item()) or bool(
        torch.any(state.raw_density_grad_f32 != 0.0).item()
    ):
        raise ValueError("fixed-site raw gradient buffers must be zero between steps")
    if bool(
        torch.any(
            state.raw_color_f32.abs()
            > state.optimizer_policy.maximum_absolute_raw_color_value
        ).item()
    ) or bool(
        torch.any(
            state.raw_density_f32.abs()
            > state.optimizer_policy.maximum_absolute_raw_density_value
        ).item()
    ):
        raise ValueError("fixed-site raw material left its declared bound")


def _validate_authorized_material_bars(
    state: PaperKineticFixedSiteMaterialState,
    authorization: PaperKineticDenseOptimizerAuthorization,
) -> None:
    _validate_material_bar_tensors(
        state,
        grad_site_rgba_f32=authorization.grad_site_rgba_f32,
        loss_f32=authorization.loss_f32,
    )


def _validate_material_bar_tensors(
    state: PaperKineticFixedSiteMaterialState,
    *,
    grad_site_rgba_f32: torch.Tensor,
    loss_f32: torch.Tensor,
) -> None:
    _require_f32(
        grad_site_rgba_f32,
        name="authorized grad_site_rgba_f32",
        shape=(state.site_count, 4),
        device=state.device,
    )
    _require_f32(
        loss_f32,
        name="authorized loss_f32",
        shape=(1,),
        device=state.device,
    )
    if grad_site_rgba_f32.requires_grad or loss_f32.requires_grad:
        raise ValueError("authorized material/loss bars must be explicit non-autograd tensors")
    if not bool(torch.isfinite(grad_site_rgba_f32).all().item()) or not bool(
        torch.isfinite(loss_f32).all().item()
    ):
        raise ValueError("authorized material/loss bars must be finite")
    _require_distinct_storage(
        *state._tensors(),
        grad_site_rgba_f32,
        loss_f32,
    )


def _decode_physical_(
    destination_rgba: torch.Tensor,
    raw_color: torch.Tensor,
    raw_density: torch.Tensor,
    *,
    parameterization: PaperKineticFixedSiteMaterialParameterization,
) -> None:
    runtime_parameterization = parameterization.runtime_parameterization
    runtime_parameterization.decode_color_(destination_rgba[:, :3], raw_color)
    runtime_parameterization.decode_density_(
        destination_rgba[:, 3],
        raw_density,
    )


def _state_generation_digest(state: PaperKineticFixedSiteMaterialState) -> str:
    # This is deliberately a live version-chain identity, not a device-to-host
    # numeric hash on every optimizer step.  Tensor identity/version signatures
    # reject unsanctioned in-place edits while live.  The raw-only restart
    # checkpoint below separately hashes exact CPU numeric content.
    return _material_generation_digest_from_fields(
        world_generation_digest=state.world_generation_digest,
        sites_content_digest=state.sites_content_digest,
        p0_material_seed_generation_digest=(
            state.p0_material_seed_generation_digest
        ),
        parameterization=state.parameterization,
        optimizer_policy=state.optimizer_policy,
        initialization_content_digest=state.initialization_content_digest,
        generation_parent_digest=state.generation_parent_digest,
        last_authorization_generation_digest=(
            state.last_authorization_generation_digest
        ),
        last_step_generation_id=state.last_step_generation_id,
        step_index=state.step_index,
    )


def _material_generation_digest_from_fields(
    *,
    world_generation_digest: str,
    sites_content_digest: str,
    p0_material_seed_generation_digest: str,
    parameterization: PaperKineticFixedSiteMaterialParameterization,
    optimizer_policy: PaperKineticFixedSiteMaterialSGDPolicy,
    initialization_content_digest: str,
    generation_parent_digest: str,
    last_authorization_generation_digest: str,
    last_step_generation_id: str,
    step_index: int,
) -> str:
    return _digest_parts(
        STATE_PROVENANCE,
        world_generation_digest,
        sites_content_digest,
        p0_material_seed_generation_digest,
        parameterization.generation_digest,
        optimizer_policy.generation_digest,
        initialization_content_digest,
        generation_parent_digest,
        last_authorization_generation_digest,
        last_step_generation_id,
        step_index,
        "material_only",
        0,
        0,
        0,
        0,
    )


def _step_receipt_digest(receipt: PaperKineticFixedSiteMaterialStepReceipt) -> str:
    return _digest_parts(
        STEP_RECEIPT_PROVENANCE,
        receipt.step_index,
        receipt.step_generation_id,
        receipt.authorization_generation_digest,
        receipt.material_generation_id_before,
        receipt.material_generation_id_after,
        receipt.loss,
        receipt.raw_color_gradient_norm,
        receipt.raw_density_gradient_norm,
        receipt.geometry_updated,
        receipt.persistent_tensor_bytes,
    )


def _checkpoint_digest(checkpoint: PaperKineticFixedSiteMaterialCheckpoint) -> str:
    return _digest_parts(
        CHECKPOINT_PROVENANCE,
        CHECKPOINT_SCHEMA,
        checkpoint.world_generation_digest,
        checkpoint.sites_content_digest,
        checkpoint.p0_material_seed_generation_digest,
        checkpoint.parameterization.generation_digest,
        checkpoint.optimizer_policy.generation_digest,
        checkpoint.initialization_content_digest,
        checkpoint.generation_parent_digest,
        checkpoint.last_authorization_generation_digest,
        checkpoint.last_step_generation_id,
        checkpoint.step_index,
        checkpoint.material_generation_id,
        checkpoint.raw_color_content_digest,
        checkpoint.raw_density_content_digest,
        checkpoint.checkpoint_tensor_bytes,
        checkpoint.persistent_frame_tensor_bytes,
        checkpoint.persistent_sample_tensor_bytes,
        checkpoint.persistent_target_tensor_bytes,
        checkpoint.persistent_prediction_tensor_bytes,
        checkpoint.optimizer_history_tensor_bytes,
    )


def _tensor_content_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(repr(tuple(value.shape)).encode("utf-8"))
    digest.update(memoryview(value.numpy()).cast("B"))
    return digest.hexdigest()


def _tensor_signature(tensor: torch.Tensor) -> tuple[object, ...]:
    storage = tensor.untyped_storage()
    return (
        id(tensor),
        int(getattr(tensor, "_version", 0)),
        int(storage.data_ptr()),
        int(storage.nbytes()),
        int(tensor.storage_offset()),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
        bool(tensor.requires_grad),
    )


def _tensor_bytes(*tensors: torch.Tensor) -> int:
    storages: dict[tuple[str, int], int] = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        storages[(str(tensor.device), int(storage.data_ptr()))] = int(storage.nbytes())
    return sum(storages.values())


def _preflight_material_state_logical_bytes(
    site_count: int,
    *,
    maximum_material_state_logical_tensor_bytes: int,
) -> None:
    _require_positive_int(site_count, name="site_count")
    _require_positive_int(
        maximum_material_state_logical_tensor_bytes,
        name="maximum_material_state_logical_tensor_bytes",
    )
    # raw RGB + raw density + physical RGBA + both raw-gradient buffers
    required = 12 * site_count * FLOAT32_LOGICAL_BYTES
    if required > maximum_material_state_logical_tensor_bytes:
        raise MemoryError(
            "fixed-site material state exceeds its explicit logical byte bound"
        )


def _preflight_checkpoint_logical_bytes(
    site_count: int,
    *,
    maximum_checkpoint_logical_tensor_bytes: int,
) -> None:
    _require_positive_int(site_count, name="site_count")
    _require_positive_int(
        maximum_checkpoint_logical_tensor_bytes,
        name="maximum_checkpoint_logical_tensor_bytes",
    )
    # raw RGB + raw density; physical material and scratch are reconstructed.
    required = 4 * site_count * FLOAT32_LOGICAL_BYTES
    if required > maximum_checkpoint_logical_tensor_bytes:
        raise MemoryError(
            "fixed-site material checkpoint exceeds its explicit logical byte bound"
        )


def _preflight_checkpoint_payload_tensor(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"checkpoint {name} must be a tensor")
    if (
        value.device.type != "cpu"
        or value.dtype != torch.float32
        or tuple(value.shape) != shape
        or not value.is_contiguous()
        or value.requires_grad
    ):
        raise ValueError(
            f"checkpoint {name} must be finite contiguous CPU float32 {shape}"
        )
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(
            f"checkpoint {name} must be finite contiguous CPU float32 {shape}"
        )


def _require_f32(
    tensor: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, ...],
    device: torch.device,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if (
        tensor.dtype != torch.float32
        or tensor.device != device
        or tuple(tensor.shape) != shape
        or not tensor.is_contiguous()
    ):
        raise ValueError(f"{name} must be contiguous float32 {shape} on {device}")


def _require_cpu_f32(
    tensor: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, ...],
) -> None:
    _require_f32(tensor, name=name, shape=shape, device=torch.device("cpu"))
    if tensor.requires_grad or not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} must be finite and non-autograd")


def _owned_checkpoint_payload_tensor(value: Any, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"checkpoint {name} must be a tensor")
    if value.device.type != "cpu" or value.dtype != torch.float32:
        raise ValueError(f"checkpoint {name} must be CPU float32")
    return value.detach().clone().contiguous()


def _require_distinct_storage(*tensors: torch.Tensor) -> None:
    keys = tuple(
        (str(tensor.device), int(tensor.untyped_storage().data_ptr()))
        for tensor in tensors
    )
    if len(set(keys)) != len(keys):
        raise ValueError("fixed-site material tensors must own distinct storage")


def _require_positive_int(value: Any, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _require_nonnegative_int(value: Any, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _require_sha256(value: str, *, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    try:
        parsed = bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest") from error
    if len(parsed) != 32 or value != value.lower():
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _digest_parts(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = repr(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


__all__ = [
    "CHECKPOINT_PROVENANCE",
    "CHECKPOINT_SCHEMA",
    "STATE_PROVENANCE",
    "STEP_RECEIPT_PROVENANCE",
    "PaperKineticFixedSiteMaterialCheckpoint",
    "PaperKineticFixedSiteMaterialParameterization",
    "PaperKineticFixedSiteMaterialSGDPolicy",
    "PaperKineticFixedSiteMaterialState",
    "PaperKineticFixedSiteMaterialStepReceipt",
    "apply_paper_kinetic_fixed_site_material_sgd_step",
    "checkpoint_paper_kinetic_fixed_site_material_state",
    "paper_kinetic_fixed_site_material_checkpoint_from_payload",
    "prepare_paper_kinetic_fixed_site_material_state",
    "restore_paper_kinetic_fixed_site_material_state",
]
