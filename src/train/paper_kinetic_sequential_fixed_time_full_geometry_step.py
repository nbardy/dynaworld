"""Fair sequential fixed-time WorldFoam full-geometry control.

This control intentionally does *not* invoke the continuous kinetic compiler.
For each selected time it discovers the exact lower-envelope owner word for
every fixed-camera ray, streams bounded spatial blocks through the existing
precompiled-length native P0 forward/VJP ABI, reduces the returned physical-
length bars into kinetic world parameters, and releases all frame-local
topology, target, prediction, and native scratch before advancing.

The expensive live topology/native reverse state is independent of requested
frame count; the selected-time scalar grid remains the expected cheap ``O(F)``
camera slice.  Discovery, native world replay, and geometry reverse work remain
linear in frame count.  That is the fair sequential same-representation
control for the compiled shared-adjoint ablation.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import torch
from paper_kinetic_runtime_paths import ensure_worldfoam_lane2_research_path

ensure_worldfoam_lane2_research_path()

from kinetic_power_word_compiler import (  # noqa: E402
    AffineKineticPowerSites,
    discover_kinetic_power_word_at_time,
)
from kinetic_stable_stratum_vjp import (  # noqa: E402
    StableStratumThresholds,
    kinetic_p0_fixed_time_physical_length_geometry_vjp,
)


STEP_PROVENANCE = "paper-kinetic-sequential-fixed-time-full-geometry-step-v1"
STEP_STATUS = "source_complete/native_runtime_unverified"
LOSS_NORMALIZATION_ID = "global_selected_rgb_mean"
FORWARD_INTO_OP_NAME = (
    "kinetic_precompiled_length_p0_lie_node_forward_into_launch_only_v1"
)
VJP_OP_NAME = "kinetic_precompiled_length_p0_lie_node_vjp_accumulate_launch_only"

_RESULT_SEAL = object()
_RECEIPT_SEAL = object()


@dataclass(frozen=True)
class PaperKineticSequentialFixedTimeMemoryPolicy:
    """Logical tensor caps; allocator/process peaks remain external evidence."""

    maximum_target_frame_logical_tensor_bytes: int
    maximum_frame_cpu_topology_logical_tensor_bytes: int
    maximum_active_device_scratch_logical_tensor_bytes: int
    maximum_geometry_d2h_logical_tensor_bytes: int
    maximum_tracks_per_native_block: int

    def assert_valid(self) -> None:
        for name, value in self.__dict__.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class PaperKineticSequentialFixedTimeStepReceipt:
    """Tensor-free aggregate of the exact per-block receipt chain."""

    selected_times_manifest_digest: str
    fixed_time_receipt_chain_digest: str
    native_callable_identity_digest: str
    requested_frame_count: int
    track_count: int
    global_site_count: int
    streamed_sample_count: int
    global_loss_element_count: int
    selected_time_grid_tensor_bytes: int
    fixed_time_direct_sample_evaluation_count: int
    sample_to_node_interaction_count: int
    native_sample_launch_count: int
    target_frame_load_count: int
    target_frame_release_count: int
    frame_scratch_release_count: int
    fixed_time_lower_envelope_discovery_call_count: int
    candidate_line_evaluation_count: int
    discovered_owner_run_count: int
    discovered_active_boundary_count: int
    native_block_count: int
    native_forward_launch_count: int
    native_forward_ordered_run_interaction_count: int
    native_full_vjp_launch_count: int
    native_vjp_ordered_run_interaction_count: int
    completion_fence_call_count: int
    geometry_length_bar_d2h_count: int
    geometry_row_vjp_call_count: int
    physical_length_reverse_interaction_count: int
    all_site_owner_margin_evaluation_count: int
    material_compact_to_global_scatter_row_count: int
    maximum_target_frame_logical_tensor_bytes: int
    maximum_frame_cpu_topology_logical_tensor_bytes: int
    maximum_active_device_scratch_logical_tensor_bytes: int
    maximum_geometry_d2h_logical_tensor_bytes: int
    generation_digest: str
    continuous_compiler_invocation_count: int = 0
    continuous_chart_count: int = 0
    retained_frame_receipt_count: int = 0
    persistent_frame_tensor_bytes: int = 0
    persistent_target_tensor_bytes: int = 0
    persistent_prediction_tensor_bytes: int = 0
    frame_by_word_reverse_state_tensor_bytes: int = 0
    allocator_peak_measured: bool = False
    provenance: str = STEP_PROVENANCE
    _seal: object = field(default=None, repr=False)

    def assert_current(self) -> None:
        expected_rows = self.requested_frame_count * self.track_count
        if (
            self._seal is not _RECEIPT_SEAL
            or self.provenance != STEP_PROVENANCE
            or not _is_sha256(self.selected_times_manifest_digest)
            or not _is_sha256(self.fixed_time_receipt_chain_digest)
            or not _is_sha256(self.native_callable_identity_digest)
            or self.requested_frame_count < 1
            or self.track_count < 1
            or self.global_site_count < 1
            or self.streamed_sample_count != expected_rows
            or self.global_loss_element_count != 3 * expected_rows
            or self.selected_time_grid_tensor_bytes
            != self.requested_frame_count * 8
            or self.fixed_time_direct_sample_evaluation_count != expected_rows
            or self.sample_to_node_interaction_count != expected_rows
            or self.native_sample_launch_count != 0
            or self.target_frame_load_count != self.requested_frame_count
            or self.target_frame_release_count != self.requested_frame_count
            or self.frame_scratch_release_count != self.requested_frame_count
            or self.fixed_time_lower_envelope_discovery_call_count != expected_rows
            or self.candidate_line_evaluation_count
            != expected_rows * self.global_site_count
            or self.discovered_owner_run_count < expected_rows
            or self.discovered_active_boundary_count
            != self.discovered_owner_run_count - expected_rows
            or self.native_block_count < self.requested_frame_count
            or self.native_forward_launch_count != self.native_block_count
            or self.native_forward_ordered_run_interaction_count
            != self.discovered_owner_run_count
            or self.native_full_vjp_launch_count != self.native_block_count
            or self.native_vjp_ordered_run_interaction_count
            != self.discovered_owner_run_count
            or self.completion_fence_call_count != self.native_block_count
            or self.geometry_length_bar_d2h_count != self.native_block_count
            or self.geometry_row_vjp_call_count != expected_rows
            or self.physical_length_reverse_interaction_count
            != self.discovered_owner_run_count
            or self.all_site_owner_margin_evaluation_count
            != 3
            * self.discovered_owner_run_count
            * (self.global_site_count - 1)
            or self.material_compact_to_global_scatter_row_count
            < self.native_block_count
            or min(
                self.maximum_target_frame_logical_tensor_bytes,
                self.maximum_frame_cpu_topology_logical_tensor_bytes,
                self.maximum_active_device_scratch_logical_tensor_bytes,
                self.maximum_geometry_d2h_logical_tensor_bytes,
            )
            < 1
            or self.continuous_compiler_invocation_count != 0
            or self.continuous_chart_count != 0
            or self.retained_frame_receipt_count != 0
            or any(
                value != 0
                for value in (
                    self.persistent_frame_tensor_bytes,
                    self.persistent_target_tensor_bytes,
                    self.persistent_prediction_tensor_bytes,
                    self.frame_by_word_reverse_state_tensor_bytes,
                )
            )
            or self.allocator_peak_measured
            or self.generation_digest != _step_receipt_digest(self)
        ):
            raise ValueError("sequential fixed-time step receipt changed")

    def accounting(self) -> Mapping[str, int | str | bool]:
        self.assert_current()
        return MappingProxyType(
            {
                name: value
                for name, value in self.__dict__.items()
                if name not in {"_seal"}
            }
        )


@dataclass(frozen=True)
class PaperKineticSequentialFixedTimeFullGeometryStepResult:
    """One complete gradient authorization for a single external SGD call."""

    step_index: int
    geometry_generation_id: str
    material_generation_id: str
    background_generation_id: str
    target_generation_id: str
    loss_f32: torch.Tensor = field(repr=False)
    grad_global_site_rgba_f32: torch.Tensor = field(repr=False)
    grad_positions0_f64_cpu: torch.Tensor = field(repr=False)
    grad_velocities_f64_cpu: torch.Tensor = field(repr=False)
    grad_weight_coefficients_f64_cpu: torch.Tensor = field(repr=False)
    receipt: PaperKineticSequentialFixedTimeStepReceipt
    accounting: Mapping[str, int | float | str | bool]
    generation_digest: str
    _tensor_signatures: tuple[tuple[object, ...], ...] = field(repr=False)
    _seal: object = field(default=None, repr=False)
    loss_normalization_id: str = LOSS_NORMALIZATION_ID
    optimizer_update_authorization_count: int = 1
    fixed_camera: bool = True
    camera_ray_gradients_enabled: bool = False
    native_runtime_verified: bool = False
    allocator_peak_measured: bool = False
    provenance: str = STEP_PROVENANCE
    runtime_status: str = STEP_STATUS

    def assert_current(self) -> None:
        tensors = (
            self.loss_f32,
            self.grad_global_site_rgba_f32,
            self.grad_positions0_f64_cpu,
            self.grad_velocities_f64_cpu,
            self.grad_weight_coefficients_f64_cpu,
        )
        site_count = self.receipt.global_site_count
        weight_count = int(self.grad_weight_coefficients_f64_cpu.shape[1])
        if (
            self._seal is not _RESULT_SEAL
            or self.provenance != STEP_PROVENANCE
            or self.runtime_status != STEP_STATUS
            or self.step_index < 0
            or any(
                not _is_sha256(value)
                for value in (
                    self.geometry_generation_id,
                    self.material_generation_id,
                    self.background_generation_id,
                    self.target_generation_id,
                )
            )
            or self.loss_normalization_id != LOSS_NORMALIZATION_ID
            or self.optimizer_update_authorization_count != 1
            or not self.fixed_camera
            or self.camera_ray_gradients_enabled
            or self.native_runtime_verified
            or self.allocator_peak_measured
            or tuple(_tensor_signature(tensor) for tensor in tensors)
            != self._tensor_signatures
            or not isinstance(self.accounting, MappingProxyType)
            or self.accounting.get("optimizer_update_authorization_count") != 1
            or self.accounting.get("continuous_compiler_invocation_count") != 0
            or self.generation_digest != _result_digest(self)
        ):
            raise ValueError("sequential fixed-time result changed")
        _require_tensor(
            self.loss_f32,
            name="loss_f32",
            dtype=torch.float32,
            device=self.grad_global_site_rgba_f32.device,
            shape=(1,),
        )
        _require_tensor(
            self.grad_global_site_rgba_f32,
            name="grad_global_site_rgba_f32",
            dtype=torch.float32,
            device=self.grad_global_site_rgba_f32.device,
            shape=(site_count, 4),
        )
        for tensor, shape, name in (
            (self.grad_positions0_f64_cpu, (site_count, 3), "position bar"),
            (self.grad_velocities_f64_cpu, (site_count, 3), "velocity bar"),
            (
                self.grad_weight_coefficients_f64_cpu,
                (site_count, weight_count),
                "weight bar",
            ),
        ):
            _require_tensor(
                tensor,
                name=name,
                dtype=torch.float64,
                device=torch.device("cpu"),
                shape=shape,
            )
            if not bool(torch.isfinite(tensor).all().item()):
                raise FloatingPointError(f"{name} is nonfinite")
        self.receipt.assert_current()


@dataclass
class _ReceiptAccumulator:
    chain: Any = field(default_factory=hashlib.sha256)
    target_frame_load_count: int = 0
    target_frame_release_count: int = 0
    frame_scratch_release_count: int = 0
    discovery_count: int = 0
    candidate_line_count: int = 0
    run_count: int = 0
    boundary_count: int = 0
    native_block_count: int = 0
    forward_count: int = 0
    forward_run_interactions: int = 0
    vjp_count: int = 0
    vjp_run_interactions: int = 0
    fence_count: int = 0
    d2h_count: int = 0
    geometry_row_count: int = 0
    reverse_interactions: int = 0
    owner_margin_evaluations: int = 0
    material_scatter_rows: int = 0
    maximum_target_bytes: int = 0
    maximum_cpu_topology_bytes: int = 0
    maximum_device_scratch_bytes: int = 0
    maximum_d2h_bytes: int = 0

    def add_digest(self, value: str) -> None:
        encoded = value.encode("ascii")
        self.chain.update(len(encoded).to_bytes(8, "big"))
        self.chain.update(encoded)


@torch.no_grad()
def run_paper_kinetic_sequential_fixed_time_full_geometry_step(
    sites: AffineKineticPowerSites,
    ray_coefficients_f64_cpu: torch.Tensor,
    selected_times_f64_cpu: torch.Tensor,
    *,
    step_index: int,
    geometry_generation_id: str,
    material_generation_id: str,
    background_generation_id: str,
    target_generation_id: str,
    near: float,
    far: float,
    global_site_rgba_f32: torch.Tensor,
    global_grad_site_rgba_f32: torch.Tensor,
    grad_positions0_f64_cpu: torch.Tensor,
    grad_velocities_f64_cpu: torch.Tensor,
    grad_weight_coefficients_f64_cpu: torch.Tensor,
    background_rgb_f32: torch.Tensor,
    target_frame_loader: Callable[[int, float], torch.Tensor],
    native_ops: Any,
    device_completion_fence: Callable[[], None],
    device_completion_fence_provenance: str,
    memory_policy: PaperKineticSequentialFixedTimeMemoryPolicy,
    optimizer_update: Callable[
        [PaperKineticSequentialFixedTimeFullGeometryStepResult], None
    ],
    thresholds: StableStratumThresholds = StableStratumThresholds(),
    physical_length_epsilon: float = 1.0e-8,
) -> PaperKineticSequentialFixedTimeFullGeometryStepResult:
    """Execute one true sequential replay and authorize one optimizer update."""

    if not isinstance(sites, AffineKineticPowerSites):
        raise TypeError("sites must be AffineKineticPowerSites")
    if not isinstance(memory_policy, PaperKineticSequentialFixedTimeMemoryPolicy):
        raise TypeError("memory_policy has the wrong type")
    memory_policy.assert_valid()
    if not isinstance(thresholds, StableStratumThresholds):
        raise TypeError("thresholds has the wrong type")
    if not callable(target_frame_loader) or not callable(optimizer_update):
        raise TypeError("target_frame_loader and optimizer_update must be callable")
    if not callable(device_completion_fence):
        raise TypeError("device_completion_fence must be callable")
    if not isinstance(device_completion_fence_provenance, str) or not (
        device_completion_fence_provenance.strip()
    ):
        raise ValueError("device completion fence provenance is required")
    if isinstance(step_index, bool) or not isinstance(step_index, int) or step_index < 0:
        raise ValueError("step_index must be a nonnegative integer")
    for name, value in (
        ("geometry_generation_id", geometry_generation_id),
        ("material_generation_id", material_generation_id),
        ("background_generation_id", background_generation_id),
        ("target_generation_id", target_generation_id),
    ):
        if not _is_sha256(value):
            raise ValueError(f"{name} must be SHA-256")
    if not math.isfinite(near) or not math.isfinite(far) or far <= near:
        raise ValueError("near/far must be finite with near < far")
    if not math.isfinite(physical_length_epsilon) or physical_length_epsilon < 0:
        raise ValueError("physical_length_epsilon must be finite and nonnegative")

    rays = _cpu_f64(ray_coefficients_f64_cpu, name="ray_coefficients_f64_cpu")
    times = _cpu_f64(selected_times_f64_cpu, name="selected_times_f64_cpu").reshape(-1)
    if rays.ndim != 2 or rays.shape[1] != 12 or rays.shape[0] < 1:
        raise ValueError("ray_coefficients_f64_cpu must have shape [P,12]")
    if times.numel() < 1 or bool(torch.any(times[1:] <= times[:-1]).item()):
        raise ValueError("selected times must be nonempty and strictly increasing")
    if not torch.equal(rays[:, 3:6], torch.zeros_like(rays[:, 3:6])) or not torch.equal(
        rays[:, 9:12], torch.zeros_like(rays[:, 9:12])
    ):
        raise ValueError("this control is fixed-camera and does not optimize ray programs")
    if geometry_generation_id != (
        _geometry_generation_id_from_normalized(
            sites,
            rays,
            near=near,
            far=far,
        )
    ):
        raise ValueError("geometry_generation_id is foreign to the live fixed-time world")
    track_count = int(rays.shape[0])
    site_count = sites.site_count
    weight_count = int(sites.weight_coefficients.shape[1])
    device = global_site_rgba_f32.device
    for tensor, shape, name in (
        (global_site_rgba_f32, (site_count, 4), "global_site_rgba_f32"),
        (
            global_grad_site_rgba_f32,
            (site_count, 4),
            "global_grad_site_rgba_f32",
        ),
        (background_rgb_f32, (3,), "background_rgb_f32"),
    ):
        _require_tensor(
            tensor,
            name=name,
            dtype=torch.float32,
            device=device,
            shape=shape,
        )
    for tensor, shape, name in (
        (grad_positions0_f64_cpu, (site_count, 3), "position bar"),
        (grad_velocities_f64_cpu, (site_count, 3), "velocity bar"),
        (
            grad_weight_coefficients_f64_cpu,
            (site_count, weight_count),
            "weight bar",
        ),
    ):
        _require_tensor(
            tensor,
            name=name,
            dtype=torch.float64,
            device=torch.device("cpu"),
            shape=shape,
        )
    if len(
        {
            (str(tensor.device), tensor.untyped_storage().data_ptr())
            for tensor in (
                global_site_rgba_f32,
                global_grad_site_rgba_f32,
                background_rgb_f32,
                grad_positions0_f64_cpu,
                grad_velocities_f64_cpu,
                grad_weight_coefficients_f64_cpu,
            )
        }
    ) != 6:
        raise ValueError("material snapshot/background/gradient bars must not alias")
    forward_into, native_vjp, native_callable_digest = _require_native_ops(
        native_ops,
        device=device,
    )
    if (
        not bool(torch.isfinite(global_site_rgba_f32).all().item())
        or bool(torch.any(global_site_rgba_f32[:, :3] < 0.0).item())
        or bool(torch.any(global_site_rgba_f32[:, :3] > 1.0).item())
        or bool(torch.any(global_site_rgba_f32[:, 3] < 0.0).item())
        or not bool(torch.isfinite(background_rgb_f32).all().item())
        or bool(torch.any(background_rgb_f32 < 0.0).item())
        or bool(torch.any(background_rgb_f32 > 1.0).item())
    ):
        raise ValueError("site material/background must be finite and physical")

    global_loss_element_count = int(times.numel()) * track_count * 3
    selected_times_digest = _tensor_content_digest(times)
    receipt_accumulator = _ReceiptAccumulator()
    loss_f32 = torch.zeros((1,), dtype=torch.float32, device=device)
    with torch.no_grad():
        global_grad_site_rgba_f32.zero_()
        grad_positions0_f64_cpu.zero_()
        grad_velocities_f64_cpu.zero_()
        grad_weight_coefficients_f64_cpu.zero_()

    for frame_index in range(int(times.numel())):
        physical_time = float(times[frame_index].item())
        target_frame = torch.as_tensor(
            target_frame_loader(frame_index, physical_time),
            dtype=torch.float32,
            device=device,
        ).detach().contiguous()
        _require_tensor(
            target_frame,
            name="target frame",
            dtype=torch.float32,
            device=device,
            shape=(track_count, 3),
        )
        if (
            not bool(torch.isfinite(target_frame).all().item())
            or bool(torch.any(target_frame < 0.0).item())
            or bool(torch.any(target_frame > 1.0).item())
        ):
            raise ValueError("target frame must be finite RGB in [0,1]")
        if any(
            target_frame.untyped_storage().data_ptr()
            == tensor.untyped_storage().data_ptr()
            for tensor in (
                global_site_rgba_f32,
                global_grad_site_rgba_f32,
                background_rgb_f32,
            )
        ):
            raise ValueError("target frame must own storage distinct from world state")
        target_bytes = _tensor_bytes(target_frame)
        if target_bytes > memory_policy.maximum_target_frame_logical_tensor_bytes:
            raise MemoryError("selected target frame exceeds its logical byte cap")
        receipt_accumulator.target_frame_load_count += 1
        receipt_accumulator.maximum_target_bytes = max(
            receipt_accumulator.maximum_target_bytes,
            target_bytes,
        )

        discovered = tuple(
            discover_kinetic_power_word_at_time(
                sites,
                rays[track_id],
                time=physical_time,
                near=near,
                far=far,
            )
            for track_id in range(track_count)
        )
        frame_cpu_topology_bytes = sum(
            _tensor_bytes(
                result.word.owners,
                result.word.left_cut_ids,
                result.word.right_cut_ids,
                result.boundary_site_pairs,
            )
            for result in discovered
        )
        if (
            frame_cpu_topology_bytes
            > memory_policy.maximum_frame_cpu_topology_logical_tensor_bytes
        ):
            raise MemoryError("fixed-time frame topology exceeds its logical byte cap")
        receipt_accumulator.maximum_cpu_topology_bytes = max(
            receipt_accumulator.maximum_cpu_topology_bytes,
            frame_cpu_topology_bytes,
        )
        receipt_accumulator.discovery_count += len(discovered)
        receipt_accumulator.candidate_line_count += len(discovered) * site_count
        receipt_accumulator.run_count += sum(result.run_count for result in discovered)
        receipt_accumulator.boundary_count += sum(
            result.active_boundary_count for result in discovered
        )

        for block_index, track_start in enumerate(
            range(0, track_count, memory_policy.maximum_tracks_per_native_block)
        ):
            track_end = min(
                track_start + memory_policy.maximum_tracks_per_native_block,
                track_count,
            )
            block_words = discovered[track_start:track_end]
            block = _prepare_fixed_time_native_block(
                block_words,
                rays[track_start:track_end],
                physical_time=physical_time,
                near=near,
                far=far,
            )
            combined_cpu_topology_bytes = (
                frame_cpu_topology_bytes + block["cpu_tensor_bytes"]
            )
            if combined_cpu_topology_bytes > (
                memory_policy.maximum_frame_cpu_topology_logical_tensor_bytes
            ):
                raise MemoryError("fixed-time native block exceeds the CPU topology cap")
            receipt_accumulator.maximum_cpu_topology_bytes = max(
                receipt_accumulator.maximum_cpu_topology_bytes,
                combined_cpu_topology_bytes,
            )
            source_site_ids_i64 = block["source_site_ids_i64"].to(
                device=device
            ).contiguous()
            word_offsets_i32 = block["word_offsets_i32"].to(device=device).contiguous()
            word_owner_i32 = block["word_owner_i32"].to(device=device).contiguous()
            node_lengths_f32 = block["node_lengths_f32"].to(device=device).contiguous()
            config_i32 = block["config_i32"].to(device=device).contiguous()
            config_f32 = torch.tensor(
                (physical_length_epsilon,),
                dtype=torch.float32,
                device=device,
            )
            compact_rgba = global_site_rgba_f32.index_select(
                0,
                source_site_ids_i64,
            ).contiguous()
            row_count = track_end - track_start
            node_chart = torch.empty(
                (row_count, 1, 4),
                dtype=torch.float32,
                device=device,
            )
            returned = forward_into(
                word_offsets_i32,
                word_owner_i32,
                node_lengths_f32,
                compact_rgba,
                config_i32,
                config_f32,
                node_chart,
                track_count=row_count,
                node_count=1,
            )
            if returned is not None:
                raise TypeError("native forward-into must return None")
            prediction, grad_node_chart = _fixed_time_loss_and_node_bar(
                node_chart[:, 0, :],
                target_frame[track_start:track_end],
                background_rgb_f32,
                global_loss_element_count=global_loss_element_count,
            )
            loss_f32.add_(prediction["loss"])
            compact_grad_rgba = torch.zeros_like(compact_rgba)
            grad_node_chart_3d = grad_node_chart[:, None, :].contiguous()
            native_result = native_vjp(
                word_offsets_i32,
                word_owner_i32,
                node_lengths_f32,
                compact_rgba,
                node_chart,
                grad_node_chart_3d,
                compact_grad_rgba,
                config_i32,
                config_f32,
                track_count=row_count,
                node_count=1,
            )
            if not isinstance(native_result, tuple) or len(native_result) != 2:
                raise TypeError("native full VJP must return (material bar, length bar)")
            returned_compact_bar, grad_lengths_f32 = native_result
            if returned_compact_bar is not compact_grad_rgba and not _same_view(
                returned_compact_bar,
                compact_grad_rgba,
            ):
                raise ValueError("native material bar must alias caller scratch")
            _require_tensor(
                grad_lengths_f32,
                name="native fixed-time length bar",
                dtype=torch.float32,
                device=device,
                shape=(1, int(word_owner_i32.numel())),
            )
            device_scratch_bytes = _tensor_bytes(
                source_site_ids_i64,
                word_offsets_i32,
                word_owner_i32,
                node_lengths_f32,
                config_i32,
                config_f32,
                compact_rgba,
                node_chart,
                grad_node_chart,
                grad_node_chart_3d,
                compact_grad_rgba,
                grad_lengths_f32,
                target_frame,
                prediction["prediction"],
                prediction["loss"],
            )
            if (
                device_scratch_bytes
                > memory_policy.maximum_active_device_scratch_logical_tensor_bytes
            ):
                raise MemoryError("active native fixed-time scratch exceeds its cap")
            global_grad_site_rgba_f32.index_add_(
                0,
                source_site_ids_i64,
                compact_grad_rgba,
            )
            if device_completion_fence() is not None:
                raise TypeError("device_completion_fence must return None")
            grad_lengths_cpu_f64 = grad_lengths_f32.detach().to(
                device="cpu",
                dtype=torch.float64,
            ).contiguous()
            d2h_bytes = _tensor_bytes(grad_lengths_cpu_f64)
            if d2h_bytes > memory_policy.maximum_geometry_d2h_logical_tensor_bytes:
                raise MemoryError("fixed-time geometry D2H exceeds its cap")
            if not bool(torch.isfinite(grad_lengths_cpu_f64).all().item()):
                raise FloatingPointError("native fixed-time length bar is nonfinite")

            word_cursor = 0
            block_geometry_rows = 0
            block_owner_margin_evaluations = 0
            block_reverse_interactions = 0
            discovery_row_digests: list[str] = []
            for local_row, discovered_word in enumerate(block_words):
                owners = discovered_word.word.owners
                word_end = word_cursor + discovered_word.run_count
                discovery_digest = _fixed_time_discovery_digest(
                    geometry_generation_id,
                    rays[track_start + local_row],
                    physical_time=physical_time,
                    near=near,
                    far=far,
                    owners=owners,
                    transition_depths=discovered_word.transition_depths,
                )
                geometry = kinetic_p0_fixed_time_physical_length_geometry_vjp(
                    sites,
                    rays[track_start + local_row],
                    time=physical_time,
                    owners=owners,
                    grad_physical_lengths=(
                        grad_lengths_cpu_f64[0, word_cursor:word_end]
                    ),
                    near=near,
                    far=far,
                    fixed_time_owner_discovery_receipt_id=discovery_digest,
                    thresholds=thresholds,
                )
                expected_lengths = block["row_lengths_f64"][local_row]
                if not torch.allclose(
                    geometry.physical_lengths,
                    expected_lengths,
                    rtol=2.0e-12,
                    atol=2.0e-12,
                ):
                    raise ValueError(
                        "fixed-time geometry recompute disagrees with exact discovery"
                    )
                grad_positions0_f64_cpu.add_(geometry.grad_positions0)
                grad_velocities_f64_cpu.add_(geometry.grad_velocities)
                grad_weight_coefficients_f64_cpu.add_(
                    geometry.grad_weight_coefficients
                )
                block_geometry_rows += 1
                block_owner_margin_evaluations += int(
                    geometry.accounting["owner_margin_evaluations"]
                )
                block_reverse_interactions += int(
                    geometry.accounting["physical_length_reverse_interactions"]
                )
                discovery_row_digests.append(discovery_digest)
                word_cursor = word_end
            if word_cursor != int(word_owner_i32.numel()):
                raise ArithmeticError("fixed-time VJP did not consume every owner run")

            block_digest = _digest_parts(
                STEP_PROVENANCE,
                "fixed-time-native-block-receipt",
                frame_index,
                physical_time,
                block_index,
                track_start,
                track_end,
                tuple(discovery_row_digests),
                int(word_owner_i32.numel()),
                int(source_site_ids_i64.numel()),
                device_scratch_bytes,
                d2h_bytes,
                block_geometry_rows,
                block_reverse_interactions,
                block_owner_margin_evaluations,
                device_completion_fence_provenance,
                native_callable_digest,
            )
            receipt_accumulator.add_digest(block_digest)
            receipt_accumulator.native_block_count += 1
            receipt_accumulator.forward_count += 1
            receipt_accumulator.forward_run_interactions += int(
                word_cursor
            )
            receipt_accumulator.vjp_count += 1
            receipt_accumulator.vjp_run_interactions += int(word_cursor)
            receipt_accumulator.fence_count += 1
            receipt_accumulator.d2h_count += 1
            receipt_accumulator.geometry_row_count += block_geometry_rows
            receipt_accumulator.reverse_interactions += block_reverse_interactions
            receipt_accumulator.owner_margin_evaluations += (
                block_owner_margin_evaluations
            )
            receipt_accumulator.material_scatter_rows += int(
                source_site_ids_i64.numel()
            )
            receipt_accumulator.maximum_device_scratch_bytes = max(
                receipt_accumulator.maximum_device_scratch_bytes,
                device_scratch_bytes,
            )
            receipt_accumulator.maximum_d2h_bytes = max(
                receipt_accumulator.maximum_d2h_bytes,
                d2h_bytes,
            )
            del (
                source_site_ids_i64,
                word_offsets_i32,
                word_owner_i32,
                node_lengths_f32,
                config_i32,
                config_f32,
                compact_rgba,
                node_chart,
                grad_node_chart,
                grad_node_chart_3d,
                compact_grad_rgba,
                grad_lengths_f32,
                grad_lengths_cpu_f64,
                prediction,
                block,
            )
        del discovered, target_frame
        receipt_accumulator.target_frame_release_count += 1
        receipt_accumulator.frame_scratch_release_count += 1

    if not bool(torch.isfinite(loss_f32).all().item()) or not bool(
        torch.isfinite(global_grad_site_rgba_f32).all().item()
    ):
        raise FloatingPointError("sequential fixed-time material result is nonfinite")
    receipt = _build_step_receipt(
        receipt_accumulator,
        selected_times_manifest_digest=selected_times_digest,
        native_callable_identity_digest=native_callable_digest,
        requested_frame_count=int(times.numel()),
        track_count=track_count,
        global_site_count=site_count,
    )
    material_grad_l2 = float(
        torch.linalg.vector_norm(global_grad_site_rgba_f32).item()
    )
    position_grad_l2 = float(
        torch.linalg.vector_norm(grad_positions0_f64_cpu).item()
    )
    velocity_grad_l2 = float(
        torch.linalg.vector_norm(grad_velocities_f64_cpu).item()
    )
    weight_grad_l2 = float(
        torch.linalg.vector_norm(grad_weight_coefficients_f64_cpu).item()
    )
    accounting = MappingProxyType(
        {
            **dict(receipt.accounting()),
            "step_index": step_index,
            "full_geometry": True,
            "fixed_camera": True,
            "camera_ray_gradients_enabled": False,
            "loss_normalization_id": LOSS_NORMALIZATION_ID,
            "global_loss_element_count": global_loss_element_count,
            "streamed_sample_count": int(times.numel()) * track_count,
            "fixed_time_direct_sample_evaluation_count": (
                int(times.numel()) * track_count
            ),
            "sample_to_node_interaction_count": int(times.numel()) * track_count,
            "native_sample_launch_count": 0,
            "material_grad_l2": material_grad_l2,
            "position_grad_l2": position_grad_l2,
            "velocity_grad_l2": velocity_grad_l2,
            "weight_grad_l2": weight_grad_l2,
            "material_grad_nonzero": material_grad_l2 > 0.0,
            "position_grad_nonzero": position_grad_l2 > 0.0,
            "velocity_grad_nonzero": velocity_grad_l2 > 0.0,
            "weight_grad_nonzero": weight_grad_l2 > 0.0,
            "optimizer_update_authorization_count": 1,
            "optimizer_update_callback_contract": "single_manual_sgd_call",
            "sequential_frame_release_required": True,
            "same_representation_native_p0_replay": True,
            "world_side_work_scaling": "O(F * P * (S log S + native_word_reverse))",
            "peak_live_frame_scaling": (
                "O(S + P + F scalar times + max_one_frame_block)"
            ),
            "selected_time_grid_tensor_bytes": _tensor_bytes(times),
            "selected_time_grid_scaling": "O(F) float64 camera-time scalars",
            "expensive_topology_and_reverse_peak_scaling": (
                "O(S + P + max_one_frame_block), excluding the O(F) scalar time grid"
            ),
            "continuous_compiler_invocation_count": 0,
            "native_runtime_verified": False,
            "allocator_peak_measured": False,
        }
    )
    tensors = (
        loss_f32,
        global_grad_site_rgba_f32,
        grad_positions0_f64_cpu,
        grad_velocities_f64_cpu,
        grad_weight_coefficients_f64_cpu,
    )
    provisional = PaperKineticSequentialFixedTimeFullGeometryStepResult(
        step_index=step_index,
        geometry_generation_id=geometry_generation_id,
        material_generation_id=material_generation_id,
        background_generation_id=background_generation_id,
        target_generation_id=target_generation_id,
        loss_f32=loss_f32,
        grad_global_site_rgba_f32=global_grad_site_rgba_f32,
        grad_positions0_f64_cpu=grad_positions0_f64_cpu,
        grad_velocities_f64_cpu=grad_velocities_f64_cpu,
        grad_weight_coefficients_f64_cpu=grad_weight_coefficients_f64_cpu,
        receipt=receipt,
        accounting=accounting,
        generation_digest="",
        _tensor_signatures=tuple(_tensor_signature(tensor) for tensor in tensors),
        _seal=_RESULT_SEAL,
    )
    result = PaperKineticSequentialFixedTimeFullGeometryStepResult(
        **{
            **provisional.__dict__,
            "generation_digest": _result_digest(provisional),
        }
    )
    result.assert_current()
    optimizer_update(result)
    result.assert_current()
    return result


def _prepare_fixed_time_native_block(
    discovered_words: tuple[Any, ...],
    rays_f64: torch.Tensor,
    *,
    physical_time: float,
    near: float,
    far: float,
) -> dict[str, Any]:
    global_owners = tuple(
        tuple(int(owner) for owner in result.word.owners.tolist())
        for result in discovered_words
    )
    source_ids = tuple(sorted({owner for row in global_owners for owner in row}))
    compact = {source_id: index for index, source_id in enumerate(source_ids)}
    offsets = [0]
    compact_owners: list[int] = []
    row_lengths = []
    for result, ray, owners in zip(
        discovered_words,
        rays_f64,
        global_owners,
        strict=True,
    ):
        compact_owners.extend(compact[owner] for owner in owners)
        offsets.append(len(compact_owners))
        cuts = (near, *(float(value) for value in result.transition_depths), far)
        direction = ray[6:9] + physical_time * ray[9:12]
        speed = torch.linalg.vector_norm(direction)
        lengths = speed * torch.tensor(
            tuple(
                right - left
                for left, right in zip(cuts[:-1], cuts[1:], strict=True)
            ),
            dtype=torch.float64,
        )
        if tuple(lengths.shape) != (result.run_count,) or bool(
            torch.any(lengths <= 0).item()
        ):
            raise ValueError("exact fixed-time discovery emitted invalid run lengths")
        row_lengths.append(lengths.contiguous())
    tensors = {
        "source_site_ids_i64": torch.tensor(source_ids, dtype=torch.int64),
        "word_offsets_i32": torch.tensor(offsets, dtype=torch.int32),
        "word_owner_i32": torch.tensor(compact_owners, dtype=torch.int32),
        "node_lengths_f32": torch.cat(row_lengths).to(dtype=torch.float32)[
            None, :
        ].contiguous(),
        "config_i32": torch.tensor(
            (
                len(discovered_words),
                1,
                len(source_ids),
                len(compact_owners),
            ),
            dtype=torch.int32,
        ),
        "row_lengths_f64": tuple(row_lengths),
    }
    if not bool(torch.isfinite(tensors["node_lengths_f32"]).all().item()) or bool(
        torch.any(tensors["node_lengths_f32"] <= 0).item()
    ):
        raise ValueError("fixed-time lengths are invalid after float32 lowering")
    tensors["cpu_tensor_bytes"] = _tensor_bytes(
        tensors["source_site_ids_i64"],
        tensors["word_offsets_i32"],
        tensors["word_owner_i32"],
        tensors["node_lengths_f32"],
        tensors["config_i32"],
        *row_lengths,
    )
    return tensors


def _fixed_time_loss_and_node_bar(
    node_chart_f32: torch.Tensor,
    target_rgb_f32: torch.Tensor,
    background_rgb_f32: torch.Tensor,
    *,
    global_loss_element_count: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    kappa = node_chart_f32[:, 0]
    velocity = node_chart_f32[:, 1:]
    phi, phi_prime = _phi_and_derivative_f32(kappa)
    beta = torch.exp(-kappa)
    prediction = phi[:, None] * velocity + beta[:, None] * background_rgb_f32
    difference = prediction - target_rgb_f32
    loss = difference.square().sum().reshape(1) / float(global_loss_element_count)
    grad_prediction = 2.0 * difference / float(global_loss_element_count)
    grad_beta = torch.sum(grad_prediction * background_rgb_f32, dim=1)
    grad_kappa = -beta * grad_beta + phi_prime * torch.sum(
        velocity * grad_prediction,
        dim=1,
    )
    grad_velocity = phi[:, None] * grad_prediction
    return {"prediction": prediction, "loss": loss}, torch.cat(
        (grad_kappa[:, None], grad_velocity),
        dim=1,
    ).contiguous()


def _phi_and_derivative_f32(
    kappa: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    small = kappa.abs() < 1.0e-4
    k2 = kappa * kappa
    k3 = k2 * kappa
    k4 = k3 * kappa
    k5 = k4 * kappa
    k6 = k5 * kappa
    series = (
        1.0
        - kappa / 2.0
        + k2 / 6.0
        - k3 / 24.0
        + k4 / 120.0
        - k5 / 720.0
        + k6 / 5040.0
    )
    series_prime = (
        -0.5
        + kappa / 3.0
        - k2 / 8.0
        + k3 / 30.0
        - k4 / 144.0
        + k5 / 840.0
    )
    safe_kappa = torch.where(small, torch.ones_like(kappa), kappa)
    numerator = -torch.expm1(-kappa)
    direct = numerator / safe_kappa
    direct_prime = (
        safe_kappa * torch.exp(-kappa) - numerator
    ) / safe_kappa.square()
    return torch.where(small, series, direct), torch.where(
        small,
        series_prime,
        direct_prime,
    )


def _build_step_receipt(
    accumulator: _ReceiptAccumulator,
    *,
    selected_times_manifest_digest: str,
    native_callable_identity_digest: str,
    requested_frame_count: int,
    track_count: int,
    global_site_count: int,
) -> PaperKineticSequentialFixedTimeStepReceipt:
    values = dict(
        selected_times_manifest_digest=selected_times_manifest_digest,
        fixed_time_receipt_chain_digest=accumulator.chain.hexdigest(),
        native_callable_identity_digest=native_callable_identity_digest,
        requested_frame_count=requested_frame_count,
        track_count=track_count,
        global_site_count=global_site_count,
        streamed_sample_count=requested_frame_count * track_count,
        global_loss_element_count=requested_frame_count * track_count * 3,
        selected_time_grid_tensor_bytes=requested_frame_count * 8,
        fixed_time_direct_sample_evaluation_count=requested_frame_count * track_count,
        sample_to_node_interaction_count=requested_frame_count * track_count,
        native_sample_launch_count=0,
        target_frame_load_count=accumulator.target_frame_load_count,
        target_frame_release_count=accumulator.target_frame_release_count,
        frame_scratch_release_count=accumulator.frame_scratch_release_count,
        fixed_time_lower_envelope_discovery_call_count=accumulator.discovery_count,
        candidate_line_evaluation_count=accumulator.candidate_line_count,
        discovered_owner_run_count=accumulator.run_count,
        discovered_active_boundary_count=accumulator.boundary_count,
        native_block_count=accumulator.native_block_count,
        native_forward_launch_count=accumulator.forward_count,
        native_forward_ordered_run_interaction_count=(
            accumulator.forward_run_interactions
        ),
        native_full_vjp_launch_count=accumulator.vjp_count,
        native_vjp_ordered_run_interaction_count=(
            accumulator.vjp_run_interactions
        ),
        completion_fence_call_count=accumulator.fence_count,
        geometry_length_bar_d2h_count=accumulator.d2h_count,
        geometry_row_vjp_call_count=accumulator.geometry_row_count,
        physical_length_reverse_interaction_count=accumulator.reverse_interactions,
        all_site_owner_margin_evaluation_count=(
            accumulator.owner_margin_evaluations
        ),
        material_compact_to_global_scatter_row_count=(
            accumulator.material_scatter_rows
        ),
        maximum_target_frame_logical_tensor_bytes=accumulator.maximum_target_bytes,
        maximum_frame_cpu_topology_logical_tensor_bytes=(
            accumulator.maximum_cpu_topology_bytes
        ),
        maximum_active_device_scratch_logical_tensor_bytes=(
            accumulator.maximum_device_scratch_bytes
        ),
        maximum_geometry_d2h_logical_tensor_bytes=accumulator.maximum_d2h_bytes,
    )
    provisional = PaperKineticSequentialFixedTimeStepReceipt(
        **values,
        generation_digest="",
        _seal=_RECEIPT_SEAL,
    )
    result = PaperKineticSequentialFixedTimeStepReceipt(
        **values,
        generation_digest=_step_receipt_digest(provisional),
        _seal=_RECEIPT_SEAL,
    )
    result.assert_current()
    return result


def _step_receipt_digest(
    receipt: PaperKineticSequentialFixedTimeStepReceipt,
) -> str:
    return _digest_parts(
        receipt.provenance,
        *(
            value
            for name, value in receipt.__dict__.items()
            if name not in {"generation_digest", "_seal", "provenance"}
        ),
    )


def _result_digest(
    result: PaperKineticSequentialFixedTimeFullGeometryStepResult,
) -> str:
    return _digest_parts(
        result.provenance,
        result.runtime_status,
        result.step_index,
        result.geometry_generation_id,
        result.material_generation_id,
        result.background_generation_id,
        result.target_generation_id,
        result.loss_normalization_id,
        result.receipt.generation_digest,
        tuple(sorted(result.accounting.items())),
        result._tensor_signatures,
        result.optimizer_update_authorization_count,
    )


def paper_kinetic_sequential_fixed_time_geometry_generation_id(
    sites: AffineKineticPowerSites,
    ray_coefficients_f64_cpu: torch.Tensor,
    *,
    near: float,
    far: float,
) -> str:
    """Derive the only accepted immutable geometry/ray generation."""

    if not isinstance(sites, AffineKineticPowerSites):
        raise TypeError("sites must be AffineKineticPowerSites")
    rays = _cpu_f64(ray_coefficients_f64_cpu, name="ray_coefficients_f64_cpu")
    if rays.ndim != 2 or rays.shape[1] != 12 or rays.shape[0] < 1:
        raise ValueError("ray_coefficients_f64_cpu must have shape [P,12]")
    if not math.isfinite(near) or not math.isfinite(far) or far <= near:
        raise ValueError("near/far must be finite with near < far")
    return _geometry_generation_id_from_normalized(
        sites,
        rays,
        near=near,
        far=far,
    )


def _geometry_generation_id_from_normalized(
    sites: AffineKineticPowerSites,
    rays: torch.Tensor,
    *,
    near: float,
    far: float,
) -> str:
    return _digest_parts(
        STEP_PROVENANCE,
        "fixed-time-world-geometry-generation",
        _tensor_content_digest(sites.positions0),
        _tensor_content_digest(sites.velocities),
        _tensor_content_digest(sites.weight_coefficients),
        _tensor_content_digest(rays),
        near,
        far,
    )


def _fixed_time_discovery_digest(
    geometry_generation_id: str,
    ray: torch.Tensor,
    *,
    physical_time: float,
    near: float,
    far: float,
    owners: torch.Tensor,
    transition_depths: tuple[Any, ...],
) -> str:
    return _digest_parts(
        STEP_PROVENANCE,
        "exact-fixed-time-lower-envelope",
        geometry_generation_id,
        _tensor_content_digest(ray.contiguous()),
        physical_time,
        near,
        far,
        tuple(int(value) for value in owners.tolist()),
        tuple((value.numerator, value.denominator) for value in transition_depths),
    )


def _require_native_ops(
    native_ops: Any,
    *,
    device: torch.device,
) -> tuple[Callable[..., Any], Callable[..., Any], str]:
    forward = getattr(native_ops, FORWARD_INTO_OP_NAME, None)
    vjp = getattr(native_ops, VJP_OP_NAME, None)
    if not callable(forward) or not callable(vjp):
        raise RuntimeError(
            "sequential fixed-time replay requires native precompiled-length "
            "forward-into and full-VJP ops"
        )
    if device.type == "mps":
        attestation = getattr(
            native_ops,
            "assert_kinetic_memory_light_compiled_abi_registered",
            None,
        )
        if not callable(attestation):
            raise RuntimeError("MPS native ops lack compiled ABI attestation")
        attestation()
    identity = _digest_parts(
        STEP_PROVENANCE,
        type(native_ops).__module__,
        type(native_ops).__qualname__,
        FORWARD_INTO_OP_NAME,
        id(forward),
        VJP_OP_NAME,
        id(vjp),
    )
    return forward, vjp, identity


def _cpu_f64(value: torch.Tensor, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float64, device="cpu").detach().clone()
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} must be finite")
    return tensor.contiguous()


def _require_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    dtype: torch.dtype,
    device: torch.device,
    shape: tuple[int, ...],
) -> None:
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.dtype != dtype
        or tensor.device != device
        or tensor.layout != torch.strided
        or tuple(tensor.shape) != shape
        or not tensor.is_contiguous()
        or tensor.requires_grad
    ):
        raise ValueError(f"{name} has invalid dtype/device/layout/shape")


def _tensor_signature(tensor: torch.Tensor) -> tuple[object, ...]:
    return (
        id(tensor),
        int(getattr(tensor, "_version", 0)),
        int(tensor.untyped_storage().data_ptr()),
        int(tensor.storage_offset()),
        tuple(int(value) for value in tensor.shape),
        tuple(int(value) for value in tensor.stride()),
        tensor.dtype,
        tensor.device,
        tensor.layout,
    )


def _tensor_content_digest(tensor: torch.Tensor) -> str:
    if tensor.device.type != "cpu" or not tensor.is_contiguous():
        raise ValueError("tensor content digest requires contiguous CPU storage")
    return hashlib.sha256(tensor.detach().numpy().tobytes(order="C")).hexdigest()


def _tensor_bytes(*tensors: torch.Tensor) -> int:
    return sum(int(tensor.numel()) * int(tensor.element_size()) for tensor in tensors)


def _same_view(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        isinstance(left, torch.Tensor)
        and left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
        and left.storage_offset() == right.storage_offset()
        and tuple(left.shape) == tuple(right.shape)
        and tuple(left.stride()) == tuple(right.stride())
    )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _digest_parts(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = repr(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


__all__ = (
    "LOSS_NORMALIZATION_ID",
    "PaperKineticSequentialFixedTimeFullGeometryStepResult",
    "PaperKineticSequentialFixedTimeMemoryPolicy",
    "PaperKineticSequentialFixedTimeStepReceipt",
    "paper_kinetic_sequential_fixed_time_geometry_generation_id",
    "run_paper_kinetic_sequential_fixed_time_full_geometry_step",
)
