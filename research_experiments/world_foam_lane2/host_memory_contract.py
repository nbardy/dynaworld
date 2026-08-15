"""Source-only host-memory accounting for the streamed WorldFoam lifecycle.

This module intentionally separates three different claims which are easy to
blur together:

* the expensive compiled replay state should depend on ``B_p`` and chart rank,
  not on the requested frame count;
* cheap sample identities/times may remain linear in ``F`` and target/ray
  payloads may remain ``B_p x K``;
* the current strict CPU proof oracle and global atlas template are *not* yet
  bounded by that runtime contract.

The byte counts below are exact tensor-payload counts for the current dtypes
and layouts.  Python container/object overhead and allocator bookkeeping are
excluded.  The strict-certificate count is deliberately only a lower bound:
it counts tuple pointer slots in the dense forward-mode construction, but not
the much larger ``Fraction``, ``_Interval``, ``_Dual``, or tuple objects.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal


@dataclass(frozen=True)
class HostMemoryAllocation:
    """One tensor-payload allocation or certified lower bound."""

    name: str
    bytes: int
    lifetime: str
    dependence: str
    status: Literal[
        "bounded",
        "cheap_linear",
        "unbounded",
        "optional",
        "partial_lower_bound",
    ]
    exact_tensor_payload: bool = True


@dataclass(frozen=True)
class WorldFoamHostMemoryDimensions:
    """Dimensions needed to audit one fixed-topology compact block.

    ``global_*`` describes the logical step/template. ``global_track_count``
    counts calibrated ``(view,pixel)`` tracks, not unique pixel coordinates;
    callers with ``V`` selected views and ``P`` pixels must pass ``V*P``.
    ``block_*`` describes
    the one spatial block retained by a native launch.  ``chart_node_counts``
    is the selected rank of every fixed-topology time chart.  The formulas
    assume the current CPU reference dtypes: float64 compiled atlases and
    int64 reference words/incidences, with float32 target/ray staging.
    """

    global_track_count: int
    global_sample_count: int
    global_site_count: int
    global_word_count: int
    global_incidence_count: int
    block_track_count: int
    sample_block_size: int
    block_site_count: int
    block_boundary_count: int
    block_word_count: int
    block_incidence_count: int
    chart_node_counts: tuple[int, ...]
    plan_pixel_count: int
    plan_sample_count: int
    image_height: int
    image_width: int
    compiled_cpu_artifact_store_max_resident_accounted_bytes: int
    spatial_block_count: int = 1
    native_topology_cache_max_entries: int = 1
    source_target_resident_bytes: int = 0
    binding_payload_bytes: int = 0
    weight_coefficient_count: int = 3
    trainable_camera_ray_count: int = 0

    def __post_init__(self) -> None:
        positive = (
            self.global_track_count,
            self.global_sample_count,
            self.global_site_count,
            self.global_word_count,
            self.block_track_count,
            self.sample_block_size,
            self.block_site_count,
            self.block_word_count,
            self.plan_pixel_count,
            self.plan_sample_count,
            self.image_height,
            self.image_width,
            self.compiled_cpu_artifact_store_max_resident_accounted_bytes,
            self.spatial_block_count,
            self.weight_coefficient_count,
        )
        if any(value < 1 for value in positive):
            raise ValueError("track/sample/site/word/image dimensions must be positive")
        nonnegative = (
            self.global_incidence_count,
            self.block_boundary_count,
            self.block_incidence_count,
            self.source_target_resident_bytes,
            self.binding_payload_bytes,
            self.native_topology_cache_max_entries,
            self.trainable_camera_ray_count,
        )
        if any(value < 0 for value in nonnegative):
            raise ValueError("incidence/boundary/external byte counts must be nonnegative")
        if not self.chart_node_counts or any(rank < 1 for rank in self.chart_node_counts):
            raise ValueError("chart_node_counts must contain positive ranks")
        if self.block_track_count > self.global_track_count:
            raise ValueError("block_track_count cannot exceed global_track_count")
        if self.block_site_count > self.global_site_count:
            raise ValueError("block_site_count cannot exceed global_site_count")
        if self.weight_coefficient_count > 3:
            raise ValueError("weight_coefficient_count cannot exceed the kinetic degree-2 ABI")
        if self.trainable_camera_ray_count > self.global_track_count:
            raise ValueError(
                "trainable_camera_ray_count cannot exceed global (view,pixel) track count"
            )

    @property
    def chart_count(self) -> int:
        return len(self.chart_node_counts)

    @property
    def total_node_count(self) -> int:
        return sum(self.chart_node_counts)

    @property
    def maximum_node_count(self) -> int:
        return max(self.chart_node_counts)

    @property
    def continuous_certificate_parameter_count(self) -> int:
        """Dense canonical-world derivative dimension used by the proof oracle."""

        return (
            5 * self.block_boundary_count
            + 12 * self.block_track_count
            + 4 * self.block_incidence_count
            + 4 * self.block_site_count
        )

    @property
    def global_kinetic_geometry_parameter_tensor_bytes(self) -> int:
        """Float64 position, velocity, and weight-polynomial coefficients."""

        return (
            8
            * self.global_site_count
            * (6 + self.weight_coefficient_count)
        )

    @property
    def global_source_parameter_tensor_bytes(self) -> int:
        """Initialized direct-kinetic world plus one physical RGBA seed.

        The lazy fixed-camera path derives affine ray coefficients inside a
        bounded spatial program block.  It therefore owns no global ``[P,12]``
        ray tensor.  Those bounded block rays are counted by
        :attr:`training_binding_tensor_bytes_per_block` instead.
        """

        return self.global_kinetic_geometry_parameter_tensor_bytes + 16 * self.global_site_count

    @property
    def global_material_step_accumulator_tensor_bytes(self) -> int:
        """Physical float32 RGBA cotangent plus the scalar float32 loss."""

        return 16 * self.global_site_count + 4

    @property
    def global_fixed_camera_geometry_gradient_tensor_bytes(self) -> int:
        """Material and direct-kinetic geometry bars with fixed cameras."""

        return (
            self.global_material_step_accumulator_tensor_bytes
            + 8
            * self.global_site_count
            * (6 + self.weight_coefficient_count)
        )

    @property
    def global_gradient_tensor_bytes(self) -> int:
        """Full direct-kinetic bars, with camera bars only when requested.

        Camera calibration is a nondefault mode.  Its affine ray cotangent is
        float64 ``[P_bar,12]``; fixed-camera training sets ``P_bar=0``.
        """

        return (
            self.global_fixed_camera_geometry_gradient_tensor_bytes
            + 96 * self.trainable_camera_ray_count
        )

    @property
    def global_decoded_material_and_raw_chain_gradient_tensor_bytes(self) -> int:
        """Exact five-tensor fixed-site material-training state.

        This is raw color/density, decoded physical RGBA, and raw
        color/density scratch gradients: twelve float32 scalars per site.
        """

        return 48 * self.global_site_count

    @property
    def global_material_training_state_plus_geometry_tensor_bytes(self) -> int:
        """Steady-state material state plus shared direct-kinetic geometry.

        This is ``120*S`` bytes for the degree-2 ABI after the external cold
        initializer and its physical material seed have been released.  It
        excludes the physical RGBA cotangent accumulated during an active
        optimizer step.
        """

        return (
            self.global_kinetic_geometry_parameter_tensor_bytes
            + self.global_decoded_material_and_raw_chain_gradient_tensor_bytes
        )

    @property
    def global_material_training_step_peak_base_tensor_bytes(self) -> int:
        """Steady state plus the live physical RGBA/loss step accumulator."""

        return (
            self.global_material_training_state_plus_geometry_tensor_bytes
            + self.global_material_step_accumulator_tensor_bytes
        )

    @property
    def global_template_tensor_bytes(self) -> int:
        """Unique tensor payload retained by the current full-P CPU template.

        For each chart this includes node times, the fit matrix, both
        ``[P,J,4]`` float64 arrays, int64 incidence, and float64 sparse depth
        coefficients.  The int64 owner/cut word tensors are shared by charts
        and counted once.
        """

        ranks = self.chart_node_counts
        return (
            64 * self.global_track_count * sum(ranks)
            + 8 * sum(rank + rank * rank for rank in ranks)
            + 48 * self.chart_count * self.global_incidence_count
            + 24 * self.global_word_count
        )

    @property
    def compact_topology_tensor_bytes(self) -> int:
        """CPU ``PreparedWorldFoamTrackBlock`` payload, including source ids."""

        return (
            16 * self.block_track_count
            + 8
            + 12 * self.block_word_count
            + 4 * self.block_incidence_count
            + 16 * self.block_boundary_count
            + 8 * self.block_site_count
        )

    @property
    def compact_world_and_atlas_tensor_bytes(self) -> int:
        """Detached compact world plus its fixed-topology float64 atlas."""

        ranks = self.chart_node_counts
        compact_world = 72 * self.block_site_count + 96 * self.block_track_count + 40 * self.block_boundary_count
        compact_atlas = (
            64 * self.block_track_count * sum(ranks)
            + 8 * sum(rank + rank * rank for rank in ranks)
            + 48 * self.chart_count * self.block_incidence_count
            + 24 * self.block_word_count
        )
        return compact_world + compact_atlas

    @property
    def staged_target_and_ray_tensor_bytes(self) -> int:
        """One float32 target ``[B_p,K,3]`` plus ray ``[B_p,K,6]`` block."""

        return 36 * self.block_track_count * self.sample_block_size

    @property
    def staged_target_tensor_bytes(self) -> int:
        """One target-only float32 ``[B_p,K,3]`` material-training block."""

        return 12 * self.block_track_count * self.sample_block_size

    @property
    def native_node_state_and_bar_tensor_bytes(self) -> int:
        """One fixed-word block's float32 node state plus its cotangent.

        This ``32*B_p*J_max`` formula is valid only when charts/blocks execute
        sequentially.  It must not be used for the row-ragged equal-rank
        bridge unless target replay or another schedule proves that only one
        native block is live.  A one-pass cross-``K`` bundle instead retains
        ``32*sum_b(R_b*J_b)`` bytes across its active heterogeneous blocks.
        """

        return 32 * self.block_track_count * self.maximum_node_count

    @property
    def native_geometry_length_bar_tensor_bytes(self) -> int:
        """Optional geometry reverse output ``float32[J_max,W_block]``."""

        return 4 * self.maximum_node_count * self.block_word_count

    @property
    def native_material_geometry_length_bar_tensor_bytes(self) -> int:
        """Material-only kinetic reverse now allocates no geometry length bar."""

        return 0

    @property
    def native_sample_payload_peak_tensor_bytes(self) -> int:
        """Float32 target/ray ``B_p x K`` peak for strict/evaluation replay.

        Strict replay retains explicit origin/direction rays for independent
        evaluation validation.  Its accumulate-only native ABI does not
        allocate or write a discarded prediction tensor.
        """

        return 36 * self.block_track_count * self.sample_block_size

    @property
    def native_material_sample_payload_peak_tensor_bytes(self) -> int:
        """Target-only float32 ``B_p x K`` payload for material training."""

        return self.staged_target_tensor_bytes

    @property
    def native_optional_prediction_tensor_bytes(self) -> int:
        """Additional float32 ``[B_p,K,3]`` payload for forward media/eval."""

        return 12 * self.block_track_count * self.sample_block_size

    @property
    def native_sample_weight_tensor_bytes(self) -> int:
        """One row-ragged float32 ``[N,J_max]`` interpolation-weight block."""

        return 4 * self.maximum_materialized_sample_count * self.maximum_node_count

    @property
    def maximum_materialized_sample_count(self) -> int:
        """Maximum flattened samples ``N=B_p*K`` in one ragged block."""

        return self.block_track_count * self.sample_block_size

    @property
    def native_sample_identity_tensor_bytes(self) -> int:
        """CPU int32 row ids plus CPU int64 flat sample ids for ``N`` rows."""

        return 12 * self.maximum_materialized_sample_count

    @property
    def native_materialized_sample_block_tensor_bytes(self) -> int:
        """Retained ragged launch block: row ids, weights, targets, flat ids."""

        return (
            self.native_sample_weight_tensor_bytes
            + self.native_material_sample_payload_peak_tensor_bytes
            + self.native_sample_identity_tensor_bytes
        )

    @property
    def native_prepared_sample_public_scratch_tensor_bytes(self) -> int:
        """Public native-prepare scratch: device int32 rows plus two configs."""

        return 4 * self.maximum_materialized_sample_count + 20

    @property
    def native_synchronized_material_sample_launch_tensor_bytes(self) -> int:
        """Selected material launch block plus its coexisting public scratch.

        For ``N <= B_p*K`` and rank ``J_max`` this is exactly
        ``4*N*J_max + 28*N + 20`` source-visible bytes.  Allocator, driver,
        command-buffer, and private-kernel storage remain outside this count.
        """

        return (
            self.native_materialized_sample_block_tensor_bytes
            + self.native_prepared_sample_public_scratch_tensor_bytes
        )

    @property
    def native_sample_time_tensor_bytes(self) -> int:
        """One CPU float64 ``[K]`` time block used to construct weights.

        The native replay state no longer owns a global ``[F]`` time copy or a
        chart-local ``[F_c]`` clone. Times have the same bounded lifetime as
        their target block.
        """

        return 8 * self.sample_block_size

    @property
    def native_compact_live_world_and_bar_tensor_bytes(self) -> int:
        """Narrow native ABI world/bar subset, not the adapter's total peak."""

        return 72 * self.block_site_count + 48 * self.block_track_count

    @property
    def native_topology_cache_tensor_bytes_per_block(self) -> int:
        """Device payload retained by one sealed material-topology token.

        The native token owns the CSR arrays but not the CPU-only source
        boundary ids.  The sealed Python token additionally retains int64
        source site and track ids for compact gathers.  This is therefore one
        ``PreparedWorldFoamTrackBlock`` payload minus ``source_boundary_ids``.
        """

        return self.compact_topology_tensor_bytes - 8 * self.block_boundary_count

    @property
    def native_topology_cache_tensor_bytes(self) -> int:
        """Uniform-block estimate for the explicitly bounded device-token LRU."""

        return (
            min(self.spatial_block_count, self.native_topology_cache_max_entries)
            * self.native_topology_cache_tensor_bytes_per_block
        )

    @property
    def training_binding_tensor_bytes_per_block(self) -> int:
        """Legacy compact P0 binding's exact private CPU tensor payload.

        The legacy binding clones compact topology, float32 site geometry and
        affine rays, node times, fit matrices, and barycentric weights.  It is
        not the current direct-kinetic ragged artifact-store accounting row.
        """

        schedule = 8 * sum(rank * rank + 2 * rank for rank in self.chart_node_counts)
        return (
            self.compact_topology_tensor_bytes
            + 20 * self.block_site_count
            + 48 * self.block_track_count
            + schedule
        )

    @property
    def unique_schedule_tensor_bytes(self) -> int:
        """One shared CPU node-time/fit/barycentric schedule payload."""

        return 8 * sum(rank * rank + 2 * rank for rank in self.chart_node_counts)

    @property
    def native_adapter_sample_peak_dominant_tensor_bytes(self) -> int:
        """Audited dominant live payload near one loss-only sample launch.

        This includes gathered materials, source ids, duplicated affine ray
        coefficients, native topology, boundary/Mobius values and bars, node
        state/bars, target/validation rays, bounded sample times, and sample
        weights. Small config, diagnostic, Python, command-buffer, and allocator
        payloads are deliberately excluded, so this is a lower bound rather
        than a peak guarantee.
        """

        return (
            76 * self.block_site_count
            + 112 * self.block_track_count
            + 60 * self.block_boundary_count
            + 36 * self.block_incidence_count
            + 12 * self.block_word_count
            + 32 * self.block_track_count * self.maximum_node_count
            + 36 * self.block_track_count * self.sample_block_size
            + self.native_sample_weight_tensor_bytes
            + self.native_sample_identity_tensor_bytes
            + self.native_prepared_sample_public_scratch_tensor_bytes
            + 8 * self.sample_block_size
        )

    @property
    def native_adapter_finalize_peak_dominant_tensor_bytes(self) -> int:
        """Audited dominant payload when the compact site gradient is emitted."""

        return (
            96 * self.block_site_count
            + 64 * self.block_track_count
            + 60 * self.block_boundary_count
            + 36 * self.block_incidence_count
            + 12 * self.block_word_count
        )

    @property
    def native_material_adapter_sample_peak_dominant_tensor_bytes(self) -> int:
        """Audited dominant sample-phase payload for material-only reverse.

        Material training freezes geometry, so the native reverse never
        allocates Mobius or boundary bars.  It retains exact ordered-transfer
        state, node cotangents, loss, and the compact RGBA bar.
        """

        return (
            76 * self.block_site_count
            + 112 * self.block_track_count
            + 40 * self.block_boundary_count
            + 20 * self.block_incidence_count
            + 12 * self.block_word_count
            + 32 * self.block_track_count * self.maximum_node_count
            + 12 * self.block_track_count * self.sample_block_size
            + self.native_sample_weight_tensor_bytes
            + self.native_sample_identity_tensor_bytes
            + self.native_prepared_sample_public_scratch_tensor_bytes
            + 8 * self.sample_block_size
        )

    @property
    def native_material_adapter_finalize_peak_dominant_tensor_bytes(self) -> int:
        """Audited material-only finalization payload without geometry bars."""

        return (
            76 * self.block_site_count
            + 64 * self.block_track_count
            + 40 * self.block_boundary_count
            + 20 * self.block_incidence_count
            + 12 * self.block_word_count
        )

    @property
    def native_audited_dominant_peak_tensor_payload_lower_bound_bytes(self) -> int:
        """Larger audited dominant phase; explicitly not measured total peak.

        A real upper bound still needs the rebuilt runtime's allocator and
        command-buffer telemetry.
        """

        return max(
            self.native_adapter_sample_peak_dominant_tensor_bytes,
            self.native_adapter_finalize_peak_dominant_tensor_bytes,
        )

    @property
    def native_material_audited_dominant_peak_tensor_payload_lower_bound_bytes(self) -> int:
        """Larger audited material-only phase, excluding allocator overhead."""

        return max(
            self.native_material_adapter_sample_peak_dominant_tensor_bytes,
            self.native_material_adapter_finalize_peak_dominant_tensor_bytes,
        )

    @property
    def staging_plan_tensor_bytes(self) -> int:
        """Resident int64 pixel/sample ids plus float32 sample times."""

        return 8 * self.plan_pixel_count + 12 * self.plan_sample_count

    @property
    def ordered_plan_peak_additional_tensor_bytes(self) -> int:
        """Ordered ids/times retained plus the transient int64 permutation."""

        return 20 * self.global_sample_count

    @property
    def decoded_rgb_frame_tensor_bytes(self) -> int:
        """One normalized float32 RGB decode, excluding PIL/NumPy overhead."""

        return 12 * self.image_height * self.image_width

    @property
    def cpu_piecewise_reference_target_bytes(self) -> int:
        """Full float64 ``[P,F,3]`` input required by the CPU reference API."""

        return 24 * self.global_track_count * self.global_sample_count

    @property
    def strict_certificate_dense_pointer_lower_bound_bytes(self) -> int:
        """Lower bound for the current dense CPU continuous-certificate peak.

        ``_dual_world`` creates ``D`` variables with two length-``D`` tangent
        tuples: at least ``2 D^2`` pointer slots.  During node linearization,
        node Jacobians and fitted coefficient tangents coexist, each carrying
        ``4 B_p J D`` interval references.  CPython pointer slots are eight
        bytes on the supported 64-bit host.  The real peak is substantially
        larger because every referenced interval owns rational endpoints.
        """

        dimension = self.continuous_certificate_parameter_count
        dual_world = 16 * dimension * dimension
        simultaneous_linearizations = 64 * self.block_track_count * self.maximum_node_count * dimension
        return max(dual_world, simultaneous_linearizations)

    @property
    def expensive_block_tensor_bytes(self) -> int:
        """Compiled block state independent of ``F`` and ``K``."""

        return self.compact_topology_tensor_bytes + self.compact_world_and_atlas_tensor_bytes

    @property
    def cheap_temporal_tensor_bytes(self) -> int:
        """Allowed global O(F) identity/time payload; Python lists are excluded.

        The bounded native ``[K]`` time block is reported separately.  Keeping
        it out of this aggregate prevents the runtime allocation table from
        counting the same live tensor twice.
        """

        return self.staging_plan_tensor_bytes + self.ordered_plan_peak_additional_tensor_bytes

    def with_global_sample_count(self, count: int) -> WorldFoamHostMemoryDimensions:
        if self.plan_sample_count % self.global_sample_count:
            raise ValueError("cannot preserve a non-rectangular plan/global sample ratio implicitly")
        view_factor = self.plan_sample_count // self.global_sample_count
        return replace(
            self,
            global_sample_count=count,
            plan_sample_count=count * view_factor,
        )

    def allocation_table(self) -> tuple[HostMemoryAllocation, ...]:
        """Return the current allocation/lifetime audit in execution order."""

        return (
            HostMemoryAllocation(
                "initialized_world_parameters",
                self.global_source_parameter_tensor_bytes,
                "initialization boundary; physical seed may release after live material-state creation",
                "O(S_global*(8+L_w)), independent of P and F; bounded block rays counted separately",
                "bounded",
            ),
            HostMemoryAllocation(
                "global_direct_kinetic_gradient_buffers",
                self.global_gradient_tensor_bytes,
                "logical optimizer step",
                "O(S_global*(6+L_w) + P_bar), independent of F; P_bar=0 for fixed cameras",
                "bounded",
            ),
            HostMemoryAllocation(
                "decoded_material_and_raw_chain_gradient_buffers",
                self.global_decoded_material_and_raw_chain_gradient_tensor_bytes,
                "material-training session lifetime",
                "12 float32 scalars/site = O(S_global), independent of F; optimizer moments excluded",
                "bounded",
            ),
            HostMemoryAllocation(
                "global_cpu_atlas_template",
                self.global_template_tensor_bytes,
                "ledger + every PreparedCompactStagedLieWorld",
                "O(P_global * sum(J_c) + C*I_global + W_global)",
                "unbounded",
            ),
            HostMemoryAllocation(
                "compact_prepared_topology",
                self.compact_topology_tensor_bytes,
                "one spatial block / certificate binding",
                "O(B_p + W_b + I_b + B_b + S_b)",
                "bounded",
            ),
            HostMemoryAllocation(
                "legacy_compact_material_training_binding_private_tensors_per_block",
                self.training_binding_tensor_bytes_per_block,
                "one reported legacy block; every spatial-block instance is retained by the legacy program",
                (
                    "reported bytes are O(B_p + W_b + I_b + B_b + S_b + sum(J_c^2)) "
                    "per block; total residency multiplies by spatial block count"
                ),
                "unbounded",
            ),
            HostMemoryAllocation(
                "compact_world_and_atlas",
                self.compact_world_and_atlas_tensor_bytes,
                "one spatial block / certificate binding",
                "O(B_p * sum(J_c) + C*I_b + W_b + S_b + B_b)",
                "bounded",
            ),
            HostMemoryAllocation(
                "continuous_certificate_dense_pointer_floor",
                self.strict_certificate_dense_pointer_lower_bound_bytes,
                "strict certification peak; released after binding",
                "Omega(max(D^2, B_p*J_max*D)); D=5B_b+12B_p+4I_b+4S_b",
                "unbounded",
                exact_tensor_payload=False,
            ),
            HostMemoryAllocation(
                "certificate_binding_json",
                self.binding_payload_bytes,
                "binding lifetime; reparsed by assert_current",
                "O(C*D) labels/facts for the current strict certificate",
                "unbounded",
            ),
            HostMemoryAllocation(
                "staging_plan_tensor_metadata",
                self.staging_plan_tensor_bytes,
                "logical step",
                "O(P_pixels + F)",
                "cheap_linear",
            ),
            HostMemoryAllocation(
                "ordered_plan_and_permutation_peak",
                self.ordered_plan_peak_additional_tensor_bytes,
                "adapter ordering; permutation is transient",
                "O(F)",
                "cheap_linear",
            ),
            HostMemoryAllocation(
                "native_sample_times",
                self.native_sample_time_tensor_bytes,
                "one synchronized K block",
                "O(K), no global F or chart-local F_c clone",
                "bounded",
            ),
            HostMemoryAllocation(
                "staged_target_and_explicit_rays",
                self.staged_target_and_ray_tensor_bytes,
                "one synchronized K block",
                "O(B_p*K)",
                "bounded",
            ),
            HostMemoryAllocation(
                "decoded_rgb_frame_tensor",
                self.decoded_rgb_frame_tensor_bytes,
                "one source decode",
                "O(H*W), one frame",
                "bounded",
            ),
            HostMemoryAllocation(
                "target_source_residency",
                self.source_target_resident_bytes,
                "target provider lifetime",
                "0 for path/video-seek; O(V*F*H*W) for resident source",
                "optional" if self.source_target_resident_bytes == 0 else "unbounded",
            ),
            HostMemoryAllocation(
                "cpu_piecewise_reference_targets",
                self.cpu_piecewise_reference_target_bytes,
                "caller-owned input for piecewise_topology_staged_lie_mse_vjp",
                "O(P_global*F)",
                "optional",
            ),
        )

    def native_runtime_allocation_table(self) -> tuple[HostMemoryAllocation, ...]:
        """Return only allocations admitted by the intended native hot path.

        Dense proof-oracle state, the full-track CPU atlas template, binding
        construction, and the full-target CPU reference API are intentionally
        absent.  This table remains logical source accounting until a rebuilt
        native extension reports allocator peaks.
        """

        return (
            HostMemoryAllocation(
                "global_kinetic_geometry_parameters",
                self.global_kinetic_geometry_parameter_tensor_bytes,
                "model lifetime",
                "O(S_global*(6+L_w)), independent of P and F; bounded block rays counted separately",
                "bounded",
            ),
            HostMemoryAllocation(
                "global_material_step_accumulator",
                self.global_material_step_accumulator_tensor_bytes,
                "logical optimizer step",
                "16*S_global+4 bytes, independent of F; selected material-only hot path",
                "bounded",
            ),
            HostMemoryAllocation(
                "decoded_material_and_raw_chain_gradient_buffers",
                self.global_decoded_material_and_raw_chain_gradient_tensor_bytes,
                "material-training session lifetime",
                "12 float32 scalars/site = O(S_global), independent of F; optimizer moments excluded",
                "bounded",
            ),
            HostMemoryAllocation(
                "bounded_compiled_cpu_artifact_store",
                self.compiled_cpu_artifact_store_max_resident_accounted_bytes,
                "direct-kinetic program lifetime under the artifact-store byte budget",
                "bounded logical tensors plus canonical metadata, independent of F",
                "bounded",
                exact_tensor_payload=False,
            ),
            HostMemoryAllocation(
                "native_cached_material_topology_tokens",
                self.native_topology_cache_tensor_bytes,
                "material-training session; explicit LRU entry/byte budget",
                "O(cache_entries * (B_p + W_b + I_b + B_b + S_b)), independent of total blocks and F",
                "bounded",
            ),
            HostMemoryAllocation(
                "native_compact_live_world_and_bars",
                self.native_compact_live_world_and_bar_tensor_bytes,
                "one spatial block",
                "O(B_p + S_b), independent of F",
                "bounded",
            ),
            HostMemoryAllocation(
                "native_node_state_and_bar",
                self.native_node_state_and_bar_tensor_bytes,
                "one chart",
                "O(B_p*J_max), independent of F and chart count",
                "bounded",
            ),
            HostMemoryAllocation(
                "native_material_geometry_length_bar",
                self.native_material_geometry_length_bar_tensor_bytes,
                "one material-only native reverse",
                "0; frozen geometry uses the dedicated no-[J,W]-output ABI",
                "bounded",
            ),
            HostMemoryAllocation(
                "native_synchronized_material_sample_launch",
                self.native_synchronized_material_sample_launch_tensor_bytes,
                "one synchronized sample block",
                (
                    "4*N*J_max+28*N+20 bytes, N<=B_p*K; retained block is "
                    "4*N*J_max+24*N and public native preparation adds 4*N+20"
                ),
                "bounded",
            ),
            HostMemoryAllocation(
                "native_sample_times",
                self.native_sample_time_tensor_bytes,
                "one synchronized sample block",
                "O(K), independent of F",
                "bounded",
            ),
            HostMemoryAllocation(
                "native_material_audited_dominant_peak_tensor_payload_lower_bound",
                self.native_material_audited_dominant_peak_tensor_payload_lower_bound_bytes,
                "maximum of audited material-only sample/finalization phases",
                (
                    "cache residency is reported separately; excludes small metadata, "
                    "Python, command buffers, allocator, and optimizer state"
                ),
                "partial_lower_bound",
            ),
            HostMemoryAllocation(
                "sample_identity_and_time_metadata",
                self.cheap_temporal_tensor_bytes,
                "logical step / adapter ordering",
                "O(F), cheap camera/sample axis",
                "cheap_linear",
            ),
            HostMemoryAllocation(
                "decoded_rgb_frame_tensor",
                self.decoded_rgb_frame_tensor_bytes,
                "one source decode",
                "O(H*W), one frame",
                "bounded",
            ),
            HostMemoryAllocation(
                "target_source_residency",
                self.source_target_resident_bytes,
                "target provider lifetime",
                "0 for path/video-seek; O(V*F*H*W) for resident source",
                "optional" if self.source_target_resident_bytes == 0 else "unbounded",
            ),
        )


__all__ = [
    "HostMemoryAllocation",
    "WorldFoamHostMemoryDimensions",
]
