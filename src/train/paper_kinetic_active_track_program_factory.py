"""Production structural compiler for fixed-camera WorldFoam tracks.

The lazy paper provider owns calibrated camera metadata and asks a track
factory for one frame-density-independent program per ``(view, pixel)``.  The
kinetic compiler below is deliberately narrower than the mathematical affine
ray ABI: the current request carries correctly resized endpoint ray witnesses,
but not the source image dimensions needed to independently reconstruct every
intermediate resized ray.  Consequently this first production factory accepts
only a camera record that is content-identical at every dataset time.

For that supported stratum the endpoint witness is a constant calibrated ray,
so its affine coefficients are exact binary64 constants.  Every camera record
is nevertheless revalidated at the compile boundary.  This is the deliberately
cheap ``O(F_dataset)`` camera slice: it prevents a stale provider certificate
or a post-seal mutable ``CameraSpec`` from admitting a moving/unsupported path.
The factory then:

* compiles continuous owner charts with the active-boundary exact compiler;
* fails closed on every unresolved topology degeneracy;
* compiles every chart to one explicitly configured fixed P0 node count; and
* returns no camera or requested-frame payload in the program.

Moving, projective-time, gauged, piecewise, or otherwise changing camera paths
are rejected.  In particular, this module never fits an affine ray through two
endpoints and pretends that the unobserved path is affine.  Continuous transfer
error certification remains a later material-snapshot gate; the returned
program honestly keeps ``continuous_forward_error_certified=False``.

This is a CPU/source component.  It launches no native kernel and retains no
request, camera grid, compiled program, or frame-axis cache on the factory.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from fractions import Fraction

import torch
from camera import CameraSpec
from paper_kinetic_runtime_paths import ensure_worldfoam_lane2_research_path

ensure_worldfoam_lane2_research_path()

from kinetic_active_owner_chart_compiler import ActiveKineticOwnerChartProgram  # noqa: E402
from kinetic_chart_transfer_bridge import (  # noqa: E402
    compile_and_bind_active_kinetic_owner_program,
)
from kinetic_multichart_transfer_program import (  # noqa: E402
    KineticMultiChartP0Program,
    compile_bound_kinetic_multichart_p0_program,
)
from paper_kinetic_lazy_program_bundles import PaperKineticTrackProgramRequest  # noqa: E402
from paper_kinetic_lazy_program_bundles import (  # noqa: E402
    PaperKineticObservationRayRecord,
)

FACTORY_PROVENANCE = "paper-kinetic-active-p0-static-camera-track-factory-v1"
CAMERA_PATH_SCOPE = "content_identical_static_calibrated_camera_records_only"
AFFINE_RAY_SCOPE = "constant_binary64_ray_from_provider_scaled_endpoint_witness"
TOPOLOGY_CERTIFICATION_SCOPE = "active_exact_continuous_owner_charts"
RANK_SELECTION_SCOPE = "fixed_node_count_from_explicit_external_provenance"
TRANSFER_CERTIFICATION_SCOPE = "deferred_until_material_snapshot"
COMPILE_ACCOUNTING_PROVENANCE = (
    "paper-kinetic-active-p0-track-compile-accounting-v1"
)

_KNOWN_CENTRAL_LENS_MODELS = {
    "pinhole": 0,
    "radial_tangential": 5,
    "opencv_fisheye": 4,
}


class PaperKineticUnsupportedCameraPathError(ValueError):
    """The calibrated camera path cannot be represented by this factory."""


class PaperKineticOwnerChartCompilationError(ValueError):
    """Exact active owner-chart compilation failed closed."""


@dataclass(frozen=True)
class PaperKineticActiveP0TrackProgramFactoryConfig:
    """Explicit structural compile policy; no dataset or frame state lives here."""

    near: float
    far: float
    node_count: int
    maximum_sites_per_track_compile: int
    maximum_charts_per_track: int
    maximum_owner_runs_per_chart: int
    rank_selection_provenance: str

    def assert_valid(self) -> None:
        if (
            isinstance(self.near, bool)
            or not isinstance(self.near, (int, float))
            or isinstance(self.far, bool)
            or not isinstance(self.far, (int, float))
        ):
            raise TypeError("paper kinetic factory near/far must be Python real scalars")
        near = float(self.near)
        far = float(self.far)
        if not math.isfinite(near) or not math.isfinite(far) or near < 0.0 or far <= near:
            raise ValueError("paper kinetic factory requires finite 0 <= near < far")
        if isinstance(self.node_count, bool) or not isinstance(self.node_count, int) or self.node_count < 2:
            raise ValueError("paper kinetic factory node_count must be an integer at least two")
        for name, value in (
            ("maximum_sites_per_track_compile", self.maximum_sites_per_track_compile),
            ("maximum_charts_per_track", self.maximum_charts_per_track),
            ("maximum_owner_runs_per_chart", self.maximum_owner_runs_per_chart),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"paper kinetic factory {name} must be a positive integer")
        if not isinstance(self.rank_selection_provenance, str) or not self.rank_selection_provenance.strip():
            raise ValueError("paper kinetic fixed rank requires nonempty selection provenance")


@dataclass(frozen=True)
class PaperKineticActiveP0TrackProgramFactory:
    """Dataset-request consumer implementing ``PaperKineticTrackProgramFactory``."""

    config: PaperKineticActiveP0TrackProgramFactoryConfig
    generation_digest: str
    provenance: str = FACTORY_PROVENANCE
    camera_path_scope: str = CAMERA_PATH_SCOPE
    affine_ray_scope: str = AFFINE_RAY_SCOPE
    topology_certification_scope: str = TOPOLOGY_CERTIFICATION_SCOPE
    rank_selection_scope: str = RANK_SELECTION_SCOPE
    transfer_certification_scope: str = TRANSFER_CERTIFICATION_SCOPE
    requested_frame_payload_retained: bool = False
    compiled_program_cache_retained: bool = False
    _seal: object = field(default=None, repr=False)

    def assert_current(self) -> None:
        self.config.assert_valid()
        if (
            self._seal is not _FACTORY_SEAL
            or self.provenance != FACTORY_PROVENANCE
            or self.camera_path_scope != CAMERA_PATH_SCOPE
            or self.affine_ray_scope != AFFINE_RAY_SCOPE
            or self.topology_certification_scope != TOPOLOGY_CERTIFICATION_SCOPE
            or self.rank_selection_scope != RANK_SELECTION_SCOPE
            or self.transfer_certification_scope != TRANSFER_CERTIFICATION_SCOPE
            or self.requested_frame_payload_retained
            or self.compiled_program_cache_retained
            or self.generation_digest != _factory_digest(self.config)
        ):
            raise ValueError("paper kinetic active P0 factory provenance changed")

    def accounting(self) -> dict[str, object]:
        """Expose the supported theorem/runtime boundary without overclaiming."""

        self.assert_current()
        return {
            "provenance": self.provenance,
            "generation_digest": self.generation_digest,
            "camera_path_scope": self.camera_path_scope,
            "affine_ray_scope": self.affine_ray_scope,
            "moving_camera_path_supported": False,
            "endpoint_affine_fit_used": False,
            "camera_path_admission_work": (
                "O(F_dataset) camera-record validation per track plus the "
                "provider cold certificate; no camera tensor is retained"
            ),
            "camera_path_admission_state_retained": "O(1) digest/boolean per view",
            "requested_frame_sampling_used_for_topology": False,
            "requested_frame_payload_retained": False,
            "compiled_program_cache_retained": False,
            "topology_compiler": "compile_active_kinetic_owner_charts",
            "topology_compile_and_bind_path": (
                "compile_and_bind_active_kinetic_owner_program"
            ),
            "active_owner_topology_compile_count_per_track": 1,
            "duplicate_source_binding_recompile_used": False,
            "topology_certification_scope": self.topology_certification_scope,
            "topology_compile_work": (
                "active-owner closure: O(U*S*R_max) predicate construction plus "
                "O(W*(S log S + S*R_max)) witness certification, excluding exact-root bit complexity"
            ),
            "exhaustive_triple_enumeration_used": False,
            "node_count": self.config.node_count,
            "maximum_sites_per_track_compile": (
                self.config.maximum_sites_per_track_compile
            ),
            "maximum_pair_cache_entries_per_track_upper_bound": (
                self.config.maximum_sites_per_track_compile
                * (self.config.maximum_sites_per_track_compile - 1)
                // 2
            ),
            "maximum_charts_per_track": self.config.maximum_charts_per_track,
            "maximum_owner_runs_per_chart": (
                self.config.maximum_owner_runs_per_chart
            ),
            "compiler_scratch_allocator_peak_measured": False,
            "rank_selection_scope": self.rank_selection_scope,
            "rank_selection_provenance": self.config.rank_selection_provenance,
            "continuous_transfer_error_certified": False,
            "transfer_certification_scope": self.transfer_certification_scope,
            "component_scope": "CPU structural compiler feeding later native lowering",
            "native_runtime_executed": False,
        }

    def memory_light_residency(self) -> dict[str, int | bool]:
        """Report only state retained by this live factory object.

        The lazy coordinator treats this method as a trust boundary.  Keep the
        report derived from the object graph instead of returning aspirational
        zeroes: a later request/program/tensor cache added to this frozen
        factory must immediately make the memory-light admission fail.
        Python allocator bytes for immutable scalar configuration are outside
        this logical-tensor report and are not presented as measured.
        """

        self.assert_current()
        report = _retained_factory_state(self)
        return {
            "retained_compile_request_count": report[0],
            "retained_compiled_program_count": report[1],
            "retained_observation_record_count": report[2],
            "retained_tensor_bytes": report[3],
            "unbounded_cache_enabled": report[4],
        }

    def compile_track(
        self,
        request: PaperKineticTrackProgramRequest,
    ) -> KineticMultiChartP0Program:
        """Compile one exact static-camera track and retain no request state."""

        self.assert_current()
        if not isinstance(request, PaperKineticTrackProgramRequest):
            raise TypeError("paper kinetic active P0 factory requires its sealed track request")
        request.assert_self_consistent()
        if request.factory_generation_digest != self.generation_digest:
            raise ValueError("paper kinetic track request belongs to a different factory generation")
        if request.pixel_index >= request.height * request.width:
            raise IndexError("paper kinetic track pixel leaves the calibrated stage image")
        if len(request.frame_times) < 2 or any(
            right <= left
            for left, right in zip(request.frame_times, request.frame_times[1:], strict=False)
        ):
            raise PaperKineticUnsupportedCameraPathError(
                "paper kinetic topology compilation requires at least two increasing camera times"
            )
        if request.world.sites.site_count > self.config.maximum_sites_per_track_compile:
            raise MemoryError(
                "paper kinetic track exceeds maximum_sites_per_track_compile before exact compilation"
            )

        ray_coefficients = _constant_affine_ray_coefficients(request)
        try:
            binding = compile_and_bind_active_kinetic_owner_program(
                request.world.sites,
                ray_coefficients,
                t_min=Fraction.from_float(float(request.frame_times[0])),
                t_max=Fraction.from_float(float(request.frame_times[-1])),
                near=Fraction.from_float(float(self.config.near)),
                far=Fraction.from_float(float(self.config.far)),
            )
        except ValueError as error:
            raise PaperKineticOwnerChartCompilationError(
                f"active owner-chart compilation failed closed: {error}"
            ) from error
        _require_passed_active_owner_program(binding.program, config=self.config)
        program = compile_bound_kinetic_multichart_p0_program(
            binding,
            node_count=self.config.node_count,
        )
        _require_factory_output(program, node_count=self.config.node_count)
        return program

    def compile_accounting(
        self,
        program: KineticMultiChartP0Program,
    ) -> dict[str, int | bool | str]:
        """Extract the exact active-compiler receipt while ``program`` is live.

        The lazy bundle compiler calls this immediately after validating each
        returned program.  The resulting mapping contains only Python scalar
        values and digests, so its caller can fold it into a rolling receipt
        without retaining this program, its charts, or any tensor payload.
        """

        self.assert_current()
        return paper_kinetic_active_p0_track_compile_accounting(program)


_FACTORY_SEAL = object()


def prepare_paper_kinetic_active_p0_track_program_factory(
    config: PaperKineticActiveP0TrackProgramFactoryConfig,
) -> PaperKineticActiveP0TrackProgramFactory:
    """Validate and cold-seal one deterministic fixed-camera factory."""

    if not isinstance(config, PaperKineticActiveP0TrackProgramFactoryConfig):
        raise TypeError("paper kinetic active P0 factory requires its config type")
    config.assert_valid()
    result = PaperKineticActiveP0TrackProgramFactory(
        config=config,
        generation_digest=_factory_digest(config),
        _seal=_FACTORY_SEAL,
    )
    result.assert_current()
    return result


def paper_kinetic_active_p0_track_compile_accounting(
    program: KineticMultiChartP0Program,
) -> dict[str, int | bool | str]:
    """Return one source-sealed receipt derived from the compiled program.

    The factory intentionally retains no cumulative counter or program cache.
    A runner that needs whole-step compiler work must call this function while
    each returned program is live and sum the receipt fields itself.
    """

    if not isinstance(program, KineticMultiChartP0Program):
        raise TypeError("active P0 compile accounting requires its program type")
    program.assert_current()
    active = program.binding.program
    if not isinstance(active, ActiveKineticOwnerChartProgram):
        raise TypeError("active P0 compile accounting requires active-compiler provenance")
    work = active.work
    values = {
        "compile_track_count": 1,
        "site_count": work.site_count,
        "certificate_round_count": work.certificate_round_count,
        "root_complement_witness_count": (
            work.root_complement_witness_count
        ),
        "witness_word_discovery_count": work.witness_word_discovery_count,
        "candidate_source_attempt_count": work.candidate_source_attempt_count,
        "unique_source_word_count": work.unique_source_word_count,
        "unique_candidate_source_count": work.unique_candidate_source_count,
        "root_isolation_call_count": work.root_isolation_call_count,
        "isolated_raw_root_count": work.isolated_raw_root_count,
        "distinct_event_guard_count": work.distinct_event_guard_count,
        "pair_difference_request_count": work.pair_difference_request_count,
        "unique_pair_difference_count": work.unique_pair_difference_count,
        "all_site_witness_check_count": work.all_site_witness_check_count,
        "algebraic_root_refinement_count": (
            work.algebraic_root_refinement_count
        ),
        "max_run_count": work.max_run_count,
        "sum_site_run_products": work.sum_site_run_products,
        "per_witness_candidate_bound_verified": (
            work.per_witness_candidate_bound_verified
        ),
        "exhaustive_triple_enumeration_used": (
            work.exhaustive_triple_enumeration_used
        ),
        "requested_frame_sampling_used": work.requested_frame_sampling_used,
        "compiler_program_generation_digest": program.generation_digest,
    }
    values["compiler_work_receipt_digest"] = _digest_parts(
        COMPILE_ACCOUNTING_PROVENANCE,
        tuple(values.items()),
    )
    values["compiler_work_receipt_provenance"] = (
        COMPILE_ACCOUNTING_PROVENANCE
    )
    return values


def _constant_affine_ray_coefficients(
    request: PaperKineticTrackProgramRequest,
) -> torch.Tensor:
    if len(request.cameras) != len(request.frame_times):
        raise ValueError("paper kinetic camera records do not cover the dataset time domain")

    # CameraSpec is a mutable record even though the containing request is a
    # frozen dataclass.  The provider's cold certificate is therefore useful
    # provenance, but it cannot replace validation at this trust boundary.
    # Scan the complete (cheap) camera slice before consulting the certificate
    # so both an originally moving path and a post-seal mutation fail with the
    # precise offending record rather than being collapsed into a stale bool.
    reference_camera_digest = _static_camera_content_digest(
        request.cameras[0],
        frame_index=0,
    )
    for frame_index, camera in enumerate(request.cameras[1:], start=1):
        if (
            _static_camera_content_digest(camera, frame_index=frame_index)
            != reference_camera_digest
        ):
            raise PaperKineticUnsupportedCameraPathError(
                f"camera record {frame_index} differs from record 0; "
                "moving/projective/gauged camera paths are unsupported"
            )
    if not request.static_camera_path_certified:
        raise PaperKineticUnsupportedCameraPathError(
            "moving/projective/gauged camera paths are unsupported by the static production factory"
        )

    witnesses = request.observations
    first = witnesses[0]
    if first.sample_time != request.frame_times[first.observation.frame_index]:
        raise ValueError("paper kinetic endpoint witness time changed")
    ray = first.ray_origin_direction
    for witness in witnesses[1:]:
        if witness.sample_time != request.frame_times[witness.observation.frame_index]:
            raise ValueError("paper kinetic endpoint witness time changed")
        if witness.camera_record_digest != first.camera_record_digest:
            raise PaperKineticUnsupportedCameraPathError(
                "static camera records produced different endpoint calibration digests"
            )
        if witness.ray_origin_direction != ray:
            raise PaperKineticUnsupportedCameraPathError(
                "moving/projective/gauged calibrated rays are unsupported; endpoint rays differ"
            )
    if not all(math.isfinite(value) for value in ray):
        raise ValueError("paper kinetic calibrated ray witness must be finite")
    direction_norm = math.sqrt(sum(value * value for value in ray[3:6]))
    if not math.isclose(direction_norm, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("paper kinetic calibrated ray direction must be unit length in binary64")
    coefficients = torch.tensor(
        (
            *ray[0:3],
            0.0,
            0.0,
            0.0,
            *ray[3:6],
            0.0,
            0.0,
            0.0,
        ),
        dtype=torch.float64,
        device="cpu",
    )
    if tuple(coefficients.shape) != (12,) or not bool(torch.isfinite(coefficients).all().item()):
        raise ArithmeticError("constant calibrated ray lowering produced invalid coefficients")
    return coefficients


def _static_camera_content_digest(camera: CameraSpec, *, frame_index: int) -> str:
    if not isinstance(camera, CameraSpec):
        raise TypeError("paper kinetic camera path must contain CameraSpec records")
    lens_model = str(camera.lens_model)
    if lens_model not in _KNOWN_CENTRAL_LENS_MODELS:
        raise PaperKineticUnsupportedCameraPathError(
            f"camera record {frame_index} uses unsupported lens model {lens_model!r}"
        )
    intrinsics = tuple(
        _finite_cpu_scalar(value, name=f"camera[{frame_index}].{name}")
        for name, value in (
            ("fx", camera.fx),
            ("fy", camera.fy),
            ("cx", camera.cx),
            ("cy", camera.cy),
        )
    )
    if intrinsics[0] <= 0.0 or intrinsics[1] <= 0.0:
        raise ValueError("paper kinetic calibrated focal lengths must be positive")
    transform = _finite_cpu_tensor(
        camera.camera_to_world,
        name=f"camera[{frame_index}].camera_to_world",
    )
    if tuple(transform.shape) != (4, 4):
        raise ValueError("paper kinetic camera_to_world must have shape [4,4]")
    if tuple(float(value) for value in transform[3].tolist()) != (0.0, 0.0, 0.0, 1.0):
        raise PaperKineticUnsupportedCameraPathError(
            "general projective camera matrices are unsupported; camera_to_world must be affine"
        )
    distortion = ()
    if camera.distortion is not None:
        normalized = _finite_cpu_tensor(
            camera.distortion,
            name=f"camera[{frame_index}].distortion",
        ).reshape(-1)
        maximum = _KNOWN_CENTRAL_LENS_MODELS[lens_model]
        if int(normalized.numel()) > maximum:
            raise ValueError(
                f"camera record {frame_index} has too many {lens_model} distortion coefficients"
            )
        if lens_model == "pinhole" and int(normalized.numel()) > 0:
            raise PaperKineticUnsupportedCameraPathError(
                "pinhole camera records with ignored distortion payloads are unsupported"
            )
        distortion = tuple(float(value) for value in normalized.tolist())
    return _digest_parts(
        "normalized-static-camera-v1",
        intrinsics,
        lens_model,
        distortion,
        tuple(float(value) for value in transform.reshape(-1).tolist()),
    )


def _finite_cpu_scalar(value: float | torch.Tensor, *, name: str) -> float:
    tensor = torch.as_tensor(value)
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be CPU-resident for structural compilation")
    if tensor.numel() != 1:
        raise ValueError(f"{name} must be scalar")
    result = float(tensor.detach().to(dtype=torch.float64).item())
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_cpu_tensor(value: object, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be CPU-resident for structural compilation")
    result = tensor.detach().to(dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise ValueError(f"{name} must be finite")
    return result


def _require_passed_active_owner_program(
    program: ActiveKineticOwnerChartProgram,
    *,
    config: PaperKineticActiveP0TrackProgramFactoryConfig,
) -> None:
    if not isinstance(program, ActiveKineticOwnerChartProgram):
        raise TypeError("paper kinetic production factory requires the active owner compiler")
    if (
        not program.passed
        or not program.continuous_time_coverage
        or not program.owner_identity_certified
        or program.unresolved_degeneracies
        or not program.charts
    ):
        reasons = tuple(degeneracy.kind for degeneracy in program.unresolved_degeneracies)
        raise PaperKineticOwnerChartCompilationError(
            f"active owner-chart compilation failed closed: degeneracies={reasons!r}"
        )
    if len(program.charts) > config.maximum_charts_per_track:
        raise MemoryError(
            "paper kinetic active compiler output exceeds maximum_charts_per_track"
        )
    if program.work.max_run_count > config.maximum_owner_runs_per_chart:
        raise MemoryError(
            "paper kinetic active compiler output exceeds maximum_owner_runs_per_chart"
        )
    if (
        program.requested_frame_sampling_used
        or program.work.requested_frame_sampling_used
        or program.work.exhaustive_triple_enumeration_used
    ):
        raise ArithmeticError("active owner-chart compiler violated the frame-free production contract")


def _require_factory_output(
    program: KineticMultiChartP0Program,
    *,
    node_count: int,
) -> None:
    program.assert_current()
    if not isinstance(program.binding.program, ActiveKineticOwnerChartProgram):
        raise ArithmeticError("paper kinetic factory output lost active-compiler provenance")
    if program.binding.compiler_provenance != "active_kinetic_owner_chart_compiler_v1":
        raise ArithmeticError("paper kinetic factory output has the wrong compiler provenance")
    if (
        program.requested_frame_sampling_used
        or program.dense_track_chart_refinement_used
        or program.continuous_forward_error_certified
        or any(chart.node_count != node_count for chart in program.charts)
    ):
        raise ArithmeticError("paper kinetic factory output violated its fixed-rank structural contract")


def _factory_digest(config: PaperKineticActiveP0TrackProgramFactoryConfig) -> str:
    return _digest_parts(
        FACTORY_PROVENANCE,
        CAMERA_PATH_SCOPE,
        AFFINE_RAY_SCOPE,
        TOPOLOGY_CERTIFICATION_SCOPE,
        RANK_SELECTION_SCOPE,
        TRANSFER_CERTIFICATION_SCOPE,
        float(config.near),
        float(config.far),
        config.node_count,
        config.maximum_sites_per_track_compile,
        config.maximum_charts_per_track,
        config.maximum_owner_runs_per_chart,
        config.rank_selection_provenance,
        tuple(sorted(_KNOWN_CENTRAL_LENS_MODELS.items())),
        1.0e-12,
    )


def _retained_factory_state(
    factory: PaperKineticActiveP0TrackProgramFactory,
) -> tuple[int, int, int, int, bool]:
    """Inspect retained fields without following unrelated global objects."""

    request_count = 0
    program_count = 0
    observation_count = 0
    tensor_storage_bytes = 0
    unbounded_cache_enabled = False
    visited: set[int] = set()
    tensor_storages: set[tuple[str, int]] = set()

    def visit(value: object, *, direct_factory_field: bool = False) -> None:
        nonlocal request_count
        nonlocal program_count
        nonlocal observation_count
        nonlocal tensor_storage_bytes
        nonlocal unbounded_cache_enabled

        if value is None or isinstance(value, (bool, int, float, str, bytes)):
            return
        identity = id(value)
        if identity in visited:
            return
        visited.add(identity)
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            key = (str(value.device), int(storage.data_ptr()))
            if key not in tensor_storages:
                tensor_storages.add(key)
                tensor_storage_bytes += int(storage.nbytes())
            return
        if isinstance(value, PaperKineticTrackProgramRequest):
            request_count += 1
        if isinstance(value, KineticMultiChartP0Program):
            program_count += 1
        if isinstance(value, PaperKineticObservationRayRecord):
            observation_count += 1
        if direct_factory_field and isinstance(value, (dict, list, set)):
            # This production factory has no bounded mutable cache policy.  A
            # newly retained mutable collection is therefore unbounded until
            # an explicit bounded contract is introduced.
            unbounded_cache_enabled = True
        if is_dataclass(value) and not isinstance(value, type):
            for item in fields(value):
                visit(getattr(value, item.name))
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                visit(key)
                visit(item)
            return
        if isinstance(value, (tuple, list, set, frozenset)):
            for item in value:
                visit(item)

    # Do not visit the factory as a dataclass: mark its direct fields so an
    # accidentally retained mutable cache is distinguished from immutable
    # nested configuration.
    for item in fields(factory):
        visit(getattr(factory, item.name), direct_factory_field=True)
    return (
        request_count,
        program_count,
        observation_count,
        tensor_storage_bytes,
        unbounded_cache_enabled,
    )


def _digest_parts(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = repr(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


__all__ = [
    "AFFINE_RAY_SCOPE",
    "CAMERA_PATH_SCOPE",
    "COMPILE_ACCOUNTING_PROVENANCE",
    "FACTORY_PROVENANCE",
    "PaperKineticActiveP0TrackProgramFactory",
    "PaperKineticActiveP0TrackProgramFactoryConfig",
    "PaperKineticOwnerChartCompilationError",
    "PaperKineticUnsupportedCameraPathError",
    "paper_kinetic_active_p0_track_compile_accounting",
    "prepare_paper_kinetic_active_p0_track_program_factory",
]
