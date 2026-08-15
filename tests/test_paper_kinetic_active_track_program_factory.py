from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any

import pytest
import torch
from camera import CameraSpec
from kinetic_active_owner_chart_compiler import ActiveKineticOwnerChartProgram
from kinetic_power_word_compiler import AffineKineticPowerSites
from paper_kinetic_active_track_program_factory import (
    CAMERA_PATH_SCOPE,
    PaperKineticActiveP0TrackProgramFactory,
    PaperKineticActiveP0TrackProgramFactoryConfig,
    PaperKineticUnsupportedCameraPathError,
    paper_kinetic_active_p0_track_compile_accounting,
    prepare_paper_kinetic_active_p0_track_program_factory,
)
from paper_kinetic_lazy_program_bundles import (
    PaperKineticObservation,
    PaperKineticTrackProgramFactory,
    PaperKineticTrackProgramRequest,
    PaperKineticWorldInitializationRequest,
    prepare_paper_kinetic_lazy_program_bundle_provider,
)
from powerfoam_training_data import PowerFoamRayProvider, PowerFoamTargetProvider


def _sha256(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class _NonresidentTargetSource:
    view_count = 1
    height = 2
    width = 3

    def __init__(self, frame_count: int) -> None:
        self.frame_count = frame_count

    def select_view_frames(
        self,
        view_indices: tuple[int, ...],
        frame_indices: tuple[int, ...],
    ) -> torch.Tensor:
        return torch.zeros(
            (len(view_indices), 3, self.height, self.width),
            dtype=torch.float32,
        )

    def residency(self) -> dict[str, Any]:
        return {
            "source_kind": "paper_kinetic_factory_contract_fixture",
            "source_device": "fixture",
            "logical_bytes": self.frame_count * 3 * self.height * self.width * 4,
            "resident_bytes": 0,
            "full_source_resident": False,
            "disk_lazy_decode": True,
        }


class _OneSiteWorldInitializer:
    provenance = "paper-kinetic-factory-one-site-world-v1"
    generation_digest = _sha256(provenance)

    def initialize_world(
        self,
        request: PaperKineticWorldInitializationRequest,
    ) -> AffineKineticPowerSites:
        request.assert_self_consistent()
        return AffineKineticPowerSites(
            positions0=torch.zeros((1, 3), dtype=torch.float64),
            velocities=torch.zeros((1, 3), dtype=torch.float64),
            weight_coefficients=torch.zeros((1, 1), dtype=torch.float64),
        )


class _TwoSiteWorldInitializer(_OneSiteWorldInitializer):
    provenance = "paper-kinetic-factory-two-site-world-v1"
    generation_digest = _sha256(provenance)

    def initialize_world(
        self,
        request: PaperKineticWorldInitializationRequest,
    ) -> AffineKineticPowerSites:
        request.assert_self_consistent()
        return AffineKineticPowerSites(
            positions0=torch.tensor(
                ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                dtype=torch.float64,
            ),
            velocities=torch.zeros((2, 3), dtype=torch.float64),
            weight_coefficients=torch.zeros((2, 1), dtype=torch.float64),
        )


class _RecordingFactory:
    def __init__(self, factory: PaperKineticActiveP0TrackProgramFactory) -> None:
        self.factory = factory
        self.provenance = factory.provenance
        self.generation_digest = factory.generation_digest
        self.requests: list[PaperKineticTrackProgramRequest] = []
        self.programs = []

    def compile_track(self, request: PaperKineticTrackProgramRequest):
        self.requests.append(request)
        program = self.factory.compile_track(request)
        self.programs.append(program)
        return program

    def compile_accounting(self, program):
        return self.factory.compile_accounting(program)


def _camera(*, translation_x: float = 0.0) -> CameraSpec:
    camera_to_world = torch.eye(4, dtype=torch.float64)
    camera_to_world[0, 3] = translation_x
    return CameraSpec(
        fx=4.0,
        fy=4.0,
        cx=1.5,
        cy=1.0,
        camera_to_world=camera_to_world,
        lens_model="pinhole",
        distortion=None,
    )


def _production_factory(
    *,
    node_count: int = 4,
    maximum_sites_per_track_compile: int = 8,
) -> PaperKineticActiveP0TrackProgramFactory:
    return prepare_paper_kinetic_active_p0_track_program_factory(
        PaperKineticActiveP0TrackProgramFactoryConfig(
            near=0.0,
            far=2.0,
            node_count=node_count,
            maximum_sites_per_track_compile=maximum_sites_per_track_compile,
            maximum_charts_per_track=16,
            maximum_owner_runs_per_chart=8,
            rank_selection_provenance="checked-in-fixed-rank-contract-test-v1",
        )
    )


def _compile_one_track(
    frame_times: tuple[float, ...],
    cameras: tuple[CameraSpec, ...],
    *,
    factory: PaperKineticActiveP0TrackProgramFactory | None = None,
    world_initializer=None,
):
    production = _production_factory() if factory is None else factory
    recording = _RecordingFactory(production)
    source = _NonresidentTargetSource(len(frame_times))
    provider = prepare_paper_kinetic_lazy_program_bundle_provider(
        dataset_generation_digest=_sha256(f"factory-fixture-{frame_times!r}"),
        target_provider=PowerFoamTargetProvider(
            source=source,
            device=torch.device("cpu"),
        ),
        ray_provider=PowerFoamRayProvider(
            cameras=(cameras,),
            height=source.height,
            width=source.width,
            device=torch.device("cpu"),
        ),
        frame_times=frame_times,
        height=source.height,
        width=source.width,
        maximum_tracks_per_bundle=1,
        maximum_observations_per_bundle=1,
        maximum_rows_per_native_block=1,
        world_initializer=(
            _OneSiteWorldInitializer()
            if world_initializer is None
            else world_initializer
        ),
        program_factory=recording,
    )
    observation = PaperKineticObservation(
        observation_id=0,
        view_index=0,
        frame_index=0,
        pixel_index=0,
    )
    bundle = next(provider.iter_spatial_bundles((observation,), device="cpu"))
    return production, recording, provider, bundle


def test_static_dataset_track_uses_active_compiler_and_fixed_nodes() -> None:
    factory, recording, provider, bundle = _compile_one_track(
        (0.0, 0.5, 1.0),
        (_camera(), _camera(), _camera()),
    )

    assert isinstance(factory, PaperKineticTrackProgramFactory)
    assert len(recording.requests) == 1
    assert len(recording.programs) == 1
    program = recording.programs[0]
    program.assert_current()
    assert isinstance(program.binding.program, ActiveKineticOwnerChartProgram)
    assert program.binding.compiler_provenance == "active_kinetic_owner_chart_compiler_v1"
    assert program.binding.program.work.exhaustive_triple_enumeration_used is False
    assert program.binding.program.work.requested_frame_sampling_used is False
    assert program.requested_frame_sampling_used is False
    assert program.dense_track_chart_refinement_used is False
    assert program.continuous_forward_error_certified is False
    assert {chart.node_count for chart in program.charts} == {4}
    assert bundle.program_generation_digests == (program.generation_digest,)
    assert bundle.factory_request_generation_digests == (
        recording.requests[0].generation_digest,
    )
    assert provider.program_factory is recording

    accounting = factory.accounting()
    assert accounting["camera_path_scope"] == CAMERA_PATH_SCOPE
    assert accounting["active_owner_topology_compile_count_per_track"] == 1
    assert accounting["duplicate_source_binding_recompile_used"] is False
    assert accounting["moving_camera_path_supported"] is False
    assert accounting["endpoint_affine_fit_used"] is False
    assert accounting["requested_frame_payload_retained"] is False
    assert accounting["compiled_program_cache_retained"] is False
    assert accounting["continuous_transfer_error_certified"] is False
    assert accounting["maximum_sites_per_track_compile"] == 8
    assert accounting["maximum_charts_per_track"] == 16
    assert accounting["maximum_owner_runs_per_chart"] == 8
    assert accounting["compiler_scratch_allocator_peak_measured"] is False
    compile_receipt = paper_kinetic_active_p0_track_compile_accounting(program)
    compiler_work = program.binding.program.work
    assert compile_receipt["compile_track_count"] == 1
    assert compile_receipt["root_complement_witness_count"] == (
        compiler_work.root_complement_witness_count
    )
    assert compile_receipt["candidate_source_attempt_count"] == (
        compiler_work.candidate_source_attempt_count
    )
    assert compile_receipt["all_site_witness_check_count"] == (
        compiler_work.all_site_witness_check_count
    )
    assert compile_receipt["unique_pair_difference_count"] == (
        compiler_work.unique_pair_difference_count
    )
    assert len(str(compile_receipt["compiler_work_receipt_digest"])) == 64
    bundle_compile_receipt = bundle.compile_receipt
    bundle_compile_receipt.assert_current(
        track_ids=bundle.track_ids,
        program_generation_digests=bundle.program_generation_digests,
        request_generation_digests=bundle.factory_request_generation_digests,
    )
    assert bundle_compile_receipt.compile_track_count == 1
    assert bundle_compile_receipt.compiler_work_receipt_count == 1
    assert bundle_compile_receipt.compiler_work_receipt_chain_link_count == 1
    assert bundle_compile_receipt.root_complement_witness_count == (
        compile_receipt["root_complement_witness_count"]
    )
    assert bundle_compile_receipt.candidate_source_attempt_count == (
        compile_receipt["candidate_source_attempt_count"]
    )
    assert bundle_compile_receipt.all_site_witness_check_count == (
        compile_receipt["all_site_witness_check_count"]
    )
    assert bundle_compile_receipt.unique_pair_difference_count == (
        compile_receipt["unique_pair_difference_count"]
    )
    assert bundle_compile_receipt.compiler_accounting_complete is True
    assert bundle_compile_receipt.all_track_receipt_digests_verified is True
    assert bundle_compile_receipt.retained_compiled_program_count == 0
    assert bundle_compile_receipt.retained_compiler_receipt_entry_count == 0
    assert bundle_compile_receipt.retained_compiler_tensor_bytes == 0
    assert len(bundle_compile_receipt.compiler_work_receipt_chain_digest) == 64
    assert factory.memory_light_residency() == {
        "retained_compile_request_count": 0,
        "retained_compiled_program_count": 0,
        "retained_observation_record_count": 0,
        "retained_tensor_bytes": 0,
        "unbounded_cache_enabled": False,
    }


def test_production_factory_retains_no_request_program_or_tensor_after_compile() -> None:
    factory, recording, _provider, _bundle = _compile_one_track(
        (0.0, 0.5, 1.0),
        (_camera(), _camera(), _camera()),
    )

    # The recording test wrapper deliberately owns these objects, proving the
    # inspection is about the production factory's retained state rather than
    # objects that merely passed through compile_track().
    assert len(recording.requests) == len(recording.programs) == 1
    assert factory.memory_light_residency() == {
        "retained_compile_request_count": 0,
        "retained_compiled_program_count": 0,
        "retained_observation_record_count": 0,
        "retained_tensor_bytes": 0,
        "unbounded_cache_enabled": False,
    }


def test_frame_density_does_not_change_the_static_structural_program() -> None:
    sparse = (0.0, 0.5, 1.0)
    dense = tuple(index / 8 for index in range(9))
    _factory_a, recording_a, provider_a, bundle_a = _compile_one_track(
        sparse,
        tuple(_camera() for _ in sparse),
    )
    _factory_b, recording_b, provider_b, bundle_b = _compile_one_track(
        dense,
        tuple(_camera() for _ in dense),
    )

    program_a = recording_a.programs[0]
    program_b = recording_b.programs[0]
    # Provider/request/cache identities remain density-specific, while the
    # compiled semantic geometry and its native lowering remain identical.
    assert provider_a.generation_digest != provider_b.generation_digest
    assert (
        recording_a.requests[0].generation_digest
        != recording_b.requests[0].generation_digest
    )
    assert bundle_a.generation_digest != bundle_b.generation_digest
    assert program_a.generation_digest == program_b.generation_digest
    assert (
        bundle_a.sampler.lowering.generation_digest
        == bundle_b.sampler.lowering.generation_digest
    )
    assert bundle_a.sampler.generation_digest == bundle_b.sampler.generation_digest
    assert program_a.structural_tensor_bytes == program_b.structural_tensor_bytes
    assert program_a.total_node_count == program_b.total_node_count
    assert program_a.binding.ray_coefficients.shape == (12,)
    assert program_b.binding.ray_coefficients.shape == (12,)
    assert torch.count_nonzero(program_a.binding.ray_coefficients[[3, 4, 5, 9, 10, 11]]) == 0
    assert torch.equal(program_a.binding.ray_coefficients, program_b.binding.ray_coefficients)


def test_intermediate_camera_change_fails_instead_of_endpoint_fitting() -> None:
    # Endpoints are identical. An endpoint-only affine fit would incorrectly
    # accept this path and erase the intermediate camera motion.
    with pytest.raises(
        PaperKineticUnsupportedCameraPathError,
        match="camera record 1 differs from record 0",
    ):
        _compile_one_track(
            (0.0, 0.5, 1.0),
            (_camera(), _camera(translation_x=0.25), _camera()),
        )


def test_one_frame_and_mutated_camera_contracts_fail_closed() -> None:
    with pytest.raises(
        PaperKineticUnsupportedCameraPathError,
        match="at least two increasing camera times",
    ):
        _compile_one_track((0.0,), (_camera(),))

    _factory, recording, _provider, _bundle = _compile_one_track(
        (0.0, 1.0),
        (_camera(), _camera()),
    )
    request = recording.requests[0]
    request.cameras[1].lens_model = "projective_matrix"  # type: ignore[assignment]
    with pytest.raises(
        PaperKineticUnsupportedCameraPathError,
        match="unsupported lens model",
    ):
        recording.factory.compile_track(request)


def test_site_budget_fails_before_exact_track_compilation() -> None:
    with pytest.raises(MemoryError, match="maximum_sites_per_track_compile"):
        _compile_one_track(
            (0.0, 1.0),
            (_camera(), _camera()),
            factory=_production_factory(maximum_sites_per_track_compile=1),
            world_initializer=_TwoSiteWorldInitializer(),
        )


def test_factory_digest_binds_rank_policy_and_rejects_unsealed_values() -> None:
    first = _production_factory(node_count=4)
    repeated = _production_factory(node_count=4)
    changed = _production_factory(node_count=8)
    assert first.generation_digest == repeated.generation_digest
    assert first.generation_digest != changed.generation_digest

    with pytest.raises(ValueError, match="node_count"):
        _production_factory(node_count=1)
    with pytest.raises(ValueError, match="provenance changed"):
        replace(first, generation_digest=_sha256("forged-factory")).assert_current()
