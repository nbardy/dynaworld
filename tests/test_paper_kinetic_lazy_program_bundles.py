from __future__ import annotations

import hashlib
from typing import Any

import pytest
import paper_kinetic_lazy_program_bundles as lazy_bundle_module
import torch
from camera import CameraSpec
from kinetic_multichart_transfer_program import compile_kinetic_multichart_p0_program
from kinetic_owner_chart_compiler import compile_exact_kinetic_owner_charts
from kinetic_power_word_compiler import AffineKineticPowerSites
from paper_kinetic_lazy_program_bundles import (
    PaperKineticTrackProgramRequest,
    PaperKineticWorldInitializationRequest,
    iter_canonical_observations_from_spacetime_batch,
    observations_from_spacetime_batch,
    prepare_paper_kinetic_lazy_bundle_construction_lifetime_slot,
    prepare_paper_kinetic_lazy_program_bundle_provider,
)
from paper_kinetic_ragged_sample_plan import PaperKineticRowRaggedSampleBlock
from paper_kinetic_sparse_sample_blocks import (
    iter_paper_kinetic_sparse_sample_blocks,
    prepare_paper_kinetic_sparse_sample_plan,
)
from paper_kinetic_step_target_frame_cache import (
    prepare_paper_kinetic_step_target_frame_cache,
)
from paper_training_types import SpacetimeBatch, SpacetimeSample
from powerfoam_training_data import (
    PowerFoamRayProvider,
    PowerFoamSelectedPixelRead,
    PowerFoamTargetProvider,
    ResidentPowerFoamTargetSource,
)


def _sha256(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class LazyTargetSource:
    view_count = 2
    frame_count = 3
    height = 2
    width = 3

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        self.full_frame_calls: list[
            tuple[tuple[int, ...], tuple[int, ...]]
        ] = []
        self.selected_pixel_calls: list[
            tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
        ] = []

    def select_view_frames(
        self,
        view_indices: tuple[int, ...],
        frame_indices: tuple[int, ...],
    ) -> torch.Tensor:
        self.calls.append((view_indices, frame_indices))
        self.full_frame_calls.append((view_indices, frame_indices))
        frames = torch.empty(
            (len(view_indices), 3, self.height, self.width),
            dtype=torch.float32,
        )
        pixels = torch.arange(self.height * self.width, dtype=torch.float32).reshape(
            self.height,
            self.width,
        )
        for position, (view_index, frame_index) in enumerate(zip(view_indices, frame_indices, strict=True)):
            base = float(100 * view_index + 10 * frame_index)
            frames[position, 0] = base + pixels
            frames[position, 1] = base + pixels + 0.25
            frames[position, 2] = base + pixels + 0.5
        return frames

    def select_view_frame_pixels_cpu(
        self,
        view_indices: tuple[int, ...],
        frame_indices: tuple[int, ...],
        pixel_indices: tuple[int, ...],
        *,
        maximum_source_decode_tensor_bytes: int,
    ) -> PowerFoamSelectedPixelRead:
        required_bytes = len(pixel_indices) * (5 * 8 + 3 * 4)
        if required_bytes > maximum_source_decode_tensor_bytes:
            raise MemoryError("fixture selected-pixel read exceeds its budget")
        self.calls.append((view_indices, frame_indices))
        self.selected_pixel_calls.append(
            (view_indices, frame_indices, pixel_indices)
        )
        rows = []
        for view_index, frame_index, pixel_index in zip(
            view_indices,
            frame_indices,
            pixel_indices,
            strict=True,
        ):
            base = float(100 * view_index + 10 * frame_index + pixel_index)
            rows.append((base, base + 0.25, base + 0.5))
        return PowerFoamSelectedPixelRead.seal(
            torch.tensor(rows, dtype=torch.float32),
            selection_mode="direct_pixels",
            source_provenance="paper_kinetic_lazy_test/direct_pixels_v1",
            source_visible_peak_logical_tensor_bytes_upper_bound=(
                required_bytes
            ),
        )

    def residency(self) -> dict[str, Any]:
        return {
            "source_kind": "paper_kinetic_lazy_test",
            "source_device": "fixture",
            "logical_bytes": self.view_count * self.frame_count * 3 * self.height * self.width * 4,
            "resident_bytes": 0,
            "full_source_resident": False,
            "disk_lazy_decode": True,
        }


class FullFrameOnlyLazyTargetSource(LazyTargetSource):
    """Compatibility source deliberately lacking a selected-pixel method."""

    select_view_frame_pixels_cpu = None


class OneSiteWorldInitializer:
    provenance = "one-site-world-initializer-v1"

    def __init__(self) -> None:
        self.generation_digest = _sha256("one-site-world-initializer-v1")
        self.calls: list[PaperKineticWorldInitializationRequest] = []
        self.sites = AffineKineticPowerSites(
            positions0=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
            velocities=torch.zeros((1, 3), dtype=torch.float64),
            weight_coefficients=torch.zeros((1, 1), dtype=torch.float64),
        )

    def initialize_world(
        self,
        request: PaperKineticWorldInitializationRequest,
    ) -> AffineKineticPowerSites:
        request.assert_self_consistent()
        self.calls.append(request)
        return self.sites


class ExactStaticRayProgramFactory:
    provenance = "exact-static-ray-program-factory-v1"

    def __init__(self) -> None:
        self.generation_digest = _sha256("exact-static-ray-program-factory-v1")
        self.calls: list[PaperKineticTrackProgramRequest] = []

    def compile_track(self, request: PaperKineticTrackProgramRequest):
        request.assert_self_consistent()
        self.calls.append(request)
        ray = request.observations[0].ray_origin_direction
        coefficients = torch.tensor(
            [*ray[0:3], 0.0, 0.0, 0.0, *ray[3:6], 0.0, 0.0, 0.0],
            dtype=torch.float64,
        )
        owner_program = compile_exact_kinetic_owner_charts(
            request.world.sites,
            coefficients,
            t_min=request.frame_times[0],
            t_max=request.frame_times[-1],
            near=0.0,
            far=2.0,
        )
        assert owner_program.passed
        return compile_kinetic_multichart_p0_program(
            owner_program,
            request.world.sites,
            coefficients,
            node_count=2,
        )


def _camera_grid() -> tuple[tuple[CameraSpec, ...], ...]:
    return tuple(
        tuple(
            CameraSpec(
                fx=4.0,
                fy=4.0,
                cx=1.5,
                cy=1.0,
                camera_to_world=torch.tensor(
                    [
                        [1.0, 0.0, 0.0, float(view)],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=torch.float64,
                ),
            )
            for _frame in range(3)
        )
        for view in range(2)
    )


def _batch() -> SpacetimeBatch:
    return SpacetimeBatch(
        samples=(
            SpacetimeSample(view_index=0, frame_index=0),
            SpacetimeSample(view_index=0, frame_index=2),
            SpacetimeSample(view_index=1, frame_index=1),
        ),
        epoch=4,
        batch_index=9,
        completes_epoch=False,
    )


def _fixture(
    *,
    maximum_tracks_per_bundle: int = 2,
    maximum_observations_per_bundle: int = 3,
    source: LazyTargetSource | None = None,
):
    source = LazyTargetSource() if source is None else source
    device = torch.device("cpu")
    target_provider = PowerFoamTargetProvider(source=source, device=device)
    ray_provider = PowerFoamRayProvider(
        cameras=_camera_grid(),
        height=source.height,
        width=source.width,
        device=device,
    )
    initializer = OneSiteWorldInitializer()
    factory = ExactStaticRayProgramFactory()
    provider = prepare_paper_kinetic_lazy_program_bundle_provider(
        dataset_generation_digest=_sha256("tiny-paper-dataset-v1"),
        target_provider=target_provider,
        ray_provider=ray_provider,
        frame_times=(0.0, 0.4, 1.0),
        height=source.height,
        width=source.width,
        maximum_tracks_per_bundle=maximum_tracks_per_bundle,
        maximum_observations_per_bundle=maximum_observations_per_bundle,
        maximum_rows_per_native_block=2,
        world_initializer=initializer,
        program_factory=factory,
    )
    observations = observations_from_spacetime_batch(
        _batch(),
        pixel_indices_by_batch_position=((0, 1), (1, 2), (3,)),
    )
    return source, ray_provider, initializer, factory, provider, observations


def test_sparse_real_observations_compile_lazily_without_cartesian_padding() -> None:
    source, _rays, initializer, factory, provider, observations = _fixture()

    assert len(initializer.calls) == 1
    assert factory.calls == []
    assert source.calls == []
    iterator = provider.iter_spatial_bundles(observations, device="cpu")
    first = next(iterator)

    assert len(factory.calls) == 2
    assert source.calls == []
    assert first.view_index == 0
    assert first.track_ids == (0, 1)
    assert first.observation_identities == (
        (0, 0, 0, 0),
        (1, 0, 0, 1),
        (2, 0, 2, 1),
    )
    assert first.observation_count == 3
    assert first.track_count == 2
    assert first.cartesian_padding_observation_count == 0
    assert first.sampler.track_ids == (0, 1)
    assert first.spatial_bundle.track_ids == (0, 1)
    assert all(row.program.binding.sites is provider.world.sites for row in first.sampler.rows)
    # Two tracks and two active frames would be four Cartesian samples.  Only
    # the three explicitly named observations exist in the bundle.
    assert first.observation_count < first.track_count * 2
    first.assert_exact_observation_coverage(tuple(record.observation for record in first.observations))

    remaining = tuple(iterator)
    assert [(bundle.view_index, bundle.track_ids, bundle.observation_count) for bundle in remaining] == [
        (0, (2,), 1),
        (1, (3,), 1),
    ]
    recovered = {identity for bundle in (first, *remaining) for identity in bundle.observation_identities}
    assert recovered == {observation.sample_identity for observation in observations}
    assert sum(bundle.observation_count for bundle in (first, *remaining)) == len(observations)
    assert source.calls == []

    accounting = provider.accounting()
    assert accounting["persistent_target_tensor_bytes"] == 0
    assert accounting["persistent_dense_ray_tensor_bytes"] == 0
    assert accounting["dense_track_frame_tensor_bytes"] == 0
    assert accounting["full_target_video_resident"] is False
    assert accounting["full_ray_video_resident"] is False
    assert accounting["bounded_sparse_sampled_observations_only"] is True
    assert accounting["dense_F_observation_residency_closed"] is False
    assert accounting["dense_F_requires_replayable_observation_source"] is True
    report = first.memory_report(requested_frame_count=300)
    assert report.persistent_frame_tensor_bytes == 0
    assert report.persistent_target_tensor_bytes == 0
    assert report.persistent_dense_ray_tensor_bytes == 0
    assert report.dense_track_frame_tensor_bytes == 0
    assert report.selected_ray_scalar_count == 18
    assert report.cartesian_padding_observation_count == 0
    assert report.observation_residency_scope == (
        "bounded_sparse_sampled_observations_only"
    )
    assert report.dense_frame_observation_streaming_implemented is False


def test_lazy_bundle_slot_retains_partial_union_construction(monkeypatch) -> None:
    _source, _rays, _initializer, _factory, provider, observations = _fixture()
    slot = prepare_paper_kinetic_lazy_bundle_construction_lifetime_slot()

    def interrupt_after_slot_install(
        sampler,
        *,
        track_ids,
        device,
        construction_lifetime=None,
    ):
        assert construction_lifetime is slot.active_lifetime
        construction_lifetime.assert_retained()
        construction_lifetime.phase = "transferring"
        raise KeyboardInterrupt("synthetic union-local transfer interruption")

    monkeypatch.setattr(
        lazy_bundle_module,
        "prepare_paper_kinetic_union_local_spatial_bundle",
        interrupt_after_slot_install,
    )
    iterator = provider.iter_canonical_spatial_bundles(
        observations,
        device="cpu",
        construction_lifetime_slot=slot,
    )
    with pytest.raises(KeyboardInterrupt, match="union-local transfer"):
        next(iterator)

    slot.assert_current()
    lifetime = slot.active_lifetime
    assert lifetime is not None
    lifetime.assert_retained()
    assert lifetime.phase == "transferring"
    assert slot.install_count == 1
    assert slot.completion_count == 0
    slot.release_active_after_completion_fence()
    assert slot.active_lifetime is None
    assert slot.release_after_completion_fence_count == 1


def test_bundle_partition_and_program_provenance_are_deterministic() -> None:
    source, _rays, _initializer, factory, provider, observations = _fixture()

    first = tuple(provider.iter_spatial_bundles(tuple(reversed(observations)), device="cpu"))
    first_call_count = len(factory.calls)
    second = tuple(provider.iter_spatial_bundles(observations, device="cpu"))

    assert first_call_count == 4
    assert len(factory.calls) == 8
    assert [bundle.generation_digest for bundle in first] == [bundle.generation_digest for bundle in second]
    assert [bundle.observation_identities for bundle in first] == [bundle.observation_identities for bundle in second]
    assert [bundle.track_ids for bundle in first] == [bundle.track_ids for bundle in second]
    assert source.calls == []
    for bundle in (*first, *second):
        bundle.assert_cold_current(provider)


def test_canonical_observation_stream_keeps_only_one_spatial_bundle_live() -> None:
    source, _rays, _initializer, factory, provider, _observations = _fixture()
    consumed: list[tuple[int, int, int, int]] = []
    canonical = iter_canonical_observations_from_spacetime_batch(
        _batch(),
        pixel_indices_by_batch_position=(range(0, 2), range(1, 3), range(3, 4)),
        image_pixel_count=6,
    )

    def tracked_observations():
        for observation in canonical:
            consumed.append(observation.sample_identity)
            yield observation

    iterator = provider.iter_canonical_spatial_bundles(
        tracked_observations(),
        device="cpu",
    )
    first = next(iterator)

    assert first.observation_identities == (
        (0, 0, 0, 0),
        (1, 0, 0, 1),
        (7, 0, 2, 1),
    )
    # The source has five observations.  Only the next track's first record is
    # consumed as bounded lookahead before the first compiled bundle is yielded.
    assert len(consumed) == 4
    assert len(factory.calls) == 2
    assert source.calls == []
    remaining = tuple(iterator)
    assert len(consumed) == 5
    assert [(bundle.view_index, bundle.track_ids) for bundle in remaining] == [
        (0, (2,)),
        (1, (3,)),
    ]
    assert provider.accounting()["provider_owned_retained_bundle_count"] == 0
    assert provider.accounting()["one_live_bundle_enforced_by_provider"] is False
    assert provider.accounting()["consumer_must_release_bundle_before_next"] is True


def test_stale_camera_world_factory_and_missing_coverage_fail_closed() -> None:
    _source, rays, initializer, factory, provider, observations = _fixture()
    bundle = next(provider.iter_spatial_bundles(observations, device="cpu"))
    covered = tuple(record.observation for record in bundle.observations)

    with pytest.raises(ValueError, match="coverage mismatch"):
        bundle.assert_exact_observation_coverage(covered[:-1])
    with pytest.raises(ValueError, match="coverage mismatch"):
        bundle.assert_exact_observation_coverage((*covered, observations[-1]))

    factory.generation_digest = _sha256("stale-factory")
    with pytest.raises(ValueError, match="factory provenance changed"):
        provider.assert_current()
    factory.generation_digest = _sha256("exact-static-ray-program-factory-v1")

    rays.cameras[0][0].camera_to_world[0, 3].add_(0.25)
    with pytest.raises(ValueError, match="calibrated camera records changed"):
        provider.assert_current()
    rays.cameras[0][0].camera_to_world[0, 3].sub_(0.25)

    initializer.sites.positions0[0, 0].add_(0.5)
    with pytest.raises(ValueError, match="world tensor content changed"):
        bundle.assert_cold_current(provider)


def test_provider_rejects_resident_targets_and_incomplete_observation_input() -> None:
    device = torch.device("cpu")
    resident = PowerFoamTargetProvider(
        source=ResidentPowerFoamTargetSource(torch.zeros((2, 3, 3, 2, 3), dtype=torch.float32)),
        device=device,
    )
    rays = PowerFoamRayProvider(
        cameras=_camera_grid(),
        height=2,
        width=3,
        device=device,
    )
    with pytest.raises(ValueError, match="nonresident target source"):
        prepare_paper_kinetic_lazy_program_bundle_provider(
            dataset_generation_digest=_sha256("resident-is-not-lazy"),
            target_provider=resident,
            ray_provider=rays,
            frame_times=(0.0, 0.4, 1.0),
            height=2,
            width=3,
            maximum_tracks_per_bundle=2,
            maximum_observations_per_bundle=3,
            maximum_rows_per_native_block=2,
            world_initializer=OneSiteWorldInitializer(),
            program_factory=ExactStaticRayProgramFactory(),
        )

    with pytest.raises(ValueError, match="cover every batch position"):
        observations_from_spacetime_batch(
            _batch(),
            pixel_indices_by_batch_position=((0,),),
        )
    with pytest.raises(ValueError, match="must be nonempty"):
        observations_from_spacetime_batch(
            _batch(),
            pixel_indices_by_batch_position=((0,), (), (1,)),
        )


def test_factory_program_must_cover_time_domain_and_replay_selected_rays() -> None:
    class WrongRayFactory(ExactStaticRayProgramFactory):
        def compile_track(self, request: PaperKineticTrackProgramRequest):
            super().compile_track(request)
            first = request.observations[0]
            wrong_ray = torch.tensor(
                [
                    first.ray_origin_direction[0] + 1.0,
                    *first.ray_origin_direction[1:3],
                    0.0,
                    0.0,
                    0.0,
                    *first.ray_origin_direction[3:6],
                    0.0,
                    0.0,
                    0.0,
                ],
                dtype=torch.float64,
            )
            owner = compile_exact_kinetic_owner_charts(
                request.world.sites,
                wrong_ray,
                t_min=request.frame_times[0],
                t_max=request.frame_times[-1],
                near=0.0,
                far=2.0,
            )
            return compile_kinetic_multichart_p0_program(
                owner,
                request.world.sites,
                wrong_ray,
                node_count=2,
            )

    source = LazyTargetSource()
    device = torch.device("cpu")
    target = PowerFoamTargetProvider(source=source, device=device)
    rays = PowerFoamRayProvider(
        cameras=_camera_grid(),
        height=2,
        width=3,
        device=device,
    )
    provider = prepare_paper_kinetic_lazy_program_bundle_provider(
        dataset_generation_digest=_sha256("wrong-ray-dataset"),
        target_provider=target,
        ray_provider=rays,
        frame_times=(0.0, 0.4, 1.0),
        height=2,
        width=3,
        maximum_tracks_per_bundle=2,
        maximum_observations_per_bundle=3,
        maximum_rows_per_native_block=2,
        world_initializer=OneSiteWorldInitializer(),
        program_factory=WrongRayFactory(),
    )
    observations = observations_from_spacetime_batch(
        _batch(),
        pixel_indices_by_batch_position=((0,), (1,), (2,)),
    )
    with pytest.raises(ValueError, match="does not reproduce a selected calibrated ray"):
        next(provider.iter_spatial_bundles(observations, device="cpu"))
    assert source.calls == []


def test_sparse_targets_stream_frame_major_into_executor_blocks() -> None:
    source, _rays, _initializer, _factory, provider, observations = _fixture()
    bundle = next(provider.iter_spatial_bundles(observations, device="cpu"))
    plan = prepare_paper_kinetic_sparse_sample_plan(
        bundle,
        provider,
        global_loss_element_count=30,
        loss_normalization_id="global-paper-step",
        maximum_samples_per_launch=1,
    )

    assert source.calls == []
    accounting = plan.accounting()
    assert accounting["frame_major_target_streaming"] is True
    assert accounting["whole_bundle_target_tuple_retained"] is False
    assert accounting["selected_target_python_scalar_count"] == 0
    assert accounting["maximum_selected_observations_per_frame"] == 2
    assert accounting["selected_frame_target_tensor_upper_bound_bytes"] == 24
    assert accounting["persistent_target_tensor_bytes"] == 0

    iterator = iter_paper_kinetic_sparse_sample_blocks(plan)
    first = next(iterator)
    assert source.calls == [((0, 0), (0, 0))]
    second = next(iterator)
    # Both first launches use frame 0.  It is decoded only once.
    assert source.calls == [((0, 0), (0, 0))]
    third = next(iterator)
    assert source.calls == [((0, 0), (0, 0)), ((0,), (2,))]
    with pytest.raises(StopIteration):
        next(iterator)
    target_reads = iterator.target_read_accounting()
    assert target_reads["selected_pixel_read_mode"] == "direct_pixels"
    assert target_reads["selected_pixel_read_acceptance_capable"] is True
    assert target_reads["selected_pixel_read_observation_count"] == 3
    assert target_reads["direct_selected_pixel_observation_count"] == 3
    assert target_reads["full_frame_fallback_observation_count"] == 0
    assert target_reads["full_frame_target_materialization_count"] == 0
    assert source.full_frame_calls == []
    assert len(source.selected_pixel_calls) == 2

    blocks = (first, second, third)
    assert all(isinstance(block, PaperKineticRowRaggedSampleBlock) for block in blocks)
    assert all(block.sample_count == 1 for block in blocks)
    assert all(block.global_loss_element_count == 30 for block in blocks)
    assert all(block.loss_scale == 1.0 / 30.0 for block in blocks)
    plan.assert_complete_launch_coverage(blocks)
    with pytest.raises(ValueError, match="missing or duplicate coverage"):
        plan.assert_complete_launch_coverage(blocks[:-1])

    target_by_record_index = {}
    for block in blocks:
        block.assert_cold_current(bundle.sampler)
        for position, record_index in enumerate(block.flat_sample_index_i64.tolist()):
            target_by_record_index[record_index] = block.target_rgb_f32[position]
    expected = (
        torch.tensor([0.0, 0.25, 0.5]),
        torch.tensor([1.0, 1.25, 1.5]),
        torch.tensor([21.0, 21.25, 21.5]),
    )
    assert tuple(sorted(target_by_record_index)) == (0, 1, 2)
    for record_index, target in enumerate(expected):
        torch.testing.assert_close(target_by_record_index[record_index], target)


def test_sparse_explicit_transfer_lifetime_blocks_advance_until_fenced() -> None:
    _source, _rays, _initializer, _factory, provider, observations = _fixture()
    bundle = next(provider.iter_spatial_bundles(observations, device="cpu"))
    plan = prepare_paper_kinetic_sparse_sample_plan(
        bundle,
        provider,
        global_loss_element_count=30,
        loss_normalization_id="explicit-transfer-lifetime",
        maximum_samples_per_launch=1,
    )
    stream = iter_paper_kinetic_sparse_sample_blocks(
        plan,
        require_explicit_transfer_settlement=True,
    )
    first = next(stream)
    lifetime = stream.active_lifetime_for(first)
    lifetime.assert_retained()
    assert lifetime.phase == "materialized"
    assert lifetime.sample_block is first
    assert lifetime.selected_frame_targets_f32 is not None
    assert lifetime.selected_pixel_read is not None
    assert lifetime.weights_f64 is not None
    assert lifetime.weights_transfer_f32 is not None
    assert lifetime.target_transfer_f32 is not None
    with pytest.raises(RuntimeError, match="settled before advancing"):
        next(stream)
    with pytest.raises(RuntimeError, match="unsettled transfer"):
        stream.close()

    stream.release_active_after_completion_fence(first)
    assert lifetime.released_after_completion_fence is True
    assert lifetime.phase == "released"
    assert lifetime.sample_block is None
    assert lifetime.selected_pixel_read is None
    assert lifetime.weights_f64 is None
    assert lifetime.weights_transfer_f32 is None
    assert lifetime.target_transfer_f32 is None
    second = next(stream)
    second_lifetime = stream.active_lifetime_for(second)
    assert second_lifetime is not lifetime
    stream.release_active_after_completion_fence(second)
    stream.close()


def test_sparse_target_fallback_is_receipted_as_full_frame_materialization() -> None:
    source = FullFrameOnlyLazyTargetSource()
    _source, _rays, _initializer, _factory, provider, observations = _fixture(
        source=source,
    )
    bundle = next(provider.iter_spatial_bundles(observations, device="cpu"))
    plan = prepare_paper_kinetic_sparse_sample_plan(
        bundle,
        provider,
        global_loss_element_count=30,
        loss_normalization_id="fallback-receipt",
        maximum_samples_per_launch=8,
    )

    stream = iter_paper_kinetic_sparse_sample_blocks(
        plan,
        maximum_source_decode_tensor_bytes=1_000,
    )
    tuple(stream)
    receipt = stream.target_read_accounting()

    assert receipt["selected_pixel_read_mode"] == "full_frame_fallback"
    assert receipt["selected_pixel_read_acceptance_capable"] is False
    assert receipt["direct_selected_pixel_observation_count"] == 0
    assert receipt["full_frame_fallback_observation_count"] == 3
    assert receipt["full_frame_target_materialization_count"] == 2
    assert receipt["peak_full_frame_materialization_tensor_bytes"] == 72
    assert len(source.full_frame_calls) == 2
    assert source.selected_pixel_calls == []


def test_dense_frame_track_residency_is_explicitly_closed() -> None:
    source, _rays, _initializer, _factory, provider, _observations = _fixture(
        maximum_observations_per_bundle=2,
    )
    dense_track_batch = SpacetimeBatch(
        samples=tuple(SpacetimeSample(view_index=0, frame_index=frame_index) for frame_index in range(3)),
        epoch=0,
        batch_index=0,
        completes_epoch=False,
    )
    dense_track = observations_from_spacetime_batch(
        dense_track_batch,
        pixel_indices_by_batch_position=((0,), (0,), (0,)),
    )

    with pytest.raises(
        ValueError,
        match="one paper kinetic track exceeds maximum_observations_per_bundle",
    ):
        next(provider.iter_spatial_bundles(dense_track, device="cpu"))
    assert source.calls == []
    assert provider.accounting()["dense_frame_observation_streaming_implemented"] is False


def test_step_target_frame_cache_decodes_shared_frames_once_across_bundles() -> None:
    source, _rays, _initializer, _factory, provider, observations = _fixture(
        maximum_tracks_per_bundle=1,
    )
    frame_bytes = provider.height * provider.width * 3 * 4
    cache = prepare_paper_kinetic_step_target_frame_cache(
        provider,
        maximum_resident_bytes=3 * frame_bytes,
    )
    block_count = 0

    for bundle in provider.iter_spatial_bundles(observations, device="cpu"):
        plan = prepare_paper_kinetic_sparse_sample_plan(
            bundle,
            provider,
            global_loss_element_count=len(observations) * 3,
            loss_normalization_id="shared-step-cache",
            maximum_samples_per_launch=2,
        )
        blocks = tuple(
            iter_paper_kinetic_sparse_sample_blocks(
                plan,
                target_frame_cache=cache,
            )
        )
        plan.assert_complete_launch_coverage(blocks)
        block_count += len(blocks)

    assert block_count > 0
    # Four spatial bundles make five bundle/frame requests, but there are only
    # three globally unique target frames in the logical step.
    assert source.calls == [
        ((0,), (0,)),
        ((0,), (2,)),
        ((1,), (1,)),
    ]
    live = cache.accounting()
    assert live["request_count"] == 5
    assert live["decode_count"] == 3
    assert live["decode_attempt_count"] == 3
    assert live["hit_count"] == 2
    assert live["cached_frame_count"] == 3
    assert live["resident_frame_tensor_bytes"] == 3 * frame_bytes
    assert live["peak_resident_frame_tensor_bytes"] == 3 * frame_bytes
    assert live["decode_each_unique_frame_at_most_once"] is True
    assert live["unbounded_fallback_enabled"] is False

    cache.close()
    closed = cache.accounting()
    assert closed["closed"] is True
    assert closed["close_count"] == 1
    assert closed["cached_frame_count"] == 0
    assert closed["resident_frame_tensor_bytes"] == 0
    assert closed["decode_count"] == 3
    with pytest.raises(ValueError, match="cache is closed"):
        cache.get_frame(provider, view_index=0, frame_index=0)


def test_step_target_frame_cache_preflights_and_stale_frames_fail_closed() -> None:
    source, _rays, _initializer, _factory, provider, _observations = _fixture()
    frame_bytes = provider.height * provider.width * 3 * 4
    too_small = prepare_paper_kinetic_step_target_frame_cache(
        provider,
        maximum_resident_bytes=frame_bytes - 1,
    )

    with pytest.raises(MemoryError, match="before decode"):
        too_small.get_frame(provider, view_index=0, frame_index=0)
    assert source.calls == []
    rejected = too_small.accounting()
    assert rejected["preflight_rejection_count"] == 1
    assert rejected["decode_attempt_count"] == 0
    assert rejected["resident_frame_tensor_bytes"] == 0
    too_small.close()

    cache = prepare_paper_kinetic_step_target_frame_cache(
        provider,
        maximum_resident_bytes=frame_bytes,
    )
    frame = cache.get_frame(provider, view_index=0, frame_index=0)
    assert source.calls == [((0,), (0,))]
    # The tensor is deliberately private; mutating it simulates a contract
    # violation and must invalidate the sealed read-only handle.
    frame._rgb_chw_f32.add_(1.0)
    with pytest.raises(ValueError, match="stale or foreign"):
        cache.get_frame(provider, view_index=0, frame_index=0)
    assert cache.accounting()["poisoned"] is True
    with pytest.raises(ValueError, match="closed after stale state"):
        cache.close()
    closed = cache.accounting()
    assert closed["closed"] is True
    assert closed["cached_frame_count"] == 0
    assert closed["resident_frame_tensor_bytes"] == 0
