from __future__ import annotations

import gc
import weakref
from dataclasses import replace

import kinetic_compiled_cpu_artifact_store as artifact_store_module
import paper_kinetic_lazy_program_bundles as lazy_program_module
import pytest
import tests.test_paper_kinetic_lazy_program_bundles as lazy_fixture
import torch
from kinetic_compiled_cpu_artifact_store import (
    PaperKineticCompiledCpuArtifactStore,
    PaperKineticCompiledCpuArtifactStorePolicy,
    compile_paper_kinetic_compiled_cpu_artifact,
    prepare_paper_kinetic_compiled_cpu_artifact_key,
    seal_paper_kinetic_compiled_cpu_artifact_from_bundle,
)
from kinetic_multichart_transfer_program import compile_kinetic_multichart_p0_program
from kinetic_owner_chart_compiler import compile_exact_kinetic_owner_charts
from paper_kinetic_lazy_program_bundles import (
    PaperKineticObservation,
    PaperKineticTrackProgramRequest,
    prepare_paper_kinetic_lazy_program_bundle_provider,
)
from powerfoam_training_data import PowerFoamRayProvider, PowerFoamTargetProvider


class _NonRetainingStaticProgramFactory:
    provenance = "bounded-store-nonretaining-static-program-factory-v1"

    def __init__(self) -> None:
        self.generation_digest = lazy_fixture._sha256(self.provenance)
        self.compile_count = 0

    def compile_track(self, request: PaperKineticTrackProgramRequest):
        request.assert_self_consistent()
        self.compile_count += 1
        ray = request.observations[0].ray_origin_direction
        coefficients = torch.tensor(
            [*ray[:3], 0.0, 0.0, 0.0, *ray[3:], 0.0, 0.0, 0.0],
            dtype=torch.float64,
        )
        owner = compile_exact_kinetic_owner_charts(
            request.world.sites,
            coefficients,
            t_min=request.frame_times[0],
            t_max=request.frame_times[-1],
            near=0.0,
            far=2.0,
        )
        assert owner.passed
        return compile_kinetic_multichart_p0_program(
            owner,
            request.world.sites,
            coefficients,
            node_count=2,
        )

    def memory_light_residency(self) -> dict[str, int | bool]:
        return {
            "retained_compile_request_count": 0,
            "retained_compiled_program_count": 0,
            "retained_observation_record_count": 0,
            "retained_tensor_bytes": 0,
            "unbounded_cache_enabled": False,
        }


def _provider(*, camera_grid=None, height=None, width=None):
    source = lazy_fixture.LazyTargetSource()
    if height is not None:
        source.height = height
    if width is not None:
        source.width = width
    target_provider = PowerFoamTargetProvider(source=source, device="cpu")
    ray_provider = PowerFoamRayProvider(
        cameras=lazy_fixture._camera_grid() if camera_grid is None else camera_grid,
        height=source.height,
        width=source.width,
        device="cpu",
    )
    factory = _NonRetainingStaticProgramFactory()
    provider = prepare_paper_kinetic_lazy_program_bundle_provider(
        dataset_generation_digest=lazy_fixture._sha256("bounded-store-dataset-v1"),
        target_provider=target_provider,
        ray_provider=ray_provider,
        frame_times=(0.0, 0.4, 1.0),
        height=source.height,
        width=source.width,
        maximum_tracks_per_bundle=2,
        maximum_observations_per_bundle=4,
        maximum_rows_per_native_block=1,
        world_initializer=lazy_fixture.OneSiteWorldInitializer(),
        program_factory=factory,
    )
    return source, factory, provider


def _observations(*frame_pixel: tuple[int, int]) -> tuple[PaperKineticObservation, ...]:
    canonical = tuple(sorted(frame_pixel, key=lambda item: (item[1], item[0])))
    return tuple(
        PaperKineticObservation(
            observation_id=index,
            view_index=0,
            frame_index=frame,
            pixel_index=pixel,
        )
        for index, (frame, pixel) in enumerate(canonical)
    )


def _compile_artifact(provider, observations, key):
    iterator = provider.iter_canonical_spatial_bundles(observations, device="cpu")
    try:
        bundle = next(iterator)
    finally:
        iterator.close()
    return seal_paper_kinetic_compiled_cpu_artifact_from_bundle(
        bundle,
        provider,
        key=key,
    )


def test_direct_cpu_compile_skips_bundle_device_mapping_and_enters_store(
    monkeypatch,
) -> None:
    target_source, factory, provider = _provider()
    store = PaperKineticCompiledCpuArtifactStore(
        PaperKineticCompiledCpuArtifactStorePolicy(
            maximum_entries=1,
            maximum_resident_accounted_bytes=10_000_000,
        )
    )
    monkeypatch.setattr(
        lazy_program_module,
        "prepare_paper_kinetic_union_local_spatial_bundle",
        lambda *_args, **_kwargs: pytest.fail(
            "direct CPU artifact compile constructed a union-local device mapping"
        ),
    )
    monkeypatch.setattr(
        lazy_program_module,
        "_materialize_lazy_bundle",
        lambda *_args, **_kwargs: pytest.fail(
            "direct CPU artifact compile entered the legacy bundle path"
        ),
    )

    acquired = store.acquire(
        provider,
        view_index=0,
        track_ids=(0, 1),
        maximum_artifact_accounted_bytes=10_000_000,
        compile_artifact=lambda key: compile_paper_kinetic_compiled_cpu_artifact(
            provider,
            key,
        ),
    )

    assert acquired.cache_status == "cold_compiled"
    assert acquired.artifact.track_ids == (0, 1)
    assert acquired.artifact.key.factory_provenance == provider.factory_provenance
    assert acquired.artifact.key.factory_generation_digest == provider.factory_generation_digest
    assert acquired.artifact.sampler.requested_frame_sampling_used_for_compile is False
    assert acquired.artifact.retained_union_device_mapping_tensor_bytes == 0
    assert acquired.artifact.retained_observation_count == 0
    assert factory.compile_count == 2
    assert target_source.calls == []


def test_direct_cpu_compile_fails_before_track_compile_on_noncontiguous_or_oversized_tracks() -> None:
    _source, factory, provider = _provider()
    noncontiguous = prepare_paper_kinetic_compiled_cpu_artifact_key(
        provider,
        view_index=0,
        track_ids=(0, 2),
    )
    with pytest.raises(ValueError, match="contiguous and increasing"):
        compile_paper_kinetic_compiled_cpu_artifact(provider, noncontiguous)
    assert factory.compile_count == 0

    oversized = prepare_paper_kinetic_compiled_cpu_artifact_key(
        provider,
        view_index=0,
        track_ids=(0, 1, 2),
    )
    with pytest.raises(MemoryError, match="bounded spatial capacity"):
        compile_paper_kinetic_compiled_cpu_artifact(provider, oversized)
    assert factory.compile_count == 0


def test_sampled_frames_do_not_enter_key_and_warm_hit_avoids_track_compile(
    monkeypatch,
) -> None:
    _source, factory, provider = _provider()
    first_observations = _observations((0, 0), (0, 1))
    later_observations = _observations((2, 0), (1, 1))
    first_key = prepare_paper_kinetic_compiled_cpu_artifact_key(
        provider,
        view_index=0,
        track_ids=(0, 1),
    )
    later_key = prepare_paper_kinetic_compiled_cpu_artifact_key(
        provider,
        view_index=0,
        track_ids=tuple(sorted({item.pixel_index for item in later_observations})),
    )
    assert first_key == later_key

    store = PaperKineticCompiledCpuArtifactStore(
        PaperKineticCompiledCpuArtifactStorePolicy(
            maximum_entries=2,
            maximum_resident_accounted_bytes=10_000_000,
        )
    )
    cold = store.acquire(
        provider,
        view_index=0,
        track_ids=(0, 1),
        maximum_artifact_accounted_bytes=10_000_000,
        compile_artifact=lambda key: _compile_artifact(
            provider,
            first_observations,
            key,
        ),
    )
    assert cold.cache_status == "cold_compiled"
    assert factory.compile_count == 2
    cold.artifact.assert_replays_selected_observations(provider, later_observations)
    monkeypatch.setattr(
        artifact_store_module,
        "build_camera_rays_at_pixels",
        lambda *_args, **_kwargs: pytest.fail("warm hit regenerated frame rays"),
    )

    warm = store.acquire(
        provider,
        view_index=0,
        track_ids=(0, 1),
        maximum_artifact_accounted_bytes=10_000_000,
        compile_artifact=lambda _key: pytest.fail("warm lookup recompiled"),
    )
    assert warm.warm_hit
    assert warm.artifact is cold.artifact
    assert warm.avoided_compile_track_count == 2
    assert factory.compile_count == 2
    report = store.report()
    assert report.hit_count == report.miss_count == 1
    assert report.cold_compiled_track_count == 2
    assert report.avoided_compile_track_count == 2
    assert report.cold_full_camera_path_validation_count == 1
    assert report.warm_identity_version_validation_count == 1
    assert report.warm_full_camera_path_validation_count == 0
    assert report.current_resident_accounted_bytes == cold.artifact.accounted_resident_bytes
    assert report.retained_observation_count == 0
    assert report.retained_dataset_frame_time_count == 0
    assert not report.device_runtime_cache_enabled


def test_budget_failure_precedes_compile_and_lru_is_bounded() -> None:
    _source, factory, provider = _provider()
    calls = 0

    def forbidden_compile(_key):
        nonlocal calls
        calls += 1
        raise AssertionError("budget preflight did not fail before compile")

    tiny = PaperKineticCompiledCpuArtifactStore(
        PaperKineticCompiledCpuArtifactStorePolicy(
            maximum_entries=1,
            maximum_resident_accounted_bytes=1,
        )
    )
    with pytest.raises(MemoryError, match="upper bound exceeds"):
        tiny.acquire(
            provider,
            view_index=0,
            track_ids=(0,),
            maximum_artifact_accounted_bytes=2,
            compile_artifact=forbidden_compile,
        )
    assert calls == 0
    assert factory.compile_count == 0
    assert tiny.report().compile_attempt_count == 0

    store = PaperKineticCompiledCpuArtifactStore(
        PaperKineticCompiledCpuArtifactStorePolicy(
            maximum_entries=1,
            maximum_resident_accounted_bytes=10_000_000,
        )
    )
    first = store.acquire(
        provider,
        view_index=0,
        track_ids=(0,),
        maximum_artifact_accounted_bytes=10_000_000,
        compile_artifact=lambda key: _compile_artifact(
            provider,
            _observations((0, 0)),
            key,
        ),
    )
    second = store.acquire(
        provider,
        view_index=0,
        track_ids=(1,),
        maximum_artifact_accounted_bytes=10_000_000,
        compile_artifact=lambda key: _compile_artifact(
            provider,
            _observations((2, 1)),
            key,
        ),
    )
    assert not first.warm_hit and not second.warm_hit
    assert second.evicted_entry_count == 1
    assert second.evicted_accounted_bytes == first.artifact.accounted_resident_bytes
    report = store.report()
    assert report.current_entry_count == 1
    assert report.current_resident_accounted_bytes == second.artifact.accounted_resident_bytes
    assert report.eviction_count == 1
    assert factory.compile_count == 2


def test_mutated_cached_tensor_fails_closed_and_is_removed() -> None:
    _source, _factory, provider = _provider()
    store = PaperKineticCompiledCpuArtifactStore(
        PaperKineticCompiledCpuArtifactStorePolicy(
            maximum_entries=1,
            maximum_resident_accounted_bytes=10_000_000,
        )
    )
    cold = store.acquire(
        provider,
        view_index=0,
        track_ids=(0,),
        maximum_artifact_accounted_bytes=10_000_000,
        compile_artifact=lambda key: _compile_artifact(
            provider,
            _observations((0, 0)),
            key,
        ),
    )
    cold.artifact.sampler.sources[0].program.charts[0].owners.add_(1)
    with pytest.raises(ValueError):
        store.acquire(
            provider,
            view_index=0,
            track_ids=(0,),
            maximum_artifact_accounted_bytes=10_000_000,
            compile_artifact=lambda _key: pytest.fail("stale hit recompiled silently"),
        )
    report = store.report()
    assert report.stale_rejection_count == 1
    assert report.current_entry_count == 0
    assert report.current_resident_accounted_bytes == 0


def test_outer_cold_camera_check_rejects_mutation_without_per_artifact_f_scan() -> None:
    _source, _factory, provider = _provider()
    store = PaperKineticCompiledCpuArtifactStore(
        PaperKineticCompiledCpuArtifactStorePolicy(
            maximum_entries=1,
            maximum_resident_accounted_bytes=10_000_000,
        )
    )
    store.acquire(
        provider,
        view_index=0,
        track_ids=(0,),
        maximum_artifact_accounted_bytes=10_000_000,
        compile_artifact=lambda key: _compile_artifact(
            provider,
            _observations((0, 0)),
            key,
        ),
    )
    provider.ray_provider.cameras[0][1].camera_to_world[0, 3] += 0.01
    warm = store.acquire(
        provider,
        view_index=0,
        track_ids=(0,),
        maximum_artifact_accounted_bytes=10_000_000,
        compile_artifact=lambda _key: pytest.fail("warm hit recompiled"),
    )
    assert warm.warm_hit
    with pytest.raises(ValueError, match="calibrated camera records changed"):
        provider.assert_current()


def test_store_retains_sampler_not_provider_bundle_or_observations() -> None:
    _source, _factory, provider = _provider()
    observations = _observations((0, 0), (2, 0))
    observation_refs = tuple(weakref.ref(item) for item in observations)
    provider_ref = weakref.ref(provider)
    store = PaperKineticCompiledCpuArtifactStore(
        PaperKineticCompiledCpuArtifactStorePolicy(
            maximum_entries=1,
            maximum_resident_accounted_bytes=10_000_000,
        )
    )
    acquired = store.acquire(
        provider,
        view_index=0,
        track_ids=(0,),
        maximum_artifact_accounted_bytes=10_000_000,
        compile_artifact=lambda key: _compile_artifact(provider, observations, key),
    )
    accounting = acquired.artifact.accounting()
    assert accounting["retained_observation_count"] == 0
    assert accounting["retained_target_tensor_bytes"] == 0
    assert accounting["retained_native_runtime_count"] == 0
    assert accounting["all_reachable_tensors_are_cpu"]

    del observations
    del provider
    gc.collect()
    assert all(reference() is None for reference in observation_refs)
    assert provider_ref() is None
    acquired.artifact.assert_current()


def test_foreign_key_and_non_affine_full_path_are_rejected() -> None:
    _source, _factory, provider = _provider()
    bundle_iterator = provider.iter_canonical_spatial_bundles(
        _observations((0, 0)),
        device="cpu",
    )
    try:
        bundle = next(bundle_iterator)
    finally:
        bundle_iterator.close()
    foreign_key = prepare_paper_kinetic_compiled_cpu_artifact_key(
        provider,
        view_index=0,
        track_ids=(1,),
    )
    with pytest.raises(ValueError):
        seal_paper_kinetic_compiled_cpu_artifact_from_bundle(
            bundle,
            provider,
            key=foreign_key,
        )

    # A sampled-frame-dependent factory can compile one frame but cannot enter
    # the observation-invariant store unless it reproduces the complete path.
    cameras = [list(path) for path in lazy_fixture._camera_grid()]
    moved = cameras[0][1]
    moved_transform = moved.camera_to_world.clone()
    moved_transform[0, 3] += 0.125
    cameras[0][1] = replace(
        moved,
        camera_to_world=moved_transform,
    )
    _source, _factory, moving_provider = _provider(camera_grid=tuple(tuple(path) for path in cameras))
    moving_iterator = moving_provider.iter_canonical_spatial_bundles(
        _observations((0, 0)),
        device="cpu",
    )
    try:
        moving_bundle = next(moving_iterator)
    finally:
        moving_iterator.close()
    with pytest.raises(ValueError, match="full calibrated camera path"):
        seal_paper_kinetic_compiled_cpu_artifact_from_bundle(
            moving_bundle,
            moving_provider,
        )

    direct_key = prepare_paper_kinetic_compiled_cpu_artifact_key(
        moving_provider,
        view_index=0,
        track_ids=(0,),
    )
    with pytest.raises(ValueError, match="full calibrated camera path"):
        compile_paper_kinetic_compiled_cpu_artifact(
            moving_provider,
            direct_key,
        )
