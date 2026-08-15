from __future__ import annotations

import gc
import weakref
from dataclasses import replace

import pytest
import tests.test_paper_kinetic_lazy_program_bundles as lazy_fixture
import torch
from kinetic_native_lazy_bundle_lane import (
    LANE_STATUS,
    materialize_paper_kinetic_native_lazy_bundle_lane,
    prepare_paper_kinetic_native_lazy_bundle_lane,
    prepare_paper_kinetic_native_lazy_bundle_lane_construction_lifetime,
)
from paper_kinetic_lazy_program_bundles import (
    observations_from_spacetime_batch,
    prepare_paper_kinetic_lazy_program_bundle_provider,
)
from powerfoam_training_data import PowerFoamRayProvider, PowerFoamTargetProvider
from test_kinetic_ragged_paper_step_cpu_fake_native import _FakeNativeOps


def _case():
    source = lazy_fixture.LazyTargetSource()
    device = torch.device("cpu")
    target_provider = PowerFoamTargetProvider(source=source, device=device)
    ray_provider = PowerFoamRayProvider(
        cameras=lazy_fixture._camera_grid(),
        height=source.height,
        width=source.width,
        device=device,
    )
    provider = prepare_paper_kinetic_lazy_program_bundle_provider(
        dataset_generation_digest=lazy_fixture._sha256(
            "native-lazy-bundle-lane-test"
        ),
        target_provider=target_provider,
        ray_provider=ray_provider,
        frame_times=(0.0, 0.4, 1.0),
        height=source.height,
        width=source.width,
        maximum_tracks_per_bundle=2,
        maximum_observations_per_bundle=3,
        maximum_rows_per_native_block=1,
        world_initializer=lazy_fixture.OneSiteWorldInitializer(),
        program_factory=lazy_fixture.ExactStaticRayProgramFactory(),
    )
    observations = observations_from_spacetime_batch(
        lazy_fixture._batch(),
        pixel_indices_by_batch_position=((0, 1), (1, 2), (3,)),
    )
    bundle = next(provider.iter_spatial_bundles(observations, device=device))
    native_ops = _FakeNativeOps()
    lane = prepare_paper_kinetic_native_lazy_bundle_lane(
        bundle,
        provider,
        native_ops,
        device=device,
        backend_provenance="cpu-contract-double/exact-production-op-object",
    )
    return provider, bundle, native_ops, lane


def test_cold_lane_builds_one_runtime_per_block_in_canonical_order() -> None:
    provider, bundle, native_ops, lane = _case()
    canonical_blocks = tuple(
        block
        for bucket in bundle.sampler.lowering.buckets
        for block in bucket.blocks
    )

    assert len(canonical_blocks) == 2
    assert lane.native_runtime_count == bundle.sampler.lowering.block_count == 2
    assert tuple(
        runtime.payload.block for runtime in lane.runtimes
    ) == canonical_blocks
    assert lane.canonical_native_block_generation_digests == tuple(
        block.generation_digest for block in canonical_blocks
    )
    assert tuple(
        binding.runtime for binding in lane.executor.bindings
    ) == lane.runtimes
    assert lane.bundle is bundle
    assert lane.runtime_status == LANE_STATUS
    assert native_ops.forward_calls == 0
    assert native_ops.material_vjp_calls == 0
    assert native_ops.sample_launch_calls == 0
    lane.assert_cold_current(provider)
    assert lane.runtime_for_native_block_digest(
        canonical_blocks[0].generation_digest
    ) is lane.runtimes[0]


def test_two_phase_lane_installs_every_runtime_transfer_lifetime_first() -> None:
    provider, bundle, native_ops, _lane = _case()
    lifetime = (
        prepare_paper_kinetic_native_lazy_bundle_lane_construction_lifetime(
            bundle,
            provider,
            native_ops,
            device="cpu",
            backend_provenance=(
                "cpu-contract-double/exact-production-op-object"
            ),
        )
    )
    lifetime.assert_retained()
    assert lifetime.phase == "installed"
    assert lifetime.provider is provider
    assert lifetime.payloads == []
    assert lifetime.runtime_lifetimes == []
    assert lifetime.runtimes == []

    lane = materialize_paper_kinetic_native_lazy_bundle_lane(lifetime)
    lifetime.assert_retained()
    assert lifetime.phase == "transferred"
    assert lifetime.provider is None
    assert len(lifetime.runtime_lifetimes) == lane.native_runtime_count
    assert tuple(lifetime.runtimes) == lane.runtimes
    assert all(
        runtime_lifetime.phase == "materialized"
        and runtime_lifetime.runtime is runtime
        and runtime_lifetime.current_transfer_source is None
        for runtime_lifetime, runtime in zip(
            lifetime.runtime_lifetimes,
            lane.runtimes,
            strict=True,
        )
    )
    assert lane._construction_lifetime is lifetime
    lane.assert_cold_current(provider)
    with pytest.raises(ValueError, match="already used"):
        materialize_paper_kinetic_native_lazy_bundle_lane(lifetime)


def test_resident_logical_bytes_are_exactly_partitioned_and_f_invariant() -> None:
    _provider, _bundle, _native_ops, lane = _case()
    sparse = lane.memory_report(5)
    dense = lane.memory_report(300)

    assert sparse.resident_logical_tensor_bytes > 0
    assert lane.resident_tensor_bytes == sparse.resident_logical_tensor_bytes
    assert sparse.bundle_logical_tensor_bytes > 0
    assert sparse.runtime_additional_logical_tensor_bytes > 0
    assert sparse.executor_additional_logical_tensor_bytes == 0
    assert sparse.resident_logical_tensor_bytes == (
        sparse.bundle_logical_tensor_bytes
        + sparse.runtime_additional_logical_tensor_bytes
        + sparse.executor_additional_logical_tensor_bytes
    )
    assert sparse.resident_logical_tensor_object_count == (
        sparse.bundle_logical_tensor_object_count
        + sparse.runtime_additional_logical_tensor_object_count
        + sparse.executor_additional_logical_tensor_object_count
    )
    assert dense.resident_logical_tensor_bytes == sparse.resident_logical_tensor_bytes
    assert (
        dense.resident_logical_tensor_object_count
        == sparse.resident_logical_tensor_object_count
    )
    assert dense.persistent_frame_tensor_bytes == 0
    assert dense.persistent_sample_tensor_bytes == 0
    assert dense.persistent_target_tensor_bytes == 0
    assert dense.persistent_prediction_tensor_bytes == 0
    assert dense.retained_provider_count == 0
    assert dense.intended_maximum_live_native_lane_count == 1
    assert dense.one_live_lane_enforced_by_lane_object is False
    assert dense.droppable_as_one_lane_unit
    assert not dense.requested_observation_count_affects_resident_bytes


def test_provider_bundle_order_and_native_abi_changes_fail_closed() -> None:
    provider, bundle, native_ops, lane = _case()
    foreign_provider, _foreign_bundle, _foreign_ops, _foreign_lane = _case()

    with pytest.raises(ValueError, match="different bundle provider"):
        lane.assert_cold_current(foreign_provider)
    stale_bundle = replace(bundle, generation_digest="stale")
    with pytest.raises(ValueError, match="lazy bundle generation changed"):
        prepare_paper_kinetic_native_lazy_bundle_lane(
            stale_bundle,
            provider,
            native_ops,
            device="cpu",
            backend_provenance="cpu-contract-double/exact-production-op-object",
        )
    reversed_lane = replace(lane, runtimes=tuple(reversed(lane.runtimes)))
    with pytest.raises(ValueError, match="identity/memory contract changed"):
        reversed_lane.assert_warm_layout()
    with pytest.raises(ValueError, match="must match its union-local bundle"):
        prepare_paper_kinetic_native_lazy_bundle_lane(
            bundle,
            provider,
            native_ops,
            device="meta",
            backend_provenance="cpu-contract-double/exact-production-op-object",
        )

    native_ops.prepare_kinetic_ragged_p0_lie_sample_block = (
        lambda *args, **kwargs: None
    )
    with pytest.raises(ValueError, match="ABI/generation contract changed"):
        lane.assert_warm_layout()


def test_lane_does_not_retain_provider_and_drops_all_lane_owned_objects() -> None:
    provider, bundle, native_ops, lane = _case()
    provider_reference = weakref.ref(provider)
    bundle_reference = weakref.ref(bundle)
    runtime_references = tuple(weakref.ref(runtime) for runtime in lane.runtimes)
    executor_reference = weakref.ref(lane.executor)
    native_ops_reference = weakref.ref(native_ops)

    del provider
    gc.collect()
    assert provider_reference() is None
    assert bundle_reference() is bundle
    assert all(reference() is not None for reference in runtime_references)
    assert executor_reference() is lane.executor

    del bundle
    del native_ops
    del lane
    gc.collect()
    assert bundle_reference() is None
    assert all(reference() is None for reference in runtime_references)
    assert executor_reference() is None
    assert native_ops_reference() is None
