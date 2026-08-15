from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import pytest
import tests.test_paper_kinetic_lazy_program_bundles as lazy_fixture
from paper_kinetic_lazy_program_bundles import PaperKineticObservation
from paper_kinetic_replayable_observations import (
    OBSERVATION_IDENTITY_LOGICAL_BYTES,
    TRACK_ID_LOGICAL_BYTES,
    PaperKineticDenseObservationMemoryPolicy,
    prepare_paper_kinetic_replayable_dense_observation_source,
)
from paper_training_types import SpacetimeBatch


def _policy(
    *,
    maximum_request_track_count: int = 2,
    maximum_chunk_observation_count: int = 2,
    maximum_chunk_observation_logical_bytes: int = 64,
    maximum_retained_frame_metadata_count: int = 64,
    maximum_retained_frame_metadata_logical_bytes: int = 4096,
) -> PaperKineticDenseObservationMemoryPolicy:
    return PaperKineticDenseObservationMemoryPolicy(
        maximum_persistent_observation_count=0,
        maximum_persistent_observation_logical_bytes=0,
        maximum_retained_frame_metadata_count=(
            maximum_retained_frame_metadata_count
        ),
        maximum_retained_frame_metadata_logical_bytes=(
            maximum_retained_frame_metadata_logical_bytes
        ),
        maximum_live_generated_observation_count=1,
        maximum_live_generated_observation_logical_bytes=(
            OBSERVATION_IDENTITY_LOGICAL_BYTES
        ),
        maximum_request_track_count=maximum_request_track_count,
        maximum_request_track_logical_bytes=(
            maximum_request_track_count * TRACK_ID_LOGICAL_BYTES
        ),
        maximum_chunk_observation_count=maximum_chunk_observation_count,
        maximum_chunk_observation_logical_bytes=(
            maximum_chunk_observation_logical_bytes
        ),
    )


def _source(**policy_overrides):
    target_source, _rays, _initializer, _factory, provider, _observations = (
        lazy_fixture._fixture()
    )
    source = prepare_paper_kinetic_replayable_dense_observation_source(
        provider,
        lazy_fixture._batch(),
        memory_policy=_policy(**policy_overrides),
    )
    return target_source, provider, source


def test_dense_manifest_replays_exact_coverage_in_bounded_spatial_chunks() -> None:
    target_source, provider, source = _source()

    accounting = source.accounting()
    assert source.observation_count == len(lazy_fixture._batch().samples) * 6
    assert accounting["logical_observation_count"] == 18
    assert accounting["persistent_observation_count"] == 0
    assert accounting["persistent_observation_logical_bytes"] == 0
    assert accounting["retained_frame_metadata_count"] == 5
    assert accounting["retained_frame_metadata_logical_bytes"] == 120
    assert accounting["effective_chunk_observation_capacity"] == 2
    assert accounting["dense_cartesian_observation_tuple_retained"] is False
    assert accounting["frame_axis_retained_by_track_request"] is False
    assert accounting["target_tensor_bytes"] == 0
    assert accounting["ray_tensor_bytes"] == 0
    assert target_source.calls == []

    session = source.open_session()
    recovered = []
    chunk_sizes = []
    for view_index in (0, 1):
        for start in range(0, provider.height * provider.width, 2):
            request = source.prepare_track_request(
                view_index=view_index,
                track_ids=range(start, start + 2),
            )
            assert not hasattr(request, "observations")
            assert not hasattr(request, "frame_indices")
            assert request.track_id_logical_bytes == 2 * TRACK_ID_LOGICAL_BYTES
            for chunk in session.iter_request_chunks(request):
                chunk.assert_self_consistent(source, request)
                assert chunk.logical_identity_bytes <= 64
                recovered.extend(
                    observation.sample_identity
                    for observation in chunk.observations
                )
                chunk_sizes.append(chunk.observation_count)

    receipt = session.seal()
    receipt.assert_current(session)
    assert receipt.observation_count == 18
    assert receipt.request_count == 6
    assert receipt.chunk_count == 9
    assert receipt.compact_manifest_digest == source.compact_manifest_digest
    assert max(chunk_sizes) == 2
    assert recovered == [
        (batch_position * 6 + pixel, view, frame, pixel)
        for view, positions in (
            (0, ((0, 0), (1, 2))),
            (1, ((2, 1),)),
        )
        for pixel in range(6)
        for batch_position, frame in positions
    ]
    assert target_source.calls == []

    replay = source.open_session()
    assert replay.compact_manifest_digest == receipt.compact_manifest_digest


class _CountedTrackSequence(Sequence[int]):
    def __init__(self, values: tuple[int, ...]) -> None:
        self.values = values
        self.item_access_count = 0

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index):
        self.item_access_count += 1
        return self.values[index]


def test_count_and_byte_budgets_fail_before_metadata_copy_or_replay() -> None:
    target_source, _rays, _initializer, _factory, provider, _observations = (
        lazy_fixture._fixture()
    )
    with pytest.raises(MemoryError, match="before allocation or replay"):
        prepare_paper_kinetic_replayable_dense_observation_source(
            provider,
            lazy_fixture._batch(),
            memory_policy=_policy(
                maximum_retained_frame_metadata_count=5,
            ),
        )
    assert target_source.calls == []

    _target_source, _provider, source = _source(
        maximum_request_track_count=2,
    )
    oversized = _CountedTrackSequence((0, 1, 2))
    with pytest.raises(MemoryError, match="before copy or replay"):
        source.prepare_track_request(view_index=0, track_ids=oversized)
    assert oversized.item_access_count == 0

    with pytest.raises(ValueError, match="cannot admit one generated identity"):
        PaperKineticDenseObservationMemoryPolicy(
            **{
                **_policy().__dict__,
                "maximum_live_generated_observation_logical_bytes": 31,
            }
        ).assert_valid()


def test_global_cursor_rejects_missing_reordered_and_abandoned_coverage() -> None:
    _target_source, _provider, source = _source(
        maximum_chunk_observation_count=1,
        maximum_chunk_observation_logical_bytes=32,
    )
    session = source.open_session()
    skipped = source.prepare_track_request(view_index=0, track_ids=(1,))
    with pytest.raises(ValueError, match="coverage cursor"):
        next(session.iter_request_chunks(skipped))
    assert session.poisoned is False

    first = source.prepare_track_request(view_index=0, track_ids=(0,))
    iterator = session.iter_request_chunks(first)
    chunk = next(iterator)
    assert chunk.observation_count == 1
    foreign = replace(
        chunk,
        observations=(
            PaperKineticObservation(
                observation_id=chunk.observations[0].observation_id,
                view_index=chunk.observations[0].view_index,
                frame_index=2,
                pixel_index=chunk.observations[0].pixel_index,
            ),
        ),
    )
    with pytest.raises(ValueError, match="changed or is foreign"):
        foreign.assert_self_consistent(source, first)
    duplicated = replace(
        chunk,
        observations=(chunk.observations[0], chunk.observations[0]),
        logical_identity_bytes=2 * OBSERVATION_IDENTITY_LOGICAL_BYTES,
    )
    with pytest.raises(ValueError, match="changed or is foreign"):
        duplicated.assert_self_consistent(source, first)
    iterator.close()
    assert session.poisoned is True
    with pytest.raises(ValueError, match="cannot be sealed"):
        session.seal()

    incomplete = source.open_session()
    request = source.prepare_track_request(view_index=0, track_ids=(0, 1))
    tuple(incomplete.iter_request_chunks(request))
    with pytest.raises(ValueError, match="exact manifest"):
        incomplete.seal()
    assert incomplete.poisoned is True


def test_frame_growth_changes_only_o_f_metadata_not_observation_residency() -> None:
    target_source, _rays, _initializer, _factory, provider, _observations = (
        lazy_fixture._fixture()
    )
    full_batch = lazy_fixture._batch()
    one_frame_batch = SpacetimeBatch(
        samples=full_batch.samples[:1],
        epoch=full_batch.epoch,
        batch_index=full_batch.batch_index,
        completes_epoch=False,
    )
    one = prepare_paper_kinetic_replayable_dense_observation_source(
        provider,
        one_frame_batch,
        memory_policy=_policy(),
    )
    three = prepare_paper_kinetic_replayable_dense_observation_source(
        provider,
        full_batch,
        memory_policy=_policy(),
    )

    assert one.observation_count == 6
    assert three.observation_count == 18
    assert one.retained_frame_metadata_count == 2
    assert three.retained_frame_metadata_count == 5
    assert one.retained_frame_metadata_logical_bytes == 48
    assert three.retained_frame_metadata_logical_bytes == 120
    assert one.accounting()["persistent_observation_count"] == 0
    assert three.accounting()["persistent_observation_count"] == 0
    assert target_source.calls == []


def test_structural_compile_identity_is_independent_of_sampled_frame_subset() -> None:
    _source, _rays, _initializer, factory, provider, _observations = (
        lazy_fixture._fixture()
    )
    first_observation = PaperKineticObservation(
        observation_id=11,
        view_index=0,
        frame_index=0,
        pixel_index=0,
    )
    last_observation = PaperKineticObservation(
        observation_id=29,
        view_index=0,
        frame_index=2,
        pixel_index=0,
    )

    first_bundle = next(
        provider.iter_canonical_spatial_bundles(
            (first_observation,),
            device="cpu",
        )
    )
    first_request = factory.calls[-1]
    last_bundle = next(
        provider.iter_canonical_spatial_bundles(
            (last_observation,),
            device="cpu",
        )
    )
    last_request = factory.calls[-1]

    assert first_bundle.observation_identities != last_bundle.observation_identities
    assert first_request.generation_digest == last_request.generation_digest
    assert first_bundle.factory_request_generation_digests == (
        last_bundle.factory_request_generation_digests
    )
    assert tuple(
        record.observation.frame_index for record in first_request.observations
    ) == (0, 2)
    assert tuple(
        record.observation.frame_index for record in last_request.observations
    ) == (0, 2)
