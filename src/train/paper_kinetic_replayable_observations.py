"""Bounded replayable dense-frame observations for kinetic WorldFoam.

The structural kinetic compiler owns tracks, not sampled frames.  This module
therefore represents a dense paper step as a compact deterministic product:

``selected (view, frame) metadata x pixel tracks``.

It never materializes that Cartesian product.  A source retains only the
``O(F)`` :class:`~paper_training_types.SpacetimeBatch` metadata and a canonical
position index.  A frame-axis-free spatial bundle requests a bounded,
contiguous set of ``(view, pixel)`` tracks; the source then replays those
tracks' observations in bounded chunks.  A session cursor proves exact global
coverage without an ``O(PF)`` set or tuple.

Every storage dimension has both a count and logical-byte budget.  The dense
metadata upper bound and request/chunk bounds are rejected before sorting,
tuple allocation, or opening an observation replay.  Logical bytes count
integer payloads, not CPython object headers; Python-allocator and device peaks
remain explicitly unmeasured.  This module creates no target, ray, or device
tensor and launches no native work.

This is the missing data-source prerequisite, not yet an end-to-end dense-F
trainer claim.  The legacy ``PaperKineticLazyProgramBundle`` still embeds
selected ray records and its sparse partitioner rejects a track longer than
``maximum_observations_per_bundle``.  Integration must pair this source with a
frame-free cached structural sampler/program artifact and adapt replayed chunks
to native sample launches; the current sparse coordinator does neither.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from paper_kinetic_lazy_program_bundles import (
    PaperKineticLazyProgramBundleProvider,
    PaperKineticObservation,
)
from paper_training_types import SpacetimeBatch

DENSE_OBSERVATION_SOURCE_PROVENANCE = (
    "paper-kinetic-replayable-dense-observation-source-v1"
)
DENSE_OBSERVATION_REQUEST_PROVENANCE = (
    "paper-kinetic-replayable-dense-observation-request-v1"
)
DENSE_OBSERVATION_CHUNK_PROVENANCE = (
    "paper-kinetic-replayable-dense-observation-chunk-v1"
)
DENSE_OBSERVATION_RECEIPT_PROVENANCE = (
    "paper-kinetic-replayable-dense-observation-receipt-v1"
)

OBSERVATION_IDENTITY_LOGICAL_BYTES = 4 * 8
TRACK_ID_LOGICAL_BYTES = 8
# One batch sample contributes two int64 coordinates, one canonical position,
# and at most one three-int64 view-span record.  The last term is a deliberate
# worst case (one selected frame per view), so preflight happens before sort.
FRAME_METADATA_UPPER_BOUND_LOGICAL_BYTES = 6 * 8

_SOURCE_SEAL = object()
_REQUEST_SEAL = object()
_CHUNK_SEAL = object()
_SESSION_SEAL = object()
_RECEIPT_SEAL = object()


@dataclass(frozen=True)
class PaperKineticDenseObservationMemoryPolicy:
    """Hard source/session bounds; total logical observations are unbounded."""

    maximum_persistent_observation_count: int
    maximum_persistent_observation_logical_bytes: int
    maximum_retained_frame_metadata_count: int
    maximum_retained_frame_metadata_logical_bytes: int
    maximum_live_generated_observation_count: int
    maximum_live_generated_observation_logical_bytes: int
    maximum_request_track_count: int
    maximum_request_track_logical_bytes: int
    maximum_chunk_observation_count: int
    maximum_chunk_observation_logical_bytes: int

    def assert_valid(self) -> None:
        for name, value in self.__dict__.items():
            _require_nonnegative_int(value, name=name)
        for name in (
            "maximum_retained_frame_metadata_count",
            "maximum_retained_frame_metadata_logical_bytes",
            "maximum_live_generated_observation_count",
            "maximum_live_generated_observation_logical_bytes",
            "maximum_request_track_count",
            "maximum_request_track_logical_bytes",
            "maximum_chunk_observation_count",
            "maximum_chunk_observation_logical_bytes",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if (
            self.maximum_live_generated_observation_count < 1
            or self.maximum_live_generated_observation_logical_bytes
            < OBSERVATION_IDENTITY_LOGICAL_BYTES
        ):
            raise ValueError("dense observation source cannot admit one generated identity")
        if (
            self.maximum_request_track_logical_bytes
            < TRACK_ID_LOGICAL_BYTES
        ):
            raise ValueError("dense observation request cannot admit one track")
        if (
            self.maximum_chunk_observation_logical_bytes
            < OBSERVATION_IDENTITY_LOGICAL_BYTES
        ):
            raise ValueError("dense observation chunk cannot admit one identity")


@dataclass(frozen=True)
class PaperKineticReplayableDenseObservationSource:
    """Compact, deterministic dense observation manifest and replay source."""

    provider: PaperKineticLazyProgramBundleProvider = field(repr=False)
    batch: SpacetimeBatch = field(repr=False)
    memory_policy: PaperKineticDenseObservationMemoryPolicy
    provider_generation_digest: str
    batch_content_digest: str
    image_pixel_count: int
    observation_count: int
    retained_frame_metadata_count: int
    retained_frame_metadata_logical_bytes: int
    compact_manifest_digest: str
    generation_digest: str
    _canonical_batch_positions: tuple[int, ...] = field(repr=False)
    _view_position_spans: tuple[tuple[int, int, int], ...] = field(repr=False)
    _provider_identity: int = field(repr=False)
    _batch_identity: int = field(repr=False)
    _batch_samples_identity: int = field(repr=False)
    _canonical_positions_identity: int = field(repr=False)
    _view_spans_identity: int = field(repr=False)
    provenance: str = DENSE_OBSERVATION_SOURCE_PROVENANCE
    _seal: object = field(default=None, repr=False)

    @property
    def selected_frame_count(self) -> int:
        return len(self.batch.samples)

    @property
    def selected_view_count(self) -> int:
        return len(self._view_position_spans)

    @property
    def canonical_view_indices(self) -> tuple[int, ...]:
        """Selected views in the exact order required by replay coverage."""

        return tuple(span[0] for span in self._view_position_spans)

    @property
    def effective_chunk_observation_capacity(self) -> int:
        return min(
            self.memory_policy.maximum_chunk_observation_count,
            self.memory_policy.maximum_chunk_observation_logical_bytes
            // OBSERVATION_IDENTITY_LOGICAL_BYTES,
        )

    def assert_warm_current(self) -> None:
        """Constant-size identity/generation check after one cold seal."""

        if (
            self._seal is not _SOURCE_SEAL
            or self.provenance != DENSE_OBSERVATION_SOURCE_PROVENANCE
            or id(self.provider) != self._provider_identity
            or id(self.batch) != self._batch_identity
            or id(self.batch.samples) != self._batch_samples_identity
            or id(self._canonical_batch_positions)
            != self._canonical_positions_identity
            or id(self._view_position_spans) != self._view_spans_identity
            or self.provider.generation_digest != self.provider_generation_digest
            or self.image_pixel_count != self.provider.height * self.provider.width
            or self.observation_count
            != self.selected_frame_count * self.image_pixel_count
            or self.retained_frame_metadata_count
            != self.selected_frame_count + self.selected_view_count
            or self.retained_frame_metadata_logical_bytes
            != 3 * 8 * self.retained_frame_metadata_count
            or self.compact_manifest_digest != _source_manifest_digest(self)
            or self.generation_digest != _source_digest(self)
        ):
            raise ValueError("replayable dense observation source metadata changed")
        self.memory_policy.assert_valid()
        self.provider.assert_warm_current()
        _enforce_source_memory_policy(
            self.memory_policy,
            selected_frame_count=self.selected_frame_count,
            selected_view_count=self.selected_view_count,
        )

    def assert_current(self) -> None:
        """Cold full provider/batch certification before opening a session."""

        self.assert_warm_current()
        self.provider.assert_current()
        if _batch_content_digest(self.batch) != self.batch_content_digest:
            raise ValueError("replayable dense observation batch changed")
        expected_positions = tuple(
            sorted(
                range(self.selected_frame_count),
                key=lambda position: (
                    self.batch.samples[position].view_index,
                    self.batch.samples[position].frame_index,
                    position,
                ),
            )
        )
        if expected_positions != self._canonical_batch_positions:
            raise ValueError("replayable dense observation canonical order changed")
        if _view_position_spans(self.batch, expected_positions) != self._view_position_spans:
            raise ValueError("replayable dense observation view spans changed")

    def prepare_track_request(
        self,
        *,
        view_index: int,
        track_ids: Sequence[int],
    ) -> PaperKineticDenseObservationTrackRequest:
        """Seal a frame-axis-free bundle request after fail-before-copy checks."""

        self.assert_warm_current()
        _require_nonnegative_int(view_index, name="view_index")
        if not isinstance(track_ids, Sequence):
            raise TypeError("dense observation track_ids must be a bounded sequence")
        track_count = len(track_ids)
        track_bytes = track_count * TRACK_ID_LOGICAL_BYTES
        if (
            track_count < 1
            or track_count > self.memory_policy.maximum_request_track_count
            or track_bytes > self.memory_policy.maximum_request_track_logical_bytes
        ):
            raise MemoryError(
                "dense observation track request exceeds its count/byte budget before copy or replay"
            )
        # Conversion happens only after both bounds above have passed.
        selected = tuple(track_ids)
        for track_id in selected:
            _require_nonnegative_int(track_id, name="track_id")
        if any(track_id >= self.image_pixel_count for track_id in selected):
            raise IndexError("dense observation track leaves the stage image")
        if any(
            right != left + 1
            for left, right in zip(selected, selected[1:], strict=False)
        ):
            raise ValueError("dense observation bundle tracks must be contiguous and increasing")
        if _source_view_span(self, view_index) is None:
            raise ValueError("dense observation request names a view absent from the batch")
        provisional = PaperKineticDenseObservationTrackRequest(
            source_generation_digest=self.generation_digest,
            view_index=view_index,
            track_ids=selected,
            track_id_logical_bytes=track_bytes,
            generation_digest="",
            _source_identity=id(self),
            _seal=_REQUEST_SEAL,
        )
        result = replace(
            provisional,
            generation_digest=_request_digest(provisional),
        )
        result.assert_current(self)
        return result

    def open_session(self) -> PaperKineticDenseObservationReplaySession:
        """Cold-certify once, then replay requests through warm checks."""

        self.assert_current()
        result = PaperKineticDenseObservationReplaySession(
            source=self,
            source_generation_digest=self.generation_digest,
            compact_manifest_digest=self.compact_manifest_digest,
            _source_identity=id(self),
            _seal=_SESSION_SEAL,
        )
        result.assert_current()
        return result

    def accounting(self) -> dict[str, int | bool | str]:
        self.assert_warm_current()
        return {
            "provenance": self.provenance,
            "provider_generation_digest": self.provider_generation_digest,
            "compact_manifest_digest": self.compact_manifest_digest,
            "compact_manifest_digest_scheme": (
                "deterministic-dense-product-descriptor-v1"
            ),
            "coordinator_ordered_identity_digest_compatible": False,
            "coordinator_dense_replay_integration_implemented": False,
            "selected_frame_count": self.selected_frame_count,
            "selected_view_count": self.selected_view_count,
            "image_pixel_count": self.image_pixel_count,
            "logical_observation_count": self.observation_count,
            "persistent_observation_count": 0,
            "persistent_observation_logical_bytes": 0,
            "retained_frame_metadata_count": self.retained_frame_metadata_count,
            "retained_frame_metadata_logical_bytes": (
                self.retained_frame_metadata_logical_bytes
            ),
            "maximum_live_generated_observation_count": 1,
            "maximum_live_generated_observation_logical_bytes": (
                OBSERVATION_IDENTITY_LOGICAL_BYTES
            ),
            "maximum_request_track_count": (
                self.memory_policy.maximum_request_track_count
            ),
            "maximum_request_track_logical_bytes": (
                self.memory_policy.maximum_request_track_logical_bytes
            ),
            "maximum_chunk_observation_count": (
                self.memory_policy.maximum_chunk_observation_count
            ),
            "maximum_chunk_observation_logical_bytes": (
                self.memory_policy.maximum_chunk_observation_logical_bytes
            ),
            "effective_chunk_observation_capacity": (
                self.effective_chunk_observation_capacity
            ),
            "dense_cartesian_observation_tuple_retained": False,
            "frame_axis_retained_by_track_request": False,
            "target_tensor_bytes": 0,
            "ray_tensor_bytes": 0,
            "device_tensor_bytes": 0,
            "target_source_access_count": 0,
            "canonical_identity_formula": (
                "observation_id=batch_position*image_pixel_count+pixel"
            ),
            "coverage_proof": (
                "contiguous view/track cursor + deterministic per-track frame replay + exact count"
            ),
            "total_observation_count_budgeted": False,
            "python_object_bytes_measured": False,
            "allocator_peak_measured": False,
        }


@dataclass(frozen=True)
class PaperKineticDenseObservationTrackRequest:
    """Bounded spatial selector containing no observations or frame axis."""

    source_generation_digest: str
    view_index: int
    track_ids: tuple[int, ...]
    track_id_logical_bytes: int
    generation_digest: str
    _source_identity: int = field(repr=False)
    provenance: str = DENSE_OBSERVATION_REQUEST_PROVENANCE
    _seal: object = field(default=None, repr=False)

    def assert_current(
        self,
        source: PaperKineticReplayableDenseObservationSource,
    ) -> None:
        source.assert_warm_current()
        if (
            self._seal is not _REQUEST_SEAL
            or self.provenance != DENSE_OBSERVATION_REQUEST_PROVENANCE
            or id(source) != self._source_identity
            or self.source_generation_digest != source.generation_digest
            or not self.track_ids
            or self.track_id_logical_bytes
            != len(self.track_ids) * TRACK_ID_LOGICAL_BYTES
            or len(self.track_ids)
            > source.memory_policy.maximum_request_track_count
            or self.track_id_logical_bytes
            > source.memory_policy.maximum_request_track_logical_bytes
            or any(
                track_id < 0 or track_id >= source.image_pixel_count
                for track_id in self.track_ids
            )
            or any(
                right != left + 1
                for left, right in zip(
                    self.track_ids,
                    self.track_ids[1:],
                    strict=False,
                )
            )
            or _source_view_span(source, self.view_index) is None
            or self.generation_digest != _request_digest(self)
        ):
            raise ValueError("dense observation track request changed or is foreign")


@dataclass(frozen=True)
class PaperKineticDenseObservationChunk:
    """One bounded identity payload; consumers must release before resuming."""

    source_generation_digest: str
    session_identity: int
    request_generation_digest: str
    chunk_index: int
    global_observation_offset: int
    observations: tuple[PaperKineticObservation, ...]
    logical_identity_bytes: int
    identities_digest: str
    generation_digest: str
    provenance: str = DENSE_OBSERVATION_CHUNK_PROVENANCE
    _seal: object = field(default=None, repr=False)

    @property
    def observation_count(self) -> int:
        return len(self.observations)

    def assert_self_consistent(
        self,
        source: PaperKineticReplayableDenseObservationSource,
        request: PaperKineticDenseObservationTrackRequest,
    ) -> None:
        source.assert_warm_current()
        request.assert_current(source)
        if (
            self._seal is not _CHUNK_SEAL
            or self.provenance != DENSE_OBSERVATION_CHUNK_PROVENANCE
            or self.source_generation_digest != source.generation_digest
            or self.request_generation_digest != request.generation_digest
            or self.chunk_index < 0
            or self.global_observation_offset < 0
            or not self.observations
            or self.observation_count
            > source.memory_policy.maximum_chunk_observation_count
            or self.logical_identity_bytes
            != self.observation_count * OBSERVATION_IDENTITY_LOGICAL_BYTES
            or self.logical_identity_bytes
            > source.memory_policy.maximum_chunk_observation_logical_bytes
            or any(
                observation.view_index != request.view_index
                or observation.pixel_index < request.track_ids[0]
                or observation.pixel_index > request.track_ids[-1]
                for observation in self.observations
            )
            or self.identities_digest != _identities_digest(self.observations)
            or self.generation_digest != _chunk_digest(self)
        ):
            raise ValueError("dense observation replay chunk changed or is foreign")
        keys = tuple(_observation_key(value) for value in self.observations)
        if any(
            right <= left for left, right in zip(keys, keys[1:], strict=False)
        ):
            raise ValueError("dense observation replay chunk is not canonical")


@dataclass
class PaperKineticDenseObservationReplaySession:
    """O(1) global coverage cursor plus one bounded active request/chunk."""

    source: PaperKineticReplayableDenseObservationSource = field(repr=False)
    source_generation_digest: str
    compact_manifest_digest: str
    _source_identity: int = field(repr=False)
    emitted_observation_count: int = 0
    request_count: int = 0
    chunk_count: int = 0
    _coverage_view_span_index: int = 0
    _next_expected_pixel: int = 0
    _active_request: bool = False
    poisoned: bool = False
    sealed: bool = False
    provenance: str = DENSE_OBSERVATION_SOURCE_PROVENANCE
    _seal: object = field(default=None, repr=False)

    def assert_current(self) -> None:
        self.source.assert_warm_current()
        if (
            self._seal is not _SESSION_SEAL
            or self.provenance != DENSE_OBSERVATION_SOURCE_PROVENANCE
            or id(self.source) != self._source_identity
            or self.source_generation_digest != self.source.generation_digest
            or self.compact_manifest_digest
            != self.source.compact_manifest_digest
            or self.emitted_observation_count < 0
            or self.request_count < 0
            or self.chunk_count < 0
            or self._coverage_view_span_index < 0
            or self._coverage_view_span_index
            > len(self.source._view_position_spans)
            or self._next_expected_pixel < 0
            or self._next_expected_pixel >= self.source.image_pixel_count
            and self._coverage_view_span_index
            < len(self.source._view_position_spans)
            or self.sealed
            and (
                self.poisoned
                or self._active_request
                or self.emitted_observation_count
                != self.source.observation_count
            )
        ):
            raise ValueError("dense observation replay session metadata changed")

    def iter_request_chunks(
        self,
        request: PaperKineticDenseObservationTrackRequest,
    ) -> Iterator[PaperKineticDenseObservationChunk]:
        """Replay one canonical bundle; premature close poisons the session."""

        self.assert_current()
        request.assert_current(self.source)
        if self.poisoned or self.sealed:
            raise ValueError("dense observation replay session is not open")
        if self._active_request:
            raise ValueError("dense observation replay session already has an active request")
        _assert_next_coverage_request(self, request)
        self._active_request = True
        completed = False
        replay = _iter_request_observations(self.source, request)
        try:
            chunk_values: list[PaperKineticObservation] = []
            seen_track_count = 0
            previous_track: int | None = None
            for observation in replay:
                _validate_replayed_observation(self.source, request, observation)
                if previous_track != observation.pixel_index:
                    seen_track_count += 1
                    previous_track = observation.pixel_index
                chunk_values.append(observation)
                if len(chunk_values) == self.source.effective_chunk_observation_capacity:
                    chunk = _seal_chunk(
                        self,
                        request,
                        tuple(chunk_values),
                    )
                    chunk_values.clear()
                    yield chunk
                    del chunk
            if chunk_values:
                chunk = _seal_chunk(self, request, tuple(chunk_values))
                chunk_values.clear()
                yield chunk
                del chunk
            if seen_track_count != len(request.track_ids):
                _fail_arithmetic(
                    "dense observation replay omitted a requested track"
                )
            self.request_count += 1
            self._next_expected_pixel = request.track_ids[-1] + 1
            if self._next_expected_pixel == self.source.image_pixel_count:
                self._coverage_view_span_index += 1
                self._next_expected_pixel = 0
            completed = True
        except BaseException:
            self.poisoned = True
            raise
        finally:
            close = getattr(replay, "close", None)
            if callable(close):
                close()
            self._active_request = False
            if not completed:
                self.poisoned = True

    def seal(self) -> PaperKineticDenseObservationReplayReceipt:
        """Prove exact dense coverage without retaining emitted identities."""

        self.assert_current()
        if self.poisoned or self.sealed or self._active_request:
            raise ValueError("dense observation replay session cannot be sealed")
        if (
            self._coverage_view_span_index
            != len(self.source._view_position_spans)
            or self._next_expected_pixel != 0
            or self.emitted_observation_count != self.source.observation_count
        ):
            self.poisoned = True
            raise ValueError("dense observation replay did not cover the exact manifest")
        self.sealed = True
        provisional = PaperKineticDenseObservationReplayReceipt(
            source_generation_digest=self.source.generation_digest,
            compact_manifest_digest=self.compact_manifest_digest,
            observation_count=self.emitted_observation_count,
            request_count=self.request_count,
            chunk_count=self.chunk_count,
            generation_digest="",
            _session_identity=id(self),
            _seal=_RECEIPT_SEAL,
        )
        result = replace(
            provisional,
            generation_digest=_receipt_digest(provisional),
        )
        result.assert_current(self)
        return result

    def accounting(self) -> dict[str, int | bool | str]:
        self.assert_current()
        return {
            "compact_manifest_digest": self.compact_manifest_digest,
            "expected_observation_count": self.source.observation_count,
            "emitted_observation_count": self.emitted_observation_count,
            "request_count": self.request_count,
            "chunk_count": self.chunk_count,
            "coverage_view_span_index": self._coverage_view_span_index,
            "next_expected_pixel": self._next_expected_pixel,
            "active_request": self._active_request,
            "poisoned": self.poisoned,
            "sealed": self.sealed,
            "session_retained_observation_count": 0,
            "session_retained_chunk_count": 0,
            "consumer_must_release_chunk_before_resume": True,
            "one_live_chunk_enforced_by_session": False,
            "exact_coverage_cursor_without_identity_set": True,
        }


@dataclass(frozen=True)
class PaperKineticDenseObservationReplayReceipt:
    source_generation_digest: str
    compact_manifest_digest: str
    observation_count: int
    request_count: int
    chunk_count: int
    generation_digest: str
    _session_identity: int = field(repr=False)
    provenance: str = DENSE_OBSERVATION_RECEIPT_PROVENANCE
    _seal: object = field(default=None, repr=False)

    def assert_current(
        self,
        session: PaperKineticDenseObservationReplaySession,
    ) -> None:
        session.assert_current()
        if (
            self._seal is not _RECEIPT_SEAL
            or self.provenance != DENSE_OBSERVATION_RECEIPT_PROVENANCE
            or id(session) != self._session_identity
            or not session.sealed
            or self.source_generation_digest != session.source.generation_digest
            or self.compact_manifest_digest
            != session.compact_manifest_digest
            or self.observation_count != session.emitted_observation_count
            or self.request_count != session.request_count
            or self.chunk_count != session.chunk_count
            or self.generation_digest != _receipt_digest(self)
        ):
            raise ValueError("dense observation replay receipt changed or is foreign")


def prepare_paper_kinetic_replayable_dense_observation_source(
    provider: PaperKineticLazyProgramBundleProvider,
    batch: SpacetimeBatch,
    *,
    memory_policy: PaperKineticDenseObservationMemoryPolicy,
) -> PaperKineticReplayableDenseObservationSource:
    """Preflight all metadata bounds, then build an ``O(F)`` dense manifest."""

    if not isinstance(provider, PaperKineticLazyProgramBundleProvider):
        raise TypeError("dense observation source requires a kinetic provider")
    if not isinstance(batch, SpacetimeBatch):
        raise TypeError("dense observation source requires a SpacetimeBatch")
    if not isinstance(memory_policy, PaperKineticDenseObservationMemoryPolicy):
        raise TypeError("dense observation source requires its memory policy")
    memory_policy.assert_valid()
    selected_frame_count = len(batch.samples)
    # Worst-case metadata bounds are checked before provider cold hashing,
    # canonical sorting, tuple construction, or observation replay.
    _enforce_source_memory_policy(
        memory_policy,
        selected_frame_count=selected_frame_count,
        selected_view_count=selected_frame_count,
    )
    provider.assert_current()
    for sample in batch.samples:
        _require_nonnegative_int(sample.view_index, name="sample.view_index")
        _require_nonnegative_int(sample.frame_index, name="sample.frame_index")
        if sample.view_index >= provider.view_count:
            raise IndexError("dense observation batch view leaves the provider")
        if sample.frame_index >= provider.frame_count:
            raise IndexError("dense observation batch frame leaves the provider")
    positions = tuple(
        sorted(
            range(selected_frame_count),
            key=lambda position: (
                batch.samples[position].view_index,
                batch.samples[position].frame_index,
                position,
            ),
        )
    )
    spans = _view_position_spans(batch, positions)
    _enforce_source_memory_policy(
        memory_policy,
        selected_frame_count=selected_frame_count,
        selected_view_count=len(spans),
    )
    batch_digest = _batch_content_digest(batch)
    image_pixel_count = provider.height * provider.width
    observation_count = selected_frame_count * image_pixel_count
    retained_count = selected_frame_count + len(spans)
    retained_bytes = retained_count * 3 * 8
    provisional = PaperKineticReplayableDenseObservationSource(
        provider=provider,
        batch=batch,
        memory_policy=memory_policy,
        provider_generation_digest=provider.generation_digest,
        batch_content_digest=batch_digest,
        image_pixel_count=image_pixel_count,
        observation_count=observation_count,
        retained_frame_metadata_count=retained_count,
        retained_frame_metadata_logical_bytes=retained_bytes,
        compact_manifest_digest="",
        generation_digest="",
        _canonical_batch_positions=positions,
        _view_position_spans=spans,
        _provider_identity=id(provider),
        _batch_identity=id(batch),
        _batch_samples_identity=id(batch.samples),
        _canonical_positions_identity=id(positions),
        _view_spans_identity=id(spans),
        _seal=_SOURCE_SEAL,
    )
    manifest_digest = _source_manifest_digest(provisional)
    with_manifest = replace(
        provisional,
        compact_manifest_digest=manifest_digest,
    )
    result = replace(
        with_manifest,
        generation_digest=_source_digest(with_manifest),
    )
    # The provider and batch were fully certified above; avoid an immediate
    # second O(VF) cold scan merely to validate freshly sealed scalar metadata.
    result.assert_warm_current()
    return result


def _enforce_source_memory_policy(
    policy: PaperKineticDenseObservationMemoryPolicy,
    *,
    selected_frame_count: int,
    selected_view_count: int,
) -> None:
    retained_count = selected_frame_count + selected_view_count
    retained_bytes = retained_count * 3 * 8
    checks = (
        (0, policy.maximum_persistent_observation_count),
        (0, policy.maximum_persistent_observation_logical_bytes),
        (retained_count, policy.maximum_retained_frame_metadata_count),
        (retained_bytes, policy.maximum_retained_frame_metadata_logical_bytes),
        (1, policy.maximum_live_generated_observation_count),
        (
            OBSERVATION_IDENTITY_LOGICAL_BYTES,
            policy.maximum_live_generated_observation_logical_bytes,
        ),
    )
    if any(actual > maximum for actual, maximum in checks):
        raise MemoryError(
            "dense observation source exceeds a count/byte budget before allocation or replay"
        )


def _assert_next_coverage_request(
    session: PaperKineticDenseObservationReplaySession,
    request: PaperKineticDenseObservationTrackRequest,
) -> None:
    if session._coverage_view_span_index >= len(session.source._view_position_spans):
        raise ValueError("dense observation replay received an extra spatial request")
    expected_view = session.source._view_position_spans[
        session._coverage_view_span_index
    ][0]
    if (
        request.view_index != expected_view
        or request.track_ids[0] != session._next_expected_pixel
    ):
        raise ValueError(
            "dense observation spatial requests must exactly advance the canonical coverage cursor"
        )


def _iter_request_observations(
    source: PaperKineticReplayableDenseObservationSource,
    request: PaperKineticDenseObservationTrackRequest,
) -> Iterator[PaperKineticObservation]:
    span = _source_view_span(source, request.view_index)
    if span is None:
        raise ValueError("dense observation request view disappeared")
    _view_index, start, stop = span
    for pixel_index in request.track_ids:
        for position_offset in range(start, stop):
            batch_position = source._canonical_batch_positions[position_offset]
            sample = source.batch.samples[batch_position]
            yield PaperKineticObservation(
                observation_id=(
                    batch_position * source.image_pixel_count + pixel_index
                ),
                view_index=request.view_index,
                frame_index=sample.frame_index,
                pixel_index=pixel_index,
            )


def _validate_replayed_observation(
    source: PaperKineticReplayableDenseObservationSource,
    request: PaperKineticDenseObservationTrackRequest,
    observation: PaperKineticObservation,
) -> None:
    if not isinstance(observation, PaperKineticObservation):
        raise TypeError("dense observation factory emitted the wrong type")
    batch_position, pixel_remainder = divmod(
        observation.observation_id,
        source.image_pixel_count,
    )
    if (
        batch_position < 0
        or batch_position >= source.selected_frame_count
        or pixel_remainder != observation.pixel_index
        or observation.view_index != request.view_index
        or observation.pixel_index < request.track_ids[0]
        or observation.pixel_index > request.track_ids[-1]
        or observation.frame_index >= source.provider.frame_count
    ):
        raise ValueError("dense observation factory emitted a foreign identity")
    sample = source.batch.samples[batch_position]
    if (
        sample.view_index != observation.view_index
        or sample.frame_index != observation.frame_index
    ):
        raise ValueError(
            "dense observation identity quotient does not name its batch view/frame"
        )


def _seal_chunk(
    session: PaperKineticDenseObservationReplaySession,
    request: PaperKineticDenseObservationTrackRequest,
    observations: tuple[PaperKineticObservation, ...],
) -> PaperKineticDenseObservationChunk:
    logical_bytes = len(observations) * OBSERVATION_IDENTITY_LOGICAL_BYTES
    provisional = PaperKineticDenseObservationChunk(
        source_generation_digest=session.source.generation_digest,
        session_identity=id(session),
        request_generation_digest=request.generation_digest,
        chunk_index=session.chunk_count,
        global_observation_offset=session.emitted_observation_count,
        observations=observations,
        logical_identity_bytes=logical_bytes,
        identities_digest=_identities_digest(observations),
        generation_digest="",
        _seal=_CHUNK_SEAL,
    )
    result = replace(provisional, generation_digest=_chunk_digest(provisional))
    result.assert_self_consistent(session.source, request)
    session.chunk_count += 1
    session.emitted_observation_count += len(observations)
    return result


def _view_position_spans(
    batch: SpacetimeBatch,
    canonical_positions: tuple[int, ...],
) -> tuple[tuple[int, int, int], ...]:
    spans: list[tuple[int, int, int]] = []
    start = 0
    while start < len(canonical_positions):
        view_index = batch.samples[canonical_positions[start]].view_index
        stop = start + 1
        while (
            stop < len(canonical_positions)
            and batch.samples[canonical_positions[stop]].view_index == view_index
        ):
            stop += 1
        spans.append((view_index, start, stop))
        start = stop
    return tuple(spans)


def _source_view_span(
    source: PaperKineticReplayableDenseObservationSource,
    view_index: int,
) -> tuple[int, int, int] | None:
    return next(
        (
            span
            for span in source._view_position_spans
            if span[0] == view_index
        ),
        None,
    )


def _batch_content_digest(batch: SpacetimeBatch) -> str:
    digest = hashlib.sha256()
    _update_digest_part(digest, "paper-kinetic-dense-batch-v1")
    _update_digest_part(digest, batch.epoch)
    _update_digest_part(digest, batch.batch_index)
    _update_digest_part(digest, batch.completes_epoch)
    _update_digest_part(digest, len(batch.samples))
    for position, sample in enumerate(batch.samples):
        _update_digest_part(
            digest,
            (position, sample.view_index, sample.frame_index),
        )
    return digest.hexdigest()


def _source_manifest_digest(
    source: PaperKineticReplayableDenseObservationSource,
) -> str:
    return _digest_parts(
        DENSE_OBSERVATION_SOURCE_PROVENANCE,
        "compact-deterministic-manifest",
        source.provider_generation_digest,
        source.batch_content_digest,
        source.image_pixel_count,
        source.observation_count,
        source.retained_frame_metadata_count,
        source.retained_frame_metadata_logical_bytes,
        "canonical=(view,pixel,frame,batch_position*P+pixel)",
    )


def _source_digest(source: PaperKineticReplayableDenseObservationSource) -> str:
    return _digest_parts(
        DENSE_OBSERVATION_SOURCE_PROVENANCE,
        source.provider_generation_digest,
        source.batch_content_digest,
        source.image_pixel_count,
        source.observation_count,
        source.compact_manifest_digest,
        tuple(source.memory_policy.__dict__.items()),
        source._provider_identity,
        source._batch_identity,
    )


def _request_digest(request: PaperKineticDenseObservationTrackRequest) -> str:
    return _digest_parts(
        DENSE_OBSERVATION_REQUEST_PROVENANCE,
        request.source_generation_digest,
        request.view_index,
        request.track_ids,
        request.track_id_logical_bytes,
        request._source_identity,
    )


def _chunk_digest(chunk: PaperKineticDenseObservationChunk) -> str:
    return _digest_parts(
        DENSE_OBSERVATION_CHUNK_PROVENANCE,
        chunk.source_generation_digest,
        chunk.session_identity,
        chunk.request_generation_digest,
        chunk.chunk_index,
        chunk.global_observation_offset,
        chunk.logical_identity_bytes,
        chunk.identities_digest,
    )


def _receipt_digest(receipt: PaperKineticDenseObservationReplayReceipt) -> str:
    return _digest_parts(
        DENSE_OBSERVATION_RECEIPT_PROVENANCE,
        receipt.source_generation_digest,
        receipt.compact_manifest_digest,
        receipt.observation_count,
        receipt.request_count,
        receipt.chunk_count,
        receipt._session_identity,
    )


def _identities_digest(
    observations: tuple[PaperKineticObservation, ...],
) -> str:
    digest = hashlib.sha256()
    _update_digest_part(digest, DENSE_OBSERVATION_CHUNK_PROVENANCE)
    _update_digest_part(digest, len(observations))
    for observation in observations:
        _update_digest_part(digest, observation.sample_identity)
    return digest.hexdigest()


def _observation_key(
    observation: PaperKineticObservation,
) -> tuple[int, int, int, int]:
    return (
        observation.view_index,
        observation.pixel_index,
        observation.frame_index,
        observation.observation_id,
    )


def _digest_parts(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        _update_digest_part(digest, part)
    return digest.hexdigest()


def _update_digest_part(digest: Any, part: object) -> None:
    encoded = repr(part).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
    digest.update(encoded)


def _require_nonnegative_int(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _fail_arithmetic(message: str) -> None:
    raise ArithmeticError(message)


__all__ = [
    "DENSE_OBSERVATION_CHUNK_PROVENANCE",
    "DENSE_OBSERVATION_RECEIPT_PROVENANCE",
    "DENSE_OBSERVATION_REQUEST_PROVENANCE",
    "DENSE_OBSERVATION_SOURCE_PROVENANCE",
    "FRAME_METADATA_UPPER_BOUND_LOGICAL_BYTES",
    "OBSERVATION_IDENTITY_LOGICAL_BYTES",
    "TRACK_ID_LOGICAL_BYTES",
    "PaperKineticDenseObservationChunk",
    "PaperKineticDenseObservationMemoryPolicy",
    "PaperKineticDenseObservationReplayReceipt",
    "PaperKineticDenseObservationReplaySession",
    "PaperKineticDenseObservationTrackRequest",
    "PaperKineticReplayableDenseObservationSource",
    "prepare_paper_kinetic_replayable_dense_observation_source",
]
