"""Strict step-scoped CPU target-frame cache for lazy kinetic paper steps.

Spatial program bundles are processed one at a time, but several bundles can
request pixels from the same decoded ``(view, frame)``.  This cache makes that
reuse explicit without introducing an unbounded target-video fallback:

* it is bound to one lazy provider identity and generation;
* a caller must supply a strict logical resident-byte budget;
* one full frame is preflighted against that budget before target decode;
* a frame key is decoded at most once for the lifetime of the cache;
* cached CPU frames are exposed only through sealed, stale-checked gather
  handles rather than as mutable public tensors;
* ``close()`` drops every frame and attempted-key record while preserving
  scalar accounting.

The byte budget is exact for retained contiguous float32 frame tensors.  It is
not an allocator-peak or Python-heap measurement.  Failed decode/validation
poisons the cache, preventing a second decode attempt for the same key.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import torch
from paper_kinetic_lazy_program_bundles import (
    PaperKineticLazyProgramBundleProvider,
)

TARGET_FRAME_CACHE_PROVENANCE = "paper-kinetic-step-target-frame-cache-v1"
TARGET_FRAME_PROVENANCE = "paper-kinetic-cached-target-frame-v1"

_CACHE_SEAL = object()
_FRAME_SEAL = object()


@dataclass(frozen=True)
class PaperKineticCachedTargetFrame:
    """Read-only-by-contract handle for one sealed CPU RGB frame."""

    view_index: int
    frame_index: int
    height: int
    width: int
    logical_tensor_bytes: int
    provider_generation_digest: str
    cache_generation_digest: str
    generation_digest: str
    _rgb_chw_f32: torch.Tensor = field(repr=False)
    _tensor_signature: tuple[object, ...] = field(repr=False)
    _cache_identity: int = field(repr=False)
    provenance: str = TARGET_FRAME_PROVENANCE
    _seal: object = field(default=None, repr=False)

    @property
    def frame_key(self) -> tuple[int, int]:
        return (self.view_index, self.frame_index)

    def assert_current(
        self,
        cache: PaperKineticStepTargetFrameCache,
        provider: PaperKineticLazyProgramBundleProvider,
    ) -> None:
        cache.assert_open_current(provider)
        if (
            self._seal is not _FRAME_SEAL
            or self.provenance != TARGET_FRAME_PROVENANCE
            or id(cache) != self._cache_identity
            or self.provider_generation_digest != provider.generation_digest
            or self.cache_generation_digest != cache.generation_digest
            or cache._frames.get(self.frame_key) is not self
            or self.logical_tensor_bytes != cache.frame_logical_tensor_bytes
            or _tensor_signature(self._rgb_chw_f32) != self._tensor_signature
            or self.generation_digest != _frame_digest(self)
        ):
            raise ValueError("paper kinetic cached target frame is stale or foreign")
        _require_frame_tensor(
            self._rgb_chw_f32,
            height=self.height,
            width=self.width,
            logical_tensor_bytes=self.logical_tensor_bytes,
        )

    def gather_pixels(
        self,
        cache: PaperKineticStepTargetFrameCache,
        provider: PaperKineticLazyProgramBundleProvider,
        pixel_indices_i64: torch.Tensor,
    ) -> torch.Tensor:
        """Return a new compact ``[K,3]`` tensor without exposing the frame."""

        self.assert_current(cache, provider)
        if (
            not isinstance(pixel_indices_i64, torch.Tensor)
            or pixel_indices_i64.device.type != "cpu"
            or pixel_indices_i64.dtype != torch.int64
            or pixel_indices_i64.ndim != 1
            or not pixel_indices_i64.is_contiguous()
            or pixel_indices_i64.numel() < 1
        ):
            raise ValueError("cached target pixel indices must be nonempty contiguous CPU int64")
        if bool(((pixel_indices_i64 < 0) | (pixel_indices_i64 >= self.height * self.width)).any()):
            raise IndexError("cached target pixel leaves the frame")
        selected = self._rgb_chw_f32.reshape(3, -1).index_select(1, pixel_indices_i64).transpose(0, 1).contiguous()
        self.assert_current(cache, provider)
        if not bool(torch.isfinite(selected).all().item()):
            raise ValueError("paper kinetic cached selected targets are nonfinite")
        return selected


@dataclass
class PaperKineticStepTargetFrameCache:
    """Mutable single-step cache with a strict no-eviction byte budget."""

    provider: PaperKineticLazyProgramBundleProvider = field(repr=False)
    maximum_resident_bytes: int
    frame_logical_tensor_bytes: int
    provider_generation_digest: str
    generation_digest: str
    _provider_identity: int = field(repr=False)
    _target_provider_identity: int = field(repr=False)
    _target_source_identity: int = field(repr=False)
    _frames: dict[tuple[int, int], PaperKineticCachedTargetFrame] = field(
        default_factory=dict,
        repr=False,
    )
    _attempted_frame_keys: set[tuple[int, int]] = field(
        default_factory=set,
        repr=False,
    )
    request_count: int = 0
    hit_count: int = 0
    decode_attempt_count: int = 0
    decode_count: int = 0
    preflight_rejection_count: int = 0
    stale_frame_rejection_count: int = 0
    resident_frame_tensor_bytes: int = 0
    peak_resident_frame_tensor_bytes: int = 0
    closed: bool = False
    poisoned: bool = False
    close_count: int = 0
    provenance: str = TARGET_FRAME_CACHE_PROVENANCE
    _seal: object = field(default=None, repr=False)

    @property
    def cached_frame_count(self) -> int:
        return len(self._frames)

    def assert_open_current(
        self,
        provider: PaperKineticLazyProgramBundleProvider,
    ) -> None:
        self._assert_metadata_current(provider, allow_closed=True)
        if self.closed:
            raise ValueError("paper kinetic target frame cache is closed")
        if self.poisoned:
            raise ValueError("paper kinetic target frame cache is poisoned")

    def get_frame(
        self,
        provider: PaperKineticLazyProgramBundleProvider,
        *,
        view_index: int,
        frame_index: int,
    ) -> PaperKineticCachedTargetFrame:
        """Return one sealed frame, decoding only after strict preflight."""

        self.assert_open_current(provider)
        _require_nonnegative_int(view_index, name="view_index")
        _require_nonnegative_int(frame_index, name="frame_index")
        if view_index >= provider.view_count or frame_index >= provider.frame_count:
            raise IndexError("paper kinetic cached frame leaves the provider grid")
        key = (view_index, frame_index)
        self.request_count += 1
        cached = self._frames.get(key)
        if cached is not None:
            self.hit_count += 1
            try:
                cached.assert_current(self, provider)
            except BaseException:
                self.stale_frame_rejection_count += 1
                self.poisoned = True
                raise
            return cached
        if key in self._attempted_frame_keys:
            self.poisoned = True
            raise ValueError("paper kinetic target frame decode key was already attempted")
        next_resident_bytes = self.resident_frame_tensor_bytes + self.frame_logical_tensor_bytes
        if next_resident_bytes > self.maximum_resident_bytes:
            self.preflight_rejection_count += 1
            raise MemoryError("paper kinetic target frame cache budget would be exceeded before decode")

        self._attempted_frame_keys.add(key)
        self.decode_attempt_count += 1
        try:
            frame = _decode_and_seal_frame(
                self,
                provider,
                view_index=view_index,
                frame_index=frame_index,
            )
            self._frames[key] = frame
            self.decode_count += 1
            self.resident_frame_tensor_bytes = next_resident_bytes
            self.peak_resident_frame_tensor_bytes = max(
                self.peak_resident_frame_tensor_bytes,
                self.resident_frame_tensor_bytes,
            )
            frame.assert_current(self, provider)
        except BaseException:
            if self._frames.pop(key, None) is not None:
                self.decode_count -= 1
                self.resident_frame_tensor_bytes -= self.frame_logical_tensor_bytes
            self.poisoned = True
            raise
        return frame

    def accounting(self) -> dict[str, int | bool | str]:
        self._assert_metadata_current(self.provider, allow_closed=True)
        return {
            "provenance": self.provenance,
            "provider_generation_digest": self.provider.generation_digest,
            "maximum_resident_bytes": self.maximum_resident_bytes,
            "frame_logical_tensor_bytes": self.frame_logical_tensor_bytes,
            "decode_scratch_upper_bound_bytes": self.frame_logical_tensor_bytes,
            "decode_insert_transient_upper_bound_bytes": (2 * self.frame_logical_tensor_bytes),
            "resident_frame_tensor_bytes": self.resident_frame_tensor_bytes,
            "peak_resident_frame_tensor_bytes": (self.peak_resident_frame_tensor_bytes),
            "cached_frame_count": self.cached_frame_count,
            "request_count": self.request_count,
            "hit_count": self.hit_count,
            "decode_attempt_count": self.decode_attempt_count,
            "decode_count": self.decode_count,
            "preflight_rejection_count": self.preflight_rejection_count,
            "stale_frame_rejection_count": self.stale_frame_rejection_count,
            "closed": self.closed,
            "poisoned": self.poisoned,
            "close_count": self.close_count,
            "decode_each_unique_frame_at_most_once": True,
            "preflight_one_frame_before_decode": True,
            "eviction_enabled": False,
            "unbounded_fallback_enabled": False,
            "cpu_float32_frames_only": True,
            "raw_frame_tensor_publicly_exposed": False,
            "allocator_peak_measured": False,
            "python_object_bytes_measured": False,
        }

    def close(self) -> None:
        """Clear all resident frames and keys, even when one frame is stale."""

        if self.closed:
            self._assert_metadata_current(self.provider, allow_closed=True)
            return
        stale_error: BaseException | None = None
        try:
            self._assert_metadata_current(self.provider)
            for frame in tuple(self._frames.values()):
                frame.assert_current(self, self.provider)
        except BaseException as error:
            stale_error = error
        finally:
            self._frames.clear()
            self._attempted_frame_keys.clear()
            self.resident_frame_tensor_bytes = 0
            self.closed = True
            self.close_count += 1
        self._assert_metadata_current(self.provider, allow_closed=True)
        if stale_error is not None:
            raise ValueError("paper kinetic target frame cache closed after stale state") from stale_error

    def _assert_metadata_current(
        self,
        provider: PaperKineticLazyProgramBundleProvider,
        *,
        allow_closed: bool = False,
    ) -> None:
        if not isinstance(provider, PaperKineticLazyProgramBundleProvider):
            raise TypeError("target frame cache requires its kinetic provider")
        expected_resident_bytes = len(self._frames) * self.frame_logical_tensor_bytes
        failed_decode_count = self.decode_attempt_count - self.decode_count
        if (
            self._seal is not _CACHE_SEAL
            or self.provenance != TARGET_FRAME_CACHE_PROVENANCE
            or id(provider) != self._provider_identity
            or id(self.provider) != self._provider_identity
            or provider.generation_digest != self.provider_generation_digest
            or id(provider.target_provider) != self._target_provider_identity
            or id(provider.target_provider.source) != self._target_source_identity
            or self.maximum_resident_bytes < 1
            or self.frame_logical_tensor_bytes != provider.height * provider.width * 3 * 4
            or self.generation_digest != _cache_digest(self)
            or not self.closed
            and self.decode_count != len(self._frames)
            or not self.closed
            and len(self._attempted_frame_keys) != self.decode_attempt_count
            or failed_decode_count not in (0, 1)
            or self.poisoned != (failed_decode_count == 1 or self.stale_frame_rejection_count > 0)
            or self.resident_frame_tensor_bytes != expected_resident_bytes
            or self.resident_frame_tensor_bytes > self.maximum_resident_bytes
            or self.peak_resident_frame_tensor_bytes < self.resident_frame_tensor_bytes
            or self.peak_resident_frame_tensor_bytes > self.maximum_resident_bytes
            or self.request_count != self.hit_count + self.decode_attempt_count + self.preflight_rejection_count
            or self.hit_count < 0
            or self.preflight_rejection_count < 0
            or self.stale_frame_rejection_count < 0
            or self.close_count != int(self.closed)
            or self.closed
            and (self._frames or self._attempted_frame_keys)
            or self.closed
            and self.resident_frame_tensor_bytes != 0
            or self.closed
            and not allow_closed
        ):
            raise ValueError("paper kinetic target frame cache metadata is stale")


def prepare_paper_kinetic_step_target_frame_cache(
    provider: PaperKineticLazyProgramBundleProvider,
    *,
    maximum_resident_bytes: int,
) -> PaperKineticStepTargetFrameCache:
    """Warm-bind an empty cache after the caller's outer cold certification.

    The step coordinator must call ``provider.assert_current()`` once before
    this boundary.  Repeating that full ``O(VF)`` camera/source certification
    for a target-only cache would turn bundle streaming back into repeated
    frame-linear metadata work; the warm check still seals every relevant
    provider/source identity and generation.
    """

    if not isinstance(provider, PaperKineticLazyProgramBundleProvider):
        raise TypeError("target frame cache requires a kinetic provider")
    provider.assert_warm_current()
    _require_positive_int(maximum_resident_bytes, name="maximum_resident_bytes")
    frame_bytes = provider.height * provider.width * 3 * 4
    provisional = PaperKineticStepTargetFrameCache(
        provider=provider,
        maximum_resident_bytes=maximum_resident_bytes,
        frame_logical_tensor_bytes=frame_bytes,
        provider_generation_digest=provider.generation_digest,
        generation_digest="",
        _provider_identity=id(provider),
        _target_provider_identity=id(provider.target_provider),
        _target_source_identity=id(provider.target_provider.source),
        _seal=_CACHE_SEAL,
    )
    provisional.generation_digest = _cache_digest(provisional)
    provisional._assert_metadata_current(provider)
    return provisional


def _decode_and_seal_frame(
    cache: PaperKineticStepTargetFrameCache,
    provider: PaperKineticLazyProgramBundleProvider,
    *,
    view_index: int,
    frame_index: int,
) -> PaperKineticCachedTargetFrame:
    flat_sample = view_index * provider.frame_count + frame_index
    decoded = provider.target_provider.select(
        torch.tensor([flat_sample], dtype=torch.int64, device="cpu"),
        height=provider.height,
        width=provider.width,
        device=torch.device("cpu"),
    )
    if (
        tuple(decoded.shape) != (1, 3, provider.height, provider.width)
        or decoded.device.type != "cpu"
        or decoded.dtype != torch.float32
        or not decoded.is_contiguous()
        or decoded.requires_grad
    ):
        raise ValueError("paper kinetic target decoder changed cached frame layout")
    # Own the retained storage.  A disk-lazy source is allowed to reuse its
    # decoder scratch; caching a view of that scratch would make later decodes
    # silently mutate earlier frames and would invalidate resident-byte counts.
    frame_tensor = decoded[0].clone(memory_format=torch.contiguous_format)
    del decoded
    _require_frame_tensor(
        frame_tensor,
        height=provider.height,
        width=provider.width,
        logical_tensor_bytes=cache.frame_logical_tensor_bytes,
    )
    if not bool(torch.isfinite(frame_tensor).all().item()):
        raise ValueError("paper kinetic cached target frame is nonfinite")
    provisional = PaperKineticCachedTargetFrame(
        view_index=view_index,
        frame_index=frame_index,
        height=provider.height,
        width=provider.width,
        logical_tensor_bytes=cache.frame_logical_tensor_bytes,
        provider_generation_digest=provider.generation_digest,
        cache_generation_digest=cache.generation_digest,
        generation_digest="",
        _rgb_chw_f32=frame_tensor,
        _tensor_signature=_tensor_signature(frame_tensor),
        _cache_identity=id(cache),
        _seal=_FRAME_SEAL,
    )
    return PaperKineticCachedTargetFrame(
        **{
            **provisional.__dict__,
            "generation_digest": _frame_digest(provisional),
        }
    )


def _cache_digest(cache: PaperKineticStepTargetFrameCache) -> str:
    return _digest_parts(
        TARGET_FRAME_CACHE_PROVENANCE,
        cache.provider.dataset_generation_digest,
        cache.provider_generation_digest,
        cache.maximum_resident_bytes,
        cache.frame_logical_tensor_bytes,
        cache._provider_identity,
        cache._target_provider_identity,
        cache._target_source_identity,
    )


def _frame_digest(frame: PaperKineticCachedTargetFrame) -> str:
    return _digest_parts(
        TARGET_FRAME_PROVENANCE,
        frame.view_index,
        frame.frame_index,
        frame.height,
        frame.width,
        frame.logical_tensor_bytes,
        frame.provider_generation_digest,
        frame.cache_generation_digest,
        frame._tensor_signature,
        frame._cache_identity,
    )


def _tensor_signature(tensor: torch.Tensor) -> tuple[object, ...]:
    storage = tensor.untyped_storage()
    return (
        id(tensor),
        int(getattr(tensor, "_version", 0)),
        int(storage.data_ptr()),
        int(storage.nbytes()),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
        tensor.layout,
        tensor.requires_grad,
    )


def _require_frame_tensor(
    tensor: torch.Tensor,
    *,
    height: int,
    width: int,
    logical_tensor_bytes: int,
) -> None:
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.device.type != "cpu"
        or tensor.dtype != torch.float32
        or tensor.layout != torch.strided
        or tuple(tensor.shape) != (3, height, width)
        or not tensor.is_contiguous()
        or tensor.requires_grad
        or tensor.storage_offset() != 0
        or tensor.numel() * tensor.element_size() != logical_tensor_bytes
        or tensor.untyped_storage().nbytes() != logical_tensor_bytes
    ):
        raise ValueError("paper kinetic cached target frame tensor layout changed")


def _digest_parts(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = repr(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _require_nonnegative_int(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _require_positive_int(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


__all__ = [
    "TARGET_FRAME_CACHE_PROVENANCE",
    "TARGET_FRAME_PROVENANCE",
    "PaperKineticCachedTargetFrame",
    "PaperKineticStepTargetFrameCache",
    "prepare_paper_kinetic_step_target_frame_cache",
]
