"""Build a bounded, identity-sealed pixel-time RGB8 target cache.

The production mapped target source consumes one payload per camera in exact
``[height,width,stored_frame,RGB]`` byte order.  This converter deliberately
does not retain a decoded video or mmap an output payload.  It:

1. hashes each already-open raw input;
2. streams one exact RGB8 frame at a time into a bounded frame-major spool;
3. transposes bounded spatial tiles into the pixel-time payload;
4. decodes that completed payload back through a separate bounded spool; and
5. independently hashes raw-decoded and cache-decoded float32 values in
   canonical ``[frame,channel,height,width]`` order.

The raw decoder receives the same open file handle that is hashed before and
after decoding.  The checked-in CLI supports a strict frame-major RGB8 raw
fixture/backend.  The separate Neural3D adapter supplies a bounded PyAV
file-object decoder without weakening this file-identity boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Protocol

from worldfoam_target_dataset_binding import (
    BINDING_SCHEMA,
    MAPPED_RGB8_LAYOUT,
    MAPPED_RGB8_SCHEMA,
    TARGET_SPLITS,
    canonical_payload_sha256,
    validate_target_dataset_binding,
)


CONVERTER_PROVENANCE = "build_worldfoam_mapped_rgb8_cache/v1"
BUILD_PLAN_SCHEMA = "dynaworld.worldfoam_mapped_rgb8_build_plan/v1"
RAW_FRAME_MAJOR_RGB8_LAYOUT = "frame_height_width_rgb_interleaved"

_MAXIMUM_CONVERTER_SOURCE_BYTES = 16 * 1024 * 1024
_HASH_CHUNK_BYTES = 1024 * 1024
_PLAN_KEYS = {
    "schema",
    "dataset_id",
    "target_split",
    "raw_dataset_manifest",
    "height",
    "width",
    "stored_frame_indices",
    "required_frame_counts",
    "logical_frame_maps",
    "views",
    "camera",
    "limits",
    "mapped_manifest_label",
    "binding_label",
}
_PLAN_FILE_KEYS = {"path", "path_label"}
_PLAN_VIEW_KEYS = {
    "view_id",
    "raw_input_path",
    "raw_input_path_label",
    "payload_label",
    "raw_layout",
    "source_frame_count",
}
_PLAN_FRAME_MAP_KEYS = {"frame_count", "source_frame_indices"}
_LIMIT_KEYS = {
    "maximum_raw_dataset_manifest_bytes",
    "maximum_raw_input_bytes_per_view",
    "maximum_total_raw_input_verification_bytes",
    "maximum_total_decode_input_bytes",
    "maximum_decoded_frame_bytes",
    "maximum_decode_hash_scratch_bytes",
    "maximum_payload_bytes_per_view",
    "maximum_total_payload_bytes",
    "maximum_transpose_scratch_bytes",
    "maximum_temporary_bytes_per_view",
    "maximum_total_output_and_temporary_bytes",
    "maximum_total_cache_verification_bytes",
    "maximum_mapped_manifest_bytes",
    "maximum_binding_bytes",
}


class OpenRawRgb8FrameDecoder(Protocol):
    """Decode one frame from the caller-owned, already-open raw file."""

    provenance: str
    uses_supplied_handle_exclusively: bool
    reads_only_through_bounded_handle_api: bool

    def decode_rgb8_frame(
        self,
        handle: BinaryIO,
        *,
        view_id: str,
        source_frame_index: int,
        height: int,
        width: int,
        maximum_decoded_frame_bytes: int,
    ) -> bytes: ...

    def close_open_file_decode(
        self,
        handle: BinaryIO,
        *,
        view_id: str,
        decode_completed: bool,
    ) -> None: ...


@dataclass(frozen=True)
class FrameMajorRgb8OpenFileDecoder:
    """Strict raw ``[source_frame,height,width,RGB]`` decoder for fixtures.

    This backend is intentionally simple and exact.  It provides a runnable
    offline converter without claiming compatibility with compressed public
    videos.  A dataset adapter can supply another ``OpenRawRgb8FrameDecoder``
    while preserving the converter's open-handle and byte-budget contracts.
    """

    source_frame_count: int
    provenance: str = "frame-major-rgb8-open-file-decoder/v1"
    uses_supplied_handle_exclusively: bool = True
    reads_only_through_bounded_handle_api: bool = True

    def decode_rgb8_frame(
        self,
        handle: BinaryIO,
        *,
        view_id: str,
        source_frame_index: int,
        height: int,
        width: int,
        maximum_decoded_frame_bytes: int,
    ) -> bytes:
        frame_bytes = _checked_product(height, width, 3, name="decoded frame bytes")
        if frame_bytes > maximum_decoded_frame_bytes:
            raise MemoryError("decoded RGB8 frame exceeds its byte cap before read")
        if (
            isinstance(self.source_frame_count, bool)
            or not isinstance(self.source_frame_count, int)
            or self.source_frame_count < 1
        ):
            raise ValueError("frame-major raw source_frame_count must be positive")
        if source_frame_index < 0 or source_frame_index >= self.source_frame_count:
            raise IndexError(
                f"raw frame {source_frame_index} for view {view_id!r} leaves its source"
            )
        expected_file_bytes = self.source_frame_count * frame_bytes
        if _open_file_size(handle) != expected_file_bytes:
            raise ValueError(
                f"frame-major raw input for view {view_id!r} has the wrong byte size"
            )
        handle.seek(source_frame_index * frame_bytes)
        decoded = handle.read(frame_bytes)
        if len(decoded) != frame_bytes:
            raise ValueError(f"raw decoder returned a short frame for view {view_id!r}")
        return decoded

    def close_open_file_decode(
        self,
        handle: BinaryIO,
        *,
        view_id: str,
        decode_completed: bool,
    ) -> None:
        del handle, view_id, decode_completed


@dataclass(frozen=True)
class WorldFoamMappedRgb8ConversionLimits:
    maximum_raw_dataset_manifest_bytes: int
    maximum_raw_input_bytes_per_view: int
    maximum_total_raw_input_verification_bytes: int
    maximum_total_decode_input_bytes: int
    maximum_decoded_frame_bytes: int
    maximum_decode_hash_scratch_bytes: int
    maximum_payload_bytes_per_view: int
    maximum_total_payload_bytes: int
    maximum_transpose_scratch_bytes: int
    maximum_temporary_bytes_per_view: int
    maximum_total_output_and_temporary_bytes: int
    maximum_total_cache_verification_bytes: int
    maximum_mapped_manifest_bytes: int
    maximum_binding_bytes: int

    def assert_valid(self) -> None:
        for name, value in self.__dict__.items():
            _positive_int(value, name=name)


@dataclass(frozen=True)
class WorldFoamRawTargetView:
    view_id: str
    raw_input_path: Path
    raw_input_path_label: str
    payload_label: str
    decoder: OpenRawRgb8FrameDecoder


@dataclass(frozen=True)
class WorldFoamMappedRgb8BuildRequest:
    output_directory: Path
    dataset_id: str
    target_split: str
    raw_dataset_manifest_path: Path
    raw_dataset_manifest_path_label: str
    views: tuple[WorldFoamRawTargetView, ...]
    height: int
    width: int
    stored_frame_indices: tuple[int, ...]
    required_frame_counts: tuple[int, ...]
    logical_frame_maps: tuple[tuple[int, tuple[int, ...]], ...]
    camera: Mapping[str, Any]
    limits: WorldFoamMappedRgb8ConversionLimits
    mapped_manifest_label: str = "mapped_rgb8_manifest.json"
    binding_label: str = "target_dataset_binding.json"


@dataclass(frozen=True)
class WorldFoamMappedRgb8BuildReceipt:
    manifest_path: Path
    binding_path: Path
    binding_sha256: str
    payload_paths: tuple[Path, ...]
    payload_sha256s: tuple[str, ...]
    raw_decoded_f32_sha256s: tuple[str, ...]
    cache_decoded_f32_sha256s: tuple[str, ...]
    exact_payload_bytes_per_view: int
    exact_total_payload_bytes: int
    total_decode_input_bytes: int
    raw_cache_decoded_f32_equality_recomputed: bool
    raw_files_hashed_and_decoded_through_same_open_handles: bool
    cache_payloads_hashed_and_verified_through_same_open_handles: bool


@dataclass(frozen=True)
class _BuiltView:
    view_id: str
    raw_input_identity: dict[str, Any]
    raw_decoded_f32_sha256: str
    cache_decoded_f32_sha256: str
    payload_identity: dict[str, Any]
    payload_temp_path: Path
    payload_final_path: Path
    decode_input_bytes: int


def _checked_product(*values: int, name: str) -> int:
    result = 1
    for value in values:
        _positive_int(value, name=name)
        result *= value
    return result


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _nonempty_trimmed(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a nonempty trimmed string")
    return value


def _target_split(value: Any) -> str:
    split = _nonempty_trimmed(value, name="target_split")
    if split not in TARGET_SPLITS:
        raise ValueError(
            f"target_split must be exactly one of {sorted(TARGET_SPLITS)}"
        )
    return split


def _portable_relative_label(value: Any, *, name: str) -> str:
    label = _nonempty_trimmed(value, name=name)
    path = PurePosixPath(label)
    if (
        "\\" in label
        or path.is_absolute()
        or path.as_posix() != label
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(ord(character) < 32 or ord(character) == 127 for character in label)
    ):
        raise ValueError(f"{name} must be a portable relative path")
    return label


def _strict_json_loads(raw: bytes, *, name: str) -> Any:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} repeats JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ValueError(f"{name} contains nonstandard JSON constant {value!r}")

    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not strict UTF-8 JSON") from error


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _file_stat_signature(stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


def _open_file_size(handle: BinaryIO) -> int:
    return int(os.fstat(handle.fileno()).st_size)


def _hash_open_file(
    handle: BinaryIO,
    *,
    maximum_bytes: int,
    expected_bytes: int | None = None,
) -> tuple[str, tuple[int, int, int, int, int]]:
    _positive_int(maximum_bytes, name="open-file hash byte cap")
    stat_before = _file_stat_signature(os.fstat(handle.fileno()))
    if stat_before[2] > maximum_bytes:
        raise MemoryError("opened file exceeds its hash-scan byte cap")
    if expected_bytes is not None and stat_before[2] != expected_bytes:
        raise ValueError("opened file size differs from its exact preflight")
    digest = hashlib.sha256()
    handle.seek(0)
    remaining = stat_before[2]
    while remaining:
        chunk = handle.read(min(_HASH_CHUNK_BYTES, remaining))
        if not chunk:
            raise ValueError("opened file ended during its bounded hash scan")
        digest.update(chunk)
        remaining -= len(chunk)
    stat_after = _file_stat_signature(os.fstat(handle.fileno()))
    if stat_before != stat_after:
        raise ValueError("opened file changed during its bounded hash scan")
    handle.seek(0)
    return digest.hexdigest(), stat_after


def _identity_from_open_file(
    handle: BinaryIO,
    *,
    path_label: str,
    maximum_bytes: int,
    expected_bytes: int | None = None,
) -> dict[str, Any]:
    digest, signature = _hash_open_file(
        handle,
        maximum_bytes=maximum_bytes,
        expected_bytes=expected_bytes,
    )
    if signature[2] < 1:
        raise ValueError("bound file identities cannot name empty files")
    return {
        "path_label": _portable_relative_label(path_label, name="file path_label"),
        "size_bytes": signature[2],
        "sha256": digest,
    }


def _flush_sync(handle: BinaryIO) -> None:
    handle.flush()
    os.fsync(handle.fileno())


class _BoundedDecodeFile:
    """Count and cap decoder reads without owning the underlying raw file."""

    def __init__(self, handle: BinaryIO, *, maximum_read_bytes: int) -> None:
        self._handle = handle
        self.maximum_read_bytes = _positive_int(
            maximum_read_bytes,
            name="decoder input byte cap",
        )
        self.read_bytes = 0

    def read(self, size: int = -1) -> bytes:
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("bounded raw decoder must request an explicit read size")
        if self.read_bytes + size > self.maximum_read_bytes:
            raise MemoryError("raw decoder input reads exceed their byte cap before read")
        value = self._handle.read(size)
        self.read_bytes += len(value)
        return value

    def readinto(self, buffer: Any) -> int:
        requested = memoryview(buffer).nbytes
        if self.read_bytes + requested > self.maximum_read_bytes:
            raise MemoryError("raw decoder input reads exceed their byte cap before read")
        count = int(self._handle.readinto(buffer))
        self.read_bytes += count
        return count

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        return int(self._handle.seek(offset, whence))

    def tell(self) -> int:
        return int(self._handle.tell())

    def fileno(self) -> int:
        return int(self._handle.fileno())

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    @property
    def closed(self) -> bool:
        return bool(self._handle.closed)

    @property
    def name(self) -> Any:
        return getattr(self._handle, "name", None)


def _update_decoded_f32_digest(
    digest: Any,
    frame_rgb8: bytes,
    *,
    height: int,
    width: int,
    maximum_scratch_bytes: int,
) -> None:
    """Hash one RGB8 frame as exact little-endian ``[channel,H,W]`` f32."""

    import numpy as np

    pixel_count = height * width
    frame_bytes = pixel_count * 3
    # The immutable RGB8 frame, one float32 channel conversion, and one
    # defensive contiguous float32 channel can overlap.
    scratch_upper_bound = frame_bytes + 2 * pixel_count * 4
    if scratch_upper_bound > maximum_scratch_bytes:
        raise MemoryError("decoded float32 hash scratch exceeds its cap before conversion")
    frame = np.frombuffer(frame_rgb8, dtype=np.uint8).reshape(height, width, 3)
    for channel_index in range(3):
        normalized = np.asarray(frame[:, :, channel_index], dtype=np.dtype("<f4"))
        normalized /= np.float32(255.0)
        contiguous = np.ascontiguousarray(normalized, dtype=np.dtype("<f4"))
        digest.update(memoryview(contiguous).cast("B"))
        del contiguous, normalized
    del frame


def _hash_frame_major_spool_decoded_f32(
    handle: BinaryIO,
    *,
    frame_count: int,
    height: int,
    width: int,
    maximum_scratch_bytes: int,
) -> str:
    frame_bytes = height * width * 3
    expected_bytes = frame_count * frame_bytes
    if _open_file_size(handle) != expected_bytes:
        raise ValueError("decoded verification spool has the wrong byte size")
    digest = hashlib.sha256()
    handle.seek(0)
    for _frame_index in range(frame_count):
        frame = handle.read(frame_bytes)
        if len(frame) != frame_bytes:
            raise ValueError("decoded verification spool ended within a frame")
        _update_decoded_f32_digest(
            digest,
            frame,
            height=height,
            width=width,
            maximum_scratch_bytes=maximum_scratch_bytes,
        )
    handle.seek(0)
    return digest.hexdigest()


def _transpose_frame_spool_to_pixel_time_payload(
    frame_spool: BinaryIO,
    payload: BinaryIO,
    *,
    frame_count: int,
    height: int,
    width: int,
    maximum_scratch_bytes: int,
) -> None:
    import numpy as np

    pixel_count = height * width
    frame_bytes = pixel_count * 3
    expected_bytes = frame_count * frame_bytes
    if _open_file_size(frame_spool) != expected_bytes:
        raise ValueError("frame-major spool changed before payload transpose")
    bytes_per_pixel_of_scratch = frame_count * 3 + 3
    pixels_per_tile = min(pixel_count, maximum_scratch_bytes // bytes_per_pixel_of_scratch)
    if pixels_per_tile < 1:
        raise MemoryError("payload transpose scratch cannot admit one pixel")
    payload.seek(0)
    for pixel_start in range(0, pixel_count, pixels_per_tile):
        tile_pixel_count = min(pixels_per_tile, pixel_count - pixel_start)
        scratch_bytes = tile_pixel_count * bytes_per_pixel_of_scratch
        if scratch_bytes > maximum_scratch_bytes:
            raise ArithmeticError("payload transpose exceeded its preflight")
        tile = np.empty((tile_pixel_count, frame_count, 3), dtype=np.uint8)
        for frame_index in range(frame_count):
            frame_spool.seek(frame_index * frame_bytes + pixel_start * 3)
            raw = frame_spool.read(tile_pixel_count * 3)
            if len(raw) != tile_pixel_count * 3:
                raise ValueError("frame-major spool ended during payload transpose")
            tile[:, frame_index, :] = np.frombuffer(raw, dtype=np.uint8).reshape(
                tile_pixel_count,
                3,
            )
        written = payload.write(memoryview(tile).cast("B"))
        if written != tile_pixel_count * frame_count * 3:
            raise OSError("pixel-time payload write was short")
        del tile
    if payload.tell() != expected_bytes:
        raise ArithmeticError("pixel-time payload transpose changed exact storage size")


def _transpose_pixel_time_payload_to_frame_spool(
    payload: BinaryIO,
    frame_spool: BinaryIO,
    *,
    frame_count: int,
    height: int,
    width: int,
    maximum_scratch_bytes: int,
) -> None:
    """Independently reconstruct frame-major RGB8 from the completed cache."""

    import numpy as np

    pixel_count = height * width
    frame_bytes = pixel_count * 3
    expected_bytes = frame_count * frame_bytes
    if _open_file_size(payload) != expected_bytes:
        raise ValueError("pixel-time payload changed before cache verification")
    bytes_per_pixel_of_scratch = frame_count * 3 + 3
    pixels_per_tile = min(pixel_count, maximum_scratch_bytes // bytes_per_pixel_of_scratch)
    if pixels_per_tile < 1:
        raise MemoryError("cache verification transpose cannot admit one pixel")
    frame_spool.seek(0)
    frame_spool.truncate(expected_bytes)
    for pixel_start in range(0, pixel_count, pixels_per_tile):
        tile_pixel_count = min(pixels_per_tile, pixel_count - pixel_start)
        payload.seek(pixel_start * frame_count * 3)
        raw = payload.read(tile_pixel_count * frame_count * 3)
        if len(raw) != tile_pixel_count * frame_count * 3:
            raise ValueError("pixel-time payload ended during cache verification")
        tile = np.frombuffer(raw, dtype=np.uint8).reshape(
            tile_pixel_count,
            frame_count,
            3,
        )
        for frame_index in range(frame_count):
            selected = np.ascontiguousarray(tile[:, frame_index, :])
            frame_spool.seek(frame_index * frame_bytes + pixel_start * 3)
            written = frame_spool.write(memoryview(selected).cast("B"))
            if written != tile_pixel_count * 3:
                raise OSError("cache verification spool write was short")
            del selected
        del tile
    _flush_sync(frame_spool)
    if _open_file_size(frame_spool) != expected_bytes:
        raise ArithmeticError("cache verification spool changed exact storage size")


def _temporary_path(output_directory: Path, label: str, role: str) -> Path:
    return output_directory / f".{Path(label).name}.{role}-{os.getpid()}"


def _safe_output_path(output_directory: Path, label: str, *, name: str) -> Path:
    canonical = _portable_relative_label(label, name=name)
    root = output_directory.resolve()
    path = (root / canonical).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} escaped the output directory") from error
    return path


def _preflight_request(
    request: WorldFoamMappedRgb8BuildRequest,
) -> tuple[int, int, dict[Path, int]]:
    request.limits.assert_valid()
    _nonempty_trimmed(request.dataset_id, name="dataset_id")
    _target_split(request.target_split)
    height = _positive_int(request.height, name="height")
    width = _positive_int(request.width, name="width")
    if not request.views:
        raise ValueError("WorldFoam target cache conversion requires at least one view")
    view_ids = tuple(view.view_id for view in request.views)
    if view_ids != tuple(sorted(set(view_ids))):
        raise ValueError("WorldFoam target cache view ids must be unique and sorted")
    stored_frames = tuple(request.stored_frame_indices)
    if (
        not stored_frames
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in stored_frames
        )
        or stored_frames != tuple(sorted(set(stored_frames)))
    ):
        raise ValueError("stored_frame_indices must be unique, increasing, and nonnegative")
    required_counts = tuple(request.required_frame_counts)
    if (
        not required_counts
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in required_counts
        )
        or required_counts != tuple(sorted(set(required_counts)))
        or required_counts[-1] != len(stored_frames)
    ):
        raise ValueError("required_frame_counts must end at the stored frame count")
    observed_map_counts = tuple(count for count, _indices in request.logical_frame_maps)
    if observed_map_counts != required_counts:
        raise ValueError("logical frame maps must follow required_frame_counts exactly")
    stored_set = set(stored_frames)
    for count, indices in request.logical_frame_maps:
        if (
            len(indices) != count
            or indices != tuple(sorted(set(indices)))
            or any(index not in stored_set for index in indices)
            or indices[0] != stored_frames[0]
            or indices[-1] != stored_frames[-1]
        ):
            raise ValueError("logical frame map changed count, order, membership, or endpoints")

    output_directory = request.output_directory.expanduser().resolve()
    manifest_path = _safe_output_path(
        output_directory,
        request.mapped_manifest_label,
        name="mapped manifest label",
    )
    binding_path = _safe_output_path(
        output_directory,
        request.binding_label,
        name="binding label",
    )
    payload_paths = tuple(
        _safe_output_path(output_directory, view.payload_label, name="payload label")
        for view in request.views
    )
    output_labels = (
        request.mapped_manifest_label,
        request.binding_label,
        *(view.payload_label for view in request.views),
    )
    if any(len(PurePosixPath(label).parts) != 1 for label in output_labels):
        raise ValueError(
            "converter v1 requires manifest, binding, and payloads in one output directory"
        )
    if len(set(payload_paths)) != len(payload_paths):
        raise ValueError("payload labels must resolve to distinct files")
    if (
        manifest_path == binding_path
        or manifest_path in payload_paths
        or binding_path in payload_paths
    ):
        raise ValueError("manifest, binding, and payload outputs must be distinct")
    for path in (manifest_path, binding_path, *payload_paths):
        if path.exists():
            raise FileExistsError(f"WorldFoam cache output already exists: {path}")

    raw_manifest = request.raw_dataset_manifest_path.expanduser().resolve()
    if not raw_manifest.is_file():
        raise FileNotFoundError(f"raw dataset manifest is not a file: {raw_manifest}")
    _portable_relative_label(
        request.raw_dataset_manifest_path_label,
        name="raw dataset manifest path label",
    )
    raw_sizes: dict[Path, int] = {raw_manifest: int(raw_manifest.stat().st_size)}
    if raw_sizes[raw_manifest] > request.limits.maximum_raw_dataset_manifest_bytes:
        raise MemoryError("raw dataset manifest exceeds its byte cap")
    for view in request.views:
        _nonempty_trimmed(view.view_id, name="view_id")
        _portable_relative_label(view.raw_input_path_label, name="raw input path label")
        _portable_relative_label(view.payload_label, name="payload label")
        if not callable(getattr(view.decoder, "decode_rgb8_frame", None)):
            raise TypeError(f"view {view.view_id!r} has no open-file RGB8 decoder")
        if not callable(getattr(view.decoder, "close_open_file_decode", None)):
            raise TypeError(f"view {view.view_id!r} has no decoder close boundary")
        _nonempty_trimmed(getattr(view.decoder, "provenance", ""), name="decoder provenance")
        if (
            getattr(view.decoder, "uses_supplied_handle_exclusively", None) is not True
            or (
                getattr(
                    view.decoder,
                    "reads_only_through_bounded_handle_api",
                    None,
                )
                is not True
            )
        ):
            raise ValueError(
                "raw decoder must exclusively use the supplied bounded open-file API"
            )
        raw_path = view.raw_input_path.expanduser().resolve()
        if not raw_path.is_file():
            raise FileNotFoundError(f"raw view input is not a file: {raw_path}")
        size = int(raw_path.stat().st_size)
        if size < 1 or size > request.limits.maximum_raw_input_bytes_per_view:
            raise MemoryError(f"raw input for view {view.view_id!r} exceeds its byte cap")
        raw_sizes[raw_path] = size

    frame_bytes = _checked_product(height, width, 3, name="frame bytes")
    payload_bytes = _checked_product(
        height,
        width,
        len(stored_frames),
        3,
        name="payload bytes",
    )
    total_payload_bytes = payload_bytes * len(request.views)
    decode_hash_scratch = frame_bytes + 2 * height * width * 4
    checks = (
        (frame_bytes, request.limits.maximum_decoded_frame_bytes, "decoded frame"),
        (
            decode_hash_scratch,
            request.limits.maximum_decode_hash_scratch_bytes,
            "decoded hash scratch",
        ),
        (payload_bytes, request.limits.maximum_payload_bytes_per_view, "per-view payload"),
        (total_payload_bytes, request.limits.maximum_total_payload_bytes, "total payload"),
        # One completed/current payload overlaps either its frame-major raw
        # spool or its independently reconstructed verification spool.  Those
        # are distinct files, so the per-view temporary-storage bound is two
        # exact payloads even though only one extra spool appears in the global
        # output-plus-temporary peak below.
        (
            2 * payload_bytes,
            request.limits.maximum_temporary_bytes_per_view,
            "per-view temporary storage",
        ),
        (
            total_payload_bytes
            + payload_bytes
            + request.limits.maximum_mapped_manifest_bytes
            + request.limits.maximum_binding_bytes,
            request.limits.maximum_total_output_and_temporary_bytes,
            "output plus temporary storage",
        ),
        (
            len(request.views) * len(stored_frames) * frame_bytes,
            request.limits.maximum_total_decode_input_bytes,
            "total selected decode input",
        ),
    )
    for actual, maximum, name in checks:
        if actual > maximum:
            raise MemoryError(f"{name} exceeds its explicit preflight cap")
    if request.limits.maximum_transpose_scratch_bytes < len(stored_frames) * 3 + 3:
        raise MemoryError("transpose scratch cannot admit one pixel")

    raw_verification_bytes = raw_sizes[raw_manifest] + 2 * sum(
        raw_sizes[view.raw_input_path.expanduser().resolve()] for view in request.views
    )
    if raw_verification_bytes > request.limits.maximum_total_raw_input_verification_bytes:
        raise MemoryError("raw-input verification scans exceed their total byte cap")
    # Per payload: one identity hash read, one cache-layout read, one verification
    # spool write, and one verification spool read.
    if (
        4 * total_payload_bytes + request.limits.maximum_mapped_manifest_bytes
        > request.limits.maximum_total_cache_verification_bytes
    ):
        raise MemoryError("cache verification I/O exceeds its total byte cap")
    return payload_bytes, total_payload_bytes, raw_sizes


def _build_one_view(
    request: WorldFoamMappedRgb8BuildRequest,
    view: WorldFoamRawTargetView,
    *,
    payload_bytes: int,
    raw_input_bytes: int,
    maximum_decode_input_bytes: int,
) -> _BuiltView:
    output_directory = request.output_directory.expanduser().resolve()
    payload_final_path = _safe_output_path(
        output_directory,
        view.payload_label,
        name="payload label",
    )
    payload_temp_path = _temporary_path(output_directory, view.payload_label, "payload")
    raw_spool_path = _temporary_path(output_directory, view.payload_label, "raw-spool")
    cache_spool_path = _temporary_path(output_directory, view.payload_label, "cache-spool")
    for path in (payload_temp_path, raw_spool_path, cache_spool_path):
        if path.exists():
            raise FileExistsError(f"converter temporary file already exists: {path}")

    raw_digest = hashlib.sha256()
    raw_identity: dict[str, Any] | None = None
    payload_identity: dict[str, Any] | None = None
    cache_digest = ""
    try:
        with view.raw_input_path.expanduser().resolve().open("rb") as raw_handle:
            raw_identity = _identity_from_open_file(
                raw_handle,
                path_label=view.raw_input_path_label,
                maximum_bytes=request.limits.maximum_raw_input_bytes_per_view,
                expected_bytes=raw_input_bytes,
            )
            initial_signature = _file_stat_signature(os.fstat(raw_handle.fileno()))
            decode_handle = _BoundedDecodeFile(
                raw_handle,
                maximum_read_bytes=maximum_decode_input_bytes,
            )
            with raw_spool_path.open("xb+") as raw_spool:
                decode_completed = False
                try:
                    for source_frame_index in request.stored_frame_indices:
                        decoded = view.decoder.decode_rgb8_frame(
                            decode_handle,
                            view_id=view.view_id,
                            source_frame_index=source_frame_index,
                            height=request.height,
                            width=request.width,
                            maximum_decoded_frame_bytes=(
                                request.limits.maximum_decoded_frame_bytes
                            ),
                        )
                        if not isinstance(decoded, bytes):
                            raise TypeError(
                                "raw decoder must return an immutable bytes frame"
                            )
                        expected_frame_bytes = request.height * request.width * 3
                        if len(decoded) != expected_frame_bytes:
                            raise ValueError(
                                "raw decoder changed exact RGB8 frame size for "
                                f"view {view.view_id!r}"
                            )
                        _update_decoded_f32_digest(
                            raw_digest,
                            decoded,
                            height=request.height,
                            width=request.width,
                            maximum_scratch_bytes=(
                                request.limits.maximum_decode_hash_scratch_bytes
                            ),
                        )
                        if raw_spool.write(decoded) != expected_frame_bytes:
                            raise OSError("raw frame spool write was short")
                        del decoded
                    decode_completed = True
                finally:
                    view.decoder.close_open_file_decode(
                        decode_handle,
                        view_id=view.view_id,
                        decode_completed=decode_completed,
                    )
                _flush_sync(raw_spool)
                if _open_file_size(raw_spool) != payload_bytes:
                    raise ArithmeticError("raw frame spool changed exact payload size")
                with payload_temp_path.open("xb+") as payload_handle:
                    _transpose_frame_spool_to_pixel_time_payload(
                        raw_spool,
                        payload_handle,
                        frame_count=len(request.stored_frame_indices),
                        height=request.height,
                        width=request.width,
                        maximum_scratch_bytes=(
                            request.limits.maximum_transpose_scratch_bytes
                        ),
                    )
                    _flush_sync(payload_handle)
                    payload_identity = _identity_from_open_file(
                        payload_handle,
                        path_label=view.payload_label,
                        maximum_bytes=request.limits.maximum_payload_bytes_per_view,
                        expected_bytes=payload_bytes,
                    )
                    payload_identity_signature = _file_stat_signature(
                        os.fstat(payload_handle.fileno())
                    )
                    # The cache-decoded identity is derived by reading the
                    # completed pixel-time payload, not by reusing raw frames or
                    # trusting the payload declaration.
                    raw_spool.close()
                    raw_spool_path.unlink()
                    with cache_spool_path.open("xb+") as cache_spool:
                        _transpose_pixel_time_payload_to_frame_spool(
                            payload_handle,
                            cache_spool,
                            frame_count=len(request.stored_frame_indices),
                            height=request.height,
                            width=request.width,
                            maximum_scratch_bytes=(
                                request.limits.maximum_transpose_scratch_bytes
                            ),
                        )
                        cache_digest = _hash_frame_major_spool_decoded_f32(
                            cache_spool,
                            frame_count=len(request.stored_frame_indices),
                            height=request.height,
                            width=request.width,
                            maximum_scratch_bytes=(
                                request.limits.maximum_decode_hash_scratch_bytes
                            ),
                        )
                    cache_spool_path.unlink()
                    # One final stat check keeps the payload identity bound to
                    # the exact same open handle used by cache verification.
                    if (
                        _open_file_size(payload_handle) != payload_bytes
                        or _file_stat_signature(os.fstat(payload_handle.fileno()))
                        != payload_identity_signature
                    ):
                        raise ValueError("payload changed during cache verification")

            repeated_sha256, final_signature = _hash_open_file(
                raw_handle,
                maximum_bytes=request.limits.maximum_raw_input_bytes_per_view,
                expected_bytes=raw_input_bytes,
            )
            if (
                repeated_sha256 != raw_identity["sha256"]
                or final_signature != initial_signature
            ):
                raise ValueError(
                    f"raw input for view {view.view_id!r} changed while decoding"
                )
        raw_decoded_digest = raw_digest.hexdigest()
        if raw_decoded_digest != cache_digest:
            raise ValueError(
                "independently decoded raw/cache float32 identities differ for "
                f"view {view.view_id!r}"
            )
        if raw_identity is None or payload_identity is None:
            raise ArithmeticError("view conversion produced no sealed file identities")
        return _BuiltView(
            view_id=view.view_id,
            raw_input_identity=raw_identity,
            raw_decoded_f32_sha256=raw_decoded_digest,
            cache_decoded_f32_sha256=cache_digest,
            payload_identity=payload_identity,
            payload_temp_path=payload_temp_path,
            payload_final_path=payload_final_path,
            decode_input_bytes=decode_handle.read_bytes,
        )
    except BaseException:
        for path in (payload_temp_path, raw_spool_path, cache_spool_path):
            path.unlink(missing_ok=True)
        raise


def _write_identity_bound_json_temp(
    path: Path,
    payload: Mapping[str, Any],
    *,
    path_label: str,
    maximum_bytes: int,
) -> dict[str, Any]:
    encoded = _canonical_json_bytes(payload)
    if not encoded or len(encoded) > maximum_bytes:
        raise MemoryError("JSON output exceeds its explicit byte cap before write")
    with path.open("xb+") as handle:
        if handle.write(encoded) != len(encoded):
            raise OSError("JSON output write was short")
        _flush_sync(handle)
        identity = _identity_from_open_file(
            handle,
            path_label=path_label,
            maximum_bytes=maximum_bytes,
            expected_bytes=len(encoded),
        )
    return identity


def _publish_no_replace(temp_path: Path, final_path: Path) -> None:
    """Publish one same-directory temporary without clobbering any file."""

    try:
        os.link(temp_path, final_path, follow_symlinks=False)
    except FileExistsError as error:
        raise FileExistsError(
            f"WorldFoam cache output appeared during conversion: {final_path}"
        ) from error
    try:
        temp_path.unlink()
    except BaseException:
        final_path.unlink(missing_ok=True)
        raise


def build_worldfoam_mapped_rgb8_cache(
    request: WorldFoamMappedRgb8BuildRequest,
) -> WorldFoamMappedRgb8BuildReceipt:
    """Build and atomically publish one strict mapped cache plus binding."""

    if not isinstance(request, WorldFoamMappedRgb8BuildRequest):
        raise TypeError("WorldFoam mapped RGB8 converter requires a build request")
    payload_bytes, total_payload_bytes, raw_sizes = _preflight_request(request)
    output_directory = request.output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_final_path = _safe_output_path(
        output_directory,
        request.mapped_manifest_label,
        name="mapped manifest label",
    )
    binding_final_path = _safe_output_path(
        output_directory,
        request.binding_label,
        name="binding label",
    )
    manifest_temp_path = _temporary_path(
        output_directory,
        request.mapped_manifest_label,
        "manifest",
    )
    binding_temp_path = _temporary_path(
        output_directory,
        request.binding_label,
        "binding",
    )
    for path in (manifest_temp_path, binding_temp_path):
        if path.exists():
            raise FileExistsError(f"converter temporary file already exists: {path}")

    built_views: list[_BuiltView] = []
    remaining_decode_input_bytes = request.limits.maximum_total_decode_input_bytes
    published_paths: list[Path] = []
    try:
        raw_manifest_path = request.raw_dataset_manifest_path.expanduser().resolve()
        with raw_manifest_path.open("rb") as handle:
            raw_dataset_manifest_identity = _identity_from_open_file(
                handle,
                path_label=request.raw_dataset_manifest_path_label,
                maximum_bytes=request.limits.maximum_raw_dataset_manifest_bytes,
                expected_bytes=raw_sizes[raw_manifest_path],
            )
        for view in request.views:
            minimum_view_decode_bytes = (
                len(request.stored_frame_indices) * request.height * request.width * 3
            )
            if remaining_decode_input_bytes < minimum_view_decode_bytes:
                raise MemoryError(
                    "remaining decoder input budget cannot admit the next view"
                )
            built_views.append(
                _build_one_view(
                    request,
                    view,
                    payload_bytes=payload_bytes,
                    raw_input_bytes=raw_sizes[view.raw_input_path.expanduser().resolve()],
                    maximum_decode_input_bytes=remaining_decode_input_bytes,
                )
            )
            remaining_decode_input_bytes -= built_views[-1].decode_input_bytes
            if remaining_decode_input_bytes < 0:
                raise ArithmeticError("decoder input accounting exceeded its preflight")

        manifest = {
            "schema": MAPPED_RGB8_SCHEMA,
            "layout": MAPPED_RGB8_LAYOUT,
            "dtype": "uint8",
            "height": request.height,
            "width": request.width,
            "stored_frame_indices": list(request.stored_frame_indices),
            "views": [
                {
                    "view_id": built.view_id,
                    "payload": built.payload_identity["path_label"],
                    "payload_bytes": built.payload_identity["size_bytes"],
                    "payload_sha256": built.payload_identity["sha256"],
                }
                for built in built_views
            ],
        }
        manifest_identity = _write_identity_bound_json_temp(
            manifest_temp_path,
            manifest,
            path_label=request.mapped_manifest_label,
            maximum_bytes=request.limits.maximum_mapped_manifest_bytes,
        )
        with Path(__file__).resolve().open("rb") as converter_source:
            converter_source_sha256, _signature = _hash_open_file(
                converter_source,
                maximum_bytes=_MAXIMUM_CONVERTER_SOURCE_BYTES,
            )

        frame_maps = [
            {
                "frame_count": frame_count,
                "source_frame_indices": list(indices),
                "logical_frame_map_sha256": canonical_payload_sha256(list(indices)),
            }
            for frame_count, indices in request.logical_frame_maps
        ]
        binding: dict[str, Any] = {
            "schema": BINDING_SCHEMA,
            "dataset_id": request.dataset_id,
            "target_split": request.target_split,
            "converter": {
                "provenance": CONVERTER_PROVENANCE,
                "source_sha256": converter_source_sha256,
            },
            "raw_dataset_manifest": raw_dataset_manifest_identity,
            "raw_views": [
                {
                    "view_id": built.view_id,
                    "raw_input": built.raw_input_identity,
                    "raw_decoded_f32_sha256": built.raw_decoded_f32_sha256,
                }
                for built in built_views
            ],
            "cache": {
                "manifest": manifest_identity,
                "format_schema": MAPPED_RGB8_SCHEMA,
                "layout": MAPPED_RGB8_LAYOUT,
                "dtype": "uint8",
                "height": request.height,
                "width": request.width,
                "stored_frame_indices": list(request.stored_frame_indices),
                "views": [
                    {
                        "view_id": built.view_id,
                        "payload": built.payload_identity,
                        "cache_decoded_f32_sha256": (
                            built.cache_decoded_f32_sha256
                        ),
                    }
                    for built in built_views
                ],
            },
            "decoded_f32_contract": {
                "dtype": "float32",
                "layout": "time_channel_height_width_contiguous_c_order",
                "byte_order": "little_endian_ieee754",
                "range": [0.0, 1.0],
                "conversion": "uint8_exact_divide_255_to_float32",
                "hash_payload": "raw_contiguous_tensor_bytes_without_metadata",
                "shape_per_view": [
                    len(request.stored_frame_indices),
                    3,
                    request.height,
                    request.width,
                ],
            },
            "camera": json.loads(_canonical_json_bytes(request.camera)),
            "logical_frame_maps": frame_maps,
            "binding_sha256": "",
        }
        binding["binding_sha256"] = canonical_payload_sha256(
            {key: value for key, value in binding.items() if key != "binding_sha256"}
        )
        validated = validate_target_dataset_binding(
            binding,
            required_frame_counts=request.required_frame_counts,
        )
        if validated != binding:
            raise ArithmeticError("binding validator changed the converter output")
        _write_identity_bound_json_temp(
            binding_temp_path,
            binding,
            path_label=request.binding_label,
            maximum_bytes=request.limits.maximum_binding_bytes,
        )

        # The binding is published last.  Its presence therefore certifies that
        # all payload and manifest renames completed successfully.
        for built in built_views:
            built.payload_final_path.parent.mkdir(parents=True, exist_ok=True)
            _publish_no_replace(built.payload_temp_path, built.payload_final_path)
            published_paths.append(built.payload_final_path)
        manifest_final_path.parent.mkdir(parents=True, exist_ok=True)
        _publish_no_replace(manifest_temp_path, manifest_final_path)
        published_paths.append(manifest_final_path)
        binding_final_path.parent.mkdir(parents=True, exist_ok=True)
        _publish_no_replace(binding_temp_path, binding_final_path)
        published_paths.append(binding_final_path)
        return WorldFoamMappedRgb8BuildReceipt(
            manifest_path=manifest_final_path,
            binding_path=binding_final_path,
            binding_sha256=binding["binding_sha256"],
            payload_paths=tuple(built.payload_final_path for built in built_views),
            payload_sha256s=tuple(
                built.payload_identity["sha256"] for built in built_views
            ),
            raw_decoded_f32_sha256s=tuple(
                built.raw_decoded_f32_sha256 for built in built_views
            ),
            cache_decoded_f32_sha256s=tuple(
                built.cache_decoded_f32_sha256 for built in built_views
            ),
            exact_payload_bytes_per_view=payload_bytes,
            exact_total_payload_bytes=total_payload_bytes,
            total_decode_input_bytes=sum(
                built.decode_input_bytes for built in built_views
            ),
            raw_cache_decoded_f32_equality_recomputed=True,
            raw_files_hashed_and_decoded_through_same_open_handles=True,
            cache_payloads_hashed_and_verified_through_same_open_handles=True,
        )
    except BaseException:
        for built in built_views:
            built.payload_temp_path.unlink(missing_ok=True)
        for path in (manifest_temp_path, binding_temp_path):
            path.unlink(missing_ok=True)
        # A failure while publishing cannot leave an apparently authoritative
        # binding or a partial cache set behind.
        for path in reversed(published_paths):
            path.unlink(missing_ok=True)
        raise


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} keys changed")


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _int_sequence(value: Any, *, name: str, positive: bool) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ValueError(f"{name} must be a nonempty sequence")
    return tuple(
        _positive_int(item, name=f"{name}[{index}]")
        if positive
        else _nonnegative_int(item, name=f"{name}[{index}]")
        for index, item in enumerate(value)
    )


def request_from_build_plan(
    payload: Mapping[str, Any],
    *,
    plan_directory: Path,
    output_directory: Path,
) -> WorldFoamMappedRgb8BuildRequest:
    """Resolve a strict JSON plan using the checked-in raw RGB8 backend."""

    plan = _mapping(payload, name="build plan")
    _require_exact_keys(plan, _PLAN_KEYS, name="build plan")
    if plan.get("schema") != BUILD_PLAN_SCHEMA:
        raise ValueError("WorldFoam mapped RGB8 build-plan schema is missing or stale")
    manifest = _mapping(plan.get("raw_dataset_manifest"), name="raw_dataset_manifest")
    _require_exact_keys(manifest, _PLAN_FILE_KEYS, name="raw_dataset_manifest")
    height = _positive_int(plan.get("height"), name="height")
    width = _positive_int(plan.get("width"), name="width")
    views_raw = plan.get("views")
    if not isinstance(views_raw, list) or not views_raw:
        raise ValueError("build plan views must be a nonempty list")
    views: list[WorldFoamRawTargetView] = []
    for index, value in enumerate(views_raw):
        view = _mapping(value, name=f"views[{index}]")
        _require_exact_keys(view, _PLAN_VIEW_KEYS, name=f"views[{index}]")
        if view.get("raw_layout") != RAW_FRAME_MAJOR_RGB8_LAYOUT:
            raise ValueError("checked-in CLI supports only strict frame-major RGB8 raw inputs")
        source_frame_count = _positive_int(
            view.get("source_frame_count"),
            name=f"views[{index}].source_frame_count",
        )
        views.append(
            WorldFoamRawTargetView(
                view_id=_nonempty_trimmed(view.get("view_id"), name="view_id"),
                raw_input_path=(
                    plan_directory
                    / _nonempty_trimmed(
                        view.get("raw_input_path"),
                        name=f"views[{index}].raw_input_path",
                    )
                ).resolve(),
                raw_input_path_label=_portable_relative_label(
                    view.get("raw_input_path_label"),
                    name="raw_input_path_label",
                ),
                payload_label=_portable_relative_label(
                    view.get("payload_label"),
                    name="payload_label",
                ),
                decoder=FrameMajorRgb8OpenFileDecoder(source_frame_count),
            )
        )
    frame_maps_raw = plan.get("logical_frame_maps")
    if not isinstance(frame_maps_raw, list) or not frame_maps_raw:
        raise ValueError("logical_frame_maps must be a nonempty list")
    frame_maps: list[tuple[int, tuple[int, ...]]] = []
    for index, value in enumerate(frame_maps_raw):
        frame_map = _mapping(value, name=f"logical_frame_maps[{index}]")
        _require_exact_keys(
            frame_map,
            _PLAN_FRAME_MAP_KEYS,
            name=f"logical_frame_maps[{index}]",
        )
        frame_maps.append(
            (
                _positive_int(frame_map.get("frame_count"), name="frame_count"),
                _int_sequence(
                    frame_map.get("source_frame_indices"),
                    name="source_frame_indices",
                    positive=False,
                ),
            )
        )
    limits_raw = _mapping(plan.get("limits"), name="limits")
    _require_exact_keys(limits_raw, _LIMIT_KEYS, name="limits")
    limits = WorldFoamMappedRgb8ConversionLimits(
        **{
            name: _positive_int(limits_raw.get(name), name=f"limits.{name}")
            for name in _LIMIT_KEYS
        }
    )
    return WorldFoamMappedRgb8BuildRequest(
        output_directory=output_directory,
        dataset_id=_nonempty_trimmed(plan.get("dataset_id"), name="dataset_id"),
        target_split=_target_split(plan.get("target_split")),
        raw_dataset_manifest_path=(
            plan_directory
            / _nonempty_trimmed(
                manifest.get("path"),
                name="raw_dataset_manifest.path",
            )
        ).resolve(),
        raw_dataset_manifest_path_label=_portable_relative_label(
            manifest.get("path_label"),
            name="raw_dataset_manifest.path_label",
        ),
        views=tuple(views),
        height=height,
        width=width,
        stored_frame_indices=_int_sequence(
            plan.get("stored_frame_indices"),
            name="stored_frame_indices",
            positive=False,
        ),
        required_frame_counts=_int_sequence(
            plan.get("required_frame_counts"),
            name="required_frame_counts",
            positive=True,
        ),
        logical_frame_maps=tuple(frame_maps),
        camera=_mapping(plan.get("camera"), name="camera"),
        limits=limits,
        mapped_manifest_label=_portable_relative_label(
            plan.get("mapped_manifest_label"),
            name="mapped_manifest_label",
        ),
        binding_label=_portable_relative_label(
            plan.get("binding_label"),
            name="binding_label",
        ),
    )


def load_bounded_build_plan(path: Path, *, maximum_plan_bytes: int) -> Mapping[str, Any]:
    _positive_int(maximum_plan_bytes, name="maximum_plan_bytes")
    with path.expanduser().resolve().open("rb") as handle:
        stat_before = _file_stat_signature(os.fstat(handle.fileno()))
        if stat_before[2] > maximum_plan_bytes:
            raise MemoryError("WorldFoam mapped RGB8 build plan exceeds its byte cap")
        raw = handle.read(maximum_plan_bytes + 1)
        stat_after = _file_stat_signature(os.fstat(handle.fileno()))
    if (
        stat_before != stat_after
        or len(raw) != stat_before[2]
        or len(raw) > maximum_plan_bytes
    ):
        raise ValueError("WorldFoam mapped RGB8 build plan changed during bounded read")
    return _mapping(_strict_json_loads(raw, name="build plan"), name="build plan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--maximum-plan-bytes", type=int, required=True)
    args = parser.parse_args()
    plan_path = args.plan.expanduser().resolve()
    plan = load_bounded_build_plan(
        plan_path,
        maximum_plan_bytes=args.maximum_plan_bytes,
    )
    request = request_from_build_plan(
        plan,
        plan_directory=plan_path.parent,
        output_directory=args.output_directory,
    )
    receipt = build_worldfoam_mapped_rgb8_cache(request)
    print(
        json.dumps(
            {
                "binding_path": str(receipt.binding_path),
                "binding_sha256": receipt.binding_sha256,
                "manifest_path": str(receipt.manifest_path),
                "payload_paths": [str(path) for path in receipt.payload_paths],
                "exact_total_payload_bytes": receipt.exact_total_payload_bytes,
                "raw_cache_decoded_f32_equality_recomputed": (
                    receipt.raw_cache_decoded_f32_equality_recomputed
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "BUILD_PLAN_SCHEMA",
    "CONVERTER_PROVENANCE",
    "FrameMajorRgb8OpenFileDecoder",
    "OpenRawRgb8FrameDecoder",
    "RAW_FRAME_MAJOR_RGB8_LAYOUT",
    "WorldFoamMappedRgb8BuildReceipt",
    "WorldFoamMappedRgb8BuildRequest",
    "WorldFoamMappedRgb8ConversionLimits",
    "WorldFoamRawTargetView",
    "build_worldfoam_mapped_rgb8_cache",
    "load_bounded_build_plan",
    "request_from_build_plan",
]
