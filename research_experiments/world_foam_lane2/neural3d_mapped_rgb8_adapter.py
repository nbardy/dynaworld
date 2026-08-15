"""Bounded Neural3D/MP4 adapter for the mapped-RGB8 cache converter.

The adapter has two deliberately separate responsibilities:

* ``PyAvOpenFileRgb8Decoder`` and ``FfmpegPipeRgb8Decoder`` decode through the
  converter-owned bounded file object.  Neither reopens an MP4 by path during
  decode, and each retains at most codec state plus one cached Python-visible
  returned RGB8 frame between calls.  Decoder allocator peaks remain
  unmeasured and are not presented as a process-memory byte cap.
* ``prepare_neural3d_mapped_rgb8_request`` turns one checked-in Neural3D
  manifest row into an exact train or heldout view cache.  Both camera lists
  are validated together and the selected cache cannot contain a view from
  the other split.

Binding v1 has no first-class decoder-provenance field.  The adapter therefore
writes a strict conversion descriptor and passes that descriptor as the
binding's hashed ``raw_dataset_manifest``.  The descriptor pins the original
dataset manifest, poses file, adapter/converter sources, exact decoder runtime,
logical-to-native frame map, and camera-generation digest.  Mixed decoder
backends or a descriptor-free public build fail closed instead of being
misrepresented by binding v1.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import select
import shutil
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

import build_worldfoam_mapped_rgb8_cache as cache_builder
from worldfoam_target_dataset_binding import TARGET_SPLITS


NEURAL3D_DESCRIPTOR_SCHEMA = (
    "dynaworld.neural3d_mapped_rgb8_conversion_descriptor/v1"
)
NEURAL3D_OFFLINE_PREFLIGHT_SCHEMA = (
    "dynaworld.neural3d_mapped_rgb8_offline_preflight/v1"
)
PYAV_DECODER_PROVENANCE = (
    "neural3d-pyav-open-file-cfr-rgb24-pillow-bilinear/v1"
)
FFMPEG_DECODER_PROVENANCE = (
    "neural3d-ffmpeg-stdin-cfr-select-rgb24-bilinear/v1"
)
NEURAL3D_POSE_SOURCE = "neural_3d_llff_opencv_relative_pinhole_v2"

_HASH_CHUNK_BYTES = 1024 * 1024
_DEFAULT_MAXIMUM_SEQUENTIAL_GAP_FRAMES = 8
_FFMPEG_IO_CHUNK_BYTES = 1024 * 1024
_FFMPEG_MAXIMUM_DIAGNOSTIC_BYTES = 1024 * 1024
_FFMPEG_PROCESS_TIMEOUT_SECONDS = 120.0
_FFMPEG_MAXIMUM_EXECUTABLE_BYTES = 512 * 1024 * 1024
_DESCRIPTOR_KEYS = {
    "schema",
    "dataset",
    "dataset_manifest",
    "dataset_record_sha256",
    "poses_bounds",
    "target_split",
    "view_ids",
    "raw_video_path_labels",
    "output_height",
    "output_width",
    "stored_logical_frame_indices",
    "logical_frame_maps",
    "native_frame_indices_by_view",
    "decoder",
    "camera_generation_digest",
    "descriptor_sha256",
}


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_positive(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _finite_nonnegative(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


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


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


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


def _stat_signature(stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


def _read_stable_file(path: Path, *, maximum_bytes: int, name: str) -> bytes:
    _positive_int(maximum_bytes, name=f"{name} byte cap")
    with path.open("rb") as handle:
        before = _stat_signature(os.fstat(handle.fileno()))
        if before[2] < 1 or before[2] > maximum_bytes:
            raise MemoryError(f"{name} exceeds its byte cap before read")
        raw = handle.read(maximum_bytes + 1)
        after = _stat_signature(os.fstat(handle.fileno()))
    if before != after or len(raw) != before[2] or len(raw) > maximum_bytes:
        raise ValueError(f"{name} changed during its bounded read")
    return raw


def _file_identity(
    path: Path,
    *,
    path_label: str,
    maximum_bytes: int,
    name: str,
) -> dict[str, Any]:
    _portable_relative_label(path_label, name=f"{name} path label")
    with path.open("rb") as handle:
        before = _stat_signature(os.fstat(handle.fileno()))
        if before[2] < 1 or before[2] > maximum_bytes:
            raise MemoryError(f"{name} exceeds its hash cap before scan")
        digest = hashlib.sha256()
        remaining = before[2]
        while remaining:
            chunk = handle.read(min(_HASH_CHUNK_BYTES, remaining))
            if not chunk:
                raise ValueError(f"{name} ended during its bounded hash scan")
            digest.update(chunk)
            remaining -= len(chunk)
        after = _stat_signature(os.fstat(handle.fileno()))
    if before != after:
        raise ValueError(f"{name} changed during its bounded hash scan")
    return {
        "path_label": path_label,
        "size_bytes": before[2],
        "sha256": digest.hexdigest(),
    }


def _path_beneath(root: Path, label: str, *, name: str) -> Path:
    canonical = _portable_relative_label(label, name=name)
    resolved_root = root.expanduser().resolve()
    path = (resolved_root / canonical).resolve()
    try:
        path.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"{name} escaped the repository root") from error
    return path


def _path_and_label_beneath(
    root: Path,
    value: Any,
    *,
    name: str,
) -> tuple[Path, str]:
    """Resolve an absolute or root-relative manifest path to a portable label."""

    raw = _nonempty_trimmed(value, name=name)
    resolved_root = root.expanduser().resolve()
    candidate = Path(raw).expanduser()
    path = candidate.resolve() if candidate.is_absolute() else (resolved_root / raw).resolve()
    try:
        relative = path.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"{name} escaped the repository root") from error
    label = _portable_relative_label(relative.as_posix(), name=f"{name} path label")
    return path, label


def _import_av() -> Any:
    try:
        import av
    except ImportError as error:
        raise ImportError(
            "PyAV is required for bounded Neural3D conversion; run this offline "
            "converter in an environment with the `av` package installed"
        ) from error
    return av


class _BoundedProbeFile:
    """Bound PyAV header reads while retaining ordinary seek semantics."""

    def __init__(self, handle: BinaryIO, *, maximum_read_bytes: int) -> None:
        self._handle = handle
        self.maximum_read_bytes = _positive_int(
            maximum_read_bytes,
            name="MP4 header read cap",
        )
        self.read_bytes = 0

    def read(self, size: int = -1) -> bytes:
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("bounded MP4 probe requires explicit read sizes")
        if self.read_bytes + size > self.maximum_read_bytes:
            raise MemoryError("MP4 header probe exceeds its byte cap before read")
        value = self._handle.read(size)
        self.read_bytes += len(value)
        return value

    def readinto(self, buffer: Any) -> int:
        requested = memoryview(buffer).nbytes
        if self.read_bytes + requested > self.maximum_read_bytes:
            raise MemoryError("MP4 header probe exceeds its byte cap before read")
        count = int(self._handle.readinto(buffer))
        self.read_bytes += count
        return count

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        return int(self._handle.seek(offset, whence))

    def tell(self) -> int:
        return int(self._handle.tell())

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    @property
    def closed(self) -> bool:
        return bool(self._handle.closed)


def _probe_mp4_header(
    path: Path,
    *,
    maximum_read_bytes: int,
) -> tuple[dict[str, Any], int]:
    """Read one exact MP4 video-stream header without decoding a frame."""

    av = _import_av()
    with path.open("rb") as raw_handle:
        signature_before = _stat_signature(os.fstat(raw_handle.fileno()))
        bounded = _BoundedProbeFile(
            raw_handle,
            maximum_read_bytes=maximum_read_bytes,
        )
        container = av.open(
            bounded,
            mode="r",
            format="mp4",
            buffer_size=min(32768, maximum_read_bytes),
        )
        try:
            streams = tuple(
                stream
                for stream in container.streams
                if str(getattr(stream, "type", "")) == "video"
            )
            if len(streams) != 1:
                raise ValueError("Neural3D MP4 must contain exactly one video stream")
            stream = streams[0]
            rate = Fraction(stream.average_rate)
            time_base = Fraction(stream.time_base)
            frame_count = int(getattr(stream, "frames", 0) or 0)
            height = int(getattr(stream, "height", 0) or 0)
            width = int(getattr(stream, "width", 0) or 0)
            if rate <= 0 or time_base <= 0:
                raise ValueError("Neural3D MP4 header has invalid timing metadata")
            if frame_count < 1 or height < 1 or width < 1:
                raise ValueError(
                    "Neural3D MP4 header must declare frame count and dimensions"
                )
            header = {
                "native_fps_numerator": rate.numerator,
                "native_fps_denominator": rate.denominator,
                "native_frame_count": frame_count,
                "native_height": height,
                "native_width": width,
                "stream_time_base_numerator": time_base.numerator,
                "stream_time_base_denominator": time_base.denominator,
                "stream_start_time": int(getattr(stream, "start_time", 0) or 0),
            }
        finally:
            container.close()
        signature_after = _stat_signature(os.fstat(raw_handle.fileno()))
    if signature_before != signature_after:
        raise ValueError("Neural3D MP4 changed during its bounded header probe")
    return header, bounded.read_bytes


def _runtime_version_payload() -> dict[str, Any]:
    av = _import_av()
    import PIL

    library_versions = getattr(av, "library_versions", {})
    if not isinstance(library_versions, Mapping) or not library_versions:
        raise RuntimeError("PyAV library_versions must be a nonempty mapping")
    normalized_libraries = {}
    for raw_name, raw_version in sorted(
        library_versions.items(),
        key=lambda item: str(item[0]),
    ):
        name = _nonempty_trimmed(str(raw_name), name="libav library name")
        version = (
            ".".join(str(part) for part in raw_version)
            if isinstance(raw_version, Sequence)
            and not isinstance(raw_version, (str, bytes))
            else str(raw_version)
        )
        version = _nonempty_trimmed(version, name=f"libav {name} version")
        if name in normalized_libraries:
            raise RuntimeError("PyAV library names collide after normalization")
        normalized_libraries[name] = version
    payload = {
        "decoder_provenance": PYAV_DECODER_PROVENANCE,
        "pyav_version": _nonempty_trimmed(
            getattr(av, "__version__", ""),
            name="PyAV version",
        ),
        "libav_versions": normalized_libraries,
        "pillow_version": _nonempty_trimmed(
            getattr(PIL, "__version__", ""),
            name="Pillow version",
        ),
        "thread_count": 1,
        "source_pixel_format": "rgb24",
        "resize": "PIL.Image.Resampling.BILINEAR",
        "logical_to_native_index": "round((start_seconds+i/sample_fps)*native_fps)",
        "variable_frame_rate_allowed": False,
    }
    return {**payload, "runtime_sha256": _canonical_sha256(payload)}


def _external_tool_identity(name: str) -> dict[str, Any]:
    """Seal one installed ffmpeg-family executable and its version output."""

    executable = shutil.which(name)
    if executable is None:
        raise ImportError(f"{name} is not installed or is absent from PATH")
    path = Path(executable).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"resolved {name} executable is not a file: {path}")
    with path.open("rb") as handle:
        before = _stat_signature(os.fstat(handle.fileno()))
        if before[2] < 1 or before[2] > _FFMPEG_MAXIMUM_EXECUTABLE_BYTES:
            raise MemoryError(f"{name} executable exceeds its identity-scan cap")
        digest = hashlib.sha256()
        remaining = before[2]
        while remaining:
            chunk = handle.read(min(_HASH_CHUNK_BYTES, remaining))
            if not chunk:
                raise ValueError(f"{name} executable ended during identity scan")
            digest.update(chunk)
            remaining -= len(chunk)
        after = _stat_signature(os.fstat(handle.fileno()))
    if before != after:
        raise ValueError(f"{name} executable changed during identity scan")
    completed = subprocess.run(
        [str(path), "-version"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=_FFMPEG_PROCESS_TIMEOUT_SECONDS,
        env={**os.environ, "LC_ALL": "C", "LANG": "C"},
    )
    output = completed.stdout
    if completed.returncode != 0:
        raise RuntimeError(f"{name} -version failed with status {completed.returncode}")
    if not output or len(output) > _FFMPEG_MAXIMUM_DIAGNOSTIC_BYTES:
        raise MemoryError(f"{name} version output exceeds its diagnostic cap")
    try:
        version_text = output.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{name} version output is not UTF-8") from error
    version_lines = version_text.splitlines()
    if not version_lines or not version_lines[0].strip():
        raise ValueError(f"{name} version output has no identity line")
    return {
        "path": str(path),
        "size_bytes": before[2],
        "sha256": digest.hexdigest(),
        "version_line": version_lines[0].strip(),
        "version_output_sha256": hashlib.sha256(output).hexdigest(),
    }


def _ffmpeg_runtime_version_payload() -> dict[str, Any]:
    payload = {
        "decoder_provenance": FFMPEG_DECODER_PROVENANCE,
        "ffmpeg": _external_tool_identity("ffmpeg"),
        "ffprobe": _external_tool_identity("ffprobe"),
        "thread_count": 1,
        "input": "pipe:0_from_converter_owned_bounded_handle",
        "output_pixel_format": "rgb24",
        "resize": "libswscale_bilinear",
        "selected_frame_filter": "exact_native_frame_index_select",
        "logical_to_native_index": (
            "round((start_seconds+i/sample_fps)*native_fps)"
        ),
        "variable_frame_rate_allowed": False,
        "whole_video_materialized": False,
    }
    return {**payload, "runtime_sha256": _canonical_sha256(payload)}


def _selected_runtime_version_payload() -> dict[str, Any]:
    """Prefer the in-process decoder; use a sealed ffmpeg pipe fallback."""

    try:
        return _runtime_version_payload()
    except ImportError as pyav_error:
        try:
            return _ffmpeg_runtime_version_payload()
        except ImportError as ffmpeg_error:
            raise ImportError(
                "no bounded Neural3D decoder runtime is available: PyAV is "
                "unavailable and ffmpeg/ffprobe are not both installed"
            ) from ExceptionGroup(
                "Neural3D decoder runtime failures",
                [pyav_error, ffmpeg_error],
            )


def _ffprobe_fraction(value: Any, *, name: str) -> Fraction:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"ffprobe {name} must be a nonempty rational string")
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError(f"ffprobe {name} is not an exact rational") from error
    if result <= 0:
        raise ValueError(f"ffprobe {name} must be positive")
    return result


def _probe_mp4_header_ffprobe(
    path: Path,
    *,
    maximum_read_bytes: int,
    runtime: Mapping[str, Any],
) -> tuple[dict[str, Any], int]:
    """Probe one MP4 with sealed ffprobe and a conservative analysis budget.

    ffprobe does not expose exact OS-level bytes read.  The returned byte count
    therefore reserves the entire configured ``probesize`` rather than
    pretending to have measured less.
    """

    cap = _positive_int(maximum_read_bytes, name="MP4 ffprobe analysis cap")
    ffprobe = runtime.get("ffprobe")
    if not isinstance(ffprobe, Mapping):
        raise ValueError("ffmpeg runtime lacks its sealed ffprobe identity")
    executable = _nonempty_trimmed(ffprobe.get("path"), name="ffprobe path")
    resolved = Path(executable).resolve()
    current = _external_tool_identity("ffprobe")
    if current != dict(ffprobe) or resolved != Path(current["path"]):
        raise RuntimeError("ffprobe runtime drifted from the conversion descriptor")
    with path.open("rb") as identity_handle:
        signature_before = _stat_signature(os.fstat(identity_handle.fileno()))
        completed = subprocess.run(
            [
                executable,
                "-v",
                "error",
                "-probesize",
                str(cap),
                "-analyzeduration",
                "0",
                "-select_streams",
                "v",
                "-show_entries",
                (
                    "stream=codec_type,avg_frame_rate,time_base,nb_frames,"
                    "height,width,start_pts"
                ),
                "-of",
                "json",
                str(path),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=_FFMPEG_PROCESS_TIMEOUT_SECONDS,
            env={**os.environ, "LC_ALL": "C", "LANG": "C"},
        )
        signature_after = _stat_signature(os.fstat(identity_handle.fileno()))
    with path.open("rb") as repeated_handle:
        repeated_signature = _stat_signature(os.fstat(repeated_handle.fileno()))
    diagnostics = completed.stderr
    if len(completed.stdout) > _FFMPEG_MAXIMUM_DIAGNOSTIC_BYTES or len(
        diagnostics
    ) > _FFMPEG_MAXIMUM_DIAGNOSTIC_BYTES:
        raise MemoryError("ffprobe output exceeds its diagnostic cap")
    if completed.returncode != 0:
        detail = diagnostics.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"ffprobe MP4 header probe failed with status {completed.returncode}: "
            f"{detail[:4096]}"
        )
    if signature_before != signature_after or signature_before != repeated_signature:
        raise ValueError("Neural3D MP4 changed during its ffprobe header probe")
    value = _strict_json_loads(completed.stdout, name="ffprobe MP4 header")
    if not isinstance(value, Mapping) or set(value) != {"streams"}:
        raise ValueError("ffprobe MP4 header must contain exactly streams")
    streams = value["streams"]
    if not isinstance(streams, list) or len(streams) != 1:
        raise ValueError("Neural3D MP4 must contain exactly one video stream")
    stream = streams[0]
    if not isinstance(stream, Mapping) or stream.get("codec_type") != "video":
        raise ValueError("ffprobe did not return one exact video stream")
    rate = _ffprobe_fraction(stream.get("avg_frame_rate"), name="average rate")
    time_base = _ffprobe_fraction(stream.get("time_base"), name="time base")
    try:
        frame_count = int(stream.get("nb_frames"))
        height = int(stream.get("height"))
        width = int(stream.get("width"))
        start_time = int(stream.get("start_pts", 0) or 0)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "ffprobe MP4 header must declare integer frame count, dimensions, and start"
        ) from error
    if frame_count < 1 or height < 1 or width < 1:
        raise ValueError("ffprobe MP4 header has invalid frame count or dimensions")
    return (
        {
            "native_fps_numerator": rate.numerator,
            "native_fps_denominator": rate.denominator,
            "native_frame_count": frame_count,
            "native_height": height,
            "native_width": width,
            "stream_time_base_numerator": time_base.numerator,
            "stream_time_base_denominator": time_base.denominator,
            "stream_start_time": start_time,
        },
        cap,
    )


def _probe_mp4_header_for_runtime(
    path: Path,
    *,
    maximum_read_bytes: int,
    runtime: Mapping[str, Any],
) -> tuple[dict[str, Any], int]:
    provenance = runtime.get("decoder_provenance")
    if provenance == PYAV_DECODER_PROVENANCE:
        return _probe_mp4_header(path, maximum_read_bytes=maximum_read_bytes)
    if provenance == FFMPEG_DECODER_PROVENANCE:
        return _probe_mp4_header_ffprobe(
            path,
            maximum_read_bytes=maximum_read_bytes,
            runtime=runtime,
        )
    raise ValueError("Neural3D runtime has an unsupported decoder provenance")


def _decoder_resource_contract(
    runtime: Mapping[str, Any],
    *,
    adapter_limits: "Neural3dMappedRgb8AdapterLimits",
    conversion_limits: cache_builder.WorldFoamMappedRgb8ConversionLimits,
) -> dict[str, Any]:
    common = {
        "adapter_limits": dict(adapter_limits.__dict__),
        "converter_limits": dict(conversion_limits.__dict__),
        "whole_video_materialized": False,
        "maximum_retained_returned_rgb8_frames_per_view": 1,
        "codec_state_byte_cap_available": False,
        "python_scratch_cap_kind": "logical_visible_buffers_not_allocator_peak",
        "allocator_peak_measured": False,
        "process_rss_cap_enforced": False,
        "scope": "python_visible_buffers_io_and_decoded_frame_work",
    }
    provenance = runtime.get("decoder_provenance")
    if provenance == PYAV_DECODER_PROVENANCE:
        return {
            "adapter_limits": common["adapter_limits"],
            "converter_limits": common["converter_limits"],
            "maximum_sequential_gap_frames": (
                _DEFAULT_MAXIMUM_SEQUENTIAL_GAP_FRAMES
            ),
            "whole_video_materialized": common["whole_video_materialized"],
            "maximum_retained_returned_rgb8_frames_per_view": common[
                "maximum_retained_returned_rgb8_frames_per_view"
            ],
            "codec_state_byte_cap_available": common[
                "codec_state_byte_cap_available"
            ],
            "python_scratch_cap_kind": common["python_scratch_cap_kind"],
            "python_scratch_upper_bound_formula": (
                "3*native_rgb8_bytes+4*output_rgb8_bytes"
            ),
            "allocator_peak_measured": common["allocator_peak_measured"],
            "process_rss_cap_enforced": common["process_rss_cap_enforced"],
            "scope": common["scope"],
        }
    if provenance == FFMPEG_DECODER_PROVENANCE:
        return {
            **common,
            "input_transport": "pipe:0_from_converter_owned_bounded_handle",
            "maximum_feeder_chunk_bytes": _FFMPEG_IO_CHUNK_BYTES,
            "maximum_stderr_capture_bytes": _FFMPEG_MAXIMUM_DIAGNOSTIC_BYTES,
            "maximum_subprocess_seconds_per_blocking_phase": (
                _FFMPEG_PROCESS_TIMEOUT_SECONDS
            ),
            "ffprobe_header_read_accounting": (
                "configured_probesize_reserved;exact_OS_bytes_not_exposed"
            ),
            "python_scratch_upper_bound_formula": "3*output_rgb8_bytes",
            "subprocess_codec_peak_measured": False,
        }
    raise ValueError("unsupported decoder runtime resource contract")


def neural3d_mapped_rgb8_offline_preflight() -> dict[str, Any]:
    """Report cache-preparation readiness without probing or decoding a dataset.

    Public training consumes already-built, rehashed train/heldout caches.  It
    must not silently fall back to eager video decode.  PyAV is preferred;
    installed ffmpeg/ffprobe provide an identity-sealed streaming fallback.
    """

    try:
        runtime = _selected_runtime_version_payload()
    except ImportError:
        runtime = None
        blockers = ["bounded_decoder_runtime_not_installed"]
    else:
        blockers = []
    payload: dict[str, Any] = {
        "schema": NEURAL3D_OFFLINE_PREFLIGHT_SCHEMA,
        "ready": not blockers,
        "blockers": blockers,
        "supported_target_splits": sorted(TARGET_SPLITS),
        "cache_schema": cache_builder.MAPPED_RGB8_SCHEMA,
        "cache_layout": cache_builder.MAPPED_RGB8_LAYOUT,
        "chunked_selected_pixel_reads_supported_by_layout": True,
        "whole_video_materialized": False,
        "runtime": runtime,
        "preflight_sha256": "",
    }
    payload["preflight_sha256"] = _canonical_sha256(
        {
            key: value
            for key, value in payload.items()
            if key != "preflight_sha256"
        }
    )
    return payload


@dataclass
class PyAvOpenFileRgb8Decoder:
    """Decode selected CFR MP4 frames through one caller-owned bounded handle."""

    expected_view_id: str
    selected_logical_frame_indices: tuple[int, ...]
    source_frame_count: int
    start_seconds: float
    sample_fps: float
    expected_native_fps_numerator: int
    expected_native_fps_denominator: int
    expected_native_frame_count: int
    expected_native_height: int
    expected_native_width: int
    expected_stream_time_base_numerator: int
    expected_stream_time_base_denominator: int
    expected_stream_start_time: int
    maximum_native_frame_bytes: int
    maximum_python_rgb_scratch_bytes: int
    maximum_decoded_native_frames: int
    expected_runtime_sha256: str
    maximum_sequential_gap_frames: int = _DEFAULT_MAXIMUM_SEQUENTIAL_GAP_FRAMES
    provenance: str = field(default=PYAV_DECODER_PROVENANCE, init=False)
    uses_supplied_handle_exclusively: bool = field(default=True, init=False)
    reads_only_through_bounded_handle_api: bool = field(default=True, init=False)
    decoded_native_frame_count: int = field(default=0, init=False)
    _container: Any = field(default=None, init=False, repr=False)
    _stream: Any = field(default=None, init=False, repr=False)
    _decode_iterator: Any = field(default=None, init=False, repr=False)
    _handle_identity: int | None = field(default=None, init=False, repr=False)
    _stream_rate: Fraction | None = field(default=None, init=False, repr=False)
    _stream_time_base: Fraction | None = field(default=None, init=False, repr=False)
    _stream_start_time: int = field(default=0, init=False, repr=False)
    _last_decoded_native_index: int = field(default=-1, init=False, repr=False)
    _last_returned_native_index: int = field(default=-1, init=False, repr=False)
    _last_returned_rgb8: bytes | None = field(default=None, init=False, repr=False)
    _request_cursor: int = field(default=0, init=False, repr=False)
    _output_shape: tuple[int, int] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        _nonempty_trimmed(self.expected_view_id, name="expected_view_id")
        selected = tuple(self.selected_logical_frame_indices)
        if (
            not selected
            or selected != tuple(sorted(set(selected)))
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                for index in selected
            )
        ):
            raise ValueError("PyAV selected logical frames must be unique and increasing")
        self.selected_logical_frame_indices = selected
        _positive_int(self.source_frame_count, name="source_frame_count")
        if selected[-1] >= self.source_frame_count:
            raise IndexError("PyAV selected logical frame leaves the declared source")
        _finite_positive(self.sample_fps, name="sample_fps")
        _finite_nonnegative(self.start_seconds, name="start_seconds")
        for name in (
            "expected_native_fps_numerator",
            "expected_native_fps_denominator",
            "expected_native_frame_count",
            "expected_native_height",
            "expected_native_width",
            "expected_stream_time_base_numerator",
            "expected_stream_time_base_denominator",
            "maximum_native_frame_bytes",
            "maximum_python_rgb_scratch_bytes",
            "maximum_decoded_native_frames",
            "maximum_sequential_gap_frames",
        ):
            _positive_int(getattr(self, name), name=name)
        if (
            isinstance(self.expected_stream_start_time, bool)
            or not isinstance(self.expected_stream_start_time, int)
        ):
            raise TypeError("expected_stream_start_time must be an integer")
        if not isinstance(self.expected_runtime_sha256, str) or (
            len(self.expected_runtime_sha256) != 64
        ) or any(
            character not in "0123456789abcdef"
            for character in self.expected_runtime_sha256
        ):
            raise ValueError("expected_runtime_sha256 must be a lowercase SHA-256")
        native_bytes = (
            self.expected_native_height * self.expected_native_width * 3
        )
        if native_bytes > self.maximum_native_frame_bytes:
            raise MemoryError("declared native RGB frame exceeds its cap")
        native_indices = tuple(self.logical_to_native_index(index) for index in selected)
        if native_indices != tuple(sorted(native_indices)):
            raise ValueError("logical-to-native frame mapping is not monotone")
        if native_indices[-1] >= self.expected_native_frame_count:
            raise IndexError("selected native frame leaves the declared MP4 stream")

    @property
    def expected_native_fps(self) -> float:
        return float(
            Fraction(
                self.expected_native_fps_numerator,
                self.expected_native_fps_denominator,
            )
        )

    def logical_to_native_index(self, logical_index: int) -> int:
        return int(
            round(
                (
                    float(self.start_seconds)
                    + float(logical_index) / float(self.sample_fps)
                )
                * float(self.expected_native_fps)
            )
        )

    def _open_session(
        self,
        handle: BinaryIO,
        *,
        height: int,
        width: int,
    ) -> None:
        runtime = _runtime_version_payload()
        if runtime["runtime_sha256"] != self.expected_runtime_sha256:
            raise RuntimeError("PyAV/Pillow runtime drifted from the bound descriptor")
        av = _import_av()
        container = av.open(handle, mode="r", format="mp4")
        try:
            streams = tuple(
                stream
                for stream in container.streams
                if str(getattr(stream, "type", "")) == "video"
            )
            if len(streams) != 1:
                raise ValueError("Neural3D MP4 must contain exactly one video stream")
            stream = streams[0]
            rate = Fraction(stream.average_rate)
            time_base = Fraction(stream.time_base)
            if rate <= 0 or time_base <= 0:
                raise ValueError("Neural3D MP4 has invalid CFR timing metadata")
            expected_rate = Fraction(
                self.expected_native_fps_numerator,
                self.expected_native_fps_denominator,
            )
            expected_time_base = Fraction(
                self.expected_stream_time_base_numerator,
                self.expected_stream_time_base_denominator,
            )
            if rate != expected_rate:
                raise ValueError("Neural3D MP4 native FPS differs from its descriptor")
            if time_base != expected_time_base:
                raise ValueError("Neural3D MP4 time base differs from its descriptor")
            native_frame_count = int(getattr(stream, "frames", 0) or 0)
            if native_frame_count != self.expected_native_frame_count:
                raise ValueError("Neural3D MP4 frame count differs from its descriptor")
            stream_start_time = int(getattr(stream, "start_time", 0) or 0)
            if stream_start_time != self.expected_stream_start_time:
                raise ValueError("Neural3D MP4 start time differs from its descriptor")
            native_height = int(getattr(stream, "height", 0))
            native_width = int(getattr(stream, "width", 0))
            if (
                native_height != self.expected_native_height
                or native_width != self.expected_native_width
            ):
                raise ValueError("Neural3D MP4 native dimensions differ from its descriptor")
            native_bytes = native_height * native_width * 3
            output_bytes = int(height) * int(width) * 3
            # This fences the source-visible Python arrays/bytes that the
            # adapter deliberately creates. PyAV, libav, NumPy, and Pillow may
            # reserve additional allocator storage; that peak requires a
            # separate process/system-memory measurement gate.
            # Conservatively cover three native-sized logical buffers and four
            # output-sized logical buffers.  The latter admit a resized Pillow
            # image, its NumPy exposure/copy, the immutable returned bytes, and
            # one transient conversion overlap.  The preceding cached return
            # is released before a distinct frame is converted below.  This is
            # a source-visible logical bound, not an allocator/RSS claim.
            logical_python_scratch = 3 * native_bytes + 4 * output_bytes
            if native_bytes > self.maximum_native_frame_bytes:
                raise MemoryError("native RGB frame exceeds its decode cap")
            if logical_python_scratch > self.maximum_python_rgb_scratch_bytes:
                raise MemoryError("PyAV logical RGB conversion scratch exceeds its cap")
            codec_context = getattr(stream, "codec_context", None)
            if codec_context is not None:
                codec_context.thread_count = 1
            self._container = container
            self._stream = stream
            self._decode_iterator = iter(container.decode(stream))
            self._handle_identity = id(handle)
            self._stream_rate = rate
            self._stream_time_base = time_base
            self._stream_start_time = stream_start_time
            self._output_shape = (int(height), int(width))
        except BaseException:
            container.close()
            raise

    def _frame_native_index(self, frame: Any) -> int:
        if frame.pts is None or self._stream_rate is None or self._stream_time_base is None:
            raise ValueError("Neural3D MP4 frame has no exact presentation timestamp")
        position = (
            Fraction(int(frame.pts) - self._stream_start_time)
            * self._stream_time_base
            * self._stream_rate
        )
        nearest = int(round(position))
        if abs(position - nearest) > Fraction(1, 1000):
            raise ValueError("Neural3D decoder encountered variable-frame-rate timing")
        return nearest

    def _seek(self, native_frame_index: int) -> None:
        if (
            self._container is None
            or self._stream is None
            or self._stream_rate is None
            or self._stream_time_base is None
        ):
            raise RuntimeError("PyAV seek requested without an open stream")
        offset = self._stream_start_time + int(
            round(
                Fraction(native_frame_index, 1)
                / self._stream_rate
                / self._stream_time_base
            )
        )
        self._container.seek(
            offset,
            backward=True,
            any_frame=False,
            stream=self._stream,
        )
        self._decode_iterator = iter(self._container.decode(self._stream))
        self._last_decoded_native_index = -1

    def _next_frame(self) -> Any:
        if self.decoded_native_frame_count >= self.maximum_decoded_native_frames:
            raise MemoryError("PyAV decoded-frame work exceeds its explicit cap")
        if self._decode_iterator is None:
            raise RuntimeError("PyAV decode iterator is not open")
        try:
            frame = next(self._decode_iterator)
        except StopIteration as error:
            raise RuntimeError("Neural3D MP4 ended before the selected frame") from error
        self.decoded_native_frame_count += 1
        return frame

    def _rgb8_bytes(self, frame: Any, *, height: int, width: int) -> bytes:
        import numpy as np
        from PIL import Image

        array = frame.to_ndarray(format="rgb24")
        expected_native_shape = (
            self.expected_native_height,
            self.expected_native_width,
            3,
        )
        if array.dtype != np.uint8 or tuple(array.shape) != expected_native_shape:
            raise ValueError("PyAV changed the declared native RGB24 frame contract")
        image = Image.fromarray(array, mode="RGB")
        if image.size != (int(width), int(height)):
            image = image.resize(
                (int(width), int(height)),
                resample=Image.Resampling.BILINEAR,
            )
        resized = np.asarray(image, dtype=np.uint8)
        if tuple(resized.shape) != (int(height), int(width), 3):
            raise ValueError("PyAV/Pillow changed the output RGB8 frame shape")
        result = resized.tobytes(order="C")
        if len(result) != int(height) * int(width) * 3:
            raise ArithmeticError("PyAV/Pillow changed exact RGB8 output bytes")
        return result

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
        if view_id != self.expected_view_id:
            raise ValueError("PyAV decoder view differs from its bound identity")
        output_bytes = _positive_int(height, name="height") * _positive_int(
            width,
            name="width",
        ) * 3
        if output_bytes > maximum_decoded_frame_bytes:
            raise MemoryError("decoded RGB8 frame exceeds its cap before decode")
        if self._request_cursor >= len(self.selected_logical_frame_indices):
            raise ValueError("PyAV decoder received more frames than its bound selection")
        expected_logical = self.selected_logical_frame_indices[self._request_cursor]
        if source_frame_index != expected_logical:
            raise ValueError("PyAV decoder request differs from its bound frame identity")
        if self._container is None:
            self._open_session(handle, height=height, width=width)
        elif id(handle) != self._handle_identity:
            raise ValueError("PyAV decoder handle changed within one camera conversion")
        elif self._output_shape != (int(height), int(width)):
            raise ValueError("PyAV decoder output dimensions changed within one view")

        target_native = self.logical_to_native_index(source_frame_index)
        if target_native == self._last_returned_native_index:
            if self._last_returned_rgb8 is None:
                raise ArithmeticError("duplicate native frame cache is missing")
            result = self._last_returned_rgb8
        else:
            # Logical requests are monotone, so a prior distinct native frame
            # can never be requested again.  Drop its immutable RGB8 cache
            # before allocating the next conversion's visible buffers.
            self._last_returned_rgb8 = None
            gap = target_native - self._last_decoded_native_index
            if self._last_decoded_native_index >= 0 and gap < 0:
                raise ValueError("PyAV native frame requests must be monotone")
            if (
                self._last_decoded_native_index < 0
                and target_native > self.maximum_sequential_gap_frames
            ) or gap > self.maximum_sequential_gap_frames:
                self._seek(target_native)
            selected_frame = None
            while selected_frame is None:
                frame = self._next_frame()
                decoded_index = self._frame_native_index(frame)
                if (
                    self._last_decoded_native_index >= 0
                    and decoded_index < self._last_decoded_native_index
                ):
                    raise ValueError("PyAV presentation order moved backwards")
                self._last_decoded_native_index = decoded_index
                if decoded_index < target_native:
                    continue
                if decoded_index > target_native:
                    raise ValueError("PyAV seek/decode skipped the exact selected frame")
                selected_frame = frame
            result = self._rgb8_bytes(selected_frame, height=height, width=width)
            self._last_returned_native_index = target_native
            self._last_returned_rgb8 = result
        self._request_cursor += 1
        return result

    def close_open_file_decode(
        self,
        handle: BinaryIO,
        *,
        view_id: str,
        decode_completed: bool,
    ) -> None:
        wrong_view = view_id != self.expected_view_id
        wrong_handle = (
            self._container is not None and id(handle) != self._handle_identity
        )
        invalid_completion_flag = not isinstance(decode_completed, bool)
        incomplete_success = (
            decode_completed is True
            and self._request_cursor != len(self.selected_logical_frame_indices)
        )
        try:
            if self._container is not None:
                self._container.close()
        finally:
            self._container = None
            self._stream = None
            self._decode_iterator = None
            self._handle_identity = None
            self._stream_rate = None
            self._stream_time_base = None
            self._last_returned_rgb8 = None
        if wrong_view:
            raise ValueError("PyAV close view differs from its bound identity")
        if wrong_handle:
            raise ValueError("PyAV close handle differs from its open decode handle")
        if invalid_completion_flag:
            raise TypeError("PyAV close requires an exact decode_completed boolean")
        if incomplete_success:
            raise ValueError("successful PyAV close did not consume its exact frame plan")


@dataclass
class FfmpegPipeRgb8Decoder:
    """Stream one MP4 through ffmpeg and retain at most one returned RGB8 frame.

    The subprocess receives bytes only from the converter-owned bounded file
    object through ``pipe:0``.  Its filter emits only the exact unique native
    frame indices named by the descriptor.  This bounds Python-visible memory
    and avoids an eager video tensor; ffmpeg/libav allocator RSS remains an
    explicitly unmeasured runtime quantity.
    """

    expected_view_id: str
    selected_logical_frame_indices: tuple[int, ...]
    source_frame_count: int
    start_seconds: float
    sample_fps: float
    expected_native_fps_numerator: int
    expected_native_fps_denominator: int
    expected_native_frame_count: int
    expected_native_height: int
    expected_native_width: int
    expected_stream_time_base_numerator: int
    expected_stream_time_base_denominator: int
    expected_stream_start_time: int
    maximum_native_frame_bytes: int
    maximum_python_rgb_scratch_bytes: int
    maximum_decoded_native_frames: int
    expected_runtime_sha256: str
    provenance: str = field(default=FFMPEG_DECODER_PROVENANCE, init=False)
    uses_supplied_handle_exclusively: bool = field(default=True, init=False)
    reads_only_through_bounded_handle_api: bool = field(default=True, init=False)
    decoded_native_frame_count: int = field(default=0, init=False)
    _native_indices: tuple[int, ...] = field(default=(), init=False, repr=False)
    _unique_native_indices: tuple[int, ...] = field(
        default=(), init=False, repr=False
    )
    _process: Any = field(default=None, init=False, repr=False)
    _feeder_thread: threading.Thread | None = field(
        default=None, init=False, repr=False
    )
    _stderr_thread: threading.Thread | None = field(
        default=None, init=False, repr=False
    )
    _handle_identity: int | None = field(default=None, init=False, repr=False)
    _output_shape: tuple[int, int] | None = field(default=None, init=False, repr=False)
    _request_cursor: int = field(default=0, init=False, repr=False)
    _unique_output_cursor: int = field(default=0, init=False, repr=False)
    _last_returned_native_index: int = field(default=-1, init=False, repr=False)
    _last_returned_rgb8: bytes | None = field(default=None, init=False, repr=False)
    _feeder_error: BaseException | None = field(default=None, init=False, repr=False)
    _stderr_error: BaseException | None = field(default=None, init=False, repr=False)
    _stderr_bytes: bytearray = field(default_factory=bytearray, init=False, repr=False)
    _stderr_overflowed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        _nonempty_trimmed(self.expected_view_id, name="expected_view_id")
        selected = tuple(self.selected_logical_frame_indices)
        if (
            not selected
            or selected != tuple(sorted(set(selected)))
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                for index in selected
            )
        ):
            raise ValueError(
                "ffmpeg selected logical frames must be unique and increasing"
            )
        self.selected_logical_frame_indices = selected
        _positive_int(self.source_frame_count, name="source_frame_count")
        if selected[-1] >= self.source_frame_count:
            raise IndexError("ffmpeg selected logical frame leaves the declared source")
        _finite_positive(self.sample_fps, name="sample_fps")
        _finite_nonnegative(self.start_seconds, name="start_seconds")
        for name in (
            "expected_native_fps_numerator",
            "expected_native_fps_denominator",
            "expected_native_frame_count",
            "expected_native_height",
            "expected_native_width",
            "expected_stream_time_base_numerator",
            "expected_stream_time_base_denominator",
            "maximum_native_frame_bytes",
            "maximum_python_rgb_scratch_bytes",
            "maximum_decoded_native_frames",
        ):
            _positive_int(getattr(self, name), name=name)
        if (
            isinstance(self.expected_stream_start_time, bool)
            or not isinstance(self.expected_stream_start_time, int)
        ):
            raise TypeError("expected_stream_start_time must be an integer")
        if not isinstance(self.expected_runtime_sha256, str) or (
            len(self.expected_runtime_sha256) != 64
        ) or any(
            character not in "0123456789abcdef"
            for character in self.expected_runtime_sha256
        ):
            raise ValueError("expected_runtime_sha256 must be a lowercase SHA-256")
        native_bytes = self.expected_native_height * self.expected_native_width * 3
        if native_bytes > self.maximum_native_frame_bytes:
            raise MemoryError("declared native RGB frame exceeds its cap")
        if self.expected_native_frame_count > self.maximum_decoded_native_frames:
            raise MemoryError(
                "ffmpeg full-stream decode can exceed the native-frame decode cap"
            )
        self._native_indices = tuple(
            self.logical_to_native_index(index) for index in selected
        )
        if self._native_indices != tuple(sorted(self._native_indices)):
            raise ValueError("logical-to-native frame mapping is not monotone")
        if self._native_indices[-1] >= self.expected_native_frame_count:
            raise IndexError("selected native frame leaves the declared MP4 stream")
        self._unique_native_indices = tuple(dict.fromkeys(self._native_indices))

    @property
    def expected_native_fps(self) -> float:
        return float(
            Fraction(
                self.expected_native_fps_numerator,
                self.expected_native_fps_denominator,
            )
        )

    def logical_to_native_index(self, logical_index: int) -> int:
        return int(
            round(
                (
                    float(self.start_seconds)
                    + float(logical_index) / float(self.sample_fps)
                )
                * float(self.expected_native_fps)
            )
        )

    def _stderr_detail(self) -> str:
        detail = bytes(self._stderr_bytes).decode("utf-8", errors="replace").strip()
        if self._stderr_overflowed:
            detail = f"{detail}\n[stderr exceeded bounded capture]".strip()
        return detail[:8192]

    def _feed_input(self, handle: BinaryIO, process: Any) -> None:
        try:
            handle.seek(0)
            source_size = int(os.fstat(handle.fileno()).st_size)
            already_read = getattr(handle, "read_bytes", None)
            maximum_read = getattr(handle, "maximum_read_bytes", None)
            if (
                isinstance(already_read, bool)
                or not isinstance(already_read, int)
                or already_read < 0
                or isinstance(maximum_read, bool)
                or not isinstance(maximum_read, int)
                or maximum_read < source_size + already_read
            ):
                raise MemoryError(
                    "bounded ffmpeg input handle cannot admit the exact source"
                )
            remaining = source_size
            while remaining:
                chunk = handle.read(min(_FFMPEG_IO_CHUNK_BYTES, remaining))
                if not chunk:
                    raise ValueError("Neural3D MP4 ended during ffmpeg pipe feed")
                remaining -= len(chunk)
                view = memoryview(chunk)
                while view:
                    if process.stdin is None:
                        raise BrokenPipeError("ffmpeg stdin disappeared during feed")
                    written = process.stdin.write(view)
                    if written is None or written < 1:
                        raise BrokenPipeError("ffmpeg stdin made no write progress")
                    view = view[written:]
            if handle.tell() != source_size:
                raise ValueError("ffmpeg input feed changed the exact source extent")
        except BaseException as error:
            self._feeder_error = error
        finally:
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except (BrokenPipeError, OSError):
                    pass

    def _drain_stderr(self, process: Any) -> None:
        try:
            if process.stderr is None:
                return
            while True:
                chunk = process.stderr.read(_FFMPEG_IO_CHUNK_BYTES)
                if not chunk:
                    break
                available = _FFMPEG_MAXIMUM_DIAGNOSTIC_BYTES - len(
                    self._stderr_bytes
                )
                if available > 0:
                    self._stderr_bytes.extend(chunk[:available])
                if len(chunk) > available:
                    self._stderr_overflowed = True
        except BaseException as error:
            self._stderr_error = error

    def _open_session(
        self,
        handle: BinaryIO,
        *,
        height: int,
        width: int,
    ) -> None:
        runtime = _ffmpeg_runtime_version_payload()
        if runtime["runtime_sha256"] != self.expected_runtime_sha256:
            raise RuntimeError("ffmpeg runtime drifted from the bound descriptor")
        output_bytes = int(height) * int(width) * 3
        if 3 * output_bytes > self.maximum_python_rgb_scratch_bytes:
            raise MemoryError("ffmpeg visible RGB output scratch exceeds its cap")
        select_expression = "+".join(
            f"eq(n\\,{index})" for index in self._unique_native_indices
        )
        ffmpeg_path = runtime["ffmpeg"]["path"]
        process = subprocess.Popen(
            [
                ffmpeg_path,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostats",
                "-threads",
                "1",
                "-i",
                "pipe:0",
                "-map",
                "0:v:0",
                "-an",
                "-sn",
                "-dn",
                "-filter_threads",
                "1",
                "-vf",
                (
                    f"select={select_expression},"
                    f"scale={int(width)}:{int(height)}:flags=bilinear"
                ),
                "-fps_mode",
                "passthrough",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            close_fds=True,
            env={**os.environ, "LC_ALL": "C", "LANG": "C"},
        )
        self._process = process
        self._handle_identity = id(handle)
        self._output_shape = (int(height), int(width))
        self._feeder_thread = threading.Thread(
            target=self._feed_input,
            args=(handle, process),
            name=f"worldfoam-ffmpeg-feed-{self.expected_view_id}",
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            args=(process,),
            name=f"worldfoam-ffmpeg-stderr-{self.expected_view_id}",
            daemon=True,
        )
        self._stderr_thread.start()
        self._feeder_thread.start()

    def _read_exact_stdout(self, expected_bytes: int) -> bytes:
        process = self._process
        if process is None or process.stdout is None:
            raise RuntimeError("ffmpeg output session is not open")
        deadline = time.monotonic() + _FFMPEG_PROCESS_TIMEOUT_SECONDS
        result = bytearray()
        descriptor = process.stdout.fileno()
        while len(result) < expected_bytes:
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                raise TimeoutError("ffmpeg timed out before one selected RGB8 frame")
            ready, _, _ = select.select(
                [descriptor],
                [],
                [],
                min(1.0, remaining_seconds),
            )
            if not ready:
                if process.poll() is None:
                    continue
            chunk = os.read(
                descriptor,
                min(_FFMPEG_IO_CHUNK_BYTES, expected_bytes - len(result)),
            )
            if not chunk:
                detail = self._stderr_detail()
                raise RuntimeError(
                    "ffmpeg ended before one exact selected RGB8 frame"
                    + (f": {detail}" if detail else "")
                )
            result.extend(chunk)
        return bytes(result)

    def _drain_remaining_stdout(self) -> int:
        process = self._process
        if process is None or process.stdout is None:
            return 0
        deadline = time.monotonic() + _FFMPEG_PROCESS_TIMEOUT_SECONDS
        descriptor = process.stdout.fileno()
        extra_bytes = 0
        while True:
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                raise TimeoutError("ffmpeg timed out while closing its output stream")
            ready, _, _ = select.select(
                [descriptor],
                [],
                [],
                min(1.0, remaining_seconds),
            )
            if not ready:
                if process.poll() is None:
                    continue
            chunk = os.read(descriptor, _FFMPEG_IO_CHUNK_BYTES)
            if not chunk:
                return extra_bytes
            extra_bytes += len(chunk)

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
        if view_id != self.expected_view_id:
            raise ValueError("ffmpeg decoder view differs from its bound identity")
        output_bytes = _positive_int(height, name="height") * _positive_int(
            width,
            name="width",
        ) * 3
        if output_bytes > maximum_decoded_frame_bytes:
            raise MemoryError("decoded RGB8 frame exceeds its cap before decode")
        if self._request_cursor >= len(self.selected_logical_frame_indices):
            raise ValueError("ffmpeg decoder received more frames than its selection")
        expected_logical = self.selected_logical_frame_indices[self._request_cursor]
        if source_frame_index != expected_logical:
            raise ValueError("ffmpeg decoder request differs from its frame identity")
        if self._process is None:
            self._open_session(handle, height=height, width=width)
        elif id(handle) != self._handle_identity:
            raise ValueError("ffmpeg decoder handle changed within one conversion")
        elif self._output_shape != (int(height), int(width)):
            raise ValueError("ffmpeg output dimensions changed within one view")

        target_native = self._native_indices[self._request_cursor]
        if target_native == self._last_returned_native_index:
            if self._last_returned_rgb8 is None:
                raise ArithmeticError("duplicate ffmpeg native frame cache is missing")
            result = self._last_returned_rgb8
        else:
            self._last_returned_rgb8 = None
            if self._unique_output_cursor >= len(self._unique_native_indices) or (
                self._unique_native_indices[self._unique_output_cursor] != target_native
            ):
                raise ArithmeticError("ffmpeg selected-frame cursor drifted")
            result = self._read_exact_stdout(output_bytes)
            self._last_returned_native_index = target_native
            self._last_returned_rgb8 = result
            self._unique_output_cursor += 1
            self.decoded_native_frame_count += 1
        self._request_cursor += 1
        return result

    def close_open_file_decode(
        self,
        handle: BinaryIO,
        *,
        view_id: str,
        decode_completed: bool,
    ) -> None:
        wrong_view = view_id != self.expected_view_id
        wrong_handle = self._process is not None and id(handle) != self._handle_identity
        invalid_completion_flag = not isinstance(decode_completed, bool)
        incomplete_success = (
            decode_completed is True
            and self._request_cursor != len(self.selected_logical_frame_indices)
        )
        process = self._process
        close_errors: list[BaseException] = []
        try:
            if process is not None:
                if decode_completed is True and not incomplete_success:
                    try:
                        extra_bytes = self._drain_remaining_stdout()
                        if extra_bytes:
                            close_errors.append(
                                ValueError(
                                    "ffmpeg emitted bytes beyond its exact selected-frame plan"
                                )
                            )
                    except BaseException as error:
                        close_errors.append(error)
                        process.terminate()
                else:
                    process.terminate()
                try:
                    return_code = process.wait(
                        timeout=_FFMPEG_PROCESS_TIMEOUT_SECONDS
                    )
                except subprocess.TimeoutExpired:
                    process.kill()
                    return_code = process.wait(
                        timeout=_FFMPEG_PROCESS_TIMEOUT_SECONDS
                    )
                    if decode_completed is True:
                        close_errors.append(TimeoutError("ffmpeg did not exit on time"))
                for thread in (self._feeder_thread, self._stderr_thread):
                    if thread is not None:
                        thread.join(timeout=_FFMPEG_PROCESS_TIMEOUT_SECONDS)
                        if thread.is_alive() and decode_completed is True:
                            close_errors.append(
                                TimeoutError("ffmpeg I/O thread did not terminate")
                            )
                if decode_completed is True and not incomplete_success:
                    if return_code != 0:
                        close_errors.append(
                            RuntimeError(
                                f"ffmpeg exited with status {return_code}: "
                                f"{self._stderr_detail()}"
                            )
                        )
                    if self._feeder_error is not None:
                        close_errors.append(self._feeder_error)
                    if self._stderr_error is not None:
                        close_errors.append(self._stderr_error)
                    if self._stderr_overflowed:
                        close_errors.append(
                            MemoryError("ffmpeg stderr exceeded its bounded capture")
                        )
                    if self._unique_output_cursor != len(
                        self._unique_native_indices
                    ):
                        close_errors.append(
                            ValueError("ffmpeg did not emit its exact unique-frame plan")
                        )
        finally:
            if process is not None:
                for stream in (process.stdout, process.stderr):
                    if stream is not None:
                        try:
                            stream.close()
                        except OSError:
                            pass
            self._process = None
            self._feeder_thread = None
            self._stderr_thread = None
            self._handle_identity = None
            self._output_shape = None
            self._last_returned_rgb8 = None
        if wrong_view:
            raise ValueError("ffmpeg close view differs from its bound identity")
        if wrong_handle:
            raise ValueError("ffmpeg close handle differs from its decode handle")
        if invalid_completion_flag:
            raise TypeError("ffmpeg close requires an exact decode_completed boolean")
        if incomplete_success:
            raise ValueError("successful ffmpeg close did not consume its frame plan")
        if close_errors:
            raise close_errors[0]


@dataclass(frozen=True)
class Neural3dMappedRgb8AdapterLimits:
    maximum_dataset_manifest_bytes: int
    maximum_poses_bounds_bytes: int
    maximum_adapter_source_bytes: int
    maximum_total_source_verification_bytes: int
    maximum_camera_tensor_bytes: int
    maximum_descriptor_bytes: int
    maximum_mp4_header_read_bytes_per_view: int
    maximum_total_mp4_header_read_bytes: int
    maximum_native_frame_bytes: int
    maximum_python_rgb_scratch_bytes: int
    maximum_decoded_native_frames_per_view: int

    def assert_valid(self) -> None:
        for name, value in self.__dict__.items():
            _positive_int(value, name=name)


@dataclass(frozen=True)
class PreparedNeural3dMappedRgb8Request:
    build_request: cache_builder.WorldFoamMappedRgb8BuildRequest
    descriptor_path: Path
    descriptor_sha256: str
    descriptor: Mapping[str, Any]
    target_split: str
    view_ids: tuple[str, ...]


@dataclass(frozen=True)
class Neural3dMappedRgb8BuildReceipt:
    cache: cache_builder.WorldFoamMappedRgb8BuildReceipt
    descriptor_path: Path
    descriptor_sha256: str
    target_split: str
    view_ids: tuple[str, ...]


def endpoint_including_logical_frame_maps(
    stored_frame_indices: Sequence[int],
    required_frame_counts: Sequence[int],
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    stored = tuple(stored_frame_indices)
    counts = tuple(required_frame_counts)
    if (
        not stored
        or stored != tuple(sorted(set(stored)))
        or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in stored
        )
    ):
        raise ValueError("stored frame indices must be unique, increasing, and nonnegative")
    if (
        not counts
        or counts != tuple(sorted(set(counts)))
        or counts[-1] != len(stored)
    ):
        raise ValueError("required frame counts must be increasing and end at stored count")
    result = []
    for count in counts:
        _positive_int(count, name="required frame count")
        if count > len(stored):
            raise ValueError("required frame count exceeds the stored cache")
        if count == 1:
            if len(stored) != 1:
                raise ValueError("one-frame map cannot preserve two physical endpoints")
            positions = (0,)
        else:
            positions = tuple(
                index * (len(stored) - 1) // (count - 1)
                for index in range(count)
            )
        if len(set(positions)) != count:
            raise ArithmeticError("endpoint-including frame map lost unique coverage")
        result.append((count, tuple(stored[position] for position in positions)))
    return tuple(result)


def _manifest_record(
    raw: bytes,
    *,
    sample_id: str,
) -> Mapping[str, Any]:
    lines = raw.splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ValueError("Neural3D JSONL manifest contains an empty row")
    records = []
    for index, line in enumerate(lines):
        value = _strict_json_loads(line, name=f"dataset manifest row {index}")
        if not isinstance(value, Mapping):
            raise TypeError(f"dataset manifest row {index} must be an object")
        if value.get("sample_id") == sample_id:
            records.append(value)
    if len(records) != 1:
        raise ValueError("Neural3D manifest must contain exactly one selected sample_id")
    return records[0]


def _checked_camera_split(
    record: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    split_views: dict[str, tuple[str, ...]] = {}
    for split in ("train", "heldout"):
        key = f"{split}_cameras"
        raw = record.get(key)
        if not isinstance(raw, list) or not raw:
            raise ValueError(f"Neural3D row requires a nonempty {key} list")
        views = tuple(
            _nonempty_trimmed(value, name=f"{split} camera id") for value in raw
        )
        if views != tuple(sorted(set(views))):
            raise ValueError(
                f"Neural3D {split} cameras must be unique and sorted"
            )
        split_views[split] = views
    overlap = sorted(set(split_views["train"]) & set(split_views["heldout"]))
    if overlap:
        raise ValueError(f"Neural3D train/heldout camera split overlaps: {overlap}")
    anchor_camera = _nonempty_trimmed(
        record.get("anchor_camera"),
        name="anchor camera",
    )
    if anchor_camera not in split_views["train"]:
        raise ValueError("Neural3D anchor camera must be one of the train views")
    return split_views["train"], split_views["heldout"], anchor_camera


def _tensor_identity_or_fail(value: Any, *, name: str) -> dict[str, Any]:
    from paper_training_protocol import tensor_content_identity

    identity = tensor_content_identity(value)
    if not isinstance(identity, dict):
        raise ArithmeticError(f"{name} produced no tensor identity")
    return identity


def _build_camera_binding(
    record: Mapping[str, Any],
    *,
    view_ids: tuple[str, ...],
    train_view_ids: tuple[str, ...],
    heldout_view_ids: tuple[str, ...],
    anchor_camera: str,
    stored_frame_indices: tuple[int, ...],
    height: int,
    width: int,
    translation_scale: float,
    provenance_payload: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    from multicam_video_data import neural_3d_camera_from_poses_bounds
    from sequence_data import normalize_frame_times

    device = torch.device("cpu")
    _, anchor_c2w = neural_3d_camera_from_poses_bounds(
        dict(record),
        anchor_camera,
        H=height,
        W=width,
        device=device,
        translation_scale=translation_scale,
    )
    intrinsics = []
    relative_w2c = []
    for view_id in view_ids:
        K, c2w = neural_3d_camera_from_poses_bounds(
            dict(record),
            view_id,
            H=height,
            W=width,
            device=device,
            translation_scale=translation_scale,
        )
        intrinsics.append(K)
        relative = torch.linalg.inv(c2w) @ anchor_c2w
        relative_w2c.append(
            relative.unsqueeze(0).repeat(len(stored_frame_indices), 1, 1)
        )
    K_grid = torch.stack(intrinsics, dim=0).contiguous()
    w2c_grid = torch.stack(relative_w2c, dim=0).contiguous()
    frame_times = torch.tensor(
        stored_frame_indices,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(-1) / float(record["fps"])
    frame_times = normalize_frame_times(frame_times).contiguous()
    frame_times_identity = _tensor_identity_or_fail(frame_times, name="frame_times")
    K_identity = _tensor_identity_or_fail(K_grid, name="K")
    w2c_identity = _tensor_identity_or_fail(w2c_grid, name="w2c")
    generation_payload = {
        "schema": "dynaworld.neural3d_camera_generation/v2",
        "train_view_ids": list(train_view_ids),
        "heldout_view_ids": list(heldout_view_ids),
        "anchor_camera": anchor_camera,
        "stored_frame_indices": list(stored_frame_indices),
        "height": height,
        "width": width,
        "translation_scale": translation_scale,
        "pose_source": NEURAL3D_POSE_SOURCE,
        "frame_times": frame_times_identity,
        # K/w2c identities remain split-specific fields in the returned
        # camera binding.  The generation digest instead seals the common
        # rig, generator inputs, and physical time grid so train and heldout
        # caches can prove that they were built under one camera convention.
        "conversion_provenance_sha256": _canonical_sha256(provenance_payload),
    }
    return {
        "view_ids": list(view_ids),
        "height": height,
        "width": width,
        "frame_times": frame_times_identity,
        "K": K_identity,
        "w2c": w2c_identity,
        "lens_models": ["pinhole"] * len(view_ids),
        "distortions": None,
        "pose_source": NEURAL3D_POSE_SOURCE,
        "camera_generation_digest": _canonical_sha256(generation_payload),
    }


def _write_descriptor(
    path: Path,
    descriptor: Mapping[str, Any],
    *,
    maximum_bytes: int,
) -> str:
    encoded = _canonical_json_bytes(descriptor)
    if not encoded or len(encoded) > maximum_bytes:
        raise MemoryError("Neural3D conversion descriptor exceeds its byte cap")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.parent / f".{path.name}.temporary-{os.getpid()}"
    if path.exists() or temp_path.exists():
        raise FileExistsError("Neural3D conversion descriptor output already exists")
    published = False
    try:
        with temp_path.open("xb+") as handle:
            if handle.write(encoded) != len(encoded):
                raise OSError("Neural3D conversion descriptor write was short")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp_path, path, follow_symlinks=False)
        except FileExistsError as error:
            raise FileExistsError(
                f"Neural3D descriptor output appeared during preparation: {path}"
            ) from error
        published = True
        temp_path.unlink()
    except BaseException:
        temp_path.unlink(missing_ok=True)
        if published:
            path.unlink(missing_ok=True)
        raise
    return hashlib.sha256(encoded).hexdigest()


def prepare_neural3d_mapped_rgb8_request(
    *,
    repository_root: Path,
    dataset_manifest_path: Path,
    dataset_manifest_path_label: str,
    sample_id: str,
    output_directory: Path,
    height: int,
    width: int,
    stored_frame_indices: Sequence[int],
    required_frame_counts: Sequence[int],
    conversion_limits: cache_builder.WorldFoamMappedRgb8ConversionLimits,
    adapter_limits: Neural3dMappedRgb8AdapterLimits,
    target_split: str = "train",
    translation_scale: float = 1.0,
    descriptor_label: str = "neural3d_conversion_descriptor.json",
    mapped_manifest_label: str = "mapped_rgb8_manifest.json",
    binding_label: str = "target_dataset_binding.json",
) -> PreparedNeural3dMappedRgb8Request:
    """Prepare one strict split-specific request without decoding an MP4."""

    adapter_limits.assert_valid()
    conversion_limits.assert_valid()
    root = repository_root.expanduser().resolve()
    manifest_path = dataset_manifest_path.expanduser().resolve()
    manifest_label = _portable_relative_label(
        dataset_manifest_path_label,
        name="dataset manifest path label",
    )
    if _path_beneath(root, manifest_label, name="dataset manifest path") != manifest_path:
        raise ValueError("dataset manifest path and path label resolve differently")
    selected_sample_id = _nonempty_trimmed(sample_id, name="sample_id")
    selected_split = _target_split(target_split)
    output_height = _positive_int(height, name="height")
    output_width = _positive_int(width, name="width")
    scale = _finite_positive(translation_scale, name="translation_scale")
    descriptor_name = _portable_relative_label(
        descriptor_label,
        name="descriptor label",
    )
    if len(PurePosixPath(descriptor_name).parts) != 1:
        raise ValueError("Neural3D descriptor must share the converter output directory")
    mapped_manifest_name = _portable_relative_label(
        mapped_manifest_label,
        name="mapped manifest label",
    )
    binding_name = _portable_relative_label(binding_label, name="binding label")
    if any(
        len(PurePosixPath(label).parts) != 1
        for label in (mapped_manifest_name, binding_name)
    ):
        raise ValueError("Neural3D converter outputs must share one output directory")
    if len({descriptor_name, mapped_manifest_name, binding_name}) != 3:
        raise ValueError("descriptor, mapped manifest, and binding labels must differ")
    output_root = output_directory.expanduser().resolve()
    if output_root.exists() and not output_root.is_dir():
        raise NotADirectoryError(f"Neural3D output path is not a directory: {output_root}")

    stored = tuple(stored_frame_indices)
    frame_maps = endpoint_including_logical_frame_maps(
        stored,
        required_frame_counts,
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Neural3D dataset manifest is not a file: {manifest_path}")
    raw_manifest = _read_stable_file(
        manifest_path,
        maximum_bytes=adapter_limits.maximum_dataset_manifest_bytes,
        name="Neural3D dataset manifest",
    )
    manifest_size = len(raw_manifest)
    record = _manifest_record(raw_manifest, sample_id=selected_sample_id)
    if record.get("dataset") != "neural_3d_video":
        raise ValueError("selected dataset row is not Neural3D Video")
    dataset_id = _nonempty_trimmed(
        record.get("dataset_name") or selected_sample_id,
        name="dataset_name",
    )
    frame_count = _positive_int(record.get("frame_count"), name="record frame_count")
    if stored[-1] >= frame_count:
        raise IndexError("stored frame selection leaves the Neural3D record")
    sample_fps = _finite_positive(record.get("fps"), name="record fps")
    native_size = record.get("source_image_size")
    if (
        not isinstance(native_size, list)
        or len(native_size) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in native_size
        )
    ):
        raise ValueError("Neural3D row requires exact source_image_size=[height,width]")
    native_height, native_width = native_size
    if native_height * native_width * 3 > adapter_limits.maximum_native_frame_bytes:
        raise MemoryError("Neural3D native RGB frame exceeds its cap before decoder creation")

    train_view_ids, heldout_view_ids, anchor_camera = _checked_camera_split(record)
    view_ids = train_view_ids if selected_split == "train" else heldout_view_ids
    payload_labels = tuple(
        _portable_relative_label(f"{view_id}.rgb8", name=f"payload {view_id}")
        for view_id in view_ids
    )
    all_output_labels = (
        descriptor_name,
        mapped_manifest_name,
        binding_name,
        *payload_labels,
    )
    if (
        any(len(PurePosixPath(label).parts) != 1 for label in all_output_labels)
        or len(set(all_output_labels)) != len(all_output_labels)
    ):
        raise ValueError("Neural3D descriptor, manifest, binding, and payloads must differ")
    for label in all_output_labels:
        path = output_root / label
        if path.exists():
            raise FileExistsError(f"Neural3D cache output already exists: {path}")
    scene_path, scene_label = _path_and_label_beneath(
        root,
        record.get("dataset_scene_dir"),
        name="dataset_scene_dir",
    )
    if not scene_path.is_dir():
        raise FileNotFoundError(f"Neural3D scene directory is missing: {scene_path}")
    poses_label = f"{scene_label}/poses_bounds.npy"
    poses_path = _path_beneath(root, poses_label, name="poses_bounds path")
    if not poses_path.is_file():
        raise FileNotFoundError(f"Neural3D poses file is missing: {poses_path}")
    poses_size = int(poses_path.stat().st_size)
    if poses_size > adapter_limits.maximum_poses_bounds_bytes:
        raise MemoryError("Neural3D poses file exceeds its cap before scan")

    source_camera = _nonempty_trimmed(
        record.get("source_camera"),
        name="source camera",
    )
    source_video_path, source_video_label = _path_and_label_beneath(
        root,
        record.get("source_video_path"),
        name="source_video_path",
    )
    source_start_seconds = _finite_nonnegative(
        record.get("source_start_seconds", 0.0),
        name="source_start_seconds",
    )
    target_camera_raw = record.get("target_camera")
    target_camera = (
        _nonempty_trimmed(target_camera_raw, name="target camera")
        if target_camera_raw is not None
        else None
    )
    target_video_path: Path | None = None
    target_video_label: str | None = None
    target_start_seconds = 0.0
    if (
        target_camera is not None
        and target_camera in view_ids
        and target_camera != source_camera
    ):
        target_video_path, target_video_label = _path_and_label_beneath(
            root,
            record.get("target_video_path"),
            name="target_video_path",
        )
        target_start_seconds = _finite_nonnegative(
            record.get("target_start_seconds", 0.0),
            name="target_start_seconds",
        )
    raw_video_labels = []
    raw_video_paths = []
    starts = []
    for view_id in view_ids:
        if view_id == source_camera:
            label = source_video_label
            path = source_video_path
            start_seconds = source_start_seconds
        elif view_id == target_camera:
            if target_video_path is None or target_video_label is None:
                raise ArithmeticError("Neural3D target video identity was not resolved")
            label = target_video_label
            path = target_video_path
            start_seconds = target_start_seconds
        else:
            label = _portable_relative_label(
                f"{scene_label}/{view_id}.mp4",
                name=f"raw video {view_id}",
            )
            path = _path_beneath(root, label, name=f"raw video {view_id}")
            # This mirrors multicam_video_data.camera_start_seconds for an
            # arbitrary Neural3D rig camera: only the explicitly named target
            # has its own target_start_seconds; other rig views share source.
            start_seconds = source_start_seconds
        if not path.is_file():
            raise FileNotFoundError(
                f"Neural3D {selected_split} video is missing: {path}"
            )
        size = int(path.stat().st_size)
        if (
            size < 1
            or size > conversion_limits.maximum_raw_input_bytes_per_view
        ):
            raise MemoryError(f"Neural3D MP4 {view_id!r} exceeds its input cap")
        raw_video_labels.append(label)
        raw_video_paths.append(path)
        starts.append(start_seconds)
    if (
        len(view_ids) * adapter_limits.maximum_mp4_header_read_bytes_per_view
        > adapter_limits.maximum_total_mp4_header_read_bytes
    ):
        raise MemoryError("Neural3D MP4 header probes exceed their total cap")

    camera_tensor_bytes = (
        len(stored)
        + len(view_ids) * 9
        + len(view_ids) * len(stored) * 16
    ) * 4
    if camera_tensor_bytes > adapter_limits.maximum_camera_tensor_bytes:
        raise MemoryError("Neural3D camera tensors exceed their cap before allocation")

    adapter_source_path = Path(__file__).resolve()
    converter_source_path = Path(cache_builder.__file__).resolve()
    source_sizes = (
        int(adapter_source_path.stat().st_size),
        int(converter_source_path.stat().st_size),
    )
    if any(
        size > adapter_limits.maximum_adapter_source_bytes
        for size in source_sizes
    ):
        raise MemoryError("Neural3D adapter/converter source exceeds its hash cap")
    verification_bytes = (
        2 * manifest_size
        + 2 * poses_size
        + sum(source_sizes)
    )
    if verification_bytes > adapter_limits.maximum_total_source_verification_bytes:
        raise MemoryError("Neural3D adapter source verification exceeds its total cap")

    manifest_identity = {
        "path_label": manifest_label,
        "size_bytes": manifest_size,
        "sha256": hashlib.sha256(raw_manifest).hexdigest(),
    }
    poses_identity = _file_identity(
        poses_path,
        path_label=poses_label,
        maximum_bytes=adapter_limits.maximum_poses_bounds_bytes,
        name="Neural3D poses_bounds",
    )
    adapter_source_identity = _file_identity(
        adapter_source_path,
        path_label="research_experiments/world_foam_lane2/neural3d_mapped_rgb8_adapter.py",
        maximum_bytes=adapter_limits.maximum_adapter_source_bytes,
        name="Neural3D adapter source",
    )
    converter_source_identity = _file_identity(
        converter_source_path,
        path_label="research_experiments/world_foam_lane2/build_worldfoam_mapped_rgb8_cache.py",
        maximum_bytes=adapter_limits.maximum_adapter_source_bytes,
        name="mapped RGB8 converter source",
    )
    runtime = _selected_runtime_version_payload()
    mp4_headers = {}
    total_header_read_bytes = 0
    for view_id, path in zip(view_ids, raw_video_paths, strict=True):
        header, read_bytes = _probe_mp4_header_for_runtime(
            path,
            maximum_read_bytes=(
                adapter_limits.maximum_mp4_header_read_bytes_per_view
            ),
            runtime=runtime,
        )
        total_header_read_bytes += read_bytes
        if total_header_read_bytes > adapter_limits.maximum_total_mp4_header_read_bytes:
            raise MemoryError("Neural3D MP4 header probes exceeded their total cap")
        if (
            header["native_height"] != native_height
            or header["native_width"] != native_width
        ):
            raise ValueError(
                f"Neural3D MP4 {view_id!r} dimensions differ from source_image_size"
            )
        mp4_headers[view_id] = header
    provenance_payload = {
        "schema": "dynaworld.neural3d_mapped_rgb8_decoder_provenance/v1",
        "adapter_source": adapter_source_identity,
        "converter_source": converter_source_identity,
        "runtime": runtime,
        "mp4_headers_by_view": mp4_headers,
        "total_mp4_header_read_bytes": total_header_read_bytes,
        "resource_contract": _decoder_resource_contract(
            runtime,
            adapter_limits=adapter_limits,
            conversion_limits=conversion_limits,
        ),
    }
    camera_generation_provenance = {
        "schema": "dynaworld.neural3d_camera_generation_provenance/v1",
        "dataset_manifest": manifest_identity,
        "dataset_record_sha256": _canonical_sha256(dict(record)),
        "poses_bounds": poses_identity,
        "adapter_source": adapter_source_identity,
        "converter_source": converter_source_identity,
        "runtime_sha256": runtime["runtime_sha256"],
    }
    camera_record = dict(record)
    camera_record["dataset_scene_dir"] = str(scene_path)
    camera = _build_camera_binding(
        camera_record,
        view_ids=view_ids,
        train_view_ids=train_view_ids,
        heldout_view_ids=heldout_view_ids,
        anchor_camera=anchor_camera,
        stored_frame_indices=stored,
        height=output_height,
        width=output_width,
        translation_scale=scale,
        provenance_payload=camera_generation_provenance,
    )
    repeated_manifest_identity = _file_identity(
        manifest_path,
        path_label=manifest_label,
        maximum_bytes=adapter_limits.maximum_dataset_manifest_bytes,
        name="Neural3D dataset manifest",
    )
    repeated_poses_identity = _file_identity(
        poses_path,
        path_label=poses_label,
        maximum_bytes=adapter_limits.maximum_poses_bounds_bytes,
        name="Neural3D poses_bounds",
    )
    if (
        repeated_manifest_identity != manifest_identity
        or repeated_poses_identity != poses_identity
    ):
        raise ValueError("Neural3D manifest or poses changed during camera generation")

    native_maps = {}
    for index, view_id in enumerate(view_ids):
        header = mp4_headers[view_id]
        native_fps = (
            header["native_fps_numerator"]
            / header["native_fps_denominator"]
        )
        native_indices = [
            int(round((starts[index] + frame / sample_fps) * native_fps))
            for frame in stored
        ]
        if native_indices != sorted(native_indices):
            raise ValueError(
                f"Neural3D native frame map is not monotone for {view_id!r}"
            )
        if (
            native_indices[0] < 0
            or native_indices[-1] >= header["native_frame_count"]
        ):
            raise IndexError(
                f"Neural3D selected native frame leaves MP4 {view_id!r}"
            )
        native_maps[view_id] = native_indices

    views_list = []
    for index, (view_id, path, label, payload_label) in enumerate(
        zip(
            view_ids,
            raw_video_paths,
            raw_video_labels,
            payload_labels,
            strict=True,
        )
    ):
        header = mp4_headers[view_id]
        decoder_class = (
            PyAvOpenFileRgb8Decoder
            if runtime["decoder_provenance"] == PYAV_DECODER_PROVENANCE
            else FfmpegPipeRgb8Decoder
        )
        decoder = decoder_class(
            expected_view_id=view_id,
            selected_logical_frame_indices=stored,
            source_frame_count=frame_count,
            start_seconds=starts[index],
            sample_fps=sample_fps,
            expected_native_fps_numerator=header["native_fps_numerator"],
            expected_native_fps_denominator=header["native_fps_denominator"],
            expected_native_frame_count=header["native_frame_count"],
            expected_native_height=native_height,
            expected_native_width=native_width,
            expected_stream_time_base_numerator=(
                header["stream_time_base_numerator"]
            ),
            expected_stream_time_base_denominator=(
                header["stream_time_base_denominator"]
            ),
            expected_stream_start_time=header["stream_start_time"],
            maximum_native_frame_bytes=adapter_limits.maximum_native_frame_bytes,
            maximum_python_rgb_scratch_bytes=(
                adapter_limits.maximum_python_rgb_scratch_bytes
            ),
            maximum_decoded_native_frames=(
                adapter_limits.maximum_decoded_native_frames_per_view
            ),
            expected_runtime_sha256=runtime["runtime_sha256"],
        )
        if [decoder.logical_to_native_index(frame) for frame in stored] != native_maps[
            view_id
        ]:
            raise ArithmeticError("Neural3D descriptor and decoder frame maps differ")
        views_list.append(
            cache_builder.WorldFoamRawTargetView(
                view_id=view_id,
                raw_input_path=path,
                raw_input_path_label=label,
                payload_label=payload_label,
                decoder=decoder,
            )
        )
    views = tuple(views_list)

    descriptor: dict[str, Any] = {
        "schema": NEURAL3D_DESCRIPTOR_SCHEMA,
        "dataset": {
            "dataset_id": dataset_id,
            "sample_id": selected_sample_id,
            "scene": _nonempty_trimmed(record.get("scene"), name="scene"),
            "frame_count": frame_count,
            "sample_fps": sample_fps,
            "native_height": native_height,
            "native_width": native_width,
            "anchor_camera": anchor_camera,
            "translation_scale": scale,
        },
        "dataset_manifest": manifest_identity,
        "dataset_record_sha256": _canonical_sha256(dict(record)),
        "poses_bounds": poses_identity,
        "target_split": selected_split,
        "view_ids": list(view_ids),
        "raw_video_path_labels": dict(zip(view_ids, raw_video_labels, strict=True)),
        "output_height": output_height,
        "output_width": output_width,
        "stored_logical_frame_indices": list(stored),
        "logical_frame_maps": [
            {"frame_count": count, "source_frame_indices": list(indices)}
            for count, indices in frame_maps
        ],
        "native_frame_indices_by_view": native_maps,
        "decoder": provenance_payload,
        "camera_generation_digest": camera["camera_generation_digest"],
        "descriptor_sha256": "",
    }
    if set(descriptor) != _DESCRIPTOR_KEYS:
        raise ArithmeticError("Neural3D descriptor keys changed")
    descriptor["descriptor_sha256"] = _canonical_sha256(
        {key: value for key, value in descriptor.items() if key != "descriptor_sha256"}
    )
    descriptor_path = output_root / descriptor_name
    descriptor_file_sha256 = _write_descriptor(
        descriptor_path,
        descriptor,
        maximum_bytes=min(
            adapter_limits.maximum_descriptor_bytes,
            conversion_limits.maximum_raw_dataset_manifest_bytes,
        ),
    )

    build_request = cache_builder.WorldFoamMappedRgb8BuildRequest(
        output_directory=output_root,
        dataset_id=dataset_id,
        target_split=selected_split,
        raw_dataset_manifest_path=descriptor_path,
        raw_dataset_manifest_path_label=descriptor_name,
        views=views,
        height=output_height,
        width=output_width,
        stored_frame_indices=stored,
        required_frame_counts=tuple(required_frame_counts),
        logical_frame_maps=frame_maps,
        camera=camera,
        limits=conversion_limits,
        mapped_manifest_label=mapped_manifest_name,
        binding_label=binding_name,
    )
    return PreparedNeural3dMappedRgb8Request(
        build_request=build_request,
        descriptor_path=descriptor_path,
        descriptor_sha256=descriptor_file_sha256,
        descriptor=descriptor,
        target_split=selected_split,
        view_ids=view_ids,
    )


def build_neural3d_mapped_rgb8_cache(
    **prepare_kwargs: Any,
) -> Neural3dMappedRgb8BuildReceipt:
    """Prepare provenance, run the converter, and clean the descriptor on failure."""

    prepared = prepare_neural3d_mapped_rgb8_request(**prepare_kwargs)
    try:
        receipt = cache_builder.build_worldfoam_mapped_rgb8_cache(
            prepared.build_request
        )
    except BaseException:
        prepared.descriptor_path.unlink(missing_ok=True)
        raise
    return Neural3dMappedRgb8BuildReceipt(
        cache=receipt,
        descriptor_path=prepared.descriptor_path,
        descriptor_sha256=prepared.descriptor_sha256,
        target_split=prepared.target_split,
        view_ids=prepared.view_ids,
    )


__all__ = [
    "FFMPEG_DECODER_PROVENANCE",
    "NEURAL3D_DESCRIPTOR_SCHEMA",
    "NEURAL3D_OFFLINE_PREFLIGHT_SCHEMA",
    "NEURAL3D_POSE_SOURCE",
    "PYAV_DECODER_PROVENANCE",
    "FfmpegPipeRgb8Decoder",
    "Neural3dMappedRgb8AdapterLimits",
    "Neural3dMappedRgb8BuildReceipt",
    "PreparedNeural3dMappedRgb8Request",
    "PyAvOpenFileRgb8Decoder",
    "build_neural3d_mapped_rgb8_cache",
    "endpoint_including_logical_frame_maps",
    "neural3d_mapped_rgb8_offline_preflight",
    "prepare_neural3d_mapped_rgb8_request",
]
