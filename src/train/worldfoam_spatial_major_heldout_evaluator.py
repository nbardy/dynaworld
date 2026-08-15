"""Spatial-major, full-temporal heldout evaluation for WorldFoam.

The public RGB8 cache is physically pixel-time major.  A frame-major evaluator
therefore causes tens of GiB of page touches and, for WorldFoam, recompiles the
same retained-depth track at every time.  This module reads and compiles bounded
pixel-track blocks across all frames, writes two explicit F_NOCACHE-compatible
raw spools (prediction float32 and target RGB8), then reconstructs one CPU frame
at a time for the canonical PSNR/SSIM/LPIPS/L1 and media operations.

The spools themselves never use mmap (the existing pixel-time source may open
one transient, receipt-accounted mapping per bounded selected read), no full
video is resident on the device, no target rays are constructed, and both
temporary spools are removed on success or failure.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from worldfoam_native4d_public_quality_row import (
    HeldoutEvaluationReceipt,
    PixelChunkRequest,
    REQUIRED_METRICS,
)


REPLAY_KIND = "worldfoam-spatial-major-heldout-evaluation-v1"
SESSION_REPLAY_KIND = "worldfoam-spatial-major-full-temporal-heldout-v1"
TARGET_READ_KIND = "worldfoam-spatial-major-target-track-read-v1"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _update_tensor_digest(digest: Any, value: Any) -> None:
    import torch

    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or not value.is_contiguous()
    ):
        raise TypeError("heldout digest requires a contiguous CPU tensor")
    digest.update(memoryview(value.numpy()).cast("B"))


def _pwrite_all(descriptor: int, data: memoryview, offset: int) -> None:
    remaining = data
    cursor = int(offset)
    while remaining:
        written = os.pwrite(descriptor, remaining, cursor)
        if written < 1:
            raise OSError("WorldFoam heldout spool write made no progress")
        remaining = remaining[written:]
        cursor += written


def _pread_exact(descriptor: int, byte_count: int, offset: int) -> bytes:
    chunks: list[bytes] = []
    remaining = int(byte_count)
    cursor = int(offset)
    while remaining:
        chunk = os.pread(descriptor, remaining, cursor)
        if not chunk:
            raise EOFError("WorldFoam heldout spool ended before its RGB payload")
        chunks.append(chunk)
        remaining -= len(chunk)
        cursor += len(chunk)
    return b"".join(chunks)


def _descriptor_sha256(descriptor: int, byte_count: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while offset < byte_count:
        chunk = _pread_exact(
            descriptor,
            min(1024 * 1024, byte_count - offset),
            offset,
        )
        digest.update(chunk)
        offset += len(chunk)
    return digest.hexdigest()


def _request_darwin_no_cache(descriptor: int) -> bool:
    if sys.platform != "darwin":
        return False
    try:
        import fcntl

        command = getattr(fcntl, "F_NOCACHE", None)
        if command is None:
            return False
        fcntl.fcntl(descriptor, command, 1)
        return True
    except (ImportError, OSError):
        return False


def _exact_rgb8(value: Any) -> Any:
    """Invert the canonical RGB8 float decoder and prove bitwise round-trip."""

    import numpy as np
    import torch

    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.float32
        or value.ndim != 3
        or value.shape[-1] != 3
        or not value.is_contiguous()
        or not bool(torch.isfinite(value).all().item())
        or float(value.min().item()) < 0.0
        or float(value.max().item()) > 1.0
    ):
        raise ValueError("WorldFoam target track block is not canonical CPU RGB")
    rgb8 = value.mul(255.0).round().to(torch.uint8).contiguous()
    decoded_np = np.asarray(rgb8.numpy(), dtype=np.float32)
    decoded_np /= 255.0
    decoded = torch.from_numpy(np.ascontiguousarray(decoded_np))
    if not torch.equal(decoded, value):
        raise ValueError("WorldFoam public target is not lossless RGB8 cache content")
    return rgb8


@dataclass(frozen=True)
class SpatialMajorHeldoutEvaluationResult:
    evaluation: HeldoutEvaluationReceipt
    spatial_replay_receipt: Mapping[str, Any]


def _require_spatial_replay_session(
    session: Any,
) -> tuple[Callable[..., Any], Callable[..., Any], int, Callable[[], Any]]:
    render = getattr(session, "render_heldout_track_block_across_frames", None)
    read_target = getattr(
        session,
        "read_heldout_target_track_block_across_frames",
        None,
    )
    block_limit = getattr(
        session,
        "maximum_heldout_tracks_per_cross_time_block",
        None,
    )
    receipt = getattr(session, "heldout_spatial_major_receipt", None)
    if not all(callable(value) for value in (render, read_target, block_limit, receipt)):
        raise TypeError(
            "WorldFoam G4-v2 heldout evaluation requires the sealed "
            "spatial-major prediction/target replay session"
        )
    limit = block_limit()
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("WorldFoam spatial-major session returned an invalid track bound")
    return render, read_target, limit, receipt


def _validate_target_read_receipt(
    value: Any,
    *,
    camera_index: int,
    pixel_ids: tuple[int, ...],
    frames: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("WorldFoam target-track read receipt is missing")
    receipt = dict(value)
    exact = {
        "schema_version": 1,
        "kind": TARGET_READ_KIND,
        "camera_index": camera_index,
        "pixel_ids": pixel_ids,
        "pixel_ids_sha256": _canonical_sha256(pixel_ids),
        "track_count": len(pixel_ids),
        "frame_count": frames,
        "observation_count": len(pixel_ids) * frames,
        "selection_mode": "direct_pixels",
        "mapping_closed_before_return": True,
        "full_frame_materialization_count": 0,
        "ray_tensor_bytes": 0,
    }
    for key, expected in exact.items():
        if receipt.get(key) != expected:
            raise ValueError(f"WorldFoam target-track receipt changed: {key}")
    for key in (
        "source_only_visible_peak_logical_tensor_bytes_upper_bound",
        "returned_target_tensor_bytes",
        "source_plus_returned_target_peak_logical_tensor_bytes_upper_bound",
        "transient_mapped_address_space_bytes",
        "requested_unique_mapped_page_count",
        "requested_mapped_page_bytes_upper_bound",
    ):
        if (
            isinstance(receipt.get(key), bool)
            or not isinstance(receipt.get(key), int)
            or receipt[key] < 0
        ):
            raise ValueError(f"WorldFoam target-track receipt count is invalid: {key}")
    if (
        receipt["returned_target_tensor_bytes"] != len(pixel_ids) * frames * 3 * 4
        or receipt["source_plus_returned_target_peak_logical_tensor_bytes_upper_bound"]
        != receipt["source_only_visible_peak_logical_tensor_bytes_upper_bound"]
        + receipt["returned_target_tensor_bytes"]
    ):
        raise ValueError("WorldFoam target-track peak accounting changed")
    if (
        not isinstance(receipt.get("source_provenance"), str)
        or not receipt["source_provenance"].strip()
        or not _is_sha256(receipt.get("generation_digest"))
        or receipt["generation_digest"]
        != _canonical_sha256(
            {key: item for key, item in receipt.items() if key != "generation_digest"}
        )
    ):
        raise ValueError("WorldFoam target-track receipt provenance changed")
    return receipt


def _validate_session_receipt(
    value: Any,
    *,
    cameras: int,
    frames: int,
    height: int,
    width: int,
    track_block_limit: int,
    render_calls: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("WorldFoam spatial-major session receipt is missing")
    receipt = dict(value)
    pixels = height * width
    target_pixels = cameras * frames * pixels
    exact = {
        "schema_version": 1,
        "kind": SESSION_REPLAY_KIND,
        "camera_count": cameras,
        "frame_count": frames,
        "image_height": height,
        "image_width": width,
        "cross_time_track_block_size": track_block_limit,
        "render_call_count": render_calls,
        "cold_track_compile_count": cameras * pixels,
        "complete_camera_record_validation_count": target_pixels,
        "native_sample_count": target_pixels,
        "native_prediction_target_observation_read_count": target_pixels,
        "spatial_target_staging_call_count": render_calls,
        "spatial_target_staging_observation_count": target_pixels,
        "target_ray_tensor_bytes": 0,
        "full_pixel_full_temporal": True,
        "frame_major_recompile_per_time_used": False,
        "prediction_spool_dtype": "float32",
    }
    for key, expected in exact.items():
        if receipt.get(key) != expected:
            raise ValueError(f"WorldFoam spatial-major session receipt changed: {key}")
    for key in (
        "admitted_site_reference_upper_bound",
        "native_bundle_count",
        "native_tracks_per_bundle_limit",
        "expected_native_bundle_count",
        "spatial_target_staging_peak_logical_bytes",
    ):
        if (
            isinstance(receipt.get(key), bool)
            or not isinstance(receipt.get(key), int)
            or receipt[key] < 0
        ):
            raise ValueError(f"WorldFoam spatial-major session count is invalid: {key}")
    for key in (
        "prediction_receipt_chain_sha256",
        "target_receipt_chain_sha256",
        "generation_digest",
    ):
        if not _is_sha256(receipt.get(key)):
            raise ValueError(f"WorldFoam spatial-major session digest is invalid: {key}")
    if (
        receipt["native_bundle_count"] != receipt["expected_native_bundle_count"]
        or receipt["native_bundle_count"] < render_calls
        or receipt["generation_digest"]
        != _canonical_sha256(
            {key: item for key, item in receipt.items() if key != "generation_digest"}
        )
    ):
        raise ValueError("WorldFoam spatial-major session coverage is not exact")
    return receipt


def validate_spatial_replay_receipt(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("spatial replay receipt must be a mapping")
    receipt = dict(value)
    required = {
        "schema_version",
        "kind",
        "camera_count",
        "frame_count",
        "image_height",
        "image_width",
        "target_pixel_count",
        "rgb_scalar_count",
        "spatial_track_count",
        "spatial_track_block_limit",
        "spatial_track_block_count",
        "maximum_observations_per_spatial_call",
        "write_superblock_track_limit",
        "write_superblock_count",
        "peak_buffered_prediction_and_target_bytes",
        "prediction_spool_bytes",
        "target_spool_bytes",
        "total_spool_bytes",
        "prediction_spool_dtype",
        "target_spool_dtype",
        "spool_shape",
        "prediction_spool_darwin_f_nocache",
        "target_spool_darwin_f_nocache",
        "spools_cleaned_before_return",
        "dense_device_video_used",
        "persistent_device_video_bytes",
        "target_ray_tensor_bytes",
        "metric_pixel_chunk_limit",
        "metric_pixel_chunk_count",
        "lpips_evaluation_count",
        "media_frame_count",
        "metric_and_media_order",
        "metric_target_spool_observation_read_count",
        "native_prediction_target_source_observation_read_count",
        "target_spool_source_observation_read_count",
        "total_target_source_observation_read_count",
        "total_target_observation_traversal_count",
        "forward_only_prediction_native_op_used",
        "heldout_wall_time_target_io_matched_across_routes",
        "track_request_manifest_sha256",
        "prediction_block_content_sha256",
        "target_block_rgb8_content_sha256",
        "prediction_spool_file_sha256",
        "target_spool_file_sha256",
        "target_read_receipt_manifest_sha256",
        "spool_read_request_manifest_sha256",
        "prediction_spool_read_content_sha256",
        "target_spool_read_content_sha256",
        "target_source_frame_sha256s",
        "target_spool_frame_read_sha256s",
        "target_source_to_spool_frame_hashes_equal",
        "metrics_sha256",
        "heldout_coverage_sha256",
        "session_receipt",
        "session_receipt_generation_digest",
        "exact_rgb8_roundtrip_verified",
        "exact_full_pixel_full_temporal_coverage",
        "one_cold_compile_per_view_pixel_track",
        "generation_digest",
    }
    if set(receipt) != required:
        raise ValueError("spatial replay receipt key set changed")
    if (
        receipt["schema_version"] != 1
        or receipt["kind"] != REPLAY_KIND
        or receipt["prediction_spool_dtype"] != "float32"
        or receipt["target_spool_dtype"] != "uint8"
        or not isinstance(receipt["prediction_spool_darwin_f_nocache"], bool)
        or not isinstance(receipt["target_spool_darwin_f_nocache"], bool)
        or receipt["spools_cleaned_before_return"] is not True
        or receipt["dense_device_video_used"] is not False
        or receipt["persistent_device_video_bytes"] != 0
        or receipt["target_ray_tensor_bytes"] != 0
        or receipt["metric_and_media_order"]
        != "camera_major_then_frame_then_ascending_pixel_chunks"
        or receipt["forward_only_prediction_native_op_used"] is not False
        or receipt["heldout_wall_time_target_io_matched_across_routes"] is not False
        or receipt["exact_rgb8_roundtrip_verified"] is not True
        or receipt["target_source_to_spool_frame_hashes_equal"] is not True
        or receipt["exact_full_pixel_full_temporal_coverage"] is not True
        or receipt["one_cold_compile_per_view_pixel_track"] is not True
    ):
        raise ValueError("spatial replay receipt semantic contract changed")
    integer_fields = tuple(
        key
        for key in required
        if key.endswith("_count")
        or key.endswith("_bytes")
        or key in {
            "camera_count",
            "frame_count",
            "image_height",
            "image_width",
            "target_pixel_count",
            "rgb_scalar_count",
            "spatial_track_block_limit",
            "maximum_observations_per_spatial_call",
            "write_superblock_track_limit",
            "metric_pixel_chunk_limit",
        }
    )
    if any(
        isinstance(receipt[key], bool)
        or not isinstance(receipt[key], int)
        or receipt[key] < 0
        for key in integer_fields
    ):
        raise ValueError("spatial replay receipt count is invalid")
    cameras = receipt["camera_count"]
    frames = receipt["frame_count"]
    pixels = receipt["image_height"] * receipt["image_width"]
    target_pixels = cameras * frames * pixels
    if (
        min(cameras, frames, pixels, receipt["spatial_track_block_limit"]) < 1
        or receipt["target_pixel_count"] != target_pixels
        or receipt["rgb_scalar_count"] != target_pixels * 3
        or receipt["spatial_track_count"] != cameras * pixels
        or receipt["maximum_observations_per_spatial_call"]
        != frames * receipt["spatial_track_block_limit"]
        or receipt["prediction_spool_bytes"] != target_pixels * 3 * 4
        or receipt["target_spool_bytes"] != target_pixels * 3
        or receipt["total_spool_bytes"]
        != receipt["prediction_spool_bytes"] + receipt["target_spool_bytes"]
        or receipt["spool_shape"] != [cameras, frames, pixels, 3]
        or receipt["lpips_evaluation_count"] != cameras * frames
        or receipt["media_frame_count"] != cameras * frames
        or receipt["metric_target_spool_observation_read_count"] != target_pixels
        or receipt["native_prediction_target_source_observation_read_count"]
        != target_pixels
        or receipt["target_spool_source_observation_read_count"] != target_pixels
        or receipt["total_target_source_observation_read_count"] != 2 * target_pixels
        or receipt["total_target_observation_traversal_count"] != 3 * target_pixels
        or receipt["spool_read_request_manifest_sha256"]
        != receipt["heldout_coverage_sha256"]
        or receipt["prediction_spool_file_sha256"]
        != receipt["prediction_spool_read_content_sha256"]
        or receipt["target_spool_file_sha256"]
        != receipt["target_spool_read_content_sha256"]
    ):
        raise ValueError("spatial replay receipt exact coverage changed")
    for key in (
        "track_request_manifest_sha256",
        "prediction_block_content_sha256",
        "target_block_rgb8_content_sha256",
        "prediction_spool_file_sha256",
        "target_spool_file_sha256",
        "target_read_receipt_manifest_sha256",
        "spool_read_request_manifest_sha256",
        "prediction_spool_read_content_sha256",
        "target_spool_read_content_sha256",
        "metrics_sha256",
        "heldout_coverage_sha256",
        "session_receipt_generation_digest",
        "generation_digest",
    ):
        if not _is_sha256(receipt[key]):
            raise ValueError(f"spatial replay receipt digest is invalid: {key}")
    session_receipt = receipt["session_receipt"]
    source_frame_hashes = receipt["target_source_frame_sha256s"]
    spool_frame_hashes = receipt["target_spool_frame_read_sha256s"]
    if (
        not isinstance(source_frame_hashes, list)
        or not isinstance(spool_frame_hashes, list)
        or len(source_frame_hashes) != cameras * frames
        or source_frame_hashes != spool_frame_hashes
        or any(not _is_sha256(item) for item in source_frame_hashes)
        or not isinstance(session_receipt, Mapping)
        or session_receipt.get("generation_digest")
        != receipt["session_receipt_generation_digest"]
        or receipt["generation_digest"]
        != _canonical_sha256(
            {key: item for key, item in receipt.items() if key != "generation_digest"}
        )
    ):
        raise ValueError("spatial replay receipt generation changed")
    return receipt


def evaluate_worldfoam_spatial_major_final_checkpoint(
    context: Any,
    *,
    session: Any,
    media_sink: Any,
    maximum_render_call_count: int,
    spool_directory: Path | None = None,
    minimum_free_bytes_after_spool: int = 512 * 1024 * 1024,
    write_superblock_track_limit: int = 1024,
    lpips_metric: Callable[[Any, Any], float] | None = None,
) -> SpatialMajorHeldoutEvaluationResult:
    """Run exact full-temporal evaluation in bounded spatial-major order."""

    import numpy as np
    import torch
    from paper_training_protocol import PaperRGBMetricAccumulator
    from perceptual_metrics import video_lpips

    render, read_target, track_block_limit, session_receipt_fn = (
        _require_spatial_replay_session(session)
    )
    for name, value in (
        ("maximum_render_call_count", maximum_render_call_count),
        ("minimum_free_bytes_after_spool", minimum_free_bytes_after_spool),
        ("write_superblock_track_limit", write_superblock_track_limit),
    ):
        minimum = 0 if name == "minimum_free_bytes_after_spool" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"WorldFoam heldout {name} is invalid")
    if write_superblock_track_limit < track_block_limit:
        raise ValueError("write superblock cannot be smaller than one track block")

    protocol = context.protocol
    frames = int(protocol.dataset.frame_count)
    cameras = len(protocol.dataset.heldout_cameras)
    height = int(protocol.final_stage.image_size.height)
    width = int(protocol.final_stage.image_size.width)
    pixels = height * width
    if min(cameras, frames, height, width) < 1:
        raise ValueError("WorldFoam heldout evaluation grid is empty")
    render_calls = cameras * math.ceil(pixels / track_block_limit)
    if render_calls > maximum_render_call_count:
        raise MemoryError(
            "WorldFoam spatial-major heldout traversal exceeds its frozen call bound"
        )
    metric_chunk_limit = int(
        getattr(
            context.work_plan,
            "heldout_maximum_pixels_per_chunk",
            context.work_plan.maximum_pixels_per_chunk,
        )
    )
    if metric_chunk_limit < 1:
        raise ValueError("heldout metric chunk bound must be positive")

    spool_shape = (cameras, frames, pixels, 3)
    target_pixels = cameras * frames * pixels
    prediction_spool_bytes = target_pixels * 3 * 4
    target_spool_bytes = target_pixels * 3
    total_spool_bytes = prediction_spool_bytes + target_spool_bytes
    directory = Path(spool_directory or tempfile.gettempdir()).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(directory).free < total_spool_bytes + minimum_free_bytes_after_spool:
        raise OSError("insufficient free space for bounded WorldFoam heldout spools")

    prediction_fd: int | None = None
    target_fd: int | None = None
    prediction_path: Path | None = None
    target_path: Path | None = None
    media_finished = False
    request_manifest = hashlib.sha256()
    prediction_write_content = hashlib.sha256()
    target_write_content = hashlib.sha256()
    target_receipt_manifest = hashlib.sha256()
    prediction_read_content = hashlib.sha256()
    target_read_content = hashlib.sha256()
    coverage = hashlib.sha256()
    target_source_frame_digests = [
        hashlib.sha256() for _index in range(cameras * frames)
    ]
    target_spool_frame_read_digests = [
        hashlib.sha256() for _index in range(cameras * frames)
    ]
    spatial_call_count = 0
    write_superblock_count = 0
    peak_buffered_bytes = 0
    prediction_no_cache = False
    target_no_cache = False
    try:
        prediction_fd, prediction_raw_path = tempfile.mkstemp(
            prefix=".worldfoam-heldout-prediction-",
            suffix=".f32.raw",
            dir=directory,
        )
        target_fd, target_raw_path = tempfile.mkstemp(
            prefix=".worldfoam-heldout-target-",
            suffix=".rgb8.raw",
            dir=directory,
        )
        prediction_path = Path(prediction_raw_path)
        target_path = Path(target_raw_path)
        os.ftruncate(prediction_fd, prediction_spool_bytes)
        os.ftruncate(target_fd, target_spool_bytes)
        prediction_no_cache = _request_darwin_no_cache(prediction_fd)
        target_no_cache = _request_darwin_no_cache(target_fd)

        for camera_index in range(cameras):
            for super_start in range(0, pixels, write_superblock_track_limit):
                super_stop = min(super_start + write_superblock_track_limit, pixels)
                buffered: list[tuple[Any, Any]] = []
                buffered_bytes = 0
                for pixel_start in range(super_start, super_stop, track_block_limit):
                    pixel_ids = tuple(
                        range(
                            pixel_start,
                            min(pixel_start + track_block_limit, super_stop),
                        )
                    )
                    request_manifest.update(
                        _canonical_bytes(
                            {
                                "camera_index": camera_index,
                                "pixel_start": pixel_start,
                                "pixel_count": len(pixel_ids),
                                "pixel_ids_sha256": _canonical_sha256(pixel_ids),
                                "frame_count": frames,
                            }
                        )
                    )
                    request_manifest.update(b"\n")
                    prediction = render(
                        camera_index=camera_index,
                        pixel_ids=pixel_ids,
                    )
                    target_result = read_target(
                        camera_index=camera_index,
                        pixel_ids=pixel_ids,
                    )
                    if (
                        not isinstance(target_result, tuple)
                        or len(target_result) != 2
                    ):
                        raise TypeError("WorldFoam target-track session result changed")
                    target, target_receipt_raw = target_result
                    if (
                        not isinstance(prediction, torch.Tensor)
                        or prediction.device.type != "cpu"
                        or prediction.dtype != torch.float32
                        or tuple(prediction.shape) != (frames, len(pixel_ids), 3)
                        or not prediction.is_contiguous()
                        or not bool(torch.isfinite(prediction).all().item())
                        or not isinstance(target, torch.Tensor)
                        or target.device.type != "cpu"
                        or target.dtype != torch.float32
                        or tuple(target.shape) != (frames, len(pixel_ids), 3)
                        or not target.is_contiguous()
                    ):
                        raise ValueError(
                            "WorldFoam spatial-major session returned an invalid block"
                        )
                    target_receipt = _validate_target_read_receipt(
                        target_receipt_raw,
                        camera_index=camera_index,
                        pixel_ids=pixel_ids,
                        frames=frames,
                    )
                    target_receipt_manifest.update(_canonical_bytes(target_receipt))
                    target_receipt_manifest.update(b"\n")
                    target_rgb8 = _exact_rgb8(target)
                    _update_tensor_digest(prediction_write_content, prediction)
                    _update_tensor_digest(target_write_content, target_rgb8)
                    buffered.append((prediction, target_rgb8))
                    buffered_bytes += (
                        prediction.numel() * prediction.element_size()
                        + target_rgb8.numel() * target_rgb8.element_size()
                    )
                    spatial_call_count += 1
                    del target
                peak_buffered_bytes = max(peak_buffered_bytes, buffered_bytes)
                for frame_index in range(frames):
                    prediction_stripe = torch.cat(
                        [prediction[frame_index] for prediction, _target in buffered],
                        dim=0,
                    ).contiguous()
                    target_stripe = torch.cat(
                        [target[frame_index] for _prediction, target in buffered],
                        dim=0,
                    ).contiguous()
                    if (
                        tuple(prediction_stripe.shape) != (super_stop - super_start, 3)
                        or tuple(target_stripe.shape) != (super_stop - super_start, 3)
                    ):
                        raise ArithmeticError("WorldFoam write superblock coverage changed")
                    observation_offset = (
                        (camera_index * frames + frame_index) * pixels + super_start
                    )
                    _pwrite_all(
                        prediction_fd,
                        memoryview(prediction_stripe.numpy()).cast("B"),
                        observation_offset * 3 * 4,
                    )
                    _pwrite_all(
                        target_fd,
                        memoryview(target_stripe.numpy()).cast("B"),
                        observation_offset * 3,
                    )
                    target_source_frame_digests[
                        camera_index * frames + frame_index
                    ].update(memoryview(target_stripe.numpy()).cast("B"))
                    del prediction_stripe, target_stripe
                write_superblock_count += 1
                del buffered
        if spatial_call_count != render_calls:
            raise ArithmeticError("WorldFoam spatial-major render-call coverage changed")
        for descriptor, expected_size in (
            (prediction_fd, prediction_spool_bytes),
            (target_fd, target_spool_bytes),
        ):
            os.fsync(descriptor)
            if os.fstat(descriptor).st_size != expected_size:
                raise ArithmeticError("WorldFoam heldout spool size changed")
        # Hash the completed files in physical frame-major byte order.  The
        # block-production digest above is intentionally a separate provenance
        # chain because pwrite traversal is spatial-major rather than file-order.
        prediction_spool_file_sha256 = _descriptor_sha256(
            prediction_fd,
            prediction_spool_bytes,
        )
        target_spool_file_sha256 = _descriptor_sha256(
            target_fd,
            target_spool_bytes,
        )

        session_receipt = _validate_session_receipt(
            session_receipt_fn(),
            cameras=cameras,
            frames=frames,
            height=height,
            width=width,
            track_block_limit=track_block_limit,
            render_calls=render_calls,
        )

        metric = video_lpips if lpips_metric is None else lpips_metric
        accumulator = PaperRGBMetricAccumulator()
        lpips_sum = 0.0
        evaluated_frames = 0
        evaluated_pixels = 0
        metric_chunks = 0
        for camera_index in range(cameras):
            for frame_index in range(frames):
                prediction_frame = torch.empty((pixels, 3), dtype=torch.float32)
                target_frame = torch.empty((pixels, 3), dtype=torch.float32)
                covered = 0
                for pixel_start in range(0, pixels, metric_chunk_limit):
                    pixel_count = min(metric_chunk_limit, pixels - pixel_start)
                    request = PixelChunkRequest(
                        split="heldout",
                        step=None,
                        sample_slot=None,
                        camera_index=camera_index,
                        frame_index=frame_index,
                        pixel_start=pixel_start,
                        pixel_count=pixel_count,
                        image_height=height,
                        image_width=width,
                    )
                    observation_offset = (
                        (camera_index * frames + frame_index) * pixels + pixel_start
                    )
                    prediction_raw = _pread_exact(
                        prediction_fd,
                        pixel_count * 3 * 4,
                        observation_offset * 3 * 4,
                    )
                    target_raw = _pread_exact(
                        target_fd,
                        pixel_count * 3,
                        observation_offset * 3,
                    )
                    prediction_chunk = torch.frombuffer(
                        bytearray(prediction_raw),
                        dtype=torch.float32,
                    ).reshape(pixel_count, 3).clone().contiguous()
                    target_np = np.frombuffer(target_raw, dtype=np.uint8).reshape(
                        pixel_count,
                        3,
                    ).astype(np.float32, copy=True)
                    target_np /= 255.0
                    target_chunk = torch.from_numpy(
                        np.ascontiguousarray(target_np)
                    ).contiguous()
                    prediction_frame[pixel_start : pixel_start + pixel_count].copy_(
                        prediction_chunk
                    )
                    target_frame[pixel_start : pixel_start + pixel_count].copy_(
                        target_chunk
                    )
                    serialized = _canonical_bytes(request.as_dict())
                    coverage.update(serialized)
                    coverage.update(b"\n")
                    prediction_read_content.update(prediction_raw)
                    target_read_content.update(target_raw)
                    target_spool_frame_read_digests[
                        camera_index * frames + frame_index
                    ].update(target_raw)
                    covered += pixel_count
                    evaluated_pixels += pixel_count
                    metric_chunks += 1
                    del prediction_raw, target_raw, prediction_chunk, target_chunk, target_np
                if covered != pixels:
                    raise ArithmeticError("WorldFoam heldout frame coverage changed")
                prediction_hwc = prediction_frame.reshape(height, width, 3)
                target_hwc = target_frame.reshape(height, width, 3)
                accumulator.update(
                    prediction_hwc.unsqueeze(0),
                    target_hwc.unsqueeze(0),
                )
                lpips_value = float(
                    metric(
                        prediction_hwc.unsqueeze(0),
                        target_hwc.unsqueeze(0),
                    )
                )
                if not math.isfinite(lpips_value) or lpips_value < 0.0:
                    raise ValueError("heldout LPIPS is non-finite or negative")
                lpips_sum += lpips_value
                media_sink.add_frame(prediction_hwc, target_hwc)
                evaluated_frames += 1
                del prediction_frame, target_frame, prediction_hwc, target_hwc
        expected_frames = cameras * frames
        if evaluated_frames != expected_frames or evaluated_pixels != target_pixels:
            raise ArithmeticError("WorldFoam heldout evaluator lost exact coverage")
        if (
            prediction_read_content.hexdigest() != prediction_spool_file_sha256
            or target_read_content.hexdigest() != target_spool_file_sha256
        ):
            raise ArithmeticError("WorldFoam spool sequential digest changed on replay")
        target_source_frame_sha256s = [
            digest.hexdigest() for digest in target_source_frame_digests
        ]
        target_spool_frame_read_sha256s = [
            digest.hexdigest() for digest in target_spool_frame_read_digests
        ]
        if target_source_frame_sha256s != target_spool_frame_read_sha256s:
            raise ArithmeticError("WorldFoam target transpose changed frame content")
        common_metrics = accumulator.metrics(prefix="heldout_eval")
        metrics = {
            "heldout_eval_psnr": float(common_metrics["heldout_eval_psnr"]),
            "heldout_eval_ssim": float(common_metrics["heldout_eval_ssim"]),
            "heldout_eval_lpips": lpips_sum / float(evaluated_frames),
            "heldout_eval_l1": float(common_metrics["heldout_eval_l1"]),
        }
        if set(metrics) != set(REQUIRED_METRICS):
            raise ArithmeticError("heldout metric key set changed")

        os.close(prediction_fd)
        prediction_fd = None
        os.close(target_fd)
        target_fd = None
        prediction_path.unlink()
        prediction_path = None
        target_path.unlink()
        target_path = None

        receipt_payload = {
            "schema_version": 1,
            "kind": REPLAY_KIND,
            "camera_count": cameras,
            "frame_count": frames,
            "image_height": height,
            "image_width": width,
            "target_pixel_count": target_pixels,
            "rgb_scalar_count": target_pixels * 3,
            "spatial_track_count": cameras * pixels,
            "spatial_track_block_limit": track_block_limit,
            "spatial_track_block_count": render_calls,
            "maximum_observations_per_spatial_call": frames * track_block_limit,
            "write_superblock_track_limit": write_superblock_track_limit,
            "write_superblock_count": write_superblock_count,
            "peak_buffered_prediction_and_target_bytes": peak_buffered_bytes,
            "prediction_spool_bytes": prediction_spool_bytes,
            "target_spool_bytes": target_spool_bytes,
            "total_spool_bytes": total_spool_bytes,
            "prediction_spool_dtype": "float32",
            "target_spool_dtype": "uint8",
            "spool_shape": list(spool_shape),
            "prediction_spool_darwin_f_nocache": prediction_no_cache,
            "target_spool_darwin_f_nocache": target_no_cache,
            "spools_cleaned_before_return": True,
            "dense_device_video_used": False,
            "persistent_device_video_bytes": 0,
            "target_ray_tensor_bytes": 0,
            "metric_pixel_chunk_limit": metric_chunk_limit,
            "metric_pixel_chunk_count": metric_chunks,
            "lpips_evaluation_count": evaluated_frames,
            "media_frame_count": evaluated_frames,
            "metric_and_media_order": (
                "camera_major_then_frame_then_ascending_pixel_chunks"
            ),
            "metric_target_spool_observation_read_count": target_pixels,
            "native_prediction_target_source_observation_read_count": (
                session_receipt["native_prediction_target_observation_read_count"]
            ),
            "target_spool_source_observation_read_count": target_pixels,
            "total_target_source_observation_read_count": 2 * target_pixels,
            "total_target_observation_traversal_count": 3 * target_pixels,
            "forward_only_prediction_native_op_used": False,
            "heldout_wall_time_target_io_matched_across_routes": False,
            "track_request_manifest_sha256": request_manifest.hexdigest(),
            "prediction_block_content_sha256": prediction_write_content.hexdigest(),
            "target_block_rgb8_content_sha256": target_write_content.hexdigest(),
            "prediction_spool_file_sha256": prediction_spool_file_sha256,
            "target_spool_file_sha256": target_spool_file_sha256,
            "target_read_receipt_manifest_sha256": (
                target_receipt_manifest.hexdigest()
            ),
            "spool_read_request_manifest_sha256": coverage.hexdigest(),
            "prediction_spool_read_content_sha256": (
                prediction_read_content.hexdigest()
            ),
            "target_spool_read_content_sha256": target_read_content.hexdigest(),
            "target_source_frame_sha256s": target_source_frame_sha256s,
            "target_spool_frame_read_sha256s": (
                target_spool_frame_read_sha256s
            ),
            "target_source_to_spool_frame_hashes_equal": True,
            "metrics_sha256": _canonical_sha256(metrics),
            "heldout_coverage_sha256": coverage.hexdigest(),
            "session_receipt": session_receipt,
            "session_receipt_generation_digest": session_receipt[
                "generation_digest"
            ],
            "exact_rgb8_roundtrip_verified": True,
            "exact_full_pixel_full_temporal_coverage": True,
            "one_cold_compile_per_view_pixel_track": True,
        }
        receipt = validate_spatial_replay_receipt(
            {
                **receipt_payload,
                "generation_digest": _canonical_sha256(receipt_payload),
            }
        )
        media_path = Path(media_sink.finish(expected_frame_count=expected_frames))
        media_finished = True
        return SpatialMajorHeldoutEvaluationResult(
            evaluation=HeldoutEvaluationReceipt(
                metrics=metrics,
                frame_count=evaluated_frames,
                pixel_count=evaluated_pixels,
                pixel_chunk_count=metric_chunks,
                coverage_sha256=coverage.hexdigest(),
                media_path=media_path,
            ),
            spatial_replay_receipt=receipt,
        )
    except BaseException:
        if not media_finished:
            media_sink.abort()
        raise
    finally:
        if prediction_fd is not None:
            os.close(prediction_fd)
        if target_fd is not None:
            os.close(target_fd)
        if prediction_path is not None:
            prediction_path.unlink(missing_ok=True)
        if target_path is not None:
            target_path.unlink(missing_ok=True)


__all__ = (
    "REPLAY_KIND",
    "SESSION_REPLAY_KIND",
    "TARGET_READ_KIND",
    "SpatialMajorHeldoutEvaluationResult",
    "evaluate_worldfoam_spatial_major_final_checkpoint",
    "validate_spatial_replay_receipt",
)
