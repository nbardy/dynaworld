"""Strict identity contract for a public mapped-RGB8 WorldFoam target cache.

The schema deliberately separates data identity from the memory gate's source
manifest.  Raw inputs may be unavailable on the eventual training host, while
the cache manifest and payloads consumed by the mapped target source can be
rehash-verified relative to the binding file without importing PyTorch or any
dataset decoder.

This module validates declared decoded-identity equality and exact cache-byte
integrity only.  It does not recompute either decoded identity, build a cache,
decode public data, or certify that a measured trial used the binding. Those
are separate converter and companion-gate responsibilities.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO


BINDING_SCHEMA = "dynaworld.worldfoam_target_dataset_binding/v1"
TARGET_SPLITS = frozenset({"train", "heldout"})
TRAIN_HELDOUT_PAIR_RECEIPT_SCHEMA = (
    "dynaworld.worldfoam_train_heldout_target_dataset_pair_receipt/v1"
)
MAPPED_RGB8_SCHEMA = "dynaworld.powerfoam_mapped_rgb8/v1"
MAPPED_RGB8_LAYOUT = "height_width_frame_rgb_interleaved"
MAXIMUM_BINDING_BYTES = 1024 * 1024
MAXIMUM_MAPPED_MANIFEST_BYTES = 1024 * 1024

_BINDING_KEYS = {
    "schema",
    "dataset_id",
    "target_split",
    "converter",
    "raw_dataset_manifest",
    "raw_views",
    "cache",
    "decoded_f32_contract",
    "camera",
    "logical_frame_maps",
    "binding_sha256",
}
_CONVERTER_KEYS = {"provenance", "source_sha256"}
_FILE_IDENTITY_KEYS = {"path_label", "size_bytes", "sha256"}
_RAW_VIEW_KEYS = {"view_id", "raw_input", "raw_decoded_f32_sha256"}
_CACHE_KEYS = {
    "manifest",
    "format_schema",
    "layout",
    "dtype",
    "height",
    "width",
    "stored_frame_indices",
    "views",
}
_CACHE_VIEW_KEYS = {"view_id", "payload", "cache_decoded_f32_sha256"}
_DECODED_F32_CONTRACT_KEYS = {
    "dtype",
    "layout",
    "byte_order",
    "range",
    "conversion",
    "hash_payload",
    "shape_per_view",
}
_CAMERA_KEYS = {
    "view_ids",
    "height",
    "width",
    "frame_times",
    "K",
    "w2c",
    "lens_models",
    "distortions",
    "pose_source",
    "camera_generation_digest",
}
_TENSOR_IDENTITY_KEYS = {
    "dtype",
    "shape",
    "bytes",
    "byte_order",
    "layout",
    "sha256",
}
_FRAME_MAP_KEYS = {
    "frame_count",
    "source_frame_indices",
    "logical_frame_map_sha256",
}
_MAPPED_MANIFEST_KEYS = {
    "schema",
    "layout",
    "dtype",
    "height",
    "width",
    "stored_frame_indices",
    "views",
}
_MAPPED_MANIFEST_VIEW_KEYS = {
    "view_id",
    "payload",
    "payload_bytes",
    "payload_sha256",
}


def canonical_payload_sha256(value: Any) -> str:
    """Hash one JSON-compatible value with the project's canonical encoding."""

    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _strict_json_loads(raw: bytes, name: str) -> Any:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> Any:
        raise ValueError(f"{name} contains nonstandard JSON constant {value!r}")

    try:
        text = raw.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not strict UTF-8 JSON") from exc


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    name: str,
) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} keys changed")


def _require_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a nonempty trimmed string")
    return value


def _require_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_target_split(value: Any, name: str = "target_split") -> str:
    split = _require_nonempty_string(value, name)
    if split not in TARGET_SPLITS:
        raise ValueError(f"{name} must be exactly one of {sorted(TARGET_SPLITS)}")
    return split


def _require_integer(value: Any, name: str, *, positive: bool) -> int:
    lower_bound = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < lower_bound:
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


def _require_int_list(
    value: Any,
    name: str,
    *,
    positive: bool,
) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty list")
    return tuple(
        _require_integer(item, f"{name}[{index}]", positive=positive)
        for index, item in enumerate(value)
    )


def _require_increasing_nonnegative_ints(value: Any, name: str) -> tuple[int, ...]:
    result = _require_int_list(value, name, positive=False)
    if tuple(sorted(set(result))) != result:
        raise ValueError(f"{name} must be unique and strictly increasing")
    return result


def _require_portable_relative_path(value: Any, name: str) -> str:
    label = _require_nonempty_string(value, name)
    path = PurePosixPath(label)
    if (
        "\\" in label
        or any(ord(character) < 32 or ord(character) == 127 for character in label)
        or path.is_absolute()
        or path.as_posix() != label
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{name} must be a canonical portable relative path")
    return label


def _validate_file_identity(value: Any, name: str) -> dict[str, Any]:
    identity = _require_mapping(value, name)
    _require_exact_keys(identity, _FILE_IDENTITY_KEYS, name)
    return {
        "path_label": _require_portable_relative_path(
            identity.get("path_label"), f"{name}.path_label"
        ),
        "size_bytes": _require_integer(
            identity.get("size_bytes"), f"{name}.size_bytes", positive=True
        ),
        "sha256": _require_sha256(identity.get("sha256"), f"{name}.sha256"),
    }


def _validate_tensor_identity(
    value: Any,
    name: str,
    *,
    expected_shape: Sequence[int] | None = None,
) -> dict[str, Any]:
    identity = _require_mapping(value, name)
    _require_exact_keys(identity, _TENSOR_IDENTITY_KEYS, name)
    if identity.get("dtype") != "torch.float32":
        raise ValueError(f"{name}.dtype must be torch.float32")
    shape = _require_int_list(identity.get("shape"), f"{name}.shape", positive=True)
    if expected_shape is not None and shape != tuple(expected_shape):
        raise ValueError(f"{name}.shape does not match the binding grid")
    expected_bytes = math.prod(shape) * 4
    if (
        _require_integer(identity.get("bytes"), f"{name}.bytes", positive=True)
        != expected_bytes
    ):
        raise ValueError(f"{name}.bytes does not match float32 shape storage")
    if identity.get("byte_order") != "native_little_endian":
        raise ValueError(f"{name}.byte_order must be native_little_endian")
    if identity.get("layout") != "contiguous_c_order":
        raise ValueError(f"{name}.layout must be contiguous_c_order")
    _require_sha256(identity.get("sha256"), f"{name}.sha256")
    return dict(identity)


def _validate_required_frame_counts(value: Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ValueError("required_frame_counts must be a nonempty sequence")
    result = tuple(
        _require_integer(item, "required_frame_counts item", positive=True)
        for item in value
    )
    if tuple(sorted(set(result))) != result:
        raise ValueError("required_frame_counts must be unique and increasing")
    return result


def validate_target_dataset_binding(
    payload: Mapping[str, Any],
    *,
    required_frame_counts: Sequence[int],
) -> dict[str, Any]:
    """Validate one self-contained public target identity without file access."""

    binding = _require_mapping(payload, "target_dataset_binding")
    _require_exact_keys(binding, _BINDING_KEYS, "target_dataset_binding")
    required_frames = _validate_required_frame_counts(required_frame_counts)
    if binding.get("schema") != BINDING_SCHEMA:
        raise ValueError("target_dataset_binding schema is missing or stale")
    _require_nonempty_string(binding.get("dataset_id"), "dataset_id")
    _require_target_split(binding.get("target_split"))

    converter = _require_mapping(binding.get("converter"), "converter")
    _require_exact_keys(converter, _CONVERTER_KEYS, "converter")
    if converter.get("provenance") != "build_worldfoam_mapped_rgb8_cache/v1":
        raise ValueError("converter provenance is missing or stale")
    _require_sha256(converter.get("source_sha256"), "converter.source_sha256")
    _validate_file_identity(binding.get("raw_dataset_manifest"), "raw_dataset_manifest")

    raw_views = binding.get("raw_views")
    if not isinstance(raw_views, list) or not raw_views:
        raise ValueError("raw_views must be a nonempty list")
    raw_view_ids: list[str] = []
    raw_decoded_digests: dict[str, str] = {}
    for index, raw_value in enumerate(raw_views):
        name = f"raw_views[{index}]"
        raw_view = _require_mapping(raw_value, name)
        _require_exact_keys(raw_view, _RAW_VIEW_KEYS, name)
        view_id = _require_nonempty_string(raw_view.get("view_id"), f"{name}.view_id")
        _validate_file_identity(raw_view.get("raw_input"), f"{name}.raw_input")
        raw_view_ids.append(view_id)
        raw_decoded_digests[view_id] = _require_sha256(
            raw_view.get("raw_decoded_f32_sha256"),
            f"{name}.raw_decoded_f32_sha256",
        )
    if tuple(raw_view_ids) != tuple(sorted(set(raw_view_ids))):
        raise ValueError("raw_views must have unique view ids in sorted order")

    cache = _require_mapping(binding.get("cache"), "cache")
    _require_exact_keys(cache, _CACHE_KEYS, "cache")
    _validate_file_identity(cache.get("manifest"), "cache.manifest")
    if (
        cache.get("format_schema") != MAPPED_RGB8_SCHEMA
        or cache.get("layout") != MAPPED_RGB8_LAYOUT
        or cache.get("dtype") != "uint8"
    ):
        raise ValueError("cache schema, layout, or dtype changed")
    height = _require_integer(cache.get("height"), "cache.height", positive=True)
    width = _require_integer(cache.get("width"), "cache.width", positive=True)
    stored_frames = _require_increasing_nonnegative_ints(
        cache.get("stored_frame_indices"), "cache.stored_frame_indices"
    )
    cache_views = cache.get("views")
    if not isinstance(cache_views, list) or not cache_views:
        raise ValueError("cache.views must be a nonempty list")
    expected_payload_bytes = height * width * len(stored_frames) * 3
    cache_view_ids: list[str] = []
    for index, cache_value in enumerate(cache_views):
        name = f"cache.views[{index}]"
        cache_view = _require_mapping(cache_value, name)
        _require_exact_keys(cache_view, _CACHE_VIEW_KEYS, name)
        view_id = _require_nonempty_string(cache_view.get("view_id"), f"{name}.view_id")
        payload_identity = _validate_file_identity(
            cache_view.get("payload"), f"{name}.payload"
        )
        if payload_identity["size_bytes"] != expected_payload_bytes:
            raise ValueError(f"{name}.payload size does not match mapped RGB8 storage")
        cache_decoded_digest = _require_sha256(
            cache_view.get("cache_decoded_f32_sha256"),
            f"{name}.cache_decoded_f32_sha256",
        )
        if raw_decoded_digests.get(view_id) != cache_decoded_digest:
            raise ValueError(f"{name} raw/cache decoded float32 identities differ")
        cache_view_ids.append(view_id)
    if tuple(cache_view_ids) != tuple(raw_view_ids):
        raise ValueError("raw and cache views must use one identical sorted order")

    decoded = _require_mapping(
        binding.get("decoded_f32_contract"), "decoded_f32_contract"
    )
    _require_exact_keys(
        decoded,
        _DECODED_F32_CONTRACT_KEYS,
        "decoded_f32_contract",
    )
    expected_decoded_literals = {
        "dtype": "float32",
        "layout": "time_channel_height_width_contiguous_c_order",
        "byte_order": "little_endian_ieee754",
        "conversion": "uint8_exact_divide_255_to_float32",
        "hash_payload": "raw_contiguous_tensor_bytes_without_metadata",
    }
    for key, expected in expected_decoded_literals.items():
        if decoded.get(key) != expected:
            raise ValueError(f"decoded_f32_contract.{key} changed")
    decoded_range = decoded.get("range")
    if (
        not isinstance(decoded_range, list)
        or len(decoded_range) != 2
        or any(isinstance(item, bool) or not isinstance(item, float) for item in decoded_range)
        or decoded_range != [0.0, 1.0]
    ):
        raise ValueError("decoded_f32_contract.range must be [0.0, 1.0]")
    decoded_shape = _require_int_list(
        decoded.get("shape_per_view"),
        "decoded_f32_contract.shape_per_view",
        positive=True,
    )
    if decoded_shape != (len(stored_frames), 3, height, width):
        raise ValueError("decoded float32 shape does not match the common cache grid")

    camera = _require_mapping(binding.get("camera"), "camera")
    _require_exact_keys(camera, _CAMERA_KEYS, "camera")
    camera_view_ids = camera.get("view_ids")
    if (
        not isinstance(camera_view_ids, list)
        or any(not isinstance(item, str) for item in camera_view_ids)
        or camera_view_ids != raw_view_ids
    ):
        raise ValueError("camera.view_ids must match the common sorted cache view order")
    if (
        _require_integer(camera.get("height"), "camera.height", positive=True) != height
        or _require_integer(camera.get("width"), "camera.width", positive=True) != width
    ):
        raise ValueError("camera resolution does not match the cache")
    view_count = len(raw_view_ids)
    frame_count = len(stored_frames)
    _validate_tensor_identity(
        camera.get("frame_times"),
        "camera.frame_times",
        expected_shape=(frame_count, 1),
    )
    _validate_tensor_identity(
        camera.get("K"),
        "camera.K",
        expected_shape=(view_count, 3, 3),
    )
    _validate_tensor_identity(
        camera.get("w2c"),
        "camera.w2c",
        expected_shape=(view_count, frame_count, 4, 4),
    )
    lens_models_raw = camera.get("lens_models")
    if not isinstance(lens_models_raw, list) or len(lens_models_raw) != view_count:
        raise ValueError("camera.lens_models must contain one nonempty model per view")
    lens_models = tuple(
        _require_nonempty_string(item, f"camera.lens_models[{index}]")
        for index, item in enumerate(lens_models_raw)
    )
    supported_lens_models = {"pinhole", "radial_tangential", "opencv_fisheye"}
    if any(item not in supported_lens_models for item in lens_models):
        raise ValueError("camera.lens_models contains an unsupported model")
    nonpinhole_models = set(lens_models) - {"pinhole"}
    if len(nonpinhole_models) > 1:
        raise ValueError("one binding cannot mix non-pinhole distortion families")
    distortions = camera.get("distortions")
    if not nonpinhole_models:
        if distortions is not None:
            raise ValueError("all-pinhole cameras must not carry distortion coefficients")
    else:
        coefficient_count = (
            5 if nonpinhole_models == {"radial_tangential"} else 4
        )
        if distortions is None:
            raise ValueError("non-pinhole cameras require distortion coefficients")
        _validate_tensor_identity(
            distortions,
            "camera.distortions",
            expected_shape=(view_count, coefficient_count),
        )
    _require_nonempty_string(camera.get("pose_source"), "camera.pose_source")
    _require_sha256(
        camera.get("camera_generation_digest"),
        "camera.camera_generation_digest",
    )

    frame_maps = binding.get("logical_frame_maps")
    if not isinstance(frame_maps, list) or not frame_maps:
        raise ValueError("logical_frame_maps must be a nonempty list")
    observed_counts: list[int] = []
    observed_maps: dict[int, tuple[int, ...]] = {}
    stored_set = set(stored_frames)
    for index, map_value in enumerate(frame_maps):
        name = f"logical_frame_maps[{index}]"
        frame_map = _require_mapping(map_value, name)
        _require_exact_keys(frame_map, _FRAME_MAP_KEYS, name)
        count = _require_integer(
            frame_map.get("frame_count"), f"{name}.frame_count", positive=True
        )
        indices = _require_increasing_nonnegative_ints(
            frame_map.get("source_frame_indices"),
            f"{name}.source_frame_indices",
        )
        if len(indices) != count:
            raise ValueError(f"{name} length does not match frame_count")
        if any(item not in stored_set for item in indices):
            raise ValueError(f"{name} leaves the common stored cache")
        if indices[0] != stored_frames[0] or indices[-1] != stored_frames[-1]:
            raise ValueError(f"{name} does not preserve the common physical endpoints")
        if canonical_payload_sha256(list(indices)) != frame_map.get(
            "logical_frame_map_sha256"
        ):
            raise ValueError(f"{name} logical frame-map digest changed")
        observed_counts.append(count)
        observed_maps[count] = indices
    if tuple(observed_counts) != required_frames:
        raise ValueError("logical frame maps do not match required_frame_counts")
    if observed_maps[required_frames[-1]] != stored_frames:
        raise ValueError("largest logical frame map must equal the complete stored cache")

    expected_binding_sha256 = canonical_payload_sha256(
        {key: value for key, value in binding.items() if key != "binding_sha256"}
    )
    if binding.get("binding_sha256") != expected_binding_sha256:
        raise ValueError("target_dataset_binding canonical digest changed")
    return dict(binding)


def _file_stat_signature(stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


def _streaming_open_file_sha256(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    handle.seek(0)
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _read_bounded_stable_file(
    path: Path,
    *,
    maximum_bytes: int,
    name: str,
) -> bytes:
    if maximum_bytes < 1:
        raise ValueError(f"{name} byte cap must be positive")
    with path.open("rb") as handle:
        stat_before = _file_stat_signature(os.fstat(handle.fileno()))
        if stat_before[2] > maximum_bytes:
            raise ValueError(f"{name} exceeds its {maximum_bytes}-byte cap")
        raw = handle.read(maximum_bytes + 1)
        stat_after = _file_stat_signature(os.fstat(handle.fileno()))
    if stat_before != stat_after or len(raw) != stat_before[2]:
        raise ValueError(f"{name} changed while reading")
    if len(raw) > maximum_bytes:
        raise ValueError(f"{name} exceeds its {maximum_bytes}-byte cap")
    return raw


def _verify_loaded_file_identity(
    raw: bytes,
    identity: Mapping[str, Any],
    name: str,
) -> None:
    if len(raw) != identity["size_bytes"]:
        raise ValueError(f"{name} size changed")
    if hashlib.sha256(raw).hexdigest() != identity["sha256"]:
        raise ValueError(f"{name} digest changed")


def _verify_file_identity(path: Path, identity: Mapping[str, Any], name: str) -> None:
    with path.open("rb") as handle:
        stat_before = _file_stat_signature(os.fstat(handle.fileno()))
        if stat_before[2] != identity["size_bytes"]:
            raise ValueError(f"{name} size changed")
        if _streaming_open_file_sha256(handle) != identity["sha256"]:
            raise ValueError(f"{name} digest changed")
        stat_after = _file_stat_signature(os.fstat(handle.fileno()))
    if stat_before != stat_after:
        raise ValueError(f"{name} changed while hashing")


def _resolve_beneath(root: Path, label: str, name: str) -> Path:
    resolved_root = root.resolve()
    candidate = (resolved_root / label).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{name} escaped its binding root") from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"{name} is not a file: {candidate}")
    return candidate


def verify_bound_cache_files(
    payload: Mapping[str, Any],
    *,
    binding_path: Path,
    required_frame_counts: Sequence[int],
) -> dict[str, Any]:
    """Rehash the exact mapped manifest and payloads consumed by training."""

    binding = validate_target_dataset_binding(
        payload,
        required_frame_counts=required_frame_counts,
    )
    cache = _require_mapping(binding["cache"], "cache")
    manifest_identity = _require_mapping(cache["manifest"], "cache.manifest")
    manifest_path = _resolve_beneath(
        binding_path.expanduser().resolve().parent,
        str(manifest_identity["path_label"]),
        "cache manifest",
    )
    raw_manifest_bytes = _read_bounded_stable_file(
        manifest_path,
        maximum_bytes=MAXIMUM_MAPPED_MANIFEST_BYTES,
        name="cache manifest",
    )
    _verify_loaded_file_identity(
        raw_manifest_bytes,
        manifest_identity,
        "cache manifest",
    )
    raw_manifest = _strict_json_loads(raw_manifest_bytes, "mapped RGB8 manifest")
    manifest = _require_mapping(raw_manifest, "mapped RGB8 manifest")
    _require_exact_keys(manifest, _MAPPED_MANIFEST_KEYS, "mapped RGB8 manifest")
    manifest_height = _require_integer(
        manifest.get("height"),
        "mapped RGB8 manifest.height",
        positive=True,
    )
    manifest_width = _require_integer(
        manifest.get("width"),
        "mapped RGB8 manifest.width",
        positive=True,
    )
    manifest_stored_frames = _require_increasing_nonnegative_ints(
        manifest.get("stored_frame_indices"),
        "mapped RGB8 manifest.stored_frame_indices",
    )
    if (
        manifest.get("schema") != cache["format_schema"]
        or manifest.get("layout") != cache["layout"]
        or manifest.get("dtype") != cache["dtype"]
        or manifest_height != cache["height"]
        or manifest_width != cache["width"]
        or manifest_stored_frames != tuple(cache["stored_frame_indices"])
    ):
        raise ValueError("mapped RGB8 manifest does not match the dataset binding")
    manifest_views = manifest.get("views")
    if not isinstance(manifest_views, list) or len(manifest_views) != len(cache["views"]):
        raise ValueError("mapped RGB8 manifest view count changed")

    payload_records: list[dict[str, Any]] = []
    for index, (manifest_value, binding_value) in enumerate(
        zip(manifest_views, cache["views"], strict=True)
    ):
        name = f"mapped RGB8 manifest views[{index}]"
        manifest_view = _require_mapping(manifest_value, name)
        _require_exact_keys(manifest_view, _MAPPED_MANIFEST_VIEW_KEYS, name)
        binding_view = _require_mapping(binding_value, f"cache.views[{index}]")
        payload_identity = _require_mapping(
            binding_view["payload"], f"cache.views[{index}].payload"
        )
        manifest_view_id = _require_nonempty_string(
            manifest_view.get("view_id"),
            f"{name}.view_id",
        )
        manifest_payload_label = _require_portable_relative_path(
            manifest_view.get("payload"),
            f"{name}.payload",
        )
        manifest_payload_bytes = _require_integer(
            manifest_view.get("payload_bytes"),
            f"{name}.payload_bytes",
            positive=True,
        )
        manifest_payload_sha256 = _require_sha256(
            manifest_view.get("payload_sha256"),
            f"{name}.payload_sha256",
        )
        if (
            manifest_view_id != binding_view["view_id"]
            or manifest_payload_label != payload_identity["path_label"]
            or manifest_payload_bytes != payload_identity["size_bytes"]
            or manifest_payload_sha256 != payload_identity["sha256"]
        ):
            raise ValueError(f"{name} does not match the dataset binding")
        payload_path = _resolve_beneath(
            manifest_path.parent,
            str(payload_identity["path_label"]),
            f"cache payload {binding_view['view_id']!r}",
        )
        _verify_file_identity(
            payload_path,
            payload_identity,
            f"cache payload {binding_view['view_id']!r}",
        )
        payload_records.append(
            {
                "view_id": binding_view["view_id"],
                "payload_bytes": payload_identity["size_bytes"],
                "payload_sha256": payload_identity["sha256"],
            }
        )
    return {
        "target_dataset_binding_sha256": binding["binding_sha256"],
        "cache_manifest_sha256": manifest_identity["sha256"],
        "cache_payload_set_sha256": canonical_payload_sha256(payload_records),
        "cache_payload_count": len(payload_records),
        "cache_payload_bytes_rehashed": True,
        "raw_cache_decoded_f32_equality_declared": True,
        "raw_cache_decoded_f32_equality_recomputed": False,
    }


def load_target_dataset_binding(
    path: Path,
    *,
    required_frame_counts: Sequence[int],
    verify_cache_files: bool = False,
) -> dict[str, Any]:
    """Load a binding and optionally rehash its mapped cache files."""

    binding_path = path.expanduser().resolve()
    if not binding_path.is_file():
        raise FileNotFoundError(f"target dataset binding is not a file: {binding_path}")
    raw_binding = _read_bounded_stable_file(
        binding_path,
        maximum_bytes=MAXIMUM_BINDING_BYTES,
        name="target dataset binding",
    )
    payload = _strict_json_loads(raw_binding, "target dataset binding")
    binding = validate_target_dataset_binding(
        _require_mapping(payload, "target_dataset_binding"),
        required_frame_counts=required_frame_counts,
    )
    if verify_cache_files:
        verify_bound_cache_files(
            binding,
            binding_path=binding_path,
            required_frame_counts=required_frame_counts,
        )
    return binding


def _validated_train_heldout_pair(
    train_payload: Mapping[str, Any],
    heldout_payload: Mapping[str, Any],
    *,
    required_frame_counts: Sequence[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the split boundary and the physical grid shared by two caches."""

    train = validate_target_dataset_binding(
        train_payload,
        required_frame_counts=required_frame_counts,
    )
    heldout = validate_target_dataset_binding(
        heldout_payload,
        required_frame_counts=required_frame_counts,
    )
    if train["target_split"] != "train" or heldout["target_split"] != "heldout":
        raise ValueError("target dataset pair must be ordered as train then heldout")
    if train["dataset_id"] != heldout["dataset_id"]:
        raise ValueError("train and heldout caches must bind one dataset_id")

    train_views = tuple(train["camera"]["view_ids"])
    heldout_views = tuple(heldout["camera"]["view_ids"])
    overlap = sorted(set(train_views) & set(heldout_views))
    if overlap:
        raise ValueError(f"train and heldout cache view ids overlap: {overlap}")

    if train["converter"] != heldout["converter"]:
        raise ValueError("train and heldout caches use different converter identities")
    train_cache = train["cache"]
    heldout_cache = heldout["cache"]
    common_cache_keys = (
        "format_schema",
        "layout",
        "dtype",
        "height",
        "width",
        "stored_frame_indices",
    )
    if any(train_cache[key] != heldout_cache[key] for key in common_cache_keys):
        raise ValueError("train and heldout caches do not share one physical cache grid")
    if train["decoded_f32_contract"] != heldout["decoded_f32_contract"]:
        raise ValueError("train and heldout caches use different decoded RGB contracts")
    if train["logical_frame_maps"] != heldout["logical_frame_maps"]:
        raise ValueError("train and heldout caches use different logical frame maps")

    train_camera = train["camera"]
    heldout_camera = heldout["camera"]
    common_camera_keys = (
        "height",
        "width",
        "frame_times",
        "pose_source",
        "camera_generation_digest",
    )
    if any(train_camera[key] != heldout_camera[key] for key in common_camera_keys):
        raise ValueError(
            "train and heldout caches do not share one camera-generation/time grid"
        )
    return train, heldout


def verify_train_heldout_target_dataset_pair(
    *,
    train_binding_path: Path,
    heldout_binding_path: Path,
    required_frame_counts: Sequence[int],
) -> dict[str, Any]:
    """Load and rehash two disjoint caches, returning one worker-safe receipt.

    The receipt binds both split-specific cache payload sets without requiring
    either cache to share a directory.  It intentionally carries no absolute
    host paths; callers retain the two input paths as runtime handles and use
    the receipt digests as the evidence identity.
    """

    train_path = train_binding_path.expanduser().resolve()
    heldout_path = heldout_binding_path.expanduser().resolve()
    if train_path == heldout_path:
        raise ValueError("train and heldout bindings must be distinct files")
    train = load_target_dataset_binding(
        train_path,
        required_frame_counts=required_frame_counts,
    )
    heldout = load_target_dataset_binding(
        heldout_path,
        required_frame_counts=required_frame_counts,
    )
    train, heldout = _validated_train_heldout_pair(
        train,
        heldout,
        required_frame_counts=required_frame_counts,
    )
    train_cache = verify_bound_cache_files(
        train,
        binding_path=train_path,
        required_frame_counts=required_frame_counts,
    )
    heldout_cache = verify_bound_cache_files(
        heldout,
        binding_path=heldout_path,
        required_frame_counts=required_frame_counts,
    )
    receipt: dict[str, Any] = {
        "schema": TRAIN_HELDOUT_PAIR_RECEIPT_SCHEMA,
        "dataset_id": train["dataset_id"],
        "train": {
            "target_dataset_binding_sha256": train["binding_sha256"],
            "view_ids": list(train["camera"]["view_ids"]),
            "cache_manifest_sha256": train_cache["cache_manifest_sha256"],
            "cache_payload_set_sha256": train_cache["cache_payload_set_sha256"],
            "cache_payload_count": train_cache["cache_payload_count"],
        },
        "heldout": {
            "target_dataset_binding_sha256": heldout["binding_sha256"],
            "view_ids": list(heldout["camera"]["view_ids"]),
            "cache_manifest_sha256": heldout_cache["cache_manifest_sha256"],
            "cache_payload_set_sha256": heldout_cache["cache_payload_set_sha256"],
            "cache_payload_count": heldout_cache["cache_payload_count"],
        },
        "common_grid": {
            "format_schema": train["cache"]["format_schema"],
            "layout": train["cache"]["layout"],
            "dtype": train["cache"]["dtype"],
            "height": train["cache"]["height"],
            "width": train["cache"]["width"],
            "stored_frame_indices": list(train["cache"]["stored_frame_indices"]),
            "logical_frame_maps_sha256": canonical_payload_sha256(
                train["logical_frame_maps"]
            ),
            "frame_times_sha256": train["camera"]["frame_times"]["sha256"],
            "pose_source": train["camera"]["pose_source"],
            "camera_generation_digest": train["camera"][
                "camera_generation_digest"
            ],
        },
        "train_heldout_view_sets_disjoint": True,
        "cache_payload_bytes_rehashed": True,
        "pair_receipt_sha256": "",
    }
    receipt["pair_receipt_sha256"] = canonical_payload_sha256(
        {
            key: value
            for key, value in receipt.items()
            if key != "pair_receipt_sha256"
        }
    )
    return receipt


__all__ = [
    "BINDING_SCHEMA",
    "MAXIMUM_BINDING_BYTES",
    "MAXIMUM_MAPPED_MANIFEST_BYTES",
    "MAPPED_RGB8_LAYOUT",
    "MAPPED_RGB8_SCHEMA",
    "TARGET_SPLITS",
    "TRAIN_HELDOUT_PAIR_RECEIPT_SCHEMA",
    "canonical_payload_sha256",
    "load_target_dataset_binding",
    "validate_target_dataset_binding",
    "verify_bound_cache_files",
    "verify_train_heldout_target_dataset_pair",
]
