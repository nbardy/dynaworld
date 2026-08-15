from __future__ import annotations

import copy
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import pytest

import worldfoam_target_dataset_binding as binding_contract


REQUIRED_FRAME_COUNTS = (2, 3)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_identity(path_label: str, payload: bytes) -> dict[str, Any]:
    return {
        "path_label": path_label,
        "size_bytes": len(payload),
        "sha256": _sha256_bytes(payload),
    }


def _tensor_identity(shape: list[int], digest_character: str) -> dict[str, Any]:
    element_count = 1
    for dimension in shape:
        element_count *= dimension
    return {
        "dtype": "torch.float32",
        "shape": shape,
        "bytes": element_count * 4,
        "byte_order": "native_little_endian",
        "layout": "contiguous_c_order",
        "sha256": digest_character * 64,
    }


def _mapped_rgb8_decoded_f32_sha256(
    payload: bytes,
    *,
    height: int,
    width: int,
    frame_count: int,
) -> str:
    digest = hashlib.sha256()
    for frame in range(frame_count):
        for channel in range(3):
            for row in range(height):
                for column in range(width):
                    offset = (((row * width + column) * frame_count + frame) * 3) + channel
                    digest.update(struct.pack("<f", payload[offset] / 255.0))
    return digest.hexdigest()


def _reseal(payload: dict[str, Any]) -> None:
    payload["binding_sha256"] = binding_contract.canonical_payload_sha256(
        {key: value for key, value in payload.items() if key != "binding_sha256"}
    )


def _write_fixture(root: Path) -> tuple[Path, dict[str, Any]]:
    payload_bytes = 2 * 3 * 3 * 3
    cache_payloads = {
        "camera_00.rgb8": bytes(range(payload_bytes)),
        "camera_01.rgb8": bytes(reversed(range(payload_bytes))),
    }
    for name, contents in cache_payloads.items():
        (root / name).write_bytes(contents)
    decoded_digests = {
        name: _mapped_rgb8_decoded_f32_sha256(
            contents,
            height=2,
            width=3,
            frame_count=3,
        )
        for name, contents in cache_payloads.items()
    }

    cache_views = [
        {
            "view_id": view_id,
            "payload": _file_identity(payload_name, cache_payloads[payload_name]),
            "cache_decoded_f32_sha256": decoded_digests[payload_name],
        }
        for view_id, payload_name in (
            ("camera_00", "camera_00.rgb8"),
            ("camera_01", "camera_01.rgb8"),
        )
    ]
    mapped_manifest = {
        "schema": "dynaworld.powerfoam_mapped_rgb8/v1",
        "layout": "height_width_frame_rgb_interleaved",
        "dtype": "uint8",
        "height": 2,
        "width": 3,
        "stored_frame_indices": [0, 2, 4],
        "views": [
            {
                "view_id": item["view_id"],
                "payload": item["payload"]["path_label"],
                "payload_bytes": item["payload"]["size_bytes"],
                "payload_sha256": item["payload"]["sha256"],
            }
            for item in cache_views
        ],
    }
    manifest_bytes = json.dumps(
        mapped_manifest,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    manifest_path = root / "mapped_rgb8_manifest.json"
    manifest_path.write_bytes(manifest_bytes)

    frame_maps = [
        {
            "frame_count": frame_count,
            "source_frame_indices": indices,
            "logical_frame_map_sha256": binding_contract.canonical_payload_sha256(
                indices
            ),
        }
        for frame_count, indices in ((2, [0, 4]), (3, [0, 2, 4]))
    ]
    binding = {
        "schema": "dynaworld.worldfoam_target_dataset_binding/v1",
        "dataset_id": "fixture/public_scene",
        "target_split": "train",
        "converter": {
            "provenance": "build_worldfoam_mapped_rgb8_cache/v1",
            "source_sha256": "a" * 64,
        },
        "raw_dataset_manifest": {
            "path_label": "dataset/manifest.json",
            "size_bytes": 17,
            "sha256": "b" * 64,
        },
        "raw_views": [
            {
                "view_id": "camera_00",
                "raw_input": {
                    "path_label": "dataset/camera_00.mp4",
                    "size_bytes": 101,
                    "sha256": "c" * 64,
                },
                "raw_decoded_f32_sha256": decoded_digests["camera_00.rgb8"],
            },
            {
                "view_id": "camera_01",
                "raw_input": {
                    "path_label": "dataset/camera_01.mp4",
                    "size_bytes": 103,
                    "sha256": "d" * 64,
                },
                "raw_decoded_f32_sha256": decoded_digests["camera_01.rgb8"],
            },
        ],
        "cache": {
            "manifest": _file_identity(
                "mapped_rgb8_manifest.json",
                manifest_bytes,
            ),
            "format_schema": "dynaworld.powerfoam_mapped_rgb8/v1",
            "layout": "height_width_frame_rgb_interleaved",
            "dtype": "uint8",
            "height": 2,
            "width": 3,
            "stored_frame_indices": [0, 2, 4],
            "views": cache_views,
        },
        "decoded_f32_contract": {
            "dtype": "float32",
            "layout": "time_channel_height_width_contiguous_c_order",
            "byte_order": "little_endian_ieee754",
            "range": [0.0, 1.0],
            "conversion": "uint8_exact_divide_255_to_float32",
            "hash_payload": "raw_contiguous_tensor_bytes_without_metadata",
            "shape_per_view": [3, 3, 2, 3],
        },
        "camera": {
            "view_ids": ["camera_00", "camera_01"],
            "height": 2,
            "width": 3,
            "frame_times": _tensor_identity([3, 1], "3"),
            "K": _tensor_identity([2, 3, 3], "4"),
            "w2c": _tensor_identity([2, 3, 4, 4], "5"),
            "lens_models": ["pinhole", "pinhole"],
            "distortions": None,
            "pose_source": "fixture_static_camera_grid",
            "camera_generation_digest": "6" * 64,
        },
        "logical_frame_maps": frame_maps,
        "binding_sha256": "",
    }
    _reseal(binding)
    binding_path = root / "target_dataset_binding.json"
    binding_path.write_text(
        json.dumps(binding, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return binding_path, binding


def _rewrite_fixture_as_heldout(
    binding_path: Path,
    binding: dict[str, Any],
    *,
    view_ids: tuple[str, str] = ("camera_02", "camera_03"),
) -> None:
    binding["target_split"] = "heldout"
    for index, view_id in enumerate(view_ids):
        binding["raw_views"][index]["view_id"] = view_id
        binding["cache"]["views"][index]["view_id"] = view_id
        binding["camera"]["view_ids"][index] = view_id
    manifest_path = binding_path.parent / "mapped_rgb8_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for index, view_id in enumerate(view_ids):
        manifest["views"][index]["view_id"] = view_id
    manifest_bytes = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)
    binding["cache"]["manifest"] = _file_identity(
        "mapped_rgb8_manifest.json",
        manifest_bytes,
    )
    _reseal(binding)
    binding_path.write_text(
        json.dumps(binding, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def test_binding_and_bound_cache_files_validate_without_raw_inputs(tmp_path: Path) -> None:
    binding_path, expected = _write_fixture(tmp_path)

    loaded = binding_contract.load_target_dataset_binding(
        binding_path,
        required_frame_counts=REQUIRED_FRAME_COUNTS,
        verify_cache_files=True,
    )
    receipt = binding_contract.verify_bound_cache_files(
        loaded,
        binding_path=binding_path,
        required_frame_counts=REQUIRED_FRAME_COUNTS,
    )

    assert loaded == expected
    assert receipt["target_dataset_binding_sha256"] == expected["binding_sha256"]
    assert receipt["cache_manifest_sha256"] == expected["cache"]["manifest"]["sha256"]
    assert receipt["cache_payload_count"] == 2
    assert len(receipt["cache_payload_set_sha256"]) == 64
    assert receipt["cache_payload_bytes_rehashed"] is True
    assert receipt["raw_cache_decoded_f32_equality_declared"] is True
    assert receipt["raw_cache_decoded_f32_equality_recomputed"] is False


def test_binding_accepts_only_exact_train_or_heldout_split(tmp_path: Path) -> None:
    _binding_path, binding = _write_fixture(tmp_path)
    binding["target_split"] = "heldout"
    _reseal(binding)
    validated = binding_contract.validate_target_dataset_binding(
        binding,
        required_frame_counts=REQUIRED_FRAME_COUNTS,
    )
    assert validated["target_split"] == "heldout"

    binding["target_split"] = "validation"
    _reseal(binding)
    with pytest.raises(ValueError, match="exactly one of"):
        binding_contract.validate_target_dataset_binding(
            binding,
            required_frame_counts=REQUIRED_FRAME_COUNTS,
        )


def test_train_heldout_pair_receipt_rehashes_disjoint_caches_on_one_grid(
    tmp_path: Path,
) -> None:
    train_root = tmp_path / "train"
    heldout_root = tmp_path / "heldout"
    train_root.mkdir()
    heldout_root.mkdir()
    train_path, train = _write_fixture(train_root)
    heldout_path, heldout = _write_fixture(heldout_root)
    _rewrite_fixture_as_heldout(heldout_path, heldout)

    receipt = binding_contract.verify_train_heldout_target_dataset_pair(
        train_binding_path=train_path,
        heldout_binding_path=heldout_path,
        required_frame_counts=REQUIRED_FRAME_COUNTS,
    )

    assert receipt["schema"] == (
        binding_contract.TRAIN_HELDOUT_PAIR_RECEIPT_SCHEMA
    )
    assert receipt["train"]["view_ids"] == ["camera_00", "camera_01"]
    assert receipt["heldout"]["view_ids"] == ["camera_02", "camera_03"]
    assert receipt["train_heldout_view_sets_disjoint"] is True
    assert receipt["cache_payload_bytes_rehashed"] is True
    assert receipt["common_grid"]["stored_frame_indices"] == [0, 2, 4]
    assert receipt["pair_receipt_sha256"] == (
        binding_contract.canonical_payload_sha256(
            {
                key: value
                for key, value in receipt.items()
                if key != "pair_receipt_sha256"
            }
        )
    )

    heldout["camera"]["camera_generation_digest"] = "7" * 64
    _reseal(heldout)
    heldout_path.write_text(
        json.dumps(heldout, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="camera-generation/time grid"):
        binding_contract.verify_train_heldout_target_dataset_pair(
            train_binding_path=train_path,
            heldout_binding_path=heldout_path,
            required_frame_counts=REQUIRED_FRAME_COUNTS,
        )


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("binding_digest", "canonical digest"),
        ("decoded_mismatch", "raw/cache decoded"),
        ("payload_size", "mapped RGB8 storage"),
        ("view_order", "identical sorted order"),
        ("frame_endpoint", "physical endpoints"),
        ("frame_map_digest", "frame-map digest"),
        ("extra_key", "keys changed"),
        ("path_escape", "portable relative path"),
        ("path_control", "portable relative path"),
        ("lens_distortion", "require distortion coefficients"),
    ),
)
def test_binding_rejects_identity_or_common_grid_drift(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    _binding_path, original = _write_fixture(tmp_path)
    payload = copy.deepcopy(original)
    if case == "binding_digest":
        payload["binding_sha256"] = "0" * 64
    elif case == "decoded_mismatch":
        payload["cache"]["views"][0]["cache_decoded_f32_sha256"] = "9" * 64
    elif case == "payload_size":
        payload["cache"]["views"][0]["payload"]["size_bytes"] -= 1
    elif case == "view_order":
        payload["cache"]["views"].reverse()
    elif case == "frame_endpoint":
        frame_map = payload["logical_frame_maps"][0]
        frame_map["source_frame_indices"] = [2, 4]
        frame_map["logical_frame_map_sha256"] = (
            binding_contract.canonical_payload_sha256([2, 4])
        )
    elif case == "frame_map_digest":
        payload["logical_frame_maps"][0]["logical_frame_map_sha256"] = "8" * 64
    elif case == "extra_key":
        payload["unexpected"] = True
    elif case == "path_escape":
        payload["cache"]["views"][0]["payload"]["path_label"] = "../camera.rgb8"
    elif case == "path_control":
        payload["cache"]["views"][0]["payload"]["path_label"] = "cache/\x00camera.rgb8"
    elif case == "lens_distortion":
        payload["camera"]["lens_models"][0] = "opencv_fisheye"
    else:
        raise AssertionError(f"unknown mutation {case}")
    if case != "binding_digest":
        _reseal(payload)

    with pytest.raises((TypeError, ValueError), match=message):
        binding_contract.validate_target_dataset_binding(
            payload,
            required_frame_counts=REQUIRED_FRAME_COUNTS,
        )


@pytest.mark.parametrize(
    "case",
    ("payload_digest", "manifest_keys", "manifest_numeric_type", "manifest_oversize"),
)
def test_bound_cache_file_verification_rejects_on_disk_drift(
    tmp_path: Path,
    case: str,
) -> None:
    binding_path, payload = _write_fixture(tmp_path)
    if case == "payload_digest":
        payload_path = tmp_path / "camera_00.rgb8"
        contents = bytearray(payload_path.read_bytes())
        contents[0] ^= 0xFF
        payload_path.write_bytes(contents)
        message = "payload.*digest changed"
    elif case in {"manifest_keys", "manifest_numeric_type"}:
        manifest_path = tmp_path / "mapped_rgb8_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if case == "manifest_keys":
            manifest["unexpected"] = True
            message = "manifest keys changed"
        else:
            manifest["height"] = 2.0
            message = "manifest.height.*positive integer"
        manifest_bytes = json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        manifest_path.write_bytes(manifest_bytes)
        payload["cache"]["manifest"] = _file_identity(
            "mapped_rgb8_manifest.json",
            manifest_bytes,
        )
        _reseal(payload)
        binding_path.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
    elif case == "manifest_oversize":
        manifest_path = tmp_path / "mapped_rgb8_manifest.json"
        manifest_bytes = b" " * (binding_contract.MAXIMUM_MAPPED_MANIFEST_BYTES + 1)
        manifest_path.write_bytes(manifest_bytes)
        payload["cache"]["manifest"] = _file_identity(
            "mapped_rgb8_manifest.json",
            manifest_bytes,
        )
        _reseal(payload)
        binding_path.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        message = "cache manifest exceeds.*byte cap"
    else:
        raise AssertionError(f"unknown mutation {case}")

    with pytest.raises(ValueError, match=message):
        binding_contract.load_target_dataset_binding(
            binding_path,
            required_frame_counts=REQUIRED_FRAME_COUNTS,
            verify_cache_files=True,
        )


def test_binding_and_manifest_loaders_reject_duplicate_keys_and_oversize_json(
    tmp_path: Path,
) -> None:
    binding_path, payload = _write_fixture(tmp_path)
    raw_binding = binding_path.read_text(encoding="utf-8")
    binding_path.write_text(
        raw_binding[:-1] + ',"schema":"duplicate"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        binding_contract.load_target_dataset_binding(
            binding_path,
            required_frame_counts=REQUIRED_FRAME_COUNTS,
        )

    binding_path, payload = _write_fixture(tmp_path)
    manifest_path = tmp_path / "mapped_rgb8_manifest.json"
    raw_manifest = manifest_path.read_text(encoding="utf-8")
    duplicate_manifest = (
        raw_manifest[:-1]
        + ',"schema":"dynaworld.powerfoam_mapped_rgb8/v1"}'
    ).encode("utf-8")
    manifest_path.write_bytes(duplicate_manifest)
    payload["cache"]["manifest"] = _file_identity(
        "mapped_rgb8_manifest.json",
        duplicate_manifest,
    )
    _reseal(payload)
    binding_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        binding_contract.load_target_dataset_binding(
            binding_path,
            required_frame_counts=REQUIRED_FRAME_COUNTS,
            verify_cache_files=True,
        )

    binding_path.write_bytes(b" " * (binding_contract.MAXIMUM_BINDING_BYTES + 1))
    with pytest.raises(ValueError, match="byte cap"):
        binding_contract.load_target_dataset_binding(
            binding_path,
            required_frame_counts=REQUIRED_FRAME_COUNTS,
        )
