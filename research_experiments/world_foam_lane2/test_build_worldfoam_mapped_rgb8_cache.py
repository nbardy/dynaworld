from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import replace
from pathlib import Path
from typing import Any, BinaryIO

import pytest

import build_worldfoam_mapped_rgb8_cache as converter
import worldfoam_target_dataset_binding as binding_contract


HEIGHT = 2
WIDTH = 3
SOURCE_FRAME_COUNT = 5
STORED_FRAMES = (0, 2, 4)
REQUIRED_FRAME_COUNTS = (2, 3)


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


def _camera(view_ids: list[str]) -> dict[str, Any]:
    return {
        "view_ids": view_ids,
        "height": HEIGHT,
        "width": WIDTH,
        "frame_times": _tensor_identity([len(STORED_FRAMES), 1], "1"),
        "K": _tensor_identity([len(view_ids), 3, 3], "2"),
        "w2c": _tensor_identity(
            [len(view_ids), len(STORED_FRAMES), 4, 4],
            "3",
        ),
        "lens_models": ["pinhole"] * len(view_ids),
        "distortions": None,
        "pose_source": "bounded-converter-fixture",
        "camera_generation_digest": "4" * 64,
    }


def _raw_frame_major_bytes(view_offset: int) -> bytes:
    values = bytearray()
    for frame in range(SOURCE_FRAME_COUNT):
        for pixel in range(HEIGHT * WIDTH):
            for channel in range(3):
                values.append((view_offset + 37 * frame + 5 * pixel + channel) % 256)
    return bytes(values)


def _expected_pixel_time_payload(raw: bytes) -> bytes:
    pixel_count = HEIGHT * WIDTH
    result = bytearray()
    for pixel in range(pixel_count):
        for frame in STORED_FRAMES:
            offset = (frame * pixel_count + pixel) * 3
            result.extend(raw[offset : offset + 3])
    return bytes(result)


def _expected_decoded_f32_sha256(raw: bytes) -> str:
    pixel_count = HEIGHT * WIDTH
    digest = hashlib.sha256()
    for frame in STORED_FRAMES:
        for channel in range(3):
            for pixel in range(pixel_count):
                offset = (frame * pixel_count + pixel) * 3 + channel
                digest.update(struct.pack("<f", raw[offset] / 255.0))
    return digest.hexdigest()


class RecordingDecoder:
    provenance = "recording-same-open-file-decoder/v1"
    uses_supplied_handle_exclusively = True
    reads_only_through_bounded_handle_api = True

    def __init__(self) -> None:
        self.delegate = converter.FrameMajorRgb8OpenFileDecoder(SOURCE_FRAME_COUNT)
        self.calls: list[tuple[int, int, str, int]] = []

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
        self.calls.append((id(handle), handle.fileno(), view_id, source_frame_index))
        return self.delegate.decode_rgb8_frame(
            handle,
            view_id=view_id,
            source_frame_index=source_frame_index,
            height=height,
            width=width,
            maximum_decoded_frame_bytes=maximum_decoded_frame_bytes,
        )

    def close_open_file_decode(
        self,
        handle: BinaryIO,
        *,
        view_id: str,
        decode_completed: bool,
    ) -> None:
        self.delegate.close_open_file_decode(
            handle,
            view_id=view_id,
            decode_completed=decode_completed,
        )


def _fixture_request(
    root: Path,
) -> tuple[
    converter.WorldFoamMappedRgb8BuildRequest,
    tuple[RecordingDecoder, ...],
    dict[str, bytes],
]:
    dataset_root = root / "dataset"
    dataset_root.mkdir()
    raw_manifest = dataset_root / "manifest.json"
    raw_manifest.write_text('{"dataset":"fixture/public"}', encoding="utf-8")
    raw_payloads = {
        "camera_00": _raw_frame_major_bytes(0),
        "camera_01": _raw_frame_major_bytes(71),
    }
    decoders = (RecordingDecoder(), RecordingDecoder())
    views = []
    for (view_id, raw), decoder in zip(raw_payloads.items(), decoders, strict=True):
        raw_path = dataset_root / f"{view_id}.rgb8"
        raw_path.write_bytes(raw)
        views.append(
            converter.WorldFoamRawTargetView(
                view_id=view_id,
                raw_input_path=raw_path,
                raw_input_path_label=f"dataset/{view_id}.rgb8",
                payload_label=f"{view_id}.rgb8",
                decoder=decoder,
            )
        )
    frame_bytes = HEIGHT * WIDTH * 3
    payload_bytes = frame_bytes * len(STORED_FRAMES)
    total_payload_bytes = payload_bytes * len(views)
    raw_verification_bytes = raw_manifest.stat().st_size + 2 * sum(
        len(value) for value in raw_payloads.values()
    )
    limits = converter.WorldFoamMappedRgb8ConversionLimits(
        maximum_raw_dataset_manifest_bytes=4096,
        maximum_raw_input_bytes_per_view=max(len(value) for value in raw_payloads.values()),
        maximum_total_raw_input_verification_bytes=raw_verification_bytes,
        maximum_total_decode_input_bytes=len(views) * len(STORED_FRAMES) * frame_bytes,
        maximum_decoded_frame_bytes=frame_bytes,
        maximum_decode_hash_scratch_bytes=frame_bytes + 2 * HEIGHT * WIDTH * 4,
        maximum_payload_bytes_per_view=payload_bytes,
        maximum_total_payload_bytes=total_payload_bytes,
        maximum_transpose_scratch_bytes=24,
        maximum_temporary_bytes_per_view=2 * payload_bytes,
        maximum_total_output_and_temporary_bytes=(
            total_payload_bytes + payload_bytes + 4096 + 16384
        ),
        maximum_total_cache_verification_bytes=4 * total_payload_bytes + 4096,
        maximum_mapped_manifest_bytes=4096,
        maximum_binding_bytes=16384,
    )
    request = converter.WorldFoamMappedRgb8BuildRequest(
        output_directory=root / "cache",
        dataset_id="fixture/public_scene",
        target_split="train",
        raw_dataset_manifest_path=raw_manifest,
        raw_dataset_manifest_path_label="dataset/manifest.json",
        views=tuple(views),
        height=HEIGHT,
        width=WIDTH,
        stored_frame_indices=STORED_FRAMES,
        required_frame_counts=REQUIRED_FRAME_COUNTS,
        logical_frame_maps=((2, (0, 4)), (3, STORED_FRAMES)),
        camera=_camera(list(raw_payloads)),
        limits=limits,
    )
    return request, decoders, raw_payloads


def test_converter_emits_exact_pixel_time_payload_manifest_and_populated_binding(
    tmp_path: Path,
) -> None:
    request, decoders, raw_payloads = _fixture_request(tmp_path)

    receipt = converter.build_worldfoam_mapped_rgb8_cache(request)

    assert receipt.raw_cache_decoded_f32_equality_recomputed is True
    assert receipt.raw_files_hashed_and_decoded_through_same_open_handles is True
    assert receipt.cache_payloads_hashed_and_verified_through_same_open_handles is True
    assert receipt.exact_payload_bytes_per_view == HEIGHT * WIDTH * len(STORED_FRAMES) * 3
    assert receipt.exact_total_payload_bytes == 2 * receipt.exact_payload_bytes_per_view
    assert receipt.total_decode_input_bytes == (
        len(raw_payloads) * len(STORED_FRAMES) * HEIGHT * WIDTH * 3
    )
    for view_id, payload_path in zip(raw_payloads, receipt.payload_paths, strict=True):
        assert payload_path.read_bytes() == _expected_pixel_time_payload(raw_payloads[view_id])
    expected_digests = tuple(
        _expected_decoded_f32_sha256(raw) for raw in raw_payloads.values()
    )
    assert receipt.raw_decoded_f32_sha256s == expected_digests
    assert receipt.cache_decoded_f32_sha256s == expected_digests
    for decoder, view_id in zip(decoders, raw_payloads, strict=True):
        assert [call[3] for call in decoder.calls] == list(STORED_FRAMES)
        assert {call[0] for call in decoder.calls} == {decoder.calls[0][0]}
        assert {call[1] for call in decoder.calls} == {decoder.calls[0][1]}
        assert {call[2] for call in decoder.calls} == {view_id}

    manifest = json.loads(receipt.manifest_path.read_text(encoding="utf-8"))
    assert manifest == {
        "schema": "dynaworld.powerfoam_mapped_rgb8/v1",
        "layout": "height_width_frame_rgb_interleaved",
        "dtype": "uint8",
        "height": HEIGHT,
        "width": WIDTH,
        "stored_frame_indices": list(STORED_FRAMES),
        "views": [
            {
                "view_id": view_id,
                "payload": f"{view_id}.rgb8",
                "payload_bytes": receipt.exact_payload_bytes_per_view,
                "payload_sha256": payload_sha256,
            }
            for view_id, payload_sha256 in zip(
                raw_payloads,
                receipt.payload_sha256s,
                strict=True,
            )
        ],
    }
    binding = binding_contract.load_target_dataset_binding(
        receipt.binding_path,
        required_frame_counts=REQUIRED_FRAME_COUNTS,
        verify_cache_files=True,
    )
    assert binding["binding_sha256"] == receipt.binding_sha256
    assert binding["converter"]["provenance"] == converter.CONVERTER_PROVENANCE
    assert binding["converter"]["source_sha256"] == hashlib.sha256(
        Path(converter.__file__).read_bytes()
    ).hexdigest()
    assert binding["raw_dataset_manifest"] == {
        "path_label": request.raw_dataset_manifest_path_label,
        "size_bytes": request.raw_dataset_manifest_path.stat().st_size,
        "sha256": hashlib.sha256(
            request.raw_dataset_manifest_path.read_bytes()
        ).hexdigest(),
    }
    assert binding["raw_views"] == [
        {
            "view_id": view.view_id,
            "raw_input": {
                "path_label": view.raw_input_path_label,
                "size_bytes": len(raw_payloads[view.view_id]),
                "sha256": hashlib.sha256(raw_payloads[view.view_id]).hexdigest(),
            },
            "raw_decoded_f32_sha256": expected_digest,
        }
        for view, expected_digest in zip(request.views, expected_digests, strict=True)
    ]
    assert binding["camera"] == request.camera
    assert binding["logical_frame_maps"] == [
        {
            "frame_count": count,
            "source_frame_indices": list(indices),
            "logical_frame_map_sha256": binding_contract.canonical_payload_sha256(
                list(indices)
            ),
        }
        for count, indices in request.logical_frame_maps
    ]
    assert [item["raw_decoded_f32_sha256"] for item in binding["raw_views"]] == list(
        expected_digests
    )
    assert [
        item["cache_decoded_f32_sha256"] for item in binding["cache"]["views"]
    ] == list(expected_digests)


def test_converter_preserves_heldout_split_and_rejects_any_other_label(
    tmp_path: Path,
) -> None:
    request, _decoders, _raw_payloads = _fixture_request(tmp_path)
    heldout_request = replace(
        request,
        target_split="heldout",
        output_directory=tmp_path / "heldout-cache",
    )

    receipt = converter.build_worldfoam_mapped_rgb8_cache(heldout_request)
    binding = binding_contract.load_target_dataset_binding(
        receipt.binding_path,
        required_frame_counts=REQUIRED_FRAME_COUNTS,
        verify_cache_files=True,
    )
    assert binding["target_split"] == "heldout"

    invalid_root = tmp_path / "invalid"
    invalid_root.mkdir()
    invalid_request, invalid_decoders, _raw_payloads = _fixture_request(invalid_root)
    with pytest.raises(ValueError, match="exactly one of"):
        converter.build_worldfoam_mapped_rgb8_cache(
            replace(invalid_request, target_split="validation")
        )
    assert all(not decoder.calls for decoder in invalid_decoders)
    assert not invalid_request.output_directory.exists()


@pytest.mark.parametrize(
    "limit_name",
    (
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
    ),
)
def test_converter_byte_caps_fail_before_decoder_or_output_creation(
    tmp_path: Path,
    limit_name: str,
) -> None:
    request, decoders, _raw_payloads = _fixture_request(tmp_path)
    lowered = replace(request.limits, **{limit_name: 1})

    with pytest.raises(MemoryError):
        converter.build_worldfoam_mapped_rgb8_cache(replace(request, limits=lowered))

    assert all(not decoder.calls for decoder in decoders)
    assert not request.output_directory.exists() or not tuple(request.output_directory.iterdir())


def test_converter_caps_payload_plus_one_active_spool_per_view(
    tmp_path: Path,
) -> None:
    request, decoders, _raw_payloads = _fixture_request(tmp_path)
    payload_bytes = HEIGHT * WIDTH * len(STORED_FRAMES) * 3
    limits = replace(
        request.limits,
        maximum_temporary_bytes_per_view=2 * payload_bytes - 1,
    )

    with pytest.raises(MemoryError, match="per-view temporary storage"):
        converter.build_worldfoam_mapped_rgb8_cache(
            replace(request, limits=limits)
        )

    assert all(not decoder.calls for decoder in decoders)
    assert not request.output_directory.exists()


def test_converter_rejects_decoder_without_bounded_handle_contract_before_output(
    tmp_path: Path,
) -> None:
    request, decoders, _raw_payloads = _fixture_request(tmp_path)

    class UnsealedDecoder(RecordingDecoder):
        provenance = "unsealed-decoder-fixture/v1"
        reads_only_through_bounded_handle_api = False

    unsealed = UnsealedDecoder()
    views = (replace(request.views[0], decoder=unsealed), *request.views[1:])

    with pytest.raises(ValueError, match="bounded open-file API"):
        converter.build_worldfoam_mapped_rgb8_cache(replace(request, views=views))

    assert not unsealed.calls
    assert all(not decoder.calls for decoder in decoders)
    assert not request.output_directory.exists()


def test_converter_rejects_decoder_without_close_boundary_before_output(
    tmp_path: Path,
) -> None:
    request, decoders, _raw_payloads = _fixture_request(tmp_path)

    class MissingCloseDecoder:
        provenance = "missing-close-boundary-fixture/v1"
        uses_supplied_handle_exclusively = True
        reads_only_through_bounded_handle_api = True

        def decode_rgb8_frame(self, _handle: BinaryIO, **_kwargs: Any) -> bytes:
            raise AssertionError("decoder without close boundary unexpectedly ran")

    views = (
        replace(request.views[0], decoder=MissingCloseDecoder()),
        *request.views[1:],
    )

    with pytest.raises(TypeError, match="no decoder close boundary"):
        converter.build_worldfoam_mapped_rgb8_cache(replace(request, views=views))

    assert all(not decoder.calls for decoder in decoders)
    assert not request.output_directory.exists()


def test_decoder_read_budget_rejects_overread_and_cleans_partial_outputs(
    tmp_path: Path,
) -> None:
    request, _decoders, _raw_payloads = _fixture_request(tmp_path)

    class OverreadDecoder(RecordingDecoder):
        provenance = "overread-decoder-fixture/v1"

        def decode_rgb8_frame(self, handle: BinaryIO, **kwargs: Any) -> bytes:
            handle.read(getattr(handle, "maximum_read_bytes") + 1)
            raise AssertionError("bounded decoder read unexpectedly succeeded")

    overread = OverreadDecoder()
    views = (replace(request.views[0], decoder=overread), *request.views[1:])

    with pytest.raises(MemoryError, match="before read"):
        converter.build_worldfoam_mapped_rgb8_cache(replace(request, views=views))

    assert not (request.output_directory / request.binding_label).exists()
    assert not request.output_directory.exists() or not tuple(request.output_directory.iterdir())


def test_converter_rejects_raw_mutation_through_the_held_open_identity(
    tmp_path: Path,
) -> None:
    request, _decoders, _raw_payloads = _fixture_request(tmp_path)
    raw_path = request.views[0].raw_input_path

    class MutatingDecoder(RecordingDecoder):
        provenance = "mutating-decoder-fixture/v1"

        def decode_rgb8_frame(self, handle: BinaryIO, **kwargs: Any) -> bytes:
            decoded = super().decode_rgb8_frame(handle, **kwargs)
            if len(self.calls) == 1:
                with raw_path.open("r+b") as mutator:
                    first = mutator.read(1)
                    mutator.seek(0)
                    mutator.write(bytes((first[0] ^ 0xFF,)))
                    mutator.flush()
                    os.fsync(mutator.fileno())
            return decoded

    mutating = MutatingDecoder()
    views = (replace(request.views[0], decoder=mutating), *request.views[1:])

    with pytest.raises(ValueError, match="changed while decoding"):
        converter.build_worldfoam_mapped_rgb8_cache(replace(request, views=views))

    assert not (request.output_directory / request.binding_label).exists()
    assert not request.output_directory.exists() or not tuple(request.output_directory.iterdir())


def test_converter_rejects_short_decoder_output_without_publishing(
    tmp_path: Path,
) -> None:
    request, _decoders, _raw_payloads = _fixture_request(tmp_path)

    class ShortDecoder(RecordingDecoder):
        provenance = "short-decoder-fixture/v1"

        def decode_rgb8_frame(self, handle: BinaryIO, **kwargs: Any) -> bytes:
            return super().decode_rgb8_frame(handle, **kwargs)[:-1]

    views = (replace(request.views[0], decoder=ShortDecoder()), *request.views[1:])

    with pytest.raises(ValueError, match="exact RGB8 frame size"):
        converter.build_worldfoam_mapped_rgb8_cache(replace(request, views=views))

    assert not (request.output_directory / request.binding_label).exists()
    assert not request.output_directory.exists() or not tuple(request.output_directory.iterdir())


def test_build_plan_is_strict_and_preserves_frame_camera_and_path_identities(
    tmp_path: Path,
) -> None:
    request, _decoders, _raw_payloads = _fixture_request(tmp_path)
    plan = {
        "schema": converter.BUILD_PLAN_SCHEMA,
        "dataset_id": request.dataset_id,
        "target_split": request.target_split,
        "raw_dataset_manifest": {
            "path": "dataset/manifest.json",
            "path_label": request.raw_dataset_manifest_path_label,
        },
        "height": request.height,
        "width": request.width,
        "stored_frame_indices": list(request.stored_frame_indices),
        "required_frame_counts": list(request.required_frame_counts),
        "logical_frame_maps": [
            {
                "frame_count": count,
                "source_frame_indices": list(indices),
            }
            for count, indices in request.logical_frame_maps
        ],
        "views": [
            {
                "view_id": view.view_id,
                "raw_input_path": f"dataset/{view.view_id}.rgb8",
                "raw_input_path_label": view.raw_input_path_label,
                "payload_label": view.payload_label,
                "raw_layout": converter.RAW_FRAME_MAJOR_RGB8_LAYOUT,
                "source_frame_count": SOURCE_FRAME_COUNT,
            }
            for view in request.views
        ],
        "camera": request.camera,
        "limits": dict(request.limits.__dict__),
        "mapped_manifest_label": request.mapped_manifest_label,
        "binding_label": request.binding_label,
    }
    resolved = converter.request_from_build_plan(
        plan,
        plan_directory=tmp_path,
        output_directory=tmp_path / "plan-cache",
    )
    assert resolved.stored_frame_indices == request.stored_frame_indices
    assert resolved.logical_frame_maps == request.logical_frame_maps
    assert resolved.camera == request.camera
    assert tuple(view.view_id for view in resolved.views) == tuple(
        view.view_id for view in request.views
    )
    assert all(
        isinstance(view.decoder, converter.FrameMajorRgb8OpenFileDecoder)
        for view in resolved.views
    )

    with pytest.raises(ValueError, match="keys changed"):
        converter.request_from_build_plan(
            {**plan, "unexpected": True},
            plan_directory=tmp_path,
            output_directory=tmp_path / "bad-cache",
        )
