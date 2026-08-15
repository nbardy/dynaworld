from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import build_worldfoam_mapped_rgb8_cache as converter
import neural3d_mapped_rgb8_adapter as adapter


def _tensor_identity(shape: list[int], fill: str) -> dict[str, Any]:
    elements = 1
    for dimension in shape:
        elements *= dimension
    return {
        "dtype": "torch.float32",
        "shape": shape,
        "bytes": elements * 4,
        "byte_order": "native_little_endian",
        "layout": "contiguous_c_order",
        "sha256": fill * 64,
    }


def _conversion_limits() -> converter.WorldFoamMappedRgb8ConversionLimits:
    return converter.WorldFoamMappedRgb8ConversionLimits(
        maximum_raw_dataset_manifest_bytes=64 * 1024,
        maximum_raw_input_bytes_per_view=1024,
        maximum_total_raw_input_verification_bytes=1024 * 1024,
        maximum_total_decode_input_bytes=1024 * 1024,
        maximum_decoded_frame_bytes=1024,
        maximum_decode_hash_scratch_bytes=4096,
        maximum_payload_bytes_per_view=4096,
        maximum_total_payload_bytes=8192,
        maximum_transpose_scratch_bytes=4096,
        maximum_temporary_bytes_per_view=4096,
        maximum_total_output_and_temporary_bytes=128 * 1024,
        maximum_total_cache_verification_bytes=128 * 1024,
        maximum_mapped_manifest_bytes=16 * 1024,
        maximum_binding_bytes=64 * 1024,
    )


def _adapter_limits() -> adapter.Neural3dMappedRgb8AdapterLimits:
    return adapter.Neural3dMappedRgb8AdapterLimits(
        maximum_dataset_manifest_bytes=64 * 1024,
        maximum_poses_bounds_bytes=64 * 1024,
        maximum_adapter_source_bytes=4 * 1024 * 1024,
        maximum_total_source_verification_bytes=16 * 1024 * 1024,
        maximum_camera_tensor_bytes=64 * 1024,
        maximum_descriptor_bytes=64 * 1024,
        maximum_mp4_header_read_bytes_per_view=4096,
        maximum_total_mp4_header_read_bytes=8192,
        maximum_native_frame_bytes=4096,
        maximum_python_rgb_scratch_bytes=16384,
        maximum_decoded_native_frames_per_view=64,
    )


def _camera_binding(view_ids: tuple[str, ...], frame_count: int) -> dict[str, Any]:
    return {
        "view_ids": list(view_ids),
        "height": 2,
        "width": 3,
        "frame_times": _tensor_identity([frame_count, 1], "1"),
        "K": _tensor_identity([len(view_ids), 3, 3], "2"),
        "w2c": _tensor_identity([len(view_ids), frame_count, 4, 4], "3"),
        "lens_models": ["pinhole"] * len(view_ids),
        "distortions": None,
        "pose_source": adapter.NEURAL3D_POSE_SOURCE,
        "camera_generation_digest": "4" * 64,
    }


def test_endpoint_including_maps_reuse_the_fixed_dataset_grid() -> None:
    maps = adapter.endpoint_including_logical_frame_maps(
        tuple(range(300)),
        (8, 64, 300),
    )

    assert maps[0] == (8, (0, 42, 85, 128, 170, 213, 256, 299))
    assert maps[-1] == (300, tuple(range(300)))
    assert all(indices[0] == 0 and indices[-1] == 299 for _count, indices in maps)


def test_camera_generation_digest_is_common_but_tensors_remain_split_specific(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch
    import multicam_video_data

    camera_ordinals = {"cam00": 0.0, "cam01": 1.0, "cam02": 2.0}

    def camera_from_poses(
        _record: dict[str, Any],
        camera_name: str,
        **_kwargs: Any,
    ):
        ordinal = camera_ordinals[camera_name]
        K = torch.eye(3, dtype=torch.float32)
        K[0, 0] = 10.0 + ordinal
        c2w = torch.eye(4, dtype=torch.float32)
        c2w[0, 3] = ordinal
        return K, c2w

    monkeypatch.setattr(
        multicam_video_data,
        "neural_3d_camera_from_poses_bounds",
        camera_from_poses,
    )
    common = dict(
        record={"fps": 30.0},
        train_view_ids=("cam00", "cam01"),
        heldout_view_ids=("cam02",),
        anchor_camera="cam00",
        stored_frame_indices=(0, 2, 4),
        height=2,
        width=3,
        translation_scale=1.0,
        provenance_payload={"source_identity": "a" * 64},
    )

    train = adapter._build_camera_binding(
        **common,
        view_ids=("cam00", "cam01"),
    )
    heldout = adapter._build_camera_binding(
        **common,
        view_ids=("cam02",),
    )

    assert train["camera_generation_digest"] == heldout[
        "camera_generation_digest"
    ]
    assert train["frame_times"] == heldout["frame_times"]
    assert train["K"]["shape"] == [2, 3, 3]
    assert heldout["K"]["shape"] == [1, 3, 3]
    assert train["K"]["sha256"] != heldout["K"]["sha256"]


def test_offline_preflight_fails_closed_when_both_decoder_runtimes_are_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_pyav() -> Any:
        raise ImportError("fixture has no PyAV")

    monkeypatch.setattr(adapter, "_import_av", missing_pyav)
    monkeypatch.setattr(adapter, "_ffmpeg_runtime_version_payload", missing_pyav)
    preflight = adapter.neural3d_mapped_rgb8_offline_preflight()

    assert preflight["ready"] is False
    assert preflight["blockers"] == ["bounded_decoder_runtime_not_installed"]
    assert preflight["supported_target_splits"] == ["heldout", "train"]
    assert preflight["cache_layout"] == "height_width_frame_rgb_interleaved"
    assert preflight["chunked_selected_pixel_reads_supported_by_layout"] is True
    assert preflight["whole_video_materialized"] is False
    assert preflight["preflight_sha256"] == adapter._canonical_sha256(
        {
            key: value
            for key, value in preflight.items()
            if key != "preflight_sha256"
        }
    )


def test_offline_preflight_uses_bounded_ffmpeg_fallback_without_pyav(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_pyav() -> Any:
        raise ImportError("fixture has no PyAV")

    runtime = {
        "decoder_provenance": adapter.FFMPEG_DECODER_PROVENANCE,
        "runtime_sha256": "f" * 64,
    }
    monkeypatch.setattr(adapter, "_import_av", missing_pyav)
    monkeypatch.setattr(
        adapter,
        "_ffmpeg_runtime_version_payload",
        lambda: runtime,
    )

    preflight = adapter.neural3d_mapped_rgb8_offline_preflight()

    assert preflight["ready"] is True
    assert preflight["blockers"] == []
    assert preflight["runtime"] == runtime
    assert preflight["whole_video_materialized"] is False


def test_pyav_decoder_uses_one_bounded_handle_and_sparse_keyframe_seek(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    class Frame:
        def __init__(self, index: int) -> None:
            self.pts = index
            self.index = index

        def to_ndarray(self, *, format: str):
            assert format == "rgb24"
            return np.full((2, 3, 3), self.index, dtype=np.uint8)

    stream = SimpleNamespace(
        type="video",
        average_rate=Fraction(4, 1),
        time_base=Fraction(1, 4),
        start_time=0,
        frames=6,
        height=2,
        width=3,
        codec_context=SimpleNamespace(thread_count=0),
    )

    class Container:
        def __init__(self) -> None:
            self.streams = (stream,)
            self.position = 0
            self.seek_offsets: list[int] = []
            self.closed = False

        def decode(self, selected_stream):
            assert selected_stream is stream
            while self.position < 6:
                index = self.position
                self.position += 1
                yield Frame(index)

        def seek(self, offset: int, **kwargs: Any) -> None:
            assert kwargs == {
                "backward": True,
                "any_frame": False,
                "stream": stream,
            }
            self.seek_offsets.append(offset)
            self.position = max(0, offset - 2)

        def close(self) -> None:
            self.closed = True

    container = Container()
    opened_handles = []

    class Av:
        @staticmethod
        def open(handle, **kwargs: Any) -> Container:
            assert kwargs == {"mode": "r", "format": "mp4"}
            opened_handles.append(handle)
            handle.seek(0)
            assert handle.read(4) == b"fake"
            return container

    monkeypatch.setattr(adapter, "_import_av", lambda: Av)
    monkeypatch.setattr(
        adapter,
        "_runtime_version_payload",
        lambda: {"runtime_sha256": "a" * 64},
    )
    decoder = adapter.PyAvOpenFileRgb8Decoder(
        expected_view_id="cam00",
        selected_logical_frame_indices=(0, 5),
        source_frame_count=6,
        start_seconds=0.0,
        sample_fps=4.0,
        expected_native_fps_numerator=4,
        expected_native_fps_denominator=1,
        expected_native_frame_count=6,
        expected_native_height=2,
        expected_native_width=3,
        expected_stream_time_base_numerator=1,
        expected_stream_time_base_denominator=4,
        expected_stream_start_time=0,
        maximum_native_frame_bytes=18,
        maximum_python_rgb_scratch_bytes=126,
        maximum_decoded_native_frames=4,
        expected_runtime_sha256="a" * 64,
        maximum_sequential_gap_frames=1,
    )
    raw_path = tmp_path / "camera.mp4"
    raw_path.write_bytes(b"fake-mp4-payload")

    with raw_path.open("rb") as raw_handle:
        bounded = converter._BoundedDecodeFile(
            raw_handle,
            maximum_read_bytes=64,
        )
        frame0 = decoder.decode_rgb8_frame(
            bounded,
            view_id="cam00",
            source_frame_index=0,
            height=2,
            width=3,
            maximum_decoded_frame_bytes=18,
        )
        frame5 = decoder.decode_rgb8_frame(
            bounded,
            view_id="cam00",
            source_frame_index=5,
            height=2,
            width=3,
            maximum_decoded_frame_bytes=18,
        )
        decoder.close_open_file_decode(
            bounded,
            view_id="cam00",
            decode_completed=True,
        )

    assert opened_handles == [bounded]
    assert frame0 == bytes([0] * 18)
    assert frame5 == bytes([5] * 18)
    assert container.seek_offsets == [5]
    assert decoder.decoded_native_frame_count == 4
    assert stream.codec_context.thread_count == 1
    assert container.closed is True


def test_pyav_decoder_fails_before_open_when_runtime_provenance_drifts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        adapter,
        "_runtime_version_payload",
        lambda: {"runtime_sha256": "b" * 64},
    )
    decoder = adapter.PyAvOpenFileRgb8Decoder(
        expected_view_id="cam00",
        selected_logical_frame_indices=(0,),
        source_frame_count=1,
        start_seconds=0.0,
        sample_fps=30.0,
        expected_native_fps_numerator=30,
        expected_native_fps_denominator=1,
        expected_native_frame_count=1,
        expected_native_height=2,
        expected_native_width=3,
        expected_stream_time_base_numerator=1,
        expected_stream_time_base_denominator=30,
        expected_stream_start_time=0,
        maximum_native_frame_bytes=18,
        maximum_python_rgb_scratch_bytes=126,
        maximum_decoded_native_frames=1,
        expected_runtime_sha256="a" * 64,
    )

    with pytest.raises(RuntimeError, match="runtime drifted"):
        decoder.decode_rgb8_frame(
            SimpleNamespace(),
            view_id="cam00",
            source_frame_index=0,
            height=2,
            width=3,
            maximum_decoded_frame_bytes=18,
        )

    cleanup_handle = SimpleNamespace()
    decoder.close_open_file_decode(
        cleanup_handle,
        view_id="cam00",
        decode_completed=False,
    )
    with pytest.raises(ValueError, match="did not consume its exact frame plan"):
        decoder.close_open_file_decode(
            cleanup_handle,
            view_id="cam00",
            decode_completed=True,
        )


def test_pyav_decoder_preflights_upscaled_visible_scratch_before_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = SimpleNamespace(
        type="video",
        average_rate=Fraction(30, 1),
        time_base=Fraction(1, 30),
        start_time=0,
        frames=1,
        height=2,
        width=3,
        codec_context=SimpleNamespace(thread_count=0),
    )

    class Container:
        streams = (stream,)
        closed = False

        def decode(self, _stream: Any):
            raise AssertionError("scratch preflight unexpectedly reached frame decode")

        def close(self) -> None:
            self.closed = True

    container = Container()

    class Av:
        @staticmethod
        def open(_handle: Any, **kwargs: Any) -> Container:
            assert kwargs == {"mode": "r", "format": "mp4"}
            return container

    monkeypatch.setattr(adapter, "_import_av", lambda: Av)
    monkeypatch.setattr(
        adapter,
        "_runtime_version_payload",
        lambda: {"runtime_sha256": "a" * 64},
    )
    # native=18 bytes, output=72 bytes, so the declared conservative visible
    # upper bound is 3*18 + 4*72 = 342 bytes.  A 341-byte cap must fail first.
    decoder = adapter.PyAvOpenFileRgb8Decoder(
        expected_view_id="cam00",
        selected_logical_frame_indices=(0,),
        source_frame_count=1,
        start_seconds=0.0,
        sample_fps=30.0,
        expected_native_fps_numerator=30,
        expected_native_fps_denominator=1,
        expected_native_frame_count=1,
        expected_native_height=2,
        expected_native_width=3,
        expected_stream_time_base_numerator=1,
        expected_stream_time_base_denominator=30,
        expected_stream_start_time=0,
        maximum_native_frame_bytes=18,
        maximum_python_rgb_scratch_bytes=341,
        maximum_decoded_native_frames=1,
        expected_runtime_sha256="a" * 64,
    )

    with pytest.raises(MemoryError, match="logical RGB conversion scratch"):
        decoder.decode_rgb8_frame(
            SimpleNamespace(),
            view_id="cam00",
            source_frame_index=0,
            height=4,
            width=6,
            maximum_decoded_frame_bytes=72,
        )

    assert container.closed is True


def test_pyav_decoder_rejects_stream_header_drift_and_closes_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = SimpleNamespace(
        type="video",
        average_rate=Fraction(30, 1),
        time_base=Fraction(1, 24),
        start_time=0,
        frames=1,
        height=2,
        width=3,
        codec_context=SimpleNamespace(thread_count=0),
    )

    class Container:
        streams = (stream,)
        closed = False

        def close(self) -> None:
            self.closed = True

    container = Container()

    class Av:
        @staticmethod
        def open(_handle: Any, **kwargs: Any) -> Container:
            assert kwargs == {"mode": "r", "format": "mp4"}
            return container

    monkeypatch.setattr(adapter, "_import_av", lambda: Av)
    monkeypatch.setattr(
        adapter,
        "_runtime_version_payload",
        lambda: {"runtime_sha256": "a" * 64},
    )
    decoder = adapter.PyAvOpenFileRgb8Decoder(
        expected_view_id="cam00",
        selected_logical_frame_indices=(0,),
        source_frame_count=1,
        start_seconds=0.0,
        sample_fps=30.0,
        expected_native_fps_numerator=30,
        expected_native_fps_denominator=1,
        expected_native_frame_count=1,
        expected_native_height=2,
        expected_native_width=3,
        expected_stream_time_base_numerator=1,
        expected_stream_time_base_denominator=30,
        expected_stream_start_time=0,
        maximum_native_frame_bytes=18,
        maximum_python_rgb_scratch_bytes=126,
        maximum_decoded_native_frames=1,
        expected_runtime_sha256="a" * 64,
    )

    with pytest.raises(ValueError, match="time base differs"):
        decoder.decode_rgb8_frame(
            SimpleNamespace(),
            view_id="cam00",
            source_frame_index=0,
            height=2,
            width=3,
            maximum_decoded_frame_bytes=18,
        )

    assert container.closed is True


def test_prepare_adapter_binds_manifest_poses_runtime_native_frames_and_camera(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene = tmp_path / "dataset" / "scene"
    scene.mkdir(parents=True)
    (scene / "poses_bounds.npy").write_bytes(b"bounded-poses-fixture")
    (scene / "cam00.mp4").write_bytes(b"camera-zero")
    (scene / "cam01.mp4").write_bytes(b"camera-one")
    (scene / "cam02.mp4").write_bytes(b"camera-two-heldout")
    target_video = tmp_path / "dataset" / "target-cam01.mp4"
    target_video.write_bytes(b"camera-one-target-window")
    record = {
        "dataset": "neural_3d_video",
        "dataset_name": "neural3d_fixture",
        "dataset_scene_dir": str(scene),
        "sample_id": "neural3d_fixture_train2",
        "scene": "fixture_scene",
        "frame_count": 5,
        "fps": 4.0,
        "source_image_size": [2, 3],
        "source_camera": "cam00",
        "source_video_path": str(scene / "cam00.mp4"),
        "source_start_seconds": 0.0,
        "target_camera": "cam01",
        "target_video_path": str(target_video),
        "target_start_seconds": 0.25,
        "train_cameras": ["cam00", "cam01"],
        "heldout_cameras": ["cam02"],
        "anchor_camera": "cam00",
        "condition_camera": "cam00",
    }
    manifest = tmp_path / "dataset" / "manifest.jsonl"
    manifest.write_text(json.dumps(record, separators=(",", ":")) + "\n")
    runtime = {
        "decoder_provenance": adapter.PYAV_DECODER_PROVENANCE,
        "runtime_sha256": "a" * 64,
    }
    monkeypatch.setattr(adapter, "_runtime_version_payload", lambda: runtime)

    def probe(_path: Path, *, maximum_read_bytes: int):
        assert maximum_read_bytes == 4096
        return (
            {
                "native_fps_numerator": 30,
                "native_fps_denominator": 1,
                "native_frame_count": 300,
                "native_height": 2,
                "native_width": 3,
                "stream_time_base_numerator": 1,
                "stream_time_base_denominator": 30,
                "stream_start_time": 0,
            },
            128,
        )

    monkeypatch.setattr(adapter, "_probe_mp4_header", probe)
    monkeypatch.setattr(
        adapter,
        "_build_camera_binding",
        lambda _record, **kwargs: _camera_binding(
            kwargs["view_ids"],
            len(kwargs["stored_frame_indices"]),
        ),
    )

    prepare_kwargs = dict(
        repository_root=tmp_path,
        dataset_manifest_path=manifest,
        dataset_manifest_path_label="dataset/manifest.jsonl",
        sample_id=record["sample_id"],
        output_directory=tmp_path / "cache",
        height=2,
        width=3,
        stored_frame_indices=(0, 2, 4),
        required_frame_counts=(2, 3),
        conversion_limits=_conversion_limits(),
        adapter_limits=_adapter_limits(),
    )
    prepared = adapter.prepare_neural3d_mapped_rgb8_request(**prepare_kwargs)

    request = prepared.build_request
    assert prepared.target_split == "train"
    assert prepared.view_ids == ("cam00", "cam01")
    assert tuple(view.view_id for view in request.views) == ("cam00", "cam01")
    assert request.logical_frame_maps == ((2, (0, 4)), (3, (0, 2, 4)))
    assert request.camera == _camera_binding(("cam00", "cam01"), 3)
    assert request.raw_dataset_manifest_path == prepared.descriptor_path
    assert hashlib.sha256(prepared.descriptor_path.read_bytes()).hexdigest() == (
        prepared.descriptor_sha256
    )
    descriptor = json.loads(prepared.descriptor_path.read_text(encoding="utf-8"))
    assert descriptor == prepared.descriptor
    assert descriptor["dataset_manifest"]["sha256"] == hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()
    assert descriptor["native_frame_indices_by_view"] == {
        "cam00": [0, 15, 30],
        "cam01": [8, 22, 38],
    }
    assert descriptor["raw_video_path_labels"] == {
        "cam00": "dataset/scene/cam00.mp4",
        "cam01": "dataset/target-cam01.mp4",
    }
    assert descriptor["decoder"]["runtime"] == runtime
    assert descriptor["decoder"]["resource_contract"] == {
        "adapter_limits": dict(_adapter_limits().__dict__),
        "converter_limits": dict(_conversion_limits().__dict__),
        "maximum_sequential_gap_frames": 8,
        "whole_video_materialized": False,
        "maximum_retained_returned_rgb8_frames_per_view": 1,
        "codec_state_byte_cap_available": False,
        "python_scratch_cap_kind": "logical_visible_buffers_not_allocator_peak",
        "python_scratch_upper_bound_formula": (
            "3*native_rgb8_bytes+4*output_rgb8_bytes"
        ),
        "allocator_peak_measured": False,
        "process_rss_cap_enforced": False,
        "scope": "python_visible_buffers_io_and_decoded_frame_work",
    }
    assert descriptor["decoder"]["adapter_source"]["sha256"] == hashlib.sha256(
        Path(adapter.__file__).read_bytes()
    ).hexdigest()
    assert all(
        isinstance(view.decoder, adapter.PyAvOpenFileRgb8Decoder)
        for view in request.views
    )
    assert all(view.decoder.expected_native_fps == 30.0 for view in request.views)
    assert all(
        view.decoder.expected_native_frame_count == 300 for view in request.views
    )
    assert [view.decoder.start_seconds for view in request.views] == [0.0, 0.25]
    assert all(
        view.decoder.expected_stream_time_base_numerator == 1
        and view.decoder.expected_stream_time_base_denominator == 30
        and view.decoder.expected_stream_start_time == 0
        for view in request.views
    )

    heldout_prepared = adapter.prepare_neural3d_mapped_rgb8_request(
        **{
            **prepare_kwargs,
            "target_split": "heldout",
            "output_directory": tmp_path / "heldout-cache",
        }
    )
    heldout_request = heldout_prepared.build_request
    assert heldout_prepared.target_split == "heldout"
    assert heldout_prepared.view_ids == ("cam02",)
    assert heldout_request.target_split == "heldout"
    assert tuple(view.view_id for view in heldout_request.views) == ("cam02",)
    assert heldout_request.camera["view_ids"] == ["cam02"]
    assert set(heldout_request.camera["view_ids"]).isdisjoint(
        request.camera["view_ids"]
    )
    assert heldout_request.camera["frame_times"] == request.camera["frame_times"]
    assert heldout_request.camera["camera_generation_digest"] == request.camera[
        "camera_generation_digest"
    ]
    assert heldout_prepared.descriptor["target_split"] == "heldout"
    assert heldout_prepared.descriptor["view_ids"] == ["cam02"]
    assert heldout_prepared.descriptor["raw_video_path_labels"] == {
        "cam02": "dataset/scene/cam02.mp4"
    }
    assert heldout_prepared.descriptor["stored_logical_frame_indices"] == [0, 2, 4]

    with pytest.raises(ValueError, match="exactly one of"):
        adapter.prepare_neural3d_mapped_rgb8_request(
            **{
                **prepare_kwargs,
                "target_split": "validation",
                "output_directory": tmp_path / "invalid-split-cache",
            }
        )

    repeated_probe_calls = []
    monkeypatch.setattr(
        adapter,
        "_probe_mp4_header",
        lambda *_args, **_kwargs: repeated_probe_calls.append(True),
    )
    with pytest.raises(FileExistsError, match="cache output already exists"):
        adapter.prepare_neural3d_mapped_rgb8_request(**prepare_kwargs)
    assert repeated_probe_calls == []


def test_prepare_adapter_rejects_unordered_views_before_descriptor_or_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene = tmp_path / "dataset" / "scene"
    scene.mkdir(parents=True)
    (scene / "poses_bounds.npy").write_bytes(b"poses")
    (scene / "cam00.mp4").write_bytes(b"zero")
    (scene / "cam01.mp4").write_bytes(b"one")
    record = {
        "dataset": "neural_3d_video",
        "dataset_name": "neural3d_fixture",
        "dataset_scene_dir": "dataset/scene",
        "sample_id": "unordered",
        "scene": "fixture_scene",
        "frame_count": 3,
        "fps": 30.0,
        "source_image_size": [2, 3],
        "source_camera": "cam00",
        "source_video_path": "dataset/scene/cam00.mp4",
        "source_start_seconds": 0.0,
        "train_cameras": ["cam01", "cam00"],
        "heldout_cameras": ["cam02"],
        "anchor_camera": "cam00",
    }
    manifest = tmp_path / "dataset" / "manifest.jsonl"
    manifest.write_text(json.dumps(record) + "\n")
    probe_calls = []
    monkeypatch.setattr(
        adapter,
        "_probe_mp4_header",
        lambda *_args, **_kwargs: probe_calls.append(True),
    )

    with pytest.raises(ValueError, match="unique and sorted"):
        adapter.prepare_neural3d_mapped_rgb8_request(
            repository_root=tmp_path,
            dataset_manifest_path=manifest,
            dataset_manifest_path_label="dataset/manifest.jsonl",
            sample_id="unordered",
            output_directory=tmp_path / "cache",
            height=2,
            width=3,
            stored_frame_indices=(0, 1, 2),
            required_frame_counts=(2, 3),
            conversion_limits=_conversion_limits(),
            adapter_limits=_adapter_limits(),
        )

    assert probe_calls == []
    assert not (tmp_path / "cache").exists()

    record["train_cameras"] = ["cam00", "cam01"]
    record["heldout_cameras"] = ["cam01"]
    manifest.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="split overlaps"):
        adapter.prepare_neural3d_mapped_rgb8_request(
            repository_root=tmp_path,
            dataset_manifest_path=manifest,
            dataset_manifest_path_label="dataset/manifest.jsonl",
            sample_id="unordered",
            output_directory=tmp_path / "overlap-cache",
            height=2,
            width=3,
            stored_frame_indices=(0, 1, 2),
            required_frame_counts=(2, 3),
            conversion_limits=_conversion_limits(),
            adapter_limits=_adapter_limits(),
            target_split="heldout",
        )
    assert probe_calls == []
    assert not (tmp_path / "overlap-cache").exists()
