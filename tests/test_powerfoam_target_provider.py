from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import multicam_val_data
import multicam_video_data
import powerfoam_training_data as training_data
import pytest
import torch
from camera import CameraSpec
from multicam_video_data import MulticamVideoFrameSource
from paper_training_protocol import normalize_image_size, resize_video_frames, tensor_content_identity
from PIL import Image
from powerfoam_training_data import (
    MappedRgb8PowerFoamTargetSource,
    PathPowerFoamTargetSource,
    PowerFoamTargetProvider,
    ResidentPowerFoamTargetSource,
    VideoSeekPowerFoamTargetSource,
)
from sequence_data import load_frame_sequence


def _normalized_frames(*, views: int = 2, frames: int = 3, height: int = 4, width: int = 5) -> torch.Tensor:
    values = torch.arange(views * frames * 3 * height * width, dtype=torch.float32)
    return values.reshape(views, frames, 3, height, width) / float(values.numel())


def _camera() -> CameraSpec:
    return CameraSpec(
        fx=4.0,
        fy=4.0,
        cx=2.0,
        cy=2.0,
        camera_to_world=torch.eye(4),
    )


def _write_mapped_rgb8_fixture(
    root: Path,
    frames_u8: torch.Tensor,
    *,
    view_ids: tuple[str, ...],
    stored_frame_indices: tuple[int, ...],
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    if tuple(frames_u8.shape[:2]) != (len(view_ids), len(stored_frame_indices)):
        raise ValueError("mapped RGB8 fixture dimensions changed")
    height, width = (int(value) for value in frames_u8.shape[-2:])
    views = []
    for view_index, view_id in enumerate(view_ids):
        payload_path = root / f"{view_id}.rgb8"
        payload = (
            frames_u8[view_index]
            .permute(2, 3, 0, 1)
            .contiguous()
            .numpy()
            .tobytes()
        )
        payload_path.write_bytes(payload)
        views.append(
            {
                "view_id": view_id,
                "payload": payload_path.name,
                "payload_bytes": len(payload),
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    manifest = {
        "schema": "dynaworld.powerfoam_mapped_rgb8/v1",
        "layout": "height_width_frame_rgb_interleaved",
        "dtype": "uint8",
        "height": height,
        "width": width,
        "stored_frame_indices": list(stored_frame_indices),
        "views": views,
    }
    manifest_path = root / "mapped_rgb8_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return manifest_path


def test_target_provider_selects_sampler_flat_indices_without_changing_rgb_values() -> None:
    frames = _normalized_frames()
    provider = PowerFoamTargetProvider.from_resident_frames(
        frames,
        device=torch.device("cpu"),
    )

    selected = provider.select(torch.tensor([5, 0, 3, 3]))

    expected = torch.stack((frames[1, 2], frames[0, 0], frames[1, 0], frames[1, 0]))
    assert torch.equal(selected, expected)
    assert selected.shape == (4, 3, 4, 5)
    assert selected.dtype == torch.float32
    assert provider.view_count == 2
    assert provider.frame_count == 3
    assert provider.sample_count == 6


def test_resident_selected_pixel_read_preserves_arbitrary_order_and_duplicates() -> None:
    frames = _normalized_frames()
    provider = PowerFoamTargetProvider(
        source=ResidentPowerFoamTargetSource(frames),
        device=torch.device("cpu"),
    )

    read = provider.select_view_frame_pixels_cpu(
        (1, 0, 1, 1, 0),
        (2, 0, 2, 1, 0),
        (19, 0, 19, 7, 0),
        maximum_source_decode_tensor_bytes=10_000,
    )

    expected = torch.stack(
        (
            frames[1, 2, :, 3, 4],
            frames[0, 0, :, 0, 0],
            frames[1, 2, :, 3, 4],
            frames[1, 1, :, 1, 2],
            frames[0, 0, :, 0, 0],
        )
    )
    assert torch.equal(read.rgb_f32_cpu, expected)
    assert read.selection_mode == "direct_pixels"
    assert read.acceptance_capable
    assert read.full_frame_materialization_count == 0
    assert read.maximum_full_frame_materialization_tensor_bytes == 0
    read.assert_valid(
        expected_observation_count=5,
        full_frame_tensor_bytes=3 * 4 * 5 * 4,
    )


def test_mapped_rgb8_selected_pixels_preserve_view_frame_map_order_and_duplicates(
    tmp_path,
) -> None:
    frames_u8 = torch.arange(2 * 3 * 3 * 4 * 5, dtype=torch.int64)
    frames_u8 = frames_u8.remainder(256).to(torch.uint8).reshape(2, 3, 3, 4, 5)
    manifest_path = _write_mapped_rgb8_fixture(
        tmp_path,
        frames_u8,
        view_ids=("cam_a", "cam_b"),
        stored_frame_indices=(0, 2, 4),
    )
    selected_u8 = frames_u8.index_select(0, torch.tensor((1, 0))).index_select(
        1,
        torch.tensor((2, 0)),
    )
    canonical_selected = training_data._normalize_rgb8_numpy_to_cpu_f32(
        selected_u8.numpy()
    )
    canonical_identity = tensor_content_identity(canonical_selected)

    class IdentityDelegate:
        view_count = 2
        frame_count = 2
        height = 4
        width = 5

        def select_view_frames(self, _views, _frames):
            raise AssertionError("selected-pixel cache used its full-frame delegate")

        def tensor_content_identity(self):
            return canonical_identity

        def residency(self):
            return {}

    source = MappedRgb8PowerFoamTargetSource.from_manifest(
        manifest_path,
        maximum_mapped_payload_bytes=10_000,
        maximum_total_payload_verification_bytes=100_000,
        expected_view_ids=("cam_b", "cam_a"),
        logical_frame_indices=(4, 0),
        full_frame_source=IdentityDelegate(),
    )
    provider = PowerFoamTargetProvider(source=source, device=torch.device("cpu"))

    read = provider.select_view_frame_pixels_cpu(
        (0, 1, 0, 0, 1),
        (0, 1, 0, 1, 1),
        (19, 0, 19, 7, 0),
        maximum_source_decode_tensor_bytes=10_000,
    )

    expected_u8 = torch.stack(
        (
            frames_u8[1, 2, :, 3, 4],
            frames_u8[0, 0, :, 0, 0],
            frames_u8[1, 2, :, 3, 4],
            frames_u8[1, 0, :, 1, 2],
            frames_u8[0, 0, :, 0, 0],
        )
    )
    expected = training_data._normalize_rgb8_numpy_to_cpu_f32(
        expected_u8.numpy()
    )
    assert torch.equal(read.rgb_f32_cpu, expected)
    assert read.selection_mode == "direct_pixels"
    assert read.acceptance_capable
    assert read.full_frame_materialization_count == 0
    assert read.transient_mapped_address_space_bytes == 4 * 5 * 3 * 3
    assert read.maximum_requested_unique_mapped_page_count >= 1
    assert read.maximum_requested_mapped_page_bytes_upper_bound == (
        read.maximum_requested_unique_mapped_page_count
        * read.mapped_page_size_bytes
    )
    assert (
        read.total_requested_unique_mapped_page_count
        >= read.maximum_requested_unique_mapped_page_count
    )
    assert read.total_requested_mapped_page_bytes_upper_bound == (
        read.total_requested_unique_mapped_page_count
        * read.mapped_page_size_bytes
    )
    assert read.mapping_closed_before_return
    assert source.residency()["resident_bytes"] == 0
    assert source.residency()["maximum_mapped_payload_bytes"] == 10_000
    assert source.residency()["maximum_total_payload_verification_bytes"] == 100_000
    assert source.residency()["mapping_lifetime"] == "one_selected_pixel_read"
    assert source.residency()["logical_frame_map_sha256"]
    assert source.tensor_content_identity() is canonical_identity


def test_mapped_rgb8_normalization_matches_canonical_numpy_decoder_for_all_bytes(
    tmp_path,
) -> None:
    import numpy as np

    byte_values = torch.arange(256, dtype=torch.int16).to(torch.uint8)
    byte_values = byte_values.reshape(1, 1, 1, 16, 16)
    frames_u8 = byte_values.expand(1, 1, 3, 16, 16).contiguous()
    manifest_path = _write_mapped_rgb8_fixture(
        tmp_path,
        frames_u8,
        view_ids=("cam",),
        stored_frame_indices=(0,),
    )
    source = MappedRgb8PowerFoamTargetSource.from_manifest(
        manifest_path,
        maximum_mapped_payload_bytes=10_000,
        maximum_total_payload_verification_bytes=100_000,
    )

    read = source.select_view_frame_pixels_cpu(
        (0,) * 256,
        (0,) * 256,
        tuple(range(256)),
        maximum_source_decode_tensor_bytes=256 * 70,
    )

    canonical = torch.from_numpy(
        np.asarray(frames_u8[0, 0].permute(1, 2, 0).reshape(-1, 3).numpy(), dtype=np.float32)
        / 255.0
    )
    assert torch.equal(read.rgb_f32_cpu, canonical)


def test_mapped_rgb8_budget_fails_before_opening_mapping(tmp_path, monkeypatch) -> None:
    frames_u8 = torch.zeros((1, 2, 3, 2, 3), dtype=torch.uint8)
    manifest_path = _write_mapped_rgb8_fixture(
        tmp_path,
        frames_u8,
        view_ids=("cam",),
        stored_frame_indices=(0, 1),
    )
    source = MappedRgb8PowerFoamTargetSource.from_manifest(
        manifest_path,
        maximum_mapped_payload_bytes=10_000,
        maximum_total_payload_verification_bytes=100_000,
    )

    def forbidden_mapping(*_args, **_kwargs):
        raise AssertionError("mapping opened before source scratch preflight")

    monkeypatch.setattr(training_data.mmap, "mmap", forbidden_mapping)
    with pytest.raises(MemoryError, match="source-decode budget"):
        source.select_view_frame_pixels_cpu(
            (0, 0),
            (0, 1),
            (0, 5),
            maximum_source_decode_tensor_bytes=133,
        )


def test_mapped_rgb8_manifest_and_payload_binding_fail_closed(tmp_path) -> None:
    frames_u8 = torch.zeros((1, 2, 3, 2, 3), dtype=torch.uint8)
    manifest_path = _write_mapped_rgb8_fixture(
        tmp_path,
        frames_u8,
        view_ids=("cam",),
        stored_frame_indices=(0, 1),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["views"][0]["payload_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="payload digest changed"):
        MappedRgb8PowerFoamTargetSource.from_manifest(
            manifest_path,
            maximum_mapped_payload_bytes=10_000,
            maximum_total_payload_verification_bytes=100_000,
        )


def test_mapped_rgb8_manifest_is_strict_and_caps_payload_before_hash_scan(
    tmp_path,
    monkeypatch,
) -> None:
    frames_u8 = torch.zeros((1, 2, 3, 2, 3), dtype=torch.uint8)
    extra_key_manifest = _write_mapped_rgb8_fixture(
        tmp_path / "extra_key",
        frames_u8,
        view_ids=("cam",),
        stored_frame_indices=(0, 1),
    )
    payload = json.loads(extra_key_manifest.read_text(encoding="utf-8"))
    payload["unbound_note"] = "not part of the cache identity"
    extra_key_manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest keys changed"):
        MappedRgb8PowerFoamTargetSource.from_manifest(
            extra_key_manifest,
            maximum_mapped_payload_bytes=10_000,
            maximum_total_payload_verification_bytes=100_000,
        )

    path_escape_manifest = _write_mapped_rgb8_fixture(
        tmp_path / "path_escape",
        frames_u8,
        view_ids=("cam",),
        stored_frame_indices=(0, 1),
    )
    payload = json.loads(path_escape_manifest.read_text(encoding="utf-8"))
    payload["views"][0]["payload"] = "../cam.rgb8"
    path_escape_manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid payload metadata"):
        MappedRgb8PowerFoamTargetSource.from_manifest(
            path_escape_manifest,
            maximum_mapped_payload_bytes=10_000,
            maximum_total_payload_verification_bytes=100_000,
        )

    control_path_manifest = _write_mapped_rgb8_fixture(
        tmp_path / "control_path",
        frames_u8,
        view_ids=("cam",),
        stored_frame_indices=(0, 1),
    )
    payload = json.loads(control_path_manifest.read_text(encoding="utf-8"))
    payload["views"][0]["payload"] = "cache/\x00cam.rgb8"
    control_path_manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid payload metadata"):
        MappedRgb8PowerFoamTargetSource.from_manifest(
            control_path_manifest,
            maximum_mapped_payload_bytes=10_000,
            maximum_total_payload_verification_bytes=100_000,
        )

    capped_manifest = _write_mapped_rgb8_fixture(
        tmp_path / "capped",
        frames_u8,
        view_ids=("cam",),
        stored_frame_indices=(0, 1),
    )

    def forbidden_hash_scan(_handle):
        raise AssertionError("payload hash scan began before the address-space cap")

    monkeypatch.setattr(training_data, "_streaming_open_file_sha256", forbidden_hash_scan)
    with pytest.raises(MemoryError, match="address-space cap"):
        MappedRgb8PowerFoamTargetSource.from_manifest(
            capped_manifest,
            maximum_mapped_payload_bytes=35,
            maximum_total_payload_verification_bytes=100_000,
        )
    with pytest.raises(MemoryError, match="verification-I/O cap"):
        MappedRgb8PowerFoamTargetSource.from_manifest(
            capped_manifest,
            maximum_mapped_payload_bytes=10_000,
            maximum_total_payload_verification_bytes=35,
        )


def test_mapped_rgb8_rejects_payload_mutation_after_construction(tmp_path) -> None:
    frames_u8 = torch.zeros((1, 2, 3, 2, 3), dtype=torch.uint8)
    manifest_path = _write_mapped_rgb8_fixture(
        tmp_path,
        frames_u8,
        view_ids=("cam",),
        stored_frame_indices=(0, 1),
    )
    source = MappedRgb8PowerFoamTargetSource.from_manifest(
        manifest_path,
        maximum_mapped_payload_bytes=10_000,
        maximum_total_payload_verification_bytes=100_000,
    )
    payload_path = tmp_path / "cam.rgb8"
    payload = bytearray(payload_path.read_bytes())
    payload[0] = 1
    payload_path.write_bytes(payload)

    with pytest.raises(ValueError, match="changed after manifest verification"):
        source.select_view_frame_pixels_cpu(
            (0,),
            (0,),
            (0,),
            maximum_source_decode_tensor_bytes=1_000,
        )


def test_frame_only_selected_pixel_fallback_is_explicitly_not_acceptance_capable() -> None:
    class FrameOnlySource:
        view_count = 1
        frame_count = 2
        height = 2
        width = 3

        def __init__(self) -> None:
            self.calls = 0

        def select_view_frames(self, view_indices, frame_indices):
            self.calls += 1
            frames = torch.empty(
                (len(view_indices), 3, self.height, self.width),
                dtype=torch.float32,
            )
            for output, frame in zip(frames, frame_indices, strict=True):
                output.copy_(
                    torch.arange(18, dtype=torch.float32).reshape(3, 2, 3)
                    + 100 * frame
                )
            return frames

        def residency(self):
            return {
                "source_kind": "frame_only_fixture",
                "source_device": "disk",
                "logical_bytes": 144,
                "resident_bytes": 0,
                "full_source_resident": False,
                "disk_lazy_decode": True,
            }

    source = FrameOnlySource()
    provider = PowerFoamTargetProvider(
        source=source,
        device=torch.device("cpu"),
    )

    with pytest.raises(MemoryError, match="source-decode budget"):
        provider.select_view_frame_pixels_cpu(
            (0, 0, 0, 0),
            (1, 0, 1, 1),
            (5, 0, 5, 1),
            maximum_source_decode_tensor_bytes=71,
        )
    assert source.calls == 0

    read = provider.select_view_frame_pixels_cpu(
        (0, 0, 0, 0),
        (1, 0, 1, 1),
        (5, 0, 5, 1),
        maximum_source_decode_tensor_bytes=10_000,
    )

    assert torch.equal(
        read.rgb_f32_cpu,
        torch.tensor(
            (
                (105.0, 111.0, 117.0),
                (0.0, 6.0, 12.0),
                (105.0, 111.0, 117.0),
                (101.0, 107.0, 113.0),
            ),
            dtype=torch.float32,
        ),
    )
    assert read.selection_mode == "full_frame_fallback"
    assert not read.acceptance_capable
    assert read.full_frame_materialization_count == 2
    assert read.maximum_full_frame_materialization_tensor_bytes == 72
    assert source.calls == 2


def test_target_provider_reads_only_selected_view_frame_pairs_and_supports_eval_resize() -> None:
    class RecordingSource:
        view_count = 4
        frame_count = 6
        height = 4
        width = 5

        def __init__(self) -> None:
            self.calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

        def select_view_frames(
            self,
            view_indices: tuple[int, ...],
            frame_indices: tuple[int, ...],
        ) -> torch.Tensor:
            self.calls.append((view_indices, frame_indices))
            values = [
                torch.full((3, self.height, self.width), view * 10 + frame, dtype=torch.float32)
                for view, frame in zip(view_indices, frame_indices, strict=True)
            ]
            return torch.stack(values)

        def residency(self) -> dict[str, Any]:
            return {
                "source_kind": "fixture_lazy_source",
                "source_device": "disk",
                "logical_bytes": 0,
                "resident_bytes": 0,
                "full_source_resident": False,
                "disk_lazy_decode": True,
            }

    source = RecordingSource()
    provider = PowerFoamTargetProvider(source=source, device=torch.device("cpu"))

    selected = provider.select_view_frames(
        torch.tensor([3, 0]),
        torch.tensor([5, 2]),
        height=2,
        width=3,
        device=torch.device("cpu"),
    )

    assert source.calls == [((3, 0), (5, 2))]
    assert selected.shape == (2, 3, 2, 3)
    assert torch.equal(selected[0], torch.full((3, 2, 3), 35.0))
    assert torch.equal(selected[1], torch.full((3, 2, 3), 2.0))
    assert provider.residency()["selection_mode"] == "selected_batch_only"
    assert provider.residency()["full_target_accelerator_resident_bytes"] == 0


def test_target_provider_resize_matches_existing_progressive_target_contract() -> None:
    frames = _normalized_frames(height=8, width=10)
    provider = PowerFoamTargetProvider.from_resident_frames(
        frames,
        device=torch.device("cpu"),
    )

    selected = provider.select(torch.tensor([4, 1]), height=4, width=6)
    native = torch.stack((frames[1, 1], frames[0, 1]))
    expected = resize_video_frames(native, normalize_image_size((4, 6)))

    assert torch.equal(selected, expected)
    assert selected.shape == (2, 3, 4, 6)


def test_path_source_decode_is_bit_exact_with_existing_sequence_loader(tmp_path) -> None:
    for frame in range(2):
        pixels = torch.arange(5 * 7 * 3, dtype=torch.uint8).reshape(5, 7, 3)
        pixels = pixels.add(frame * 17)
        Image.fromarray(pixels.numpy()).save(tmp_path / f"frame_{frame:03d}.png")
    sequence = load_frame_sequence(
        tmp_path,
        target_size=4,
        image_crop_mode="center_square",
    )
    provider = PowerFoamTargetProvider(
        source=PathPowerFoamTargetSource(
            frame_paths=(sequence.frame_paths,),
            image_crop_modes=(sequence.image_crop_mode,),
            height=4,
            width=4,
        ),
        device=torch.device("cpu"),
    )

    selected = provider.select(torch.tensor([1, 0]))

    assert torch.equal(selected, sequence.frames[[1, 0]])


def test_selected_mp4_decoder_seeks_only_requested_native_frames(tmp_path, monkeypatch) -> None:
    import numpy as np

    video_path = tmp_path / "camera.mp4"
    video_path.touch()
    bgr_frames = [np.full((2, 3, 3), (index, index + 10, index + 20), dtype=np.uint8) for index in range(6)]

    class Capture:
        def __init__(self) -> None:
            self.position = 0
            self.set_positions: list[int] = []
            self.read_count = 0

        def isOpened(self) -> bool:
            return True

        def get(self, prop: int) -> float:
            if prop == Cv2.CAP_PROP_FPS:
                return 10.0
            if prop == Cv2.CAP_PROP_FRAME_COUNT:
                return float(len(bgr_frames))
            return 0.0

        def set(self, prop: int, value: float) -> None:
            assert prop == Cv2.CAP_PROP_POS_FRAMES
            self.position = int(value)
            self.set_positions.append(self.position)

        def read(self):
            self.read_count += 1
            frame = bgr_frames[self.position]
            self.position += 1
            return True, frame

        def release(self) -> None:
            pass

    capture = Capture()

    class Cv2:
        CAP_PROP_FPS = 1
        CAP_PROP_FRAME_COUNT = 2
        CAP_PROP_POS_FRAMES = 3
        COLOR_BGR2RGB = 4

        @staticmethod
        def VideoCapture(_path: str) -> Capture:
            return capture

        @staticmethod
        def cvtColor(frame, code: int):
            assert code == Cv2.COLOR_BGR2RGB
            return frame[..., ::-1].copy()

    monkeypatch.setattr(multicam_val_data, "_import_cv2", lambda: Cv2)
    selected = multicam_val_data.load_multicam_val_selected_camera_frames(
        video_path=video_path,
        start_seconds=0.0,
        fps=10.0,
        frame_count=6,
        sample_indices=(4, 1, 4),
        target_size=(2, 3),
    )

    assert capture.set_positions == [4, 1, 4]
    assert capture.read_count == 3
    assert torch.equal(
        selected[:, :, 0, 0],
        torch.tensor(
            [[24, 14, 4], [21, 11, 1], [24, 14, 4]],
            dtype=torch.float32,
        ).div(255.0),
    )


def test_video_seek_source_constructs_without_decode_and_preserves_tensor_identity(
    tmp_path,
    monkeypatch,
) -> None:
    paths = tuple(tmp_path / f"cam{view}.mp4" for view in range(2))
    for path in paths:
        path.touch()
    frame_sources = tuple(
        MulticamVideoFrameSource(
            camera_name=f"cam{view}",
            video_path=path,
            start_seconds=0.0,
            sample_fps=30.0,
            source_frame_count=8,
            selected_frame_indices=(1, 4, 7),
            height=2,
            width=3,
        )
        for view, path in enumerate(paths)
    )
    calls = []

    def decode_selected(**kwargs) -> torch.Tensor:
        calls.append((Path(kwargs["video_path"]).stem, tuple(kwargs["sample_indices"])))
        view = int(Path(kwargs["video_path"]).stem.removeprefix("cam"))
        return torch.stack(
            [
                torch.full((3, 2, 3), view * 100 + frame, dtype=torch.float32).div(255.0)
                for frame in kwargs["sample_indices"]
            ]
        )

    monkeypatch.setattr(training_data, "load_multicam_val_selected_camera_frames", decode_selected)
    source = VideoSeekPowerFoamTargetSource(frame_sources)
    provider = PowerFoamTargetProvider(source=source, device=torch.device("cpu"))
    assert calls == []
    assert provider.residency()["resident_bytes"] == 0
    assert provider.residency()["source_kind"] == "video_seek_mp4"

    selected = provider.select(torch.tensor([5, 0, 4, 5]))
    assert calls == [("cam1", (7, 4, 7)), ("cam0", (1,))]
    assert torch.equal(
        selected[:, 0, 0, 0],
        torch.tensor([107, 1, 104, 107], dtype=torch.float32).div(255.0),
    )

    eager = torch.stack(
        [
            torch.stack(
                [torch.full((3, 2, 3), view * 100 + frame, dtype=torch.float32).div(255.0) for frame in (1, 4, 7)]
            )
            for view in range(2)
        ]
    )
    assert source.tensor_content_identity(chunk_frames=2) == tensor_content_identity(eager)


def test_target_provider_fails_closed_on_invalid_selection_and_source_contract() -> None:
    provider = PowerFoamTargetProvider.from_resident_frames(
        _normalized_frames(),
        device=torch.device("cpu"),
    )

    with pytest.raises(ValueError, match="at least one"):
        provider.select(torch.empty(0, dtype=torch.long))
    with pytest.raises(IndexError, match="outside"):
        provider.select(torch.tensor([6]))
    with pytest.raises(ValueError, match="equally sized"):
        provider.select_view_frames([0, 1], [0])
    with pytest.raises(IndexError, match="view index"):
        provider.select_view_frames([2], [0])
    with pytest.raises(IndexError, match="frame index"):
        provider.select_view_frames([0], [3])
    with pytest.raises(ValueError, match="positive dimensions"):
        provider.select(torch.tensor([0]), height=0)
    with pytest.raises(ValueError, match="float32"):
        PowerFoamTargetProvider.from_resident_frames(
            torch.zeros((1, 1, 3, 2, 2), dtype=torch.uint8),
            device=torch.device("cpu"),
        )


def test_paper_loader_drops_compatibility_targets_and_accounts_resident_fallback(monkeypatch) -> None:
    train_frames = _normalized_frames(views=2, frames=3, height=4, width=4)
    heldout_frames = _normalized_frames(views=1, frames=3, height=4, width=4)
    bundle = SimpleNamespace(
        train_frames=train_frames,
        heldout_frames=heldout_frames,
        train_K=torch.eye(3).repeat(2, 1, 1),
        heldout_K=torch.eye(3).repeat(1, 1, 1),
        train_w2c=torch.eye(4).repeat(2, 3, 1, 1),
        heldout_w2c=torch.eye(4).repeat(1, 3, 1, 1),
        train_lens_models=None,
        heldout_lens_models=None,
        train_distortions=None,
        heldout_distortions=None,
        train_view_count=2,
        heldout_view_count=1,
        frame_count=3,
        condition_sequence=SimpleNamespace(
            frames=train_frames[0],
            frame_times=torch.linspace(0.0, 1.0, 3),
            video_fps=30.0,
        ),
        metadata={"sample_id": "fixture"},
        train_camera_names=["a", "b"],
        heldout_camera_names=["c"],
        pose_source="fixture",
        anchor_c2w=None,
    )
    train_cameras = ((_camera(),) * 3,) * 2
    heldout_cameras = ((_camera(),) * 3,)
    monkeypatch.setattr(training_data, "load_multicam_video_bundle", lambda **_kwargs: bundle)
    monkeypatch.setattr(training_data, "cameras_from_K_w2c", lambda *_args, **_kwargs: train_cameras)
    monkeypatch.setattr(
        training_data,
        "heldout_cameras_from_K_w2c",
        lambda *_args, **_kwargs: heldout_cameras,
    )

    data = training_data.load_powerfoam_training_data(
        {
            "render": {"render_size": 4, "image_size": [4, 4]},
            "data": {"frame_source": "multicam_val"},
            "camera": {},
            "model": {"init_from_video": True},
            "paper_protocol": {"enabled": True},
        },
        torch.device("cpu"),
    )

    train_provider = data["sample_target_provider"]
    heldout_provider = data["heldout_target_provider"]
    assert data["targets"] is None
    assert data["heldout_targets"] is None
    assert torch.equal(train_provider.select(torch.tensor([5, 0])), train_frames.flatten(0, 1)[[5, 0]])
    assert torch.equal(heldout_provider.select(torch.tensor([2])), heldout_frames.flatten(0, 1)[[2]])
    assert data["target_residency"]["train"]["resident_bytes"] == train_frames.untyped_storage().nbytes()
    assert data["target_residency"]["heldout"]["resident_bytes"] == heldout_frames.untyped_storage().nbytes()
    assert data["target_residency"]["train"]["full_target_accelerator_resident_bytes"] == 0
    assert data["target_residency"]["train"]["compatibility_tensor_resident_bytes"] == 0
    assert (
        data["target_residency"]["train"]["effective_decoded_target_resident_bytes"]
        == train_frames.untyped_storage().nbytes()
    )
    assert "resident provider" in data["target_residency"]["limitation"]
    assert data["init_frames_residency"]["shares_train_target_storage"] is True


def _write_rgb_grid(
    root: Path,
    colors: tuple[tuple[tuple[int, int, int], ...], ...],
) -> tuple[tuple[Path, ...], ...]:
    root.mkdir(parents=True)
    paths = []
    for view, view_colors in enumerate(colors):
        view_paths = []
        for frame, color in enumerate(view_colors):
            path = root / f"view_{view}_frame_{frame}.png"
            Image.new("RGB", (6, 5), color).save(path)
            view_paths.append(path)
        paths.append(tuple(view_paths))
    return tuple(paths)


def _decoded_color_grid(
    colors: tuple[tuple[tuple[int, int, int], ...], ...],
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    return torch.stack(
        [
            torch.stack(
                [
                    torch.tensor(color, dtype=torch.float32).div(255.0).reshape(3, 1, 1).expand(3, height, width)
                    for color in view_colors
                ]
            )
            for view_colors in colors
        ]
    )


def test_paper_loader_uses_complete_frame_path_grid_as_lazy_source(tmp_path, monkeypatch) -> None:
    train_colors = (
        ((10, 20, 30), (40, 50, 60), (70, 80, 90)),
        ((15, 25, 35), (45, 55, 65), (75, 85, 95)),
    )
    heldout_colors = (((100, 110, 120), (130, 140, 150), (160, 170, 180)),)
    train_paths = _write_rgb_grid(tmp_path / "train", train_colors)
    heldout_paths = _write_rgb_grid(tmp_path / "heldout", heldout_colors)
    train_frames = _decoded_color_grid(train_colors, height=4, width=4)
    heldout_frames = _decoded_color_grid(heldout_colors, height=4, width=4)
    train_sequences = tuple(SimpleNamespace(frame_paths=paths, image_crop_mode="resize") for paths in train_paths)
    heldout_sequences = tuple(SimpleNamespace(frame_paths=paths, image_crop_mode="resize") for paths in heldout_paths)
    bundle = SimpleNamespace(
        train_frames=train_frames,
        heldout_frames=heldout_frames,
        train_sequences=train_sequences,
        heldout_sequences=heldout_sequences,
        train_K=torch.eye(3).repeat(2, 1, 1),
        heldout_K=torch.eye(3).repeat(1, 1, 1),
        train_w2c=torch.eye(4).repeat(2, 3, 1, 1),
        heldout_w2c=torch.eye(4).repeat(1, 3, 1, 1),
        train_lens_models=None,
        heldout_lens_models=None,
        train_distortions=None,
        heldout_distortions=None,
        train_view_count=2,
        heldout_view_count=1,
        frame_count=3,
        condition_sequence=SimpleNamespace(
            frames=train_frames[0],
            frame_times=torch.linspace(0.0, 1.0, 3),
            video_fps=30.0,
        ),
        metadata={"sample_id": "path_fixture"},
        train_camera_names=["a", "b"],
        heldout_camera_names=["c"],
        pose_source="fixture",
        anchor_c2w=None,
    )
    train_cameras = ((_camera(),) * 3,) * 2
    heldout_cameras = ((_camera(),) * 3,)
    monkeypatch.setattr(training_data, "load_multicam_video_bundle", lambda **_kwargs: bundle)
    monkeypatch.setattr(training_data, "cameras_from_K_w2c", lambda *_args, **_kwargs: train_cameras)
    monkeypatch.setattr(
        training_data,
        "heldout_cameras_from_K_w2c",
        lambda *_args, **_kwargs: heldout_cameras,
    )

    data = training_data.load_powerfoam_training_data(
        {
            "render": {"render_size": 4, "image_size": [4, 4]},
            "data": {"frame_source": "multicam_val"},
            "camera": {},
            "model": {"init_from_video": True},
            "paper_protocol": {"enabled": True},
        },
        torch.device("cpu"),
    )

    assert torch.equal(
        data["sample_target_provider"].select(torch.tensor([5, 0])),
        train_frames.flatten(0, 1)[[5, 0]],
    )
    assert torch.equal(
        data["heldout_target_provider"].select(torch.tensor([2])),
        heldout_frames.flatten(0, 1)[[2]],
    )
    train_residency = data["target_residency"]["train"]
    assert train_residency["source_kind"] == "path_backed_images"
    assert train_residency["resident_bytes"] == 0
    assert train_residency["disk_lazy_decode"] is True
    assert train_residency["compatibility_tensor_resident_bytes"] == 0
    assert train_residency["effective_decoded_target_resident_bytes"] == 0
    assert train_residency["provider_can_replace_compatibility_tensor"] is True
    assert data["targets"] is None
    assert data["heldout_targets"] is None
    assert data["init_frames"].untyped_storage().data_ptr() != train_frames.untyped_storage().data_ptr()
    assert data["init_frames_resident_bytes"] == train_frames[0].numel() * train_frames.element_size()
    assert data["init_frames_residency"]["shares_train_target_storage"] is False


def test_paper_loader_builds_neural3d_mp4_providers_without_full_decode(
    tmp_path,
    monkeypatch,
) -> None:
    train_path = tmp_path / "cam04.mp4"
    heldout_path = tmp_path / "cam06.mp4"
    train_path.touch()
    heldout_path.touch()
    record = {
        "dataset": "neural_3d_video",
        "sample_id": "coffee_martini_fixture",
        "source_camera": "cam04",
        "target_camera": "cam06",
        "source_video_path": str(train_path),
        "target_video_path": str(heldout_path),
        "source_start_seconds": 0.0,
        "target_start_seconds": 0.0,
        "fps": 30.0,
        "frame_count": 5,
        "train_cameras": ["cam04"],
        "heldout_cameras": ["cam06"],
        "anchor_camera": "cam04",
        "condition_camera": "cam04",
    }
    monkeypatch.setattr(multicam_video_data, "select_multicam_record", lambda _cfg: record)

    def forbidden_full_decode(*_args, **_kwargs):
        raise AssertionError("metadata-only paper loading must not decode a complete MP4")

    monkeypatch.setattr(multicam_video_data, "load_camera_video", forbidden_full_decode)
    decode_calls = []

    def decode_selected(**kwargs) -> torch.Tensor:
        camera_value = 4 if Path(kwargs["video_path"]).stem == "cam04" else 6
        decode_calls.append((Path(kwargs["video_path"]).stem, tuple(kwargs["sample_indices"])))
        return torch.stack(
            [
                torch.full((3, 2, 3), camera_value * 10 + frame, dtype=torch.float32).div(255.0)
                for frame in kwargs["sample_indices"]
            ]
        )

    monkeypatch.setattr(training_data, "load_multicam_val_selected_camera_frames", decode_selected)
    data = training_data.load_powerfoam_training_data(
        {
            "render": {"render_size": 3, "image_size": [2, 3]},
            "data": {
                "frame_source": "multicam_val",
                "frame_indices": [4, 1],
            },
            "camera": {"rig_init": "orthogonal_origin"},
            "model": {
                "init_from_video": True,
                "init_point_cloud_path": "already_selected_initializer.ply",
            },
            "paper_protocol": {"enabled": True},
        },
        torch.device("cpu"),
    )

    assert decode_calls == []
    assert data["targets"] is None
    assert data["heldout_targets"] is None
    assert data["init_frames"] is None
    assert data["paper_dataset_bundle"] is None
    assert callable(data["paper_dataset_bundle_builder"])
    assert data["target_residency"]["train"]["source_kind"] == "video_seek_mp4"
    assert data["target_residency"]["train"]["resident_bytes"] == 0
    assert data["target_residency"]["heldout"]["resident_bytes"] == 0

    train = data["sample_target_provider"].select(torch.tensor([1, 0]))
    heldout = data["heldout_target_provider"].select(torch.tensor([0, 1]))
    assert decode_calls == [("cam04", (1, 4)), ("cam06", (4, 1))]
    assert torch.equal(
        train[:, 0, 0, 0],
        torch.tensor([41, 44], dtype=torch.float32).div(255.0),
    )
    assert torch.equal(
        heldout[:, 0, 0, 0],
        torch.tensor([64, 61], dtype=torch.float32).div(255.0),
    )
    assert data["sample_frame_indices"].tolist() == [0, 1]
    assert data["heldout_frame_indices"].tolist() == [0, 1]

    decode_calls.clear()
    identity = training_data.resolve_powerfoam_paper_dataset_bundle(data)
    eager_train = torch.stack(
        [torch.full((3, 2, 3), value, dtype=torch.float32).div(255.0) for value in (44, 41)]
    ).unsqueeze(0)
    eager_heldout = torch.stack(
        [torch.full((3, 2, 3), value, dtype=torch.float32).div(255.0) for value in (64, 61)]
    ).unsqueeze(0)
    assert identity["train_frames"] == tensor_content_identity(eager_train)
    assert identity["heldout_frames"] == tensor_content_identity(eager_heldout)
    assert decode_calls == [("cam04", (4, 1)), ("cam06", (4, 1))]
    assert training_data.resolve_powerfoam_paper_dataset_bundle(data) is identity
    assert decode_calls == [("cam04", (4, 1)), ("cam06", (4, 1))]
