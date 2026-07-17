from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from runtime_types import SequenceData
from sequence_data import _image_to_tensor, load_video_sequence, load_video_window_sequence
from video_feature_cache import sample_cache_key


class _FakeCapture:
    def __init__(self, _path: str) -> None:
        self._used = False

    def isOpened(self) -> bool:
        return True

    def get(self, prop: int) -> float:
        if prop == _FakeCv2.CAP_PROP_FPS:
            return 4.0
        if prop == _FakeCv2.CAP_PROP_FRAME_COUNT:
            return 1.0
        return 0.0

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._used:
            return False, None
        self._used = True
        frame_bgr = np.zeros((4, 4, 3), dtype=np.uint8)
        frame_bgr[..., 0] = 255
        return True, frame_bgr

    def release(self) -> None:
        return None


class _FakeCv2:
    CAP_PROP_FPS = 5
    CAP_PROP_FRAME_COUNT = 7
    CAP_PROP_POS_FRAMES = 1
    CAP_PROP_POS_MSEC = 2
    COLOR_BGR2RGB = 42

    @staticmethod
    def VideoCapture(path: str) -> _FakeCapture:
        return _FakeCapture(path)

    @staticmethod
    def cvtColor(frame: np.ndarray, _code: int) -> np.ndarray:
        return frame[..., ::-1]


def test_load_video_sequence_allows_single_frame(monkeypatch) -> None:
    monkeypatch.setattr("sequence_data._import_cv2", lambda: _FakeCv2)

    sequence = load_video_sequence(Path("single_frame.mp4"), target_size=4, max_frames=1)

    assert sequence.frame_count == 1
    assert sequence.selected_frame_count == 1
    assert sequence.all_frame_count == 1
    assert sequence.frame_times.shape == (1, 1)
    assert float(sequence.frame_times[0, 0]) == 0.0


class _FakeWindowCapture:
    def __init__(self, _path: str) -> None:
        self._index = 0
        self._frame_count = 10

    def isOpened(self) -> bool:
        return True

    def get(self, prop: int) -> float:
        if prop == _FakeCv2.CAP_PROP_FPS:
            return 4.0
        if prop == _FakeCv2.CAP_PROP_FRAME_COUNT:
            return float(self._frame_count)
        return 0.0

    def set(self, prop: int, value: float) -> bool:
        if prop == _FakeCv2.CAP_PROP_POS_FRAMES:
            self._index = int(value)
        elif prop == _FakeCv2.CAP_PROP_POS_MSEC:
            self._index = int(round(value / 1000.0 * 4.0))
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._index >= self._frame_count:
            return False, None
        frame_bgr = np.full((4, 4, 3), self._index, dtype=np.uint8)
        self._index += 1
        return True, frame_bgr

    def release(self) -> None:
        return None


def test_load_video_window_sequence_samples_requested_times(monkeypatch) -> None:
    monkeypatch.setattr("sequence_data._import_cv2", lambda: _FakeCv2)
    monkeypatch.setattr(_FakeCv2, "VideoCapture", lambda path: _FakeWindowCapture(path))

    sequence = load_video_window_sequence(
        Path("window.mp4"),
        target_size=4,
        start_seconds=0.5,
        duration_seconds=2.0,
        fps=2.0,
        frame_count=3,
    )

    assert sequence.frame_count == 3
    assert sequence.selected_frame_count == 3
    assert sequence.all_frame_count == 10
    assert sequence.video_fps == 2.0
    assert sequence.frame_source == "explicit_video_window"
    assert [record["frame_index"] for record in sequence.records] == [2, 4, 6]


def test_load_video_window_sequence_reuses_frame_cache(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("sequence_data._import_cv2", lambda: _FakeCv2)
    monkeypatch.setattr(_FakeCv2, "VideoCapture", lambda path: _FakeWindowCapture(path))

    first = load_video_window_sequence(
        Path("window.mp4"),
        target_size=4,
        start_seconds=0.5,
        duration_seconds=2.0,
        fps=2.0,
        frame_count=3,
        frame_cache_dir=tmp_path,
    )

    def _fail_capture(_path: str):
        raise AssertionError("cache hit should not reopen the source video")

    monkeypatch.setattr(_FakeCv2, "VideoCapture", _fail_capture)
    second = load_video_window_sequence(
        Path("window.mp4"),
        target_size=4,
        start_seconds=0.5,
        duration_seconds=2.0,
        fps=2.0,
        frame_count=3,
        frame_cache_dir=tmp_path,
    )

    assert len(list(tmp_path.glob("*.pt"))) == 1
    assert torch.equal(second.frames, first.frames)
    assert torch.equal(second.frame_times, first.frame_times)
    assert second.records == first.records


def test_image_to_tensor_center_square_crops_before_resize() -> None:
    image = Image.fromarray(np.zeros((4, 6, 3), dtype=np.uint8))
    pixels = image.load()
    for y in range(4):
        for x in range(6):
            pixels[x, y] = (x * 20, y * 20, 0)

    tensor = _image_to_tensor(image, target_size=4, image_crop_mode="center_square")

    # Width 6, height 4 center-square crop removes x=0 and x=5 before resize.
    red_channel = tensor[0]
    assert float(red_channel[:, 0].mean()) == torch.tensor(20 / 255).item()
    assert float(red_channel[:, -1].mean()) == torch.tensor(80 / 255).item()


def test_video_feature_cache_key_distinguishes_video_windows() -> None:
    frames = torch.zeros((3, 3, 4, 4), dtype=torch.float32)
    frame_times = torch.zeros((3, 1), dtype=torch.float32)
    common = {
        "frames": frames,
        "frame_times": frame_times,
        "video_fps": 2.0,
        "frame_source": "explicit_video_window",
        "source_path": Path("same_video.mp4"),
        "selected_frame_count": 3,
        "all_frame_count": 10,
    }
    first = SequenceData(
        **common,
        records=(
            {"timestamp_seconds": 0.5, "frame_index": 2},
            {"timestamp_seconds": 1.0, "frame_index": 4},
            {"timestamp_seconds": 1.5, "frame_index": 6},
        ),
    )
    second = SequenceData(
        **common,
        records=(
            {"timestamp_seconds": 1.0, "frame_index": 4},
            {"timestamp_seconds": 1.5, "frame_index": 6},
            {"timestamp_seconds": 2.0, "frame_index": 8},
        ),
    )

    feature_cfg = {
        "extractor": "vjepa_torchhub",
        "model_id": "vjepa2_1_vit_base_384",
        "sample_cache_key": "unit-test",
        "cache_version": 1,
        "vjepa_crop_size": 256,
    }
    assert sample_cache_key(first, feature_cfg) != sample_cache_key(second, feature_cfg)
