from __future__ import annotations

from pathlib import Path

import numpy as np

from sequence_data import load_video_sequence


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
