from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

import video_feature_cache
from runtime_types import SequenceData
from video_feature_cache import VideoFeatureCache, infer_feature_channels


class _CountingExtractor:
    def __init__(self, value: float) -> None:
        self.value = value
        self.calls = 0

    def __call__(self, _sequence_data: SequenceData) -> dict[str, torch.Tensor]:
        self.calls += 1
        return {"tokens": torch.full((1, 2, 4), self.value, dtype=torch.float32)}


def _sequence(source_path: Path) -> SequenceData:
    source_path.write_bytes(b"video")
    return SequenceData(
        frames=torch.zeros(2, 3, 4, 4),
        frame_times=torch.linspace(0, 1, 2).view(2, 1),
        video_fps=4.0,
        frame_source="explicit_video",
        source_path=source_path,
        selected_frame_count=2,
        all_frame_count=2,
    )


def _cache_cfg(cache_dir: Path, sample_key: str = "v1") -> dict[str, Any]:
    return {
        "cache_dir": str(cache_dir),
        "extractor": "rgb_pyramid",
        "save_dtype": "float32",
        "keep_in_memory": False,
        "sample_cache_key": sample_key,
    }


def test_video_feature_cache_hit_miss_and_key_busting(
    monkeypatch,
    tmp_path: Path,
) -> None:
    sequence = _sequence(tmp_path / "clip.mp4")
    first_extractor = _CountingExtractor(value=1.0)
    monkeypatch.setattr(video_feature_cache, "build_feature_extractor", lambda *_args, **_kwargs: first_extractor)

    cache = VideoFeatureCache(_cache_cfg(tmp_path / "cache", sample_key="v1"), device="cpu")
    first_path = cache.cache_path(sequence)
    first = cache.load_or_bake(sequence)

    assert first_path.exists()
    assert first_extractor.calls == 1
    assert torch.all(first["tokens"] == 1.0)

    second_extractor = _CountingExtractor(value=2.0)
    monkeypatch.setattr(video_feature_cache, "build_feature_extractor", lambda *_args, **_kwargs: second_extractor)
    second_cache = VideoFeatureCache(_cache_cfg(tmp_path / "cache", sample_key="v1"), device="cpu")
    cached = second_cache.load_or_bake(sequence)

    assert second_extractor.calls == 0
    assert torch.all(cached["tokens"] == 1.0)

    busted_cache = VideoFeatureCache(_cache_cfg(tmp_path / "cache", sample_key="v2"), device="cpu")
    busted_path = busted_cache.cache_path(sequence)
    busted = busted_cache.load_or_bake(sequence)

    assert busted_path != first_path
    assert second_extractor.calls == 1
    assert torch.all(busted["tokens"] == 2.0)


def test_infer_feature_channels_for_token_and_video_layouts() -> None:
    channels = infer_feature_channels(
        {
            "tokens": torch.zeros(1, 8, 32),
            "feature_video": torch.zeros(1, 4, 16, 8, 8),
        }
    )

    assert channels == {"tokens": 32, "feature_video": 4}
