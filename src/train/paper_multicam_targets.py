from __future__ import annotations

import hashlib
import json
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

from multicam_val_data import load_multicam_val_selected_camera_frames
from multicam_video_data import MulticamVideoFrameSource


@dataclass
class PaperMulticamTargetProvider:
    """Bounded CPU target decode for Paper A multicamera MP4 inputs.

    The provider preserves the eager loader's logical ``[view,time,C,H,W]``
    values, but retains at most ``cache_capacity_frames`` decoded frames.  It
    groups misses by camera so one paper batch or evaluation chunk needs at
    most one video-open/decode call per participating view.
    """

    frame_sources: tuple[MulticamVideoFrameSource, ...]
    cache_capacity_frames: int = 8
    _cache: OrderedDict[tuple[int, int], torch.Tensor] = field(
        default_factory=OrderedDict,
        init=False,
        repr=False,
    )
    _decode_call_count: int = field(default=0, init=False, repr=False)
    _decoded_frame_count: int = field(default=0, init=False, repr=False)
    _requested_frame_count: int = field(default=0, init=False, repr=False)
    _cache_hit_count: int = field(default=0, init=False, repr=False)
    _peak_cache_resident_frames: int = field(default=0, init=False, repr=False)
    _peak_request_frame_count: int = field(default=0, init=False, repr=False)
    _peak_decode_batch_frames: int = field(default=0, init=False, repr=False)
    _identity_pass_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.frame_sources:
            raise ValueError("paper target provider requires at least one camera")
        if (
            isinstance(self.cache_capacity_frames, bool)
            or not isinstance(self.cache_capacity_frames, int)
            or self.cache_capacity_frames < 1
        ):
            raise ValueError("paper target cache capacity must be a positive integer")
        reference = self.frame_sources[0]
        expected = (
            len(reference.selected_frame_indices),
            reference.height,
            reference.width,
        )
        if any(
            (
                len(source.selected_frame_indices),
                source.height,
                source.width,
            )
            != expected
            for source in self.frame_sources
        ):
            raise ValueError(
                "paper target camera sources must share frame count and dimensions"
            )

    @property
    def view_count(self) -> int:
        return len(self.frame_sources)

    @property
    def frame_count(self) -> int:
        return len(self.frame_sources[0].selected_frame_indices)

    @property
    def height(self) -> int:
        return int(self.frame_sources[0].height)

    @property
    def width(self) -> int:
        return int(self.frame_sources[0].width)

    @property
    def frame_tensor_bytes(self) -> int:
        return 3 * self.height * self.width * 4

    @staticmethod
    def _indices(values: Sequence[int] | torch.Tensor) -> tuple[int, ...]:
        return tuple(
            int(value)
            for value in torch.as_tensor(
                values,
                device="cpu",
                dtype=torch.long,
            )
            .reshape(-1)
            .tolist()
        )

    @torch.no_grad()
    def select_view_frames(
        self,
        view_indices: Sequence[int] | torch.Tensor,
        frame_indices: Sequence[int] | torch.Tensor,
    ) -> torch.Tensor:
        views = self._indices(view_indices)
        frames = self._indices(frame_indices)
        if not views or len(views) != len(frames):
            raise ValueError(
                "paper target view/frame selections must be non-empty and equally sized"
            )
        invalid_view = next(
            (value for value in views if value < 0 or value >= self.view_count),
            None,
        )
        if invalid_view is not None:
            raise IndexError(
                f"paper target view index {invalid_view} is outside "
                f"[0, {self.view_count})"
            )
        invalid_frame = next(
            (value for value in frames if value < 0 or value >= self.frame_count),
            None,
        )
        if invalid_frame is not None:
            raise IndexError(
                f"paper target frame index {invalid_frame} is outside "
                f"[0, {self.frame_count})"
            )

        keys = tuple(zip(views, frames, strict=True))
        unique_keys = tuple(dict.fromkeys(keys))
        resolved: dict[tuple[int, int], torch.Tensor] = {
            key: self._cache[key] for key in unique_keys if key in self._cache
        }
        self._cache_hit_count += sum(key in self._cache for key in keys)

        missing = tuple(key for key in unique_keys if key not in resolved)
        for view in dict.fromkeys(key[0] for key in missing):
            view_keys = tuple(key for key in missing if key[0] == view)
            source = self.frame_sources[view]
            native_indices = tuple(
                source.selected_frame_indices[frame] for _, frame in view_keys
            )
            decoded = load_multicam_val_selected_camera_frames(
                video_path=source.video_path,
                start_seconds=source.start_seconds,
                fps=source.sample_fps,
                frame_count=source.source_frame_count,
                sample_indices=native_indices,
                target_size=(source.height, source.width),
                device=torch.device("cpu"),
            )
            expected_shape = (len(view_keys), 3, self.height, self.width)
            if (
                decoded.device.type != "cpu"
                or decoded.dtype != torch.float32
                or tuple(decoded.shape) != expected_shape
            ):
                raise ValueError(
                    "paper target decoder violated its CPU float32 contract: "
                    f"expected {expected_shape}, got {decoded.dtype} "
                    f"{tuple(decoded.shape)} on {decoded.device}"
                )
            for key, frame in zip(view_keys, decoded, strict=True):
                # A clone prevents one cached frame from retaining the whole
                # decoded batch's storage after its siblings are evicted.
                resolved[key] = frame.clone()
            self._decode_call_count += 1
            self._decoded_frame_count += len(view_keys)
            self._peak_decode_batch_frames = max(
                self._peak_decode_batch_frames,
                len(view_keys),
            )

        selected = torch.stack([resolved[key] for key in keys], dim=0)
        for key in unique_keys:
            self._cache[key] = resolved[key]
            self._cache.move_to_end(key)
        while len(self._cache) > self.cache_capacity_frames:
            self._cache.popitem(last=False)

        self._requested_frame_count += len(keys)
        self._peak_request_frame_count = max(
            self._peak_request_frame_count,
            len(keys),
        )
        self._peak_cache_resident_frames = max(
            self._peak_cache_resident_frames,
            len(self._cache),
        )
        return selected

    def clear_cache(self) -> None:
        self._cache.clear()

    @torch.no_grad()
    def tensor_content_identity(self, *, chunk_frames: int = 16) -> dict[str, Any]:
        """Hash the logical eager tensor while retaining bounded frame chunks."""

        if (
            isinstance(chunk_frames, bool)
            or not isinstance(chunk_frames, int)
            or chunk_frames < 1
        ):
            raise ValueError("paper target identity chunk size must be positive")
        metadata = {
            "dtype": str(torch.float32),
            "shape": [
                self.view_count,
                self.frame_count,
                3,
                self.height,
                self.width,
            ],
            "bytes": (
                self.view_count
                * self.frame_count
                * self.frame_tensor_bytes
            ),
            "byte_order": f"native_{sys.byteorder}_endian",
            "layout": "contiguous_c_order",
        }
        digest = hashlib.sha256()
        digest.update(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        digest.update(b"\n")
        self.clear_cache()
        try:
            for view in range(self.view_count):
                for start in range(0, self.frame_count, chunk_frames):
                    stop = min(start + chunk_frames, self.frame_count)
                    chunk = self.select_view_frames(
                        (view,) * (stop - start),
                        tuple(range(start, stop)),
                    )
                    digest.update(memoryview(chunk.contiguous().numpy()).cast("B"))
                    del chunk
        finally:
            self.clear_cache()
        self._identity_pass_count += 1
        return {**metadata, "sha256": digest.hexdigest()}

    def accounting(self) -> dict[str, Any]:
        resident_frames = len(self._cache)
        transient_frames_bound = (
            self.cache_capacity_frames
            + 3
            * max(
                self._peak_request_frame_count,
                self._peak_decode_batch_frames,
            )
        )
        return {
            "schema_version": 1,
            "source_kind": "paper_video_seek_bounded_lru",
            "source_device": "disk",
            "output_device": "cpu",
            "logical_bytes": (
                self.view_count
                * self.frame_count
                * self.frame_tensor_bytes
            ),
            "resident_bytes": resident_frames * self.frame_tensor_bytes,
            "full_source_resident": False,
            "disk_lazy_decode": True,
            "cache_capacity_frames": self.cache_capacity_frames,
            "cache_capacity_bytes": (
                self.cache_capacity_frames * self.frame_tensor_bytes
            ),
            "cache_resident_frames": resident_frames,
            "peak_cache_resident_frames": self._peak_cache_resident_frames,
            "peak_cache_resident_bytes": (
                self._peak_cache_resident_frames * self.frame_tensor_bytes
            ),
            "decode_call_count": self._decode_call_count,
            "decoded_frame_count": self._decoded_frame_count,
            "requested_frame_count": self._requested_frame_count,
            "cache_hit_count": self._cache_hit_count,
            "peak_request_frame_count": self._peak_request_frame_count,
            "peak_decode_batch_frames": self._peak_decode_batch_frames,
            # A miss can temporarily retain the decoder batch, independent
            # per-frame cache clones, the ordered return stack, and the prior
            # LRU.  This is deliberately more conservative than cache bytes.
            "peak_transient_cpu_frames_conservative_bound": transient_frames_bound,
            "peak_transient_cpu_bytes_conservative_bound": (
                transient_frames_bound * self.frame_tensor_bytes
            ),
            "identity_pass_count": self._identity_pass_count,
            "preserves_logical_order_and_duplicates": True,
            "full_video_tensor_materialization_count": 0,
        }


def load_grouped_frozen_target_frames(provider, frame_indices, *, start, stop, chunk_frames):
    """Load one device chunk, prefetching only a bounded group of selected CPU frames."""
    prefetch_frames = (provider.cache_capacity_frames // chunk_frames) * chunk_frames
    if prefetch_frames > chunk_frames and start % prefetch_frames == 0:
        upcoming = frame_indices[start : start + prefetch_frames]
        # The returned batch is released here; only the bounded LRU survives.
        # This call must remain inside the consumer's target-loading timer.
        provider.select_view_frames((0,) * len(upcoming), upcoming)
    selected = frame_indices[start:stop]
    return provider.select_view_frames((0,) * len(selected), selected)


__all__ = ["PaperMulticamTargetProvider", "load_grouped_frozen_target_frames"]
