from __future__ import annotations

import hashlib
import json
import mmap
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from camera import CameraSpec
from multicam_val_data import load_multicam_val_selected_camera_frames
from multicam_video_data import (
    MulticamVideoFrameSource,
    cameras_from_K_w2c,
    heldout_cameras_from_K_w2c,
    load_multicam_video_bundle,
)
from paper_training_protocol import (
    normalize_image_size,
    paper_dataset_bundle_identity,
    resize_video_frames,
)
from powerfoam_geometry import powerfoam_rays_from_camera, powerfoam_rays_from_camera_grid
from powerfoam_training import flatten_multiview_powerfoam_samples
from sequence_data import load_video_sequence


class PowerFoamTargetSource(Protocol):
    """Selected-read source contract; a disk decoder can replace the resident source."""

    @property
    def view_count(self) -> int: ...

    @property
    def frame_count(self) -> int: ...

    @property
    def height(self) -> int: ...

    @property
    def width(self) -> int: ...

    def select_view_frames(
        self,
        view_indices: tuple[int, ...],
        frame_indices: tuple[int, ...],
    ) -> torch.Tensor: ...

    def residency(self) -> dict[str, Any]: ...


_SELECTED_PIXEL_READ_SEAL = object()
_SELECTED_PIXEL_READ_MODES = {
    "direct_pixels",
    "certified_bounded_region",
    "full_frame_fallback",
}


@dataclass(frozen=True)
class PowerFoamSelectedPixelRead:
    """Sealed CPU ``[N,3]`` read with explicit materialization provenance.

    ``direct_pixels`` and ``certified_bounded_region`` are the only modes that
    can support the memory-scaling acceptance claim.  The fallback remains a
    compatibility path for compressed/image sources, but records every full
    frame it materialized instead of presenting that work as pixel-linear.
    """

    rgb_f32_cpu: torch.Tensor
    selection_mode: str
    source_provenance: str
    observation_count: int
    source_visible_peak_logical_tensor_bytes_upper_bound: int
    full_frame_materialization_count: int
    maximum_full_frame_materialization_tensor_bytes: int
    bounded_region_materialization_count: int
    maximum_bounded_region_materialization_tensor_bytes: int
    transient_mapped_address_space_bytes: int
    maximum_requested_unique_mapped_page_count: int
    total_requested_unique_mapped_page_count: int
    mapped_page_size_bytes: int
    maximum_requested_mapped_page_bytes_upper_bound: int
    total_requested_mapped_page_bytes_upper_bound: int
    mapping_closed_before_return: bool
    preserves_request_order_and_duplicates: bool
    _tensor_identity: int
    _tensor_signature: tuple[object, ...]
    _seal: object

    @classmethod
    def seal(
        cls,
        rgb_f32_cpu: torch.Tensor,
        *,
        selection_mode: str,
        source_provenance: str,
        source_visible_peak_logical_tensor_bytes_upper_bound: int,
        full_frame_materialization_count: int = 0,
        maximum_full_frame_materialization_tensor_bytes: int = 0,
        bounded_region_materialization_count: int = 0,
        maximum_bounded_region_materialization_tensor_bytes: int = 0,
        transient_mapped_address_space_bytes: int = 0,
        maximum_requested_unique_mapped_page_count: int = 0,
        total_requested_unique_mapped_page_count: int = 0,
        mapped_page_size_bytes: int = 0,
        maximum_requested_mapped_page_bytes_upper_bound: int = 0,
        total_requested_mapped_page_bytes_upper_bound: int = 0,
        mapping_closed_before_return: bool = True,
    ) -> PowerFoamSelectedPixelRead:
        tensor = torch.as_tensor(rgb_f32_cpu)
        result = cls(
            rgb_f32_cpu=tensor,
            selection_mode=str(selection_mode),
            source_provenance=str(source_provenance),
            observation_count=int(tensor.shape[0]) if tensor.ndim == 2 else -1,
            source_visible_peak_logical_tensor_bytes_upper_bound=int(
                source_visible_peak_logical_tensor_bytes_upper_bound
            ),
            full_frame_materialization_count=int(full_frame_materialization_count),
            maximum_full_frame_materialization_tensor_bytes=int(
                maximum_full_frame_materialization_tensor_bytes
            ),
            bounded_region_materialization_count=int(
                bounded_region_materialization_count
            ),
            maximum_bounded_region_materialization_tensor_bytes=int(
                maximum_bounded_region_materialization_tensor_bytes
            ),
            transient_mapped_address_space_bytes=int(
                transient_mapped_address_space_bytes
            ),
            maximum_requested_unique_mapped_page_count=int(
                maximum_requested_unique_mapped_page_count
            ),
            total_requested_unique_mapped_page_count=int(
                total_requested_unique_mapped_page_count
            ),
            mapped_page_size_bytes=int(mapped_page_size_bytes),
            maximum_requested_mapped_page_bytes_upper_bound=int(
                maximum_requested_mapped_page_bytes_upper_bound
            ),
            total_requested_mapped_page_bytes_upper_bound=int(
                total_requested_mapped_page_bytes_upper_bound
            ),
            mapping_closed_before_return=bool(mapping_closed_before_return),
            preserves_request_order_and_duplicates=True,
            _tensor_identity=id(tensor),
            _tensor_signature=_selected_pixel_tensor_signature(tensor),
            _seal=_SELECTED_PIXEL_READ_SEAL,
        )
        result.assert_valid()
        return result

    @property
    def acceptance_capable(self) -> bool:
        return (
            self.selection_mode
            in {"direct_pixels", "certified_bounded_region"}
            and self.full_frame_materialization_count == 0
            and self.maximum_full_frame_materialization_tensor_bytes == 0
        )

    def assert_valid(
        self,
        *,
        expected_observation_count: int | None = None,
        full_frame_tensor_bytes: int | None = None,
    ) -> None:
        tensor = self.rgb_f32_cpu
        logical_bytes = self.observation_count * 3 * 4
        if (
            self._seal is not _SELECTED_PIXEL_READ_SEAL
            or self.selection_mode not in _SELECTED_PIXEL_READ_MODES
            or not self.source_provenance.strip()
            or tensor.device.type != "cpu"
            or tensor.dtype != torch.float32
            or tensor.ndim != 2
            or tuple(tensor.shape) != (self.observation_count, 3)
            or self.observation_count < 1
            or not tensor.is_contiguous()
            or id(tensor) != self._tensor_identity
            or _selected_pixel_tensor_signature(tensor) != self._tensor_signature
            or not self.preserves_request_order_and_duplicates
            or self.source_visible_peak_logical_tensor_bytes_upper_bound
            < logical_bytes
            or self.full_frame_materialization_count < 0
            or self.maximum_full_frame_materialization_tensor_bytes < 0
            or self.bounded_region_materialization_count < 0
            or self.maximum_bounded_region_materialization_tensor_bytes < 0
            or self.transient_mapped_address_space_bytes < 0
            or self.maximum_requested_unique_mapped_page_count < 0
            or self.total_requested_unique_mapped_page_count < 0
            or self.mapped_page_size_bytes < 0
            or self.maximum_requested_mapped_page_bytes_upper_bound < 0
            or self.total_requested_mapped_page_bytes_upper_bound < 0
        ):
            raise ValueError("selected-pixel source violated its sealed CPU RGB contract")
        mapped_receipt_values = (
            self.transient_mapped_address_space_bytes,
            self.maximum_requested_unique_mapped_page_count,
            self.total_requested_unique_mapped_page_count,
            self.mapped_page_size_bytes,
            self.maximum_requested_mapped_page_bytes_upper_bound,
            self.total_requested_mapped_page_bytes_upper_bound,
        )
        if self.transient_mapped_address_space_bytes > 0:
            if (
                self.maximum_requested_unique_mapped_page_count < 1
                or self.total_requested_unique_mapped_page_count
                < self.maximum_requested_unique_mapped_page_count
                or self.mapped_page_size_bytes < 1
                or self.maximum_requested_mapped_page_bytes_upper_bound
                != self.maximum_requested_unique_mapped_page_count
                * self.mapped_page_size_bytes
                or self.total_requested_mapped_page_bytes_upper_bound
                != self.total_requested_unique_mapped_page_count
                * self.mapped_page_size_bytes
                or not self.mapping_closed_before_return
            ):
                raise ValueError("selected-pixel mapped-source receipt is incomplete")
        elif any(mapped_receipt_values) or not self.mapping_closed_before_return:
            raise ValueError("selected-pixel read reported an invalid empty mapping receipt")
        if (
            expected_observation_count is not None
            and self.observation_count != int(expected_observation_count)
        ):
            raise ValueError("selected-pixel source changed requested observation coverage")
        if self.selection_mode == "direct_pixels":
            if any(
                value != 0
                for value in (
                    self.full_frame_materialization_count,
                    self.maximum_full_frame_materialization_tensor_bytes,
                    self.bounded_region_materialization_count,
                    self.maximum_bounded_region_materialization_tensor_bytes,
                )
            ):
                raise ValueError("direct-pixel read reported a hidden region/frame materialization")
        elif self.selection_mode == "certified_bounded_region":
            if (
                self.full_frame_materialization_count != 0
                or self.maximum_full_frame_materialization_tensor_bytes != 0
                or self.bounded_region_materialization_count < 1
                or self.maximum_bounded_region_materialization_tensor_bytes < 12
                or self.source_visible_peak_logical_tensor_bytes_upper_bound
                < logical_bytes
                + self.maximum_bounded_region_materialization_tensor_bytes
            ):
                raise ValueError("bounded-region read omitted its bounded materialization proof")
            if (
                full_frame_tensor_bytes is not None
                and self.maximum_bounded_region_materialization_tensor_bytes
                >= int(full_frame_tensor_bytes)
            ):
                raise ValueError("bounded-region read materialized a full-frame-sized region")
        elif (
            self.full_frame_materialization_count < 1
            or self.maximum_full_frame_materialization_tensor_bytes < 12
            or self.bounded_region_materialization_count != 0
            or self.maximum_bounded_region_materialization_tensor_bytes != 0
            or self.source_visible_peak_logical_tensor_bytes_upper_bound
            < logical_bytes + self.maximum_full_frame_materialization_tensor_bytes
        ):
            raise ValueError("full-frame fallback omitted its materialization accounting")
        if (
            full_frame_tensor_bytes is not None
            and self.maximum_full_frame_materialization_tensor_bytes
            not in {0, int(full_frame_tensor_bytes)}
        ):
            raise ValueError("selected-pixel full-frame byte accounting changed")


def _selected_pixel_tensor_signature(tensor: torch.Tensor) -> tuple[object, ...]:
    return (
        tuple(int(value) for value in tensor.shape),
        tuple(int(value) for value in tensor.stride()),
        str(tensor.dtype),
        str(tensor.device),
        int(tensor.storage_offset()),
        int(tensor.untyped_storage().data_ptr()),
        int(getattr(tensor, "_version", 0)),
    )


def _normalize_rgb8_numpy_to_cpu_f32(array: Any) -> torch.Tensor:
    """Match the canonical image/video decoder's NumPy float32 division."""

    import numpy as np

    normalized = np.asarray(array, dtype=np.float32)
    normalized /= 255.0
    return torch.from_numpy(np.ascontiguousarray(normalized))


@dataclass(frozen=True)
class ResidentPowerFoamTargetSource:
    """Current bridge: decoded RGB remains resident on CPU, never on the accelerator."""

    frames: torch.Tensor

    def __post_init__(self) -> None:
        if self.frames.ndim != 5 or int(self.frames.shape[2]) != 3:
            raise ValueError("PowerFoam target source requires normalized RGB frames [view, frame, 3, height, width]")
        if min(int(value) for value in self.frames.shape) < 1:
            raise ValueError("PowerFoam target source dimensions must be positive")
        if self.frames.device.type != "cpu":
            raise ValueError("resident PowerFoam target source must be CPU-backed")
        if self.frames.dtype != torch.float32:
            raise ValueError("PowerFoam target source requires normalized float32 RGB frames")

    @property
    def view_count(self) -> int:
        return int(self.frames.shape[0])

    @property
    def frame_count(self) -> int:
        return int(self.frames.shape[1])

    @property
    def height(self) -> int:
        return int(self.frames.shape[-2])

    @property
    def width(self) -> int:
        return int(self.frames.shape[-1])

    def select_view_frames(
        self,
        view_indices: tuple[int, ...],
        frame_indices: tuple[int, ...],
    ) -> torch.Tensor:
        views = torch.tensor(view_indices, device="cpu", dtype=torch.long)
        frames = torch.tensor(frame_indices, device="cpu", dtype=torch.long)
        return self.frames[views, frames]

    def select_view_frame_pixels_cpu(
        self,
        view_indices: tuple[int, ...],
        frame_indices: tuple[int, ...],
        pixel_indices: tuple[int, ...],
        *,
        maximum_source_decode_tensor_bytes: int,
    ) -> PowerFoamSelectedPixelRead:
        """Gather exact pixels without flattening/copying the resident video."""

        observation_count = len(pixel_indices)
        required_peak_bytes = observation_count * (5 * 8 + 3 * 4)
        if required_peak_bytes > int(maximum_source_decode_tensor_bytes):
            raise MemoryError(
                "resident selected-pixel read exceeds its source-decode budget"
            )
        views = torch.tensor(view_indices, device="cpu", dtype=torch.long)
        frames = torch.tensor(frame_indices, device="cpu", dtype=torch.long)
        pixels = torch.tensor(pixel_indices, device="cpu", dtype=torch.long)
        rows = torch.div(pixels, self.width, rounding_mode="floor")
        columns = torch.remainder(pixels, self.width)
        selected = self.frames[views, frames, :, rows, columns].contiguous()
        return PowerFoamSelectedPixelRead.seal(
            selected,
            selection_mode="direct_pixels",
            source_provenance="resident_cpu_tensor/direct_advanced_index_v1",
            source_visible_peak_logical_tensor_bytes_upper_bound=(
                required_peak_bytes
            ),
        )

    def residency(self) -> dict[str, Any]:
        return {
            "source_kind": "resident_cpu_tensor",
            "source_device": "cpu",
            "logical_bytes": int(self.frames.numel() * self.frames.element_size()),
            "resident_bytes": int(self.frames.untyped_storage().nbytes()),
            "full_source_resident": True,
            "disk_lazy_decode": False,
        }


@dataclass(frozen=True)
class PathPowerFoamTargetSource:
    """Decode only selected RGB image paths; no decoded target video stays resident."""

    frame_paths: tuple[tuple[Path, ...], ...]
    image_crop_modes: tuple[str, ...]
    height: int
    width: int

    def __post_init__(self) -> None:
        if not self.frame_paths or not self.frame_paths[0]:
            raise ValueError("path-backed PowerFoam targets require a non-empty view/frame grid")
        frame_count = len(self.frame_paths[0])
        if any(len(paths) != frame_count for paths in self.frame_paths):
            raise ValueError("path-backed PowerFoam target views must have equal frame counts")
        if len(self.image_crop_modes) != len(self.frame_paths):
            raise ValueError("path-backed PowerFoam targets require one crop mode per view")
        if int(self.height) < 1 or int(self.width) < 1:
            raise ValueError("path-backed PowerFoam target dimensions must be positive")
        missing = next(
            (path for paths in self.frame_paths for path in paths if not path.is_file()),
            None,
        )
        if missing is not None:
            raise FileNotFoundError(f"PowerFoam target frame does not exist: {missing}")
        unsupported = next(
            (
                mode
                for mode in self.image_crop_modes
                if str(mode or "resize").lower() not in {"resize", "none", "center_square", "center_crop", "center"}
            ),
            None,
        )
        if unsupported is not None:
            raise ValueError(f"unsupported PowerFoam target crop mode: {unsupported!r}")

    @property
    def view_count(self) -> int:
        return len(self.frame_paths)

    @property
    def frame_count(self) -> int:
        return len(self.frame_paths[0])

    def _decode(self, path: Path, *, image_crop_mode: str) -> torch.Tensor:
        import numpy as np
        from PIL import Image

        with Image.open(path) as image:
            rgb = image.convert("RGB")
            mode = str(image_crop_mode or "resize").lower()
            if mode in {"center_square", "center_crop", "center"}:
                source_width, source_height = rgb.size
                side = min(source_width, source_height)
                left = (source_width - side) // 2
                top = (source_height - side) // 2
                rgb = rgb.crop((left, top, left + side, top + side))
            resampling = getattr(Image, "Resampling", Image).BILINEAR
            rgb = rgb.resize((int(self.width), int(self.height)), resample=resampling)
            array = np.asarray(rgb, dtype=np.uint8)
        return _normalize_rgb8_numpy_to_cpu_f32(array).permute(2, 0, 1).contiguous()

    def select_view_frames(
        self,
        view_indices: tuple[int, ...],
        frame_indices: tuple[int, ...],
    ) -> torch.Tensor:
        decoded: dict[tuple[int, int], torch.Tensor] = {}
        selected = []
        for view, frame in zip(view_indices, frame_indices, strict=True):
            key = (view, frame)
            if key not in decoded:
                decoded[key] = self._decode(
                    self.frame_paths[view][frame],
                    image_crop_mode=self.image_crop_modes[view],
                )
            selected.append(decoded[key])
        return torch.stack(selected)

    def residency(self) -> dict[str, Any]:
        return {
            "source_kind": "path_backed_images",
            "source_device": "disk",
            "logical_bytes": self.view_count * self.frame_count * 3 * self.height * self.width * 4,
            "resident_bytes": 0,
            "full_source_resident": False,
            "disk_lazy_decode": True,
            "path_count": self.view_count * self.frame_count,
        }


@dataclass(frozen=True)
class VideoSeekPowerFoamTargetSource:
    """Decode only requested logical frames from synchronized camera MP4s."""

    frame_sources: tuple[MulticamVideoFrameSource, ...]

    def __post_init__(self) -> None:
        if not self.frame_sources:
            raise ValueError("video-seek PowerFoam targets require at least one camera")
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
            raise ValueError("video-seek PowerFoam camera sources must share frame count and dimensions")

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

    def select_view_frames(
        self,
        view_indices: tuple[int, ...],
        frame_indices: tuple[int, ...],
    ) -> torch.Tensor:
        output: list[torch.Tensor | None] = [None] * len(view_indices)
        for view in dict.fromkeys(view_indices):
            slots = [index for index, value in enumerate(view_indices) if value == view]
            source = self.frame_sources[view]
            logical_indices = tuple(source.selected_frame_indices[frame_indices[index]] for index in slots)
            decoded = load_multicam_val_selected_camera_frames(
                video_path=source.video_path,
                start_seconds=source.start_seconds,
                fps=source.sample_fps,
                frame_count=source.source_frame_count,
                sample_indices=logical_indices,
                target_size=(source.height, source.width),
                device=torch.device("cpu"),
            )
            for slot, frame in zip(slots, decoded, strict=True):
                output[slot] = frame
        if any(frame is None for frame in output):
            raise RuntimeError("video-seek target decoder did not fill every requested sample")
        return torch.stack([frame for frame in output if frame is not None])

    def tensor_content_identity(self, *, chunk_frames: int = 16) -> dict[str, Any]:
        """Reproduce the eager decoded tensor hash with bounded frame residency."""

        if int(chunk_frames) < 1:
            raise ValueError("video target identity chunk_frames must be positive")
        metadata = {
            "dtype": str(torch.float32),
            "shape": [self.view_count, self.frame_count, 3, self.height, self.width],
            "bytes": self.view_count * self.frame_count * 3 * self.height * self.width * 4,
            "byte_order": f"native_{sys.byteorder}_endian",
            "layout": "contiguous_c_order",
        }
        digest = hashlib.sha256()
        digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
        for view in range(self.view_count):
            for start in range(0, self.frame_count, int(chunk_frames)):
                stop = min(start + int(chunk_frames), self.frame_count)
                frames = self.select_view_frames(
                    (view,) * (stop - start),
                    tuple(range(start, stop)),
                )
                digest.update(memoryview(frames.contiguous().numpy()).cast("B"))
        return {**metadata, "sha256": digest.hexdigest()}

    def residency(self) -> dict[str, Any]:
        return {
            "source_kind": "video_seek_mp4",
            "source_device": "disk",
            "logical_bytes": self.view_count * self.frame_count * 3 * self.height * self.width * 4,
            "resident_bytes": 0,
            "full_source_resident": False,
            "disk_lazy_decode": True,
            "video_count": self.view_count,
            "selected_frame_count_per_video": self.frame_count,
        }


_MAPPED_RGB8_SCHEMA = "dynaworld.powerfoam_mapped_rgb8/v1"
_MAPPED_RGB8_LAYOUT = "height_width_frame_rgb_interleaved"
_MAXIMUM_MAPPED_RGB8_MANIFEST_BYTES = 1024 * 1024
_MAPPED_RGB8_MANIFEST_KEYS = {
    "schema",
    "layout",
    "dtype",
    "height",
    "width",
    "stored_frame_indices",
    "views",
}
_MAPPED_RGB8_VIEW_KEYS = {
    "view_id",
    "payload",
    "payload_bytes",
    "payload_sha256",
}


def _streaming_open_file_sha256(handle: Any) -> str:
    digest = hashlib.sha256()
    handle.seek(0)
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object repeats key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> Any:
    raise ValueError(f"mapped RGB8 manifest contains nonstandard JSON constant {value!r}")


def _file_stat_signature(stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


@dataclass(frozen=True)
class MappedRgb8PowerFoamTargetSource:
    """Pixel-time RGB8 cache mapped only during one selected-pixel read.

    Payloads use ``[height,width,stored_frame,RGB]`` byte order.  This makes a
    spatial track's temporal samples contiguous while avoiding a persistent
    mapping or decoded video.  Full-frame reads deliberately delegate to the
    original image/video source because this layout is training-path specific.
    """

    manifest_path: Path
    manifest_sha256: str
    view_ids: tuple[str, ...]
    payload_paths: tuple[Path, ...]
    payload_sha256s: tuple[str, ...]
    payload_bytes: tuple[int, ...]
    payload_stat_signatures: tuple[tuple[int, int, int, int, int], ...]
    maximum_mapped_payload_bytes: int
    maximum_total_payload_verification_bytes: int
    stored_frame_indices: tuple[int, ...]
    logical_frame_indices: tuple[int, ...]
    logical_to_stored_indices: tuple[int, ...]
    height: int
    width: int
    full_frame_source: PowerFoamTargetSource | None = None

    @classmethod
    def from_manifest(
        cls,
        manifest_path: Path | str,
        *,
        maximum_mapped_payload_bytes: int,
        maximum_total_payload_verification_bytes: int,
        expected_view_ids: tuple[str, ...] | None = None,
        logical_frame_indices: tuple[int, ...] | None = None,
        full_frame_source: PowerFoamTargetSource | None = None,
    ) -> MappedRgb8PowerFoamTargetSource:
        path = Path(manifest_path)
        with path.open("rb") as handle:
            manifest_stat_before = _file_stat_signature(os.fstat(handle.fileno()))
            if manifest_stat_before[2] > _MAXIMUM_MAPPED_RGB8_MANIFEST_BYTES:
                raise MemoryError("mapped RGB8 manifest exceeds its byte cap")
            raw_manifest = handle.read(_MAXIMUM_MAPPED_RGB8_MANIFEST_BYTES + 1)
            manifest_stat_after = _file_stat_signature(os.fstat(handle.fileno()))
        if (
            len(raw_manifest) > _MAXIMUM_MAPPED_RGB8_MANIFEST_BYTES
            or manifest_stat_before != manifest_stat_after
            or len(raw_manifest) != manifest_stat_after[2]
        ):
            raise ValueError("mapped RGB8 manifest changed during bounded read")
        manifest_sha256 = hashlib.sha256(raw_manifest).hexdigest()
        payload = json.loads(
            raw_manifest,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_nonfinite_json_constant,
        )
        if not isinstance(payload, dict):
            raise ValueError("mapped RGB8 manifest must be a JSON object")
        if set(payload) != _MAPPED_RGB8_MANIFEST_KEYS:
            raise ValueError("mapped RGB8 manifest keys changed")
        if (
            isinstance(maximum_mapped_payload_bytes, bool)
            or not isinstance(maximum_mapped_payload_bytes, int)
            or maximum_mapped_payload_bytes < 1
        ):
            raise ValueError("mapped RGB8 payload-address-space cap must be positive")
        if (
            isinstance(maximum_total_payload_verification_bytes, bool)
            or not isinstance(maximum_total_payload_verification_bytes, int)
            or maximum_total_payload_verification_bytes < 1
        ):
            raise ValueError("mapped RGB8 total payload-verification cap must be positive")
        if (
            payload.get("schema") != _MAPPED_RGB8_SCHEMA
            or payload.get("layout") != _MAPPED_RGB8_LAYOUT
            or payload.get("dtype") != "uint8"
        ):
            raise ValueError("mapped RGB8 manifest has an unsupported schema/layout/dtype")
        height = payload.get("height")
        width = payload.get("width")
        stored_frames_raw = payload.get("stored_frame_indices")
        views_raw = payload.get("views")
        if (
            isinstance(height, bool)
            or not isinstance(height, int)
            or height < 1
            or isinstance(width, bool)
            or not isinstance(width, int)
            or width < 1
            or not isinstance(stored_frames_raw, list)
            or not stored_frames_raw
            or not isinstance(views_raw, list)
            or not views_raw
        ):
            raise ValueError("mapped RGB8 manifest dimensions, frames, or views are invalid")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in stored_frames_raw):
            raise ValueError("mapped RGB8 stored frame indices must be integers")
        stored_frame_indices = tuple(int(value) for value in stored_frames_raw)
        if (
            any(value < 0 for value in stored_frame_indices)
            or tuple(sorted(set(stored_frame_indices))) != stored_frame_indices
        ):
            raise ValueError("mapped RGB8 stored frame indices must be unique and increasing")

        records: dict[str, dict[str, Any]] = {}
        for record in views_raw:
            if not isinstance(record, dict):
                raise ValueError("mapped RGB8 view records must be objects")
            if set(record) != _MAPPED_RGB8_VIEW_KEYS:
                raise ValueError("mapped RGB8 view-record keys changed")
            view_id = record.get("view_id")
            if (
                not isinstance(view_id, str)
                or not view_id.strip()
                or view_id != view_id.strip()
                or view_id in records
            ):
                raise ValueError("mapped RGB8 view ids must be unique nonempty strings")
            records[view_id] = record
        if tuple(records) != tuple(sorted(records)):
            raise ValueError("mapped RGB8 manifest views must be sorted by view id")
        if expected_view_ids is None:
            selected_view_ids = tuple(records)
        else:
            if any(not isinstance(value, str) for value in expected_view_ids):
                raise ValueError("mapped RGB8 expected view ids must be strings")
            selected_view_ids = tuple(expected_view_ids)
        if (
            not selected_view_ids
            or len(set(selected_view_ids)) != len(selected_view_ids)
            or any(not value or value != value.strip() for value in selected_view_ids)
        ):
            raise ValueError("mapped RGB8 expected view ids must be unique and nonempty")
        missing_view = next((view for view in selected_view_ids if view not in records), None)
        if missing_view is not None:
            raise KeyError(f"mapped RGB8 manifest has no view {missing_view!r}")

        expected_payload_bytes = height * width * len(stored_frame_indices) * 3
        if expected_payload_bytes > maximum_mapped_payload_bytes:
            raise MemoryError("mapped RGB8 payload exceeds its address-space cap")
        expected_verification_bytes = expected_payload_bytes * len(selected_view_ids)
        if expected_verification_bytes > maximum_total_payload_verification_bytes:
            raise MemoryError("mapped RGB8 payload set exceeds its verification-I/O cap")
        payload_paths: list[Path] = []
        payload_sha256s: list[str] = []
        payload_sizes: list[int] = []
        payload_stat_signatures: list[tuple[int, int, int, int, int]] = []
        payload_root = path.parent.resolve()
        for view_id in selected_view_ids:
            record = records[view_id]
            relative_payload = record.get("payload")
            declared_bytes = record.get("payload_bytes")
            declared_sha256 = record.get("payload_sha256")
            if (
                not isinstance(relative_payload, str)
                or not relative_payload.strip()
                or any(
                    ord(character) < 32 or ord(character) == 127
                    for character in relative_payload
                )
                or Path(relative_payload).is_absolute()
                or ".." in Path(relative_payload).parts
                or Path(relative_payload).as_posix() != relative_payload
                or isinstance(declared_bytes, bool)
                or not isinstance(declared_bytes, int)
                or declared_bytes != expected_payload_bytes
                or not _is_sha256(declared_sha256)
            ):
                raise ValueError(f"mapped RGB8 view {view_id!r} has invalid payload metadata")
            payload_path = (path.parent / relative_payload).resolve()
            try:
                payload_path.relative_to(payload_root)
            except ValueError as error:
                raise ValueError(
                    f"mapped RGB8 payload escaped its manifest directory for view {view_id!r}"
                ) from error
            with payload_path.open("rb") as handle:
                stat_before = _file_stat_signature(os.fstat(handle.fileno()))
                if stat_before[2] != declared_bytes:
                    raise ValueError(f"mapped RGB8 payload size changed for view {view_id!r}")
                if _streaming_open_file_sha256(handle) != declared_sha256:
                    raise ValueError(f"mapped RGB8 payload digest changed for view {view_id!r}")
                stat_after = _file_stat_signature(os.fstat(handle.fileno()))
            if stat_before != stat_after:
                raise ValueError(f"mapped RGB8 payload changed while verifying view {view_id!r}")
            payload_paths.append(payload_path)
            payload_sha256s.append(declared_sha256)
            payload_sizes.append(declared_bytes)
            payload_stat_signatures.append(stat_after)

        if logical_frame_indices is None:
            selected_logical_frames = stored_frame_indices
        else:
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in logical_frame_indices
            ):
                raise ValueError("mapped RGB8 logical frame selection must use integers")
            selected_logical_frames = tuple(logical_frame_indices)
        if not selected_logical_frames or len(set(selected_logical_frames)) != len(selected_logical_frames):
            raise ValueError("mapped RGB8 logical frame selection must be unique and nonempty")
        stored_lookup = {frame: index for index, frame in enumerate(stored_frame_indices)}
        missing_frame = next(
            (frame for frame in selected_logical_frames if frame not in stored_lookup),
            None,
        )
        if missing_frame is not None:
            raise KeyError(f"mapped RGB8 cache has no stored frame {missing_frame}")
        logical_to_stored = tuple(stored_lookup[frame] for frame in selected_logical_frames)
        result = cls(
            manifest_path=path,
            manifest_sha256=manifest_sha256,
            view_ids=selected_view_ids,
            payload_paths=tuple(payload_paths),
            payload_sha256s=tuple(payload_sha256s),
            payload_bytes=tuple(payload_sizes),
            payload_stat_signatures=tuple(payload_stat_signatures),
            maximum_mapped_payload_bytes=maximum_mapped_payload_bytes,
            maximum_total_payload_verification_bytes=(
                maximum_total_payload_verification_bytes
            ),
            stored_frame_indices=stored_frame_indices,
            logical_frame_indices=selected_logical_frames,
            logical_to_stored_indices=logical_to_stored,
            height=height,
            width=width,
            full_frame_source=full_frame_source,
        )
        result._assert_delegate_compatible()
        return result

    @property
    def view_count(self) -> int:
        return len(self.view_ids)

    @property
    def frame_count(self) -> int:
        return len(self.logical_frame_indices)

    def _assert_delegate_compatible(self) -> None:
        if self.full_frame_source is None:
            return
        if (
            int(self.full_frame_source.view_count) != self.view_count
            or int(self.full_frame_source.frame_count) != self.frame_count
            or int(self.full_frame_source.height) != self.height
            or int(self.full_frame_source.width) != self.width
        ):
            raise ValueError("mapped RGB8 full-frame delegate changed the selected grid")

    def select_view_frames(
        self,
        view_indices: tuple[int, ...],
        frame_indices: tuple[int, ...],
    ) -> torch.Tensor:
        if self.full_frame_source is None:
            raise RuntimeError(
                "pixel-time RGB8 cache requires an explicit full-frame delegate for frame reads"
            )
        return self.full_frame_source.select_view_frames(view_indices, frame_indices)

    def select_view_frame_pixels_cpu(
        self,
        view_indices: tuple[int, ...],
        frame_indices: tuple[int, ...],
        pixel_indices: tuple[int, ...],
        *,
        maximum_source_decode_tensor_bytes: int,
    ) -> PowerFoamSelectedPixelRead:
        import numpy as np

        observation_count = len(pixel_indices)
        if (
            observation_count < 1
            or len(view_indices) != observation_count
            or len(frame_indices) != observation_count
        ):
            raise ValueError("mapped RGB8 selected-pixel request is empty or ragged")
        invalid_view = next(
            (value for value in view_indices if value < 0 or value >= self.view_count),
            None,
        )
        invalid_frame = next(
            (value for value in frame_indices if value < 0 or value >= self.frame_count),
            None,
        )
        invalid_pixel = next(
            (value for value in pixel_indices if value < 0 or value >= self.height * self.width),
            None,
        )
        if invalid_view is not None:
            raise IndexError("mapped RGB8 selected-pixel request left its view grid")
        if invalid_frame is not None:
            raise IndexError("mapped RGB8 selected-pixel request left its frame grid")
        if invalid_pixel is not None:
            raise IndexError("mapped RGB8 selected-pixel request left its pixel grid")

        # Output, five int64 index families, NumPy's advanced-index RGB8 copy,
        # our detached RGB8 copy, and the temporary normalized float32
        # selection can overlap. The mmap and its page coverage are reported
        # separately because address space is not a logical tensor allocation.
        source_visible_peak_bytes = observation_count * (
            12 + 5 * 8 + 2 * 3 + 12
        )
        if source_visible_peak_bytes > int(maximum_source_decode_tensor_bytes):
            raise MemoryError("mapped RGB8 selected-pixel read exceeds its source-decode budget")

        page_size = int(mmap.PAGESIZE)
        positions_by_view: dict[int, list[int]] = {}
        for position, view in enumerate(view_indices):
            positions_by_view.setdefault(int(view), []).append(position)
        maximum_mapped_bytes = max(self.payload_bytes[view] for view in positions_by_view)
        maximum_page_count = 0
        total_page_count = 0
        for view, positions in positions_by_view.items():
            touched_pages: set[int] = set()
            for position in positions:
                pixel = int(pixel_indices[position])
                stored_frame = self.logical_to_stored_indices[int(frame_indices[position])]
                byte_offset = (pixel * len(self.stored_frame_indices) + stored_frame) * 3
                touched_pages.add(byte_offset // page_size)
                touched_pages.add((byte_offset + 2) // page_size)
            maximum_page_count = max(maximum_page_count, len(touched_pages))
            total_page_count += len(touched_pages)

        output = torch.empty((observation_count, 3), dtype=torch.float32, device="cpu")
        stored_frame_count = len(self.stored_frame_indices)
        for view, positions in positions_by_view.items():
            pixels = np.asarray([pixel_indices[position] for position in positions], dtype=np.int64)
            rows = pixels // self.width
            columns = pixels % self.width
            stored_frames = np.asarray(
                [self.logical_to_stored_indices[frame_indices[position]] for position in positions],
                dtype=np.int64,
            )
            with self.payload_paths[view].open("rb") as handle:
                if (
                    _file_stat_signature(os.fstat(handle.fileno()))
                    != self.payload_stat_signatures[view]
                ):
                    raise ValueError("mapped RGB8 payload changed after manifest verification")
                with mmap.mmap(handle.fileno(), length=0, access=mmap.ACCESS_READ) as mapping:
                    payload = np.ndarray(
                        (self.height, self.width, stored_frame_count, 3),
                        dtype=np.uint8,
                        buffer=mapping,
                    )
                    selected_u8 = np.array(
                        payload[rows, columns, stored_frames, :],
                        dtype=np.uint8,
                        copy=True,
                        order="C",
                    )
                    del payload
                if (
                    _file_stat_signature(os.fstat(handle.fileno()))
                    != self.payload_stat_signatures[view]
                ):
                    raise ValueError("mapped RGB8 payload changed during selected-pixel read")
            selected_f32 = _normalize_rgb8_numpy_to_cpu_f32(selected_u8)
            destination = torch.tensor(positions, dtype=torch.long, device="cpu")
            output.index_copy_(0, destination, selected_f32)
            del destination, selected_f32, selected_u8, stored_frames, columns, rows, pixels

        return PowerFoamSelectedPixelRead.seal(
            output.contiguous(),
            selection_mode="direct_pixels",
            source_provenance=(
                f"mapped_rgb8_pixel_time_v1/manifest_sha256={self.manifest_sha256}/"
                f"maximum_mapped_payload_bytes={self.maximum_mapped_payload_bytes}/"
                "maximum_total_payload_verification_bytes="
                f"{self.maximum_total_payload_verification_bytes}"
            ),
            source_visible_peak_logical_tensor_bytes_upper_bound=(
                source_visible_peak_bytes
            ),
            transient_mapped_address_space_bytes=maximum_mapped_bytes,
            maximum_requested_unique_mapped_page_count=maximum_page_count,
            total_requested_unique_mapped_page_count=total_page_count,
            mapped_page_size_bytes=page_size,
            maximum_requested_mapped_page_bytes_upper_bound=(
                maximum_page_count * page_size
            ),
            total_requested_mapped_page_bytes_upper_bound=(
                total_page_count * page_size
            ),
            mapping_closed_before_return=True,
        )

    def residency(self) -> dict[str, Any]:
        frame_map_sha256 = hashlib.sha256(
            json.dumps(
                self.logical_frame_indices,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return {
            "source_kind": "mapped_rgb8_pixel_time_v1",
            "source_device": "disk",
            "logical_bytes": self.view_count * self.frame_count * 3 * self.height * self.width * 4,
            "raw_storage_bytes": sum(self.payload_bytes),
            "maximum_mapped_payload_bytes": self.maximum_mapped_payload_bytes,
            "maximum_total_payload_verification_bytes": (
                self.maximum_total_payload_verification_bytes
            ),
            "construction_payload_verification_bytes": sum(self.payload_bytes),
            "construction_full_payload_scan": True,
            "resident_bytes": 0,
            "full_source_resident": False,
            "disk_lazy_decode": True,
            "layout": _MAPPED_RGB8_LAYOUT,
            "dtype": "uint8",
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "maximum_manifest_bytes": _MAXIMUM_MAPPED_RGB8_MANIFEST_BYTES,
            "payload_sha256s": self.payload_sha256s,
            "view_ids": self.view_ids,
            "stored_frame_count": len(self.stored_frame_indices),
            "selected_frame_count": self.frame_count,
            "logical_frame_map_sha256": frame_map_sha256,
            "mapping_lifetime": "one_selected_pixel_read",
            "mapping_closed_before_return": True,
            "requested_page_coverage_is_not_residency_measurement": True,
            "system_page_cache_peak_measured": False,
            "full_frame_reads_delegated": self.full_frame_source is not None,
        }

    def tensor_content_identity(self) -> dict[str, Any]:
        """Return the canonical decoded-f32 tensor identity via the delegate.

        A manifest/frame-map digest is provenance, not a substitute for the
        canonical content hash used to compare paper lanes. The deferred MP4
        delegate computes that exact hash in bounded frame chunks. Cache-to-
        delegate equality remains a separate converter/dataset-binding gate.
        """

        identity_builder = getattr(
            self.full_frame_source,
            "tensor_content_identity",
            None,
        )
        if not callable(identity_builder):
            raise RuntimeError(
                "mapped RGB8 paper identity requires a canonical full-frame "
                "identity delegate"
            )
        identity = identity_builder()
        expected_shape = [
            self.view_count,
            self.frame_count,
            3,
            self.height,
            self.width,
        ]
        if (
            not isinstance(identity, dict)
            or identity.get("dtype") != str(torch.float32)
            or identity.get("shape") != expected_shape
            or identity.get("bytes")
            != self.view_count
            * self.frame_count
            * 3
            * self.height
            * self.width
            * 4
            or identity.get("byte_order")
            != f"native_{sys.byteorder}_endian"
            or identity.get("layout") != "contiguous_c_order"
            or not _is_sha256(identity.get("sha256"))
        ):
            raise ValueError("mapped RGB8 identity delegate changed the canonical tensor contract")
        return identity


@dataclass(frozen=True)
class PowerFoamTargetProvider:
    """Select camera-time RGB batches before optional resize/device transfer."""

    source: PowerFoamTargetSource
    device: torch.device

    @classmethod
    def from_resident_frames(
        cls,
        frames: torch.Tensor,
        *,
        device: torch.device,
    ) -> PowerFoamTargetProvider:
        return cls(source=ResidentPowerFoamTargetSource(frames), device=device)

    @property
    def view_count(self) -> int:
        return int(self.source.view_count)

    @property
    def frame_count(self) -> int:
        return int(self.source.frame_count)

    @property
    def sample_count(self) -> int:
        return self.view_count * self.frame_count

    @property
    def height(self) -> int:
        return int(self.source.height)

    @property
    def width(self) -> int:
        return int(self.source.width)

    def residency(self) -> dict[str, Any]:
        source = dict(self.source.residency())
        source.update(
            {
                "output_device": str(self.device),
                "selection_mode": "selected_batch_only",
                "full_target_accelerator_resident_bytes": 0,
                "selected_pixel_api_available": True,
                "native_selected_pixel_method_available": callable(
                    getattr(self.source, "select_view_frame_pixels_cpu", None)
                ),
            }
        )
        return source

    @property
    def native_selected_pixel_method_available(self) -> bool:
        return callable(getattr(self.source, "select_view_frame_pixels_cpu", None))

    @torch.no_grad()
    def select_view_frame_pixels_cpu(
        self,
        view_indices: tuple[int, ...] | list[int] | torch.Tensor,
        frame_indices: tuple[int, ...] | list[int] | torch.Tensor,
        pixel_indices: tuple[int, ...] | list[int] | torch.Tensor,
        *,
        maximum_source_decode_tensor_bytes: int,
    ) -> PowerFoamSelectedPixelRead:
        """Read one ordered RGB row per ``(view, frame, pixel)`` tuple.

        Sources with a native selected-pixel method avoid full-frame
        materialization.  Legacy image/video sources use an explicitly marked
        full-frame compatibility fallback; that mode is never paper-memory
        acceptance-capable.
        """

        views = tuple(
            int(value)
            for value in torch.as_tensor(
                view_indices,
                device="cpu",
                dtype=torch.long,
            ).reshape(-1).tolist()
        )
        frames = tuple(
            int(value)
            for value in torch.as_tensor(
                frame_indices,
                device="cpu",
                dtype=torch.long,
            ).reshape(-1).tolist()
        )
        pixels = tuple(
            int(value)
            for value in torch.as_tensor(
                pixel_indices,
                device="cpu",
                dtype=torch.long,
            ).reshape(-1).tolist()
        )
        if not views or len(views) != len(frames) or len(views) != len(pixels):
            raise ValueError(
                "PowerFoam selected-pixel view/frame/pixel inputs must be "
                "non-empty and equally sized"
            )
        if (
            isinstance(maximum_source_decode_tensor_bytes, bool)
            or not isinstance(maximum_source_decode_tensor_bytes, int)
            or maximum_source_decode_tensor_bytes < 1
        ):
            raise ValueError("selected-pixel source-decode budget must be positive")
        invalid_view = next(
            (value for value in views if value < 0 or value >= self.view_count),
            None,
        )
        if invalid_view is not None:
            raise IndexError(
                f"PowerFoam target view index {invalid_view} is outside "
                f"[0, {self.view_count})"
            )
        invalid_frame = next(
            (value for value in frames if value < 0 or value >= self.frame_count),
            None,
        )
        if invalid_frame is not None:
            raise IndexError(
                f"PowerFoam target frame index {invalid_frame} is outside "
                f"[0, {self.frame_count})"
            )
        pixel_count = self.height * self.width
        invalid_pixel = next(
            (value for value in pixels if value < 0 or value >= pixel_count),
            None,
        )
        if invalid_pixel is not None:
            raise IndexError(
                f"PowerFoam target pixel index {invalid_pixel} is outside "
                f"[0, {pixel_count})"
            )

        native_read = getattr(self.source, "select_view_frame_pixels_cpu", None)
        if callable(native_read):
            result = native_read(
                views,
                frames,
                pixels,
                maximum_source_decode_tensor_bytes=(
                    maximum_source_decode_tensor_bytes
                ),
            )
            if not isinstance(result, PowerFoamSelectedPixelRead):
                raise TypeError(
                    "native selected-pixel source returned an unsealed payload"
                )
            result.assert_valid(
                expected_observation_count=len(views),
                full_frame_tensor_bytes=3 * self.height * self.width * 4,
            )
            if (
                result.source_visible_peak_logical_tensor_bytes_upper_bound
                > maximum_source_decode_tensor_bytes
            ):
                raise MemoryError(
                    "native selected-pixel source exceeded its preflight budget"
                )
            return result

        positions_by_frame: dict[tuple[int, int], list[int]] = {}
        for position, key in enumerate(zip(views, frames, strict=True)):
            positions_by_frame.setdefault(key, []).append(position)
        full_frame_bytes = 3 * self.height * self.width * 4
        maximum_frame_observation_count = max(
            len(positions) for positions in positions_by_frame.values()
        )
        peak_logical_bytes = (
            len(views) * 3 * 4
            + full_frame_bytes
            + maximum_frame_observation_count * (4 * 8 + 3 * 4)
        )
        if peak_logical_bytes > maximum_source_decode_tensor_bytes:
            raise MemoryError(
                "full-frame selected-pixel fallback exceeds its source-decode budget"
            )
        output = torch.empty(
            (len(views), 3),
            dtype=torch.float32,
            device="cpu",
        )
        for (view, frame), positions in positions_by_frame.items():
            decoded = self.source.select_view_frames((view,), (frame,))
            expected_shape = (1, 3, self.height, self.width)
            if (
                tuple(decoded.shape) != expected_shape
                or decoded.dtype != torch.float32
                or decoded.device.type != "cpu"
            ):
                raise ValueError(
                    "PowerFoam full-frame selected-pixel fallback violated its "
                    f"CPU float32 {expected_shape} contract"
                )
            destination = torch.tensor(positions, dtype=torch.long, device="cpu")
            selected_pixels = torch.tensor(
                [pixels[position] for position in positions],
                dtype=torch.long,
                device="cpu",
            )
            selected_rows = torch.div(
                selected_pixels,
                self.width,
                rounding_mode="floor",
            )
            selected_columns = torch.remainder(selected_pixels, self.width)
            selected = decoded[0].permute(1, 2, 0)[
                selected_rows,
                selected_columns,
            ].contiguous()
            output.index_copy_(0, destination, selected)
            del selected
            del selected_columns
            del selected_rows
            del selected_pixels
            del destination
            del decoded
        source_kind = str(self.source.residency().get("source_kind", "unknown"))
        return PowerFoamSelectedPixelRead.seal(
            output,
            selection_mode="full_frame_fallback",
            source_provenance=f"{source_kind}/one_frame_at_a_time_fallback_v1",
            source_visible_peak_logical_tensor_bytes_upper_bound=peak_logical_bytes,
            full_frame_materialization_count=len(positions_by_frame),
            maximum_full_frame_materialization_tensor_bytes=full_frame_bytes,
        )

    @torch.no_grad()
    def select(
        self,
        sample_indices: torch.Tensor,
        *,
        height: int | None = None,
        width: int | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        flat_indices = sample_indices.detach().to(device="cpu", dtype=torch.long).reshape(-1)
        if flat_indices.numel() < 1:
            raise ValueError("PowerFoam target selection requires at least one sample")
        if bool(((flat_indices < 0) | (flat_indices >= self.sample_count)).any()):
            invalid = int(flat_indices[((flat_indices < 0) | (flat_indices >= self.sample_count))][0])
            raise IndexError(f"PowerFoam target sample index {invalid} is outside [0, {self.sample_count})")
        view_indices = tuple(int(value) // self.frame_count for value in flat_indices.tolist())
        frame_indices = tuple(int(value) % self.frame_count for value in flat_indices.tolist())
        return self.select_view_frames(
            view_indices,
            frame_indices,
            height=height,
            width=width,
            device=device,
        )

    @torch.no_grad()
    def select_view_frames(
        self,
        view_indices: tuple[int, ...] | list[int] | torch.Tensor,
        frame_indices: tuple[int, ...] | list[int] | torch.Tensor,
        *,
        height: int | None = None,
        width: int | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        views = tuple(
            int(value) for value in torch.as_tensor(view_indices, device="cpu", dtype=torch.long).reshape(-1).tolist()
        )
        frames = tuple(
            int(value) for value in torch.as_tensor(frame_indices, device="cpu", dtype=torch.long).reshape(-1).tolist()
        )
        if not views or len(views) != len(frames):
            raise ValueError("PowerFoam target view/frame selections must be non-empty and equally sized")
        invalid_view = next((value for value in views if value < 0 or value >= self.view_count), None)
        if invalid_view is not None:
            raise IndexError(f"PowerFoam target view index {invalid_view} is outside [0, {self.view_count})")
        invalid_frame = next((value for value in frames if value < 0 or value >= self.frame_count), None)
        if invalid_frame is not None:
            raise IndexError(f"PowerFoam target frame index {invalid_frame} is outside [0, {self.frame_count})")
        target_height = self.height if height is None else int(height)
        target_width = self.width if width is None else int(width)
        if target_height < 1 or target_width < 1:
            raise ValueError("PowerFoam target selection requires positive dimensions")
        selected = self.source.select_view_frames(views, frames)
        expected_shape = (len(views), 3, self.height, self.width)
        if tuple(selected.shape) != expected_shape or selected.dtype != torch.float32:
            raise ValueError(
                "PowerFoam target source violated its normalized RGB selection contract: "
                f"expected float32 {expected_shape}, got {selected.dtype} {tuple(selected.shape)}"
            )
        if (target_height, target_width) != (self.height, self.width):
            selected = resize_video_frames(
                selected,
                normalize_image_size((target_height, target_width)),
            )
        return selected.to(
            device=self.device if device is None else device,
            dtype=torch.float32,
        )


@dataclass(frozen=True)
class PowerFoamRayProvider:
    cameras: tuple[tuple[CameraSpec, ...], ...]
    height: int
    width: int
    device: torch.device

    @property
    def view_count(self) -> int:
        return len(self.cameras)

    @property
    def frame_count(self) -> int:
        return len(self.cameras[0]) if self.cameras else 0

    @property
    def sample_count(self) -> int:
        return self.view_count * self.frame_count

    def select(
        self,
        sample_indices: torch.Tensor,
        *,
        height: int | None = None,
        width: int | None = None,
    ) -> torch.Tensor:
        flat_indices = sample_indices.detach().to(device="cpu", dtype=torch.long).tolist()
        if not flat_indices:
            raise ValueError("PowerFoam ray selection requires at least one sample")
        target_height = self.height if height is None else int(height)
        target_width = self.width if width is None else int(width)
        if target_height < 1 or target_width < 1:
            raise ValueError("PowerFoam ray selection requires positive dimensions")
        sx = float(target_width) / float(self.width)
        sy = float(target_height) / float(self.height)
        rays = []
        for index in flat_indices:
            if index < 0 or index >= self.sample_count:
                raise IndexError(f"PowerFoam ray sample index {index} is outside [0, {self.sample_count})")
            view, frame = divmod(int(index), self.frame_count)
            camera = self.cameras[view][frame]
            scaled_camera = CameraSpec(
                fx=camera.fx * sx,
                fy=camera.fy * sy,
                cx=camera.cx * sx,
                cy=camera.cy * sy,
                camera_to_world=camera.camera_to_world,
                lens_model=camera.lens_model,
                distortion=camera.distortion,
            )
            rays.append(
                powerfoam_rays_from_camera(
                    scaled_camera,
                    height=target_height,
                    width=target_width,
                    device=self.device,
                )
            )
        return torch.cat(rays, dim=0)


def _path_target_source(
    sequences: Any,
    frames: torch.Tensor,
) -> PathPowerFoamTargetSource | None:
    if not isinstance(sequences, (tuple, list)) or len(sequences) != int(frames.shape[0]):
        return None
    path_grid = tuple(tuple(Path(path) for path in getattr(sequence, "frame_paths", ())) for sequence in sequences)
    if any(len(paths) != int(frames.shape[1]) for paths in path_grid):
        return None
    if any(not path.is_file() for paths in path_grid for path in paths):
        return None
    return PathPowerFoamTargetSource(
        frame_paths=path_grid,
        image_crop_modes=tuple(str(getattr(sequence, "image_crop_mode", "resize")) for sequence in sequences),
        height=int(frames.shape[-2]),
        width=int(frames.shape[-1]),
    )


def _target_provider(
    frames: torch.Tensor,
    sequences: Any,
    *,
    device: torch.device,
    video_frame_sources: tuple[MulticamVideoFrameSource, ...] = (),
) -> PowerFoamTargetProvider:
    base_source: PowerFoamTargetSource
    if video_frame_sources:
        base_source = VideoSeekPowerFoamTargetSource(video_frame_sources)
    else:
        path_source = _path_target_source(
            sequences,
            frames,
        )
        base_source = (
            ResidentPowerFoamTargetSource(frames)
            if path_source is None
            else path_source
        )
    return PowerFoamTargetProvider(
        source=base_source,
        device=device,
    )


def _target_split_residency(
    provider: PowerFoamTargetProvider,
    compatibility_targets: torch.Tensor | None,
) -> dict[str, Any]:
    residency = provider.residency()
    compatibility_bytes = 0 if compatibility_targets is None else int(compatibility_targets.untyped_storage().nbytes())
    residency.update(
        {
            "compatibility_tensor_resident_bytes": compatibility_bytes,
            "effective_decoded_target_resident_bytes": max(
                int(residency["resident_bytes"]),
                compatibility_bytes,
            ),
            "compatibility_tensor_required_by_current_trainer": False,
            "provider_can_replace_compatibility_tensor": not bool(residency["full_source_resident"]),
        }
    )
    return residency


def resolve_powerfoam_paper_dataset_bundle(training_data: dict[str, Any]) -> dict[str, Any] | None:
    """Resolve a deferred decoded-target identity after bounded training/evaluation."""

    identity = training_data.get("paper_dataset_bundle")
    if identity is not None:
        return identity
    builder = training_data.get("paper_dataset_bundle_builder")
    if builder is None:
        return None
    identity = builder()
    if not isinstance(identity, dict):
        raise TypeError("paper dataset bundle identity builder must return a dictionary")
    training_data["paper_dataset_bundle"] = identity
    return identity


def load_powerfoam_training_data(cfg: dict[str, Any], device: torch.device) -> dict[str, Any]:
    render_size = int(cfg["render"]["render_size"])
    image_size = normalize_image_size(cfg["render"]["image_size"])
    frame_source = str(cfg["data"]["frame_source"])
    if frame_source == "multicam_val":
        stream_rays = bool(cfg.get("paper_protocol", {}).get("enabled", False))
        bundle = load_multicam_video_bundle(
            data_cfg=cfg["data"],
            camera_cfg=cfg["camera"],
            target_size=(image_size.height, image_size.width),
            device=device,
            frame_device=torch.device("cpu") if stream_rays else device,
            defer_video_frames=stream_rays,
        )
        train_cameras = cameras_from_K_w2c(
            bundle.train_K,
            bundle.train_w2c,
            lens_models=bundle.train_lens_models,
            distortions=bundle.train_distortions,
        )
        if stream_rays:
            train_ray_provider = PowerFoamRayProvider(
                cameras=train_cameras,
                height=image_size.height,
                width=image_size.width,
                device=device,
            )
            train_target_provider = _target_provider(
                bundle.train_frames,
                getattr(bundle, "train_sequences", ()),
                device=device,
                video_frame_sources=getattr(bundle, "train_frame_sources", ()),
            )
            targets = None
            sample_frame_indices = torch.arange(bundle.frame_count, device=device, dtype=torch.long).repeat(
                bundle.train_view_count
            )
            sample_rays = None
        else:
            train_rays = powerfoam_rays_from_camera_grid(
                train_cameras,
                height=image_size.height,
                width=image_size.width,
                device=device,
            )
            targets, sample_frame_indices, sample_rays = flatten_multiview_powerfoam_samples(
                bundle.train_frames.to(device=device, dtype=torch.float32),
                train_rays,
            )
            train_ray_provider = None
            train_target_provider = None

        heldout_targets = None
        heldout_frame_indices = None
        heldout_rays = None
        heldout_ray_provider = None
        heldout_target_provider = None
        if bundle.heldout_frames is not None and bundle.heldout_K is not None and bundle.heldout_w2c is not None:
            heldout_camera_grid = heldout_cameras_from_K_w2c(
                bundle.heldout_K,
                bundle.heldout_w2c,
                lens_models=bundle.heldout_lens_models,
                distortions=bundle.heldout_distortions,
            )
            if stream_rays:
                heldout_ray_provider = PowerFoamRayProvider(
                    cameras=heldout_camera_grid,
                    height=image_size.height,
                    width=image_size.width,
                    device=device,
                )
                heldout_target_provider = _target_provider(
                    bundle.heldout_frames,
                    getattr(bundle, "heldout_sequences", ()),
                    device=device,
                    video_frame_sources=getattr(bundle, "heldout_frame_sources", ()),
                )
                heldout_targets = None
                heldout_frame_indices = torch.arange(bundle.frame_count, device=device, dtype=torch.long).repeat(
                    bundle.heldout_view_count
                )
                heldout_rays = None
            else:
                heldout_ray_grid = powerfoam_rays_from_camera_grid(
                    heldout_camera_grid,
                    height=image_size.height,
                    width=image_size.width,
                    device=device,
                )
                heldout_targets, heldout_frame_indices, heldout_rays = flatten_multiview_powerfoam_samples(
                    bundle.heldout_frames.to(device=device, dtype=torch.float32),
                    heldout_ray_grid,
                )

        init_frames = None
        needs_video_init = bool(cfg.get("model", {}).get("init_from_video", False)) and not bool(
            cfg.get("model", {}).get("init_point_cloud_path")
        )
        if needs_video_init:
            if (
                stream_rays
                and train_target_provider is not None
                and not bool(train_target_provider.residency()["full_source_resident"])
            ):
                condition_camera = (
                    bundle.metadata.get("condition_camera") or bundle.train_camera_names[0]
                    if bundle.metadata is not None
                    else bundle.train_camera_names[0]
                )
                condition_view = bundle.train_camera_names.index(str(condition_camera))
                init_frames = train_target_provider.select_view_frames(
                    (condition_view,) * bundle.frame_count,
                    tuple(range(bundle.frame_count)),
                    device=torch.device("cpu"),
                )
            else:
                init_frames = bundle.condition_sequence.frames.detach().to(device="cpu")
        init_shares_train_storage = (
            init_frames is not None
            and bundle.train_frames.device.type != "meta"
            and init_frames.untyped_storage().data_ptr() == bundle.train_frames.untyped_storage().data_ptr()
        )
        deferred_target_sources = (
            VideoSeekPowerFoamTargetSource,
            MappedRgb8PowerFoamTargetSource,
        )
        deferred_target_identities = train_target_provider is not None and isinstance(
            train_target_provider.source, deferred_target_sources
        )
        if deferred_target_identities:
            if heldout_target_provider is None or not isinstance(
                heldout_target_provider.source,
                deferred_target_sources,
            ):
                raise ValueError(
                    "deferred train and heldout targets must both expose bounded identities"
                )

            def build_paper_dataset_bundle() -> dict[str, Any]:
                return paper_dataset_bundle_identity(
                    bundle,
                    image_size=image_size,
                    decoded_frame_identities={
                        "train_frames": train_target_provider.source.tensor_content_identity(),
                        "heldout_frames": heldout_target_provider.source.tensor_content_identity(),
                    },
                )

            paper_dataset_bundle = None
            paper_dataset_bundle_builder = build_paper_dataset_bundle
        else:
            paper_dataset_bundle = paper_dataset_bundle_identity(
                bundle,
                image_size=image_size,
            )
            paper_dataset_bundle_builder = None
        return {
            "targets": targets,
            "sample_frame_indices": sample_frame_indices,
            "sample_rays": sample_rays,
            "sample_ray_provider": train_ray_provider,
            "sample_target_provider": train_target_provider,
            "heldout_targets": heldout_targets,
            "heldout_frame_indices": heldout_frame_indices,
            "heldout_rays": heldout_rays,
            "heldout_ray_provider": heldout_ray_provider,
            "heldout_target_provider": heldout_target_provider,
            "target_residency": {
                "train": (
                    None if train_target_provider is None else _target_split_residency(train_target_provider, targets)
                ),
                "heldout": (
                    None
                    if heldout_target_provider is None
                    else _target_split_residency(heldout_target_provider, heldout_targets)
                ),
                "limitation": (
                    "targets are eagerly materialized on the training device"
                    if train_target_provider is None
                    else (
                        "video-backed splits remain fully decoded on CPU inside the resident provider"
                        if bool(train_target_provider.residency()["full_source_resident"])
                        else (
                            "MP4 and complete frame-path targets decode only selected batches; "
                            "video initialization may still materialize the condition view when explicitly requested"
                        )
                    )
                ),
            },
            "init_frames": init_frames,
            "init_frames_resident_bytes": (0 if init_frames is None else int(init_frames.untyped_storage().nbytes())),
            "init_frames_residency": {
                "enabled": init_frames is not None,
                "resident_bytes": (0 if init_frames is None else int(init_frames.untyped_storage().nbytes())),
                "shares_train_target_storage": (init_shares_train_storage),
            },
            "frame_count": bundle.frame_count,
            "train_view_count": bundle.train_view_count,
            "video_fps": float(bundle.condition_sequence.video_fps),
            "source_label": str(bundle.metadata.get("sample_id")) if bundle.metadata else "multicam_val",
            "train_views": bundle.train_camera_names,
            "heldout_views": bundle.heldout_camera_names or [],
            "pose_source": bundle.pose_source,
            "paper_dataset_bundle": paper_dataset_bundle,
            "paper_dataset_bundle_builder": paper_dataset_bundle_builder,
            "world_to_model": None
            if bundle.anchor_c2w is None
            else torch.linalg.inv(bundle.anchor_c2w.detach().to(device="cpu", dtype=torch.float32)),
            "point_cloud_visibility_train_K": bundle.train_K.detach().to(device="cpu", dtype=torch.float32),
            "point_cloud_visibility_train_w2c": bundle.train_w2c.detach().to(device="cpu", dtype=torch.float32),
            "point_cloud_visibility_train_lens_models": bundle.train_lens_models,
            "point_cloud_visibility_train_distortions": None
            if bundle.train_distortions is None
            else bundle.train_distortions.detach().to(device="cpu", dtype=torch.float32),
        }

    if cfg["data"]["video_path"] is None:
        raise ValueError("data.video_path is required unless data.frame_source is 'multicam_val'.")
    sequence = load_video_sequence(
        cfg["data"]["video_path"],
        target_size=render_size,
        max_frames=int(cfg["data"]["max_frames"]),
        frame_source=frame_source,
    )
    targets = sequence.frames.to(device=device, dtype=torch.float32)
    init_frames = targets.detach().cpu()
    return {
        "targets": targets,
        "sample_frame_indices": torch.arange(targets.size(0), device=device, dtype=torch.long),
        "sample_rays": None,
        "sample_ray_provider": None,
        "sample_target_provider": None,
        "heldout_targets": None,
        "heldout_frame_indices": None,
        "heldout_rays": None,
        "heldout_ray_provider": None,
        "heldout_target_provider": None,
        "target_residency": {
            "train": None,
            "heldout": None,
            "limitation": "targets are eagerly materialized on the training device",
        },
        "init_frames": init_frames,
        "init_frames_resident_bytes": int(init_frames.untyped_storage().nbytes()),
        "init_frames_residency": {
            "enabled": True,
            "resident_bytes": int(init_frames.untyped_storage().nbytes()),
            "shares_train_target_storage": targets.device.type == "cpu",
        },
        "frame_count": int(targets.size(0)),
        "train_view_count": 1,
        "video_fps": float(sequence.video_fps),
        "source_label": str(cfg["data"]["video_path"]),
        "train_views": [],
        "heldout_views": [],
        "pose_source": None,
        "world_to_model": None,
        "point_cloud_visibility_train_K": None,
        "point_cloud_visibility_train_w2c": None,
        "point_cloud_visibility_train_lens_models": None,
        "point_cloud_visibility_train_distortions": None,
    }
