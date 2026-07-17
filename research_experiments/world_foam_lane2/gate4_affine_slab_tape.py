from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from gate4_moving_ray_slab_compiler import (
    _frame_time,
    event_set_for_ray,
    slab_ranges,
)
from gate1_realray_per_sample_reference import make_boundaries_4d


GATE4_NEAR_CUT_ID = -1
GATE4_FAR_CUT_ID = -2
GATE4_ENABLE_EXPERIMENTAL_NATIVE_CUT_PREP_DELTA = False
GATE4_ENABLE_EXPERIMENTAL_NATIVE_SORTED_DELTA = False


@dataclass(frozen=True)
class Gate4AffineSlabTape:
    layout: str
    candidate_order: str
    view_count: int
    height: int
    width: int
    track_count: int
    row_count: int
    frame_count: int
    time_slab_count: int
    tile_shape: list[int] | None
    tile_grid_shape: list[int] | None
    frame_t: torch.Tensor
    ray_coeff: torch.Tensor
    explicit_rays: torch.Tensor
    row_index: torch.Tensor
    row_offsets: torch.Tensor
    candidate_ids: torch.Tensor
    candidate_depth_coeffs: torch.Tensor
    per_frame_event_sum: int
    missing_sample_events: int
    extra_candidate_events: int
    candidate_replay_iterations: int
    candidate_depth_order: dict[str, Any]
    direct_boundary_iterations: int
    compiled_boundary_tests: int
    candidate_count: int
    max_candidates_per_row: int
    avg_candidates_per_row: float
    empty_row_count: int
    max_origin_residual: float
    max_direction_residual: float

    @property
    def candidate_depth_num(self) -> torch.Tensor:
        return self.candidate_depth_coeffs[:, :2].contiguous()

    def candidate_depth_den(self, *, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        return self.candidate_depth_coeffs[:, 2:].contiguous().to(dtype=dtype)

    def to_legacy_bundle(self) -> dict[str, Any]:
        return {
            "layout": self.layout,
            "candidate_order": self.candidate_order,
            "view_count": self.view_count,
            "height": self.height,
            "width": self.width,
            "track_count": self.track_count,
            "row_count": self.row_count,
            "tile_shape": self.tile_shape,
            "tile_grid_shape": self.tile_grid_shape,
            "frame_t": self.frame_t,
            "ray_coeff": self.ray_coeff,
            "explicit_rays": self.explicit_rays,
            "row_index": self.row_index,
            "row_offsets": self.row_offsets,
            "candidate_ids": self.candidate_ids,
            "candidate_depth_coeffs": self.candidate_depth_coeffs,
            "per_frame_event_sum": self.per_frame_event_sum,
            "missing_sample_events": self.missing_sample_events,
            "extra_candidate_events": self.extra_candidate_events,
            "candidate_replay_iterations": self.candidate_replay_iterations,
            "candidate_depth_order": self.candidate_depth_order,
            "direct_boundary_iterations": self.direct_boundary_iterations,
            "compiled_boundary_tests": self.compiled_boundary_tests,
            "candidate_count": self.candidate_count,
            "max_candidates_per_row": self.max_candidates_per_row,
            "avg_candidates_per_row": self.avg_candidates_per_row,
            "empty_row_count": self.empty_row_count,
            "max_origin_residual": self.max_origin_residual,
            "max_direction_residual": self.max_direction_residual,
        }


@dataclass(frozen=True)
class Gate4EndpointRunRecord:
    owner: int
    left_cut_id: int
    right_cut_id: int


@dataclass(frozen=True)
class Gate4EndpointDeltaReplaceTape:
    base_offsets_i32: torch.Tensor
    base_owner_i32: torch.Tensor
    base_left_i32: torch.Tensor
    base_right_i32: torch.Tensor
    track_change_offsets_i32: torch.Tensor
    change_frame_i32: torch.Tensor
    change_offsets_i32: torch.Tensor
    change_owner_i32: torch.Tensor
    change_left_i32: torch.Tensor
    change_right_i32: torch.Tensor
    base_record_i32: torch.Tensor | None = None
    change_record_i32: torch.Tensor | None = None

    @property
    def storage_bytes(self) -> int:
        tensors = (
            self.base_offsets_i32,
            self.base_owner_i32,
            self.base_left_i32,
            self.base_right_i32,
            self.track_change_offsets_i32,
            self.change_frame_i32,
            self.change_offsets_i32,
            self.change_owner_i32,
            self.change_left_i32,
            self.change_right_i32,
        )
        if self.base_record_i32 is not None:
            tensors += (self.base_record_i32,)
        if self.change_record_i32 is not None:
            tensors += (self.change_record_i32,)
        return int(sum(t.numel() * t.element_size() for t in tensors))


@dataclass
class _DeltaReplaceTensorChunks:
    base_offsets: list[torch.Tensor]
    base_owner: list[torch.Tensor]
    base_left: list[torch.Tensor]
    base_right: list[torch.Tensor]
    track_change_offsets: list[torch.Tensor]
    change_frame: list[torch.Tensor]
    change_offsets: list[torch.Tensor]
    change_owner: list[torch.Tensor]
    change_left: list[torch.Tensor]
    change_right: list[torch.Tensor]
    base_record: list[torch.Tensor] | None
    change_record: list[torch.Tensor] | None
    base_record_count: int = 0
    change_event_count: int = 0
    change_record_count: int = 0
    native_result_count: int = 0


def _empty_i32_tensor() -> torch.Tensor:
    return torch.empty((0,), dtype=torch.int32)


def _i32_cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().to(dtype=torch.int32).contiguous()


def _new_delta_replace_tensor_chunks(*, include_packed_records: bool) -> _DeltaReplaceTensorChunks:
    zero = torch.zeros((1,), dtype=torch.int32)
    return _DeltaReplaceTensorChunks(
        base_offsets=[zero],
        base_owner=[],
        base_left=[],
        base_right=[],
        track_change_offsets=[zero],
        change_frame=[],
        change_offsets=[zero],
        change_owner=[],
        change_left=[],
        change_right=[],
        base_record=[] if include_packed_records else None,
        change_record=[] if include_packed_records else None,
    )


def _cat_i32_chunks(chunks: list[torch.Tensor]) -> torch.Tensor:
    if not chunks:
        return _empty_i32_tensor()
    return torch.cat(chunks).to(dtype=torch.int32).contiguous()


def _pack_cut_id_i32_value(cut_id: int) -> int:
    if int(cut_id) < -2:
        raise ValueError("packed endpoint records only support cut ids -1, -2, or nonnegative boundary ids")
    if int(cut_id) == GATE4_NEAR_CUT_ID:
        return 0
    if int(cut_id) == GATE4_FAR_CUT_ID:
        return 1
    code = int(cut_id) + 2
    if code > 4095:
        raise ValueError("packed endpoint records support cut codes up to 4095")
    return code


def _pack_endpoint_records_i32_values(
    *,
    owner: list[int],
    left: list[int],
    right: list[int],
) -> list[int]:
    if len(owner) != len(left) or len(owner) != len(right):
        raise ValueError("packed endpoint record lists must have matching lengths")
    packed: list[int] = []
    for owner_id, left_cut_id, right_cut_id in zip(owner, left, right, strict=True):
        if int(owner_id) < -1 or int(owner_id) > 255:
            raise ValueError("packed endpoint records support owner ids in [-1, 255]")
        owner_code = 0 if int(owner_id) < 0 else int(owner_id)
        value = (
            owner_code
            | (_pack_cut_id_i32_value(int(left_cut_id)) << 8)
            | (_pack_cut_id_i32_value(int(right_cut_id)) << 20)
        )
        if value > 2_147_483_647:
            raise ValueError("packed endpoint record exceeded signed int32 range")
        packed.append(value)
    return packed


def _materialize_packed_record_list(
    *,
    name: str,
    record: list[int] | None,
    owner: list[int],
    left: list[int],
    right: list[int],
) -> torch.Tensor | None:
    if record is None:
        return None
    if len(record) == len(owner):
        return torch.tensor(record, dtype=torch.int32)
    if len(record) == 0:
        return torch.tensor(
            _pack_endpoint_records_i32_values(owner=owner, left=left, right=right),
            dtype=torch.int32,
        )
    raise ValueError(f"{name} length {len(record)} did not match unpacked record length {len(owner)}")


def _cut_arrays_from_ordered_depth_ids(
    *,
    depths: np.ndarray,
    boundary_ids: np.ndarray,
    near: float,
    far: float,
    epsilon: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray]:
    if depths.size:
        keep = np.ones(int(depths.shape[0]), dtype=bool)
        if depths.shape[0] > 1:
            keep[1:] = np.abs(np.diff(depths)) > float(epsilon)
        kept_depths = depths[keep].astype(np.float64, copy=False)
        kept_ids = boundary_ids[keep].astype(np.int64, copy=False)
        cut_depths = np.empty(int(kept_depths.shape[0]) + 2, dtype=np.float64)
        cut_ids = np.empty(int(kept_ids.shape[0]) + 2, dtype=np.int64)
        cut_depths[0] = float(near)
        cut_depths[-1] = float(far)
        cut_depths[1:-1] = kept_depths
        cut_ids[0] = GATE4_NEAR_CUT_ID
        cut_ids[-1] = GATE4_FAR_CUT_ID
        cut_ids[1:-1] = kept_ids
        return cut_depths, cut_ids
    return (
        np.array([near, far], dtype=np.float64),
        np.array([GATE4_NEAR_CUT_ID, GATE4_FAR_CUT_ID], dtype=np.int64),
    )


def _first_nonempty_segment_index(cut_depths: np.ndarray, *, epsilon: float = 1.0e-8) -> int | None:
    segment_count = int(cut_depths.shape[0]) - 1
    segment_index = 0
    while (
        segment_index < segment_count
        and float(cut_depths[segment_index + 1] - cut_depths[segment_index]) <= float(epsilon)
    ):
        segment_index += 1
    return segment_index if segment_index < segment_count else None


def _owner_indices_for_points(
    *,
    points: np.ndarray,
    site_xyz: np.ndarray,
    site_t: np.ndarray,
    site_weight: np.ndarray,
) -> np.ndarray:
    if points.size == 0:
        return np.empty((0,), dtype=np.int64)
    delta_xyz = points[:, None, :3] - site_xyz[None, :, :]
    delta_t = points[:, 3:4] - site_t.reshape(1, -1)
    power = np.sum(delta_xyz * delta_xyz, axis=2) + delta_t * delta_t - site_weight.reshape(1, -1)
    return np.argmin(power, axis=1).astype(np.int64, copy=False)


def _boundary_arrays(boundaries: tuple[Any, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normal_xyz = np.array(
        [[float(boundary.nx), float(boundary.ny), float(boundary.nz)] for boundary in boundaries],
        dtype=np.float64,
    )
    nt = np.array([float(boundary.nt) for boundary in boundaries], dtype=np.float64)
    bias = np.array([float(boundary.b) for boundary in boundaries], dtype=np.float64)
    return normal_xyz, nt, bias


def _boundary_depth_coefficients_for_rows(
    *,
    ray_coeff: np.ndarray,
    boundaries: tuple[Any, ...],
) -> np.ndarray:
    """Return `[track_count, boundary_count, 4]` affine depth coefficients."""
    if ray_coeff.size == 0:
        return np.empty((0, len(boundaries), 4), dtype=np.float64)
    normal_xyz, nt, bias = _boundary_arrays(boundaries)
    origin_base = ray_coeff[:, 0:3]
    origin_slope = ray_coeff[:, 3:6]
    direction_base = ray_coeff[:, 6:9]
    direction_slope = ray_coeff[:, 9:12]
    numer_base = -(origin_base @ normal_xyz.T + bias.reshape(1, -1))
    numer_slope = -(origin_slope @ normal_xyz.T + nt.reshape(1, -1))
    denom_base = direction_base @ normal_xyz.T
    denom_slope = direction_slope @ normal_xyz.T
    return np.stack((numer_base, numer_slope, denom_base, denom_slope), axis=2)


def _compiled_slab_events_from_coeffs(
    *,
    coeffs: np.ndarray,
    t0: float,
    t1: float,
    near: float,
    far: float,
    invalid_epsilon: float,
    residual_depth_padding: float,
) -> np.ndarray:
    denom0 = coeffs[:, 2] + coeffs[:, 3] * float(t0)
    denom1 = coeffs[:, 2] + coeffs[:, 3] * float(t1)
    conservative = (np.abs(denom0) < float(invalid_epsilon)) | (np.abs(denom1) < float(invalid_epsilon))
    conservative |= (denom0 * denom1) < 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        depth0 = (coeffs[:, 0] + coeffs[:, 1] * float(t0)) / denom0
        depth1 = (coeffs[:, 0] + coeffs[:, 1] * float(t1)) / denom1
    lo = np.minimum(depth0, depth1) - float(residual_depth_padding)
    hi = np.maximum(depth0, depth1) + float(residual_depth_padding)
    in_range = np.maximum(lo, float(near)) <= np.minimum(hi, float(far))
    return np.flatnonzero(conservative | in_range).astype(np.int64, copy=False)


def _compiled_slab_event_mask_from_coeffs(
    *,
    coeffs: np.ndarray,
    t0: float,
    t1: float,
    near: float,
    far: float,
    invalid_epsilon: float,
    residual_depth_padding: float,
) -> np.ndarray:
    denom0 = coeffs[:, :, 2] + coeffs[:, :, 3] * float(t0)
    denom1 = coeffs[:, :, 2] + coeffs[:, :, 3] * float(t1)
    conservative = (np.abs(denom0) < float(invalid_epsilon)) | (np.abs(denom1) < float(invalid_epsilon))
    conservative |= (denom0 * denom1) < 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        depth0 = (coeffs[:, :, 0] + coeffs[:, :, 1] * float(t0)) / denom0
        depth1 = (coeffs[:, :, 0] + coeffs[:, :, 1] * float(t1)) / denom1
    lo = np.minimum(depth0, depth1) - float(residual_depth_padding)
    hi = np.maximum(depth0, depth1) + float(residual_depth_padding)
    in_range = np.maximum(lo, float(near)) <= np.minimum(hi, float(far))
    return conservative | in_range


def _candidate_csr_from_per_track_event_masks(
    *,
    masks_by_slab: list[np.ndarray],
    all_boundary_coeffs: np.ndarray,
    slabs: list[tuple[float, float]],
    candidate_order: str,
    invalid_epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not masks_by_slab:
        return (
            np.array([0], dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    row_count = int(masks_by_slab[0].shape[0])
    time_slabs = int(len(masks_by_slab))
    row_parts: list[np.ndarray] = []
    slab_parts: list[np.ndarray] = []
    boundary_parts: list[np.ndarray] = []
    for slab_id, mask in enumerate(masks_by_slab):
        row_ids, boundary_ids = np.nonzero(mask)
        if row_ids.size == 0:
            continue
        row_parts.append(row_ids.astype(np.int64, copy=False))
        slab_parts.append(np.full(row_ids.shape, int(slab_id), dtype=np.int64))
        boundary_parts.append(boundary_ids.astype(np.int64, copy=False))
    if not row_parts:
        row_slab_count = row_count * time_slabs
        return (
            np.zeros(row_slab_count + 1, dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0, 4), dtype=np.float32),
            np.zeros(row_slab_count, dtype=np.int64),
        )

    row_ids = np.concatenate(row_parts)
    slab_ids = np.concatenate(slab_parts)
    boundary_ids = np.concatenate(boundary_parts)
    row_slab_ids = row_ids * time_slabs + slab_ids
    if candidate_order == "boundary-id":
        order = np.lexsort((boundary_ids, row_slab_ids))
    elif candidate_order == "slab-mid-depth":
        selected_coeffs = all_boundary_coeffs[row_ids, boundary_ids]
        slab_mid = np.array([0.5 * (t0 + t1) for t0, t1 in slabs], dtype=np.float64)
        t_mid = slab_mid[slab_ids]
        denom = selected_coeffs[:, 2] + selected_coeffs[:, 3] * t_mid
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = (selected_coeffs[:, 0] + selected_coeffs[:, 1] * t_mid) / denom
        invalid = np.abs(denom) < float(invalid_epsilon)
        order = np.lexsort((boundary_ids, depth, invalid.astype(np.int64), row_slab_ids))
    else:
        raise ValueError("candidate_order must be 'boundary-id' or 'slab-mid-depth'")

    row_slab_ids = row_slab_ids[order]
    row_ids = row_ids[order]
    boundary_ids = boundary_ids[order]
    row_slab_count = row_count * time_slabs
    counts = np.bincount(row_slab_ids, minlength=row_slab_count).astype(np.int64, copy=False)
    offsets = np.empty(row_slab_count + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    coeffs = all_boundary_coeffs[row_ids, boundary_ids].astype(np.float32, copy=False)
    return offsets, boundary_ids.astype(np.int64, copy=False), coeffs, counts


def _iter_single_slab_sorted_depth_id_chunks(
    *,
    row_offsets: np.ndarray,
    candidate_ids: np.ndarray,
    coeffs: np.ndarray,
    frame_t: np.ndarray,
    row_index: np.ndarray,
    near: float,
    far: float,
    invalid_epsilon: float,
    chunk_size: int = 128,
) -> Any:
    frame_count = int(frame_t.shape[0])
    max_row = int(row_offsets.shape[0]) - 1
    column_template: np.ndarray | None = None
    t_values = frame_t.reshape(1, 1, frame_count)
    for track_begin in range(0, int(row_index.shape[0]), int(chunk_size)):
        track_end = min(track_begin + int(chunk_size), int(row_index.shape[0]))
        rows = row_index[track_begin:track_end]
        valid_rows = (rows >= 0) & (rows < max_row)
        counts = np.zeros(int(rows.shape[0]), dtype=np.int64)
        if np.any(valid_rows):
            valid_row_ids = rows[valid_rows]
            counts[valid_rows] = row_offsets[valid_row_ids + 1] - row_offsets[valid_row_ids]
        max_count = int(counts.max(initial=0))
        if max_count <= 0:
            yield (
                track_begin,
                np.empty((int(rows.shape[0]), 0, frame_count), dtype=np.float64),
                np.empty((int(rows.shape[0]), 0, frame_count), dtype=np.int64),
                np.zeros((int(rows.shape[0]), frame_count), dtype=np.int64),
            )
            continue

        chunk_coeffs = np.zeros((int(rows.shape[0]), max_count, 4), dtype=np.float64)
        chunk_ids = np.zeros((int(rows.shape[0]), max_count), dtype=np.int64)
        for local_index, row in enumerate(rows):
            if not valid_rows[local_index]:
                continue
            count = int(counts[local_index])
            if count <= 0:
                continue
            begin = int(row_offsets[int(row)])
            end = begin + count
            chunk_coeffs[local_index, :count] = coeffs[begin:end]
            chunk_ids[local_index, :count] = candidate_ids[begin:end]

        if column_template is None or int(column_template.shape[1]) < max_count:
            column_template = np.arange(max_count, dtype=np.int64).reshape(1, max_count, 1)
        valid_slots = column_template[:, :max_count, :] < counts.reshape(-1, 1, 1)
        denom = chunk_coeffs[:, :, 2:3] + chunk_coeffs[:, :, 3:4] * t_values
        numer = chunk_coeffs[:, :, 0:1] + chunk_coeffs[:, :, 1:2] * t_values
        with np.errstate(divide="ignore", invalid="ignore"):
            depths = numer / denom
        valid = (
            valid_slots
            & (np.abs(denom) >= float(invalid_epsilon))
            & np.isfinite(depths)
            & (depths >= float(near))
            & (depths <= float(far))
        )
        valid_counts = np.sum(valid, axis=1).astype(np.int64, copy=False)
        order = np.argsort(np.where(valid, depths, np.inf), axis=1, kind="mergesort")
        sorted_depths = np.take_along_axis(depths, order, axis=1)
        sorted_ids = np.take_along_axis(np.broadcast_to(chunk_ids[:, :, None], depths.shape), order, axis=1)
        yield track_begin, sorted_depths, sorted_ids, valid_counts


def _owner_run_records_from_cut_arrays(
    *,
    cut_depths: np.ndarray,
    cut_ids: np.ndarray,
    boundary_left: np.ndarray,
    boundary_right: np.ndarray,
    boundary_other_by_owner: np.ndarray | None = None,
    start_segment: int,
    initial_owner: int,
) -> tuple[tuple[int, int, int], ...]:
    segment_count = int(cut_ids.shape[0]) - 1
    if start_segment >= segment_count:
        return tuple()

    current_owner = int(initial_owner)
    records: list[tuple[int, int, int]] = []
    cursor = start_segment
    while cursor < segment_count:
        following_boundary_ids = cut_ids[cursor + 1 : -1]
        next_cut_index = segment_count
        if following_boundary_ids.size:
            if boundary_other_by_owner is None:
                left_ids = boundary_left[following_boundary_ids]
                right_ids = boundary_right[following_boundary_ids]
                owner_hit = np.flatnonzero((left_ids == int(current_owner)) | (right_ids == int(current_owner)))
            else:
                owner_hit = np.flatnonzero(boundary_other_by_owner[int(current_owner), following_boundary_ids] >= 0)
            if owner_hit.size:
                next_cut_index = int(cursor + 1 + owner_hit[0])
        if float(cut_depths[next_cut_index] - cut_depths[cursor]) > 1.0e-8:
            records.append((int(current_owner), int(cut_ids[cursor]), int(cut_ids[next_cut_index])))
        if next_cut_index >= segment_count:
            break
        boundary_id = int(cut_ids[next_cut_index])
        left_site = int(boundary_left[boundary_id])
        right_site = int(boundary_right[boundary_id])
        if int(current_owner) == left_site:
            current_owner = right_site
        elif int(current_owner) == right_site:
            current_owner = left_site
        cursor = next_cut_index
    return tuple(records)


def _owner_run_record_objects_from_cut_arrays(
    *,
    cut_depths: np.ndarray,
    cut_ids: np.ndarray,
    boundary_left: np.ndarray,
    boundary_right: np.ndarray,
    boundary_other_by_owner: np.ndarray | None = None,
    start_segment: int,
    initial_owner: int,
) -> tuple[Gate4EndpointRunRecord, ...]:
    return tuple(
        Gate4EndpointRunRecord(owner=owner, left_cut_id=left_cut_id, right_cut_id=right_cut_id)
        for owner, left_cut_id, right_cut_id in _owner_run_records_from_cut_arrays(
            cut_depths=cut_depths,
            cut_ids=cut_ids,
            boundary_left=boundary_left,
            boundary_right=boundary_right,
            boundary_other_by_owner=boundary_other_by_owner,
            start_segment=start_segment,
            initial_owner=initial_owner,
        )
    )


def _records_from_frame_work(
    *,
    frame_work: list[tuple[np.ndarray, np.ndarray, int | None]],
    initial_owners: list[int | None],
    boundary_left: np.ndarray,
    boundary_right: np.ndarray,
    boundary_other_by_owner: np.ndarray | None = None,
) -> list[tuple[Gate4EndpointRunRecord, ...]]:
    frames: list[tuple[Gate4EndpointRunRecord, ...]] = []
    for (cut_depths, cut_ids, start_segment), initial_owner in zip(frame_work, initial_owners, strict=True):
        if start_segment is None or initial_owner is None:
            frames.append(tuple())
            continue
        frames.append(
            _owner_run_record_objects_from_cut_arrays(
                cut_depths=cut_depths,
                cut_ids=cut_ids,
                boundary_left=boundary_left,
                boundary_right=boundary_right,
                boundary_other_by_owner=boundary_other_by_owner,
                start_segment=int(start_segment),
                initial_owner=int(initial_owner),
            )
        )
    return frames


def _append_delta_track_rows(
    *,
    frame_work: list[tuple[np.ndarray, np.ndarray, int | None]],
    initial_owners: list[int | None],
    boundary_left: np.ndarray,
    boundary_right: np.ndarray,
    boundary_other_by_owner: np.ndarray | None,
    base_offsets: list[int],
    base_owner: list[int],
    base_left: list[int],
    base_right: list[int],
    track_change_offsets: list[int],
    change_frame: list[int],
    change_offsets: list[int],
    change_owner: list[int],
    change_left: list[int],
    change_right: list[int],
) -> None:
    previous: tuple[tuple[int, int, int], ...] | None = None
    for frame_id, ((cut_depths, cut_ids, start_segment), initial_owner) in enumerate(
        zip(frame_work, initial_owners, strict=True)
    ):
        if start_segment is None or initial_owner is None:
            current: tuple[tuple[int, int, int], ...] = tuple()
        else:
            current = _owner_run_records_from_cut_arrays(
                cut_depths=cut_depths,
                cut_ids=cut_ids,
                boundary_left=boundary_left,
                boundary_right=boundary_right,
                boundary_other_by_owner=boundary_other_by_owner,
                start_segment=int(start_segment),
                initial_owner=int(initial_owner),
            )
        if frame_id == 0:
            for owner, left_cut_id, right_cut_id in current:
                base_owner.append(owner)
                base_left.append(left_cut_id)
                base_right.append(right_cut_id)
            base_offsets.append(len(base_owner))
            previous = current
            continue
        if current == previous:
            continue
        change_frame.append(frame_id)
        for owner, left_cut_id, right_cut_id in current:
            change_owner.append(owner)
            change_left.append(left_cut_id)
            change_right.append(right_cut_id)
        change_offsets.append(len(change_owner))
        previous = current
    if previous is None:
        base_offsets.append(len(base_owner))
    track_change_offsets.append(len(change_frame))


def _boundary_other_by_owner(
    *,
    boundary_left: np.ndarray,
    boundary_right: np.ndarray,
    site_count: int,
) -> np.ndarray:
    boundary_count = int(boundary_left.shape[0])
    other = np.full((int(site_count), boundary_count), -1, dtype=np.int64)
    boundary_ids = np.arange(boundary_count, dtype=np.int64)
    other[boundary_left, boundary_ids] = boundary_right
    other[boundary_right, boundary_ids] = boundary_left
    return other


def _gate4_delta_replace_from_cuts_cpu_op() -> Any | None:
    try:
        return torch.ops.world_foam_lane2_fused_slab_v0.gate4_delta_replace_from_cuts_cpu
    except (AttributeError, RuntimeError):
        return None


def _gate4_delta_replace_packed_from_cuts_cpu_op() -> Any | None:
    try:
        return torch.ops.world_foam_lane2_fused_slab_v0.gate4_delta_replace_packed_from_cuts_cpu
    except (AttributeError, RuntimeError):
        return None


def _gate4_cut_arrays_from_sorted_cpu_op() -> Any | None:
    try:
        return torch.ops.world_foam_lane2_fused_slab_v0.gate4_cut_arrays_from_sorted_cpu
    except (AttributeError, RuntimeError):
        return None


def _gate4_delta_replace_from_sorted_cpu_op() -> Any | None:
    try:
        return torch.ops.world_foam_lane2_fused_slab_v0.gate4_delta_replace_from_sorted_cpu
    except (AttributeError, RuntimeError):
        return None


def _gate4_delta_replace_packed_from_sorted_cpu_op() -> Any | None:
    try:
        return torch.ops.world_foam_lane2_fused_slab_v0.gate4_delta_replace_packed_from_sorted_cpu
    except (AttributeError, RuntimeError):
        return None


def _gate4_delta_replace_packed_from_coeff_csr_cpu_op() -> Any | None:
    try:
        return torch.ops.world_foam_lane2_fused_slab_v0.gate4_delta_replace_packed_from_coeff_csr_cpu
    except (AttributeError, RuntimeError):
        return None


def _unpack_native_delta_result(result: Any) -> tuple[Any, ...]:
    if len(result) == 12:
        return tuple(result)
    if len(result) == 10:
        return (
            result[0],
            result[1],
            result[2],
            result[3],
            None,
            result[4],
            result[5],
            result[6],
            result[7],
            result[8],
            result[9],
            None,
        )
    raise ValueError(f"native Gate4 delta result returned {len(result)} tensors, expected 10 or 12")


def _append_native_delta_tensor_result(
    *,
    result: Any,
    chunks: _DeltaReplaceTensorChunks,
) -> None:
    (
        chunk_base_offsets,
        chunk_base_owner,
        chunk_base_left,
        chunk_base_right,
        chunk_base_record,
        chunk_track_change_offsets,
        chunk_change_frame,
        chunk_change_offsets,
        chunk_change_owner,
        chunk_change_left,
        chunk_change_right,
        chunk_change_record,
    ) = _unpack_native_delta_result(result)

    base_record_offset = int(chunks.base_record_count)
    change_event_offset = int(chunks.change_event_count)
    change_record_offset = int(chunks.change_record_count)

    chunk_base_offsets = _i32_cpu_tensor(chunk_base_offsets)
    chunk_base_owner = _i32_cpu_tensor(chunk_base_owner)
    chunk_base_left = _i32_cpu_tensor(chunk_base_left)
    chunk_base_right = _i32_cpu_tensor(chunk_base_right)
    chunk_track_change_offsets = _i32_cpu_tensor(chunk_track_change_offsets)
    chunk_change_frame = _i32_cpu_tensor(chunk_change_frame)
    chunk_change_offsets = _i32_cpu_tensor(chunk_change_offsets)
    chunk_change_owner = _i32_cpu_tensor(chunk_change_owner)
    chunk_change_left = _i32_cpu_tensor(chunk_change_left)
    chunk_change_right = _i32_cpu_tensor(chunk_change_right)

    chunks.base_offsets.append(chunk_base_offsets[1:] + base_record_offset)
    chunks.base_owner.append(chunk_base_owner)
    chunks.base_left.append(chunk_base_left)
    chunks.base_right.append(chunk_base_right)
    if chunks.base_record is not None:
        if chunk_base_record is None:
            raise ValueError("native packed delta result did not include base_record_i32")
        chunks.base_record.append(_i32_cpu_tensor(chunk_base_record))
    chunks.track_change_offsets.append(chunk_track_change_offsets[1:] + change_event_offset)
    chunks.change_frame.append(chunk_change_frame)
    chunks.change_offsets.append(chunk_change_offsets[1:] + change_record_offset)
    chunks.change_owner.append(chunk_change_owner)
    chunks.change_left.append(chunk_change_left)
    chunks.change_right.append(chunk_change_right)
    if chunks.change_record is not None:
        if chunk_change_record is None:
            raise ValueError("native packed delta result did not include change_record_i32")
        chunks.change_record.append(_i32_cpu_tensor(chunk_change_record))

    chunks.base_record_count += int(chunk_base_owner.numel())
    chunks.change_event_count += int(chunk_change_frame.numel())
    chunks.change_record_count += int(chunk_change_owner.numel())
    chunks.native_result_count += 1


def _append_native_delta_result(
    *,
    result: Any,
    base_offsets: list[int],
    base_owner: list[int],
    base_left: list[int],
    base_right: list[int],
    track_change_offsets: list[int],
    change_frame: list[int],
    change_offsets: list[int],
    change_owner: list[int],
    change_left: list[int],
    change_right: list[int],
    base_record: list[int] | None = None,
    change_record: list[int] | None = None,
) -> None:
    (
        chunk_base_offsets,
        chunk_base_owner,
        chunk_base_left,
        chunk_base_right,
        chunk_base_record,
        chunk_track_change_offsets,
        chunk_change_frame,
        chunk_change_offsets,
        chunk_change_owner,
        chunk_change_left,
        chunk_change_right,
        chunk_change_record,
    ) = _unpack_native_delta_result(result)

    base_record_offset = len(base_owner)
    change_event_offset = len(change_frame)
    change_record_offset = len(change_owner)
    base_offsets.extend(base_record_offset + int(value) for value in chunk_base_offsets.tolist()[1:])
    base_owner.extend(int(value) for value in chunk_base_owner.tolist())
    base_left.extend(int(value) for value in chunk_base_left.tolist())
    base_right.extend(int(value) for value in chunk_base_right.tolist())
    if base_record is not None:
        if chunk_base_record is None:
            raise ValueError("native packed delta result did not include base_record_i32")
        base_record.extend(int(value) for value in chunk_base_record.tolist())
    track_change_offsets.extend(change_event_offset + int(value) for value in chunk_track_change_offsets.tolist()[1:])
    change_frame.extend(int(value) for value in chunk_change_frame.tolist())
    change_offsets.extend(change_record_offset + int(value) for value in chunk_change_offsets.tolist()[1:])
    change_owner.extend(int(value) for value in chunk_change_owner.tolist())
    change_left.extend(int(value) for value in chunk_change_left.tolist())
    change_right.extend(int(value) for value in chunk_change_right.tolist())
    if change_record is not None:
        if chunk_change_record is None:
            raise ValueError("native packed delta result did not include change_record_i32")
        change_record.extend(int(value) for value in chunk_change_record.tolist())


def _append_native_delta_chunk(
    *,
    cut_depth_parts: list[np.ndarray],
    cut_id_parts: list[np.ndarray],
    cut_offsets: list[int],
    start_segments: list[int],
    initial_owners: list[int],
    boundary_other_by_owner: np.ndarray,
    frame_count: int,
    base_offsets: list[int],
    base_owner: list[int],
    base_left: list[int],
    base_right: list[int],
    track_change_offsets: list[int],
    change_frame: list[int],
    change_offsets: list[int],
    change_owner: list[int],
    change_left: list[int],
    change_right: list[int],
    base_record: list[int] | None = None,
    change_record: list[int] | None = None,
    tensor_chunks: _DeltaReplaceTensorChunks | None = None,
) -> bool:
    op = _gate4_delta_replace_packed_from_cuts_cpu_op() if base_record is not None else _gate4_delta_replace_from_cuts_cpu_op()
    if op is None:
        return False
    cut_depths = (
        np.concatenate(cut_depth_parts).astype(np.float64, copy=False)
        if cut_depth_parts
        else np.empty((0,), dtype=np.float64)
    )
    cut_ids = (
        np.concatenate(cut_id_parts).astype(np.int64, copy=False)
        if cut_id_parts
        else np.empty((0,), dtype=np.int64)
    )
    result = op(
        torch.from_numpy(np.ascontiguousarray(cut_depths)),
        torch.from_numpy(np.ascontiguousarray(cut_ids)),
        torch.tensor(cut_offsets, dtype=torch.int64),
        torch.tensor(start_segments, dtype=torch.int64),
        torch.tensor(initial_owners, dtype=torch.int64),
        torch.from_numpy(np.ascontiguousarray(boundary_other_by_owner)),
        int(frame_count),
        1.0e-8,
    )
    if tensor_chunks is not None:
        _append_native_delta_tensor_result(result=result, chunks=tensor_chunks)
    else:
        _append_native_delta_result(
            result=result,
            base_offsets=base_offsets,
            base_owner=base_owner,
            base_left=base_left,
            base_right=base_right,
            track_change_offsets=track_change_offsets,
            change_frame=change_frame,
            change_offsets=change_offsets,
            change_owner=change_owner,
            change_left=change_left,
            change_right=change_right,
            base_record=base_record,
            change_record=change_record,
        )
    return True


def _append_native_cutprep_delta_chunk(
    *,
    chunk_depths: np.ndarray,
    chunk_ids: np.ndarray,
    chunk_valid_counts: np.ndarray,
    row_active: np.ndarray,
    ray_coeff: np.ndarray,
    frame_t: np.ndarray,
    site_xyz: np.ndarray,
    site_t: np.ndarray,
    site_weight: np.ndarray,
    boundary_other_by_owner: np.ndarray,
    frame_count: int,
    near: float,
    far: float,
    base_offsets: list[int],
    base_owner: list[int],
    base_left: list[int],
    base_right: list[int],
    track_change_offsets: list[int],
    change_frame: list[int],
    change_offsets: list[int],
    change_owner: list[int],
    change_left: list[int],
    change_right: list[int],
    base_record: list[int] | None = None,
    change_record: list[int] | None = None,
    tensor_chunks: _DeltaReplaceTensorChunks | None = None,
) -> bool:
    cutprep_op = _gate4_cut_arrays_from_sorted_cpu_op()
    delta_op = (
        _gate4_delta_replace_packed_from_cuts_cpu_op()
        if base_record is not None
        else _gate4_delta_replace_from_cuts_cpu_op()
    )
    if cutprep_op is None or delta_op is None:
        return False
    cut_depths, cut_ids, cut_offsets, start_segments, initial_owners = cutprep_op(
        torch.from_numpy(np.ascontiguousarray(chunk_depths, dtype=np.float64)),
        torch.from_numpy(np.ascontiguousarray(chunk_ids, dtype=np.int64)),
        torch.from_numpy(np.ascontiguousarray(chunk_valid_counts, dtype=np.int64)),
        torch.from_numpy(np.ascontiguousarray(row_active, dtype=np.int64)),
        torch.from_numpy(np.ascontiguousarray(ray_coeff, dtype=np.float64)),
        torch.from_numpy(np.ascontiguousarray(frame_t, dtype=np.float64)),
        torch.from_numpy(np.ascontiguousarray(site_xyz, dtype=np.float64)),
        torch.from_numpy(np.ascontiguousarray(site_t, dtype=np.float64)),
        torch.from_numpy(np.ascontiguousarray(site_weight, dtype=np.float64)),
        int(frame_count),
        float(near),
        float(far),
        1.0e-6,
        1.0e-8,
    )
    result = delta_op(
        cut_depths,
        cut_ids,
        cut_offsets,
        start_segments,
        initial_owners,
        torch.from_numpy(np.ascontiguousarray(boundary_other_by_owner, dtype=np.int64)),
        int(frame_count),
        1.0e-8,
    )
    if tensor_chunks is not None:
        _append_native_delta_tensor_result(result=result, chunks=tensor_chunks)
    else:
        _append_native_delta_result(
            result=result,
            base_offsets=base_offsets,
            base_owner=base_owner,
            base_left=base_left,
            base_right=base_right,
            track_change_offsets=track_change_offsets,
            change_frame=change_frame,
            change_offsets=change_offsets,
            change_owner=change_owner,
            change_left=change_left,
            change_right=change_right,
            base_record=base_record,
            change_record=change_record,
        )
    return True


def _append_native_sorted_delta_chunk(
    *,
    chunk_depths: np.ndarray,
    chunk_ids: np.ndarray,
    chunk_valid_counts: np.ndarray,
    row_active: np.ndarray,
    ray_coeff: np.ndarray,
    frame_t: np.ndarray,
    site_xyz: np.ndarray,
    site_t: np.ndarray,
    site_weight: np.ndarray,
    boundary_other_by_owner: np.ndarray,
    frame_count: int,
    near: float,
    far: float,
    base_offsets: list[int],
    base_owner: list[int],
    base_left: list[int],
    base_right: list[int],
    track_change_offsets: list[int],
    change_frame: list[int],
    change_offsets: list[int],
    change_owner: list[int],
    change_left: list[int],
    change_right: list[int],
    base_record: list[int] | None = None,
    change_record: list[int] | None = None,
    tensor_chunks: _DeltaReplaceTensorChunks | None = None,
) -> bool:
    op = _gate4_delta_replace_packed_from_sorted_cpu_op() if base_record is not None else _gate4_delta_replace_from_sorted_cpu_op()
    if op is None:
        return False
    result = op(
        torch.from_numpy(np.ascontiguousarray(chunk_depths, dtype=np.float64)),
        torch.from_numpy(np.ascontiguousarray(chunk_ids, dtype=np.int64)),
        torch.from_numpy(np.ascontiguousarray(chunk_valid_counts, dtype=np.int64)),
        torch.from_numpy(np.ascontiguousarray(row_active, dtype=np.int64)),
        torch.from_numpy(np.ascontiguousarray(ray_coeff, dtype=np.float64)),
        torch.from_numpy(np.ascontiguousarray(frame_t, dtype=np.float64)),
        torch.from_numpy(np.ascontiguousarray(site_xyz, dtype=np.float64)),
        torch.from_numpy(np.ascontiguousarray(site_t, dtype=np.float64)),
        torch.from_numpy(np.ascontiguousarray(site_weight, dtype=np.float64)),
        torch.from_numpy(np.ascontiguousarray(boundary_other_by_owner, dtype=np.int64)),
        int(frame_count),
        float(near),
        float(far),
        1.0e-6,
        1.0e-8,
    )
    if tensor_chunks is not None:
        _append_native_delta_tensor_result(result=result, chunks=tensor_chunks)
    else:
        _append_native_delta_result(
            result=result,
            base_offsets=base_offsets,
            base_owner=base_owner,
            base_left=base_left,
            base_right=base_right,
            track_change_offsets=track_change_offsets,
            change_frame=change_frame,
            change_offsets=change_offsets,
            change_owner=change_owner,
            change_left=change_left,
            change_right=change_right,
            base_record=base_record,
            change_record=change_record,
        )
    return True


def _delta_replace_tape_from_lists(
    *,
    base_offsets: list[int],
    base_owner: list[int],
    base_left: list[int],
    base_right: list[int],
    track_change_offsets: list[int],
    change_frame: list[int],
    change_offsets: list[int],
    change_owner: list[int],
    change_left: list[int],
    change_right: list[int],
    base_record: list[int] | None = None,
    change_record: list[int] | None = None,
) -> Gate4EndpointDeltaReplaceTape:
    return Gate4EndpointDeltaReplaceTape(
        base_offsets_i32=torch.tensor(base_offsets, dtype=torch.int32),
        base_owner_i32=torch.tensor(base_owner, dtype=torch.int32),
        base_left_i32=torch.tensor(base_left, dtype=torch.int32),
        base_right_i32=torch.tensor(base_right, dtype=torch.int32),
        track_change_offsets_i32=torch.tensor(track_change_offsets, dtype=torch.int32),
        change_frame_i32=torch.tensor(change_frame, dtype=torch.int32),
        change_offsets_i32=torch.tensor(change_offsets, dtype=torch.int32),
        change_owner_i32=torch.tensor(change_owner, dtype=torch.int32),
        change_left_i32=torch.tensor(change_left, dtype=torch.int32),
        change_right_i32=torch.tensor(change_right, dtype=torch.int32),
        base_record_i32=_materialize_packed_record_list(
            name="base_record_i32",
            record=base_record,
            owner=base_owner,
            left=base_left,
            right=base_right,
        ),
        change_record_i32=_materialize_packed_record_list(
            name="change_record_i32",
            record=change_record,
            owner=change_owner,
            left=change_left,
            right=change_right,
        ),
    )


def _delta_replace_tape_from_tensor_chunks(chunks: _DeltaReplaceTensorChunks) -> Gate4EndpointDeltaReplaceTape:
    return Gate4EndpointDeltaReplaceTape(
        base_offsets_i32=_cat_i32_chunks(chunks.base_offsets),
        base_owner_i32=_cat_i32_chunks(chunks.base_owner),
        base_left_i32=_cat_i32_chunks(chunks.base_left),
        base_right_i32=_cat_i32_chunks(chunks.base_right),
        track_change_offsets_i32=_cat_i32_chunks(chunks.track_change_offsets),
        change_frame_i32=_cat_i32_chunks(chunks.change_frame),
        change_offsets_i32=_cat_i32_chunks(chunks.change_offsets),
        change_owner_i32=_cat_i32_chunks(chunks.change_owner),
        change_left_i32=_cat_i32_chunks(chunks.change_left),
        change_right_i32=_cat_i32_chunks(chunks.change_right),
        base_record_i32=_cat_i32_chunks(chunks.base_record) if chunks.base_record is not None else None,
        change_record_i32=_cat_i32_chunks(chunks.change_record) if chunks.change_record is not None else None,
    )


def build_gate4_endpoint_run_sequences(
    *,
    tape: Gate4AffineSlabTape,
    sites: tuple[Any, ...],
    near: float,
    far: float,
    invalid_epsilon: float,
) -> list[list[tuple[Gate4EndpointRunRecord, ...]]]:
    """Build endpoint owner/cut-id rows from a compiled Gate4 affine slab tape.

    This reuses the slab tape's conservative per-track candidate rows, so it
    avoids the slow per-frame all-boundary scan used by the older endpoint
    record packer. The resulting rows are frozen-geometry RGB-MSE records:
    ownership and cut ids are fixed, while RGBA/density stays live in Metal.
    """
    if tape.layout != "per-track":
        raise ValueError("Gate4 endpoint rows require per-track affine slab tape")
    if tape.row_count != tape.track_count:
        raise ValueError("Gate4 endpoint rows require row_count == track_count")
    if tape.missing_sample_events != 0:
        raise ValueError("Gate4 endpoint rows require zero missing sample events in the affine slab tape")
    if near >= far:
        raise ValueError("near must be less than far")
    if invalid_epsilon <= 0.0:
        raise ValueError("invalid_epsilon must be positive")

    row_offsets = tape.row_offsets.detach().cpu().to(dtype=torch.long).numpy()
    candidate_ids = tape.candidate_ids.detach().cpu().to(dtype=torch.long).numpy()
    coeffs = tape.candidate_depth_coeffs.detach().cpu().to(dtype=torch.float64).numpy()
    frame_t = tape.frame_t.detach().cpu().to(dtype=torch.float64).numpy()
    row_index = tape.row_index.detach().cpu().to(dtype=torch.long).numpy()
    ray_coeff = tape.ray_coeff.detach().cpu().to(dtype=torch.float64).numpy()
    boundary_pairs = tuple((int(boundary.left), int(boundary.right)) for boundary in make_boundaries_4d(sites))
    boundary_left = np.array([left for left, _right in boundary_pairs], dtype=np.int64)
    boundary_right = np.array([right for _left, right in boundary_pairs], dtype=np.int64)
    boundary_other = _boundary_other_by_owner(
        boundary_left=boundary_left,
        boundary_right=boundary_right,
        site_count=len(sites),
    )
    site_xyz = np.array([(float(site.x), float(site.y), float(site.z)) for site in sites], dtype=np.float64)
    site_t = np.array([float(site.t) for site in sites], dtype=np.float64)
    site_weight = np.array([float(site.weight) for site in sites], dtype=np.float64)

    sequences: list[list[tuple[Gate4EndpointRunRecord, ...]]] = [
        [tuple() for _frame_id in range(tape.frame_count)] for _track_id in range(tape.track_count)
    ]
    if tape.time_slab_count == 1:
        for track_begin, chunk_depths, chunk_ids, chunk_valid_counts in _iter_single_slab_sorted_depth_id_chunks(
            row_offsets=row_offsets,
            candidate_ids=candidate_ids,
            coeffs=coeffs,
            frame_t=frame_t,
            row_index=row_index,
            near=float(near),
            far=float(far),
            invalid_epsilon=float(invalid_epsilon),
        ):
            track_end = min(track_begin + int(chunk_valid_counts.shape[0]), tape.track_count)
            owner_points: list[tuple[float, float, float, float]] = []
            owner_refs: list[tuple[int, int]] = []
            chunk_frame_work: list[list[tuple[np.ndarray, np.ndarray, int | None]]] = []
            chunk_initial_owners: list[list[int | None]] = []
            for local_index, track_id in enumerate(range(track_begin, track_end)):
                row = int(row_index[track_id])
                if row < 0 or row >= tape.row_count:
                    chunk_frame_work.append([])
                    chunk_initial_owners.append([])
                    continue
                track_ray_coeff = ray_coeff[track_id]
                frame_work: list[tuple[np.ndarray, np.ndarray, int | None]] = []
                track_initial_owners: list[int | None] = [None for _ in range(tape.frame_count)]
                for frame_id in range(tape.frame_count):
                    t = float(frame_t[frame_id])
                    valid_count = int(chunk_valid_counts[local_index, frame_id])
                    sorted_depths = (
                        chunk_depths[local_index, :valid_count, frame_id]
                        if valid_count
                        else np.empty((0,), dtype=np.float64)
                    )
                    sorted_ids = (
                        chunk_ids[local_index, :valid_count, frame_id]
                        if valid_count
                        else np.empty((0,), dtype=np.int64)
                    )
                    cut_depths, cut_ids = _cut_arrays_from_ordered_depth_ids(
                        depths=sorted_depths,
                        boundary_ids=sorted_ids,
                        near=float(near),
                        far=float(far),
                    )
                    start_segment = _first_nonempty_segment_index(cut_depths)
                    if start_segment is not None:
                        ox = float(track_ray_coeff[0] + track_ray_coeff[3] * t)
                        oy = float(track_ray_coeff[1] + track_ray_coeff[4] * t)
                        oz = float(track_ray_coeff[2] + track_ray_coeff[5] * t)
                        dx = float(track_ray_coeff[6] + track_ray_coeff[9] * t)
                        dy = float(track_ray_coeff[7] + track_ray_coeff[10] * t)
                        dz = float(track_ray_coeff[8] + track_ray_coeff[11] * t)
                        first_mid = 0.5 * float(cut_depths[start_segment] + cut_depths[start_segment + 1])
                        owner_refs.append((local_index, len(frame_work)))
                        owner_points.append((ox + dx * first_mid, oy + dy * first_mid, oz + dz * first_mid, t))
                    frame_work.append((cut_depths, cut_ids, start_segment))
                chunk_frame_work.append(frame_work)
                chunk_initial_owners.append(track_initial_owners)
            if owner_points:
                owner_values = _owner_indices_for_points(
                    points=np.array(owner_points, dtype=np.float64),
                    site_xyz=site_xyz,
                    site_t=site_t,
                    site_weight=site_weight,
                )
                for (local_index, frame_work_index), owner in zip(owner_refs, owner_values, strict=True):
                    chunk_initial_owners[local_index][frame_work_index] = int(owner)
            for local_index, track_id in enumerate(range(track_begin, track_end)):
                if not chunk_frame_work[local_index]:
                    continue
                sequences[track_id] = _records_from_frame_work(
                    frame_work=chunk_frame_work[local_index],
                    initial_owners=chunk_initial_owners[local_index],
                    boundary_left=boundary_left,
                    boundary_right=boundary_right,
                    boundary_other_by_owner=boundary_other,
                )
        return sequences

    for track_id in range(tape.track_count):
        row = int(row_index[track_id])
        if row < 0 or row >= tape.row_count:
            continue
        track_ray_coeff = ray_coeff[track_id]
        frame_work: list[tuple[np.ndarray, np.ndarray, int | None]] = []
        owner_points: list[tuple[float, float, float, float]] = []
        owner_work_indices: list[int] = []
        for frame_id in range(tape.frame_count):
            t = float(frame_t[frame_id])
            sorted_depths = np.empty((0,), dtype=np.float64)
            sorted_ids = np.empty((0,), dtype=np.int64)
            slab_id = min(int(math.floor(t * tape.time_slab_count)), tape.time_slab_count - 1)
            csr_row = row * tape.time_slab_count + slab_id
            begin = int(row_offsets[csr_row])
            end = int(row_offsets[csr_row + 1])
            if end > begin:
                local_coeffs = coeffs[begin:end]
                denom = local_coeffs[:, 2] + local_coeffs[:, 3] * t
                numer = local_coeffs[:, 0] + local_coeffs[:, 1] * t
                with np.errstate(divide="ignore", invalid="ignore"):
                    depths = numer / denom
                valid = (
                    (np.abs(denom) >= float(invalid_epsilon))
                    & np.isfinite(depths)
                    & (depths >= float(near))
                    & (depths <= float(far))
                )
                valid_indices = np.flatnonzero(valid)
                if valid_indices.size:
                    valid_depths = depths[valid_indices]
                    order = np.argsort(valid_depths, kind="mergesort")
                    ordered_indices = valid_indices[order]
                    sorted_depths = depths[ordered_indices]
                    sorted_ids = candidate_ids[begin:end][ordered_indices]
            cut_depths, cut_ids = _cut_arrays_from_ordered_depth_ids(
                depths=sorted_depths,
                boundary_ids=sorted_ids,
                near=float(near),
                far=float(far),
            )
            start_segment = _first_nonempty_segment_index(cut_depths)
            ox = float(track_ray_coeff[0] + track_ray_coeff[3] * t)
            oy = float(track_ray_coeff[1] + track_ray_coeff[4] * t)
            oz = float(track_ray_coeff[2] + track_ray_coeff[5] * t)
            dx = float(track_ray_coeff[6] + track_ray_coeff[9] * t)
            dy = float(track_ray_coeff[7] + track_ray_coeff[10] * t)
            dz = float(track_ray_coeff[8] + track_ray_coeff[11] * t)
            if start_segment is not None:
                first_mid = 0.5 * float(cut_depths[start_segment] + cut_depths[start_segment + 1])
                owner_work_indices.append(len(frame_work))
                owner_points.append((ox + dx * first_mid, oy + dy * first_mid, oz + dz * first_mid, t))
            frame_work.append((cut_depths, cut_ids, start_segment))
        initial_owners: list[int | None] = [None for _ in frame_work]
        if owner_points:
            owner_values = _owner_indices_for_points(
                points=np.array(owner_points, dtype=np.float64),
                site_xyz=site_xyz,
                site_t=site_t,
                site_weight=site_weight,
            )
            for work_index, owner in zip(owner_work_indices, owner_values, strict=True):
                initial_owners[work_index] = int(owner)
        sequences[track_id] = _records_from_frame_work(
            frame_work=frame_work,
            initial_owners=initial_owners,
            boundary_left=boundary_left,
            boundary_right=boundary_right,
            boundary_other_by_owner=boundary_other,
        )
    return sequences


def build_gate4_endpoint_delta_replace_tape(
    *,
    tape: Gate4AffineSlabTape,
    sites: tuple[Any, ...],
    near: float,
    far: float,
    invalid_epsilon: float,
    experimental_native_cut_prep_delta: bool = False,
    experimental_native_sorted_delta: bool = False,
    experimental_native_emitted_pack_records: bool = False,
) -> Gate4EndpointDeltaReplaceTape:
    """Build packed delta-replace endpoint records directly from a Gate4 affine slab tape."""
    if tape.layout != "per-track":
        raise ValueError("Gate4 endpoint rows require per-track affine slab tape")
    if tape.row_count != tape.track_count:
        raise ValueError("Gate4 endpoint rows require row_count == track_count")
    if tape.missing_sample_events != 0:
        raise ValueError("Gate4 endpoint rows require zero missing sample events in the affine slab tape")
    if near >= far:
        raise ValueError("near must be less than far")
    if invalid_epsilon <= 0.0:
        raise ValueError("invalid_epsilon must be positive")
    if bool(experimental_native_emitted_pack_records) and tape.time_slab_count != 1:
        raise ValueError("native emitted packed endpoint records currently require a single Gate4 time slab")

    row_offsets = tape.row_offsets.detach().cpu().to(dtype=torch.long).numpy()
    candidate_ids = tape.candidate_ids.detach().cpu().to(dtype=torch.long).numpy()
    coeffs = tape.candidate_depth_coeffs.detach().cpu().to(dtype=torch.float64).numpy()
    frame_t = tape.frame_t.detach().cpu().to(dtype=torch.float64).numpy()
    row_index = tape.row_index.detach().cpu().to(dtype=torch.long).numpy()
    ray_coeff = tape.ray_coeff.detach().cpu().to(dtype=torch.float64).numpy()
    boundary_pairs = tuple((int(boundary.left), int(boundary.right)) for boundary in make_boundaries_4d(sites))
    boundary_left = np.array([left for left, _right in boundary_pairs], dtype=np.int64)
    boundary_right = np.array([right for _left, right in boundary_pairs], dtype=np.int64)
    boundary_other = _boundary_other_by_owner(
        boundary_left=boundary_left,
        boundary_right=boundary_right,
        site_count=len(sites),
    )
    site_xyz = np.array([(float(site.x), float(site.y), float(site.z)) for site in sites], dtype=np.float64)
    site_t = np.array([float(site.t) for site in sites], dtype=np.float64)
    site_weight = np.array([float(site.weight) for site in sites], dtype=np.float64)

    base_offsets = [0]
    base_owner: list[int] = []
    base_left: list[int] = []
    base_right: list[int] = []
    track_change_offsets = [0]
    change_frame: list[int] = []
    change_offsets = [0]
    change_owner: list[int] = []
    change_left: list[int] = []
    change_right: list[int] = []
    base_record: list[int] | None = [] if bool(experimental_native_emitted_pack_records) else None
    change_record: list[int] | None = [] if bool(experimental_native_emitted_pack_records) else None
    native_tensor_chunks = _new_delta_replace_tensor_chunks(
        include_packed_records=bool(experimental_native_emitted_pack_records)
    )

    if tape.time_slab_count == 1:
        direct_csr_op = _gate4_delta_replace_packed_from_coeff_csr_cpu_op()
        if (
            base_record is not None
            and direct_csr_op is not None
            and (bool(experimental_native_sorted_delta) or GATE4_ENABLE_EXPERIMENTAL_NATIVE_SORTED_DELTA)
        ):
            result = direct_csr_op(
                torch.from_numpy(np.ascontiguousarray(row_offsets, dtype=np.int64)),
                torch.from_numpy(np.ascontiguousarray(candidate_ids, dtype=np.int64)),
                torch.from_numpy(np.ascontiguousarray(coeffs, dtype=np.float64)),
                torch.from_numpy(np.ascontiguousarray(row_index, dtype=np.int64)),
                torch.from_numpy(np.ascontiguousarray(ray_coeff, dtype=np.float64)),
                torch.from_numpy(np.ascontiguousarray(frame_t, dtype=np.float64)),
                torch.from_numpy(np.ascontiguousarray(site_xyz, dtype=np.float64)),
                torch.from_numpy(np.ascontiguousarray(site_t, dtype=np.float64)),
                torch.from_numpy(np.ascontiguousarray(site_weight, dtype=np.float64)),
                torch.from_numpy(np.ascontiguousarray(boundary_other, dtype=np.int64)),
                int(tape.frame_count),
                float(near),
                float(far),
                float(invalid_epsilon),
                1.0e-6,
                1.0e-8,
            )
            _append_native_delta_tensor_result(result=result, chunks=native_tensor_chunks)
            return _delta_replace_tape_from_tensor_chunks(native_tensor_chunks)
        for track_begin, chunk_depths, chunk_ids, chunk_valid_counts in _iter_single_slab_sorted_depth_id_chunks(
            row_offsets=row_offsets,
            candidate_ids=candidate_ids,
            coeffs=coeffs,
            frame_t=frame_t,
            row_index=row_index,
            near=float(near),
            far=float(far),
            invalid_epsilon=float(invalid_epsilon),
        ):
            track_end = min(track_begin + int(chunk_valid_counts.shape[0]), tape.track_count)
            sorted_op_available = (
                (bool(experimental_native_sorted_delta) or GATE4_ENABLE_EXPERIMENTAL_NATIVE_SORTED_DELTA)
                and (
                    _gate4_delta_replace_packed_from_sorted_cpu_op() is not None
                    if base_record is not None
                    else _gate4_delta_replace_from_sorted_cpu_op() is not None
                )
            )
            if sorted_op_available:
                row_active = (row_index[track_begin:track_end] >= 0) & (row_index[track_begin:track_end] < tape.row_count)
                _append_native_sorted_delta_chunk(
                    chunk_depths=chunk_depths,
                    chunk_ids=chunk_ids,
                    chunk_valid_counts=chunk_valid_counts,
                    row_active=row_active.astype(np.int64, copy=False),
                    ray_coeff=ray_coeff[track_begin:track_end],
                    frame_t=frame_t,
                    site_xyz=site_xyz,
                    site_t=site_t,
                    site_weight=site_weight,
                    boundary_other_by_owner=boundary_other,
                    frame_count=tape.frame_count,
                    near=float(near),
                    far=float(far),
                    base_offsets=base_offsets,
                    base_owner=base_owner,
                    base_left=base_left,
                    base_right=base_right,
                    track_change_offsets=track_change_offsets,
                    change_frame=change_frame,
                    change_offsets=change_offsets,
                    change_owner=change_owner,
                    change_left=change_left,
                    change_right=change_right,
                    base_record=base_record,
                    change_record=change_record,
                    tensor_chunks=native_tensor_chunks,
                )
                continue
            cut_prep_op_available = (
                (bool(experimental_native_cut_prep_delta) or GATE4_ENABLE_EXPERIMENTAL_NATIVE_CUT_PREP_DELTA)
                and _gate4_cut_arrays_from_sorted_cpu_op() is not None
                and (
                    _gate4_delta_replace_packed_from_cuts_cpu_op() is not None
                    if base_record is not None
                    else _gate4_delta_replace_from_cuts_cpu_op() is not None
                )
            )
            if cut_prep_op_available:
                row_active = (row_index[track_begin:track_end] >= 0) & (row_index[track_begin:track_end] < tape.row_count)
                _append_native_cutprep_delta_chunk(
                    chunk_depths=chunk_depths,
                    chunk_ids=chunk_ids,
                    chunk_valid_counts=chunk_valid_counts,
                    row_active=row_active.astype(np.int64, copy=False),
                    ray_coeff=ray_coeff[track_begin:track_end],
                    frame_t=frame_t,
                    site_xyz=site_xyz,
                    site_t=site_t,
                    site_weight=site_weight,
                    boundary_other_by_owner=boundary_other,
                    frame_count=tape.frame_count,
                    near=float(near),
                    far=float(far),
                    base_offsets=base_offsets,
                    base_owner=base_owner,
                    base_left=base_left,
                    base_right=base_right,
                    track_change_offsets=track_change_offsets,
                    change_frame=change_frame,
                    change_offsets=change_offsets,
                    change_owner=change_owner,
                    change_left=change_left,
                    change_right=change_right,
                    base_record=base_record,
                    change_record=change_record,
                    tensor_chunks=native_tensor_chunks,
                )
                continue
            owner_points: list[tuple[float, float, float, float]] = []
            owner_refs: list[Any] = []
            native_op_available = (
                _gate4_delta_replace_packed_from_cuts_cpu_op() is not None
                if base_record is not None
                else _gate4_delta_replace_from_cuts_cpu_op() is not None
            )
            if native_op_available:
                cut_depth_parts: list[np.ndarray] = []
                cut_id_parts: list[np.ndarray] = []
                cut_offsets = [0]
                start_segments: list[int] = []
                initial_owners: list[int] = []
            else:
                chunk_frame_work: list[list[tuple[np.ndarray, np.ndarray, int | None]]] = []
                chunk_initial_owners: list[list[int | None]] = []
            for local_index, track_id in enumerate(range(track_begin, track_end)):
                row = int(row_index[track_id])
                if row < 0 or row >= tape.row_count:
                    if native_op_available:
                        for _frame_id in range(tape.frame_count):
                            cut_offsets.append(cut_offsets[-1])
                            start_segments.append(-1)
                            initial_owners.append(-1)
                    else:
                        chunk_frame_work.append([])
                        chunk_initial_owners.append([])
                    continue
                track_ray_coeff = ray_coeff[track_id]
                if not native_op_available:
                    frame_work: list[tuple[np.ndarray, np.ndarray, int | None]] = []
                    track_initial_owners: list[int | None] = [None for _ in range(tape.frame_count)]
                for frame_id in range(tape.frame_count):
                    t = float(frame_t[frame_id])
                    valid_count = int(chunk_valid_counts[local_index, frame_id])
                    sorted_depths = (
                        chunk_depths[local_index, :valid_count, frame_id]
                        if valid_count
                        else np.empty((0,), dtype=np.float64)
                    )
                    sorted_ids = (
                        chunk_ids[local_index, :valid_count, frame_id]
                        if valid_count
                        else np.empty((0,), dtype=np.int64)
                    )
                    cut_depths, cut_ids = _cut_arrays_from_ordered_depth_ids(
                        depths=sorted_depths,
                        boundary_ids=sorted_ids,
                        near=float(near),
                        far=float(far),
                    )
                    start_segment = _first_nonempty_segment_index(cut_depths)
                    if native_op_available:
                        cut_depth_parts.append(cut_depths)
                        cut_id_parts.append(cut_ids)
                        cut_offsets.append(cut_offsets[-1] + int(cut_depths.shape[0]))
                        start_segments.append(-1 if start_segment is None else int(start_segment))
                        initial_owners.append(-1)
                    if start_segment is not None:
                        ox = float(track_ray_coeff[0] + track_ray_coeff[3] * t)
                        oy = float(track_ray_coeff[1] + track_ray_coeff[4] * t)
                        oz = float(track_ray_coeff[2] + track_ray_coeff[5] * t)
                        dx = float(track_ray_coeff[6] + track_ray_coeff[9] * t)
                        dy = float(track_ray_coeff[7] + track_ray_coeff[10] * t)
                        dz = float(track_ray_coeff[8] + track_ray_coeff[11] * t)
                        first_mid = 0.5 * float(cut_depths[start_segment] + cut_depths[start_segment + 1])
                        owner_refs.append(
                            local_index * tape.frame_count + frame_id
                            if native_op_available
                            else (local_index, len(frame_work))
                        )
                        owner_points.append((ox + dx * first_mid, oy + dy * first_mid, oz + dz * first_mid, t))
                    if not native_op_available:
                        frame_work.append((cut_depths, cut_ids, start_segment))
                if not native_op_available:
                    chunk_frame_work.append(frame_work)
                    chunk_initial_owners.append(track_initial_owners)
            if owner_points:
                owner_values = _owner_indices_for_points(
                    points=np.array(owner_points, dtype=np.float64),
                    site_xyz=site_xyz,
                    site_t=site_t,
                    site_weight=site_weight,
                )
                if native_op_available:
                    for work_index, owner in zip(owner_refs, owner_values, strict=True):
                        initial_owners[int(work_index)] = int(owner)
                else:
                    for (local_index, frame_work_index), owner in zip(owner_refs, owner_values, strict=True):
                        chunk_initial_owners[local_index][frame_work_index] = int(owner)
            if native_op_available:
                _append_native_delta_chunk(
                    cut_depth_parts=cut_depth_parts,
                    cut_id_parts=cut_id_parts,
                    cut_offsets=cut_offsets,
                    start_segments=start_segments,
                    initial_owners=initial_owners,
                    boundary_other_by_owner=boundary_other,
                    frame_count=tape.frame_count,
                    base_offsets=base_offsets,
                    base_owner=base_owner,
                    base_left=base_left,
                    base_right=base_right,
                    track_change_offsets=track_change_offsets,
                    change_frame=change_frame,
                    change_offsets=change_offsets,
                    change_owner=change_owner,
                    change_left=change_left,
                    change_right=change_right,
                    base_record=base_record,
                    change_record=change_record,
                    tensor_chunks=native_tensor_chunks,
                )
            else:
                for local_index in range(track_end - track_begin):
                    _append_delta_track_rows(
                        frame_work=chunk_frame_work[local_index],
                        initial_owners=chunk_initial_owners[local_index],
                        boundary_left=boundary_left,
                        boundary_right=boundary_right,
                        boundary_other_by_owner=boundary_other,
                        base_offsets=base_offsets,
                        base_owner=base_owner,
                        base_left=base_left,
                        base_right=base_right,
                        track_change_offsets=track_change_offsets,
                        change_frame=change_frame,
                        change_offsets=change_offsets,
                        change_owner=change_owner,
                        change_left=change_left,
                        change_right=change_right,
                    )
        if native_tensor_chunks.native_result_count > 0:
            return _delta_replace_tape_from_tensor_chunks(native_tensor_chunks)
        return _delta_replace_tape_from_lists(
            base_offsets=base_offsets,
            base_owner=base_owner,
            base_left=base_left,
            base_right=base_right,
            track_change_offsets=track_change_offsets,
            change_frame=change_frame,
            change_offsets=change_offsets,
            change_owner=change_owner,
            change_left=change_left,
            change_right=change_right,
            base_record=base_record,
            change_record=change_record,
        )

    for track_id in range(tape.track_count):
        row = int(row_index[track_id])
        if row < 0 or row >= tape.row_count:
            _append_delta_track_rows(
                frame_work=[],
                initial_owners=[],
                boundary_left=boundary_left,
                boundary_right=boundary_right,
                boundary_other_by_owner=boundary_other,
                base_offsets=base_offsets,
                base_owner=base_owner,
                base_left=base_left,
                base_right=base_right,
                track_change_offsets=track_change_offsets,
                change_frame=change_frame,
                change_offsets=change_offsets,
                change_owner=change_owner,
                change_left=change_left,
                change_right=change_right,
            )
            continue
        track_ray_coeff = ray_coeff[track_id]
        frame_work: list[tuple[np.ndarray, np.ndarray, int | None]] = []
        owner_points: list[tuple[float, float, float, float]] = []
        owner_work_indices: list[int] = []
        for frame_id in range(tape.frame_count):
            t = float(frame_t[frame_id])
            sorted_depths = np.empty((0,), dtype=np.float64)
            sorted_ids = np.empty((0,), dtype=np.int64)
            slab_id = min(int(math.floor(t * tape.time_slab_count)), tape.time_slab_count - 1)
            csr_row = row * tape.time_slab_count + slab_id
            begin = int(row_offsets[csr_row])
            end = int(row_offsets[csr_row + 1])
            if end > begin:
                local_coeffs = coeffs[begin:end]
                denom = local_coeffs[:, 2] + local_coeffs[:, 3] * t
                numer = local_coeffs[:, 0] + local_coeffs[:, 1] * t
                with np.errstate(divide="ignore", invalid="ignore"):
                    depths = numer / denom
                valid = (
                    (np.abs(denom) >= float(invalid_epsilon))
                    & np.isfinite(depths)
                    & (depths >= float(near))
                    & (depths <= float(far))
                )
                valid_indices = np.flatnonzero(valid)
                if valid_indices.size:
                    valid_depths = depths[valid_indices]
                    order = np.argsort(valid_depths, kind="mergesort")
                    ordered_indices = valid_indices[order]
                    sorted_depths = depths[ordered_indices]
                    sorted_ids = candidate_ids[begin:end][ordered_indices]
            cut_depths, cut_ids = _cut_arrays_from_ordered_depth_ids(
                depths=sorted_depths,
                boundary_ids=sorted_ids,
                near=float(near),
                far=float(far),
            )
            start_segment = _first_nonempty_segment_index(cut_depths)
            ox = float(track_ray_coeff[0] + track_ray_coeff[3] * t)
            oy = float(track_ray_coeff[1] + track_ray_coeff[4] * t)
            oz = float(track_ray_coeff[2] + track_ray_coeff[5] * t)
            dx = float(track_ray_coeff[6] + track_ray_coeff[9] * t)
            dy = float(track_ray_coeff[7] + track_ray_coeff[10] * t)
            dz = float(track_ray_coeff[8] + track_ray_coeff[11] * t)
            if start_segment is not None:
                first_mid = 0.5 * float(cut_depths[start_segment] + cut_depths[start_segment + 1])
                owner_work_indices.append(len(frame_work))
                owner_points.append((ox + dx * first_mid, oy + dy * first_mid, oz + dz * first_mid, t))
            frame_work.append((cut_depths, cut_ids, start_segment))
        initial_owners: list[int | None] = [None for _ in frame_work]
        if owner_points:
            owner_values = _owner_indices_for_points(
                points=np.array(owner_points, dtype=np.float64),
                site_xyz=site_xyz,
                site_t=site_t,
                site_weight=site_weight,
            )
            for work_index, owner in zip(owner_work_indices, owner_values, strict=True):
                initial_owners[work_index] = int(owner)
        _append_delta_track_rows(
            frame_work=frame_work,
            initial_owners=initial_owners,
            boundary_left=boundary_left,
            boundary_right=boundary_right,
            boundary_other_by_owner=boundary_other,
            base_offsets=base_offsets,
            base_owner=base_owner,
            base_left=base_left,
            base_right=base_right,
            track_change_offsets=track_change_offsets,
            change_frame=change_frame,
            change_offsets=change_offsets,
            change_owner=change_owner,
            change_left=change_left,
            change_right=change_right,
        )

    return _delta_replace_tape_from_lists(
        base_offsets=base_offsets,
        base_owner=base_owner,
        base_left=base_left,
        base_right=base_right,
        track_change_offsets=track_change_offsets,
        change_frame=change_frame,
        change_offsets=change_offsets,
        change_owner=change_owner,
        change_left=change_left,
        change_right=change_right,
        base_record=base_record,
        change_record=change_record,
    )


def build_gate4_boundary_depth_coefficients(
    *,
    tape: Gate4AffineSlabTape,
    boundaries: tuple[Any, ...],
) -> torch.Tensor:
    """Return `[track_count * boundary_count, 4]` depth coefficients for endpoint-record kernels."""
    ray_coeff = tape.ray_coeff.detach().cpu().to(dtype=torch.float64).numpy()
    coeffs = _boundary_depth_coefficients_for_rows(ray_coeff=ray_coeff, boundaries=boundaries)
    return torch.from_numpy(coeffs.reshape(-1, 4)).to(dtype=torch.float32).contiguous()


def _track_time_values(frame_indices: torch.Tensor, *, view: int, frame_count: int) -> torch.Tensor:
    return torch.tensor(
        [_frame_time(int(frame_indices[view * frame_count + frame].item()), frame_count) for frame in range(frame_count)],
        dtype=torch.float64,
    )


def _ray_tuple(ray: torch.Tensor) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    return (
        (float(ray[0].item()), float(ray[1].item()), float(ray[2].item())),
        (float(ray[3].item()), float(ray[4].item()), float(ray[5].item())),
    )


def _fit_all_linear_ray_tracks(
    *,
    rays: torch.Tensor,
    frame_t: torch.Tensor,
    view_count: int,
    frame_count: int,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    sample_count, height, width, payload = rays.shape
    if payload != 6:
        raise ValueError(f"sample rays must have payload dimension 6, got {payload}")
    if sample_count != int(view_count) * int(frame_count):
        raise ValueError("sample count must equal view_count * frame_count")

    rays_by_track = (
        rays.reshape(int(view_count), int(frame_count), int(height), int(width), 6)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )
    samples64 = rays_by_track.to(dtype=torch.float64)
    times64 = frame_t.to(dtype=torch.float64)
    if int(frame_count) == 1:
        slope = torch.zeros_like(samples64[..., 0, :])
        base = samples64[..., 0, :]
    else:
        t_centered = times64 - times64.mean()
        denom = torch.sum(t_centered * t_centered)
        sample_mean = samples64.mean(dim=3)
        if float(denom.item()) <= 0.0:
            slope = torch.zeros_like(sample_mean)
        else:
            slope = torch.sum(
                t_centered.reshape(1, 1, 1, int(frame_count), 1)
                * (samples64 - sample_mean.unsqueeze(3)),
                dim=3,
            ) / denom
        base = sample_mean - slope * times64.mean()
    predicted = base.unsqueeze(3) + times64.reshape(1, 1, 1, int(frame_count), 1) * slope.unsqueeze(3)
    residual = (predicted - samples64).abs()
    ray_coeff = torch.cat(
        (
            base[..., :3],
            slope[..., :3],
            base[..., 3:],
            slope[..., 3:],
        ),
        dim=-1,
    ).reshape(-1, 12).contiguous()
    explicit_rays = rays_by_track.reshape(-1, 6).to(dtype=torch.float32).contiguous()
    return (
        ray_coeff,
        explicit_rays,
        float(residual[..., :3].max().item()),
        float(residual[..., 3:].max().item()),
    )


def build_gate4_affine_slab_tape(
    *,
    boundaries: tuple[Any, ...],
    rays: torch.Tensor,
    frame_indices: torch.Tensor,
    frame_count: int,
    time_slabs: int,
    near: float,
    far: float,
    invalid_epsilon: float,
    residual_depth_padding: float,
    layout: str,
    tile_h: int,
    tile_w: int,
    candidate_order: str,
    sample_validation: str = "full",
) -> Gate4AffineSlabTape:
    rays = rays.detach().cpu().to(dtype=torch.float32)
    frame_indices = frame_indices.detach().cpu().to(dtype=torch.long)
    sample_count, height, width, payload = rays.shape
    if payload != 6:
        raise ValueError(f"sample rays must have payload dimension 6, got {payload}")
    if sample_count % frame_count != 0:
        raise ValueError("sample count must be view_count * frame_count")
    if time_slabs <= 0:
        raise ValueError("time_slabs must be positive")
    if layout not in {"tiled", "per-track"}:
        raise ValueError("layout must be 'tiled' or 'per-track'")
    if candidate_order not in {"boundary-id", "slab-mid-depth"}:
        raise ValueError("candidate_order must be 'boundary-id' or 'slab-mid-depth'")
    if candidate_order == "slab-mid-depth" and layout != "per-track":
        raise ValueError("slab-mid-depth candidate order currently requires per-track layout")
    if sample_validation not in {"full", "skip"}:
        raise ValueError("sample_validation must be 'full' or 'skip'")
    if tile_h <= 0 or tile_w <= 0:
        raise ValueError("tile_h and tile_w must be positive")

    view_count = int(sample_count // frame_count)
    frame_t = _track_time_values(frame_indices, view=0, frame_count=frame_count).to(dtype=torch.float32)
    for view in range(1, view_count):
        view_t = _track_time_values(frame_indices, view=view, frame_count=frame_count).to(dtype=torch.float32)
        if not torch.allclose(frame_t, view_t):
            raise ValueError("Gate 4 affine slab tape expects each train view to use the same frame times")

    track_count = int(view_count * height * width)
    tiles_y = (int(height) + tile_h - 1) // tile_h
    tiles_x = (int(width) + tile_w - 1) // tile_w
    row_count = track_count if layout == "per-track" else int(view_count * tiles_y * tiles_x)
    fast_per_track_skip = bool(layout == "per-track" and sample_validation == "skip")
    row_sets: list[list[set[int]]] = [] if fast_per_track_skip else [[set() for _ in range(time_slabs)] for _ in range(row_count)]
    row_index: list[int] = []
    row_index_np_i32: np.ndarray | None = None
    slabs = slab_ranges(time_slabs)
    ray_coeff_tensor, explicit_rays_f32, max_origin_residual, max_direction_residual = _fit_all_linear_ray_tracks(
        rays=rays,
        frame_t=frame_t,
        view_count=view_count,
        frame_count=frame_count,
    )

    if fast_per_track_skip:
        row_index_np_i32 = np.arange(track_count, dtype=np.int32)
    else:
        for view in range(view_count):
            for y in range(height):
                for x in range(width):
                    track_id = view * int(height) * int(width) + y * int(width) + x
                    row_id = (
                        track_id
                        if layout == "per-track"
                        else view * tiles_y * tiles_x + (y // tile_h) * tiles_x + (x // tile_w)
                    )
                    row_index.append(row_id)

    ray_coeff_np = ray_coeff_tensor.detach().cpu().numpy()
    all_boundary_coeffs = _boundary_depth_coefficients_for_rows(ray_coeff=ray_coeff_np, boundaries=boundaries)

    offsets = [0]
    ids: list[int] = []
    coeffs: list[tuple[float, float, float, float]] = []
    row_candidate_counts: list[int] = []
    fast_offsets_np: np.ndarray | None = None
    fast_ids_np: np.ndarray | None = None
    fast_coeffs_np: np.ndarray | None = None
    fast_counts_np: np.ndarray | None = None
    if fast_per_track_skip:
        masks_by_slab = [
            _compiled_slab_event_mask_from_coeffs(
                coeffs=all_boundary_coeffs,
                t0=t0,
                t1=t1,
                near=near,
                far=far,
                invalid_epsilon=invalid_epsilon,
                residual_depth_padding=residual_depth_padding,
            )
            for t0, t1 in slabs
        ]
        fast_offsets_np, fast_ids_np, fast_coeffs_np, fast_counts_np = _candidate_csr_from_per_track_event_masks(
            masks_by_slab=masks_by_slab,
            all_boundary_coeffs=all_boundary_coeffs,
            slabs=slabs,
            candidate_order=candidate_order,
            invalid_epsilon=invalid_epsilon,
        )
    else:
        for track_id, row_id in enumerate(row_index):
            track_coeffs = all_boundary_coeffs[track_id]
            for slab_id, (t0, t1) in enumerate(slabs):
                events = _compiled_slab_events_from_coeffs(
                    coeffs=track_coeffs,
                    t0=t0,
                    t1=t1,
                    near=near,
                    far=far,
                    invalid_epsilon=invalid_epsilon,
                    residual_depth_padding=residual_depth_padding,
                )
                row_sets[row_id][slab_id].update(int(event) for event in events)
        for row in range(row_count):
            for slab_id in range(time_slabs):
                if candidate_order == "slab-mid-depth":
                    t_mid = 0.5 * (slabs[slab_id][0] + slabs[slab_id][1])
                    candidate_array = np.array(sorted(row_sets[row][slab_id]), dtype=np.int64)
                    if candidate_array.size:
                        row_coeffs = all_boundary_coeffs[row, candidate_array]
                        denom = row_coeffs[:, 2] + row_coeffs[:, 3] * t_mid
                        with np.errstate(divide="ignore", invalid="ignore"):
                            depth = (row_coeffs[:, 0] + row_coeffs[:, 1] * t_mid) / denom
                        invalid = np.abs(denom) < invalid_epsilon
                        order = np.lexsort((candidate_array, depth, invalid.astype(np.int64)))
                        candidates = [int(candidate_array[index]) for index in order]
                    else:
                        candidates = []
                else:
                    candidates = sorted(row_sets[row][slab_id])
                ids.extend(candidates)
                if layout == "per-track":
                    coeffs.extend(tuple(float(value) for value in all_boundary_coeffs[row, boundary_id]) for boundary_id in candidates)
                row_candidate_counts.append(len(candidates))
                offsets.append(len(ids))

    per_frame_event_sum = 0
    missing_sample_events = 0
    extra_candidate_events = 0
    candidate_replay_iterations = 0
    depth_order_tolerance = 1.0e-6
    depth_order_checked_samples = 0
    depth_order_valid_depth_values = 0
    depth_order_adjacent_pairs = 0
    depth_order_adjacent_inversions = 0
    depth_order_samples_with_adjacent_inversions = 0
    depth_order_max_adjacent_inversions_per_sample = 0
    depth_order_max_depth_drop = 0.0
    if sample_validation == "full":
        for view in range(view_count):
            for y in range(height):
                for x in range(width):
                    track_id = view * int(height) * int(width) + y * int(width) + x
                    row = row_index[track_id]
                    for frame in range(frame_count):
                        sample_index = view * frame_count + frame
                        t = float(frame_t[frame].item())
                        slab_id = min(int(math.floor(t * time_slabs)), time_slabs - 1)
                        origin, direction = _ray_tuple(rays[sample_index, y, x])
                        sample_events, _invalid = event_set_for_ray(
                            boundaries=boundaries,
                            origin=origin,
                            direction=direction,
                            t=t,
                            near=near,
                            far=far,
                            invalid_epsilon=invalid_epsilon,
                        )
                        candidates = row_sets[row][slab_id]
                        per_frame_event_sum += len(sample_events)
                        missing_sample_events += len(sample_events - candidates)
                        extra_candidate_events += len(candidates - sample_events)
                        candidate_replay_iterations += len(candidates)
                        if layout == "per-track":
                            depth_order_checked_samples += 1
                            csr_row = row * time_slabs + slab_id
                            begin = offsets[csr_row]
                            end = offsets[csr_row + 1]
                            valid_depths: list[float] = []
                            for cursor in range(begin, end):
                                numer_base, numer_slope, denom_base, denom_slope = coeffs[cursor]
                                denom = denom_base + denom_slope * t
                                if abs(denom) < invalid_epsilon:
                                    continue
                                depth = (numer_base + numer_slope * t) / denom
                                if math.isfinite(depth) and near <= depth <= far:
                                    valid_depths.append(float(depth))
                            depth_order_valid_depth_values += len(valid_depths)
                            if len(valid_depths) > 1:
                                sample_inversions = 0
                                depth_order_adjacent_pairs += len(valid_depths) - 1
                                for left, right in zip(valid_depths[:-1], valid_depths[1:], strict=True):
                                    drop = left - right
                                    if drop > depth_order_tolerance:
                                        sample_inversions += 1
                                        depth_order_max_depth_drop = max(depth_order_max_depth_drop, float(drop))
                                if sample_inversions:
                                    depth_order_adjacent_inversions += sample_inversions
                                    depth_order_samples_with_adjacent_inversions += 1
                                    depth_order_max_adjacent_inversions_per_sample = max(
                                        depth_order_max_adjacent_inversions_per_sample,
                                        sample_inversions,
                                    )
    else:
        slab_frame_counts = np.zeros(time_slabs, dtype=np.int64)
        for frame in range(frame_count):
            t = float(frame_t[frame].item())
            slab_id = min(int(math.floor(t * time_slabs)), time_slabs - 1)
            slab_frame_counts[slab_id] += 1
        if fast_counts_np is not None:
            count_grid = fast_counts_np.reshape(row_count, time_slabs)
            candidate_replay_iterations = int((count_grid * slab_frame_counts.reshape(1, time_slabs)).sum())
        else:
            row_track_counts = np.bincount(np.asarray(row_index, dtype=np.int64), minlength=row_count)
            for row in range(row_count):
                for slab_id in range(time_slabs):
                    row_slab_index = row * time_slabs + slab_id
                    candidate_replay_iterations += (
                        int(row_candidate_counts[row_slab_index])
                        * int(row_track_counts[row])
                        * int(slab_frame_counts[slab_id])
                    )

    if row_index_np_i32 is not None:
        row_index_i32 = torch.from_numpy(row_index_np_i32).contiguous()
    else:
        row_index_i32 = torch.tensor(row_index, dtype=torch.int32).contiguous()
    if fast_offsets_np is not None and fast_ids_np is not None and fast_coeffs_np is not None:
        row_offsets_i32 = torch.from_numpy(fast_offsets_np.astype(np.int32, copy=False)).contiguous()
        candidate_ids_i32 = torch.from_numpy(fast_ids_np.astype(np.int32, copy=False)).contiguous()
        candidate_depth_coeff_f32 = torch.from_numpy(fast_coeffs_np.astype(np.float32, copy=False)).reshape(-1, 4).contiguous()
        counts = torch.from_numpy(fast_counts_np.astype(np.int64, copy=False)).contiguous()
    else:
        row_offsets_i32 = torch.tensor(offsets, dtype=torch.int32).contiguous()
        candidate_ids_i32 = torch.tensor(ids, dtype=torch.int32).contiguous()
        candidate_depth_coeff_f32 = torch.tensor(coeffs, dtype=torch.float32).reshape(-1, 4).contiguous()
        counts = torch.tensor(row_candidate_counts, dtype=torch.int64)
    ray_coeff_f32 = ray_coeff_tensor.to(dtype=torch.float32).contiguous()

    return Gate4AffineSlabTape(
        layout=layout,
        candidate_order=candidate_order,
        view_count=view_count,
        height=int(height),
        width=int(width),
        track_count=track_count,
        row_count=row_count,
        frame_count=frame_count,
        time_slab_count=time_slabs,
        tile_shape=[tile_h, tile_w] if layout == "tiled" else None,
        tile_grid_shape=[view_count, tiles_y, tiles_x] if layout == "tiled" else None,
        frame_t=frame_t.contiguous(),
        ray_coeff=ray_coeff_f32,
        explicit_rays=explicit_rays_f32,
        row_index=row_index_i32,
        row_offsets=row_offsets_i32,
        candidate_ids=candidate_ids_i32,
        candidate_depth_coeffs=candidate_depth_coeff_f32,
        per_frame_event_sum=int(per_frame_event_sum),
        missing_sample_events=int(missing_sample_events),
        extra_candidate_events=int(extra_candidate_events),
        candidate_replay_iterations=int(candidate_replay_iterations),
        candidate_depth_order={
            "sample_validation": sample_validation,
            "missing_sample_events_authoritative": bool(sample_validation == "full"),
            "checked_samples": int(depth_order_checked_samples),
            "valid_depth_values": int(depth_order_valid_depth_values),
            "adjacent_pairs": int(depth_order_adjacent_pairs),
            "adjacent_inversions": int(depth_order_adjacent_inversions),
            "samples_with_adjacent_inversions": int(depth_order_samples_with_adjacent_inversions),
            "max_adjacent_inversions_per_sample": int(depth_order_max_adjacent_inversions_per_sample),
            "max_depth_drop": float(depth_order_max_depth_drop),
            "tolerance": float(depth_order_tolerance),
            "ordered_append_safe": bool(depth_order_checked_samples > 0 and depth_order_adjacent_inversions == 0),
        },
        direct_boundary_iterations=int(track_count * frame_count * len(boundaries)),
        compiled_boundary_tests=int(track_count * time_slabs * len(boundaries)),
        candidate_count=int(candidate_ids_i32.numel()),
        max_candidates_per_row=int(counts.max().item()) if counts.numel() else 0,
        avg_candidates_per_row=float(counts.to(dtype=torch.float32).mean().item()) if counts.numel() else 0.0,
        empty_row_count=int((counts == 0).sum().item()) if counts.numel() else 0,
        max_origin_residual=float(max_origin_residual),
        max_direction_residual=float(max_direction_residual),
    )
