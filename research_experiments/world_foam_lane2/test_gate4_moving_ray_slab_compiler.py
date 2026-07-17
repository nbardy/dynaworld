from __future__ import annotations

import math
import unittest
from unittest import mock

import numpy as np
import torch

import gate4_affine_slab_tape as gate4_tape_module
from gate1_realray_per_sample_reference import (
    EPS,
    Site4D,
    dedupe_sorted_depths,
    make_boundaries_4d,
    owner_at_4d,
    render_one_ray,
)
from gate4_moving_ray_slab_compiler import (
    LinearRayTrack,
    boundary_depth_coefficients,
    compiled_slab_event_set_for_track,
    event_set_for_ray,
    track_ray_at,
)
from gate4_affine_slab_tape import (
    GATE4_FAR_CUT_ID,
    GATE4_NEAR_CUT_ID,
    _boundary_other_by_owner,
    _cut_arrays_from_ordered_depth_ids,
    _first_nonempty_segment_index,
    _iter_single_slab_sorted_depth_id_chunks,
    _owner_indices_for_points,
    _owner_run_records_from_cut_arrays,
    build_gate4_affine_slab_tape,
    build_gate4_boundary_depth_coefficients,
    build_gate4_endpoint_delta_replace_tape,
    build_gate4_endpoint_run_sequences,
)
from probe_endpoint_record_delta_replay import pack_endpoint_record_delta_replace_tape
from probe_owner_run_boundary_tape import _build_owner_run_sequences


def _render_one_ray_from_compiled_candidates(
    *,
    sites: tuple[Site4D, ...],
    track: LinearRayTrack,
    candidate_ids: set[int],
    t: float,
    near: float,
    far: float,
    invalid_epsilon: float,
    transmittance_threshold: float,
) -> tuple[tuple[float, float, float], float, float, int, int]:
    boundaries = make_boundaries_4d(sites)
    depths: list[float] = []
    invalid = 0
    for boundary_id in candidate_ids:
        numer_base, numer_slope, denom_base, denom_slope = boundary_depth_coefficients(
            boundaries[boundary_id],
            track,
        )
        denom = denom_base + denom_slope * t
        if abs(denom) < invalid_epsilon:
            invalid += 1
            continue
        depth = (numer_base + numer_slope * t) / denom
        if near <= depth <= far:
            depths.append(depth)
    cuts = [near, *dedupe_sorted_depths(depths), far]
    origin, direction = track_ray_at(track, t)
    rgb = [0.0, 0.0, 0.0]
    alpha_accum = 0.0
    depth_accum = 0.0
    transmittance = 1.0
    segment_count = 0
    for depth0, depth1 in zip(cuts, cuts[1:]):
        if depth1 - depth0 <= EPS:
            continue
        mid = 0.5 * (depth0 + depth1)
        site = sites[
            owner_at_4d(
                sites,
                x=origin[0] + direction[0] * mid,
                y=origin[1] + direction[1] * mid,
                z=origin[2] + direction[2] * mid,
                t=t,
            )
        ]
        density = max(float(site.rgba[3]), 0.0)
        segment_alpha = 1.0 - math.exp(-density * (depth1 - depth0))
        contribution = transmittance * segment_alpha
        rgb[0] += contribution * float(site.rgba[0])
        rgb[1] += contribution * float(site.rgba[1])
        rgb[2] += contribution * float(site.rgba[2])
        alpha_accum += contribution
        depth_accum += contribution * mid
        transmittance *= 1.0 - segment_alpha
        segment_count += 1
        if transmittance <= transmittance_threshold:
            break
    expected_depth = depth_accum / max(alpha_accum, EPS) if alpha_accum > 0.0 else far
    return (rgb[0], rgb[1], rgb[2]), alpha_accum, expected_depth, segment_count, invalid


class Gate4MovingRaySlabCompilerTest(unittest.TestCase):
    def test_native_endpoint_record_packer_matches_bit_layout(self) -> None:
        try:
            pack_op = torch.ops.world_foam_lane2_fused_slab_v0.pack_endpoint_records_i32_cpu
        except (AttributeError, RuntimeError):
            self.skipTest("world_foam_lane2_fused_slab_v0 extension not available")

        owner = torch.tensor([0, 5, -1, 255], dtype=torch.int32)
        left = torch.tensor([GATE4_NEAR_CUT_ID, GATE4_FAR_CUT_ID, 0, 4093], dtype=torch.int32)
        right = torch.tensor([GATE4_FAR_CUT_ID, 7, 12, 3], dtype=torch.int32)
        expected = torch.tensor(
            [
                1 << 20,
                5 | (1 << 8) | (9 << 20),
                (2 << 8) | (14 << 20),
                255 | (4095 << 8) | (5 << 20),
            ],
            dtype=torch.int32,
        )
        torch.testing.assert_close(pack_op(owner, left, right), expected)

        with self.assertRaises(RuntimeError):
            pack_op(torch.tensor([256], dtype=torch.int32), left[:1], right[:1])
        with self.assertRaises(RuntimeError):
            pack_op(owner[:1], torch.tensor([-3], dtype=torch.int32), right[:1])

    def _high_cap_moving_fixture(self) -> tuple[tuple[Site4D, ...], torch.Tensor, torch.Tensor]:
        sites: list[Site4D] = []
        for index in range(24):
            angle = 2.0 * math.pi * float(index) / 24.0
            radius = 0.15 + 0.08 * float(index % 6)
            sites.append(
                Site4D(
                    x=radius * math.cos(angle),
                    y=radius * math.sin(angle),
                    z=0.6 + 0.13 * float(index % 8),
                    t=float(index % 5) / 4.0,
                    weight=0.002 * float((index * 7) % 11),
                    rgba=(0.1 + 0.03 * float(index), 0.2, 0.3, 0.8 + 0.01 * float(index)),
                )
            )

        frame_count = 16
        xs = torch.linspace(-0.45, 0.45, 12)
        ys = torch.linspace(-0.35, 0.35, 12)
        frames: list[list[list[list[float]]]] = []
        for frame_id in range(frame_count):
            tau = float(frame_id) / float(frame_count - 1) - 0.5
            frame: list[list[list[float]]] = []
            for y_value in ys:
                row: list[list[float]] = []
                for x_value in xs:
                    origin = torch.tensor([0.08 * tau, -0.03 * tau, 0.02 * tau])
                    direction = torch.tensor([float(x_value) + 0.02 * tau, float(y_value) - 0.01 * tau, 1.0])
                    row.append(torch.cat([origin, direction]).tolist())
                frame.append(row)
            frames.append(frame)
        return tuple(sites), torch.tensor(frames, dtype=torch.float32), torch.arange(frame_count, dtype=torch.long)

    def test_highcap_single_slab_sorted_rows_match_cut_array_delta_records(self) -> None:
        sites, rays, frame_indices = self._high_cap_moving_fixture()
        boundaries = make_boundaries_4d(sites)
        frame_count = int(rays.shape[0])
        near = 0.1
        far = 4.0
        invalid_epsilon = 1.0e-7
        tape = build_gate4_affine_slab_tape(
            boundaries=boundaries,
            rays=rays,
            frame_indices=frame_indices,
            frame_count=frame_count,
            time_slabs=1,
            near=near,
            far=far,
            invalid_epsilon=invalid_epsilon,
            residual_depth_padding=0.001,
            layout="per-track",
            tile_h=1,
            tile_w=1,
            candidate_order="slab-mid-depth",
            sample_validation="skip",
        )
        self.assertEqual(tape.missing_sample_events, 0)
        self.assertGreaterEqual(tape.max_candidates_per_row, 200)

        endpoint_sequences = build_gate4_endpoint_run_sequences(
            tape=tape,
            sites=sites,
            near=near,
            far=far,
            invalid_epsilon=invalid_epsilon,
        )
        direct_delta = build_gate4_endpoint_delta_replace_tape(
            tape=tape,
            sites=sites,
            near=near,
            far=far,
            invalid_epsilon=invalid_epsilon,
        )
        packed_delta = pack_endpoint_record_delta_replace_tape(endpoint_sequences, frame_count=frame_count)
        for attr in (
            "base_offsets_i32",
            "base_owner_i32",
            "base_left_i32",
            "base_right_i32",
            "track_change_offsets_i32",
            "change_frame_i32",
            "change_offsets_i32",
            "change_owner_i32",
            "change_left_i32",
            "change_right_i32",
        ):
            torch.testing.assert_close(getattr(direct_delta, attr), getattr(packed_delta, attr))
        if gate4_tape_module._gate4_cut_arrays_from_sorted_cpu_op() is not None:
            native_cutprep_delta = build_gate4_endpoint_delta_replace_tape(
                tape=tape,
                sites=sites,
                near=near,
                far=far,
                invalid_epsilon=invalid_epsilon,
                experimental_native_cut_prep_delta=True,
            )
            for attr in (
                "base_offsets_i32",
                "base_owner_i32",
                "base_left_i32",
                "base_right_i32",
                "track_change_offsets_i32",
                "change_frame_i32",
                "change_offsets_i32",
                "change_owner_i32",
                "change_left_i32",
                "change_right_i32",
            ):
                torch.testing.assert_close(getattr(native_cutprep_delta, attr), getattr(packed_delta, attr))
        if gate4_tape_module._gate4_delta_replace_packed_from_cuts_cpu_op() is not None:
            native_emitted_packed_delta = build_gate4_endpoint_delta_replace_tape(
                tape=tape,
                sites=sites,
                near=near,
                far=far,
                invalid_epsilon=invalid_epsilon,
                experimental_native_emitted_pack_records=True,
            )
            for attr in (
                "base_offsets_i32",
                "base_owner_i32",
                "base_left_i32",
                "base_right_i32",
                "track_change_offsets_i32",
                "change_frame_i32",
                "change_offsets_i32",
                "change_owner_i32",
                "change_left_i32",
                "change_right_i32",
            ):
                torch.testing.assert_close(getattr(native_emitted_packed_delta, attr), getattr(packed_delta, attr))
            self.assertIsNotNone(native_emitted_packed_delta.base_record_i32)
            self.assertIsNotNone(native_emitted_packed_delta.change_record_i32)
            pack_op = torch.ops.world_foam_lane2_fused_slab_v0.pack_endpoint_records_i32_cpu
            torch.testing.assert_close(
                native_emitted_packed_delta.base_record_i32,
                pack_op(
                    packed_delta.base_owner_i32.contiguous(),
                    packed_delta.base_left_i32.contiguous(),
                    packed_delta.base_right_i32.contiguous(),
                ),
            )
            torch.testing.assert_close(
                native_emitted_packed_delta.change_record_i32,
                pack_op(
                    packed_delta.change_owner_i32.contiguous(),
                    packed_delta.change_left_i32.contiguous(),
                    packed_delta.change_right_i32.contiguous(),
                ),
            )
        direct_csr_op = gate4_tape_module._gate4_delta_replace_packed_from_coeff_csr_cpu_op()
        if direct_csr_op is not None:
            direct_csr_call_count = 0

            def counted_direct_csr_op(*args, **kwargs):
                nonlocal direct_csr_call_count
                direct_csr_call_count += 1
                return direct_csr_op(*args, **kwargs)

            with mock.patch.object(
                gate4_tape_module,
                "_gate4_delta_replace_packed_from_coeff_csr_cpu_op",
                return_value=counted_direct_csr_op,
            ):
                native_direct_csr_packed_delta = build_gate4_endpoint_delta_replace_tape(
                    tape=tape,
                    sites=sites,
                    near=near,
                    far=far,
                    invalid_epsilon=invalid_epsilon,
                    experimental_native_sorted_delta=True,
                    experimental_native_emitted_pack_records=True,
                )
            self.assertEqual(direct_csr_call_count, 1)
            for attr in (
                "base_offsets_i32",
                "base_owner_i32",
                "base_left_i32",
                "base_right_i32",
                "track_change_offsets_i32",
                "change_frame_i32",
                "change_offsets_i32",
                "change_owner_i32",
                "change_left_i32",
                "change_right_i32",
            ):
                torch.testing.assert_close(getattr(native_direct_csr_packed_delta, attr), getattr(packed_delta, attr))
            self.assertIsNotNone(native_direct_csr_packed_delta.base_record_i32)
            self.assertIsNotNone(native_direct_csr_packed_delta.change_record_i32)
            pack_op = torch.ops.world_foam_lane2_fused_slab_v0.pack_endpoint_records_i32_cpu
            torch.testing.assert_close(
                native_direct_csr_packed_delta.base_record_i32,
                pack_op(
                    packed_delta.base_owner_i32.contiguous(),
                    packed_delta.base_left_i32.contiguous(),
                    packed_delta.base_right_i32.contiguous(),
                ),
            )
            torch.testing.assert_close(
                native_direct_csr_packed_delta.change_record_i32,
                pack_op(
                    packed_delta.change_owner_i32.contiguous(),
                    packed_delta.change_left_i32.contiguous(),
                    packed_delta.change_right_i32.contiguous(),
                ),
            )
        if gate4_tape_module._gate4_delta_replace_from_sorted_cpu_op() is not None:
            native_sorted_delta = build_gate4_endpoint_delta_replace_tape(
                tape=tape,
                sites=sites,
                near=near,
                far=far,
                invalid_epsilon=invalid_epsilon,
                experimental_native_sorted_delta=True,
            )
            for attr in (
                "base_offsets_i32",
                "base_owner_i32",
                "base_left_i32",
                "base_right_i32",
                "track_change_offsets_i32",
                "change_frame_i32",
                "change_offsets_i32",
                "change_owner_i32",
                "change_left_i32",
                "change_right_i32",
            ):
                torch.testing.assert_close(getattr(native_sorted_delta, attr), getattr(packed_delta, attr))
            pack_op = torch.ops.world_foam_lane2_fused_slab_v0.pack_endpoint_records_i32_cpu
            if gate4_tape_module._gate4_delta_replace_packed_from_sorted_cpu_op() is not None:
                native_sorted_packed_delta = build_gate4_endpoint_delta_replace_tape(
                    tape=tape,
                    sites=sites,
                    near=near,
                    far=far,
                    invalid_epsilon=invalid_epsilon,
                    experimental_native_sorted_delta=True,
                    experimental_native_emitted_pack_records=True,
                )
                for attr in (
                    "base_offsets_i32",
                    "base_owner_i32",
                    "base_left_i32",
                    "base_right_i32",
                    "track_change_offsets_i32",
                    "change_frame_i32",
                    "change_offsets_i32",
                    "change_owner_i32",
                    "change_left_i32",
                    "change_right_i32",
                ):
                    torch.testing.assert_close(getattr(native_sorted_packed_delta, attr), getattr(packed_delta, attr))
                self.assertIsNotNone(native_sorted_packed_delta.base_record_i32)
                self.assertIsNotNone(native_sorted_packed_delta.change_record_i32)
                torch.testing.assert_close(
                    native_sorted_packed_delta.base_record_i32,
                    pack_op(
                        packed_delta.base_owner_i32.contiguous(),
                        packed_delta.base_left_i32.contiguous(),
                        packed_delta.base_right_i32.contiguous(),
                    ),
                )
                torch.testing.assert_close(
                    native_sorted_packed_delta.change_record_i32,
                    pack_op(
                        packed_delta.change_owner_i32.contiguous(),
                        packed_delta.change_left_i32.contiguous(),
                        packed_delta.change_right_i32.contiguous(),
                    ),
                )
            with (
                mock.patch.object(
                    gate4_tape_module,
                    "_gate4_delta_replace_packed_from_sorted_cpu_op",
                    return_value=None,
                ),
                mock.patch.object(
                    gate4_tape_module,
                    "_gate4_delta_replace_packed_from_cuts_cpu_op",
                    return_value=None,
                ),
            ):
                python_fallback_sorted_packed_delta = build_gate4_endpoint_delta_replace_tape(
                    tape=tape,
                    sites=sites,
                    near=near,
                    far=far,
                    invalid_epsilon=invalid_epsilon,
                    experimental_native_sorted_delta=True,
                    experimental_native_emitted_pack_records=True,
                )
            for attr in (
                "base_offsets_i32",
                "base_owner_i32",
                "base_left_i32",
                "base_right_i32",
                "track_change_offsets_i32",
                "change_frame_i32",
                "change_offsets_i32",
                "change_owner_i32",
                "change_left_i32",
                "change_right_i32",
            ):
                torch.testing.assert_close(getattr(python_fallback_sorted_packed_delta, attr), getattr(packed_delta, attr))
            self.assertIsNotNone(python_fallback_sorted_packed_delta.base_record_i32)
            self.assertIsNotNone(python_fallback_sorted_packed_delta.change_record_i32)
            self.assertEqual(
                python_fallback_sorted_packed_delta.storage_bytes,
                packed_delta.storage_bytes
                + python_fallback_sorted_packed_delta.base_record_i32.numel()
                * python_fallback_sorted_packed_delta.base_record_i32.element_size()
                + python_fallback_sorted_packed_delta.change_record_i32.numel()
                * python_fallback_sorted_packed_delta.change_record_i32.element_size(),
            )
            torch.testing.assert_close(
                python_fallback_sorted_packed_delta.base_record_i32,
                pack_op(
                    packed_delta.base_owner_i32.contiguous(),
                    packed_delta.base_left_i32.contiguous(),
                    packed_delta.base_right_i32.contiguous(),
                ),
            )
            torch.testing.assert_close(
                python_fallback_sorted_packed_delta.change_record_i32,
                pack_op(
                    packed_delta.change_owner_i32.contiguous(),
                    packed_delta.change_left_i32.contiguous(),
                    packed_delta.change_right_i32.contiguous(),
                ),
            )
            if gate4_tape_module._gate4_delta_replace_packed_from_cuts_cpu_op() is not None:
                with mock.patch.object(
                    gate4_tape_module,
                    "_gate4_delta_replace_packed_from_sorted_cpu_op",
                    return_value=None,
                ):
                    fallback_sorted_packed_delta = build_gate4_endpoint_delta_replace_tape(
                        tape=tape,
                        sites=sites,
                        near=near,
                        far=far,
                        invalid_epsilon=invalid_epsilon,
                        experimental_native_sorted_delta=True,
                        experimental_native_emitted_pack_records=True,
                    )
                for attr in (
                    "base_offsets_i32",
                    "base_owner_i32",
                    "base_left_i32",
                    "base_right_i32",
                    "track_change_offsets_i32",
                    "change_frame_i32",
                    "change_offsets_i32",
                    "change_owner_i32",
                    "change_left_i32",
                    "change_right_i32",
                ):
                    torch.testing.assert_close(getattr(fallback_sorted_packed_delta, attr), getattr(packed_delta, attr))
                self.assertIsNotNone(fallback_sorted_packed_delta.base_record_i32)
                self.assertIsNotNone(fallback_sorted_packed_delta.change_record_i32)
                torch.testing.assert_close(
                    fallback_sorted_packed_delta.base_record_i32,
                    pack_op(
                        packed_delta.base_owner_i32.contiguous(),
                        packed_delta.base_left_i32.contiguous(),
                        packed_delta.base_right_i32.contiguous(),
                    ),
                )
                torch.testing.assert_close(
                    fallback_sorted_packed_delta.change_record_i32,
                    pack_op(
                        packed_delta.change_owner_i32.contiguous(),
                        packed_delta.change_left_i32.contiguous(),
                        packed_delta.change_right_i32.contiguous(),
                    ),
                )

        row_offsets = tape.row_offsets.detach().cpu().to(dtype=torch.long).numpy()
        candidate_ids = tape.candidate_ids.detach().cpu().to(dtype=torch.long).numpy()
        coeffs = tape.candidate_depth_coeffs.detach().cpu().to(dtype=torch.float64).numpy()
        frame_t = tape.frame_t.detach().cpu().to(dtype=torch.float64).numpy()
        row_index = tape.row_index.detach().cpu().to(dtype=torch.long).numpy()
        ray_coeff = tape.ray_coeff.detach().cpu().to(dtype=torch.float64).numpy()
        boundary_left = np.array([boundary.left for boundary in boundaries], dtype=np.int64)
        boundary_right = np.array([boundary.right for boundary in boundaries], dtype=np.int64)
        boundary_other = _boundary_other_by_owner(
            boundary_left=boundary_left,
            boundary_right=boundary_right,
            site_count=len(sites),
        )
        site_xyz = np.array([(site.x, site.y, site.z) for site in sites], dtype=np.float64)
        site_t = np.array([site.t for site in sites], dtype=np.float64)
        site_weight = np.array([site.weight for site in sites], dtype=np.float64)

        duplicate_frame_count = 0
        no_dedupe_mismatch_count = 0
        for track_begin, chunk_depths, chunk_ids, chunk_valid_counts in _iter_single_slab_sorted_depth_id_chunks(
            row_offsets=row_offsets,
            candidate_ids=candidate_ids,
            coeffs=coeffs,
            frame_t=frame_t,
            row_index=row_index,
            near=near,
            far=far,
            invalid_epsilon=invalid_epsilon,
        ):
            track_end = min(track_begin + int(chunk_valid_counts.shape[0]), tape.track_count)
            for local_index, track_id in enumerate(range(track_begin, track_end)):
                if row_index[track_id] < 0:
                    continue
                for frame_id in range(frame_count):
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
                    if valid_count > 1 and np.any(np.abs(np.diff(sorted_depths)) <= 1.0e-6):
                        duplicate_frame_count += 1
                    cut_depths, cut_ids = _cut_arrays_from_ordered_depth_ids(
                        depths=sorted_depths,
                        boundary_ids=sorted_ids,
                        near=near,
                        far=far,
                    )
                    start_segment = _first_nonempty_segment_index(cut_depths)
                    if start_segment is None:
                        continue
                    t_value = float(frame_t[frame_id])
                    track_ray_coeff = ray_coeff[track_id]
                    midpoint = 0.5 * float(cut_depths[start_segment] + cut_depths[start_segment + 1])
                    owner_point = (
                        float(track_ray_coeff[0] + track_ray_coeff[3] * t_value)
                        + float(track_ray_coeff[6] + track_ray_coeff[9] * t_value) * midpoint,
                        float(track_ray_coeff[1] + track_ray_coeff[4] * t_value)
                        + float(track_ray_coeff[7] + track_ray_coeff[10] * t_value) * midpoint,
                        float(track_ray_coeff[2] + track_ray_coeff[5] * t_value)
                        + float(track_ray_coeff[8] + track_ray_coeff[11] * t_value) * midpoint,
                        t_value,
                    )
                    initial_owner = int(
                        _owner_indices_for_points(
                            points=np.array([owner_point], dtype=np.float64),
                            site_xyz=site_xyz,
                            site_t=site_t,
                            site_weight=site_weight,
                        )[0]
                    )
                    expected = _owner_run_records_from_cut_arrays(
                        cut_depths=cut_depths,
                        cut_ids=cut_ids,
                        boundary_left=boundary_left,
                        boundary_right=boundary_right,
                        boundary_other_by_owner=boundary_other,
                        start_segment=start_segment,
                        initial_owner=initial_owner,
                    )

                    no_dedupe_depths = np.empty(valid_count + 2, dtype=np.float64)
                    no_dedupe_ids = np.empty(valid_count + 2, dtype=np.int64)
                    no_dedupe_depths[0] = near
                    no_dedupe_depths[-1] = far
                    no_dedupe_ids[0] = GATE4_NEAR_CUT_ID
                    no_dedupe_ids[-1] = GATE4_FAR_CUT_ID
                    if valid_count:
                        no_dedupe_depths[1:-1] = sorted_depths
                        no_dedupe_ids[1:-1] = sorted_ids
                    no_dedupe_start = _first_nonempty_segment_index(no_dedupe_depths)
                    if no_dedupe_start is None:
                        no_dedupe_records = tuple()
                    else:
                        no_dedupe_midpoint = 0.5 * float(
                            no_dedupe_depths[no_dedupe_start] + no_dedupe_depths[no_dedupe_start + 1]
                        )
                        no_dedupe_point = (
                            float(track_ray_coeff[0] + track_ray_coeff[3] * t_value)
                            + float(track_ray_coeff[6] + track_ray_coeff[9] * t_value) * no_dedupe_midpoint,
                            float(track_ray_coeff[1] + track_ray_coeff[4] * t_value)
                            + float(track_ray_coeff[7] + track_ray_coeff[10] * t_value) * no_dedupe_midpoint,
                            float(track_ray_coeff[2] + track_ray_coeff[5] * t_value)
                            + float(track_ray_coeff[8] + track_ray_coeff[11] * t_value) * no_dedupe_midpoint,
                            t_value,
                        )
                        no_dedupe_owner = int(
                            _owner_indices_for_points(
                                points=np.array([no_dedupe_point], dtype=np.float64),
                                site_xyz=site_xyz,
                                site_t=site_t,
                                site_weight=site_weight,
                            )[0]
                        )
                        no_dedupe_records = _owner_run_records_from_cut_arrays(
                            cut_depths=no_dedupe_depths,
                            cut_ids=no_dedupe_ids,
                            boundary_left=boundary_left,
                            boundary_right=boundary_right,
                            boundary_other_by_owner=boundary_other,
                            start_segment=no_dedupe_start,
                            initial_owner=no_dedupe_owner,
                        )
                    if no_dedupe_records != expected:
                        no_dedupe_mismatch_count += 1

        self.assertGreater(duplicate_frame_count, 0)
        self.assertGreater(no_dedupe_mismatch_count, 0)

    def test_affine_moving_ray_slab_covers_sample_events(self) -> None:
        sites = (
            Site4D(x=0.0, y=0.0, z=1.0, t=0.0, weight=0.0, rgba=(1.0, 0.0, 0.0, 1.0)),
            Site4D(x=1.0, y=0.0, z=3.0, t=1.0, weight=0.0, rgba=(0.0, 1.0, 0.0, 1.0)),
        )
        boundaries = make_boundaries_4d(sites)
        track = LinearRayTrack(
            origin_base=(0.0, 0.0, 0.0),
            origin_slope=(1.0, 0.0, 0.0),
            direction_base=(0.0, 0.0, 1.0),
            direction_slope=(0.0, 0.0, 0.0),
            max_origin_residual=0.0,
            max_direction_residual=0.0,
        )
        compiled_events, conservative = compiled_slab_event_set_for_track(
            boundaries=boundaries,
            track=track,
            t0=0.0,
            t1=1.0,
            near=1.0,
            far=3.0,
            invalid_epsilon=1.0e-7,
            residual_depth_padding=0.0,
        )
        self.assertEqual(conservative, 0)
        self.assertEqual(compiled_events, {0})
        for time_value in (0.0, 0.5, 1.0):
            origin, direction = track_ray_at(track, time_value)
            sample_events, invalid = event_set_for_ray(
                boundaries=boundaries,
                origin=origin,
                direction=direction,
                t=time_value,
                near=1.0,
                far=3.0,
                invalid_epsilon=1.0e-7,
            )
            self.assertEqual(invalid, 0)
            self.assertTrue(sample_events.issubset(compiled_events))

    def test_denominator_crossing_is_conservative(self) -> None:
        sites = (
            Site4D(x=0.0, y=0.0, z=1.0, t=0.0, weight=0.0, rgba=(1.0, 0.0, 0.0, 1.0)),
            Site4D(x=1.0, y=0.0, z=1.0, t=0.0, weight=0.0, rgba=(0.0, 1.0, 0.0, 1.0)),
        )
        boundaries = make_boundaries_4d(sites)
        track = LinearRayTrack(
            origin_base=(0.0, 0.0, 0.0),
            origin_slope=(0.0, 0.0, 0.0),
            direction_base=(-0.5, 0.0, 1.0),
            direction_slope=(1.0, 0.0, 0.0),
            max_origin_residual=0.0,
            max_direction_residual=0.0,
        )
        compiled_events, conservative = compiled_slab_event_set_for_track(
            boundaries=boundaries,
            track=track,
            t0=0.0,
            t1=1.0,
            near=0.0,
            far=4.0,
            invalid_epsilon=1.0e-7,
            residual_depth_padding=0.0,
        )
        self.assertEqual(compiled_events, {0})
        self.assertEqual(conservative, 1)

    def test_compiled_candidates_render_like_per_frame_realray_reference(self) -> None:
        sites = (
            Site4D(x=-0.15, y=0.0, z=0.9, t=0.0, weight=0.0, rgba=(1.0, 0.1, 0.0, 1.3)),
            Site4D(x=0.45, y=0.0, z=1.9, t=0.5, weight=0.0, rgba=(0.0, 0.8, 0.1, 0.9)),
            Site4D(x=1.0, y=0.0, z=2.8, t=1.0, weight=0.0, rgba=(0.1, 0.1, 1.0, 1.1)),
        )
        boundaries = make_boundaries_4d(sites)
        track = LinearRayTrack(
            origin_base=(-0.25, 0.0, 0.0),
            origin_slope=(0.7, 0.0, 0.0),
            direction_base=(0.1, 0.0, 1.0),
            direction_slope=(0.05, 0.0, 0.0),
            max_origin_residual=0.0,
            max_direction_residual=0.0,
        )
        compiled_events, conservative = compiled_slab_event_set_for_track(
            boundaries=boundaries,
            track=track,
            t0=0.0,
            t1=1.0,
            near=0.1,
            far=4.0,
            invalid_epsilon=1.0e-7,
            residual_depth_padding=0.0,
        )
        self.assertEqual(conservative, 0)
        self.assertGreaterEqual(len(compiled_events), 1)

        for time_value in (0.0, 0.25, 0.5, 0.75, 1.0):
            origin, direction = track_ray_at(track, time_value)
            direct = render_one_ray(
                sites=sites,
                boundaries=boundaries,
                origin=origin,
                direction=direction,
                t=time_value,
                near=0.1,
                far=4.0,
                invalid_epsilon=1.0e-7,
                transmittance_threshold=0.0,
            )
            compiled = _render_one_ray_from_compiled_candidates(
                sites=sites,
                track=track,
                candidate_ids=compiled_events,
                t=time_value,
                near=0.1,
                far=4.0,
                invalid_epsilon=1.0e-7,
                transmittance_threshold=0.0,
            )
            for direct_rgb, compiled_rgb in zip(direct[0], compiled[0], strict=True):
                self.assertAlmostEqual(direct_rgb, compiled_rgb, places=6)
            self.assertAlmostEqual(direct[1], compiled[1], places=6)
            self.assertAlmostEqual(direct[2], compiled[2], places=6)
            self.assertEqual(direct[3], compiled[3])

    def test_affine_slab_tape_materializes_metal_ready_csr(self) -> None:
        sites = (
            Site4D(x=-0.15, y=0.0, z=0.9, t=0.0, weight=0.0, rgba=(1.0, 0.1, 0.0, 1.3)),
            Site4D(x=0.45, y=0.0, z=1.9, t=0.5, weight=0.0, rgba=(0.0, 0.8, 0.1, 0.9)),
            Site4D(x=1.0, y=0.0, z=2.8, t=1.0, weight=0.0, rgba=(0.1, 0.1, 1.0, 1.1)),
        )
        boundaries = make_boundaries_4d(sites)
        track = LinearRayTrack(
            origin_base=(-0.25, 0.0, 0.0),
            origin_slope=(0.7, 0.0, 0.0),
            direction_base=(0.1, 0.0, 1.0),
            direction_slope=(0.05, 0.0, 0.0),
            max_origin_residual=0.0,
            max_direction_residual=0.0,
        )
        frame_count = 5
        frame_times = torch.linspace(0.0, 1.0, frame_count, dtype=torch.float32)
        rays = torch.tensor(
            [
                [[[*track_ray_at(track, float(t.item()))[0], *track_ray_at(track, float(t.item()))[1]]]]
                for t in frame_times
            ],
            dtype=torch.float32,
        )
        tape = build_gate4_affine_slab_tape(
            boundaries=boundaries,
            rays=rays,
            frame_indices=torch.arange(frame_count, dtype=torch.long),
            frame_count=frame_count,
            time_slabs=1,
            near=0.1,
            far=4.0,
            invalid_epsilon=1.0e-7,
            residual_depth_padding=0.0,
            layout="per-track",
            tile_h=1,
            tile_w=1,
            candidate_order="slab-mid-depth",
        )

        self.assertEqual(tape.row_index.dtype, torch.int32)
        self.assertEqual(tape.row_offsets.dtype, torch.int32)
        self.assertEqual(tape.candidate_ids.dtype, torch.int32)
        self.assertEqual(tape.candidate_depth_coeffs.shape[1], 4)
        self.assertEqual(tape.candidate_depth_num.shape[1], 2)
        self.assertEqual(tape.candidate_depth_den().dtype, torch.float16)
        self.assertEqual(tape.row_offsets.numel(), tape.row_count * tape.time_slab_count + 1)
        self.assertEqual(tape.row_offsets[-1].item(), tape.candidate_ids.numel())
        self.assertEqual(tape.ray_coeff.shape, (1, 12))
        self.assertEqual(tape.explicit_rays.shape, (frame_count, 6))
        self.assertEqual(tape.missing_sample_events, 0)

        row = int(tape.row_index[0].item())
        begin = int(tape.row_offsets[row].item())
        end = int(tape.row_offsets[row + 1].item())
        candidate_ids = set(int(value) for value in tape.candidate_ids[begin:end].tolist())
        self.assertGreaterEqual(len(candidate_ids), 1)
        for t in frame_times:
            time_value = float(t.item())
            direct = render_one_ray(
                sites=sites,
                boundaries=boundaries,
                origin=track_ray_at(track, time_value)[0],
                direction=track_ray_at(track, time_value)[1],
                t=time_value,
                near=0.1,
                far=4.0,
                invalid_epsilon=1.0e-7,
                transmittance_threshold=0.0,
            )
            compiled = _render_one_ray_from_compiled_candidates(
                sites=sites,
                track=track,
                candidate_ids=candidate_ids,
                t=time_value,
                near=0.1,
                far=4.0,
                invalid_epsilon=1.0e-7,
                transmittance_threshold=0.0,
            )
            for direct_rgb, compiled_rgb in zip(direct[0], compiled[0], strict=True):
                self.assertAlmostEqual(direct_rgb, compiled_rgb, places=6)
            self.assertAlmostEqual(direct[1], compiled[1], places=6)
            self.assertAlmostEqual(direct[2], compiled[2], places=6)

    def test_affine_slab_tape_builds_endpoint_records_matching_slow_owner_runs(self) -> None:
        sites = (
            Site4D(x=-0.15, y=0.0, z=0.9, t=0.0, weight=0.0, rgba=(1.0, 0.1, 0.0, 1.3)),
            Site4D(x=0.45, y=0.0, z=1.9, t=0.5, weight=0.0, rgba=(0.0, 0.8, 0.1, 0.9)),
            Site4D(x=1.0, y=0.0, z=2.8, t=1.0, weight=0.0, rgba=(0.1, 0.1, 1.0, 1.1)),
        )
        boundaries = make_boundaries_4d(sites)
        track = LinearRayTrack(
            origin_base=(-0.25, 0.0, 0.0),
            origin_slope=(0.7, 0.0, 0.0),
            direction_base=(0.1, 0.0, 1.0),
            direction_slope=(0.05, 0.0, 0.0),
            max_origin_residual=0.0,
            max_direction_residual=0.0,
        )
        frame_count = 5
        frame_times = torch.linspace(0.0, 1.0, frame_count, dtype=torch.float32)
        rays = torch.tensor(
            [
                [[[*track_ray_at(track, float(t.item()))[0], *track_ray_at(track, float(t.item()))[1]]]]
                for t in frame_times
            ],
            dtype=torch.float32,
        )
        frame_indices = torch.arange(frame_count, dtype=torch.long)
        tape = build_gate4_affine_slab_tape(
            boundaries=boundaries,
            rays=rays,
            frame_indices=frame_indices,
            frame_count=frame_count,
            time_slabs=1,
            near=0.1,
            far=4.0,
            invalid_epsilon=1.0e-7,
            residual_depth_padding=0.0,
            layout="per-track",
            tile_h=1,
            tile_w=1,
            candidate_order="slab-mid-depth",
        )
        skipped_validation_tape = build_gate4_affine_slab_tape(
            boundaries=boundaries,
            rays=rays,
            frame_indices=frame_indices,
            frame_count=frame_count,
            time_slabs=1,
            near=0.1,
            far=4.0,
            invalid_epsilon=1.0e-7,
            residual_depth_padding=0.0,
            layout="per-track",
            tile_h=1,
            tile_w=1,
            candidate_order="slab-mid-depth",
            sample_validation="skip",
        )
        self.assertEqual(tape.candidate_depth_order["sample_validation"], "full")
        self.assertTrue(tape.candidate_depth_order["missing_sample_events_authoritative"])
        self.assertEqual(skipped_validation_tape.candidate_depth_order["sample_validation"], "skip")
        self.assertFalse(skipped_validation_tape.candidate_depth_order["missing_sample_events_authoritative"])
        self.assertEqual(skipped_validation_tape.missing_sample_events, 0)
        self.assertEqual(skipped_validation_tape.candidate_replay_iterations, tape.candidate_replay_iterations)
        torch.testing.assert_close(skipped_validation_tape.row_offsets, tape.row_offsets)
        torch.testing.assert_close(skipped_validation_tape.candidate_ids, tape.candidate_ids)
        torch.testing.assert_close(skipped_validation_tape.candidate_depth_coeffs, tape.candidate_depth_coeffs)

        endpoint_sequences = build_gate4_endpoint_run_sequences(
            tape=tape,
            sites=sites,
            near=0.1,
            far=4.0,
            invalid_epsilon=1.0e-7,
        )
        skipped_validation_endpoint_sequences = build_gate4_endpoint_run_sequences(
            tape=skipped_validation_tape,
            sites=sites,
            near=0.1,
            far=4.0,
            invalid_epsilon=1.0e-7,
        )
        direct_delta = build_gate4_endpoint_delta_replace_tape(
            tape=skipped_validation_tape,
            sites=sites,
            near=0.1,
            far=4.0,
            invalid_epsilon=1.0e-7,
        )
        packed_delta = pack_endpoint_record_delta_replace_tape(endpoint_sequences, frame_count=frame_count)
        for attr in (
            "base_offsets_i32",
            "base_owner_i32",
            "base_left_i32",
            "base_right_i32",
            "track_change_offsets_i32",
            "change_frame_i32",
            "change_offsets_i32",
            "change_owner_i32",
            "change_left_i32",
            "change_right_i32",
        ):
            torch.testing.assert_close(getattr(direct_delta, attr), getattr(packed_delta, attr))
        slow_sequences, _sample_meta = _build_owner_run_sequences(
            sites=sites,
            boundaries=boundaries,
            rays=rays,
            frame_indices=frame_indices,
            frame_count=frame_count,
            near=0.1,
            far=4.0,
            invalid_epsilon=1.0e-7,
            transmittance_threshold=0.0,
            site_rgba=torch.tensor([site.rgba for site in sites], dtype=torch.float32),
        )

        self.assertEqual(skipped_validation_endpoint_sequences, endpoint_sequences)
        self.assertEqual(len(endpoint_sequences), 1)
        self.assertEqual(len(endpoint_sequences[0]), frame_count)
        self.assertEqual(len(endpoint_sequences), len(slow_sequences))
        for endpoint_frames, slow_frames in zip(endpoint_sequences, slow_sequences, strict=True):
            for endpoint_records, slow_records in zip(endpoint_frames, slow_frames, strict=True):
                self.assertGreaterEqual(len(endpoint_records), 1)
                self.assertEqual(endpoint_records[0].left_cut_id, GATE4_NEAR_CUT_ID)
                self.assertEqual(endpoint_records[-1].right_cut_id, GATE4_FAR_CUT_ID)
                for left, right in zip(endpoint_records[:-1], endpoint_records[1:], strict=True):
                    self.assertNotEqual(left.owner, right.owner)
                endpoint_rows = tuple(
                    (record.owner, record.left_cut_id, record.right_cut_id) for record in endpoint_records
                )
                slow_rows = tuple((record.owner, record.left_cut_id, record.right_cut_id) for record in slow_records)
                self.assertEqual(endpoint_rows, slow_rows)

        coeffs = build_gate4_boundary_depth_coefficients(tape=tape, boundaries=boundaries)
        self.assertEqual(coeffs.shape, (tape.track_count * len(boundaries), 4))
        expected_coeffs = torch.tensor(
            boundary_depth_coefficients(boundaries[0], track),
            dtype=torch.float32,
        )
        torch.testing.assert_close(coeffs[0], expected_coeffs)

    def test_affine_endpoint_records_match_slow_owner_runs_for_multitrack_slab_tape(self) -> None:
        sites = (
            Site4D(x=-0.4, y=0.0, z=0.7, t=0.0, weight=0.0, rgba=(1.0, 0.0, 0.0, 1.2)),
            Site4D(x=0.2, y=0.1, z=1.3, t=0.2, weight=0.02, rgba=(0.1, 0.9, 0.0, 0.8)),
            Site4D(x=0.8, y=-0.1, z=2.2, t=0.7, weight=0.01, rgba=(0.0, 0.2, 1.0, 1.1)),
            Site4D(x=-0.2, y=0.35, z=3.1, t=1.0, weight=0.03, rgba=(0.9, 0.9, 0.1, 0.7)),
            Site4D(x=0.6, y=-0.3, z=4.1, t=0.4, weight=0.0, rgba=(0.4, 0.1, 0.8, 0.9)),
        )
        boundaries = make_boundaries_4d(sites)
        tracks = (
            LinearRayTrack(
                origin_base=(-0.5, 0.0, 0.0),
                origin_slope=(0.9, 0.2, 0.0),
                direction_base=(0.05, 0.01, 1.0),
                direction_slope=(0.1, -0.02, 0.0),
                max_origin_residual=0.0,
                max_direction_residual=0.0,
            ),
            LinearRayTrack(
                origin_base=(0.3, 0.1, 0.0),
                origin_slope=(-0.4, -0.15, 0.0),
                direction_base=(-0.04, 0.0, 1.0),
                direction_slope=(0.03, 0.04, 0.0),
                max_origin_residual=0.0,
                max_direction_residual=0.0,
            ),
        )
        frame_count = 6
        frame_times = torch.linspace(0.0, 1.0, frame_count, dtype=torch.float32)
        rays = torch.tensor(
            [
                [
                    [
                        [*track_ray_at(tracks[0], float(t.item()))[0], *track_ray_at(tracks[0], float(t.item()))[1]],
                        [*track_ray_at(tracks[1], float(t.item()))[0], *track_ray_at(tracks[1], float(t.item()))[1]],
                    ]
                ]
                for t in frame_times
            ],
            dtype=torch.float32,
        )
        frame_indices = torch.arange(frame_count, dtype=torch.long)
        tape = build_gate4_affine_slab_tape(
            boundaries=boundaries,
            rays=rays,
            frame_indices=frame_indices,
            frame_count=frame_count,
            time_slabs=2,
            near=0.1,
            far=5.0,
            invalid_epsilon=1.0e-7,
            residual_depth_padding=0.01,
            layout="per-track",
            tile_h=1,
            tile_w=1,
            candidate_order="slab-mid-depth",
        )
        self.assertEqual(tape.missing_sample_events, 0)

        endpoint_sequences = build_gate4_endpoint_run_sequences(
            tape=tape,
            sites=sites,
            near=0.1,
            far=5.0,
            invalid_epsilon=1.0e-7,
        )
        slow_sequences, _sample_meta = _build_owner_run_sequences(
            sites=sites,
            boundaries=boundaries,
            rays=rays,
            frame_indices=frame_indices,
            frame_count=frame_count,
            near=0.1,
            far=5.0,
            invalid_epsilon=1.0e-7,
            transmittance_threshold=0.0,
            site_rgba=torch.tensor([site.rgba for site in sites], dtype=torch.float32),
        )

        self.assertEqual(len(endpoint_sequences), len(slow_sequences))
        for endpoint_frames, slow_frames in zip(endpoint_sequences, slow_sequences, strict=True):
            for endpoint_records, slow_records in zip(endpoint_frames, slow_frames, strict=True):
                endpoint_rows = tuple(
                    (record.owner, record.left_cut_id, record.right_cut_id) for record in endpoint_records
                )
                slow_rows = tuple((record.owner, record.left_cut_id, record.right_cut_id) for record in slow_records)
                self.assertEqual(endpoint_rows, slow_rows)


if __name__ == "__main__":
    unittest.main()
