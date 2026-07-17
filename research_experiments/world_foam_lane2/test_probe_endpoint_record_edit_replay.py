from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from probe_endpoint_record_edit_replay import (
    OP_DELETE,
    OP_INSERT,
    OP_REPLACE,
    _apply_ops,
    _edit_script,
    _track_boundary_coefficients,
    pack_endpoint_record_block_edit_tape,
    pack_endpoint_record_edit_tape,
)
from probe_endpoint_record_delta_replay import (
    build_delta_replace_chunk_change_offsets,
    build_delta_replace_chunk_owner_lists,
    pack_endpoint_record_delta_replace_tape,
)
from torch_world_foam_lane2_fused_slab import (
    RealRayReplayConfig,
    endpoint_record_delta_replace_coeff16_i16x3_framegroup16_mse_autograd,
    endpoint_record_delta_replace_coeff16_i16x3_framegroup16_materialized_mse_vjp_direct_atomic_rgb_only,
    endpoint_record_delta_replace_coeff16_i16x3_framegroup16_mse_vjp_direct_atomic_rgb_only,
    endpoint_record_delta_replace_coeff16_i16x3_framegroup16_ownerreduce_mse_vjp_direct_atomic_rgb_only,
    endpoint_record_delta_replace_coeff16_i16x3_mse_vjp_direct_atomic_rgb_only,
    endpoint_record_delta_replace_coeff16_packed_framegroup16_mse_vjp_direct_atomic_rgb_only,
    endpoint_record_delta_replace_coeff16_i16x4_framegroup16_mse_vjp_direct_atomic_rgb_only,
    endpoint_record_delta_replace_coeff16_i16x4_mse_vjp_direct_atomic_rgb_only,
    endpoint_record_delta_replace_coeff16_mse_vjp_direct_atomic_rgb_only,
    endpoint_record_delta_replace_rgba_depth_replay,
    endpoint_record_edit_block_coeff16_rgba_depth_replay,
    endpoint_record_edit_block_coeff16_i16_mse_vjp_direct_atomic_rgb_only,
    endpoint_record_edit_block_coeff16_i16x3_mse_vjp_direct_atomic_rgb_only,
    endpoint_record_edit_block_coeff16_mse_vjp_direct_atomic_rgb_only,
    endpoint_record_edit_block_coeff16_packed_mse_vjp_direct_atomic_rgb_only,
    endpoint_record_edit_block_coeff16_vjp_direct_atomic_rgb_only,
    endpoint_record_edit_block4_rgba_depth_replay,
    endpoint_record_edit_block_coeff_rgba_depth_replay,
    endpoint_record_edit_block_coeff_rgb_replay,
    endpoint_record_edit_block_coeff_mse_vjp_direct_atomic_rgb_only,
    endpoint_record_edit_block_coeff_vjp_direct_atomic_rgb_only,
    endpoint_record_edit_coeff16_mse_vjp_direct_atomic_rgb_only,
    endpoint_record_edit_mse_vjp_direct_atomic_rgb_only,
    endpoint_record_edit_rgba_depth_replay,
    endpoint_record_edit_vjp_direct_atomic_rgb_only,
)


def _pack_record(owner: int, left: int, right: int) -> int:
    def cut_code(cut: int) -> int:
        if cut == -1:
            return 0
        if cut == -2:
            return 1
        if cut < 0:
            raise ValueError("packed test record only supports -1, -2, or nonnegative cuts")
        return cut + 2

    return int(max(owner, 0) | (cut_code(left) << 8) | (cut_code(right) << 20))


def _record(owner: int, left: int, right: int) -> SimpleNamespace:
    return SimpleNamespace(owner=owner, left_cut_id=left, right_cut_id=right)


def _record_i16x3(*records: tuple[int, int, int]) -> torch.Tensor:
    return torch.tensor(records, device=torch.device("mps"), dtype=torch.int16).reshape(-1)


def _tape_i16x3(owner_i32: torch.Tensor, left_i32: torch.Tensor, right_i32: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (
            owner_i32.to(dtype=torch.int16),
            left_i32.to(dtype=torch.int16),
            right_i32.to(dtype=torch.int16),
        ),
        dim=1,
    ).reshape(-1)


def _tape_i16cols(owner_i32: torch.Tensor, left_i32: torch.Tensor, right_i32: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (
            owner_i32.to(dtype=torch.int16),
            left_i32.to(dtype=torch.int16),
            right_i32.to(dtype=torch.int16),
        ),
        dim=0,
    )


def _tape_i16x4(owner_i32: torch.Tensor, left_i32: torch.Tensor, right_i32: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (
            owner_i32.to(dtype=torch.int16),
            left_i32.to(dtype=torch.int16),
            right_i32.to(dtype=torch.int16),
            torch.zeros_like(owner_i32, dtype=torch.int16),
        ),
        dim=1,
    ).reshape(-1)


def _tape_packed_i32(owner_i32: torch.Tensor, left_i32: torch.Tensor, right_i32: torch.Tensor) -> torch.Tensor:
    return torch.tensor(
        [
            _pack_record(int(owner), int(left), int(right))
            for owner, left, right in zip(owner_i32.tolist(), left_i32.tolist(), right_i32.tolist(), strict=True)
        ],
        device=torch.device("mps"),
        dtype=torch.int32,
    )


class EndpointRecordEditReplayTests(unittest.TestCase):
    def test_edit_script_handles_insert_delete_and_replace(self) -> None:
        left = ((1, -1, 2), (3, 2, 4), (5, 4, -2))
        right = ((1, -1, 2), (7, 2, 6), (9, 6, 8), (5, 4, -2))

        ops = _edit_script(left, right)

        self.assertEqual(_apply_ops(left, ops), right)
        self.assertEqual([op[0] for op in ops], [OP_REPLACE, OP_INSERT])
        self.assertEqual([op[1] for op in ops], [1, 2])

    def test_pack_edit_tape_skips_unchanged_frames_and_records_ops(self) -> None:
        sequences = [
            [
                (_record(1, -1, 2), _record(3, 2, -2)),
                (_record(1, -1, 2), _record(3, 2, -2)),
                (_record(1, -1, 2), _record(4, 2, 5), _record(3, 5, -2)),
            ],
            [
                (_record(2, -1, 4), _record(6, 4, -2)),
                (_record(2, -1, -2),),
                (_record(2, -1, -2),),
            ],
        ]

        tape = pack_endpoint_record_edit_tape(sequences, frame_count=3)

        self.assertEqual(tape.base_offsets_i32.tolist(), [0, 2, 4])
        self.assertEqual(tape.base_owner_i32.tolist(), [1, 3, 2, 6])
        self.assertEqual(tape.track_change_offsets_i32.tolist(), [0, 1, 2])
        self.assertEqual(tape.change_frame_i32.tolist(), [2, 1])
        self.assertEqual(tape.op_offsets_i32.tolist(), [0, 2, 4])
        self.assertEqual(tape.op_type_i32.tolist(), [OP_REPLACE, OP_INSERT, OP_REPLACE, OP_DELETE])
        self.assertEqual(tape.op_pos_i32.tolist(), [1, 2, 0, 1])
        self.assertEqual(tape.op_owner_i32.tolist(), [4, 3, 2, -1])
        self.assertEqual(tape.op_left_i32.tolist(), [2, 5, -1, -1])
        self.assertEqual(tape.op_right_i32.tolist(), [5, -2, -2, -1])
        self.assertEqual(tape.changed_records, 4)

        block2 = pack_endpoint_record_block_edit_tape(sequences, frame_count=3, block_size=2)

        self.assertEqual(block2.block_size, 2)
        self.assertEqual(block2.block_count, 2)
        self.assertEqual(block2.anchor_offsets_i32.tolist(), [0, 2, 5, 7, 8])
        self.assertEqual(block2.anchor_owner_i32.tolist(), [1, 3, 1, 4, 3, 2, 6, 2])
        self.assertEqual(block2.track_block_change_offsets_i32.tolist(), [0, 0, 0, 0, 1, 1])
        self.assertEqual(block2.change_frame_i32.tolist(), [1])
        self.assertEqual(block2.op_offsets_i32.tolist(), [0, 2])
        self.assertEqual(block2.op_type_i32.tolist(), [OP_REPLACE, OP_DELETE])
        self.assertEqual(block2.op_pos_i32.tolist(), [0, 1])
        self.assertEqual(block2.op_owner_i32.tolist(), [2, -1])
        self.assertEqual(block2.op_left_i32.tolist(), [-1, -1])
        self.assertEqual(block2.op_right_i32.tolist(), [-2, -1])
        self.assertEqual(block2.changed_records, 1)

    def test_track_boundary_coefficients_fit_linear_depth(self) -> None:
        boundaries = (SimpleNamespace(nx=0.0, ny=0.0, nz=1.0, nt=0.0, b=-2.0),)
        track_rays = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.5, 0.0, 0.0, 1.0],
                ]
            ],
            dtype=torch.float32,
        )
        frame_t = torch.tensor([0.0, 1.0], dtype=torch.float32)

        coeffs = _track_boundary_coefficients(boundaries=boundaries, track_rays=track_rays, frame_t=frame_t)

        self.assertEqual(tuple(coeffs.shape), (1, 4))
        self.assertTrue(torch.allclose(coeffs[0], torch.tensor([2.0, -0.5, 1.0, 0.0]), atol=1.0e-6))

    def test_pack_block4_edit_tape_anchors_each_track_block(self) -> None:
        sequences = [
            [
                (_record(1, -1, 2), _record(3, 2, -2)),
                (_record(1, -1, 2), _record(3, 2, -2)),
                (_record(1, -1, 2), _record(4, 2, 5), _record(3, 5, -2)),
            ],
            [
                (_record(2, -1, 4), _record(6, 4, -2)),
                (_record(2, -1, -2),),
                (_record(2, -1, -2),),
            ],
        ]

        tape = pack_endpoint_record_block_edit_tape(sequences, frame_count=3, block_size=4)

        self.assertEqual(tape.block_size, 4)
        self.assertEqual(tape.block_count, 1)
        self.assertEqual(tape.anchor_offsets_i32.tolist(), [0, 2, 4])
        self.assertEqual(tape.anchor_owner_i32.tolist(), [1, 3, 2, 6])
        self.assertEqual(tape.track_block_change_offsets_i32.tolist(), [0, 1, 1, 2])
        self.assertEqual(tape.change_frame_i32.tolist(), [2, 1])
        self.assertEqual(tape.op_offsets_i32.tolist(), [0, 2, 4])
        self.assertEqual(tape.op_type_i32.tolist(), [OP_REPLACE, OP_INSERT, OP_REPLACE, OP_DELETE])
        self.assertEqual(tape.op_pos_i32.tolist(), [1, 2, 0, 1])
        self.assertEqual(tape.op_owner_i32.tolist(), [4, 3, 2, -1])
        self.assertEqual(tape.op_left_i32.tolist(), [2, 5, -1, -1])
        self.assertEqual(tape.op_right_i32.tolist(), [5, -2, -2, -1])
        self.assertEqual(tape.changed_records, 4)

    @unittest.skipUnless(torch.backends.mps.is_available(), "requires local MPS Metal backend")
    def test_block_coeff_rejects_cut_depths_outside_near_far_like_block4(self) -> None:
        device = torch.device("mps")
        boundary_f32 = torch.tensor([[0.0, 0.0, 1.0, 0.0, -2.0]], device=device, dtype=torch.float32)
        coeff_f32 = torch.tensor([[2.0, 0.0, 1.0, 0.0]], device=device, dtype=torch.float32)
        rays_f32 = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]], device=device, dtype=torch.float32)
        frame_t_f32 = torch.tensor([0.0], device=device, dtype=torch.float32)
        anchor_offsets_i32 = torch.tensor([0, 2], device=device, dtype=torch.int32)
        anchor_owner_i32 = torch.tensor([0, 1], device=device, dtype=torch.int32)
        anchor_left_i32 = torch.tensor([-1, 0], device=device, dtype=torch.int32)
        anchor_right_i32 = torch.tensor([0, -2], device=device, dtype=torch.int32)
        track_block_change_offsets_i32 = torch.tensor([0, 0], device=device, dtype=torch.int32)
        empty_i32 = torch.empty((0,), device=device, dtype=torch.int32)
        op_offsets_i32 = torch.tensor([0], device=device, dtype=torch.int32)
        site_rgba_f32 = torch.tensor(
            [[0.25, 0.50, 0.75, 1.0], [0.75, 0.25, 0.50, 1.0]],
            device=device,
            dtype=torch.float32,
        )
        config = RealRayReplayConfig(near=0.0, far=1.0, invalid_epsilon=1.0e-7, transmittance_threshold=1.0e-4)

        block4 = endpoint_record_edit_block4_rgba_depth_replay(
            boundary_f32,
            rays_f32,
            frame_t_f32,
            anchor_offsets_i32,
            anchor_owner_i32,
            anchor_left_i32,
            anchor_right_i32,
            track_block_change_offsets_i32,
            empty_i32,
            op_offsets_i32,
            empty_i32,
            empty_i32,
            empty_i32,
            empty_i32,
            empty_i32,
            site_rgba_f32,
            config,
            track_count=1,
            frame_count=1,
            block_size=4,
        )
        block_coeff = endpoint_record_edit_block_coeff_rgba_depth_replay(
            coeff_f32,
            frame_t_f32,
            anchor_offsets_i32,
            anchor_owner_i32,
            anchor_left_i32,
            anchor_right_i32,
            track_block_change_offsets_i32,
            empty_i32,
            op_offsets_i32,
            empty_i32,
            empty_i32,
            empty_i32,
            empty_i32,
            empty_i32,
            site_rgba_f32,
            config,
            track_count=1,
            frame_count=1,
            boundary_count=1,
            block_size=4,
        )
        torch.mps.synchronize()

        for coeff_part, block4_part in zip(block_coeff, block4, strict=True):
            self.assertTrue(torch.allclose(coeff_part.cpu(), block4_part.cpu(), atol=1.0e-6, rtol=0.0))
        self.assertTrue(torch.allclose(block_coeff[0].cpu(), torch.zeros((1, 1, 3)), atol=1.0e-6, rtol=0.0))

    @unittest.skipUnless(torch.backends.mps.is_available(), "requires local MPS Metal backend")
    def test_block_coeff16_matches_f32_replay_and_rgb_vjp_on_simple_row(self) -> None:
        device = torch.device("mps")
        coeff_f32 = torch.tensor([[2.0, -0.25, 1.0, 0.0]], device=device, dtype=torch.float32)
        coeff_f16 = coeff_f32.to(dtype=torch.float16)
        frame_t_f32 = torch.tensor([0.0, 1.0], device=device, dtype=torch.float32)
        anchor_offsets_i32 = torch.tensor([0, 1], device=device, dtype=torch.int32)
        anchor_owner_i32 = torch.tensor([0], device=device, dtype=torch.int32)
        anchor_left_i32 = torch.tensor([-1], device=device, dtype=torch.int32)
        anchor_right_i32 = torch.tensor([0], device=device, dtype=torch.int32)
        track_block_change_offsets_i32 = torch.tensor([0, 0], device=device, dtype=torch.int32)
        empty_i32 = torch.empty((0,), device=device, dtype=torch.int32)
        op_offsets_i32 = torch.tensor([0], device=device, dtype=torch.int32)
        site_rgba_f32 = torch.tensor([[0.25, 0.50, 0.75, 1.0]], device=device, dtype=torch.float32)
        grad_rgb_f32 = torch.tensor([[[0.1, 0.2, -0.3], [0.05, -0.1, 0.2]]], device=device, dtype=torch.float32)
        config = RealRayReplayConfig(near=0.0, far=3.0, invalid_epsilon=1.0e-7, transmittance_threshold=1.0e-4)

        f32 = endpoint_record_edit_block_coeff_rgba_depth_replay(
            coeff_f32,
            frame_t_f32,
            anchor_offsets_i32,
            anchor_owner_i32,
            anchor_left_i32,
            anchor_right_i32,
            track_block_change_offsets_i32,
            empty_i32,
            op_offsets_i32,
            empty_i32,
            empty_i32,
            empty_i32,
            empty_i32,
            empty_i32,
            site_rgba_f32,
            config,
            track_count=1,
            frame_count=2,
            boundary_count=1,
            block_size=4,
        )
        f16 = endpoint_record_edit_block_coeff16_rgba_depth_replay(
            coeff_f16,
            frame_t_f32,
            anchor_offsets_i32,
            anchor_owner_i32,
            anchor_left_i32,
            anchor_right_i32,
            track_block_change_offsets_i32,
            empty_i32,
            op_offsets_i32,
            empty_i32,
            empty_i32,
            empty_i32,
            empty_i32,
            empty_i32,
            site_rgba_f32,
            config,
            track_count=1,
            frame_count=2,
            boundary_count=1,
            block_size=4,
        )
        rgb_only = endpoint_record_edit_block_coeff_rgb_replay(
            coeff_f32,
            frame_t_f32,
            anchor_offsets_i32,
            anchor_owner_i32,
            anchor_left_i32,
            anchor_right_i32,
            track_block_change_offsets_i32,
            empty_i32,
            op_offsets_i32,
            empty_i32,
            empty_i32,
            empty_i32,
            empty_i32,
            empty_i32,
            site_rgba_f32,
            config,
            track_count=1,
            frame_count=2,
            boundary_count=1,
            block_size=4,
        )
        grad_f32 = endpoint_record_edit_block_coeff_vjp_direct_atomic_rgb_only(
            coeff_f32,
            frame_t_f32,
            anchor_offsets_i32,
            anchor_owner_i32,
            anchor_left_i32,
            anchor_right_i32,
            track_block_change_offsets_i32,
            empty_i32,
            op_offsets_i32,
            empty_i32,
            empty_i32,
            empty_i32,
            empty_i32,
            empty_i32,
            site_rgba_f32,
            grad_rgb_f32,
            config,
            track_count=1,
            frame_count=2,
            boundary_count=1,
            block_size=4,
        )
        grad_f16 = endpoint_record_edit_block_coeff16_vjp_direct_atomic_rgb_only(
            coeff_f16,
            frame_t_f32,
            anchor_offsets_i32,
            anchor_owner_i32,
            anchor_left_i32,
            anchor_right_i32,
            track_block_change_offsets_i32,
            empty_i32,
            op_offsets_i32,
            empty_i32,
            empty_i32,
            empty_i32,
            empty_i32,
            empty_i32,
            site_rgba_f32,
            grad_rgb_f32,
            config,
            track_count=1,
            frame_count=2,
            boundary_count=1,
            block_size=4,
        )
        torch.mps.synchronize()

        for f16_part, f32_part in zip(f16, f32, strict=True):
            self.assertTrue(torch.allclose(f16_part.cpu(), f32_part.cpu(), atol=2.0e-4, rtol=0.0))
        self.assertTrue(torch.allclose(rgb_only.cpu(), f32[0].cpu(), atol=1.0e-6, rtol=0.0))
        self.assertTrue(torch.allclose(grad_f16.cpu(), grad_f32.cpu(), atol=2.0e-4, rtol=0.0))

    @unittest.skipUnless(torch.backends.mps.is_available(), "requires local MPS Metal backend")
    def test_block_coeff16_packed_fused_mse_matches_unpacked_replace(self) -> None:
        device = torch.device("mps")
        coeff_f16 = torch.tensor([[2.0, -0.25, 1.0, 0.0]], device=device, dtype=torch.float16)
        frame_t_f32 = torch.tensor([0.0, 1.0], device=device, dtype=torch.float32)
        anchor_offsets_i32 = torch.tensor([0, 1], device=device, dtype=torch.int32)
        anchor_owner_i32 = torch.tensor([0], device=device, dtype=torch.int32)
        anchor_left_i32 = torch.tensor([-1], device=device, dtype=torch.int32)
        anchor_right_i32 = torch.tensor([0], device=device, dtype=torch.int32)
        track_block_change_offsets_i32 = torch.tensor([0, 1], device=device, dtype=torch.int32)
        change_frame_i32 = torch.tensor([1], device=device, dtype=torch.int32)
        op_offsets_i32 = torch.tensor([0, 1], device=device, dtype=torch.int32)
        op_type_i32 = torch.tensor([OP_REPLACE], device=device, dtype=torch.int32)
        op_pos_i32 = torch.tensor([0], device=device, dtype=torch.int32)
        op_owner_i32 = torch.tensor([0], device=device, dtype=torch.int32)
        op_left_i32 = torch.tensor([-1], device=device, dtype=torch.int32)
        op_right_i32 = torch.tensor([-2], device=device, dtype=torch.int32)
        anchor_record_i32 = torch.tensor([_pack_record(0, -1, 0)], device=device, dtype=torch.int32)
        op_record_i32 = torch.tensor([_pack_record(0, -1, -2)], device=device, dtype=torch.int32)
        site_rgba_f32 = torch.tensor([[0.25, 0.50, 0.75, 1.0]], device=device, dtype=torch.float32)
        target = torch.zeros((1, 2, 3), device=device, dtype=torch.float32)
        config = RealRayReplayConfig(near=0.0, far=3.0, invalid_epsilon=1.0e-7, transmittance_threshold=1.0e-4)

        unpacked_loss, unpacked_grad = endpoint_record_edit_block_coeff16_mse_vjp_direct_atomic_rgb_only(
            coeff_f16,
            frame_t_f32,
            anchor_offsets_i32,
            anchor_owner_i32,
            anchor_left_i32,
            anchor_right_i32,
            track_block_change_offsets_i32,
            change_frame_i32,
            op_offsets_i32,
            op_type_i32,
            op_pos_i32,
            op_owner_i32,
            op_left_i32,
            op_right_i32,
            site_rgba_f32,
            target,
            config,
            track_count=1,
            frame_count=2,
            boundary_count=1,
            block_size=4,
        )
        packed_loss, packed_grad = endpoint_record_edit_block_coeff16_packed_mse_vjp_direct_atomic_rgb_only(
            coeff_f16,
            frame_t_f32,
            anchor_offsets_i32,
            anchor_record_i32,
            track_block_change_offsets_i32,
            change_frame_i32,
            op_offsets_i32,
            op_type_i32,
            op_pos_i32,
            op_record_i32,
            site_rgba_f32,
            target,
            config,
            track_count=1,
            frame_count=2,
            boundary_count=1,
            block_size=4,
        )
        torch.mps.synchronize()

        self.assertTrue(torch.allclose(packed_loss.cpu(), unpacked_loss.cpu(), atol=1.0e-6, rtol=0.0))
        self.assertTrue(torch.allclose(packed_grad.cpu(), unpacked_grad.cpu(), atol=1.0e-6, rtol=0.0))
        self.assertGreater(float(packed_grad[:, 3].detach().abs().sum().cpu().item()), 0.0)

    @unittest.skipUnless(torch.backends.mps.is_available(), "requires local MPS Metal backend")
    def test_block_coeff16_i16_fused_mse_matches_unpacked_replace(self) -> None:
        device = torch.device("mps")
        coeff_f16 = torch.tensor([[2.0, -0.25, 1.0, 0.0]], device=device, dtype=torch.float16)
        frame_t_f32 = torch.tensor([0.0, 1.0], device=device, dtype=torch.float32)
        anchor_offsets_i32 = torch.tensor([0, 1], device=device, dtype=torch.int32)
        anchor_owner_i32 = torch.tensor([0], device=device, dtype=torch.int32)
        anchor_left_i32 = torch.tensor([-1], device=device, dtype=torch.int32)
        anchor_right_i32 = torch.tensor([0], device=device, dtype=torch.int32)
        track_block_change_offsets_i32 = torch.tensor([0, 1], device=device, dtype=torch.int32)
        change_frame_i32 = torch.tensor([1], device=device, dtype=torch.int32)
        op_offsets_i32 = torch.tensor([0, 1], device=device, dtype=torch.int32)
        op_type_i32 = torch.tensor([OP_REPLACE], device=device, dtype=torch.int32)
        op_pos_i32 = torch.tensor([0], device=device, dtype=torch.int32)
        op_owner_i32 = torch.tensor([0], device=device, dtype=torch.int32)
        op_left_i32 = torch.tensor([-1], device=device, dtype=torch.int32)
        op_right_i32 = torch.tensor([-2], device=device, dtype=torch.int32)
        site_rgba_f32 = torch.tensor([[0.25, 0.50, 0.75, 1.0]], device=device, dtype=torch.float32)
        target = torch.zeros((1, 2, 3), device=device, dtype=torch.float32)
        config = RealRayReplayConfig(near=0.0, far=3.0, invalid_epsilon=1.0e-7, transmittance_threshold=1.0e-4)

        unpacked_loss, unpacked_grad = endpoint_record_edit_block_coeff16_mse_vjp_direct_atomic_rgb_only(
            coeff_f16,
            frame_t_f32,
            anchor_offsets_i32,
            anchor_owner_i32,
            anchor_left_i32,
            anchor_right_i32,
            track_block_change_offsets_i32,
            change_frame_i32,
            op_offsets_i32,
            op_type_i32,
            op_pos_i32,
            op_owner_i32,
            op_left_i32,
            op_right_i32,
            site_rgba_f32,
            target,
            config,
            track_count=1,
            frame_count=2,
            boundary_count=1,
            block_size=4,
        )
        i16_loss, i16_grad = endpoint_record_edit_block_coeff16_i16_mse_vjp_direct_atomic_rgb_only(
            coeff_f16,
            frame_t_f32,
            anchor_offsets_i32,
            anchor_owner_i32.to(dtype=torch.int16),
            anchor_left_i32.to(dtype=torch.int16),
            anchor_right_i32.to(dtype=torch.int16),
            track_block_change_offsets_i32,
            change_frame_i32,
            op_offsets_i32,
            op_type_i32,
            op_pos_i32,
            op_owner_i32.to(dtype=torch.int16),
            op_left_i32.to(dtype=torch.int16),
            op_right_i32.to(dtype=torch.int16),
            site_rgba_f32,
            target,
            config,
            track_count=1,
            frame_count=2,
            boundary_count=1,
            block_size=4,
        )
        torch.mps.synchronize()

        self.assertTrue(torch.allclose(i16_loss.cpu(), unpacked_loss.cpu(), atol=1.0e-6, rtol=0.0))
        self.assertTrue(torch.allclose(i16_grad.cpu(), unpacked_grad.cpu(), atol=1.0e-6, rtol=0.0))
        self.assertGreater(float(i16_grad[:, 3].detach().abs().sum().cpu().item()), 0.0)

    @unittest.skipUnless(torch.backends.mps.is_available(), "requires local MPS Metal backend")
    def test_block_coeff16_i16x3_fused_mse_matches_unpacked_replace(self) -> None:
        device = torch.device("mps")
        coeff_f16 = torch.tensor([[2.0, -0.25, 1.0, 0.0]], device=device, dtype=torch.float16)
        frame_t_f32 = torch.tensor([0.0, 1.0], device=device, dtype=torch.float32)
        anchor_offsets_i32 = torch.tensor([0, 1], device=device, dtype=torch.int32)
        anchor_owner_i32 = torch.tensor([0], device=device, dtype=torch.int32)
        anchor_left_i32 = torch.tensor([-1], device=device, dtype=torch.int32)
        anchor_right_i32 = torch.tensor([0], device=device, dtype=torch.int32)
        track_block_change_offsets_i32 = torch.tensor([0, 1], device=device, dtype=torch.int32)
        change_frame_i32 = torch.tensor([1], device=device, dtype=torch.int32)
        op_offsets_i32 = torch.tensor([0, 1], device=device, dtype=torch.int32)
        op_type_i32 = torch.tensor([OP_REPLACE], device=device, dtype=torch.int32)
        op_pos_i32 = torch.tensor([0], device=device, dtype=torch.int32)
        op_owner_i32 = torch.tensor([0], device=device, dtype=torch.int32)
        op_left_i32 = torch.tensor([-1], device=device, dtype=torch.int32)
        op_right_i32 = torch.tensor([-2], device=device, dtype=torch.int32)
        anchor_record_i16 = _record_i16x3((0, -1, 0))
        op_record_i16 = _record_i16x3((0, -1, -2))
        site_rgba_f32 = torch.tensor([[0.25, 0.50, 0.75, 1.0]], device=device, dtype=torch.float32)
        target = torch.zeros((1, 2, 3), device=device, dtype=torch.float32)
        config = RealRayReplayConfig(near=0.0, far=3.0, invalid_epsilon=1.0e-7, transmittance_threshold=1.0e-4)

        unpacked_loss, unpacked_grad = endpoint_record_edit_block_coeff16_mse_vjp_direct_atomic_rgb_only(
            coeff_f16,
            frame_t_f32,
            anchor_offsets_i32,
            anchor_owner_i32,
            anchor_left_i32,
            anchor_right_i32,
            track_block_change_offsets_i32,
            change_frame_i32,
            op_offsets_i32,
            op_type_i32,
            op_pos_i32,
            op_owner_i32,
            op_left_i32,
            op_right_i32,
            site_rgba_f32,
            target,
            config,
            track_count=1,
            frame_count=2,
            boundary_count=1,
            block_size=4,
        )
        i16x3_loss, i16x3_grad = endpoint_record_edit_block_coeff16_i16x3_mse_vjp_direct_atomic_rgb_only(
            coeff_f16,
            frame_t_f32,
            anchor_offsets_i32,
            anchor_record_i16,
            track_block_change_offsets_i32,
            change_frame_i32,
            op_offsets_i32,
            op_type_i32,
            op_pos_i32,
            op_record_i16,
            site_rgba_f32,
            target,
            config,
            track_count=1,
            frame_count=2,
            boundary_count=1,
            block_size=4,
        )
        torch.mps.synchronize()

        self.assertTrue(torch.allclose(i16x3_loss.cpu(), unpacked_loss.cpu(), atol=1.0e-6, rtol=0.0))
        self.assertTrue(torch.allclose(i16x3_grad.cpu(), unpacked_grad.cpu(), atol=1.0e-6, rtol=0.0))
        self.assertGreater(float(i16x3_grad[:, 3].detach().abs().sum().cpu().item()), 0.0)

    @unittest.skipUnless(torch.backends.mps.is_available(), "requires local MPS Metal backend")
    def test_block_coeff_fused_mse_vjp_matches_render_loss_and_manual_vjp(self) -> None:
        device = torch.device("mps")
        coeff_f32 = torch.tensor([[2.0, -0.25, 1.0, 0.0]], device=device, dtype=torch.float32)
        coeff_f16 = coeff_f32.to(dtype=torch.float16)
        frame_t_f32 = torch.tensor([0.0, 1.0], device=device, dtype=torch.float32)
        anchor_offsets_i32 = torch.tensor([0, 1], device=device, dtype=torch.int32)
        anchor_owner_i32 = torch.tensor([0], device=device, dtype=torch.int32)
        anchor_left_i32 = torch.tensor([-1], device=device, dtype=torch.int32)
        anchor_right_i32 = torch.tensor([0], device=device, dtype=torch.int32)
        track_block_change_offsets_i32 = torch.tensor([0, 0], device=device, dtype=torch.int32)
        empty_i32 = torch.empty((0,), device=device, dtype=torch.int32)
        op_offsets_i32 = torch.tensor([0], device=device, dtype=torch.int32)
        site_rgba_f32 = torch.tensor([[0.25, 0.50, 0.75, 1.0]], device=device, dtype=torch.float32)
        config = RealRayReplayConfig(near=0.0, far=3.0, invalid_epsilon=1.0e-7, transmittance_threshold=1.0e-4)

        rgb = endpoint_record_edit_block_coeff_rgb_replay(
            coeff_f32,
            frame_t_f32,
            anchor_offsets_i32,
            anchor_owner_i32,
            anchor_left_i32,
            anchor_right_i32,
            track_block_change_offsets_i32,
            empty_i32,
            op_offsets_i32,
            empty_i32,
            empty_i32,
            empty_i32,
            empty_i32,
            empty_i32,
            site_rgba_f32,
            config,
            track_count=1,
            frame_count=2,
            boundary_count=1,
            block_size=4,
        )
        target = torch.zeros_like(rgb)
        manual_loss = (rgb - target).square().mean()
        grad_rgb = (2.0 / float(rgb.numel())) * (rgb - target)
        manual_grad = endpoint_record_edit_block_coeff_vjp_direct_atomic_rgb_only(
            coeff_f32,
            frame_t_f32,
            anchor_offsets_i32,
            anchor_owner_i32,
            anchor_left_i32,
            anchor_right_i32,
            track_block_change_offsets_i32,
            empty_i32,
            op_offsets_i32,
            empty_i32,
            empty_i32,
            empty_i32,
            empty_i32,
            empty_i32,
            site_rgba_f32,
            grad_rgb.contiguous(),
            config,
            track_count=1,
            frame_count=2,
            boundary_count=1,
            block_size=4,
        )
        fused_loss, fused_grad = endpoint_record_edit_block_coeff_mse_vjp_direct_atomic_rgb_only(
            coeff_f32,
            frame_t_f32,
            anchor_offsets_i32,
            anchor_owner_i32,
            anchor_left_i32,
            anchor_right_i32,
            track_block_change_offsets_i32,
            empty_i32,
            op_offsets_i32,
            empty_i32,
            empty_i32,
            empty_i32,
            empty_i32,
            empty_i32,
            site_rgba_f32,
            target,
            config,
            track_count=1,
            frame_count=2,
            boundary_count=1,
            block_size=4,
        )
        rgb16, _alpha16, _depth16 = endpoint_record_edit_block_coeff16_rgba_depth_replay(
            coeff_f16,
            frame_t_f32,
            anchor_offsets_i32,
            anchor_owner_i32,
            anchor_left_i32,
            anchor_right_i32,
            track_block_change_offsets_i32,
            empty_i32,
            op_offsets_i32,
            empty_i32,
            empty_i32,
            empty_i32,
            empty_i32,
            empty_i32,
            site_rgba_f32,
            config,
            track_count=1,
            frame_count=2,
            boundary_count=1,
            block_size=4,
        )
        manual_loss16 = (rgb16 - target).square().mean()
        grad_rgb16 = (2.0 / float(rgb16.numel())) * (rgb16 - target)
        manual_grad16 = endpoint_record_edit_block_coeff16_vjp_direct_atomic_rgb_only(
            coeff_f16,
            frame_t_f32,
            anchor_offsets_i32,
            anchor_owner_i32,
            anchor_left_i32,
            anchor_right_i32,
            track_block_change_offsets_i32,
            empty_i32,
            op_offsets_i32,
            empty_i32,
            empty_i32,
            empty_i32,
            empty_i32,
            empty_i32,
            site_rgba_f32,
            grad_rgb16.contiguous(),
            config,
            track_count=1,
            frame_count=2,
            boundary_count=1,
            block_size=4,
        )
        fused_loss16, fused_grad16 = endpoint_record_edit_block_coeff16_mse_vjp_direct_atomic_rgb_only(
            coeff_f16,
            frame_t_f32,
            anchor_offsets_i32,
            anchor_owner_i32,
            anchor_left_i32,
            anchor_right_i32,
            track_block_change_offsets_i32,
            empty_i32,
            op_offsets_i32,
            empty_i32,
            empty_i32,
            empty_i32,
            empty_i32,
            empty_i32,
            site_rgba_f32,
            target,
            config,
            track_count=1,
            frame_count=2,
            boundary_count=1,
            block_size=4,
        )
        zero_loss, zero_grad = endpoint_record_edit_block_coeff_mse_vjp_direct_atomic_rgb_only(
            coeff_f32,
            frame_t_f32,
            anchor_offsets_i32,
            anchor_owner_i32,
            anchor_left_i32,
            anchor_right_i32,
            track_block_change_offsets_i32,
            empty_i32,
            op_offsets_i32,
            empty_i32,
            empty_i32,
            empty_i32,
            empty_i32,
            empty_i32,
            site_rgba_f32,
            rgb.detach().contiguous(),
            config,
            track_count=1,
            frame_count=2,
            boundary_count=1,
            block_size=4,
        )
        torch.mps.synchronize()

        self.assertTrue(torch.allclose(fused_loss.cpu().reshape(()), manual_loss.cpu(), atol=1.0e-6, rtol=0.0))
        self.assertTrue(torch.allclose(fused_grad.cpu(), manual_grad.cpu(), atol=1.0e-6, rtol=0.0))
        self.assertTrue(torch.allclose(fused_loss16.cpu().reshape(()), manual_loss16.cpu(), atol=1.0e-6, rtol=0.0))
        self.assertTrue(torch.allclose(fused_grad16.cpu(), manual_grad16.cpu(), atol=1.0e-6, rtol=0.0))
        self.assertGreater(float(fused_grad[:, 3].detach().abs().sum().cpu().item()), 0.0)
        self.assertTrue(torch.allclose(zero_loss.cpu(), torch.zeros((1,)), atol=1.0e-7, rtol=0.0))
        self.assertTrue(torch.allclose(zero_grad.cpu(), torch.zeros((1, 4)), atol=1.0e-7, rtol=0.0))

    @unittest.skipUnless(torch.backends.mps.is_available(), "requires local MPS Metal backend")
    def test_edit_fused_mse_vjp_matches_render_loss_and_manual_vjp(self) -> None:
        device = torch.device("mps")
        sequences = [[(_record(0, -1, 0),), (_record(0, -1, 0),)]]
        tape = pack_endpoint_record_edit_tape(sequences, frame_count=2)
        boundaries = (SimpleNamespace(nx=0.0, ny=0.0, nz=1.0, nt=0.0, b=-2.0),)
        boundary_f32 = torch.tensor([[0.0, 0.0, 1.0, 0.0, -2.0]], device=device, dtype=torch.float32)
        rays_f32 = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.5, 0.0, 0.0, 1.0],
                ]
            ],
            device=device,
            dtype=torch.float32,
        )
        frame_t_f32 = torch.tensor([0.0, 1.0], device=device, dtype=torch.float32)
        coeff_f16 = _track_boundary_coefficients(
            boundaries=boundaries,
            track_rays=rays_f32.detach().cpu(),
            frame_t=frame_t_f32.detach().cpu(),
        ).to(device=device, dtype=torch.float16)
        site_rgba_f32 = torch.tensor([[0.25, 0.50, 0.75, 1.0]], device=device, dtype=torch.float32)
        config = RealRayReplayConfig(near=0.0, far=3.0, invalid_epsilon=1.0e-7, transmittance_threshold=1.0e-4)

        tape_args = (
            boundary_f32,
            rays_f32,
            frame_t_f32,
            tape.base_offsets_i32.to(device=device),
            tape.base_owner_i32.to(device=device),
            tape.base_left_i32.to(device=device),
            tape.base_right_i32.to(device=device),
            tape.track_change_offsets_i32.to(device=device),
            tape.change_frame_i32.to(device=device),
            tape.op_offsets_i32.to(device=device),
            tape.op_type_i32.to(device=device),
            tape.op_pos_i32.to(device=device),
            tape.op_owner_i32.to(device=device),
            tape.op_left_i32.to(device=device),
            tape.op_right_i32.to(device=device),
        )
        coeff_tape_args = (
            coeff_f16,
            frame_t_f32,
            tape.base_offsets_i32.to(device=device),
            tape.base_owner_i32.to(device=device),
            tape.base_left_i32.to(device=device),
            tape.base_right_i32.to(device=device),
            tape.track_change_offsets_i32.to(device=device),
            tape.change_frame_i32.to(device=device),
            tape.op_offsets_i32.to(device=device),
            tape.op_type_i32.to(device=device),
            tape.op_pos_i32.to(device=device),
            tape.op_owner_i32.to(device=device),
            tape.op_left_i32.to(device=device),
            tape.op_right_i32.to(device=device),
        )
        rgb, _alpha, _depth = endpoint_record_edit_rgba_depth_replay(
            *tape_args,
            site_rgba_f32,
            config,
            track_count=1,
            frame_count=2,
        )
        target = torch.zeros_like(rgb)
        manual_loss = (rgb - target).square().mean()
        grad_rgb = (2.0 / float(rgb.numel())) * (rgb - target)
        manual_grad = endpoint_record_edit_vjp_direct_atomic_rgb_only(
            *tape_args,
            site_rgba_f32,
            grad_rgb.contiguous(),
            config,
            track_count=1,
            frame_count=2,
        )
        fused_loss, fused_grad = endpoint_record_edit_mse_vjp_direct_atomic_rgb_only(
            *tape_args,
            site_rgba_f32,
            target,
            config,
            track_count=1,
            frame_count=2,
        )
        coeff16_loss, coeff16_grad = endpoint_record_edit_coeff16_mse_vjp_direct_atomic_rgb_only(
            *coeff_tape_args,
            site_rgba_f32,
            target,
            config,
            track_count=1,
            frame_count=2,
            boundary_count=1,
        )
        zero_loss, zero_grad = endpoint_record_edit_mse_vjp_direct_atomic_rgb_only(
            *tape_args,
            site_rgba_f32,
            rgb.detach().contiguous(),
            config,
            track_count=1,
            frame_count=2,
        )
        torch.mps.synchronize()

        self.assertTrue(torch.allclose(fused_loss.cpu().reshape(()), manual_loss.cpu(), atol=1.0e-6, rtol=0.0))
        self.assertTrue(torch.allclose(fused_grad.cpu(), manual_grad.cpu(), atol=1.0e-6, rtol=0.0))
        self.assertTrue(torch.allclose(coeff16_loss.cpu().reshape(()), fused_loss.cpu().reshape(()), atol=1.0e-6, rtol=0.0))
        self.assertTrue(torch.allclose(coeff16_grad.cpu(), fused_grad.cpu(), atol=1.0e-6, rtol=0.0))
        self.assertGreater(float(fused_grad[:, 3].detach().abs().sum().cpu().item()), 0.0)
        self.assertTrue(torch.allclose(zero_loss.cpu(), torch.zeros((1,)), atol=1.0e-7, rtol=0.0))
        self.assertTrue(torch.allclose(zero_grad.cpu(), torch.zeros((1, 4)), atol=1.0e-7, rtol=0.0))

    @unittest.skipUnless(torch.backends.mps.is_available(), "requires local MPS Metal backend")
    def test_delta_replace_coeff16_fused_mse_matches_raw_edit_on_changed_row(self) -> None:
        device = torch.device("mps")
        sequences = [
            [
                (_record(0, -1, 0), _record(1, 0, -2)),
                (_record(1, -1, 0), _record(0, 0, -2)),
            ]
        ]
        edit_tape = pack_endpoint_record_edit_tape(sequences, frame_count=2)
        delta_tape = pack_endpoint_record_delta_replace_tape(sequences, frame_count=2)
        boundaries = (SimpleNamespace(nx=0.0, ny=0.0, nz=1.0, nt=0.0, b=-2.0),)
        boundary_f32 = torch.tensor([[0.0, 0.0, 1.0, 0.0, -2.0]], device=device, dtype=torch.float32)
        rays_f32 = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.5, 0.0, 0.0, 1.0],
                ]
            ],
            device=device,
            dtype=torch.float32,
        )
        frame_t_f32 = torch.tensor([0.0, 1.0], device=device, dtype=torch.float32)
        coeff_f16 = _track_boundary_coefficients(
            boundaries=boundaries,
            track_rays=rays_f32.detach().cpu(),
            frame_t=frame_t_f32.detach().cpu(),
        ).to(device=device, dtype=torch.float16)
        site_rgba_f32 = torch.tensor(
            [[0.25, 0.50, 0.75, 1.0], [0.75, 0.25, 0.50, 1.5]],
            device=device,
            dtype=torch.float32,
        )
        config = RealRayReplayConfig(near=0.0, far=3.0, invalid_epsilon=1.0e-7, transmittance_threshold=1.0e-4)
        edit_args = (
            coeff_f16,
            frame_t_f32,
            edit_tape.base_offsets_i32.to(device=device),
            edit_tape.base_owner_i32.to(device=device),
            edit_tape.base_left_i32.to(device=device),
            edit_tape.base_right_i32.to(device=device),
            edit_tape.track_change_offsets_i32.to(device=device),
            edit_tape.change_frame_i32.to(device=device),
            edit_tape.op_offsets_i32.to(device=device),
            edit_tape.op_type_i32.to(device=device),
            edit_tape.op_pos_i32.to(device=device),
            edit_tape.op_owner_i32.to(device=device),
            edit_tape.op_left_i32.to(device=device),
            edit_tape.op_right_i32.to(device=device),
        )
        delta_args = (
            coeff_f16,
            frame_t_f32,
            delta_tape.base_offsets_i32.to(device=device),
            delta_tape.base_owner_i32.to(device=device),
            delta_tape.base_left_i32.to(device=device),
            delta_tape.base_right_i32.to(device=device),
            delta_tape.track_change_offsets_i32.to(device=device),
            delta_tape.change_frame_i32.to(device=device),
            delta_tape.change_offsets_i32.to(device=device),
            delta_tape.change_owner_i32.to(device=device),
            delta_tape.change_left_i32.to(device=device),
            delta_tape.change_right_i32.to(device=device),
        )
        delta_forward, _alpha, _depth = endpoint_record_delta_replace_rgba_depth_replay(
            boundary_f32,
            rays_f32,
            frame_t_f32,
            delta_tape.base_offsets_i32.to(device=device),
            delta_tape.base_owner_i32.to(device=device),
            delta_tape.base_left_i32.to(device=device),
            delta_tape.base_right_i32.to(device=device),
            delta_tape.track_change_offsets_i32.to(device=device),
            delta_tape.change_frame_i32.to(device=device),
            delta_tape.change_offsets_i32.to(device=device),
            delta_tape.change_owner_i32.to(device=device),
            delta_tape.change_left_i32.to(device=device),
            delta_tape.change_right_i32.to(device=device),
            site_rgba_f32,
            config,
            track_count=1,
            frame_count=2,
        )
        target = torch.zeros_like(delta_forward)
        edit_loss, edit_grad = endpoint_record_edit_coeff16_mse_vjp_direct_atomic_rgb_only(
            *edit_args,
            site_rgba_f32,
            target,
            config,
            track_count=1,
            frame_count=2,
            boundary_count=1,
        )
        delta_loss, delta_grad = endpoint_record_delta_replace_coeff16_mse_vjp_direct_atomic_rgb_only(
            *delta_args,
            site_rgba_f32,
            target,
            config,
            track_count=1,
            frame_count=2,
            boundary_count=1,
        )
        delta_i16x3_loss, delta_i16x3_grad = endpoint_record_delta_replace_coeff16_i16x3_mse_vjp_direct_atomic_rgb_only(
            coeff_f16,
            frame_t_f32,
            delta_tape.base_offsets_i32.to(device=device),
            _tape_i16x3(delta_tape.base_owner_i32, delta_tape.base_left_i32, delta_tape.base_right_i32).to(
                device=device
            ),
            delta_tape.track_change_offsets_i32.to(device=device),
            delta_tape.change_frame_i32.to(device=device),
            delta_tape.change_offsets_i32.to(device=device),
            _tape_i16x3(delta_tape.change_owner_i32, delta_tape.change_left_i32, delta_tape.change_right_i32).to(
                device=device
            ),
            site_rgba_f32,
            target,
            config,
            track_count=1,
            frame_count=2,
            boundary_count=1,
        )
        chunk_offsets_2 = build_delta_replace_chunk_change_offsets(delta_tape, frame_count=2)
        (
            delta_i16x3_framegroup_loss,
            delta_i16x3_framegroup_grad,
        ) = endpoint_record_delta_replace_coeff16_i16x3_framegroup16_mse_vjp_direct_atomic_rgb_only(
            coeff_f16,
            frame_t_f32,
            delta_tape.base_offsets_i32.to(device=device),
            _tape_i16x3(delta_tape.base_owner_i32, delta_tape.base_left_i32, delta_tape.base_right_i32).to(
                device=device
            ),
            delta_tape.track_change_offsets_i32.to(device=device),
            chunk_offsets_2.to(device=device, dtype=torch.int16),
            delta_tape.change_frame_i32.to(device=device),
            delta_tape.change_offsets_i32.to(device=device),
            _tape_i16x3(delta_tape.change_owner_i32, delta_tape.change_left_i32, delta_tape.change_right_i32).to(
                device=device
            ),
            site_rgba_f32,
            target,
            config,
            track_count=1,
            frame_count=2,
            boundary_count=1,
        )
        site_rgba_autograd = site_rgba_f32.detach().clone().requires_grad_(True)
        delta_i16x3_framegroup_autograd_loss = (
            endpoint_record_delta_replace_coeff16_i16x3_framegroup16_mse_autograd(
                coeff_f16,
                frame_t_f32,
                delta_tape.base_offsets_i32.to(device=device),
                _tape_i16x3(delta_tape.base_owner_i32, delta_tape.base_left_i32, delta_tape.base_right_i32).to(
                    device=device
                ),
                delta_tape.track_change_offsets_i32.to(device=device),
                chunk_offsets_2.to(device=device, dtype=torch.int16),
                delta_tape.change_frame_i32.to(device=device),
                delta_tape.change_offsets_i32.to(device=device),
                _tape_i16x3(delta_tape.change_owner_i32, delta_tape.change_left_i32, delta_tape.change_right_i32).to(
                    device=device
                ),
                site_rgba_autograd,
                target,
                config,
                track_count=1,
                frame_count=2,
                boundary_count=1,
            )
        )
        delta_i16x3_framegroup_autograd_loss.backward()
        delta_i16x4_loss, delta_i16x4_grad = endpoint_record_delta_replace_coeff16_i16x4_mse_vjp_direct_atomic_rgb_only(
            coeff_f16,
            frame_t_f32,
            delta_tape.base_offsets_i32.to(device=device),
            _tape_i16x4(delta_tape.base_owner_i32, delta_tape.base_left_i32, delta_tape.base_right_i32).to(
                device=device
            ),
            delta_tape.track_change_offsets_i32.to(device=device),
            delta_tape.change_frame_i32.to(device=device),
            delta_tape.change_offsets_i32.to(device=device),
            _tape_i16x4(delta_tape.change_owner_i32, delta_tape.change_left_i32, delta_tape.change_right_i32).to(
                device=device
            ),
            site_rgba_f32,
            target,
            config,
            track_count=1,
            frame_count=2,
            boundary_count=1,
        )
        (
            delta_i16x4_framegroup_loss,
            delta_i16x4_framegroup_grad,
        ) = endpoint_record_delta_replace_coeff16_i16x4_framegroup16_mse_vjp_direct_atomic_rgb_only(
            coeff_f16,
            frame_t_f32,
            delta_tape.base_offsets_i32.to(device=device),
            _tape_i16x4(delta_tape.base_owner_i32, delta_tape.base_left_i32, delta_tape.base_right_i32).to(
                device=device
            ),
            delta_tape.track_change_offsets_i32.to(device=device),
            build_delta_replace_chunk_change_offsets(delta_tape, frame_count=2).to(device=device, dtype=torch.int16),
            delta_tape.change_frame_i32.to(device=device),
            delta_tape.change_offsets_i32.to(device=device),
            _tape_i16x4(delta_tape.change_owner_i32, delta_tape.change_left_i32, delta_tape.change_right_i32).to(
                device=device
            ),
            site_rgba_f32,
            target,
            config,
            track_count=1,
            frame_count=2,
            boundary_count=1,
        )
        torch.mps.synchronize()

        self.assertTrue(torch.allclose(delta_loss.cpu().reshape(()), edit_loss.cpu().reshape(()), atol=1.0e-6, rtol=0.0))
        self.assertTrue(torch.allclose(delta_grad.cpu(), edit_grad.cpu(), atol=1.0e-6, rtol=0.0))
        self.assertTrue(
            torch.allclose(delta_i16x3_loss.cpu().reshape(()), delta_loss.cpu().reshape(()), atol=1.0e-6, rtol=0.0)
        )
        self.assertTrue(torch.allclose(delta_i16x3_grad.cpu(), delta_grad.cpu(), atol=1.0e-6, rtol=0.0))
        self.assertTrue(
            torch.allclose(
                delta_i16x3_framegroup_loss.cpu().reshape(()),
                delta_loss.cpu().reshape(()),
                atol=1.0e-6,
                rtol=0.0,
            )
        )
        self.assertTrue(torch.allclose(delta_i16x3_framegroup_grad.cpu(), delta_grad.cpu(), atol=1.0e-6, rtol=0.0))
        self.assertTrue(
            torch.allclose(
                delta_i16x3_framegroup_autograd_loss.detach().cpu(),
                delta_i16x3_framegroup_loss.cpu().reshape(()),
                atol=1.0e-6,
                rtol=0.0,
            )
        )
        self.assertTrue(
            torch.allclose(site_rgba_autograd.grad.detach().cpu(), delta_i16x3_framegroup_grad.cpu(), atol=1.0e-6, rtol=0.0)
        )
        self.assertTrue(
            torch.allclose(delta_i16x4_loss.cpu().reshape(()), delta_loss.cpu().reshape(()), atol=1.0e-6, rtol=0.0)
        )
        self.assertTrue(torch.allclose(delta_i16x4_grad.cpu(), delta_grad.cpu(), atol=1.0e-6, rtol=0.0))
        self.assertTrue(
            torch.allclose(
                delta_i16x4_framegroup_loss.cpu().reshape(()),
                delta_loss.cpu().reshape(()),
                atol=1.0e-6,
                rtol=0.0,
            )
        )
        self.assertTrue(torch.allclose(delta_i16x4_framegroup_grad.cpu(), delta_grad.cpu(), atol=1.0e-6, rtol=0.0))
        self.assertGreater(float(delta_grad[:, 3].detach().abs().sum().cpu().item()), 0.0)

    def _assert_delta_replace_framegroup_matches_scalar(
        self,
        sequences: list[list[tuple[SimpleNamespace, ...]]],
        *,
        frame_count: int,
        boundaries: tuple[SimpleNamespace, ...],
        site_rgba_rows: list[list[float]],
        atol: float = 3.0e-6,
    ) -> None:
        device = torch.device("mps")
        track_count = len(sequences)
        boundary_count = len(boundaries)
        delta_tape = pack_endpoint_record_delta_replace_tape(sequences, frame_count=frame_count)
        chunk_offsets = build_delta_replace_chunk_change_offsets(delta_tape, frame_count=frame_count)
        owner_offsets, owner_ids = build_delta_replace_chunk_owner_lists(
            delta_tape,
            frame_count=frame_count,
            site_count=len(site_rgba_rows),
        )

        frame_t_f32 = torch.linspace(0.0, 1.0, frame_count, device=device, dtype=torch.float32)
        rays_f32 = torch.zeros((track_count, frame_count, 6), device=device, dtype=torch.float32)
        for track_id in range(track_count):
            rays_f32[track_id, :, 0] = 0.03 * float(track_id)
            rays_f32[track_id, :, 2] = torch.linspace(
                0.0 + 0.05 * float(track_id),
                0.45 + 0.05 * float(track_id),
                frame_count,
                device=device,
            )
            rays_f32[track_id, :, 5] = 1.0
        coeff_f16 = _track_boundary_coefficients(
            boundaries=boundaries,
            track_rays=rays_f32.detach().cpu(),
            frame_t=frame_t_f32.detach().cpu(),
        ).to(device=device, dtype=torch.float16)
        site_rgba_f32 = torch.tensor(site_rgba_rows, device=device, dtype=torch.float32)
        target = torch.linspace(
            0.05,
            0.95,
            track_count * frame_count * 3,
            device=device,
            dtype=torch.float32,
        ).reshape(track_count, frame_count, 3)
        config = RealRayReplayConfig(near=0.0, far=3.5, invalid_epsilon=1.0e-7, transmittance_threshold=1.0e-4)

        scalar_loss, scalar_grad = endpoint_record_delta_replace_coeff16_i16x3_mse_vjp_direct_atomic_rgb_only(
            coeff_f16,
            frame_t_f32,
            delta_tape.base_offsets_i32.to(device=device),
            _tape_i16x3(delta_tape.base_owner_i32, delta_tape.base_left_i32, delta_tape.base_right_i32).to(
                device=device
            ),
            delta_tape.track_change_offsets_i32.to(device=device),
            delta_tape.change_frame_i32.to(device=device),
            delta_tape.change_offsets_i32.to(device=device),
            _tape_i16x3(delta_tape.change_owner_i32, delta_tape.change_left_i32, delta_tape.change_right_i32).to(
                device=device
            ),
            site_rgba_f32,
            target,
            config,
            track_count=track_count,
            frame_count=frame_count,
            boundary_count=boundary_count,
        )
        framegroup_loss, framegroup_grad = endpoint_record_delta_replace_coeff16_i16x3_framegroup16_mse_vjp_direct_atomic_rgb_only(
            coeff_f16,
            frame_t_f32,
            delta_tape.base_offsets_i32.to(device=device),
            _tape_i16x3(delta_tape.base_owner_i32, delta_tape.base_left_i32, delta_tape.base_right_i32).to(
                device=device
            ),
            delta_tape.track_change_offsets_i32.to(device=device),
            chunk_offsets.to(device=device, dtype=torch.int16),
            delta_tape.change_frame_i32.to(device=device),
            delta_tape.change_offsets_i32.to(device=device),
            _tape_i16x3(delta_tape.change_owner_i32, delta_tape.change_left_i32, delta_tape.change_right_i32).to(
                device=device
            ),
            site_rgba_f32,
            target,
            config,
            track_count=track_count,
            frame_count=frame_count,
            boundary_count=boundary_count,
        )
        ownerreduce_loss, ownerreduce_grad = (
            endpoint_record_delta_replace_coeff16_i16x3_framegroup16_ownerreduce_mse_vjp_direct_atomic_rgb_only(
                coeff_f16,
                frame_t_f32,
                delta_tape.base_offsets_i32.to(device=device),
                _tape_i16x3(delta_tape.base_owner_i32, delta_tape.base_left_i32, delta_tape.base_right_i32).to(
                    device=device
                ),
                delta_tape.track_change_offsets_i32.to(device=device),
                chunk_offsets.to(device=device, dtype=torch.int16),
                owner_offsets.to(device=device, dtype=torch.int32),
                owner_ids.to(device=device, dtype=torch.int16),
                delta_tape.change_frame_i32.to(device=device),
                delta_tape.change_offsets_i32.to(device=device),
                _tape_i16x3(
                    delta_tape.change_owner_i32,
                    delta_tape.change_left_i32,
                    delta_tape.change_right_i32,
                ).to(device=device),
                site_rgba_f32,
                target,
                config,
                track_count=track_count,
                frame_count=frame_count,
                boundary_count=boundary_count,
            )
        )
        i16cols_op = torch.ops.world_foam_lane2_fused_slab_v0.endpoint_record_delta_replace_coeff16_i16cols_framegroup16_mse_vjp_direct_atomic_rgb_only
        i16cols_framegroup_loss, i16cols_framegroup_grad = i16cols_op(
            coeff_f16,
            frame_t_f32,
            delta_tape.base_offsets_i32.to(device=device),
            _tape_i16cols(delta_tape.base_owner_i32, delta_tape.base_left_i32, delta_tape.base_right_i32).to(
                device=device
            ),
            delta_tape.track_change_offsets_i32.to(device=device),
            chunk_offsets.to(device=device, dtype=torch.int16),
            delta_tape.change_frame_i32.to(device=device),
            delta_tape.change_offsets_i32.to(device=device),
            _tape_i16cols(delta_tape.change_owner_i32, delta_tape.change_left_i32, delta_tape.change_right_i32).to(
                device=device
            ),
            site_rgba_f32,
            target,
            torch.tensor(
                [
                    boundary_count,
                    track_count,
                    frame_count,
                    int(site_rgba_f32.shape[0]),
                    int(delta_tape.base_owner_i32.numel()),
                    int(delta_tape.change_frame_i32.numel()),
                    int(delta_tape.change_owner_i32.numel()),
                ],
                device=device,
                dtype=torch.int32,
            ),
            torch.tensor(
                [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
                device=device,
                dtype=torch.float32,
            ),
        )
        framegroup64_op = torch.ops.world_foam_lane2_fused_slab_v0.endpoint_record_delta_replace_coeff16_i16x3_framegroup64_mse_vjp_direct_atomic_rgb_only
        framegroup64_loss, framegroup64_grad = framegroup64_op(
            coeff_f16,
            frame_t_f32,
            delta_tape.base_offsets_i32.to(device=device),
            _tape_i16x3(delta_tape.base_owner_i32, delta_tape.base_left_i32, delta_tape.base_right_i32).to(
                device=device
            ),
            delta_tape.track_change_offsets_i32.to(device=device),
            build_delta_replace_chunk_change_offsets(delta_tape, frame_count=frame_count, chunk_size=64).to(
                device=device,
                dtype=torch.int16,
            ),
            delta_tape.change_frame_i32.to(device=device),
            delta_tape.change_offsets_i32.to(device=device),
            _tape_i16x3(delta_tape.change_owner_i32, delta_tape.change_left_i32, delta_tape.change_right_i32).to(
                device=device
            ),
            site_rgba_f32,
            target,
            torch.tensor(
                [
                    boundary_count,
                    track_count,
                    frame_count,
                    int(site_rgba_f32.shape[0]),
                    int(delta_tape.base_owner_i32.numel()),
                    int(delta_tape.change_frame_i32.numel()),
                    int(delta_tape.change_owner_i32.numel()),
                ],
                device=device,
                dtype=torch.int32,
            ),
            torch.tensor(
                [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
                device=device,
                dtype=torch.float32,
            ),
        )
        packed_framegroup_loss, packed_framegroup_grad = (
            endpoint_record_delta_replace_coeff16_packed_framegroup16_mse_vjp_direct_atomic_rgb_only(
                coeff_f16,
                frame_t_f32,
                delta_tape.base_offsets_i32.to(device=device),
                _tape_packed_i32(delta_tape.base_owner_i32, delta_tape.base_left_i32, delta_tape.base_right_i32),
                delta_tape.track_change_offsets_i32.to(device=device),
                chunk_offsets.to(device=device, dtype=torch.int16),
                delta_tape.change_frame_i32.to(device=device),
                delta_tape.change_offsets_i32.to(device=device),
                _tape_packed_i32(
                    delta_tape.change_owner_i32,
                    delta_tape.change_left_i32,
                    delta_tape.change_right_i32,
                ),
                site_rgba_f32,
                target,
                config,
                track_count=track_count,
                frame_count=frame_count,
                boundary_count=boundary_count,
            )
        )
        materialized_loss, materialized_grad = (
            endpoint_record_delta_replace_coeff16_i16x3_framegroup16_materialized_mse_vjp_direct_atomic_rgb_only(
                coeff_f16,
                frame_t_f32,
                delta_tape.base_offsets_i32.to(device=device),
                _tape_i16x3(delta_tape.base_owner_i32, delta_tape.base_left_i32, delta_tape.base_right_i32).to(
                    device=device
                ),
                delta_tape.track_change_offsets_i32.to(device=device),
                build_delta_replace_chunk_change_offsets(delta_tape, frame_count=frame_count, chunk_size=16).to(
                    device=device,
                    dtype=torch.int16,
                ),
                delta_tape.change_frame_i32.to(device=device),
                delta_tape.change_offsets_i32.to(device=device),
                _tape_i16x3(delta_tape.change_owner_i32, delta_tape.change_left_i32, delta_tape.change_right_i32).to(
                    device=device
                ),
                site_rgba_f32,
                target,
                config,
                track_count=track_count,
                frame_count=frame_count,
                boundary_count=boundary_count,
            )
        )
        i16x4_framegroup_loss, i16x4_framegroup_grad = (
            endpoint_record_delta_replace_coeff16_i16x4_framegroup16_mse_vjp_direct_atomic_rgb_only(
                coeff_f16,
                frame_t_f32,
                delta_tape.base_offsets_i32.to(device=device),
                _tape_i16x4(delta_tape.base_owner_i32, delta_tape.base_left_i32, delta_tape.base_right_i32).to(
                    device=device
                ),
                delta_tape.track_change_offsets_i32.to(device=device),
                chunk_offsets.to(device=device, dtype=torch.int16),
                delta_tape.change_frame_i32.to(device=device),
                delta_tape.change_offsets_i32.to(device=device),
                _tape_i16x4(delta_tape.change_owner_i32, delta_tape.change_left_i32, delta_tape.change_right_i32).to(
                    device=device
                ),
                site_rgba_f32,
                target,
                config,
                track_count=track_count,
                frame_count=frame_count,
                boundary_count=boundary_count,
            )
        )
        torch.mps.synchronize()

        scalar_loss_cpu = scalar_loss.cpu().reshape(())
        framegroup_loss_cpu = framegroup_loss.cpu().reshape(())
        ownerreduce_loss_cpu = ownerreduce_loss.cpu().reshape(())
        i16cols_framegroup_loss_cpu = i16cols_framegroup_loss.cpu().reshape(())
        framegroup64_loss_cpu = framegroup64_loss.cpu().reshape(())
        packed_framegroup_loss_cpu = packed_framegroup_loss.cpu().reshape(())
        materialized_loss_cpu = materialized_loss.cpu().reshape(())
        i16x4_framegroup_loss_cpu = i16x4_framegroup_loss.cpu().reshape(())
        scalar_grad_cpu = scalar_grad.cpu()
        framegroup_grad_cpu = framegroup_grad.cpu()
        ownerreduce_grad_cpu = ownerreduce_grad.cpu()
        i16cols_framegroup_grad_cpu = i16cols_framegroup_grad.cpu()
        framegroup64_grad_cpu = framegroup64_grad.cpu()
        packed_framegroup_grad_cpu = packed_framegroup_grad.cpu()
        materialized_grad_cpu = materialized_grad.cpu()
        i16x4_framegroup_grad_cpu = i16x4_framegroup_grad.cpu()
        loss_diff = float((framegroup_loss_cpu - scalar_loss_cpu).abs().item())
        grad_diff = float((framegroup_grad_cpu - scalar_grad_cpu).abs().max().item())
        ownerreduce_loss_diff = float((ownerreduce_loss_cpu - scalar_loss_cpu).abs().item())
        ownerreduce_grad_diff = float((ownerreduce_grad_cpu - scalar_grad_cpu).abs().max().item())
        i16cols_loss_diff = float((i16cols_framegroup_loss_cpu - scalar_loss_cpu).abs().item())
        i16cols_grad_diff = float((i16cols_framegroup_grad_cpu - scalar_grad_cpu).abs().max().item())
        framegroup64_loss_diff = float((framegroup64_loss_cpu - scalar_loss_cpu).abs().item())
        framegroup64_grad_diff = float((framegroup64_grad_cpu - scalar_grad_cpu).abs().max().item())
        packed_loss_diff = float((packed_framegroup_loss_cpu - scalar_loss_cpu).abs().item())
        packed_grad_diff = float((packed_framegroup_grad_cpu - scalar_grad_cpu).abs().max().item())
        materialized_loss_diff = float((materialized_loss_cpu - scalar_loss_cpu).abs().item())
        materialized_grad_diff = float((materialized_grad_cpu - scalar_grad_cpu).abs().max().item())
        i16x4_loss_diff = float((i16x4_framegroup_loss_cpu - scalar_loss_cpu).abs().item())
        i16x4_grad_diff = float((i16x4_framegroup_grad_cpu - scalar_grad_cpu).abs().max().item())
        self.assertLessEqual(loss_diff, atol)
        self.assertLessEqual(grad_diff, atol)
        self.assertLessEqual(ownerreduce_loss_diff, atol)
        self.assertLessEqual(ownerreduce_grad_diff, atol)
        self.assertLessEqual(i16cols_loss_diff, atol)
        self.assertLessEqual(i16cols_grad_diff, atol)
        self.assertLessEqual(framegroup64_loss_diff, atol)
        self.assertLessEqual(framegroup64_grad_diff, atol)
        self.assertLessEqual(packed_loss_diff, atol)
        self.assertLessEqual(packed_grad_diff, atol)
        self.assertLessEqual(materialized_loss_diff, atol)
        self.assertLessEqual(materialized_grad_diff, atol)
        self.assertLessEqual(i16x4_loss_diff, atol)
        self.assertLessEqual(i16x4_grad_diff, atol)
        self.assertGreater(float(framegroup_grad_cpu.abs().sum().item()), 0.0)
        self.assertGreater(float(ownerreduce_grad_cpu.abs().sum().item()), 0.0)
        self.assertGreater(float(i16cols_framegroup_grad_cpu.abs().sum().item()), 0.0)
        self.assertGreater(float(framegroup64_grad_cpu.abs().sum().item()), 0.0)
        self.assertGreater(float(packed_framegroup_grad_cpu.abs().sum().item()), 0.0)
        self.assertGreater(float(materialized_grad_cpu.abs().sum().item()), 0.0)
        self.assertGreater(float(i16x4_framegroup_grad_cpu.abs().sum().item()), 0.0)

    def test_delta_replace_framegroup_chunk_offsets_match_scalar_after_first_chunk(self) -> None:
        device = torch.device("mps")
        frame_count = 20
        frames = []
        for frame_id in range(frame_count):
            if frame_id < 5:
                frames.append((_record(0, -1, 0), _record(1, 0, -2)))
            elif frame_id < 17:
                frames.append((_record(1, -1, 0), _record(0, 0, -2)))
            else:
                frames.append((_record(0, -1, -2),))
        delta_tape = pack_endpoint_record_delta_replace_tape([frames], frame_count=frame_count)
        chunk_offsets = build_delta_replace_chunk_change_offsets(delta_tape, frame_count=frame_count)

        self.assertEqual(chunk_offsets.tolist(), [0, 2])

        boundaries = (SimpleNamespace(nx=0.0, ny=0.0, nz=1.0, nt=0.0, b=-2.0),)
        frame_t_f32 = torch.linspace(0.0, 1.0, frame_count, device=device, dtype=torch.float32)
        rays_f32 = torch.zeros((1, frame_count, 6), device=device, dtype=torch.float32)
        rays_f32[:, :, 2] = torch.linspace(0.0, 0.5, frame_count, device=device)
        rays_f32[:, :, 5] = 1.0
        coeff_f16 = _track_boundary_coefficients(
            boundaries=boundaries,
            track_rays=rays_f32.detach().cpu(),
            frame_t=frame_t_f32.detach().cpu(),
        ).to(device=device, dtype=torch.float16)
        site_rgba_f32 = torch.tensor(
            [[0.25, 0.50, 0.75, 1.0], [0.75, 0.25, 0.50, 1.5]],
            device=device,
            dtype=torch.float32,
        )
        target = torch.zeros((1, frame_count, 3), device=device, dtype=torch.float32)
        config = RealRayReplayConfig(near=0.0, far=3.0, invalid_epsilon=1.0e-7, transmittance_threshold=1.0e-4)

        scalar_loss, scalar_grad = endpoint_record_delta_replace_coeff16_i16x3_mse_vjp_direct_atomic_rgb_only(
            coeff_f16,
            frame_t_f32,
            delta_tape.base_offsets_i32.to(device=device),
            _tape_i16x3(delta_tape.base_owner_i32, delta_tape.base_left_i32, delta_tape.base_right_i32).to(
                device=device
            ),
            delta_tape.track_change_offsets_i32.to(device=device),
            delta_tape.change_frame_i32.to(device=device),
            delta_tape.change_offsets_i32.to(device=device),
            _tape_i16x3(delta_tape.change_owner_i32, delta_tape.change_left_i32, delta_tape.change_right_i32).to(
                device=device
            ),
            site_rgba_f32,
            target,
            config,
            track_count=1,
            frame_count=frame_count,
            boundary_count=1,
        )
        framegroup_loss, framegroup_grad = (
            endpoint_record_delta_replace_coeff16_i16x3_framegroup16_mse_vjp_direct_atomic_rgb_only(
                coeff_f16,
                frame_t_f32,
                delta_tape.base_offsets_i32.to(device=device),
                _tape_i16x3(delta_tape.base_owner_i32, delta_tape.base_left_i32, delta_tape.base_right_i32).to(
                    device=device
                ),
                delta_tape.track_change_offsets_i32.to(device=device),
                chunk_offsets.to(device=device, dtype=torch.int16),
                delta_tape.change_frame_i32.to(device=device),
                delta_tape.change_offsets_i32.to(device=device),
                _tape_i16x3(delta_tape.change_owner_i32, delta_tape.change_left_i32, delta_tape.change_right_i32).to(
                    device=device
                ),
                site_rgba_f32,
                target,
                config,
                track_count=1,
                frame_count=frame_count,
                boundary_count=1,
            )
        )
        torch.mps.synchronize()

        self.assertTrue(
            torch.allclose(framegroup_loss.cpu().reshape(()), scalar_loss.cpu().reshape(()), atol=1.0e-6, rtol=0.0)
        )
        self.assertTrue(torch.allclose(framegroup_grad.cpu(), scalar_grad.cpu(), atol=1.0e-6, rtol=0.0))

    def test_delta_replace_framegroup_empty_rows_match_scalar(self) -> None:
        frame_count = 40
        frames = []
        for frame_id in range(frame_count):
            if frame_id < 8:
                frames.append(())
            elif frame_id < 20:
                frames.append((_record(0, -1, -2),))
            elif frame_id < 32:
                frames.append(())
            else:
                frames.append((_record(1, -1, -2),))

        self._assert_delta_replace_framegroup_matches_scalar(
            [frames],
            frame_count=frame_count,
            boundaries=(SimpleNamespace(nx=0.0, ny=0.0, nz=1.0, nt=0.0, b=-2.0),),
            site_rgba_rows=[
                [0.25, 0.50, 0.75, 1.0],
                [0.75, 0.25, 0.50, 1.5],
            ],
        )

    def test_delta_replace_framegroup_rowref_reduce_128_multitrack_matches_scalar(self) -> None:
        frame_count = 128
        sequences = [[], []]
        for frame_id in range(frame_count):
            if frame_id < 16:
                sequences[0].append((_record(0, -1, 0), _record(1, 0, 1), _record(2, 1, -2)))
            elif frame_id < 80:
                sequences[0].append((_record(2, -1, 0), _record(0, 0, 1), _record(1, 1, -2)))
            else:
                sequences[0].append((_record(1, -1, 1), _record(0, 1, -2)))

            if frame_id < 32:
                sequences[1].append((_record(1, -1, 0), _record(0, 0, -2)))
            elif frame_id < 96:
                sequences[1].append((_record(0, -1, 1), _record(2, 1, -2)))
            else:
                sequences[1].append((_record(2, -1, 0), _record(1, 0, -2)))

        self._assert_delta_replace_framegroup_matches_scalar(
            sequences,
            frame_count=frame_count,
            boundaries=(
                SimpleNamespace(nx=0.0, ny=0.0, nz=1.0, nt=0.0, b=-1.35),
                SimpleNamespace(nx=0.0, ny=0.0, nz=1.0, nt=0.0, b=-2.45),
            ),
            site_rgba_rows=[
                [0.25, 0.50, 0.75, 1.0],
                [0.75, 0.25, 0.50, 1.5],
                [0.40, 0.80, 0.20, 0.8],
                [0.10, 0.35, 0.90, 1.2],
            ],
        )

    def test_delta_replace_framegroup_rowref_reduce_128_matches_scalar(self) -> None:
        device = torch.device("mps")
        frame_count = 128
        frames = []
        for frame_id in range(frame_count):
            if frame_id < 16:
                frames.append((_record(0, -1, 0), _record(1, 0, -2)))
            elif frame_id < 80:
                frames.append((_record(1, -1, 0), _record(0, 0, -2)))
            else:
                frames.append((_record(0, -1, -2),))
        delta_tape = pack_endpoint_record_delta_replace_tape([frames], frame_count=frame_count)
        chunk_offsets = build_delta_replace_chunk_change_offsets(delta_tape, frame_count=frame_count)

        self.assertEqual(chunk_offsets.tolist(), [0, 1, 1, 2, 2])

        boundaries = (SimpleNamespace(nx=0.0, ny=0.0, nz=1.0, nt=0.0, b=-2.0),)
        frame_t_f32 = torch.linspace(0.0, 1.0, frame_count, device=device, dtype=torch.float32)
        rays_f32 = torch.zeros((1, frame_count, 6), device=device, dtype=torch.float32)
        rays_f32[:, :, 2] = torch.linspace(0.0, 0.5, frame_count, device=device)
        rays_f32[:, :, 5] = 1.0
        coeff_f16 = _track_boundary_coefficients(
            boundaries=boundaries,
            track_rays=rays_f32.detach().cpu(),
            frame_t=frame_t_f32.detach().cpu(),
        ).to(device=device, dtype=torch.float16)
        site_rgba_f32 = torch.tensor(
            [[0.25, 0.50, 0.75, 1.0], [0.75, 0.25, 0.50, 1.5]],
            device=device,
            dtype=torch.float32,
        )
        target = torch.zeros((1, frame_count, 3), device=device, dtype=torch.float32)
        config = RealRayReplayConfig(near=0.0, far=3.0, invalid_epsilon=1.0e-7, transmittance_threshold=1.0e-4)

        scalar_loss, scalar_grad = endpoint_record_delta_replace_coeff16_i16x3_mse_vjp_direct_atomic_rgb_only(
            coeff_f16,
            frame_t_f32,
            delta_tape.base_offsets_i32.to(device=device),
            _tape_i16x3(delta_tape.base_owner_i32, delta_tape.base_left_i32, delta_tape.base_right_i32).to(
                device=device
            ),
            delta_tape.track_change_offsets_i32.to(device=device),
            delta_tape.change_frame_i32.to(device=device),
            delta_tape.change_offsets_i32.to(device=device),
            _tape_i16x3(delta_tape.change_owner_i32, delta_tape.change_left_i32, delta_tape.change_right_i32).to(
                device=device
            ),
            site_rgba_f32,
            target,
            config,
            track_count=1,
            frame_count=frame_count,
            boundary_count=1,
        )
        framegroup_loss, framegroup_grad = (
            endpoint_record_delta_replace_coeff16_i16x3_framegroup16_mse_vjp_direct_atomic_rgb_only(
                coeff_f16,
                frame_t_f32,
                delta_tape.base_offsets_i32.to(device=device),
                _tape_i16x3(delta_tape.base_owner_i32, delta_tape.base_left_i32, delta_tape.base_right_i32).to(
                    device=device
                ),
                delta_tape.track_change_offsets_i32.to(device=device),
                chunk_offsets.to(device=device, dtype=torch.int16),
                delta_tape.change_frame_i32.to(device=device),
                delta_tape.change_offsets_i32.to(device=device),
                _tape_i16x3(delta_tape.change_owner_i32, delta_tape.change_left_i32, delta_tape.change_right_i32).to(
                    device=device
                ),
                site_rgba_f32,
                target,
                config,
                track_count=1,
                frame_count=frame_count,
                boundary_count=1,
            )
        )
        torch.mps.synchronize()

        self.assertTrue(
            torch.allclose(framegroup_loss.cpu().reshape(()), scalar_loss.cpu().reshape(()), atol=1.0e-6, rtol=0.0)
        )
        self.assertTrue(torch.allclose(framegroup_grad.cpu(), scalar_grad.cpu(), atol=1.0e-6, rtol=0.0))

    def test_delta_replace_framegroup_above_reduce_cap_128_matches_scalar(self) -> None:
        frame_count = 128
        frames = []
        for frame_id in range(frame_count):
            if frame_id < 48:
                frames.append((_record(17, -1, 0), _record(2, 0, -2)))
            elif frame_id < 96:
                frames.append((_record(18, -1, 0), _record(1, 0, -2)))
            else:
                frames.append((_record(19, -1, 0), _record(0, 0, -2)))

        site_rgba_rows = [
            [
                0.1 + 0.03 * float(site_id % 5),
                0.2 + 0.02 * float(site_id % 7),
                0.3 + 0.01 * float(site_id % 11),
                0.5 + 0.05 * float(site_id % 4),
            ]
            for site_id in range(20)
        ]
        self._assert_delta_replace_framegroup_matches_scalar(
            [frames],
            frame_count=frame_count,
            boundaries=(SimpleNamespace(nx=0.0, ny=0.0, nz=1.0, nt=0.0, b=-2.0),),
            site_rgba_rows=site_rgba_rows,
        )


if __name__ == "__main__":
    unittest.main()
