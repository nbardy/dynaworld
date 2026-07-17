from __future__ import annotations

import unittest

import torch

from probe_endpoint_run_tape import compress_same_owner_endpoint_runs, replay_endpoint_run_tape_torch
from probe_fused_slab_segment_tape import SegmentTape
from torch_world_foam_lane2_fused_slab import (
    RealRayReplayConfig,
    endpoint_run_mse_vjp_direct_atomic_rgb_only,
    endpoint_run_rgba_depth_replay,
    endpoint_run_vjp_direct_atomic_grad_only,
    segment_tape_mse_vjp_direct_atomic_rgb_only,
    segment_tape_nomids_mse_vjp_direct_atomic_rgb_only,
    segment_tape_rgba_depth_replay,
    segment_tape_vjp_direct_atomic_grad_only,
)


class EndpointRunTapeTest(unittest.TestCase):
    def _toy_segment_tape(self) -> SegmentTape:
        return SegmentTape(
            owners_i32=torch.tensor([[[0, 0, 1, 1]]], dtype=torch.int32),
            lengths_f32=torch.tensor([[[1.0, 2.0, 1.0, 1.0]]], dtype=torch.float32),
            mids_f32=torch.tensor([[[0.5, 2.0, 3.5, 4.5]]], dtype=torch.float32),
            counts_i32=torch.tensor([[4]], dtype=torch.int32),
            active_counts_i32=torch.tensor([[4]], dtype=torch.int32),
            frame_t_f32=torch.tensor([0.0], dtype=torch.float32),
            track_count=1,
            frame_count=1,
            max_segments=4,
        )

    def test_compresses_contiguous_same_owner_segments_to_endpoints(self) -> None:
        endpoint = compress_same_owner_endpoint_runs(self._toy_segment_tape())

        self.assertEqual(endpoint.offsets_i32.tolist(), [0, 2])
        self.assertEqual(endpoint.owners_i32.tolist(), [0, 1])
        self.assertEqual(endpoint.starts_f32.tolist(), [0.0, 3.0])
        self.assertEqual(endpoint.ends_f32.tolist(), [3.0, 5.0])

    def test_torch_replay_uses_continuous_absorption_depth(self) -> None:
        endpoint = compress_same_owner_endpoint_runs(self._toy_segment_tape())
        site_rgba = torch.tensor(
            [
                [0.2, 0.4, 0.6, 0.5],
                [0.8, 0.1, 0.3, 0.25],
            ],
            dtype=torch.float32,
        )
        rgb, alpha, depth = replay_endpoint_run_tape_torch(
            tape=endpoint,
            site_rgba_f32=site_rgba,
            track_count=1,
            frame_count=1,
            far=9.0,
            transmittance_threshold=0.0,
        )

        first_alpha = 1.0 - torch.exp(torch.tensor(-1.5))
        first_mass = first_alpha / 0.5 - 3.0 * torch.exp(torch.tensor(-1.5))
        second_trans_before = torch.exp(torch.tensor(-1.5))
        second_alpha = 1.0 - torch.exp(torch.tensor(-0.5))
        second_mass = 3.0 * second_alpha + second_alpha / 0.25 - 2.0 * torch.exp(torch.tensor(-0.5))
        expected_alpha = first_alpha + second_trans_before * second_alpha
        expected_depth = (first_mass + second_trans_before * second_mass) / expected_alpha
        expected_rgb = first_alpha * site_rgba[0, :3] + second_trans_before * second_alpha * site_rgba[1, :3]

        self.assertTrue(torch.allclose(alpha[0, 0], expected_alpha, atol=1.0e-6))
        self.assertTrue(torch.allclose(depth[0, 0], expected_depth, atol=1.0e-6))
        self.assertTrue(torch.allclose(rgb[0, 0], expected_rgb, atol=1.0e-6))

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is required for endpoint-run fused MSE parity")
    def test_fused_mse_matches_replay_plus_rgb_vjp(self) -> None:
        ops = torch.ops.world_foam_lane2_fused_slab_v0
        if not hasattr(ops, "endpoint_run_mse_vjp_direct_atomic_rgb_only"):
            self.skipTest("world_foam_lane2_fused_slab_v0 endpoint-run fused MSE op is not built")
        device = torch.device("mps")
        endpoint = compress_same_owner_endpoint_runs(self._toy_segment_tape())
        offsets = endpoint.offsets_i32.to(device=device).contiguous()
        owners = endpoint.owners_i32.to(device=device).contiguous()
        starts = endpoint.starts_f32.to(device=device).contiguous()
        ends = endpoint.ends_f32.to(device=device).contiguous()
        site_rgba = torch.tensor(
            [
                [0.2, 0.4, 0.6, 0.5],
                [0.8, 0.1, 0.3, 0.25],
            ],
            dtype=torch.float32,
            device=device,
        )
        target_rgb = torch.tensor([[[0.05, 0.2, 0.7]]], dtype=torch.float32, device=device)
        config = RealRayReplayConfig(near=0.0, far=9.0, invalid_epsilon=1.0e-7, transmittance_threshold=0.0)

        rgb, alpha, depth = endpoint_run_rgba_depth_replay(
            offsets,
            owners,
            starts,
            ends,
            site_rgba,
            config,
            track_count=1,
            frame_count=1,
        )
        ref_loss = (rgb - target_rgb).square().mean()
        grad_rgb = (2.0 / float(target_rgb.numel())) * (rgb - target_rgb)
        ref_grad = endpoint_run_vjp_direct_atomic_grad_only(
            offsets,
            owners,
            starts,
            ends,
            site_rgba,
            grad_rgb.contiguous(),
            torch.zeros_like(alpha),
            torch.zeros_like(depth),
            config,
            track_count=1,
            frame_count=1,
        )
        fused_loss, fused_grad = endpoint_run_mse_vjp_direct_atomic_rgb_only(
            offsets,
            owners,
            starts,
            ends,
            site_rgba,
            target_rgb,
            config,
            track_count=1,
            frame_count=1,
        )
        torch.mps.synchronize()

        torch.testing.assert_close(fused_loss.reshape(()).cpu(), ref_loss.cpu(), rtol=1.0e-5, atol=1.0e-6)
        torch.testing.assert_close(fused_grad.cpu(), ref_grad.cpu(), rtol=1.0e-4, atol=1.0e-5)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is required for segment-tape fused MSE parity")
    def test_segment_tape_fused_mse_matches_replay_plus_rgb_vjp(self) -> None:
        ops = torch.ops.world_foam_lane2_fused_slab_v0
        if not hasattr(ops, "segment_tape_mse_vjp_direct_atomic_rgb_only"):
            self.skipTest("world_foam_lane2_fused_slab_v0 segment-tape fused MSE op is not built")
        device = torch.device("mps")
        offsets = torch.tensor([0, 2], dtype=torch.int32, device=device)
        owners = torch.tensor([0, 1], dtype=torch.int32, device=device)
        lengths = torch.tensor([3.0, 2.0], dtype=torch.float32, device=device)
        mids = torch.tensor([1.5, 4.0], dtype=torch.float32, device=device)
        site_rgba = torch.tensor(
            [
                [0.2, 0.4, 0.6, 0.5],
                [0.8, 0.1, 0.3, 0.25],
            ],
            dtype=torch.float32,
            device=device,
        )
        target_rgb = torch.tensor([[[0.05, 0.2, 0.7]]], dtype=torch.float32, device=device)
        config = RealRayReplayConfig(near=0.0, far=9.0, invalid_epsilon=1.0e-7, transmittance_threshold=0.0)

        rgb, alpha, depth = segment_tape_rgba_depth_replay(
            offsets,
            owners,
            lengths,
            mids,
            site_rgba,
            config,
            track_count=1,
            frame_count=1,
        )
        ref_loss = (rgb - target_rgb).square().mean()
        grad_rgb = (2.0 / float(target_rgb.numel())) * (rgb - target_rgb)
        ref_grad = segment_tape_vjp_direct_atomic_grad_only(
            offsets,
            owners,
            lengths,
            mids,
            site_rgba,
            grad_rgb.contiguous(),
            torch.zeros_like(alpha),
            torch.zeros_like(depth),
            config,
            track_count=1,
            frame_count=1,
        )
        fused_loss, fused_grad = segment_tape_mse_vjp_direct_atomic_rgb_only(
            offsets,
            owners,
            lengths,
            mids,
            site_rgba,
            target_rgb,
            config,
            track_count=1,
            frame_count=1,
        )
        torch.mps.synchronize()

        torch.testing.assert_close(fused_loss.reshape(()).cpu(), ref_loss.cpu(), rtol=1.0e-5, atol=1.0e-6)
        torch.testing.assert_close(fused_grad.cpu(), ref_grad.cpu(), rtol=1.0e-4, atol=1.0e-5)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is required for segment-tape no-mid fused MSE parity")
    def test_segment_tape_nomids_fused_mse_matches_mid_fused_path(self) -> None:
        ops = torch.ops.world_foam_lane2_fused_slab_v0
        if not hasattr(ops, "segment_tape_mse_vjp_direct_atomic_rgb_only"):
            self.skipTest("world_foam_lane2_fused_slab_v0 segment-tape fused MSE op is not built")
        device = torch.device("mps")
        offsets = torch.tensor([0, 2], dtype=torch.int32, device=device)
        owners = torch.tensor([0, 1], dtype=torch.int32, device=device)
        lengths = torch.tensor([3.0, 2.0], dtype=torch.float32, device=device)
        mids = torch.tensor([1.5, 4.0], dtype=torch.float32, device=device)
        site_rgba = torch.tensor(
            [
                [0.2, 0.4, 0.6, 0.5],
                [0.8, 0.1, 0.3, 0.25],
            ],
            dtype=torch.float32,
            device=device,
        )
        target_rgb = torch.tensor([[[0.05, 0.2, 0.7]]], dtype=torch.float32, device=device)
        config = RealRayReplayConfig(near=0.0, far=9.0, invalid_epsilon=1.0e-7, transmittance_threshold=0.0)

        mid_loss, mid_grad = segment_tape_mse_vjp_direct_atomic_rgb_only(
            offsets,
            owners,
            lengths,
            mids,
            site_rgba,
            target_rgb,
            config,
            track_count=1,
            frame_count=1,
        )
        nomid_loss, nomid_grad = segment_tape_nomids_mse_vjp_direct_atomic_rgb_only(
            offsets,
            owners,
            lengths,
            site_rgba,
            target_rgb,
            config,
            track_count=1,
            frame_count=1,
        )
        torch.mps.synchronize()

        torch.testing.assert_close(nomid_loss.cpu(), mid_loss.cpu(), rtol=0.0, atol=0.0)
        torch.testing.assert_close(nomid_grad.cpu(), mid_grad.cpu(), rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
