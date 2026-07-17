from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch


DYNAWORLD = Path(__file__).resolve().parents[2]
VARIANT = DYNAWORLD / "third_party" / "fast-mac-gsplat" / "variants" / "world_foam_lane2_fused_slab_v0"
TOOLS = VARIANT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
if str(VARIANT) not in sys.path:
    sys.path.insert(0, str(VARIANT))

from train_eval_fused_slab_mixed_mps import _slice_loaded_training_data, _target_rgb_track_major  # noqa: E402
from torch_world_foam_lane2_fused_slab.ops import (  # noqa: E402
    MAX_REALRAY_BOUNDARIES,
    MAX_REALRAY_FUSED_MSE_BOUNDARIES,
    RealRayReplayConfig,
    _validate_csr_offsets_cpu,
    fused_slab_affine_coeff16_cap224_mse_vjp_direct_atomic_rgb_only,
    fused_slab_affine_coeff16_densitymask_mse_vjp_direct_atomic_rgb_only,
    fused_slab_affine_coeff16_mse_vjp_direct_atomic_rgb_only,
    fused_slab_affine_coeff16_mse_vjp_direct_atomic_sample_reduce_rgb_only,
    fused_slab_affine_coeff16_mse_vjp_direct_atomic_framegroup16_cached_rgb_only,
    fused_slab_affine_coeff16_mse_vjp_direct_atomic_sitecache_rgb_only,
    fused_slab_affine_coeff16_ownerkeep_i16_mse_vjp_direct_atomic_rgb_only,
    fused_slab_affine_coeff16_ownerkeep_mse_vjp_direct_atomic_rgb_only,
    fused_slab_affine_coeff16_ownerupdate_i16_mse_vjp_direct_atomic_rgb_only,
    fused_slab_affine_coeff16_ownerupdate_mse_vjp_direct_atomic_rgb_only,
    fused_slab_affine_coeff16_sortnet_mse_vjp_direct_atomic_rgb_only,
    fused_slab_affine_coeff16_mse_vjp_direct_atomic_track_rgb_only,
    fused_slab_affine_coeff16_realray_rgba_depth_replay,
    fused_slab_affine_num32_den16_mse_vjp_direct_atomic_rgb_only,
    fused_slab_affine_num32_den16_mse_vjp_direct_atomic_track_rgb_only,
    fused_slab_affine_num32_den16_realray_rgba_depth_replay,
)


class TrainEvalFusedSlabMixedMpsTests(unittest.TestCase):
    @staticmethod
    def _reference_rgb_from_depths(
        depths: torch.Tensor,
        sites: torch.Tensor,
        site_rgba: torch.Tensor,
        config: RealRayReplayConfig,
    ) -> torch.Tensor:
        rgb = torch.zeros(3, dtype=torch.float32)
        transmittance = 1.0
        previous_depth = float(config.near)
        for next_depth_t in torch.sort(depths).values.tolist() + [float(config.far)]:
            next_depth = float(next_depth_t)
            length = next_depth - previous_depth
            if length > 1.0e-8 and transmittance > config.transmittance_threshold:
                mid_depth = 0.5 * (previous_depth + next_depth)
                distances = (sites[:, 0]).square() + (sites[:, 1]).square() + (sites[:, 2] - mid_depth).square()
                owner = int(torch.argmin(distances - sites[:, 4]).item())
                density = max(float(site_rgba[owner, 3].item()), 0.0)
                segment_trans = math.exp(-density * length)
                weight = transmittance * (1.0 - segment_trans)
                rgb += float(weight) * site_rgba[owner, :3]
                transmittance *= segment_trans
            previous_depth = next_depth
        return rgb

    def test_slice_loaded_training_data_keeps_view_major_prefix_frames(self) -> None:
        frame_indices = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)
        targets = torch.arange(8, dtype=torch.float32).reshape(8, 1, 1, 1)
        rays = torch.arange(8 * 6, dtype=torch.float32).reshape(8, 1, 1, 6)
        heldout_indices = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        heldout_targets = torch.arange(10, 14, dtype=torch.float32).reshape(4, 1, 1, 1)
        heldout_rays = torch.arange(4 * 6, dtype=torch.float32).reshape(4, 1, 1, 6)
        data = {
            "targets": targets,
            "sample_frame_indices": frame_indices,
            "sample_rays": rays,
            "heldout_targets": heldout_targets,
            "heldout_frame_indices": heldout_indices,
            "heldout_rays": heldout_rays,
            "init_frames": torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3),
            "frame_count": 4,
        }

        sliced = _slice_loaded_training_data(data, frame_count=2)

        torch.testing.assert_close(sliced["targets"].flatten(), torch.tensor([0.0, 1.0, 4.0, 5.0]))
        torch.testing.assert_close(sliced["sample_frame_indices"], torch.tensor([0, 1, 0, 1]))
        torch.testing.assert_close(sliced["sample_rays"][:, 0, 0, 0], torch.tensor([0.0, 6.0, 24.0, 30.0]))
        torch.testing.assert_close(sliced["heldout_targets"].flatten(), torch.tensor([10.0, 11.0]))
        torch.testing.assert_close(sliced["heldout_frame_indices"], torch.tensor([0, 1]))
        torch.testing.assert_close(sliced["heldout_rays"][:, 0, 0, 0], torch.tensor([0.0, 6.0]))
        torch.testing.assert_close(sliced["init_frames"], data["init_frames"][:2])
        self.assertEqual(sliced["frame_count"], 2)

    def test_target_rgb_track_major_inverts_render_layout(self) -> None:
        targets = torch.arange(2 * 3 * 3 * 2 * 2, dtype=torch.float32).reshape(6, 3, 2, 2)

        track_major = _target_rgb_track_major(targets, view_count=2, frame_count=3, height=2, width=2)

        self.assertEqual(tuple(track_major.shape), (8, 3, 3))
        torch.testing.assert_close(track_major[0], targets.reshape(2, 3, 3, 2, 2)[0, :, :, 0, 0])
        torch.testing.assert_close(track_major[3], targets.reshape(2, 3, 3, 2, 2)[0, :, :, 1, 1])
        torch.testing.assert_close(track_major[4], targets.reshape(2, 3, 3, 2, 2)[1, :, :, 0, 0])

    def test_target_rgb_track_major_rejects_wrong_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "target_rgb must have shape"):
            _target_rgb_track_major(torch.zeros((2, 3, 1, 1)), view_count=2, frame_count=2, height=1, width=1)

    def test_fused_mse_csr_guard_accepts_site24_candidate_row(self) -> None:
        row_offsets = torch.tensor([0, 222], dtype=torch.int32)

        _validate_csr_offsets_cpu(
            row_offsets,
            candidate_count=222,
            max_boundaries=MAX_REALRAY_FUSED_MSE_BOUNDARIES,
        )

        self.assertEqual(MAX_REALRAY_BOUNDARIES, 128)
        self.assertEqual(MAX_REALRAY_FUSED_MSE_BOUNDARIES, 256)
        with self.assertRaisesRegex(ValueError, "cap 128"):
            _validate_csr_offsets_cpu(row_offsets, candidate_count=222)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is required for the Metal owner-update regression")
    def test_coeff16_ownerupdate_mse_matches_sample_parallel_with_ambiguous_cut(self) -> None:
        ops = torch.ops.world_foam_lane2_fused_slab_v0
        if not hasattr(ops, "fused_slab_affine_coeff16_ownerupdate_mse_vjp_direct_atomic_rgb_only"):
            self.skipTest("world_foam_lane2_fused_slab_v0 coeff16 ownerupdate fused MSE extension is not built")
        if not hasattr(ops, "fused_slab_affine_coeff16_ownerupdate_i16_mse_vjp_direct_atomic_rgb_only"):
            self.skipTest("world_foam_lane2_fused_slab_v0 coeff16 ownerupdate-i16 fused MSE extension is not built")
        if not hasattr(ops, "fused_slab_affine_coeff16_ownerkeep_mse_vjp_direct_atomic_rgb_only"):
            self.skipTest("world_foam_lane2_fused_slab_v0 coeff16 ownerkeep fused MSE extension is not built")
        if not hasattr(ops, "fused_slab_affine_coeff16_ownerkeep_i16_mse_vjp_direct_atomic_rgb_only"):
            self.skipTest("world_foam_lane2_fused_slab_v0 coeff16 ownerkeep-i16 fused MSE extension is not built")
        device = torch.device("mps")
        config = RealRayReplayConfig(near=0.0, far=1.0, invalid_epsilon=1.0e-6, transmittance_threshold=0.0)
        row_index = torch.tensor([0], device=device, dtype=torch.int32)
        row_offsets = torch.tensor([0, 3], device=device, dtype=torch.int32)
        candidate_ids = torch.tensor([0, 1, 2], device=device, dtype=torch.int32)
        boundary_pairs = torch.tensor([[0, 1], [0, 2], [1, 2]], device=device, dtype=torch.int32)
        depth_coeff16 = torch.tensor(
            [
                [0.35, 0.0, 1.0, 0.0],
                [0.50, 0.0, 1.0, 0.0],
                [0.65, 0.0, 1.0, 0.0],
            ],
            device=device,
            dtype=torch.float16,
        )
        sites = torch.tensor(
            [
                [0.0, 0.0, 0.20, 0.0, 0.0],
                [0.0, 0.0, 0.50, 0.0, 0.0],
                [0.0, 0.0, 0.80, 0.0, 0.0],
            ],
            device=device,
            dtype=torch.float32,
        )
        site_rgba = torch.tensor(
            [
                [0.90, 0.10, 0.05, 0.60],
                [0.10, 0.85, 0.15, -0.45],
                [0.05, 0.20, 0.95, 0.70],
            ],
            device=device,
            dtype=torch.float32,
        )
        ray_coeff = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
            device=device,
            dtype=torch.float32,
        )
        frame_t = torch.tensor([0.0], device=device, dtype=torch.float32)
        target_rgb = torch.zeros((1, 1, 3), device=device, dtype=torch.float32)

        sample_loss, sample_grad = fused_slab_affine_coeff16_mse_vjp_direct_atomic_rgb_only(
            row_index,
            row_offsets,
            depth_coeff16,
            sites,
            site_rgba,
            ray_coeff,
            frame_t,
            target_rgb,
            config,
            time_slab_count=1,
            row_count=1,
        )
        ownerupdate_loss, ownerupdate_grad = fused_slab_affine_coeff16_ownerupdate_mse_vjp_direct_atomic_rgb_only(
            row_index,
            row_offsets,
            candidate_ids,
            depth_coeff16,
            boundary_pairs,
            sites,
            site_rgba,
            ray_coeff,
            frame_t,
            target_rgb,
            config,
            time_slab_count=1,
            row_count=1,
        )
        ownerupdate_i16_loss, ownerupdate_i16_grad = (
            fused_slab_affine_coeff16_ownerupdate_i16_mse_vjp_direct_atomic_rgb_only(
                row_index,
                row_offsets,
                candidate_ids.to(dtype=torch.int16),
                depth_coeff16,
                boundary_pairs.to(dtype=torch.int16),
                sites,
                site_rgba,
                ray_coeff,
                frame_t,
                target_rgb,
                config,
                time_slab_count=1,
                row_count=1,
            )
        )
        ownerkeep_loss, ownerkeep_grad = fused_slab_affine_coeff16_ownerkeep_mse_vjp_direct_atomic_rgb_only(
            row_index,
            row_offsets,
            candidate_ids,
            depth_coeff16,
            boundary_pairs,
            sites,
            site_rgba,
            ray_coeff,
            frame_t,
            target_rgb,
            config,
            time_slab_count=1,
            row_count=1,
        )
        ownerkeep_i16_loss, ownerkeep_i16_grad = (
            fused_slab_affine_coeff16_ownerkeep_i16_mse_vjp_direct_atomic_rgb_only(
                row_index,
                row_offsets,
                candidate_ids.to(dtype=torch.int16),
                depth_coeff16,
                boundary_pairs.to(dtype=torch.int16),
                sites,
                site_rgba,
                ray_coeff,
                frame_t,
                target_rgb,
                config,
                time_slab_count=1,
                row_count=1,
            )
        )
        torch.mps.synchronize()

        torch.testing.assert_close(ownerupdate_loss.detach().cpu(), sample_loss.detach().cpu(), atol=5.0e-5, rtol=5.0e-5)
        torch.testing.assert_close(ownerupdate_grad.detach().cpu(), sample_grad.detach().cpu(), atol=5.0e-5, rtol=5.0e-5)
        torch.testing.assert_close(
            ownerupdate_i16_loss.detach().cpu(),
            sample_loss.detach().cpu(),
            atol=5.0e-5,
            rtol=5.0e-5,
        )
        torch.testing.assert_close(
            ownerupdate_i16_grad.detach().cpu(),
            sample_grad.detach().cpu(),
            atol=5.0e-5,
            rtol=5.0e-5,
        )
        torch.testing.assert_close(ownerkeep_loss.detach().cpu(), sample_loss.detach().cpu(), atol=5.0e-5, rtol=5.0e-5)
        torch.testing.assert_close(ownerkeep_grad.detach().cpu(), sample_grad.detach().cpu(), atol=5.0e-5, rtol=5.0e-5)
        torch.testing.assert_close(
            ownerkeep_i16_loss.detach().cpu(),
            sample_loss.detach().cpu(),
            atol=5.0e-5,
            rtol=5.0e-5,
        )
        torch.testing.assert_close(
            ownerkeep_i16_grad.detach().cpu(),
            sample_grad.detach().cpu(),
            atol=5.0e-5,
            rtol=5.0e-5,
        )

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is required for the Metal sample-reduce regression")
    def test_coeff16_sample_reduce_mse_matches_sample_parallel(self) -> None:
        ops = torch.ops.world_foam_lane2_fused_slab_v0
        if not hasattr(ops, "fused_slab_affine_coeff16_mse_vjp_direct_atomic_sample_reduce_rgb_only"):
            self.skipTest("world_foam_lane2_fused_slab_v0 coeff16 sample-reduce fused MSE extension is not built")
        device = torch.device("mps")
        config = RealRayReplayConfig(near=0.0, far=1.0, invalid_epsilon=1.0e-6, transmittance_threshold=0.0)
        row_index = torch.tensor([0], device=device, dtype=torch.int32)
        row_offsets = torch.tensor([0, 5], device=device, dtype=torch.int32)
        depth_coeff16 = torch.tensor(
            [
                [0.18, 0.0, 1.0, 0.0],
                [0.31, 0.0, 1.0, 0.0],
                [0.48, 0.0, 1.0, 0.0],
                [0.67, 0.0, 1.0, 0.0],
                [0.82, 0.0, 1.0, 0.0],
            ],
            device=device,
            dtype=torch.float16,
        )
        sites = torch.tensor(
            [
                [0.0, 0.0, 0.15, 0.0, 0.0],
                [0.0, 0.0, 0.40, 0.0, 0.0],
                [0.0, 0.0, 0.72, 0.0, 0.0],
            ],
            device=device,
            dtype=torch.float32,
        )
        site_rgba = torch.tensor(
            [
                [0.90, 0.10, 0.05, 0.60],
                [0.10, 0.85, 0.15, 0.45],
                [0.05, 0.20, 0.95, 0.70],
            ],
            device=device,
            dtype=torch.float32,
        )
        ray_coeff = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
            device=device,
            dtype=torch.float32,
        )
        frame_t = torch.tensor([0.0], device=device, dtype=torch.float32)
        target_rgb = torch.tensor([[[0.15, 0.08, 0.25]]], device=device, dtype=torch.float32)

        sample_loss, sample_grad = fused_slab_affine_coeff16_mse_vjp_direct_atomic_rgb_only(
            row_index,
            row_offsets,
            depth_coeff16,
            sites,
            site_rgba,
            ray_coeff,
            frame_t,
            target_rgb,
            config,
            time_slab_count=1,
            row_count=1,
        )
        reduce_loss, reduce_grad = fused_slab_affine_coeff16_mse_vjp_direct_atomic_sample_reduce_rgb_only(
            row_index,
            row_offsets,
            depth_coeff16,
            sites,
            site_rgba,
            ray_coeff,
            frame_t,
            target_rgb,
            config,
            time_slab_count=1,
            row_count=1,
        )
        torch.mps.synchronize()

        torch.testing.assert_close(reduce_loss.detach().cpu(), sample_loss.detach().cpu(), atol=5.0e-5, rtol=5.0e-5)
        torch.testing.assert_close(reduce_grad.detach().cpu(), sample_grad.detach().cpu(), atol=5.0e-5, rtol=5.0e-5)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is required for the Metal framegroup cached regression")
    def test_coeff16_framegroup16_cached_mse_matches_sample_parallel_multi_frame(self) -> None:
        ops = torch.ops.world_foam_lane2_fused_slab_v0
        if not hasattr(ops, "fused_slab_affine_coeff16_mse_vjp_direct_atomic_framegroup16_cached_rgb_only"):
            self.skipTest("world_foam_lane2_fused_slab_v0 coeff16 framegroup16 cached fused MSE extension is not built")
        if not hasattr(ops, "fused_slab_affine_coeff16_mse_vjp_direct_atomic_sitecache_rgb_only"):
            self.skipTest("world_foam_lane2_fused_slab_v0 coeff16 sitecache fused MSE extension is not built")
        if not hasattr(ops, "fused_slab_affine_coeff16_cap224_mse_vjp_direct_atomic_rgb_only"):
            self.skipTest("world_foam_lane2_fused_slab_v0 coeff16 cap224 fused MSE extension is not built")
        if not hasattr(ops, "fused_slab_affine_coeff16_densitymask_mse_vjp_direct_atomic_rgb_only"):
            self.skipTest("world_foam_lane2_fused_slab_v0 coeff16 densitymask fused MSE extension is not built")
        device = torch.device("mps")
        config = RealRayReplayConfig(near=0.0, far=1.0, invalid_epsilon=1.0e-6, transmittance_threshold=0.0)
        row_index = torch.tensor([0], device=device, dtype=torch.int32)
        row_offsets = torch.tensor([0, 5], device=device, dtype=torch.int32)
        depth_coeff16 = torch.tensor(
            [
                [0.18, 0.0, 1.0, 0.0],
                [0.31, 0.0, 1.0, 0.0],
                [0.48, 0.0, 1.0, 0.0],
                [0.67, 0.0, 1.0, 0.0],
                [0.82, 0.0, 1.0, 0.0],
            ],
            device=device,
            dtype=torch.float16,
        )
        sites = torch.tensor(
            [
                [0.0, 0.0, 0.15, 0.0, 0.0],
                [0.0, 0.0, 0.40, 0.0, 0.0],
                [0.0, 0.0, 0.72, 0.0, 0.0],
            ],
            device=device,
            dtype=torch.float32,
        )
        site_rgba = torch.tensor(
            [
                [0.90, 0.10, 0.05, 0.60],
                [0.10, 0.85, 0.15, 0.45],
                [0.05, 0.20, 0.95, 0.70],
            ],
            device=device,
            dtype=torch.float32,
        )
        ray_coeff = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
            device=device,
            dtype=torch.float32,
        )
        frame_t = torch.tensor([0.0, 0.25, 0.50, 0.75], device=device, dtype=torch.float32)
        target_rgb = torch.tensor(
            [[[0.15, 0.08, 0.25], [0.10, 0.20, 0.12], [0.30, 0.05, 0.10], [0.05, 0.18, 0.22]]],
            device=device,
            dtype=torch.float32,
        )

        sample_loss, sample_grad = fused_slab_affine_coeff16_mse_vjp_direct_atomic_rgb_only(
            row_index,
            row_offsets,
            depth_coeff16,
            sites,
            site_rgba,
            ray_coeff,
            frame_t,
            target_rgb,
            config,
            time_slab_count=1,
            row_count=1,
        )
        cached_loss, cached_grad = fused_slab_affine_coeff16_mse_vjp_direct_atomic_framegroup16_cached_rgb_only(
            row_index,
            row_offsets,
            depth_coeff16,
            sites,
            site_rgba,
            ray_coeff,
            frame_t,
            target_rgb,
            config,
            time_slab_count=1,
            row_count=1,
        )
        sitecache_loss, sitecache_grad = fused_slab_affine_coeff16_mse_vjp_direct_atomic_sitecache_rgb_only(
            row_index,
            row_offsets,
            depth_coeff16,
            sites,
            site_rgba,
            ray_coeff,
            frame_t,
            target_rgb,
            config,
            time_slab_count=1,
            row_count=1,
        )
        cap224_loss, cap224_grad = fused_slab_affine_coeff16_cap224_mse_vjp_direct_atomic_rgb_only(
            row_index,
            row_offsets,
            depth_coeff16,
            sites,
            site_rgba,
            ray_coeff,
            frame_t,
            target_rgb,
            config,
            time_slab_count=1,
            row_count=1,
        )
        densitymask_loss, densitymask_grad = fused_slab_affine_coeff16_densitymask_mse_vjp_direct_atomic_rgb_only(
            row_index,
            row_offsets,
            depth_coeff16,
            sites,
            site_rgba,
            ray_coeff,
            frame_t,
            target_rgb,
            config,
            time_slab_count=1,
            row_count=1,
        )
        torch.mps.synchronize()

        torch.testing.assert_close(cached_loss.detach().cpu(), sample_loss.detach().cpu(), atol=5.0e-5, rtol=5.0e-5)
        torch.testing.assert_close(cached_grad.detach().cpu(), sample_grad.detach().cpu(), atol=5.0e-5, rtol=5.0e-5)
        torch.testing.assert_close(sitecache_loss.detach().cpu(), sample_loss.detach().cpu(), atol=5.0e-5, rtol=5.0e-5)
        torch.testing.assert_close(sitecache_grad.detach().cpu(), sample_grad.detach().cpu(), atol=5.0e-5, rtol=5.0e-5)
        torch.testing.assert_close(cap224_loss.detach().cpu(), sample_loss.detach().cpu(), atol=5.0e-5, rtol=5.0e-5)
        torch.testing.assert_close(cap224_grad.detach().cpu(), sample_grad.detach().cpu(), atol=5.0e-5, rtol=5.0e-5)
        torch.testing.assert_close(
            densitymask_loss.detach().cpu(), sample_loss.detach().cpu(), atol=5.0e-5, rtol=5.0e-5
        )
        torch.testing.assert_close(
            densitymask_grad.detach().cpu(), sample_grad.detach().cpu(), atol=5.0e-5, rtol=5.0e-5
        )

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is required for the Metal high-cap replay regression")
    def test_fused_mse_highcap_metal_replays_candidates_beyond_128(self) -> None:
        ops = torch.ops.world_foam_lane2_fused_slab_v0
        if not hasattr(ops, "fused_slab_affine_num32_den16_realray_rgba_depth_replay"):
            self.skipTest("world_foam_lane2_fused_slab_v0 extension is not built")
        if not hasattr(ops, "fused_slab_affine_coeff16_mse_vjp_direct_atomic_rgb_only"):
            self.skipTest("world_foam_lane2_fused_slab_v0 coeff16 fused MSE extension is not built")
        if not hasattr(ops, "fused_slab_affine_coeff16_sortnet_mse_vjp_direct_atomic_rgb_only"):
            self.skipTest("world_foam_lane2_fused_slab_v0 coeff16 sortnet fused MSE extension is not built")
        device = torch.device("mps")
        config = RealRayReplayConfig(near=0.0, far=1.0, invalid_epsilon=1.0e-6, transmittance_threshold=0.0)
        early_depths = torch.linspace(0.10, 0.20, 128, dtype=torch.float32)
        late_depths = torch.linspace(0.70, 0.95, 12, dtype=torch.float32)
        depths = torch.cat([early_depths[::2], early_depths[1::2], late_depths])
        row_index = torch.tensor([0], device=device, dtype=torch.int32)
        row_offsets = torch.tensor([0, depths.numel()], device=device, dtype=torch.int32)
        depth_num = torch.stack([depths, torch.zeros_like(depths)], dim=1).to(device)
        depth_den = torch.stack([torch.ones_like(depths), torch.zeros_like(depths)], dim=1).to(device=device, dtype=torch.float16)
        depth_coeff16 = torch.cat((depth_num.to(dtype=torch.float16), depth_den), dim=1).contiguous()
        sites = torch.tensor(
            [
                [0.0, 0.0, 0.25, 0.0, 0.0],
                [0.0, 0.0, 1.00, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        site_rgba = torch.tensor(
            [
                [0.85, 0.05, 0.10, 0.55],
                [0.05, 0.80, 0.10, 0.75],
            ],
            dtype=torch.float32,
        )
        ray_coeff = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
            device=device,
            dtype=torch.float32,
        )
        frame_t = torch.tensor([0.0], device=device, dtype=torch.float32)
        target_rgb = torch.zeros((1, 1, 3), device=device, dtype=torch.float32)

        expected_rgb = self._reference_rgb_from_depths(depths, sites, site_rgba, config)
        truncated_rgb = self._reference_rgb_from_depths(depths[:MAX_REALRAY_BOUNDARIES], sites, site_rgba, config)
        self.assertGreater(float((expected_rgb - truncated_rgb).abs().max().item()), 1.0e-2)

        replay_rgb, _, _ = fused_slab_affine_num32_den16_realray_rgba_depth_replay(
            row_index,
            row_offsets,
            depth_num,
            depth_den,
            sites.to(device),
            site_rgba.to(device),
            ray_coeff,
            frame_t,
            config,
            time_slab_count=1,
            row_count=1,
        )
        fused_loss, fused_grad = fused_slab_affine_num32_den16_mse_vjp_direct_atomic_rgb_only(
            row_index,
            row_offsets,
            depth_num,
            depth_den,
            sites.to(device),
            site_rgba.to(device),
            ray_coeff,
            frame_t,
            target_rgb,
            config,
            time_slab_count=1,
            row_count=1,
        )
        track_loss, track_grad = fused_slab_affine_num32_den16_mse_vjp_direct_atomic_track_rgb_only(
            row_index,
            row_offsets,
            depth_num,
            depth_den,
            sites.to(device),
            site_rgba.to(device),
            ray_coeff,
            frame_t,
            target_rgb,
            config,
            time_slab_count=1,
            row_count=1,
        )
        coeff16_replay_rgb, _, _ = fused_slab_affine_coeff16_realray_rgba_depth_replay(
            row_index,
            row_offsets,
            depth_coeff16,
            sites.to(device),
            site_rgba.to(device),
            ray_coeff,
            frame_t,
            config,
            time_slab_count=1,
            row_count=1,
        )
        coeff16_loss, coeff16_grad = fused_slab_affine_coeff16_mse_vjp_direct_atomic_rgb_only(
            row_index,
            row_offsets,
            depth_coeff16,
            sites.to(device),
            site_rgba.to(device),
            ray_coeff,
            frame_t,
            target_rgb,
            config,
            time_slab_count=1,
            row_count=1,
        )
        coeff16_sortnet_loss, coeff16_sortnet_grad = fused_slab_affine_coeff16_sortnet_mse_vjp_direct_atomic_rgb_only(
            row_index,
            row_offsets,
            depth_coeff16,
            sites.to(device),
            site_rgba.to(device),
            ray_coeff,
            frame_t,
            target_rgb,
            config,
            time_slab_count=1,
            row_count=1,
        )
        coeff16_track_loss, coeff16_track_grad = fused_slab_affine_coeff16_mse_vjp_direct_atomic_track_rgb_only(
            row_index,
            row_offsets,
            depth_coeff16,
            sites.to(device),
            site_rgba.to(device),
            ray_coeff,
            frame_t,
            target_rgb,
            config,
            time_slab_count=1,
            row_count=1,
        )
        torch.mps.synchronize()

        torch.testing.assert_close(replay_rgb.detach().cpu().reshape(3), expected_rgb, atol=5.0e-5, rtol=5.0e-5)
        expected_loss = expected_rgb.square().mean()
        torch.testing.assert_close(fused_loss.detach().cpu().reshape(()), expected_loss, atol=5.0e-5, rtol=5.0e-5)
        torch.testing.assert_close(track_loss.detach().cpu(), fused_loss.detach().cpu(), atol=5.0e-5, rtol=5.0e-5)
        torch.testing.assert_close(track_grad.detach().cpu(), fused_grad.detach().cpu(), atol=5.0e-5, rtol=5.0e-5)
        coeff16_expected_loss = coeff16_replay_rgb.detach().cpu().reshape(3).square().mean()
        torch.testing.assert_close(
            coeff16_loss.detach().cpu().reshape(()),
            coeff16_expected_loss,
            atol=5.0e-5,
            rtol=5.0e-5,
        )
        torch.testing.assert_close(
            coeff16_track_loss.detach().cpu(),
            coeff16_loss.detach().cpu(),
            atol=5.0e-5,
            rtol=5.0e-5,
        )
        torch.testing.assert_close(
            coeff16_sortnet_loss.detach().cpu(),
            coeff16_loss.detach().cpu(),
            atol=5.0e-5,
            rtol=5.0e-5,
        )
        torch.testing.assert_close(
            coeff16_sortnet_grad.detach().cpu(),
            coeff16_grad.detach().cpu(),
            atol=5.0e-5,
            rtol=5.0e-5,
        )
        torch.testing.assert_close(
            coeff16_track_grad.detach().cpu(),
            coeff16_grad.detach().cpu(),
            atol=5.0e-5,
            rtol=5.0e-5,
        )


if __name__ == "__main__":
    unittest.main()
