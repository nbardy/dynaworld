from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

import compare_endpoint_run_record_edit_train_eval as compare_mod
import train_eval_owner_run_tape as train_eval_mod
from probe_endpoint_record_delta_replay import build_delta_replace_frame_row_descriptors


def _fake_payload(*, tape_mode: str, partial_out_json: Path | None = None, **_kwargs: object) -> dict[str, object]:
    timings_s = {
        "endpoint-run": (0.0130, 0.0038, 0.0059),
        "endpoint-record-edit": (0.0112, 0.0035, 0.0049),
        "endpoint-record-edit-fused-mse": (0.0062, 0.0, 0.0030),
        "endpoint-record-delta-replace-coeff16-i16x3-framegroup16-fused-mse": (0.0047, 0.0, 0.0024),
        "endpoint-record-delta-replace-coeff16-i16x4-framegroup16-fused-mse": (0.0051, 0.0, 0.0027),
        "endpoint-record-edit-block4": (0.0091, 0.0024, 0.0050),
        "endpoint-record-edit-block-coeff": (0.0080, 0.0025, 0.0036),
        "endpoint-record-edit-block-coeff-rgb": (0.0070, 0.0018, 0.0035),
        "endpoint-record-edit-block-coeff-fused-mse": (0.0055, 0.0, 0.0029),
        "endpoint-record-edit-block-coeff16": (0.0156, 0.0030, 0.0080),
    }
    storage = {
        "endpoint-run": 0.111,
        "endpoint-record-edit": 0.026,
        "endpoint-record-edit-fused-mse": 0.026,
        "endpoint-record-delta-replace-coeff16-i16x3-framegroup16-fused-mse": 0.032,
        "endpoint-record-delta-replace-coeff16-i16x4-framegroup16-fused-mse": 0.041,
        "endpoint-record-edit-block4": 0.044,
        "endpoint-record-edit-block-coeff": 0.181,
        "endpoint-record-edit-block-coeff-rgb": 0.181,
        "endpoint-record-edit-block-coeff-fused-mse": 0.181,
        "endpoint-record-edit-block-coeff16": 0.112,
    }
    total_s, render_s, backward_s = timings_s[tape_mode]
    row = {
        "frame_count": 16,
        "step_summary": {
            "total": {"mean_s": total_s},
            "render": {"mean_s": render_s},
            "backward": {"mean_s": backward_s},
        },
        "final_heldout_psnr": 13.25,
        "train_selected_tape_segments_vs_full": 0.103,
        "train_selected_tape_storage_vs_full": storage[tape_mode],
        "train_endpoint_record_edit_storage_vs_endpoint_run": 0.235
        if tape_mode != "endpoint-run"
        else 0.0,
        "train_endpoint_record_block4_storage_vs_endpoint_run": 0.395
        if tape_mode in {
            "endpoint-record-edit-block4",
            "endpoint-record-edit-block-coeff",
            "endpoint-record-edit-block-coeff-rgb",
            "endpoint-record-edit-block-coeff-fused-mse",
            "endpoint-record-edit-block-coeff16",
        }
        else 0.0,
    }
    payload = {
        "status": "ok",
        "optimizer_mode": "autograd",
        "tape_mode": tape_mode,
        "rows": [row],
    }
    if partial_out_json is not None:
        partial_out_json.parent.mkdir(parents=True, exist_ok=True)
        partial_out_json.write_text(json.dumps({"tape_mode": tape_mode, "rows": [row]}) + "\n", encoding="utf-8")
    return payload


class CompareEndpointRunRecordEditTrainEvalTests(unittest.TestCase):
    def test_repeat_loaded_frames_expands_view_major_samples(self) -> None:
        targets = torch.arange(4, dtype=torch.float32).reshape(4, 1, 1, 1)
        rays = torch.arange(4 * 6, dtype=torch.float32).reshape(4, 1, 1, 6)
        frame_indices = torch.tensor([0, 1, 0, 1], dtype=torch.long)

        expanded_targets, expanded_rays, expanded_indices, repeated = train_eval_mod._fit_loaded_frame_count(
            split_name="train",
            targets=targets,
            rays=rays,
            frame_indices=frame_indices,
            loaded_frame_count=2,
            requested_frame_count=5,
            allow_repeat_loaded_frames=True,
        )

        self.assertTrue(repeated)
        self.assertEqual(expanded_targets.shape[0], 10)
        self.assertEqual(expanded_rays.shape[0], 10)
        self.assertEqual(expanded_targets[:, 0, 0, 0].tolist(), [0, 1, 0, 1, 0, 2, 3, 2, 3, 2])
        self.assertEqual(expanded_indices.tolist(), [0, 1, 2, 3, 4, 0, 1, 2, 3, 4])

    def test_repeat_loaded_frames_requires_explicit_opt_in(self) -> None:
        targets = torch.zeros((4, 1, 1, 1), dtype=torch.float32)
        rays = torch.zeros((4, 1, 1, 6), dtype=torch.float32)
        frame_indices = torch.tensor([0, 1, 0, 1], dtype=torch.long)

        with self.assertRaisesRegex(ValueError, "--repeat-loaded-frames"):
            train_eval_mod._fit_loaded_frame_count(
                split_name="heldout",
                targets=targets,
                rays=rays,
                frame_indices=frame_indices,
                loaded_frame_count=2,
                requested_frame_count=5,
                allow_repeat_loaded_frames=False,
            )

    def test_summarize_steps_records_robust_outlier_fields(self) -> None:
        summary = train_eval_mod._summarize_steps(
            [
                {"total": 0.001, "render": 0.0004},
                {"total": 0.002, "render": 0.0005},
                {"total": 0.003, "render": 0.0006},
                {"total": 0.100, "render": 0.0300},
            ]
        )

        self.assertEqual(summary["total"]["count"], 4)
        self.assertAlmostEqual(summary["total"]["median_s"], 0.0025)
        self.assertAlmostEqual(summary["total"]["p90_s"], 0.100)
        self.assertGreater(summary["total"]["max_to_median_ratio"], 30.0)

    def test_coeff16_selected_storage_counts_half_precision_sidecar(self) -> None:
        selected = SimpleNamespace(storage_bytes=900)
        edit = SimpleNamespace(storage_bytes=200)
        block_edit = SimpleNamespace(storage_bytes=320)
        coeff = torch.zeros((5, 4), dtype=torch.float32)

        coeff32_storage = train_eval_mod._selected_tape_storage_bytes(
            tape_mode="endpoint-record-edit-block-coeff",
            selected=selected,
            endpoint_record_edit=edit,
            endpoint_record_block_edit=block_edit,
            coeff_f32=coeff,
        )
        coeff16_storage = train_eval_mod._selected_tape_storage_bytes(
            tape_mode="endpoint-record-edit-block-coeff16",
            selected=selected,
            endpoint_record_edit=edit,
            endpoint_record_block_edit=block_edit,
            coeff_f32=coeff,
        )

        self.assertEqual(coeff32_storage, 320 + coeff.numel() * coeff.element_size())
        self.assertEqual(coeff16_storage, 320 + coeff.numel() * 2)
        self.assertNotEqual(coeff16_storage, selected.storage_bytes)
        self.assertEqual(
            train_eval_mod._selected_tape_storage_bytes(
                tape_mode="endpoint-record-edit-block-coeff-rgb",
                selected=selected,
                endpoint_record_edit=edit,
                endpoint_record_block_edit=block_edit,
                coeff_f32=coeff,
            ),
            coeff32_storage,
        )
        self.assertEqual(
            train_eval_mod._selected_tape_storage_bytes(
                tape_mode="endpoint-record-edit-block-coeff-fused-mse",
                selected=selected,
                endpoint_record_edit=edit,
                endpoint_record_block_edit=block_edit,
                coeff_f32=coeff,
            ),
            coeff32_storage,
        )

    def test_selected_coeff_storage_bytes_match_selected_mode_precision(self) -> None:
        coeff = torch.zeros((5, 4), dtype=torch.float32)

        self.assertEqual(
            train_eval_mod._selected_coeff_storage_bytes(
                tape_mode="endpoint-record-edit-block-coeff",
                coeff_f32=coeff,
            ),
            coeff.numel() * coeff.element_size(),
        )
        self.assertEqual(
            train_eval_mod._selected_coeff_storage_bytes(
                tape_mode="endpoint-record-edit-block-coeff16-fused-mse",
                coeff_f32=coeff,
            ),
            coeff.numel() * 2,
        )
        self.assertEqual(
            train_eval_mod._selected_coeff_storage_bytes(
                tape_mode=train_eval_mod.DELTA_AUTO_FRAMEGROUP16_MODE,
                coeff_f32=coeff,
            ),
            coeff.numel() * 2,
        )
        self.assertEqual(
            train_eval_mod._selected_coeff_storage_bytes(
                tape_mode="endpoint-run",
                coeff_f32=coeff,
            ),
            0,
        )

    def test_selected_device_tensor_storage_breakdown_splits_coeff_from_resident_noncoeff(self) -> None:
        selected_device = {
            "delta_coeff_f16": torch.zeros((7, 4), dtype=torch.float16),
            "delta_base_record_i32": torch.zeros((11,), dtype=torch.int32),
            "track_chunk_change_offsets_i16": torch.zeros((5,), dtype=torch.int16),
            "delta_packed_framegroup16_launch_only_fused_mse": True,
            "delta_launch_frame_count": 16,
        }

        breakdown = train_eval_mod._selected_device_tensor_storage_breakdown(selected_device, device_type=None)

        coeff_bytes = 7 * 4 * 2
        noncoeff_bytes = 11 * 4 + 5 * 2
        self.assertEqual(breakdown["coeff_bytes"], coeff_bytes)
        self.assertEqual(breakdown["noncoeff_bytes"], noncoeff_bytes)
        self.assertEqual(breakdown["total_bytes"], coeff_bytes + noncoeff_bytes)
        self.assertEqual(
            breakdown["by_key"],
            {
                "delta_base_record_i32": 11 * 4,
                "delta_coeff_f16": coeff_bytes,
                "track_chunk_change_offsets_i16": 5 * 2,
            },
        )

    def test_affine_candidate_resident_storage_is_not_counted_as_coeff_sidecar(self) -> None:
        selected_device = {
            "affine_candidate_depth_num_f32": torch.zeros((3, 2), dtype=torch.float32),
            "affine_candidate_depth_den_f16": torch.zeros((3, 2), dtype=torch.float16),
            "affine_ray_f32": torch.zeros((2, 12), dtype=torch.float32),
            "affine_candidate_fused_mse": True,
            "affine_time_slab_count": 1,
        }

        breakdown = train_eval_mod._selected_device_tensor_storage_breakdown(selected_device, device_type=None)

        expected_total = 3 * 2 * 4 + 3 * 2 * 2 + 2 * 12 * 4
        self.assertEqual(breakdown["coeff_bytes"], 0)
        self.assertEqual(breakdown["noncoeff_bytes"], expected_total)
        self.assertEqual(breakdown["total_bytes"], expected_total)

    def test_frame_row_descriptors_encode_begin_len_and_source_per_real_frame(self) -> None:
        delta = SimpleNamespace(
            base_offsets_i32=torch.tensor([0, 2, 3], dtype=torch.int32),
            base_owner_i32=torch.zeros((3,), dtype=torch.int32),
            base_left_i32=torch.zeros((3,), dtype=torch.int32),
            base_right_i32=torch.zeros((3,), dtype=torch.int32),
            track_change_offsets_i32=torch.tensor([0, 2, 3], dtype=torch.int32),
            change_frame_i32=torch.tensor([2, 4, 1], dtype=torch.int32),
            change_offsets_i32=torch.tensor([0, 1, 3, 4], dtype=torch.int32),
            change_owner_i32=torch.zeros((4,), dtype=torch.int32),
            change_left_i32=torch.zeros((4,), dtype=torch.int32),
            change_right_i32=torch.zeros((4,), dtype=torch.int32),
        )

        row_begin, row_len_source = build_delta_replace_frame_row_descriptors(delta, frame_count=5)

        change_bit = 0x4000
        self.assertEqual(row_begin.dtype, torch.int32)
        self.assertEqual(row_len_source.dtype, torch.int16)
        self.assertEqual(row_begin.tolist(), [0, 0, 0, 0, 1, 2, 3, 3, 3, 3])
        self.assertEqual(
            row_len_source.tolist(),
            [
                2,
                2,
                change_bit | 1,
                change_bit | 1,
                change_bit | 2,
                1,
                change_bit | 1,
                change_bit | 1,
                change_bit | 1,
                change_bit | 1,
            ],
        )

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS required for launch-only op parity")
    def test_unchecked_launch_only_packed_framegroup_matches_checked_launch_only(self) -> None:
        ops = torch.ops.world_foam_lane2_fused_slab_v0
        checked_name = (
            "endpoint_record_delta_replace_coeff16_packed_framegroup16_launch_only_mse_vjp_direct_atomic_rgb_only"
        )
        unchecked_name = (
            "endpoint_record_delta_replace_coeff16_packed_framegroup16_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only"
        )
        reduce32_name = (
            "endpoint_record_delta_replace_coeff16_packed_framegroup16_reduce32_launch_only_mse_vjp_direct_atomic_rgb_only"
        )
        reduce32_unchecked_name = (
            "endpoint_record_delta_replace_coeff16_packed_framegroup16_reduce32_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only"
        )
        rowselect32_name = (
            "endpoint_record_delta_replace_coeff16_packed_framegroup16_rowselect32_launch_only_mse_vjp_direct_atomic_rgb_only"
        )
        rowselect32_unchecked_name = (
            "endpoint_record_delta_replace_coeff16_packed_framegroup16_rowselect32_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only"
        )
        rowdesc_name = (
            "endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc_launch_only_mse_vjp_direct_atomic_rgb_only"
        )
        rowdesc_unchecked_name = (
            "endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only"
        )
        rowdesc32_name = (
            "endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc32_launch_only_mse_vjp_direct_atomic_rgb_only"
        )
        rowdesc32_unchecked_name = (
            "endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc32_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only"
        )
        if (
            not hasattr(ops, checked_name)
            or not hasattr(ops, unchecked_name)
            or not hasattr(ops, reduce32_name)
            or not hasattr(ops, reduce32_unchecked_name)
            or not hasattr(ops, rowselect32_name)
            or not hasattr(ops, rowselect32_unchecked_name)
            or not hasattr(ops, rowdesc_name)
            or not hasattr(ops, rowdesc_unchecked_name)
            or not hasattr(ops, rowdesc32_name)
            or not hasattr(ops, rowdesc32_unchecked_name)
        ):
            self.fail("world_foam_lane2_fused_slab_v0 packed framegroup launch-only ops are not built")

        device = torch.device("mps")
        boundary_count = 1
        track_count = 1
        frame_count = 2
        site_count = 24
        owner_id = 17
        base_record_i32 = train_eval_mod._pack_endpoint_records_i32(
            torch.tensor([owner_id], dtype=torch.int32),
            torch.tensor([-1], dtype=torch.int32),
            torch.tensor([-2], dtype=torch.int32),
            site_count=site_count,
            boundary_count=boundary_count,
        ).to(device=device)
        base_offsets_i32 = torch.tensor([0, 1], dtype=torch.int32, device=device)
        track_change_offsets_i32 = torch.tensor([0, 0], dtype=torch.int32, device=device)
        track_chunk_change_offsets_i16 = torch.tensor([0, 0], dtype=torch.int16, device=device)
        change_frame_i32 = torch.empty((0,), dtype=torch.int32, device=device)
        change_offsets_i32 = torch.tensor([0], dtype=torch.int32, device=device)
        change_record_i32 = torch.empty((0,), dtype=torch.int32, device=device)
        coeff_f16 = torch.zeros((track_count * boundary_count, 4), dtype=torch.float16, device=device)
        frame_t_f32 = torch.tensor([0.0, 1.0], dtype=torch.float32, device=device)
        site_rgba_cpu = torch.zeros((site_count, 4), dtype=torch.float32)
        site_rgba_cpu[:, 0] = torch.linspace(0.05, 0.29, site_count)
        site_rgba_cpu[:, 1] = torch.linspace(0.15, 0.39, site_count)
        site_rgba_cpu[:, 2] = torch.linspace(0.25, 0.49, site_count)
        site_rgba_cpu[:, 3] = 1.25
        site_rgba_f32 = site_rgba_cpu.to(device=device)
        target_rgb_f32 = torch.tensor([[[0.1, 0.2, 0.3], [0.25, 0.35, 0.45]]], dtype=torch.float32, device=device)
        config_i32 = torch.tensor(
            [boundary_count, track_count, frame_count, site_count, 1, 0, 0],
            dtype=torch.int32,
            device=device,
        )
        config_f32 = torch.tensor([0.1, 1.0, 1.0e-6, 1.0e-4], dtype=torch.float32, device=device)

        checked_loss, checked_grad = getattr(ops, checked_name)(
            coeff_f16,
            frame_t_f32,
            base_offsets_i32,
            base_record_i32,
            track_change_offsets_i32,
            track_chunk_change_offsets_i16,
            change_frame_i32,
            change_offsets_i32,
            change_record_i32,
            site_rgba_f32,
            target_rgb_f32,
            config_i32,
            config_f32,
            boundary_count,
            track_count,
            frame_count,
            site_count,
            1,
            0,
            0,
        )
        unchecked_loss, unchecked_grad = getattr(ops, unchecked_name)(
            coeff_f16,
            frame_t_f32,
            base_offsets_i32,
            base_record_i32,
            track_change_offsets_i32,
            track_chunk_change_offsets_i16,
            change_frame_i32,
            change_offsets_i32,
            change_record_i32,
            site_rgba_f32,
            target_rgb_f32,
            config_i32,
            config_f32,
            track_count,
            frame_count,
            site_count,
        )
        reduce32_loss, reduce32_grad = getattr(ops, reduce32_name)(
            coeff_f16,
            frame_t_f32,
            base_offsets_i32,
            base_record_i32,
            track_change_offsets_i32,
            track_chunk_change_offsets_i16,
            change_frame_i32,
            change_offsets_i32,
            change_record_i32,
            site_rgba_f32,
            target_rgb_f32,
            config_i32,
            config_f32,
            boundary_count,
            track_count,
            frame_count,
            site_count,
            1,
            0,
            0,
        )
        reduce32_unchecked_loss, reduce32_unchecked_grad = getattr(ops, reduce32_unchecked_name)(
            coeff_f16,
            frame_t_f32,
            base_offsets_i32,
            base_record_i32,
            track_change_offsets_i32,
            track_chunk_change_offsets_i16,
            change_frame_i32,
            change_offsets_i32,
            change_record_i32,
            site_rgba_f32,
            target_rgb_f32,
            config_i32,
            config_f32,
            track_count,
            frame_count,
            site_count,
        )
        rowselect32_loss, rowselect32_grad = getattr(ops, rowselect32_name)(
            coeff_f16,
            frame_t_f32,
            base_offsets_i32,
            base_record_i32,
            track_change_offsets_i32,
            track_chunk_change_offsets_i16,
            change_frame_i32,
            change_offsets_i32,
            change_record_i32,
            site_rgba_f32,
            target_rgb_f32,
            config_i32,
            config_f32,
            boundary_count,
            track_count,
            frame_count,
            site_count,
            1,
            0,
            0,
        )
        rowselect32_unchecked_loss, rowselect32_unchecked_grad = getattr(ops, rowselect32_unchecked_name)(
            coeff_f16,
            frame_t_f32,
            base_offsets_i32,
            base_record_i32,
            track_change_offsets_i32,
            track_chunk_change_offsets_i16,
            change_frame_i32,
            change_offsets_i32,
            change_record_i32,
            site_rgba_f32,
            target_rgb_f32,
            config_i32,
            config_f32,
            track_count,
            frame_count,
            site_count,
        )
        row_begin_i32, row_len_source_i16 = build_delta_replace_frame_row_descriptors(
            SimpleNamespace(
                base_offsets_i32=torch.tensor([0, 1], dtype=torch.int32),
                track_change_offsets_i32=torch.tensor([0, 0], dtype=torch.int32),
                change_frame_i32=torch.empty((0,), dtype=torch.int32),
                change_offsets_i32=torch.tensor([0], dtype=torch.int32),
            ),
            frame_count=frame_count,
        )
        rowdesc_loss, rowdesc_grad = getattr(ops, rowdesc_name)(
            coeff_f16,
            frame_t_f32,
            row_begin_i32.to(device=device),
            row_len_source_i16.to(device=device),
            base_record_i32,
            change_record_i32,
            site_rgba_f32,
            target_rgb_f32,
            config_i32,
            config_f32,
            boundary_count,
            track_count,
            frame_count,
            site_count,
            1,
            0,
        )
        rowdesc_unchecked_loss, rowdesc_unchecked_grad = getattr(ops, rowdesc_unchecked_name)(
            coeff_f16,
            frame_t_f32,
            row_begin_i32.to(device=device),
            row_len_source_i16.to(device=device),
            base_record_i32,
            change_record_i32,
            site_rgba_f32,
            target_rgb_f32,
            config_i32,
            config_f32,
            track_count,
            frame_count,
            site_count,
        )
        rowdesc32_loss, rowdesc32_grad = getattr(ops, rowdesc32_name)(
            coeff_f16,
            frame_t_f32,
            row_begin_i32.to(device=device),
            row_len_source_i16.to(device=device),
            base_record_i32,
            change_record_i32,
            site_rgba_f32,
            target_rgb_f32,
            config_i32,
            config_f32,
            boundary_count,
            track_count,
            frame_count,
            site_count,
            1,
            0,
        )
        rowdesc32_unchecked_loss, rowdesc32_unchecked_grad = getattr(ops, rowdesc32_unchecked_name)(
            coeff_f16,
            frame_t_f32,
            row_begin_i32.to(device=device),
            row_len_source_i16.to(device=device),
            base_record_i32,
            change_record_i32,
            site_rgba_f32,
            target_rgb_f32,
            config_i32,
            config_f32,
            track_count,
            frame_count,
            site_count,
        )

        self.assertTrue(torch.allclose(checked_loss.cpu(), unchecked_loss.cpu(), atol=1.0e-7, rtol=0.0))
        self.assertTrue(torch.allclose(checked_grad.cpu(), unchecked_grad.cpu(), atol=1.0e-6, rtol=0.0))
        self.assertTrue(torch.allclose(checked_loss.cpu(), reduce32_loss.cpu(), atol=1.0e-7, rtol=0.0))
        self.assertTrue(torch.allclose(checked_grad.cpu(), reduce32_grad.cpu(), atol=1.0e-6, rtol=0.0))
        self.assertTrue(torch.allclose(checked_loss.cpu(), reduce32_unchecked_loss.cpu(), atol=1.0e-7, rtol=0.0))
        self.assertTrue(torch.allclose(checked_grad.cpu(), reduce32_unchecked_grad.cpu(), atol=1.0e-6, rtol=0.0))
        self.assertTrue(torch.allclose(checked_loss.cpu(), rowselect32_loss.cpu(), atol=1.0e-7, rtol=0.0))
        self.assertTrue(torch.allclose(checked_grad.cpu(), rowselect32_grad.cpu(), atol=1.0e-6, rtol=0.0))
        self.assertTrue(torch.allclose(checked_loss.cpu(), rowselect32_unchecked_loss.cpu(), atol=1.0e-7, rtol=0.0))
        self.assertTrue(torch.allclose(checked_grad.cpu(), rowselect32_unchecked_grad.cpu(), atol=1.0e-6, rtol=0.0))
        self.assertTrue(torch.allclose(checked_loss.cpu(), rowdesc_loss.cpu(), atol=1.0e-7, rtol=0.0))
        self.assertTrue(torch.allclose(checked_grad.cpu(), rowdesc_grad.cpu(), atol=1.0e-6, rtol=0.0))
        self.assertTrue(torch.allclose(checked_loss.cpu(), rowdesc_unchecked_loss.cpu(), atol=1.0e-7, rtol=0.0))
        self.assertTrue(torch.allclose(checked_grad.cpu(), rowdesc_unchecked_grad.cpu(), atol=1.0e-6, rtol=0.0))
        self.assertTrue(torch.allclose(checked_loss.cpu(), rowdesc32_loss.cpu(), atol=1.0e-7, rtol=0.0))
        self.assertTrue(torch.allclose(checked_grad.cpu(), rowdesc32_grad.cpu(), atol=1.0e-6, rtol=0.0))
        self.assertTrue(torch.allclose(checked_loss.cpu(), rowdesc32_unchecked_loss.cpu(), atol=1.0e-7, rtol=0.0))
        self.assertTrue(torch.allclose(checked_grad.cpu(), rowdesc32_unchecked_grad.cpu(), atol=1.0e-6, rtol=0.0))

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS required for packed-device layout check")
    def test_minimal_delta_fused_device_layout_keeps_only_index_tables_on_mps(self) -> None:
        delta = SimpleNamespace(
            base_offsets_i32=torch.tensor([0, 2], dtype=torch.int32),
            base_owner_i32=torch.tensor([0, 1], dtype=torch.int32),
            track_change_offsets_i32=torch.tensor([0, 1], dtype=torch.int32),
            change_frame_i32=torch.tensor([1], dtype=torch.int32),
            change_offsets_i32=torch.tensor([0, 2], dtype=torch.int32),
            change_owner_i32=torch.tensor([2, 3], dtype=torch.int32),
        )
        frame_t_f32 = torch.tensor([0.0, 1.0], dtype=torch.float32)

        selected_device = train_eval_mod._move_endpoint_record_delta_replace_minimal_fused_tape_to_mps(
            delta=delta,
            frame_t_f32=frame_t_f32,
        )

        self.assertEqual(
            set(selected_device),
            set(train_eval_mod._MINIMAL_DELTA_FUSED_DEVICE_TENSOR_KEYS),
        )
        for forbidden_key in (
            "boundary_f32",
            "rays_f32",
            "base_owner_i32",
            "base_left_i32",
            "base_right_i32",
            "change_owner_i32",
            "change_left_i32",
            "change_right_i32",
        ):
            self.assertNotIn(forbidden_key, selected_device)
        for value in selected_device.values():
            self.assertEqual(value.device.type, "mps")
            self.assertTrue(value.is_contiguous())

    def test_validate_packed_endpoint_record_tensor_accepts_shader_contract(self) -> None:
        record = torch.tensor([2097152, 1049089], dtype=torch.int32)

        validated = train_eval_mod._validate_packed_endpoint_record_tensor(
            "packed_base_record_i32",
            record,
            expected_shape=torch.Size([2]),
        )

        self.assertIs(validated, record)

    def test_validate_packed_endpoint_record_tensor_rejects_bad_dtype(self) -> None:
        record = torch.tensor([2097152, 1049089], dtype=torch.int64)

        with self.assertRaisesRegex(ValueError, "packed_base_record_i32.*torch.int32"):
            train_eval_mod._validate_packed_endpoint_record_tensor(
                "packed_base_record_i32",
                record,
                expected_shape=torch.Size([2]),
            )

    def test_validate_packed_endpoint_record_tensor_rejects_bad_shape(self) -> None:
        record = torch.zeros((3,), dtype=torch.int32)

        with self.assertRaisesRegex(ValueError, "packed_base_record_i32 shape"):
            train_eval_mod._validate_packed_endpoint_record_tensor(
                "packed_base_record_i32",
                record,
                expected_shape=torch.Size([2]),
            )

    def test_validate_packed_endpoint_record_tensor_rejects_non_contiguous(self) -> None:
        record = torch.arange(4, dtype=torch.int32)[::2]

        with self.assertRaisesRegex(ValueError, "packed_base_record_i32 must be contiguous"):
            train_eval_mod._validate_packed_endpoint_record_tensor(
                "packed_base_record_i32",
                record,
                expected_shape=torch.Size([2]),
            )

    def test_validate_packed_endpoint_record_tensor_rejects_non_cpu_device(self) -> None:
        record = torch.empty((2,), dtype=torch.int32, device="meta")

        with self.assertRaisesRegex(ValueError, "packed_base_record_i32 must be a CPU tensor"):
            train_eval_mod._validate_packed_endpoint_record_tensor(
                "packed_base_record_i32",
                record,
                expected_shape=torch.Size([2]),
            )

    def test_validate_packed_endpoint_record_tensor_rejects_negative_record(self) -> None:
        record = torch.tensor([-1], dtype=torch.int32)

        with self.assertRaisesRegex(ValueError, "packed_base_record_i32 must contain nonnegative"):
            train_eval_mod._validate_packed_endpoint_record_tensor(
                "packed_base_record_i32",
                record,
                expected_shape=torch.Size([1]),
                site_count=2,
                boundary_count=1,
            )

    def test_validate_packed_endpoint_record_tensor_rejects_owner_out_of_range(self) -> None:
        record = torch.tensor([3], dtype=torch.int32)

        with self.assertRaisesRegex(ValueError, "packed_base_record_i32 owner code must be < site_count=2"):
            train_eval_mod._validate_packed_endpoint_record_tensor(
                "packed_base_record_i32",
                record,
                expected_shape=torch.Size([1]),
                site_count=2,
                boundary_count=1,
            )

    def test_validate_packed_endpoint_record_tensor_rejects_boundary_out_of_range(self) -> None:
        record = torch.tensor([3 << 8], dtype=torch.int32)

        with self.assertRaisesRegex(ValueError, "packed_base_record_i32 left cut id must be < boundary_count=1"):
            train_eval_mod._validate_packed_endpoint_record_tensor(
                "packed_base_record_i32",
                record,
                expected_shape=torch.Size([1]),
                site_count=2,
                boundary_count=1,
            )

    def test_validate_endpoint_delta_index_tables_accepts_shader_contract(self) -> None:
        delta = SimpleNamespace(
            base_offsets_i32=torch.tensor([0, 2], dtype=torch.int32),
            track_change_offsets_i32=torch.tensor([0, 1], dtype=torch.int32),
            change_frame_i32=torch.tensor([1], dtype=torch.int32),
            change_offsets_i32=torch.tensor([0, 2], dtype=torch.int32),
            base_owner_i32=torch.zeros((2,), dtype=torch.int32),
            change_owner_i32=torch.zeros((2,), dtype=torch.int32),
        )

        tables = train_eval_mod._validate_endpoint_delta_index_tables(delta)

        self.assertIs(tables["base_offsets_i32"], delta.base_offsets_i32)
        self.assertIs(tables["track_change_offsets_i32"], delta.track_change_offsets_i32)
        self.assertIs(tables["change_frame_i32"], delta.change_frame_i32)
        self.assertIs(tables["change_offsets_i32"], delta.change_offsets_i32)

    def test_validate_endpoint_delta_index_tables_rejects_mismatched_track_counts(self) -> None:
        delta = SimpleNamespace(
            base_offsets_i32=torch.tensor([0, 1, 2], dtype=torch.int32),
            track_change_offsets_i32=torch.tensor([0, 1], dtype=torch.int32),
            change_frame_i32=torch.tensor([1], dtype=torch.int32),
            change_offsets_i32=torch.tensor([0, 2], dtype=torch.int32),
            base_owner_i32=torch.zeros((2,), dtype=torch.int32),
            change_owner_i32=torch.zeros((2,), dtype=torch.int32),
        )

        with self.assertRaisesRegex(ValueError, "matching track counts"):
            train_eval_mod._validate_endpoint_delta_index_tables(delta)

    def test_validate_endpoint_delta_index_tables_rejects_change_offset_length_mismatch(self) -> None:
        delta = SimpleNamespace(
            base_offsets_i32=torch.tensor([0, 2], dtype=torch.int32),
            track_change_offsets_i32=torch.tensor([0, 1], dtype=torch.int32),
            change_frame_i32=torch.tensor([1], dtype=torch.int32),
            change_offsets_i32=torch.tensor([0, 1, 2], dtype=torch.int32),
            base_owner_i32=torch.zeros((2,), dtype=torch.int32),
            change_owner_i32=torch.zeros((2,), dtype=torch.int32),
        )

        with self.assertRaisesRegex(ValueError, "change_offsets_i32 length"):
            train_eval_mod._validate_endpoint_delta_index_tables(delta)

    def test_validate_endpoint_delta_index_tables_rejects_nonmonotonic_offsets(self) -> None:
        delta = SimpleNamespace(
            base_offsets_i32=torch.tensor([0, 2, 1], dtype=torch.int32),
            track_change_offsets_i32=torch.tensor([0, 0, 1], dtype=torch.int32),
            change_frame_i32=torch.tensor([1], dtype=torch.int32),
            change_offsets_i32=torch.tensor([0, 2], dtype=torch.int32),
            base_owner_i32=torch.zeros((1,), dtype=torch.int32),
            change_owner_i32=torch.zeros((2,), dtype=torch.int32),
        )

        with self.assertRaisesRegex(ValueError, "base_offsets_i32 must be monotonic"):
            train_eval_mod._validate_endpoint_delta_index_tables(delta)

    def test_validate_endpoint_delta_index_tables_rejects_final_offset_mismatch(self) -> None:
        delta = SimpleNamespace(
            base_offsets_i32=torch.tensor([0, 2], dtype=torch.int32),
            track_change_offsets_i32=torch.tensor([0, 1], dtype=torch.int32),
            change_frame_i32=torch.tensor([1], dtype=torch.int32),
            change_offsets_i32=torch.tensor([0, 1], dtype=torch.int32),
            base_owner_i32=torch.zeros((2,), dtype=torch.int32),
            change_owner_i32=torch.zeros((2,), dtype=torch.int32),
        )

        with self.assertRaisesRegex(ValueError, "change_offsets_i32 final offset"):
            train_eval_mod._validate_endpoint_delta_index_tables(delta)

    def test_validate_endpoint_delta_index_tables_rejects_bad_owner_dtype(self) -> None:
        delta = SimpleNamespace(
            base_offsets_i32=torch.tensor([0, 2], dtype=torch.int32),
            track_change_offsets_i32=torch.tensor([0, 1], dtype=torch.int32),
            change_frame_i32=torch.tensor([1], dtype=torch.int32),
            change_offsets_i32=torch.tensor([0, 2], dtype=torch.int32),
            base_owner_i32=torch.zeros((2,), dtype=torch.int64),
            change_owner_i32=torch.zeros((2,), dtype=torch.int32),
        )

        with self.assertRaisesRegex(ValueError, "base_owner_i32 must be torch.int32"):
            train_eval_mod._validate_endpoint_delta_index_tables(delta)

    def test_pack_endpoint_records_i32_rejects_bad_component_dtype(self) -> None:
        owner = torch.zeros((2,), dtype=torch.int64)
        left = torch.zeros((2,), dtype=torch.int32)
        right = torch.zeros((2,), dtype=torch.int32)

        with self.assertRaisesRegex(ValueError, "packed endpoint record owner_i32 must be torch.int32"):
            train_eval_mod._pack_endpoint_records_i32(owner, left, right)

    def test_pack_endpoint_records_i32_rejects_component_shape_mismatch(self) -> None:
        owner = torch.zeros((2,), dtype=torch.int32)
        left = torch.zeros((3,), dtype=torch.int32)
        right = torch.zeros((2,), dtype=torch.int32)

        with self.assertRaisesRegex(ValueError, "owner/left/right tensors must have matching shapes"):
            train_eval_mod._pack_endpoint_records_i32(owner, left, right)

    def test_pack_endpoint_records_i16x3_rejects_noncontiguous_component(self) -> None:
        owner = torch.zeros((2,), dtype=torch.int32)
        left = torch.zeros((2,), dtype=torch.int32)
        right = torch.arange(4, dtype=torch.int32)[::2]

        with self.assertRaisesRegex(ValueError, "int16x3 endpoint record right_i32 must be contiguous"):
            train_eval_mod._pack_endpoint_records_i16x3(owner, left, right)

    def test_pack_endpoint_records_i32_rejects_owner_out_of_site_range(self) -> None:
        owner = torch.tensor([2], dtype=torch.int32)
        left = torch.tensor([-1], dtype=torch.int32)
        right = torch.tensor([-2], dtype=torch.int32)

        with self.assertRaisesRegex(ValueError, "owner_i32 must be < site_count=2"):
            train_eval_mod._pack_endpoint_records_i32(
                owner,
                left,
                right,
                site_count=2,
                boundary_count=1,
            )

    def test_pack_endpoint_records_i32_rejects_cut_out_of_boundary_range(self) -> None:
        owner = torch.tensor([0], dtype=torch.int32)
        left = torch.tensor([1], dtype=torch.int32)
        right = torch.tensor([-2], dtype=torch.int32)

        with self.assertRaisesRegex(ValueError, "left_i32 must be < boundary_count=1"):
            train_eval_mod._pack_endpoint_records_i32(
                owner,
                left,
                right,
                site_count=2,
                boundary_count=1,
            )

    def test_delta_framegroup16_selected_storage_counts_i16_records_and_chunk_offsets(self) -> None:
        selected = SimpleNamespace(storage_bytes=900)
        delta = SimpleNamespace(
            base_offsets_i32=torch.zeros((2,), dtype=torch.int32),
            track_change_offsets_i32=torch.zeros((2,), dtype=torch.int32),
            change_frame_i32=torch.zeros((3,), dtype=torch.int32),
            change_offsets_i32=torch.zeros((4,), dtype=torch.int32),
            base_owner_i32=torch.zeros((2,), dtype=torch.int32),
            change_owner_i32=torch.zeros((5,), dtype=torch.int32),
        )
        coeff = torch.zeros((7, 4), dtype=torch.float32)

        storage = train_eval_mod._selected_tape_storage_bytes(
            tape_mode="endpoint-record-delta-replace-coeff16-i16x3-framegroup16-fused-mse",
            selected=selected,
            endpoint_record_edit=None,
            endpoint_record_delta_replace=delta,
            endpoint_record_block_edit=None,
            coeff_f32=coeff,
            extra_storage_bytes=18,
        )

        int32_tables = (2 + 2 + 3 + 4) * 4
        i16x3_records = (2 + 5) * 6
        coeff16 = coeff.numel() * 2
        self.assertEqual(storage, int32_tables + i16x3_records + coeff16 + 18)
        self.assertNotEqual(storage, selected.storage_bytes)
        self.assertEqual(
            train_eval_mod._selected_tape_storage_bytes(
                tape_mode="endpoint-record-delta-replace-coeff16-i16x4-framegroup16-fused-mse",
                selected=selected,
                endpoint_record_edit=None,
                endpoint_record_delta_replace=delta,
                endpoint_record_block_edit=None,
                coeff_f32=coeff,
                extra_storage_bytes=18,
            ),
            int32_tables + (2 + 5) * 8 + coeff16 + 18,
        )

    def test_rowdesc_selected_storage_counts_descriptors_not_legacy_delta_indexes(self) -> None:
        selected = SimpleNamespace(storage_bytes=900)
        delta = SimpleNamespace(
            base_offsets_i32=torch.zeros((2,), dtype=torch.int32),
            track_change_offsets_i32=torch.zeros((2,), dtype=torch.int32),
            change_frame_i32=torch.zeros((3,), dtype=torch.int32),
            change_offsets_i32=torch.zeros((4,), dtype=torch.int32),
            base_owner_i32=torch.zeros((2,), dtype=torch.int32),
            change_owner_i32=torch.zeros((5,), dtype=torch.int32),
        )
        coeff = torch.zeros((7, 4), dtype=torch.float32)

        storage = train_eval_mod._selected_tape_storage_bytes(
            tape_mode=train_eval_mod.DELTA_PACKED_FRAMEGROUP16_MODE,
            selected=selected,
            endpoint_record_edit=None,
            endpoint_record_delta_replace=delta,
            endpoint_record_block_edit=None,
            coeff_f32=coeff,
            extra_storage_bytes=18,
            include_delta_index_storage=False,
        )

        packed_records = (2 + 5) * 4
        coeff16 = coeff.numel() * 2
        self.assertEqual(storage, packed_records + coeff16 + 18)

    def test_compare_writes_mode_progress_partial(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            partial_path = Path(tmpdir) / "paired.partial.json"
            with patch.object(compare_mod, "run_train_eval", side_effect=_fake_payload) as fake_run:
                payload = compare_mod.compare(
                    config_path=Path("dummy.jsonc"),
                    frame_counts=(2, 4, 8, 16),
                    render_size=32,
                    site_count=12,
                    near=0.1,
                    far=6.0,
                    density=10.0,
                    invalid_epsilon=1.0e-6,
                    transmittance_threshold=1.0e-4,
                    synthetic_motion=compare_mod.SyntheticRayMotion(
                        origin_velocity=(0.08, 0.0, 0.02),
                        direction_velocity=(0.02, 0.0, 0.0),
                    ),
                    steps=5,
                    warmup_steps=1,
                    lr=0.03,
                    beta1=0.9,
                    beta2=0.999,
                    adam_eps=1.0e-8,
                    optimizer_mode="autograd",
                    segment_tape_vjp_mode="direct_atomic_grad_only",
                    include_block4=True,
                    include_block_coeff=True,
                    include_block_coeff_rgb=False,
                    include_block_coeff16=False,
                    edit_block_size=4,
                    partial_out_json=partial_path,
                )

            self.assertEqual(payload["status"], "ok")
            self.assertLess(payload["ratios"]["block_coeff_to_endpoint_total_16f"], 1.0)
            self.assertLess(payload["ratios"]["block_coeff_to_block4_total_16f"], 1.0)
            self.assertEqual(fake_run.call_count, 4)
            self.assertTrue(partial_path.exists())
            partial = json.loads(partial_path.read_text(encoding="utf-8"))
            self.assertEqual(
                partial["completed_modes"],
                [
                    "endpoint-run",
                    "endpoint-record-edit",
                    "endpoint-record-edit-block4",
                    "endpoint-record-edit-block-coeff",
                ],
            )
            self.assertEqual(sorted(partial["results"]), sorted(partial["requested_modes"]))
            self.assertFalse(partial["allow_repeat_loaded_frames"])
            for mode in partial["requested_modes"]:
                self.assertTrue((partial_path.parent / f"paired.partial.{mode}.rows.partial.json").exists())

    def test_compare_records_edit_fused_mse_mode(self) -> None:
        with patch.object(compare_mod, "run_train_eval", side_effect=_fake_payload) as fake_run:
            payload = compare_mod.compare(
                config_path=Path("dummy.jsonc"),
                frame_counts=(16,),
                render_size=32,
                site_count=12,
                near=0.1,
                far=6.0,
                density=10.0,
                invalid_epsilon=1.0e-6,
                transmittance_threshold=1.0e-4,
                synthetic_motion=compare_mod.SyntheticRayMotion(
                    origin_velocity=(0.08, 0.0, 0.02),
                    direction_velocity=(0.02, 0.0, 0.0),
                ),
                steps=5,
                warmup_steps=1,
                lr=0.03,
                beta1=0.9,
                beta2=0.999,
                adam_eps=1.0e-8,
                optimizer_mode="manual-vjp",
                segment_tape_vjp_mode="direct_atomic_grad_only",
                include_block4=False,
                include_block_coeff=False,
                include_block_coeff_rgb=False,
                include_block_coeff16=False,
                edit_block_size=4,
                include_edit_fused_mse=True,
            )

        self.assertEqual(fake_run.call_count, 3)
        self.assertIn("endpoint-record-edit-fused-mse", payload["summary_16f"])
        self.assertLess(payload["ratios"]["edit_fused_mse_to_endpoint_total_16f"], 1.0)
        self.assertLess(payload["ratios"]["edit_fused_mse_to_edit_total_16f"], 1.0)
        self.assertTrue(payload["acceptance"]["edit_fused_mse_psnr_matches"])
        self.assertTrue(payload["acceptance"]["edit_fused_mse_storage_below_endpoint"])

    def test_compare_records_delta_framegroup16_fused_mse_mode(self) -> None:
        mode = "endpoint-record-delta-replace-coeff16-i16x3-framegroup16-fused-mse"
        with patch.object(compare_mod, "run_train_eval", side_effect=_fake_payload) as fake_run:
            payload = compare_mod.compare(
                config_path=Path("dummy.jsonc"),
                frame_counts=(16,),
                render_size=32,
                site_count=12,
                near=0.1,
                far=6.0,
                density=10.0,
                invalid_epsilon=1.0e-6,
                transmittance_threshold=1.0e-4,
                synthetic_motion=compare_mod.SyntheticRayMotion(
                    origin_velocity=(0.08, 0.0, 0.02),
                    direction_velocity=(0.02, 0.0, 0.0),
                ),
                steps=5,
                warmup_steps=1,
                lr=0.03,
                beta1=0.9,
                beta2=0.999,
                adam_eps=1.0e-8,
                optimizer_mode="manual-vjp",
                segment_tape_vjp_mode="direct_atomic_grad_only",
                include_block4=False,
                include_block_coeff=False,
                include_block_coeff_rgb=False,
                include_block_coeff16=False,
                edit_block_size=4,
                include_delta_framegroup16_fused_mse=True,
            )

        self.assertEqual(fake_run.call_count, 3)
        self.assertIn(mode, payload["summary_16f"])
        self.assertEqual(payload["summary_16f"][mode]["render_ms"], 0.0)
        self.assertLess(payload["ratios"]["delta_framegroup16_to_endpoint_total_16f"], 1.0)
        self.assertLess(payload["ratios"]["delta_framegroup16_to_edit_total_16f"], 1.0)
        self.assertTrue(payload["acceptance"]["endpoint_record_delta_framegroup16_fused_mse_ok"])
        self.assertTrue(payload["acceptance"]["delta_framegroup16_psnr_matches"])
        self.assertTrue(payload["acceptance"]["delta_framegroup16_storage_below_endpoint"])
        self.assertIn("Delta-framegroup16 fused-MSE is faster than endpoint-run", payload["conclusion"])

    def test_compare_can_include_delta_i16x4_framegroup16_fused_mse_mode(self) -> None:
        mode = "endpoint-record-delta-replace-coeff16-i16x4-framegroup16-fused-mse"
        with patch.object(compare_mod, "run_train_eval", side_effect=_fake_payload) as fake_run:
            payload = compare_mod.compare(
                config_path=Path("dummy.jsonc"),
                frame_counts=(16,),
                render_size=32,
                site_count=12,
                near=0.1,
                far=6.0,
                density=10.0,
                invalid_epsilon=1.0e-6,
                transmittance_threshold=1.0e-4,
                synthetic_motion=compare_mod.SyntheticRayMotion(
                    origin_velocity=(0.08, 0.0, 0.02),
                    direction_velocity=(0.02, 0.0, 0.0),
                ),
                steps=5,
                warmup_steps=1,
                lr=0.03,
                beta1=0.9,
                beta2=0.999,
                adam_eps=1.0e-8,
                optimizer_mode="manual-vjp",
                segment_tape_vjp_mode="direct_atomic_grad_only",
                include_block4=False,
                include_block_coeff=False,
                include_block_coeff_rgb=False,
                include_block_coeff16=False,
                edit_block_size=4,
                include_delta_i16x4_framegroup16_fused_mse=True,
            )

        self.assertEqual(fake_run.call_count, 3)
        self.assertIn(mode, payload["summary_16f"])
        self.assertEqual(payload["summary_16f"][mode]["render_ms"], 0.0)
        self.assertLess(payload["summary_16f"][mode]["total_ms"], payload["summary_16f"]["endpoint-run"]["total_ms"])

    def test_compare_passes_repeat_loaded_frames_flag(self) -> None:
        with patch.object(compare_mod, "run_train_eval", side_effect=_fake_payload) as fake_run:
            payload = compare_mod.compare(
                config_path=Path("dummy.jsonc"),
                frame_counts=(32,),
                render_size=32,
                site_count=12,
                near=0.1,
                far=6.0,
                density=10.0,
                invalid_epsilon=1.0e-6,
                transmittance_threshold=1.0e-4,
                synthetic_motion=compare_mod.SyntheticRayMotion(
                    origin_velocity=(0.08, 0.0, 0.02),
                    direction_velocity=(0.02, 0.0, 0.0),
                ),
                steps=1,
                warmup_steps=0,
                lr=0.03,
                beta1=0.9,
                beta2=0.999,
                adam_eps=1.0e-8,
                optimizer_mode="autograd",
                segment_tape_vjp_mode="direct_atomic_grad_only",
                include_block4=False,
                include_block_coeff=False,
                include_block_coeff_rgb=False,
                include_block_coeff16=False,
                edit_block_size=4,
                allow_repeat_loaded_frames=True,
            )

        self.assertTrue(payload["allow_repeat_loaded_frames"])
        self.assertIn("repeated loaded frames", payload["scope"])
        self.assertTrue(all(call.kwargs["allow_repeat_loaded_frames"] for call in fake_run.call_args_list))

    def test_compare_records_block_coeff_rgb_mode(self) -> None:
        with patch.object(compare_mod, "run_train_eval", side_effect=_fake_payload):
            payload = compare_mod.compare(
                config_path=Path("dummy.jsonc"),
                frame_counts=(16,),
                render_size=32,
                site_count=12,
                near=0.1,
                far=6.0,
                density=10.0,
                invalid_epsilon=1.0e-6,
                transmittance_threshold=1.0e-4,
                synthetic_motion=compare_mod.SyntheticRayMotion(
                    origin_velocity=(0.08, 0.0, 0.02),
                    direction_velocity=(0.02, 0.0, 0.0),
                ),
                steps=5,
                warmup_steps=1,
                lr=0.03,
                beta1=0.9,
                beta2=0.999,
                adam_eps=1.0e-8,
                optimizer_mode="autograd",
                segment_tape_vjp_mode="direct_atomic_grad_only",
                include_block4=False,
                include_block_coeff=True,
                include_block_coeff_rgb=True,
                include_block_coeff16=False,
                edit_block_size=4,
            )

        self.assertEqual(payload["status"], "ok")
        self.assertIn("endpoint-record-edit-block-coeff-rgb", payload["summary_16f"])
        self.assertLess(payload["ratios"]["block_coeff_rgb_to_block_coeff_render_16f"], 1.0)
        self.assertTrue(payload["acceptance"]["block_coeff_rgb_psnr_matches"])

    def test_compare_records_block_coeff_fused_mse_mode(self) -> None:
        with patch.object(compare_mod, "run_train_eval", side_effect=_fake_payload):
            payload = compare_mod.compare(
                config_path=Path("dummy.jsonc"),
                frame_counts=(16,),
                render_size=32,
                site_count=12,
                near=0.1,
                far=6.0,
                density=10.0,
                invalid_epsilon=1.0e-6,
                transmittance_threshold=1.0e-4,
                synthetic_motion=compare_mod.SyntheticRayMotion(
                    origin_velocity=(0.08, 0.0, 0.02),
                    direction_velocity=(0.02, 0.0, 0.0),
                ),
                steps=5,
                warmup_steps=1,
                lr=0.03,
                beta1=0.9,
                beta2=0.999,
                adam_eps=1.0e-8,
                optimizer_mode="manual-vjp",
                segment_tape_vjp_mode="direct_atomic_grad_only",
                include_block4=False,
                include_block_coeff=True,
                include_block_coeff_rgb=True,
                include_block_coeff16=False,
                edit_block_size=4,
                include_block_coeff_fused_mse=True,
            )

        self.assertEqual(payload["status"], "ok")
        self.assertIn("endpoint-record-edit-block-coeff-fused-mse", payload["summary_16f"])
        self.assertEqual(payload["summary_16f"]["endpoint-record-edit-block-coeff-fused-mse"]["render_ms"], 0.0)
        self.assertLess(payload["ratios"]["block_coeff_fused_mse_to_block_coeff_rgb_total_16f"], 1.0)
        self.assertTrue(payload["acceptance"]["block_coeff_fused_mse_psnr_matches"])

    def test_summary_16f_prefers_16_row_when_32_is_present(self) -> None:
        payload = _fake_payload(tape_mode="endpoint-run")
        rows = payload["rows"]
        self.assertIsInstance(rows, list)
        row32 = dict(rows[0])
        row32["frame_count"] = 32
        row32["step_summary"] = {
            "total": {"mean_s": 99.0},
            "render": {"mean_s": 88.0},
            "backward": {"mean_s": 77.0},
        }
        rows.append(row32)

        summary = compare_mod._summary_16f(payload)

        self.assertEqual(summary["frame_count"], 16)
        self.assertEqual(summary["total_ms"], 13.0)

    def test_compare_records_block_coeff16_negative_speed_read(self) -> None:
        with patch.object(compare_mod, "run_train_eval", side_effect=_fake_payload):
            payload = compare_mod.compare(
                config_path=Path("dummy.jsonc"),
                frame_counts=(16,),
                render_size=32,
                site_count=12,
                near=0.1,
                far=6.0,
                density=10.0,
                invalid_epsilon=1.0e-6,
                transmittance_threshold=1.0e-4,
                synthetic_motion=compare_mod.SyntheticRayMotion(
                    origin_velocity=(0.08, 0.0, 0.02),
                    direction_velocity=(0.02, 0.0, 0.0),
                ),
                steps=5,
                warmup_steps=1,
                lr=0.03,
                beta1=0.9,
                beta2=0.999,
                adam_eps=1.0e-8,
                optimizer_mode="manual-vjp",
                segment_tape_vjp_mode="direct_atomic_grad_only",
                include_block4=True,
                include_block_coeff=True,
                include_block_coeff_rgb=False,
                include_block_coeff16=True,
                edit_block_size=4,
            )

        self.assertEqual(payload["status"], "ok")
        self.assertGreater(payload["ratios"]["block_coeff16_to_endpoint_total_16f"], 1.0)
        self.assertGreater(payload["ratios"]["block_coeff16_to_block_coeff_total_16f"], 1.0)
        self.assertFalse(payload["acceptance"]["block_coeff16_total_not_slower_than_endpoint"])
        self.assertIn("Block-coeff16 is slower than endpoint-run", payload["conclusion"])


if __name__ == "__main__":
    unittest.main()
