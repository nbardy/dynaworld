from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

import probe_delta_framegroup_variant_timing as probe


class ProbeDeltaFramegroupVariantTimingTests(unittest.TestCase):
    def test_run_probe_forwards_diagnostic_and_launch_only_flags_to_frame_cases(self) -> None:
        calls: list[dict[str, object]] = []

        def fake_frame_case(frame_count: int, **kwargs: object) -> dict[str, object]:
            calls.append({"frame_count": frame_count, **kwargs})
            return {
                "variants": {
                    "packed_framegroup32_launch_only": {
                        "mean_ms": 1.0,
                        "trimmed_mean_ms": 1.0,
                        "median_ms": 1.0,
                    }
                }
            }

        with (
            mock.patch.object(probe.torch.backends.mps, "is_available", return_value=True),
            mock.patch.object(probe, "_frame_case", side_effect=fake_frame_case),
        ):
            payload = probe.run_probe(
                frame_counts=(2, 4),
                warmup=3,
                steps=5,
                track_repeats=7,
                prewarm_sweep=True,
                interleave_variants=True,
                include_diagnostic_packed_variants=True,
                include_launch_only_variants=True,
            )

        self.assertEqual(payload["status"], "ok")
        self.assertTrue(payload["include_diagnostic_packed_variants"])
        self.assertTrue(payload["include_launch_only_variants"])
        self.assertEqual(len(calls), 4)
        self.assertEqual([call["frame_count"] for call in calls], [2, 4, 2, 4])
        for call in calls:
            self.assertTrue(call["include_diagnostic_packed_variants"])
            self.assertTrue(call["include_launch_only_variants"])
            self.assertTrue(call["interleave_variants"])
            self.assertEqual(call["track_repeats"], 7)
        self.assertEqual(calls[0]["warmup"], 1)
        self.assertEqual(calls[0]["steps"], 1)
        self.assertEqual(calls[2]["warmup"], 3)
        self.assertEqual(calls[2]["steps"], 5)
        self.assertTrue(
            payload["scales"]["packed_framegroup32_launch_only"]["mean_is_sublinear_vs_frame_count"]
        )

    def test_packed_launch_only_op_builds_config_tensors_and_scalar_metadata(self) -> None:
        captured: dict[str, object] = {}

        def fake_launch(*args: object) -> tuple[torch.Tensor, torch.Tensor]:
            captured["args"] = args
            return torch.tensor(1.25), torch.tensor([0.5, 0.75])

        fake_ops = SimpleNamespace(fake_launch=fake_launch)
        coeff = torch.zeros((6, 4), dtype=torch.float16)
        frame_t = torch.zeros((3,), dtype=torch.float32)
        base_offsets = torch.tensor([0, 2, 4], dtype=torch.int32)
        base_packed = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
        track_change_offsets = torch.tensor([0, 1, 2], dtype=torch.int32)
        chunk_offsets = torch.tensor([0, 1, 2, 2], dtype=torch.int16)
        change_frame = torch.tensor([1, 2], dtype=torch.int32)
        change_offsets = torch.tensor([0, 2, 4], dtype=torch.int32)
        change_packed = torch.tensor([5, 6, 7, 8], dtype=torch.int32)
        site_rgba = torch.zeros((5, 4), dtype=torch.float32)
        target_rgb = torch.zeros((2, 3, 3), dtype=torch.float32)
        config = probe.RealRayReplayConfig(
            near=0.25,
            far=3.5,
            invalid_epsilon=1.0e-6,
            transmittance_threshold=1.0e-4,
        )

        with mock.patch.object(
            probe.torch.ops,
            "world_foam_lane2_fused_slab_v0",
            fake_ops,
            create=True,
        ):
            loss, grad = probe._packed_launch_only_op(
                "fake_launch",
                coeff=coeff,
                frame_t=frame_t,
                base_offsets=base_offsets,
                base_packed=base_packed,
                track_change_offsets=track_change_offsets,
                chunk_offsets=chunk_offsets,
                change_frame=change_frame,
                change_offsets=change_offsets,
                change_packed=change_packed,
                site_rgba=site_rgba,
                target_rgb=target_rgb,
                config=config,
                boundary_count=3,
                track_count=2,
                frame_count=3,
            )

        self.assertEqual(float(loss.item()), 1.25)
        self.assertEqual(grad.tolist(), [0.5, 0.75])
        args = captured["args"]
        self.assertIs(args[0], coeff)
        self.assertIs(args[3], base_packed)
        config_i32 = args[11]
        config_f32 = args[12]
        self.assertIsInstance(config_i32, torch.Tensor)
        self.assertIsInstance(config_f32, torch.Tensor)
        self.assertEqual(config_i32.tolist(), [3, 2, 3, 5, 4, 2, 4])
        for actual, expected in zip(config_f32.tolist(), [0.25, 3.5, 1.0e-6, 1.0e-4], strict=True):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(args[13:], (3, 2, 3, 5, 4, 2, 4))


if __name__ == "__main__":
    unittest.main()
