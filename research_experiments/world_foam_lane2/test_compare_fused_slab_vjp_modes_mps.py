from __future__ import annotations

import sys
import unittest
from pathlib import Path


DYNAWORLD = Path(__file__).resolve().parents[2]
TOOLS = DYNAWORLD / "third_party" / "fast-mac-gsplat" / "variants" / "world_foam_lane2_fused_slab_v0" / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from compare_fused_slab_vjp_modes_mps import _parse_modes, summarize_mode_rows  # noqa: E402


class CompareFusedSlabVjpModesMpsTests(unittest.TestCase):
    def test_parse_modes_accepts_fused_mse_rgb_only(self) -> None:
        self.assertEqual(_parse_modes("direct_atomic_rgb_only,fused_mse_rgb_only"), ("direct_atomic_rgb_only", "fused_mse_rgb_only"))

    def test_parse_modes_rejects_unknown_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown VJP mode"):
            _parse_modes("direct_atomic_grad_only,nope")

    def test_summarize_mode_rows_reports_scales(self) -> None:
        rows = [
            {
                "status": "ok",
                "frame_count": 2,
                "final_train_psnr": 10.0,
                "final_heldout_psnr": 9.0,
                "step_summary": {
                    "total": {"median_s": 0.010},
                    "render": {"median_s": 0.003},
                    "backward": {"median_s": 0.004},
                    "optimizer": {"median_s": 0.001},
                },
                "wall_timing": {"total_run_s": 1.0, "train_loop_s": 0.5},
            },
            {
                "status": "ok",
                "frame_count": 16,
                "final_train_psnr": 11.0,
                "final_heldout_psnr": 10.0,
                "step_summary": {
                    "total": {"median_s": 0.015},
                    "render": {"median_s": 0.004},
                    "backward": {"median_s": 0.006},
                    "optimizer": {"median_s": 0.001},
                },
                "wall_timing": {"total_run_s": 1.2, "train_loop_s": 0.7},
            },
        ]

        summary = summarize_mode_rows(rows)

        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["frame_counts"], [2, 16])
        self.assertAlmostEqual(summary["total_median_ms_by_frame"]["2"], 10.0)
        self.assertAlmostEqual(summary["backward_median_ms_by_frame"]["16"], 6.0)
        self.assertAlmostEqual(summary["total_median_scale_first_to_last"], 1.5)
        self.assertAlmostEqual(summary["backward_median_scale_first_to_last"], 1.5)
        self.assertEqual(summary["train_psnr_by_frame"], {"2": 10.0, "16": 11.0})


if __name__ == "__main__":
    unittest.main()
