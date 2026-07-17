from __future__ import annotations

import unittest

from compare_delta_framegroup_i16x3_i16x4_train_eval import summarize_pair


def _row(*, frame_count: int, total_ms: float, backward_ms: float, psnr: float, storage: int) -> dict[str, object]:
    return {
        "frame_count": frame_count,
        "step_summary": {
            "total": {"mean_s": total_ms / 1000.0, "median_s": total_ms / 1000.0},
            "backward": {"mean_s": backward_ms / 1000.0, "median_s": backward_ms / 1000.0},
        },
        "final_heldout_psnr": psnr,
        "train_selected_tape_storage_bytes": storage,
    }


def _payload(
    *,
    total_scale: float,
    backward_scale: float,
    storage_scale: float,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "status": "ok",
        "total_step_scale_first_to_last": total_scale,
        "backward_scale_first_to_last": backward_scale,
        "selected_tape_storage_scale_first_to_last": storage_scale,
        "rows": rows,
    }


class CompareDeltaFramegroupI16x3I16x4Tests(unittest.TestCase):
    def test_summarize_pair_reports_ratios_and_candidate(self) -> None:
        i16x3 = _payload(
            total_scale=1.4,
            backward_scale=1.5,
            storage_scale=1.01,
            rows=[
                _row(frame_count=16, total_ms=4.0, backward_ms=3.0, psnr=14.0, storage=1000),
                _row(frame_count=32, total_ms=5.6, backward_ms=4.5, psnr=14.1, storage=1010),
            ],
        )
        i16x4 = _payload(
            total_scale=1.3,
            backward_scale=1.4,
            storage_scale=1.01,
            rows=[
                _row(frame_count=16, total_ms=4.1, backward_ms=3.05, psnr=14.00001, storage=1100),
                _row(frame_count=32, total_ms=5.5, backward_ms=4.4, psnr=14.10001, storage=1111),
            ],
        )

        summary = summarize_pair(i16x3=i16x3, i16x4=i16x4, frame_counts=(16, 32))

        self.assertAlmostEqual(summary["ratios_by_frame"]["16"]["i16x4_over_i16x3_total_mean"], 1.025)
        self.assertAlmostEqual(summary["ratios_by_frame"]["32"]["i16x4_over_i16x3_backward_mean"], 4.4 / 4.5)
        self.assertAlmostEqual(summary["ratios_by_frame"]["16"]["i16x4_over_i16x3_storage"], 1.1)
        self.assertLess(summary["max_psnr_delta"], 1.0e-4)
        self.assertTrue(summary["i16x4_speed_promotion_candidate"])

    def test_summarize_pair_rejects_missing_frame(self) -> None:
        i16x3 = _payload(
            total_scale=1.0,
            backward_scale=1.0,
            storage_scale=1.0,
            rows=[_row(frame_count=16, total_ms=4.0, backward_ms=3.0, psnr=14.0, storage=1000)],
        )
        i16x4 = _payload(
            total_scale=1.0,
            backward_scale=1.0,
            storage_scale=1.0,
            rows=[_row(frame_count=16, total_ms=4.1, backward_ms=3.1, psnr=14.0, storage=1000)],
        )

        with self.assertRaisesRegex(ValueError, "requested frame counts"):
            summarize_pair(i16x3=i16x3, i16x4=i16x4, frame_counts=(16, 32))


if __name__ == "__main__":
    unittest.main()
