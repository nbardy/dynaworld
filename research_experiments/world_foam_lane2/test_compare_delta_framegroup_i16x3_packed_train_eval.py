from __future__ import annotations

import unittest

from compare_delta_framegroup_i16x3_packed_train_eval import _combine_single_frame_payloads, summarize_auto_selector, summarize_pair
from train_eval_owner_run_tape import (
    DELTA_AUTO_FRAMEGROUP16_MODE,
    DELTA_I16X3_FRAMEGROUP16_MODE,
    DELTA_PACKED_FRAMEGROUP16_MODE,
    DELTA_PACKED_FRAMEGROUP16_SMALLRUN16_MODE,
    _effective_native_emitted_pack_records,
    _resolve_delta_framegroup16_auto_mode,
)


def _row(
    *,
    frame_count: int,
    total_ms: float,
    backward_ms: float,
    psnr: float,
    storage: int,
    resolved_mode: str | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "frame_count": frame_count,
        "step_summary": {
            "total": {"mean_s": total_ms / 1000.0, "median_s": total_ms / 1000.0},
            "backward": {"mean_s": backward_ms / 1000.0, "median_s": backward_ms / 1000.0},
            "render": {"mean_s": 0.0, "median_s": 0.0},
        },
        "final_heldout_psnr": psnr,
        "train_selected_tape_storage_bytes": storage,
        "train_selected_tape_segments": storage,
        "train_full_storage_bytes": storage * 10,
        "train_full_segments": storage * 10,
        "train_owner_run_segments": max(storage // 2, 1),
        "train_endpoint_record_edit_ops": max(storage // 3, 1),
        "status": "ok",
    }
    if resolved_mode is not None:
        row["tape_mode_resolved"] = resolved_mode
    return row


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


class CompareDeltaFramegroupI16x3PackedTests(unittest.TestCase):
    def test_auto_mode_only_emits_packed_records_when_resolved_mode_uses_them(self) -> None:
        packed = _resolve_delta_framegroup16_auto_mode(DELTA_AUTO_FRAMEGROUP16_MODE, frame_count=64)
        i16x3 = _resolve_delta_framegroup16_auto_mode(DELTA_AUTO_FRAMEGROUP16_MODE, frame_count=128)
        smallrun16 = _resolve_delta_framegroup16_auto_mode(
            DELTA_AUTO_FRAMEGROUP16_MODE,
            frame_count=64,
            prefer_smallrun16=True,
        )

        self.assertEqual(packed, DELTA_PACKED_FRAMEGROUP16_MODE)
        self.assertEqual(i16x3, DELTA_I16X3_FRAMEGROUP16_MODE)
        self.assertEqual(smallrun16, DELTA_PACKED_FRAMEGROUP16_SMALLRUN16_MODE)
        self.assertTrue(_effective_native_emitted_pack_records(requested=True, resolved_tape_mode=packed))
        self.assertTrue(_effective_native_emitted_pack_records(requested=True, resolved_tape_mode=smallrun16))
        self.assertFalse(_effective_native_emitted_pack_records(requested=True, resolved_tape_mode=i16x3))
        self.assertFalse(_effective_native_emitted_pack_records(requested=False, resolved_tape_mode=packed))

    def test_summarize_pair_reports_storage_win_but_rejects_slow_packed(self) -> None:
        i16x3 = _payload(
            total_scale=1.4,
            backward_scale=1.5,
            storage_scale=1.01,
            rows=[
                _row(frame_count=16, total_ms=4.0, backward_ms=3.0, psnr=14.0, storage=1000),
                _row(frame_count=32, total_ms=5.6, backward_ms=4.5, psnr=14.1, storage=1010),
            ],
        )
        packed = _payload(
            total_scale=1.2,
            backward_scale=1.3,
            storage_scale=1.01,
            rows=[
                _row(frame_count=16, total_ms=5.0, backward_ms=4.0, psnr=14.00001, storage=900),
                _row(frame_count=32, total_ms=5.4, backward_ms=4.4, psnr=14.10001, storage=910),
            ],
        )

        summary = summarize_pair(i16x3=i16x3, packed=packed, frame_counts=(16, 32))

        self.assertAlmostEqual(summary["ratios_by_frame"]["16"]["packed_over_i16x3_total_mean"], 1.25)
        self.assertAlmostEqual(summary["ratios_by_frame"]["32"]["packed_over_i16x3_backward_mean"], 4.4 / 4.5)
        self.assertAlmostEqual(summary["ratios_by_frame"]["16"]["packed_over_i16x3_storage"], 0.9)
        self.assertTrue(summary["packed_storage_below_i16x3"])
        self.assertFalse(summary["packed_speed_promotion_candidate"])

    def test_summarize_pair_can_promote_when_speed_quality_and_scaling_match(self) -> None:
        i16x3 = _payload(
            total_scale=1.4,
            backward_scale=1.5,
            storage_scale=1.01,
            rows=[
                _row(frame_count=16, total_ms=4.0, backward_ms=3.0, psnr=14.0, storage=1000),
                _row(frame_count=32, total_ms=5.6, backward_ms=4.5, psnr=14.1, storage=1010),
            ],
        )
        packed = _payload(
            total_scale=1.3,
            backward_scale=1.4,
            storage_scale=1.01,
            rows=[
                _row(frame_count=16, total_ms=3.9, backward_ms=2.9, psnr=14.00001, storage=900),
                _row(frame_count=32, total_ms=5.4, backward_ms=4.4, psnr=14.10001, storage=910),
            ],
        )

        summary = summarize_pair(i16x3=i16x3, packed=packed, frame_counts=(16, 32))

        self.assertTrue(summary["packed_speed_promotion_candidate"])
        self.assertTrue(summary["packed_storage_below_i16x3"])

    def test_summarize_pair_rejects_missing_frame(self) -> None:
        i16x3 = _payload(
            total_scale=1.0,
            backward_scale=1.0,
            storage_scale=1.0,
            rows=[_row(frame_count=16, total_ms=4.0, backward_ms=3.0, psnr=14.0, storage=1000)],
        )
        packed = _payload(
            total_scale=1.0,
            backward_scale=1.0,
            storage_scale=1.0,
            rows=[_row(frame_count=16, total_ms=4.1, backward_ms=3.1, psnr=14.0, storage=900)],
        )

        with self.assertRaisesRegex(ValueError, "requested frame counts"):
            summarize_pair(i16x3=i16x3, packed=packed, frame_counts=(16, 32))

    def test_combine_single_frame_payloads_rebuilds_scale_and_acceptance(self) -> None:
        payloads = [
            _payload(
                total_scale=1.0,
                backward_scale=1.0,
                storage_scale=1.0,
                rows=[_row(frame_count=64, total_ms=10.0, backward_ms=8.0, psnr=14.0, storage=1000)],
            ),
            _payload(
                total_scale=1.0,
                backward_scale=1.0,
                storage_scale=1.0,
                rows=[_row(frame_count=128, total_ms=12.0, backward_ms=9.0, psnr=14.1, storage=1010)],
            ),
        ]

        combined = _combine_single_frame_payloads(
            tape_mode="endpoint-record-delta-replace-coeff16-packed-framegroup16-fused-mse",
            frame_counts=(64, 128),
            payloads=payloads,
        )

        self.assertEqual(combined["status"], "ok")
        self.assertEqual(combined["frame_counts"], [64, 128])
        self.assertAlmostEqual(combined["total_step_scale_first_to_last"], 1.2)
        self.assertAlmostEqual(combined["backward_scale_first_to_last"], 9.0 / 8.0)
        self.assertAlmostEqual(combined["selected_tape_storage_scale_first_to_last"], 1.01)
        self.assertTrue(combined["acceptance"]["total_step_sublinear_vs_frames"])
        self.assertTrue(combined["acceptance"]["selected_tape_storage_below_full_at_max_frame"])

    def test_summarize_auto_selector_reports_policy_and_oracle_ratio(self) -> None:
        i16x3 = _payload(
            total_scale=1.0,
            backward_scale=1.0,
            storage_scale=1.1,
            rows=[
                _row(frame_count=64, total_ms=10.0, backward_ms=8.0, psnr=14.0, storage=1000),
                _row(frame_count=128, total_ms=8.0, backward_ms=6.0, psnr=14.1, storage=1030),
            ],
        )
        packed = _payload(
            total_scale=2.0,
            backward_scale=2.0,
            storage_scale=1.1,
            rows=[
                _row(frame_count=64, total_ms=7.0, backward_ms=5.0, psnr=14.0, storage=900),
                _row(frame_count=128, total_ms=12.0, backward_ms=9.0, psnr=14.1, storage=990),
            ],
        )
        auto = _payload(
            total_scale=8.0 / 7.0,
            backward_scale=6.0 / 5.0,
            storage_scale=1030 / 900,
            rows=[
                _row(
                    frame_count=64,
                    total_ms=7.0,
                    backward_ms=5.0,
                    psnr=14.0,
                    storage=900,
                    resolved_mode="endpoint-record-delta-replace-coeff16-packed-framegroup16-fused-mse",
                ),
                _row(
                    frame_count=128,
                    total_ms=8.0,
                    backward_ms=6.0,
                    psnr=14.1,
                    storage=1030,
                    resolved_mode="endpoint-record-delta-replace-coeff16-i16x3-framegroup16-fused-mse",
                ),
            ],
        )

        summary = summarize_auto_selector(i16x3=i16x3, packed=packed, auto=auto, frame_counts=(64, 128))

        self.assertTrue(summary["auto_matches_expected_policy"])
        self.assertAlmostEqual(summary["max_auto_over_best_component_total_mean_ratio"], 1.0)
        self.assertAlmostEqual(summary["max_auto_over_best_component_backward_mean_ratio"], 1.0)
        self.assertTrue(summary["auto_oracle_candidate"])


if __name__ == "__main__":
    unittest.main()
