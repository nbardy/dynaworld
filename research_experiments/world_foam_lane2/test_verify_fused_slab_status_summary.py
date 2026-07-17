from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import verify_fused_slab_status_summary as verify_mod


SUMMARY_PATH = (
    Path(__file__).resolve().parent
    / "results"
    / "2026-05-15_fused_slab_mixed_status_summary.json"
)


def _verify_payload(payload: dict[str, object]) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmpdir:
        summary_path = Path(tmpdir) / "summary.json"
        summary_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        return verify_mod.verify(
            argparse.Namespace(
                summary_json=summary_path,
                expected_winner="direct_atomic_grad_only",
                max_psnr_spread=1.0e-3,
            )
        )


class VerifyFusedSlabStatusSummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        if not SUMMARY_PATH.exists():
            self.skipTest(f"missing generated summary fixture: {SUMMARY_PATH}")
        self.summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    def test_current_summary_fixture_is_valid(self) -> None:
        result = _verify_payload(self.summary)
        self.assertEqual(result["status"], "ok", result["failures"])
        i16x4 = self.summary["framegroup16_i16x4_compare"]
        self.assertTrue(i16x4["available"])
        self.assertFalse(i16x4["i16x4_speed_promotion_candidate"])
        self.assertFalse(i16x4["i16x4_total_sublinear_claim"])
        packed = self.summary["framegroup16_packed_prewarm_compare"]
        self.assertTrue(packed["available"])
        self.assertTrue(packed["packed_speed_promotion_candidate"])
        self.assertTrue(packed["packed_storage_below_i16x3"])
        packed_broad = self.summary["framegroup16_packed_broad_compare"]
        self.assertTrue(packed_broad["available"])
        self.assertFalse(packed_broad["packed_speed_promotion_candidate"])
        self.assertTrue(packed_broad["speed_rejected_by_128"])

    def test_repeat20_rejects_block_coeff_losing_speed_win(self) -> None:
        payload = copy.deepcopy(self.summary)
        repeat = payload["endpoint_record_edit_block_coeff_repeat20_16f"]
        repeat["ratios"]["block_coeff_to_endpoint_total_16f"] = 1.05
        repeat["acceptance"]["block_coeff_total_not_slower_than_endpoint"] = False
        repeat["block_coeff_speed_read"] = "not_faster_or_not_measured"

        result = _verify_payload(payload)

        self.assertEqual(result["status"], "failed")
        joined = "\n".join(result["failures"])
        self.assertIn("20-step 16f repeat should preserve block-coeff faster than endpoint-run", joined)
        self.assertIn("20-step 16f block-coeff acceptance block_coeff_total_not_slower_than_endpoint", joined)
        self.assertIn("20-step 16f repeat must record block-coeff faster-than-endpoint speed read", joined)

    def test_repeat20_rejects_raw_edit_marked_repeatably_fast(self) -> None:
        payload = copy.deepcopy(self.summary)
        repeat = payload["endpoint_record_edit_block_coeff_repeat20_16f"]
        repeat["ratios"]["edit_to_endpoint_total_16f"] = 0.95
        repeat["acceptance"]["edit_total_not_slower_than_endpoint"] = True
        repeat["speed_read"] = "not_slower_in_this_smoke_run_but_not_stable"

        result = _verify_payload(payload)

        self.assertEqual(result["status"], "failed")
        joined = "\n".join(result["failures"])
        self.assertIn("20-step 16f repeat should preserve that raw edit is slower in this run", joined)
        self.assertIn("20-step 16f repeat should preserve raw edit as slower in acceptance", joined)
        self.assertIn("20-step 16f repeat must record raw edit slower speed read", joined)

    def test_rejects_framegroup_lossreduce_missing_scope_boundary(self) -> None:
        payload = copy.deepcopy(self.summary)
        framegroup = payload["framegroup16_lossreduce_render32"]
        framegroup["full_trainer_claim"] = True
        framegroup["completion_claim"] = True

        result = _verify_payload(payload)

        self.assertEqual(result["status"], "failed")
        joined = "\n".join(result["failures"])
        self.assertIn("framegroup16 loss-reduction guardrail must not claim completion", joined)
        self.assertIn("framegroup16 loss-reduction guardrail must not claim full trainer coverage", joined)

    def test_rejects_framegroup_lossreduce_128f_outlier_regression(self) -> None:
        payload = copy.deepcopy(self.summary)
        framegroup = payload["framegroup16_lossreduce_render32"]
        framegroup["mixed_128_total_max_ms"] = 12.0
        framegroup["confirm_128only"]["total_median_ms"] = 6.0

        result = _verify_payload(payload)

        self.assertEqual(result["status"], "failed")
        joined = "\n".join(result["failures"])
        self.assertIn("framegroup16 loss-reduction mixed 128f total max exceeds outlier guard", joined)
        self.assertIn("framegroup16 loss-reduction 128-only total median exceeds guard", joined)

    def test_rejects_framegroup_compare_speed_regression(self) -> None:
        payload = copy.deepcopy(self.summary)
        framegroup = payload["framegroup16_compare_render32_speedscale"]
        framegroup["ratios_by_frame"]["128"]["total"] = 0.90

        result = _verify_payload(payload)

        self.assertEqual(result["status"], "failed")
        self.assertIn("framegroup16 compare 128f total ratio exceeds guard", "\n".join(result["failures"]))

    def test_rejects_framegroup_compare_scope_or_claim_regression(self) -> None:
        payload = copy.deepcopy(self.summary)
        framegroup = payload["framegroup16_compare_render32_speedscale"]
        framegroup["full_trainer_claim"] = True
        framegroup["scope"] = "paired smoke"

        result = _verify_payload(payload)

        self.assertEqual(result["status"], "failed")
        joined = "\n".join(result["failures"])
        self.assertIn("framegroup16 compare guardrail must not claim full trainer coverage", joined)
        self.assertIn("framegroup16 compare scope must keep benchmark and repeated-frame caveats", joined)

    def test_rejects_framegroup_compare_loaded_boundary_regression(self) -> None:
        payload = copy.deepcopy(self.summary)
        framegroup = payload["framegroup16_compare_render32_speedscale"]
        framegroup["real_loaded_frame_counts"] = [16, 32]
        framegroup["repeated_frame_counts"] = [64, 128]
        framegroup["repeat_scope_by_frame"]["32"] = "real loaded frame count"

        result = _verify_payload(payload)

        self.assertEqual(result["status"], "failed")
        joined = "\n".join(result["failures"])
        self.assertIn("framegroup16 compare real-loaded rows must remain only 16f", joined)
        self.assertIn("framegroup16 compare repeated-fixture rows must remain 32/64/128f", joined)
        self.assertIn("framegroup16 compare 32f row must keep repeated-fixture scope", joined)

    def test_rejects_real32_compare_lost_sublinear_claim(self) -> None:
        payload = copy.deepcopy(self.summary)
        framegroup = payload["framegroup16_real32_render32_compare"]
        framegroup["real_frame_sublinear_claim"] = False
        framegroup["total_sublinear_real_frames"] = False
        framegroup["conclusion"] = "real-loaded compare is not sublinear"

        result = _verify_payload(payload)

        self.assertEqual(result["status"], "failed")
        joined = "\n".join(result["failures"])
        self.assertIn("real32 framegroup compare must preserve measured real-frame sublinear scaling", joined)
        self.assertIn("real32 framegroup compare must preserve total-sublinear win", joined)
        self.assertIn("real32 framegroup compare conclusion must keep sublinear result and scope caveats", joined)

    def test_rejects_real32_compare_repeated_boundary_regression(self) -> None:
        payload = copy.deepcopy(self.summary)
        framegroup = payload["framegroup16_real32_render32_compare"]
        framegroup["real_loaded_frame_counts"] = [16]
        framegroup["repeated_frame_counts"] = [32]
        framegroup["repeat_scope_by_frame"]["32"] = "synthetic repeated-fixture speed-scaling smoke"

        result = _verify_payload(payload)

        self.assertEqual(result["status"], "failed")
        joined = "\n".join(result["failures"])
        self.assertIn("real32 framegroup compare real-loaded rows must remain 16/32f", joined)
        self.assertIn("real32 framegroup compare must not include repeated-fixture rows", joined)
        self.assertIn("real32 framegroup compare 32f row must stay real-loaded", joined)

    def test_rejects_real32_compare_speed_regression(self) -> None:
        payload = copy.deepcopy(self.summary)
        framegroup = payload["framegroup16_real32_render32_compare"]
        framegroup["ratios_by_frame"]["32"]["total"] = 0.90

        result = _verify_payload(payload)

        self.assertEqual(result["status"], "failed")
        self.assertIn("real32 framegroup compare 32f total ratio exceeds guard", "\n".join(result["failures"]))

    def test_rejects_i16x4_compare_accidental_promotion(self) -> None:
        payload = copy.deepcopy(self.summary)
        framegroup = payload["framegroup16_i16x4_compare"]
        i16x4_mode = "endpoint-record-delta-replace-coeff16-i16x4-framegroup16-fused-mse"
        framegroup["i16x4_speed_promotion_candidate"] = True
        framegroup["i16x4_total_sublinear_claim"] = True
        framegroup["mode_statuses"][i16x4_mode] = "ok"
        framegroup["conclusion"] = "promoted"

        result = _verify_payload(payload)

        self.assertEqual(result["status"], "failed")
        joined = "\n".join(result["failures"])
        self.assertIn("i16x4 framegroup must remain a non-promotion candidate", joined)
        self.assertIn("i16x4 framegroup total-sublinear claim must remain false", joined)
        self.assertIn("i16x4 framegroup i16x4 mode status must stay failed until explicit promotion", joined)
        self.assertIn("i16x4 framegroup conclusion must keep non-promotion and scope caveats", joined)

    def test_rejects_i16x4_compare_storage_regression(self) -> None:
        payload = copy.deepcopy(self.summary)
        framegroup = payload["framegroup16_i16x4_compare"]
        framegroup["max_i16x4_over_i16x3_storage_ratio"] = 1.20
        framegroup["ratios_by_frame"]["32"]["i16x4_over_i16x3_storage"] = 1.20

        result = _verify_payload(payload)

        self.assertEqual(result["status"], "failed")
        joined = "\n".join(result["failures"])
        self.assertIn("i16x4 framegroup max_i16x4_over_i16x3_storage_ratio exceeds guard", joined)
        self.assertIn("i16x4 framegroup 32f i16x4_over_i16x3_storage exceeds guard", joined)

    def test_rejects_i16x4_prewarm_compare_accidental_promotion(self) -> None:
        payload = copy.deepcopy(self.summary)
        framegroup = payload["framegroup16_i16x4_prewarm_compare"]
        framegroup["i16x4_speed_promotion_candidate"] = True
        framegroup["speed_rejected_by_ratio"] = False
        framegroup["max_i16x4_over_i16x3_total_mean_ratio"] = 0.95
        framegroup["max_i16x4_over_i16x3_backward_mean_ratio"] = 0.95
        framegroup["ratios_by_frame"]["32"]["i16x4_over_i16x3_total_mean"] = 0.95
        framegroup["ratios_by_frame"]["32"]["i16x4_over_i16x3_backward_mean"] = 0.95

        result = _verify_payload(payload)

        self.assertEqual(result["status"], "failed")
        joined = "\n".join(result["failures"])
        self.assertIn("i16x4 prewarm compare must remain a non-promotion candidate", joined)
        self.assertIn("i16x4 prewarm compare must keep ratio-based non-promotion", joined)
        self.assertIn("i16x4 prewarm compare must preserve a total/backward ratio above promotion guard", joined)
        self.assertIn("i16x4 prewarm compare 32f total ratio must stay above promotion guard", joined)
        self.assertIn("i16x4 prewarm compare 32f backward ratio must stay above promotion guard", joined)

    def test_rejects_packed_compare_lost_candidate_speed(self) -> None:
        payload = copy.deepcopy(self.summary)
        framegroup = payload["framegroup16_packed_prewarm_compare"]
        framegroup["packed_speed_promotion_candidate"] = False
        framegroup["max_packed_over_i16x3_total_mean_ratio"] = 1.01
        framegroup["ratios_by_frame"]["32"]["packed_over_i16x3_total_mean"] = 1.01

        result = _verify_payload(payload)

        self.assertEqual(result["status"], "failed")
        joined = "\n".join(result["failures"])
        self.assertIn("packed prewarm compare must preserve speed-candidate evidence", joined)
        self.assertIn("packed prewarm compare max total ratio exceeds candidate guard", joined)
        self.assertIn("packed prewarm compare 32f packed_over_i16x3_total_mean exceeds candidate guard", joined)

    def test_rejects_packed_compare_scope_or_storage_regression(self) -> None:
        payload = copy.deepcopy(self.summary)
        framegroup = payload["framegroup16_packed_prewarm_compare"]
        framegroup["full_trainer_claim"] = True
        framegroup["packed_storage_below_i16x3"] = False
        framegroup["max_packed_over_i16x3_storage_ratio"] = 1.0
        framegroup["ratios_by_frame"]["16"]["packed_over_i16x3_storage"] = 1.0
        framegroup["conclusion"] = "promoted"

        result = _verify_payload(payload)

        self.assertEqual(result["status"], "failed")
        joined = "\n".join(result["failures"])
        self.assertIn("packed prewarm compare must not claim full trainer coverage", joined)
        self.assertIn("packed prewarm compare must preserve storage-below-i16x3 evidence", joined)
        self.assertIn("packed prewarm compare storage ratio must stay below i16x3", joined)
        self.assertIn("packed prewarm compare 16f storage ratio must stay below i16x3", joined)
        self.assertIn("packed prewarm compare conclusion must keep candidate and scope caveats", joined)

    def test_rejects_packed_broad_compare_accidental_promotion(self) -> None:
        payload = copy.deepcopy(self.summary)
        framegroup = payload["framegroup16_packed_broad_compare"]
        framegroup["packed_speed_promotion_candidate"] = True
        framegroup["speed_rejected_by_128"] = False
        framegroup["max_packed_over_i16x3_total_mean_ratio"] = 0.90
        framegroup["max_packed_over_i16x3_backward_mean_ratio"] = 0.90
        framegroup["ratios_by_frame"]["128"]["packed_over_i16x3_total_mean"] = 0.90
        framegroup["ratios_by_frame"]["128"]["packed_over_i16x3_backward_mean"] = 0.90

        result = _verify_payload(payload)

        self.assertEqual(result["status"], "failed")
        joined = "\n".join(result["failures"])
        self.assertIn("packed broad compare must remain a non-promotion candidate", joined)
        self.assertIn("packed broad compare must keep 128f speed rejection", joined)
        self.assertIn("packed broad compare must preserve total/backward ratio above promotion guard", joined)
        self.assertIn("packed broad compare 128f total ratio must preserve speed rejection", joined)
        self.assertIn("packed broad compare 128f backward ratio must preserve speed rejection", joined)

    def test_rejects_packed_broad_compare_lost_64f_win_or_scope(self) -> None:
        payload = copy.deepcopy(self.summary)
        framegroup = payload["framegroup16_packed_broad_compare"]
        framegroup["full_trainer_claim"] = True
        framegroup["ratios_by_frame"]["64"]["packed_over_i16x3_total_mean"] = 0.95
        framegroup["ratios_by_frame"]["64"]["packed_over_i16x3_backward_mean"] = 0.95
        framegroup["conclusion"] = "promoted"

        result = _verify_payload(payload)

        self.assertEqual(result["status"], "failed")
        joined = "\n".join(result["failures"])
        self.assertIn("packed broad compare must not claim full trainer coverage", joined)
        self.assertIn("packed broad compare 64f total ratio must preserve speed win", joined)
        self.assertIn("packed broad compare 64f backward ratio must preserve speed win", joined)
        self.assertIn("packed broad compare conclusion must keep broad non-promotion", joined)
        self.assertIn("packed broad compare conclusion must keep scope caveats", joined)

    def test_rejects_framegroup_autograd_smoke_regression(self) -> None:
        payload = copy.deepcopy(self.summary)
        smoke = payload["framegroup16_autograd_smoke"]
        smoke["optimizer_mode"] = "manual-vjp"
        smoke["acceptance"]["gradients_nonzero"] = False
        smoke["row"]["fused_loss_vjp_ms"] = 0.0
        smoke["world_foam_objective_adapter"]["name"] = "DirectMetalWrapper"

        result = _verify_payload(payload)

        self.assertEqual(result["status"], "failed")
        joined = "\n".join(result["failures"])
        self.assertIn("framegroup16 autograd smoke must use optimizer_mode=autograd", joined)
        self.assertIn("framegroup16 autograd smoke must preserve gradients_nonzero", joined)
        self.assertIn("framegroup16 autograd smoke row.fused_loss_vjp_ms is not positive finite", joined)
        self.assertIn("framegroup16 autograd smoke adapter name must be 'WorldFoamFrozenRGBMSEObjective'", joined)

    def test_rejects_framegroup_autograd_speedscale_regression(self) -> None:
        payload = copy.deepcopy(self.summary)
        speedscale = payload["framegroup16_autograd_speedscale"]
        speedscale["real_loaded_frame_counts"] = [16]
        speedscale["repeated_frame_counts"] = [32, 64, 128]
        speedscale["total_scale_first_to_last"] = 3.0
        speedscale["world_foam_objective_adapter_rows_all_match"] = False
        speedscale["by_frame"]["128"]["parameter_update_abs_max"] = 0.0
        speedscale["by_frame"]["128"]["world_foam_objective_adapter"] = None
        speedscale["repeat_scope_by_frame"]["32"] = "synthetic repeated-fixture speed-scaling smoke"

        result = _verify_payload(payload)

        self.assertEqual(result["status"], "failed")
        joined = "\n".join(result["failures"])
        self.assertIn("framegroup16 autograd speedscale real-loaded rows must remain 16/32f", joined)
        self.assertIn("framegroup16 autograd speedscale repeated rows must remain 64/128f", joined)
        self.assertIn("framegroup16 autograd speedscale 32f row must stay real-loaded", joined)
        self.assertIn("framegroup16 autograd speedscale total_scale_first_to_last exceeds guarded threshold", joined)
        self.assertIn("framegroup16 autograd speedscale adapter metadata must be present on every row", joined)
        self.assertIn(
            "framegroup16 autograd speedscale 128f parameter_update_abs_max is not positive finite",
            joined,
        )
        self.assertIn("framegroup16 autograd speedscale 128f missing WorldFoamFrozenRGBMSEObjective adapter metadata", joined)


if __name__ == "__main__":
    unittest.main()
