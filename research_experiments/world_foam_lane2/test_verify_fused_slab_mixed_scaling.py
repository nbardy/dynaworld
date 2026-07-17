from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import verify_fused_slab_mixed_scaling as verify_mod


def _args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "frame_counts": "2,4,8,16",
        "best_mode": "direct_atomic_grad_only",
        "required_modes": ",".join(verify_mod.DEFAULT_REQUIRED_MODES),
        "max_total_scale": 1.6,
        "max_render_scale": 1.35,
        "max_backward_scale": 2.2,
        "max_psnr_spread": 1.0e-3,
        "max_realray_boundaries": verify_mod.DEFAULT_MAX_REALRAY_BOUNDARIES,
        "max_vjp_grad_rel_error": 2.0e-6,
        "train_eval_json": None,
        "smoke_json": None,
        "framegroup_frame_counts": "16,32,64,128",
        "framegroup_lossreduce_json": verify_mod.DEFAULT_FRAMEGROUP_LOSSREDUCE_ARTIFACT,
        "framegroup_128only_json": verify_mod.DEFAULT_FRAMEGROUP_128ONLY_ARTIFACT,
        "framegroup_max_total_scale": 1.5,
        "framegroup_max_backward_scale": 1.65,
        "framegroup_max_storage_scale": 1.10,
        "framegroup_max_mixed_128_total_max_ms": 7.5,
        "framegroup_max_128only_total_median_ms": 4.5,
        "framegroup_max_128only_total_max_ms": 8.5,
        "framegroup_max_128only_backward_median_ms": 3.75,
        "compare_frame_counts": "16,32,64,128",
        "compare_smoke_json": verify_mod.DEFAULT_FRAMEGROUP_COMPARE_SMOKE_ARTIFACT,
        "compare_max_framegroup_to_endpoint_total_16f": 0.75,
        "compare_max_framegroup_to_endpoint_backward_16f": 0.95,
        "compare_max_psnr_delta": 1.0e-3,
        "compare_max_framegroup_storage_vs_full_16f": 0.15,
        "compare_max_framegroup_total_scale": 3.25,
        "compare_max_framegroup_backward_scale": 3.75,
        "compare_max_framegroup_storage_scale": 1.10,
        "compare_max_framegroup_to_endpoint_total_all_frames": 0.75,
        "compare_max_psnr_delta_all_frames": 5.0e-3,
        "compare_render_size": 32,
        "compare_site_count": 12,
        "real32_frame_counts": "16,32",
        "real32_compare_json": verify_mod.DEFAULT_FRAMEGROUP_REAL32_COMPARE_ARTIFACT,
        "real32_max_framegroup_to_endpoint_total_all_frames": 0.75,
        "real32_max_framegroup_to_endpoint_backward_all_frames": 0.95,
        "real32_max_psnr_delta_all_frames": 1.0e-3,
        "real32_max_framegroup_total_scale": 2.25,
        "real32_max_framegroup_backward_scale": 2.35,
        "real32_max_framegroup_storage_scale": 1.10,
        "real32_render_size": 32,
        "real32_site_count": 12,
        "i16x4_compare_frame_counts": "16,32",
        "i16x4_compare_json": verify_mod.DEFAULT_FRAMEGROUP_I16X4_COMPARE_ARTIFACT,
        "i16x4_max_over_i16x3_total_mean_ratio": 1.05,
        "i16x4_max_over_i16x3_backward_mean_ratio": 1.05,
        "i16x4_max_over_i16x3_storage_ratio": 1.08,
        "i16x4_max_psnr_delta": 1.0e-4,
        "out_json": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _write_temp_json(tmpdir: str, name: str, payload: dict[str, object]) -> Path:
    path = Path(tmpdir) / name
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


class VerifyFusedSlabMixedScalingTests(unittest.TestCase):
    def setUp(self) -> None:
        required = [
            *verify_mod.DEFAULT_TRAIN_EVAL_ARTIFACTS,
            *verify_mod.DEFAULT_SMOKE_ARTIFACTS,
            verify_mod.DEFAULT_FRAMEGROUP_LOSSREDUCE_ARTIFACT,
            verify_mod.DEFAULT_FRAMEGROUP_128ONLY_ARTIFACT,
            verify_mod.DEFAULT_FRAMEGROUP_COMPARE_SMOKE_ARTIFACT,
            verify_mod.DEFAULT_FRAMEGROUP_REAL32_COMPARE_ARTIFACT,
            verify_mod.DEFAULT_FRAMEGROUP_I16X4_COMPARE_ARTIFACT,
        ]
        missing = [path for path in required if not path.exists()]
        if missing:
            self.skipTest(f"missing generated verifier fixtures: {missing}")

    def test_current_artifacts_are_valid(self) -> None:
        result = verify_mod.verify(_args())
        self.assertEqual(result["status"], "ok", result["failures"])
        framegroup = result["framegroup_lossreduce"]
        self.assertLess(framegroup["total_scale_first_to_last"], 1.5)
        self.assertLess(framegroup["mixed_128_total_max_ms"], 7.5)
        compare = result["framegroup_compare_smoke"]
        self.assertLess(compare["total_ratio_16f"], 0.75)
        self.assertLess(compare["ratios_by_frame"]["128"]["total"], 0.75)
        self.assertLess(compare["total_scale_first_to_last"], 3.25)
        self.assertEqual(compare["render_size"], 32)
        self.assertEqual(compare["site_count"], 12)
        self.assertEqual(compare["loaded_frame_count"], 16)
        self.assertEqual(compare["real_loaded_frame_counts"], [16])
        self.assertEqual(compare["repeated_frame_counts"], [32, 64, 128])
        self.assertIn("not a stable benchmark", compare["scope"])
        real32 = result["framegroup_real32_compare"]
        self.assertEqual(real32["real_loaded_frame_counts"], [16, 32])
        self.assertEqual(real32["repeated_frame_counts"], [])
        self.assertLess(real32["ratios_by_frame"]["32"]["total"], 0.75)
        self.assertTrue(real32["total_sublinear_real_frames"])
        self.assertTrue(real32["backward_sublinear_real_frames"])
        self.assertTrue(real32["real_frame_sublinear_claim"])
        i16x4 = result["framegroup_i16x4_compare"]
        self.assertFalse(i16x4["i16x4_speed_promotion_candidate"])
        self.assertEqual(i16x4["mode_statuses"]["endpoint-record-delta-replace-coeff16-i16x3-framegroup16-fused-mse"], "ok")
        self.assertEqual(
            i16x4["mode_statuses"]["endpoint-record-delta-replace-coeff16-i16x4-framegroup16-fused-mse"],
            "failed",
        )
        self.assertLess(i16x4["max_i16x4_over_i16x3_total_mean_ratio"], 1.05)
        self.assertLess(i16x4["max_i16x4_over_i16x3_backward_mean_ratio"], 1.05)
        self.assertLess(i16x4["max_i16x4_over_i16x3_storage_ratio"], 1.08)

    def test_rejects_lossreduce_mixed_128_outlier_regression(self) -> None:
        payload = copy.deepcopy(
            json.loads(verify_mod.DEFAULT_FRAMEGROUP_LOSSREDUCE_ARTIFACT.read_text(encoding="utf-8"))
        )
        payload["rows"][-1]["step_summary"]["total"]["max_s"] = 0.020
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_path = _write_temp_json(tmpdir, "bad_framegroup.json", payload)
            result = verify_mod.verify(_args(framegroup_lossreduce_json=bad_path))

        self.assertEqual(result["status"], "failed")
        self.assertIn("128f mixed-sweep total max", "\n".join(result["failures"]))

    def test_rejects_lossreduce_128only_median_regression(self) -> None:
        payload = copy.deepcopy(
            json.loads(verify_mod.DEFAULT_FRAMEGROUP_128ONLY_ARTIFACT.read_text(encoding="utf-8"))
        )
        payload["rows"][0]["step_summary"]["total"]["median_s"] = 0.006
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_path = _write_temp_json(tmpdir, "bad_128only.json", payload)
            result = verify_mod.verify(_args(framegroup_128only_json=bad_path))

        self.assertEqual(result["status"], "failed")
        self.assertIn("128-only total median", "\n".join(result["failures"]))

    def test_rejects_compare_smoke_speed_regression(self) -> None:
        payload = copy.deepcopy(
            json.loads(verify_mod.DEFAULT_FRAMEGROUP_COMPARE_SMOKE_ARTIFACT.read_text(encoding="utf-8"))
        )
        payload["ratios"]["delta_framegroup16_to_endpoint_total_16f"] = 0.90
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_path = _write_temp_json(tmpdir, "bad_compare.json", payload)
            result = verify_mod.verify(_args(compare_smoke_json=bad_path))

        self.assertEqual(result["status"], "failed")
        self.assertIn("framegroup compare total ratio", "\n".join(result["failures"]))

    def test_rejects_compare_smoke_all_frame_speed_regression(self) -> None:
        payload = copy.deepcopy(
            json.loads(verify_mod.DEFAULT_FRAMEGROUP_COMPARE_SMOKE_ARTIFACT.read_text(encoding="utf-8"))
        )
        mode = "endpoint-record-delta-replace-coeff16-i16x3-framegroup16-fused-mse"
        payload["summary_by_frame"]["128"][mode]["total_ms"] = (
            payload["summary_by_frame"]["128"]["endpoint-run"]["total_ms"] * 0.90
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_path = _write_temp_json(tmpdir, "bad_compare_128.json", payload)
            result = verify_mod.verify(_args(compare_smoke_json=bad_path))

        self.assertEqual(result["status"], "failed")
        self.assertIn("frame 128 framegroup total ratio", "\n".join(result["failures"]))

    def test_rejects_compare_smoke_scale_regression(self) -> None:
        payload = copy.deepcopy(
            json.loads(verify_mod.DEFAULT_FRAMEGROUP_COMPARE_SMOKE_ARTIFACT.read_text(encoding="utf-8"))
        )
        fg = payload["results"]["endpoint-record-delta-replace-coeff16-i16x3-framegroup16-fused-mse"]
        fg["total_step_scale_first_to_last"] = 3.5
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_path = _write_temp_json(tmpdir, "bad_compare_scale.json", payload)
            result = verify_mod.verify(_args(compare_smoke_json=bad_path))

        self.assertEqual(result["status"], "failed")
        self.assertIn("framegroup total scale", "\n".join(result["failures"]))

    def test_rejects_compare_smoke_all_frame_psnr_regression(self) -> None:
        payload = copy.deepcopy(
            json.loads(verify_mod.DEFAULT_FRAMEGROUP_COMPARE_SMOKE_ARTIFACT.read_text(encoding="utf-8"))
        )
        mode = "endpoint-record-delta-replace-coeff16-i16x3-framegroup16-fused-mse"
        payload["summary_by_frame"]["64"][mode]["heldout_psnr"] -= 0.02
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_path = _write_temp_json(tmpdir, "bad_compare_all_frame_psnr.json", payload)
            result = verify_mod.verify(_args(compare_smoke_json=bad_path))

        self.assertEqual(result["status"], "failed")
        self.assertIn("frame 64 framegroup PSNR delta", "\n".join(result["failures"]))

    def test_rejects_compare_smoke_wrong_render_size(self) -> None:
        payload = copy.deepcopy(
            json.loads(verify_mod.DEFAULT_FRAMEGROUP_COMPARE_SMOKE_ARTIFACT.read_text(encoding="utf-8"))
        )
        fg = payload["results"]["endpoint-record-delta-replace-coeff16-i16x3-framegroup16-fused-mse"]
        fg["render_size"] = 16
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_path = _write_temp_json(tmpdir, "bad_compare_render_size.json", payload)
            result = verify_mod.verify(_args(compare_smoke_json=bad_path))

        self.assertEqual(result["status"], "failed")
        self.assertIn("framegroup result render_size must be 32", "\n".join(result["failures"]))

    def test_rejects_compare_smoke_missing_repeated_row_marker(self) -> None:
        payload = copy.deepcopy(
            json.loads(verify_mod.DEFAULT_FRAMEGROUP_COMPARE_SMOKE_ARTIFACT.read_text(encoding="utf-8"))
        )
        fg = payload["results"]["endpoint-record-delta-replace-coeff16-i16x3-framegroup16-fused-mse"]
        for row in fg["rows"]:
            if row["frame_count"] == 32:
                row["repeat_loaded_frames"] = False
                row["repeat_loaded_frames_scope"] = "real loaded frame count"
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_path = _write_temp_json(tmpdir, "bad_compare_repeat_marker.json", payload)
            result = verify_mod.verify(_args(compare_smoke_json=bad_path))

        self.assertEqual(result["status"], "failed")
        self.assertIn("frame 32 must be marked as repeated-fixture", "\n".join(result["failures"]))

    def test_rejects_compare_smoke_missing_scope_boundary(self) -> None:
        payload = copy.deepcopy(
            json.loads(verify_mod.DEFAULT_FRAMEGROUP_COMPARE_SMOKE_ARTIFACT.read_text(encoding="utf-8"))
        )
        payload["scope"] = "paired current-process smoke"
        payload["repeat_loaded_frames"] = False
        fg = payload["results"]["endpoint-record-delta-replace-coeff16-i16x3-framegroup16-fused-mse"]
        fg["quality_claim"] = True
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_path = _write_temp_json(tmpdir, "bad_scope.json", payload)
            result = verify_mod.verify(_args(compare_smoke_json=bad_path))

        self.assertEqual(result["status"], "failed")
        joined = "\n".join(result["failures"])
        self.assertIn("repeated-loaded-frame scope", joined)
        self.assertIn("scope must keep benchmark and repeated-fixture caveats", joined)
        self.assertIn("must not claim full trainer or quality", joined)

    def test_rejects_compare_smoke_psnr_regression(self) -> None:
        payload = copy.deepcopy(
            json.loads(verify_mod.DEFAULT_FRAMEGROUP_COMPARE_SMOKE_ARTIFACT.read_text(encoding="utf-8"))
        )
        mode = "endpoint-record-delta-replace-coeff16-i16x3-framegroup16-fused-mse"
        payload["summary_16f"][mode]["heldout_psnr"] -= 0.05
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_path = _write_temp_json(tmpdir, "bad_psnr.json", payload)
            result = verify_mod.verify(_args(compare_smoke_json=bad_path))

        self.assertEqual(result["status"], "failed")
        self.assertIn("framegroup PSNR delta", "\n".join(result["failures"]))

    def test_rejects_real32_repeated_frame_regression(self) -> None:
        payload = copy.deepcopy(
            json.loads(verify_mod.DEFAULT_FRAMEGROUP_REAL32_COMPARE_ARTIFACT.read_text(encoding="utf-8"))
        )
        mode = "endpoint-record-delta-replace-coeff16-i16x3-framegroup16-fused-mse"
        payload["repeat_loaded_frames"] = True
        for row in payload["results"][mode]["rows"]:
            if row["frame_count"] == 32:
                row["loaded_frame_count"] = 16
                row["repeat_loaded_frames"] = True
                row["repeat_loaded_frames_scope"] = "synthetic repeated-fixture speed-scaling smoke"
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_path = _write_temp_json(tmpdir, "bad_real32_repeat.json", payload)
            result = verify_mod.verify(_args(real32_compare_json=bad_path))

        self.assertEqual(result["status"], "failed")
        joined = "\n".join(result["failures"])
        self.assertIn("real32 compare must not use repeated loaded frames", joined)
        self.assertIn("real32 frame 32 must be loaded as itself", joined)

    def test_rejects_real32_speed_regression(self) -> None:
        payload = copy.deepcopy(
            json.loads(verify_mod.DEFAULT_FRAMEGROUP_REAL32_COMPARE_ARTIFACT.read_text(encoding="utf-8"))
        )
        mode = "endpoint-record-delta-replace-coeff16-i16x3-framegroup16-fused-mse"
        payload["summary_by_frame"]["32"][mode]["total_ms"] = (
            payload["summary_by_frame"]["32"]["endpoint-run"]["total_ms"] * 0.90
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_path = _write_temp_json(tmpdir, "bad_real32_speed.json", payload)
            result = verify_mod.verify(_args(real32_compare_json=bad_path))

        self.assertEqual(result["status"], "failed")
        self.assertIn("real32 frame 32 framegroup total ratio", "\n".join(result["failures"]))

    def test_rejects_i16x4_accidental_promotion(self) -> None:
        payload = copy.deepcopy(
            json.loads(verify_mod.DEFAULT_FRAMEGROUP_I16X4_COMPARE_ARTIFACT.read_text(encoding="utf-8"))
        )
        payload["summary"]["i16x4_speed_promotion_candidate"] = True
        payload["summary"]["i16x4_total_sublinear"] = True
        payload["summary"]["i16x4_backward_sublinear"] = True
        payload["mode_statuses"]["endpoint-record-delta-replace-coeff16-i16x4-framegroup16-fused-mse"] = "ok"
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_path = _write_temp_json(tmpdir, "bad_i16x4_promotion.json", payload)
            result = verify_mod.verify(_args(i16x4_compare_json=bad_path))

        self.assertEqual(result["status"], "failed")
        joined = "\n".join(result["failures"])
        self.assertIn("i16x4 mode status must stay failed", joined)
        self.assertIn("i16x4 must not be marked as a speed promotion candidate", joined)
        self.assertIn("i16x4 total sublinear flag must remain false", joined)

    def test_rejects_i16x4_storage_regression(self) -> None:
        payload = copy.deepcopy(
            json.loads(verify_mod.DEFAULT_FRAMEGROUP_I16X4_COMPARE_ARTIFACT.read_text(encoding="utf-8"))
        )
        payload["summary"]["ratios_by_frame"]["32"]["i16x4_over_i16x3_storage"] = 1.20
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_path = _write_temp_json(tmpdir, "bad_i16x4_storage.json", payload)
            result = verify_mod.verify(_args(i16x4_compare_json=bad_path))

        self.assertEqual(result["status"], "failed")
        self.assertIn("frame 32 i16x4 storage ratio", "\n".join(result["failures"]))


if __name__ == "__main__":
    unittest.main()
