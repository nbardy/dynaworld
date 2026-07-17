from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import verify_gate4_affine_train_eval as verify_mod


def _args(path: Path, **overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "artifact": path,
        "frame_counts": "2,4,8,16",
        "render_size": 32,
        "site_count": 12,
        "vjp_mode": "direct_atomic_grad_only",
        "min_train_psnr": 8.0,
        "min_heldout_psnr": 8.0,
        "max_total_scale": 2.0,
        "max_backward_scale": 2.5,
        "require_alpha_depth_aux_loss": False,
        "require_median_timing": False,
        "max_total_median_scale": 2.0,
        "max_backward_median_scale": 2.5,
        "max_row_mean_to_median": 2.5,
        "max_row_max_to_median": 8.0,
        "max_tape_storage_scale": 1.10,
        "boundary_ratio_tolerance": 1.0e-7,
        "out_json": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _step_summary(frame: int) -> dict[str, dict[str, float | int]]:
    total_s = {2: 0.010, 4: 0.012, 8: 0.014, 16: 0.016}[frame]
    backward_s = {2: 0.004, 4: 0.005, 8: 0.006, 16: 0.008}[frame]
    return {
        "render": {
            "count": 5,
            "mean_s": total_s * 0.35,
            "median_s": total_s * 0.34,
            "min_s": total_s * 0.30,
            "max_s": total_s * 0.40,
        },
        "loss_eval": {
            "count": 5,
            "mean_s": total_s * 0.05,
            "median_s": total_s * 0.05,
            "min_s": total_s * 0.04,
            "max_s": total_s * 0.06,
        },
        "backward": {
            "count": 5,
            "mean_s": backward_s,
            "median_s": backward_s,
            "min_s": backward_s * 0.90,
            "max_s": backward_s * 1.10,
        },
        "optimizer": {
            "count": 5,
            "mean_s": total_s * 0.08,
            "median_s": total_s * 0.08,
            "min_s": total_s * 0.06,
            "max_s": total_s * 0.10,
        },
        "total": {
            "count": 5,
            "mean_s": total_s,
            "median_s": total_s,
            "min_s": total_s * 0.95,
            "max_s": total_s * 1.05,
        },
    }


def _row(frame: int) -> dict[str, object]:
    return {
        "frame_count": frame,
        "render_size": 32,
        "site_count": 12,
        "status": "ok",
        "final_train_psnr": 12.0 + frame * 0.01,
        "final_heldout_psnr": 11.0 + frame * 0.01,
        "first_grad_abs_sum": 1.25,
        "first_alpha_output_grad_abs_sum": 0.0,
        "first_depth_output_grad_abs_sum": 0.0,
        "parameter_update_abs_max": 0.03,
        "vjp_mode": "direct_atomic_grad_only",
        "loss_terms": {
            "alpha_aux_weight": 0.0,
            "depth_aux_weight": 0.0,
            "alpha_depth_aux_active": False,
        },
        "step_summary": _step_summary(frame),
        "train_mixed_tape_storage_bytes": 1_000_000 + frame * 100,
        "heldout_mixed_tape_storage_bytes": 800_000 + frame * 100,
        "train_explicit_ray_storage_bytes": 49_152 * frame,
        "heldout_explicit_ray_storage_bytes": 24_576 * frame,
        "train_compiled_boundary_test_ratio": 1.0 / float(frame),
        "heldout_compiled_boundary_test_ratio": 1.0 / float(frame),
        "acceptance": {
            "loss_decreased": True,
            "gradients_nonzero": True,
            "parameters_updated": True,
            "candidate_rows_under_metal_cap": True,
            "zero_missing_sample_events": True,
            "outputs_are_finite": True,
            "alpha_depth_aux_vjp_seed_nonzero": True,
        },
    }


def _payload() -> dict[str, object]:
    return {
        "benchmark": "world_foam_lane2_fused_slab_mixed_train_eval_mps",
        "status": "ok",
        "gate": "mixed_num32_den16_affine_moving_ray_site_rgba_train_eval",
        "device": "mps",
        "frame_counts": [2, 4, 8, 16],
        "render_size": 32,
        "time_slabs": 1,
        "layout": "per-track",
        "candidate_order": "slab-mid-depth",
        "gradient_scope": "frozen_geometry_site_rgba_only_mixed_num32_den16_vjp",
        "loss_scope": "rgb_mse_plus_optional_alpha_depth_aux",
        "full_trainer_claim": False,
        "full_geometry_gradient_claim": False,
        "quality_claim": False,
        "vjp_mode": "direct_atomic_grad_only",
        "alpha_aux_weight": 0.0,
        "depth_aux_weight": 0.0,
        "rows": [_row(frame) for frame in (2, 4, 8, 16)],
    }


def _write_payload(tmpdir: str, payload: dict[str, object]) -> Path:
    path = Path(tmpdir) / "artifact.json"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


class VerifyGate4AffineTrainEvalTests(unittest.TestCase):
    def test_accepts_scoped_train_eval_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_payload(tmpdir, _payload())
            result = verify_mod.verify(_args(path))

        self.assertEqual(result["status"], "ok", result["failures"])
        self.assertLess(result["total_mean_scale_first_to_last"], 2.0)
        self.assertLess(result["train_mixed_tape_storage_scale_first_to_last"], 1.10)
        self.assertEqual(result["train_explicit_ray_storage_scale_first_to_last"], 8.0)
        self.assertEqual(result["vjp_mode"], "direct_atomic_grad_only")

    def test_accepts_ownerupdate_train_eval_mode(self) -> None:
        payload = _payload()
        payload["gradient_scope"] = "frozen_geometry_site_rgba_only_mixed_num32_den16_vjp_direct_atomic_grad_only_ownerupdate"
        payload["vjp_mode"] = "direct_atomic_grad_only_ownerupdate"
        for row in payload["rows"]:
            row["vjp_mode"] = "direct_atomic_grad_only_ownerupdate"
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_payload(tmpdir, payload)
            result = verify_mod.verify(_args(path, vjp_mode="direct_atomic_grad_only_ownerupdate"))

        self.assertEqual(result["status"], "ok", result["failures"])
        self.assertEqual(result["vjp_mode"], "direct_atomic_grad_only_ownerupdate")
        self.assertEqual(
            result["gradient_scope"],
            "frozen_geometry_site_rgba_only_mixed_num32_den16_vjp_direct_atomic_grad_only_ownerupdate",
        )

    def test_accepts_fused_mse_rgb_only_train_eval_mode(self) -> None:
        payload = _payload()
        payload["gradient_scope"] = "frozen_geometry_site_rgba_only_mixed_num32_den16_vjp_fused_mse_rgb_only"
        payload["vjp_mode"] = "fused_mse_rgb_only"
        for row in payload["rows"]:
            row["vjp_mode"] = "fused_mse_rgb_only"
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_payload(tmpdir, payload)
            result = verify_mod.verify(_args(path, vjp_mode="fused_mse_rgb_only"))

        self.assertEqual(result["status"], "ok", result["failures"])
        self.assertEqual(result["vjp_mode"], "fused_mse_rgb_only")
        self.assertEqual(
            result["gradient_scope"],
            "frozen_geometry_site_rgba_only_mixed_num32_den16_vjp_fused_mse_rgb_only",
        )

    def test_rejects_wrong_top_level_vjp_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_payload(tmpdir, _payload())
            result = verify_mod.verify(_args(path, vjp_mode="direct_atomic_grad_only_ownerupdate"))

        self.assertEqual(result["status"], "failed")
        self.assertIn("vjp_mode", "\n".join(result["failures"]))

    def test_rejects_row_vjp_mode_drift(self) -> None:
        payload = _payload()
        payload["rows"][2]["vjp_mode"] = "direct_atomic_grad_only_ownerupdate"
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_payload(tmpdir, payload)
            result = verify_mod.verify(_args(path))

        self.assertEqual(result["status"], "failed")
        self.assertIn("row vjp_mode", "\n".join(result["failures"]))

    def test_accepts_required_alpha_depth_aux_loss(self) -> None:
        payload = _payload()
        payload["alpha_aux_weight"] = 0.01
        payload["depth_aux_weight"] = 0.02
        for row in payload["rows"]:
            row["first_alpha_output_grad_abs_sum"] = 0.5
            row["first_depth_output_grad_abs_sum"] = 0.25
            row["loss_terms"] = {
                "alpha_aux_weight": 0.01,
                "depth_aux_weight": 0.02,
                "alpha_depth_aux_active": True,
            }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_payload(tmpdir, payload)
            result = verify_mod.verify(_args(path, require_alpha_depth_aux_loss=True))

        self.assertEqual(result["status"], "ok", result["failures"])
        self.assertEqual(result["alpha_aux_weight"], 0.01)
        self.assertEqual(result["depth_aux_weight"], 0.02)

    def test_rejects_missing_alpha_depth_aux_seed_when_required(self) -> None:
        payload = _payload()
        payload["alpha_aux_weight"] = 0.01
        payload["depth_aux_weight"] = 0.02
        for row in payload["rows"]:
            row["first_alpha_output_grad_abs_sum"] = 0.5
            row["first_depth_output_grad_abs_sum"] = 0.25
            row["loss_terms"] = {
                "alpha_aux_weight": 0.01,
                "depth_aux_weight": 0.02,
                "alpha_depth_aux_active": True,
            }
        payload["rows"][1]["first_depth_output_grad_abs_sum"] = 0.0
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_payload(tmpdir, payload)
            result = verify_mod.verify(_args(path, require_alpha_depth_aux_loss=True))

        self.assertEqual(result["status"], "failed")
        self.assertIn("first_depth_output_grad_abs_sum", "\n".join(result["failures"]))

    def test_rejects_promoted_scope_claims(self) -> None:
        payload = _payload()
        payload["full_trainer_claim"] = True
        payload["quality_claim"] = True
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_payload(tmpdir, payload)
            result = verify_mod.verify(_args(path))

        self.assertEqual(result["status"], "failed")
        failures = "\n".join(result["failures"])
        self.assertIn("full_trainer_claim must be false", failures)
        self.assertIn("quality_claim must be false", failures)

    def test_rejects_false_acceptance_flag(self) -> None:
        payload = _payload()
        payload["rows"][1]["acceptance"]["loss_decreased"] = False
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_payload(tmpdir, payload)
            result = verify_mod.verify(_args(path))

        self.assertEqual(result["status"], "failed")
        self.assertIn("acceptance loss_decreased is not true", "\n".join(result["failures"]))

    def test_rejects_low_psnr(self) -> None:
        payload = _payload()
        payload["rows"][2]["final_heldout_psnr"] = 7.5
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_payload(tmpdir, payload)
            result = verify_mod.verify(_args(path))

        self.assertEqual(result["status"], "failed")
        self.assertIn("final_heldout_psnr", "\n".join(result["failures"]))

    def test_rejects_wrong_boundary_ratio(self) -> None:
        payload = _payload()
        payload["rows"][3]["train_compiled_boundary_test_ratio"] = 0.25
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_payload(tmpdir, payload)
            result = verify_mod.verify(_args(path))

        self.assertEqual(result["status"], "failed")
        self.assertIn("train_compiled_boundary_test_ratio", "\n".join(result["failures"]))

    def test_rejects_nonsublinear_total_or_backward_scale(self) -> None:
        payload = _payload()
        payload["rows"][3]["step_summary"]["total"]["mean_s"] = 0.090
        payload["rows"][3]["step_summary"]["backward"]["mean_s"] = 0.040
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_payload(tmpdir, payload)
            result = verify_mod.verify(_args(path))

        self.assertEqual(result["status"], "failed")
        failures = "\n".join(result["failures"])
        self.assertIn("total mean scale", failures)
        self.assertIn("backward mean scale", failures)

    def test_rejects_mixed_tape_storage_growth(self) -> None:
        payload = _payload()
        payload["rows"][3]["train_mixed_tape_storage_bytes"] = 2_000_000
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_payload(tmpdir, payload)
            result = verify_mod.verify(_args(path))

        self.assertEqual(result["status"], "failed")
        self.assertIn("train mixed tape storage scale", "\n".join(result["failures"]))

    def test_rejects_missing_median_timing_when_required(self) -> None:
        payload = _payload()
        del payload["rows"][0]["step_summary"]["total"]["median_s"]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_payload(tmpdir, payload)
            result = verify_mod.verify(_args(path, require_median_timing=True))

        self.assertEqual(result["status"], "failed")
        self.assertIn("step_summary.total.median_s", "\n".join(result["failures"]))

    def test_rejects_spiky_median_timing_when_required(self) -> None:
        payload = _payload()
        payload["rows"][1]["step_summary"]["total"]["max_s"] = 0.100
        payload["rows"][1]["step_summary"]["backward"]["mean_s"] = 0.030
        payload["rows"][1]["step_summary"]["backward"]["median_s"] = 0.004
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_payload(tmpdir, payload)
            result = verify_mod.verify(_args(path, require_median_timing=True))

        self.assertEqual(result["status"], "failed")
        failures = "\n".join(result["failures"])
        self.assertIn("step_summary.total max/median", failures)
        self.assertIn("step_summary.backward mean/median", failures)

    def test_rejects_non_linear_explicit_ray_storage_scale(self) -> None:
        payload = _payload()
        payload["rows"][3]["heldout_explicit_ray_storage_bytes"] = 24_576 * 8
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_payload(tmpdir, payload)
            result = verify_mod.verify(_args(path))

        self.assertEqual(result["status"], "failed")
        self.assertIn("heldout explicit ray storage scale", "\n".join(result["failures"]))


if __name__ == "__main__":
    unittest.main()
