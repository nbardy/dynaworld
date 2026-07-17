from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import verify_gate4_affine_tape_bridge as verify_mod


def _args(path: Path, **overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "artifact": path,
        "frame_counts": "2,4,8,16",
        "render_size": 32,
        "site_count": 12,
        "max_mixed_error": 5.0e-4,
        "max_affine_residual": 1.0e-5,
        "max_mixed_storage_scale": 1.10,
        "boundary_ratio_tolerance": 1.0e-7,
        "require_ownerupdate": False,
        "require_vjp_seed_mode": None,
        "require_coeff16_rejected": True,
        "out_json": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _row(frame: int) -> dict[str, object]:
    return {
        "frames": frame,
        "render_size": 32,
        "site_count": 12,
        "missing_sample_events": 0,
        "max_candidates_per_row": 32,
        "linear_fit": {
            "max_origin_residual": 0.0,
            "max_direction_residual": 0.0,
        },
        "compiled_boundary_test_ratio": 1.0 / float(frame),
        "total_mixed_fused_storage_bytes": 1_000_000 - frame * 1_000,
        "explicit_ray_storage_bytes": 49_152 * frame,
    }


def _payload() -> dict[str, object]:
    return {
        "benchmark": "world_foam_lane2_fused_slab_affine_realray_mps_smoke",
        "status": "ok",
        "frame_counts": [2, 4, 8, 16],
        "render_size": 32,
        "time_slabs": 1,
        "layout": "per-track",
        "candidate_order": "slab-mid-depth",
        "include_vjp": True,
        "gradient_scope": "mixed_num32_den16_site_rgba_vjp_rgb_seed",
        "vjp_seed_mode": "rgb",
        "include_ownerupdate": False,
        "quality_claim": False,
        "training_claim": False,
        "max_realray_boundaries": 128,
        "tolerance": 5.0e-4,
        "mixed_max_error": 1.0e-4,
        "coeff16_diagnostics": {
            "max_error": 0.02,
            "within_approx_tolerance": False,
        },
        "mixed_vjp_direct_diagnostics": {
            "within_grad_tolerance": True,
        },
        "mixed_vjp_direct_grad_only_diagnostics": {
            "within_grad_tolerance": True,
        },
        "mixed_vjp_direct_rgb_only_diagnostics": {
            "has_expected_seed_behavior": True,
        },
        "mixed_vjp_direct_track_diagnostics": {
            "within_grad_tolerance": True,
        },
        "autograd_vjp_diagnostics": {
            "general_modes_match_reduce": True,
            "rgb_only_has_expected_seed_behavior": True,
        },
        "ownerupdate_diagnostics": {
            "checked": False,
            "max_error": None,
            "within_strict_tolerance": None,
        },
        "mixed_vjp_direct_grad_only_ownerupdate_diagnostics": {
            "checked": False,
            "max_grad_delta_vs_reduce": None,
            "max_grad_rel_delta_vs_reduce": None,
            "within_grad_tolerance": None,
        },
        "acceptance": {
            "zero_missing_sample_events": True,
            "candidate_rows_under_metal_cap": True,
            "matches_explicit_realray": True,
            "mixed_matches_explicit_realray": True,
            "mixed_vjp_direct_matches_reduce_grad": True,
        },
        "rows": [_row(frame) for frame in (2, 4, 8, 16)],
    }


def _write_payload(tmpdir: str, payload: dict[str, object]) -> Path:
    path = Path(tmpdir) / "artifact.json"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


class VerifyGate4AffineTapeBridgeTests(unittest.TestCase):
    def test_accepts_scoped_gate4_bridge_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_payload(tmpdir, _payload())
            result = verify_mod.verify(_args(path))

        self.assertEqual(result["status"], "ok", result["failures"])
        self.assertTrue(result["coeff16_rejected"])
        self.assertLess(result["mixed_storage_scale_first_to_last"], 1.10)
        self.assertEqual(result["ownerupdate_scope"]["ownerupdate_checked"], False)

    def test_rejects_ownerupdate_acceptance_when_ownerupdate_was_not_checked(self) -> None:
        payload = _payload()
        payload["acceptance"]["mixed_vjp_direct_grad_only_ownerupdate_matches_reduce_grad"] = True
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_payload(tmpdir, payload)
            result = verify_mod.verify(_args(path))

        self.assertEqual(result["status"], "failed")
        self.assertIn("must not appear when include_ownerupdate is false", "\n".join(result["failures"]))

    def test_rejects_promoted_coeff16_path(self) -> None:
        payload = _payload()
        payload["coeff16_diagnostics"]["within_approx_tolerance"] = True
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_payload(tmpdir, payload)
            result = verify_mod.verify(_args(path))

        self.assertEqual(result["status"], "failed")
        self.assertIn("pure coeff16 path must remain rejected", "\n".join(result["failures"]))

    def test_rejects_missing_sample_events(self) -> None:
        payload = _payload()
        payload["rows"][2]["missing_sample_events"] = 1
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_payload(tmpdir, payload)
            result = verify_mod.verify(_args(path))

        self.assertEqual(result["status"], "failed")
        self.assertIn("missing_sample_events must be zero", "\n".join(result["failures"]))

    def test_rejects_wrong_boundary_ratio(self) -> None:
        payload = _payload()
        payload["rows"][3]["compiled_boundary_test_ratio"] = 0.25
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_payload(tmpdir, payload)
            result = verify_mod.verify(_args(path))

        self.assertEqual(result["status"], "failed")
        self.assertIn("compiled boundary ratio", "\n".join(result["failures"]))

    def test_accepts_ownerupdate_scope_when_explicitly_checked(self) -> None:
        payload = _payload()
        payload["include_ownerupdate"] = True
        payload["ownerupdate_diagnostics"] = {
            "checked": True,
            "max_error": 1.0e-6,
            "within_strict_tolerance": True,
        }
        payload["mixed_vjp_direct_grad_only_ownerupdate_diagnostics"] = {
            "checked": True,
            "max_grad_delta_vs_reduce": 1.0e-6,
            "max_grad_rel_delta_vs_reduce": 1.0e-7,
            "within_grad_tolerance": True,
        }
        payload["acceptance"]["ownerupdate_matches_explicit_realray"] = True
        payload["acceptance"]["mixed_vjp_direct_grad_only_ownerupdate_matches_reduce_grad"] = True
        payload["acceptance"]["mixed_vjp_direct_grad_only_ownerupdate_gradients_finite"] = True
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_payload(tmpdir, payload)
            result = verify_mod.verify(_args(path))

        self.assertEqual(result["status"], "ok", result["failures"])
        self.assertTrue(result["ownerupdate_scope"]["ownerupdate_checked"])
        self.assertTrue(result["ownerupdate_scope"]["ownerupdate_vjp_checked"])

    def test_rejects_missing_ownerupdate_when_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_payload(tmpdir, _payload())
            result = verify_mod.verify(_args(path, require_ownerupdate=True))

        self.assertEqual(result["status"], "failed")
        self.assertIn("ownerupdate must be included", "\n".join(result["failures"]))

    def test_rejects_wrong_vjp_seed_mode_when_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_payload(tmpdir, _payload())
            result = verify_mod.verify(_args(path, require_vjp_seed_mode="rgba-depth"))

        self.assertEqual(result["status"], "failed")
        self.assertIn("vjp_seed_mode", "\n".join(result["failures"]))

    def test_accepts_required_vjp_seed_mode(self) -> None:
        payload = _payload()
        payload["gradient_scope"] = "mixed_num32_den16_site_rgba_vjp_rgba-depth_seed"
        payload["vjp_seed_mode"] = "rgba-depth"
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_payload(tmpdir, payload)
            result = verify_mod.verify(_args(path, require_vjp_seed_mode="rgba-depth"))

        self.assertEqual(result["status"], "ok", result["failures"])
        self.assertEqual(result["vjp_seed_mode"], "rgba-depth")


if __name__ == "__main__":
    unittest.main()
