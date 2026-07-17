from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import verify_worldfoam_rebuilt_native_smokes as verify_mod


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _basic_payload(*, status: str = "ok") -> dict[str, object]:
    return {
        "status": status,
        "benchmark": "unit_benchmark",
        "quality_claim": False,
        "training_claim": False,
        "acceptance": {"unit_acceptance": True},
    }


def _slab_owner_payload(*, status: str = "ok", layout: str = "per-track") -> dict[str, object]:
    return {
        "status": status,
        "benchmark": "world_foam_lane2_fused_slab_affine_realray_mps_smoke",
        "quality_claim": False,
        "training_claim": False,
        "layout": layout,
        "include_ownerupdate": True,
        "include_vjp": True,
        "acceptance": {
            "ownerupdate_matches_explicit_realray": True,
            "mixed_vjp_direct_grad_only_ownerupdate_gradients_finite": True,
        },
        "ownerupdate_diagnostics": {
            "checked": True,
            "max_error": 1.0e-5,
            "within_strict_tolerance": True,
        },
        "mixed_vjp_direct_grad_only_ownerupdate_diagnostics": {
            "checked": True,
            "within_grad_tolerance": True,
        },
    }


def _known_invalid_payload() -> dict[str, object]:
    return {
        "status": "failed",
        "benchmark": "world_foam_lane2_fused_slab_affine_realray_mps_smoke",
        "layout": "tiled",
        "include_ownerupdate": True,
        "acceptance": {
            "ownerupdate_matches_explicit_realray": False,
            "mixed_vjp_direct_grad_only_ownerupdate_gradients_finite": False,
        },
        "ownerupdate_diagnostics": {
            "checked": True,
            "max_error": None,
            "within_strict_tolerance": False,
        },
    }


class VerifyWorldFoamRebuiltNativeSmokesTests(unittest.TestCase):
    def test_real_rebuilt_native_smoke_bundle_passes(self) -> None:
        result = verify_mod.verify()

        self.assertEqual(result["status"], "ok", result["failures"])
        self.assertEqual(result["required_count"], 7)
        self.assertEqual(
            result["known_invalid_tiled_ownerupdate"]["classification"],
            "expected_invalid_tiled_ownerupdate",
        )

    def test_rejects_failed_required_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "required.json"
            _write_json(path, _basic_payload(status="failed"))
            specs = ({"label": "unit", "path": path, "benchmark": "unit_benchmark"},)

            result = verify_mod.verify(specs=specs, known_invalid_artifact=Path(tmpdir) / "absent.json")

        self.assertEqual(result["status"], "failed")
        self.assertIn("status is 'failed'", "\n".join(result["failures"]))

    def test_validates_required_ownerupdate_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "owner.json"
            _write_json(path, _slab_owner_payload())
            specs = (
                {
                    "label": "owner",
                    "path": path,
                    "benchmark": "world_foam_lane2_fused_slab_affine_realray_mps_smoke",
                    "layout": "per-track",
                    "include_ownerupdate": True,
                    "include_vjp": True,
                    "ownerupdate_checked": True,
                },
            )

            result = verify_mod.verify(specs=specs, known_invalid_artifact=Path(tmpdir) / "absent.json")

        self.assertEqual(result["status"], "ok", result["failures"])

    def test_classifies_expected_invalid_tiled_ownerupdate_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            required = Path(tmpdir) / "required.json"
            invalid = Path(tmpdir) / "invalid.json"
            _write_json(required, _basic_payload())
            _write_json(invalid, _known_invalid_payload())
            specs = ({"label": "unit", "path": required, "benchmark": "unit_benchmark"},)

            result = verify_mod.verify(specs=specs, known_invalid_artifact=invalid)

        self.assertEqual(result["status"], "ok", result["failures"])
        self.assertEqual(
            result["known_invalid_tiled_ownerupdate"]["classification"],
            "expected_invalid_tiled_ownerupdate",
        )

    def test_rejects_unexpected_shape_at_known_invalid_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            required = Path(tmpdir) / "required.json"
            invalid = Path(tmpdir) / "invalid.json"
            _write_json(required, _basic_payload())
            _write_json(invalid, _slab_owner_payload(status="ok", layout="per-track"))
            specs = ({"label": "unit", "path": required, "benchmark": "unit_benchmark"},)

            result = verify_mod.verify(specs=specs, known_invalid_artifact=invalid)

        self.assertEqual(result["status"], "failed")
        self.assertIn("known invalid artifact no longer has the expected failed shape", "\n".join(result["failures"]))


if __name__ == "__main__":
    unittest.main()
