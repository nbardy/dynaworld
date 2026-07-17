from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import verify_framegroup16_timing_robust as verify_mod


def _summary(mean_ms: float, median_ms: float, max_ms: float) -> dict[str, float]:
    return {
        "mean_s": mean_ms / 1000.0,
        "median_s": median_ms / 1000.0,
        "max_s": max_ms / 1000.0,
        "p90_s": max_ms / 1000.0,
    }


def _row(
    frame_count: int,
    total_ms: float,
    backward_ms: float,
    storage: int = 1000,
    topology_storage: int | None = None,
    coeff_storage: int | None = None,
    mps_resident_storage: int | None = None,
    mps_resident_noncoeff_storage: int | None = None,
    mps_resident_coeff_storage: int | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "frame_count": frame_count,
        "status": "ok",
        "step_summary": {
            "total": _summary(total_ms, total_ms, total_ms * 1.2),
            "backward": _summary(backward_ms, backward_ms, backward_ms * 1.2),
        },
        "train_selected_tape_storage_bytes": storage,
        "final_heldout_psnr": 14.0,
    }
    if topology_storage is not None:
        row["train_selected_tape_topology_storage_bytes"] = topology_storage
    if coeff_storage is not None:
        row["train_endpoint_record_coeff_storage_bytes"] = coeff_storage
    if mps_resident_storage is not None:
        row["train_selected_tape_mps_resident_storage_bytes"] = mps_resident_storage
    if mps_resident_noncoeff_storage is not None:
        row["train_selected_tape_mps_resident_noncoeff_storage_bytes"] = mps_resident_noncoeff_storage
    if mps_resident_coeff_storage is not None:
        row["train_endpoint_record_coeff_mps_resident_storage_bytes"] = mps_resident_coeff_storage
    return row


def _artifact(rows: list[dict[str, object]], status: str = "ok") -> dict[str, object]:
    return {
        "status": status,
        "tape_mode": verify_mod.PROMOTED_TAPE_MODE,
        "rows": rows,
    }


def _verify(
    payload: dict[str, object],
    *,
    confirm: dict[str, object] | None = None,
    allow: bool = False,
    expected_frames: tuple[int, ...] = (16, 32, 64, 128),
    expect_payload_bool: tuple[tuple[str, bool], ...] = (),
) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        artifact_path = root / "artifact.json"
        artifact_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        confirm_path = None
        if confirm is not None:
            confirm_path = root / "confirm.json"
            confirm_path.write_text(json.dumps(confirm) + "\n", encoding="utf-8")
        reference_path = None
        reference = payload.get("reference")
        if isinstance(reference, dict):
            reference_path = root / "reference.json"
            reference_path.write_text(json.dumps(reference) + "\n", encoding="utf-8")
        return verify_mod.verify(
            argparse.Namespace(
                artifact=artifact_path,
                confirm_artifact=confirm_path,
                reference_artifact=reference_path,
                expected_frames=expected_frames,
                expected_tape_mode=verify_mod.PROMOTED_TAPE_MODE,
                expect_payload_bool=expect_payload_bool,
                max_total_scale=2.0,
                max_backward_scale=2.0,
                max_storage_scale=1.10,
                max_topology_storage_scale=1.10,
                max_coeff_storage_scale=1.10,
                max_mps_resident_storage_scale=1.10,
                max_mps_resident_noncoeff_storage_scale=1.10,
                max_mps_resident_coeff_storage_scale=1.10,
                max_row_mean_to_median=2.5,
                max_row_max_to_median=8.0,
                max_confirm_total_median_ms=8.0,
                max_confirm_backward_median_ms=8.0,
                max_confirm_total_max_ms=12.0,
                max_reference_total_median_ratio=1.20,
                max_reference_backward_median_ratio=1.20,
                allow_confirmed_outliers=allow,
            )
        )


class VerifyFramegroup16TimingRobustTests(unittest.TestCase):
    def test_accepts_clean_sublinear_speedscale_artifact(self) -> None:
        payload = _artifact(
            [
                _row(16, 3.0, 2.5, 1000),
                _row(32, 3.5, 3.0, 1010),
                _row(64, 3.8, 3.2, 1020),
                _row(128, 4.4, 3.8, 1030),
            ]
        )

        result = _verify(payload)

        self.assertEqual(result["status"], "ok", result)
        self.assertTrue(result["clean_speedscale_artifact"])
        self.assertTrue(result["promoted_path_not_regressed"])

    def test_reports_topology_and_coeff_storage_scales_when_present(self) -> None:
        payload = _artifact(
            [
                _row(16, 3.0, 2.5, 1000, topology_storage=120, coeff_storage=800),
                _row(32, 3.5, 3.0, 1010, topology_storage=123, coeff_storage=800),
                _row(64, 3.8, 3.2, 1020, topology_storage=126, coeff_storage=800),
                _row(128, 4.4, 3.8, 1030, topology_storage=130, coeff_storage=800),
            ]
        )

        result = _verify(payload)

        self.assertEqual(result["status"], "ok", result)
        self.assertAlmostEqual(result["storage_scale"], 1.03)
        self.assertAlmostEqual(result["topology_storage_scale"], 130 / 120)
        self.assertAlmostEqual(result["coeff_storage_scale"], 1.0)
        self.assertEqual(result["rows"]["16"]["topology_storage_bytes"], 120)
        self.assertEqual(result["rows"]["16"]["coeff_storage_bytes"], 800)

    def test_zero_optional_storage_sidecar_reports_zero_scale(self) -> None:
        payload = _artifact(
            [
                _row(16, 3.0, 2.5, 1000, topology_storage=120, coeff_storage=0),
                _row(32, 3.5, 3.0, 1010, topology_storage=123, coeff_storage=0),
                _row(64, 3.8, 3.2, 1020, topology_storage=126, coeff_storage=0),
                _row(128, 4.4, 3.8, 1030, topology_storage=130, coeff_storage=0),
            ]
        )

        result = _verify(payload)

        self.assertEqual(result["status"], "ok", result)
        self.assertAlmostEqual(result["topology_storage_scale"], 130 / 120)
        self.assertEqual(result["coeff_storage_scale"], 0.0)

    def test_reports_mps_resident_storage_scales_when_present(self) -> None:
        payload = _artifact(
            [
                _row(
                    16,
                    3.0,
                    2.5,
                    1000,
                    mps_resident_storage=720,
                    mps_resident_noncoeff_storage=120,
                    mps_resident_coeff_storage=600,
                ),
                _row(
                    32,
                    3.5,
                    3.0,
                    1010,
                    mps_resident_storage=724,
                    mps_resident_noncoeff_storage=124,
                    mps_resident_coeff_storage=600,
                ),
                _row(
                    64,
                    3.8,
                    3.2,
                    1020,
                    mps_resident_storage=728,
                    mps_resident_noncoeff_storage=128,
                    mps_resident_coeff_storage=600,
                ),
                _row(
                    128,
                    4.4,
                    3.8,
                    1030,
                    mps_resident_storage=732,
                    mps_resident_noncoeff_storage=132,
                    mps_resident_coeff_storage=600,
                ),
            ]
        )

        result = _verify(payload)

        self.assertEqual(result["status"], "ok", result)
        self.assertAlmostEqual(result["mps_resident_storage_scale"], 732 / 720)
        self.assertAlmostEqual(result["mps_resident_noncoeff_storage_scale"], 132 / 120)
        self.assertAlmostEqual(result["mps_resident_coeff_storage_scale"], 1.0)
        self.assertEqual(result["rows"]["16"]["mps_resident_storage_bytes"], 720)
        self.assertEqual(result["rows"]["16"]["mps_resident_noncoeff_storage_bytes"], 120)
        self.assertEqual(result["rows"]["16"]["mps_resident_coeff_storage_bytes"], 600)

    def test_rejects_mps_resident_noncoeff_storage_scale_regression(self) -> None:
        payload = _artifact(
            [
                _row(
                    16,
                    3.0,
                    2.5,
                    1000,
                    mps_resident_storage=700,
                    mps_resident_noncoeff_storage=100,
                    mps_resident_coeff_storage=600,
                ),
                _row(
                    32,
                    3.5,
                    3.0,
                    1010,
                    mps_resident_storage=720,
                    mps_resident_noncoeff_storage=120,
                    mps_resident_coeff_storage=600,
                ),
                _row(
                    64,
                    3.8,
                    3.2,
                    1020,
                    mps_resident_storage=740,
                    mps_resident_noncoeff_storage=140,
                    mps_resident_coeff_storage=600,
                ),
                _row(
                    128,
                    4.4,
                    3.8,
                    1030,
                    mps_resident_storage=760,
                    mps_resident_noncoeff_storage=160,
                    mps_resident_coeff_storage=600,
                ),
            ]
        )

        result = _verify(payload)

        self.assertEqual(result["status"], "failed")
        self.assertIn("MPS resident non-coefficient storage scale", "\n".join(result["failures"]))

    def test_accepts_expected_payload_bools_on_top_level_and_rows(self) -> None:
        payload = _artifact(
            [
                _row(16, 3.0, 2.5, 1000),
                _row(32, 3.5, 3.0, 1010),
                _row(64, 3.8, 3.2, 1020),
                _row(128, 4.4, 3.8, 1030),
            ]
        )
        expected = {
            "experimental_kernel_order_packed_delta_device": True,
            "experimental_launch_only_packed_delta": True,
            "experimental_unchecked_launch_only_packed_delta": True,
            "experimental_rowdesc_launch_only_packed_delta": True,
        }
        payload.update(expected)
        for row in payload["rows"]:
            row.update(expected)

        result = _verify(payload, expect_payload_bool=tuple(expected.items()))

        self.assertEqual(result["status"], "ok", result)
        self.assertEqual(result["expected_payload_bools"], expected)

    def test_rejects_expected_payload_bool_mismatch(self) -> None:
        payload = _artifact(
            [
                _row(16, 3.0, 2.5, 1000),
                _row(32, 3.5, 3.0, 1010),
                _row(64, 3.8, 3.2, 1020),
                _row(128, 4.4, 3.8, 1030),
            ]
        )
        payload["experimental_rowdesc_launch_only_packed_delta"] = True
        for row in payload["rows"]:
            row["experimental_rowdesc_launch_only_packed_delta"] = True
        payload["rows"][2]["experimental_rowdesc_launch_only_packed_delta"] = False

        result = _verify(
            payload,
            expect_payload_bool=(("experimental_rowdesc_launch_only_packed_delta", True),),
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("expected experimental_rowdesc_launch_only_packed_delta=True", "\n".join(result["failures"]))

    def test_rejects_optional_storage_sidecar_that_grows_from_zero(self) -> None:
        payload = _artifact(
            [
                _row(16, 3.0, 2.5, 1000, topology_storage=0, coeff_storage=800),
                _row(32, 3.5, 3.0, 1010, topology_storage=1, coeff_storage=800),
                _row(64, 3.8, 3.2, 1020, topology_storage=1, coeff_storage=800),
                _row(128, 4.4, 3.8, 1030, topology_storage=1, coeff_storage=800),
            ]
        )

        result = _verify(payload)

        self.assertEqual(result["status"], "failed")
        self.assertIn("topology storage scale", "\n".join(result["failures"]))

    def test_rejects_topology_storage_scale_regression_even_when_total_storage_passes(self) -> None:
        payload = _artifact(
            [
                _row(16, 3.0, 2.5, 1000, topology_storage=100, coeff_storage=800),
                _row(32, 3.5, 3.0, 1010, topology_storage=110, coeff_storage=790),
                _row(64, 3.8, 3.2, 1020, topology_storage=120, coeff_storage=780),
                _row(128, 4.4, 3.8, 1030, topology_storage=140, coeff_storage=770),
            ]
        )

        result = _verify(payload)

        self.assertEqual(result["status"], "failed")
        self.assertIn("topology storage scale", "\n".join(result["failures"]))

    def test_rejects_contaminated_full_sweep_without_confirmation(self) -> None:
        bad_128 = _row(128, 300.0, 280.0, 1030)
        payload = _artifact(
            [
                _row(16, 3.0, 2.5, 1000),
                _row(32, 3.5, 3.0, 1010),
                _row(64, 3.8, 3.2, 1020),
                bad_128,
            ],
            status="failed",
        )

        result = _verify(payload)

        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["clean_speedscale_artifact"])
        self.assertFalse(result["promoted_path_not_regressed"])
        self.assertIn("top-level status", "\n".join(result["contamination"]))

    def test_classifies_confirmed_outlier_without_calling_it_clean_speedscale(self) -> None:
        bad_32 = _row(32, 60.0, 55.0, 1010)
        bad_32["step_summary"]["total"] = _summary(60.0, 3.4, 300.0)
        bad_32["step_summary"]["backward"] = _summary(55.0, 3.0, 290.0)
        payload = _artifact(
            [
                _row(16, 3.0, 2.5, 1000),
                bad_32,
                _row(64, 3.8, 3.2, 1020),
                _row(128, 300.0, 280.0, 1030),
            ],
            status="failed",
        )
        confirm = _artifact([_row(128, 4.5, 4.0, 1030)])

        result = _verify(payload, confirm=confirm, allow=True)

        self.assertEqual(result["status"], "confirmed_outlier", result)
        self.assertFalse(result["clean_speedscale_artifact"])
        self.assertTrue(result["promoted_path_not_regressed"])
        self.assertGreater(len(result["contamination"]), 0)
        self.assertLess(result["substituted_last_frame_scales"]["total_mean_scale"], 2.0)

    def test_rejects_confirmed_outlier_when_confirmation_is_slow(self) -> None:
        payload = _artifact(
            [
                _row(16, 3.0, 2.5, 1000),
                _row(32, 3.5, 3.0, 1010),
                _row(64, 3.8, 3.2, 1020),
                _row(128, 300.0, 280.0, 1030),
            ],
            status="failed",
        )
        confirm = _artifact([_row(128, 20.0, 18.0, 1030)])

        result = _verify(payload, confirm=confirm, allow=True)

        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["promoted_path_not_regressed"])
        self.assertIn("confirmation total median", "\n".join(result["failures"]))

    def test_reference_artifact_rejects_sublinear_but_slow_sweep(self) -> None:
        reference = _artifact(
            [
                _row(16, 3.0, 2.5, 1000),
                _row(32, 3.5, 3.0, 1010),
                _row(64, 3.8, 3.2, 1020),
                _row(128, 4.4, 3.8, 1030),
            ]
        )
        payload = _artifact(
            [
                _row(16, 7.0, 6.0, 1000),
                _row(32, 5.6, 4.9, 1010),
                _row(64, 5.0, 4.2, 1020),
                _row(128, 4.5, 3.9, 1030),
            ]
        )
        payload["reference"] = reference

        result = _verify(payload)

        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["clean_speedscale_artifact"])
        self.assertFalse(result["promoted_path_not_regressed"])
        self.assertIn("reference", "\n".join(result["failures"]))

    def test_reference_artifact_rejects_slow_single_frame_spot(self) -> None:
        reference = _artifact([_row(16, 3.0, 2.5, 1000)])
        payload = _artifact([_row(16, 27.5, 20.7, 1000)])
        payload["reference"] = reference

        result = _verify(payload, expected_frames=(16,))

        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["clean_speedscale_artifact"])
        self.assertIn("16f total median", "\n".join(result["failures"]))

    def test_rejects_contended_benchmark_environment(self) -> None:
        payload = _artifact(
            [
                _row(16, 3.0, 2.5, 1000),
                _row(32, 3.5, 3.0, 1010),
                _row(64, 3.8, 3.2, 1020),
                _row(128, 4.4, 3.8, 1030),
            ]
        )
        payload["benchmark_environment"] = {
            "status": "contended",
            "contending_processes": [{"pid": 147, "command": "python -m pytest tests/"}],
        }

        result = _verify(payload)

        self.assertEqual(result["status"], "failed")
        self.assertIn("benchmark_environment", "\n".join(result["contamination"]))

    def test_accepts_background_benchmark_environment(self) -> None:
        payload = _artifact(
            [
                _row(16, 3.0, 2.5, 1000),
                _row(32, 3.5, 3.0, 1010),
                _row(64, 3.8, 3.2, 1020),
                _row(128, 4.4, 3.8, 1030),
            ]
        )
        payload["benchmark_environment"] = {
            "status": "background",
            "background_processes": [{"pid": 5391, "pcpu": 0.1, "command": "python -m sky.server.server"}],
            "blocking_processes": [],
        }

        result = _verify(payload)

        self.assertEqual(result["status"], "ok", result)
        self.assertEqual(result["contamination"], [])


if __name__ == "__main__":
    unittest.main()
