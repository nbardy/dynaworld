from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import verify_gate4_affine_candidate_csr_train_eval as verify_mod


def _args(path: Path, **overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "artifact": path,
        "frame_counts": "2,4,8,16",
        "render_size": 16,
        "site_count": 24,
        "min_train_psnr": 8.0,
        "min_heldout_psnr": 8.0,
        "max_total_scale": 2.0,
        "max_backward_scale": 2.0,
        "max_total_median_scale": 2.0,
        "max_backward_median_scale": 2.0,
        "max_storage_scale": 1.10,
        "max_noncoeff_storage_scale": 1.10,
        "max_candidate_scale": 1.10,
        "max_candidates_per_row": 256,
        "max_row_mean_to_median": 2.0,
        "max_row_max_to_median": 4.0,
        "allow_contended": False,
        "out_json": None,
        "tape_mode": verify_mod.TAPE_MODE,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _step_summary(frame: int) -> dict[str, dict[str, float | int]]:
    total_s = {2: 0.0045, 4: 0.0048, 8: 0.0042, 16: 0.0044}[frame]
    backward_s = {2: 0.0039, 4: 0.0040, 8: 0.0036, 16: 0.0038}[frame]
    return {
        "render": {"count": 5, "mean_s": 0.0, "median_s": 0.0, "min_s": 0.0, "max_s": 0.0},
        "loss_eval": {"count": 5, "mean_s": 0.0, "median_s": 0.0, "min_s": 0.0, "max_s": 0.0},
        "backward": {
            "count": 5,
            "mean_s": backward_s,
            "median_s": backward_s,
            "min_s": backward_s * 0.95,
            "max_s": backward_s * 1.05,
        },
        "optimizer": {
            "count": 5,
            "mean_s": 0.0005,
            "median_s": 0.0005,
            "min_s": 0.00045,
            "max_s": 0.00055,
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
    storage = {2: 708_604, 4: 706_044, 8: 702_756, 16: 703_020}[frame]
    candidates = {2: 84_930, 4: 84_609, 8: 84_196, 16: 84_225}[frame]
    return {
        "frame_count": frame,
        "status": "ok",
        "tape_mode": verify_mod.TAPE_MODE,
        "render_size": 16,
        "site_count": 24,
        "endpoint_record_source": "gate4-affine",
        "gate4_affine_candidate_csr_fused_mse": True,
        "gate4_affine_candidate_csr_coeff16_fused_mse": True,
        "gate4_affine_candidate_csr_trackmse_fused_mse": False,
        "gate4_affine_candidate_csr_cap224_fused_mse": False,
        "gate4_affine_candidate_csr_densitymask_fused_mse": False,
        "gate4_affine_candidate_csr_sample_reduce_fused_mse": False,
        "gate4_affine_candidate_csr_sortnet_fused_mse": False,
        "gate4_affine_candidate_csr_framegroup16_cached_fused_mse": False,
        "gate4_affine_candidate_csr_sitecache_fused_mse": False,
        "gate4_affine_candidate_csr_ownerupdate_fused_mse": False,
        "gate4_affine_candidate_csr_ownerupdate_i16_fused_mse": False,
        "gate4_affine_candidate_csr_ownerkeep_fused_mse": False,
        "final_train_psnr": 14.0,
        "final_heldout_psnr": 13.5,
        "first_grad_abs_sum": 0.3,
        "parameter_update_abs_max": 0.1,
        "step_summary": _step_summary(frame),
        "train_selected_tape_mps_resident_storage_bytes": storage,
        "train_selected_tape_mps_resident_noncoeff_storage_bytes": storage,
        "gate4_endpoint_train_metadata": {
            "candidate_count": candidates,
            "max_candidates_per_row": 224,
        },
        "acceptance": {
            "loss_decreased": True,
            "gradients_nonzero": True,
            "parameters_updated": True,
            "selected_tape_segments_below_full": True,
            "owner_run_segments_below_full": True,
            "selected_tape_vjp_under_segment_cap": True,
            "owner_run_vjp_under_segment_cap": True,
            "outputs_are_finite": True,
        },
    }


def _payload() -> dict[str, object]:
    return {
        "benchmark": "world_foam_lane2_segment_tape_train_eval_mps",
        "status": "ok",
        "tape_mode": verify_mod.TAPE_MODE,
        "gate4_affine_candidate_csr_fused_mse": True,
        "gate4_affine_candidate_csr_coeff16_fused_mse": True,
        "gate4_affine_candidate_csr_trackmse_fused_mse": False,
        "gate4_affine_candidate_csr_cap224_fused_mse": False,
        "gate4_affine_candidate_csr_densitymask_fused_mse": False,
        "gate4_affine_candidate_csr_sample_reduce_fused_mse": False,
        "gate4_affine_candidate_csr_sortnet_fused_mse": False,
        "gate4_affine_candidate_csr_framegroup16_cached_fused_mse": False,
        "gate4_affine_candidate_csr_sitecache_fused_mse": False,
        "gate4_affine_candidate_csr_ownerupdate_fused_mse": False,
        "gate4_affine_candidate_csr_ownerupdate_i16_fused_mse": False,
        "gate4_affine_candidate_csr_ownerkeep_fused_mse": False,
        "gate4_affine_candidate_csr_ownerkeep_i16_fused_mse": False,
        "endpoint_record_source": "gate4-affine",
        "optimizer_mode": "manual-vjp",
        "full_trainer_claim": False,
        "full_geometry_gradient_claim": False,
        "quality_claim": False,
        "render_size": 16,
        "site_count": 24,
        "frame_counts": [2, 4, 8, 16],
        "benchmark_environment": {"status": "background"},
        "rows": [_row(frame) for frame in (2, 4, 8, 16)],
    }


def _write_payload(tmpdir: str, payload: dict[str, object]) -> Path:
    path = Path(tmpdir) / "artifact.json"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


class VerifyGate4AffineCandidateCSRTrainEvalTests(unittest.TestCase):
    def test_accepts_clean_candidate_csr_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_payload(tmpdir, _payload())
            result = verify_mod.verify(_args(path))

        self.assertEqual(result["status"], "ok", result["failures"])
        self.assertLess(result["resident_noncoeff_storage_scale"], 1.10)
        self.assertLess(result["backward_scale"], 2.0)
        self.assertEqual(result["benchmark_environment_status"], "background")

    def test_rejects_contended_environment_by_default(self) -> None:
        payload = _payload()
        payload["benchmark_environment"] = {"status": "contended"}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_payload(tmpdir, payload)
            result = verify_mod.verify(_args(path))

        self.assertEqual(result["status"], "failed")
        self.assertIn("benchmark_environment", "\n".join(result["failures"]))

    def test_can_allow_contended_for_diagnostic_artifacts(self) -> None:
        payload = _payload()
        payload["benchmark_environment"] = {"status": "contended"}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_payload(tmpdir, payload)
            result = verify_mod.verify(_args(path, allow_contended=True))

        self.assertEqual(result["status"], "ok", result["failures"])
        self.assertEqual(result["contamination"], ["benchmark_environment status is 'contended'"])

    def test_accepts_single_frame_count_shape_smoke_without_sublinear_claim(self) -> None:
        payload = _payload()
        payload["frame_counts"] = [2]
        payload["rows"] = [payload["rows"][0]]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_payload(tmpdir, payload)
            result = verify_mod.verify(_args(path, frame_counts="2"))

        self.assertEqual(result["status"], "ok", result["failures"])
        self.assertFalse(result["scale_gate_required"])
        self.assertEqual(result["frame_scale"], 1.0)

    def test_rejects_candidate_storage_growth(self) -> None:
        payload = _payload()
        payload["rows"][-1]["train_selected_tape_mps_resident_noncoeff_storage_bytes"] = 2_000_000
        payload["rows"][-1]["train_selected_tape_mps_resident_storage_bytes"] = 2_000_000
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_payload(tmpdir, payload)
            result = verify_mod.verify(_args(path))

        self.assertEqual(result["status"], "failed")
        self.assertIn("resident_noncoeff_storage_scale", "\n".join(result["failures"]))

    def test_rejects_trackmse_when_sample_parallel_mode_is_required(self) -> None:
        payload = _payload()
        payload["tape_mode"] = verify_mod.GATE4_AFFINE_CANDIDATE_COEFF16_TRACK_MSE_FUSED_MSE_MODE
        payload["gate4_affine_candidate_csr_trackmse_fused_mse"] = True
        for row in payload["rows"]:
            row["tape_mode"] = verify_mod.GATE4_AFFINE_CANDIDATE_COEFF16_TRACK_MSE_FUSED_MSE_MODE
            row["gate4_affine_candidate_csr_trackmse_fused_mse"] = True
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_payload(tmpdir, payload)
            result = verify_mod.verify(_args(path))

        self.assertEqual(result["status"], "failed")
        self.assertIn("unexpected tape_mode", "\n".join(result["failures"]))

    def test_accepts_ownerupdate_when_requested(self) -> None:
        payload = _payload()
        payload["tape_mode"] = verify_mod.GATE4_AFFINE_CANDIDATE_COEFF16_OWNERUPDATE_FUSED_MSE_MODE
        payload["gate4_affine_candidate_csr_ownerupdate_fused_mse"] = True
        for row in payload["rows"]:
            row["tape_mode"] = verify_mod.GATE4_AFFINE_CANDIDATE_COEFF16_OWNERUPDATE_FUSED_MSE_MODE
            row["gate4_affine_candidate_csr_ownerupdate_fused_mse"] = True
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_payload(tmpdir, payload)
            result = verify_mod.verify(
                _args(path, tape_mode=verify_mod.GATE4_AFFINE_CANDIDATE_COEFF16_OWNERUPDATE_FUSED_MSE_MODE)
            )

        self.assertEqual(result["status"], "ok", result["failures"])

    def test_accepts_ownerkeep_when_requested(self) -> None:
        payload = _payload()
        payload["tape_mode"] = verify_mod.GATE4_AFFINE_CANDIDATE_COEFF16_OWNERKEEP_FUSED_MSE_MODE
        payload["gate4_affine_candidate_csr_ownerkeep_fused_mse"] = True
        for row in payload["rows"]:
            row["tape_mode"] = verify_mod.GATE4_AFFINE_CANDIDATE_COEFF16_OWNERKEEP_FUSED_MSE_MODE
            row["gate4_affine_candidate_csr_ownerkeep_fused_mse"] = True
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_payload(tmpdir, payload)
            result = verify_mod.verify(
                _args(path, tape_mode=verify_mod.GATE4_AFFINE_CANDIDATE_COEFF16_OWNERKEEP_FUSED_MSE_MODE)
            )

        self.assertEqual(result["status"], "ok", result["failures"])

    def test_accepts_ownerupdate_i16_when_requested(self) -> None:
        payload = _payload()
        payload["tape_mode"] = verify_mod.GATE4_AFFINE_CANDIDATE_COEFF16_OWNERUPDATE_I16_FUSED_MSE_MODE
        payload["gate4_affine_candidate_csr_ownerupdate_i16_fused_mse"] = True
        for row in payload["rows"]:
            row["tape_mode"] = verify_mod.GATE4_AFFINE_CANDIDATE_COEFF16_OWNERUPDATE_I16_FUSED_MSE_MODE
            row["gate4_affine_candidate_csr_ownerupdate_i16_fused_mse"] = True
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_payload(tmpdir, payload)
            result = verify_mod.verify(
                _args(path, tape_mode=verify_mod.GATE4_AFFINE_CANDIDATE_COEFF16_OWNERUPDATE_I16_FUSED_MSE_MODE)
            )

        self.assertEqual(result["status"], "ok", result["failures"])

    def test_accepts_ownerkeep_i16_when_requested(self) -> None:
        payload = _payload()
        payload["tape_mode"] = verify_mod.GATE4_AFFINE_CANDIDATE_COEFF16_OWNERKEEP_I16_FUSED_MSE_MODE
        payload["gate4_affine_candidate_csr_ownerkeep_i16_fused_mse"] = True
        for row in payload["rows"]:
            row["tape_mode"] = verify_mod.GATE4_AFFINE_CANDIDATE_COEFF16_OWNERKEEP_I16_FUSED_MSE_MODE
            row["gate4_affine_candidate_csr_ownerkeep_i16_fused_mse"] = True
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_payload(tmpdir, payload)
            result = verify_mod.verify(
                _args(path, tape_mode=verify_mod.GATE4_AFFINE_CANDIDATE_COEFF16_OWNERKEEP_I16_FUSED_MSE_MODE)
            )

        self.assertEqual(result["status"], "ok", result["failures"])

    def test_accepts_sample_reduce_when_requested(self) -> None:
        payload = _payload()
        payload["tape_mode"] = verify_mod.GATE4_AFFINE_CANDIDATE_COEFF16_SAMPLE_REDUCE_FUSED_MSE_MODE
        payload["gate4_affine_candidate_csr_sample_reduce_fused_mse"] = True
        for row in payload["rows"]:
            row["tape_mode"] = verify_mod.GATE4_AFFINE_CANDIDATE_COEFF16_SAMPLE_REDUCE_FUSED_MSE_MODE
            row["gate4_affine_candidate_csr_sample_reduce_fused_mse"] = True
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_payload(tmpdir, payload)
            result = verify_mod.verify(
                _args(path, tape_mode=verify_mod.GATE4_AFFINE_CANDIDATE_COEFF16_SAMPLE_REDUCE_FUSED_MSE_MODE)
            )

        self.assertEqual(result["status"], "ok", result["failures"])

    def test_accepts_cap224_when_requested(self) -> None:
        payload = _payload()
        payload["tape_mode"] = verify_mod.GATE4_AFFINE_CANDIDATE_COEFF16_CAP224_FUSED_MSE_MODE
        payload["gate4_affine_candidate_csr_cap224_fused_mse"] = True
        for row in payload["rows"]:
            row["tape_mode"] = verify_mod.GATE4_AFFINE_CANDIDATE_COEFF16_CAP224_FUSED_MSE_MODE
            row["gate4_affine_candidate_csr_cap224_fused_mse"] = True
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_payload(tmpdir, payload)
            result = verify_mod.verify(
                _args(path, tape_mode=verify_mod.GATE4_AFFINE_CANDIDATE_COEFF16_CAP224_FUSED_MSE_MODE)
            )

        self.assertEqual(result["status"], "ok", result["failures"])

    def test_accepts_densitymask_when_requested(self) -> None:
        payload = _payload()
        payload["tape_mode"] = verify_mod.GATE4_AFFINE_CANDIDATE_COEFF16_DENSITYMASK_FUSED_MSE_MODE
        payload["gate4_affine_candidate_csr_densitymask_fused_mse"] = True
        for row in payload["rows"]:
            row["tape_mode"] = verify_mod.GATE4_AFFINE_CANDIDATE_COEFF16_DENSITYMASK_FUSED_MSE_MODE
            row["gate4_affine_candidate_csr_densitymask_fused_mse"] = True
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_payload(tmpdir, payload)
            result = verify_mod.verify(
                _args(path, tape_mode=verify_mod.GATE4_AFFINE_CANDIDATE_COEFF16_DENSITYMASK_FUSED_MSE_MODE)
            )

        self.assertEqual(result["status"], "ok", result["failures"])

    def test_accepts_sortnet_when_requested(self) -> None:
        payload = _payload()
        payload["tape_mode"] = verify_mod.GATE4_AFFINE_CANDIDATE_COEFF16_SORTNET_FUSED_MSE_MODE
        payload["gate4_affine_candidate_csr_sortnet_fused_mse"] = True
        for row in payload["rows"]:
            row["tape_mode"] = verify_mod.GATE4_AFFINE_CANDIDATE_COEFF16_SORTNET_FUSED_MSE_MODE
            row["gate4_affine_candidate_csr_sortnet_fused_mse"] = True
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_payload(tmpdir, payload)
            result = verify_mod.verify(
                _args(path, tape_mode=verify_mod.GATE4_AFFINE_CANDIDATE_COEFF16_SORTNET_FUSED_MSE_MODE)
            )

        self.assertEqual(result["status"], "ok", result["failures"])

    def test_accepts_framegroup16_cached_when_requested(self) -> None:
        payload = _payload()
        payload["tape_mode"] = verify_mod.GATE4_AFFINE_CANDIDATE_COEFF16_FRAMEGROUP16_CACHED_FUSED_MSE_MODE
        payload["gate4_affine_candidate_csr_framegroup16_cached_fused_mse"] = True
        for row in payload["rows"]:
            row["tape_mode"] = verify_mod.GATE4_AFFINE_CANDIDATE_COEFF16_FRAMEGROUP16_CACHED_FUSED_MSE_MODE
            row["gate4_affine_candidate_csr_framegroup16_cached_fused_mse"] = True
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_payload(tmpdir, payload)
            result = verify_mod.verify(
                _args(path, tape_mode=verify_mod.GATE4_AFFINE_CANDIDATE_COEFF16_FRAMEGROUP16_CACHED_FUSED_MSE_MODE)
            )

        self.assertEqual(result["status"], "ok", result["failures"])

    def test_accepts_sitecache_when_requested(self) -> None:
        payload = _payload()
        payload["tape_mode"] = verify_mod.GATE4_AFFINE_CANDIDATE_COEFF16_SITECACHE_FUSED_MSE_MODE
        payload["gate4_affine_candidate_csr_sitecache_fused_mse"] = True
        for row in payload["rows"]:
            row["tape_mode"] = verify_mod.GATE4_AFFINE_CANDIDATE_COEFF16_SITECACHE_FUSED_MSE_MODE
            row["gate4_affine_candidate_csr_sitecache_fused_mse"] = True
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_payload(tmpdir, payload)
            result = verify_mod.verify(
                _args(path, tape_mode=verify_mod.GATE4_AFFINE_CANDIDATE_COEFF16_SITECACHE_FUSED_MSE_MODE)
            )

        self.assertEqual(result["status"], "ok", result["failures"])


if __name__ == "__main__":
    unittest.main()
