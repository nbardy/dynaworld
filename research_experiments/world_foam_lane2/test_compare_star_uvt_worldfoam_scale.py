from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import compare_star_uvt_worldfoam_scale as compare_mod
from compare_star_uvt_worldfoam_scale import compare_summaries, run_gate, summarize_star_rows, summarize_worldfoam


class CompareStarUvtWorldFoamScaleTests(unittest.TestCase):
    def test_periodic_mps_export_monitor_blocks_compare_promotion(self) -> None:
        command = (
            "uv run python scripts/run_btc15m_overnight_shadow_monitor.py "
            "--run-id btc15m_toto_context64 --toto-export-device mps "
            "--toto-export-with-runtime-deps"
        )

        self.assertTrue(
            compare_mod._benchmark_process_blocks_promotion(
                command=command,
                pcpu=0.0,
                blocking_cpu_threshold=5.0,
                hard_keywords=("pytest", "torch", "metal", "mps"),
            )
        )
        self.assertEqual(
            compare_mod._benchmark_process_block_reason(
                command=command,
                pcpu=0.0,
                blocking_cpu_threshold=5.0,
                hard_keywords=("pytest", "torch", "metal", "mps"),
            ),
            "periodic_mps_exporter",
        )

    def test_capture_ignores_current_process_ancestor_chain(self) -> None:
        ps_stdout = "\n".join(
            [
                (
                    "20 1 0.0 0.0 "
                    "rtk sh -lc PYTHONPATH=src/train .venv/bin/python "
                    "research_experiments/world_foam_lane2/compare_star_uvt_worldfoam_scale.py "
                    "--worldfoam-artifact results/local_mac_powerfoam_metal_smoke.json"
                ),
                (
                    "30 20 0.0 0.0 "
                    "/bin/zsh -lc PYTHONPATH=src/train .venv/bin/python "
                    "research_experiments/world_foam_lane2/compare_star_uvt_worldfoam_scale.py "
                    "--worldfoam-artifact results/local_mac_powerfoam_metal_smoke.json"
                ),
                (
                    "40 30 0.0 0.1 "
                    "/opt/homebrew/bin/python "
                    "research_experiments/world_foam_lane2/compare_star_uvt_worldfoam_scale.py "
                    "--require-benchmark-environment-ok"
                ),
            ]
        )
        with (
            mock.patch.object(compare_mod.os, "getpid", return_value=40),
            mock.patch.object(compare_mod.os, "getppid", return_value=30),
            mock.patch.object(
                compare_mod.subprocess,
                "run",
                return_value=mock.Mock(stdout=ps_stdout + "\n"),
            ),
        ):
            environment = compare_mod.capture_benchmark_environment()

        self.assertEqual(environment["status"], "ok")
        self.assertEqual(environment["blocking_processes"], [])
        self.assertEqual(environment["background_processes"], [])

    def test_unchecked_environment_blocks_compare_promotion(self) -> None:
        self.assertTrue(compare_mod.benchmark_environment_blocks_promotion({"status": "unchecked"}))
        self.assertFalse(compare_mod.benchmark_environment_blocks_promotion({"status": "background"}))
        self.assertFalse(compare_mod.benchmark_environment_blocks_promotion({"status": "ok"}))

    def test_summarizes_star_direct_atomic_rows(self) -> None:
        rows = [
            {
                "frames": 2,
                "sample_emission_mode": "direct_atomic",
                "summary": {
                    "total_ms": {"median": 10.0},
                    "backward_ms": {"median": 6.0},
                    "forward_ms": {"median": 2.0},
                    "direct_grad_tube_count": {"median": 128.0},
                },
            },
            {
                "frames": 16,
                "sample_emission_mode": "direct_atomic",
                "summary": {
                    "total_ms": {"median": 20.0},
                    "backward_ms": {"median": 9.0},
                    "forward_ms": {"median": 3.0},
                    "direct_grad_tube_count": {"median": 128.0},
                },
            },
        ]

        summary = summarize_star_rows(rows)

        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["frame_counts"], [2, 16])
        self.assertEqual(summary["total_median_scale_first_to_last"], 2.0)
        self.assertEqual(summary["backward_median_scale_first_to_last"], 1.5)
        self.assertEqual(summary["direct_grad_tube_count_by_frame"], {"2": 128.0, "16": 128.0})

    def test_summarizes_worldfoam_fused_artifact_and_compares_ratios(self) -> None:
        rows = []
        for frame, total_s, backward_s, mixed_storage, explicit_storage in (
            (2, 0.002, 0.001, 1000.0, 200.0),
            (16, 0.003, 0.0015, 1100.0, 1600.0),
        ):
            rows.append(
                {
                    "frame_count": frame,
                    "final_train_psnr": 13.0,
                    "final_heldout_psnr": 14.0,
                    "train_mixed_tape_storage_bytes": mixed_storage,
                    "train_explicit_ray_storage_bytes": explicit_storage,
                    "step_summary": {
                        "total": {"median_s": total_s},
                        "backward": {"median_s": backward_s},
                    },
                    "wall_timing": {
                        "build_train_tape_s": 0.5,
                        "build_heldout_tape_s": 0.25,
                        "train_loop_s": 0.1,
                    },
                }
            )
        worldfoam = summarize_worldfoam({"status": "ok", "vjp_mode": "fused_mse_rgb_only", "rows": rows})
        star = {
            "frame_counts": [2, 16],
            "total_median_ms_by_frame": {"2": 10.0, "16": 18.0},
            "backward_median_ms_by_frame": {"2": 6.0, "16": 9.0},
            "total_median_scale_first_to_last": 1.8,
            "backward_median_scale_first_to_last": 1.5,
        }

        comparison = compare_summaries(star, worldfoam)

        self.assertEqual(worldfoam["status"], "ok")
        self.assertEqual(worldfoam["total_median_ms_by_frame"], {"2": 2.0, "16": 3.0})
        self.assertEqual(worldfoam["train_mixed_tape_storage_scale_first_to_last"], 1.1)
        self.assertEqual(worldfoam["train_explicit_ray_storage_scale_first_to_last"], 8.0)
        self.assertEqual(comparison["total_median_ms_ratio_star_over_worldfoam_by_frame"], {"2": 5.0, "16": 6.0})
        self.assertEqual(comparison["backward_median_ms_ratio_star_over_worldfoam_by_frame"], {"2": 6.0, "16": 6.0})

    def test_summarizes_candidate_csr_worldfoam_artifact(self) -> None:
        rows = []
        for frame, total_s, backward_s, storage, ray_storage, candidates in (
            (2, 0.004, 0.0035, 1_048_324.0, 196_608.0, 84_930.0),
            (16, 0.0042, 0.0037, 1_039_920.0, 1_572_864.0, 84_225.0),
        ):
            rows.append(
                {
                    "frame_count": frame,
                    "final_train_psnr": 14.0,
                    "final_heldout_psnr": 13.5,
                    "train_selected_tape_mps_resident_noncoeff_storage_bytes": storage,
                    "train_selected_tape_mps_resident_storage_by_key": {"affine_ray_f32": ray_storage},
                    "gate4_endpoint_train_metadata": {
                        "candidate_count": candidates,
                        "max_candidates_per_row": 224,
                    },
                    "step_summary": {
                        "total": {"median_s": total_s},
                        "backward": {"median_s": backward_s},
                    },
                }
            )

        summary = summarize_worldfoam(
            {
                "status": "ok",
                "tape_mode": "gate4-affine-candidate-num32-den16-fused-mse",
                "gate4_affine_candidate_csr_fused_mse": True,
                "rows": rows,
            }
        )

        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["worldfoam_family"], "gate4_affine_candidate_csr")
        self.assertEqual(summary["train_mixed_tape_storage_bytes_by_frame"]["2"], 1_048_324.0)
        self.assertEqual(summary["train_explicit_ray_storage_bytes_by_frame"]["16"], 1_572_864.0)
        self.assertLess(summary["candidate_count_scale_first_to_last"], 1.0)
        self.assertEqual(summary["max_candidates_per_row_by_frame"], {"2": 224.0, "16": 224.0})

    def test_worldfoam_summary_accepts_trackmse_candidate_csr_mode(self) -> None:
        rows = [
            {
                "frame_count": 2,
                "final_train_psnr": 14.0,
                "final_heldout_psnr": 13.5,
                "train_selected_tape_mps_resident_noncoeff_storage_bytes": 1024,
                "train_selected_tape_mps_resident_storage_by_key": {"affine_ray_f32": 2048},
                "gate4_endpoint_train_metadata": {
                    "candidate_count": 100,
                    "max_candidates_per_row": 16,
                },
                "step_summary": {
                    "total": {"median_s": 0.001},
                    "backward": {"median_s": 0.0005},
                },
            }
        ]

        summary = summarize_worldfoam(
            {
                "status": "ok",
                "tape_mode": "gate4-affine-candidate-num32-den16-trackmse-fused-mse",
                "rows": rows,
            }
        )

        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["worldfoam_family"], "gate4_affine_candidate_csr")

    def test_worldfoam_summary_accepts_framebitmask_owner_run_factorized_mode(self) -> None:
        rows = [
            {
                "frame_count": 2,
                "loaded_frame_count": 2,
                "repeat_loaded_frames": False,
                "final_train_psnr": 14.0,
                "final_heldout_psnr": 13.5,
                "train_selected_tape_mps_resident_noncoeff_storage_bytes": 1024,
                "train_selected_tape_schema_storage_bytes": 2048,
                "step_summary": {
                    "total": {"median_s": 0.001},
                    "backward": {"median_s": 0.0005},
                },
            },
            {
                "frame_count": 32,
                "loaded_frame_count": 16,
                "repeat_loaded_frames": True,
                "repeat_loaded_frames_scope": "synthetic repeated-fixture speed-scaling smoke",
                "final_train_psnr": 14.1,
                "final_heldout_psnr": 13.6,
                "train_selected_tape_mps_resident_noncoeff_storage_bytes": 4096,
                "train_selected_tape_schema_storage_bytes": 8192,
                "step_summary": {
                    "total": {"median_s": 0.0015},
                    "backward": {"median_s": 0.0007},
                },
            },
        ]

        summary = summarize_worldfoam(
            {
                "status": "ok",
                "tape_mode": "owner-run-delta-packed-factorized-framebitmask-recompute-fused-mse-nomid",
                "rows": rows,
            }
        )

        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["worldfoam_family"], "owner_run_factorized")
        self.assertEqual(summary["train_mixed_tape_storage_bytes_by_frame"]["32"], 4096.0)
        self.assertEqual(summary["train_mixed_tape_storage_scale_first_to_last"], 4.0)
        self.assertEqual(summary["loaded_frame_count_by_frame"], {"2": 2, "32": 16})
        self.assertEqual(summary["repeat_loaded_frames_by_frame"], {"2": False, "32": True})
        self.assertEqual(
            summary["repeat_loaded_frames_scope_by_frame"]["32"],
            "synthetic repeated-fixture speed-scaling smoke",
        )

    def test_clean_compare_rejects_contended_worldfoam_artifact(self) -> None:
        rows = []
        for frame in (2, 16):
            rows.append(
                {
                    "frame_count": frame,
                    "final_train_psnr": 14.0,
                    "final_heldout_psnr": 13.5,
                    "train_selected_tape_mps_resident_noncoeff_storage_bytes": 1024,
                    "step_summary": {
                        "total": {"median_s": 0.001},
                        "backward": {"median_s": 0.0005},
                    },
                }
            )
        artifact = {
            "status": "ok",
            "tape_mode": "owner-run-delta-packed-factorized-framebitmask-recompute-fused-mse-nomid",
            "benchmark_environment": {"status": "contended"},
            "acceptance": {"all_rows_ok": True},
            "rows": rows,
        }
        star_rows = [
            {
                "frames": 2,
                "sample_emission_mode": "direct_atomic",
                "summary": {"total_ms": {"median": 1.0}, "backward_ms": {"median": 0.5}},
            },
            {
                "frames": 16,
                "sample_emission_mode": "direct_atomic",
                "summary": {"total_ms": {"median": 1.5}, "backward_ms": {"median": 0.7}},
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "worldfoam.json"
            artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
            args = argparse.Namespace(
                video_path=Path("unused.mp4"),
                worldfoam_artifact=artifact_path,
                frame_counts="2,16",
                star_target_size=64,
                star_tube_count=224,
                star_seed=5,
                star_spatial_precision=0.125,
                star_temporal_precision=2.0,
                star_opacity=0.7,
                star_tile_t=1,
                star_tile_capacity=128,
                star_lr=0.12,
                steps=2,
                warmup_steps=1,
                star_pair_count_every=0,
                star_repeat_loaded_frames=False,
                require_clean_worldfoam_artifact=True,
                require_benchmark_environment_ok=False,
                wait_for_benchmark_environment_ok_timeout_s=0.0,
                wait_for_benchmark_environment_ok_poll_s=15.0,
                post_run_benchmark_environment_settle_s=0.0,
            )

            with (
                mock.patch(
                    "compare_star_uvt_worldfoam_scale.capture_benchmark_environment",
                    return_value={"status": "ok"},
                ),
                mock.patch("compare_star_uvt_worldfoam_scale.run_star_cases", return_value=star_rows),
            ):
                payload = run_gate(args)

        self.assertEqual(payload["status"], "failed")
        self.assertIn("WorldFoam artifact benchmark_environment is contended", payload["failures"])
        self.assertEqual(payload["worldfoam"]["summary"]["benchmark_environment_status"], "contended")

    def test_clean_compare_rejects_missing_worldfoam_acceptance_before_star_run(self) -> None:
        rows = []
        for frame in (2, 16):
            rows.append(
                {
                    "frame_count": frame,
                    "final_train_psnr": 14.0,
                    "final_heldout_psnr": 13.5,
                    "train_selected_tape_mps_resident_noncoeff_storage_bytes": 1024,
                    "step_summary": {
                        "total": {"median_s": 0.001},
                        "backward": {"median_s": 0.0005},
                    },
                }
            )
        artifact = {
            "status": "ok",
            "tape_mode": "owner-run-delta-packed-factorized-framebitmask-recompute-fused-mse-nomid",
            "benchmark_environment": {"status": "background"},
            "rows": rows,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "worldfoam.json"
            artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
            args = argparse.Namespace(
                video_path=Path("unused.mp4"),
                worldfoam_artifact=artifact_path,
                frame_counts="2,16",
                star_target_size=64,
                star_tube_count=224,
                star_seed=5,
                star_spatial_precision=0.125,
                star_temporal_precision=2.0,
                star_opacity=0.7,
                star_tile_t=1,
                star_tile_capacity=128,
                star_lr=0.12,
                steps=2,
                warmup_steps=1,
                star_pair_count_every=0,
                star_repeat_loaded_frames=False,
                require_clean_worldfoam_artifact=True,
                require_benchmark_environment_ok=False,
                wait_for_benchmark_environment_ok_timeout_s=0.0,
                wait_for_benchmark_environment_ok_poll_s=15.0,
                post_run_benchmark_environment_settle_s=0.0,
            )

            with (
                mock.patch(
                    "compare_star_uvt_worldfoam_scale.capture_benchmark_environment",
                    return_value={"status": "ok"},
                ),
                mock.patch("compare_star_uvt_worldfoam_scale.run_star_cases") as star_mock,
            ):
                payload = run_gate(args)

        self.assertEqual(payload["status"], "failed")
        self.assertIn("WorldFoam artifact acceptance is missing", payload["failures"])
        star_mock.assert_not_called()

    def test_clean_compare_rejects_unchecked_worldfoam_artifact(self) -> None:
        rows = []
        for frame in (2, 16):
            rows.append(
                {
                    "frame_count": frame,
                    "final_train_psnr": 14.0,
                    "final_heldout_psnr": 13.5,
                    "train_selected_tape_mps_resident_noncoeff_storage_bytes": 1024,
                    "step_summary": {
                        "total": {"median_s": 0.001},
                        "backward": {"median_s": 0.0005},
                    },
                }
            )
        artifact = {
            "status": "ok",
            "tape_mode": "owner-run-delta-packed-factorized-framebitmask-recompute-fused-mse-nomid",
            "benchmark_environment": {"status": "unchecked", "error": "ps failed"},
            "acceptance": {"all_rows_ok": True},
            "rows": rows,
        }
        star_rows = [
            {
                "frames": 2,
                "sample_emission_mode": "direct_atomic",
                "summary": {"total_ms": {"median": 1.0}, "backward_ms": {"median": 0.5}},
            },
            {
                "frames": 16,
                "sample_emission_mode": "direct_atomic",
                "summary": {"total_ms": {"median": 1.5}, "backward_ms": {"median": 0.7}},
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "worldfoam.json"
            artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
            args = argparse.Namespace(
                video_path=Path("unused.mp4"),
                worldfoam_artifact=artifact_path,
                frame_counts="2,16",
                star_target_size=64,
                star_tube_count=224,
                star_seed=5,
                star_spatial_precision=0.125,
                star_temporal_precision=2.0,
                star_opacity=0.7,
                star_tile_t=1,
                star_tile_capacity=128,
                star_lr=0.12,
                steps=2,
                warmup_steps=1,
                star_pair_count_every=0,
                star_repeat_loaded_frames=False,
                require_clean_worldfoam_artifact=True,
                require_benchmark_environment_ok=False,
                wait_for_benchmark_environment_ok_timeout_s=0.0,
                wait_for_benchmark_environment_ok_poll_s=15.0,
                post_run_benchmark_environment_settle_s=0.0,
            )

            with (
                mock.patch(
                    "compare_star_uvt_worldfoam_scale.capture_benchmark_environment",
                    return_value={"status": "ok"},
                ),
                mock.patch("compare_star_uvt_worldfoam_scale.run_star_cases", return_value=star_rows),
            ):
                payload = run_gate(args)

        self.assertEqual(payload["status"], "failed")
        self.assertIn(
            "WorldFoam artifact benchmark_environment is not promotable: unchecked",
            payload["failures"],
        )

    def test_clean_compare_rejects_unchecked_start_environment_before_star_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "worldfoam.json"
            artifact_path.write_text(json.dumps({"status": "ok", "rows": []}), encoding="utf-8")
            args = argparse.Namespace(
                video_path=Path("unused.mp4"),
                worldfoam_artifact=artifact_path,
                frame_counts="2,16",
                star_target_size=64,
                star_tube_count=224,
                star_seed=5,
                star_spatial_precision=0.125,
                star_temporal_precision=2.0,
                star_opacity=0.7,
                star_tile_t=1,
                star_tile_capacity=128,
                star_lr=0.12,
                steps=2,
                warmup_steps=1,
                star_pair_count_every=0,
                star_repeat_loaded_frames=False,
                require_clean_worldfoam_artifact=True,
                require_benchmark_environment_ok=True,
                wait_for_benchmark_environment_ok_timeout_s=0.0,
                wait_for_benchmark_environment_ok_poll_s=15.0,
                post_run_benchmark_environment_settle_s=0.0,
            )

            with (
                mock.patch(
                    "compare_star_uvt_worldfoam_scale.capture_benchmark_environment",
                    return_value={"status": "unchecked", "error": "ps failed"},
                ),
                mock.patch("compare_star_uvt_worldfoam_scale.run_star_cases") as star_mock,
            ):
                payload = run_gate(args)

        self.assertEqual(payload["status"], "failed")
        self.assertIn(
            "benchmark environment was not promotable before STAR run: unchecked",
            payload["failures"],
        )
        star_mock.assert_not_called()

    def test_clean_compare_waits_for_quiet_start_environment(self) -> None:
        rows = []
        for frame in (2, 16):
            rows.append(
                {
                    "frame_count": frame,
                    "final_train_psnr": 14.0,
                    "final_heldout_psnr": 13.5,
                    "train_selected_tape_mps_resident_noncoeff_storage_bytes": 1024,
                    "step_summary": {
                        "total": {"median_s": 0.001},
                        "backward": {"median_s": 0.0005},
                    },
                }
            )
        artifact = {
            "status": "ok",
            "tape_mode": "owner-run-delta-packed-factorized-framebitmask-recompute-fused-mse-nomid",
            "benchmark_environment": {"status": "background"},
            "acceptance": {"all_rows_ok": True},
            "rows": rows,
        }
        star_rows = [
            {
                "frames": 2,
                "sample_emission_mode": "direct_atomic",
                "summary": {"total_ms": {"median": 1.0}, "backward_ms": {"median": 0.5}},
            },
            {
                "frames": 16,
                "sample_emission_mode": "direct_atomic",
                "summary": {"total_ms": {"median": 1.5}, "backward_ms": {"median": 0.7}},
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "worldfoam.json"
            artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
            args = argparse.Namespace(
                video_path=Path("unused.mp4"),
                worldfoam_artifact=artifact_path,
                frame_counts="2,16",
                star_target_size=64,
                star_tube_count=224,
                star_seed=5,
                star_spatial_precision=0.125,
                star_temporal_precision=2.0,
                star_opacity=0.7,
                star_tile_t=1,
                star_tile_capacity=128,
                star_lr=0.12,
                steps=2,
                warmup_steps=1,
                star_pair_count_every=0,
                star_repeat_loaded_frames=True,
                require_clean_worldfoam_artifact=True,
                require_benchmark_environment_ok=True,
                wait_for_benchmark_environment_ok_timeout_s=1.0,
                wait_for_benchmark_environment_ok_poll_s=0.1,
                post_run_benchmark_environment_settle_s=0.0,
            )

            with (
                mock.patch(
                    "compare_star_uvt_worldfoam_scale.capture_benchmark_environment",
                    side_effect=[
                        {"status": "contended", "blocking_processes": [{"pid": 1}]},
                        {"status": "ok"},
                        {"status": "ok"},
                    ],
                ),
                mock.patch("compare_star_uvt_worldfoam_scale.time.sleep") as sleep_mock,
                mock.patch("compare_star_uvt_worldfoam_scale.run_star_cases", return_value=star_rows) as star_mock,
            ):
                payload = run_gate(args)

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["benchmark_environment"]["status"], "ok")
        sleep_mock.assert_called_once_with(0.1)
        star_mock.assert_called_once()
        self.assertTrue(star_mock.call_args.kwargs["repeat_loaded_frames"])


if __name__ == "__main__":
    unittest.main()
