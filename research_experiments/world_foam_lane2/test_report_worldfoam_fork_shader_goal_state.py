from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import report_worldfoam_fork_shader_goal_state as report_mod
import run_worldfoam_next_mps_candidate as launcher
import verify_worldfoam_next_mps_candidate_result as verifier


FRAME_COUNTS = [2, 4, 8, 16, 32]


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _fresh_checked_at() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _ok_artifact(path: Path) -> None:
    stem = path.stem
    if "source" in stem:
        _write(path, _source_wiring_payload(path.parent))
    elif "import" in stem:
        _write(path, _import_registration_payload(path.parent))
    elif "smoke" in stem:
        _write(path, _smoke_bundle_payload(path.parent))
    else:
        _write(path, {"status": "ok", "failures": []})


def _source_wiring_payload(base: Path) -> dict[str, object]:
    variant_root = base / "variants"
    variant_root.mkdir(parents=True, exist_ok=True)
    variants = []
    for variant, package in report_mod.DEFAULT_VARIANTS:
        variants.append(
            {
                "variant": variant,
                "package": package,
                "status": "ok",
                "failures": [],
                "schema_count": 3,
                "impl_count": 3,
                "impl_target_count": 3,
                "python_ops_ref_count": 3,
                "loaded_metal_file_count": 1,
                "loaded_metal_kernel_count": 2,
                "host_kernel_ref_count": 2,
                "host_kernel_field_count": 2,
                "initialized_kernel_field_count": 2,
                "metal_kernel_count": 2,
            }
        )
    return {
        "status": "ok",
        "failures": [],
        "variant_root": str(variant_root),
        "variant_count": len(variants),
        "variants": variants,
    }


def _import_registration_payload(base: Path) -> dict[str, object]:
    variant_root = base / "variants"
    variant_root.mkdir(parents=True, exist_ok=True)
    variants = []
    for variant, package in report_mod.DEFAULT_VARIANTS:
        extension_library = variant_root / variant / package / "_C.cpython-311-darwin.so"
        extension_library.parent.mkdir(parents=True, exist_ok=True)
        extension_library.write_bytes(b"test-extension")
        variants.append(
            {
                "variant": variant,
                "package": package,
                "status": "ok",
                "failures": [],
                "schema_count": 3,
                "registered_schema_count": 3,
                "missing_registered_schemas": [],
                "extension_library": str(extension_library),
                "import_error": "",
                "extension_load_error": "",
                "compiled_source_count": 4,
            }
        )
    return {
        "status": "ok",
        "failures": [],
        "variant_root": str(variant_root),
        "variant_count": len(variants),
        "variants": variants,
    }


def _smoke_bundle_payload(base: Path) -> dict[str, object]:
    required = []
    for spec in report_mod.REQUIRED_ARTIFACTS:
        artifact_path = base / f"{spec['label']}.json"
        _write(artifact_path, {"status": "ok", "benchmark": spec["benchmark"]})
        required.append(
            {
                "label": spec["label"],
                "path": str(artifact_path),
                "status": "ok",
                "failures": [],
                "benchmark": spec["benchmark"],
                "artifact_status": "ok",
            }
        )
    invalid_path = base / "known_invalid_tiled_ownerupdate.json"
    _write(invalid_path, {"status": "failed"})
    return {
        "status": "ok",
        "failures": [],
        "quality_claim": False,
        "speed_claim": False,
        "scope": "rebuilt_native_variant_smoke_artifacts_only",
        "required_count": len(required),
        "required": required,
        "known_invalid_tiled_ownerupdate": {
            "path": str(invalid_path),
            "status": "ok",
            "classification": "expected_invalid_tiled_ownerupdate",
            "failures": [],
            "present": True,
        },
    }


def _blocked_summary(path: Path, *, status: str = "preflight_contended") -> None:
    _write(
        path,
        {
            "status": status,
            "history_jsonl": "next.history.jsonl",
            "preflight_process_sample_limit": 32,
            "preflight_blocking_process_count": 3,
            "preflight_blocking_process_sample_count": 2,
            "preflight_blocking_process_unlisted_count": 1,
            "preflight_contending_process_count": 3,
            "preflight_contending_process_sample_count": 2,
            "preflight_contending_process_unlisted_count": 1,
            "preflight_blocking_reasons": ["high_cpu", "periodic_mps_exporter"],
            "preflight_attempt_count": 2,
            "preflight_retry_timeout_s": 180.0,
            "preflight_stability_samples_requested": 3,
            "preflight_stability_samples_completed": 1,
            "preflight_stability_ok": False,
            "preflight_blocking_processes": [
                {
                    "pid": 7002,
                    "ppid": 6978,
                    "stat": "R",
                    "elapsed": "25:28",
                    "block_reason": "high_cpu",
                    "pcpu": 188.9,
                    "pmem": 1.8,
                    "command": "python train_node_curve_program_flow_v2.py",
                },
                {
                    "pid": 54895,
                    "ppid": 54881,
                    "stat": "S+",
                    "elapsed": "02:20:11",
                    "block_reason": "periodic_mps_exporter",
                    "pcpu": 0.0,
                    "pmem": 0.1,
                    "command": "python scripts/run_btc15m_overnight_shadow_monitor.py",
                },
            ],
            "preflight_external_blocker_summary": {
                "blocking_kind_counts": {"high_cpu_external_job": 1, "periodic_mps_exporter": 2},
                "blocking_reason_counts": {"high_cpu": 1, "periodic_mps_exporter": 2},
                "manual_next_actions": [
                    "rerun only after the benchmark preflight reports a quiet external-process window",
                    "wait for or manually pause periodic ai_trader/TOTO MPS exporter work",
                ],
            },
            "planned_worldfoam_artifact": "planned.worldfoam.json",
        },
    )


def _blocker_diagnosis(path: Path, *, summary_json: Path) -> None:
    _write(
        path,
        {
            "status": "blocked",
            "summary_json": str(summary_json),
            "checked_at": "2026-05-21T04:12:00+07:00",
            "process_sample_limit": 32,
            "blocker_count": 3,
            "blocker_sample_count": 2,
            "blocker_unlisted_count": 1,
            "contending_process_count": 3,
            "contending_process_sample_count": 2,
            "contending_process_unlisted_count": 1,
            "live_blocker_count": 2,
            "recent_output_blocker_count": 1,
            "live_or_recent_blocker_count": 2,
            "category_counts": {
                "ai_trader_toto_mps_exporter": 1,
                "ai_trader_toto_worker": 1,
                "font_maker_random_stroke_train": 1,
            },
            "live_category_counts": {
                "ai_trader_toto_mps_exporter": 1,
                "font_maker_random_stroke_train": 1,
            },
            "active_cpu_category_counts": {"font_maker_random_stroke_train": 1},
            "summary_cpu_active_category_counts": {"font_maker_random_stroke_train": 1},
            "live_cpu_over_preflight_threshold_category_counts": {"font_maker_random_stroke_train": 1},
            "recent_output_category_counts": {"ai_trader_toto_mps_exporter": 1},
            "max_estimated_remaining_s_by_category": {"ai_trader_toto_mps_exporter": 34789.0},
            "blockers": [
                {
                    "pid": 7002,
                    "ppid": 6978,
                    "stat": "R",
                    "elapsed": "25:28",
                    "block_reason": "high_cpu",
                    "pcpu": 188.9,
                    "pmem": 1.8,
                    "command": "python train_node_curve_program_flow_v2.py",
                    "category": "font_maker_random_stroke_train",
                    "pid_live": True,
                    "live_pcpu": 190.2,
                },
                {
                    "pid": 54895,
                    "ppid": 54881,
                    "stat": "S+",
                    "elapsed": "02:20:11",
                    "block_reason": "periodic_mps_exporter",
                    "pcpu": 0.0,
                    "pmem": 0.1,
                    "command": "/usr/bin/SCREEN -dmS toto_floor001_postfix_20260520T171609Z zsh -lc python scripts/run_btc15m_overnight_shadow_monitor.py",
                    "category": "ai_trader_toto_mps_exporter",
                    "pid_live": True,
                    "recent_output_count": 12,
                    "declared_duration_hours": 12.0,
                    "elapsed_s": 8411.0,
                    "estimated_remaining_s": 34789.0,
                    "estimated_done_at": "2026-05-21T13:51:49+07:00",
                },
            ],
        },
    )


def _benchmark_environment_snapshot() -> dict[str, object]:
    return {
        "status": "background",
        "pid": 12345,
        "keywords": ["python", "pytest", "torch", "metal", "mps"],
        "hard_keywords": ["torch", "mps"],
        "blocking_cpu_threshold": verifier.EXPECTED_BLOCKING_CPU_THRESHOLD,
        "general_blocking_cpu_threshold": verifier.EXPECTED_GENERAL_BLOCKING_CPU_THRESHOLD,
        "blocking_process_count": 0,
        "contending_process_count": 0,
        "background_process_count": 0,
        "blocking_processes": [],
        "contending_processes": [],
        "background_processes": [],
    }


def _benchmark_environment() -> dict[str, object]:
    return {
        "status": "background",
        "start": _benchmark_environment_snapshot(),
        "end": _benchmark_environment_snapshot(),
    }


def _worldfoam_artifact_payload() -> dict[str, object]:
    rows = []
    for index, frame_count in enumerate(FRAME_COUNTS):
        offset = 0.001 * index
        rows.append(
            {
                "status": "ok",
                "frame_count": frame_count,
                "loaded_frame_count": frame_count,
                "render_size": verifier.REQUIRED_RENDER_SIZE,
                "site_count": verifier.REQUIRED_SITE_COUNT,
                "steps": verifier.REQUIRED_STEPS,
                "warmup_steps": verifier.REQUIRED_WARMUP_STEPS,
                "repeat_loaded_frames": False,
                "final_train_psnr": 13.0 + offset,
                "final_heldout_psnr": 14.0 + offset,
                "final_train_l1": 0.18,
                "final_heldout_l1": 0.15,
                "step_summary": {
                    "total": {"mean_s": 0.010 + offset},
                    "backward": {"mean_s": 0.004 + offset},
                    "render": {"mean_s": 0.001 + offset},
                },
            }
        )
    return {
        "status": "ok",
        "benchmark": verifier.EXPECTED_BENCHMARK,
        "benchmark_environment": _benchmark_environment(),
        "device": "mps",
        "frame_counts": FRAME_COUNTS,
        "frame_scale_first_to_last": 16.0,
        "total_step_scale_first_to_last": 1.4,
        "backward_scale_first_to_last": 2.0,
        "render_scale_first_to_last": 5.0,
        "render_size": verifier.REQUIRED_RENDER_SIZE,
        "site_count": verifier.REQUIRED_SITE_COUNT,
        "site_initialization": "legacy_pixel_mean",
        "tape_mode": launcher.DEFAULT_TAPE_MODE,
        "optimizer_mode": verifier.EXPECTED_OPTIMIZER_MODE,
        "endpoint_record_source": verifier.EXPECTED_ENDPOINT_RECORD_SOURCE,
        "experimental_selected_only_owner_run_delta_prep": True,
        "experimental_native_owner_run_cutwalk_delta": True,
        "allow_repeat_loaded_frames": False,
        "acceptance": {
            "all_rows_ok": True,
            "total_step_sublinear_vs_frames": True,
            "render_sublinear_vs_frames": True,
            "backward_sublinear_vs_frames": True,
            "selected_tape_segments_below_full_at_max_frame": True,
            "selected_tape_storage_below_full_at_max_frame": True,
            "owner_run_segments_below_full_at_max_frame": True,
        },
        "rows": rows,
    }


def _verified_summary(summary_path: Path, artifact_path: Path, *, embedded_result: bool = True) -> None:
    _write(artifact_path, _worldfoam_artifact_payload())
    payload: dict[str, object] = {
            "benchmark": "world_foam_next_mps_candidate_launch",
            "status": "train_eval_ok",
            "execute": True,
            "failures": [],
            "readiness_status": "ok",
            "ready_for_quiet_mps_quality_speed_run": True,
            "quality_claim": False,
            "speed_claim": False,
            "mps_quality_speed_artifact_required": True,
            "next_mps_candidate": "legacy_pixel_mean",
            "preflight_returncode": 0,
            "preflight_benchmark_environment_status": "background",
            "preflight_benchmark_environment": _benchmark_environment_snapshot(),
            "preflight_blocking_process_count": 0,
            "preflight_contending_process_count": 0,
            "preflight_stability_samples_requested": verifier.MIN_STABILITY_SAMPLES,
            "preflight_stability_samples_completed": verifier.MIN_STABILITY_SAMPLES,
            "preflight_stability_ok": True,
            "train_eval_returncode": 0,
            "planned_worldfoam_artifact": str(artifact_path),
            "train_eval_command": [
                "python",
                "train_eval_owner_run_tape.py",
                "--frame-counts",
                ",".join(str(count) for count in FRAME_COUNTS),
                "--render-size",
                str(verifier.REQUIRED_RENDER_SIZE),
                "--site-count",
                str(verifier.REQUIRED_SITE_COUNT),
                "--site-initialization",
                "legacy_pixel_mean",
                "--steps",
                str(verifier.REQUIRED_STEPS),
                "--warmup-steps",
                str(verifier.REQUIRED_WARMUP_STEPS),
                "--optimizer-mode",
                verifier.EXPECTED_OPTIMIZER_MODE,
                "--tape-mode",
                launcher.DEFAULT_TAPE_MODE,
                "--endpoint-record-source",
                verifier.EXPECTED_ENDPOINT_RECORD_SOURCE,
                "--require-benchmark-environment-ok",
                "--out-json",
                str(artifact_path),
            ],
    }
    if embedded_result:
        payload.update(
            {
                "verify_result": True,
                "result_verifier_returncode": 0,
                "result_verifier_payload": {
                    "status": "ok",
                    "failures": [],
                    "summary_path": str(summary_path),
                    "worldfoam_artifact": str(artifact_path),
                    "artifact_checks_skipped": False,
                },
                "result_verifier_command": [
                    "python",
                    str(report_mod.ROOT / "research_experiments/world_foam_lane2/verify_worldfoam_next_mps_candidate_result.py"),
                    str(summary_path),
                ],
            }
        )
    _write(summary_path, payload)


class ReportWorldFoamForkShaderGoalStateTests(unittest.TestCase):
    def test_real_goal_state_is_complete_after_verified_clean_gate(self) -> None:
        result = report_mod.audit()

        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["objective_complete"])
        self.assertTrue(result["shader_fork_smoke_state_fixed"])
        self.assertFalse(result["missing_requirements"]["clean_real32_mps_psnr_speed_sublinear_gate"])
        self.assertTrue(result["artifacts"]["next_mps_quality_speed"]["complete"])
        self.assertEqual(result["artifacts"]["next_mps_quality_speed"]["result_verifier_status"], "ok")

    def test_reports_blocked_when_shader_gates_pass_but_preflight_is_contended(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.json"
            imports = root / "imports.json"
            smoke = root / "smoke.json"
            next_mps = root / "next.json"
            diagnosis = root / "diagnosis.json"
            for path in (source, imports, smoke):
                _ok_artifact(path)
            _blocked_summary(next_mps)
            _blocker_diagnosis(diagnosis, summary_json=next_mps)

            result = report_mod.audit(
                source_json=source,
                import_json=imports,
                smoke_bundle_json=smoke,
                next_mps_summary_json=next_mps,
                blocker_diagnosis_json=diagnosis,
                current_benchmark_environment_probe={
                    "available": True,
                    "status": "ok",
                    "blocks_promotion": False,
                    "returncode": 0,
                },
            )

        self.assertEqual(result["status"], "blocked_external_environment")
        self.assertFalse(result["objective_complete"])
        self.assertEqual(
            result["artifacts"]["next_mps_quality_speed"]["blocking_kind_counts"],
            {"high_cpu_external_job": 1, "periodic_mps_exporter": 1},
        )
        self.assertEqual(
            result["artifacts"]["next_mps_quality_speed"]["blocking_reason_counts"],
            {"high_cpu": 1, "periodic_mps_exporter": 1},
        )
        self.assertIn(
            "wait for or manually pause periodic ai_trader/TOTO MPS exporter work",
            result["artifacts"]["next_mps_quality_speed"]["manual_next_actions"],
        )
        self.assertEqual(result["artifacts"]["next_mps_quality_speed"]["preflight_attempt_count"], 2)
        self.assertEqual(result["artifacts"]["next_mps_quality_speed"]["history_jsonl"], "next.history.jsonl")
        self.assertEqual(result["artifacts"]["next_mps_quality_speed"]["preflight_process_sample_limit"], 32)
        self.assertEqual(result["artifacts"]["next_mps_quality_speed"]["preflight_blocking_processes"][0]["pid"], 7002)
        self.assertEqual(result["artifacts"]["next_mps_quality_speed"]["preflight_blocking_processes"][0]["stat"], "R")
        self.assertEqual(
            result["artifacts"]["next_mps_quality_speed"]["preflight_blocking_processes"][0]["elapsed"],
            "25:28",
        )
        self.assertEqual(result["artifacts"]["next_mps_quality_speed"]["preflight_blocking_process_sample_count"], 2)
        self.assertEqual(result["artifacts"]["next_mps_quality_speed"]["preflight_blocking_process_unlisted_count"], 1)
        self.assertEqual(result["artifacts"]["next_mps_quality_speed"]["preflight_contending_process_sample_count"], 2)
        self.assertEqual(result["artifacts"]["next_mps_quality_speed"]["preflight_contending_process_unlisted_count"], 1)
        self.assertEqual(result["artifacts"]["next_mps_quality_speed"]["preflight_retry_timeout_s"], 180.0)
        self.assertEqual(result["artifacts"]["next_mps_quality_speed"]["preflight_stability_samples_requested"], 3)
        self.assertEqual(result["artifacts"]["next_mps_quality_speed"]["preflight_stability_samples_completed"], 1)
        self.assertIs(result["artifacts"]["next_mps_quality_speed"]["preflight_stability_ok"], False)
        self.assertEqual(
            result["artifacts"]["next_mps_quality_speed"]["preflight_blocking_reasons"],
            ["high_cpu", "periodic_mps_exporter"],
        )
        self.assertEqual(result["artifacts"]["next_mps_quality_speed"]["result_verifier_status"], "failed")
        self.assertTrue(result["artifacts"]["next_mps_quality_speed"]["result_verifier_artifact_checks_skipped"])
        self.assertEqual(result["artifacts"]["live_blocker_diagnosis"]["process_sample_limit"], 32)
        self.assertIn(
            "summary status is not train_eval_ok: 'preflight_contended'",
            result["artifacts"]["next_mps_quality_speed"]["result_verifier_failures"],
        )
        live_diagnosis = result["artifacts"]["live_blocker_diagnosis"]
        self.assertTrue(live_diagnosis["available"])
        self.assertTrue(live_diagnosis["matches_next_mps_summary"])
        self.assertEqual(live_diagnosis["checked_at"], "2026-05-21T04:12:00+07:00")
        self.assertIn("diagnosis_fresh", live_diagnosis)
        self.assertEqual(live_diagnosis["live_blocker_count"], 2)
        self.assertEqual(
            live_diagnosis["live_cpu_over_preflight_threshold_category_counts"],
            {"font_maker_random_stroke_train": 1},
        )
        self.assertEqual(
            live_diagnosis["live_category_counts"],
            {"ai_trader_toto_mps_exporter": 1, "font_maker_random_stroke_train": 1},
        )
        self.assertEqual(
            live_diagnosis["summary_cpu_active_category_counts"],
            {"font_maker_random_stroke_train": 1},
        )
        self.assertEqual(live_diagnosis["recent_output_category_counts"], {"ai_trader_toto_mps_exporter": 1})
        self.assertEqual(
            live_diagnosis["blocking_screen_session_names"],
            ["toto_floor001_postfix_20260520T171609Z"],
        )
        self.assertEqual(
            live_diagnosis["max_estimated_remaining_s_by_category"],
            {"ai_trader_toto_mps_exporter": 34789.0},
        )
        self.assertEqual(live_diagnosis["blockers"][0]["pid"], 7002)
        self.assertEqual(live_diagnosis["blockers"][1]["estimated_remaining_s"], 34789.0)
        rerun_plan = result["clean_mps_rerun_plan"]
        self.assertTrue(rerun_plan["requires_quiet_window"])
        self.assertFalse(rerun_plan["ready_to_run_now"])
        self.assertEqual(
            rerun_plan["blocking_conditions"],
            ["live_or_recent_external_blockers_present", "live_blocker_diagnosis_stale"],
        )
        self.assertEqual(rerun_plan["latest_blocker_checked_at"], "2026-05-21T04:12:00+07:00")
        self.assertEqual(rerun_plan["run_after_estimated_done_at"], "2026-05-21T13:51:49+07:00")
        self.assertEqual(rerun_plan["run_after_estimated_done_at_scope"], "live_blocker_diagnosis_only")
        self.assertTrue(rerun_plan["run_after_estimated_done_at_requires_reprobe"])
        self.assertEqual(
            rerun_plan["live_max_estimated_remaining_s_by_category"],
            {"ai_trader_toto_mps_exporter": 34789.0},
        )
        self.assertFalse(rerun_plan["current_benchmark_environment_has_independent_blockers"])
        self.assertEqual(rerun_plan["live_blocker_status"], "blocked")
        self.assertEqual(rerun_plan["live_blocker_count"], 2)
        self.assertEqual(
            rerun_plan["live_blocking_category_counts"],
            {"ai_trader_toto_mps_exporter": 1, "font_maker_random_stroke_train": 1},
        )
        self.assertEqual(
            rerun_plan["live_blocking_screen_session_names"],
            ["toto_floor001_postfix_20260520T171609Z"],
        )
        self.assertEqual(
            rerun_plan["preflight_sample_category_counts"],
            {
                "ai_trader_toto_mps_exporter": 1,
                "ai_trader_toto_worker": 1,
                "font_maker_random_stroke_train": 1,
            },
        )
        self.assertEqual(
            rerun_plan["live_recent_output_category_counts"],
            {"ai_trader_toto_mps_exporter": 1},
        )
        self.assertEqual(rerun_plan["command"][:2], ["rtk", "env"])
        self.assertIn("--execute", rerun_plan["command"])
        self.assertIn("--verify-result", rerun_plan["command"])
        self.assertIn("--preflight-stability-samples", rerun_plan["command"])
        self.assertIn("--preflight-retry-timeout-s", rerun_plan["command"])
        self.assertIn(
            "use clean_mps_rerun_plan.command for the guarded real32 MPS PSNR/speed/sublinear gate",
            result["next_actions"],
        )

    def test_marks_live_blocker_diagnosis_stale_when_checked_at_is_old(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.json"
            imports = root / "imports.json"
            smoke = root / "smoke.json"
            next_mps = root / "next.json"
            diagnosis = root / "diagnosis.json"
            for path in (source, imports, smoke):
                _ok_artifact(path)
            _blocked_summary(next_mps)
            _blocker_diagnosis(diagnosis, summary_json=next_mps)

            result = report_mod.audit(
                source_json=source,
                import_json=imports,
                smoke_bundle_json=smoke,
                next_mps_summary_json=next_mps,
                blocker_diagnosis_json=diagnosis,
                max_blocker_diagnosis_age_s=60.0,
                now=datetime.fromisoformat("2026-05-21T04:20:00+07:00"),
            )

        live_diagnosis = result["artifacts"]["live_blocker_diagnosis"]
        self.assertFalse(live_diagnosis["diagnosis_fresh"])
        self.assertEqual(live_diagnosis["diagnosis_age_s"], 480.0)
        self.assertEqual(live_diagnosis["diagnosis_max_age_s"], 60.0)
        self.assertEqual(
            live_diagnosis["failures"],
            ["blocker diagnosis is stale: age_s=480.0 > max_age_s=60.0"],
        )

    def test_recomputes_blocker_kinds_from_process_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.json"
            imports = root / "imports.json"
            smoke = root / "smoke.json"
            next_mps = root / "next.json"
            for path in (source, imports, smoke):
                _ok_artifact(path)
            _write(
                next_mps,
                {
                    "status": "preflight_contended",
                    "preflight_blocking_processes": [
                        {
                            "pid": 75,
                            "block_reason": "high_cpu",
                            "command": "python scripts/train_kalshi_btc15m_sft.py --input /tmp/btc15m.csv",
                        },
                        {
                            "pid": 76,
                            "block_reason": "periodic_mps_exporter",
                            "command": "python scripts/run_btc15m_overnight_shadow_monitor.py --run-id toto",
                        },
                    ],
                    "preflight_external_blocker_summary": {
                        "blocking_kind_counts": {"periodic_mps_exporter": 2},
                        "blocking_reason_counts": {"high_cpu": 1, "periodic_mps_exporter": 1},
                        "manual_next_actions": ["stale embedded action"],
                    },
                },
            )

            result = report_mod.audit(
                source_json=source,
                import_json=imports,
                smoke_bundle_json=smoke,
                next_mps_summary_json=next_mps,
            )

        next_mps_status = result["artifacts"]["next_mps_quality_speed"]
        self.assertEqual(
            next_mps_status["blocking_kind_counts"],
            {"ai_trader_btc15m_sft": 1, "periodic_mps_exporter": 1},
        )
        self.assertIn(
            "wait for ai_trader BTC15M SFT pytest/training workers to finish",
            next_mps_status["manual_next_actions"],
        )

    def test_reports_blocked_while_retrying_preflight_wait(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.json"
            imports = root / "imports.json"
            smoke = root / "smoke.json"
            next_mps = root / "next.json"
            for path in (source, imports, smoke):
                _ok_artifact(path)
            _blocked_summary(next_mps, status="preflight_retry_waiting")

            result = report_mod.audit(
                source_json=source,
                import_json=imports,
                smoke_bundle_json=smoke,
                next_mps_summary_json=next_mps,
            )

        self.assertEqual(result["status"], "blocked_external_environment")
        self.assertFalse(result["objective_complete"])
        self.assertTrue(result["missing_requirements"]["clean_real32_mps_psnr_speed_sublinear_gate"])
        self.assertEqual(result["artifacts"]["next_mps_quality_speed"]["status"], "preflight_retry_waiting")
        self.assertEqual(result["artifacts"]["next_mps_quality_speed"]["result_verifier_status"], "failed")
        self.assertEqual(result["failures"], [])

    def test_reports_ready_for_clean_gate_when_only_historical_preflight_blockers_remain(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.json"
            imports = root / "imports.json"
            smoke = root / "smoke.json"
            next_mps = root / "next.json"
            diagnosis = root / "diagnosis.json"
            for path in (source, imports, smoke):
                _ok_artifact(path)
            _blocked_summary(next_mps)
            _write(
                diagnosis,
                {
                    "status": "no_live_or_recent_blockers_found",
                    "summary_json": str(next_mps),
                    "checked_at": _fresh_checked_at(),
                    "blocker_count": 3,
                    "blocker_sample_count": 3,
                    "blocker_unlisted_count": 0,
                    "live_blocker_count": 0,
                    "recent_output_blocker_count": 0,
                    "live_or_recent_blocker_count": 0,
                    "category_counts": {
                        "ai_trader_toto_mps_exporter": 2,
                        "ai_trader_toto_worker": 1,
                    },
                    "live_category_counts": {},
                    "recent_output_category_counts": {},
                    "summary_cpu_active_category_counts": {"ai_trader_toto_worker": 1},
                    "blockers": [
                        {
                            "pid": 7805,
                            "category": "ai_trader_toto_worker",
                            "pid_live": False,
                            "summary_cpu_active": True,
                            "recent_output_count": 0,
                        }
                    ],
                },
            )

            result = report_mod.audit(
                source_json=source,
                import_json=imports,
                smoke_bundle_json=smoke,
                next_mps_summary_json=next_mps,
                blocker_diagnosis_json=diagnosis,
                current_benchmark_environment_probe={
                    "available": True,
                    "status": "ok",
                    "blocks_promotion": False,
                    "returncode": 0,
                },
            )

        self.assertEqual(result["status"], "incomplete_ready_for_clean_mps_gate")
        self.assertFalse(result["objective_complete"])
        self.assertTrue(result["clean_mps_rerun_plan"]["ready_to_run_now"])
        self.assertEqual(result["clean_mps_rerun_plan"]["blocking_conditions"], [])
        self.assertTrue(result["clean_mps_rerun_plan"]["run_after_estimated_done_at_requires_reprobe"])
        self.assertFalse(result["clean_mps_rerun_plan"]["current_benchmark_environment_has_independent_blockers"])
        self.assertEqual(result["clean_mps_rerun_plan"]["current_benchmark_environment_status"], "ok")
        self.assertFalse(result["clean_mps_rerun_plan"]["current_benchmark_environment_blocks_promotion"])
        self.assertTrue(result["clean_mps_rerun_plan"]["embedded_result_verification"])
        self.assertEqual(result["clean_mps_rerun_plan"]["acceptance_verifier_required_status"], "ok")
        self.assertEqual(
            result["clean_mps_rerun_plan"]["acceptance_verifier_command_template"][-2:],
            [
                "research_experiments/world_foam_lane2/verify_worldfoam_next_mps_candidate_result.py",
                "<launch_summary_json>",
            ],
        )
        self.assertEqual(result["clean_mps_rerun_plan"]["live_blocking_category_counts"], {})
        self.assertEqual(
            result["clean_mps_rerun_plan"]["preflight_sample_category_counts"],
            {"ai_trader_toto_mps_exporter": 2, "ai_trader_toto_worker": 1},
        )

    def test_current_environment_probe_blocks_ready_status_even_after_old_blockers_clear(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.json"
            imports = root / "imports.json"
            smoke = root / "smoke.json"
            next_mps = root / "next.json"
            diagnosis = root / "diagnosis.json"
            for path in (source, imports, smoke):
                _ok_artifact(path)
            _blocked_summary(next_mps)
            _write(
                diagnosis,
                {
                    "status": "no_live_or_recent_blockers_found",
                    "summary_json": str(next_mps),
                    "checked_at": _fresh_checked_at(),
                    "live_blocker_count": 0,
                    "recent_output_blocker_count": 0,
                    "live_or_recent_blocker_count": 0,
                    "category_counts": {"ai_trader_toto_mps_exporter": 2},
                    "live_category_counts": {},
                    "recent_output_category_counts": {},
                    "blockers": [],
                },
            )

            result = report_mod.audit(
                source_json=source,
                import_json=imports,
                smoke_bundle_json=smoke,
                next_mps_summary_json=next_mps,
                blocker_diagnosis_json=diagnosis,
                current_benchmark_environment_probe={
                    "available": True,
                    "status": "contended",
                    "blocks_promotion": True,
                    "returncode": 2,
                    "blocking_process_count": 2,
                    "blocking_processes": [
                        {
                            "pid": 13496,
                            "ppid": 13480,
                            "stat": "R",
                            "elapsed": "02:01:33",
                            "block_reason": "high_cpu",
                            "pcpu": 190.0,
                            "pmem": 2.1,
                            "command": "python /Users/nicholasbardy/git/font_maker/train_node_curve_program_flow_v2.py --device mps",
                        },
                        {
                            "pid": 54895,
                            "ppid": 54881,
                            "stat": "S+",
                            "elapsed": "02:20:11",
                            "block_reason": "periodic_mps_exporter",
                            "pcpu": 0.0,
                            "pmem": 0.1,
                            "command": "/usr/bin/SCREEN -dmS toto_floor001_postfix_20260520T171609Z zsh -lc python scripts/run_btc15m_overnight_shadow_monitor.py --run-id toto",
                        },
                    ],
                },
            )

        self.assertEqual(result["status"], "blocked_external_environment")
        self.assertFalse(result["clean_mps_rerun_plan"]["ready_to_run_now"])
        self.assertEqual(result["clean_mps_rerun_plan"]["current_benchmark_environment_status"], "contended")
        self.assertTrue(result["clean_mps_rerun_plan"]["current_benchmark_environment_blocks_promotion"])
        self.assertEqual(
            result["clean_mps_rerun_plan"]["blocking_conditions"],
            ["current_benchmark_environment_contended"],
        )
        self.assertEqual(
            result["clean_mps_rerun_plan"]["wait_reason"],
            "current benchmark environment probe must report ok/background before a clean MPS gate run",
        )
        current_probe = result["artifacts"]["current_benchmark_environment_probe"]
        self.assertEqual(
            current_probe["blocking_kind_counts"],
            {"font_maker_random_stroke_train": 1, "periodic_mps_exporter": 1},
        )
        self.assertEqual(
            current_probe["blocking_reason_counts"],
            {"high_cpu": 1, "periodic_mps_exporter": 1},
        )
        self.assertEqual(
            current_probe["blocking_screen_session_names"],
            ["toto_floor001_postfix_20260520T171609Z"],
        )
        self.assertEqual(current_probe["blocking_process_sample"][0]["pid"], 13496)
        self.assertIn(
            "wait for font_maker random-stroke training to finish or pause it",
            current_probe["manual_next_actions"],
        )
        rerun_plan = result["clean_mps_rerun_plan"]
        self.assertEqual(rerun_plan["current_benchmark_environment_blocking_process_count"], 2)
        self.assertTrue(rerun_plan["run_after_estimated_done_at_requires_reprobe"])
        self.assertTrue(rerun_plan["current_benchmark_environment_has_independent_blockers"])
        self.assertEqual(
            rerun_plan["current_benchmark_environment_blocking_kind_counts"],
            {"font_maker_random_stroke_train": 1, "periodic_mps_exporter": 1},
        )
        self.assertEqual(
            rerun_plan["current_benchmark_environment_blocking_screen_session_names"],
            ["toto_floor001_postfix_20260520T171609Z"],
        )
        self.assertIn(
            "wait for or manually pause periodic ai_trader/TOTO MPS exporter work",
            rerun_plan["current_benchmark_environment_manual_next_actions"],
        )

    def test_reports_failed_prerequisite_when_shader_gate_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.json"
            imports = root / "imports.json"
            smoke = root / "smoke.json"
            next_mps = root / "next.json"
            _ok_artifact(source)
            _write(imports, {"status": "failed", "failures": ["import failed"]})
            _ok_artifact(smoke)
            _blocked_summary(next_mps)

            result = report_mod.audit(
                source_json=source,
                import_json=imports,
                smoke_bundle_json=smoke,
                next_mps_summary_json=next_mps,
            )

        self.assertEqual(result["status"], "failed_prerequisite")
        self.assertFalse(result["shader_fork_smoke_state_fixed"])
        self.assertIn("import", "\n".join(result["failures"]))

    def test_status_ok_stub_does_not_satisfy_shader_fork_gates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.json"
            imports = root / "imports.json"
            smoke = root / "smoke.json"
            next_mps = root / "next.json"
            _write(source, {"status": "ok", "failures": []})
            _ok_artifact(imports)
            _ok_artifact(smoke)
            _blocked_summary(next_mps)

            result = report_mod.audit(
                source_json=source,
                import_json=imports,
                smoke_bundle_json=smoke,
                next_mps_summary_json=next_mps,
            )

        self.assertEqual(result["status"], "failed_prerequisite")
        self.assertFalse(result["shader_fork_smoke_state_fixed"])
        self.assertFalse(result["fixed_requirements"]["native_source_wiring"])
        self.assertIn(
            f"source: variant_count is None, expected {len(report_mod.DEFAULT_VARIANTS)}",
            result["failures"],
        )
        self.assertIn("source: variants is missing or not a list", result["failures"])

    def test_missing_referenced_shader_artifact_paths_fail_shader_fork_gates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.json"
            imports = root / "imports.json"
            smoke = root / "smoke.json"
            next_mps = root / "next.json"
            _ok_artifact(source)
            _ok_artifact(imports)
            _ok_artifact(smoke)
            _blocked_summary(next_mps)
            imports_payload = json.loads(imports.read_text(encoding="utf-8"))
            imports_payload["variants"][0]["extension_library"] = str(root / "missing_extension.so")
            _write(imports, imports_payload)
            smoke_payload = json.loads(smoke.read_text(encoding="utf-8"))
            smoke_payload["required"][0]["path"] = str(root / "missing_smoke.json")
            _write(smoke, smoke_payload)

            result = report_mod.audit(
                source_json=source,
                import_json=imports,
                smoke_bundle_json=smoke,
                next_mps_summary_json=next_mps,
            )

        self.assertEqual(result["status"], "failed_prerequisite")
        self.assertFalse(result["fixed_requirements"]["native_import_registration"])
        self.assertFalse(result["fixed_requirements"]["rebuilt_native_smoke_bundle"])
        self.assertIn(
            f"import: {report_mod.DEFAULT_VARIANTS[0][0]}: extension_library does not exist",
            result["failures"],
        )
        self.assertIn(
            "smoke_bundle: direct_power_boundary: artifact path is missing or does not exist",
            result["failures"],
        )

    def test_reports_complete_only_when_all_shader_gates_and_next_mps_are_verified(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.json"
            imports = root / "imports.json"
            smoke = root / "smoke.json"
            next_mps = root / "next.json"
            artifact = root / "candidate.worldfoam.json"
            for path in (source, imports, smoke):
                _ok_artifact(path)
            _verified_summary(next_mps, artifact)

            result = report_mod.audit(
                source_json=source,
                import_json=imports,
                smoke_bundle_json=smoke,
                next_mps_summary_json=next_mps,
            )

        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["objective_complete"])
        self.assertEqual(result["artifacts"]["next_mps_quality_speed"]["result_verifier_status"], "ok")
        self.assertTrue(result["artifacts"]["next_mps_quality_speed"]["embedded_result_verification_complete"])

    def test_does_not_complete_when_clean_artifact_was_not_embedded_verified(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.json"
            imports = root / "imports.json"
            smoke = root / "smoke.json"
            next_mps = root / "next.json"
            artifact = root / "candidate.worldfoam.json"
            for path in (source, imports, smoke):
                _ok_artifact(path)
            _verified_summary(next_mps, artifact, embedded_result=False)

            result = report_mod.audit(
                source_json=source,
                import_json=imports,
                smoke_bundle_json=smoke,
                next_mps_summary_json=next_mps,
            )

        self.assertEqual(result["status"], "incomplete_missing_clean_mps_gate")
        self.assertFalse(result["objective_complete"])
        self.assertEqual(result["artifacts"]["next_mps_quality_speed"]["result_verifier_status"], "ok")
        self.assertFalse(result["artifacts"]["next_mps_quality_speed"]["embedded_result_verification_complete"])
        self.assertIn(
            "next-MPS summary was not launched with verify_result=true",
            result["artifacts"]["next_mps_quality_speed"]["embedded_result_verifier_failures"],
        )
        self.assertIn(
            "next_mps: next-MPS embedded result_verifier_payload is missing",
            result["failures"],
        )

    def test_does_not_complete_when_embedded_verifier_payload_targets_other_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.json"
            imports = root / "imports.json"
            smoke = root / "smoke.json"
            next_mps = root / "next.json"
            artifact = root / "candidate.worldfoam.json"
            for path in (source, imports, smoke):
                _ok_artifact(path)
            _verified_summary(next_mps, artifact)
            payload = json.loads(next_mps.read_text(encoding="utf-8"))
            verifier_payload = payload["result_verifier_payload"]
            assert isinstance(verifier_payload, dict)
            verifier_payload["worldfoam_artifact"] = str(root / "other.worldfoam.json")
            _write(next_mps, payload)

            result = report_mod.audit(
                source_json=source,
                import_json=imports,
                smoke_bundle_json=smoke,
                next_mps_summary_json=next_mps,
            )

        self.assertEqual(result["status"], "incomplete_missing_clean_mps_gate")
        self.assertFalse(result["objective_complete"])
        self.assertEqual(result["artifacts"]["next_mps_quality_speed"]["result_verifier_status"], "ok")
        self.assertIn(
            "next-MPS embedded result_verifier_payload worldfoam_artifact does not match plan",
            result["artifacts"]["next_mps_quality_speed"]["embedded_result_verifier_failures"],
        )

    def test_does_not_complete_on_legacy_result_verified_stub(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.json"
            imports = root / "imports.json"
            smoke = root / "smoke.json"
            next_mps = root / "next.json"
            for path in (source, imports, smoke):
                _ok_artifact(path)
            _write(next_mps, {"status": "result_verified"})

            result = report_mod.audit(
                source_json=source,
                import_json=imports,
                smoke_bundle_json=smoke,
                next_mps_summary_json=next_mps,
            )

        self.assertEqual(result["status"], "incomplete_missing_clean_mps_gate")
        self.assertFalse(result["objective_complete"])
        self.assertEqual(result["artifacts"]["next_mps_quality_speed"]["result_verifier_status"], "failed")
        self.assertIn("summary status is not train_eval_ok: 'result_verified'", "\n".join(result["failures"]))

    def test_default_next_mps_summary_uses_newest_launcher_summary_not_train_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_artifact = root / "2026-05-21_worldfoam_next_mps_goal_continuation_clean_retry.json"
            older_summary = root / "2026-05-21_worldfoam_next_mps_goal_continuation_preflight.launch_summary.json"
            old_named_newer_summary = (
                root / "2026-05-21_worldfoam_next_mps_goal_continuation_clean_retry.launch_summary.json"
            )
            live_preflight_summary = root / "2026-05-21_worldfoam_next_mps_live_preflight_0348.launch_summary.json"
            for path in (train_artifact, older_summary, old_named_newer_summary, live_preflight_summary):
                _write(path, {"status": "preflight_contended"})
            os.utime(older_summary, ns=(1_000_000_000, 1_000_000_000))
            os.utime(old_named_newer_summary, ns=(2_000_000_000, 2_000_000_000))
            os.utime(live_preflight_summary, ns=(3_000_000_000, 3_000_000_000))

            result = report_mod.default_next_mps_summary_json(root)

        self.assertEqual(result, live_preflight_summary)


if __name__ == "__main__":
    unittest.main()
