from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import refresh_worldfoam_fork_shader_goal_state as refresh_mod


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _ok(path: Path) -> None:
    stem = path.stem
    if "source" in stem:
        variant_root = path.parent / "variants"
        variant_root.mkdir(parents=True, exist_ok=True)
        variants = [
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
            for variant, package in refresh_mod.goal_report.DEFAULT_VARIANTS
        ]
        _write(
            path,
            {
                "status": "ok",
                "failures": [],
                "variant_root": str(variant_root),
                "variant_count": len(variants),
                "variants": variants,
            },
        )
    elif "import" in stem:
        variant_root = path.parent / "variants"
        variant_root.mkdir(parents=True, exist_ok=True)
        variants = []
        for variant, package in refresh_mod.goal_report.DEFAULT_VARIANTS:
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
        _write(
            path,
            {
                "status": "ok",
                "failures": [],
                "variant_root": str(variant_root),
                "variant_count": len(variants),
                "variants": variants,
            },
        )
    elif "smoke" in stem:
        required = []
        for spec in refresh_mod.goal_report.REQUIRED_ARTIFACTS:
            artifact_path = path.parent / f"{spec['label']}.json"
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
        invalid_path = path.parent / "known_invalid_tiled_ownerupdate.json"
        _write(invalid_path, {"status": "failed"})
        _write(
            path,
            {
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
                },
            },
        )
    else:
        _write(path, {"status": "ok", "failures": []})


class RefreshWorldFoamForkShaderGoalStateTests(unittest.TestCase):
    def test_current_benchmark_environment_probe_marks_contended_preflight_as_blocking(self) -> None:
        with mock.patch.object(
            refresh_mod.launcher,
            "_run_json_command",
            return_value=(
                2,
                {
                    "status": "contended",
                    "blocking_process_count": 1,
                    "contending_process_count": 1,
                    "background_process_count": 0,
                    "process_sample_limit": 32,
                    "blocking_processes": [
                        {
                            "pid": 99,
                            "block_reason": "periodic_mps_exporter",
                            "command": "python scripts/run_btc15m_overnight_shadow_monitor.py --run-id toto",
                        }
                    ],
                    "contending_processes": [{"pid": 99}],
                    "background_processes": [],
                },
                "{}",
                "",
            ),
        ):
            probe = refresh_mod._current_benchmark_environment_probe(wait_timeout_s=0.0, wait_poll_s=1.0)

        self.assertTrue(probe["available"])
        self.assertEqual(probe["status"], "contended")
        self.assertTrue(probe["blocks_promotion"])
        self.assertEqual(probe["returncode"], 2)
        self.assertEqual(probe["blocking_processes"][0]["pid"], 99)
        self.assertEqual(probe["blocking_kind_counts"], {"periodic_mps_exporter": 1})
        self.assertEqual(probe["blocking_reason_counts"], {"periodic_mps_exporter": 1})
        self.assertIn(
            "wait for or manually pause periodic ai_trader/TOTO MPS exporter work",
            probe["manual_next_actions"],
        )

    def test_refresh_writes_diagnosis_before_goal_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.json"
            imports = root / "imports.json"
            smoke = root / "smoke.json"
            next_mps = root / "next.launch_summary.json"
            diagnosis = root / "diagnosis.json"
            goal = root / "goal.json"
            for path in (source, imports, smoke):
                _ok(path)
            _write(
                next_mps,
                {
                    "benchmark": "world_foam_next_mps_candidate_launch",
                    "status": "preflight_contended",
                    "execute": True,
                    "readiness_status": "ok",
                    "ready_for_quiet_mps_quality_speed_run": True,
                    "next_mps_candidate": "legacy_pixel_mean",
                    "preflight_returncode": 2,
                    "preflight_benchmark_environment_status": "contended",
                    "preflight_benchmark_environment": {
                        "status": "contended",
                        "blocking_cpu_threshold": 5.0,
                        "general_blocking_cpu_threshold": 75.0,
                        "blocking_process_count": 1,
                        "contending_process_count": 1,
                        "keywords": ["python", "torch", "mps"],
                        "hard_keywords": ["torch", "mps"],
                    },
                    "preflight_blocking_process_count": 1,
                    "preflight_contending_process_count": 1,
                    "preflight_stability_samples_requested": 3,
                    "preflight_stability_samples_completed": 1,
                    "preflight_stability_ok": False,
                    "train_eval_returncode": None,
                    "planned_worldfoam_artifact": str(root / "missing.worldfoam.json"),
                    "train_eval_command": [
                        "python",
                        "train_eval_owner_run_tape.py",
                        "--frame-counts",
                        "2,4,8,16,32",
                        "--site-initialization",
                        "legacy_pixel_mean",
                        "--optimizer-mode",
                        "manual-vjp",
                        "--tape-mode",
                        "native-cutwalk",
                        "--endpoint-record-source",
                        "slow-owner-run",
                        "--require-benchmark-environment-ok",
                        "--out-json",
                        str(root / "missing.worldfoam.json"),
                    ],
                    "preflight_blocking_processes": [
                        {
                            "pid": 987654321,
                            "ppid": 1,
                            "stat": "R",
                            "elapsed": "01:02",
                            "pcpu": 190.0,
                            "pmem": 1.0,
                            "block_reason": "high_cpu",
                            "command": "python train_node_curve_program_flow_v2.py",
                        }
                    ],
                },
            )

            with mock.patch.object(
                refresh_mod,
                "_current_benchmark_environment_probe",
                return_value={
                    "available": True,
                    "status": "ok",
                    "blocks_promotion": False,
                    "returncode": 0,
                    "blocking_process_count": 0,
                    "contending_process_count": 0,
                    "background_process_count": 0,
                },
            ) as current_probe:
                report = refresh_mod.refresh(
                    source_json=source,
                    import_json=imports,
                    smoke_bundle_json=smoke,
                    next_mps_summary_json=next_mps,
                    blocker_diagnosis_json=diagnosis,
                    out_json=goal,
                    recent_seconds=1.0,
                )

            diagnosis_payload = json.loads(diagnosis.read_text(encoding="utf-8"))
            goal_payload = json.loads(goal.read_text(encoding="utf-8"))

        self.assertEqual(report["status"], "incomplete_ready_for_clean_mps_gate")
        self.assertEqual(goal_payload["status"], "incomplete_ready_for_clean_mps_gate")
        self.assertEqual(
            Path(diagnosis_payload["summary_json"]).resolve(strict=False),
            next_mps.resolve(strict=False),
        )
        live_diagnosis = goal_payload["artifacts"]["live_blocker_diagnosis"]
        self.assertTrue(live_diagnosis["available"])
        self.assertTrue(live_diagnosis["matches_next_mps_summary"])
        self.assertEqual(
            Path(live_diagnosis["summary_json"]).resolve(strict=False),
            next_mps.resolve(strict=False),
        )
        self.assertEqual(live_diagnosis["blocker_count"], 1)
        self.assertEqual(live_diagnosis["status"], "no_live_or_recent_blockers_found")
        self.assertEqual(live_diagnosis["live_or_recent_blocker_count"], 0)
        self.assertTrue(goal_payload["clean_mps_rerun_plan"]["ready_to_run_now"])
        self.assertEqual(
            goal_payload["artifacts"]["current_benchmark_environment_probe"]["status"],
            "ok",
        )
        current_probe.assert_called_once()


if __name__ == "__main__":
    unittest.main()
